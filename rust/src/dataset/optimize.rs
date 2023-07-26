// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Table maintenance for optimizing table layout.

use std::collections::HashMap;
use std::ops::{AddAssign, Range};
use std::sync::Arc;

use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{Stream, StreamExt};

use crate::format::Manifest;
use crate::Result;
use crate::{format::Fragment, Dataset};

use super::fragment::FileFragment;
use super::{write_fragments, write_manifest_file, WriteMode, WriteParams};

#[derive(Debug, Clone)]
pub struct CompactionOptions {
    /// Target number of rows per file. Defaults to 1 million.
    ///
    /// This is used to determine which fragments need compaction, as any
    /// fragments that have fewer rows than this value will be candidates for
    /// compaction.
    target_rows_per_fragment: usize,
    /// Max number of rows per group
    ///
    /// This does not affect which fragments need compaction, but does affect
    /// how they are re-written if selected.
    max_rows_per_group: usize,
    /// Whether to compact fragments with deletions so there are no deletions.
    /// Defaults to true.
    materialize_deletions: bool,
    /// The fraction of rows that need to be deleted in a fragment before
    /// materializing the deletions. Defaults to 10% (0.1). Setting to zero (or
    /// lower) will materialize deletions for all fragments with deletions.
    /// Setting above 1.0 will never materialize deletions.
    materialize_deletion_threshold: f32,
    /// The number of concurrent jobs. Defaults to the number of CPUs.
    num_concurrent_jobs: usize,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            // Matching defaults fro WriteParams
            target_rows_per_fragment: 1024 * 1024,
            max_rows_per_group: 1024,
            materialize_deletions: true,
            materialize_deletion_threshold: 0.1,
            num_concurrent_jobs: num_cpus::get(),
        }
    }
}

impl CompactionOptions {
    pub fn validate(&mut self) {
        // If threshold is 100%, same as turning off deletion materialization.
        if self.materialize_deletions && self.materialize_deletion_threshold > 1.0 {
            self.materialize_deletions = false;
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompactionMetrics {
    fragments_removed: usize,
    fragments_added: usize,
    files_removed: usize,
    files_added: usize,
}

impl AddAssign for CompactionMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.fragments_removed += rhs.fragments_removed;
        self.fragments_added += rhs.fragments_added;
        self.files_removed += rhs.files_removed;
        self.files_added += rhs.files_added;
    }
}

/// Compacts the files in the dataset without reordering them.
///
/// This does a few things:
///  * Removes deleted rows from fragments.
///  * Removed dropped columns from fragments.
///  * Merges fragments that are too small.
///
/// This method tries to preserve the insertion order of rows in the dataset.
///
/// If no compaction is needed, this method will not make a new version of the table.
pub async fn compact_files(
    dataset: &mut Dataset,
    mut options: CompactionOptions,
) -> Result<CompactionMetrics> {
    // First, validate the arguments.
    options.validate();

    // Then, build a plan about which fragments to compact, and in what groups
    let compaction_plan: CompactionPlan = plan_compaction(dataset, &options).await?;

    // Finally, run a compaction job to compact the fragments. This works by:
    // Scanning the fragments in each group, and writing the rows to a new file
    // until we have reached the appropriate size. Then, we move to writing a new
    // file.
    let mut metrics = CompactionMetrics::default();

    // If nothing to compact, don't make a commit.
    if compaction_plan.rewrite_groups.is_empty() {
        return Ok(metrics);
    }

    // Once all the files are written, we collect the metadata and commit.
    let mut result_stream = futures::stream::iter(compaction_plan.fragments_iter())
        .map(|task| rewrite_files(task, &options))
        .buffer_unordered(options.num_concurrent_jobs);

    let mut current_fragment_id = dataset
        .manifest
        .max_fragment_id()
        .map(|max| max + 1)
        .unwrap_or(0);

    // Mapping of replacement ranges to the new fragments
    let mut new_fragments: HashMap<Range<usize>, Vec<Fragment>> = HashMap::new();

    while let Some(result) = result_stream.next().await {
        let mut result = result?;

        // Update the metrics
        metrics += result.metrics;

        for fragment in &mut result.new_fragments {
            fragment.id = current_fragment_id;
            current_fragment_id += 1;
        }

        let RewriteResult {
            new_fragments: fragments,
            replace_range,
            ..
        } = result;
        new_fragments.insert(replace_range, fragments);
    }

    let final_fragments = compaction_plan.build_fragment_list(new_fragments);

    // Commit the dataset transaction
    let indices = if metrics.fragments_removed == dataset.get_fragments().len() {
        // All fragments were replaced, so the index is totally invalid
        None
    } else {
        Some(dataset.load_indices().await?)
    };

    // TODO: replace this with from_previous
    let mut manifest = Manifest::new(dataset.schema(), Arc::new(final_fragments));

    manifest.version = dataset
        .latest_manifest()
        .await
        .map(|m| m.version + 1)
        .unwrap_or(1);

    write_manifest_file(
        &dataset.object_store,
        &dataset.base,
        &mut manifest,
        indices,
        Default::default(),
    )
    .await?;

    dataset.manifest = Arc::new(manifest);

    // Finally, we return the metrics.
    Ok(metrics)
}

// TODO: ideally these metrics should already be in the manifest, so we don't
// have to scan during compaction.

/// Information about a fragment used to decide it's fate in compaction
struct FragmentMetrics {
    /// The number of original rows in the fragment
    pub fragment_length: usize,
    /// The number of rows that have been deleted
    pub num_deletions: usize,
}

impl FragmentMetrics {
    /// The fraction of rows that have been deleted
    fn deletion_percentage(&self) -> f32 {
        if self.fragment_length > 0 {
            self.num_deletions as f32 / self.fragment_length as f32
        } else {
            0.0
        }
    }
}

async fn collect_metrics(fragment: &FileFragment) -> Result<FragmentMetrics> {
    let fragment_length = fragment.fragment_length();
    let num_deletions = fragment.count_deletions();
    let (fragment_length, num_deletions) =
        futures::future::try_join(fragment_length, num_deletions).await?;
    Ok(FragmentMetrics {
        fragment_length,
        num_deletions,
    })
}

struct CompactionPlan {
    pub fragments: Vec<FileFragment>,
    pub rewrite_groups: Vec<Range<usize>>,
    pub keep_groups: Vec<Range<usize>>,
}

struct CompactionTask<'a> {
    pub fragments: &'a [FileFragment],
    pub replace_range: Range<usize>,
}

impl CompactionPlan {
    fn fragments_iter(&self) -> impl Iterator<Item = CompactionTask> {
        self.rewrite_groups.iter().map(|range| CompactionTask {
            fragments: &self.fragments[range.clone()],
            replace_range: range.clone(),
        })
    }

    fn build_fragment_list(
        &self,
        mut new_fragments: HashMap<Range<usize>, Vec<Fragment>>,
    ) -> Vec<Fragment> {
        let mut fragments = Vec::with_capacity(self.fragments.len());

        let mut i = 0;
        let mut rewrite_iter = self.rewrite_groups.iter().peekable();
        let mut keep_iter = self.keep_groups.iter().peekable();

        while i < self.fragments.len() {
            if i == rewrite_iter.peek().map(|r| r.start).unwrap_or(usize::MAX) {
                let range = rewrite_iter.next().unwrap();
                fragments.extend(new_fragments.remove(range).unwrap());
                i = range.end;
            } else {
                let range = keep_iter.next().unwrap();
                fragments.extend(
                    self.fragments[range.clone()]
                        .iter()
                        .map(|f| f.metadata.clone()),
                );
                i = range.end;
            }
        }

        fragments
    }
}

struct CompactionPlanBuilder {
    fragments: Vec<FileFragment>,
    rewrite_groups: Vec<Range<usize>>,
    keep_groups: Vec<Range<usize>>,
    current_group_start: usize,
    current_group_size: usize,
    current_is_rewrite: bool,
    i: usize,
}

impl From<CompactionPlanBuilder> for CompactionPlan {
    fn from(mut builder: CompactionPlanBuilder) -> Self {
        // Finish last group
        if builder.current_group_size > 0 {
            if builder.current_is_rewrite && builder.current_group_size > 1 {
                builder
                    .rewrite_groups
                    .push(builder.current_group_start..builder.fragments.len());
            } else {
                builder
                    .keep_groups
                    .push(builder.current_group_start..builder.fragments.len());
            }
        }

        let CompactionPlanBuilder {
            fragments,
            rewrite_groups,
            keep_groups,
            ..
        } = builder;

        Self {
            fragments,
            rewrite_groups,
            keep_groups,
        }
    }
}

impl CompactionPlanBuilder {
    fn with_capacity(n: usize) -> Self {
        Self {
            fragments: Vec::with_capacity(n),
            rewrite_groups: Vec::new(),
            keep_groups: Vec::new(),
            current_group_start: 0,
            current_group_size: 0,
            current_is_rewrite: false,
            i: 0,
        }
    }

    fn push(&mut self, fragment: FileFragment, is_candidate: bool) {
        self.fragments.push(fragment);

        if self.current_is_rewrite != is_candidate {
            if self.current_group_size > 0 {
                if self.current_is_rewrite && self.current_group_size > 1 {
                    self.rewrite_groups.push(self.current_group_start..self.i);
                } else {
                    self.keep_groups.push(self.current_group_start..self.i);
                }
            }
            self.current_group_start = self.i;
            self.current_is_rewrite = is_candidate;
            self.current_group_size = 1;
        } else {
            self.current_group_size += 1;
        }

        self.i += 1;
    }
}

/// Formulate a plan to compact the files in a dataset
///
/// Returns a list of groups of files that should be compacted together. The groups
/// are separated and internally ordered such that they can preserve the existing
/// order of the dataset.
async fn plan_compaction(dataset: &Dataset, options: &CompactionOptions) -> Result<CompactionPlan> {
    // We assume here that get_fragments is returning the fragments in a
    // meaningful order that we want to preserve.
    let mut fragment_metrics = futures::stream::iter(dataset.get_fragments())
        .map(|fragment| async move {
            match collect_metrics(&fragment).await {
                Ok(metrics) => Ok((fragment, metrics)),
                Err(e) => Err(e),
            }
        })
        .buffered(num_cpus::get() * 2);

    let mut compaction_plan = CompactionPlanBuilder::with_capacity(fragment_metrics.size_hint().0);

    while let Some(res) = fragment_metrics.next().await {
        let (fragment, metrics) = res?;

        // If either the fragment is too small or has too many deletions, we
        // consider it a candidate for compaction.
        if metrics.fragment_length < options.target_rows_per_fragment
            || (options.materialize_deletions
                && metrics.deletion_percentage() > options.materialize_deletion_threshold)
        {
            compaction_plan.push(fragment, true);
        } else {
            compaction_plan.push(fragment, false);
        }
    }

    Ok(compaction_plan.into())
}

#[derive(Debug)]
struct RewriteResult {
    metrics: CompactionMetrics,
    new_fragments: Vec<Fragment>,
    replace_range: Range<usize>,
}

async fn rewrite_files(
    task: CompactionTask<'_>,
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let mut metrics = CompactionMetrics::default();

    if task.fragments.is_empty() {
        return Ok(RewriteResult {
            metrics,
            new_fragments: Vec::new(),
            replace_range: task.replace_range,
        });
    }

    let dataset = task.fragments[0].dataset();
    let fragments = task
        .fragments
        .iter()
        .map(|fragment| fragment.metadata.clone())
        .collect();
    let mut scanner = dataset.scan();
    scanner.with_fragments(fragments);

    let data = SendableRecordBatchStream::from(scanner.try_into_stream().await?);

    let params = WriteParams {
        max_rows_per_file: options.target_rows_per_fragment,
        max_rows_per_group: options.max_rows_per_group,
        mode: WriteMode::Append,
        ..Default::default()
    };
    let new_fragments = write_fragments(
        dataset.object_store.clone(),
        &dataset.base,
        dataset.schema(),
        data,
        params,
    )
    .await?;

    metrics.files_removed = task
        .fragments
        .iter()
        .map(|f| f.metadata.files.len() + f.metadata.deletion_file.is_some() as usize)
        .sum();
    metrics.fragments_removed = task.fragments.len();
    metrics.fragments_added = new_fragments.len();
    metrics.files_added = new_fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();

    Ok(RewriteResult {
        metrics,
        new_fragments,
        replace_range: task.replace_range,
    })
}

#[cfg(test)]
mod tests {

    use arrow_array::{Float32Array, Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use arrow_select::concat::concat_batches;
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use super::*;

    fn sample_data() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from_iter_values(0..10_000))],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_compact_empty() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // Compact an empty table
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), Arc::new(schema));
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.rewrite_groups.len(), 0);

        let metrics = compact_files(&mut dataset, CompactionOptions::default())
            .await
            .unwrap();

        assert_eq!(metrics, CompactionMetrics::default());
        assert_eq!(dataset.manifest.version, 1);
    }

    #[tokio::test]
    async fn test_compact_all_good() {
        // Compact a table with nothing to do
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        // Just one file
        let write_params = WriteParams {
            max_rows_per_file: 10_000,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // There's only one file, so we can't compact any more if we wanted to.
        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.rewrite_groups.len(), 0);

        // Now split across multiple files
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 3_000,
            max_rows_per_group: 1_000,
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.rewrite_groups.len(), 0);
    }

    #[tokio::test]
    async fn test_compact_many() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();

        // Create a table with 3 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1200))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 400,
            ..Default::default()
        };
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Append 2 large fragments (1k rows)
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(1200, 2000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Delete 1 row from first large fragment
        dataset.delete("a = 1300").await.unwrap();

        // Delete 20% of rows from second large fragment
        dataset.delete("a >= 2400 AND a < 2600").await.unwrap();

        // Append 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(3200, 600))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 300,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Create compaction plan
        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.rewrite_groups.len(), 2);
        assert_eq!(plan.rewrite_groups[0].len(), 3);
        assert_eq!(plan.rewrite_groups[1].len(), 3);

        assert_eq!(plan.rewrite_groups[0], 0..3);
        assert_eq!(plan.rewrite_groups[1], 4..7);

        // Run compaction
        let metrics = compact_files(&mut dataset, options).await.unwrap();

        // Assert on metrics
        assert_eq!(metrics.fragments_removed, 6);
        assert_eq!(metrics.fragments_added, 4);
        assert_eq!(metrics.files_removed, 7); // 6 data files + 1 deletion file
        assert_eq!(metrics.files_added, 4);

        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id())
            .collect::<Vec<_>>();
        // Fragment ids are assigned on task completion, but that isn't deterministic.
        // But we can say the old fragment id=3 should be in the middle, and all
        // the other ids should be greater than 6.
        assert_eq!(fragment_ids[2], 3);
        assert!(fragment_ids.iter().all(|id| *id > 6 || *id == 3));
        dataset.validate().await.unwrap();
    }

    #[tokio::test]
    async fn test_compact_data_files() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();

        // Create a table with 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        // Just one file
        let write_params = WriteParams {
            max_rows_per_file: 5_000,
            max_rows_per_group: 1_000,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Add a column
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]);

        let data = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..10_000)),
                Arc::new(Float32Array::from_iter_values(
                    (0..10_000).map(|x| x as f32 * std::f32::consts::PI),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());

        dataset.merge(reader, "a", "a").await.unwrap();

        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.rewrite_groups.len(), 1);
        assert_eq!(plan.rewrite_groups[0].len(), 2);

        let metrics = compact_files(&mut dataset, CompactionOptions::default())
            .await
            .unwrap();

        assert_eq!(metrics.files_removed, 4); // 2 fragments with 2 data files
        assert_eq!(metrics.files_added, 1); // 1 fragment with 1 data file
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        // Assert order unchanged and data is all there.
        let scanner = dataset.scan();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let scanned_data = concat_batches(&batches[0].schema(), &batches).unwrap();

        assert_eq!(scanned_data, data);
    }
}
