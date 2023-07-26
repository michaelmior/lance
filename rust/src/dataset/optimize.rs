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

use std::ops::{AddAssign, Range};
use std::sync::Arc;

use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;

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
    let compaction_groups: Vec<Vec<FileFragment>> = plan_compaction(dataset, &options).await?;

    // Finally, run a compaction job to compact the fragments. This works by:
    // Scanning the fragments in each group, and writing the rows to a new file
    // until we have reached the appropriate size. Then, we move to writing a new
    // file.
    let mut metrics = CompactionMetrics::default();

    // If nothing to compact, don't make a commit.
    if compaction_groups.is_empty() {
        return Ok(metrics);
    }

    // Once all the files are written, we collect the metadata and commit.
    let mut result_stream = futures::stream::iter(compaction_groups)
        .map(|group| rewrite_files(group, &options))
        .buffer_unordered(options.num_concurrent_jobs);

    let existing_fragments: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(|f| f.metadata)
        .collect();
    let mut current_fragment_id = dataset
        .manifest
        .max_fragment_id()
        .map(|max| max + 1)
        .unwrap_or(0);

    let mut final_fragments: Vec<Fragment> = Vec::new();
    let mut existing_frag_i = 0;

    while let Some(result) = result_stream.next().await {
        let result = result?;

        // Update the metrics
        metrics += result.metrics;

        // We want to replace the old fragments with the new ones, but we we'd
        // like to preserve the order.
        for existing_fragment in existing_fragments[existing_frag_i..].iter() {
            dbg!(existing_fragment.id, &result.replaced_fragment_ids);
            if existing_fragment.id as u32 == *result.replaced_fragment_ids.first().unwrap() {
                break;
            }
            
            final_fragments.push(existing_fragment.clone());
        }
        // Skip the fragments we are replacing
        existing_frag_i += result.replaced_fragment_ids.len();

        // Add new fragments
        for mut new_fragment in result.new_fragments {
            new_fragment.id = current_fragment_id;
            current_fragment_id += 1;
            final_fragments.push(new_fragment);
        }
    }

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

    /// The number of rows that are still in the fragment
    fn num_rows(&self) -> usize {
        self.fragment_length - self.num_deletions
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

/// Formulate a plan to compact the files in a dataset
///
/// Returns a list of groups of files that should be compacted together. The groups
/// are separated and internally ordered such that they can preserve the existing
/// order of the dataset.
async fn plan_compaction(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<Vec<Vec<FileFragment>>> {
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

    let mut groups = Vec::new();
    let mut current_group = Vec::new();

    while let Some(res) = fragment_metrics.next().await {
        let (fragment, metrics) = res?;

        // If the fragment is too small, add it to the current group.
        if metrics.fragment_length < options.target_rows_per_fragment
            || (options.materialize_deletions
                && metrics.deletion_percentage() > options.materialize_deletion_threshold)
        {
            dbg!("Adding fragment to group", fragment.id(), metrics.num_rows(), metrics.deletion_percentage());
            // If the fragment has deletions, and we are materializing deletions,
            // add it to the current group.
            current_group.push(fragment);
        } else {
            dbg!("skipping fragment", fragment.id());
            // Otherwise, add the current group to the list of groups, and start
            // a new group with this fragment.
            if !current_group.is_empty() {
                groups.push(std::mem::take(&mut current_group));
            }
        }
    }
    
    // Add final group
    groups.push(current_group);

    // Cleanup: remove any lone files we don't have reason to compact.
    let mut to_drop = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        if group.len() == 1 && group[0].metadata.deletion_file.is_none() {
            to_drop.push(i);
        }
    }
    for i in to_drop {
        groups.remove(i);
    }

    Ok(groups)
}

#[derive(Debug)]
struct RewriteResult {
    metrics: CompactionMetrics,
    new_fragments: Vec<Fragment>,
    replaced_fragment_ids: Vec<u32>,
}

async fn rewrite_files(
    group: Vec<FileFragment>,
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let mut metrics = CompactionMetrics::default();

    if group.is_empty() {
        return Ok(RewriteResult {
            metrics,
            new_fragments: Vec::new(),
            replaced_fragment_ids: Vec::new(),
        });
    }

    let dataset = group[0].dataset();
    let fragments = group
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

    metrics.files_removed = group
        .iter()
        .map(|f| f.metadata.files.len() + f.metadata.deletion_file.is_some() as usize)
        .sum();
    metrics.fragments_removed = group.len();
    metrics.fragments_added = new_fragments.len();
    metrics.files_added = new_fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();

    Ok(RewriteResult {
        metrics,
        new_fragments,
        replaced_fragment_ids: group.iter().map(|f| f.id() as u32).collect(),
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
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
        assert_eq!(plan.len(), 0);

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
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema().clone());
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
        assert_eq!(plan.len(), 0);

        // Now split across multiple files
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema().clone());
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
        assert_eq!(plan.len(), 0);
    }

    #[tokio::test]
    async fn test_compact_many() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();
        
        // Create a table with 3 small fragments
        let reader = RecordBatchIterator::new(vec![
            Ok(data.slice(0, 1200))], data.schema().clone());
        let write_params = WriteParams {
            max_rows_per_file: 400,
            ..Default::default()
        };
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Append 2 large fragments (1k rows)
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(1200, 2000))], data.schema().clone());
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
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(3200, 600))], data.schema().clone());
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
        let plan = plan_compaction(&dataset, &options)
            .await
            .unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].len(), 3);
        assert_eq!(plan[1].len(), 3);

        assert_eq!(plan[0].iter().map(|f| f.id()).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(plan[1].iter().map(|f| f.id()).collect::<Vec<_>>(), vec![4, 5, 6]);

        // Run compaction
        let metrics = compact_files(&mut dataset, options)
            .await
            .unwrap();

        // Assert on metrics
        assert_eq!(metrics.fragments_removed, 6);
        assert_eq!(metrics.fragments_added, 4);
        assert_eq!(metrics.files_removed, 7); // 6 data files + 1 deletion file
        assert_eq!(metrics.files_added, 4);

        let fragment_ids = dataset.get_fragments().iter().map(|f| f.id()).collect::<Vec<_>>();
        assert_eq!(fragment_ids, vec![7, 8, 3, 9, 10]);
    }

    #[tokio::test]
    async fn test_compact_data_files() {
        // Create a table with 2 small fragments

        // Add a column

        // Create compaction plan
        // Assert both files compacted

        // Run compaction

        // Assert files reduced to 1 per fragment
        // assert just one fragment

        // Assert order unchanged and data is all there.
    }
}
