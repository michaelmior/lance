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

use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance::utils::kmeans::KMeans;
use lance::utils::testing::generate_random_array;

fn bench_train(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    let dimension: i32 = 128;
    let array = generate_random_array(1024 * 4 * dimension as usize);

    c.bench_function("train_128d_4k", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = KMeans::new(&array, 256, 25, 50).await;
        })
    });

    #[cfg(feature = "faiss")]
    c.bench_function("train_128d_4k_faiss", |b| {
        use arrow_array::{cast::as_primitive_array, Float32Array};
        use faiss::cluster::kmeans_clustering;

        let array = data.as_ref().values();
        let f32array: &Float32Array = as_primitive_array(&array);
        b.iter(|| {
            kmeans_clustering(dimension as u32, 256, f32array.values()).unwrap();
        })
    });

    let array = generate_random_array(1024 * 64 * dimension as usize);
    c.bench_function("train_128d_65535", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = KMeans::new(&array, 256, 25, 50).await;
        })
    });

    #[cfg(feature = "faiss")]
    c.bench_function("train_128d_65535_faiss", |b| {
        use arrow_array::{cast::as_primitive_array, Float32Array};
        use faiss::cluster::kmeans_clustering;

        let array = data.as_ref().values();
        let f32array: &Float32Array = as_primitive_array(&array);
        b.iter(|| {
            kmeans_clustering(dimension as u32, 256, f32array.values()).unwrap();
        })
    });

    let dimension = 8;
    let array = generate_random_array(1024 * 64 * dimension as usize);
    c.bench_function("train_8d_65535", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = KMeans::new(&array, 256, 25, 50).await;
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
    .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_train);
// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_train);
criterion_main!(benches);
