//! Criterion benchmarks for the eager tensor kernels.
//!
//! Run with `cargo bench`. Throughput is reported in bytes/second of input
//! processed, so results can be compared against memory bandwidth.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datalab::tensor::Tensor;

const SIZES: &[usize] = &[1_000, 1_000_000];

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f64");
    for &n in SIZES {
        let a = Tensor::from_fn(n, |i| i as f64);
        let b = Tensor::from_fn(n, |i| (2 * i) as f64);
        group.throughput(Throughput::Bytes((2 * n * size_of::<f64>()) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(&a) + black_box(&b));
        });
    }
    group.finish();
}

fn bench_mul_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_scalar_f64");
    for &n in SIZES {
        let a = Tensor::from_fn(n, |i| i as f64);
        group.throughput(Throughput::Bytes((n * size_of::<f64>()) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(&a) * black_box(1.5));
        });
    }
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_f64");
    for &n in SIZES {
        let a = Tensor::from_fn(n, |i| (i % 7) as f64);
        group.throughput(Throughput::Bytes((n * size_of::<f64>()) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(&a).sum());
        });
    }
    group.finish();
}

/// Eager pipeline vs the same pipeline through the lazy engine: measures the
/// engine's overhead (batching, per-batch allocation/copy, boxed dispatch).
fn bench_lazy_vs_eager(c: &mut Criterion) {
    let n = 1_000_000;
    let mut group = c.benchmark_group("map_sum_f64_1M");
    group.throughput(Throughput::Bytes((n * size_of::<f64>()) as u64));

    let data = Tensor::from_fn(n, |i| (i % 100) as f64);

    group.bench_function("eager", |bench| {
        bench.iter(|| black_box(&data).map(|x| x * 2.0).sum());
    });

    // Same as "eager" plus the source clone the lazy path is forced to pay
    // (`lazy()` consumes the tensor): the fair apples-to-apples baseline.
    group.bench_function("eager_cloned", |bench| {
        bench.iter(|| black_box(&data).clone().map(|x| x * 2.0).sum());
    });

    // Sequential engine: measures the pure engine overhead (batching,
    // boxed dispatch) without threading.
    group.bench_function("lazy_seq", |bench| {
        bench.iter(|| {
            black_box(&data)
                .clone()
                .lazy()
                .with_threads(1)
                .map(|x| x * 2.0)
                .sum()
                .collect()
                .unwrap()
                .item()
        });
    });

    // Parallel engine: 256 KiB batches give the morsel machinery enough
    // morsels (32 for 1M f64) to spread across cores.
    group.bench_function("lazy_par", |bench| {
        bench.iter(|| {
            black_box(&data)
                .clone()
                .lazy()
                .with_batch_bytes(256 * 1024)
                .map(|x| x * 2.0)
                .sum()
                .collect()
                .unwrap()
                .item()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_mul_scalar,
    bench_sum,
    bench_lazy_vs_eager
);
criterion_main!(benches);
