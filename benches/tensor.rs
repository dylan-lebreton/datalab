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

criterion_group!(benches, bench_add, bench_mul_scalar, bench_sum);
criterion_main!(benches);
