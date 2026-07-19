//! Low-level compute kernels over contiguous slices.
//!
//! Kernels are datalab's innermost compute loops: plain functions over
//! contiguous slices, written once and reused everywhere — by the eager
//! [`Tensor`](crate::tensor::Tensor) operations today, and by the streaming
//! engine's batch operators tomorrow. Keeping them at the slice level makes
//! them trivially testable, benchmarkable, and lets the compiler
//! auto-vectorize the loops (the zip-style iteration compiles without bounds
//! checks).
//!
//! Binary kernels require all slices to have the same length and panic
//! otherwise: validating lengths is the caller's contract, kept out of the
//! inner loop.

use std::ops::{Add, Mul, Sub};

use crate::view::Element;

/// Panics with a clear message unless both lengths equal `out_len`.
fn check_lens(a_len: usize, b_len: usize, out_len: usize) {
    assert!(
        a_len == b_len && b_len == out_len,
        "kernel length mismatch: a={a_len}, b={b_len}, out={out_len}"
    );
}

/// Writes `a[i] + b[i]` into `out[i]` for every index.
///
/// # Panics
///
/// Panics if the three slices do not have the same length.
///
/// # Examples
///
/// ```
/// let (a, b, mut out) = ([1.0f64, 2.0], [10.0f64, 20.0], [0.0f64; 2]);
/// datalab::kernel::add(&a, &b, &mut out);
/// assert_eq!(out, [11.0, 22.0]);
/// ```
pub fn add<T: Element + Add<Output = T>>(a: &[T], b: &[T], out: &mut [T]) {
    check_lens(a.len(), b.len(), out.len());
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
}

/// Writes `a[i] - b[i]` into `out[i]` for every index.
///
/// # Panics
///
/// Panics if the three slices do not have the same length.
///
/// # Examples
///
/// ```
/// let (a, b, mut out) = ([10i32, 20], [1i32, 2], [0i32; 2]);
/// datalab::kernel::sub(&a, &b, &mut out);
/// assert_eq!(out, [9, 18]);
/// ```
pub fn sub<T: Element + Sub<Output = T>>(a: &[T], b: &[T], out: &mut [T]) {
    check_lens(a.len(), b.len(), out.len());
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x - y;
    }
}

/// Writes `a[i] * b[i]` into `out[i]` for every index (element-wise product).
///
/// # Panics
///
/// Panics if the three slices do not have the same length.
///
/// # Examples
///
/// ```
/// let (a, b, mut out) = ([2.0f32, 3.0], [4.0f32, 5.0], [0.0f32; 2]);
/// datalab::kernel::mul(&a, &b, &mut out);
/// assert_eq!(out, [8.0, 15.0]);
/// ```
pub fn mul<T: Element + Mul<Output = T>>(a: &[T], b: &[T], out: &mut [T]) {
    check_lens(a.len(), b.len(), out.len());
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x * y;
    }
}

/// Writes `a[i] * scalar` into `out[i]` for every index.
///
/// # Panics
///
/// Panics if the two slices do not have the same length.
///
/// # Examples
///
/// ```
/// let (a, mut out) = ([1.0f64, -2.0], [0.0f64; 2]);
/// datalab::kernel::mul_scalar(&a, 3.0, &mut out);
/// assert_eq!(out, [3.0, -6.0]);
/// ```
pub fn mul_scalar<T: Element + Mul<Output = T>>(a: &[T], scalar: T, out: &mut [T]) {
    check_lens(a.len(), a.len(), out.len());
    for (o, &x) in out.iter_mut().zip(a) {
        *o = x * scalar;
    }
}

/// Number of independent accumulators in the base case of [`sum`]. Breaking
/// the single dependency chain into 8 lanes is what lets the compiler
/// vectorize the reduction.
const SUM_LANES: usize = 8;

/// Length up to which [`sum`] uses the blocked base case directly; longer
/// inputs are split recursively (pairwise summation).
const SUM_BLOCK: usize = 256;

/// Returns the sum of all elements, starting from `T::default()` (zero for
/// every primitive).
///
/// Uses **pairwise summation** with a multi-accumulator base case: the input
/// is split recursively into halves, and short blocks are reduced with 8
/// independent accumulator lanes. The order of additions is fixed and
/// deterministic, but differs from a strict left-to-right fold — for floats
/// the rounding error grows in `O(log n)` instead of `O(n)`, i.e. this is
/// both faster (vectorizable) and more accurate than the naive loop. For
/// integers the result is identical to the naive loop.
///
/// # Examples
///
/// ```
/// assert_eq!(datalab::kernel::sum(&[1.0f64, 2.5, -0.5]), 3.0);
/// assert_eq!(datalab::kernel::sum::<i32>(&[]), 0);
/// ```
#[must_use]
pub fn sum<T: Element + Add<Output = T> + Default>(a: &[T]) -> T {
    if a.len() <= SUM_BLOCK {
        return sum_block(a);
    }
    // Split on a lane-aligned midpoint so every base-case block (except the
    // final one) is a whole number of lanes.
    let mid = (a.len() / 2).next_multiple_of(SUM_LANES);
    sum(&a[..mid]) + sum(&a[mid..])
}

/// Base case of [`sum`]: reduces a short block with independent accumulator
/// lanes, then combines the lanes and folds the remainder.
fn sum_block<T: Element + Add<Output = T> + Default>(a: &[T]) -> T {
    let mut lanes = [T::default(); SUM_LANES];
    let chunks = a.chunks_exact(SUM_LANES);
    let remainder = chunks.remainder();
    for chunk in chunks {
        for (lane, &x) in lanes.iter_mut().zip(chunk) {
            *lane = *lane + x;
        }
    }
    let total = lanes.into_iter().fold(T::default(), |acc, lane| acc + lane);
    remainder.iter().fold(total, |acc, &x| acc + x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_elementwise() {
        let mut out = [0i64; 3];
        add(&[1, 2, 3], &[10, 20, 30], &mut out);
        assert_eq!(out, [11, 22, 33]);
    }

    #[test]
    fn sub_elementwise() {
        let mut out = [0i64; 2];
        sub(&[10, 20], &[1, 2], &mut out);
        assert_eq!(out, [9, 18]);
    }

    #[test]
    fn mul_elementwise() {
        let mut out = [0i64; 2];
        mul(&[2, 3], &[4, 5], &mut out);
        assert_eq!(out, [8, 15]);
    }

    #[test]
    fn mul_scalar_scales() {
        let mut out = [0.0f64; 3];
        mul_scalar(&[1.0, -2.0, 0.5], 4.0, &mut out);
        assert_eq!(out, [4.0, -8.0, 2.0]);
    }

    #[test]
    fn sum_accumulates() {
        assert_eq!(sum(&[1u32, 2, 3]), 6);
        assert_eq!(sum::<f64>(&[]), 0.0);
    }

    #[test]
    fn sum_handles_every_length_boundary() {
        // Cover: empty, below/at/above the lane count, at/around the block
        // size, and deep into the recursive pairwise path.
        for len in [0, 1, 7, 8, 9, 255, 256, 257, 1000, 4096, 100_000] {
            let data: Vec<u64> = (1..=len as u64).collect();
            let expected = len as u64 * (len as u64 + 1) / 2;
            assert_eq!(sum(&data), expected, "len = {len}");
        }
    }

    #[test]
    fn sum_matches_naive_float_sum_closely() {
        let data: Vec<f64> = (0..10_000).map(|i| (i as f64).sin()).collect();
        let naive: f64 = data.iter().sum();
        let ours = sum(&data);
        assert!((ours - naive).abs() < 1e-9, "ours={ours}, naive={naive}");
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn add_panics_on_length_mismatch() {
        let mut out = [0i32; 2];
        add(&[1, 2, 3], &[1, 2], &mut out);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mul_scalar_panics_on_length_mismatch() {
        let mut out = [0i32; 3];
        mul_scalar(&[1, 2], 5, &mut out);
    }

    #[test]
    fn empty_slices_are_fine() {
        let mut out: [f64; 0] = [];
        add(&[], &[], &mut out);
        assert_eq!(sum::<f64>(&[]), 0.0);
    }
}
