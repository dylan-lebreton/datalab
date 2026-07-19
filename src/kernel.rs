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

/// Returns the sum of all elements, starting from `T::default()` (zero for
/// every primitive).
///
/// The accumulation is strict left-to-right. For floats this is the plain
/// sequential semantics (not pairwise/compensated); fancier reductions can be
/// added later without changing callers.
///
/// # Examples
///
/// ```
/// assert_eq!(datalab::kernel::sum(&[1.0f64, 2.5, -0.5]), 3.0);
/// assert_eq!(datalab::kernel::sum::<i32>(&[]), 0);
/// ```
#[must_use]
pub fn sum<T: Element + Add<Output = T> + Default>(a: &[T]) -> T {
    a.iter().fold(T::default(), |acc, &x| acc + x)
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
