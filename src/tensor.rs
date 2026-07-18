//! Owned, contiguous, typed 1-D tensors.
//!
//! A [`Tensor`] owns a [`Storage`] and interprets it as a fixed-length,
//! contiguous run of elements of type `T`. It is datalab's materialized
//! ("eager") object: all the data is there, in one contiguous allocation,
//! which keeps scans cache-friendly and SIMD-friendly.
//!
//! Tensors have a **fixed length**: values can be mutated in place (through
//! [`Tensor::as_mut_slice`] or [`Tensor::view_mut`]), but a tensor never grows
//! or shrinks. Growable structures and streams of batches are separate,
//! higher-level concerns.
//!
//! This module contains no `unsafe` code: it composes the safe APIs of
//! [`Storage`] and the typed views.

use std::fmt;
use std::marker::PhantomData;

use crate::storage::{STORAGE_ALIGN, Storage};
use crate::view::{Element, View, ViewMut};

/// An owned, contiguous, fixed-length 1-D tensor of elements `T`.
///
/// # Examples
///
/// ```
/// use datalab::tensor::Tensor;
///
/// let mut tensor = Tensor::<f64>::zeros(3);
/// tensor.as_mut_slice()[0] = 1.5;
/// assert_eq!(tensor.as_slice(), &[1.5, 0.0, 0.0]);
/// assert_eq!(tensor.len(), 3);
/// ```
#[derive(Clone)]
pub struct Tensor<T: Element> {
    /// Invariant: the byte length is a multiple of `size_of::<T>()` and the
    /// allocation is aligned for `T`, so the storage is always viewable as
    /// `[T]`. Every constructor goes through [`Tensor::zeros`], which
    /// guarantees both.
    storage: Storage,
    _elem: PhantomData<T>,
}

impl<T: Element> Tensor<T> {
    /// Creates a tensor of `len` elements, all set to the all-zero bit pattern
    /// (numeric zero for every primitive).
    ///
    /// # Panics
    ///
    /// Panics if the size in bytes overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let tensor = Tensor::<i32>::zeros(4);
    /// assert_eq!(tensor.as_slice(), &[0, 0, 0, 0]);
    /// ```
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("tensor size in bytes overflows usize");
        // The default alignment already covers every primitive; the `max`
        // keeps the invariant for exotic user-defined `Element` types.
        let align = STORAGE_ALIGN.max(align_of::<T>());
        Self {
            storage: Storage::zeroed_aligned(bytes, align),
            _elem: PhantomData,
        }
    }

    /// Creates a tensor holding a copy of `elements`.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let tensor = Tensor::from_elements(&[1.5f64, -2.0]);
    /// assert_eq!(tensor.as_slice(), &[1.5, -2.0]);
    /// ```
    #[must_use]
    pub fn from_elements(elements: &[T]) -> Self {
        let mut tensor = Self::zeros(elements.len());
        tensor.as_mut_slice().copy_from_slice(elements);
        tensor
    }

    /// Creates a tensor of `len` elements where element `i` is `f(i)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let squares = Tensor::from_fn(4, |i| (i * i) as u32);
    /// assert_eq!(squares.as_slice(), &[0, 1, 4, 9]);
    /// ```
    #[must_use]
    pub fn from_fn(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
        let mut tensor = Self::zeros(len);
        for (i, slot) in tensor.as_mut_slice().iter_mut().enumerate() {
            *slot = f(i);
        }
        tensor
    }

    /// Returns the number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// assert_eq!(Tensor::<f64>::zeros(5).len(), 5);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.storage.len() / size_of::<T>()
    }

    /// Returns `true` if the tensor holds no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// assert!(Tensor::<f64>::zeros(0).is_empty());
    /// assert!(!Tensor::<f64>::zeros(1).is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Returns the elements as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let tensor = Tensor::from_elements(&[1u8, 2, 3]);
    /// assert_eq!(tensor.as_slice().iter().sum::<u8>(), 6);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.view().into_slice()
    }

    /// Returns the elements as a mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let mut tensor = Tensor::<u8>::zeros(2);
    /// tensor.as_mut_slice()[1] = 7;
    /// assert_eq!(tensor.as_slice(), &[0, 7]);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.view_mut().into_slice_mut()
    }

    /// Returns an immutable typed view of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let tensor = Tensor::from_elements(&[1i64, 2]);
    /// assert_eq!(tensor.view().len(), 2);
    /// ```
    #[inline]
    #[must_use]
    pub fn view(&self) -> View<'_, T> {
        View::new(&self.storage).expect("Tensor invariant: storage is viewable as [T]")
    }

    /// Returns a mutable typed view of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let mut tensor = Tensor::<i64>::zeros(2);
    /// tensor.view_mut()[0] = 5;
    /// assert_eq!(tensor.as_slice(), &[5, 0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn view_mut(&mut self) -> ViewMut<'_, T> {
        ViewMut::new(&mut self.storage).expect("Tensor invariant: storage is viewable as [T]")
    }

    /// Returns a reference to the underlying byte storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let tensor = Tensor::<f64>::zeros(2);
    /// assert_eq!(tensor.storage().len(), 16);
    /// ```
    #[inline]
    #[must_use]
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Consumes the tensor, returning the underlying byte storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let storage = Tensor::from_elements(&[1u8, 2]).into_storage();
    /// assert_eq!(storage.as_bytes(), &[1, 2]);
    /// ```
    #[must_use]
    pub fn into_storage(self) -> Storage {
        self.storage
    }
}

impl<T: Element + PartialEq> PartialEq for Tensor<T> {
    /// Two tensors are equal when their elements are equal (element semantics,
    /// e.g. `NaN != NaN` for floats — same as comparing slices).
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Element + Eq> Eq for Tensor<T> {}

impl<T: Element> Default for Tensor<T> {
    fn default() -> Self {
        Self::zeros(0)
    }
}

impl<T: Element + fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: Element> From<&[T]> for Tensor<T> {
    fn from(elements: &[T]) -> Self {
        Self::from_elements(elements)
    }
}

impl<T: Element> FromIterator<T> for Tensor<T> {
    /// Collects an iterator into a tensor.
    ///
    /// The elements are first gathered into an intermediate `Vec` (the final
    /// length must be known before the aligned allocation is made).
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elements: Vec<T> = iter.into_iter().collect();
        Self::from_elements(&elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_is_all_zero() {
        let tensor = Tensor::<f64>::zeros(4);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.as_slice(), &[0.0; 4]);
    }

    #[test]
    fn from_elements_roundtrips() {
        let tensor = Tensor::from_elements(&[1i32, -2, 3]);
        assert_eq!(tensor.len(), 3);
        assert_eq!(tensor.as_slice(), &[1, -2, 3]);
    }

    #[test]
    fn from_fn_fills_by_index() {
        let tensor = Tensor::from_fn(5, |i| i as u64 * 10);
        assert_eq!(tensor.as_slice(), &[0, 10, 20, 30, 40]);
    }

    #[test]
    fn from_iterator_collects() {
        let tensor: Tensor<i32> = (1..=3).collect();
        assert_eq!(tensor.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn values_are_mutable_in_place() {
        let mut tensor = Tensor::<f32>::zeros(3);
        tensor.as_mut_slice()[1] = 2.5;
        tensor.view_mut()[2] = -1.0;
        assert_eq!(tensor.as_slice(), &[0.0, 2.5, -1.0]);
    }

    #[test]
    fn clone_is_a_deep_copy() {
        let original = Tensor::from_elements(&[1u8, 2]);
        let mut copy = original.clone();
        copy.as_mut_slice()[0] = 9;
        assert_eq!(original.as_slice(), &[1, 2]);
        assert_eq!(copy.as_slice(), &[9, 2]);
    }

    #[test]
    fn equality_uses_element_semantics() {
        assert_eq!(Tensor::from_elements(&[1i32, 2]), Tensor::from_elements(&[1i32, 2]));
        assert_ne!(Tensor::from_elements(&[1i32, 2]), Tensor::from_elements(&[1i32, 3]));
        // NaN != NaN, like slices of floats.
        let nan = Tensor::from_elements(&[f64::NAN]);
        assert_ne!(nan, nan.clone());
    }

    #[test]
    fn empty_tensor() {
        let tensor = Tensor::<f64>::default();
        assert!(tensor.is_empty());
        assert_eq!(tensor.len(), 0);
        assert_eq!(tensor.as_slice(), &[] as &[f64]);
    }

    #[test]
    fn storage_length_is_len_times_element_size() {
        let tensor = Tensor::<f64>::zeros(3);
        assert_eq!(tensor.storage().len(), 3 * size_of::<f64>());
        assert_eq!(tensor.into_storage().len(), 24);
    }

    #[test]
    fn from_slice_conversion() {
        let tensor: Tensor<u16> = (&[7u16, 8][..]).into();
        assert_eq!(tensor.as_slice(), &[7, 8]);
    }

    #[test]
    fn debug_prints_elements() {
        let tensor = Tensor::from_elements(&[1u8, 2]);
        assert_eq!(format!("{tensor:?}"), "[1, 2]");
    }
}
