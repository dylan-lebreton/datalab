//! Typed views over raw [`Storage`] bytes.
//!
//! A view is a non-owning, typed lens: it reinterprets the raw bytes of a
//! [`Storage`] as a slice of some element type `T` (an [`Element`]), so callers
//! can read and write numbers instead of bytes. Creating a view is checked and
//! zero-copy: it borrows the storage and reinterprets its bytes in place.
//!
//! [`View`] gives read-only access; [`ViewMut`] also allows writing. Both
//! dereference to `[T]`, so the whole slice API (indexing, iteration, `len`, …)
//! is available.
//!
//! [`Storage`]: crate::storage::Storage

use std::error::Error;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::slice;

use crate::storage::Storage;

/// Types that can be safely reinterpreted from raw bytes.
///
/// Elements are plain data through and through: the `Copy + Send + Sync +
/// 'static` supertraits mean an element owns no resources and can freely
/// cross thread boundaries — which the engine relies on to move batches
/// around (and, later, to execute plans in parallel).
///
/// # Safety
///
/// Implementing this trait asserts that the type is `Copy`, has a non-zero
/// size, has no padding bytes, and that **every** bit pattern is a valid value
/// — so reinterpreting arbitrary bytes as `Self` can never produce an invalid
/// value. This holds for the integer and floating-point primitives, but not
/// for types with invalid bit patterns such as `bool` or `char`, nor for
/// zero-sized types.
pub unsafe trait Element: Copy + Send + Sync + 'static {}

macro_rules! impl_element {
    ($($t:ty),* $(,)?) => {
        $(
            // SAFETY: every bit pattern is a valid value for this primitive, it
            // is `Copy`, and it has no padding.
            unsafe impl Element for $t {}
        )*
    };
}

impl_element!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);

impl Storage {
    /// Creates a storage holding a copy of `elements`, laid out contiguously
    /// with the default alignment.
    ///
    /// This is the typed counterpart of [`Storage::from_bytes`]: the resulting
    /// storage is always viewable as `[T]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    /// use datalab::view::View;
    ///
    /// let storage = Storage::from_elements(&[1.5f64, -2.0]);
    /// let view = View::<f64>::new(&storage).unwrap();
    /// assert_eq!(&*view, &[1.5, -2.0]);
    /// ```
    #[must_use]
    pub fn from_elements<T: Element>(elements: &[T]) -> Self {
        // SAFETY: `T: Element` guarantees no padding, so every byte of the
        // slice is initialized; `u8` has alignment 1, and the length in bytes
        // of an existing slice cannot overflow `isize`.
        let bytes =
            unsafe { slice::from_raw_parts(elements.as_ptr().cast::<u8>(), size_of_val(elements)) };
        Self::from_bytes(bytes)
    }
}

/// The reason a typed view could not be created over a [`Storage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewError {
    /// The byte length is not a whole multiple of the element size.
    SizeMismatch {
        /// Number of bytes held by the storage.
        byte_len: usize,
        /// Size, in bytes, of one element.
        element_size: usize,
    },
    /// The storage is not aligned strongly enough for the element type.
    Misaligned {
        /// Alignment, in bytes, required by the element type.
        required: usize,
    },
    /// The storage is not writable in place (memory-mapped, or shared by a
    /// clone/slice), so no mutable view can be created. Promote it first
    /// with [`Storage::make_mut`](crate::storage::Storage::make_mut).
    ReadOnly,
}

impl fmt::Display for ViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeMismatch {
                byte_len,
                element_size,
            } => write!(
                f,
                "byte length {byte_len} is not a multiple of element size {element_size}"
            ),
            Self::Misaligned { required } => {
                write!(f, "storage is not aligned to {required} bytes")
            }
            Self::ReadOnly => {
                write!(
                    f,
                    "storage is read-only (memory-mapped or shared); promote it with Storage::make_mut"
                )
            }
        }
    }
}

impl Error for ViewError {}

/// Validates that `bytes` can be reinterpreted as `[T]` and returns the element
/// count.
fn checked_len<T: Element>(bytes: &[u8]) -> Result<usize, ViewError> {
    // Backstop for the `Element` contract: a zero-sized type would divide by
    // zero below, so reject it at compile time instead.
    const { assert!(size_of::<T>() != 0, "Element types must not be zero-sized") };
    let element_size = size_of::<T>();
    if !bytes.len().is_multiple_of(element_size) {
        return Err(ViewError::SizeMismatch {
            byte_len: bytes.len(),
            element_size,
        });
    }
    // Alignment only matters when there is data to point at.
    if !bytes.is_empty() && !(bytes.as_ptr() as usize).is_multiple_of(align_of::<T>()) {
        return Err(ViewError::Misaligned {
            required: align_of::<T>(),
        });
    }
    Ok(bytes.len() / element_size)
}

/// An immutable, typed view over a [`Storage`]'s bytes.
///
/// Dereferences to `[T]`, so every slice method is available.
///
/// # Examples
///
/// ```
/// use datalab::storage::Storage;
/// use datalab::view::View;
///
/// // 16 bytes = two f64 values, zero-initialized.
/// let storage = Storage::zeroed(16);
/// let view = View::<f64>::new(&storage).unwrap();
/// assert_eq!(view.len(), 2);
/// assert_eq!(view[0], 0.0);
/// ```
#[derive(Clone, Copy)]
pub struct View<'a, T: Element> {
    data: &'a [T],
}

impl<'a, T: Element> View<'a, T> {
    /// Creates a typed view over `storage`.
    ///
    /// # Errors
    ///
    /// Returns [`ViewError`] if the storage's byte length is not a multiple of
    /// `size_of::<T>()`, or if it is not aligned for `T`.
    pub fn new(storage: &'a Storage) -> Result<Self, ViewError> {
        let bytes = storage.as_bytes();
        let len = checked_len::<T>(bytes)?;
        let data = if len == 0 {
            &[]
        } else {
            // SAFETY: `checked_len` verified the length is a multiple of the
            // element size and the base pointer is aligned for `T`; `T: Element`
            // guarantees any bit pattern is valid; the bytes are initialized and
            // stay valid and immutable for `'a` through the borrow of `storage`.
            unsafe { slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len) }
        };
        Ok(Self { data })
    }

    /// Consumes the view, returning the underlying slice for the storage's
    /// full borrow lifetime `'a` (longer than a `Deref` borrow of the view).
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    /// use datalab::view::View;
    ///
    /// let storage = Storage::from_elements(&[1u32, 2]);
    /// let slice: &[u32] = View::<u32>::new(&storage).unwrap().into_slice();
    /// assert_eq!(slice, &[1, 2]);
    /// ```
    #[inline]
    #[must_use]
    pub fn into_slice(self) -> &'a [T] {
        self.data
    }
}

impl<T: Element> Deref for View<'_, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data
    }
}

impl<T: Element + fmt::Debug> fmt::Debug for View<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.data.iter()).finish()
    }
}

/// A mutable, typed view over a [`Storage`]'s bytes.
///
/// Dereferences to `[T]` (mutably too), so every slice method is available.
///
/// # Examples
///
/// ```
/// use datalab::storage::Storage;
/// use datalab::view::ViewMut;
///
/// let mut storage = Storage::zeroed(16); // two f64
/// let mut view = ViewMut::<f64>::new(&mut storage).unwrap();
/// view[0] = 1.5;
/// view[1] = -2.0;
/// assert_eq!(&*view, &[1.5, -2.0]);
/// ```
pub struct ViewMut<'a, T: Element> {
    data: &'a mut [T],
}

impl<'a, T: Element> ViewMut<'a, T> {
    /// Creates a mutable typed view over `storage`.
    ///
    /// # Errors
    ///
    /// Returns [`ViewError`] if the storage's byte length is not a multiple of
    /// `size_of::<T>()`, if it is not aligned for `T`, or if it is read-only
    /// ([`ViewError::ReadOnly`], e.g. memory-mapped or shared — even for an
    /// empty storage).
    pub fn new(storage: &'a mut Storage) -> Result<Self, ViewError> {
        let len = checked_len::<T>(storage.as_bytes())?;
        // Even a zero-length mutable view requires a writable storage: the
        // read-only gate must not depend on the length.
        let bytes = storage.as_bytes_mut().ok_or(ViewError::ReadOnly)?;
        let data = if len == 0 {
            &mut []
        } else {
            // SAFETY: `checked_len` verified length and alignment; `T: Element`
            // makes any bit pattern valid; `&mut storage` guarantees exclusive
            // access to initialized bytes for `'a`.
            unsafe { slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), len) }
        };
        Ok(Self { data })
    }

    /// Consumes the view, returning the underlying mutable slice for the
    /// storage's full borrow lifetime `'a`.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    /// use datalab::view::ViewMut;
    ///
    /// let mut storage = Storage::from_elements(&[1u32, 2]);
    /// let slice: &mut [u32] = ViewMut::<u32>::new(&mut storage).unwrap().into_slice_mut();
    /// slice[0] = 9;
    /// assert_eq!(slice, &[9, 2]);
    /// ```
    #[inline]
    #[must_use]
    pub fn into_slice_mut(self) -> &'a mut [T] {
        self.data
    }
}

impl<T: Element> Deref for ViewMut<'_, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.data
    }
}

impl<T: Element> DerefMut for ViewMut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.data
    }
}

impl<T: Element + fmt::Debug> fmt::Debug for ViewMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.data.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_storage_reads_as_zeros() {
        let storage = Storage::zeroed(16);
        let view = View::<f64>::new(&storage).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(&*view, &[0.0, 0.0]);
    }

    #[test]
    fn write_then_read_roundtrips() {
        let mut storage = Storage::zeroed(16);
        {
            let mut view = ViewMut::<f64>::new(&mut storage).unwrap();
            view[0] = 1.5;
            view[1] = -2.0;
        }
        let view = View::<f64>::new(&storage).unwrap();
        assert_eq!(&*view, &[1.5, -2.0]);
    }

    #[test]
    fn element_count_depends_on_type_size() {
        let storage = Storage::zeroed(12);
        assert_eq!(View::<u8>::new(&storage).unwrap().len(), 12);
        assert_eq!(View::<i32>::new(&storage).unwrap().len(), 3);
    }

    #[test]
    fn slice_api_is_available_through_deref() {
        let mut storage = Storage::zeroed(16);
        {
            let mut view = ViewMut::<i32>::new(&mut storage).unwrap();
            for (i, slot) in view.iter_mut().enumerate() {
                *slot = i as i32;
            }
        }
        let view = View::<i32>::new(&storage).unwrap();
        assert_eq!(view.iter().sum::<i32>(), (0..4).sum::<i32>());
        assert_eq!(view.get(2), Some(&2));
    }

    #[test]
    fn size_mismatch_is_reported() {
        let storage = Storage::from_bytes(&[0; 5]);
        let err = View::<f64>::new(&storage).unwrap_err();
        assert_eq!(
            err,
            ViewError::SizeMismatch {
                byte_len: 5,
                element_size: 8,
            }
        );
    }

    #[test]
    fn empty_storage_yields_empty_view() {
        let storage = Storage::zeroed(0);
        let view = View::<f64>::new(&storage).unwrap();
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
    }

    #[test]
    fn from_elements_roundtrips_through_a_view() {
        let storage = Storage::from_elements(&[1i32, -2, 3]);
        assert_eq!(storage.len(), 3 * size_of::<i32>());
        let view = View::<i32>::new(&storage).unwrap();
        assert_eq!(&*view, &[1, -2, 3]);
    }

    #[test]
    fn from_elements_of_empty_slice_is_empty() {
        let storage = Storage::from_elements::<f64>(&[]);
        assert!(storage.is_empty());
    }

    #[test]
    fn view_is_copyable() {
        let storage = Storage::from_elements(&[1u8, 2]);
        let view = View::<u8>::new(&storage).unwrap();
        let copy = view;
        assert_eq!(&*view, &*copy);
    }

    #[test]
    fn error_displays_a_message() {
        let err = ViewError::SizeMismatch {
            byte_len: 5,
            element_size: 8,
        };
        assert!(err.to_string().contains("not a multiple"));
        assert!(ViewError::ReadOnly.to_string().contains("read-only"));
    }

    #[test]
    fn mutable_view_over_empty_shared_storage_is_refused() {
        // The read-only gate holds even when there is nothing to mutate.
        let mut storage = Storage::zeroed(0);
        let _shared = storage.clone();
        assert_eq!(
            ViewMut::<f64>::new(&mut storage).unwrap_err(),
            ViewError::ReadOnly
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // file-backed mmap is not supported under miri
    fn mutable_view_over_read_only_storage_is_refused() {
        let mut storage = Storage::from_elements(&[1.0f64, 2.0]);
        storage.spill_to_disk(std::env::temp_dir()).unwrap();
        assert_eq!(ViewMut::<f64>::new(&mut storage).unwrap_err(), ViewError::ReadOnly);
        // Read views still work fine.
        assert_eq!(&*View::<f64>::new(&storage).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn end_to_end_file_to_typed_view() {
        // The core out-of-core promise, in miniature: bytes on disk, viewed
        // as typed numbers with zero copy, reduced by a kernel.
        let elements: Vec<f64> = (0..1000).map(f64::from).collect();
        let mut storage = Storage::from_elements(&elements);
        storage.spill_to_disk(std::env::temp_dir()).unwrap();

        let view = View::<f64>::new(&storage).unwrap();
        assert_eq!(view.len(), 1000);
        assert_eq!(crate::kernel::sum(&view), 999.0 * 1000.0 / 2.0);
    }
}
