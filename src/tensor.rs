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

use std::error::Error;
use std::fmt;
use std::io;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
use std::path::Path;

use crate::kernel;
use crate::storage::{STORAGE_ALIGN, Storage};
use crate::view::{Element, View, ViewError, ViewMut};

/// The reason a tensor could not be created from a file.
#[derive(Debug)]
pub enum TensorFileError {
    /// Opening or mapping the file failed.
    Io(io::Error),
    /// The file's bytes cannot be viewed as the requested element type (e.g.
    /// its size is not a whole number of elements).
    View(ViewError),
}

impl fmt::Display for TensorFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "cannot map file: {err}"),
            Self::View(err) => write!(f, "file bytes do not form a tensor: {err}"),
        }
    }
}

impl Error for TensorFileError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::View(err) => Some(err),
        }
    }
}

impl From<io::Error> for TensorFileError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<ViewError> for TensorFileError {
    fn from(err: ViewError) -> Self {
        Self::View(err)
    }
}

/// An owned, contiguous, fixed-length 1-D tensor of elements `T`.
///
/// Cloning is **zero-copy** (the clone shares the bytes, Arrow-style): while
/// shared, neither tensor is writable in place — mutate through the explicit
/// [`Tensor::make_mut`], which copies on write. See
/// [`Storage`](crate::storage) for the sharing model.
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
    /// bytes are aligned for `T`, so the storage is always viewable as `[T]`.
    /// In-memory constructors go through [`Tensor::zeros`] which guarantees
    /// both; [`Tensor::map_file`] validates the invariant before accepting a
    /// file (mappings are page-aligned).
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

    /// Wraps an existing storage without copying.
    ///
    /// Internal: callers must provide a storage upholding the invariant
    /// (whole number of `T`-aligned elements), which is re-checked here.
    pub(crate) fn from_storage(storage: Storage) -> Self {
        View::<T>::new(&storage).expect("storage must be viewable as [T]");
        Self {
            storage,
            _elem: PhantomData,
        }
    }

    /// Opens `path` as a **read-only**, disk-resident tensor.
    ///
    /// The file is memory-mapped, not read up front: the OS loads pages on
    /// first access and evicts them under memory pressure, so the tensor may
    /// be far larger than the available RAM. Reads (`as_slice`, `view`,
    /// kernels, operators) work as usual; mutation requires an explicit
    /// [`Tensor::make_mut`] first (see [`Tensor::is_writable`]).
    ///
    /// See [`Storage::map_file`] for the file-stability contract.
    ///
    /// # Errors
    ///
    /// Returns [`TensorFileError::Io`] if the file cannot be opened or
    /// mapped, and [`TensorFileError::View`] if its byte length is not a
    /// whole number of `T` elements.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use datalab::tensor::Tensor;
    ///
    /// // A 40 GB weight file on a 16 GB machine: opens instantly.
    /// let weights = Tensor::<f32>::map_file("model-weights.bin")?;
    /// let norm = weights.map(|w| w * w).sum();
    /// # Ok::<(), datalab::tensor::TensorFileError>(())
    /// ```
    pub fn map_file(path: impl AsRef<Path>) -> Result<Self, TensorFileError> {
        let storage = Storage::map_file(path)?;
        // Validate the invariant up front (mappings are page-aligned, so in
        // practice only the size check can fail).
        View::<T>::new(&storage)?;
        Ok(Self {
            storage,
            _elem: PhantomData,
        })
    }

    /// Moves the elements into a temporary file inside `dir`, releasing the
    /// tensor's RAM while keeping it readable (memory-mapped).
    ///
    /// The tensor becomes read-only; promote it back with
    /// [`Tensor::make_mut`]. See [`Storage::spill_to_disk`] for details.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from writing or mapping the temporary file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use datalab::tensor::Tensor;
    ///
    /// let mut batch = Tensor::from_elements(&[1.0f64, 2.0]);
    /// batch.spill_to_disk(std::env::temp_dir())?; // RAM released
    /// assert_eq!(batch.sum(), 3.0);               // still readable
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn spill_to_disk(&mut self, dir: impl AsRef<Path>) -> io::Result<()> {
        self.storage.spill_to_disk(dir)
    }

    /// Returns `true` if the elements can be mutated in place.
    ///
    /// Tensors built in memory are writable while uniquely owned; tensors
    /// backed by a file ([`Tensor::map_file`], [`Tensor::spill_to_disk`]) or
    /// currently **shared** (cloned zero-copy) are not, until promoted with
    /// [`Tensor::make_mut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// assert!(Tensor::<f64>::zeros(2).is_writable());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_writable(&self) -> bool {
        self.storage.is_writable()
    }

    /// Ensures the tensor is uniquely writable, then returns its mutable
    /// elements.
    ///
    /// If the tensor is file-backed or shared, its bytes are first
    /// **copied** into a fresh heap allocation — an explicit `O(len)`
    /// copy-on-write (sharers keep the original). Uniquely-owned writable
    /// tensors are returned as-is. The copy is aligned for `T` (and to the
    /// default storage alignment), so the tensor invariant survives the
    /// promotion whatever the element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let mut tensor = Tensor::from_elements(&[1u32, 2]);
    /// tensor.make_mut()[0] = 9;
    /// assert_eq!(tensor.as_slice(), &[9, 2]);
    /// ```
    pub fn make_mut(&mut self) -> &mut [T] {
        if !self.is_writable() {
            // The untyped storage cannot know `T`'s alignment; realign the
            // copy here, exactly as `Tensor::zeros` does.
            self.storage = Storage::from_bytes_aligned(
                self.storage.as_bytes(),
                STORAGE_ALIGN.max(align_of::<T>()),
            );
        }
        self.as_mut_slice()
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
    /// # Panics
    ///
    /// Panics if the tensor is not writable in place (file-backed or
    /// shared): check [`Tensor::is_writable`] or promote with
    /// [`Tensor::make_mut`] first.
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
    /// # Panics
    ///
    /// Panics if the tensor is not writable in place (file-backed or
    /// shared): check [`Tensor::is_writable`] or promote with
    /// [`Tensor::make_mut`] first.
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
        match ViewMut::new(&mut self.storage) {
            Ok(view) => view,
            Err(ViewError::ReadOnly) => {
                panic!(
                    "tensor is not writable in place (file-backed or shared); \
                     promote it with make_mut()"
                )
            }
            Err(_) => unreachable!("Tensor invariant: storage is viewable as [T]"),
        }
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

    /// Returns the single element of a one-element tensor.
    ///
    /// The typical way to read a lazy reduction's result:
    /// `plan.sum().collect()?.item()` (mirrors Polars' `item()`).
    ///
    /// # Panics
    ///
    /// Panics if the tensor does not hold exactly one element.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// assert_eq!(Tensor::from_elements(&[42i32]).item(), 42);
    /// ```
    #[must_use]
    pub fn item(&self) -> T {
        assert_eq!(
            self.len(),
            1,
            "item() requires a tensor of exactly one element, got {}",
            self.len()
        );
        self.as_slice()[0]
    }

    /// Returns the sum of all elements (zero for an empty tensor).
    ///
    /// Uses [`kernel::sum`]: deterministic pairwise summation — vectorizable,
    /// and more accurate than a naive left-to-right fold for floats.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let tensor = Tensor::from_elements(&[1.0f64, 2.5, -0.5]);
    /// assert_eq!(tensor.sum(), 3.0);
    /// ```
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: Add<Output = T> + Default,
    {
        kernel::sum(self.as_slice())
    }

    /// Returns a new tensor where element `i` is `f(self[i])`.
    ///
    /// The output element type may differ from the input's.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let lengths = Tensor::from_elements(&[1.5f64, -2.0]);
    /// let rounded: Tensor<i64> = lengths.map(|x| x.round() as i64);
    /// assert_eq!(rounded.as_slice(), &[2, -2]);
    /// ```
    #[must_use]
    pub fn map<U: Element>(&self, mut f: impl FnMut(T) -> U) -> Tensor<U> {
        let mut out = Tensor::<U>::zeros(self.len());
        for (o, &x) in out.as_mut_slice().iter_mut().zip(self.as_slice()) {
            *o = f(x);
        }
        out
    }
}

/// Element-wise addition: `&a + &b`.
///
/// # Panics
///
/// Panics if the tensors do not have the same length.
impl<T: Element + Add<Output = T>> Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Tensor<T> {
        let mut out = Tensor::zeros(self.len());
        kernel::add(self.as_slice(), rhs.as_slice(), out.as_mut_slice());
        out
    }
}

/// Element-wise subtraction: `&a - &b`.
///
/// # Panics
///
/// Panics if the tensors do not have the same length.
impl<T: Element + Sub<Output = T>> Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Tensor<T> {
        let mut out = Tensor::zeros(self.len());
        kernel::sub(self.as_slice(), rhs.as_slice(), out.as_mut_slice());
        out
    }
}

/// Element-wise product: `&a * &b`.
///
/// # Panics
///
/// Panics if the tensors do not have the same length.
impl<T: Element + Mul<Output = T>> Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Tensor<T> {
        let mut out = Tensor::zeros(self.len());
        kernel::mul(self.as_slice(), rhs.as_slice(), out.as_mut_slice());
        out
    }
}

/// Scalar multiplication: `&a * scalar`.
impl<T: Element + Mul<Output = T>> Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, scalar: T) -> Tensor<T> {
        let mut out = Tensor::zeros(self.len());
        kernel::mul_scalar(self.as_slice(), scalar, out.as_mut_slice());
        out
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
    fn clone_is_zero_copy_with_copy_on_write() {
        let original = Tensor::from_elements(&[1u8, 2]);
        let mut copy = original.clone();
        // Shared: same bytes, no in-place mutation.
        assert_eq!(copy.as_slice().as_ptr(), original.as_slice().as_ptr());
        assert!(!copy.is_writable());
        // Explicit copy-on-write diverges the clone, sharers keep theirs.
        copy.make_mut()[0] = 9;
        assert_eq!(original.as_slice(), &[1, 2]);
        assert_eq!(copy.as_slice(), &[9, 2]);
        assert!(original.is_writable()); // unique again
    }

    #[test]
    #[should_panic(expected = "not writable in place")]
    fn mutating_a_shared_tensor_panics_with_a_clear_message() {
        let mut tensor = Tensor::from_elements(&[1u8, 2]);
        let _shared = tensor.clone();
        let _ = tensor.as_mut_slice();
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

    #[test]
    fn elementwise_operators() {
        let a = Tensor::from_elements(&[1.0f64, 2.0, 3.0]);
        let b = Tensor::from_elements(&[10.0f64, 20.0, 30.0]);
        assert_eq!((&a + &b).as_slice(), &[11.0, 22.0, 33.0]);
        assert_eq!((&b - &a).as_slice(), &[9.0, 18.0, 27.0]);
        assert_eq!((&a * &b).as_slice(), &[10.0, 40.0, 90.0]);
        assert_eq!((&a * 2.0).as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn add_panics_on_length_mismatch() {
        let a = Tensor::from_elements(&[1i32, 2]);
        let b = Tensor::from_elements(&[1i32, 2, 3]);
        let _ = &a + &b;
    }

    #[test]
    fn sum_and_map() {
        let tensor = Tensor::from_elements(&[1i64, -2, 3]);
        assert_eq!(tensor.sum(), 2);
        let doubled = tensor.map(|x| x * 2);
        assert_eq!(doubled.as_slice(), &[2, -4, 6]);
        let as_f64: Tensor<f64> = tensor.map(|x| x as f64);
        assert_eq!(as_f64.as_slice(), &[1.0, -2.0, 3.0]);
    }

    /// Creates a unique temp-file path for file-based tests.
    fn temp_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "datalab-tensor-test-{tag}-{}-{id}",
            std::process::id()
        ))
    }

    #[test]
    #[cfg_attr(miri, ignore)] // file-backed mmap is not supported under miri
    fn map_file_reads_tensor_from_disk() {
        let source = Tensor::from_elements(&[1.5f64, -2.0, 4.5]);
        let path = temp_path("map");
        std::fs::write(&path, source.storage().as_bytes()).unwrap();

        let mapped = Tensor::<f64>::map_file(&path).unwrap();
        assert_eq!(mapped.as_slice(), &[1.5, -2.0, 4.5]);
        assert_eq!(mapped.sum(), 4.0);
        assert!(!mapped.is_writable());
        drop(mapped);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn map_file_rejects_partial_elements() {
        let path = temp_path("badsize");
        std::fs::write(&path, [0u8; 5]).unwrap(); // not a multiple of 8
        let err = Tensor::<f64>::map_file(&path).unwrap_err();
        assert!(matches!(err, TensorFileError::View(_)));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access is blocked by miri isolation
    fn map_file_of_missing_file_is_io_error() {
        let err = Tensor::<f64>::map_file(temp_path("missing")).unwrap_err();
        assert!(matches!(err, TensorFileError::Io(_)));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn spill_keeps_tensor_readable_and_make_mut_promotes() {
        let mut tensor = Tensor::from_elements(&[1u32, 2, 3]);
        tensor.spill_to_disk(std::env::temp_dir()).unwrap();
        assert!(!tensor.is_writable());
        assert_eq!(tensor.as_slice(), &[1, 2, 3]);
        assert_eq!(tensor.sum(), 6);

        tensor.make_mut()[0] = 9;
        assert!(tensor.is_writable());
        assert_eq!(tensor.as_slice(), &[9, 2, 3]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[should_panic(expected = "not writable in place")]
    fn mutating_a_spilled_tensor_panics_with_a_clear_message() {
        let mut tensor = Tensor::from_elements(&[1u32, 2]);
        tensor.spill_to_disk(std::env::temp_dir()).unwrap();
        let _ = tensor.as_mut_slice();
    }

    #[test]
    fn tensor_file_error_displays() {
        let err = TensorFileError::View(crate::view::ViewError::ReadOnly);
        assert!(err.to_string().contains("tensor"));
        assert!(std::error::Error::source(&err).is_some());
    }
}
