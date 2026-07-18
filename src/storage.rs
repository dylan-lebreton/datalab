//! Raw, contiguous byte storage — the lowest-level building block.
//!
//! A [`Storage`] owns a contiguous region of bytes with a known length and
//! says nothing about how those bytes are interpreted; interpreting them as a
//! typed, shaped array is the job of a view layered on top.
//!
//! The bytes live in a heap allocation that is **aligned** (see
//! [`STORAGE_ALIGN`]) so they can later be reinterpreted as `f64`/`i32`/… and
//! fed to SIMD instructions without unaligned-access penalties. The alignment
//! is a construction parameter: callers that know their needs can request a
//! specific alignment via [`Storage::zeroed_aligned`], while [`Storage::zeroed`]
//! uses a sensible per-architecture default.
//!
//! For now the bytes always live in RAM. The backing store is meant to become
//! pluggable later — memory-mapped files, spill-to-disk, shared memory —
//! behind this same API, without callers noticing.

use std::alloc::{self, Layout};
use std::fmt;
use std::ptr::NonNull;
use std::slice;

/// Default alignment, in bytes, of a [`Storage`] allocated with
/// [`Storage::zeroed`].
///
/// This is always **at least 64** (enough for every scalar we store and for
/// wide SIMD registers). On `aarch64` (e.g. Apple Silicon) it is 128, matching
/// the cache-line size to avoid false sharing between neighbouring buffers.
///
/// The value is chosen at compile time from the target architecture, so it
/// adapts automatically to the machine without any user action. The table
/// mirrors the one maintained by `crossbeam-utils`' `CachePadded`.
pub const STORAGE_ALIGN: usize = if cfg!(target_arch = "aarch64") {
    128
} else {
    64
};

/// Owns a contiguous region of raw, untyped bytes on an aligned allocation.
///
/// `Storage` is the lowest-level building block of datalab: it holds a
/// contiguous run of bytes with a known length and says nothing about how they
/// are interpreted. Interpreting them as a typed, shaped array is the job of a
/// view layered on top.
///
/// # Examples
///
/// ```
/// use datalab::storage::Storage;
///
/// let mut storage = Storage::zeroed(4);
/// storage.as_bytes_mut()[1] = 42;
/// assert_eq!(storage.as_bytes(), &[0, 42, 0, 0]);
/// ```
pub struct Storage {
    /// Pointer to the first byte. For an empty storage this is a dangling but
    /// well-aligned pointer and no allocation is held.
    ptr: NonNull<u8>,
    /// Number of bytes.
    len: usize,
    /// Alignment the allocation was created with; needed to free it correctly.
    align: usize,
}

// SAFETY: `Storage` uniquely owns its heap allocation and exposes no interior
// mutability, so — exactly like `Vec<u8>` — it is safe to move it across
// threads and to share `&Storage` between threads.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    /// Creates a storage of `len` zero-initialized bytes, using the default
    /// alignment [`STORAGE_ALIGN`].
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::zeroed(3);
    /// assert_eq!(storage.as_bytes(), &[0, 0, 0]);
    /// ```
    pub fn zeroed(len: usize) -> Self {
        Self::zeroed_aligned(len, STORAGE_ALIGN)
    }

    /// Creates a storage of `len` zero-initialized bytes with a caller-chosen
    /// `align`.
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two, or if the total allocation size
    /// would overflow `isize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::zeroed_aligned(16, 32);
    /// assert_eq!(storage.alignment(), 32);
    /// assert_eq!(storage.as_bytes().as_ptr() as usize % 32, 0);
    /// ```
    pub fn zeroed_aligned(len: usize, align: usize) -> Self {
        assert!(
            align.is_power_of_two(),
            "alignment must be a power of two, got {align}"
        );

        if len == 0 {
            // No allocation for an empty storage: a dangling (but non-null)
            // pointer is a valid base for a zero-length slice.
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                align,
            };
        }

        let layout = Layout::from_size_align(len, align).expect("allocation size overflow");
        // SAFETY: `layout` has a non-zero size (checked `len != 0` above).
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self { ptr, len, align }
    }

    /// Creates a storage holding a copy of `bytes`, using the default alignment.
    ///
    /// The bytes are copied into a freshly aligned allocation, so the result is
    /// aligned to [`STORAGE_ALIGN`] regardless of the source's alignment.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::from_bytes(&[1, 2, 3]);
    /// assert_eq!(storage.len(), 3);
    /// assert_eq!(storage.as_bytes(), &[1, 2, 3]);
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut storage = Self::zeroed(bytes.len());
        storage.as_bytes_mut().copy_from_slice(bytes);
        storage
    }

    /// Returns the number of bytes held by this storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// assert_eq!(Storage::zeroed(8).len(), 8);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the storage holds no bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// assert!(Storage::zeroed(0).is_empty());
    /// assert!(!Storage::zeroed(1).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the alignment, in bytes, of the underlying allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::{Storage, STORAGE_ALIGN};
    ///
    /// assert_eq!(Storage::zeroed(8).alignment(), STORAGE_ALIGN);
    /// ```
    pub fn alignment(&self) -> usize {
        self.align
    }

    /// Returns an immutable view of the raw bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::from_bytes(&[1, 2, 3]);
    /// assert_eq!(storage.as_bytes(), &[1, 2, 3]);
    /// ```
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `ptr` points to `len` contiguous, initialized bytes (zeroed at
        // allocation) that stay valid and immutable for the lifetime of `&self`.
        // For `len == 0`, a dangling pointer is a valid base for an empty slice.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable view of the raw bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let mut storage = Storage::zeroed(2);
    /// storage.as_bytes_mut()[0] = 7;
    /// assert_eq!(storage.as_bytes(), &[7, 0]);
    /// ```
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` points to `len` contiguous, initialized bytes that stay
        // valid for the lifetime of `&mut self`, and `&mut self` guarantees
        // exclusive access. For `len == 0`, a dangling pointer is valid.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.len != 0 {
            let layout =
                Layout::from_size_align(self.len, self.align).expect("layout was valid on alloc");
            // SAFETY: `ptr` was allocated in `zeroed_aligned` with exactly this
            // layout (same non-zero `len` and `align`) and is freed only once.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        let mut cloned = Self::zeroed_aligned(self.len, self.align);
        cloned.as_bytes_mut().copy_from_slice(self.as_bytes());
        cloned
    }
}

impl PartialEq for Storage {
    /// Two storages are equal when they hold the same bytes; alignment is not
    /// compared.
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for Storage {}

impl Default for Storage {
    fn default() -> Self {
        Self::zeroed(0)
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Storage")
            .field("len", &self.len)
            .field("align", &self.align)
            .finish()
    }
}

impl From<&[u8]> for Storage {
    fn from(bytes: &[u8]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl AsRef<[u8]> for Storage {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_allocates_len_zero_bytes() {
        let storage = Storage::zeroed(8);
        assert_eq!(storage.len(), 8);
        assert!(!storage.is_empty());
        assert!(storage.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn from_bytes_preserves_content() {
        let storage = Storage::from_bytes(&[1, 2, 3]);
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn empty_storage_is_empty() {
        let storage = Storage::zeroed(0);
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert_eq!(storage.as_bytes(), &[] as &[u8]);
    }

    #[test]
    fn as_bytes_mut_allows_in_place_edits() {
        let mut storage = Storage::zeroed(4);
        storage.as_bytes_mut()[1] = 42;
        assert_eq!(storage.as_bytes(), &[0, 42, 0, 0]);
    }

    #[test]
    fn default_alignment_is_respected() {
        let storage = Storage::zeroed(128);
        assert_eq!(storage.alignment(), STORAGE_ALIGN);
        assert_eq!(storage.as_bytes().as_ptr() as usize % STORAGE_ALIGN, 0);
    }

    #[test]
    fn requested_alignment_is_respected() {
        let storage = Storage::zeroed_aligned(64, 32);
        assert_eq!(storage.alignment(), 32);
        assert_eq!(storage.as_bytes().as_ptr() as usize % 32, 0);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn non_power_of_two_alignment_panics() {
        let _ = Storage::zeroed_aligned(8, 3);
    }

    #[test]
    fn clone_is_a_deep_copy() {
        let original = Storage::from_bytes(&[9, 8, 7]);
        let mut copy = original.clone();
        assert_eq!(copy, original);
        copy.as_bytes_mut()[0] = 0;
        assert_ne!(copy, original);
        assert_eq!(original.as_bytes(), &[9, 8, 7]);
    }

    #[test]
    fn equality_compares_bytes() {
        assert_eq!(Storage::from_bytes(&[1, 2]), Storage::from_bytes(&[1, 2]));
        assert_ne!(Storage::from_bytes(&[1, 2]), Storage::from_bytes(&[1, 3]));
    }

    #[test]
    fn from_slice_conversion_matches_from_bytes() {
        let storage: Storage = (&[4, 5, 6][..]).into();
        assert_eq!(storage, Storage::from_bytes(&[4, 5, 6]));
    }

    #[test]
    fn as_ref_exposes_bytes() {
        let storage = Storage::from_bytes(&[4, 5]);
        let slice: &[u8] = storage.as_ref();
        assert_eq!(slice, &[4, 5]);
    }

    #[test]
    fn default_is_empty() {
        assert!(Storage::default().is_empty());
    }

    #[test]
    fn storage_align_covers_every_scalar() {
        assert!(STORAGE_ALIGN.is_power_of_two());
        assert!(STORAGE_ALIGN >= 64);
        assert!(STORAGE_ALIGN >= std::mem::align_of::<u64>());
        assert!(STORAGE_ALIGN >= std::mem::align_of::<f64>());
        assert!(STORAGE_ALIGN >= std::mem::align_of::<u128>());
    }
}
