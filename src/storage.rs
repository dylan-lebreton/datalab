//! Raw, contiguous byte storage — the lowest-level building block.
//!
//! A [`Storage`] owns a contiguous region of bytes with a known length and
//! says nothing about how those bytes are interpreted; interpreting them as a
//! typed, shaped array is the job of a view layered on top.
//!
//! # Backings
//!
//! Where the bytes live is pluggable behind this one API:
//!
//! - **Heap** — an owned, writable allocation in RAM, **aligned** (see
//!   [`STORAGE_ALIGN`]) so the bytes can be reinterpreted as `f64`/`i32`/…
//!   and fed to SIMD instructions without unaligned-access penalties.
//! - **Memory-mapped file** — a read-only window over a file on disk
//!   ([`Storage::map_file`]). Pages are loaded lazily by the OS on first
//!   access and evicted under memory pressure, so storages far larger than
//!   RAM stay usable with bounded memory.
//!
//! [`Storage::spill_to_disk`] moves a heap storage's bytes into a temporary
//! memory-mapped file, releasing the RAM while keeping the bytes readable —
//! the primitive a memory-budgeted engine needs. [`Storage::make_mut`] goes
//! the other way, promoting a read-only mapped storage back to a writable
//! heap copy. Both directions are explicit: nothing here copies silently.

use std::alloc::{self, Layout};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::process;
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;

/// Default alignment, in bytes, of a [`Storage`] allocated with
/// [`Storage::zeroed`].
///
/// This is always **at least 64**: enough for every scalar we store, for the
/// widest SIMD registers (AVX-512), for the x86-64 cache line, and matching
/// the 64-byte convention used by Apache Arrow. On `aarch64` (e.g. Apple
/// Silicon) it is 128 — the cache-line size there — so neighbouring buffers
/// never share a line (false sharing).
///
/// The value is chosen at compile time from the target architecture, so it
/// adapts automatically to the machine without any user action.
pub const STORAGE_ALIGN: usize = if cfg!(target_arch = "aarch64") {
    128
} else {
    64
};

/// Counter making spill file names unique within the process.
static SPILL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Where a [`Storage`]'s bytes physically live.
enum Backing {
    /// Owned, writable, aligned heap allocation.
    Heap {
        /// Pointer to the first byte. For an empty storage this is a dangling
        /// but well-aligned pointer and no allocation is held.
        ptr: NonNull<u8>,
        /// Number of bytes.
        len: usize,
        /// Alignment the allocation was created with; needed to free it.
        align: usize,
    },
    /// Read-only, page-aligned memory-mapped file.
    Mmap(Mmap),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Self::Heap { ptr, len, align } = *self
            && len != 0
        {
            let layout = Layout::from_size_align(len, align).expect("layout was valid on alloc");
            // SAFETY: `ptr` was allocated with exactly this layout (same
            // non-zero `len` and `align`) and is freed only once, here.
            unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }
}

/// Owns a contiguous region of raw, untyped bytes.
///
/// `Storage` is the lowest-level building block of datalab: it holds a
/// contiguous run of bytes with a known length and says nothing about how they
/// are interpreted. Interpreting them as a typed, shaped array is the job of a
/// view layered on top. The bytes may live in RAM or in a memory-mapped file —
/// see the [module documentation](self) for the available backings.
///
/// # Examples
///
/// ```
/// use datalab::storage::Storage;
///
/// let mut storage = Storage::zeroed(4);
/// storage.as_bytes_mut().unwrap()[1] = 42;
/// assert_eq!(storage.as_bytes(), &[0, 42, 0, 0]);
/// ```
pub struct Storage {
    backing: Backing,
}

// SAFETY: both backings are safe to move across threads: the heap allocation
// is uniquely owned with no interior mutability (like `Vec<u8>`), and
// `memmap2::Mmap` is itself `Send`.
unsafe impl Send for Storage {}

// SAFETY: `&Storage` only permits reads and there is no interior mutability;
// `memmap2::Mmap` is itself `Sync`.
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
    #[must_use]
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
    #[must_use]
    pub fn zeroed_aligned(len: usize, align: usize) -> Self {
        assert!(
            align.is_power_of_two(),
            "alignment must be a power of two, got {align}"
        );

        if len == 0 {
            // No allocation for an empty storage: a dangling (but non-null)
            // pointer is a valid base for a zero-length slice.
            return Self {
                backing: Backing::Heap {
                    ptr: NonNull::dangling(),
                    len: 0,
                    align,
                },
            };
        }

        let layout = Layout::from_size_align(len, align).expect("allocation size overflow");
        // SAFETY: `layout` has a non-zero size (checked `len != 0` above).
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self {
            backing: Backing::Heap { ptr, len, align },
        }
    }

    /// Creates a storage holding a copy of `bytes`, using the default alignment.
    ///
    /// The bytes are copied into a freshly aligned heap allocation, so the
    /// result is writable and aligned to [`STORAGE_ALIGN`] regardless of the
    /// source's alignment.
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
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut storage = Self::zeroed(bytes.len());
        storage
            .as_bytes_mut()
            .expect("freshly allocated heap storage is writable")
            .copy_from_slice(bytes);
        storage
    }

    /// Opens `path` as a **read-only**, memory-mapped storage.
    ///
    /// No bytes are read up front: the OS loads pages lazily on first access
    /// and evicts them under memory pressure, so the storage may be far larger
    /// than the available RAM while keeping memory usage bounded. Mapping an
    /// empty file yields an ordinary empty (heap) storage.
    ///
    /// The resulting storage is not writable: [`Storage::as_bytes_mut`]
    /// returns `None`. To modify the bytes, promote them to a heap copy with
    /// [`Storage::make_mut`].
    ///
    /// # Errors
    ///
    /// Returns any I/O error from opening or mapping the file.
    ///
    /// # File stability
    ///
    /// The file must not be modified or truncated by this or another process
    /// while the storage exists: the OS reflects such changes into the
    /// mapping, which can crash reads (`SIGBUS`) or produce inconsistent
    /// data. This is the standard memory-mapping contract (the same one
    /// accepted by e.g. safetensors and ripgrep).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use datalab::storage::Storage;
    ///
    /// let weights = Storage::map_file("model-weights.bin")?;
    /// let first_bytes = &weights.as_bytes()[..8]; // only this page is loaded
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn map_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        if file.metadata()?.len() == 0 {
            // A zero-length mapping is invalid; an empty heap storage is
            // indistinguishable from the outside.
            return Ok(Self::zeroed(0));
        }
        // SAFETY: we create a read-only mapping of a file we just opened. The
        // caller accepts the documented file-stability contract above.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self {
            backing: Backing::Mmap(mmap),
        })
    }

    /// Moves the bytes into a temporary file inside `dir` and releases the
    /// heap allocation, keeping the bytes readable through a memory mapping.
    ///
    /// This is the spill primitive: after the call, the storage's RAM is
    /// returned to the system and reads go through the OS page cache. The
    /// storage becomes read-only ([`Storage::as_bytes_mut`] returns `None`);
    /// promote it back with [`Storage::make_mut`] if needed.
    ///
    /// The temporary file is unlinked immediately after mapping, so it is
    /// reclaimed automatically when the storage is dropped (or on process
    /// exit), and never needs manual cleanup. Spilling an empty or already
    /// spilled storage is a no-op.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from creating, writing or mapping the file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use datalab::storage::Storage;
    ///
    /// let mut storage = Storage::from_bytes(&[1, 2, 3]);
    /// storage.spill_to_disk(std::env::temp_dir())?; // RAM released
    /// assert_eq!(storage.as_bytes(), &[1, 2, 3]);   // still readable
    /// assert!(storage.as_bytes_mut().is_none());    // but read-only now
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn spill_to_disk(&mut self, dir: impl AsRef<Path>) -> io::Result<()> {
        match &self.backing {
            Backing::Mmap(_) => Ok(()),
            Backing::Heap { len: 0, .. } => Ok(()),
            Backing::Heap { .. } => {
                let id = SPILL_COUNTER.fetch_add(1, Ordering::Relaxed);
                let path = dir
                    .as_ref()
                    .join(format!("datalab-spill-{}-{id}", process::id()));
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&path)?;
                file.write_all(self.as_bytes())?;
                // SAFETY: read-only mapping of a file this process just wrote
                // and, once unlinked below, no other process can reach.
                let mmap = unsafe { Mmap::map(&file)? };
                // Unlink now: on Unix the data lives until the mapping is
                // dropped, and the file needs no manual cleanup. Best-effort:
                // on platforms that forbid deleting an open file, it lingers
                // until process exit.
                let _ = fs::remove_file(&path);
                self.backing = Backing::Mmap(mmap);
                Ok(())
            }
        }
    }

    /// Returns `true` if the bytes can be mutated in place (heap backing).
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// assert!(Storage::zeroed(4).is_writable());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_writable(&self) -> bool {
        matches!(self.backing, Backing::Heap { .. })
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
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.backing {
            Backing::Heap { len, .. } => *len,
            Backing::Mmap(mmap) => mmap.len(),
        }
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
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alignment, in bytes, of the first byte.
    ///
    /// For a heap storage this is the alignment the allocation was created
    /// with. For a memory-mapped storage it is the actual alignment of the
    /// mapped address (mappings are page-aligned, so this is at least the
    /// page size — 4 KiB or more).
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::{Storage, STORAGE_ALIGN};
    ///
    /// assert_eq!(Storage::zeroed(8).alignment(), STORAGE_ALIGN);
    /// ```
    #[must_use]
    pub fn alignment(&self) -> usize {
        match &self.backing {
            Backing::Heap { align, .. } => *align,
            Backing::Mmap(mmap) => {
                let addr = mmap.as_ptr() as usize;
                debug_assert_ne!(addr, 0);
                1 << addr.trailing_zeros()
            }
        }
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
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backing {
            // SAFETY: `ptr` points to `len` contiguous, initialized bytes that
            // stay valid and immutable for the lifetime of `&self`. For
            // `len == 0`, a dangling pointer is a valid base for an empty
            // slice.
            Backing::Heap { ptr, len, .. } => unsafe {
                slice::from_raw_parts(ptr.as_ptr(), *len)
            },
            Backing::Mmap(mmap) => mmap,
        }
    }

    /// Returns a mutable view of the raw bytes, or `None` if the storage is
    /// read-only (memory-mapped).
    ///
    /// Read-only storages never mutate silently: promote them explicitly with
    /// [`Storage::make_mut`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let mut storage = Storage::zeroed(2);
    /// storage.as_bytes_mut().unwrap()[0] = 7;
    /// assert_eq!(storage.as_bytes(), &[7, 0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match &mut self.backing {
            Backing::Heap { ptr, len, .. } => {
                // SAFETY: `ptr` points to `len` contiguous, initialized bytes
                // that stay valid for the lifetime of `&mut self`, and
                // `&mut self` guarantees exclusive access. For `len == 0`, a
                // dangling pointer is valid.
                let bytes = unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), *len) };
                Some(bytes)
            }
            Backing::Mmap(_) => None,
        }
    }

    /// Ensures the storage is writable, then returns its mutable bytes.
    ///
    /// If the storage is memory-mapped (read-only), its bytes are first
    /// **copied** into a fresh heap allocation — an explicit `O(len)` cost,
    /// after which the mapping is released. Heap storages are returned as-is.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use datalab::storage::Storage;
    ///
    /// let mut storage = Storage::map_file("data.bin")?; // read-only
    /// storage.make_mut()[0] = 42;                       // explicit copy, then write
    /// assert!(storage.is_writable());
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn make_mut(&mut self) -> &mut [u8] {
        if let Backing::Mmap(_) = &self.backing {
            let promoted = Self::from_bytes(self.as_bytes());
            *self = promoted; // the mapping is dropped here
        }
        self.as_bytes_mut()
            .expect("heap storage is always writable")
    }
}

impl Clone for Storage {
    /// Deep-copies the bytes into a new **heap** storage, so a clone is always
    /// writable — even when cloning a memory-mapped storage.
    fn clone(&self) -> Self {
        match &self.backing {
            Backing::Heap { len, align, .. } => {
                let mut cloned = Self::zeroed_aligned(*len, *align);
                cloned
                    .as_bytes_mut()
                    .expect("freshly allocated heap storage is writable")
                    .copy_from_slice(self.as_bytes());
                cloned
            }
            Backing::Mmap(_) => Self::from_bytes(self.as_bytes()),
        }
    }
}

impl PartialEq for Storage {
    /// Two storages are equal when they hold the same bytes; backing and
    /// alignment are not compared.
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
        let backing = match &self.backing {
            Backing::Heap { .. } => "heap",
            Backing::Mmap(_) => "mmap",
        };
        f.debug_struct("Storage")
            .field("backing", &backing)
            .field("len", &self.len())
            .finish()
    }
}

impl From<&[u8]> for Storage {
    fn from(bytes: &[u8]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl AsRef<[u8]> for Storage {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a unique temp-file path for file-based tests.
    fn temp_path(tag: &str) -> std::path::PathBuf {
        let id = SPILL_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("datalab-test-{tag}-{}-{id}", process::id()))
    }

    #[test]
    fn zeroed_allocates_len_zero_bytes() {
        let storage = Storage::zeroed(8);
        assert_eq!(storage.len(), 8);
        assert!(!storage.is_empty());
        assert!(storage.is_writable());
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
        storage.as_bytes_mut().unwrap()[1] = 42;
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
        copy.as_bytes_mut().unwrap()[0] = 0;
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
        // Compile-time contract: the default alignment covers every scalar.
        const { assert!(STORAGE_ALIGN.is_power_of_two()) };
        const { assert!(STORAGE_ALIGN >= 64) };
        const { assert!(STORAGE_ALIGN >= align_of::<u64>()) };
        const { assert!(STORAGE_ALIGN >= align_of::<f64>()) };
        const { assert!(STORAGE_ALIGN >= align_of::<u128>()) };
    }

    #[test]
    #[cfg_attr(miri, ignore)] // file-backed mmap is not supported under miri
    fn map_file_reads_file_bytes() {
        let path = temp_path("map");
        fs::write(&path, [10u8, 20, 30]).unwrap();
        let storage = Storage::map_file(&path).unwrap();
        assert_eq!(storage.as_bytes(), &[10, 20, 30]);
        assert!(!storage.is_writable());
        assert!(storage.alignment() >= 4096, "mappings are page-aligned");
        drop(storage);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn map_file_of_missing_file_errors() {
        assert!(Storage::map_file(temp_path("missing")).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn map_file_of_empty_file_is_empty() {
        let path = temp_path("empty");
        fs::write(&path, []).unwrap();
        let storage = Storage::map_file(&path).unwrap();
        assert!(storage.is_empty());
        fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn mapped_storage_refuses_mutation() {
        let path = temp_path("romut");
        fs::write(&path, [1u8, 2]).unwrap();
        let mut storage = Storage::map_file(&path).unwrap();
        assert!(storage.as_bytes_mut().is_none());
        fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn make_mut_promotes_mapped_storage_to_heap() {
        let path = temp_path("promote");
        fs::write(&path, [1u8, 2, 3]).unwrap();
        let mut storage = Storage::map_file(&path).unwrap();
        storage.make_mut()[0] = 9;
        assert!(storage.is_writable());
        assert_eq!(storage.as_bytes(), &[9, 2, 3]);
        // The file itself is untouched.
        assert_eq!(fs::read(&path).unwrap(), [1, 2, 3]);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn make_mut_on_heap_storage_is_free() {
        let mut storage = Storage::from_bytes(&[5, 6]);
        storage.make_mut()[1] = 7;
        assert_eq!(storage.as_bytes(), &[5, 7]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn spill_keeps_bytes_readable_but_not_writable() {
        let mut storage = Storage::from_bytes(&[1, 2, 3, 4]);
        storage.spill_to_disk(std::env::temp_dir()).unwrap();
        assert_eq!(storage.as_bytes(), &[1, 2, 3, 4]);
        assert!(!storage.is_writable());
        // Spilling again is a no-op.
        storage.spill_to_disk(std::env::temp_dir()).unwrap();
        assert_eq!(storage.as_bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn spill_of_empty_storage_is_a_noop() {
        let mut storage = Storage::zeroed(0);
        storage.spill_to_disk(std::env::temp_dir()).unwrap();
        assert!(storage.is_empty());
        assert!(storage.is_writable());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn spill_then_make_mut_round_trips() {
        let mut storage = Storage::from_bytes(&[7, 8]);
        storage.spill_to_disk(std::env::temp_dir()).unwrap();
        storage.make_mut()[0] = 1;
        assert!(storage.is_writable());
        assert_eq!(storage.as_bytes(), &[1, 8]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn clone_of_mapped_storage_is_writable() {
        let mut storage = Storage::from_bytes(&[3, 1, 4]);
        storage.spill_to_disk(std::env::temp_dir()).unwrap();
        let mut clone = storage.clone();
        assert!(clone.is_writable());
        assert_eq!(clone, storage);
        clone.as_bytes_mut().unwrap()[0] = 0;
        assert_eq!(storage.as_bytes(), &[3, 1, 4]);
    }
}
