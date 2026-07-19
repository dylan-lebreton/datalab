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
//! the primitive a memory-budgeted engine needs.
//!
//! # Sharing and copy-on-write
//!
//! The underlying allocation is reference-counted (the Arrow/Polars model):
//! [`Storage::clone`] and [`Storage::slice`] are **zero-copy** — they share
//! the same bytes and cost an atomic increment. In exchange, in-place
//! mutation requires **unique** ownership of a heap backing:
//! [`Storage::as_bytes_mut`] returns `None` while the bytes are shared or
//! memory-mapped, and [`Storage::make_mut`] performs the **explicit**
//! copy-on-write promotion. Cheap things are silent, costly things are
//! explicit — never the other way around.

use std::alloc::{self, Layout};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::process;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::Arc;
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

/// Where a [`Storage`]'s bytes physically live. Shared behind an [`Arc`];
/// dropped (and freed) when the last sharer goes away.
enum Backing {
    /// Owned, writable (when uniquely held), aligned heap allocation.
    Heap {
        /// Pointer to the first byte. For an empty storage this is a dangling
        /// but non-null pointer and no allocation is held.
        ptr: NonNull<u8>,
        /// Number of bytes of the allocation.
        len: usize,
        /// Alignment the allocation was created with; needed to free it.
        align: usize,
    },
    /// Read-only, page-aligned memory-mapped file.
    Mmap(Mmap),
}

// SAFETY: the heap allocation is owned by exactly one `Backing` (no interior
// mutability; mutation goes through `Arc::get_mut`, i.e. exclusive access),
// and `memmap2::Mmap` is itself `Send`.
unsafe impl Send for Backing {}

// SAFETY: shared references to a `Backing` only permit reads, and
// `memmap2::Mmap` is itself `Sync`.
unsafe impl Sync for Backing {}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Self::Heap { ptr, len, align } = *self
            && len != 0
        {
            let layout = Layout::from_size_align(len, align).expect("layout was valid on alloc");
            // SAFETY: `ptr` was allocated with exactly this layout (same
            // non-zero `len` and `align`) and is freed only once, here — the
            // `Arc` guarantees this runs when the last sharer is dropped.
            unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }
}

/// Owns (possibly jointly) a contiguous region of raw, untyped bytes.
///
/// `Storage` is the lowest-level building block of datalab: it exposes a
/// window (`offset`/`len`) over a reference-counted allocation and says
/// nothing about how the bytes are interpreted. Interpreting them as a
/// typed, shaped array is the job of a view layered on top. The bytes may
/// live in RAM or in a memory-mapped file, and are shared zero-copy by
/// [`Storage::clone`] and [`Storage::slice`] — see the
/// [module documentation](self).
///
/// # Examples
///
/// ```
/// use datalab::storage::Storage;
///
/// let mut storage = Storage::zeroed(4);
/// storage.as_bytes_mut().unwrap()[1] = 42;
/// assert_eq!(storage.as_bytes(), &[0, 42, 0, 0]);
///
/// let shared = storage.clone();              // zero-copy
/// assert!(storage.as_bytes_mut().is_none()); // shared => not writable
/// drop(shared);
/// assert!(storage.as_bytes_mut().is_some()); // unique again
/// ```
pub struct Storage {
    /// Invariant: `offset + len <= backing.len()`.
    backing: Arc<Backing>,
    /// Start of this storage's window into the backing.
    offset: usize,
    /// Number of bytes of this storage's window.
    len: usize,
}

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

        let backing = if len == 0 {
            // No allocation for an empty storage: a dangling (but non-null)
            // pointer is a valid base for a zero-length slice. It is placed
            // at address `align` so the pointer really carries the alignment
            // `alignment()` reports, even with nothing to point at.
            Backing::Heap {
                ptr: NonNull::new(ptr::without_provenance_mut(align))
                    .expect("align is a non-zero power of two"),
                len: 0,
                align,
            }
        } else {
            let layout = Layout::from_size_align(len, align).expect("allocation size overflow");
            // SAFETY: `layout` has a non-zero size (checked `len != 0` above).
            let ptr = unsafe { alloc::alloc_zeroed(layout) };
            let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
            Backing::Heap { ptr, len, align }
        };
        Self {
            backing: Arc::new(backing),
            offset: 0,
            len,
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
        Self::from_bytes_aligned(bytes, STORAGE_ALIGN)
    }

    /// Creates a storage holding a copy of `bytes` with a caller-chosen
    /// `align`.
    ///
    /// This is the copying counterpart of [`Storage::zeroed_aligned`]: the
    /// bytes land in a fresh, writable heap allocation aligned to `align`.
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two, or if the total allocation
    /// size would overflow `isize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::from_bytes_aligned(&[1, 2, 3], 4096);
    /// assert_eq!(storage.as_bytes(), &[1, 2, 3]);
    /// assert_eq!(storage.alignment(), 4096);
    /// ```
    #[must_use]
    pub fn from_bytes_aligned(bytes: &[u8], align: usize) -> Self {
        let mut storage = Self::zeroed_aligned(bytes.len(), align);
        storage
            .as_bytes_mut()
            .expect("freshly allocated heap storage is uniquely writable")
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
        let len = mmap.len();
        Ok(Self {
            backing: Arc::new(Backing::Mmap(mmap)),
            offset: 0,
            len,
        })
    }

    /// Returns a **zero-copy** sub-storage over `offset..offset + len` of
    /// this storage's bytes.
    ///
    /// The slice shares the underlying allocation (an atomic increment, no
    /// bytes moved). While any sharer is alive, neither storage is writable
    /// in place — see the [module documentation](self).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` exceeds this storage's length.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::from_bytes(&[1, 2, 3, 4, 5]);
    /// let window = storage.slice(1, 3);
    /// assert_eq!(window.as_bytes(), &[2, 3, 4]);
    /// ```
    #[must_use]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "slice {offset}..{} out of bounds of storage of length {}",
            offset.saturating_add(len),
            self.len
        );
        Self {
            backing: Arc::clone(&self.backing),
            offset: self.offset + offset,
            len,
        }
    }

    /// Moves the bytes into a temporary file inside `dir` and releases this
    /// handle's reference to the heap allocation, keeping the bytes readable
    /// through a memory mapping.
    ///
    /// This is the spill primitive: after the call, reads go through the OS
    /// page cache and the storage is read-only ([`Storage::as_bytes_mut`]
    /// returns `None`); promote it back with [`Storage::make_mut`] if
    /// needed. Other storages sharing the old allocation are unaffected (the
    /// RAM itself is returned to the system when the last sharer lets go).
    ///
    /// On Unix the temporary file is created readable by the owning user
    /// only and unlinked immediately after mapping, so it is reclaimed
    /// automatically when the storage is dropped (or on process exit) and
    /// never needs manual cleanup. On platforms that forbid deleting a
    /// mapped file (e.g. Windows), the file persists in `dir` until it is
    /// removed externally. Spilling an empty or already spilled storage is a
    /// no-op.
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
        if matches!(&*self.backing, Backing::Mmap(_)) || self.len == 0 {
            return Ok(());
        }
        let id = SPILL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = dir
            .as_ref()
            .join(format!("datalab-spill-{}-{id}", process::id()));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        // Spilled bytes may be sensitive and `dir` may be shared (`/tmp`):
        // keep the file private to the owning user.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path)?;
        file.write_all(self.as_bytes())?;
        // SAFETY: read-only mapping of a file this process just wrote and,
        // once unlinked below, no other process can reach.
        let mmap = unsafe { Mmap::map(&file)? };
        // Unlink now: on Unix the data lives until the mapping is dropped,
        // and the file needs no manual cleanup. Best-effort: on platforms
        // that forbid deleting an open file, it lingers until process exit.
        let _ = fs::remove_file(&path);
        let len = mmap.len();
        self.backing = Arc::new(Backing::Mmap(mmap));
        self.offset = 0;
        self.len = len;
        Ok(())
    }

    /// Returns `true` if the bytes can be mutated in place: the storage is
    /// heap-backed **and** not currently shared with any clone or slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::zeroed(4);
    /// assert!(storage.is_writable());
    /// let shared = storage.clone();
    /// assert!(!storage.is_writable()); // shared until `shared` is dropped
    /// ```
    #[inline]
    #[must_use]
    pub fn is_writable(&self) -> bool {
        Arc::strong_count(&self.backing) == 1 && matches!(&*self.backing, Backing::Heap { .. })
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
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the guaranteed alignment, in bytes, of the first byte of this
    /// storage's window.
    ///
    /// For a whole heap storage this is the alignment the allocation was
    /// created with; for a memory-mapped one, the actual alignment of the
    /// mapped address (page-aligned, so at least 4 KiB). Slicing at an
    /// `offset` lowers the guarantee to the alignment of that offset.
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
        let base = match &*self.backing {
            Backing::Heap { align, .. } => *align,
            Backing::Mmap(mmap) => {
                let addr = mmap.as_ptr() as usize;
                debug_assert_ne!(addr, 0);
                1 << addr.trailing_zeros()
            }
        };
        if self.offset == 0 {
            base
        } else {
            base.min(1 << self.offset.trailing_zeros())
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
        match &*self.backing {
            // SAFETY: the struct invariant guarantees `offset + len` lies
            // within the allocation; the bytes are initialized and stay valid
            // and immutable for the lifetime of `&self` (shared `Arc`). For
            // an empty window, any base pointer is valid.
            Backing::Heap { ptr, .. } => unsafe {
                slice::from_raw_parts(ptr.as_ptr().add(self.offset), self.len)
            },
            Backing::Mmap(mmap) => &mmap[self.offset..self.offset + self.len],
        }
    }

    /// Returns a mutable view of the raw bytes, or `None` if the storage is
    /// not writable in place (memory-mapped, or shared by a clone/slice).
    ///
    /// Nothing mutates or copies silently: promote a non-writable storage
    /// explicitly with [`Storage::make_mut`] instead.
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
        let offset = self.offset;
        let len = self.len;
        // `Arc::get_mut` returns `None` while the backing is shared — the
        // copy-on-write gate.
        match Arc::get_mut(&mut self.backing)? {
            Backing::Heap { ptr, .. } => {
                // SAFETY: the struct invariant guarantees `offset + len` lies
                // within the allocation; `Arc::get_mut` proved exclusive
                // access for the lifetime of `&mut self`. For an empty
                // window, any base pointer is valid.
                let bytes = unsafe { slice::from_raw_parts_mut(ptr.as_ptr().add(offset), len) };
                Some(bytes)
            }
            Backing::Mmap(_) => None,
        }
    }

    /// Ensures the storage is uniquely writable, then returns its mutable
    /// bytes.
    ///
    /// If the storage is memory-mapped or shared, its window is first
    /// **copied** into a fresh heap allocation — an explicit `O(len)`
    /// copy-on-write, after which this handle owns the copy exclusively
    /// (other sharers keep the original bytes). Uniquely-owned heap storages
    /// are returned as-is.
    ///
    /// The copy preserves the alignment the backing allocation was created
    /// with, so a guarantee established via [`Storage::zeroed_aligned`]
    /// survives copy-on-write. Memory-mapped backings (whose page alignment
    /// was never requested by the caller) are copied at the default
    /// [`STORAGE_ALIGN`].
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let mut storage = Storage::from_bytes(&[1, 2]);
    /// let shared = storage.clone();
    /// storage.make_mut()[0] = 9;            // explicit copy, then write
    /// assert_eq!(storage.as_bytes(), &[9, 2]);
    /// assert_eq!(shared.as_bytes(), &[1, 2]); // sharers keep the original
    /// ```
    pub fn make_mut(&mut self) -> &mut [u8] {
        if !self.is_writable() {
            let align = match &*self.backing {
                Backing::Heap { align, .. } => *align,
                Backing::Mmap(_) => STORAGE_ALIGN,
            };
            *self = Self::from_bytes_aligned(self.as_bytes(), align);
        }
        self.as_bytes_mut()
            .expect("a uniquely-owned heap storage is writable")
    }
}

impl Clone for Storage {
    /// **Zero-copy**: the clone shares the same bytes (an atomic increment).
    /// While both are alive, neither is writable in place; mutate through
    /// the explicit [`Storage::make_mut`]. See the
    /// [module documentation](self).
    fn clone(&self) -> Self {
        Self {
            backing: Arc::clone(&self.backing),
            offset: self.offset,
            len: self.len,
        }
    }
}

impl PartialEq for Storage {
    /// Two storages are equal when they hold the same bytes; backing,
    /// sharing and alignment are not compared.
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
        let backing = match &*self.backing {
            Backing::Heap { .. } => "heap",
            Backing::Mmap(_) => "mmap",
        };
        f.debug_struct("Storage")
            .field("backing", &backing)
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("shared", &(Arc::strong_count(&self.backing) > 1))
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
    fn empty_storage_pointer_honours_the_reported_alignment() {
        let storage = Storage::zeroed_aligned(0, 64);
        assert_eq!(storage.alignment(), 64);
        assert_eq!(storage.as_bytes().as_ptr() as usize % 64, 0);
    }

    #[test]
    fn from_bytes_aligned_respects_alignment() {
        let storage = Storage::from_bytes_aligned(&[7, 8, 9], 4096);
        assert_eq!(storage.as_bytes(), &[7, 8, 9]);
        assert_eq!(storage.alignment(), 4096);
        assert_eq!(storage.as_bytes().as_ptr() as usize % 4096, 0);
    }

    #[test]
    fn make_mut_preserves_the_requested_alignment() {
        let mut storage = Storage::zeroed_aligned(32, 4096);
        let shared = storage.clone();
        storage.make_mut()[0] = 1; // copy-on-write
        assert_eq!(storage.alignment(), 4096);
        assert_eq!(storage.as_bytes().as_ptr() as usize % 4096, 0);
        assert_eq!(shared.as_bytes()[0], 0);
    }

    #[test]
    fn clone_is_zero_copy_and_blocks_writes_while_shared() {
        let mut storage = Storage::from_bytes(&[9, 8, 7]);
        let shared = storage.clone();
        // Same bytes, same underlying allocation.
        assert_eq!(shared, storage);
        assert_eq!(shared.as_bytes().as_ptr(), storage.as_bytes().as_ptr());
        // Shared => neither handle is writable in place.
        assert!(!storage.is_writable());
        assert!(storage.as_bytes_mut().is_none());
        // Dropping the sharer restores writability.
        drop(shared);
        assert!(storage.is_writable());
        storage.as_bytes_mut().unwrap()[0] = 0;
        assert_eq!(storage.as_bytes(), &[0, 8, 7]);
    }

    #[test]
    fn make_mut_on_shared_storage_copies_on_write() {
        let mut storage = Storage::from_bytes(&[1, 2]);
        let shared = storage.clone();
        storage.make_mut()[0] = 9;
        assert_eq!(storage.as_bytes(), &[9, 2]);
        assert_eq!(shared.as_bytes(), &[1, 2]); // the sharer is untouched
        assert!(shared.is_writable()); // and unique again
    }

    #[test]
    fn slice_is_zero_copy() {
        let storage = Storage::from_bytes(&[1, 2, 3, 4, 5]);
        let window = storage.slice(1, 3);
        assert_eq!(window.as_bytes(), &[2, 3, 4]);
        assert_eq!(
            window.as_bytes().as_ptr(),
            storage.as_bytes()[1..].as_ptr()
        );
        // A slice of a slice composes offsets.
        let inner = window.slice(2, 1);
        assert_eq!(inner.as_bytes(), &[4]);
        // Empty slice at the end is fine.
        assert!(storage.slice(5, 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn out_of_bounds_slice_panics() {
        let _ = Storage::from_bytes(&[1, 2, 3]).slice(2, 2);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn overflowing_slice_bounds_panic_with_the_right_message() {
        let _ = Storage::from_bytes(&[1, 2, 3]).slice(usize::MAX, 2);
    }

    #[test]
    fn slicing_lowers_the_alignment_guarantee() {
        let storage = Storage::zeroed(64);
        assert_eq!(storage.slice(0, 16).alignment(), STORAGE_ALIGN);
        assert_eq!(storage.slice(8, 16).alignment(), 8);
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
    fn make_mut_on_unique_heap_storage_is_free() {
        let mut storage = Storage::from_bytes(&[5, 6]);
        let before = storage.as_bytes().as_ptr();
        storage.make_mut()[1] = 7;
        assert_eq!(storage.as_bytes(), &[5, 7]);
        assert_eq!(storage.as_bytes().as_ptr(), before); // no reallocation
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
    fn spill_of_shared_storage_leaves_sharers_untouched() {
        let mut storage = Storage::from_bytes(&[3, 1, 4]);
        let shared = storage.clone();
        storage.spill_to_disk(std::env::temp_dir()).unwrap();
        assert_eq!(storage.as_bytes(), &[3, 1, 4]);
        assert!(shared.is_writable()); // the sharer became unique heap again
        assert_eq!(shared.as_bytes(), &[3, 1, 4]);
    }
}
