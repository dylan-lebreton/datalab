//! Raw, contiguous byte storage — the lowest-level building block.
//!
//! A [`Storage`] owns a contiguous region of bytes with a known length and
//! says nothing about how those bytes are interpreted; interpreting them as a
//! typed, shaped array is the job of a view layered on top.
//!
//! For now the bytes always live in RAM (the heap). The backing store is meant
//! to become pluggable later — memory-mapped files, spill-to-disk, shared
//! memory — behind this same API, without callers noticing.

/// Owns a contiguous region of raw, untyped bytes.
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
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Storage {
    bytes: Vec<u8>,
}

impl Storage {
    /// Creates a storage of `len` zero-initialized bytes.
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
        Self {
            bytes: vec![0u8; len],
        }
    }

    /// Creates a storage that takes ownership of already-existing bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::from_bytes(vec![1, 2, 3]);
    /// assert_eq!(storage.len(), 3);
    /// ```
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
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
        self.bytes.len()
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
        self.bytes.is_empty()
    }

    /// Returns an immutable view of the raw bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::storage::Storage;
    ///
    /// let storage = Storage::from_bytes(vec![1, 2, 3]);
    /// assert_eq!(storage.as_bytes(), &[1, 2, 3]);
    /// ```
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
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
        &mut self.bytes
    }
}

impl From<Vec<u8>> for Storage {
    fn from(bytes: Vec<u8>) -> Self {
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
        let storage = Storage::from_bytes(vec![1, 2, 3]);
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn empty_storage_is_empty() {
        let storage = Storage::zeroed(0);
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
    }

    #[test]
    fn as_bytes_mut_allows_in_place_edits() {
        let mut storage = Storage::zeroed(4);
        storage.as_bytes_mut()[1] = 42;
        assert_eq!(storage.as_bytes(), &[0, 42, 0, 0]);
    }

    #[test]
    fn from_vec_conversion_matches_from_bytes() {
        let storage: Storage = vec![9, 8, 7].into();
        assert_eq!(storage, Storage::from_bytes(vec![9, 8, 7]));
    }

    #[test]
    fn as_ref_exposes_bytes() {
        let storage = Storage::from_bytes(vec![4, 5]);
        let slice: &[u8] = storage.as_ref();
        assert_eq!(slice, &[4, 5]);
    }

    #[test]
    fn default_is_empty() {
        assert!(Storage::default().is_empty());
    }
}
