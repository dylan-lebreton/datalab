//! The lazy, streaming execution engine — v1.
//!
//! A [`LazyTensor`] is a **plan**: a description of computations to run
//! later, built by chaining methods (like a Polars `LazyFrame`). The rule is
//! uniform, with no exception to memorize:
//!
//! > **Nothing executes until [`LazyTensor::collect`] or
//! > [`LazyTensor::sink_file`] is called.** Every other method — including
//! > reductions like [`LazyTensor::sum`] — only extends the plan.
//!
//! `collect` runs the plan and materializes the result as a [`Tensor`];
//! `sink_file` runs the plan and streams the result to a file **without ever
//! materializing it**.
//!
//! Execution is *batched*: the source produces small contiguous [`Tensor`]
//! batches (sized in bytes, see [`LazyTensor::with_batch_bytes`]), each
//! operator transforms a batch into a new batch, and the terminal consumes
//! them one by one. At any instant only a couple of batches are resident, so
//! memory stays bounded regardless of the source size — a file far larger
//! than RAM streams through comfortably.
//!
//! ```
//! use datalab::lazy;
//!
//! let total = lazy::generate(1_000, |i| i as f64)
//!     .map(|x| x * 2.0)
//!     .sum()          // still lazy: a 1-element plan
//!     .collect()?     // the only thing that executes
//!     .item();
//! assert_eq!(total, 999_000.0);
//! # Ok::<(), datalab::lazy::EngineError>(())
//! ```
//!
//! # Design notes (and how this scales later)
//!
//! **The plan is data, not types.** The chain of operations is stored as a
//! plain `Vec` of nodes so it can be inspected ([`LazyTensor::explain`]) and,
//! later, optimized (e.g. fusing consecutive `map`s into one pass). The
//! public API stays fully typed (`LazyTensor<T>`); inside, each operation is
//! a type-erased `batch -> batch` function whose types were checked at
//! construction.
//!
//! **The plan is internal plumbing, not a user data structure.** A plan is a
//! recipe — a handful of nodes, a few bytes each — while the data it
//! describes may be terabytes. It is deliberately a plain `Vec`, not a
//! datalab structure: the engine never builds itself out of the user-facing
//! types layered on top of it.
//!
//! **Pull now, push later.** Execution is a single-threaded *pull* loop
//! (each terminal drains the source through the operators). This is the
//! simplest correct model, but it cannot parallelize across cores: that
//! requires a *push* (morsel-driven) executor, where sources push batches to
//! a pool of workers — the migration Polars had to make for its streaming
//! engine. The seam is prepared: operators are pure `batch -> batch`
//! functions with `Send` bounds that never know who drives them, so moving
//! to a push executor replaces only the driving loop, not the operators.
//!
//! **Linear chain now, graph later.** A plan is currently a straight line
//! (source → ops → terminal). Binary operations between two lazy tensors
//! (e.g. `&a + &b` lazily) require the plan to become a DAG with multiple
//! sources; the `Vec<OpNode>` then becomes a node arena with indices. This
//! is an intended, documented evolution — not a rewrite: sources, operators
//! and terminals keep their shapes.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::ops::Add;
use std::path::{Path, PathBuf};

use crate::kernel;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorFileError};
use crate::view::{Element, View};

/// Default target size of a batch, in bytes (a few MiB: large enough to
/// amortize per-batch overhead and feed SIMD, small enough to stay
/// cache-friendly and keep memory bounded).
pub const DEFAULT_BATCH_BYTES: usize = 4 * 1024 * 1024;

/// A type-erased batch: a boxed `Tensor<T>` for some [`Element`] type `T`.
///
/// Types are checked when the plan is built, then erased so heterogeneous
/// chains (`f32 -> f64 -> i64`) can share one representation.
type Batch = Box<dyn Any + Send>;

/// A pull-based stream of batches — the execution side of a source.
trait BatchStream {
    /// Produces the next batch, or `None` when the source is exhausted.
    fn next_batch(&mut self) -> Option<Batch>;
}

/// Builds a fresh [`BatchStream`], given the target batch size in bytes.
type StreamFactory = Box<dyn FnOnce(usize) -> Result<Box<dyn BatchStream>, EngineError> + Send>;

/// The source node of a plan: a human-readable label plus a factory that
/// instantiates the actual stream when execution starts.
struct SourceNode {
    label: String,
    make: StreamFactory,
}

/// An operator node: a label plus a pure, type-erased `batch -> batch`
/// function. Operators never know who drives them (pull today, push later).
struct OpNode {
    label: &'static str,
    apply: Box<dyn Fn(Batch) -> Batch + Send>,
}

/// The reason a plan failed to execute.
#[derive(Debug)]
pub enum EngineError {
    /// Opening or validating the source failed.
    Source(TensorFileError),
    /// Writing the sink failed.
    Io(io::Error),
    /// The sink path refers to the same file the plan is scanning; executing
    /// would overwrite the source while reading it.
    SinkIntoSource,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(err) => write!(f, "cannot open plan source: {err}"),
            Self::Io(err) => write!(f, "sink failed: {err}"),
            Self::SinkIntoSource => {
                write!(f, "cannot sink into the file the plan is scanning")
            }
        }
    }
}

impl Error for EngineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::SinkIntoSource => None,
        }
    }
}

impl From<TensorFileError> for EngineError {
    fn from(err: TensorFileError) -> Self {
        Self::Source(err)
    }
}

impl From<io::Error> for EngineError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// A lazy computation plan producing elements of type `T`.
///
/// Built by [`scan_file`], [`generate`] or [`Tensor::lazy`], extended with
/// operations like [`LazyTensor::map`], and executed by a terminal
/// ([`collect`](LazyTensor::collect), [`sum`](LazyTensor::sum),
/// [`sink_file`](LazyTensor::sink_file)). See the [module docs](self).
#[must_use = "a LazyTensor is only a plan; call collect(), sum() or sink_file() to execute it"]
pub struct LazyTensor<T: Element> {
    source: SourceNode,
    ops: Vec<OpNode>,
    batch_bytes: usize,
    /// Set when the source is a file scan; used to refuse sinking into it.
    source_path: Option<PathBuf>,
    _out: PhantomData<T>,
}

impl<T: Element> fmt::Debug for LazyTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyTensor[{}]", self.explain().replace('\n', " "))
    }
}

/// Creates a lazy plan whose source is a raw binary file of `T` elements.
///
/// Nothing is read (the file is not even opened) until a terminal runs the
/// plan; the file is then memory-mapped and streamed batch by batch, so it
/// may be far larger than RAM. The format is the raw native byte
/// representation of the elements — exactly what [`LazyTensor::sink_file`]
/// and [`Tensor::map_file`] use, so the three compose.
///
/// # Examples
///
/// ```no_run
/// use datalab::lazy;
///
/// // 40 GB of f32 weights on disk, a few MiB of RAM used.
/// let norm = lazy::scan_file::<f32>("model-weights.bin")
///     .map(|w| w * w)
///     .sum()          // still lazy
///     .collect()?     // executes, streaming
///     .item();
/// # Ok::<(), datalab::lazy::EngineError>(())
/// ```
pub fn scan_file<T: Element>(path: impl AsRef<Path>) -> LazyTensor<T> {
    let path = path.as_ref().to_path_buf();
    let label = format!(
        "scan_file({:?}) as {}",
        path.display(),
        std::any::type_name::<T>()
    );
    let make_path = path.clone();
    LazyTensor {
        source: SourceNode {
            label,
            make: Box::new(move |batch_bytes| {
                let storage = Storage::map_file(&make_path).map_err(TensorFileError::Io)?;
                // Validate up front that the bytes form whole elements.
                View::<T>::new(&storage).map_err(TensorFileError::View)?;
                Ok(Box::new(StorageStream::<T> {
                    storage,
                    pos: 0,
                    batch_elems: batch_elems::<T>(batch_bytes),
                    _elem: PhantomData,
                }))
            }),
        },
        ops: Vec::new(),
        batch_bytes: DEFAULT_BATCH_BYTES,
        source_path: Some(path),
        _out: PhantomData,
    }
}

/// Creates a lazy plan whose source generates `len` elements with `f(i)`.
///
/// Elements are produced batch by batch during execution; the full sequence
/// never exists in memory at once.
///
/// # Examples
///
/// ```
/// use datalab::lazy;
///
/// let squares = lazy::generate(4, |i| (i * i) as u32).collect()?;
/// assert_eq!(squares.as_slice(), &[0, 1, 4, 9]);
/// # Ok::<(), datalab::lazy::EngineError>(())
/// ```
pub fn generate<T: Element>(
    len: usize,
    f: impl Fn(usize) -> T + Send + 'static,
) -> LazyTensor<T> {
    let label = format!("generate(len={len}) as {}", std::any::type_name::<T>());
    LazyTensor {
        source: SourceNode {
            label,
            make: Box::new(move |batch_bytes| {
                Ok(Box::new(GenerateStream {
                    f,
                    next: 0,
                    len,
                    batch_elems: batch_elems::<T>(batch_bytes),
                    _elem: PhantomData,
                }))
            }),
        },
        ops: Vec::new(),
        batch_bytes: DEFAULT_BATCH_BYTES,
        source_path: None,
        _out: PhantomData,
    }
}

impl<T: Element> Tensor<T> {
    /// Turns this tensor into a lazy plan sourcing from its elements.
    ///
    /// The tensor is moved into the plan and streamed batch by batch during
    /// execution. For data that lives in a file, prefer [`scan_file`], which
    /// does not require building a tensor first.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::tensor::Tensor;
    ///
    /// let doubled = Tensor::from_elements(&[1.0f64, 2.5])
    ///     .lazy()
    ///     .map(|x| x * 2.0)
    ///     .collect()?;
    /// assert_eq!(doubled.as_slice(), &[2.0, 5.0]);
    /// # Ok::<(), datalab::lazy::EngineError>(())
    /// ```
    pub fn lazy(self) -> LazyTensor<T> {
        let label = format!(
            "tensor(len={}) as {}",
            self.len(),
            std::any::type_name::<T>()
        );
        LazyTensor {
            source: SourceNode {
                label,
                make: Box::new(move |batch_bytes| {
                    Ok(Box::new(StorageStream::<T> {
                        storage: self.into_storage(),
                        pos: 0,
                        batch_elems: batch_elems::<T>(batch_bytes),
                        _elem: PhantomData,
                    }))
                }),
            },
            ops: Vec::new(),
            batch_bytes: DEFAULT_BATCH_BYTES,
            source_path: None,
            _out: PhantomData,
        }
    }
}

impl<T: Element> LazyTensor<T> {
    /// Sets the target batch size in bytes (default
    /// [`DEFAULT_BATCH_BYTES`]). Clamped so a batch holds at least one
    /// element.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::lazy;
    ///
    /// let sum = lazy::generate(10, |i| i as u64)
    ///     .with_batch_bytes(16) // tiny batches: 2 u64 per batch
    ///     .sum()
    ///     .collect()?
    ///     .item();
    /// assert_eq!(sum, 45);
    /// # Ok::<(), datalab::lazy::EngineError>(())
    /// ```
    pub fn with_batch_bytes(mut self, bytes: usize) -> Self {
        self.batch_bytes = bytes;
        self
    }

    /// Appends an element-wise transformation to the plan.
    ///
    /// Nothing runs now; during execution each batch is transformed into a
    /// new batch of `U`. (Consecutive `map`s currently run as separate
    /// passes over each small batch; fusing them into a single pass is a
    /// planned optimizer step — the plan-as-data representation exists
    /// precisely to enable it.)
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::lazy;
    ///
    /// let plan = lazy::generate(3, |i| i as i64).map(|x| x + 10);
    /// assert_eq!(plan.collect()?.as_slice(), &[10, 11, 12]);
    /// # Ok::<(), datalab::lazy::EngineError>(())
    /// ```
    pub fn map<U: Element>(mut self, f: impl Fn(T) -> U + Send + Sync + 'static) -> LazyTensor<U> {
        self.ops.push(OpNode {
            label: "map",
            apply: Box::new(move |batch: Batch| -> Batch {
                let input = batch
                    .downcast::<Tensor<T>>()
                    .expect("engine invariant: batch type matches the plan chain");
                Box::new(input.map(&f))
            }),
        });
        LazyTensor {
            source: self.source,
            ops: self.ops,
            batch_bytes: self.batch_bytes,
            source_path: self.source_path,
            _out: PhantomData,
        }
    }

    /// Renders the plan as a human-readable string, one node per line.
    ///
    /// Terminals are not part of the stored plan (they are the call that
    /// executes it).
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::lazy;
    ///
    /// let plan = lazy::generate(10, |i| i as f64).map(|x| x * 2.0);
    /// let text = plan.explain();
    /// assert!(text.contains("generate"));
    /// assert!(text.contains("map"));
    /// ```
    #[must_use]
    pub fn explain(&self) -> String {
        let mut out = self.source.label.clone();
        for op in &self.ops {
            out.push_str("\n  -> ");
            out.push_str(op.label);
        }
        out
    }

    /// Runs the plan and materializes the result as a [`Tensor`].
    ///
    /// By definition this holds the **entire result** in memory — use it
    /// when the result fits. To keep memory bounded end to end, finish with
    /// [`LazyTensor::sum`]-style reductions or [`LazyTensor::sink_file`]
    /// instead.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] if the source cannot be opened or validated.
    pub fn collect(self) -> Result<Tensor<T>, EngineError> {
        let mut batches: Vec<Tensor<T>> = Vec::new();
        self.run(|batch| batches.push(batch))?;

        let total: usize = batches.iter().map(Tensor::len).sum();
        let mut out = Tensor::<T>::zeros(total);
        let slice = out.as_mut_slice();
        let mut pos = 0;
        for batch in &batches {
            slice[pos..pos + batch.len()].copy_from_slice(batch.as_slice());
            pos += batch.len();
        }
        Ok(out)
    }

    /// Appends a streaming sum reduction to the plan, yielding a **lazy**
    /// one-element tensor (zero for an empty source).
    ///
    /// Nothing executes now — like every non-terminal, `sum` only extends
    /// the plan (same rule as Polars' `LazyFrame::sum`). On execution, each
    /// batch is reduced with [`kernel::sum`] (pairwise) and dropped before
    /// the next one is produced, so memory stays bounded; per-batch partial
    /// sums are added in order, making the result deterministic for a given
    /// batch size. Get the scalar with `.collect()?.item()`.
    ///
    /// Configure [`LazyTensor::with_batch_bytes`] *before* `sum`: the
    /// reduction stage keeps the batch size it was built with.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::lazy;
    ///
    /// let plan = lazy::generate(100, |i| i as u64).sum(); // still lazy
    /// assert_eq!(plan.collect()?.item(), 4950);
    /// # Ok::<(), datalab::lazy::EngineError>(())
    /// ```
    pub fn sum(self) -> LazyTensor<T>
    where
        T: Add<Output = T> + Default,
    {
        // The reduction wraps the whole plan built so far into a new source
        // that, once executed, drains it and emits a single 1-element batch.
        // In explain() it still reads as one linear chain ending in "-> sum".
        let label = format!("{}\n  -> sum", self.explain());
        let batch_bytes = self.batch_bytes;
        let source_path = self.source_path.clone();
        LazyTensor {
            source: SourceNode {
                label,
                make: Box::new(move |_outer_batch_bytes| {
                    let mut total = T::default();
                    self.run(|batch| total = total + kernel::sum(batch.as_slice()))?;
                    Ok(Box::new(OnceStream { value: Some(total) }))
                }),
            },
            ops: Vec::new(),
            batch_bytes,
            source_path,
            _out: PhantomData,
        }
    }

    /// Runs the plan and streams the result to a file, without ever
    /// materializing it: each batch is written then dropped, so memory stays
    /// bounded regardless of the result size.
    ///
    /// The format is the raw native byte representation of the elements,
    /// readable back with [`scan_file`] or [`Tensor::map_file`]. Any
    /// existing file at `path` is overwritten.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] if the source cannot be opened, if writing
    /// fails, or if `path` is the very file the plan scans
    /// ([`EngineError::SinkIntoSource`] — sinking into the source would
    /// overwrite it while reading it).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use datalab::lazy;
    ///
    /// lazy::scan_file::<f32>("weights.bin")
    ///     .map(|w| w * 0.5)
    ///     .sink_file("weights-halved.bin")?;
    /// # Ok::<(), datalab::lazy::EngineError>(())
    /// ```
    pub fn sink_file(self, path: impl AsRef<Path>) -> Result<(), EngineError> {
        let path = path.as_ref();
        if let Some(source) = &self.source_path {
            // Best-effort: canonicalization fails harmlessly if either path
            // does not exist yet.
            if let (Ok(a), Ok(b)) = (fs::canonicalize(source), fs::canonicalize(path))
                && a == b
            {
                return Err(EngineError::SinkIntoSource);
            }
        }

        let mut file = fs::File::create(path)?;
        let mut write_error: Option<io::Error> = None;
        self.run(|batch| {
            if write_error.is_none()
                && let Err(err) = file.write_all(batch.storage().as_bytes())
            {
                write_error = Some(err);
            }
        })?;
        if let Some(err) = write_error {
            return Err(EngineError::Io(err));
        }
        file.flush()?;
        Ok(())
    }

    /// The pull loop: drains the source through the operators, handing each
    /// resulting batch to `consume`. This is the only place that drives
    /// execution — swapping it for a push (parallel) executor later leaves
    /// sources, operators and terminals untouched.
    fn run(self, mut consume: impl FnMut(Tensor<T>)) -> Result<(), EngineError> {
        let mut stream = (self.source.make)(self.batch_bytes)?;
        while let Some(mut batch) = stream.next_batch() {
            for op in &self.ops {
                batch = (op.apply)(batch);
            }
            let tensor = batch
                .downcast::<Tensor<T>>()
                .expect("engine invariant: final batch type matches the plan output");
            consume(*tensor);
        }
        Ok(())
    }
}

/// Computes how many `T` elements fit the byte target (at least one).
fn batch_elems<T: Element>(batch_bytes: usize) -> usize {
    (batch_bytes / size_of::<T>()).max(1)
}

/// Stream over a storage viewable as `[T]` (in-memory or memory-mapped):
/// yields consecutive windows as **zero-copy slices** of the shared storage
/// (no bytes are moved to produce a batch; downstream operators read the
/// window and write into fresh outputs).
struct StorageStream<T: Element> {
    storage: Storage,
    /// Position in elements.
    pos: usize,
    batch_elems: usize,
    _elem: PhantomData<T>,
}

impl<T: Element> BatchStream for StorageStream<T> {
    fn next_batch(&mut self) -> Option<Batch> {
        let total = self.storage.len() / size_of::<T>();
        if self.pos >= total {
            return None;
        }
        let end = (self.pos + self.batch_elems).min(total);
        let window = self
            .storage
            .slice(self.pos * size_of::<T>(), (end - self.pos) * size_of::<T>());
        self.pos = end;
        Some(Box::new(Tensor::<T>::from_storage(window)))
    }
}

/// Stream that yields a single one-element batch (the result of a
/// reduction), then ends.
struct OnceStream<T: Element> {
    value: Option<T>,
}

impl<T: Element> BatchStream for OnceStream<T> {
    fn next_batch(&mut self) -> Option<Batch> {
        let value = self.value.take()?;
        Some(Box::new(Tensor::from_elements(&[value])))
    }
}

/// Stream that generates elements on the fly with a function of the index.
struct GenerateStream<T, F> {
    f: F,
    next: usize,
    len: usize,
    batch_elems: usize,
    _elem: PhantomData<T>,
}

impl<T: Element, F: Fn(usize) -> T + Send> BatchStream for GenerateStream<T, F> {
    fn next_batch(&mut self) -> Option<Batch> {
        if self.next >= self.len {
            return None;
        }
        let start = self.next;
        let end = (start + self.batch_elems).min(self.len);
        let batch = Tensor::from_fn(end - start, |i| (self.f)(start + i));
        self.next = end;
        Some(Box::new(batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a unique temp-file path for file-based tests.
    fn temp_path(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("datalab-lazy-test-{tag}-{}-{id}", std::process::id()))
    }

    #[test]
    fn generate_collect_matches_eager() {
        let lazy = generate(1000, |i| i as f64).collect().unwrap();
        let eager = Tensor::from_fn(1000, |i| i as f64);
        assert_eq!(lazy, eager);
    }

    #[test]
    fn many_small_batches_preserve_order_and_values() {
        // 8-byte batches => 1 f64 per batch => 100 batches.
        let out = generate(100, |i| i as f64)
            .with_batch_bytes(8)
            .map(|x| x + 1.0)
            .collect()
            .unwrap();
        assert_eq!(out, Tensor::from_fn(100, |i| (i + 1) as f64));
    }

    #[test]
    fn map_changes_element_type() {
        let out = generate(4, |i| i as f64)
            .map(|x| (x * 10.0) as i64)
            .collect()
            .unwrap();
        assert_eq!(out.as_slice(), &[0, 10, 20, 30]);
    }

    #[test]
    fn sum_stays_lazy_then_streams_and_matches_eager() {
        let plan = generate(10_000, |i| (i % 7) as f64)
            .with_batch_bytes(256)
            .sum(); // nothing has executed yet
        assert!(plan.explain().contains("-> sum"));
        let lazy_sum = plan.collect().unwrap().item();
        let eager_sum = Tensor::from_fn(10_000, |i| (i % 7) as f64).sum();
        assert_eq!(lazy_sum, eager_sum);
    }

    #[test]
    fn sum_is_chainable_like_any_plan() {
        // A reduction yields a 1-element lazy tensor: still mappable.
        let result = generate(10, |i| i as f64)
            .sum()
            .map(|total| total / 10.0)
            .collect()
            .unwrap();
        assert_eq!(result.item(), 4.5);
    }

    #[test]
    fn tensor_lazy_roundtrip() {
        let out = Tensor::from_elements(&[1i32, 2, 3])
            .lazy()
            .map(|x| x * x)
            .collect()
            .unwrap();
        assert_eq!(out.as_slice(), &[1, 4, 9]);
    }

    #[test]
    fn empty_source_yields_empty_results() {
        assert!(generate(0, |i| i as f64).collect().unwrap().is_empty());
        assert_eq!(generate(0, |i| i as f64).sum().collect().unwrap().item(), 0.0);
    }

    #[test]
    fn explain_lists_source_and_ops() {
        let plan = generate(10, |i| i as f64).map(|x| x * 2.0).map(|x| x + 1.0);
        let text = plan.explain();
        assert!(text.contains("generate(len=10)"));
        assert_eq!(text.matches("map").count(), 2);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // file-backed mmap is not supported under miri
    fn scan_file_streams_a_written_file() {
        let path = temp_path("scan");
        let source = Tensor::from_fn(5000, |i| i as f64);
        fs::write(&path, source.storage().as_bytes()).unwrap();

        let total = scan_file::<f64>(&path)
            .with_batch_bytes(1024)
            .map(|x| x * 2.0)
            .sum()
            .collect()
            .unwrap()
            .item();
        assert_eq!(total, 2.0 * source.sum());
        fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn scan_file_of_missing_file_errors_at_execution() {
        let plan = scan_file::<f64>(temp_path("missing")).sum(); // still no error: lazy
        assert!(matches!(plan.collect(), Err(EngineError::Source(_))));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn scan_file_rejects_partial_elements() {
        let path = temp_path("badsize");
        fs::write(&path, [0u8; 5]).unwrap();
        let result = scan_file::<f64>(&path).collect();
        assert!(matches!(result, Err(EngineError::Source(_))));
        fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn sink_file_roundtrips_through_scan() {
        let input = temp_path("sink-in");
        let output = temp_path("sink-out");
        let source = Tensor::from_fn(3000, |i| i as f64);
        fs::write(&input, source.storage().as_bytes()).unwrap();

        // file -> map -> other file, in small batches.
        scan_file::<f64>(&input)
            .with_batch_bytes(512)
            .map(|x| x + 0.5)
            .sink_file(&output)
            .unwrap();

        let result = Tensor::<f64>::map_file(&output).unwrap();
        assert_eq!(result.len(), 3000);
        assert_eq!(result.as_slice()[10], 10.5);
        fs::remove_file(&input).unwrap();
        fs::remove_file(&output).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn sink_into_the_scanned_file_is_refused() {
        let path = temp_path("selfsink");
        fs::write(&path, Tensor::from_elements(&[1.0f64]).storage().as_bytes()).unwrap();
        let result = scan_file::<f64>(&path).sink_file(&path);
        assert!(matches!(result, Err(EngineError::SinkIntoSource)));
        // The source file is intact.
        assert_eq!(Tensor::<f64>::map_file(&path).unwrap().as_slice(), &[1.0]);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn sink_of_generated_plan_writes_readable_file() {
        let path = temp_path("gen-sink");
        generate(100, |i| i as u32).sink_file(&path).unwrap();
        let back = Tensor::<u32>::map_file(&path).unwrap();
        assert_eq!(back, Tensor::from_fn(100, |i| i as u32));
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn engine_error_displays() {
        assert!(EngineError::SinkIntoSource.to_string().contains("sink"));
    }
}
