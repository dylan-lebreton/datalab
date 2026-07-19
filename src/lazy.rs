//! The lazy, streaming execution engine — v2: plan DAG.
//!
//! A [`LazyTensor`] is a **plan**: a description of computations to run
//! later, built by chaining methods (like a Polars `LazyFrame`). The rule is
//! uniform, with no exception to memorize:
//!
//! > **Nothing executes until [`LazyTensor::collect`] or
//! > [`LazyTensor::sink_file`] is called.** Every other method — including
//! > reductions like [`LazyTensor::sum`] and binary operations like
//! > `a + b` — only extends the plan.
//!
//! `collect` runs the plan and materializes the result as a [`Tensor`];
//! `sink_file` runs the plan and streams the result to a file **without ever
//! materializing it**.
//!
//! Execution is *batched*: each source produces small contiguous [`Tensor`]
//! batches (sized in bytes, see [`LazyTensor::with_batch_bytes`]), each
//! operator transforms batches into new batches, and the terminal consumes
//! them one by one. At any instant only a few batches are resident, so
//! memory stays bounded regardless of the source size — a file far larger
//! than RAM streams through comfortably.
//!
//! ```
//! use datalab::lazy;
//!
//! let a = lazy::generate(1_000, |i| i as f64);
//! let b = lazy::generate(1_000, |i| (2 * i) as f64);
//! let total = (a + b)      // still lazy: a two-source plan
//!     .sum()               // still lazy: a 1-element plan
//!     .collect()?          // the only thing that executes
//!     .item();
//! assert_eq!(total, 3.0 * 999.0 * 1_000.0 / 2.0);
//! # Ok::<(), datalab::lazy::EngineError>(())
//! ```
//!
//! # Design notes (and how this scales later)
//!
//! **The plan is data, not types.** The plan is a **node arena** (a `Vec` of
//! nodes addressed by indices) so it can be inspected
//! ([`LazyTensor::explain`]) and, later, optimized (e.g. fusing consecutive
//! `map`s into one pass). The public API stays fully typed
//! (`LazyTensor<T>`); inside, each operation is a type-erased
//! batch-to-batch function whose types were checked at construction.
//!
//! **The plan is internal plumbing, not a user data structure.** A plan is a
//! recipe — a handful of nodes, a few bytes each — while the data it
//! describes may be terabytes. It is deliberately a plain `Vec`, not a
//! datalab structure: the engine never builds itself out of the user-facing
//! types layered on top of it.
//!
//! **Pull now, push later.** Execution is a single-threaded *pull* loop
//! (each terminal drains its upstream streams). This is the simplest
//! correct model, but it cannot parallelize across cores: that requires a
//! *push* (morsel-driven) executor, where sources push batches to a pool of
//! workers — the migration Polars had to make for its streaming engine. The
//! seam is prepared: operators are pure batch-to-batch streams with `Send`
//! bounds that never know who drives them, so moving to a push executor
//! replaces only the driving loop, not the operators.
//!
//! **Tree now, shared DAG later.** Binary operations give a plan multiple
//! sources, so the arena forms a *tree* (every node feeds exactly one
//! consumer). Sharing one subplan between several consumers — a true DAG
//! with fan-out — additionally requires reference-counted edges and batch
//! broadcasting; the arena representation is already shaped for it.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
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

/// A pull-based stream of batches — the execution side of a plan node.
trait BatchStream {
    /// Produces the next batch, `Ok(None)` when the stream is exhausted, or
    /// an error (e.g. two zipped plans disagreeing on length).
    fn next_batch(&mut self) -> Result<Option<Batch>, EngineError>;
}

/// Builds a fresh source [`BatchStream`], given the target batch size in
/// bytes.
type StreamFactory = Box<dyn FnOnce(usize) -> Result<Box<dyn BatchStream>, EngineError> + Send>;

/// Builds the stream of a binary node from its two input streams.
type ZipFactory =
    Box<dyn FnOnce(Box<dyn BatchStream>, Box<dyn BatchStream>) -> Box<dyn BatchStream> + Send>;

/// Builds the stream of a reduction node from its input stream.
type ReduceFactory = Box<dyn FnOnce(Box<dyn BatchStream>) -> Box<dyn BatchStream> + Send>;

/// Index of a node in a plan's arena.
type NodeId = usize;

/// One node of a plan.
enum PlanNode {
    /// Produces batches out of thin air (a file scan, a generator, a
    /// materialized tensor).
    Source {
        /// Human-readable description for [`LazyTensor::explain`].
        label: String,
        /// Set when the source scans a file; used to refuse sinking into it.
        path: Option<PathBuf>,
        /// Instantiates the actual stream when execution starts.
        make: StreamFactory,
    },
    /// Transforms each input batch into one output batch, element-wise.
    Map {
        input: NodeId,
        /// Pure, type-erased batch-to-batch function (types checked at
        /// construction). Never knows who drives it (pull today, push later).
        apply: Box<dyn Fn(Batch) -> Batch + Send>,
    },
    /// Combines two inputs element-wise (re-chunking as needed).
    Zip {
        left: NodeId,
        right: NodeId,
        label: &'static str,
        make: ZipFactory,
    },
    /// Drains its input and emits a single (1-element) batch.
    Reduce {
        input: NodeId,
        label: &'static str,
        make: ReduceFactory,
    },
}

impl PlanNode {
    /// Shifts every child index by `offset` — used when two arenas are
    /// merged by a binary operation.
    fn shift_children(&mut self, offset: usize) {
        match self {
            Self::Source { .. } => {}
            Self::Map { input, .. } | Self::Reduce { input, .. } => *input += offset,
            Self::Zip { left, right, .. } => {
                *left += offset;
                *right += offset;
            }
        }
    }
}

/// The reason a plan failed to execute.
#[derive(Debug)]
pub enum EngineError {
    /// Opening or validating a source failed.
    Source(TensorFileError),
    /// Writing the sink failed.
    Io(io::Error),
    /// The sink path refers to a file the plan is scanning; executing
    /// would overwrite the source while reading it.
    SinkIntoSource,
    /// A binary operation combined two plans that produced different
    /// numbers of elements (detected during execution, when one side ends
    /// before the other).
    LengthMismatch,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(err) => write!(f, "cannot open plan source: {err}"),
            Self::Io(err) => write!(f, "sink failed: {err}"),
            Self::SinkIntoSource => {
                write!(f, "cannot sink into a file the plan is scanning")
            }
            Self::LengthMismatch => {
                write!(
                    f,
                    "length mismatch: one side of a binary operation ended before the other"
                )
            }
        }
    }
}

impl Error for EngineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::SinkIntoSource | Self::LengthMismatch => None,
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
/// Built by [`scan_file`], [`generate`] or [`Tensor::lazy`]; extended with
/// operations like [`LazyTensor::map`], reductions like
/// [`LazyTensor::sum`], and binary operators (`a + b`, `a - b`, `a * b`,
/// [`LazyTensor::zip_with`]); executed by [`collect`](LazyTensor::collect)
/// or [`sink_file`](LazyTensor::sink_file). See the [module docs](self).
#[must_use = "a LazyTensor is only a plan; call collect() or sink_file() to execute it"]
pub struct LazyTensor<T: Element> {
    /// Node arena. Invariant: `root` is a valid index, every node's
    /// children are valid indices of earlier nodes, and every node feeds
    /// exactly one consumer (the plan is a tree).
    nodes: Vec<PlanNode>,
    /// The node producing this plan's output.
    root: NodeId,
    batch_bytes: usize,
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
    let make: StreamFactory = Box::new(move |batch_bytes| {
        let storage = Storage::map_file(&make_path).map_err(TensorFileError::Io)?;
        // Validate up front that the bytes form whole elements.
        View::<T>::new(&storage).map_err(TensorFileError::View)?;
        Ok(Box::new(StorageStream::<T> {
            storage,
            pos: 0,
            batch_elems: batch_elems::<T>(batch_bytes),
            _elem: PhantomData,
        }))
    });
    LazyTensor {
        nodes: vec![PlanNode::Source {
            label,
            path: Some(path),
            make,
        }],
        root: 0,
        batch_bytes: DEFAULT_BATCH_BYTES,
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
    let make: StreamFactory = Box::new(move |batch_bytes| {
        Ok(Box::new(GenerateStream {
            f,
            next: 0,
            len,
            batch_elems: batch_elems::<T>(batch_bytes),
            _elem: PhantomData,
        }))
    });
    LazyTensor {
        nodes: vec![PlanNode::Source {
            label,
            path: None,
            make,
        }],
        root: 0,
        batch_bytes: DEFAULT_BATCH_BYTES,
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
        let make: StreamFactory = Box::new(move |batch_bytes| {
            Ok(Box::new(StorageStream::<T> {
                storage: self.into_storage(),
                pos: 0,
                batch_elems: batch_elems::<T>(batch_bytes),
                _elem: PhantomData,
            }))
        });
        LazyTensor {
            nodes: vec![PlanNode::Source {
                label,
                path: None,
                make,
            }],
            root: 0,
            batch_bytes: DEFAULT_BATCH_BYTES,
            _out: PhantomData,
        }
    }
}

impl<T: Element> LazyTensor<T> {
    /// Sets the target batch size in bytes (default
    /// [`DEFAULT_BATCH_BYTES`]). Clamped so a batch holds at least one
    /// element.
    ///
    /// The setting configures the **whole plan**, wherever it is called in
    /// the chain. When two plans are combined by a binary operation, the
    /// smaller of the two settings wins (the tighter memory bound).
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
        let apply = Box::new(move |batch: Batch| -> Batch {
            let input = batch
                .downcast::<Tensor<T>>()
                .expect("engine invariant: batch type matches the plan chain");
            Box::new(input.map(&f))
        });
        self.nodes.push(PlanNode::Map {
            input: self.root,
            apply,
        });
        LazyTensor {
            root: self.nodes.len() - 1,
            nodes: self.nodes,
            batch_bytes: self.batch_bytes,
            _out: PhantomData,
        }
    }

    /// Combines two plans element-wise with `f`, yielding a lazy plan of
    /// the results.
    ///
    /// The element types of the two sides (and of the output) may all
    /// differ. During execution the two streams are pulled in lockstep and
    /// **re-chunked**: their batches need not line up, the zip consumes the
    /// overlap of the current batches and buffers the remainder — memory
    /// stays bounded.
    ///
    /// The `+`, `-` and `*` operators are shorthands for `zip_with` with
    /// the matching arithmetic (backed by the SIMD-friendly
    /// [`kernel`] loops).
    ///
    /// # Errors (at execution)
    ///
    /// Executing the combined plan fails with
    /// [`EngineError::LengthMismatch`] if the two sides produce different
    /// numbers of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::lazy;
    ///
    /// let a = lazy::generate(3, |i| i as f64);
    /// let b = lazy::generate(3, |i| (i + 1) as u32);
    /// let ratio = a.zip_with(b, |x, y| x / f64::from(y)).collect()?;
    /// assert_eq!(ratio.as_slice(), &[0.0, 0.5, 2.0 / 3.0]);
    /// # Ok::<(), datalab::lazy::EngineError>(())
    /// ```
    pub fn zip_with<U: Element, V: Element>(
        self,
        other: LazyTensor<U>,
        f: impl Fn(T, U) -> V + Send + Sync + 'static,
    ) -> LazyTensor<V> {
        self.zip_nodes(other, "zip_with", move |l, r| {
            Tensor::from_fn(l.len(), |i| f(l[i], r[i]))
        })
    }

    /// Merges `other`'s arena into this plan's and roots a `Zip` node over
    /// both — the shared implementation of every binary operation. `f`
    /// combines two same-length windows into one output batch.
    fn zip_nodes<U: Element, V: Element>(
        mut self,
        other: LazyTensor<U>,
        label: &'static str,
        f: impl Fn(&[T], &[U]) -> Tensor<V> + Send + 'static,
    ) -> LazyTensor<V> {
        let offset = self.nodes.len();
        let right_root = other.root + offset;
        self.nodes.extend(other.nodes.into_iter().map(|mut node| {
            node.shift_children(offset);
            node
        }));
        let make: ZipFactory = Box::new(move |left, right| {
            Box::new(ZipStream {
                left: Chunked::<T>::new(left),
                right: Chunked::<U>::new(right),
                f,
                _out: PhantomData::<V>,
            })
        });
        self.nodes.push(PlanNode::Zip {
            left: self.root,
            right: right_root,
            label,
            make,
        });
        LazyTensor {
            root: self.nodes.len() - 1,
            nodes: self.nodes,
            batch_bytes: self.batch_bytes.min(other.batch_bytes),
            _out: PhantomData,
        }
    }

    /// Renders the plan as a human-readable tree, one node per line, with
    /// the **last** operation first (the root) and sources as leaves.
    ///
    /// Terminals are not part of the stored plan (they are the call that
    /// executes it).
    ///
    /// # Examples
    ///
    /// ```
    /// use datalab::lazy;
    ///
    /// let a = lazy::generate(10, |i| i as f64).map(|x| x * 2.0);
    /// let b = lazy::generate(10, |i| i as f64);
    /// let text = (a + b).sum().explain();
    /// assert!(text.contains("sum"));
    /// assert!(text.contains("add"));
    /// assert!(text.contains("map"));
    /// assert!(text.contains("generate"));
    /// ```
    #[must_use]
    pub fn explain(&self) -> String {
        let mut out = String::new();
        render(&self.nodes, self.root, "", &mut out);
        out.pop(); // trailing newline
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
    /// Returns [`EngineError`] if a source cannot be opened or validated,
    /// or if a binary operation combined plans of different lengths.
    pub fn collect(self) -> Result<Tensor<T>, EngineError> {
        let mut batches: Vec<Tensor<T>> = Vec::new();
        self.run(|batch| {
            batches.push(batch);
            Ok(())
        })?;

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
    /// # Examples
    ///
    /// ```
    /// use datalab::lazy;
    ///
    /// let plan = lazy::generate(100, |i| i as u64).sum(); // still lazy
    /// assert_eq!(plan.collect()?.item(), 4950);
    /// # Ok::<(), datalab::lazy::EngineError>(())
    /// ```
    pub fn sum(mut self) -> LazyTensor<T>
    where
        T: Add<Output = T> + Default,
    {
        let make: ReduceFactory = Box::new(|inner| {
            Box::new(SumStream::<T> {
                inner: Some(inner),
                _elem: PhantomData,
            })
        });
        self.nodes.push(PlanNode::Reduce {
            input: self.root,
            label: "sum",
            make,
        });
        LazyTensor {
            root: self.nodes.len() - 1,
            nodes: self.nodes,
            batch_bytes: self.batch_bytes,
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
    /// Returns [`EngineError`] if a source cannot be opened, if writing
    /// fails, if a binary operation combined plans of different lengths, or
    /// if `path` refers to any file the plan scans — also through symlinks
    /// or hard links ([`EngineError::SinkIntoSource`] — sinking into a
    /// source would overwrite it while reading it).
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
        let sinks_into_source = self.nodes.iter().any(|node| {
            matches!(node, PlanNode::Source { path: Some(source), .. } if is_same_file(source, path))
        });
        if sinks_into_source {
            return Err(EngineError::SinkIntoSource);
        }

        let mut file = fs::File::create(path)?;
        self.run(|batch| {
            file.write_all(batch.storage().as_bytes())
                .map_err(EngineError::Io)
        })?;
        file.flush()?;
        Ok(())
    }

    /// The pull loop: builds the stream tree from the arena, drains it, and
    /// hands each resulting batch to `consume`; an error from `consume`
    /// aborts the drain immediately (e.g. a full disk stops a sink at the
    /// first failed write). This is the only place that drives execution —
    /// swapping it for a push (parallel) executor later leaves sources,
    /// operators and terminals untouched.
    fn run(
        self,
        mut consume: impl FnMut(Tensor<T>) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let mut nodes: Vec<Option<PlanNode>> = self.nodes.into_iter().map(Some).collect();
        let mut stream = build_stream(&mut nodes, self.root, self.batch_bytes)?;
        while let Some(batch) = stream.next_batch()? {
            let tensor = batch
                .downcast::<Tensor<T>>()
                .expect("engine invariant: final batch type matches the plan output");
            consume(*tensor)?;
        }
        Ok(())
    }
}

/// Element-wise addition of two plans: `a + b`, lazily.
///
/// Operands are consumed (plans are single-use recipes). Executing the
/// result fails with [`EngineError::LengthMismatch`] if the operands
/// produce different numbers of elements.
impl<T: Element + Add<Output = T>> Add for LazyTensor<T> {
    type Output = LazyTensor<T>;

    fn add(self, rhs: Self) -> LazyTensor<T> {
        self.zip_nodes(rhs, "add", |l, r| {
            let mut out = Tensor::zeros(l.len());
            kernel::add(l, r, out.as_mut_slice());
            out
        })
    }
}

/// Element-wise subtraction of two plans: `a - b`, lazily.
///
/// Operands are consumed (plans are single-use recipes). Executing the
/// result fails with [`EngineError::LengthMismatch`] if the operands
/// produce different numbers of elements.
impl<T: Element + Sub<Output = T>> Sub for LazyTensor<T> {
    type Output = LazyTensor<T>;

    fn sub(self, rhs: Self) -> LazyTensor<T> {
        self.zip_nodes(rhs, "sub", |l, r| {
            let mut out = Tensor::zeros(l.len());
            kernel::sub(l, r, out.as_mut_slice());
            out
        })
    }
}

/// Element-wise product of two plans: `a * b`, lazily.
///
/// Operands are consumed (plans are single-use recipes). Executing the
/// result fails with [`EngineError::LengthMismatch`] if the operands
/// produce different numbers of elements.
impl<T: Element + Mul<Output = T>> Mul for LazyTensor<T> {
    type Output = LazyTensor<T>;

    fn mul(self, rhs: Self) -> LazyTensor<T> {
        self.zip_nodes(rhs, "mul", |l, r| {
            let mut out = Tensor::zeros(l.len());
            kernel::mul(l, r, out.as_mut_slice());
            out
        })
    }
}

/// Recursively instantiates the stream of node `id`, taking each node out
/// of the arena (every node is used exactly once — the plan is a tree).
fn build_stream(
    nodes: &mut [Option<PlanNode>],
    id: NodeId,
    batch_bytes: usize,
) -> Result<Box<dyn BatchStream>, EngineError> {
    let node = nodes[id]
        .take()
        .expect("plan invariant: every node feeds exactly one consumer");
    match node {
        PlanNode::Source { make, .. } => make(batch_bytes),
        PlanNode::Map { input, apply } => Ok(Box::new(MapStream {
            inner: build_stream(nodes, input, batch_bytes)?,
            apply,
        })),
        PlanNode::Zip {
            left, right, make, ..
        } => {
            let left = build_stream(nodes, left, batch_bytes)?;
            let right = build_stream(nodes, right, batch_bytes)?;
            Ok(make(left, right))
        }
        PlanNode::Reduce { input, make, .. } => {
            Ok(make(build_stream(nodes, input, batch_bytes)?))
        }
    }
}

/// Writes the `explain` line of node `id` (label, then children indented
/// under `prefix` with tree connectors).
fn render(nodes: &[PlanNode], id: NodeId, prefix: &str, out: &mut String) {
    let (label, children): (&str, Vec<NodeId>) = match &nodes[id] {
        PlanNode::Source { label, .. } => (label, Vec::new()),
        PlanNode::Map { input, .. } => ("map", vec![*input]),
        PlanNode::Zip {
            left, right, label, ..
        } => (label, vec![*left, *right]),
        PlanNode::Reduce { input, label, .. } => (label, vec![*input]),
    };
    out.push_str(label);
    out.push('\n');
    let mut iter = children.into_iter().peekable();
    while let Some(child) = iter.next() {
        let last = iter.peek().is_none();
        out.push_str(prefix);
        out.push_str(if last { "└─ " } else { "├─ " });
        let child_prefix = format!("{prefix}{}", if last { "   " } else { "│  " });
        render(nodes, child, &child_prefix, out);
    }
}

/// Computes how many `T` elements fit the byte target (at least one).
fn batch_elems<T: Element>(batch_bytes: usize) -> usize {
    (batch_bytes / size_of::<T>()).max(1)
}

/// Returns `true` when `a` and `b` refer to the same underlying file — also
/// through symlinks or hard links. Best-effort: `false` when either path
/// cannot be inspected (e.g. it does not exist yet).
fn is_same_file(a: &Path, b: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (fs::metadata(a), fs::metadata(b)) {
            (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        matches!(
            (fs::canonicalize(a), fs::canonicalize(b)),
            (Ok(a), Ok(b)) if a == b
        )
    }
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
    fn next_batch(&mut self) -> Result<Option<Batch>, EngineError> {
        let total = self.storage.len() / size_of::<T>();
        if self.pos >= total {
            return Ok(None);
        }
        let end = (self.pos + self.batch_elems).min(total);
        let window = self
            .storage
            .slice(self.pos * size_of::<T>(), (end - self.pos) * size_of::<T>());
        self.pos = end;
        Ok(Some(Box::new(Tensor::<T>::from_storage(window))))
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
    fn next_batch(&mut self) -> Result<Option<Batch>, EngineError> {
        if self.next >= self.len {
            return Ok(None);
        }
        let start = self.next;
        let end = (start + self.batch_elems).min(self.len);
        let batch = Tensor::from_fn(end - start, |i| (self.f)(start + i));
        self.next = end;
        Ok(Some(Box::new(batch)))
    }
}

/// Stream applying a type-erased batch-to-batch function to its input.
struct MapStream {
    inner: Box<dyn BatchStream>,
    apply: Box<dyn Fn(Batch) -> Batch + Send>,
}

impl BatchStream for MapStream {
    fn next_batch(&mut self) -> Result<Option<Batch>, EngineError> {
        Ok(self.inner.next_batch()?.map(|batch| (self.apply)(batch)))
    }
}

/// Stream that drains its input on the first pull, summing every batch,
/// then yields the single 1-element result.
struct SumStream<T: Element> {
    /// Taken on the first pull; `None` afterwards.
    inner: Option<Box<dyn BatchStream>>,
    _elem: PhantomData<T>,
}

impl<T: Element + Add<Output = T> + Default> BatchStream for SumStream<T> {
    fn next_batch(&mut self) -> Result<Option<Batch>, EngineError> {
        let Some(mut inner) = self.inner.take() else {
            return Ok(None);
        };
        let mut total = T::default();
        while let Some(batch) = inner.next_batch()? {
            let tensor = batch
                .downcast::<Tensor<T>>()
                .expect("engine invariant: batch type matches the plan chain");
            total = total + kernel::sum(tensor.as_slice());
        }
        Ok(Some(Box::new(Tensor::from_elements(&[total]))))
    }
}

/// Adapter turning a batch stream into a stream of typed windows that can
/// be consumed at any granularity: [`Chunked::peek`] exposes the unconsumed
/// window of the current batch (pulling the next batch as needed) and
/// [`Chunked::advance`] marks elements as consumed. At most one upstream
/// batch is held at a time, so memory stays bounded.
struct Chunked<T: Element> {
    stream: Box<dyn BatchStream>,
    /// The batch currently being consumed, and how many of its elements
    /// have been consumed already.
    pending: Option<(Tensor<T>, usize)>,
    /// Set once the upstream stream is exhausted.
    done: bool,
}

impl<T: Element> Chunked<T> {
    fn new(stream: Box<dyn BatchStream>) -> Self {
        Self {
            stream,
            pending: None,
            done: false,
        }
    }

    /// Returns the current unconsumed window, pulling from the upstream as
    /// needed; `None` means the stream is exhausted.
    fn peek(&mut self) -> Result<Option<&[T]>, EngineError> {
        loop {
            let consumed = self
                .pending
                .as_ref()
                .is_none_or(|(batch, offset)| *offset >= batch.len());
            if !consumed {
                break;
            }
            if self.done {
                self.pending = None;
                return Ok(None);
            }
            match self.stream.next_batch()? {
                Some(batch) => {
                    let tensor = batch
                        .downcast::<Tensor<T>>()
                        .expect("engine invariant: batch type matches the plan chain");
                    self.pending = Some((*tensor, 0));
                }
                None => self.done = true,
            }
        }
        Ok(self
            .pending
            .as_ref()
            .map(|(batch, offset)| &batch.as_slice()[*offset..]))
    }

    /// Marks the first `n` elements of the current window as consumed.
    fn advance(&mut self, n: usize) {
        if let Some((_, offset)) = &mut self.pending {
            *offset += n;
        }
    }
}

/// Stream combining two upstreams element-wise. The upstreams' batches need
/// not line up: each pull consumes the overlap of the two current batches
/// (re-chunking), so the output batch size is the smaller of the two
/// windows and memory stays bounded.
struct ZipStream<T: Element, U: Element, V: Element, F> {
    left: Chunked<T>,
    right: Chunked<U>,
    /// Combines two same-length windows into one output batch.
    f: F,
    _out: PhantomData<V>,
}

impl<T, U, V, F> BatchStream for ZipStream<T, U, V, F>
where
    T: Element,
    U: Element,
    V: Element,
    F: Fn(&[T], &[U]) -> Tensor<V> + Send,
{
    fn next_batch(&mut self) -> Result<Option<Batch>, EngineError> {
        let (out, n) = match (self.left.peek()?, self.right.peek()?) {
            (None, None) => return Ok(None),
            (Some(_), None) | (None, Some(_)) => return Err(EngineError::LengthMismatch),
            (Some(l), Some(r)) => {
                let n = l.len().min(r.len());
                ((self.f)(&l[..n], &r[..n]), n)
            }
        };
        self.left.advance(n);
        self.right.advance(n);
        Ok(Some(Box::new(out)))
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
        assert!(plan.explain().contains("sum"));
        let lazy_sum = plan.collect().unwrap().item();
        let eager_sum = Tensor::from_fn(10_000, |i| (i % 7) as f64).sum();
        assert_eq!(lazy_sum, eager_sum);
    }

    #[test]
    fn with_batch_bytes_after_sum_configures_the_whole_plan() {
        // Same plan method rules everywhere: configuring the batch size
        // after the reduction still drives the inner drain.
        let total = generate(100, |i| i as u64)
            .sum()
            .with_batch_bytes(8) // 1 u64 per batch
            .collect()
            .unwrap()
            .item();
        assert_eq!(total, 4950);
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
    fn explain_renders_binary_plans_as_a_tree() {
        let a = generate(4, |i| i as f64);
        let b = generate(4, |i| i as f64).map(|x| x * 2.0);
        let text = (a + b).sum().explain();
        assert!(text.contains("sum"));
        assert!(text.contains("add"));
        assert!(text.contains("├─"), "binary nodes branch: {text}");
        assert!(text.contains("└─"), "last children close the branch: {text}");
        assert_eq!(text.matches("generate(len=4)").count(), 2);
    }

    #[test]
    fn lazy_binary_ops_match_eager() {
        let ea = Tensor::from_fn(1000, |i| (i % 13) as f64);
        let eb = Tensor::from_fn(1000, |i| (i % 7) as f64);
        // 64-byte batches (8 f64) exercise the re-chunking path.
        let lazy = |t: &Tensor<f64>| t.clone().lazy().with_batch_bytes(64);
        assert_eq!((lazy(&ea) + lazy(&eb)).collect().unwrap(), &ea + &eb);
        assert_eq!((lazy(&ea) - lazy(&eb)).collect().unwrap(), &ea - &eb);
        assert_eq!((lazy(&ea) * lazy(&eb)).collect().unwrap(), &ea * &eb);
    }

    #[test]
    fn zip_with_rechunks_streams_of_different_batch_granularity() {
        // With 64-byte batches, f64 batches hold 8 elements but u8 batches
        // hold 64: the zip must re-chunk to the overlap.
        let a = generate(100, |i| i as f64).with_batch_bytes(64);
        let b = generate(100, |i| i as u8).with_batch_bytes(64);
        let out = a.zip_with(b, |x, y| x + f64::from(y)).collect().unwrap();
        assert_eq!(out, Tensor::from_fn(100, |i| 2.0 * i as f64));
    }

    #[test]
    fn binary_op_on_plans_of_different_lengths_errors() {
        let short_left = (generate(5, |i| i as f64) + generate(7, |i| i as f64)).collect();
        assert!(matches!(short_left, Err(EngineError::LengthMismatch)));
        let short_right = (generate(7, |i| i as f64) + generate(5, |i| i as f64)).collect();
        assert!(matches!(short_right, Err(EngineError::LengthMismatch)));
    }

    #[test]
    fn binary_op_of_empty_plans_is_empty() {
        let out = (generate(0, |i| i as f64) + generate(0, |i| i as f64))
            .collect()
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn map_after_binary_op_transforms_the_combined_stream() {
        let out = (generate(4, |i| i as i64) + generate(4, |i| i as i64))
            .map(|x| x * 10)
            .collect()
            .unwrap();
        assert_eq!(out.as_slice(), &[0, 20, 40, 60]);
    }

    #[test]
    fn zero_batch_bytes_still_makes_progress() {
        // The batch size is clamped to at least one element.
        let out = generate(3, |i| i as u64).with_batch_bytes(0).collect().unwrap();
        assert_eq!(out.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn reductions_compose_with_binary_ops() {
        // Two 1-element plans (reductions) can be combined like any others.
        let diff = (generate(10, |i| i as f64).sum() - generate(10, |_| 1.0).sum())
            .collect()
            .unwrap()
            .item();
        assert_eq!(diff, 45.0 - 10.0);
    }

    #[test]
    fn sum_after_binary_op_streams_and_matches_eager() {
        let total = (generate(10_000, |i| i as f64).with_batch_bytes(256)
            + generate(10_000, |i| i as f64))
        .sum()
        .collect()
        .unwrap()
        .item();
        assert_eq!(total, 9_999.0 * 10_000.0);
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
    #[cfg(unix)]
    #[cfg_attr(miri, ignore)]
    fn sink_into_a_hard_link_of_the_scanned_file_is_refused() {
        let path = temp_path("hardlink-src");
        let link = temp_path("hardlink-dst");
        fs::write(&path, Tensor::from_elements(&[1.0f64]).storage().as_bytes()).unwrap();
        fs::hard_link(&path, &link).unwrap();
        let result = scan_file::<f64>(&path).sink_file(&link);
        assert!(matches!(result, Err(EngineError::SinkIntoSource)));
        // The source file is intact.
        assert_eq!(Tensor::<f64>::map_file(&path).unwrap().as_slice(), &[1.0]);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&link).unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn sink_into_any_source_of_a_binary_plan_is_refused() {
        let a = temp_path("bin-a");
        let b = temp_path("bin-b");
        fs::write(&a, Tensor::from_elements(&[1.0f64]).storage().as_bytes()).unwrap();
        fs::write(&b, Tensor::from_elements(&[2.0f64]).storage().as_bytes()).unwrap();
        let plan = scan_file::<f64>(&a) + scan_file::<f64>(&b);
        assert!(matches!(plan.sink_file(&b), Err(EngineError::SinkIntoSource)));
        // Both source files are intact.
        assert_eq!(Tensor::<f64>::map_file(&b).unwrap().as_slice(), &[2.0]);
        fs::remove_file(&a).unwrap();
        fs::remove_file(&b).unwrap();
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
        assert!(EngineError::LengthMismatch.to_string().contains("length"));
    }
}
