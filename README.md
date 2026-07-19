# datalab

[![Crates.io](https://img.shields.io/crates/v/datalab.svg)](https://crates.io/crates/datalab)
[![Docs.rs](https://docs.rs/datalab/badge.svg)](https://docs.rs/datalab)
[![License: Apache-2.0 OR MIT](https://img.shields.io/crates/l/datalab.svg)](#license)

**Highly-optimized, streaming-capable data structures for Rust — trees, tensors, and more.**

> **Status: pre-alpha.** datalab is in early development. The foundations are
> being built brick by brick and the public API is **not stable yet**. The
> `0.1.0` release on crates.io reserves the name; it does not yet provide a
> usable API. Watch this space.

---

## Table of contents

- [Why datalab](#why-datalab)
- [Vision](#vision)
- [Use cases](#use-cases)
- [Design principles](#design-principles)
- [Architecture](#architecture)
- [Data types](#data-types)
- [Roadmap](#roadmap)
- [Getting started](#getting-started)
- [Development conventions](#development-conventions)
- [License](#license)

---

## Why datalab

Anyone who has done serious data work has hit the same wall: **the data does not
fit in memory.** In NumPy or pandas, a tensor or a dataframe that is slightly too
big for RAM simply crashes the process. The usual answers — buy more RAM, rent a
cluster, rewrite everything — are expensive and out of reach for many.

[Polars](https://pola.rs/) showed a better way for dataframes: a **lazy,
streaming, out-of-core engine** that processes data in bounded memory. It is
excellent — but it is limited to **tabular data**, and in practice its streaming
still *breaks* on certain operations, at which point memory blows up anyway.

Meanwhile, entire fields work with data that is **not** a table:

- **AI / ML** needs huge dense **tensors** (model weights, embeddings, training
  batches) that routinely exceed RAM and VRAM.
- **Genetics / bioinformatics** needs **trees** (phylogenies) and **sparse
  matrices** far bigger than memory.
- **Geospatial** work needs **points, paths, trajectories** and spatial indexes.

These communities deserve the same superpower Polars gives to dataframes.
**datalab is that superpower, generalized.**

## Vision

The genius of Polars is not the DataFrame — it is the **engine** underneath
(a lazy query plan, an optimizer, and a streaming out-of-core executor over
chunked storage). The DataFrame is just one *frontend* bolted on top.

datalab **decouples that engine from the tabular frontend** and exposes it to
many domains:

> **One streaming engine. Many data structures.**
> Tensors, trees, graphs, geospatial types, and dataframes — all built on a
> single out-of-core core, with an API that is **lazy when you need to stream
> and eager when you have the RAM.**

Streaming is **adaptive**, never mandatory:

- **Enough RAM?** → eager execution, maximum performance.
- **Not enough?** → streaming from disk: slower, but it *completes* instead of
  crashing.

This is a deliberate act of **democratization**: an expert on a modest laptop
should be able to run the computation their science requires, even if it takes
longer, without a cluster budget.

## Use cases

| Domain | Structure | Pain today | datalab |
|---|---|---|---|
| AI / ML | Tensors | Weights/embeddings exceed RAM/VRAM | Streamable, disk-resident tensors |
| Genetics | Trees, sparse matrices | Phylogenies too large for memory | Out-of-core trees & sparse data |
| Geospatial | Points, trajectories | Fragmented, slow tooling | Native geo types + spatial indexes |
| Data analysis | DataFrames | Polars streaming can break | Never-break streaming dataframes |

## A taste

```rust
use datalab::lazy;

// A binary file of f32 weights, potentially far larger than RAM.
// Nothing executes until collect() or sink_file() — no exceptions.
let norm = lazy::scan_file::<f32>("model-weights.bin")
    .map(|w| w * w)
    .sum()          // still lazy: a 1-element plan
    .collect()?     // executes: streams in small batches, bounded memory
    .item();

lazy::scan_file::<f32>("model-weights.bin")
    .map(|w| w * 0.5)
    .sink_file("weights-halved.bin")?;  // file -> file, never materialized
```

## Design principles

1. **Streaming is a transverse property, not a feature of one structure.**
   Every structure is designed to run out-of-core from day one.

2. **Streaming never breaks — by construction.** "Seeing all the data" means
   *processing* it, not *holding* it in RAM. No operator is admitted to the
   engine unless it has a bounded-memory (spillable) execution path, using
   proven external-memory algorithms (external merge sort, partitioned hash
   join, external aggregation). This is the founding invariant.

3. **Separate storage from interpretation.** Raw bytes (`Storage`) are one
   thing; reading them as typed, shaped values (a view) is another. The same
   bytes can be interpreted many ways, shared, or spilled — without copies.

4. **Separate storage from compute.** *Where the bytes live* (RAM, mmap, spill,
   shared memory) and *who computes on them* (one core, many cores, a GPU, and
   — much later — a cluster) are two pluggable seams. Chunks stream between them.

5. **Trees and graphs are views over tensors of indices.** Following the
   idiomatic Rust arena model, structure is expressed as index tensors
   (CSR-like), which unifies trees, graphs, and sparse tensors — and makes them
   streamable, because their very definition lives in streamable storage.

6. **Small core, composable everything.** A tiny set of primitives plus
   composition; the large "zoo" of logical data types is added as **extension
   types** without touching the core.

7. **Single machine first, cluster later.** Build a best-in-class single-node,
   chunk-oriented engine. Because it is already chunk-oriented, a future
   distributed layer can orchestrate on top without a rewrite. We do not pretend
   distribution is free — it is a deliberate, later decision.

## Architecture

datalab is organized in layers. Higher layers depend only on the abstractions
of lower layers, never on their internals.

```
   Frontends    Tensor · Tree/Graph · Geo · DataFrame     nice APIs per domain
   ─────────────────────────────────────────────────────
   Engine       lazy plan + optimizer + streaming exec    never-break out-of-core
   ─────────────────────────────────────────────────────
   Structures   tensor (view + shape) · tree (indices)
   ─────────────────────────────────────────────────────
   View         typed lens: read bytes as f64/i32/bool…
   ─────────────────────────────────────────────────────
   Storage      contiguous bytes (+ pluggable backing)    the atom
   ─────────────────────────────────────────────────────
   Bytes        alignment, bit-packing, memory
```

- **Storage** — owns a contiguous region of raw, untyped bytes behind one
  stable API, with pluggable backings: aligned RAM or a read-only
  memory-mapped file, with explicit spill-to-disk and promote-to-heap
  transitions between them (shared memory later). Allocations are
  reference-counted (Arrow-style): clones and slices are zero-copy, and
  mutation requires unique ownership or an explicit copy-on-write
  (`make_mut`) — cheap things are silent, costly things are explicit.
- **View** — a typed, shaped, non-owning lens over a `Storage`. This is where a
  logical data type interprets the bytes, and where the RAM-vs-out-of-core
  duality lives (resident random-access view vs streaming chunk iterator).
- **Engine** — turns a lazy plan into a streaming execution over chunks, with a
  memory budget and spill-to-disk.
- **Frontends** — translate domain operations into engine plans.

### On Apache Arrow

datalab aims to be **Arrow-*interoperable* at the boundaries** (to exchange with
Polars, read Parquet, etc.) rather than **Arrow-*native* at the core**. Arrow is
tabular-first; datalab's core is tensor-first and out-of-core-native. Interop is
an edge adapter to be added when the DataFrame frontend is built.

## Data types

The dtype system is a small physical core plus a large set of logical
**extension types** (a physical layout + logical semantics), keeping the core
minimal.

- **Base (Polars/Arrow parity):** integers (incl. 128-bit), unsigned, floats,
  `Decimal`, `Bool`, `String`, `Binary`, temporal (`Date`, `Datetime`,
  `Duration`, `Time`), nested (`List`, `Array`, `Struct`, `Map`),
  `Categorical`/`Enum`, `Null`.
- **AI-oriented:** `f16`/`bf16`, `FP8`/`FP4`, quantized types (scale/zero-point),
  fixed-size **embedding/vector** types with similarity operations.
- **Temporal, extended:** an **anchored interval** type — a dated interval that
  knows its start *and* end (not just a magnitude), with interval-algebra
  operations (overlap, contains, intersection).
- **Geospatial:** `Point` (lon/lat), `LineString`, and **spatiotemporal
  trajectories** (a point moving over time), with CRS metadata and spatial
  indexing.
- **Scientific:** quantities with **units** and dimensional analysis, complex
  numbers, sparse vectors/matrices.

## Roadmap

- [x] Project setup, dual license, first crates.io release (name reserved)
- [x] `Storage` — aligned, contiguous byte storage (RAM-backed)
- [x] `View` / `ViewMut` — typed, zero-copy interpretation of bytes
- [x] `Tensor` — owned, contiguous, typed 1-D tensor
- [x] Element-wise kernels + benchmarks (criterion)
- [x] Pluggable backing store (mmap, spill-to-disk)
- [x] `Tensor::map_file` / `Tensor::spill_to_disk` — disk-resident tensors,
      larger than RAM
- [x] Lazy engine v1 — plan-as-data, batched pull execution
      (`scan_file` / `.lazy()` / `generate` → `map` → `collect` / `sum` /
      `sink_file`), bounded memory end to end
- [x] Engine v2: plan DAG — node-arena plans, lazy element-wise binary
      operations (`a + b`, `zip_with`) with re-chunking zip streams
- [x] Engine v2: parallel execution — morsel-parallel stages (producer,
      worker pool, ordered reassembly), results identical whatever the
      thread count
- [ ] Engine v2: pipeline breakers with spill (sort, group_by, join)
- [ ] N-D tensor (shape/strides)
- [ ] Trees as views over index tensors
- [ ] Additional frontends (DataFrame, Geo) and Arrow interop

## Getting started

datalab requires **Rust 1.94+** (edition 2024).

```bash
# build
cargo build

# run tests (unit + doctests)
cargo test

# build the documentation
cargo doc --open
```

Performance-sensitive work must be measured in release mode
(`cargo test --release`, `cargo bench`). This repository is configured for
optimized local builds (fat LTO, single codegen unit, and `target-cpu=native`
via `.cargo/config.toml`).

## Development conventions

**Language.** All in-repository text — code, comments, doc comments, README,
commit messages — is written in **English**.

**Commits.** [Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): message` (e.g. `feat(storage): add aligned backing`).

**Documentation.** Documentation is mandatory: the crate enables
`#![deny(missing_docs)]`, so every public item must be documented, and every
public item carries a runnable `# Examples` doctest. The public API follows the
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

**Quality bar.** Code must pass `cargo test` (including doctests) and
`cargo clippy --all-targets` with no warnings before being committed. `unsafe`
is confined, justified with a `// SAFETY:` comment (enforced by lint), wrapped
in a safe API, and checked with [Miri](https://github.com/rust-lang/miri).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
