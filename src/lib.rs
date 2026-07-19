//! # datalab
//!
//! Highly-optimized, streaming-capable data structures for Rust
//! (trees, tensors, stacks, and more).
//!
//! **Status:** early development — the public API is not stable yet.

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(clippy::undocumented_unsafe_blocks)]

pub mod kernel;
pub mod storage;
pub mod tensor;
pub mod view;
