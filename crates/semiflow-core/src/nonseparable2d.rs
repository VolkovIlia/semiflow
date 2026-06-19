//! `NonSeparable2DChernoff` — thin compatibility shim (v2.2 Wave C, ADR-0058).
//!
//! In v2.2 the implementation was unified into
//! [`crate::nonseparable_mixed::NonSeparableMixedChernoff`]. This module
//! re-exports the type alias; the v0.7.0-compatible `new` constructor lives on
//! the underlying `NonSeparableMixedChernoff` impl (delegates to `with_scalar_c`).
//!
//! ## What changed in v2.2
//!
//! The 514-LoC palindromic-5-leg implementation previously here was consolidated
//! into `nonseparable_mixed.rs` together with `nonseparable2d_aniso.rs` (ADR-0058
//! SUPERSEDES ADR-0033 "keep both"). All unit tests and slope gates pass byte-identical.
//!
//! See math.md §10.7-bis (original §), §10.7-ter (generalisation), §18 (refactor).

pub use crate::nonseparable_mixed::{NonSeparable2DChernoff, NonSeparableMixedChernoff};
