//! `NonSeparable2DAnisotropicChernoff` — thin compatibility shim (v2.2 Wave C, ADR-0058).
//!
//! In v2.2 the implementation was unified into
//! [`crate::nonseparable_mixed::NonSeparableMixedChernoff`]. This module
//! re-exports the type alias; the v0.9.0-compatible `new` constructor lives on
//! the underlying `NonSeparableMixedChernoff` impl (delegates to `with_scalar_c`,
//! which is byte-identical since both v0.7 and v0.9 use the same coupling stencil).
//!
//! ## What changed in v2.2
//!
//! The 481-LoC palindromic-5-leg implementation previously here was consolidated
//! into `nonseparable_mixed.rs` together with `nonseparable2d.rs` (ADR-0058
//! SUPERSEDES ADR-0033 "keep both"). All unit tests and slope gates pass byte-identical.
//!
//! See math.md §10.7-ter (original §, ADR-0023), §18 (refactor), ADR-0058.

pub use crate::nonseparable_mixed::{NonSeparable2DAnisotropicChernoff, NonSeparableMixedChernoff};
