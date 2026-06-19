//! [`SemiflowStatus`] — C-compatible status codes for FFI entry points.
//!
//! Every `extern "C"` function returns one of these variants.  The enum is
//! `#[repr(C)]` and **not** `#[non_exhaustive]` — the C ABI requires a stable
//! integer representation.

use semiflow_core::SemiflowError;

/// Status codes returned by all `smf_*` C functions.
///
/// Integer values are stable ABI; do not reorder.  New variants (gap or
/// append) require a major version bump per ADR-0028.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemiflowStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// Grid geometry mismatch (e.g. `n < 4`, `xmin >= xmax`, wrong length).
    GridMismatch = 1,
    /// Non-finite value (NaN or Inf) encountered in input data.
    NanInf = 2,
    /// Argument violates a domain precondition other than NaN/Inf or grid.
    OutOfDomain = 3,
    /// Grid resolution too coarse for the requested Chernoff shift.
    BoundaryFailure = 4,
    /// A required pointer argument was null.
    NullPtr = 5,
    /// CFL bound violated for the truncated-exp K=4 power series.
    CflViolated = 6,
    /// Iterative solver did not converge within the iteration cap.
    ConvergenceFailed = 7,
    /// Requested feature is not supported in this build.
    Unsupported = 8,
    /// A Rust panic was caught at the FFI boundary.
    ///
    /// This indicates an internal bug; please file an issue.
    Panic = 99,
}

impl From<&SemiflowError> for SemiflowStatus {
    fn from(err: &SemiflowError) -> Self {
        match err {
            SemiflowError::DomainViolation { what, value } => map_domain_violation(what, *value),
            SemiflowError::GridUnderresolved { .. } => SemiflowStatus::BoundaryFailure,
            // Both convergence-related variants map to the same status code.
            SemiflowError::ConvergenceFailed { .. } | SemiflowError::AdaptiveStepRejected { .. } => {
                SemiflowStatus::ConvergenceFailed
            }
            SemiflowError::Unsupported { .. } => SemiflowStatus::Unsupported,
            SemiflowError::CflViolated { .. } => SemiflowStatus::CflViolated,
            // Non-exhaustive guard: future variants map to OutOfDomain.
            _ => SemiflowStatus::OutOfDomain,
        }
    }
}

/// Map a `DomainViolation` to a finer-grained status code.
///
/// Heuristic: non-finite `value` → `NanInf`; `what` containing "grid",
/// "n must", "values.len", "xmin", or "xmax" → `GridMismatch`; everything
/// else → `OutOfDomain`.
///
/// The "xmin"/"xmax" substrings catch `Grid1D::new` messages such as
/// `"xmin must be finite"`, `"xmax must be finite"`, and `"xmin must be < xmax"`.
///
/// Note: "n must" also matches `ChernoffSemigroup::new`'s `n_steps` check, but
/// that path is unreachable from the Wave A FFI surface — `handle.rs` hardcodes
/// `n_steps = 100`. If a future entry point exposes `n_steps`, this routing
/// will need disambiguation.
fn map_domain_violation(what: &str, value: f64) -> SemiflowStatus {
    if !value.is_finite() {
        return SemiflowStatus::NanInf;
    }
    if what.contains("grid")
        || what.contains("n must")
        || what.contains("values.len")
        || what.contains("xmin")
        || what.contains("xmax")
    {
        return SemiflowStatus::GridMismatch;
    }
    SemiflowStatus::OutOfDomain
}
