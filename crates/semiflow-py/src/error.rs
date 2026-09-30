//! Python exception type for `semiflow-py`.
//!
//! A single `SemiflowError(Exception)` class is exposed with a `kind: str`
//! discriminator attribute, mirroring `SemiflowStatus` variant names from
//! `crates/semiflow-ffi/src/status.rs` in string form.  This follows the
//! `subprocess.CalledProcessError` / `urllib.error.HTTPError` convention:
//! one exception class, discriminated by an attribute, is more forward-
//! compatible than one class per variant.
//!
//! `SemiflowError::from_core` converts a `semiflow::SemiflowError` to a
//! `PyErr`, mapping error variants to the same `kind` strings that the C ABI
//! `smf_status_str()` returns.

use pyo3::{exceptions::PyException, prelude::*};
use semiflow::SemiflowError as CoreError;

pyo3::create_exception!(
    semiflow,
    SemiflowError,
    PyException,
    "Exception raised by semiflow-py operations.

Attributes
----------
kind : str
    Discriminator string matching the C-ABI `SemiflowStatus` names:
    ``GridMismatch``, ``NanInf``, ``OutOfDomain``, ``BoundaryFailure``,
    ``CflViolated``, ``ConvergenceFailed``, ``Unsupported``, ``Panic``.
"
);

/// Build a `PyErr` for `SemiflowError` with an explicit `kind` and `msg`.
///
/// The message keeps its `[Kind] ` prefix (backward compatible); the documented
/// `kind` attribute is set on the exception instance.
pub(crate) fn new_pyerr(kind: &str, msg: &str) -> PyErr {
    with_kind(SemiflowError::new_err(format!("[{kind}] {msg}")), kind)
}

/// Build a `PyErr` for a Rust panic caught at the `PyO3` boundary.
pub(crate) fn new_panic_pyerr() -> PyErr {
    with_kind(
        SemiflowError::new_err("[Panic] internal Rust panic — please file an issue"),
        "Panic",
    )
}

/// Attach the `kind` discriminator attribute declared in the class docstring
/// and in `__init__.pyi` (it used to exist only as the message prefix).
fn with_kind(err: PyErr, kind: &str) -> PyErr {
    Python::attach(|py| {
        // Setting an attribute on a fresh exception instance cannot fail in
        // practice; if it ever did, the error still carries the `[Kind]` prefix.
        let _ = err.value(py).setattr("kind", kind);
    });
    err
}

/// Convert a `semiflow::SemiflowError` to a Python `SemiflowError` `PyErr`.
///
/// Mirrors `crates/semiflow-ffi/src/status.rs` `From<&SemiflowError>` mapping.
pub(crate) fn from_core(err: &CoreError) -> PyErr {
    let (kind, msg) = classify_core_error(err);
    new_pyerr(kind, &msg)
}

/// Map a core error to `(kind, human-readable-message)`.
fn classify_core_error(err: &CoreError) -> (&'static str, String) {
    match err {
        CoreError::DomainViolation { what, value } => map_domain_violation(what, *value),
        CoreError::GridUnderresolved { .. } => ("BoundaryFailure", format!("{err}")),
        CoreError::ConvergenceFailed { .. } | CoreError::AdaptiveStepRejected { .. } => {
            ("ConvergenceFailed", format!("{err}"))
        }
        CoreError::Unsupported { .. } => ("Unsupported", format!("{err}")),
        CoreError::CflViolated { .. } => ("CflViolated", format!("{err}")),
        _ => ("OutOfDomain", format!("{err}")),
    }
}

/// Heuristic: non-finite value → `NanInf`; grid-related what → `GridMismatch`.
fn map_domain_violation(what: &str, value: f64) -> (&'static str, String) {
    if !value.is_finite() {
        return ("NanInf", format!("{what}: {value}"));
    }
    if what.contains("grid")
        || what.contains("n must")
        || what.contains("values.len")
        || what.contains("xmin")
        || what.contains("xmax")
    {
        return ("GridMismatch", what.to_owned());
    }
    ("OutOfDomain", what.to_owned())
}
