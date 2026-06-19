//! JS error conversion for `semiflow-wasm`.
//!
//! A single `Error` value with a `.kind` string property is returned to JS
//! callers, mirroring the `[kind] message` convention in `semiflow-py` and the
//! `SemiflowStatus` C enum names from `semiflow-ffi`. Same kind strings across
//! all three bindings: `GridMismatch`, `NanInf`, `OutOfDomain`,
//! `BoundaryFailure`, `CflViolated`, `ConvergenceFailed`, `Unsupported`,
//! `Panic`.
//!
//! `js_sys::Error` carries a `message` string (readable in `.message`).
//! The `.kind` discriminator is attached via `js_sys::Reflect::set` so that
//! JS callers can match on `err.kind` without parsing the message string.

use js_sys::Reflect;
use semiflow_core::SemiflowError as CoreError;
use wasm_bindgen::prelude::*;

/// Convert a `semiflow_core::SemiflowError` to a `JsValue` error with `.kind`.
///
/// The returned value is a `js_sys::Error` extended with a `kind` property.
pub(crate) fn err_to_js(e: &CoreError) -> JsValue {
    let (kind, msg) = classify(e);
    make_js_error(kind, &msg)
}

/// Build a JS Error with a `kind` discriminator from a static kind string.
pub(crate) fn make_js_error(kind: &str, msg: &str) -> JsValue {
    let full = format!("[{kind}] {msg}");
    let err = js_sys::Error::new(&full);
    let _ = Reflect::set(&err, &"kind".into(), &kind.into());
    err.into()
}

/// Map a core error to `(kind, human-readable-message)` strings.
fn classify(err: &CoreError) -> (&'static str, String) {
    match err {
        CoreError::DomainViolation { what, value } => classify_domain(what, *value),
        CoreError::GridUnderresolved { .. } => ("BoundaryFailure", format!("{err}")),
        CoreError::ConvergenceFailed { .. } | CoreError::AdaptiveStepRejected { .. } => {
            ("ConvergenceFailed", format!("{err}"))
        }
        CoreError::Unsupported { .. } => ("Unsupported", format!("{err}")),
        CoreError::CflViolated { .. } => ("CflViolated", format!("{err}")),
        _ => ("OutOfDomain", format!("{err}")),
    }
}

/// Heuristic: non-finite value → `NanInf`; grid-related label → `GridMismatch`.
///
/// "n must" is only a grid signal when combined with "Grid" or "grid" in the
/// message — e.g. "`Grid3D::new_generic`: x.n must be >= 2".  Standalone "n must"
/// (e.g. "Evolver n must be >= 1") must map to "`OutOfDomain`".
fn classify_domain(what: &str, value: f64) -> (&'static str, String) {
    if !value.is_finite() {
        return ("NanInf", format!("{what}: {value}"));
    }
    let is_grid_n = (what.contains("Grid") || what.contains("grid")) && what.contains("n must");
    if what.contains("grid")
        || is_grid_n
        || what.contains("values.len")
        || what.contains("xmin")
        || what.contains("xmax")
    {
        return ("GridMismatch", what.to_owned());
    }
    ("OutOfDomain", what.to_owned())
}
