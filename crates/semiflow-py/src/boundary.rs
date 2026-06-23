//! Translate the Python-facing `boundary='...'` kwarg into
//! `semiflow::BoundaryPolicy`.

use pyo3::prelude::*;
use semiflow::BoundaryPolicy;

use crate::error::new_pyerr;

/// Parse a `boundary='...'` Python kwarg into a [`BoundaryPolicy`].
///
/// Accepted (case-insensitive):
/// - `"reflect"` (default) — mirror at boundary
/// - `"periodic"` — wrap with period `(n-1)·dx`
/// - `"zero"` — return 0.0 outside domain
/// - `"linear"` — linear extrapolation from boundary nodes
///
/// # Errors
/// Returns `SemiflowError(kind='OutOfDomain')` for any unrecognised string,
/// listing all valid options in the message.
pub(crate) fn parse_boundary(s: &str) -> PyResult<BoundaryPolicy> {
    match s.to_ascii_lowercase().as_str() {
        "reflect" => Ok(BoundaryPolicy::Reflect),
        "periodic" => Ok(BoundaryPolicy::Periodic),
        "zero" => Ok(BoundaryPolicy::ZeroExtend),
        "linear" => Ok(BoundaryPolicy::LinearExtrapolate),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!(
                "unknown boundary policy {other:?}; \
                 valid options are: \"reflect\", \"periodic\", \"zero\", \"linear\""
            ),
        )),
    }
}
