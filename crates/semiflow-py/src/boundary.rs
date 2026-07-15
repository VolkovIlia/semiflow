//! Translate the Python-facing `boundary='...'` kwarg into
//! `semiflow::BoundaryPolicy`.

use pyo3::prelude::*;
use semiflow::BoundaryPolicy;

use crate::error::new_pyerr;

/// Parse a `boundary='...'` Python kwarg into a [`BoundaryPolicy`].
///
/// Accepted (case-insensitive):
///
/// | String | Policy | Typical use |
/// |--------|--------|-------------|
/// | `"reflect"` (default) | `BoundaryPolicy::Reflect` | General PDEs; default G1/G2 oracle requirement |
/// | `"periodic"` | `BoundaryPolicy::Periodic` | Periodic domains (torus, FFT-compatible) |
/// | `"zero"` | `BoundaryPolicy::ZeroExtend` | Solutions that vanish at boundary (barrier options, puts in log-price space near deep OTM) |
/// | `"linear"` | `BoundaryPolicy::LinearExtrapolate` | **Asymptotically-linear far-field** (European calls, linear ramps) |
///
/// ## `"linear"` for finance / far-field BCs
///
/// `"linear"` is the **recommended far-field closure for option-pricing PDEs
/// with asymptotically-linear payoffs** such as European calls.  At the call
/// far-field, `V ≈ S − Ke^{−rτ}`, so `V_SS → 0` and linear extrapolation is
/// exact to leading order.  Validated on `Shift1D.with_arrays` with
/// `S ∈ [0, 4K]`, n = 1025: ATM relative error ≈ 8.5e-5.
///
/// For inhomogeneous Dirichlet (fixing u to a non-zero constant at the
/// boundary), the core `BoundaryPolicy::Dirichlet { value }` variant exists in
/// the Rust library but is not yet exposed through this Python string parser.
/// Until it is, `"linear"` is the correct idiom for all asymptotically-linear
/// payoffs; a future version will add `"dirichlet"` / `("dirichlet", value)`
/// support (see issue #20).
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
