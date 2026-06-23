//! PyO3 binding for `VarCoefTt` — additive-separable variable-coefficient
//! TT evolver (ADR-0178, math §52.10, issue #2).
//!
//! Mirrors [`crate::tt_py`] for `TtEvolver` / `TtState`.
//!
//! ## API
//!
//! ```python
//! import numpy as np
//! from semiflow import TtState, VarCoefTtEvolver
//!
//! n = 16; d = 2
//! xs = np.linspace(-3.0, 3.0, n)
//! a  = [np.ones(n) * 0.5 for _ in range(d)]   # const a_j > 0
//! b  = [np.zeros(n)       for _ in range(d)]   # zero drift
//! v  = [np.zeros(n)       for _ in range(d)]   # zero reaction
//! dom = [(-3.0, 3.0)] * d
//! ev = VarCoefTtEvolver(a, b, v, dom, eps_round=1e-10)
//! state = TtState([np.exp(-xs**2) for _ in range(d)])
//! ev.evolve(state, t_final=0.1, n_steps=4)
//! ```
//!
//! ## Error mapping
//!
//! `VarCoefOutOfClass` → `SemiflowError(kind='OutOfDomain')` (via `from_core`).

use pyo3::prelude::*;
use pyo3::types::PyList;
use semiflow_core::VarCoefTt;

use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;
use crate::tt_py::PyTtState;

// ---------------------------------------------------------------------------
// VarCoefTtEvolver pyclass
// ---------------------------------------------------------------------------

/// Additive-separable variable-coefficient TT evolver (ADR-0178, math §52.10).
///
/// Evolves a [`TtState`] by `exp(τ·L)` where `L = Σⱼ Lⱼ`,
/// `Lⱼ = ∂_{xⱼ}(aⱼ·∂_{xⱼ}) + bⱼ·∂_{xⱼ} + vⱼ` (additive-separable only).
///
/// Parameters
/// ----------
/// a_axis : list[list[float]]
///     Per-axis diffusion `aⱼ(xⱼ)` — each inner list has length nⱼ,
///     all entries strictly positive.
/// b_axis : list[list[float]]
///     Per-axis drift `bⱼ(xⱼ)` — each inner list length nⱼ.
/// v_axis : list[list[float]]
///     Per-axis reaction `vⱼ(xⱼ)` — each inner list length nⱼ or empty
///     (empty means zero reaction on that axis).
/// domain : list[tuple[float, float]]
///     Per-axis `(lo, hi)` domain bounds.  ``hi > lo`` required.
/// eps_round : float
///     TT-rounding tolerance applied after each step (finite, >= 0).
///
/// Raises
/// ------
/// SemiflowError
///     ``kind='OutOfDomain'`` — ``d == 0``, shape mismatch, ``nⱼ < 2``,
///     or any ``a_axis[j][i] <= 0``.
///     ``kind='NanInf'`` — non-finite value in any coefficient.
#[pyclass(name = "VarCoefTtEvolver")]
pub struct PyVarCoefTtEvolver {
    inner: VarCoefTt<f64>,
}

#[pymethods]
impl PyVarCoefTtEvolver {
    /// Construct a `VarCoefTtEvolver`.
    #[new]
    fn new(
        a_axis: &Bound<'_, PyList>,
        b_axis: &Bound<'_, PyList>,
        v_axis: &Bound<'_, PyList>,
        domain: &Bound<'_, PyList>,
        eps_round: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let a = extract_ragged(a_axis, "a_axis")?;
            let b = extract_ragged(b_axis, "b_axis")?;
            let v = extract_ragged(v_axis, "v_axis")?;
            let dom = extract_domain(domain, "domain")?;
            if !eps_round.is_finite() {
                return Err(new_pyerr("NanInf", "VarCoefTtEvolver: eps_round is non-finite"));
            }
            let ev = VarCoefTt::<f64>::new(a, b, v, dom, eps_round)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner: ev })
        })
    }

    /// Number of axes this evolver was built for.
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Evolve `state` in-place for `t_final` using `n_steps` steps.
    ///
    /// Parameters
    /// ----------
    /// state : TtState
    ///     Mutable carrier state (same `TtState` as `TtEvolver.evolve`).
    /// t_final : float
    ///     Total evolution time (>= 0, finite).
    /// n_steps : int
    ///     Number of time steps (>= 1).
    ///
    /// Raises
    /// ------
    /// SemiflowError
    ///     ``kind='OutOfDomain'`` — ``n_steps == 0``, non-finite/negative
    ///     ``t_final``, or ``ev.ndim() != state.ndim()``.
    fn evolve(
        &self,
        state: &mut PyTtState,
        t_final: f64,
        n_steps: usize,
    ) -> PyResult<()> {
        catch_panic_py!({
            if n_steps == 0 {
                return Err(new_pyerr("OutOfDomain", "VarCoefTtEvolver.evolve: n_steps must be >= 1"));
            }
            if !t_final.is_finite() || t_final < 0.0 {
                return Err(new_pyerr("OutOfDomain", "VarCoefTtEvolver.evolve: t_final must be finite >= 0"));
            }
            if self.inner.ndim() != state.inner.ndim() {
                return Err(new_pyerr("OutOfDomain", "VarCoefTtEvolver.evolve: evolver ndim != state ndim"));
            }
            self.inner.evolve(t_final, n_steps, &mut state.inner);
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Extract a `list[list[float]]` as `Vec<Vec<f64>>` with finite-value checks.
fn extract_ragged(list: &Bound<'_, PyList>, ctx: &str) -> PyResult<Vec<Vec<f64>>> {
    let d = list.len();
    if d == 0 {
        return Err(new_pyerr("OutOfDomain", &format!("{ctx}: list is empty (d must be >= 1)")));
    }
    let mut out = Vec::with_capacity(d);
    for (j, item) in list.iter().enumerate() {
        let v: Vec<f64> = item.extract::<Vec<f64>>().map_err(|_| {
            new_pyerr("OutOfDomain", &format!("{ctx}: axis {j} is not a list[float]"))
        })?;
        for &x in &v {
            if !x.is_finite() {
                return Err(new_pyerr("NanInf", &format!("{ctx}: NaN/Inf in axis {j}")));
            }
        }
        out.push(v);
    }
    Ok(out)
}

/// Extract `list[tuple[float, float]]` as `Vec<(f64, f64)>`.
fn extract_domain(list: &Bound<'_, PyList>, ctx: &str) -> PyResult<Vec<(f64, f64)>> {
    let d = list.len();
    if d == 0 {
        return Err(new_pyerr("OutOfDomain", &format!("{ctx}: domain list is empty")));
    }
    let mut out = Vec::with_capacity(d);
    for (j, item) in list.iter().enumerate() {
        let pair: (f64, f64) = item.extract::<(f64, f64)>().map_err(|_| {
            new_pyerr("OutOfDomain", &format!("{ctx}: domain[{j}] is not a (float, float) tuple"))
        })?;
        if !pair.0.is_finite() || !pair.1.is_finite() {
            return Err(new_pyerr("NanInf", &format!("{ctx}: domain[{j}] contains NaN/Inf")));
        }
        if pair.0 >= pair.1 {
            return Err(new_pyerr("OutOfDomain", &format!("{ctx}: domain[{j}].lo >= hi")));
        }
        out.push(pair);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `VarCoefTtEvolver` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVarCoefTtEvolver>()?;
    Ok(())
}
