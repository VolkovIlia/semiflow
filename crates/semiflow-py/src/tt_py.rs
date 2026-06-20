//! v9 S³ `PyO3` binding for `TtState` and `TtEvolver` (tensor-train Chernoff).
//!
//! Mirrors `crates/semiflow-ffi/src/tt_ffi.rs` in host-idiomatic numpy form.
//! Contract: `contracts/semiflow-ffi.s3-carrier-handle.yaml` (ttstate / `tt_evolver` groups).
//!
//! ## API summary
//!
//! ```python
//! import numpy as np
//! from semiflow import TtState, TtEvolver
//!
//! slices = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
//! state = TtState(slices)           # rank-1 separable IC
//! ev = TtEvolver(a=[0.5,0.5], b=[0.0,0.0], c=0.0,
//!                dom_min=[-3.0,-3.0], dom_max=[3.0,3.0], eps_round=1e-10)
//! ev.evolve(state, t_final=0.1, n_steps=4)
//! v = state.inner_separable([np.ones(3), np.ones(3)])
//! assert v == v  # finite
//! ```
//!
//! ## Error model
//!
//! Raises `SemiflowError` with `kind` matching `SemiflowStatus` variant names.
//!
//! ## GIL policy
//!
//! GIL is held throughout (TT-Chernoff evolve is already fast per-step; the
//! correctness-first approach mirrors `ReverseHeat1D`).

use pyo3::prelude::*;
use pyo3::types::PyList;
use semiflow_core::{TtChernoff, TtState};

use crate::error::new_pyerr;
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// TtState pyclass
// ---------------------------------------------------------------------------

/// Tensor-train state built from rank-1 separable per-axis slices (v9, §52).
///
/// Storage: O(d·n·r²) — curse-escaped for diagonal-A Gaussian diffusion.
///
/// Parameters
/// ----------
/// slices : list[numpy.ndarray[float64]]
///     Per-axis 1-D float64 arrays (at least one element each).
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` — empty list or any slice is empty.
///     ``kind='NanInf'`` — any slice contains NaN or Inf.
#[pyclass(name = "TtState")]
pub struct PyTtState {
    pub inner: TtState<f64>,
}

#[pymethods]
impl PyTtState {
    /// Build a rank-1 separable `TtState` from per-axis numpy slices.
    #[new]
    fn new(slices: &Bound<'_, PyList>) -> PyResult<Self> {
        catch_panic_py!({
            let vecs = extract_slices(slices, "TtState")?;
            Ok(Self { inner: TtState::<f64>::rank1_separable(vecs) })
        })
    }

    /// Number of modes (dimensions d).
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Mode size `n_j` for axis `j`.
    ///
    /// Raises `SemiflowError(kind='OutOfDomain')` if `j >= ndim()`.
    fn n_j(&self, j: usize) -> PyResult<usize> {
        catch_panic_py!({
            if j >= self.inner.ndim() {
                return Err(new_pyerr("OutOfDomain", "TtState.n_j: axis out of range"));
            }
            Ok(self.inner.n_j(j))
        })
    }

    /// Peak bond rank (max over internal bonds).
    fn peak_rank(&self) -> usize {
        self.inner.peak_rank()
    }

    /// Total number of stored scalars (working-set size).
    fn storage_size(&self) -> usize {
        self.inner.storage_size()
    }

    /// Separable inner product `⟨f, u⟩` for a list of per-axis numpy vectors.
    ///
    /// Parameters
    /// ----------
    /// functionals : list[numpy.ndarray[float64]]
    ///     One 1-D float64 array per axis; length of each must match `n_j(axis)`.
    ///
    /// Returns
    /// -------
    /// float
    ///     The scalar projection value.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='GridMismatch'`` — list length != ndim or slice lengths mismatch.
    ///     ``kind='NanInf'`` — NaN/Inf in any functional.
    fn inner_separable(
        &self,
        functionals: &Bound<'_, PyList>,
    ) -> PyResult<f64> {
        catch_panic_py!({
            let vecs = extract_slices(functionals, "TtState.inner_separable")?;
            if vecs.len() != self.inner.ndim() {
                return Err(new_pyerr(
                    "GridMismatch",
                    "TtState.inner_separable: functionals length != ndim",
                ));
            }
            for (j, v) in vecs.iter().enumerate() {
                if v.len() != self.inner.n_j(j) {
                    return Err(new_pyerr(
                        "GridMismatch",
                        &format!("TtState.inner_separable: axis {j} length mismatch"),
                    ));
                }
            }
            Ok(self.inner.inner_separable(&vecs))
        })
    }
}

// ---------------------------------------------------------------------------
// TtEvolver pyclass
// ---------------------------------------------------------------------------

/// Tensor-train Chernoff evolver for separable diagonal-A diffusion (v9, §52).
///
/// Parameters
/// ----------
/// a : list[float]
///     Per-axis diffusion coefficients (all >= 0, finite).
/// b : list[float]
///     Per-axis drift coefficients (finite).
/// c : float
///     Scalar reaction coefficient (finite).
/// `dom_min` : list[float]
///     Per-axis domain lower bounds.
/// `dom_max` : list[float]
///     Per-axis domain upper bounds (each > corresponding `dom_min`).
/// `eps_round` : float
///     TT-rounding tolerance (finite, >= 0).
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` — empty axis list.
///     ``kind='NanInf'`` — non-finite or negative `a[j]`, non-finite `b[j]`/`c`/domain.
#[pyclass(name = "TtEvolver")]
pub struct PyTtEvolver {
    inner: TtChernoff<f64>,
}

#[pymethods]
impl PyTtEvolver {
    /// Construct a `TtEvolver`.
    #[new]
    #[allow(clippy::needless_pass_by_value)] // PyO3 FromPyObject requires owned Vec
    fn new(
        a: Vec<f64>,
        b: Vec<f64>,
        c: f64,
        dom_min: Vec<f64>,
        dom_max: Vec<f64>,
        eps_round: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let ev = build_tt_evolver(&a, &b, c, &dom_min, &dom_max, eps_round)?;
            Ok(Self { inner: ev })
        })
    }

    /// Number of axes this evolver was built for.
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Evolve `state` in-place for time `t_final` using `n_steps` Chernoff steps.
    ///
    /// Parameters
    /// ----------
    /// state : `TtState`
    ///     Mutable carrier state (rank-1 IC or previous result).
    /// `t_final` : float
    ///     Total evolution time (>= 0, finite).
    /// `n_steps` : int
    ///     Number of Chernoff time steps (>= 1).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='OutOfDomain'`` — `n_steps` == 0, `t_final` non-finite/negative,
    ///     or `ev.ndim() != state.ndim()`.
    fn evolve(
        &self,
        state: &mut PyTtState,
        t_final: f64,
        n_steps: usize,
    ) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_args(t_final, n_steps, "TtEvolver.evolve")?;
            if self.inner.ndim() != state.inner.ndim() {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "TtEvolver.evolve: evolver ndim != state ndim",
                ));
            }
            self.inner.evolve(t_final, n_steps, &mut state.inner);
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Extract a list of numpy arrays as `Vec<Vec<f64>>` with finite-value checks.
fn extract_slices(list: &Bound<'_, PyList>, ctx: &str) -> PyResult<Vec<Vec<f64>>> {
    let n = list.len();
    if n == 0 {
        return Err(new_pyerr("GridMismatch", &format!("{ctx}: slices list is empty")));
    }
    let mut out = Vec::with_capacity(n);
    for (j, item) in list.iter().enumerate() {
        let v: Vec<f64> = item.extract::<Vec<f64>>().map_err(|_| {
            new_pyerr("GridMismatch", &format!("{ctx}: axis {j} is not a float64 array"))
        })?;
        if v.is_empty() {
            return Err(new_pyerr("GridMismatch", &format!("{ctx}: axis {j} is empty")));
        }
        for &x in &v {
            if !x.is_finite() {
                return Err(new_pyerr("NanInf", &format!("{ctx}: NaN/Inf in axis {j}")));
            }
        }
        out.push(v);
    }
    Ok(out)
}

/// Build a `TtChernoff<f64>` with input validation mirroring `tt_ffi.rs`.
fn build_tt_evolver(
    a: &[f64],
    b: &[f64],
    c: f64,
    dom_min: &[f64],
    dom_max: &[f64],
    eps_round: f64,
) -> PyResult<TtChernoff<f64>> {
    if a.is_empty() {
        return Err(new_pyerr("GridMismatch", "TtEvolver: axis list is empty"));
    }
    if !c.is_finite() || !eps_round.is_finite() {
        return Err(new_pyerr("NanInf", "TtEvolver: c or eps_round is non-finite"));
    }
    for (j, &v) in a.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            return Err(new_pyerr("NanInf", &format!("TtEvolver: a[{j}] must be finite >= 0")));
        }
    }
    for (j, &v) in b.iter().enumerate() {
        if !v.is_finite() {
            return Err(new_pyerr("NanInf", &format!("TtEvolver: b[{j}] is non-finite")));
        }
    }
    let domain: Vec<(f64, f64)> = dom_min
        .iter()
        .zip(dom_max.iter())
        .enumerate()
        .map(|(j, (&lo, &hi))| {
            if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                Err(new_pyerr("NanInf", &format!("TtEvolver: domain[{j}] invalid")))
            } else {
                Ok((lo, hi))
            }
        })
        .collect::<PyResult<_>>()?;
    Ok(TtChernoff::new(a.to_vec(), b.to_vec(), c, domain, eps_round))
}

/// Validate `t_final` and `n_steps` for evolve methods.
fn validate_evolve_args(t_final: f64, n_steps: usize, ctx: &str) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", &format!("{ctx}: n_steps must be >= 1")));
    }
    if !t_final.is_finite() || t_final < 0.0 {
        return Err(new_pyerr("OutOfDomain", &format!("{ctx}: t_final must be finite >= 0")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `TtState` and `TtEvolver` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTtState>()?;
    m.add_class::<PyTtEvolver>()?;
    Ok(())
}
