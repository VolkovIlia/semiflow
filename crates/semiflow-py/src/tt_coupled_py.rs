//! v9 S³ `PyO3` binding for `TtCoupledEvolver` (coupled tensor-train Chernoff).
//!
//! Mirrors `crates/semiflow-ffi/src/tt_coupled_ffi.rs`.
//! Contract: `contracts/semiflow-ffi.s3-carrier-handle.yaml` (`tt_coupled` group).
//!
//! ## Coupling topology
//!
//! - `None`         — no coupling; bit-identical to `TtEvolver` (Gate C invariant).
//! - `Tridiagonal(rho)` — nearest-neighbour chain `(j, j+1, rho)` for all j.
//! - `Pairs(list)`  — explicit `[(j, k, rho), ...]` pairs; adjacent-only (`|k-j|==1`).
//!
//! ## Fail-loud walls (same as FFI, C-4 contract)
//!
//! - Any `b_j != 0` → `SemiflowError(kind='OutOfDomain')`.
//! - Non-adjacent pairs `|k-j| != 1` → `SemiflowError(kind='OutOfDomain')`.
//! - Non-SPD pair block `a[j]*a[k] - rho^2 <= 0` → `SemiflowError(kind='OutOfDomain')`.
//!
//! ## Example
//!
//! ```python
//! state = TtState([np.array([1.0, 0.0, 0.0])] * 2)
//! ev = TtCoupledEvolver(
//!     a=[0.5, 0.5], b=[0.0, 0.0], c=0.0,
//!     coupling=("Tridiagonal", 0.3),
//!     dom_min=[-3.0, -3.0], dom_max=[3.0, 3.0], eps_round=1e-10,
//! )
//! ev.evolve(state, t_final=0.1, n_steps=4)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use semiflow::{CoupledTtChernoff, CouplingTopology};

use crate::error::new_pyerr;
use crate::panic::catch_panic_py;
use crate::tt_py::PyTtState;

// ---------------------------------------------------------------------------
// TtCoupledEvolver pyclass
// ---------------------------------------------------------------------------

/// Coupled tensor-train Chernoff evolver (v9, §52.9, ADR-0162).
///
/// Extends `TtEvolver` with stable pair-factor coupling `exp(τ·L_pair)`.
/// With `coupling=None`, behaviour is bit-identical to `TtEvolver` (Gate C).
///
/// Parameters
/// ----------
/// a : list[float]       — per-axis diffusion (finite, >= 0).
/// b : list[float]       — per-axis drift (must be 0.0 — wall: drift deferred).
/// c : float             — scalar reaction (finite).
/// coupling : tuple
///     One of:
///       ``("None",)``                       — no coupling.
///       ``("Tridiagonal", rho)``            — nearest-neighbour chain.
///       ``("Pairs", [(j, k, rho), ...])``  — explicit adjacent pairs.
/// `dom_min` : list[float] — per-axis domain lower bounds.
/// `dom_max` : list[float] — per-axis domain upper bounds.
/// `eps_round` : float   — TT-rounding tolerance (finite, >= 0).
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='OutOfDomain'`` — any b[j] != 0, non-adjacent pair, non-SPD block.
///     ``kind='GridMismatch'`` — empty axis list or invalid domain.
///     ``kind='NanInf'`` — non-finite inputs.
#[pyclass(name = "TtCoupledEvolver")]
pub struct PyTtCoupledEvolver {
    inner: CoupledTtChernoff<f64>,
}

#[pymethods]
impl PyTtCoupledEvolver {
    /// Construct a `TtCoupledEvolver`.
    #[new]
    #[allow(clippy::needless_pass_by_value)] // PyO3 FromPyObject requires owned Vec
    #[allow(clippy::too_many_arguments)] // PyO3 constructor mirrors the 7-param FFI surface
    fn new(
        a: Vec<f64>,
        b: Vec<f64>,
        c: f64,
        coupling: &Bound<'_, PyTuple>,
        dom_min: Vec<f64>,
        dom_max: Vec<f64>,
        eps_round: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let n_axes = a.len();
            validate_coeffs(&a, &b, c, eps_round, "TtCoupledEvolver")?;
            let domain = build_domain(&dom_min, &dom_max, "TtCoupledEvolver")?;
            let topology = decode_topology(coupling, n_axes, "TtCoupledEvolver")?;
            precheck_walls(&b, &topology, &a, "TtCoupledEvolver")?;
            let ev = CoupledTtChernoff::new(a, b, c, topology, domain, eps_round);
            Ok(Self { inner: ev })
        })
    }

    /// Number of axes this evolver was built for.
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Evolve `state` in-place for time `t_final` using `n_steps` Chernoff steps.
    ///
    /// Same carrier (`TtState`) as `TtEvolver.evolve`.
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
            validate_evolve_args(t_final, n_steps, "TtCoupledEvolver.evolve")?;
            if self.inner.ndim() != state.inner.ndim() {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "TtCoupledEvolver.evolve: evolver ndim != state ndim",
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

/// Validate scalar coefficients for finiteness and non-negativity of a.
fn validate_coeffs(a: &[f64], b: &[f64], c: f64, eps: f64, ctx: &str) -> PyResult<()> {
    if a.is_empty() {
        return Err(new_pyerr("GridMismatch", &format!("{ctx}: axis list is empty")));
    }
    if !c.is_finite() || !eps.is_finite() {
        return Err(new_pyerr("NanInf", &format!("{ctx}: c or eps_round is non-finite")));
    }
    for (j, &v) in a.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            return Err(new_pyerr("NanInf", &format!("{ctx}: a[{j}] must be finite >= 0")));
        }
    }
    for (j, &v) in b.iter().enumerate() {
        if !v.is_finite() {
            return Err(new_pyerr("NanInf", &format!("{ctx}: b[{j}] is non-finite")));
        }
    }
    Ok(())
}

/// Build domain vec from parallel min/max slices.
fn build_domain(dom_min: &[f64], dom_max: &[f64], ctx: &str) -> PyResult<Vec<(f64, f64)>> {
    dom_min
        .iter()
        .zip(dom_max.iter())
        .enumerate()
        .map(|(j, (&lo, &hi))| {
            if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                Err(new_pyerr("NanInf", &format!("{ctx}: domain[{j}] invalid")))
            } else {
                Ok((lo, hi))
            }
        })
        .collect()
}

/// Decode a Python coupling tuple into `CouplingTopology<f64>`.
fn decode_topology(
    tup: &Bound<'_, PyTuple>,
    n_axes: usize,
    ctx: &str,
) -> PyResult<CouplingTopology<f64>> {
    let tag: String = tup.get_item(0)?.extract()?;
    match tag.as_str() {
        "None" => Ok(CouplingTopology::None),
        "Tridiagonal" => {
            let rho: f64 = tup.get_item(1)?.extract().map_err(|_| {
                new_pyerr("NanInf", &format!("{ctx}: Tridiagonal rho must be float"))
            })?;
            if !rho.is_finite() {
                return Err(new_pyerr("NanInf", &format!("{ctx}: Tridiagonal rho is non-finite")));
            }
            Ok(CouplingTopology::Tridiagonal(rho))
        }
        "Pairs" => {
            let raw: Vec<(usize, usize, f64)> = tup
                .get_item(1)?
                .extract::<Vec<(usize, usize, f64)>>()
                .map_err(|_| {
                    new_pyerr(
                        "GridMismatch",
                        &format!("{ctx}: Pairs expects list of (j, k, rho) tuples"),
                    )
                })?;
            for &(j, k, rho) in &raw {
                if j >= n_axes || k >= n_axes {
                    return Err(new_pyerr("GridMismatch", &format!("{ctx}: pair axis index out of range")));
                }
                if !rho.is_finite() {
                    return Err(new_pyerr("NanInf", &format!("{ctx}: pair rho is non-finite")));
                }
            }
            Ok(CouplingTopology::Pairs(raw))
        }
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!("{ctx}: unknown coupling tag '{other}'; expected None/Tridiagonal/Pairs"),
        )),
    }
}

/// Precheck fail-loud walls (mirrors `tt_coupled_ffi.rs::precheck_coupled_walls`).
fn precheck_walls(
    b: &[f64],
    topology: &CouplingTopology<f64>,
    a: &[f64],
    ctx: &str,
) -> PyResult<()> {
    // Wall 1: drift b != 0 deferred.
    for (j, &bj) in b.iter().enumerate() {
        if bj != 0.0 {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("{ctx}: b[{j}] != 0 (drift deferred to v9.2.0)"),
            ));
        }
    }
    // Wall 2: non-adjacent pairs; Wall 3: non-SPD block.
    if let CouplingTopology::Pairs(ref ps) = topology {
        for &(j, k, rho) in ps {
            let (lo, hi) = if j < k { (j, k) } else { (k, j) };
            if hi != lo + 1 {
                return Err(new_pyerr(
                    "OutOfDomain",
                    &format!("{ctx}: pair ({j},{k}) is not adjacent"),
                ));
            }
            let det = a[lo] * a[hi] - rho * rho;
            if det <= 0.0 {
                return Err(new_pyerr(
                    "OutOfDomain",
                    &format!("{ctx}: pair ({lo},{hi}) block is not SPD (det={det:.3e})"),
                ));
            }
        }
    }
    Ok(())
}

/// Validate evolve arguments.
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

/// Register `TtCoupledEvolver` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTtCoupledEvolver>()?;
    Ok(())
}
