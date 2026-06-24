//! v8.1.0 `PyO3` binding for `AdjointFokkerPlanckChernoff` (C2, ADR-0138, ADR-0107 Amdt 1).
//!
//! Implements `AdjointFokkerPlanckV8` — a stateless-per-call Python class that
//! applies adjoint Fokker-Planck Chernoff steps on M(ℝ) represented as two flat
//! numpy arrays (positions, weights).
//!
//! ## NARROW scope (§38.3, ADR-0107 AMENDMENT 1 NORMATIVE)
//!
//! Adjoint (weak-*) Fokker-Planck on M(ℝ). D=1 constant-coefficient 4-Dirac
//! pushforward (Lemma A.1, §38.3). Dirac count grows ×4 per step. Forward
//! kernel = `DiffusionChernoff` (Brownian benchmark).
//!
//! ## ABI-safety invariant (ADR-0138 hard constraint)
//!
//! `MeasureState`<f64,1> never crosses the boundary. The caller passes and
//! receives two parallel numpy float64 arrays (positions, weights). The
//! kernel is stateless per call (same kernel object can be called many times).
//!
//! ## GIL policy (ADR-0031 three-phase)
//!
//! `.step` releases the GIL via `py.detach` around the multi-step push
//! (Lemma A.1 arithmetic — pure Rust, no Python callbacks).
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/wasm.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unused_self
)]

use numpy::ToPyArray;
use pyo3::prelude::*;
use semiflow::{
    AdjointFokkerPlanckChernoff, ChernoffFunction, DiffusionChernoff, Grid1D, MeasureState,
    ScratchPool, State,
};

// `ChernoffFunction` is used via `apply_into`; `State` is used via `zero_into`/`axpy_into`.
// Both are needed at trait-method call sites.
use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// Inner Rust state
// ---------------------------------------------------------------------------

/// Stored coefficients for GIL-off rebuild (stateless per `.step` call).
///
/// The kernel is rebuilt inside `run_steps` (GIL-off, `py.detach`); this
/// struct holds only the scalar coefficients (per-crate dup, ADR-0028 Amdt 2).
struct AdjointFpInner {
    a: f64,
    b: f64,
    c: f64,
}

// ---------------------------------------------------------------------------
// AdjointFokkerPlanckV8 pyclass
// ---------------------------------------------------------------------------

/// Adjoint Fokker-Planck Chernoff on M(ℝ) — flat two-buffer interface (v8.1.0).
///
/// Applies adjoint (weak-*) Fokker-Planck Chernoff steps: each Dirac `δ_x`
/// is pushed to four children (Lemma A.1, §38.3):
///
///   S*(τ) `δ_x` = ¼δ_{x+h} + ¼δ_{x-h} + ½δ_{x+k} + `τc·δ_x`
///
/// where ``h = 2√(aτ)`` and ``k = 2bτ``. The measure is passed as two
/// parallel numpy float64 arrays; ``MeasureState`` never crosses the boundary.
///
/// **NARROW scope**: D=1 constant-coefficient 4-Dirac pushforward (§38.3).
/// Dirac count grows ×4 per step. Forward kernel = `DiffusionChernoff` (Brownian).
///
/// Parameters
/// ----------
/// a : float
///     Diffusion coefficient (``h = 2√(aτ)``). Must be finite.
/// b : float
///     Drift coefficient (``k = 2bτ``). Must be finite.
/// c : float
///     Reaction coefficient (mass factor ``1 + τc``). Must be finite.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' — invalid geometry for the forward kernel.
///     kind='`OutOfDomain`'  — non-finite coefficient.
#[pyclass(name = "AdjointFokkerPlanckV8")]
pub struct PyAdjointFokkerPlanckV8 {
    inner: AdjointFpInner,
}

#[pymethods]
impl PyAdjointFokkerPlanckV8 {
    #[new]
    fn new(a: f64, b: f64, c: f64) -> PyResult<Self> {
        catch_panic_py!({
            let inner = build_inner(a, b, c).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Apply ``n_steps`` adjoint Fokker-Planck steps to the input measure.
    ///
    /// The GIL is released during the Lemma A.1 push (ADR-0031).
    /// ``MeasureState`` never crosses the boundary (ADR-0138).
    ///
    /// Parameters
    /// ----------
    /// tau : float
    ///     Step size (``> 0``, finite).
    /// positions : array-like
    ///     Dirac positions, 1-D float64, length ``n_diracs``.
    /// weights : array-like
    ///     Dirac weights, 1-D float64, same length as ``positions``.
    /// `n_steps` : int, optional
    ///     Number of steps (default 1).
    ///
    /// Returns
    /// -------
    /// tuple[np.ndarray, np.ndarray]
    ///     ``(positions, weights)`` after applying ``n_steps``.
    ///     Length grows by factor 4 per step (Lemma A.1, §38.3).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`'  — tau <= 0, non-finite, or `n_steps` == 0.
    ///     kind='`GridMismatch`' — len(positions) != len(weights).
    #[pyo3(signature = (tau, positions, weights, n_steps = 1))]
    fn step<'py>(
        &self,
        py: Python<'py>,
        tau: f64,
        positions: &Bound<'_, pyo3::types::PyAny>,
        weights: &Bound<'_, pyo3::types::PyAny>,
        n_steps: usize,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    )> {
        catch_panic_py!({
            validate_tau(tau)?;
            if n_steps == 0 {
                return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
            }
            // Phase 1: extract input under GIL.
            let pos_vec = extract_f64_vec(positions)?;
            let wts_vec = extract_f64_vec(weights)?;
            if pos_vec.len() != wts_vec.len() {
                return Err(new_pyerr(
                    "GridMismatch",
                    "len(positions) must equal len(weights)",
                ));
            }
            let a = self.inner.a;
            let b = self.inner.b;
            let c = self.inner.c;
            // Phase 2: multi-step push — release GIL.
            let result = py.detach(|| run_steps(a, b, c, tau, &pos_vec, &wts_vec, n_steps));
            let (out_pos, out_wts) = result.map_err(|e| from_core(&e))?;
            // Phase 3: marshal to numpy under GIL.
            Ok((
                out_pos.as_slice().to_pyarray(py),
                out_wts.as_slice().to_pyarray(py),
            ))
        })
    }

    /// Return total variation ``‖ρ‖_TV = Σ|w_i|`` of the input measure.
    fn total_variation(
        &self,
        positions: &Bound<'_, pyo3::types::PyAny>,
        weights: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<f64> {
        catch_panic_py!({
            let pos_vec = extract_f64_vec(positions)?;
            let wts_vec = extract_f64_vec(weights)?;
            if pos_vec.len() != wts_vec.len() {
                return Err(new_pyerr("GridMismatch", "len(positions) != len(weights)"));
            }
            let rho = build_measure(&pos_vec, &wts_vec);
            Ok(rho.total_variation())
        })
    }

    /// Return second moment ``⟨x², ρ⟩ = Σ x_i² w_i``.
    fn second_moment(
        &self,
        positions: &Bound<'_, pyo3::types::PyAny>,
        weights: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<f64> {
        catch_panic_py!({
            let pos_vec = extract_f64_vec(positions)?;
            let wts_vec = extract_f64_vec(weights)?;
            if pos_vec.len() != wts_vec.len() {
                return Err(new_pyerr("GridMismatch", "len(positions) != len(weights)"));
            }
            let rho = build_measure(&pos_vec, &wts_vec);
            Ok(rho.second_moment())
        })
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust multi-step push (GIL-off)
// ---------------------------------------------------------------------------

/// Run `n_steps` of the adjoint push — executed GIL-off under `py.detach`.
fn run_steps(
    a: f64,
    b: f64,
    c: f64,
    tau: f64,
    pos_in: &[f64],
    wts_in: &[f64],
    n_steps: usize,
) -> Result<(Vec<f64>, Vec<f64>), semiflow::SemiflowError> {
    let grid = Grid1D::new(-4.0_f64, 4.0, 32)?;
    let fwd = DiffusionChernoff::new_const_a(a, a, grid);
    let kernel = AdjointFokkerPlanckChernoff::new(fwd, a, b, c);
    let mut rho = build_measure(pos_in, wts_in);
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        let mut rho_next = MeasureState::<f64, 1>::dirac([0.0_f64], 0.0);
        kernel.apply_into(tau, &rho, &mut rho_next, &mut pool)?;
        rho = rho_next;
    }
    Ok(rho.to_flat_buffers_d1())
}

// ---------------------------------------------------------------------------
// Builder and validators
// ---------------------------------------------------------------------------

fn build_inner(a: f64, b: f64, c: f64) -> Result<AdjointFpInner, semiflow::SemiflowError> {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "adjoint_fp: a, b, c must be finite",
            value: a,
        });
    }
    Ok(AdjointFpInner { a, b, c })
}

fn build_measure(positions: &[f64], weights: &[f64]) -> MeasureState<f64, 1> {
    let mut m = MeasureState::<f64, 1>::dirac([0.0_f64], 0.0);
    m.zero_into();
    for (&p, &w) in positions.iter().zip(weights.iter()) {
        let atom = MeasureState::<f64, 1>::dirac([p], w);
        m.axpy_into(1.0, &atom);
    }
    m
}

fn extract_f64_vec(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "positions/weights must be numpy.ndarray[float64] or sequence of floats",
        )
    })
}

fn validate_tau(tau: f64) -> PyResult<()> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "tau must be finite and > 0"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `AdjointFokkerPlanckV8` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAdjointFokkerPlanckV8>()?;
    Ok(())
}
