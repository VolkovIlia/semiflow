//! v8.1.0 `PyO3` binding for `ComplexTripleJump` (F4, ADR-0138, ADR-0136 Amdt 2).
//!
//! Implements `ComplexTripleJumpV8` — a Python class that applies the order-4
//! complex triple-jump over the filiform-N5 step-4 Carnot group.
//!
//! ## NARROW scope (§1.4, `V8_1_TIER3_BINDING_DESIGN.md` NORMATIVE)
//!
//! Order-4 complex-time triple-jump over the filiform-N5 step-4 Carnot Strang.
//! D=5 ONLY (filiform Lie algebra is fixed). Complex substeps are internal;
//! only the real projection `Re(Ψ(τ)f)` is exposed. NaN/Inf or τ<0 → `OutOfDomain`.
//!
//! ## ABI-safety invariant (ADR-0138 hard constraint)
//!
//! No `Complex<f64>`, `CplxGridFn5`, or `SemiflowComplex` type crosses the
//! boundary. `apply_complex` is FORBIDDEN at the boundary. Only `apply_real`
//! (real input → real output via `GridFnND<f64,5>`) is exposed. The flat
//! `f64` buffer is the only wire format.
//!
//! ## GIL policy (ADR-0031 three-phase)
//!
//! `.apply_real` releases the GIL via `py.detach` around the triple complex
//! Strang sweep (pure Rust, no Python callbacks).
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/wasm.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_truncation)]

use numpy::ToPyArray;
use pyo3::prelude::*;
use semiflow_core::{
    grid_nd::{GridFnND, GridND},
    ComplexTripleJump, Grid1D,
};

use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// Constants (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

const D: usize = 5;

// ---------------------------------------------------------------------------
// Inner Rust state
// ---------------------------------------------------------------------------

/// Inner Rust state for `ComplexTripleJumpV8` (heap-owned by pyclass).
///
/// Stores grid geometry for GIL-off rebuilds (stateless per `.apply_real` call).
struct CplxTripleInner {
    domain_lo: f64,
    domain_hi: f64,
    n_per_axis: usize,
}

// ---------------------------------------------------------------------------
// ComplexTripleJumpV8 pyclass
// ---------------------------------------------------------------------------

/// Order-4 complex triple-jump, filiform-N5 Carnot, D=5 only (v8.1.0, ADR-0138).
///
/// Applies one order-4 step via `Ψ(τ) = K(γ⋆τ) ∘ K((1−2γ⋆)τ) ∘ K(γ⋆τ)` where
/// K is the filiform-N5 palindromic Strang (math.md §28.bis.8, ADR-0136 Amdt 2).
/// Complex substeps are internal; only the real projection `Re(Ψ(τ)f)` is exposed.
///
/// **NARROW scope**: filiform-N5, D=5 ONLY. ``apply_complex`` / ``CplxGridFn5``
/// are NOT exposed (ABI-safety invariant, ADR-0138). FFI/WASM surfaces
/// are deferred-within-v8.1.
///
/// Parameters
/// ----------
/// `domain_lo` : float
///     Lower bound of each axis (finite, same for all 5 axes).
/// `domain_hi` : float
///     Upper bound (``> domain_lo``).
/// `n_per_axis` : int
///     Grid nodes per axis (``>= 4``).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' — invalid domain or `n_per_axis` < 4.
///     kind='`OutOfDomain`'  — non-finite domain bound or γ⋆ check fails.
#[pyclass(name = "ComplexTripleJumpV8")]
pub struct PyComplexTripleJumpV8 {
    inner: CplxTripleInner,
}

#[pymethods]
impl PyComplexTripleJumpV8 {
    #[new]
    fn new(domain_lo: f64, domain_hi: f64, n_per_axis: usize) -> PyResult<Self> {
        catch_panic_py!({
            validate_domain(domain_lo, domain_hi, n_per_axis)?;
            Ok(Self {
                inner: CplxTripleInner {
                    domain_lo,
                    domain_hi,
                    n_per_axis,
                },
            })
        })
    }

    /// Apply one order-4 step and return the real projection `Re(Ψ(τ)f)`.
    ///
    /// The GIL is released during the triple complex Strang sweep (ADR-0031).
    /// ``Complex<f64>``/``CplxGridFn5`` never cross the boundary (ADR-0138).
    ///
    /// Parameters
    /// ----------
    /// tau : float
    ///     Step size (``>= 0``, finite).
    /// u0 : array-like
    ///     Flat 5-D grid function, 1-D float64, length ``n_per_axis^5``.
    ///
    /// Returns
    /// -------
    /// `NDArray`[np.float64]
    ///     Real projection ``Re(Ψ(τ)f)``, flat float64, same length as ``u0``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`GridMismatch`' — ``len(u0) != n_per_axis^5``.
    ///     kind='`OutOfDomain`'  — ``tau < 0`` or non-finite.
    fn apply_real<'py>(
        &self,
        py: Python<'py>,
        tau: f64,
        u0: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        catch_panic_py!({
            validate_tau(tau)?;
            // Phase 1: extract input under GIL.
            let u0_vec = extract_f64_vec(u0)?;
            let expected = self.inner.n_per_axis.pow(D as u32);
            if u0_vec.len() != expected {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("len(u0)={} != n_per_axis^5={}", u0_vec.len(), expected),
                ));
            }
            let lo = self.inner.domain_lo;
            let hi = self.inner.domain_hi;
            let n = self.inner.n_per_axis;
            // Phase 2: pure-Rust compute — release GIL.
            let result: Result<Vec<f64>, semiflow_core::SemiflowError> =
                py.detach(|| run_cplx_triple(lo, hi, n, tau, &u0_vec));
            let out = result.map_err(|e| from_core(&e))?;
            // Phase 3: marshal to numpy under GIL.
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Verify that γ⋆ satisfies the cubic ``2γ³+(1−2γ)³=0`` with Re>0.
    ///
    /// Returns ``True`` iff the residual is < 1e-12 and both Re(γ⋆) > 0
    /// and Re(1−2γ⋆) > 0.
    #[staticmethod]
    fn verify_gamma_star() -> bool {
        ComplexTripleJump::verify_gamma_star()
    }

    /// Return the total number of grid points (``n_per_axis^5``).
    fn size(&self) -> usize {
        self.inner.n_per_axis.pow(D as u32)
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust compute (GIL-off)
// ---------------------------------------------------------------------------

/// Run one complex triple-jump step — executed GIL-off under `py.detach`.
fn run_cplx_triple(
    lo: f64,
    hi: f64,
    n: usize,
    tau: f64,
    u0: &[f64],
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let grid = build_grid(lo, hi, n)?;
    let src = GridFnND::new(grid, u0.to_vec())?;
    let kernel = ComplexTripleJump::new()?;
    let out = kernel.apply_real(tau, &src)?;
    Ok(out.values)
}

// ---------------------------------------------------------------------------
// Builders (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

fn build_grid(lo: f64, hi: f64, n: usize) -> Result<GridND<f64, D>, semiflow_core::SemiflowError> {
    let ax = Grid1D::new(lo, hi, n)?;
    GridND::new([ax; D])
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

fn validate_domain(lo: f64, hi: f64, n: usize) -> PyResult<()> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(new_pyerr("OutOfDomain", "domain bounds must be finite"));
    }
    if lo >= hi {
        return Err(new_pyerr("GridMismatch", "domain_lo must be < domain_hi"));
    }
    if n < 4 {
        return Err(new_pyerr("GridMismatch", "n_per_axis must be >= 4"));
    }
    Ok(())
}

fn validate_tau(tau: f64) -> PyResult<()> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(new_pyerr("OutOfDomain", "tau must be finite and >= 0"));
    }
    Ok(())
}

fn extract_f64_vec(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "u0 must be numpy.ndarray[float64] or sequence of floats",
        )
    })
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `ComplexTripleJumpV8` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyComplexTripleJumpV8>()?;
    Ok(())
}
