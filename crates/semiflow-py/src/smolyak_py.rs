//! v8.1.0 `PyO3` binding for `SmolyakGridND<f64,6>` (C1, ADR-0138, ADR-0123 Amdt 1).
//!
//! Implements `SmolyakD6V8` — a stateless Python class that evolves a 6-D
//! flat numpy float64 array via the sparse-grid Smolyak Chernoff step.
//!
//! ## NARROW scope (§1.3, `V8_1_TIER3_BINDING_DESIGN.md` NORMATIVE)
//!
//! Sparse-grid Smolyak quadrature, D=6, unit-diffusion (a=I, b=0, c=0) only.
//! Variable coefficients are NOT bound (TIER-3). Weights carry signs; F(0)=I
//! verified at construction. Default level ℓ=D+3=9.
//!
//! ## ABI-safety invariant (ADR-0138 hard constraint)
//!
//! No `GridFnND<f64,6>` or N-D Rust state type crosses the boundary.
//! The caller passes and receives a flat numpy float64 array of length n^6.
//! The Python side reshapes to 6-D as needed; Rust sees only flat slices.
//!
//! ## GIL policy (ADR-0031 three-phase)
//!
//! `.apply` releases the GIL via `py.detach` around the Smolyak quadrature
//! (pure Rust, no Python callbacks).
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/wasm.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_truncation, clippy::unused_self)]

use numpy::ToPyArray;
use pyo3::prelude::*;
use semiflow::{
    grid_nd::{GridFnND, GridND},
    smolyak::SmolyakGridND,
    ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};

use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// Constants (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

const D: usize = 6;
const DEFAULT_LEVEL: usize = D + 3; // ℓ=9 → 533 nodes

// ---------------------------------------------------------------------------
// Inner Rust state
// ---------------------------------------------------------------------------

/// Inner Rust state for `SmolyakD6V8` (heap-owned by pyclass).
///
/// Stores geometry for GIL-off rebuilds (stateless per `.apply` call).
struct SmolyakInner {
    domain_lo: f64,
    domain_hi: f64,
    n_per_axis: usize,
}

// ---------------------------------------------------------------------------
// SmolyakD6V8 pyclass
// ---------------------------------------------------------------------------

/// Sparse-grid Smolyak Chernoff kernel, D=6, unit-diffusion (v8.1.0, ADR-0138).
///
/// Applies one or more Chernoff steps of the unit-diffusion Smolyak operator
/// to a flat 6-D numpy float64 array.  The 6-D tensor state stays inside Rust;
/// only the flat `f64` buffer crosses the boundary.
///
/// **NARROW scope**: unit a=I, b=0, c=0 only. Variable coefficients are NOT
/// bound (TIER-3). Default level ℓ=9 (D+3=9 → 533 Smolyak nodes vs 4096
/// tensor baseline). FFI/WASM surfaces are deferred-within-v8.1 (ADR-0138).
///
/// Parameters
/// ----------
/// `domain_lo` : float
///     Lower bound of each axis (same for all 6 axes, finite).
/// `domain_hi` : float
///     Upper bound of each axis (``> domain_lo``).
/// `n_per_axis` : int
///     Number of grid nodes per axis (``>= 4``).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' — invalid domain or `n_per_axis` < 4.
///     kind='`OutOfDomain`'  — non-finite domain bound.
#[pyclass(name = "SmolyakD6V8")]
pub struct PySmolyakD6V8 {
    inner: SmolyakInner,
}

#[pymethods]
impl PySmolyakD6V8 {
    #[new]
    fn new(domain_lo: f64, domain_hi: f64, n_per_axis: usize) -> PyResult<Self> {
        catch_panic_py!({
            validate_domain(domain_lo, domain_hi, n_per_axis)?;
            Ok(Self {
                inner: SmolyakInner {
                    domain_lo,
                    domain_hi,
                    n_per_axis,
                },
            })
        })
    }

    /// Apply ``n_steps`` Smolyak Chernoff steps and return the flat result.
    ///
    /// The GIL is released during the Smolyak quadrature (ADR-0031).
    /// ``GridFnND<f64,6>`` never crosses the boundary (ADR-0138).
    ///
    /// Parameters
    /// ----------
    /// tau : float
    ///     Step size (``>= 0``, finite).
    /// u0 : array-like
    ///     Flat 6-D grid function, 1-D float64, length ``n_per_axis^6``.
    /// `n_steps` : int, optional
    ///     Number of Chernoff steps (default 1, must be >= 1).
    ///
    /// Returns
    /// -------
    /// `NDArray`[np.float64]
    ///     Flat output after applying ``n_steps``, same length as ``u0``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`GridMismatch`' — ``len(u0) != n_per_axis^6``.
    ///     kind='`OutOfDomain`'  — ``tau < 0``, non-finite, or `n_steps` == 0.
    #[pyo3(signature = (tau, u0, n_steps = 1))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        tau: f64,
        u0: &Bound<'_, pyo3::types::PyAny>,
        n_steps: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        catch_panic_py!({
            validate_tau(tau)?;
            if n_steps == 0 {
                return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
            }
            // Phase 1: extract input under GIL.
            let u0_vec = extract_f64_vec(u0)?;
            let expected = self.inner.n_per_axis.pow(D as u32);
            if u0_vec.len() != expected {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("len(u0)={} != n_per_axis^6={}", u0_vec.len(), expected),
                ));
            }
            let lo = self.inner.domain_lo;
            let hi = self.inner.domain_hi;
            let n = self.inner.n_per_axis;
            // Phase 2: pure-Rust compute — release GIL.
            let result: Result<Vec<f64>, semiflow::SemiflowError> =
                py.detach(|| run_smolyak(lo, hi, n, tau, &u0_vec, n_steps));
            let out = result.map_err(|e| from_core(&e))?;
            // Phase 3: marshal to numpy under GIL.
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Return the Smolyak node count (sparse grid size).
    fn n_nodes(&self) -> PyResult<usize> {
        catch_panic_py!({
            let k = build_kernel(
                self.inner.domain_lo,
                self.inner.domain_hi,
                self.inner.n_per_axis,
            )
            .map_err(|e| from_core(&e))?;
            Ok(k.n_nodes())
        })
    }

    /// Return the Smolyak level ℓ (default D+3=9).
    fn level(&self) -> usize {
        DEFAULT_LEVEL
    }

    /// Return the total number of grid points (``n_per_axis^6``).
    fn size(&self) -> usize {
        self.inner.n_per_axis.pow(D as u32)
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust compute (GIL-off)
// ---------------------------------------------------------------------------

/// Run `n_steps` Smolyak steps — executed GIL-off under `py.detach`.
fn run_smolyak(
    lo: f64,
    hi: f64,
    n: usize,
    tau: f64,
    u0: &[f64],
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let kernel = build_kernel(lo, hi, n)?;
    let grid = build_grid(lo, hi, n)?;
    let src_fn = GridFnND::new(grid.clone(), u0.to_vec())?;
    let mut src = src_fn;
    let mut dst = GridFnND::new(grid, vec![0.0_f64; u0.len()])?;
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut pool)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Builders (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

fn build_grid(lo: f64, hi: f64, n: usize) -> Result<GridND<f64, D>, semiflow::SemiflowError> {
    let ax = Grid1D::new(lo, hi, n)?;
    GridND::new([ax; D])
}

fn build_kernel(
    lo: f64,
    hi: f64,
    n: usize,
) -> Result<SmolyakGridND<f64, D>, semiflow::SemiflowError> {
    let grid = build_grid(lo, hi, n)?;
    SmolyakGridND::with_level(
        |_x: &[f64; D], a: &mut SquareMatrix<f64, D>| {
            for i in 0..D {
                a.set(i, i, 1.0);
            }
        },
        |_x: &[f64; D], b: &mut [f64; D]| {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        },
        |_x: &[f64; D]| 0.0_f64,
        grid,
        DEFAULT_LEVEL,
    )
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

/// Register `SmolyakD6V8` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySmolyakD6V8>()?;
    Ok(())
}
