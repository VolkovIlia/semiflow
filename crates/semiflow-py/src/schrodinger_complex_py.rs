//! Wave P2 — complex Schrödinger (`SchrodingerComplex1D`).
//!
//! Exposes `SchrödingerChernoffComplex<num_complex::Complex<f64>>` (ADR-0079
//! Option B, math.md §30.3) as `SchrodingerComplex1D`.  Unlike the real-pair
//! `Schrodinger1D` (ADR-0057), this kernel operates natively in `ℂ` via the
//! Cayley–Crank-Nicolson map — no split into re/im grids.
//!
//! ## Kernel
//!
//! | pyclass | Core type | Notes |
//! |---------|-----------|-------|
//! | `SchrodingerComplex1D` | `SchrödingerChernoffComplex<Complex<f64>>` | M5; order 2, unitary |
//!
//! ## GIL policy
//!
//! Three-phase pattern (ADR-0031): validate + copy under GIL → `py.detach`
//! compute → write result under GIL.  `SchrödingerChernoffComplex` is `Send +
//! Sync` (only `Vec<f64>`, `Grid1D<f64>`, `PhantomData`; no `RefCell`), so
//! no `Mutex` wrapper is required (contrast `Schrodinger1D`).

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::unused_self
)]

use std::sync::Mutex;

use numpy::{Complex64, PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{ChernoffSemigroup, Grid1D, GridFnComplex1D, SchrödingerChernoffComplex};

use crate::{
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Type alias — concrete complex type used throughout this module.
// ---------------------------------------------------------------------------

/// f64-valued complex number (numpy `complex128`).
/// `numpy::Complex64` is `num_complex::Complex<f64>` (re-exported directly).
type C64 = Complex64;

// ---------------------------------------------------------------------------
// Inner state type
// ---------------------------------------------------------------------------

/// Internal storage for `SchrodingerComplex1D`.
///
/// We store the potential `v_at_node` as a `Vec<f64>` so that the kernel can
/// be reconstructed cheaply inside `py.detach` without requiring `Clone` on
/// `SchrödingerChernoffComplex` (which doesn't derive it).  The kernel itself
/// is reconstructed each `evolve` call inside the GIL-released window.
/// Grid geometry is recovered from `state.grid`.
pub(crate) struct SchrodingerComplexInner {
    /// Pre-sampled potential values `V(x_0), …, V(x_{N-1})`.
    pub v_at_node: Vec<f64>,
    /// Current wavefunction state `ψ(x_i) ∈ ℂ`.  Also carries the grid.
    pub state: GridFnComplex1D<C64>,
}

// ---------------------------------------------------------------------------
// SchrodingerComplex1D Python class
// ---------------------------------------------------------------------------

/// 1-D Schrödinger equation with native complex state: ``iψ_t = (−½Δ + V)ψ``.
///
/// Backed by ``SchrödingerChernoffComplex`` (ADR-0079 Option B, math.md §30.3):
/// palindromic Strang splitting with Cayley–Crank-Nicolson kinetic step.
/// Globally order 2; exactly unitary (‖ψ(t)‖₂ = ‖ψ(0)‖₂ to machine precision).
///
/// Unlike ``Schrodinger1D`` (real-pair split), this class stores the wavefunction
/// as a native ``complex128`` array and avoids the re/im split overhead.
///
/// Raises ``SemiflowError`` on invalid inputs or evolution failures.
#[pyclass(name = "SchrodingerComplex1D")]
pub struct PySchrodingerComplex1D {
    /// Mutex ensures `&mut self` in evolve does not alias inner data.
    inner: Mutex<SchrodingerComplexInner>,
    /// Grid spacing `dx = (xmax − xmin) / (n − 1)`, cached for `norm_squared`.
    dx: f64,
    /// Number of grid nodes.
    n: usize,
}

#[pymethods]
impl PySchrodingerComplex1D {
    /// Create a free-particle state (``V = 0``) from a ``complex128`` initial array.
    ///
    /// Parameters
    /// ----------
    /// xmin : float
    ///     Left boundary.
    /// xmax : float
    ///     Right boundary (must be > xmin).
    /// n : int
    ///     Number of grid nodes (must be >= 4).
    /// psi0 : `NDArray`[complex128]
    ///     Initial wavefunction; length n; all finite.
    /// boundary : str, optional
    ///     Boundary policy (keyword-only); one of ``"reflect"`` (default),
    ///     ``"periodic"``, ``"zero"``, ``"linear"``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`GridMismatch`' if xmin >= xmax or n < 4 or len(psi0) != n.
    ///     kind='`NanInf`' if psi0 contains NaN or Inf.
    ///     kind='`OutOfDomain`' if boundary is not recognised.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, psi0, *, boundary = "reflect"))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        psi0: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let psi_vec = extract_complex128_vec(psi0)?;
            let inner = build_zero_v(xmin, xmax, n, policy, &psi_vec).map_err(|e| from_core(&e))?;
            let dx = compute_dx(xmin, xmax, n);
            Ok(Self {
                inner: Mutex::new(inner),
                dx,
                n,
            })
        })
    }

    /// Create state with a pre-sampled real potential ``V(x)`` and complex128 ψ₀.
    ///
    /// Parameters
    /// ----------
    /// v : `NDArray`[float64]
    ///     Pre-sampled ``V(x_i)`` values; length n; all finite.
    /// psi0 : `NDArray`[complex128]
    ///     Initial wavefunction; length n; all finite.
    /// boundary : str, optional
    ///     Boundary policy (keyword-only); default ``"reflect"``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`' on invalid inputs.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, v, psi0, *, boundary = "reflect"))]
    fn with_potential(
        xmin: f64,
        xmax: f64,
        n: usize,
        v: &Bound<'_, PyAny>,
        psi0: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let psi_vec = extract_complex128_vec(psi0)?;
            let inner = build_with_v(xmin, xmax, n, policy, v, &psi_vec)?;
            let dx = compute_dx(xmin, xmax, n);
            Ok(Self {
                inner: Mutex::new(inner),
                dx,
                n,
            })
        })
    }

    /// Advance the wavefunction by time ``t`` using ``n_steps`` Chernoff steps.
    ///
    /// Mutates self in-place; returns ``None``.  The GIL is released during the
    /// inner pure-Rust compute loop (ADR-0031 three-phase pattern).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time to advance.  Must be non-negative and finite.
    /// `n_steps` : int, optional
    ///     Number of Chernoff steps (default 200).  Must be >= 1.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if ``t < 0``, ``t`` is non-finite, or ``n_steps == 0``.
    #[pyo3(signature = (t, n_steps = 200))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            // Phase 1: validate + extract under GIL.
            validate_evolve_params(t, n_steps)?;
            let (v_at_node, values, grid) = {
                let guard = self
                    .inner
                    .lock()
                    .expect("SchrodingerComplex1D mutex poisoned");
                (
                    guard.v_at_node.clone(),
                    guard.state.values.clone(),
                    guard.state.grid,
                )
            };

            // Phase 2: pure-Rust compute (GIL released via py.detach).
            // Kernel is reconstructed from v_at_node inside the released window.
            let result: Result<Vec<C64>, _> =
                py.detach(|| compute_complex(v_at_node, grid, values, t, n_steps));
            let new_values = result.map_err(|e| from_core(&e))?;

            // Phase 3: write result back under GIL.
            let mut guard = self
                .inner
                .lock()
                .expect("SchrodingerComplex1D mutex poisoned");
            guard.state.values = new_values;
            Ok(())
        })
    }

    /// Return current wavefunction as a ``complex128`` numpy array of length n.
    ///
    /// Returns a copy of the internal state.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        catch_panic_py!({
            let guard = self
                .inner
                .lock()
                .expect("SchrodingerComplex1D mutex poisoned");
            let buf: Vec<Complex64> = guard
                .state
                .values
                .iter()
                .map(|z| Complex64::new(z.re, z.im))
                .collect();
            Ok(buf.as_slice().to_pyarray(py))
        })
    }

    /// Return the number of grid nodes n.
    fn __len__(&self) -> usize {
        self.n
    }

    /// Return the approximation order (always 2 for palindromic Strang).
    fn order(&self) -> u32 {
        2
    }

    /// Return ``Σ |ψ_i|² · dx`` — grid-spacing-weighted squared L2 norm.
    ///
    /// Equals 1.0 for a normalised wavefunction; used to verify unitarity.
    fn norm_squared(&self) -> f64 {
        let guard = self
            .inner
            .lock()
            .expect("SchrodingerComplex1D mutex poisoned");
        guard
            .state
            .values
            .iter()
            .fold(0.0_f64, |acc, z| acc + z.norm_sqr())
            * self.dx
    }
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn build_zero_v(
    xmin: f64,
    xmax: f64,
    n: usize,
    policy: semiflow::BoundaryPolicy,
    psi: &[C64],
) -> Result<SchrodingerComplexInner, semiflow::SemiflowError> {
    validate_psi_finite(psi)?;
    let grid = Grid1D::<f64>::new(xmin, xmax, n)?.with_boundary(policy);
    // Validate kernel construction (zero potential always succeeds).
    SchrödingerChernoffComplex::<C64>::new(grid, |_| 0.0)?;
    let v_at_node = vec![0.0_f64; n];
    let state = GridFnComplex1D::<C64>::new(grid, psi.to_vec())?;
    Ok(SchrodingerComplexInner { v_at_node, state })
}

fn build_with_v(
    xmin: f64,
    xmax: f64,
    n: usize,
    policy: semiflow::BoundaryPolicy,
    v_arr: &Bound<'_, PyAny>,
    psi: &[C64],
) -> PyResult<SchrodingerComplexInner> {
    validate_psi_finite(psi).map_err(|e| from_core(&e))?;
    let v_vec = v_arr
        .extract::<Vec<f64>>()
        .map_err(|_| new_pyerr("GridMismatch", "v must be numpy.ndarray[float64]"))?;
    if v_vec.len() != n {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("len(v) = {} but n = {n}", v_vec.len()),
        ));
    }
    for &vi in &v_vec {
        if !vi.is_finite() {
            return Err(new_pyerr("NanInf", "v contains NaN or Inf"));
        }
    }
    let grid = Grid1D::<f64>::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    // Validate kernel construction eagerly.
    let v_slice = v_vec.clone();
    let v_arc = std::sync::Arc::new(v_slice);
    let v_arc2 = v_arc.clone();
    let dx_check = grid.dx();
    let kernel_check = SchrödingerChernoffComplex::<C64>::new(grid, move |x: f64| {
        let idx = ((x - xmin) / dx_check).round() as usize;
        v_arc2[idx.min(v_arc2.len().saturating_sub(1))]
    })
    .map_err(|e| from_core(&e))?;
    drop(kernel_check);
    let state = GridFnComplex1D::<C64>::new(grid, psi.to_vec()).map_err(|e| from_core(&e))?;
    Ok(SchrodingerComplexInner {
        v_at_node: v_arc.to_vec(),
        state,
    })
}

// ---------------------------------------------------------------------------
// GIL-free compute helper (called inside py.detach)
// ---------------------------------------------------------------------------

/// Reconstruct kernel from stored potential, then evolve.
///
/// Called inside `py.detach`; all types are `Send + Sync`.
fn compute_complex(
    v_at_node: Vec<f64>,
    grid: Grid1D<f64>,
    values: Vec<C64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<C64>, semiflow::SemiflowError> {
    let v_arc = std::sync::Arc::new(v_at_node);
    let v_arc2 = v_arc.clone();
    let dx = grid.dx();
    let xmin = grid.xmin;
    let kernel = SchrödingerChernoffComplex::<C64>::new(grid, move |x: f64| {
        let idx = ((x - xmin) / dx).round() as usize;
        v_arc2[idx.min(v_arc2.len().saturating_sub(1))]
    })?;
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let state = GridFnComplex1D::<C64>::new(grid, values)?;
    Ok(sg.evolve(t, &state)?.values)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_evolve_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn validate_psi_finite(psi: &[C64]) -> Result<(), semiflow::SemiflowError> {
    for z in psi {
        if !z.re.is_finite() || !z.im.is_finite() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "psi0 contains NaN or Inf",
                value: f64::NAN,
            });
        }
    }
    Ok(())
}

/// Extract a Python ``complex128`` array into a `Vec<C64>`.
///
/// Mirrors `schrodinger.rs`'s `split_complex_array` but returns a `Vec<C64>`
/// instead of `(Vec<f64>, Vec<f64>)`.
fn extract_complex128_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<C64>> {
    let re_attr = obj
        .getattr("real")
        .map_err(|_| new_pyerr("GridMismatch", "psi0 must be a numpy.ndarray[complex128]"))?;
    let im_attr = obj
        .getattr("imag")
        .map_err(|_| new_pyerr("GridMismatch", "psi0 must be a numpy.ndarray[complex128]"))?;
    let re: Vec<f64> = re_attr
        .extract::<Vec<f64>>()
        .map_err(|_| new_pyerr("GridMismatch", "psi0.real must be extractable as float64"))?;
    let im: Vec<f64> = im_attr
        .extract::<Vec<f64>>()
        .map_err(|_| new_pyerr("GridMismatch", "psi0.imag must be extractable as float64"))?;
    if re.len() != im.len() {
        return Err(new_pyerr(
            "GridMismatch",
            "psi0.real and psi0.imag length mismatch",
        ));
    }
    Ok(re
        .into_iter()
        .zip(im)
        .map(|(r, i)| Complex64::new(r, i))
        .collect())
}

fn compute_dx(xmin: f64, xmax: f64, n: usize) -> f64 {
    if n > 1 {
        (xmax - xmin) / (n as f64 - 1.0)
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register Wave P2 pyclasses into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PySchrodingerComplex1D>()?;
    Ok(())
}
