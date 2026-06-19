//! `Schrodinger1D` — Schrödinger equation `iψ_t = (−Δ + V(x))ψ` for Python.
//!
//! Backed by [`SchrodingerChernoff<f64>`] (palindromic Strang splitting,
//! globally order 2, unitary by construction — ADR-0057).
//! Follows the three-phase GIL-release pattern (ADR-0031).
//!
//! `SchrodingerChernoff` contains a `RefCell` for its working buffers, which
//! makes it `!Sync`.  `PyO3` requires `#[pyclass]` to be `Send + Sync`, so the
//! inner state is wrapped in `Mutex<Schrodinger1DInner>`.
//!
//! Builders, compute helpers, and extraction utilities live in `schrodinger_helpers.rs`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::type_complexity)]

use std::sync::Mutex;

use numpy::{Complex64, PyArray1};
use pyo3::prelude::*;
use semiflow_core::{SchrodingerChernoff, SchrodingerState};

use crate::{
    boundary::parse_boundary,
    error::from_core,
    panic::catch_panic_py,
    schrodinger_helpers::{
        assemble_complex128, build_schrodinger_with_v, build_schrodinger_zero_v, compute_dx,
        compute_schrodinger, extract_f64_vec, split_complex_array, validate_evolve_params,
    },
};

// ---------------------------------------------------------------------------
// Inner state type
// ---------------------------------------------------------------------------

/// Internal storage for `Schrodinger1D`.
///
/// `SchrodingerChernoff<f64>` is `Send` but `!Sync` (contains `RefCell`).
/// Wrapped in `Mutex` so `Schrodinger1D` satisfies `PyO3`'s `Send + Sync` bound.
pub(crate) struct Schrodinger1DInner {
    /// Chernoff kernel (kinetic + potential); cloned on each `evolve`.
    pub chernoff: SchrodingerChernoff<f64>,
    /// Current wavefunction state (`psi_re + i·psi_im`).
    pub state: SchrodingerState<f64>,
}

// ---------------------------------------------------------------------------
// Schrodinger1D Python class
// ---------------------------------------------------------------------------

/// 1-D Schrödinger equation state: `iψ_t = (−Δ + V(x))ψ`.
///
/// Backed by `SchrodingerChernoff<f64>` — palindromic Strang splitting,
/// globally order 2, unitary by construction (ADR-0057, §17 of math spec).
///
/// Raises `SemiflowError` on invalid inputs or evolution failures.
#[pyclass(name = "Schrodinger1D")]
pub struct Schrodinger1D {
    /// Mutex-wrapped inner state (`SchrodingerChernoff` is `!Sync` via `RefCell`).
    inner: Mutex<Schrodinger1DInner>,
    /// Grid spacing `dx = (xmax - xmin) / (n - 1)`, cached for `norm_squared`.
    dx: f64,
    /// Number of grid nodes.
    n: usize,
}

#[pymethods]
impl Schrodinger1D {
    /// Create a free-particle state (`V = 0`) from a complex128 array.
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
    ///     Boundary policy; one of ``"reflect"`` (default), ``"periodic"``,
    ///     ``"zero"``, ``"linear"``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`' on invalid inputs.
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
            let (re, im) = split_complex_array(psi0)?;
            let inner = build_schrodinger_zero_v(xmin, xmax, n, policy, &re, &im)
                .map_err(|e| from_core(&e))?;
            let dx = compute_dx(xmin, xmax, n);
            Ok(Schrodinger1D {
                inner: Mutex::new(inner),
                dx,
                n,
            })
        })
    }

    /// Create a free-particle state from separate real/imag float64 arrays.
    ///
    /// Equivalent to `Schrodinger1D(xmin, xmax, n, psi_re + 1j*psi_im, ...)`.
    ///
    /// Parameters
    /// ----------
    /// `psi_re` : `NDArray`[float64]
    ///     Real part of ψ₀; length n; all finite.
    /// `psi_im` : `NDArray`[float64]
    ///     Imaginary part of ψ₀; length n; all finite.
    /// boundary : str, optional
    ///     Boundary policy (keyword-only); default ``"reflect"``.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, psi_re, psi_im, *, boundary = "reflect"))]
    fn from_parts(
        xmin: f64,
        xmax: f64,
        n: usize,
        psi_re: &Bound<'_, PyAny>,
        psi_im: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let re = extract_f64_vec(psi_re, "psi_re")?;
            let im = extract_f64_vec(psi_im, "psi_im")?;
            let inner = build_schrodinger_zero_v(xmin, xmax, n, policy, &re, &im)
                .map_err(|e| from_core(&e))?;
            let dx = compute_dx(xmin, xmax, n);
            Ok(Schrodinger1D {
                inner: Mutex::new(inner),
                dx,
                n,
            })
        })
    }

    /// Create state with a pre-sampled potential `V(x)` and complex128 ψ₀.
    ///
    /// Parameters
    /// ----------
    /// v : `NDArray`[float64]
    ///     Pre-sampled `V(x_i)` values; length n; all finite.
    /// psi0 : `NDArray`[complex128]
    ///     Initial wavefunction; length n; all finite.
    /// boundary : str, optional
    ///     Boundary policy (keyword-only); default ``"reflect"``.
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
            let (re, im) = split_complex_array(psi0)?;
            let inner = build_schrodinger_with_v(xmin, xmax, n, policy, v, &re, &im)?;
            let dx = compute_dx(xmin, xmax, n);
            Ok(Schrodinger1D {
                inner: Mutex::new(inner),
                dx,
                n,
            })
        })
    }

    /// Create state with potential `V(x)` and separate real/imag float64 arrays.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, v, psi_re, psi_im, *, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_potential_parts(
        xmin: f64,
        xmax: f64,
        n: usize,
        v: &Bound<'_, PyAny>,
        psi_re: &Bound<'_, PyAny>,
        psi_im: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let re = extract_f64_vec(psi_re, "psi_re")?;
            let im = extract_f64_vec(psi_im, "psi_im")?;
            let inner = build_schrodinger_with_v(xmin, xmax, n, policy, v, &re, &im)?;
            let dx = compute_dx(xmin, xmax, n);
            Ok(Schrodinger1D {
                inner: Mutex::new(inner),
                dx,
                n,
            })
        })
    }

    /// Advance the wavefunction by time ``t`` using ``n_steps`` Chernoff steps.
    ///
    /// Mutates self in-place; returns ``None``.  The GIL is released during
    /// the inner pure-Rust compute loop (ADR-0031 three-phase pattern).
    ///
    /// **Negative ``t`` (D2 — ADR-0113)**: ``t`` may be negative for backward
    /// (time-reversed) unitary evolution.  The palindromic Strang kernel satisfies
    /// ``S(−τ) = S(τ)⁻¹`` exactly (verified round-trip residual 1.19e-13 at
    /// n=128, T=1.0, 200 steps each way).  Norm is preserved to machine precision.
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time to advance.  Must be finite; may be negative for backward evolution.
    /// `n_steps` : int, optional
    ///     Number of Chernoff steps (default 200).  Must be >= 1.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if ``t`` is non-finite or ``n_steps == 0``.
    #[pyo3(signature = (t, n_steps = 200))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            // Phase 1: validate + clone buffers (under GIL).
            validate_evolve_params(t, n_steps)?;
            let (chernoff, re_in, im_in, grid) = {
                let guard = self.inner.lock().expect("Schrodinger1D mutex poisoned");
                (
                    guard.chernoff.clone(),
                    guard.state.psi_re.values.clone(),
                    guard.state.psi_im.values.clone(),
                    guard.state.psi_re.grid,
                )
            };

            // Phase 2: pure-Rust compute (GIL released via py.detach).
            let result: Result<(Vec<f64>, Vec<f64>), _> =
                py.detach(|| compute_schrodinger(chernoff, grid, re_in, im_in, t, n_steps));
            let (re_out, im_out) = result.map_err(|e| from_core(&e))?;

            // Phase 3: update state (under GIL).
            let mut guard = self.inner.lock().expect("Schrodinger1D mutex poisoned");
            guard.state.psi_re.values = re_out;
            guard.state.psi_im.values = im_out;
            Ok(())
        })
    }

    /// Return current wavefunction as a complex128 numpy array of length n.
    ///
    /// Returns a copy of the internal state; mutations do not affect this object.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        catch_panic_py!({
            let guard = self.inner.lock().expect("Schrodinger1D mutex poisoned");
            let re = &guard.state.psi_re.values;
            let im = &guard.state.psi_im.values;
            assemble_complex128(py, re, im)
        })
    }

    /// Return `(psi_re, psi_im)` as two float64 numpy arrays of length n.
    fn values_parts<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        use numpy::ToPyArray;
        catch_panic_py!({
            let guard = self.inner.lock().expect("Schrodinger1D mutex poisoned");
            let re = guard.state.psi_re.values.as_slice().to_pyarray(py);
            let im = guard.state.psi_im.values.as_slice().to_pyarray(py);
            Ok((re, im))
        })
    }

    /// Return the number of grid nodes.
    fn __len__(&self) -> usize {
        self.n
    }

    /// Return `Σ |ψ_i|² · dx` — grid-spacing-weighted squared L2 norm.
    fn norm_squared(&self) -> f64 {
        let guard = self.inner.lock().expect("Schrodinger1D mutex poisoned");
        let raw: f64 = guard
            .state
            .psi_re
            .values
            .iter()
            .zip(guard.state.psi_im.values.iter())
            .fold(0.0_f64, |acc, (&r, &i)| acc + r * r + i * i);
        raw * self.dx
    }
}
