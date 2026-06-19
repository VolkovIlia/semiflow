//! `#[pyclass] Heat1D` — 1-D heat equation state exposed to Python.
//!
//! Wraps `SemiflowStateInner` (heap-allocated semigroup + current grid function)
//! and exposes an idiomatic Python API:
//!   `Heat1D(xmin, xmax, n, u0)` — constructor
//!   `evolve(t, n_steps=100)` — advance by time `t` (GIL released during compute)
//!   `evolve_chunked(t, total_steps, chunk_steps)` — chunked GIL-cooperative evolve
//!   `values()` — return current state as `numpy.ndarray[float64]`
//!   `__len__()` — number of grid nodes
//!
//! `#![allow(unsafe_code)]` is required because the `PyO3` `#[pyclass]` and
//! `#[pymethods]` proc-macros expand `unsafe` blocks inside this file.
//!
//! ## GIL-release pattern (ADR-0031)
//!
//! `Heat1D::evolve` releases the GIL during the pure-Rust inner compute loop
//! via `py.detach` (`PyO3` 0.28 equivalent of `allow_threads`).  The three-phase
//! structure is:
//!
//! 1. **Pre-flight (under GIL)**: validate params, copy input buffer into
//!    owned `Vec<f64>`, extract Rust-owned chernoff function pointer.
//! 2. **Compute (GIL released)**: call pure-Rust `ChernoffSemigroup::evolve`
//!    inside `py.detach`; returns owned `Vec<f64>`.
//! 3. **Post-flight (under GIL)**: update `self.inner` from result `Vec`.
//!
//! `Send + Sync` for `ChernoffSemigroup<DiffusionChernoff<f64>, GridFn1D<f64>>`
//! is verified at compile time by `static_assertions::assert_impl_all!` in
//! `crate::send_assertions`.
//!
//! ## Module split (suckless budget)
//!
//! Helper functions (`make_coeff_closure`, `build_heat1d_from_arrays`,
//! `build_presampled_closure`, `compute_chunk`) live in `crate::state_1d_chunked`
//! to keep this file ≤500 lines.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_precision_loss, clippy::needless_pass_by_value)]

use numpy::{PyArray1, ToPyArray};
use pyo3::{prelude::*, types::PyAnyMethods};

use crate::{
    boundary::parse_boundary,
    dtype_dispatch::{parse_dtype, Dtype},
    error::{from_core, new_pyerr},
    graph_heat_f32::compute_heat1d_f32,
    handle::{build_heat_unit, SemiflowStateInner},
    panic::catch_panic_py,
    state_1d_chunked::{build_heat1d_from_arrays, compute_chunk, make_coeff_closure},
};

// ---------------------------------------------------------------------------
// Heat1D Python class
// ---------------------------------------------------------------------------

/// 1-D heat equation state.
///
/// Solves `∂_t u = ∂_{xx} u` (unit diffusion `a = 1`) on `[xmin, xmax]`
/// with `n` uniformly-spaced nodes.
///
/// Thread safety: `Heat1D` is **not** thread-safe.  Access from multiple
/// threads concurrently requires external locking.  (The GIL is released
/// during `evolve`; Python threads may run concurrently but must not share
/// one `Heat1D` instance without an external lock.)
///
/// Parameters
/// ----------
/// xmin : float
///     Left boundary; must be finite.
/// xmax : float
///     Right boundary; must be finite and > `xmin`.
/// n : int
///     Number of grid nodes (must be ≥ 4).
/// u0 : numpy.ndarray[float64]
///     Initial condition; 1-D float64 array of length exactly `n`.
///     All elements must be finite (no NaN, no Inf).
///
/// Raises
/// ------
/// `SemiflowError`
///     `kind='GridMismatch'` — `n < 4`, `xmin >= xmax`, or `len(u0) != n`.
///     `kind='NanInf'` — `u0` contains NaN or Inf.
///     `kind='OutOfDomain'` — other domain precondition violated.
#[pyclass(name = "Heat1D")]
pub struct Heat1D {
    pub(crate) inner: SemiflowStateInner,
    pub(crate) dtype: Dtype,
}

impl Heat1D {
    /// Construct from already-built components.
    ///
    /// Used by `crate::state_1d_chunked` helpers to create `Heat1D` without
    /// going through the public Python constructor.
    pub(crate) fn from_inner(inner: SemiflowStateInner, dtype: Dtype) -> Self {
        Heat1D { inner, dtype }
    }
}

#[pymethods]
impl Heat1D {
    /// Create a new heat-equation state.
    ///
    /// Parameters
    /// ----------
    /// boundary : str, optional
    ///     Boundary policy (keyword-only).  One of ``"reflect"`` (default),
    ///     ``"periodic"``, ``"zero"``, ``"linear"``.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, boundary = "reflect", dtype = "f64"))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        boundary: &str,
        dtype: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let dt = parse_dtype(dtype)?;
            let slice = extract_f64_slice(u0)?;
            let inner =
                build_heat_unit(xmin, xmax, n, 100, &slice, policy).map_err(|e| from_core(&e))?;
            Ok(Heat1D { inner, dtype: dt })
        })
    }

    /// Advance the state by time `t` using `n_steps` Chernoff iterations.
    ///
    /// The GIL is released via `py.detach` during the inner pure-Rust compute
    /// loop so that concurrent Python threads (e.g. Jupyter UI,
    /// `ThreadPoolExecutor`) can make progress. See ADR-0031 for the
    /// three-phase design.
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time to advance.  Must be non-negative and finite.
    /// `n_steps` : int, optional
    ///     Number of Chernoff steps (default 100).  Must be ≥ 1.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``OutOfDomain`` if `t < 0` or `n_steps == 0`.
    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// When ``dtype="f32"`` was set at construction the compute runs in f32;
    /// the internal state is updated from the f32 result (cast back to f64).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            // Phase 1: validate + extract (GIL held)
            validate_evolve_params(t, n_steps)?;
            let input_values: Vec<f64> = self.inner.current.values.clone();

            let result_values: Vec<f64> = match self.dtype {
                Dtype::F64 => {
                    let chernoff_func = self.inner.semigroup.func.clone();
                    let grid = self.inner.current.grid;
                    let chernoff_for_sg = chernoff_func.clone();
                    let res: Result<Vec<f64>, _> =
                        py.detach(|| compute_evolve(chernoff_func, grid, input_values, t, n_steps));
                    let vals = res.map_err(|e| from_core(&e))?;
                    // Rebuild semigroup for next call.
                    let sg = semiflow_core::ChernoffSemigroup::new(chernoff_for_sg, n_steps)
                        .map_err(|e| from_core(&e))?;
                    self.inner.semigroup = sg;
                    vals
                }
                Dtype::F32 => {
                    let xmin = self.inner.current.grid.xmin;
                    let xmax = self.inner.current.grid.xmax;
                    let n = self.inner.current.values.len();
                    py.detach(|| compute_heat1d_f32(xmin, xmax, n, &input_values, t, n_steps))
                        .map_err(|e| from_core(&e))?
                }
            };

            // Phase 3: update internal state
            self.inner.current.values = result_values;
            Ok(())
        })
    }

    /// Chunked GIL-cooperative evolve (ADR-0141).
    ///
    /// Runs `total_steps` Chernoff iterations in chunks of `chunk_steps`,
    /// releasing the GIL (`py.detach`) for each chunk's pure-Rust compute
    /// and re-acquiring it to call the optional `progress(done, total)` callback.
    ///
    /// **0-ULP parity**: result is bit-identical to `evolve(t, total_steps)`.
    /// Chunking changes only *when* the GIL is released, not the arithmetic.
    ///
    /// **GIL-safety**: `progress` is called only while the GIL is held.
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time to advance.  Must be non-negative and finite.
    /// `total_steps` : int
    ///     Total number of Chernoff steps.  Must be ≥ 1.
    /// `chunk_steps` : int
    ///     Number of steps per GIL-release window.  Must be ≥ 1.
    ///     Smaller values increase GIL-yield frequency (better cooperative
    ///     scheduling) at the cost of slightly more overhead.
    /// progress : callable, optional
    ///     Called as ``progress(done: int, total: int)`` after each chunk
    ///     completes (GIL is held at that point).  May raise; the exception
    ///     propagates and stops the evolution cleanly (partial state is NOT
    ///     written back).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='OutOfDomain'`` if ``t < 0``, ``total_steps == 0``, or
    ///     ``chunk_steps == 0``.
    /// Any exception raised by `progress` propagates verbatim.
    #[pyo3(signature = (t, total_steps, chunk_steps, progress = None))]
    fn evolve_chunked(
        &mut self,
        py: Python<'_>,
        t: f64,
        total_steps: usize,
        chunk_steps: usize,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_params(t, total_steps)?;
            if chunk_steps == 0 {
                return Err(new_pyerr("OutOfDomain", "chunk_steps must be >= 1"));
            }
            let chernoff_func = self.inner.semigroup.func.clone();
            let grid = self.inner.current.grid;
            let tau = t / total_steps as f64;
            let mut buf: Vec<f64> = self.inner.current.values.clone();
            let mut done: usize = 0;
            while done < total_steps {
                let k = (total_steps - done).min(chunk_steps);
                let func_k = chernoff_func.clone();
                let buf_in = buf.clone();
                // Phase 2: compute k steps (GIL released)
                buf = py
                    .detach(|| compute_chunk(func_k, grid, buf_in, tau, k))
                    .map_err(|e| from_core(&e))?;
                done += k;
                // Phase 3: optional progress callback (GIL held)
                if let Some(ref cb) = progress {
                    cb.call1(py, (done, total_steps))?;
                }
            }
            // Rebuild semigroup for next call (mirrors evolve pattern).
            let sg = semiflow_core::ChernoffSemigroup::new(chernoff_func, total_steps)
                .map_err(|e| from_core(&e))?;
            self.inner.semigroup = sg;
            self.inner.current.values = buf;
            Ok(())
        })
    }

    /// Return the current grid values as a 1-D ``numpy.ndarray[float64]``.
    ///
    /// Returns a **copy** of the internal state; mutations to the returned
    /// array do not affect this `Heat1D` object.  Dtype is always ``float64``,
    /// length is always ``len(self)``.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let arr = self.inner.current.values.as_slice().to_pyarray(py);
            Ok(arr)
        })
    }

    /// Return the number of grid nodes (same as ``len(u0)`` passed to the
    /// constructor).
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }

    /// Create a `Heat1D` with a variable diffusion coefficient `a(x)` via
    /// Python callables.
    ///
    /// **Performance note (ADR-0031 / ADR-0034)**: each evaluation of `a`,
    /// `a_prime`, or `a_double_prime` re-acquires the GIL (~2-5 µs on `CPython`
    /// 3.11). This **defeats** the GIL-release optimisation of `evolve` (ADR-0031).
    /// For a 1024-node grid with 100 Chernoff steps the callback overhead is
    /// ~2.5M × 2-5 µs ≈ 5-12 s, versus ~50 ms for the unit-`a` path.
    /// If throughput matters, consider:
    ///   * Pre-sampled array approach: `Heat1D.with_a_array(a_values, ...)` [v0.13.0+]
    ///   * Zero-overhead C path: `Heat1D.with_a_cfunction(ctypes.CFUNCTYPE(...))` [v0.13.0+]
    ///   * Drop to `semiflow-ffi` directly via `ctypes` for the C-callback path.
    ///
    /// Parameters
    /// ----------
    /// xmin : float
    ///     Left boundary; must be finite.
    /// xmax : float
    ///     Right boundary; must be finite and > `xmin`.
    /// n : int
    ///     Number of grid nodes (must be ≥ 4).
    /// a : callable
    ///     Diffusion coefficient ``a(x: float) -> float``. Must be positive
    ///     everywhere on ``[xmin, xmax]`` (strict ellipticity).
    /// `a_prime` : callable
    ///     First derivative ``a'(x: float) -> float``.
    /// `a_double_prime` : callable
    ///     Second derivative ``a''(x: float) -> float``.
    /// `a_norm_bound` : float
    ///     Upper bound for ``‖a‖_∞`` (used for diagnostics; must be > 0).
    /// u0 : numpy.ndarray[float64]
    ///     Initial condition; 1-D float64 array of length exactly `n`.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     `kind='GridMismatch'` if `n < 4`, `xmin >= xmax`, or `len(u0) != n`.
    ///     `kind='NanInf'` if `u0` contains NaN or Inf.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, a, a_prime, a_double_prime,
                        a_norm_bound, u0, *, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_a_function(
        xmin: f64,
        xmax: f64,
        n: usize,
        a: Py<PyAny>,
        a_prime: Py<PyAny>,
        a_double_prime: Py<PyAny>,
        a_norm_bound: f64,
        u0: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            use crate::handle::build_heat_closure;
            let policy = parse_boundary(boundary)?;
            let slice = extract_f64_slice(u0)?;
            let a_fn = make_coeff_closure(a);
            let a_prime_fn = make_coeff_closure(a_prime);
            let a_double_prime_fn = make_coeff_closure(a_double_prime);
            let inner = build_heat_closure(
                xmin,
                xmax,
                n,
                100,
                a_norm_bound,
                &slice,
                a_fn,
                a_prime_fn,
                a_double_prime_fn,
                policy,
            )
            .map_err(|e| from_core(&e))?;
            Ok(Heat1D {
                inner,
                dtype: Dtype::F64,
            })
        })
    }

    /// Create a `Heat1D` from pre-sampled diffusion coefficient arrays.
    ///
    /// **GIL-zero-cost advantage**: unlike `with_a_function`, the coefficient
    /// closures produced here are pure-Rust closures backed by `Arc<Vec<f64>>`
    /// data.  They do **not** re-acquire the GIL during `evolve`, so the
    /// three-phase GIL-release pattern of `evolve` (ADR-0031) is fully
    /// effective.  For large grids or many time steps, this can be 10× faster
    /// than the Python-callable path.
    ///
    /// Parameters
    /// ----------
    /// xmin, xmax : float
    ///     Grid domain boundaries.
    /// n : int
    ///     Number of grid nodes.
    /// a : numpy.ndarray[float64]
    ///     Pre-sampled `a(x_i)` values, length `n`.  All values must be finite.
    /// u0 : numpy.ndarray[float64]
    ///     Initial condition, length `n`.
    /// `a_prime` : numpy.ndarray[float64], optional
    ///     Pre-sampled `a'(x_i)`.  If ``None``, computed via 4th-order FD.
    /// `a_double_prime` : numpy.ndarray[float64], optional
    ///     Pre-sampled `a''(x_i)`.  If ``None``, computed via 4th-order FD.
    /// `a_norm_bound` : float, optional
    ///     Upper bound on `‖a‖_∞`.  If ``None``, set to ``1.1 * max(a)``.
    /// boundary : str, optional
    ///     Boundary policy (keyword-only); default ``"reflect"``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='GridMismatch'`` if any array length != n.
    ///     ``kind='NanInf'`` if any array contains NaN or Inf.
    ///     ``kind='OutOfDomain'`` for invalid boundary string.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, a, u0, *,
                        a_prime = None, a_double_prime = None,
                        a_norm_bound = None, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_a_array(
        xmin: f64,
        xmax: f64,
        n: usize,
        a: &Bound<'_, PyAny>,
        u0: &Bound<'_, PyAny>,
        a_prime: Option<&Bound<'_, PyAny>>,
        a_double_prime: Option<&Bound<'_, PyAny>>,
        a_norm_bound: Option<f64>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            build_heat1d_from_arrays(
                xmin,
                xmax,
                n,
                a,
                u0,
                a_prime,
                a_double_prime,
                a_norm_bound,
                boundary,
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Phase 1 helper: input validation (no Python types involved)
// ---------------------------------------------------------------------------

/// Validate `t` and `n_steps` before GIL release.
///
/// Extracted so that Phase 1 logic is ≤50 lines and testable in isolation.
pub(crate) fn validate_evolve_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2 helper: pure-Rust compute (called inside py.allow_threads)
// ---------------------------------------------------------------------------

/// Run the Chernoff iteration on an owned buffer, returning an owned result.
///
/// No Python types cross this boundary.  All parameters are `Send + Sync`.
///
/// # Errors
/// Propagates [`semiflow_core::SemiflowError`] from `ChernoffSemigroup`.
fn compute_evolve(
    chernoff_func: semiflow_core::DiffusionChernoff<f64>,
    grid: semiflow_core::Grid1D<f64>,
    input: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    use semiflow_core::{ChernoffSemigroup, GridFn1D};
    let sg = ChernoffSemigroup::new(chernoff_func, n_steps)?;
    let f = GridFn1D::new(grid, input)?;
    let result = sg.evolve(t, &f)?;
    Ok(result.values)
}

// ---------------------------------------------------------------------------
// Utility: extract f64 slice from any numpy-compatible Python object
// ---------------------------------------------------------------------------

/// Convert any Python array-like to a `Vec<f64>`.
///
/// Accepts `numpy.ndarray` (dtype float64, 1-D) and any Python sequence of
/// floats.  Fails with `TypeError` for unsupported types.
///
/// `pub(crate)` so that `crate::state_1d_chunked` can call this via
/// `crate::state::extract_f64_slice` (re-exported from `state.rs`).
pub(crate) fn extract_f64_slice(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // Try numpy array first (common scientific-Python path).
    if let Ok(arr) = obj.extract::<Vec<f64>>() {
        return Ok(arr);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "u0 must be a numpy.ndarray[float64] or a sequence of floats",
    ))
}
