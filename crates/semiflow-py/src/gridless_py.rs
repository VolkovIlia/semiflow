//! v9 S³ `PyO3` binding for `MeasureState` and `GridlessEvolver` (D=1 monomorphic).
//!
//! Mirrors `crates/semiflow-ffi/src/gridless_ffi.rs` in host-idiomatic numpy form.
//! Contract: `contracts/semiflow-ffi.s3-carrier-handle.yaml` (measurestate / gridless groups).
//!
//! ## D-monomorphism
//!
//! Only D=1 is compiled (same as the FFI layer v9.2.0).  Passing `dim != 1`
//! raises `SemiflowError(kind='Unsupported')`.
//!
//! ## API summary
//!
//! ```python
//! import numpy as np
//! from semiflow import MeasureState, GridlessEvolver
//!
//! pos = np.array([0.0, 1.0])
//! wts = np.array([0.5, 0.5])
//! ms = MeasureState(positions=pos, weights=wts, dim=1)
//!
//! ev = GridlessEvolver(a=0.5, b=0.0, c=0.0, voronoi_cap=64)
//! ev.evolve(ms, t_final=0.1, n_steps=4)
//!
//! m_pos, m_wts = ms.marginal(axis=0)
//! tv = ms.total_variation()
//! m2 = ms.second_moment()
//! assert tv > 0.0
//! ```

use numpy::ToPyArray;
use pyo3::prelude::*;
use semiflow_core::{chernoff::ChernoffFunction, GridlessChernoff, MeasureState, ParticleReduction,
                    ScratchPool};

use crate::error::new_pyerr;
use crate::panic::catch_panic_py;

/// Compiled dimension (mirrors `COMPILED_D` in `gridless_ffi.rs`).
const COMPILED_D: usize = 1;

/// (positions, weights) pair returned by `MeasureState.marginal`.
type MarginalPair<'py> = (
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
);

// ---------------------------------------------------------------------------
// MeasureState pyclass
// ---------------------------------------------------------------------------

/// Sparse weighted-Dirac particle ensemble on ℝ (D=1, v9, §50).
///
/// Represents a signed measure `ρ = Σ_i w_i δ_{x_i}` as a particle set.
/// Curse-escape: the `3^D` dense tree is never materialised; only sparse
/// marginals and scalar observables cross the Python boundary.
///
/// Parameters
/// ----------
/// positions : numpy.ndarray[float64]
///     Flat array of particle positions, length `n_part` (D=1).
/// weights : numpy.ndarray[float64]
///     Signed weights, length `n_part`.
/// dim : int
///     Must equal 1 (compiled D); any other value raises ``kind='Unsupported'``.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='Unsupported'`` — dim != 1.
///     ``kind='GridMismatch'`` — `n_part` == 0 or lengths mismatch.
///     ``kind='NanInf'`` — NaN/Inf in positions or weights.
#[pyclass(name = "MeasureState")]
pub struct PyMeasureState {
    pub inner: MeasureState<f64, 1>,
}

#[pymethods]
impl PyMeasureState {
    /// Construct a `MeasureState` from parallel position/weight numpy arrays.
    #[new]
    #[allow(clippy::needless_pass_by_value)] // PyO3 FromPyObject requires owned Vec
    fn new(
        positions: Vec<f64>,
        weights: Vec<f64>,
        dim: usize,
    ) -> PyResult<Self> {
        catch_panic_py!({
            if dim != COMPILED_D {
                return Err(new_pyerr("Unsupported", "MeasureState: dim must be 1 (compiled D)"));
            }
            if positions.is_empty() || weights.is_empty() {
                return Err(new_pyerr("GridMismatch", "MeasureState: positions/weights must be non-empty"));
            }
            if positions.len() != weights.len() {
                return Err(new_pyerr("GridMismatch", "MeasureState: positions and weights lengths differ"));
            }
            let ms = build_measure_state(&positions, &weights)?;
            Ok(Self { inner: ms })
        })
    }

    /// Number of Dirac atoms.
    fn n_diracs(&self) -> usize {
        self.inner.n_diracs()
    }

    /// Total-variation norm `‖ρ‖_TV`.
    fn total_variation(&self) -> f64 {
        self.inner.total_variation()
    }

    /// Second moment `⟨x², ρ⟩` — tightness monitor (§38.5).
    fn second_moment(&self) -> f64 {
        self.inner.second_moment()
    }

    /// Return the marginal projection onto `axis` as (positions, weights) arrays.
    ///
    /// Parameters
    /// ----------
    /// axis : int
    ///     Axis to project onto. Must be 0 for D=1.
    ///
    /// Returns
    /// -------
    /// (positions, weights) : (numpy.ndarray[float64], numpy.ndarray[float64])
    ///     Both have length `n_diracs()`.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='OutOfDomain'`` — axis >= `COMPILED_D`.
    fn marginal<'py>(
        &self,
        py: Python<'py>,
        axis: usize,
    ) -> PyResult<MarginalPair<'py>> {
        catch_panic_py!({
            if axis >= COMPILED_D {
                return Err(new_pyerr("OutOfDomain", "MeasureState.marginal: axis >= 1 (COMPILED_D)"));
            }
            let diracs = self.inner.diracs();
            let mut pos_out = Vec::with_capacity(diracs.len());
            let mut wt_out = Vec::with_capacity(diracs.len());
            for (pos, w) in diracs {
                pos_out.push(pos[axis]);
                wt_out.push(*w);
            }
            Ok((
                pos_out.as_slice().to_pyarray(py),
                wt_out.as_slice().to_pyarray(py),
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// GridlessEvolver pyclass
// ---------------------------------------------------------------------------

/// Gridless particle-ensemble Chernoff evolver (D=1, v9, §50, ADR-0155).
///
/// Advances `MeasureState` via the 1-D 3-branch Chernoff kernel:
///   `(x, w) → (x+h, ¼w) + (x-h, ¼w) + (x+k, ½w)` with `R_P` cap.
///
/// Parameters
/// ----------
/// a : float
///     Diffusion coefficient (>= 0, finite). D=1 scalar.
/// b : float
///     Drift coefficient (finite). D=1 scalar.
/// c : float
///     Reaction coefficient (finite).
/// `voronoi_cap` : int, optional
///     Particle cap for `WeightedVoronoi` reduction (default 64, must be >= 1).
///     Pass ``voronoi_cap=0`` to request ``GaussianBackground`` (pass-through stub).
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='NanInf'`` — non-finite or negative `a`, non-finite `b`/`c`.
///     ``kind='OutOfDomain'`` — `voronoi_cap == 0` with `WeightedVoronoi` selected
///     (this is reserved; pass `voronoi_cap=0` to select `GaussianBackground`).
#[pyclass(name = "GridlessEvolver")]
pub struct PyGridlessEvolver {
    inner: GridlessChernoff<f64, 1>,
}

#[pymethods]
impl PyGridlessEvolver {
    /// Construct a `GridlessEvolver`.
    ///
    /// `voronoi_cap` controls the `WeightedVoronoi` particle cap.  Pass
    /// ``gaussian_background=True`` to use the `GaussianBackground` stub instead.
    #[new]
    #[pyo3(signature = (a, b, c, voronoi_cap=64, gaussian_background=false))]
    fn new(
        a: f64,
        b: f64,
        c: f64,
        voronoi_cap: usize,
        gaussian_background: bool,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_gridless_coeffs(a, b, c, "GridlessEvolver")?;
            let reduction = if gaussian_background {
                ParticleReduction::GaussianBackground
            } else {
                if voronoi_cap == 0 {
                    return Err(new_pyerr("OutOfDomain", "GridlessEvolver: voronoi_cap must be >= 1"));
                }
                ParticleReduction::WeightedVoronoi { cap: voronoi_cap }
            };
            let ev = GridlessChernoff::<f64, 1>::new([a], [b], c, reduction);
            Ok(Self { inner: ev })
        })
    }

    /// Apply one Chernoff step of size `tau` to `src`, writing result into `dst`.
    ///
    /// Parameters
    /// ----------
    /// tau : float     — step size (>= 0, finite).
    /// src : `MeasureState`  — read-only source.
    /// dst : `MeasureState`  — overwritten with push-forward.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='OutOfDomain'`` — `tau` < 0 or non-finite.
    fn apply(
        &self,
        tau: f64,
        src: &PyMeasureState,
        dst: &mut PyMeasureState,
    ) -> PyResult<()> {
        catch_panic_py!({
            if !tau.is_finite() || tau < 0.0 {
                return Err(new_pyerr("OutOfDomain", "GridlessEvolver.apply: tau must be finite >= 0"));
            }
            let mut pool = ScratchPool::<f64>::new();
            self.inner
                .apply_into(tau, &src.inner, &mut dst.inner, &mut pool)
                .map_err(|e| new_pyerr("OutOfDomain", &format!("{e}")))?;
            Ok(())
        })
    }

    /// Evolve `state` in-place for time `t_final` using `n_steps` Chernoff steps.
    ///
    /// Uses two alternating scratch buffers (mirrors `gridless_ffi.rs`).
    ///
    /// Parameters
    /// ----------
    /// state : `MeasureState` — modified in-place.
    /// `t_final` : float    — total time (>= 0, finite).
    /// `n_steps` : int      — number of steps (>= 1).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='OutOfDomain'`` — `n_steps` == 0 or `t_final` non-finite/negative.
    fn evolve(
        &self,
        state: &mut PyMeasureState,
        t_final: f64,
        n_steps: usize,
    ) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_args(t_final, n_steps, "GridlessEvolver.evolve")?;
            #[allow(clippy::cast_precision_loss)] // n_steps <= u32::MAX in practice
            let tau = t_final / n_steps as f64;
            let mut buf_a = state.inner.clone();
            let mut buf_b = state.inner.clone();
            let mut pool = ScratchPool::<f64>::new();
            let mut a_is_src = true;
            for _ in 0..n_steps {
                let res = if a_is_src {
                    self.inner.apply_into(tau, &buf_a, &mut buf_b, &mut pool)
                } else {
                    self.inner.apply_into(tau, &buf_b, &mut buf_a, &mut pool)
                };
                res.map_err(|e| new_pyerr("OutOfDomain", &format!("{e}")))?;
                a_is_src = !a_is_src;
            }
            state.inner = if a_is_src { buf_a } else { buf_b };
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Build a `MeasureState<f64, 1>` from flat position/weight vecs.
fn build_measure_state(
    pos: &[f64],
    wts: &[f64],
) -> PyResult<MeasureState<f64, 1>> {
    let n = pos.len();
    let mut particles: Vec<([f64; 1], f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let p = pos[i];
        let w = wts[i];
        if !p.is_finite() || !w.is_finite() {
            return Err(new_pyerr("NanInf", "MeasureState: NaN/Inf in positions or weights"));
        }
        particles.push(([p], w));
    }
    Ok(MeasureState::<f64, 1>::from_particles(&particles))
}

/// Validate scalar D=1 coefficients.
fn validate_gridless_coeffs(a: f64, b: f64, c: f64, ctx: &str) -> PyResult<()> {
    if !a.is_finite() || a < 0.0 {
        return Err(new_pyerr("NanInf", &format!("{ctx}: a must be finite >= 0")));
    }
    if !b.is_finite() {
        return Err(new_pyerr("NanInf", &format!("{ctx}: b is non-finite")));
    }
    if !c.is_finite() {
        return Err(new_pyerr("NanInf", &format!("{ctx}: c is non-finite")));
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

/// Register `MeasureState` and `GridlessEvolver` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMeasureState>()?;
    m.add_class::<PyGridlessEvolver>()?;
    Ok(())
}
