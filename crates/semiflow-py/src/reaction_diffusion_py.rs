//! `ReactionDiffusion1D/2D/3D` — semilinear reaction–diffusion systems (ADR-0208/0209).
//!
//! `∂ₜu_k = D_k Δu_k + f_k(t, x, u)` for `K` species on a 1-D/2-D/3-D grid: each
//! species diffuses with its own constant `D_k` (`DiffusionChernoff`, Strang
//! across axes), the reaction is a built-in model or any vectorised callable.
//! Strang splitting (order 2); `richardson=L` extrapolates in the step count
//! (order `1 + L`). The GIL is released for the whole evolution.

// Binding layer: PyO3 wrapper patterns.
#![allow(
    unsafe_code,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value
)]

use std::{marker::PhantomData, sync::Arc};

use numpy::{ndarray::ArrayD, ToPyArray};
use pyo3::{prelude::*, types::PyDict};
use semiflow::{
    extrapolate_into, ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, Grid3D, GridFn1D,
    GridFn2D, GridFn3D, Growth, Kinetics, NodalField, ReactionDiffusion, ScratchPool,
    SemiflowError as CoreError, Species, SpeciesEngine, State, Strang2D, Strang3D,
};

use crate::{
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
    rd_kinetics_py::{ArcKinetics, PyKinetics},
    rd_setup_py::{build_geometry, extract_fields, parse_reaction, Geometry},
};

/// Kinetics, diffusivities and clock shared by the three classes.
struct RdCore {
    diff: Vec<f64>,
    kinetics: Arc<dyn Kinetics<f64>>,
    py_kinetics: Option<Arc<PyKinetics>>,
    substeps: u32,
    t: f64,
}

/// State of one reaction–diffusion system.
struct RdState {
    core: RdCore,
    geom: Geometry,
    /// One flat field (x fastest) per species.
    values: Vec<Vec<f64>>,
}

/// Shared constructor body.
#[allow(clippy::too_many_arguments)]
fn new_state(
    py: Python<'_>,
    axes: &[(f64, f64, usize)],
    u0: &Bound<'_, PyAny>,
    diffusivity: Vec<f64>,
    reaction: &Bound<'_, PyAny>,
    params: Option<&Bound<'_, PyDict>>,
    boundary: &str,
    substeps: u32,
    t0: f64,
) -> PyResult<RdState> {
    let geom = build_geometry(axes, parse_boundary(boundary)?)?;
    let values = extract_fields(py, u0, &geom.field_shape())?;
    let k = values.len();
    if diffusivity.len() != k || diffusivity.iter().any(|d| !d.is_finite() || *d < 0.0) {
        return Err(new_pyerr(
            "OutOfDomain",
            &format!("diffusivity must hold K = {k} finite values >= 0, got {diffusivity:?}"),
        ));
    }
    if !t0.is_finite() || substeps == 0 {
        return Err(new_pyerr(
            "OutOfDomain",
            "t0 must be finite and substeps >= 1",
        ));
    }
    let (kinetics, py_kinetics) = parse_reaction(reaction, params, k, axes.len())?;
    let core = RdCore {
        diff: diffusivity,
        kinetics,
        py_kinetics,
        substeps,
        t: t0,
    };
    Ok(RdState { core, geom, values })
}

/// Run `n` steps (or `levels` Richardson levels) GIL-free on fields of type `S`.
fn run<S: NodalField<f64> + Clone + Send>(
    core: &RdCore,
    engines: Vec<SpeciesEngine<'static, f64, S>>,
    fields: Vec<S>,
    (t, n, levels): (f64, usize, usize),
) -> Result<Vec<S>, CoreError> {
    let kin = Box::new(ArcKinetics(Arc::clone(&core.kinetics)));
    let rd = ReactionDiffusion::new(engines, kin)?.with_reaction_substeps(core.substeps);
    let src = Species::new(fields);
    let mut dst = src.clone();
    let mut scratch = ScratchPool::new();
    if levels <= 1 {
        rd.evolve_into(core.t, t, n, &src, &mut dst, &mut scratch)?;
    } else {
        extrapolate_into(&rd, core.t, t, n, levels, &src, &mut dst, &mut scratch)?;
    }
    Ok(dst.fields)
}

/// Build fields and engines of type `S`, [`run`], and flatten the result.
fn solve<S: NodalField<f64> + Clone + Send>(
    core: &RdCore,
    vals: Vec<Vec<f64>>,
    job: (f64, usize, usize),
    field: impl Fn(Vec<f64>) -> S,
    engine: impl Fn(f64) -> SpeciesEngine<'static, f64, S>,
    flat: impl Fn(S) -> Vec<f64>,
) -> Result<Vec<Vec<f64>>, CoreError> {
    let engines = core.diff.iter().map(|&d| engine(d)).collect();
    let fields = vals.into_iter().map(field).collect();
    run(core, engines, fields, job).map(|v| v.into_iter().map(flat).collect())
}

/// Diffusion engine along one axis (`D > 0`).
fn axis_engine(d: f64, g: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new_const_a(d, d, g)
}

/// The exact `D = 0` "diffusion": the identity (bit-exact, no sampling).
struct Frozen<S>(PhantomData<fn() -> S>);

impl<S: State<f64>> ChernoffFunction<f64> for Frozen<S> {
    type S = S;
    fn apply_into(
        &self,
        _tau: f64,
        src: &S,
        dst: &mut S,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), CoreError> {
        dst.copy_from(src);
        Ok(())
    }
    fn order(&self) -> u32 {
        u32::MAX
    }
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

/// `Frozen` for `D = 0`, else `build()`.
fn engine_or_frozen<S: State<f64> + 'static>(
    d: f64,
    build: impl FnOnce() -> SpeciesEngine<'static, f64, S>,
) -> SpeciesEngine<'static, f64, S> {
    if d == 0.0 {
        Box::new(Frozen::<S>(PhantomData))
    } else {
        build()
    }
}

/// `D_k ∂ₓₓ` engine of a 1-D species.
fn engine_1d(d: f64, g: Grid1D<f64>) -> SpeciesEngine<'static, f64, GridFn1D<f64>> {
    engine_or_frozen(d, || Box::new(axis_engine(d, g)))
}

/// `D_k Δ` engine of a 2-D species (Strang across the axes).
fn engine_2d(d: f64, g: Grid2D<f64>) -> SpeciesEngine<'static, f64, GridFn2D<f64>> {
    engine_or_frozen(d, || {
        Box::new(Strang2D::new(axis_engine(d, g.x), axis_engine(d, g.y)))
    })
}

/// `D_k Δ` engine of a 3-D species (Strang across the axes).
fn engine_3d(diff: f64, grid: Grid3D<f64>) -> SpeciesEngine<'static, f64, GridFn3D<f64>> {
    let axis = |g: Grid1D<f64>| axis_engine(diff, g);
    engine_or_frozen(diff, || {
        Box::new(Strang3D::new(axis(grid.x), axis(grid.y), axis(grid.z)))
    })
}

/// The pure-Rust part of `evolve` (runs GIL-free).
fn compute(
    core: &RdCore,
    geom: &Geometry,
    vals: Vec<Vec<f64>>,
    job: (f64, usize, usize),
) -> Result<Vec<Vec<f64>>, CoreError> {
    match *geom {
        Geometry::D1(g) => solve(
            core,
            vals,
            job,
            |values| GridFn1D { values, grid: g },
            |d| engine_1d(d, g),
            |f| f.values,
        ),
        Geometry::D2(g) => solve(
            core,
            vals,
            job,
            |values| GridFn2D { values, grid: g },
            |d| engine_2d(d, g),
            |f| f.values,
        ),
        Geometry::D3(g) => solve(
            core,
            vals,
            job,
            |values| GridFn3D { values, grid: g },
            |d| engine_3d(d, g),
            |f| f.values,
        ),
    }
}

impl RdState {
    /// Evolve over `[t, t + dt]` with `n` steps; GIL released.
    fn evolve(&mut self, py: Python<'_>, dt: f64, steps: usize, levels: usize) -> PyResult<()> {
        if !dt.is_finite()
            || dt < 0.0
            || steps == 0
            || !(1..=semiflow::MAX_RICHARDSON_LEVELS).contains(&levels)
        {
            return Err(new_pyerr(
                "OutOfDomain",
                "need finite t >= 0, n_steps >= 1, 1 <= richardson <= 6",
            ));
        }
        let (core, geom, vals) = (&self.core, &self.geom, self.values.clone());
        let result = py.detach(|| compute(core, geom, vals, (dt, steps, levels)));
        match result {
            Ok(values) => {
                self.values = values;
                self.core.t += dt;
                Ok(())
            }
            Err(e) => Err(self
                .core
                .py_kinetics
                .as_ref()
                .and_then(|p| p.take_error())
                .unwrap_or_else(|| from_core(&e))),
        }
    }

    /// `(K, *field_shape)` numpy copy of the state.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut shape = vec![self.values.len()];
        shape.extend(self.geom.field_shape());
        let arr = ArrayD::from_shape_vec(shape, self.values.concat())
            .map_err(|_| new_pyerr("GridMismatch", "internal reshape failure"))?;
        Ok(arr.to_pyarray(py).into_any())
    }

    fn nodes(&self) -> usize {
        self.values.first().map_or(0, Vec::len)
    }
}

/// Generate a Python class over [`RdState`]: the given constructor plus the
/// shared methods (one `#[pymethods]` block per class).
macro_rules! rd_class {
    ($name:ident, $py:literal, $doc:literal, { $($ctor:tt)* }) => {
        #[doc = $doc]
        #[pyclass(name = $py)]
        pub struct $name {
            state: RdState,
        }

        #[pymethods]
        impl $name {
            $($ctor)*

            /// Advance by `t` with `n_steps` Strang steps (`richardson=L`: `L`
            /// levels of step-count extrapolation, order `1 + L`). GIL released.
            #[pyo3(signature = (t, n_steps = 100, richardson = 1))]
            fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize, richardson: usize) -> PyResult<()> {
                catch_panic_py!(self.state.evolve(py, t, n_steps, richardson))
            }

            /// Current state, shape `(K, *grid_shape)` (x fastest, last axis).
            fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                self.state.values(py)
            }

            /// Current time (`t0` plus every evolved interval).
            #[getter]
            fn time(&self) -> f64 {
                self.state.core.t
            }

            /// Number of species `K`.
            #[getter]
            fn species(&self) -> usize {
                self.state.values.len()
            }

            /// Grid nodes per species.
            fn __len__(&self) -> usize {
                self.state.nodes()
            }
        }
    };
}

rd_class!(
    ReactionDiffusion1D,
    "ReactionDiffusion1D",
    "1-D system `∂ₜu_k = D_k ∂ₓₓu_k + f_k(t, x, u)`; `u0` of shape `(K, n)`.",
    {
        #[new]
        #[pyo3(signature = (xmin, xmax, n, u0, *, diffusivity, reaction, params = None, boundary = "reflect", substeps = 1, t0 = 0.0))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            xmin: f64,
            xmax: f64,
            n: usize,
            u0: &Bound<'_, PyAny>,
            diffusivity: Vec<f64>,
            reaction: &Bound<'_, PyAny>,
            params: Option<&Bound<'_, PyDict>>,
            boundary: &str,
            substeps: u32,
            t0: f64,
        ) -> PyResult<Self> {
            let axes = [(xmin, xmax, n)];
            catch_panic_py!(new_state(
                py,
                &axes,
                u0,
                diffusivity,
                reaction,
                params,
                boundary,
                substeps,
                t0
            )
            .map(|state| Self { state }))
        }
    }
);

rd_class!(
    ReactionDiffusion2D,
    "ReactionDiffusion2D",
    "2-D system on `[xmin, xmax]×[ymin, ymax]`; `u0` of shape `(K, ny, nx)`.",
    {
        #[new]
        #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, u0, *, diffusivity, reaction, params = None, boundary = "reflect", substeps = 1, t0 = 0.0))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            xmin: f64,
            xmax: f64,
            nx: usize,
            ymin: f64,
            ymax: f64,
            ny: usize,
            u0: &Bound<'_, PyAny>,
            diffusivity: Vec<f64>,
            reaction: &Bound<'_, PyAny>,
            params: Option<&Bound<'_, PyDict>>,
            boundary: &str,
            substeps: u32,
            t0: f64,
        ) -> PyResult<Self> {
            let axes = [(xmin, xmax, nx), (ymin, ymax, ny)];
            catch_panic_py!(new_state(
                py,
                &axes,
                u0,
                diffusivity,
                reaction,
                params,
                boundary,
                substeps,
                t0
            )
            .map(|state| Self { state }))
        }
    }
);

rd_class!(
    ReactionDiffusion3D,
    "ReactionDiffusion3D",
    "3-D system on a box; `u0` of shape `(K, nz, ny, nx)`.",
    {
        #[new]
        #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, u0, *, diffusivity, reaction, params = None, boundary = "reflect", substeps = 1, t0 = 0.0))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            xmin: f64,
            xmax: f64,
            nx: usize,
            ymin: f64,
            ymax: f64,
            ny: usize,
            zmin: f64,
            zmax: f64,
            nz: usize,
            u0: &Bound<'_, PyAny>,
            diffusivity: Vec<f64>,
            reaction: &Bound<'_, PyAny>,
            params: Option<&Bound<'_, PyDict>>,
            boundary: &str,
            substeps: u32,
            t0: f64,
        ) -> PyResult<Self> {
            let axes = [(xmin, xmax, nx), (ymin, ymax, ny), (zmin, zmax, nz)];
            catch_panic_py!(new_state(
                py,
                &axes,
                u0,
                diffusivity,
                reaction,
                params,
                boundary,
                substeps,
                t0
            )
            .map(|state| Self { state }))
        }
    }
);

/// Richardson weights for step counts `(j+1)·n`, `j < levels` (ADR-0207).
#[pyfunction]
fn richardson_weights(order: u32, levels: usize) -> PyResult<Vec<f64>> {
    semiflow::richardson_weights(order, levels).map_err(|e| from_core(&e))
}

/// Register the reaction–diffusion classes.
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ReactionDiffusion1D>()?;
    m.add_class::<ReactionDiffusion2D>()?;
    m.add_class::<ReactionDiffusion3D>()?;
    m.add_function(wrap_pyfunction!(richardson_weights, m)?)
}
