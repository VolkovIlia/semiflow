//! `ReactionDiffusion1D/2D/3D` — semilinear reaction–diffusion systems (ADR-0208/0209).
//!
//! `∂ₜu_k = D_k Δu_k + f_k(t, x, u)` for `K` species on a 1-D/2-D/3-D grid: each
//! species diffuses with its own constant `D_k` (`DiffusionChernoff`, Strang
//! across axes), the reaction is a built-in model or any vectorised callable.
//! A `K×K` `diffusivity` matrix selects cross-diffusion `∂ₜu = DΔu + f` (ADR-0213:
//! modal decomposition `D = PΛP⁻¹`, verified; complex, negative or defective `D`
//! rejected). Strang splitting (order 2); `richardson=L` extrapolates in the step count
//! (order `1 + L`). `reaction_solver="rosenbrock"` integrates stiff kinetics with
//! adaptive ROS2 (`rtol`, `atol`; `jacobian(t, x, u) -> (K, K, N)` for a callable
//! reaction, finite differences otherwise; ADR-0212). The GIL is released for the
//! whole evolution.

// Binding layer: PyO3 wrapper patterns.
#![allow(
    unsafe_code,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value
)]

use std::sync::Arc;

use numpy::{ndarray::ArrayD, ToPyArray};
use pyo3::{prelude::*, types::PyDict};
use semiflow::{
    extrapolate_into, CrossDiffusion, GridFn1D, GridFn2D, GridFn3D, Kinetics, NodalField,
    ReactionDiffusion, ReactionSolver, ScratchPool, SemiflowError as CoreError, Species,
    SpeciesEngine,
};

use crate::{
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
    rd_engines_py::{engine_1d, engine_2d, engine_3d},
    rd_kinetics_py::{ArcKinetics, PyKinetics},
    rd_setup_py::{
        build_geometry, extract_fields, parse_diffusivity, parse_reaction, parse_solver, Geometry,
        StiffOptions,
    },
};

/// Kinetics, diffusivities and clock shared by the three classes.
struct RdCore {
    /// Engine coefficients: `D_k`, or the eigenvalues of a cross-diffusion `D`.
    diff: Vec<f64>,
    /// The decomposition of a `K×K` diffusivity matrix (ADR-0213).
    cross: Option<CrossDiffusion<f64>>,
    kinetics: Arc<dyn Kinetics<f64>>,
    py_kinetics: Option<Arc<PyKinetics>>,
    substeps: u32,
    solver: ReactionSolver,
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
    diffusivity: &Bound<'_, PyAny>,
    reaction: &Bound<'_, PyAny>,
    params: Option<&Bound<'_, PyDict>>,
    boundary: &str,
    (substeps, t0): (u32, f64),
    stiff: &StiffOptions<'_, '_>,
) -> PyResult<RdState> {
    let geom = build_geometry(axes, parse_boundary(boundary)?)?;
    let values = extract_fields(py, u0, &geom.field_shape())?;
    let k = values.len();
    let (diff, cross) = parse_diffusivity(diffusivity, k)?;
    if !t0.is_finite() || substeps == 0 {
        return Err(new_pyerr(
            "OutOfDomain",
            "t0 must be finite and substeps >= 1",
        ));
    }
    let solver = parse_solver(stiff)?;
    let (kinetics, py_kinetics) =
        parse_reaction(reaction, params, stiff.jacobian, (k, axes.len()))?;
    let core = RdCore {
        diff,
        cross,
        kinetics,
        py_kinetics,
        substeps,
        solver,
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
    let mut rd = ReactionDiffusion::new(engines, kin)?
        .with_reaction_substeps(core.substeps)
        .with_reaction_solver(core.solver);
    if let Some(cd) = &core.cross {
        rd = rd.with_cross_diffusion(cd.clone())?;
    }
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
// The class attribute (`pyclass` with its Python name) is written at each call site, where
// `xtask doc-check` (README ↔ registered classes) can see the literal name.
macro_rules! rd_class {
    (#[$class:meta] $name:ident, $doc:literal, { $($ctor:tt)* }) => {
        #[doc = $doc]
        #[$class]
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
    #[pyclass(name = "ReactionDiffusion1D")]
    ReactionDiffusion1D,
    "1-D system `∂ₜu_k = D_k ∂ₓₓu_k + f_k(t, x, u)`; `u0` of shape `(K, n)`.",
    {
        #[new]
        #[pyo3(signature = (xmin, xmax, n, u0, *, diffusivity, reaction, params = None, boundary = "reflect", substeps = 1, t0 = 0.0, reaction_solver = "rk4", rtol = 1e-6, atol = 1e-9, jacobian = None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            xmin: f64,
            xmax: f64,
            n: usize,
            u0: &Bound<'_, PyAny>,
            diffusivity: &Bound<'_, PyAny>,
            reaction: &Bound<'_, PyAny>,
            params: Option<&Bound<'_, PyDict>>,
            boundary: &str,
            substeps: u32,
            t0: f64,
            reaction_solver: &str,
            rtol: f64,
            atol: f64,
            jacobian: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            let stiff = StiffOptions {
                solver: reaction_solver,
                rtol,
                atol,
                jacobian,
            };
            let axes = [(xmin, xmax, n)];
            catch_panic_py!(new_state(
                py,
                &axes,
                u0,
                diffusivity,
                reaction,
                params,
                boundary,
                (substeps, t0),
                &stiff
            )
            .map(|state| Self { state }))
        }
    }
);

rd_class!(
    #[pyclass(name = "ReactionDiffusion2D")]
    ReactionDiffusion2D,
    "2-D system on `[xmin, xmax]×[ymin, ymax]`; `u0` of shape `(K, ny, nx)`.",
    {
        #[new]
        #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, u0, *, diffusivity, reaction, params = None, boundary = "reflect", substeps = 1, t0 = 0.0, reaction_solver = "rk4", rtol = 1e-6, atol = 1e-9, jacobian = None))]
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
            diffusivity: &Bound<'_, PyAny>,
            reaction: &Bound<'_, PyAny>,
            params: Option<&Bound<'_, PyDict>>,
            boundary: &str,
            substeps: u32,
            t0: f64,
            reaction_solver: &str,
            rtol: f64,
            atol: f64,
            jacobian: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            let stiff = StiffOptions {
                solver: reaction_solver,
                rtol,
                atol,
                jacobian,
            };
            let axes = [(xmin, xmax, nx), (ymin, ymax, ny)];
            catch_panic_py!(new_state(
                py,
                &axes,
                u0,
                diffusivity,
                reaction,
                params,
                boundary,
                (substeps, t0),
                &stiff
            )
            .map(|state| Self { state }))
        }
    }
);

rd_class!(
    #[pyclass(name = "ReactionDiffusion3D")]
    ReactionDiffusion3D,
    "3-D system on a box; `u0` of shape `(K, nz, ny, nx)`.",
    {
        #[new]
        #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, u0, *, diffusivity, reaction, params = None, boundary = "reflect", substeps = 1, t0 = 0.0, reaction_solver = "rk4", rtol = 1e-6, atol = 1e-9, jacobian = None))]
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
            diffusivity: &Bound<'_, PyAny>,
            reaction: &Bound<'_, PyAny>,
            params: Option<&Bound<'_, PyDict>>,
            boundary: &str,
            substeps: u32,
            t0: f64,
            reaction_solver: &str,
            rtol: f64,
            atol: f64,
            jacobian: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            let stiff = StiffOptions {
                solver: reaction_solver,
                rtol,
                atol,
                jacobian,
            };
            let axes = [(xmin, xmax, nx), (ymin, ymax, ny), (zmin, zmax, nz)];
            catch_panic_py!(new_state(
                py,
                &axes,
                u0,
                diffusivity,
                reaction,
                params,
                boundary,
                (substeps, t0),
                &stiff
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
