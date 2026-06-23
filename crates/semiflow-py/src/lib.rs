//! `semiflow` — `PyO3` Python bindings for `semiflow-core`.
//!
//! ## Contents
//!
//! - `Heat1D` — 1-D heat-equation state (unit diffusion `a = 1.0`).
//! - `SemiflowError` — discriminated exception class with a `kind: str`
//!   attribute.
//! - `version()` — crate version string.
//!
//! ## Error model
//!
//! All fallible operations raise `SemiflowError`.  The `.kind` attribute is a
//! string matching the `SemiflowStatus` C-ABI names from `semiflow-ffi`:
//! `"GridMismatch"`, `"NanInf"`, `"OutOfDomain"`, `"BoundaryFailure"`,
//! `"CflViolated"`, `"ConvergenceFailed"`, `"Unsupported"`, `"Panic"`.
//!
//! ```python
//! from semiflow import Heat1D, SemiflowError
//! try:
//!     state.evolve(-1.0)
//! except SemiflowError as e:
//!     if e.kind == "OutOfDomain":
//!         ...
//! ```
//!
//! ## GIL policy
//!
//! `Heat1D.evolve` releases the GIL during the inner pure-Rust compute loop
//! (ADR-0031); the three-phase design is documented in `state.rs`.
//!
//! ## Safety note
//!
//! `#![allow(unsafe_code)]` is required: the `#[pymodule]` proc-macro
//! expands `unsafe` blocks inside this file (`PyO3` initialisation code).
//!
//! ## Scope (v0.9.0-beta binding-parity wave)
//!
//! Broad parity with `semiflow-core` across the following families:
//!
//! - **1D diffusion** — `Heat1D`, `Heat1D4th/6th`, `TruncatedExp/4th`,
//!   `DriftReaction1D`, `Shift1D`, `Strang1D`.
//! - **2D/3D Strang tensor product** — `Heat2D/3D`, `Heat2DVarA/3DVarA`.
//! - **Non-separable / anisotropic** — `NonSeparable2D`, `NonSeparable2DAniso`,
//!   `AnisotropicShiftND2/3`.
//! - **High-dimensional sparse grid** — `SmolyakD6`.
//! - **Boundary conditions** — `Killing1D`, `Reflected1D`, `Robin1D`,
//!   `Resolvent1D`, `KilledDirichlet1D`, `ObstacleChernoff1D`.
//! - **Schrödinger** — real and complex variants.
//! - **Matrix diffusion** — `MatrixDiffusion1D`.
//! - **Nonautonomous / resolvent** — `Howland1D`, `Subordinated1D`,
//!   `ResolventJumpChernoff` (1D/2D/3D).
//! - **Manifold** — `ManifoldChernoff` (Torus, Sphere2, Hyperbolic2).
//! - **Hypoelliptic / sub-Riemannian** — Heisenberg, Kolmogorov, Engel.
//! - **Graph** — `GraphHeat`, `GraphHeat4th`, `MagnusGraphHeat`,
//!   `VarCoefGraphHeat`, `QuantumGraphHeat`, `StrangGraph`.
//! - **S³ flagship carriers** (ADR-0171) — `TtEvolver`, `TtCoupledEvolver`,
//!   `GridlessEvolver`.
//! - **Adjoint / Greeks / adaptive** — `AdjointFokkerPlanck`,
//!   `EvolverHeat1DGreeksV3`, `AdaptivePI`, `Adjoint1D`.
//! - **Carnot / point evaluation** — `ComplexTripleJump`, `PointEval`.
//!
//! **PyO3-only deferrals:** `ObstacleND`, `ObstacleGamma`, `GraphTraj`,
//! Laplacian introspection, and `GraphAdjoint` dense read-back are not yet
//! exposed (closure / dense-matrix surfaces require additional ABI design).
//!
//! See ADR-0028 for the binding split rationale and ABI stability roadmap.

#![allow(unsafe_code)]

use pyo3::prelude::*;

mod adaptive;
mod adjoint;
mod adjoint_fp_py;
mod anisotropic_nd;
mod anisotropic_nd2;
mod anisotropic_nd3;
mod anisotropic_nd_helpers;
mod bc_kernels;
mod bc_kernels2;
mod boundary;
mod carnot_complex_py;
mod coeff;
mod coeff2d;
mod diffusion_extra;
mod diffusion_extra2;
mod diffusion_hi;
mod drift_reaction_py;
mod drift_reaction_zeta4_py;
mod dtype_dispatch;
mod error;
mod geometry;
mod geometry_hypoelliptic;
mod graph_adjoint;
mod graph_extra;
mod graph_extra_heat;
mod graph_heat_f32;
mod graph_py;
mod graph_sensitivity_py;
mod graph_v2_4;
mod greeks_py;
mod handle;
mod hormander_py;
mod laplacian_introspect;
mod magnus6;
mod magnus_graph_py;
mod nonseparable;
mod obstacle_build;
mod obstacle_gamma_py;
mod obstacle_py;
mod panic;
mod resolvent_jump_nd_py;
mod resolvent_jump_py;
mod reverse_ad_py;
mod schrodinger;
mod schrodinger_complex_py;
mod schrodinger_helpers;
mod send_assertions;
mod shift1d_py;
mod smolyak_py;
mod state;
mod state_1d_chunked;
mod structured;
mod structured_matrix;
mod structured_point;
mod structured_traj;
mod subordinated_py;
mod time_dependent;
mod v3;
mod gridless_py;
mod tt_coupled_py;
mod tt_py;
mod tt_varcoef_py;
mod wentzell_helpers;
mod wentzell_py;
mod expmv_py;
mod killing_soft_py;
mod matrix_2d3d_py;
mod zeta4_py;
mod zeta6_py;

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

/// `PyO3` module definition for `semiflow`.
///
/// Registration is split into three helper functions (suckless fn-size limit).
/// Each helper handles a cohesive wave of pyclasses; see comments therein.
#[pymodule]
fn semiflow(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_core_v2(py, m)?;
    register_adr_111_waves(py, m)?;
    register_v6_v8(py, m)?;
    Ok(())
}

/// Register v2.x core + graph surface (`Heat1D`, Graph*, ADR-0055/0057/0058/0059).
fn register_core_v2(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<state::Heat1D>()?;
    m.add_class::<state::Heat2D>()?;
    m.add_class::<state::Heat3D>()?;
    m.add_class::<diffusion_hi::Heat1D4th>()?;
    m.add_class::<diffusion_hi::Heat1D6th>()?;
    m.add_class::<drift_reaction_py::DriftReaction1D>()?;
    m.add_class::<shift1d_py::Shift1D>()?;
    m.add_class::<schrodinger::Schrodinger1D>()?;
    m.add_class::<graph_py::GraphPath>()?;
    m.add_class::<graph_py::GraphHeat>()?;
    m.add_class::<magnus_graph_py::MagnusGraphHeat>()?;
    m.add_class::<graph_extra::PyGraph>()?;
    m.add_class::<graph_extra::PyLaplacian>()?;
    m.add_class::<graph_extra::GraphHeat4th>()?;
    m.add_class::<graph_extra::VarCoefGraphHeat>()?;
    m.add_class::<magnus6::MagnusGraphHeat6>()?;
    m.add_class::<graph_v2_4::GraphHeat6>()?;
    m.add_class::<graph_v2_4::VarCoefMagnusGraph>()?;
    m.add_class::<nonseparable::NonSeparable2D>()?;
    m.add_class::<adjoint::Adjoint>()?;
    m.add_class::<graph_adjoint::GraphAdjoint>()?;
    m.add_class::<adaptive::AdaptivePI>()?;
    m.add("SemiflowError", py.get_type::<error::SemiflowError>())?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    // v3.0 surface (ADR-0076 — additive; v2 surface above unchanged)
    v3::register(py, m)?;
    // v4.1 Phase D — PyO3 parity for new Rust APIs
    hormander_py::register(py, m)?;
    zeta4_py::register(py, m)?;
    zeta6_py::register(py, m)?;
    // bind-remaining-operators wave
    expmv_py::register(py, m)?;
    drift_reaction_zeta4_py::register(py, m)?;
    killing_soft_py::register(py, m)?;
    matrix_2d3d_py::register(py, m)
}

/// Register ADR-0111 Wave P1–P7 pyclasses.
fn register_adr_111_waves(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // P1: 1-D diffusion completeness
    diffusion_extra::register(py, m)?;
    // P2: complex Schrödinger
    schrodinger_complex_py::register(py, m)?;
    // P3: boundary-condition kernels
    bc_kernels::register(py, m)?;
    // P4: nonautonomous + subordinated
    time_dependent::register(py, m)?;
    // P5: geometry — manifold + hypoelliptic
    geometry::register(py, m)?;
    // P6: quantum graphs, matrix diffusion, point-eval, graph traj
    structured::register(py, m)?;
    // P7: multi-D anisotropic + 2D/3D variable-coefficient constructors
    anisotropic_nd::register(py, m)
}

/// Register v6.3–v8.1 pyclasses (obstacle, Greeks, resolvent jump, FP, Smolyak, Carnot).
fn register_v6_v8(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    graph_sensitivity_py::register(py, m)?;
    obstacle_py::register(py, m)?;
    greeks_py::register(py, m)?;
    resolvent_jump_py::register(py, m)?;
    resolvent_jump_nd_py::register(py, m)?;
    adjoint_fp_py::register(py, m)?;
    smolyak_py::register(py, m)?;
    carnot_complex_py::register(py, m)?;
    obstacle_gamma_py::register(py, m)?;
    wentzell_py::register(py, m)?;
    // v9.0.0 Shift B — reverse-mode AD (math §51.5, ADR-0156)
    reverse_ad_py::register(py, m)?;
    // v9 S³ flagship carriers — tensor-train + gridless particle (ADR-0171)
    tt_py::register(py, m)?;
    tt_coupled_py::register(py, m)?;
    gridless_py::register(py, m)?;
    // VarCoefTt (issue #2, ADR-0178): additive-separable variable-coefficient TT evolver
    tt_varcoef_py::register(py, m)
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Return the `semiflow-py` crate version string (e.g. ``"0.10.0"``).
///
/// Matches the Cargo package version baked in at compile time.
/// Identical to calling ``importlib.metadata.version("semiflow-pde")`` but
/// available without the import overhead.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
