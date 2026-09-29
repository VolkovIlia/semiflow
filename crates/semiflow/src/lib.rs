#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

// The engines share coefficient closures through `alloc::sync::Arc`, which needs
// pointer-sized atomics (ADR-0200). Say so instead of failing with dozens of
// unresolved `alloc::sync` imports on targets such as `thumbv6m-none-eabi`.
#[cfg(not(target_has_atomic = "ptr"))]
compile_error!(
    "semiflow requires a target with pointer-sized atomics (it uses alloc::sync::Arc); \
     e.g. thumbv7m/thumbv7em/riscv32imac work, thumbv6m does not"
);

// Unit tests run on the host; when the crate is built `no_std` (`--no-default-features`)
// they still need `std` for the test harness, printing and `format!`/`vec!`.
#[cfg(test)]
#[macro_use]
extern crate std;

pub mod adaptive;
pub mod adjoint;
pub mod adjoint_fp;
pub mod approximation;
pub mod axis;
pub mod boundary;
pub(crate) mod boundary_value;
pub mod carnot_complex;
pub(crate) mod carnot_complex_helpers;
pub mod carnot_stepk;
pub(crate) mod carnot_stepk_helpers;
pub mod chernoff;
pub mod complex;
pub mod conservative;
pub mod conservative_assemble;
pub mod controller;
pub mod diffusion;
pub mod diffusion4;
pub mod diffusion4_zeta4;
pub(crate) mod diffusion4_zeta4_stencil_ho;
pub mod diffusion6;
pub mod diffusion6_zeta6;
pub mod diffusion8_zeta8;
mod diffusion_storage;
pub(crate) mod diffusion_zeta_common;
pub mod drift_reaction;
pub mod drift_reaction_zeta4;
pub mod dual;
pub mod entry_sensitivity;
pub mod error;
pub mod etdrk4;
pub(crate) mod etdrk4_helpers;
pub mod expmv;
pub mod float;
pub(crate) mod gen_quadrature;
pub mod general_operator;
pub mod generator_action;
pub mod graph;
pub mod graph_adjoint_presampled;
pub mod graph_batched;
mod graph_batched_tests;
pub mod graph_frechet;
pub mod graph_heat;
pub mod graph_heat4;
pub mod graph_heat6;
pub mod graph_krylov;
pub mod graph_sensitivity;
pub(crate) mod graph_sensitivity_helpers;
mod graph_sensitivity_tests;
pub mod graph_signal;
pub mod graph_traj;
pub mod graph_var_coef;
pub mod grid;
pub mod grid2d;
pub mod grid3d;
pub mod grid_batched;
pub(crate) mod grid_chebyshev;
pub(crate) mod grid_chebyshev_nodes;
pub(crate) mod grid_chebyshev_octonic;
pub(crate) mod grid_chebyshev_septic;
pub(crate) mod grid_cubic;
pub mod grid_fn;
pub mod grid_fn2d;
pub mod grid_fn3d;
pub mod grid_nd;
pub mod gridless;
pub(crate) mod gridless_reduce;
pub mod hdr;
pub mod heisenberg_kernel;
pub mod hormander;
pub mod hormander_engel;
pub(crate) mod hormander_engel_helpers;
pub(crate) mod hormander_heisenberg;
pub mod howland;
pub(crate) mod interp_stencil;
pub mod killed_dirichlet;
pub mod killing;
pub mod killing_order2;
pub mod killing_soft;
pub mod magnus6_graph;
pub mod magnus_graph;
pub mod magnus_graph_adjoint;
pub(crate) mod magnus_graph_helpers;
#[cfg(test)]
mod magnus_graph_tests;
pub mod manifold;
pub mod manifold_chernoff;
pub mod manifold_hyperbolic;
pub mod manifold_kahler;
pub mod mass_operator;
pub mod matrix_2d3d;
pub(crate) mod matrix_inv;
pub(crate) mod matrix_pade;
pub(crate) mod matrix_pade_complex;
pub(crate) mod matrix_strang;
pub mod matrix_system;
pub mod matrix_system_complex;
pub mod multilayer;
pub mod nonlinearity;
pub mod nonseparable2d;
pub mod nonseparable2d_aniso;
pub mod nonseparable_mixed;
pub mod nonseparable_mixed_closure;
pub mod obstacle;
pub mod obstacle_gamma;
pub mod obstacle_nd;
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
#[doc(hidden)]
pub mod parallel1d;
#[cfg(not(feature = "parallel"))]
pub(crate) mod parallel1d;
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
#[doc(hidden)]
pub mod parallel_pool;
pub(crate) mod pcg;
pub(crate) mod pencil;
pub mod phi_action;
pub(crate) mod phi_action_helpers;
pub mod point_eval;
pub mod quantum_graph;
pub(crate) mod quantum_graph_data; // internal helpers; no public re-export
pub mod quantum_schrodinger;
pub mod reflection;
pub mod reflection_regions;
pub mod resolvent;
pub mod resolvent_complex;
pub mod resolvent_jump;
pub mod resolvent_jump_nd;
pub(crate) mod resolvent_quad;
/// Residual gate-wrapper and `Sampleable<GridFn1D>` impl (suckless split from `resolvent`).
pub(crate) mod resolvent_residual;
pub mod reverse_ad;
/// DoF-aligned region partition for K>1 reverse-AD (§51.10, ADR-0177).
pub mod reverse_region;
/// Backward sweep internals for `reverse_ad` (additive split, ≤500-line cap).
pub(crate) mod reverse_sweep;
pub mod robin;
pub mod schrodinger;
pub mod schrodinger_complex;
pub(crate) mod schrodinger_complex_state;
pub mod scratch;
pub mod shift1d;
pub mod shift1d_vjp;
pub mod shift_nd;
pub mod shift_nd_adaptive;
pub(crate) mod shift_nd_gauss;
pub mod shift_nd_zeta2;
#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
#[doc(hidden)]
pub mod simd;
pub mod smolyak;
pub mod state;
pub mod strang;
pub mod strang2d;
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
#[doc(hidden)]
pub mod strang2d_parallel;
pub mod strang2d_pencil;
pub mod strang3d;
pub mod strang3d_axislift;
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
#[doc(hidden)]
pub mod strang3d_parallel;
pub mod strang_graph;
pub mod subordinated;
pub mod symmetric_operator;
pub mod truncated_exp;
pub mod truncated_exp4;
pub mod truncated_exp4_cached;
pub mod tt_chernoff;
pub mod tt_core;
pub mod tt_coupled;
pub(crate) mod tt_coupled_pair;
#[cfg(feature = "s3-poc")]
pub mod tt_dense_coupling;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_dense_coupling;
pub(crate) mod tt_dense_expm;
pub mod tt_varcoef;

#[cfg(feature = "s3-poc")]
pub mod tt_drift_spectral;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_drift_spectral;
#[cfg(feature = "s3-poc")]
pub mod tt_nonlinear_spectral;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_nonlinear_spectral;
#[cfg(feature = "s3-poc")]
pub mod tt_nonsep_varcoef;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_nonsep_varcoef;
pub(crate) mod tt_spectral;
#[cfg(feature = "s3-poc")]
pub mod tt_varcoef_spectral;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_varcoef_spectral;

// S³ sibling API modules (boundary-as-type wrappers, ADR-0169).
#[cfg(feature = "s3-poc")]
pub mod tt_dense_coupling_api;
#[cfg(feature = "s3-poc")]
pub mod tt_drift_spectral_api;
#[cfg(feature = "s3-poc")]
pub mod tt_nonlinear_spectral_api;
#[cfg(feature = "s3-poc")]
pub mod tt_nonsep_varcoef_api;
pub mod varcoef_magnus_graph;
pub mod wentzell;

pub use crate::{
    adaptive::{AdaptiveOutcome, AdaptivePI},
    adjoint::{AdjointApply, AdjointChernoff},
    adjoint_fp::{AdjointFokkerPlanckChernoff, Adjointable, MeasureState},
    approximation::{assert_in_subspace, ApproximationSubspace, LadderRung},
    axis::{Axis, AxisLift},
    carnot_complex::{ComplexTripleJump, CplxGridFn5, GAMMA_STAR},
    carnot_stepk::{Filiform5X1, Filiform5X2},
    chernoff::{ApplyChernoffExt, ChernoffFunction, ChernoffSemigroup, Evolver, Growth},
    complex::SemiflowComplex,
    conservative::{steady_state_dirichlet_1d, ConservativeDiffusionChernoff},
    conservative_assemble::{assemble_conservative_csr_1d, assemble_conservative_csr_nd},
    controller::{ClassicalPI, H211bFilter, StepController},
    diffusion::DiffusionChernoff,
    diffusion4::Diffusion4thChernoff,
    diffusion4_zeta4::Diffusion4thZeta4Chernoff,
    diffusion6::Diffusion6thChernoff,
    diffusion6_zeta6::Diffusion6thZeta6Chernoff,
    diffusion8_zeta8::Diffusion8thZeta8Chernoff,
    drift_reaction::DriftReactionChernoff,
    drift_reaction_zeta4::DriftReactionZeta4Chernoff,
    dual::Dual,
    entry_sensitivity::EntrySensitivity,
    error::SemiflowError,
    etdrk4::Etdrk4,
    expmv::DiffusionExpmvChernoff,
    float::SemiflowFloat,
    generator_action::{DivFormGenerator, GeneratorAction, NegLaplacianGenerator},
    graph::{Graph, Laplacian, LaplacianKind},
    graph_adjoint_presampled::{
        fill_abscissa_times, PreSampledLaplacianSeq, PreSampledMagnusAdj, PreSampledVarCoefAdj,
    },
    graph_frechet::graph_expmv_frechet,
    graph_heat::GraphHeatChernoff,
    graph_heat4::GraphHeat4thChernoff,
    graph_heat6::GraphHeat6thChernoff,
    graph_krylov::{
        dense_graph_expmv_ref, graph_expmv_krylov, graph_expmv_matvec_count, GraphKrylovChernoff,
        KrylovPath, MAX_DENSE_N,
    },
    graph_sensitivity::{
        adjoint_state_gradient, apply_edge_weight_deriv, magnus_step_jvp_into,
        EdgeWeightSensitivity, GeneratorSensitivity, NodeTimescaleSensitivity,
    },
    graph_signal::{CsrRowIter, GraphSignal},
    graph_traj::{GraphTraj, SegmentWeightFn, MAX_GRAPH_TRAJ_SEGMENTS},
    graph_var_coef::VarCoefGraphHeatChernoff,
    grid::{BoundaryPolicy, Grid1D, InterpKind, OobPolicy},
    grid2d::Grid2D,
    grid3d::Grid3D,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    grid_fn3d::GridFn3D,
    grid_nd::{GridFnND, GridND},
    gridless::{GridlessChernoff, ParticleReduction},
    hdr::HdrSnapshot,
    heisenberg_kernel::heisenberg_heat_kernel,
    hormander::{
        HeisenbergGroup, HeisenbergX, HeisenbergY, HypoellipticChernoff, KolmogorovDiffusionX1,
        KolmogorovDriftX0, KolmogorovPhaseSpace, VectorField,
    },
    hormander_engel::{EngelX1, EngelX2},
    howland::{HowlandLift, HowlandState, TimedChernoffFunction},
    killed_dirichlet::KilledDirichletChernoff,
    killing::{BallRegion, BoxRegion, KillingChernoff, KillingRegion},
    killing_order2::DirichletHeat2ndChernoff,
    killing_soft::{ClosureKillingRate, Killing2ndChernoff, KillingRate},
    magnus6_graph::MagnusGraphHeat6thChernoff,
    magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff},
    manifold::{BoundedGeometryManifold, Hyperbolic2, Sphere2, Torus},
    manifold_chernoff::ManifoldChernoff,
    manifold_kahler::FubiniStudyCp1,
    mass_operator::{dense_massk_expmv_ref, mass_lumped_evolve, MassKOperator, TriangularFactor},
    matrix_2d3d::{
        MatrixDiffusionChernoff2D, MatrixDiffusionChernoff3D, MatrixGridFn2D, MatrixGridFn3D,
    },
    matrix_system::{MatrixDiffusionChernoff, MatrixGridFn1D},
    matrix_system_complex::{MatrixDiffusionChernoffComplex, MatrixGridFnComplex1D},
    multilayer::{multilayer_evolve, Layer, MassWeightedConservativeChernoff, MultilayerStack},
    nonlinearity::{AllenCahn, Nonlinearity, NonlinearityDiff},
    nonseparable2d::NonSeparable2DChernoff,
    nonseparable2d_aniso::NonSeparable2DAnisotropicChernoff,
    nonseparable_mixed::NonSeparableMixedChernoff,
    obstacle::{ClosureObstacle, ConstantObstacle, Obstacle, ObstacleChernoff},
    obstacle_nd::ObstacleChernoffND,
    phi_action::{dense_phi_aug_ref, phi_action, phi_action_batched, PHI_MAX},
    point_eval::{sample_gridfn2d, PointEval},
    quantum_graph::{KirchhoffVertex, QuantumGraph, QuantumGraphHeatChernoff, QuantumGraphSignal},
    quantum_schrodinger::{QuantumGraphComplexSignal, QuantumSchrödingerChernoff},
    reflection::{HalfSpaceRegion, ReflectedHeatChernoff, ReflectingRegion},
    resolvent::{
        LaplaceChernoffResolvent, LaplaceChernoffResolventResidual, LaplaceQuadrature, Sampleable,
    },
    resolvent_complex::EvalComplex,
    resolvent_jump::ResolventJumpChernoff,
    resolvent_jump_nd::{ResolventJumpChernoff2D, ResolventJumpChernoff3D},
    reverse_ad::{
        forward_with_checkpoints, recompute_segment, step_jacobian_col, CheckpointSchedule,
        ReverseChernoff, TransposeApply,
    },
    reverse_region::RegionMap,
    robin::{HalfSpaceRobin, RobinHeatChernoff, RobinRegion},
    schrodinger::{SchrodingerChernoff, SchrodingerState},
    schrodinger_complex::SchrödingerChernoffComplex,
    schrodinger_complex_state::GridFnComplex1D,
    scratch::{ScratchPool, ScratchVec},
    shift1d::ShiftChernoff1D,
    shift_nd::{AnisotropicShiftChernoffND, GaussHermiteTensor, SquareMatrix},
    shift_nd_adaptive::AnisotropicShiftAdaptiveQ,
    shift_nd_zeta2::AnisotropicShiftZeta2ND,
    smolyak::SmolyakGridND,
    state::{Discrete, HilbertState, State},
    strang::StrangSplit,
    strang2d::Strang2D,
    strang3d::Strang3D,
    strang3d_axislift::AxisLift3D,
    strang_graph::StrangSplitGraph,
    subordinated::{
        GammaSubordinator, InverseGaussianSubordinator, LevySubordinator, StableSubordinator,
        SubordinatedChernoff,
    },
    symmetric_operator::{dense_csr_expmv_ref, SymmetricLinearOp, SymmetricOperator},
    truncated_exp::TruncatedExpDiffusionChernoff,
    truncated_exp4::{TruncatedExp4WithCache, TruncatedExp4thDiffusionChernoff},
    // TT-Chernoff (v9.0.0 Shift C, §52, ADR-0159): diagonal-A Gaussian class, rank ≤ d/2.
    tt_chernoff::{TtChernoff, TtState},
    // CoupledTtChernoff (v9.1.0, ADR-0159 Am.1): cross-axis D1_j⊗D1_k pair bonds.
    tt_coupled::{CoupledTtChernoff, CouplingTopology},
    // VarCoefTt (#2, ADR-0178): additive-separable var-coef TT; rank-1-preserving.
    tt_varcoef::VarCoefTt,
    varcoef_magnus_graph::{compute_rho_bar, VarCoefMagnusGraphHeatChernoff, WeightAtTime},
    wentzell::{DynamicWentzellChernoff, HalfSpaceWentzell, WentzellRegion},
};
// ── S³ public surface (v9.2.0, ADR-0169) ────────────────────────────────────
// All six tokens are behind the non-default `s3-poc` feature.
// Each wrapper enforces its class boundary at construction time (boundary-as-type).
#[cfg(feature = "s3-poc")]
#[cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]
pub use crate::{
    tt_dense_coupling_api::S3DenseCouplingEvolver,
    tt_drift_spectral_api::S3DriftSpectralEvolver,
    tt_nonlinear_spectral::Reaction,
    tt_nonlinear_spectral_api::{S3BurgersColeHopf, S3ReactionDiffusion},
    tt_nonsep_varcoef::{CoefRole, CpCoef, CpTerm},
    tt_nonsep_varcoef_api::S3NonSepVarCoefEvolver,
    tt_varcoef_spectral::{AxisCoef, S3VarCoefEvolver},
};
/// Drain all thread-local parallel scratch pools on the calling thread.
///
/// Combines [`strang2d_parallel::drain_thread_local_pools_2d`] and
/// [`strang3d_parallel::drain_thread_local_pools_3d`]. After this call, the
/// pools are empty (both `f64` and `f32` pools); the next parallel step will
/// re-allocate (one buffer per thread) and then settle back to steady-state
/// capacity.
///
/// **Test hook only** — not part of the stable v1.0.0 API.
/// Gated on `feature = "parallel"`.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn drain_thread_local_pools() {
    strang2d_parallel::drain_thread_local_pools_2d();
    strang3d_parallel::drain_thread_local_pools_3d();
}
