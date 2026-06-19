//! # `semiflow-core` — Chernoff approximations of operator semigroups
//!
//! Implements formula (6) of Theorem 6 from
//! I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) 124–135,
//! [DOI 10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q),
//! [arXiv:2301.06765](https://arxiv.org/abs/2301.06765).
//!
//! The library is `no_std`-compatible (requires `alloc`). Feature flags:
//! `std` (error trait), `simd` (AVX2/NEON; default-on), `parallel` (multi-thread
//! `Strang2D`), `linear-interp` (linear boundary interpolation).
//!
//! ## Quickstart
//!
//! See [`crates/semiflow-core/README.md`](https://docs.rs/semiflow-core) for a
//! fuller introduction and a worked advection-diffusion example.
//!
//! ## Exports
//!
//! ### v0.1.0
//!
//! - **Formula (6)** via [`ShiftChernoff1D`]: the four-term Chernoff function
//!   for `L = a(x)∂²_x + b(x)∂_x + c(x)`.
//! - **Chernoff iteration** via [`ChernoffSemigroup`]: `(S(t/n))^n f`.
//! - **Heat-kernel oracle** (G1-legacy/G2-legacy regression tests): use
//!   `ShiftChernoff1D { a: |_| 0.5, b: |_| 0.0, c: |_| 0.0 }` with
//!   initial datum `exp(-x²)` and oracle `(1+2t)^{-1/2} exp(-x²/(1+2t))`.
//!
//! ### v0.2.0 — operator splitting `L = A + B` (ADR-0006)
//!
//! - **Diffusion Chernoff** via [`DiffusionChernoff`]: 5-point order-2 formula
//!   for `A = a(x)∂²_x`.
//! - **Drift-reaction Chernoff** via [`DriftReactionChernoff`]: exact
//!   characteristic-flow formula for `B = b(x)∂_x + c(x)`.
//! - **Strang composition** via [`StrangSplit`]: `Φ(τ) = D(τ/2) ∘ R(τ) ∘ D(τ/2)`,
//!   global order 2 (G3-strang gate: slope ≤ −1.95).
//! - **Advection-diffusion oracle** (G1/G2/G3-strang acceptance tests):
//!   `∂_t u = ½ ∂_xx u + ½ ∂_x u`, oracle `u(1,x) = 3^{-1/2} exp(-(x+0.5)²/3)`.
//!
//! ### v0.5.0 — 2D tensor-product (ADR-0012)
//!
//! - Tensor geometry [`Grid2D`] (row-major, x fast axis).
//! - 2D state [`GridFn2D`] (`impl State`, single `Vec<f64>`).
//! - Per-axis lift adapter [`AxisLift`] with [`Axis::X`] / [`Axis::Y`].
//! - 2D palindromic Strang [`Strang2D`]: `Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2)`,
//!   global order 2 for separable `L = Lx ⊗ I + I ⊗ Ly`.
//! - Closed-form 2D heat oracle in `tests/heat_2d_oracle.rs`.
//!
//! See `contracts/semiflow-core.math.md` §10 (Theorem 7), `tensor.yaml`,
//! and `docs/adr/0012-tensor-product-2d.md`.
//!
//! ### v0.9.0 — 3D tensor-product, generic-over-float, non-separable 2D (ADR-0024, ADR-0023, ADR-0025)
//!
//! - 3D tensor geometry [`Grid3D`] (x-fastest, `idx(i,j,k) = k·nx·ny + j·nx + i`).
//! - 3D state [`GridFn3D`] (`impl State`, single `Vec<f64>`).
//! - Per-axis lift adapter [`AxisLift3D`] with extended [`Axis::Z`] variant.
//! - 3D palindromic Strang [`Strang3D`]:
//!   `Sx(τ/2) ∘ Sy(τ/2) ∘ Sz(τ) ∘ Sy(τ/2) ∘ Sx(τ/2)`,
//!   global order = min(order per axis) by Theorem 7' (Lemma 10.1, BCH residue = 0).
//! - Closed-form 3D heat oracle in `tests/heat_3d_oracle.rs`.
//! - Anisotropic non-separable 2D via [`NonSeparable2DAnisotropicChernoff`] (ADR-0023).
//! - All grid and Chernoff types are now generic over [`SemiflowFloat`] (`f32`/`f64`).
//!
//! See `contracts/semiflow-core.math.md` §10.7-ter, §10.8; `docs/adr/0024-tensor-3d.md`.
//!
//! ### v9.0.0 — third S-curve: reverse AD, tensor-train carrier, gridless particle
//!
//! Three additive shifts; no existing kernel semantics change.
//!
//! - **Shift B — [`ReverseChernoff`] + [`CheckpointSchedule`]** (§51, ADR-0156):
//!   reverse-mode differentiable Chernoff layer via binomial checkpointing.
//!   The adjoint of `(F(τ))ⁿ` is the *transposed* product of the same shift-and-scale
//!   steps — algebraically exact, no implicit backward solve, no external autodiff crate.
//!   Binomial checkpointing achieves `O(√n)` peak memory (slope 0.39, `G_REVERSE_AD_CHECKPOINT`
//!   PASS).  The full gradient matches central-difference and forward-mode `Dual<F>` (§46)
//!   to 0 ULP (`G_REVERSE_AD_GRADIENT` PASS, cross-mode parity).  **Narrow scope**:
//!   constructable only for `C: LinearChernoffFamily`; transpose-exactness is not claimed
//!   for variable-coefficient or nonlinear kernels.
//!
//! - **Shift C — [`TtChernoff`] + [`TtState`]** (§52, ADR-0159):
//!   tensor-train state carrier that escapes the curse of dimensionality for the
//!   **linear diagonal-`A` Gaussian class**.  Each Chernoff step applies d rank-O(1)
//!   per-axis TT-operators (Kazeev–Khoromskij) followed by deterministic TT-rounding
//!   (in-tree one-sided Jacobi SVD, no LAPACK, no new dependency).  For constant diagonal
//!   `A`, TT-rank is algebraically capped at r ≤ d/2 (Rohrbach–Dolgov–Grasedyck–Scheichl),
//!   giving storage `O(d³·n)` — polynomial in d.  The rank-1 case (`r=1`) reduces exactly
//!   to the shipped Strang⊗ tensor product.  Validated for d ∈ {4, 6, 8, 10}
//!   (`G_TT_CHERNOFF_DIMSCALING`, `slow-tests`).  Off-diagonal / variable-coefficient /
//!   nonlinear operators are research-track.
//!
//! - **[`GridlessChernoff`] + [`ParticleReduction`]** (§50, ADR-0155):
//!   deterministic branching particle evolver on a `MeasureState` ensemble — the §38
//!   adjoint push-forward on weighted Diracs, implemented at O(d·P) per step independent
//!   of any ambient grid.  Validated at d=2 (`G_GRIDLESS_DIM_SCALING` PASS, sup-error
//!   1.197e-3 < 5e-3).  The pre-registered variance go/no-go fired (`G_GRIDLESS_VARIANCE`
//!   NO-GO, 1.417× MSE ratio at d=2, below the ≥2× gate); high-dimensional curse
//!   re-enters through the particle-reduction grid at d≥3 — an intrinsic limit of the
//!   particle representation.  This result is **retained as a documented negative** and a
//!   correct d=2 primitive; high-dimensional gridless evolution is research-track
//!   (see ADR-0155 Amendment 1 and the separate [`TtChernoff`] resolution in §52).
//!
//! See `contracts/semiflow-core.math.md` §50–52; `docs/adr/0155`, `0156`, `0159`.
//!
//! ### v9.2.0 — S³ honest-scope public API (ADR-0169, `s3-poc` feature)
//!
//! Promotes the five S³ POC evolvers from `pub(crate)` to a curated public surface
//! behind the non-default `s3-poc` feature.  Three honesty layers are enforced:
//!
//! 1. **Type wall** — boundary-as-type wrapper constructors accept only in-class
//!    arguments; out-of-class operators are unconstructible at the type level.
//! 2. **Feature gate** — all six tokens are `#[cfg(feature = "s3-poc")]`; a build
//!    without the feature sees none of them.
//! 3. **Rustdoc stanza** — every public S³ type carries a normative
//!    `## Proven boundary` section citing the RELEASE-BLOCKING gate and the
//!    mathematical scope.
//!
//! Six tokens: [`S3DriftSpectralEvolver`], [`S3DenseCouplingEvolver`],
//! [`S3VarCoefEvolver`], [`S3NonSepVarCoefEvolver`], [`S3BurgersColeHopf`],
//! [`S3ReactionDiffusion`].  Container types: [`AxisCoef`], [`CpTerm`], [`CpCoef`],
//! [`CoefRole`], [`Reaction`].  See `docs/adr/0169-s3-honest-scope-public-api-promotion.md`.
//!
//! See `contracts/semiflow-core.math.md` for the full mathematical specification.
//!
//! ## Quick start
//!
//! ```rust
//! use semiflow_core::{Grid1D, GridFn1D, ShiftChernoff1D, ChernoffSemigroup};
//!
//! // Heat equation: ∂_t u = 0.5 ∂_xx u
//! let grid  = Grid1D::new(-10.0, 10.0, 1000)
//!     .expect("grid bounds and node count are valid");
//! let func  = ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.0, grid);
//! let semi  = ChernoffSemigroup::new(func, 100)
//!     .expect("n=100 satisfies the n >= 1 precondition");
//! let u0    = GridFn1D::from_fn(grid, |x| (-x * x).exp());
//! let u1    = semi.evolve(1.0, &u0)
//!     .expect("evolve should not fail for valid inputs");
//!
//! // Oracle: u(1,x) = (3)^{-1/2} exp(-x²/3)
//! let oracle = GridFn1D::from_fn(grid, |x| (3.0_f64).sqrt().recip() * (-(x*x)/3.0).exp());
//! use semiflow_core::State;
//! let mut diff = u1.clone();
//! diff.axpy(-1.0, &oracle);
//! assert!(diff.norm_sup() < 5e-4, "G1 tolerance exceeded: {}", diff.norm_sup());
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

pub mod adaptive;
pub mod adjoint;
pub mod adjoint_fp;
pub mod approximation;
pub mod axis;
pub mod boundary;
pub mod carnot_complex;
pub(crate) mod carnot_complex_helpers;
pub mod carnot_stepk;
pub(crate) mod carnot_stepk_helpers;
pub mod chernoff;
pub mod complex;
pub mod controller;
pub mod diffusion;
pub mod diffusion4;
pub mod diffusion4_zeta4;
pub(crate) mod diffusion4_zeta4_stencil_ho;
pub mod diffusion6;
pub mod diffusion6_zeta6;
pub mod diffusion8_zeta8;
mod diffusion_storage;
pub mod drift_reaction;
pub mod drift_reaction_zeta4;
pub mod dual;
pub mod error;
pub mod expmv;
pub mod float;
pub(crate) mod gen_quadrature;
pub mod graph;
pub mod graph_heat;
pub mod graph_heat4;
pub mod graph_heat6;
pub mod graph_sensitivity;
pub(crate) mod graph_sensitivity_helpers;
mod graph_sensitivity_tests;
pub mod graph_signal;
pub mod graph_traj;
pub mod graph_var_coef;
pub mod grid;
pub mod grid2d;
pub mod grid3d;
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
pub mod killed_dirichlet;
pub mod killing;
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
pub mod matrix_2d3d;
pub(crate) mod matrix_inv;
pub(crate) mod matrix_pade;
pub(crate) mod matrix_pade_complex;
pub(crate) mod matrix_strang;
pub mod matrix_system;
pub mod matrix_system_complex;
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
pub(crate) mod pencil;
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
/// Backward sweep internals for `reverse_ad` (additive split, ≤500-line cap).
pub(crate) mod reverse_sweep;
pub mod robin;
pub mod schrodinger;
pub mod schrodinger_complex;
pub(crate) mod schrodinger_complex_state;
pub mod scratch;
pub mod shift1d;
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
pub mod strang3d;
pub mod strang3d_axislift;
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
#[doc(hidden)]
pub mod strang3d_parallel;
pub mod strang_graph;
pub mod subordinated;
pub mod truncated_exp;
pub mod truncated_exp4;
pub mod truncated_exp4_cached;
pub mod tt_chernoff;
pub mod tt_core;
pub mod tt_coupled;
pub(crate) mod tt_coupled_pair;
pub(crate) mod tt_dense_expm;
#[cfg(feature = "s3-poc")]
pub mod tt_dense_coupling;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_dense_coupling;

#[cfg(feature = "s3-poc")]
pub mod tt_drift_spectral;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_drift_spectral;

#[cfg(feature = "s3-poc")]
pub mod tt_nonlinear_spectral;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_nonlinear_spectral;

pub(crate) mod tt_spectral;

#[cfg(feature = "s3-poc")]
pub mod tt_nonsep_varcoef;
#[cfg(not(feature = "s3-poc"))]
pub(crate) mod tt_nonsep_varcoef;

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
pub mod tt_nonsep_varcoef_api;
#[cfg(feature = "s3-poc")]
pub mod tt_nonlinear_spectral_api;
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
    error::SemiflowError,
    expmv::DiffusionExpmvChernoff,
    float::SemiflowFloat,
    graph::{Graph, Laplacian, LaplacianKind},
    graph_heat::GraphHeatChernoff,
    graph_heat4::GraphHeat4thChernoff,
    graph_heat6::GraphHeat6thChernoff,
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
    killing_soft::{ClosureKillingRate, Killing2ndChernoff, KillingRate},
    magnus6_graph::MagnusGraphHeat6thChernoff,
    magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff},
    manifold::{BoundedGeometryManifold, Hyperbolic2, Sphere2, Torus},
    manifold_chernoff::ManifoldChernoff,
    manifold_kahler::FubiniStudyCp1,
    matrix_2d3d::{
        MatrixDiffusionChernoff2D, MatrixDiffusionChernoff3D, MatrixGridFn2D, MatrixGridFn3D,
    },
    matrix_system::{MatrixDiffusionChernoff, MatrixGridFn1D},
    matrix_system_complex::{MatrixDiffusionChernoffComplex, MatrixGridFnComplex1D},
    nonseparable2d::NonSeparable2DChernoff,
    nonseparable2d_aniso::NonSeparable2DAnisotropicChernoff,
    nonseparable_mixed::NonSeparableMixedChernoff,
    obstacle::{ClosureObstacle, ConstantObstacle, Obstacle, ObstacleChernoff},
    obstacle_nd::ObstacleChernoffND,
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
    truncated_exp::TruncatedExpDiffusionChernoff,
    truncated_exp4::{TruncatedExp4WithCache, TruncatedExp4thDiffusionChernoff},
    // TT-Chernoff (v9.0.0 Shift C): TT representation of the Chernoff semigroup.
    // Escapes the exponential curse (polynomial TT-rank) for linear diagonal-A
    // (Gaussian class). See math §50 / contracts §50.2 / ADR-0159.
    tt_chernoff::{TtChernoff, TtState},
    // CoupledTtChernoff (v9.1.0 Shift C RESOLUTION): genuine cross-axis coupling.
    // Applies D1_j⊗D1_k pair-bond operators; grows rank from a rank-1 IC.
    // See math §52.9 / ADR-0159 Amendment 1.
    tt_coupled::{CoupledTtChernoff, CouplingTopology},
    varcoef_magnus_graph::{compute_rho_bar, VarCoefMagnusGraphHeatChernoff, WeightAtTime},
    wentzell::{DynamicWentzellChernoff, HalfSpaceWentzell, WentzellRegion},
};

// ── S³ public surface (v9.2.0, ADR-0169) ────────────────────────────────────
// All six tokens are behind the non-default `s3-poc` feature.
// Each wrapper enforces its class boundary at construction time (boundary-as-type).
#[cfg(feature = "s3-poc")]
#[cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]
pub use crate::{
    tt_drift_spectral_api::S3DriftSpectralEvolver,
    tt_dense_coupling_api::S3DenseCouplingEvolver,
    tt_varcoef_spectral::{AxisCoef, S3VarCoefEvolver},
    tt_nonsep_varcoef::{CoefRole, CpCoef, CpTerm},
    tt_nonsep_varcoef_api::S3NonSepVarCoefEvolver,
    tt_nonlinear_spectral::Reaction,
    tt_nonlinear_spectral_api::{S3BurgersColeHopf, S3ReactionDiffusion},
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
