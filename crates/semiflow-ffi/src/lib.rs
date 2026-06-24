//! C ABI bindings for `semiflow-core` (experimental, v0.10.0).
//!
//! ## Status
//!
//! This crate is **experimental**: the ABI is not stabilised until v1.0.0
//! (see ADR-0028, `docs/adr/0028-ffi-pyo3-wasm-v0_10.md`).
//!
//! ## Build requirement
//!
//! Always build the `cdylib` with `--profile release-ffi`:
//!
//! ```text
//! cargo build -p semiflow-ffi --profile release-ffi
//! ```
//!
//! The workspace `[profile.release]` has `panic = "abort"` which turns
//! `catch_unwind` into a no-op, breaking the FFI panic boundary.
//! `[profile.release-ffi]` overrides this to `panic = "unwind"`.
//!
//! ## Scope (v0.9.0-beta binding-parity wave)
//!
//! Near-full parity with `semiflow-core` across the following families:
//!
//! - **1D diffusion** — standard (`smf_heat1d_*`), higher-order ζ-ladder
//!   (`Diffusion4th/6th/8th`, `Zeta4/6th`), truncated-exp (`TruncExp/4th`),
//!   drift-reaction (`DriftReaction1D`), shift (`Shift1D`), Strang split.
//! - **2D/3D Strang tensor product** — `Heat2D/3D`, variable-coef (`VarA`).
//! - **Non-separable** — `NonSeparable2D`, `NonSeparable2DAniso`,
//!   `AnisotropicShiftND2/3`.
//! - **High-dimensional sparse grid** — `SmolyakD6`.
//! - **Boundary conditions** — `Killing1D`, `Reflected1D`, `Robin1D`,
//!   `Resolvent1D`, `KilledDirichlet1D`, `ObstacleChernoff` (1D).
//! - **Schrödinger** — real (`Schrodinger1D`) and complex
//!   (`SchrodingerComplex1D`).
//! - **Matrix diffusion** — `MatrixDiffusion1D`.
//! - **Nonautonomous / resolvent** — `Howland1D`, `Subordinated1D`,
//!   `ResolventJumpChernoff` (1D/2D/3D).
//! - **Manifold** — `ManifoldChernoff` (Torus, Sphere2, Hyperbolic2).
//! - **Hypoelliptic / sub-Riemannian** — Heisenberg, Kolmogorov, Engel.
//! - **Graph** — `GraphHeatChernoff`, `GraphHeat4th`,
//!   `MagnusGraphHeat`, `VarCoefGraphHeat`, `VarCoefMagnusGraphHeat`,
//!   `QuantumGraphHeatChernoff`, `StrangGraph`.
//! - **S³ flagship carriers** (ADR-0171) — `TtEvolver/TtState`,
//!   `TtCoupledEvolver`, `GridlessEvolver/MeasureState`.
//! - **Adjoint / Greeks / adaptive** — `AdjointFokkerPlanck`,
//!   `EvolverHeat1DGreeksV3`, `AdaptivePI`, `Adjoint1D`.
//! - **Carnot / point evaluation** — `ComplexTripleJump`, `PointEval`.
//!
//! **Documented deferrals (ABI-unsafe or closure-capture surfaces):**
//! `ObstacleND`, `ObstacleGamma`, `GraphTraj`, Laplacian introspection,
//! and `GraphAdjoint` read-back — dense-matrix / closure reads are not
//! expressible in a stable C ABI.
//!
//! See ADR-0028 for the binding split rationale and ABI stability roadmap.

#[macro_use]
mod panic;

pub mod adjoint_fp_ffi;
pub mod bc_ffi;
pub mod bc_ffi2;
pub mod bc_ffi3;
pub mod cdr_ffi;
pub mod diffusion_extra_ffi;
pub mod diffusion_hi_ffi;
pub mod diffusion_hi_zeta_ffi;
pub mod drift_reaction_zeta4_ffi;
pub mod expmv_ffi;
pub mod ffi;
pub mod graph_ffi;
pub mod graph_ffi_v2_4;
pub mod greeks;
pub mod gridless_ffi;
pub(crate) mod handle;
pub mod killing_soft_ffi;
pub mod matrix_2d3d_ffi;
pub mod matrix_ffi;
pub mod resolvent_jump_ffi;
pub mod resolvent_jump_nd_ffi;
pub mod schrodinger_complex_ffi;
pub mod schrodinger_ffi;
pub mod status;
pub mod strang_ffi;
pub mod strang_nd_2d_ffi;
pub mod strang_nd_3d_ffi;
pub mod tt_coupled_ffi;
pub mod tt_ffi;
pub mod v3;
pub mod wentzell_ffi;

pub use adjoint_fp_ffi::{
    smf_adjoint_fp_free_v3, smf_adjoint_fp_new_brownian_1d_v3, smf_adjoint_fp_step_v3,
    SmfAdjointFpV3,
};
pub use bc_ffi::{
    smf_killing1d_evolve, smf_killing1d_free, smf_killing1d_new, smf_killing1d_size,
    smf_killing1d_values, smf_reflected1d_evolve, smf_reflected1d_free, smf_reflected1d_new,
    smf_reflected1d_size, smf_reflected1d_values, SmfKilling1D, SmfReflected1D,
};
pub use bc_ffi2::{
    smf_killed_dir1d_apply, smf_killed_dir1d_free, smf_killed_dir1d_new, smf_killed_dir1d_size,
    smf_killed_dir1d_values, smf_resolvent1d_eval, smf_resolvent1d_free, smf_resolvent1d_new,
    smf_resolvent1d_size, smf_robin1d_evolve, smf_robin1d_free, smf_robin1d_new, smf_robin1d_size,
    smf_robin1d_values, SmfKilledDir1D, SmfResolvent1D, SmfRobin1D,
};
pub use bc_ffi3::{
    smf_dirichlet_heat2nd1d_evolve, smf_dirichlet_heat2nd1d_free, smf_dirichlet_heat2nd1d_new,
    smf_dirichlet_heat2nd1d_size, smf_dirichlet_heat2nd1d_values, SmfDirichletHeat2nd1D,
};
pub use cdr_ffi::{
    smf_drift_reaction_evolve, smf_drift_reaction_free, smf_drift_reaction_new,
    smf_drift_reaction_size, smf_drift_reaction_values, smf_shift1d_evolve, smf_shift1d_free,
    smf_shift1d_new, smf_shift1d_size, smf_shift1d_values, SmfDriftReaction, SmfShift1D,
};
pub use diffusion_extra_ffi::{
    smf_trunc_exp4_evolve, smf_trunc_exp4_free, smf_trunc_exp4_new, smf_trunc_exp4_size,
    smf_trunc_exp4_values, smf_trunc_exp_evolve, smf_trunc_exp_free, smf_trunc_exp_new,
    smf_trunc_exp_size, smf_trunc_exp_values, SmfTruncExp, SmfTruncExp4,
};
pub use diffusion_hi_ffi::{
    smf_heat1d_4th_evolve, smf_heat1d_4th_free, smf_heat1d_4th_new, smf_heat1d_4th_size,
    smf_heat1d_4th_values, smf_heat1d_6th_evolve, smf_heat1d_6th_free, smf_heat1d_6th_new,
    smf_heat1d_6th_size, smf_heat1d_6th_values, SmfHeat1D4th, SmfHeat1D6th,
};
pub use diffusion_hi_zeta_ffi::{
    smf_heat1d_zeta4_evolve, smf_heat1d_zeta4_free, smf_heat1d_zeta4_new, smf_heat1d_zeta4_size,
    smf_heat1d_zeta4_values, smf_heat1d_zeta6_evolve, smf_heat1d_zeta6_free, smf_heat1d_zeta6_new,
    smf_heat1d_zeta6_size, smf_heat1d_zeta6_values, smf_heat1d_zeta8_evolve, smf_heat1d_zeta8_free,
    smf_heat1d_zeta8_new, smf_heat1d_zeta8_size, smf_heat1d_zeta8_values, SmfHeat1DZeta4,
    SmfHeat1DZeta6, SmfHeat1DZeta8,
};
pub use drift_reaction_zeta4_ffi::{
    smf_drift_reaction_zeta4_evolve, smf_drift_reaction_zeta4_free, smf_drift_reaction_zeta4_new,
    smf_drift_reaction_zeta4_size, smf_drift_reaction_zeta4_values, SmfDriftReactionZeta4,
};
pub use expmv_ffi::{
    smf_expmv1d_evolve, smf_expmv1d_free, smf_expmv1d_new, smf_expmv1d_size, smf_expmv1d_values,
    SmfExpmv1D,
};
pub use ffi::*;
pub use graph_ffi::*;
pub use graph_ffi_v2_4::*;
pub use greeks::{
    smf_greeks_evolver_free_v3, smf_greeks_evolver_new_heat_1d_unit_v3, smf_heat1d_greeks_v3,
    SmfGreeksEvolverV3,
};
pub use handle::SemiflowState;
pub use killing_soft_ffi::{
    smf_killing2nd_evolve, smf_killing2nd_free, smf_killing2nd_new, smf_killing2nd_size,
    smf_killing2nd_values, SmfKilling2nd,
};
pub use matrix_2d3d_ffi::{
    smf_matrix2d_evolve, smf_matrix2d_free, smf_matrix2d_new, smf_matrix2d_size,
    smf_matrix2d_values, smf_matrix3d_evolve, smf_matrix3d_free, smf_matrix3d_new,
    smf_matrix3d_size, smf_matrix3d_values, SmfMatrix2D, SmfMatrix3D,
};
pub use matrix_ffi::{
    smf_matrix_diffusion_evolve, smf_matrix_diffusion_free, smf_matrix_diffusion_new,
    smf_matrix_diffusion_size, smf_matrix_diffusion_values, SmfMatrixDiffusion1D,
};
pub use resolvent_jump_ffi::{
    smf_resolvent_jump_apply_v3, smf_resolvent_jump_free_v3,
    smf_resolvent_jump_new_heat_1d_unit_v3, SmfResolventJumpV3,
};
pub use resolvent_jump_nd_ffi::{
    smf_resolvent_jump_2d_apply_v3, smf_resolvent_jump_2d_free_v3,
    smf_resolvent_jump_2d_new_heat_unit_v3, smf_resolvent_jump_3d_apply_v3,
    smf_resolvent_jump_3d_free_v3, smf_resolvent_jump_3d_new_heat_unit_v3, SmfResolventJump2DV3,
    SmfResolventJump3DV3,
};
pub use schrodinger_complex_ffi::{
    smf_schrodinger_cx_evolve, smf_schrodinger_cx_free, smf_schrodinger_cx_new,
    smf_schrodinger_cx_size, smf_schrodinger_cx_values, SmfSchrodingerComplex1D,
};
pub use schrodinger_ffi::{
    smf_schrodinger_evolve, smf_schrodinger_free, smf_schrodinger_new, smf_schrodinger_size,
    smf_schrodinger_values, SmfSchrodinger1D,
};
pub use status::SemiflowStatus;
pub use strang_ffi::{
    smf_strang1d_evolve, smf_strang1d_free, smf_strang1d_new, smf_strang1d_size,
    smf_strang1d_values, SmfStrang1D,
};
pub use strang_nd_2d_ffi::{
    smf_heat2d_evolve, smf_heat2d_free, smf_heat2d_new, smf_heat2d_size, smf_heat2d_vara_evolve,
    smf_heat2d_vara_free, smf_heat2d_vara_new, smf_heat2d_vara_size, SmfHeat2D, SmfHeat2DVarA,
};
pub use strang_nd_3d_ffi::{
    smf_heat3d_evolve, smf_heat3d_free, smf_heat3d_new, smf_heat3d_size, smf_heat3d_vara_evolve,
    smf_heat3d_vara_free, smf_heat3d_vara_new, smf_heat3d_vara_size, SmfHeat3D, SmfHeat3DVarA,
};
pub use v3::{
    smf_evolver_evolve_into_v3, smf_evolver_free_v3, smf_evolver_new_heat_1d_unit_v3,
    smf_evolver_size_v3, smf_evolver_values_v3, smf_growth_v3, SmfEvolverV3, SmfGrowthV3,
};
pub use wentzell_ffi::{
    smf_wentzell_evolve_v3, smf_wentzell_evolver_free_v3, smf_wentzell_evolver_new_heat_1d_unit_v3,
    SmfWentzellEvolverV3,
};

// Round 6 — non-separable 2D, anisotropic-ND, Smolyak 6D
pub mod aniso_nd2_ffi;
pub mod aniso_nd3_ffi;
pub mod nonsep_ffi;
pub mod smolyak_ffi;
pub use aniso_nd2_ffi::{
    smf_aniso_nd2_evolve, smf_aniso_nd2_free, smf_aniso_nd2_new, smf_aniso_nd2_size,
    smf_aniso_nd2_values, SmfAnisoND2,
};
pub use aniso_nd3_ffi::{
    smf_aniso_nd3_evolve, smf_aniso_nd3_free, smf_aniso_nd3_new, smf_aniso_nd3_size,
    smf_aniso_nd3_values, SmfAnisoND3,
};
pub use nonsep_ffi::{
    smf_nonsep2d_aniso_evolve, smf_nonsep2d_aniso_free, smf_nonsep2d_aniso_new,
    smf_nonsep2d_aniso_size, smf_nonsep2d_aniso_values, smf_nonsep2d_evolve, smf_nonsep2d_free,
    smf_nonsep2d_new, smf_nonsep2d_size, smf_nonsep2d_values, SmfNonSep2D, SmfNonSep2DAniso,
};
pub use smolyak_ffi::{
    smf_smolyak_d6_apply, smf_smolyak_d6_free, smf_smolyak_d6_n_nodes, smf_smolyak_d6_new,
    smf_smolyak_d6_size, SmfSmolyakD6,
};

// Round 7 — Howland lift, subordinated heat, manifold Chernoff
pub mod howland_ffi;
pub mod manifold_ffi;
pub mod subordinated_ffi;
pub use howland_ffi::{
    smf_howland1d_evolve, smf_howland1d_free, smf_howland1d_new, smf_howland1d_size,
    smf_howland1d_values, SmfHowland1D,
};
pub use manifold_ffi::{
    smf_manifold2d_evolve, smf_manifold2d_free, smf_manifold2d_new, smf_manifold2d_size,
    smf_manifold2d_values, SmfManifold2D,
};
pub use subordinated_ffi::{
    smf_subordinated1d_evolve, smf_subordinated1d_free, smf_subordinated1d_new,
    smf_subordinated1d_size, smf_subordinated1d_values, SmfSubordinated1D,
};

// Round 8 — hypoelliptic / sub-Riemannian Chernoff engines
pub mod hypoelliptic_engel_ffi;
pub mod hypoelliptic_ffi;
pub use hypoelliptic_engel_ffi::{
    smf_hypo_engel_evolve, smf_hypo_engel_free, smf_hypo_engel_new, smf_hypo_engel_size,
    smf_hypo_engel_values, SmfHypoEngel,
};
pub use hypoelliptic_ffi::{
    smf_hypo_heisenberg_free, smf_hypo_heisenberg_kernel, smf_hypo_heisenberg_new,
    smf_hypo_heisenberg_order, smf_hypo_kolmogorov_evolve, smf_hypo_kolmogorov_free,
    smf_hypo_kolmogorov_new, smf_hypo_kolmogorov_size, smf_hypo_kolmogorov_values,
    SmfHypoHeisenberg, SmfHypoKolmogorov,
};

// Round 10 — graph-heat family parity (GraphHeat4th, MagnusGraphHeat6, VarCoefGraphHeat)
pub mod graph_heat_extra_ffi;
pub mod graph_vc_ghc_ffi;
pub use graph_heat_extra_ffi::{
    smf_ghc4_apply_into, smf_ghc4_current, smf_ghc4_drop, smf_ghc4_new, smf_mghc6_apply_into,
    smf_mghc6_current, smf_mghc6_drop, smf_mghc6_new, SmfGhc4, SmfMghc6,
};
pub use graph_vc_ghc_ffi::{
    smf_vc_ghc_apply_into, smf_vc_ghc_current, smf_vc_ghc_drop, smf_vc_ghc_new, SmfVcGhc,
};

// Round 11 — niche engines (QuantumGraphHeat, StrangGraph, ComplexTripleJump, PointEval)
pub mod carnot_ffi;
pub mod point_eval_ffi;
pub mod quantum_graph_ffi;
pub mod strang_graph_ffi;
pub use carnot_ffi::{
    smf_carnot_ctj_apply_real, smf_carnot_ctj_drop, smf_carnot_ctj_new, smf_carnot_ctj_size,
    smf_carnot_ctj_verify_gamma_star, SmfCarnotCtj,
};
pub use point_eval_ffi::{
    smf_point_eval_drop, smf_point_eval_eval_at, smf_point_eval_new, smf_point_eval_size,
    SmfPointEval,
};
pub use quantum_graph_ffi::{
    smf_qgheat_drop, smf_qgheat_evolve, smf_qgheat_new, smf_qgheat_set_state, smf_qgheat_size,
    smf_qgheat_values, smf_qgraph_drop, smf_qgraph_from_edges, smf_qgraph_n_edges,
    smf_qgraph_n_per_edge, smf_qgraph_path, smf_qgraph_star, smf_qgraph_total_len, SmfQuantumGraph,
    SmfQuantumGraphHeat,
};
pub use strang_graph_ffi::{
    smf_strang_graph_cycle_new, smf_strang_graph_drop, smf_strang_graph_evolve,
    smf_strang_graph_n_nodes, smf_strang_graph_order, smf_strang_graph_path_new, SmfStrangGraph,
};

// S³ flagship carriers (ADR-0171) — TtEvolver/TtState, TtCoupledEvolver, GridlessEvolver/MeasureState
// VarCoefTt (issue #2, ADR-0178): additive-separable variable-coefficient TT evolver
pub mod tt_varcoef_ffi;
pub use gridless_ffi::{
    smf_gridless_apply, smf_gridless_evolve, smf_gridless_free, smf_gridless_new,
    smf_measurestate_free, smf_measurestate_marginal, smf_measurestate_n_diracs,
    smf_measurestate_new, smf_measurestate_second_moment, smf_measurestate_total_variation,
    SmfGridlessEvolver, SmfMeasureState,
};
pub use tt_coupled_ffi::{
    smf_tt_coupled_evolve, smf_tt_coupled_free, smf_tt_coupled_new, SmfTtCoupledEvolver,
};
pub use tt_ffi::{
    smf_tt_evolver_evolve, smf_tt_evolver_free, smf_tt_evolver_new, smf_ttstate_free,
    smf_ttstate_inner_separable, smf_ttstate_n_j, smf_ttstate_ndim, smf_ttstate_new_separable,
    smf_ttstate_peak_rank, smf_ttstate_storage_size, SmfTtEvolver, SmfTtState,
};
pub use tt_varcoef_ffi::{
    smf_varcoef_tt_evolver_evolve, smf_varcoef_tt_evolver_free, smf_varcoef_tt_evolver_ndim,
    smf_varcoef_tt_evolver_new, SmfVarCoefTtEvolver,
};

// Round 9 — obstacle, adjoint, and adaptive-stepping engines
pub mod adaptive_ffi;
pub mod adjoint_ffi;
pub mod obstacle_ffi;
pub use adaptive_ffi::{
    smf_adaptive_pi_evolve, smf_adaptive_pi_free, smf_adaptive_pi_new, smf_adaptive_pi_size,
    smf_adaptive_pi_values, SmfAdaptivePI,
};
pub use adjoint_ffi::{
    smf_adjoint1d_evolve, smf_adjoint1d_free, smf_adjoint1d_new, smf_adjoint1d_order,
    smf_adjoint1d_size, smf_adjoint1d_values, SmfAdjoint1D,
};
pub use obstacle_ffi::{
    smf_obstacle1d_evolve, smf_obstacle1d_free, smf_obstacle1d_new, smf_obstacle1d_size,
    smf_obstacle1d_values, SmfObstacle1D,
};

// C-parity pass — Laplacian introspection + GraphTraj (degenerate)
pub mod laplacian_ffi;
pub use laplacian_ffi::{
    smf_free_buf_f64, smf_free_buf_usize, smf_graph_laplacian_combinatorial,
    smf_graph_laplacian_normalized, smf_graph_traj_free, smf_graph_traj_n_nodes,
    smf_graph_traj_n_segments, smf_graph_traj_new, smf_graph_traj_t_horizon, smf_laplacian_col_idx,
    smf_laplacian_free, smf_laplacian_is_combinatorial, smf_laplacian_is_normalized,
    smf_laplacian_n_nodes, smf_laplacian_row_ptr, smf_laplacian_spectral_bound,
    smf_laplacian_to_dense, smf_laplacian_vals, SmfGraphTraj, SmfLaplacian,
};

// C-parity pass — ObstacleGammaV8 + ObstacleNDV8 (ADR-0153 TIER-2)
pub mod obstacle_gamma_ffi;
pub use obstacle_gamma_ffi::{
    smf_free_buf_u8, smf_obstacle_gamma_free, smf_obstacle_gamma_inactive_gamma,
    smf_obstacle_gamma_new_array, smf_obstacle_gamma_new_const, smf_obstacle_gamma_size,
    SmfObstacleGamma,
};

pub mod obstacle_nd_ffi;
pub use obstacle_nd_ffi::{
    smf_obstacle_nd2_apply, smf_obstacle_nd2_free, smf_obstacle_nd2_new, smf_obstacle_nd2_shape,
    SmfObstacleND2,
};

// ADR-0180 — pre-sampled graph state-adjoint (batched time-grid, GL₄-aware)
pub mod graph_adjoint_ffi;
pub use graph_adjoint_ffi::{
    smf_graph_adjoint_abscissa_times, smf_graph_adjoint_evolve_state_adjoint,
    smf_graph_adjoint_free, smf_graph_adjoint_n_nodes, smf_graph_adjoint_new_presampled,
    smf_graph_adjoint_new_presampled_varcoef, SmfGraphAdjoint,
};
