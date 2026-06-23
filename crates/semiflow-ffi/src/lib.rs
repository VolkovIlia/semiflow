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

pub mod bc_ffi;
pub mod bc_ffi2;
pub mod bc_ffi3;
pub mod adjoint_fp_ffi;
pub mod cdr_ffi;
pub mod matrix_ffi;
pub mod schrodinger_complex_ffi;
pub mod schrodinger_ffi;
pub mod diffusion_extra_ffi;
pub mod diffusion_hi_ffi;
pub mod diffusion_hi_zeta_ffi;
pub mod ffi;
pub mod graph_ffi;
pub mod graph_ffi_v2_4;
pub mod greeks;
pub(crate) mod handle;
pub mod resolvent_jump_ffi;
pub mod resolvent_jump_nd_ffi;
pub mod status;
pub mod strang_ffi;
pub mod strang_nd_2d_ffi;
pub mod strang_nd_3d_ffi;
pub mod v3;
pub mod wentzell_ffi;
pub mod tt_ffi;
pub mod tt_coupled_ffi;
pub mod gridless_ffi;
pub mod expmv_ffi;
pub mod drift_reaction_zeta4_ffi;
pub mod killing_soft_ffi;
pub mod matrix_2d3d_ffi;

pub use expmv_ffi::{
    smf_expmv1d_new, smf_expmv1d_evolve, smf_expmv1d_values,
    smf_expmv1d_size, smf_expmv1d_free, SmfExpmv1D,
};
pub use drift_reaction_zeta4_ffi::{
    smf_drift_reaction_zeta4_new, smf_drift_reaction_zeta4_evolve,
    smf_drift_reaction_zeta4_values, smf_drift_reaction_zeta4_size,
    smf_drift_reaction_zeta4_free, SmfDriftReactionZeta4,
};
pub use killing_soft_ffi::{
    smf_killing2nd_new, smf_killing2nd_evolve, smf_killing2nd_values,
    smf_killing2nd_size, smf_killing2nd_free, SmfKilling2nd,
};
pub use matrix_2d3d_ffi::{
    smf_matrix2d_new, smf_matrix2d_evolve, smf_matrix2d_values,
    smf_matrix2d_size, smf_matrix2d_free, SmfMatrix2D,
    smf_matrix3d_new, smf_matrix3d_evolve, smf_matrix3d_values,
    smf_matrix3d_size, smf_matrix3d_free, SmfMatrix3D,
};

pub use adjoint_fp_ffi::{
    smf_adjoint_fp_free_v3, smf_adjoint_fp_new_brownian_1d_v3, smf_adjoint_fp_step_v3,
    SmfAdjointFpV3,
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
pub use diffusion_hi_ffi::{
    smf_heat1d_4th_evolve, smf_heat1d_4th_free, smf_heat1d_4th_new, smf_heat1d_4th_size,
    smf_heat1d_4th_values, smf_heat1d_6th_evolve, smf_heat1d_6th_free, smf_heat1d_6th_new,
    smf_heat1d_6th_size, smf_heat1d_6th_values, SmfHeat1D4th, SmfHeat1D6th,
};
pub use diffusion_hi_zeta_ffi::{
    smf_heat1d_zeta4_evolve, smf_heat1d_zeta4_free, smf_heat1d_zeta4_new,
    smf_heat1d_zeta4_size, smf_heat1d_zeta4_values, smf_heat1d_zeta6_evolve,
    smf_heat1d_zeta6_free, smf_heat1d_zeta6_new, smf_heat1d_zeta6_size,
    smf_heat1d_zeta6_values, smf_heat1d_zeta8_evolve, smf_heat1d_zeta8_free,
    smf_heat1d_zeta8_new, smf_heat1d_zeta8_size, smf_heat1d_zeta8_values, SmfHeat1DZeta4,
    SmfHeat1DZeta6, SmfHeat1DZeta8,
};
pub use ffi::*;
pub use graph_ffi::*;
pub use graph_ffi_v2_4::*;
pub use greeks::{
    smf_greeks_evolver_free_v3, smf_greeks_evolver_new_heat_1d_unit_v3, smf_heat1d_greeks_v3,
    SmfGreeksEvolverV3,
};
pub use handle::SemiflowState;
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
pub use status::SemiflowStatus;
pub use v3::{
    smf_evolver_evolve_into_v3, smf_evolver_free_v3, smf_evolver_new_heat_1d_unit_v3,
    smf_evolver_size_v3, smf_evolver_values_v3, smf_growth_v3, SmfEvolverV3, SmfGrowthV3,
};
pub use wentzell_ffi::{
    smf_wentzell_evolve_v3, smf_wentzell_evolver_free_v3, smf_wentzell_evolver_new_heat_1d_unit_v3,
    SmfWentzellEvolverV3,
};
pub use matrix_ffi::{
    smf_matrix_diffusion_evolve, smf_matrix_diffusion_free, smf_matrix_diffusion_new,
    smf_matrix_diffusion_size, smf_matrix_diffusion_values, SmfMatrixDiffusion1D,
};
pub use schrodinger_ffi::{
    smf_schrodinger_evolve, smf_schrodinger_free, smf_schrodinger_new, smf_schrodinger_size,
    smf_schrodinger_values, SmfSchrodinger1D,
};
pub use schrodinger_complex_ffi::{
    smf_schrodinger_cx_evolve, smf_schrodinger_cx_free, smf_schrodinger_cx_new,
    smf_schrodinger_cx_size, smf_schrodinger_cx_values, SmfSchrodingerComplex1D,
};
pub use bc_ffi::{
    smf_killing1d_evolve, smf_killing1d_free, smf_killing1d_new, smf_killing1d_size,
    smf_killing1d_values, SmfKilling1D,
    smf_reflected1d_evolve, smf_reflected1d_free, smf_reflected1d_new, smf_reflected1d_size,
    smf_reflected1d_values, SmfReflected1D,
};
pub use bc_ffi3::{
    smf_dirichlet_heat2nd1d_evolve, smf_dirichlet_heat2nd1d_free, smf_dirichlet_heat2nd1d_new,
    smf_dirichlet_heat2nd1d_size, smf_dirichlet_heat2nd1d_values, SmfDirichletHeat2nd1D,
};
pub use bc_ffi2::{
    smf_robin1d_evolve, smf_robin1d_free, smf_robin1d_new, smf_robin1d_size,
    smf_robin1d_values, SmfRobin1D,
    smf_resolvent1d_eval, smf_resolvent1d_free, smf_resolvent1d_new, smf_resolvent1d_size,
    SmfResolvent1D,
    smf_killed_dir1d_apply, smf_killed_dir1d_free, smf_killed_dir1d_new,
    smf_killed_dir1d_size, smf_killed_dir1d_values, SmfKilledDir1D,
};

// Round 6 — non-separable 2D, anisotropic-ND, Smolyak 6D
pub mod nonsep_ffi;
pub mod aniso_nd2_ffi;
pub mod aniso_nd3_ffi;
pub mod smolyak_ffi;
pub use nonsep_ffi::{
    smf_nonsep2d_new, smf_nonsep2d_evolve, smf_nonsep2d_size,
    smf_nonsep2d_values, smf_nonsep2d_free, SmfNonSep2D,
    smf_nonsep2d_aniso_new, smf_nonsep2d_aniso_evolve, smf_nonsep2d_aniso_size,
    smf_nonsep2d_aniso_values, smf_nonsep2d_aniso_free, SmfNonSep2DAniso,
};
pub use aniso_nd2_ffi::{
    smf_aniso_nd2_new, smf_aniso_nd2_evolve, smf_aniso_nd2_size,
    smf_aniso_nd2_values, smf_aniso_nd2_free, SmfAnisoND2,
};
pub use aniso_nd3_ffi::{
    smf_aniso_nd3_new, smf_aniso_nd3_evolve, smf_aniso_nd3_size,
    smf_aniso_nd3_values, smf_aniso_nd3_free, SmfAnisoND3,
};
pub use smolyak_ffi::{
    smf_smolyak_d6_new, smf_smolyak_d6_apply, smf_smolyak_d6_size,
    smf_smolyak_d6_n_nodes, smf_smolyak_d6_free, SmfSmolyakD6,
};

// Round 7 — Howland lift, subordinated heat, manifold Chernoff
pub mod howland_ffi;
pub mod subordinated_ffi;
pub mod manifold_ffi;
pub use howland_ffi::{
    smf_howland1d_new, smf_howland1d_evolve, smf_howland1d_values,
    smf_howland1d_size, smf_howland1d_free, SmfHowland1D,
};
pub use subordinated_ffi::{
    smf_subordinated1d_new, smf_subordinated1d_evolve, smf_subordinated1d_values,
    smf_subordinated1d_size, smf_subordinated1d_free, SmfSubordinated1D,
};
pub use manifold_ffi::{
    smf_manifold2d_new, smf_manifold2d_evolve, smf_manifold2d_values,
    smf_manifold2d_size, smf_manifold2d_free, SmfManifold2D,
};

// Round 8 — hypoelliptic / sub-Riemannian Chernoff engines
pub mod hypoelliptic_ffi;
pub mod hypoelliptic_engel_ffi;
pub use hypoelliptic_ffi::{
    smf_hypo_heisenberg_new, smf_hypo_heisenberg_order, smf_hypo_heisenberg_kernel,
    smf_hypo_heisenberg_free, SmfHypoHeisenberg,
    smf_hypo_kolmogorov_new, smf_hypo_kolmogorov_evolve, smf_hypo_kolmogorov_values,
    smf_hypo_kolmogorov_size, smf_hypo_kolmogorov_free, SmfHypoKolmogorov,
};
pub use hypoelliptic_engel_ffi::{
    smf_hypo_engel_new, smf_hypo_engel_evolve, smf_hypo_engel_values,
    smf_hypo_engel_size, smf_hypo_engel_free, SmfHypoEngel,
};

// Round 10 — graph-heat family parity (GraphHeat4th, MagnusGraphHeat6, VarCoefGraphHeat)
pub mod graph_heat_extra_ffi;
pub mod graph_vc_ghc_ffi;
pub use graph_heat_extra_ffi::{
    smf_ghc4_new, smf_ghc4_apply_into, smf_ghc4_current, smf_ghc4_drop, SmfGhc4,
    smf_mghc6_new, smf_mghc6_apply_into, smf_mghc6_current, smf_mghc6_drop, SmfMghc6,
};
pub use graph_vc_ghc_ffi::{
    smf_vc_ghc_new, smf_vc_ghc_apply_into, smf_vc_ghc_current, smf_vc_ghc_drop, SmfVcGhc,
};

// Round 11 — niche engines (QuantumGraphHeat, StrangGraph, ComplexTripleJump, PointEval)
pub mod quantum_graph_ffi;
pub mod strang_graph_ffi;
pub mod carnot_ffi;
pub mod point_eval_ffi;
pub use quantum_graph_ffi::{
    smf_qgraph_path, smf_qgraph_star, smf_qgraph_from_edges,
    smf_qgraph_n_edges, smf_qgraph_n_per_edge, smf_qgraph_total_len, smf_qgraph_drop,
    SmfQuantumGraph,
    smf_qgheat_new, smf_qgheat_set_state, smf_qgheat_evolve,
    smf_qgheat_values, smf_qgheat_size, smf_qgheat_drop, SmfQuantumGraphHeat,
};
pub use strang_graph_ffi::{
    smf_strang_graph_path_new, smf_strang_graph_cycle_new,
    smf_strang_graph_evolve, smf_strang_graph_n_nodes, smf_strang_graph_order,
    smf_strang_graph_drop, SmfStrangGraph,
};
pub use carnot_ffi::{
    smf_carnot_ctj_new, smf_carnot_ctj_apply_real, smf_carnot_ctj_size,
    smf_carnot_ctj_verify_gamma_star, smf_carnot_ctj_drop, SmfCarnotCtj,
};
pub use point_eval_ffi::{
    smf_point_eval_new, smf_point_eval_eval_at, smf_point_eval_size,
    smf_point_eval_drop, SmfPointEval,
};

// S³ flagship carriers (ADR-0171) — TtEvolver/TtState, TtCoupledEvolver, GridlessEvolver/MeasureState
// VarCoefTt (issue #2, ADR-0178): additive-separable variable-coefficient TT evolver
pub mod tt_varcoef_ffi;
pub use tt_varcoef_ffi::{
    smf_varcoef_tt_evolver_new, smf_varcoef_tt_evolver_evolve,
    smf_varcoef_tt_evolver_ndim, smf_varcoef_tt_evolver_free,
    SmfVarCoefTtEvolver,
};
pub use tt_ffi::{
    smf_ttstate_new_separable, smf_ttstate_free, smf_ttstate_ndim, smf_ttstate_n_j,
    smf_ttstate_peak_rank, smf_ttstate_storage_size, smf_ttstate_inner_separable,
    smf_tt_evolver_new, smf_tt_evolver_evolve, smf_tt_evolver_free,
    SmfTtState, SmfTtEvolver,
};
pub use tt_coupled_ffi::{
    smf_tt_coupled_new, smf_tt_coupled_evolve, smf_tt_coupled_free,
    SmfTtCoupledEvolver,
};
pub use gridless_ffi::{
    smf_measurestate_new, smf_measurestate_free, smf_measurestate_n_diracs,
    smf_measurestate_total_variation, smf_measurestate_second_moment,
    smf_measurestate_marginal,
    smf_gridless_new, smf_gridless_apply, smf_gridless_evolve, smf_gridless_free,
    SmfMeasureState, SmfGridlessEvolver,
};

// Round 9 — obstacle, adjoint, and adaptive-stepping engines
pub mod obstacle_ffi;
pub mod adjoint_ffi;
pub mod adaptive_ffi;
pub use obstacle_ffi::{
    smf_obstacle1d_new, smf_obstacle1d_evolve, smf_obstacle1d_values,
    smf_obstacle1d_size, smf_obstacle1d_free, SmfObstacle1D,
};
pub use adjoint_ffi::{
    smf_adjoint1d_new, smf_adjoint1d_evolve, smf_adjoint1d_values,
    smf_adjoint1d_size, smf_adjoint1d_order, smf_adjoint1d_free, SmfAdjoint1D,
};
pub use adaptive_ffi::{
    smf_adaptive_pi_new, smf_adaptive_pi_evolve, smf_adaptive_pi_values,
    smf_adaptive_pi_size, smf_adaptive_pi_free, SmfAdaptivePI,
};

// C-parity pass — Laplacian introspection + GraphTraj (degenerate)
pub mod laplacian_ffi;
pub use laplacian_ffi::{
    smf_graph_laplacian_combinatorial, smf_graph_laplacian_normalized,
    smf_laplacian_free, smf_laplacian_n_nodes,
    smf_laplacian_is_combinatorial, smf_laplacian_is_normalized,
    smf_laplacian_spectral_bound,
    smf_laplacian_row_ptr, smf_laplacian_col_idx, smf_laplacian_vals,
    smf_laplacian_to_dense,
    smf_free_buf_usize, smf_free_buf_f64,
    SmfLaplacian,
    smf_graph_traj_new, smf_graph_traj_free,
    smf_graph_traj_n_nodes, smf_graph_traj_n_segments, smf_graph_traj_t_horizon,
    SmfGraphTraj,
};

// C-parity pass — ObstacleGammaV8 + ObstacleNDV8 (ADR-0153 TIER-2)
pub mod obstacle_gamma_ffi;
pub use obstacle_gamma_ffi::{
    smf_obstacle_gamma_new_const, smf_obstacle_gamma_new_array,
    smf_obstacle_gamma_free, smf_obstacle_gamma_size,
    smf_obstacle_gamma_inactive_gamma,
    smf_free_buf_u8,
    SmfObstacleGamma,
};

pub mod obstacle_nd_ffi;
pub use obstacle_nd_ffi::{
    smf_obstacle_nd2_new, smf_obstacle_nd2_free,
    smf_obstacle_nd2_shape, smf_obstacle_nd2_apply,
    SmfObstacleND2,
};

// ADR-0180 — pre-sampled graph state-adjoint (batched time-grid, GL₄-aware)
pub mod graph_adjoint_ffi;
pub use graph_adjoint_ffi::{
    smf_graph_adjoint_abscissa_times,
    smf_graph_adjoint_new_presampled,
    smf_graph_adjoint_new_presampled_varcoef,
    smf_graph_adjoint_evolve_state_adjoint,
    smf_graph_adjoint_n_nodes,
    smf_graph_adjoint_free,
    SmfGraphAdjoint,
};
