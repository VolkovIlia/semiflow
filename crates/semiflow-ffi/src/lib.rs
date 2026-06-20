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
//! ## Scope (v0.10.0 Wave A)
//!
//! Only 1-D heat with `a(x) = 1.0` (constant diffusion) is exposed.
//! Variable-coefficient `a(x)` requires a runtime callback; deferred to
//! v0.11.0. See [`crate::ffi::smf_state_new_heat_1d_unit`].

#[macro_use]
mod panic;

pub mod bc_ffi;
pub mod bc_ffi2;
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
