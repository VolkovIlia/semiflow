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
