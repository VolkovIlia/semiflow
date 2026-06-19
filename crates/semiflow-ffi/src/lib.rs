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
pub mod ffi;
pub mod graph_ffi;
pub mod graph_ffi_v2_4;
pub mod greeks;
pub(crate) mod handle;
pub mod resolvent_jump_ffi;
pub mod resolvent_jump_nd_ffi;
pub mod status;
pub mod v3;
pub mod wentzell_ffi;

pub use adjoint_fp_ffi::{
    smf_adjoint_fp_free_v3, smf_adjoint_fp_new_brownian_1d_v3, smf_adjoint_fp_step_v3,
    SmfAdjointFpV3,
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
