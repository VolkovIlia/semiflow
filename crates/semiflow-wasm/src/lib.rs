//! WebAssembly bindings for `semiflow-core` (experimental, v0.10.0 Wave C).
//!
//! Exposes a single [`Heat1D`] JS class for 1-D heat with constant diffusion
//! `a = 1`, mirroring `semiflow-ffi` and `semiflow-py`.  Distribution via npm
//! is deferred to v0.11.0; v0.10.0 ships CI-built artifacts only.
//!
//! # Error model
//!
//! All fallible methods throw a JS `Error` whose `.kind` property is a string
//! matching the `SemiflowStatus` C-ABI names used by `semiflow-ffi` and
//! `semiflow-py`.  The mapping is:
//!
//! | `.kind` string      | Meaning                                              |
//! |---------------------|------------------------------------------------------|
//! | `"GridMismatch"`    | Grid geometry error (`n < 4`, `xmin >= xmax`, wrong buffer length) |
//! | `"NanInf"`          | NaN or Inf in input data                             |
//! | `"OutOfDomain"`     | Domain precondition violated (`t < 0`, `n_steps == 0`, etc.) |
//! | `"BoundaryFailure"` | Grid resolution too coarse for the Chernoff shift    |
//! | `"CflViolated"`     | CFL bound violated for the truncated-exp K=4 series  |
//! | `"ConvergenceFailed"` | Iterative solver did not converge                  |
//! | `"Unsupported"`     | Feature not available in this build                  |
//! | `"Panic"`           | Internal Rust panic (file a bug)                     |
//!
//! Example JS dispatch:
//! ```js
//! try {
//!   state.evolve(t, n_steps);
//! } catch (e) {
//!   if (e.kind === "OutOfDomain") { /* handle */ }
//!   else { throw e; }
//! }
//! ```
//!
//! # Panic boundary
//!
//! `wasm-bindgen` routes Rust panics through `__wbindgen_throw` to JS, so
//! this crate uses workspace `[profile.release]` (`panic = "abort"`) and does
//! NOT wrap calls in `std::panic::catch_unwind`.  Better diagnostics in
//! development are provided by [`panic_hook_init`] (installs
//! `console_error_panic_hook`).
//!
//! This intentionally diverges from `semiflow-ffi`'s `[profile.release-ffi]`
//! (`panic = "unwind"`); see ADR-0028 Amendment 1.

#![allow(unsafe_code)]

use wasm_bindgen::prelude::*;

mod adjoint_fp_wasm;
mod error;
mod graph_wasm;
mod graph_wasm_hi;
mod greeks_wasm;
mod handle;
mod resolvent_jump_nd_wasm;
mod resolvent_jump_wasm;
mod reverse_ad_wasm;
mod state;
mod v3;
mod wentzell_wasm;

#[cfg(feature = "full")]
mod diffusion_hi_wasm;
#[cfg(feature = "full")]
mod diffusion_extra_wasm;
#[cfg(feature = "full")]
mod matrix_diffusion_wasm;
#[cfg(feature = "full")]
mod schrodinger_complex_wasm;
#[cfg(feature = "full")]
mod schrodinger_wasm;
#[cfg(feature = "full")]
mod bc_wasm;
#[cfg(feature = "full")]
mod strang_nd_wasm;
#[cfg(feature = "full")]
mod nonsep_wasm;
#[cfg(feature = "full")]
mod aniso_nd_wasm;
#[cfg(feature = "full")]
mod smolyak_wasm;
#[cfg(feature = "full")]
mod howland_wasm;
#[cfg(feature = "full")]
mod subordinated_wasm;
#[cfg(feature = "full")]
mod manifold_wasm;
#[cfg(feature = "full")]
mod hypoelliptic_wasm;

pub use adjoint_fp_wasm::AdjointFokkerPlanckV8;
pub use graph_wasm::{GraphHeat, GraphPath};
pub use graph_wasm_hi::GraphHeat6;
pub use greeks_wasm::EvolverHeat1DGreeksV3;
pub use resolvent_jump_nd_wasm::{ResolventJump2DV8, ResolventJump3DV8};
pub use resolvent_jump_wasm::ResolventJumpV8;
pub use reverse_ad_wasm::ReverseHeat1D;
pub use state::Heat1D;
pub use v3::{EvolverHeat1DUnitV3, GrowthV3};
pub use wentzell_wasm::{GammaFamily, WentzellV8};

#[cfg(feature = "full")]
pub use diffusion_hi_wasm::{Heat1D4th, Heat1D6th, Heat1DZeta4, Heat1DZeta6, Heat1DZeta8};
#[cfg(feature = "full")]
pub use diffusion_extra_wasm::{
    DriftReaction1D, Shift1D, Strang1D, TruncatedExp1D, TruncatedExp4th1D,
};
#[cfg(feature = "full")]
pub use matrix_diffusion_wasm::MatrixDiffusion1D;
#[cfg(feature = "full")]
pub use schrodinger_complex_wasm::SchrodingerComplex1D;
#[cfg(feature = "full")]
pub use schrodinger_wasm::Schrodinger1D;
#[cfg(feature = "full")]
pub use bc_wasm::{KilledDirichlet1D, Killing1D, Reflected1D, Resolvent1D, Robin1D};
#[cfg(feature = "full")]
pub use strang_nd_wasm::{Heat2D, Heat2DVarA, Heat3D, Heat3DVarA};
#[cfg(feature = "full")]
pub use nonsep_wasm::{NonSeparable2D, NonSeparable2DAniso};
#[cfg(feature = "full")]
pub use aniso_nd_wasm::{AnisotropicShiftND2, AnisotropicShiftND3};
#[cfg(feature = "full")]
pub use smolyak_wasm::SmolyakD6;
#[cfg(feature = "full")]
pub use howland_wasm::Howland1D;
#[cfg(feature = "full")]
pub use subordinated_wasm::Subordinated1D;
#[cfg(feature = "full")]
pub use manifold_wasm::Manifold2D;
#[cfg(feature = "full")]
pub use hypoelliptic_wasm::{
    HypoellipticChernoffEngel, HypoellipticChernoffHeisenberg, HypoellipticChernoffKolmogorov,
};

/// Return the `semiflow-wasm` crate version string (e.g. `"0.10.0"`).
///
/// Matches the Cargo package version baked in at compile time.
#[must_use]
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// Install `console_error_panic_hook` for readable Rust panic messages in JS.
///
/// Call once at application startup (e.g. in your WASM initialisation block).
/// Subsequent calls are no-ops — the hook is installed at most once via
/// `std::sync::Once`.
///
/// In production builds (`panic = "abort"`, ADR-0028 Amendment 1) Rust panics
/// route directly through `__wbindgen_throw`; this hook adds a formatted
/// backtrace to the thrown string, which is useful during development but
/// harmless in production.
#[wasm_bindgen]
pub fn panic_hook_init() {
    console_error_panic_hook::set_once();
}
