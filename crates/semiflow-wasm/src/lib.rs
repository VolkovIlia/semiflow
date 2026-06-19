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
