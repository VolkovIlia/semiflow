//! WebAssembly bindings for `semiflow-core` (experimental, v0.9.0-beta).
//!
//! Exposes a broad set of `semiflow-core` engine families as JS classes via
//! `wasm-bindgen`, mirroring `semiflow-ffi` and `semiflow-py`.
//!
//! ## Cargo features — `full` vs default (lite)
//!
//! The default build ("lite") exposes only the lightweight baseline engines
//! (1D heat, graph path/heat, resolvent-jump, reverse-AD, Greeks, Wentzell,
//! adjoint Fokker–Planck) and keeps the raw Wasm binary to **≈ 768 KB**.
//!
//! Building with `--features full` adds all heavy-grid, multi-dimensional,
//! boundary-condition, and hypoelliptic engines, bringing the raw binary to
//! **≈ 1.4 MB**.  See `[features]` in `Cargo.toml`.
//!
//! ## Engine surface — default (lite) build
//!
//! - `Heat1D`, `GraphPath`, `GraphHeat`, `GraphHeat6`.
//! - `ResolventJumpV8`, `ResolventJump2DV8`, `ResolventJump3DV8`.
//! - `ReverseHeat1D`, `EvolverHeat1DGreeksV3`, `GrowthV3`,
//!   `EvolverHeat1DUnitV3`.
//! - `WentzellV8`, `GammaFamily`.
//! - `AdjointFokkerPlanckV8`.
//!
//! ## Engine surface — `--features full` additions
//!
//! - **Higher-order 1D** — `Heat1D4th/6th`, `Heat1DZeta4/6/8`,
//!   `TruncatedExp1D`, `TruncatedExp4th1D`, `DriftReaction1D`, `Shift1D`,
//!   `Strang1D`.
//! - **Matrix / Schrödinger** — `MatrixDiffusion1D`, `Schrodinger1D`,
//!   `SchrodingerComplex1D`.
//! - **Boundary conditions** — `Killing1D`, `Reflected1D`, `Robin1D`,
//!   `Resolvent1D`, `KilledDirichlet1D`.
//! - **2D/3D tensor** — `Heat2D/3D`, `Heat2DVarA/3DVarA`.
//! - **Non-separable / anisotropic** — `NonSeparable2D`, `NonSeparable2DAniso`,
//!   `AnisotropicShiftND2/3`.
//! - **High-dimensional** — `SmolyakD6`.
//! - **Nonautonomous** — `Howland1D`, `Subordinated1D`.
//! - **Manifold** — `Manifold2D` (Torus, Sphere2, Hyperbolic2).
//! - **Hypoelliptic** — Heisenberg, Kolmogorov, Engel.
//! - **Graph extensions** — `GraphHeat4th`, `VarCoefGraphHeat`,
//!   `MagnusGraphHeat`, `MagnusGraphHeat6`, `VarCoefMagnusGraph`,
//!   `QuantumGraph`, `QuantumGraphHeat`, `StrangGraph`.
//! - **Other** — `Obstacle1D`, `Adjoint1D`, `AdaptivePI1D`,
//!   `ComplexTripleJump`, `PointEval`.
//!
//! **Documented deferrals:** `ObstacleND`, `ObstacleGamma`, `GraphTraj`,
//! Laplacian introspection, and `GraphAdjoint` dense read-back (same reasons
//! as `semiflow-ffi` — closures and dense matrices are not ABI-safe).
//! S³ carriers (`TtEvolver`, `GridlessEvolver`) are C-ABI-accessible
//! (ADR-0171) but not yet wired to WASM; deferred to a follow-up release.
//!
//! Distribution via npm is managed by `release-wasm.yml`.
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
mod tt_wasm;
mod tt_coupled_wasm;
mod gridless_wasm;

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
#[cfg(feature = "full")]
mod obstacle_wasm;
#[cfg(feature = "full")]
mod adjoint_wasm;
#[cfg(feature = "full")]
mod adaptive_wasm;
#[cfg(feature = "full")]
mod graph_heat_extra_wasm;
#[cfg(feature = "full")]
mod graph_magnus_wasm;
#[cfg(feature = "full")]
mod quantum_graph_wasm;
#[cfg(feature = "full")]
mod strang_graph_wasm;
#[cfg(feature = "full")]
mod carnot_wasm;
#[cfg(feature = "full")]
mod point_eval_wasm;

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
pub use tt_wasm::{TtState, TtEvolver};
pub use tt_coupled_wasm::TtCoupledEvolver;
pub use gridless_wasm::{MeasureState, GridlessEvolver};

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
#[cfg(feature = "full")]
pub use obstacle_wasm::ObstacleChernoffWasm; // Rust type name; JS sees `Obstacle1D` via js_name
#[cfg(feature = "full")]
pub use adjoint_wasm::Adjoint1D;
#[cfg(feature = "full")]
pub use adaptive_wasm::AdaptivePI1D;
#[cfg(feature = "full")]
pub use graph_heat_extra_wasm::{GraphHeat4thWasm, VarCoefGraphHeatWasm};
#[cfg(feature = "full")]
pub use graph_magnus_wasm::{
    MagnusGraphHeatWasm, MagnusGraphHeat6Wasm, VarCoefMagnusGraphWasm,
};
#[cfg(feature = "full")]
pub use quantum_graph_wasm::{QuantumGraphWasm, QuantumGraphHeatWasm};
#[cfg(feature = "full")]
pub use strang_graph_wasm::StrangGraphWasm;
#[cfg(feature = "full")]
pub use carnot_wasm::ComplexTripleJumpWasm;
#[cfg(feature = "full")]
pub use point_eval_wasm::PointEvalWasm;

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
