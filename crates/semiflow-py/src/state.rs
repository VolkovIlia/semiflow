//! Heat-equation Python classes — split into per-dimension submodules.
//!
//! Public surface (re-exported here so `crate::state::Heat1D` etc. still
//! resolve from `lib.rs` without any change to `pymodule` registration):
//!   - [`Heat1D`] — 1-D heat equation (unit and variable `a`)
//!   - [`Heat2D`] — 2-D heat equation (Strang splitting)
//!   - [`Heat3D`] — 3-D heat equation (Strang splitting)
//!
//! Implementation lives in the private submodules below.

#![allow(unsafe_code)]

#[path = "state_1d.rs"]
mod state_1d;
#[path = "state_2d.rs"]
mod state_2d;
#[path = "state_3d.rs"]
mod state_3d;

pub(crate) use state_1d::extract_f64_slice;
pub use state_1d::Heat1D;
pub use state_2d::Heat2D;
pub use state_3d::Heat3D;
