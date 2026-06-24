//! Nonautonomous heat via Howland lift — WebAssembly binding (`full` feature).
//!
//! | JS class    | Core type                                | Python mirror |
//! |-------------|------------------------------------------|---------------|
//! | `Howland1D` | `HowlandLift<DiffusionChernoff<f64>>` | `Howland1D`   |
//!
//! ## Design
//!
//! Mirrors `semiflow-py` `Howland1D` (M11, `time_dependent.rs`).
//! `n_steps = n_t − 1` so `tau = t_horizon / (n_t − 1) = delta_s` exactly
//! (matched-step requirement §23.4).  `evolve()` is parameter-free.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    diffusion::DiffusionChernoff,
    howland::{HowlandLift, HowlandState},
    ChernoffSemigroup, Grid1D, GridFn1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Coefficient stubs (unit diffusion, no drift/reaction)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_hw(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_hw(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type HowlandKernel = HowlandLift<DiffUnit, f64>;
type HowlandTuple = (HowlandKernel, HowlandState<GridFn1D<f64>, f64>, Grid1D<f64>);

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn extract_u0_hw(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
    if u0.length() as usize != n {
        return Err(make_js_error("GridMismatch", "u0.length() must equal n"));
    }
    let mut buf = vec![0.0f64; n];
    u0.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(buf)
}

fn vec_to_js_hw(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_howland_wasm(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    n_t: usize,
    t_horizon: f64,
) -> Result<HowlandTuple, semiflow::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n)?;
    let diff = DiffusionChernoff::new(unit_a_hw, zero_hw, zero_hw, 1.0, grid);
    let lift = HowlandLift::new(diff, t_horizon, n_t)?;
    let slice = GridFn1D::new(grid, u0.to_vec())?;
    let samples: Vec<GridFn1D<f64>> = (0..n_t).map(|_| slice.clone()).collect();
    let state = HowlandState::new(samples)?;
    Ok((lift, state, grid))
}

// ---------------------------------------------------------------------------
// GIL-free compute helper
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn step_howland(
    lift: &HowlandKernel,
    state: &HowlandState<GridFn1D<f64>, f64>,
) -> Result<HowlandState<GridFn1D<f64>, f64>, semiflow::SemiflowError> {
    let n_steps = lift.n_t() - 1;
    let sg = ChernoffSemigroup::new(lift.clone(), n_steps)?;
    sg.evolve(lift.delta_s() * n_steps as f64, state)
}

// ---------------------------------------------------------------------------
// Howland1D — JS class
// ---------------------------------------------------------------------------

/// 1-D nonautonomous heat via Howland lift (M11).
///
/// Backed by `HowlandLift<DiffusionChernoff<f64>>`.  At construction the
/// time grid `[0, t_horizon]` is divided into `n_t − 1` intervals of size
/// `delta_s = t_horizon / (n_t − 1)`.  Calling `evolve()` advances by
/// one full `t_horizon` sweep (matched-step requirement §23.4).
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Howland1D {
    lift: HowlandKernel,
    state: HowlandState<GridFn1D<f64>, f64>,
    grid: Grid1D<f64>,
    n_t: usize,
    t_horizon: f64,
}

#[wasm_bindgen]
impl Howland1D {
    /// Construct `Howland1D`.
    ///
    /// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
    /// - `n` — spatial grid nodes (≥ 4).
    /// - `u0` — `Float64Array` of length `n`; initial condition (finite).
    /// - `n_t` — temporal grid points (≥ 2, default 11).
    /// - `t_horizon` — time horizon T (> 0, default 0.1).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        n_t: usize,
        t_horizon: f64,
    ) -> Result<Howland1D, JsValue> {
        if n_t < 2 {
            return Err(make_js_error("OutOfDomain", "n_t must be >= 2"));
        }
        if !t_horizon.is_finite() || t_horizon <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "t_horizon must be finite and > 0",
            ));
        }
        let buf = extract_u0_hw(u0, n)?;
        let (lift, state, grid) =
            build_howland_wasm(xmin, xmax, n, &buf, n_t, t_horizon).map_err(|e| err_to_js(&e))?;
        Ok(Howland1D {
            lift,
            state,
            grid,
            n_t,
            t_horizon,
        })
    }

    /// Advance by one full `t_horizon` sweep.
    ///
    /// Uses `n_steps = n_t − 1` Chernoff steps with `tau = delta_s`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self) -> Result<(), JsValue> {
        let next = step_howland(&self.lift, &self.state).map_err(|e| err_to_js(&e))?;
        self.state = next;
        Ok(())
    }

    /// Return the last time slice `u(t_horizon, ·)` as `Float64Array` of length `n`.
    #[must_use]
    pub fn values(&self) -> Float64Array {
        let last = &self.state.samples[self.n_t - 1];
        vec_to_js_hw(&last.values)
    }

    /// `delta_s = t_horizon / (n_t − 1)`.
    #[must_use]
    pub fn delta_s(&self) -> f64 {
        self.lift.delta_s()
    }

    /// Number of temporal grid points `n_t`.
    #[must_use]
    pub fn n_t(&self) -> usize {
        self.n_t
    }

    /// Time horizon `T` set at construction.
    #[must_use]
    pub fn t_horizon(&self) -> f64 {
        self.t_horizon
    }

    /// Approximation order (always 1).
    #[must_use]
    pub fn order(&self) -> u32 {
        1
    }

    /// Number of spatial grid nodes `n`.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.grid.n
    }
}
