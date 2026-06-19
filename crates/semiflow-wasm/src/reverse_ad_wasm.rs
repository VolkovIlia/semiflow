//! v9.0.0 WASM binding for `ReverseChernoff.value_and_grad` (math §51.5,
//! ADR-0156, Shift B).
//!
//! Exposes `ReverseHeat1D` — a JS class wrapping
//! `semiflow_core::ReverseChernoff<f64>` for the constant-a
//! `DiffusionChernoff` kernel (narrow scope, §51.5).
//!
//! ## JS API
//!
//! ```js
//! const rc = new ReverseHeat1D(0.4, -4.0, 4.0, 24, 8);
//! const u0     = new Float64Array(24); // fill with exp(-x^2)
//! const target = new Float64Array(24); // fill with 0
//! const result = rc.valueAndGrad(0.05, u0, target);
//! // result[0]: loss (value), result[1]: ∂J/∂θ (gradient)
//! ```
//!
//! The return is a plain JS object `{ value: f64, grad: f64 }` — a
//! `Float64Array` of length 2 (`[value, grad]`) is used under the hood
//! to avoid allocating a JS object on the Rust side (wasm-bindgen limitation
//! on returning structs with named fields without `JsValue`).
//!
//! ## Error model (`.kind`)
//!
//! - `"GridMismatch"` — `n_grid` < 4 or xmin >= xmax or array length mismatch.
//! - `"OutOfDomain"`  — theta <= 0, `n_steps` == 0, tau <= 0.
//! - `"NanInf"`       — NaN/Inf in u0 or target.
//!
//! ## Panic boundary (ADR-0028 Amendment 1)
//!
//! `[profile.release]` uses `panic = "abort"` — NO `catch_unwind`.
//!
//! ## Scope (NARROW — §51.5)
//!
//! Constant-a `DiffusionChernoff` ONLY; θ is the uniform diffusivity.
//! Variable-coefficient and nonlinear kernels are out of scope for v9.0.0.
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication of kernel construction required.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_errors_doc,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use wasm_bindgen::prelude::*;

use semiflow_core::{
    CheckpointSchedule, DiffusionChernoff, Dual, Grid1D, GridFn1D, InterpKind, ReverseChernoff,
};

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Kernel construction (per-crate duplicate, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

/// Build a `DiffusionChernoff<f64>` for constant `a(x) ≡ theta` (`CubicHermite`).
fn build_f64_kernel_wasm(theta: f64, grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::with_closure(move |_| theta, |_| 0.0_f64, |_| 0.0_f64, theta, grid)
}

/// Construct a `ReverseChernoff<f64>` from scalar params.
///
/// Both f64 and Dual<f64> grids are built here (`CubicHermite`, per `reverse_ad.rs` tests).
fn build_reverse_chernoff_wasm(
    theta: f64,
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_steps: usize,
) -> Result<ReverseChernoff<f64>, semiflow_core::error::SemiflowError> {
    // f64 grid.
    let grid_f64 = Grid1D::<f64>::new(xmin, xmax, n_grid)?.with_interp(InterpKind::CubicHermite);

    // Dual<f64> grid.
    let grid_dual =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(xmin), Dual::constant(xmax), n_grid)?
            .with_interp(InterpKind::CubicHermite);

    let kernel_f64 = build_f64_kernel_wasm(theta, grid_f64);

    let kernel_dual = DiffusionChernoff::<Dual<f64>>::with_closure(
        move |_: Dual<f64>| Dual::variable(theta),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        theta,
        grid_dual,
    );

    let schedule = CheckpointSchedule::sqrt_n(n_steps);
    Ok(ReverseChernoff::new(kernel_f64, kernel_dual, schedule))
}

// ---------------------------------------------------------------------------
// ReverseHeat1D JS class
// ---------------------------------------------------------------------------

/// Reverse-mode AD evolver for constant-a 1-D heat (v9.0.0, math §51, ADR-0156).
///
/// Computes `(J, ∂J/∂θ)` where `J(θ) = ‖(F_θ(τ))ⁿ u₀ − target‖²` via the
/// K=1 forward-mode Dual path (§51.4; 0-ULP parity with forward AD by construction).
///
/// ## NARROW scope (§51.5)
///
/// Constant-a `DiffusionChernoff` ONLY.  Variable-coefficient kernels are out
/// of scope for v9.0.0.
///
/// ## Error model
///
/// All methods return `Result<..., JsValue>`.  The `JsValue` is a JS Error with
/// a `.kind` discriminator (see module-level doc).
#[wasm_bindgen]
pub struct ReverseHeat1D {
    theta: f64,
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_steps: usize,
}

#[wasm_bindgen]
impl ReverseHeat1D {
    /// Construct a `ReverseHeat1D` evolver.
    ///
    /// Parameters
    /// ----------
    /// theta   : number — diffusivity θ > 0.
    /// xmin    : number — left domain boundary.
    /// xmax    : number — right domain boundary (xmax > xmin).
    /// nGrid   : number — grid nodes (>= 4).
    /// nSteps  : number — Chernoff steps per `valueAndGrad` call (>= 1).
    #[wasm_bindgen(constructor)]
    pub fn new(
        theta: f64,
        xmin: f64,
        xmax: f64,
        n_grid: usize,
        n_steps: usize,
    ) -> Result<ReverseHeat1D, JsValue> {
        if !theta.is_finite() || theta <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "ReverseHeat1D: theta must be finite and > 0",
            ));
        }
        if n_steps == 0 {
            return Err(make_js_error(
                "OutOfDomain",
                "ReverseHeat1D: nSteps must be >= 1",
            ));
        }
        // Validate grid.
        Grid1D::<f64>::new(xmin, xmax, n_grid).map_err(|e| err_to_js(&e))?;
        Ok(Self {
            theta,
            xmin,
            xmax,
            n_grid,
            n_steps,
        })
    }

    /// Compute `(J, ∂J/∂θ)` for the scalar diffusivity parameter.
    ///
    /// **K=1 path**: forward-mode `Dual<f64>` (§51.4; 0-ULP parity with core).
    ///
    /// Parameters
    /// ----------
    /// tau    : number       — per-step time increment (> 0, finite).
    /// u0     : `Float64Array` — initial condition, length nGrid.
    /// target : `Float64Array` — target state, length nGrid.
    ///
    /// Returns
    /// -------
    /// `Float64Array` of length 2: `[value, grad]`
    ///   - `[0]` = L² loss `‖(F_θ(τ))ⁿ u₀ − target‖²`.
    ///   - `[1]` = `∂J/∂θ` (scalar, 0-ULP vs core).
    #[wasm_bindgen(js_name = "valueAndGrad")]
    pub fn value_and_grad(
        &self,
        tau: f64,
        u0: &js_sys::Float64Array,
        target: &js_sys::Float64Array,
    ) -> Result<js_sys::Float64Array, JsValue> {
        validate_tau_wasm(tau, "ReverseHeat1D.valueAndGrad")?;
        let (rc, u0_fn, target_fn) = build_wasm_rc_inputs(
            self.theta,
            self.xmin,
            self.xmax,
            self.n_grid,
            self.n_steps,
            u0,
            target,
            "ReverseHeat1D.valueAndGrad",
        )?;
        let (value, grad) = rc
            .value_and_grad_k1(tau, self.n_steps, &u0_fn, &target_fn)
            .map_err(|e| err_to_js(&e))?;
        let arr = js_sys::Float64Array::new_with_length(2);
        arr.set_index(0, value);
        arr.set_index(1, grad);
        Ok(arr)
    }

    /// Compute `(J, grad_vec)` for a K-vector of parameters in ONE backward pass.
    ///
    /// **K-vector path (§51.9, ADR-0156 Amendment 1)**: the genuine cotangent backward
    /// sweep accumulates all K gradient components in one backward walk — O(1) trajectory
    /// passes independent of K. This is the capability asserted by `G_REVERSE_AD_ADVANTAGE`.
    ///
    /// ## 0-ULP parity
    ///
    /// Byte-identical to the Rust core `value_and_grad` for the same K, τ, u0, target
    /// (per `G_BINDING_REVERSE_AD_PARITY` sub-test 4; same arithmetic path).
    ///
    /// Parameters
    /// ----------
    /// tau    : number       — per-step time increment (> 0, finite).
    /// theta  : `Float64Array` — K diffusivity parameters, length >= 1.
    /// u0     : `Float64Array` — initial condition, length nGrid.
    /// target : `Float64Array` — target state, length nGrid.
    ///
    /// Returns
    /// -------
    /// `Float64Array` of length K+1: `[value, grad_0, grad_1, …, grad_{K-1}]`
    ///   - `[0]`   = L² loss `‖(F_θ(τ))ⁿ u₀ − target‖²`.
    ///   - `[1..K+1]` = gradient components `∂J/∂θ_p`.
    #[wasm_bindgen(js_name = "valueAndGradKvec")]
    pub fn value_and_grad_kvec(
        &self,
        tau: f64,
        theta: &js_sys::Float64Array,
        u0: &js_sys::Float64Array,
        target: &js_sys::Float64Array,
    ) -> Result<js_sys::Float64Array, JsValue> {
        validate_tau_wasm(tau, "ReverseHeat1D.valueAndGradKvec")?;
        let k = theta.length() as usize;
        if k == 0 {
            return Err(make_js_error(
                "OutOfDomain",
                "ReverseHeat1D.valueAndGradKvec: theta must be non-empty",
            ));
        }
        let mut theta_buf = vec![0.0f64; k];
        theta.copy_to(&mut theta_buf);
        validate_finite_wasm(&theta_buf, "theta")?;
        let (rc, u0_fn, target_fn) = build_wasm_rc_inputs(
            self.theta,
            self.xmin,
            self.xmax,
            self.n_grid,
            self.n_steps,
            u0,
            target,
            "ReverseHeat1D.valueAndGradKvec",
        )?;
        // K-vector backward sweep — ONE pass for all K gradients.
        let (value, grad_vec) = rc
            .value_and_grad(tau, self.n_steps, &u0_fn, &target_fn, &theta_buf)
            .map_err(|e| err_to_js(&e))?;
        // Pack result as Float64Array([value, grad_0, ..., grad_{K-1}]).
        let arr = js_sys::Float64Array::new_with_length((k + 1) as u32);
        arr.set_index(0, value);
        for (i, &g) in grad_vec.iter().enumerate() {
            arr.set_index((i + 1) as u32, g);
        }
        Ok(arr)
    }

    /// Return the diffusivity parameter θ.
    #[must_use]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Return the number of Chernoff steps.
    #[wasm_bindgen(js_name = "nSteps")]
    #[must_use]
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// Return the number of grid nodes.
    #[wasm_bindgen(js_name = "nGrid")]
    #[must_use]
    pub fn n_grid(&self) -> usize {
        self.n_grid
    }
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

/// Validate `tau > 0` and finite; returns `OutOfDomain` on failure.
fn validate_tau_wasm(tau: f64, ctx: &str) -> Result<(), JsValue> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(make_js_error(
            "OutOfDomain",
            &format!("{ctx}: tau must be finite and > 0"),
        ));
    }
    Ok(())
}

/// Copy + validate JS `Float64Array` pair, then build `ReverseChernoff` + `GridFn1D` pair.
fn build_wasm_rc_inputs(
    theta: f64,
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_steps: usize,
    u0: &js_sys::Float64Array,
    target: &js_sys::Float64Array,
    ctx: &str,
) -> Result<(ReverseChernoff<f64>, GridFn1D<f64>, GridFn1D<f64>), JsValue> {
    if u0.length() as usize != n_grid {
        return Err(make_js_error(
            "GridMismatch",
            &format!("{ctx}: u0.length must equal nGrid"),
        ));
    }
    if target.length() as usize != n_grid {
        return Err(make_js_error(
            "GridMismatch",
            &format!("{ctx}: target.length must equal nGrid"),
        ));
    }
    let mut u0_buf = vec![0.0f64; n_grid];
    u0.copy_to(&mut u0_buf);
    let mut target_buf = vec![0.0f64; n_grid];
    target.copy_to(&mut target_buf);
    validate_finite_wasm(&u0_buf, "u0")?;
    validate_finite_wasm(&target_buf, "target")?;
    let rc = build_reverse_chernoff_wasm(theta, xmin, xmax, n_grid, n_steps)
        .map_err(|e| err_to_js(&e))?;
    let grid = Grid1D::<f64>::new(xmin, xmax, n_grid)
        .map_err(|e| err_to_js(&e))?
        .with_interp(InterpKind::CubicHermite);
    let u0_fn = GridFn1D::new(grid, u0_buf).map_err(|e| err_to_js(&e))?;
    let target_fn = GridFn1D::new(grid, target_buf).map_err(|e| err_to_js(&e))?;
    Ok((rc, u0_fn, target_fn))
}

fn validate_finite_wasm(v: &[f64], name: &str) -> Result<(), JsValue> {
    for &x in v {
        if !x.is_finite() {
            return Err(make_js_error(
                "NanInf",
                &format!("ReverseHeat1D: {name} contains NaN or Inf ({x})"),
            ));
        }
    }
    Ok(())
}
