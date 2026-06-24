//! WASM binding for Dual-AD Greeks (Δ, Γ) of 1D unit-diffusion heat.
//!
//! Implements `EvolverHeat1DGreeksV3` (§4, `V8_PHASE5_BINDING_GREEKS_DESIGN.md)`:
//! `.greeks(t)` returns `{ value, delta, gamma }` — three `Float64Array`s.
//!
//! ## AD path
//!
//! A single `Dual<Dual<f64>>` hyper-dual sweep computes value/Δ/Γ together
//! (§1, math.md §46.4). θ is the diffusion-scale; only the `scale` arg is
//! seeded via `with_closure` — `a'`/`a''` fn-closures carry zero duals.
//!
//! - `value[i]  = u.values[i].value.value`
//! - `delta[i]  = u.values[i].tangent.value`   (∂u/∂θ, outer tangent)
//! - `gamma[i]  = u.values[i].tangent.tangent` (∂²u/∂θ², hyper component)
//!
//! Profile: `[profile.release]` (`panic = "abort"`) per ADR-0028 Amendment 1.
//! No `catch_unwind` — all error paths return `Err(JsValue)`.
//!
//! ADR-0028 Amendment 2: per-crate duplication — no shared util between
//! semiflow-ffi/semiflow-py/semiflow-wasm.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::assigning_clones,
    clippy::cast_possible_truncation,
    clippy::missing_errors_doc,
    clippy::type_complexity
)]

use js_sys::{Object, Reflect};
use semiflow::{dual::Dual, DiffusionChernoff, Grid1D, GridFn1D};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Type alias for readability
// ---------------------------------------------------------------------------

type HyperDual = Dual<Dual<f64>>;

// ---------------------------------------------------------------------------
// GreeksInnerWasm — heap-owned state
// ---------------------------------------------------------------------------

/// Inner Rust state: grid geometry + θ seed + current value-only chain state.
///
/// Mirrors `GreeksInner` in `semiflow-py/src/greeks_py.rs` (per-crate dup).
struct GreeksInnerWasm {
    /// Number of Chernoff iterations.
    n_chernoff: usize,
    /// Grid geometry (f64; shared between sweeps).
    grid_f64: Grid1D<f64>,
    /// Diffusion-scale θ at which Δ/Γ are evaluated.
    theta: f64,
    /// Current state (value-only, f64; updated after each `.greeks()` call).
    current: Vec<f64>,
}

// ---------------------------------------------------------------------------
// EvolverHeat1DGreeksV3 WASM class
// ---------------------------------------------------------------------------

/// v8.0.0 Hyper-dual Greeks evolver for unit-diffusion heat (ADR-0133 A3).
///
/// Computes `{value, delta, gamma}` — the solution and its first/second
/// derivatives w.r.t. the diffusion-scale parameter θ — via a single
/// `Dual<Dual<f64>>` hyper-dual sweep (math §46.4).
///
/// ## JS Example
///
/// ```js
/// import init, { EvolverHeat1DGreeksV3 } from "@semiflow/wasm";
/// await init();
/// const N = 64;
/// const u0 = new Float64Array(N).map((_, i) => {
///   const x = -5 + 10*i/(N-1);
///   return Math.exp(-x*x);
/// });
/// const ev = new EvolverHeat1DGreeksV3(-5, 5, N, u0, 32, 0.5);
/// const { value, delta, gamma } = ev.greeks(0.05);
/// // value, delta, gamma are each Float64Array of length N
/// ```
///
/// ## Error model (`.kind` discriminator)
///
/// - `"GridMismatch"` — invalid geometry or `u0.length != n_grid`.
/// - `"NanInf"`       — non-finite value in `u0` or `scale_theta`.
/// - `"OutOfDomain"`  — `n_chernoff == 0` or `t < 0`.
#[wasm_bindgen]
pub struct EvolverHeat1DGreeksV3 {
    inner: GreeksInnerWasm,
}

#[wasm_bindgen]
impl EvolverHeat1DGreeksV3 {
    /// Construct a Greeks evolver for unit-diffusion heat.
    ///
    /// ## Parameters
    /// - `domain_lo`   — left boundary (finite).
    /// - `domain_hi`   — right boundary (finite, > `domain_lo`).
    /// - `n_grid`      — number of grid nodes (>= 4).
    /// - `u0`          — `Float64Array` of length `n_grid`; all finite.
    /// - `n_chernoff`  — Chernoff iteration count (>= 1).
    /// - `scale_theta` — diffusion-scale θ (default 0.5).
    #[wasm_bindgen(constructor)]
    pub fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &js_sys::Float64Array,
        n_chernoff: usize,
        scale_theta: f64,
    ) -> Result<EvolverHeat1DGreeksV3, JsValue> {
        if u0.length() as usize != n_grid {
            return Err(make_js_error("GridMismatch", "u0.length must equal n_grid"));
        }
        let mut buf = vec![0.0f64; n_grid];
        u0.copy_to(&mut buf);
        let inner = build_greeks_inner(domain_lo, domain_hi, n_grid, &buf, n_chernoff, scale_theta)
            .map_err(|e| err_to_js(&e))?;
        Ok(EvolverHeat1DGreeksV3 { inner })
    }

    /// Advance by `t`; return `{ value, delta, gamma }` as a JS object.
    ///
    /// Each field is a `Float64Array` of length `size()`.  The internal
    /// current state is updated to the primal-value result.
    ///
    /// ## Parameters
    /// - `t` — time step (>= 0, finite).
    ///
    /// ## Errors
    /// - `.kind = "OutOfDomain"` — `t < 0` or non-finite.
    pub fn greeks(&mut self, t: f64) -> Result<JsValue, JsValue> {
        if !t.is_finite() || t < 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
        }
        let (values, deltas, gammas) = run_hyper_dual_sweep(
            self.inner.grid_f64,
            &self.inner.current,
            self.inner.n_chernoff,
            self.inner.theta,
            t,
        )
        .map_err(|e| err_to_js(&e))?;

        self.inner.current = values.clone();
        Ok(make_greeks_object(&values, &deltas, &gammas))
    }

    /// Return the number of grid nodes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.current.len()
    }

    /// Return the Chernoff iteration count.
    #[wasm_bindgen(js_name = "nChernoff")]
    #[must_use]
    pub fn n_chernoff(&self) -> usize {
        self.inner.n_chernoff
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust hyper-dual sweep
// ---------------------------------------------------------------------------

/// Run one `Dual<Dual<f64>>` Chernoff sweep; return demultiplexed buffers.
///
/// Mirrors `run_hyper_dual_sweep` in `semiflow-py/src/greeks_py.rs` exactly,
/// per the per-crate duplication rule (ADR-0028 Amendment 2).
///
/// `with_closure` is used to capture `theta_seeded` — fn-ptrs cannot capture.
fn run_hyper_dual_sweep(
    grid_f64: Grid1D<f64>,
    u0_f64: &[f64],
    n_chernoff: usize,
    theta: f64,
    t: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), semiflow::SemiflowError> {
    let lo = Dual::constant(Dual::constant(grid_f64.xmin));
    let hi = Dual::constant(Dual::constant(grid_f64.xmax));
    let grid_hd = Grid1D::<HyperDual>::new_generic(lo, hi, grid_f64.n)?;

    // Seed θ: both inner and outer tangents = 1 → ∂/∂θ and ∂²/∂θ².
    let theta_seeded: HyperDual = Dual::variable(Dual::variable(theta));

    let chernoff = DiffusionChernoff::with_closure(
        move |_: HyperDual| theta_seeded,
        |_: HyperDual| Dual::constant(Dual::constant(0.0_f64)),
        |_: HyperDual| Dual::constant(Dual::constant(0.0_f64)),
        theta,
        grid_hd,
    );

    let u0_hd: Vec<HyperDual> = u0_f64
        .iter()
        .map(|&v| Dual::constant(Dual::constant(v)))
        .collect();
    let mut current: GridFn1D<HyperDual> = GridFn1D::new_generic(grid_hd, u0_hd)?;

    #[allow(clippy::cast_precision_loss)]
    let tau_f64 = t / (n_chernoff as f64);
    let tau: HyperDual = Dual::constant(Dual::constant(tau_f64));

    for _ in 0..n_chernoff {
        current = chernoff.apply_f(tau, &current)?;
    }
    Ok(demux_hyper_dual(&current.values))
}

// ---------------------------------------------------------------------------
// Demultiplex helper
// ---------------------------------------------------------------------------

/// Demultiplex hyper-dual buffer into (value, delta, gamma) f64 vecs.
///
/// - `value[i]  = h.value.value`     — primal
/// - `delta[i]  = h.tangent.value`   — ∂u/∂θ (outer tangent)
/// - `gamma[i]  = h.tangent.tangent` — ∂²u/∂θ² (hyper component)
fn demux_hyper_dual(hds: &[HyperDual]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = hds.len();
    let mut values = Vec::with_capacity(n);
    let mut deltas = Vec::with_capacity(n);
    let mut gammas = Vec::with_capacity(n);
    for hd in hds {
        values.push(hd.value.value);
        deltas.push(hd.tangent.value);
        gammas.push(hd.tangent.tangent);
    }
    (values, deltas, gammas)
}

// ---------------------------------------------------------------------------
// JS object builder
// ---------------------------------------------------------------------------

/// Build `{ value: Float64Array, delta: Float64Array, gamma: Float64Array }`.
fn make_greeks_object(values: &[f64], deltas: &[f64], gammas: &[f64]) -> JsValue {
    let n = values.len() as u32;
    let arr_v = js_sys::Float64Array::new_with_length(n);
    arr_v.copy_from(values);
    let arr_d = js_sys::Float64Array::new_with_length(n);
    arr_d.copy_from(deltas);
    let arr_g = js_sys::Float64Array::new_with_length(n);
    arr_g.copy_from(gammas);
    let obj = Object::new();
    let _ = Reflect::set(&obj, &"value".into(), &arr_v);
    let _ = Reflect::set(&obj, &"delta".into(), &arr_d);
    let _ = Reflect::set(&obj, &"gamma".into(), &arr_g);
    obj.into()
}

// ---------------------------------------------------------------------------
// Builder helper
// ---------------------------------------------------------------------------

fn build_greeks_inner(
    lo: f64,
    hi: f64,
    n_grid: usize,
    u0: &[f64],
    n_chernoff: usize,
    theta: f64,
) -> Result<GreeksInnerWasm, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    validate_theta(theta)?;
    if n_chernoff == 0 {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "n_chernoff must be >= 1",
            value: 0.0,
        });
    }
    let grid_f64 = Grid1D::new(lo, hi, n_grid)?;
    Ok(GreeksInnerWasm {
        n_chernoff,
        grid_f64,
        theta,
        current: u0.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_u0_finite(u0: &[f64]) -> Result<(), semiflow::SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

fn validate_theta(theta: f64) -> Result<(), semiflow::SemiflowError> {
    if !theta.is_finite() {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "scale_theta must be finite",
            value: theta,
        });
    }
    Ok(())
}
