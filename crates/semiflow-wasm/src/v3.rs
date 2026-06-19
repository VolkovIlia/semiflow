//! v3.0 WASM surface (ADR-0076, Wave F, Approach A).
//!
//! Wraps `semiflow_core` v3 types ([`Evolver`], [`Growth<f64>`], `apply_into`)
//! for JavaScript callers.  **Additive** to the existing v2 JS classes; the
//! v2 compatibility shim layer was hard-removed at v4.0 (ADR-0084).
//!
//! ## v3 JS classes
//!
//! - [`GrowthV3`] — JavaScript mirror of `Growth<f64>`.  Has `.multiplier`
//!   and `.omega` getter properties.
//!
//! - [`EvolverHeat1DUnitV3`] — wraps `Evolver<DiffusionChernoff<f64>, f64>`
//!   with a `current: GridFn1D<f64>` in-place state.  Mirrors the Wave D FFI
//!   pattern (opaque Rust state + JS-visible methods) and Wave E `PyO3` pattern
//!   (zero-alloc `evolve_into`, `values()` copy).
//!
//! ## Panic boundary
//!
//! WASM uses `[profile.release]` (`panic = "abort"`) per ADR-0028 Amendment 1
//! — NO `catch_unwind`.  All error paths return `Err(JsValue)`.
//!
//! ## Send + Sync
//!
//! WASM is single-threaded by spec (`wasm32-unknown-unknown`); `Send + Sync`
//! bounds are vacuously safe here.  `DiffusionChernoff<f64>` and `GridFn1D<f64>`
//! already implement `Send + Sync`; `EvolverHeat1DUnitV3` needs no extra
//! `unsafe impl`.

#![allow(unsafe_code)]

use wasm_bindgen::prelude::*;

use semiflow_core::{
    ChernoffFunction, DiffusionChernoff, Evolver, Grid1D, GridFn1D, Growth, ScratchPool,
};

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// GrowthV3 — JavaScript mirror of Growth<f64>
// ---------------------------------------------------------------------------

/// Growth bound `‖S(τ)‖ ≤ M · exp(ω · τ)` (v3.0, ADR-0074 / ADR-0076).
///
/// JS-facing class mirroring `Rust Growth<f64>`.
///
/// ## Properties
///
/// - `multiplier` — `M ≥ 1.0`.  For unit diffusion: `1.0`.
/// - `omega` — `ω` (finite).  For unit diffusion: `0.0`.
///
/// ## JS Example
///
/// ```js
/// const ev = new EvolverHeat1DUnitV3(-1.0, 1.0, 64, u0, 32);
/// const g = ev.growth();
/// console.log(g.multiplier, g.omega);  // 1 0
/// ```
#[wasm_bindgen]
pub struct GrowthV3 {
    /// M in `‖S(τ)‖ ≤ M · exp(ω · τ)`.
    multiplier: f64,
    /// ω in `‖S(τ)‖ ≤ M · exp(ω · τ)`.
    omega: f64,
}

#[wasm_bindgen]
impl GrowthV3 {
    /// `M ≥ 1.0`; for unit diffusion: `1.0`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// ω (finite); for unit diffusion: `0.0`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn omega(&self) -> f64 {
        self.omega
    }
}

impl From<Growth<f64>> for GrowthV3 {
    fn from(g: Growth<f64>) -> Self {
        Self {
            multiplier: g.multiplier,
            omega: g.omega,
        }
    }
}

// ---------------------------------------------------------------------------
// EvolverHeat1DUnitV3 — inner Rust state (private)
// ---------------------------------------------------------------------------

/// Inner state for `EvolverHeat1DUnitV3` (Rust-private, heap-owned by wasm-bindgen).
///
/// Mirrors `crates/semiflow-ffi/src/v3.rs` `EvolverInnerV3` and
/// `crates/semiflow-py/src/v3.rs` `EvolverInner`.
struct EvolverInnerV3 {
    evolver: Evolver<DiffusionChernoff<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// EvolverHeat1DUnitV3 JS class
// ---------------------------------------------------------------------------

/// v3.0 Evolver for the unit-diffusion heat equation (ADR-0076, Wave F).
///
/// Solves `∂_t u = ∂²u` on `[domain_lo, domain_hi]` with `n_grid` nodes
/// and `n_chernoff` Chernoff iterations per `evolveInto` call.
///
/// This is the **v3-native** WASM class wrapping
/// `Evolver<DiffusionChernoff<f64>, f64>` directly (zero-alloc `apply_into`
/// hot path).  For the v2 allocating API, use `Heat1D` which is preserved
/// for 12-month compat per ADR-0035 §9.
///
/// ## JS Example
///
/// ```js
/// import init, { EvolverHeat1DUnitV3 } from "@semiflow/wasm";
/// await init();
/// const u0 = new Float64Array(64).fill(0).map((_, i) => Math.exp(-i/10));
/// const ev = new EvolverHeat1DUnitV3(-1.0, 1.0, 64, u0, 32);
/// const out = new Float64Array(64);
/// ev.evolveInto(0.05, out);
/// const g = ev.growth();
/// console.log(g.multiplier, g.omega);
/// ```
#[wasm_bindgen]
pub struct EvolverHeat1DUnitV3 {
    inner: EvolverInnerV3,
}

#[wasm_bindgen]
impl EvolverHeat1DUnitV3 {
    /// Create a v3 Evolver for unit-diffusion heat.
    ///
    /// ## Parameters
    /// - `domain_lo` — left boundary; must be finite.
    /// - `domain_hi` — right boundary; must be finite and > `domain_lo`.
    /// - `n_grid` — number of grid nodes; must be ≥ 4.
    /// - `u0` — `Float64Array` of length exactly `n_grid`; all elements finite.
    /// - `n_chernoff` — Chernoff iteration count; must be ≥ 1.
    ///
    /// ## Errors
    /// - `.kind = "GridMismatch"` — geometry invalid or `u0.length != n_grid`.
    /// - `.kind = "NanInf"` — `u0` contains NaN or Inf.
    /// - `.kind = "OutOfDomain"` — `n_chernoff == 0`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &js_sys::Float64Array,
        n_chernoff: usize,
    ) -> Result<EvolverHeat1DUnitV3, JsValue> {
        if u0.length() as usize != n_grid {
            return Err(make_js_error(
                "GridMismatch",
                "u0.length() must equal n_grid",
            ));
        }
        let mut buf = vec![0.0f64; n_grid];
        u0.copy_to(&mut buf);
        let inner = build_evolver_v3(domain_lo, domain_hi, n_grid, n_chernoff, &buf)
            .map_err(|e| err_to_js(&e))?;
        Ok(EvolverHeat1DUnitV3 { inner })
    }

    /// Evolve the current state in-place by time `t`.
    ///
    /// Writes the evolved values into `out` (zero-alloc).  The internal
    /// current state is updated to the result.
    ///
    /// ## Parameters
    /// - `t` — time to advance; must be ≥ 0 and finite.
    /// - `out` — `Float64Array` of length `size()`; filled with result.
    ///
    /// ## Errors
    /// - `.kind = "GridMismatch"` — `out.length != size()`.
    /// - `.kind = "OutOfDomain"` — `t < 0` or non-finite.
    #[wasm_bindgen(js_name = "evolveInto")]
    pub fn evolve_into(&mut self, t: f64, out: &js_sys::Float64Array) -> Result<(), JsValue> {
        if !t.is_finite() || t < 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
        }
        let expected = self.inner.current.values.len();
        if out.length() as usize != expected {
            return Err(make_js_error(
                "GridMismatch",
                "out.length() must equal size()",
            ));
        }
        let mut dst = self.inner.current.clone();
        let mut scratch = ScratchPool::<f64>::new();
        self.inner
            .evolver
            .evolve_into(t, &self.inner.current, &mut dst, &mut scratch)
            .map_err(|e| err_to_js(&e))?;
        self.inner.current = dst;
        #[allow(clippy::cast_possible_truncation)]
        out.copy_from(&self.inner.current.values);
        Ok(())
    }

    /// Return the current state as a new `Float64Array` (copy).
    ///
    /// Mutations to the returned array do not affect the `EvolverHeat1DUnitV3` state.
    #[must_use]
    pub fn values(&self) -> js_sys::Float64Array {
        let v = &self.inner.current.values;
        #[allow(clippy::cast_possible_truncation)]
        let arr = js_sys::Float64Array::new_with_length(v.len() as u32);
        arr.copy_from(v);
        arr
    }

    /// Return the number of grid nodes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.current.values.len()
    }

    /// Return the growth bound of the underlying `DiffusionChernoff` kernel.
    ///
    /// For unit diffusion: `multiplier = 1.0`, `omega = 0.0`.
    #[must_use]
    pub fn growth(&self) -> GrowthV3 {
        let g: Growth<f64> = self.inner.evolver.func.growth();
        GrowthV3::from(g)
    }

    /// Return the Chernoff iteration count.
    #[wasm_bindgen(js_name = "nChernoff")]
    #[must_use]
    pub fn n_chernoff(&self) -> usize {
        self.inner.evolver.n
    }
}

// ---------------------------------------------------------------------------
// Private builder helper
// ---------------------------------------------------------------------------

/// Build an `EvolverInnerV3` for unit-diffusion heat on `[lo, hi]`.
///
/// Mirrors `crates/semiflow-ffi/src/v3.rs` `build_evolver_heat_unit` and
/// `crates/semiflow-py/src/v3.rs` `build_evolver_inner`.
fn build_evolver_v3(
    lo: f64,
    hi: f64,
    n_grid: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<EvolverInnerV3, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(lo, hi, n_grid)?;
    let chernoff = DiffusionChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let evolver = Evolver::new(chernoff, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(EvolverInnerV3 { evolver, current })
}

// ---------------------------------------------------------------------------
// Coefficient stubs (a = 1.0, a' = 0, a'' = 0)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}

extern "Rust" fn zero_d(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

/// Return `DomainViolation` if any element of `u0` is non-finite.
fn validate_u0_finite(u0: &[f64]) -> Result<(), semiflow_core::SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(semiflow_core::SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}
