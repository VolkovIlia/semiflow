//! Boundary-condition Chernoff kernels for WebAssembly (`full` feature).
//!
//! Exposes JS classes mirroring the Python `semiflow-py` binding:
//!
//! | JS class             | Core type                                                           | M#  |
//! |----------------------|---------------------------------------------------------------------|-----|
//! | `Killing1D`          | `KillingChernoff<DiffusionChernoff, BoxRegion<f64,1>>`              | M8  |
//! | `Reflected1D`        | `ReflectedHeatChernoff<DiffusionChernoff, HalfSpaceRegion<f64,1>>` | M9  |
//! | `Robin1D`            | `RobinHeatChernoff<DiffusionChernoff, HalfSpaceRobin<f64,1>>`      | M10 |
//! | `Resolvent1D`        | `LaplaceChernoffResolvent<DiffusionChernoff, f64>`                  | M7  |
//! | `KilledDirichlet1D`  | `Evolver<KilledDirichletChernoff>`                                  |     |
//!
//! `DirichletHeat2nd1D` (M11, §21.9) lives in `bc_wasm2.rs` (suckless split).
//!
//! Error model: same `.kind`-tagged JS `Error` as `Heat1D` — see crate docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`; validate first.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    killing::{BoxRegion, KillingChernoff},
    reflection::{HalfSpaceRegion, ReflectedHeatChernoff},
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
    robin::{HalfSpaceRobin, RobinHeatChernoff},
    ChernoffSemigroup, DiffusionChernoff, Evolver, Grid1D, GridFn1D, InterpKind,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Unit coefficient stubs (fn pointer, not closure — matches Python binding)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_bc(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_bc(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Shared validation helpers
// ---------------------------------------------------------------------------

fn extract_u0(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
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

fn validate_evolve(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn fn_to_js(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type KillingKernel = KillingChernoff<DiffUnit, BoxRegion<f64, 1>>;
type ReflectedKernel = ReflectedHeatChernoff<DiffUnit, HalfSpaceRegion<f64, 1>>;
type RobinKernel = RobinHeatChernoff<DiffUnit, HalfSpaceRobin<f64, 1>>;
type ResolventKernel = LaplaceChernoffResolvent<DiffUnit, f64>;
type KilledDirichletKernel = semiflow::killed_dirichlet::KilledDirichletChernoff;

// ===========================================================================
// Killing1D
// ===========================================================================

/// 1-D heat with absorbing Dirichlet BC via Feynman-Kac killing (M8).
///
/// Solves `∂_t u = ∂²u` with `u = 0` outside the box `[lo, hi)`.
/// Backed by `KillingChernoff<DiffusionChernoff, BoxRegion>` (order 1).
///
/// ## Parameters
/// - `xmin`, `xmax` — domain bounds (`xmin < xmax`, finite).
/// - `n`    — grid nodes (≥ 4).
/// - `u0`   — `Float64Array` of length `n`, all finite.
/// - `lo`, `hi` — killing box bounds (defaults: middle half of domain).
#[wasm_bindgen]
pub struct Killing1D {
    sg: ChernoffSemigroup<KillingKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Killing1D {
    /// Create a new `Killing1D` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        lo: f64,
        hi: f64,
    ) -> Result<Killing1D, JsValue> {
        let buf = extract_u0(u0, n)?;
        let range = xmax - xmin;
        let lo_eff = if lo.is_finite() {
            lo
        } else {
            xmin + range * 0.25
        };
        let hi_eff = if hi.is_finite() {
            hi
        } else {
            xmax - range * 0.25
        };
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
        let region = BoxRegion::<f64, 1>::new([lo_eff], [hi_eff]).map_err(|e| err_to_js(&e))?;
        let kernel = KillingChernoff::new(diff, region).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Killing1D { sg, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let func = self.sg.func.clone();
        let sg = ChernoffSemigroup::new(func, n_steps).map_err(|e| err_to_js(&e))?;
        self.current = sg.evolve(t, &self.current).map_err(|e| err_to_js(&e))?;
        self.sg = sg;
        Ok(())
    }

    /// Return current grid values as a new `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js(&self.current.values)
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}

// ===========================================================================
// Reflected1D
// ===========================================================================

/// 1-D heat with Neumann BC via the image method (M9).
///
/// Solves `∂_t u = ∂²u` with zero-flux `∂_x u = 0` at `origin`.
/// Backed by `ReflectedHeatChernoff<DiffusionChernoff, HalfSpaceRegion>` (order 2).
///
/// ## Parameters
/// - `xmin`, `xmax` — domain bounds.
/// - `n`      — grid nodes (≥ 4).
/// - `u0`     — `Float64Array` of length `n`, all finite.
/// - `origin` — reflecting boundary point (default = `xmin`).
#[wasm_bindgen]
pub struct Reflected1D {
    sg: ChernoffSemigroup<ReflectedKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Reflected1D {
    /// Create a new `Reflected1D` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        origin: f64,
    ) -> Result<Reflected1D, JsValue> {
        let buf = extract_u0(u0, n)?;
        let origin_eff = if origin.is_finite() { origin } else { xmin };
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
        let region =
            HalfSpaceRegion::<f64, 1>::new([origin_eff], [1.0]).map_err(|e| err_to_js(&e))?;
        let kernel = ReflectedHeatChernoff::new(diff, region).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Reflected1D { sg, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let func = self.sg.func.clone();
        let sg = ChernoffSemigroup::new(func, n_steps).map_err(|e| err_to_js(&e))?;
        self.current = sg.evolve(t, &self.current).map_err(|e| err_to_js(&e))?;
        self.sg = sg;
        Ok(())
    }

    /// Return current grid values as a new `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js(&self.current.values)
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}

// ===========================================================================
// Robin1D
// ===========================================================================

/// 1-D heat with Robin BC via the skew image method (M10).
///
/// Solves `∂_t u = ∂²u` with `α·u(0) − β·∂_x u(0) = 0` at `origin`.
/// Backed by `RobinHeatChernoff<DiffusionChernoff, HalfSpaceRobin>` (order 1).
///
/// ## Parameters
/// - `xmin`, `xmax` — domain bounds.
/// - `n`      — grid nodes (≥ 4).
/// - `u0`     — `Float64Array` of length `n`, all finite.
/// - `alpha`  — Robin coefficient on u (≥ 0, default 1.0).
/// - `beta`   — Robin coefficient on ∂_n u (> 0, default 1.0).
/// - `origin` — Robin boundary point (default = `xmin`).
#[wasm_bindgen]
pub struct Robin1D {
    sg: ChernoffSemigroup<RobinKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Robin1D {
    /// Create a new `Robin1D` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        alpha: f64,
        beta: f64,
        origin: f64,
    ) -> Result<Robin1D, JsValue> {
        let buf = extract_u0(u0, n)?;
        let origin_eff = if origin.is_finite() { origin } else { xmin };
        // Mirror Python: downgrade to CubicHermite so interp_generic
        // can service ghost reflection calls (same note as bc_kernels2.rs).
        let grid = Grid1D::new(xmin, xmax, n)
            .map_err(|e| err_to_js(&e))?
            .with_interp(InterpKind::CubicHermite);
        let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
        let region = HalfSpaceRobin::<f64, 1>::new([origin_eff], [1.0], alpha, beta)
            .map_err(|e| err_to_js(&e))?;
        let kernel = RobinHeatChernoff::new(diff, region).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Robin1D { sg, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let func = self.sg.func.clone();
        let sg = ChernoffSemigroup::new(func, n_steps).map_err(|e| err_to_js(&e))?;
        self.current = sg.evolve(t, &self.current).map_err(|e| err_to_js(&e))?;
        self.sg = sg;
        Ok(())
    }

    /// Return current grid values as a new `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js(&self.current.values)
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}

// ===========================================================================
// Resolvent1D
// ===========================================================================

/// 1-D Laplace-Chernoff resolvent `(λI − ∂²)⁻¹ g` (M7).
///
/// Computes `R̃(λ) g = ∫₀^∞ exp(−λt) S(t)g dt` via Gauss-Laguerre-32
/// (Remizov 2025 Vladikavkaz Thm 3).
///
/// ## Parameters
/// - `xmin`, `xmax` — domain bounds.
/// - `n`          — grid nodes (≥ 4).
/// - `nChernoff`  — Chernoff truncation for inner heat semigroup (default 32).
#[wasm_bindgen]
pub struct Resolvent1D {
    resolvent: ResolventKernel,
    grid: Grid1D<f64>,
}

#[wasm_bindgen]
impl Resolvent1D {
    /// Create a new `Resolvent1D`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, n_chernoff: usize) -> Result<Resolvent1D, JsValue> {
        if n_chernoff == 0 {
            return Err(make_js_error("OutOfDomain", "nChernoff must be >= 1"));
        }
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
        let resolvent =
            LaplaceChernoffResolvent::new(diff, n_chernoff, LaplaceQuadrature::GaussLaguerre32)
                .map_err(|e| err_to_js(&e))?;
        Ok(Resolvent1D { resolvent, grid })
    }

    /// Evaluate `R̃(lambda) g`; return result as `Float64Array`.
    ///
    /// ## Parameters
    /// - `lambda` — resolvent parameter (> 0, finite).
    /// - `g`      — `Float64Array` of length `size()`, all finite.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn eval(&self, lambda: f64, g: &Float64Array) -> Result<Float64Array, JsValue> {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "lambda must be finite and > 0",
            ));
        }
        let n = self.grid.n;
        if g.length() as usize != n {
            return Err(make_js_error("GridMismatch", "g.length must equal n"));
        }
        let mut g_buf = vec![0.0f64; n];
        g.copy_to(&mut g_buf);
        for &v in &g_buf {
            if !v.is_finite() {
                return Err(make_js_error("NanInf", "g contains NaN or Inf"));
            }
        }
        let gfn = GridFn1D::new(self.grid, g_buf).map_err(|e| err_to_js(&e))?;
        let result = self
            .resolvent
            .eval(lambda, &gfn)
            .map_err(|e| err_to_js(&e))?;
        Ok(fn_to_js(&result.values))
    }

    /// Number of grid nodes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.grid.n
    }
}

// ===========================================================================
// KilledDirichlet1D
// ===========================================================================

/// 1-D heat with absorbing Dirichlet BC via Crank–Nicolson Cayley map.
///
/// `u|∂R = 0`; order 2 (math §44.ter, ADR-0135 Amendment 2).
/// Mirrors `KilledDirichlet1D` from `semiflow-py` `greeks_py.rs`.
///
/// ## Parameters
/// - `domainLo`, `domainHi` — domain boundaries.
/// - `nGrid`    — grid nodes (≥ 3).
/// - `u0`       — `Float64Array` of length `nGrid`, all finite.
/// - `nChernoff` — Chernoff iteration count (≥ 1).
#[wasm_bindgen]
pub struct KilledDirichlet1D {
    evolver: Evolver<KilledDirichletKernel>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl KilledDirichlet1D {
    /// Create a new `KilledDirichlet1D` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &Float64Array,
        n_chernoff: usize,
    ) -> Result<KilledDirichlet1D, JsValue> {
        let buf = extract_u0(u0, n_grid)?;
        if n_chernoff == 0 {
            return Err(make_js_error("OutOfDomain", "nChernoff must be >= 1"));
        }
        let grid = Grid1D::new(domain_lo, domain_hi, n_grid).map_err(|e| err_to_js(&e))?;
        let kernel = KilledDirichletKernel::new(|_| 1.0_f64, |_| 0.0_f64, grid)
            .map_err(|e| err_to_js(&e))?;
        let evolver = Evolver::new(kernel, n_chernoff).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(KilledDirichlet1D { evolver, current })
    }

    /// Advance by `t`; return evolved grid as `Float64Array`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn apply(&mut self, t: f64) -> Result<Float64Array, JsValue> {
        if !t.is_finite() || t < 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
        }
        let result = self
            .evolver
            .evolve(t, &self.current)
            .map_err(|e| err_to_js(&e))?;
        self.current = result;
        Ok(fn_to_js(&self.current.values))
    }

    /// Number of grid nodes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.current.values.len()
    }
}
