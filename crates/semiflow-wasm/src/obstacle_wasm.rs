//! Round-9 obstacle engine for WebAssembly (`full` feature).
//!
//! | JS class    | Core type                                              | Python mirror      |
//! |-------------|--------------------------------------------------------|--------------------|
//! | `Obstacle1D`| `ObstacleChernoff<DiffusionChernoff, *Obstacle, f64>`  | `ObstacleChernoff` |
//!
//! ## Design
//!
//! Mirrors `semiflow-py` `ObstacleChernoff` (`obstacle_py.rs`), D=1 only.
//! WASM boundary uses `Float64Array` (copy semantics); no numpy dependency.
//! Two obstacle variants: constant `level` (fast path) or per-node `obstacle_array`.
//! Strang-split path (`b ≠ 0 || c ≠ 0`) is NOT exposed here — WASM exposes the
//! fast-path pure-diffusion inner only (matching C FFI `obs_buf`/`level` surface).
//!
//! ## Supported constructor combinations
//!
//! | `obstacle_array` | `level`  | Kernel built                           |
//! |------------------|----------|----------------------------------------|
//! | `null`           | finite   | `ConstantObstacle(level)`              |
//! | `Float64Array`   | ignored  | `WasmArrayObstacle` (per-node floor)   |
//!
//! `a > 0` (diffusion), `b = 0`, `c = 0` always (fast path). Strang-split deferred.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    ChernoffFunction, ConstantObstacle, DiffusionChernoff, Grid1D, GridFn1D, Obstacle,
    ObstacleChernoff, ScratchPool, SemiflowError,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// WasmArrayObstacle — local per-node floor (mirrors FfiArrayObstacle)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct WasmArrayObstacle {
    values: Vec<f64>,
}

impl WasmArrayObstacle {
    fn new(values: Vec<f64>) -> Result<Self, SemiflowError> {
        for &v in &values {
            if !v.is_finite() {
                return Err(SemiflowError::DomainViolation {
                    what: "obstacle_array contains NaN or Inf",
                    value: v,
                });
            }
        }
        Ok(Self { values })
    }
}

#[allow(clippy::cast_precision_loss)] // usize → f64 for error reporting only
impl Obstacle<f64> for WasmArrayObstacle {
    fn value_at(&self, _point: &[f64]) -> f64 {
        0.0
    }

    fn project_in_place(&self, dst: &mut GridFn1D<f64>) -> Result<(), SemiflowError> {
        if dst.values.len() != self.values.len() {
            return Err(SemiflowError::DomainViolation {
                what: "WasmArrayObstacle: length != grid n",
                value: self.values.len() as f64,
            });
        }
        for (v, &g) in dst.values.iter_mut().zip(self.values.iter()) {
            if *v < g {
                *v = g;
            }
        }
        Ok(())
    }

    fn active_set_into(&self, w: &GridFn1D<f64>, active: &mut [bool]) -> Result<(), SemiflowError> {
        if active.len() != w.grid.n || active.len() != self.values.len() {
            return Err(SemiflowError::DomainViolation {
                what: "WasmArrayObstacle::active_set_into: length mismatch",
                value: active.len() as f64,
            });
        }
        for (flag, (wv, gv)) in active
            .iter_mut()
            .zip(w.values.iter().zip(self.values.iter()))
        {
            *flag = *wv > *gv;
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type ConstKernel = ObstacleChernoff<DiffUnit, ConstantObstacle<f64>, f64>;
type ArrayKernel = ObstacleChernoff<DiffUnit, WasmArrayObstacle, f64>;

// ---------------------------------------------------------------------------
// ObstacleVariant — avoids Box<dyn ChernoffFunction>
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
enum ObstacleVariant {
    Const(ConstKernel),
    Array(ArrayKernel),
}

impl ObstacleVariant {
    fn apply_step(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        match self {
            Self::Const(k) => k.apply_into(tau, src, dst, scratch),
            Self::Array(k) => k.apply_into(tau, src, dst, scratch),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn extract_f64_buf(arr: &Float64Array, n: usize, label: &str) -> Result<Vec<f64>, JsValue> {
    if arr.length() as usize != n {
        return Err(make_js_error(
            "GridMismatch",
            &format!("{label} length must equal n"),
        ));
    }
    let mut buf = vec![0.0f64; n];
    arr.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error(
                "NanInf",
                &format!("{label} contains NaN or Inf"),
            ));
        }
    }
    Ok(buf)
}

fn fn_to_js_obs(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

fn validate_evolve_obs(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_obs_variant(
    xmin: f64,
    xmax: f64,
    n: usize,
    a: f64,
    level: f64,
    obs_arr: Option<Vec<f64>>,
) -> Result<(ObstacleVariant, Grid1D<f64>), JsValue> {
    let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
    let diff = DiffusionChernoff::new_const_a(a, a, grid);
    let variant = if let Some(vals) = obs_arr {
        if vals.len() != n {
            return Err(make_js_error(
                "GridMismatch",
                "obstacle_array length must equal n",
            ));
        }
        let obs = WasmArrayObstacle::new(vals).map_err(|e| err_to_js(&e))?;
        let k = ObstacleChernoff::new(diff, obs).map_err(|e| err_to_js(&e))?;
        ObstacleVariant::Array(k)
    } else {
        let obs = ConstantObstacle::new(level).map_err(|e| err_to_js(&e))?;
        let k = ObstacleChernoff::new(diff, obs).map_err(|e| err_to_js(&e))?;
        ObstacleVariant::Const(k)
    };
    Ok((variant, grid))
}

// ---------------------------------------------------------------------------
// Pure-Rust evolve helper
// ---------------------------------------------------------------------------

fn run_obs_evolve(
    kernel: &ObstacleVariant,
    src_vals: Vec<f64>,
    grid: Grid1D<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    #[allow(clippy::cast_precision_loss)]
    let tau = t / n_steps as f64;
    let mut src = GridFn1D::new(grid, src_vals)?;
    let mut dst = src.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        kernel.apply_step(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Obstacle1D — JS class
// ---------------------------------------------------------------------------

/// 1-D obstacle / variational-inequality Chernoff evolver (math §44).
///
/// Generator: ``L = a·∂_xx`` (constant diffusion, drift-free fast path).
/// JS name `Obstacle1D` matches `Killing1D` / `Reflected1D` / `Robin1D`.
///
/// ## Constructor
///
/// ```js
/// // Constant-floor obstacle (most common — option pricing):
/// new Obstacle1D(xmin, xmax, n, u0, a, level)
///
/// // Per-node floor via obstacle_array (pass null for level when using array):
/// new Obstacle1D(xmin, xmax, n, u0, a, 0.0, obstacle_array)
/// ```
///
/// Parameters:
/// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
/// - `n` — grid nodes (≥ 4).
/// - `u0` — `Float64Array` of length `n`, all finite.
/// - `a` — diffusion coefficient (> 0).
/// - `level` — constant floor `g ≡ level` (used when `obstacle_array` is `null`).
/// - `obstacle_array` — optional `Float64Array` of length `n` (per-node floor).
///   When provided, `level` is ignored and the array obstacle is used.
///
/// Strang-split (`b`, `c` ≠ 0) is not exposed; those paths match the FFI
/// fast path (`a` only). For the full surface use the Python binding.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen(js_name = "Obstacle1D")]
pub struct ObstacleChernoffWasm {
    kernel: ObstacleVariant,
    current: GridFn1D<f64>,
}

#[wasm_bindgen(js_class = "Obstacle1D")]
impl ObstacleChernoffWasm {
    /// Construct `Obstacle1D`.
    ///
    /// Pass `obstacle_array = null` (or omit it) to use the constant-floor
    /// `level` path.  Pass a `Float64Array` of length `n` to override with
    /// a per-node floor (ignores `level`).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[allow(clippy::too_many_arguments)] // mirrors Python/FFI signatures (7 params)
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        a: f64,
        level: f64,
        obstacle_array: Option<Float64Array>,
    ) -> Result<ObstacleChernoffWasm, JsValue> {
        if !a.is_finite() || a <= 0.0 {
            return Err(make_js_error("OutOfDomain", "a must be finite and > 0"));
        }
        let u0_buf = extract_f64_buf(u0, n, "u0")?;
        let obs_opt = if let Some(arr) = obstacle_array {
            let v = extract_f64_buf(&arr, n, "obstacle_array")?;
            Some(v)
        } else if level.is_finite() {
            None
        } else {
            return Err(make_js_error("OutOfDomain", "level must be finite"));
        };
        let (kernel, grid) = build_obs_variant(xmin, xmax, n, a, level, obs_opt)?;
        let current = GridFn1D::new(grid, u0_buf).map_err(|e| err_to_js(&e))?;
        Ok(ObstacleChernoffWasm { kernel, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// Returns updated `Float64Array` of length `n` (copy).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<Float64Array, JsValue> {
        validate_evolve_obs(t, n_steps)?;
        let grid = self.current.grid;
        let src = self.current.values.clone();
        let out = run_obs_evolve(&self.kernel, src, grid, t, n_steps).map_err(|e| err_to_js(&e))?;
        self.current.values.clone_from(&out);
        Ok(fn_to_js_obs(&out))
    }

    /// Return current grid values as `Float64Array` of length `n` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js_obs(&self.current.values)
    }

    /// Approximation order (always 1; §44.4 projection cap).
    #[must_use]
    pub fn order(&self) -> u32 {
        1
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}
