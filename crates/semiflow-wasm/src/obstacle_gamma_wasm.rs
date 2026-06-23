//! WASM JS classes for `ObstacleGammaV8` and `ObstacleND2` (C-parity pass,
//! `full` feature, ADR-0028/0171, ADR-0153 TIER-2).
//!
//! Mirrors `semiflow-py` `PyObstacleGammaV8` + `PyObstacleNDV8`
//! (`obstacle_gamma_py.rs`).
//!
//! ## JS classes
//!
//! ### `ObstacleGammaV8`
//!
//! ```js
//! import init, { ObstacleGammaV8 } from "@semiflow/wasm";
//! await init();
//! // Constant-floor variant:
//! const og = ObstacleGammaV8.newConst(0.0, 2.0, 64, 0.5);
//! // Per-node array variant:
//! const arr = new Float64Array(64).fill(0.5);
//! const og2 = ObstacleGammaV8.newArray(0.0, 2.0, 64, arr);
//!
//! const v = new Float64Array(64);
//! // ... fill v with value-function data ...
//! const { gamma, defined, count } = og.inactiveGamma(v);
//! console.log(count);  // nodes where Γ is defined
//! ```
//!
//! ### `ObstacleND2`
//!
//! ```js
//! import init, { ObstacleND2 } from "@semiflow/wasm";
//! await init();
//! const nd = new ObstacleND2(0.0, 1.0, 8, 0.0, 1.0, 8, 0.0);
//! const [nx, ny] = nd.shape();   // [8, 8]
//! const v = new Float64Array(64);
//! const out = nd.apply(0.01, v); // Float64Array length 64
//! ```
//!
//! ## Error model
//!
//! `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1); no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::{Float64Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

use semiflow::{
    grid_nd::{GridFnND, GridND},
    shift_nd::AnisotropicShiftChernoffND,
    ChernoffFunction, ConstantObstacle, DiffusionChernoff, Grid1D, GridFn1D,
    Obstacle, ObstacleChernoff, ObstacleChernoffND, ScratchPool, SemiflowError, SquareMatrix,
};

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// WasmGammaArrayObstacle — per-node floor (mirrors FFI / PyO3 counterparts)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct WasmGammaArrayObstacle {
    values: Vec<f64>,
}

impl WasmGammaArrayObstacle {
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

#[allow(clippy::cast_precision_loss)]
impl Obstacle<f64> for WasmGammaArrayObstacle {
    fn value_at(&self, _point: &[f64]) -> f64 {
        0.0
    }

    fn project_in_place(&self, dst: &mut GridFn1D<f64>) -> Result<(), SemiflowError> {
        if dst.values.len() != self.values.len() {
            return Err(SemiflowError::DomainViolation {
                what: "WasmGammaArrayObstacle: length != grid n",
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
                what: "WasmGammaArrayObstacle::active_set_into: length mismatch",
                value: active.len() as f64,
            });
        }
        for (flag, (wv, gv)) in active.iter_mut().zip(w.values.iter().zip(self.values.iter())) {
            *flag = *wv > *gv;
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// Type aliases + GammaVariant
// ---------------------------------------------------------------------------

type GammaConst = ObstacleChernoff<DiffusionChernoff<f64>, ConstantObstacle<f64>, f64>;
type GammaArray = ObstacleChernoff<DiffusionChernoff<f64>, WasmGammaArrayObstacle, f64>;

#[allow(clippy::large_enum_variant)]
enum GammaVariant {
    Const(GammaConst),
    Array(GammaArray),
}

impl GammaVariant {
    fn apply_gamma(
        &self,
        v: &GridFn1D<f64>,
        gamma: &mut GridFn1D<f64>,
        defined: &mut [bool],
    ) -> Result<usize, SemiflowError> {
        match self {
            Self::Const(k) => k.apply_inactive_gamma_into(v, gamma, defined),
            Self::Array(k) => k.apply_inactive_gamma_into(v, gamma, defined),
        }
    }
}

// ---------------------------------------------------------------------------
// ObstacleGammaV8 JS class
// ---------------------------------------------------------------------------

/// Inactive-set Γ = V″ primitive (v8.3.0, ADR-0153 §4.1).
///
/// **Honesty (NORMATIVE)**: `defined[i] == 0` means Γ is REFUSED at node `i`;
/// NOT "Γ = 0". Callers MUST check `defined[i]` before reading `gamma[i]`.
///
/// # Errors
/// Throws `.kind`-tagged JS `Error`.
#[wasm_bindgen(js_name = "ObstacleGammaV8")]
pub struct ObstacleGammaV8Wasm {
    kernel: GammaVariant,
    grid: Grid1D<f64>,
}

#[wasm_bindgen(js_class = "ObstacleGammaV8")]
impl ObstacleGammaV8Wasm {
    /// Construct with a constant obstacle floor.
    ///
    /// Parameters: `domainLo`, `domainHi`, `nGrid` (>= 4), `level` (finite).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(js_name = "newConst")]
    pub fn new_const(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        level: f64,
    ) -> Result<ObstacleGammaV8Wasm, JsValue> {
        validate_gamma_domain(domain_lo, domain_hi, n_grid)?;
        if !level.is_finite() {
            return Err(make_js_error("NanInf", "level must be finite"));
        }
        let grid = Grid1D::new(domain_lo, domain_hi, n_grid).map_err(|e| err_to_js(&e))?;
        let diff =
            DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0_f64, grid);
        let obs = ConstantObstacle::new(level).map_err(|e| err_to_js(&e))?;
        let kernel: GammaConst = ObstacleChernoff::new(diff, obs).map_err(|e| err_to_js(&e))?;
        Ok(ObstacleGammaV8Wasm { kernel: GammaVariant::Const(kernel), grid })
    }

    /// Construct with a per-node array obstacle floor.
    ///
    /// Parameters: `domainLo`, `domainHi`, `nGrid` (>= 4),
    /// `obstacleArray` (`Float64Array` of length `nGrid`, all finite).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(js_name = "newArray")]
    pub fn new_array(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        obstacle_array: &Float64Array,
    ) -> Result<ObstacleGammaV8Wasm, JsValue> {
        validate_gamma_domain(domain_lo, domain_hi, n_grid)?;
        if obstacle_array.length() as usize != n_grid {
            return Err(make_js_error(
                "GridMismatch",
                "obstacle_array length must equal n_grid",
            ));
        }
        let mut obs_buf = vec![0.0f64; n_grid];
        obstacle_array.copy_to(&mut obs_buf);
        let grid = Grid1D::new(domain_lo, domain_hi, n_grid).map_err(|e| err_to_js(&e))?;
        let diff =
            DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0_f64, grid);
        let obs = WasmGammaArrayObstacle::new(obs_buf).map_err(|e| err_to_js(&e))?;
        let kernel: GammaArray = ObstacleChernoff::new(diff, obs).map_err(|e| err_to_js(&e))?;
        Ok(ObstacleGammaV8Wasm { kernel: GammaVariant::Array(kernel), grid })
    }

    /// Number of grid nodes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.grid.n
    }

    /// Compute inactive-set Γ = V″ on the OPEN continuation set.
    ///
    /// Returns a JS object `{ gamma: Float64Array, defined: Uint8Array, count: number }`.
    /// - `gamma[i]` — central-difference V″ where `defined[i] == 1`.
    /// - `defined[i]` — `1` = Γ defined; `0` = REFUSED (do NOT read `gamma[i]`).
    /// - `count` — number of nodes where Γ is defined.
    ///
    /// # Errors
    /// Throws `.kind`-tagged JS `Error` if `v.length != nGrid`.
    #[wasm_bindgen(js_name = "inactiveGamma")]
    pub fn inactive_gamma(&self, v: &Float64Array) -> Result<JsValue, JsValue> {
        let n = self.grid.n;
        if v.length() as usize != n {
            return Err(make_js_error("GridMismatch", "v length does not match nGrid"));
        }
        let mut v_buf = vec![0.0f64; n];
        v.copy_to(&mut v_buf);
        let v_fn = GridFn1D::new(self.grid, v_buf).map_err(|e| err_to_js(&e))?;
        let mut gamma_fn = v_fn.zeroed_like();
        let mut defined_bool = vec![false; n];
        let count = self
            .kernel
            .apply_gamma(&v_fn, &mut gamma_fn, &mut defined_bool)
            .map_err(|e| err_to_js(&e))?;
        // Build gamma Float64Array.
        #[allow(clippy::cast_possible_truncation)]
        let gamma_arr = Float64Array::new_with_length(n as u32);
        gamma_arr.copy_from(&gamma_fn.values);
        // Build defined Uint8Array.
        let defined_u8: Vec<u8> = defined_bool.iter().map(|&b| b as u8).collect();
        #[allow(clippy::cast_possible_truncation)]
        let defined_arr = Uint8Array::new_with_length(n as u32);
        defined_arr.copy_from(&defined_u8);
        // Return { gamma, defined, count }.
        let obj = Object::new();
        Reflect::set(&obj, &"gamma".into(), &gamma_arr.into()).map_err(|e| e)?;
        Reflect::set(&obj, &"defined".into(), &defined_arr.into()).map_err(|e| e)?;
        #[allow(clippy::cast_precision_loss)]
        Reflect::set(&obj, &"count".into(), &(count as f64).into()).map_err(|e| e)?;
        Ok(obj.into())
    }
}

// ---------------------------------------------------------------------------
// ObstacleND2 JS class (D=2)
// ---------------------------------------------------------------------------

type Nd2Kernel =
    ObstacleChernoffND<AnisotropicShiftChernoffND<f64, 2>, ConstantObstacle<f64>, f64, 2>;

/// D=2 projective-splitting obstacle evolver (v8.3.0, ADR-0153 §4.2).
///
/// Layout: flat axis-0-fastest (`idx(i,j) = i + j*nx`).
/// Pass `v.flat(order="F")` (or `F`-order ravel) from Python/numpy; use
/// `Float64Array` directly in JS.
///
/// # Errors
/// Throws `.kind`-tagged JS `Error`.
#[wasm_bindgen(js_name = "ObstacleND2")]
pub struct ObstacleND2Wasm {
    level: f64,
    grid_nd: GridND<f64, 2>,
    nx: usize,
    ny: usize,
}

#[wasm_bindgen(js_class = "ObstacleND2")]
impl ObstacleND2Wasm {
    /// Construct a D=2 obstacle evolver.
    ///
    /// Parameters: `xmin`, `xmax`, `nx`, `ymin`, `ymax`, `ny`, `level`.
    /// Requires `nx * ny >= 25`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        level: f64,
    ) -> Result<ObstacleND2Wasm, JsValue> {
        validate_nd2_domain(xmin, xmax, ymin, ymax, level)?;
        let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
        let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
        let grid_nd = GridND::<f64, 2>::new([gx, gy]).map_err(|e| err_to_js(&e))?;
        Ok(ObstacleND2Wasm { level, grid_nd, nx, ny })
    }

    /// Return `[nx, ny]` as a two-element array.
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        vec![self.nx, self.ny]
    }

    /// Apply one Chernoff step `Π_g ∘ S(Δτ)`.
    ///
    /// `v` must be `Float64Array` of length `nx*ny` (axis-0-fastest).
    /// Returns `Float64Array` of the same length.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` if `v.length != nx*ny` or `tau <= 0`.
    pub fn apply(&self, tau: f64, v: &Float64Array) -> Result<Float64Array, JsValue> {
        let n = self.nx * self.ny;
        if v.length() as usize != n {
            return Err(make_js_error("GridMismatch", "v length must equal nx*ny"));
        }
        if !tau.is_finite() || tau <= 0.0 {
            return Err(make_js_error("OutOfDomain", "tau must be finite and > 0"));
        }
        let mut v_buf = vec![0.0f64; n];
        v.copy_to(&mut v_buf);
        let result = run_nd2_step(self.grid_nd.clone(), &v_buf, self.level, tau)
            .map_err(|e| err_to_js(&e))?;
        #[allow(clippy::cast_possible_truncation)]
        let out = Float64Array::new_with_length(n as u32);
        out.copy_from(&result);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust step helper (mirrors run_nd_step_2d in obstacle_gamma_py.rs)
// ---------------------------------------------------------------------------

fn run_nd2_step(
    grid_nd: GridND<f64, 2>,
    v_vals: &[f64],
    level: f64,
    tau: f64,
) -> Result<Vec<f64>, SemiflowError> {
    let src = GridFnND::new(grid_nd.clone(), v_vals.to_vec())?;
    let mut dst = GridFnND::new(grid_nd.clone(), vec![0.0_f64; v_vals.len()])?;
    let inner = AnisotropicShiftChernoffND::<f64, 2>::new(
        |_x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            a.set(0, 1, 0.0);
            a.set(1, 0, 0.0);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x: &[f64; 2]| 0.0_f64,
        grid_nd.clone(),
    )?;
    let obs = ConstantObstacle::new(level)?;
    let kernel: Nd2Kernel = ObstacleChernoffND::new(inner, obs)?;
    let mut scratch = ScratchPool::new();
    kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
    Ok(dst.values)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_gamma_domain(lo: f64, hi: f64, n: usize) -> Result<(), JsValue> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(make_js_error("NanInf", "domain bounds must be finite"));
    }
    if lo >= hi {
        return Err(make_js_error("OutOfDomain", "domainLo must be < domainHi"));
    }
    if n < 4 {
        return Err(make_js_error("OutOfDomain", "nGrid must be >= 4"));
    }
    Ok(())
}

fn validate_nd2_domain(xmin: f64, xmax: f64, ymin: f64, ymax: f64, level: f64) -> Result<(), JsValue> {
    if !xmin.is_finite() || !xmax.is_finite() || !ymin.is_finite() || !ymax.is_finite() {
        return Err(make_js_error("NanInf", "domain bounds must be finite"));
    }
    if xmin >= xmax {
        return Err(make_js_error("OutOfDomain", "xmin must be < xmax"));
    }
    if ymin >= ymax {
        return Err(make_js_error("OutOfDomain", "ymin must be < ymax"));
    }
    if !level.is_finite() {
        return Err(make_js_error("NanInf", "level must be finite"));
    }
    Ok(())
}
