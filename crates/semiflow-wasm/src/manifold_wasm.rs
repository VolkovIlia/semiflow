//! Riemannian manifold Chernoff — WebAssembly binding (`full` feature).
//!
//! | JS class     | Core type                      | Python mirror |
//! |--------------|--------------------------------|---------------|
//! | `Manifold2D` | `ManifoldChernoff<M, f64>`     | `Manifold2D`  |
//!
//! ## Backend tag (u32)
//!
//! Consistent with FFI Round-7 ordering:
//! - `0` = `Torus` — flat 2-torus T² (R ≡ 0, Python default).
//! - `1` = `Sphere2` — 2-sphere S²(r).
//! - `2` = `Hyperbolic2` — Poincaré disk H²(s).
//!
//! ## Buffer layout
//!
//! Flat row-major `Float64Array` of length `nx * ny`; `values[j*nx + i] ≈ u(x_i, y_j)`.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow_core::{
    manifold::{Hyperbolic2, Sphere2, Torus},
    manifold_chernoff::ManifoldChernoff,
    ChernoffFunction, ChernoffSemigroup, Grid1D, Grid2D, GridFn2D, ScratchPool,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// ManifoldEnum — binding-side dispatch (mirrors semiflow-py ManifoldEnum)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum ManifoldEnum {
    Sphere(ManifoldChernoff<Sphere2<f64>, f64>),
    Hyperbolic(ManifoldChernoff<Hyperbolic2<f64>, f64>),
    Torus(ManifoldChernoff<Torus<f64, 2>, f64>),
}

// Safety: all three inner types are Send+Sync (contains only f64 scalars
// and PhantomData; verified in semiflow-py send_assertions.rs).
unsafe impl Send for ManifoldEnum {}
unsafe impl Sync for ManifoldEnum {}

impl ChernoffFunction<f64> for ManifoldEnum {
    type S = GridFn2D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn2D<f64>,
        dst: &mut GridFn2D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow_core::SemiflowError> {
        match self {
            ManifoldEnum::Sphere(k) => k.apply_into(tau, src, dst, scratch),
            ManifoldEnum::Hyperbolic(k) => k.apply_into(tau, src, dst, scratch),
            ManifoldEnum::Torus(k) => k.apply_into(tau, src, dst, scratch),
        }
    }

    fn order(&self) -> u32 {
        match self {
            ManifoldEnum::Sphere(k) => k.order(),
            ManifoldEnum::Hyperbolic(k) => k.order(),
            ManifoldEnum::Torus(k) => k.order(),
        }
    }

    fn growth(&self) -> semiflow_core::chernoff::Growth<f64> {
        match self {
            ManifoldEnum::Sphere(k) => k.growth(),
            ManifoldEnum::Hyperbolic(k) => k.growth(),
            ManifoldEnum::Torus(k) => k.growth(),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend selection (u32 tag: 0=Torus, 1=Sphere2, 2=Hyperbolic2)
// ---------------------------------------------------------------------------

fn parse_manifold_tag(
    tag: u32,
    radius: f64,
    curvature_correction: bool,
) -> Result<ManifoldEnum, JsValue> {
    match tag {
        0 => {
            let m = Torus::<f64, 2>::unit();
            Ok(ManifoldEnum::Torus(ManifoldChernoff::new(m, curvature_correction)))
        }
        1 => {
            let m = Sphere2::with_radius(radius).map_err(|e| err_to_js(&e))?;
            Ok(ManifoldEnum::Sphere(ManifoldChernoff::new(m, curvature_correction)))
        }
        2 => {
            let m = Hyperbolic2::with_scale(radius).map_err(|e| err_to_js(&e))?;
            Ok(ManifoldEnum::Hyperbolic(ManifoldChernoff::new(m, curvature_correction)))
        }
        _ => Err(make_js_error(
            "Unsupported",
            "manifold_tag must be 0 (torus), 1 (sphere2), or 2 (hyperbolic2)",
        )),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn extract_flat(u0: &Float64Array, expected: usize) -> Result<Vec<f64>, JsValue> {
    if u0.length() as usize != expected {
        return Err(make_js_error("GridMismatch", "u0.length() must equal nx * ny"));
    }
    let mut buf = vec![0.0f64; expected];
    u0.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(buf)
}

fn validate_evolve_mfd(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn vec_to_js_mfd(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn build_manifold_wasm(
    x0min: f64, x0max: f64, nx: usize,
    x1min: f64, x1max: f64, ny: usize,
    u0: &[f64],
    kernel: ManifoldEnum,
) -> Result<(ManifoldEnum, GridFn2D<f64>), semiflow_core::SemiflowError> {
    if u0.len() != nx * ny {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "u0 length must equal nx * ny",
            value: u0.len() as f64,
        });
    }
    let gx = Grid1D::new(x0min, x0max, nx)?;
    let gy = Grid1D::new(x1min, x1max, ny)?;
    let grid = Grid2D::new(gx, gy);
    let current = GridFn2D::new_generic(grid, u0.to_vec())?;
    Ok((kernel, current))
}

// ---------------------------------------------------------------------------
// Manifold2D — JS class
// ---------------------------------------------------------------------------

/// 2-D Riemannian manifold Chernoff approximation (M13).
///
/// Backed by `ManifoldChernoff<M, f64>` (MMRS 2023, math §24, ADR-0071).
/// Backend selected by `manifold_tag`:
/// - `0` = torus (T², flat, R ≡ 0; Python default).
/// - `1` = sphere2 (S²(r)); `radius` sets sphere radius.
/// - `2` = hyperbolic2 (H²(s)); `radius` sets scale.
///
/// Layout: flat row-major `Float64Array` length `nx * ny`;
/// `values[j*nx + i] ≈ u(x0_i, x1_j)`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Manifold2D {
    kernel: ManifoldEnum,
    current: GridFn2D<f64>,
    nx: usize,
    ny: usize,
}

#[wasm_bindgen]
impl Manifold2D {
    /// Construct `Manifold2D`.
    ///
    /// - `x0min`, `x0max`, `nx` — axis-0 grid (nx ≥ 4).
    /// - `x1min`, `x1max`, `ny` — axis-1 grid (ny ≥ 4).
    /// - `u0` — `Float64Array` of length `nx * ny`.
    /// - `manifold_tag` — `0`=torus, `1`=sphere2, `2`=hyperbolic2.
    /// - `radius` — sphere/hyperbolic scale parameter (> 0, default 1.0).
    /// - `curvature_correction` — apply R/12 correction (default true).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x0min: f64, x0max: f64, nx: usize,
        x1min: f64, x1max: f64, ny: usize,
        u0: &Float64Array,
        manifold_tag: u32,
        radius: f64,
        curvature_correction: bool,
    ) -> Result<Manifold2D, JsValue> {
        let buf = extract_flat(u0, nx * ny)?;
        let kernel = parse_manifold_tag(manifold_tag, radius, curvature_correction)?;
        let (kernel, current) =
            build_manifold_wasm(x0min, x0max, nx, x1min, x1max, ny, &buf, kernel)
                .map_err(|e| err_to_js(&e))?;
        Ok(Manifold2D { kernel, current, nx, ny })
    }

    /// Advance state by time `t` using `n_steps` Chernoff steps.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve_mfd(t, n_steps)?;
        let sg = ChernoffSemigroup::new(self.kernel.clone(), n_steps).map_err(|e| err_to_js(&e))?;
        let next = sg.evolve(t, &self.current).map_err(|e| err_to_js(&e))?;
        self.current = next;
        Ok(())
    }

    /// Return current chart values as `Float64Array` of length `nx * ny` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        vec_to_js_mfd(&self.current.values)
    }

    /// Approximation order (2 with `curvature_correction` for sphere2/hyperbolic2; else 1).
    #[must_use]
    pub fn order(&self) -> u32 {
        self.kernel.order()
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize { self.nx }

    /// Y-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> usize { self.ny }

    /// Total grid nodes (`nx * ny`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize { self.nx * self.ny }
}
