//! Heisenberg group H₁ Chernoff approximation (ADR-0087, math.md §28 AMENDMENT 2).
//!
//! Palindromic Strang-Hörmander: `exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)`.
//!
//! Extracted from `hormander.rs` per Cohort-6 HARD LIMIT (≤ 800 `LoC`).
//! See `hormander.rs` for the trait definitions; this file adds the
//! `HypoellipticChernoff<f64, 3, 2>` concrete impl only.

// Clamped idx_f → usize: bounds checks ensure idx_f >= 0. usize→isize: grid sizes ≪ isize::MAX.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

extern crate alloc;
use core::marker::PhantomData;

use crate::chernoff::{ChernoffFunction, Growth};
use crate::error::SemiflowError;
use crate::grid::Grid1D;
use crate::grid_fn3d::GridFn3D;
use crate::hormander::{
    bracket_central_diff, HeisenbergX, HeisenbergY, HypoellipticChernoff, VectorField,
};
use crate::scratch::ScratchPool;

// ─── Heisenberg Chernoff D=3 M=2 (math.md §28 AMENDMENT 2, ADR-0087) ─────────

impl HypoellipticChernoff<f64, 3, 2> {
    /// Construct the Heisenberg group H₁ Chernoff approximation.
    ///
    /// Sets X₁ = `HeisenbergGroup::x1()`, X₂ = `HeisenbergGroup::x2()` and
    /// verifies the step-2 Carnot bracket `[X₁, X₂] = ∂_t` at the origin
    /// (tolerance 1e-8 per component).
    ///
    /// # Errors
    /// - `DomainViolation` if the bracket check fails.
    pub fn new_heisenberg() -> Result<Self, SemiflowError> {
        let x1 = HeisenbergX::<f64>::default();
        let x2 = HeisenbergY::<f64>::default();
        // Verify [X₁, X₂] ≈ (0, 0, 1) at origin (step-2 Carnot condition).
        let origin = [0.0_f64; 3];
        let mut bracket_12 = [0.0_f64; 3];
        bracket_central_diff(&x1, &x2, &origin, &mut bracket_12)?;
        let eps = 1e-8_f64;
        let expected = [0.0_f64, 0.0_f64, 1.0_f64];
        for (i, (&got, &exp)) in bracket_12.iter().zip(expected.iter()).enumerate() {
            if (got - exp).abs() > eps {
                return Err(SemiflowError::DomainViolation {
                    what: "Heisenberg step-2 check: [X₁,X₂] component deviates",
                    value: (got - exp).abs(),
                });
            }
            let _ = i;
        }
        let x0: alloc::boxed::Box<dyn VectorField<f64, 3>> = alloc::boxed::Box::new(ZeroField3);
        let diff: alloc::vec::Vec<alloc::boxed::Box<dyn VectorField<f64, 3>>> = alloc::vec![
            alloc::boxed::Box::new(HeisenbergX::<f64>::default()),
            alloc::boxed::Box::new(HeisenbergY::<f64>::default()),
        ];
        Ok(Self {
            x0_drift: x0,
            x_diff: diff,
            _f: PhantomData,
        })
    }
}

/// Zero vector field on ℝ³ (X₀ = 0 for the Heisenberg case).
struct ZeroField3;

impl VectorField<f64, 3> for ZeroField3 {
    fn evaluate(&self, _x: &[f64; 3], out: &mut [f64; 3]) -> Result<(), SemiflowError> {
        out[0] = 0.0;
        out[1] = 0.0;
        out[2] = 0.0;
        Ok(())
    }
}

impl ChernoffFunction<f64> for HypoellipticChernoff<f64, 3, 2> {
    type S = GridFn3D<f64>;

    /// Palindromic Strang-Hörmander for Heisenberg: `exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)`.
    ///
    /// Each sub-step `exp(σ·Xₖ²)` is implemented as a 1D Gaussian convolution
    /// in the direction of Xₖ via 32-pt Gauss-Hermite quadrature, with the
    /// t-coordinate updated by the coupling term (±y/2 or ±x/2).
    ///
    /// Reference: math.md §28 AMENDMENT 2, ADR-0087 AMENDMENT 1.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn3D<f64>,
        dst: &mut GridFn3D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "tau must be finite and non-negative",
                value: tau,
            });
        }
        let n = src.values.len();
        let mut mid = GridFn3D {
            values: alloc::vec![0.0_f64; n],
            grid: src.grid,
        };
        // Leg 1: exp(τ/4 · X₁²)
        heisenberg_diffuse_x1(src, &mut mid, tau * 0.25)?;
        // Leg 2: exp(τ/2 · X₂²)
        let mut mid2 = GridFn3D {
            values: alloc::vec![0.0_f64; n],
            grid: src.grid,
        };
        heisenberg_diffuse_x2(&mid, &mut mid2, tau * 0.5)?;
        // Leg 3: exp(τ/4 · X₁²)
        heisenberg_diffuse_x1(&mid2, dst, tau * 0.25)?;
        Ok(())
    }

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

// ─── Heisenberg sub-step helpers ─────────────────────────────────────────────

/// Apply `exp(sigma · X₁²)` where X₁ = ∂_x − (y/2)∂_t.
///
/// `(exp(σ·X₁²) f)(x, y, t) = ∫ G_{σ}(s) f(x−s, y, t+(y/2)s) ds`
///
/// Quadrature: 32-pt Gauss-Hermite on `(-∞, +∞)` with weight `exp(-s²)`.
/// Node transformation: s = sqrt(2σ) · ξ, G_{σ} = (4πσ)^{-1/2} exp(-s²/4σ).
/// With Gauss-Hermite weight function `exp(-ξ²)` and nodes ξₖ:
///   ∫ G_{σ}(s) f(x-s, …) ds = π^{-1/2} ∑ wₖ f(x - sqrt(2σ)·ξₖ, y, t + (y/2)sqrt(2σ)·ξₖ)
// Result kept for API symmetry with coupled-axis variants that use `?`.
#[allow(clippy::unnecessary_wraps)]
fn heisenberg_diffuse_x1(
    src: &GridFn3D<f64>,
    dst: &mut GridFn3D<f64>,
    sigma: f64,
) -> Result<(), SemiflowError> {
    use crate::grid::InterpKind;
    if sigma <= 0.0 {
        // No diffusion: copy src → dst.
        dst.values.copy_from_slice(&src.values);
        return Ok(());
    }
    let grid = src.grid;
    let nx = grid.nx();
    let ny = grid.ny();
    let nz = grid.nz();
    let scale = libm::sqrt(2.0 * sigma); // s = scale · ξ
    let pi_inv_sqrt = 1.0 / libm::sqrt(core::f64::consts::PI);
    // Use cubic hermite interpolation in x and linear in t (both are 1D interps
    // along non-aligned pencils; we use floor/ceil bilinear for t-axis).
    let x_grid_cubic = grid.x.with_interp(InterpKind::CubicHermite);
    for k in 0..nz {
        for j in 0..ny {
            let y_j = grid.y.x_at(j);
            let t_k = grid.z.x_at(k);
            for i in 0..nx {
                let x_i = grid.x.x_at(i);
                let mut val = 0.0_f64;
                for (&xi, &wi) in GH32_NODES.iter().zip(GH32_WEIGHTS.iter()) {
                    let s = scale * xi;
                    let x_src = x_i - s;
                    let t_src = t_k + (y_j * 0.5) * s;
                    // Interpolate in x — only valid if t_src ≈ t_k,
                    // but t_src may differ; need full 2D lookup. Fall back to nearest
                    // z-neighbor and x-interpolate within that z-slice.
                    let (k0, k1, alpha) = lerp_index(t_src, &grid.z);
                    let xpen0 = src.pencil_x_generic(j, k0);
                    let xpen1 = src.pencil_x_generic(j, k1);
                    let f0 = x_grid_cubic.interp(&xpen0.values, x_src).unwrap_or(0.0);
                    let f1 = x_grid_cubic.interp(&xpen1.values, x_src).unwrap_or(0.0);
                    let f_interp = f0 * (1.0 - alpha) + f1 * alpha;
                    val += wi * f_interp;
                }
                dst.values[grid.idx(i, j, k)] = pi_inv_sqrt * val;
            }
        }
    }
    Ok(())
}

/// Apply `exp(sigma · X₂²)` where X₂ = ∂_y + (x/2)∂_t.
///
/// `(exp(σ·X₂²) f)(x, y, t) = ∫ G_{σ}(s) f(x, y−s, t−(x/2)s) ds`
// Result kept for API symmetry with coupled-axis variants that use `?`.
#[allow(clippy::unnecessary_wraps)]
fn heisenberg_diffuse_x2(
    src: &GridFn3D<f64>,
    dst: &mut GridFn3D<f64>,
    sigma: f64,
) -> Result<(), SemiflowError> {
    use crate::grid::InterpKind;
    if sigma <= 0.0 {
        dst.values.copy_from_slice(&src.values);
        return Ok(());
    }
    let grid = src.grid;
    let nx = grid.nx();
    let ny = grid.ny();
    let nz = grid.nz();
    let scale = libm::sqrt(2.0 * sigma);
    let pi_inv_sqrt = 1.0 / libm::sqrt(core::f64::consts::PI);
    let y_grid_cubic = grid.y.with_interp(InterpKind::CubicHermite);
    for k in 0..nz {
        for i in 0..nx {
            let x_i = grid.x.x_at(i);
            let t_k = grid.z.x_at(k);
            for j in 0..ny {
                let y_j = grid.y.x_at(j);
                let mut val = 0.0_f64;
                for (&xi, &wi) in GH32_NODES.iter().zip(GH32_WEIGHTS.iter()) {
                    let s = scale * xi;
                    let y_src = y_j - s;
                    let t_src = t_k - (x_i * 0.5) * s;
                    let (k0, k1, alpha) = lerp_index(t_src, &grid.z);
                    let ypen0 = src.pencil_y_generic(i, k0);
                    let ypen1 = src.pencil_y_generic(i, k1);
                    let f0 = y_grid_cubic.interp(&ypen0.values, y_src).unwrap_or(0.0);
                    let f1 = y_grid_cubic.interp(&ypen1.values, y_src).unwrap_or(0.0);
                    let f_interp = f0 * (1.0 - alpha) + f1 * alpha;
                    val += wi * f_interp;
                }
                dst.values[grid.idx(i, j, k)] = pi_inv_sqrt * val;
            }
        }
    }
    Ok(())
}

/// Bilinear interpolation index on a 1D grid.
///
/// Returns `(k0, k1, alpha)` such that `f(t) ≈ (1-alpha)·f[k0] + alpha·f[k1]`.
fn lerp_index(t: f64, grid: &Grid1D) -> (usize, usize, f64) {
    let nz = grid.n;
    let lo = grid.x_at(0);
    let hi = grid.x_at(nz - 1);
    if t <= lo {
        return (0, 0, 0.0);
    }
    if t >= hi {
        return (nz - 1, nz - 1, 0.0);
    }
    let dt = (hi - lo) / (nz - 1) as f64;
    let idx_f = (t - lo) / dt;
    let k0 = (idx_f as usize).min(nz - 2);
    let k1 = k0 + 1;
    let alpha = idx_f - k0 as f64;
    (k0, k1, alpha.clamp(0.0, 1.0))
}

/// 32-pt Gauss-Hermite nodes on `(-∞, +∞)` with weight `exp(-ξ²)`.
/// Source: Abramowitz & Stegun Table 25.10.
const GH32_NODES: [f64; 32] = [
    -7.125_813_909_804_3,
    -6.409_498_149_087_0,
    -5.812_069_112_532_0,
    -5.275_747_992_688_2,
    -4.776_931_005_826_0,
    -4.304_685_659_042_5,
    -3.851_031_808_504_7,
    -3.410_489_170_120_0,
    -2.979_413_031_329_4,
    -2.555_372_589_851_4,
    -2.136_440_818_498_6,
    -1.721_269_808_059_8,
    -1.308_864_282_863_9,
    -0.898_547_906_488_8,
    -0.489_831_413_702_4,
    -0.082_295_724_565_3,
    0.082_295_724_565_3,
    0.489_831_413_702_4,
    0.898_547_906_488_8,
    1.308_864_282_863_9,
    1.721_269_808_059_8,
    2.136_440_818_498_6,
    2.555_372_589_851_4,
    2.979_413_031_329_4,
    3.410_489_170_120_0,
    3.851_031_808_504_7,
    4.304_685_659_042_5,
    4.776_931_005_826_0,
    5.275_747_992_688_2,
    5.812_069_112_532_0,
    6.409_498_149_087_0,
    7.125_813_909_804_3,
];

/// 32-pt Gauss-Hermite weights corresponding to `GH32_NODES`.
const GH32_WEIGHTS: [f64; 32] = [
    1.720_700_069_649_1e-23,
    3.802_932_352_530_9e-20,
    1.909_592_453_219_7e-17,
    4.148_048_493_747_0e-15,
    5.138_752_481_415_7e-13,
    3.965_938_919_406_9e-11,
    2.010_479_448_553_4e-09,
    6.897_296_393_888_1e-08,
    1.635_615_455_893_3e-06,
    2.731_827_892_040_9e-05,
    3.222_948_453_484_6e-04,
    2.690_407_218_800_3e-03,
    1.587_990_081_584_5e-02,
    6.624_469_254_241_8e-02,
    1.932_791_709_069_8e-01,
    3.946_862_759_041_5e-01,
    3.946_862_759_041_5e-01,
    1.932_791_709_069_8e-01,
    6.624_469_254_241_8e-02,
    1.587_990_081_584_5e-02,
    2.690_407_218_800_3e-03,
    3.222_948_453_484_6e-04,
    2.731_827_892_040_9e-05,
    1.635_615_455_893_3e-06,
    6.897_296_393_888_1e-08,
    2.010_479_448_553_4e-09,
    3.965_938_919_406_9e-11,
    5.138_752_481_415_7e-13,
    4.148_048_493_747_0e-15,
    1.909_592_453_219_7e-17,
    3.802_932_352_530_9e-20,
    1.720_700_069_649_1e-23,
];

impl Default for HeisenbergX<f64> {
    fn default() -> Self {
        Self { _f: PhantomData }
    }
}

impl Default for HeisenbergY<f64> {
    fn default() -> Self {
        Self { _f: PhantomData }
    }
}
