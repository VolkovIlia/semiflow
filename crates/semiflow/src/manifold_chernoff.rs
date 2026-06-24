//! `ManifoldChernoff<M, F>` — Riemannian Chernoff wrapper (math.md §24, ADR-0071).
//!
//! Implements `ChernoffFunction<f64>` for D=2 manifolds via Gauss-Hermite
//! quadrature on the tangent space `T_xM`. Shipped backends:
//! `Sphere2<f64>`, `Hyperbolic2<f64>`, `Torus<f64, 2>` (v2.8),
//! `FubiniStudyCp1<f64>` (v7.0.0, ADR-0129).
//!
//! State type `S = GridFn2D<f64>` — a function sampled on a 2D chart grid.
//!
//! References: Mazzucchi-Moretti-Remizov-Smolyanov 2023 *Math. Nachr.* Thm 1.

use core::marker::PhantomData;

#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::{
    chernoff::Growth,
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn2d::GridFn2D,
    manifold::{BoundedGeometryManifold, Hyperbolic2, Sphere2, Torus},
    manifold_kahler::FubiniStudyCp1,
    scratch::ScratchPool,
    ChernoffFunction,
};

// ─── Gauss-Hermite 5-pt quadrature (∫ e^{-x²} f(x) dx) ──────────────────────
//
// Nodes & weights from Abramowitz & Stegun Table 25.10 (GH-5, physicists).
// Normalisation: Σ w_k = √π.

const GH5_NODES: [f64; 5] = [
    -2.020_182_870_456_086,
    -0.958_572_464_613_819,
    0.0,
    0.958_572_464_613_819,
    2.020_182_870_456_086,
];
const GH5_WEIGHTS: [f64; 5] = [
    0.019_953_242_059_046,
    0.393_619_323_152_241,
    0.945_308_720_482_942,
    0.393_619_323_152_241,
    0.019_953_242_059_046,
];

// ─── ManifoldChernoff<M, F> ───────────────────────────────────────────────────

/// Chernoff function on a D=2 Riemannian manifold (math.md §24, ADR-0071).
///
/// Applies a Gaussian kernel on the tangent space `T_xM` per chart node x,
/// pushed forward to the manifold via `exp_x`. With `with_curvature_correction`
/// the output is multiplied by `[1 + (τ/12)·R(x)]` (MMRS 2023 Theorem 1),
/// lifting the order from 1 to 2.
///
/// **State type**: `GridFn2D<f64>` — a function sampled on a 2D chart grid.
/// The chart axes are backend-specific: for `Sphere2`, the grid spans
/// (θ, φ); for `Torus<f64,2>` and `Hyperbolic2` their respective charts.
///
/// **v2.8 scope**: D=2 manifolds only. D=1 (`Torus<f64,1>`) and D≥3
/// deferred to v2.9+ pending a generic `ManifoldStateMap` associated type.
#[derive(Debug, Clone)]
pub struct ManifoldChernoff<M, F = f64>
where
    M: BoundedGeometryManifold<F>,
    F: SemiflowFloat,
{
    manifold: M,
    with_curvature_correction: bool,
    _f: PhantomData<F>,
}

impl<M, F> ManifoldChernoff<M, F>
where
    M: BoundedGeometryManifold<F>,
    F: SemiflowFloat,
{
    /// Construct a `ManifoldChernoff` wrapper.
    ///
    /// - `manifold`: a `BoundedGeometryManifold<F>` backend.
    /// - `with_curvature_correction`: if `true`, applies the `[1+(τ/12)·R(x)]`
    ///   order-2 lift (MMRS 2023 Thm 1); if `false`, returns order-1.
    #[must_use]
    pub fn new(manifold: M, with_curvature_correction: bool) -> Self {
        Self {
            manifold,
            with_curvature_correction,
            _f: PhantomData,
        }
    }
}

// ─── Shared apply logic (D=2, GH-5 tensor product) ───────────────────────────

/// Apply the manifold Chernoff step at one chart node (col, row).
///
/// Integrates `u(exp_x(v))` over `T_xM` using GH-5 ⊗ GH-5, then normalises.
/// Chart coords: `x = [grid.x.x_at(col), grid.y.x_at(row)]`.
///
/// `scale0` and `scale1` convert GH node units to coordinate tangent components.
/// For a metric `g = diag(g_{00}, g_{11})` at x, the orthonormal tangent basis
/// gives `v_i = (2√τ · node_i) / √g_{ii}`. The caller computes these scales.
///
/// # Curvature correction (`with_correction=true`)
/// Applies `output = (1 + τ·R(x)/12) · base` per MMRS 2023 formula (24.1).
/// The curvature factor `[1 + τR/12]` lifts the per-step approximation order
/// from O(τ²) to O(τ³), achieving overall Chernoff convergence order O(τ²)
/// per Theorem 24.1. The effective generator of the corrected Chernoff product
/// is `Δ_M + R/12`; test oracles must account for this shift.
///
/// # Prefactor
/// `(4πτ)^{-1} · (2√τ)² = 1/π` for d=2.
// 8 args: manifold, τ, src, col, row, correction flag, and 2 coord scales — all required.
#[allow(clippy::too_many_arguments)]
fn apply_at_node<M: BoundedGeometryManifold<f64>>(
    manifold: &M,
    tau: f64,
    src: &GridFn2D<f64>,
    col: usize,
    row: usize,
    with_correction: bool,
    scale0: f64, // 2√τ / √g_{00}  (coord scale for axis 0)
    scale1: f64, // 2√τ / √g_{11}  (coord scale for axis 1)
) -> f64 {
    let nx = src.grid.nx();
    let ny = src.grid.ny();
    let coord0 = src.grid.x.x_at(col); // first chart coord
    let coord1 = src.grid.y.x_at(row); // second chart coord
    let chart_x = [coord0, coord1];

    let mut acc = 0.0f64;
    let mut y_out = [0.0f64; 2];
    let mut tangent_v = [0.0f64; 2];

    for (&s_node, &w_s) in GH5_NODES.iter().zip(GH5_WEIGHTS.iter()) {
        for (&t_node, &w_t) in GH5_NODES.iter().zip(GH5_WEIGHTS.iter()) {
            tangent_v[0] = scale0 * s_node;
            tangent_v[1] = scale1 * t_node;
            if manifold.exp_map(&chart_x, &tangent_v, &mut y_out).is_err() {
                continue;
            }
            let val = bilinear_sample(src, nx, ny, y_out[0], y_out[1]);
            acc += w_s * w_t * val;
        }
    }
    // Prefactor: (4πτ)^{-d/2} * (2√τ)^d = π^{-d/2} for d=2 → 1/π.
    let base = acc / core::f64::consts::PI;

    if with_correction {
        let curvature = manifold.scalar_curvature(&chart_x);
        base * (1.0 + (tau / 12.0) * curvature)
    } else {
        base
    }
}

/// Bilinear interpolation of src at chart position (cx, cy).
///
/// Clamps to boundary for out-of-range coords (nearest-valid-node).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn bilinear_sample(src: &GridFn2D<f64>, nx: usize, ny: usize, cx: f64, cy: f64) -> f64 {
    // Find fractional grid index along x-axis.
    let xi = (cx - src.grid.x.xmin) / (src.grid.x.xmax - src.grid.x.xmin) * (nx - 1) as f64;
    let yi = (cy - src.grid.y.xmin) / (src.grid.y.xmax - src.grid.y.xmin) * (ny - 1) as f64;
    // Clamp xi/yi to [0, nx-2] / [0, ny-2] before floor → usize cast.
    let xi = xi.clamp(0.0, (nx - 2) as f64);
    let yi = yi.clamp(0.0, (ny - 2) as f64);
    let i0 = xi.floor() as usize;
    let j0 = yi.floor() as usize;
    let fx = xi - i0 as f64;
    let fy = yi - j0 as f64;
    // Clamp fractional parts to [0,1].
    let fx = fx.clamp(0.0, 1.0);
    let fy = fy.clamp(0.0, 1.0);
    let v00 = src.values[j0 * nx + i0];
    let v10 = src.values[j0 * nx + i0 + 1];
    let v01 = src.values[(j0 + 1) * nx + i0];
    let v11 = src.values[(j0 + 1) * nx + i0 + 1];
    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

// ─── ChernoffFunction impl for Sphere2<f64> ────────────────────────────────────

impl ChernoffFunction<f64> for ManifoldChernoff<Sphere2<f64>, f64> {
    type S = GridFn2D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        // S²(r): metric g = diag(r², r²·sin²θ). Orthonormal basis:
        //   e_θ = (1/r)·∂_θ  → coord scale = 2√τ / r
        //   e_φ = (1/(r·sinθ))·∂_φ → coord scale = 2√τ / (r·sinθ)
        let r = self.manifold.radius;
        let sqrt_tau = tau.sqrt();
        let scale_theta = 2.0 * sqrt_tau / r;
        let nx = src.grid.nx();
        let ny = src.grid.ny();
        for row in 0..ny {
            for col in 0..nx {
                let theta = src.grid.x.x_at(col);
                let sin_theta = theta.sin().max(1e-10); // guard pole
                let scale_phi = 2.0 * sqrt_tau / (r * sin_theta);
                let val = apply_at_node(
                    &self.manifold,
                    tau,
                    src,
                    col,
                    row,
                    self.with_curvature_correction,
                    scale_theta,
                    scale_phi,
                );
                dst.values[row * nx + col] = val;
            }
        }
        Ok(())
    }

    fn order(&self) -> u32 {
        if self.with_curvature_correction {
            2
        } else {
            1
        }
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

// ─── ChernoffFunction impl for Hyperbolic2<f64> ────────────────────────────────

impl ChernoffFunction<f64> for ManifoldChernoff<Hyperbolic2<f64>, f64> {
    type S = GridFn2D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        // H²(scale): metric g = (2·scale/(1-|z|²))² · diag(1,1).
        // Orthonormal scale = 2√τ · (1-|z|²) / (2·scale).
        let sc = self.manifold.scale;
        let sqrt_tau = tau.sqrt();
        let nx = src.grid.nx();
        let ny = src.grid.ny();
        for row in 0..ny {
            for col in 0..nx {
                let u = src.grid.x.x_at(col);
                let w = src.grid.y.x_at(row);
                let one_minus_r_sq = (1.0 - (u * u + w * w)).max(1e-10);
                let ortho_scale = sqrt_tau * one_minus_r_sq / sc;
                let val = apply_at_node(
                    &self.manifold,
                    tau,
                    src,
                    col,
                    row,
                    self.with_curvature_correction,
                    ortho_scale,
                    ortho_scale,
                );
                dst.values[row * nx + col] = val;
            }
        }
        Ok(())
    }

    fn order(&self) -> u32 {
        if self.with_curvature_correction {
            2
        } else {
            1
        }
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

// ─── ChernoffFunction impl for Torus<f64, 2> ──────────────────────────────────

impl ChernoffFunction<f64> for ManifoldChernoff<Torus<f64, 2>, f64> {
    type S = GridFn2D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        // Flat torus T²: metric g = diag(1, 1). Orthonormal scale = 2√τ.
        let scale = 2.0 * tau.sqrt();
        let nx = src.grid.nx();
        let ny = src.grid.ny();
        for row in 0..ny {
            for col in 0..nx {
                let val = apply_at_node(
                    &self.manifold,
                    tau,
                    src,
                    col,
                    row,
                    self.with_curvature_correction,
                    scale,
                    scale,
                );
                dst.values[row * nx + col] = val;
            }
        }
        Ok(())
    }

    fn order(&self) -> u32 {
        // R ≡ 0 on torus; correction is identity. Report order 1 either way.
        1
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

// ─── ChernoffFunction impl for FubiniStudyCp1<f64> (ADR-0129) ─────────────────

impl ChernoffFunction<f64> for ManifoldChernoff<FubiniStudyCp1<f64>, f64> {
    type S = GridFn2D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        // Fubini-Study CP¹: g_FS = σ² I where σ = 2/(1+u²+v²) = 2/(1+r²).
        // Orthonormal tangent scale: 2√τ / σ = 2√τ · (1+r²)/2 = √τ · (1+r²).
        // Both axes share the same conformal factor (conformally flat).
        let sqrt_tau = tau.sqrt();
        let nx = src.grid.nx();
        let ny = src.grid.ny();
        for row in 0..ny {
            for col in 0..nx {
                let u = src.grid.x.x_at(col);
                let v = src.grid.y.x_at(row);
                let r_sq = u * u + v * v;
                let ortho_scale = sqrt_tau * (1.0 + r_sq);
                let val = apply_at_node(
                    &self.manifold,
                    tau,
                    src,
                    col,
                    row,
                    self.with_curvature_correction,
                    ortho_scale,
                    ortho_scale,
                );
                dst.values[row * nx + col] = val;
            }
        }
        Ok(())
    }

    fn order(&self) -> u32 {
        if self.with_curvature_correction {
            2
        } else {
            1
        }
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    include!("manifold_chernoff_tests_mod.rs");
}
