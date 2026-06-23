//! B5 (ADR-0129) — CP¹ Fubini-Study Kähler manifold backend (math.md §24.7-bis).
//!
//! `FubiniStudyCp1<F>` implements `BoundedGeometryManifold<F>` for the complex
//! projective line CP¹ with the Fubini-Study metric in the affine chart z = u + iv:
//!
//! ```text
//! g_FS = 4 / (1 + |z|²)² · (du² + dv²)
//! ```
//!
//! This metric is **isometric to the round S²** via stereographic projection (ADR-0129
//! PRE-FLIGHT, `scripts/verify_complex_kahler.py` 3/3 PASS). Scalar curvature R = 2
//! (constant). The `ManifoldChernoff<FubiniStudyCp1, F>` wrapper therefore reuses the
//! shipped R/12 machinery from math.md §24.2 verbatim — no new convergence theory.
//!
//! The "Kähler" label is honest at the SCALAR heat level (complex affine chart,
//! Fubini-Study metric). Holomorphic line-bundle sections remain a future deferral.
//!
//! # Chart and coordinate conventions
//!
//! Points: `x = [u, v]` with z = u + iv ∈ ℂ (affine chart of CP¹; north pole excluded).
//! Tangent: `v = [v_u, v_v]` in the chart basis (∂_u, ∂_v).
//!
//! # Coordinate helper formulae (stereographic, math.md §24.7-bis)
//!
//! Inverse stereographic (chart → unit S²): let r² = u² + v², σ = 1 + r².
//! ```text
//! P = (2u/σ, 2v/σ, (r²-1)/σ)
//! ```
//! Forward stereographic (unit S² → chart, from north pole; requires Z ≠ 1):
//! ```text
//! (u, v) = (X/(1+Z), Y/(1+Z))
//! ```
//! Note: the Fubini-Study normalisation uses projection *from the south pole* (Z = −1),
//! matching the convention where the Gauss curvature K = 1 gives scalar R = 2.
//!
//! # References
//!
//! - ADR-0129, `scripts/verify_complex_kahler.py`.
//! - MMRS 2023 *Math. Nachr.* Thm 1 (R/12 correction, applies verbatim via isometry).
//! - Sakai 1996 §III.4 (geodesic `exp_map` on S² — same geometry via isometry).

use num_traits::Float;

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    manifold::{rodrigues_3d, validate_exp_inputs, validate_pt_inputs, BoundedGeometryManifold},
};

// ─── FubiniStudyCp1<F> ────────────────────────────────────────────────────────

/// CP¹ with Fubini-Study metric in the affine chart z = u + iv (ADR-0129).
///
/// Affine chart: `x = [u, v]` with z = u + iv ∈ ℂ.
/// Metric: `g_FS = 4/(1+|z|²)² · (du² + dv²)`.
/// Scalar curvature: **R = 2** (constant; S²-isometric).
/// Injectivity radius: π (diameter of CP¹ in the FS metric).
///
/// `ManifoldChernoff<FubiniStudyCp1>` with `with_curvature_correction = true`
/// achieves convergence order 2 via the MMRS 2023 R/12 machinery.
///
/// **Scope boundary**: scalar Laplace-Beltrami heat only. Holomorphic line-bundle
/// sections (genuinely complex-valued state) are a future item.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FubiniStudyCp1<F: SemiflowFloat = f64> {
    _f: core::marker::PhantomData<F>,
}

impl<F: SemiflowFloat> Default for FubiniStudyCp1<F> {
    fn default() -> Self {
        Self {
            _f: core::marker::PhantomData,
        }
    }
}

impl<F: SemiflowFloat> FubiniStudyCp1<F> {
    /// Construct the unit CP¹ Fubini-Study backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: SemiflowFloat> BoundedGeometryManifold<F> for FubiniStudyCp1<F> {
    fn dim(&self) -> usize {
        2
    }

    fn injectivity_radius(&self) -> F {
        // CP¹ with Fubini-Study has diameter π (half the diameter of S² with R=2).
        // Same as unit S² with radius 1: inj = π·r = π.
        F::from(core::f64::consts::PI).unwrap_or_else(F::zero)
    }

    /// `exp_x(v)` via 3D great-circle embedding (Sakai 1996 §III.4, isometry to S²).
    ///
    /// Algorithm: stereographic chart → unit S² → great-circle step → chart.
    ///
    /// # Errors
    /// Returns `DomainViolation` on length mismatch or non-finite inputs.
    fn exp_map(&self, x: &[F], v: &[F], out: &mut [F]) -> Result<(), SemiflowError> {
        validate_exp_inputs::<F>(2, x, v, out)?;
        let (u0, v0) = (x[0], x[1]);
        let (vu, vv) = (v[0], v[1]);
        // 1. Inverse stereographic: chart (u,v) → unit S² point P.
        let p = stereo_inv::<F>(u0, v0);
        // 2. Convert chart tangent (v_u, v_v) to 3D tangent V at P.
        //    V = v_u * dP/du + v_v * dP/dv  (Jacobian of stereo_inv).
        let (j_u, j_v) = stereo_inv_jacobian::<F>(u0, v0);
        let v3 = [
            vu * j_u[0] + vv * j_v[0],
            vu * j_u[1] + vv * j_v[1],
            vu * j_u[2] + vv * j_v[2],
        ];
        // 3. |V| = geodesic arc angle α.
        let norm_v = Float::sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2]);
        if norm_v < F::epsilon() {
            out[0] = u0;
            out[1] = v0;
            return Ok(());
        }
        // 4. Q = P·cos(α) + (V/|V|)·sin(α).
        let ca = Float::cos(norm_v);
        let sa = Float::sin(norm_v);
        let inv = F::one() / norm_v;
        let q = [
            p[0] * ca + v3[0] * inv * sa,
            p[1] * ca + v3[1] * inv * sa,
            p[2] * ca + v3[2] * inv * sa,
        ];
        // 5. Forward stereographic: unit S² → chart.
        let (u_out, v_out) = stereo_proj::<F>(q);
        out[0] = u_out;
        out[1] = v_out;
        Ok(())
    }

    /// Parallel transport via Rodrigues rotation (Sakai 1996 §III.4).
    ///
    /// # Errors
    /// Returns `DomainViolation` on length mismatch or non-finite inputs.
    fn parallel_transport(
        &self,
        x: &[F],
        y: &[F],
        v: &[F],
        out: &mut [F],
    ) -> Result<(), SemiflowError> {
        validate_pt_inputs::<F>(2, x, y, v, out)?;
        // Lift to 3D, apply SO(3) parallel transport, project back.
        let px = stereo_inv::<F>(x[0], x[1]);
        let qy = stereo_inv::<F>(y[0], y[1]);
        // Convert chart tangent v at x to 3D.
        let (j_u, j_v) = stereo_inv_jacobian::<F>(x[0], x[1]);
        let v3 = [
            v[0] * j_u[0] + v[1] * j_v[0],
            v[0] * j_u[1] + v[1] * j_v[1],
            v[0] * j_u[2] + v[1] * j_v[2],
        ];
        // Rodrigues rotation axis n = P × Q / |P × Q|, angle ψ = arccos(P·Q).
        let dot = (px[0] * qy[0] + px[1] * qy[1] + px[2] * qy[2])
            .min(F::one())
            .max(-F::one());
        let psi = Float::acos(dot);
        if Float::abs(psi) < F::epsilon() {
            out[0] = v[0];
            out[1] = v[1];
            return Ok(());
        }
        let cx = px[1] * qy[2] - px[2] * qy[1];
        let cy = px[2] * qy[0] - px[0] * qy[2];
        let cz = px[0] * qy[1] - px[1] * qy[0];
        let inv_sin = F::one() / Float::sin(psi);
        let n = [cx * inv_sin, cy * inv_sin, cz * inv_sin];
        let w3 = rodrigues_3d::<F>(v3, n, psi);
        // Project w3 from T_y S² back to chart tangent at y.
        let w2 = stereo_proj_tangent::<F>(y[0], y[1], w3);
        out[0] = w2[0];
        out[1] = w2[1];
        Ok(())
    }

    fn scalar_curvature(&self, _x: &[F]) -> F {
        // Fubini-Study CP¹: R = 2 (constant; S²-isometric, verified by
        // scripts/verify_complex_kahler.py sub-check 1).
        F::one() + F::one()
    }

    fn volume_element_log(&self, x: &[F]) -> F {
        // √det g = 4/(1+r²)² for g_FS = 4/σ² * I; log = 2·log(2) - 2·log(1+r²).
        let two = F::one() + F::one();
        let r_sq = x[0] * x[0] + x[1] * x[1];
        let log_two = Float::ln(two);
        let sigma = F::one() + r_sq;
        let log_sigma = Float::ln(sigma.max(F::epsilon()));
        two * log_two - two * log_sigma
    }
}

// ─── Stereographic geometry helpers ──────────────────────────────────────────

/// Inverse stereographic projection from chart (u,v) to unit S² (from south pole).
///
/// P = (2u/σ, 2v/σ, (r²−1)/σ) with σ = 1 + r², r² = u²+v².
#[inline]
fn stereo_inv<F: SemiflowFloat>(u: F, v: F) -> [F; 3] {
    let two = F::one() + F::one();
    let r_sq = u * u + v * v;
    let sigma = F::one() + r_sq;
    [two * u / sigma, two * v / sigma, (r_sq - F::one()) / sigma]
}

/// Jacobian columns of `stereo_inv` at (u,v).
///
/// Returns (∂P/∂u, ∂P/∂v) as 3D vectors.
#[inline]
fn stereo_inv_jacobian<F: SemiflowFloat>(u: F, v: F) -> ([F; 3], [F; 3]) {
    let two = F::one() + F::one();
    let four = two + two;
    let r_sq = u * u + v * v;
    let sigma = F::one() + r_sq;
    let sigma_sq = sigma * sigma;
    // ∂P/∂u = ( 2(1+v²-u²)/σ², -4uv/σ², 4u/σ² )
    let j_u = [
        two * (F::one() + v * v - u * u) / sigma_sq,
        -four * u * v / sigma_sq,
        four * u / sigma_sq,
    ];
    // ∂P/∂v = ( -4uv/σ², 2(1+u²-v²)/σ², 4v/σ² )
    let j_v = [
        -four * u * v / sigma_sq,
        two * (F::one() + u * u - v * v) / sigma_sq,
        four * v / sigma_sq,
    ];
    (j_u, j_v)
}

/// Forward stereographic projection from unit S² to chart (from north pole, Z ≠ 1).
///
/// (u, v) = (X/(1−Z), Y/(1−Z)).
/// This is the inverse of `stereo_inv`: `stereo_proj(stereo_inv(u,v)) = (u,v)`.
/// Z = 1 (north pole) is the point at infinity of the affine chart; clamped away.
#[inline]
fn stereo_proj<F: SemiflowFloat>(p: [F; 3]) -> (F, F) {
    // p[2] = Z; 1-Z must be positive (north pole Z=1 excluded from chart).
    let one_minus_z = (F::one() - p[2]).max(F::epsilon());
    (p[0] / one_minus_z, p[1] / one_minus_z)
}

/// Pull back 3D tangent `w` at point `stereo_inv(u,v)` on unit S² to chart tangent.
///
/// Solves `V = v_u·J_u + v_v·J_v` for `(v_u, v_v)`, where `J_u = ∂P/∂u`, `J_v = ∂P/∂v`.
/// `J_u ⊥ J_v` and `|J_u|² = |J_v|² = 4/σ²` (see `manifold_kahler.rs` derivation)
/// so `v_u = w·J_u / |J_u|²`, `v_v = w·J_v / |J_v|²`.
#[inline]
fn stereo_proj_tangent<F: SemiflowFloat>(u: F, v: F, w: [F; 3]) -> [F; 2] {
    let (j_u, j_v) = stereo_inv_jacobian::<F>(u, v);
    // |j_u|² = |j_v|² = 4/σ² (from the FS metric identity).
    let norm_sq = j_u[0] * j_u[0] + j_u[1] * j_u[1] + j_u[2] * j_u[2];
    let norm_sq = if norm_sq < F::epsilon() {
        F::epsilon()
    } else {
        norm_sq
    };
    let dot_u = w[0] * j_u[0] + w[1] * j_u[1] + w[2] * j_u[2];
    let dot_v = w[0] * j_v[0] + w[1] * j_v[1] + w[2] * j_v[2];
    [dot_u / norm_sq, dot_v / norm_sq]
}

// ─── Inline unit tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fs_scalar_curvature_is_two() {
        let m = FubiniStudyCp1::<f64>::new();
        // constant R = 2 everywhere
        for x in &[[0.0, 0.0], [1.0, 0.5], [-0.7, 0.3], [3.0, -2.0]] {
            let r = m.scalar_curvature(x.as_slice());
            assert!((r - 2.0).abs() < 1e-14, "R={r} at x={x:?}");
        }
    }

    #[test]
    fn fs_exp_map_zero_v_is_identity() {
        let m = FubiniStudyCp1::<f64>::new();
        let mut out = [0.0f64; 2];
        m.exp_map(&[0.5, 0.3], &[0.0, 0.0], &mut out).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-12, "u={}", out[0]);
        assert!((out[1] - 0.3).abs() < 1e-12, "v={}", out[1]);
    }

    #[test]
    fn stereo_roundtrip_at_origin() {
        let p = stereo_inv::<f64>(0.0, 0.0);
        // (0,0) → south pole (0,0,-1) in this convention
        assert!((p[2] + 1.0).abs() < 1e-14, "Z={}", p[2]);
        let (u, v) = stereo_proj::<f64>(p);
        assert!(u.abs() < 1e-12, "u={u}");
        assert!(v.abs() < 1e-12, "v={v}");
    }

    #[test]
    fn stereo_roundtrip_general() {
        for (u0, v0) in &[(1.0_f64, 0.5), (-0.7, 0.3), (2.0, -1.0)] {
            let p = stereo_inv::<f64>(*u0, *v0);
            // Check p is on unit sphere.
            let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            assert!((r2 - 1.0).abs() < 1e-13, "|p|²={r2} at ({u0},{v0})");
            // Check roundtrip.
            let (u1, v1) = stereo_proj::<f64>(p);
            assert!((u1 - u0).abs() < 1e-12, "u0={u0} u1={u1}");
            assert!((v1 - v0).abs() < 1e-12, "v0={v0} v1={v1}");
        }
    }

    #[test]
    fn fs_injectivity_radius() {
        let m = FubiniStudyCp1::<f64>::new();
        let pi = core::f64::consts::PI;
        assert!((m.injectivity_radius() - pi).abs() < 1e-12);
    }
}
