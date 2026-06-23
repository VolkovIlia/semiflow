//! A4 — Riemannian Manifold Chernoff (math.md §24, ADR-0071).
//!
//! `BoundedGeometryManifold<F>` trait + three closed-form backends:
//! `Torus<F, D>` (R ≡ 0), `Sphere2<F>` (R ≡ 2/r²), `Hyperbolic2<F>` (R ≡ −2/s²).
//!
//! `ManifoldChernoff<M, F>` wrapper ships in Wave B.
//!
//! References: MMRS 2023 *Math. Nachr.* Thm 1; Sakai 1996 §III.4; Anderson 2005 §3.3.

use num_traits::Float;

use crate::{error::SemiflowError, float::SemiflowFloat};

// Re-export Hyperbolic2 from sibling module (extracted for file-size compliance).
pub use crate::manifold_hyperbolic::Hyperbolic2;

// ─── BoundedGeometryManifold<F> trait ────────────────────────────────────────

/// Riemannian manifold of bounded geometry (math.md §24.3, ADR-0071).
///
/// Hypotheses: B1 `injectivity_radius() > 0`, B2 `scalar_curvature` bounded,
/// B3 `‖∇^k R‖_∞ < ∞` for k ≤ 2 (needed for the order-2 MMRS correction).
///
/// v2.8 backends: `Torus<F, D>`, `Sphere2<F>`, `Hyperbolic2<F>`.
/// **Not object-safe** (const-generic `D` on `Torus`).
pub trait BoundedGeometryManifold<F: SemiflowFloat = f64>: Send + Sync + 'static {
    /// Manifold dimension d. MUST be ≥ 1.
    fn dim(&self) -> usize;

    /// Uniform infimum of injectivity radius `inf_{x ∈ M} inj(x)`.
    ///
    /// MUST be strictly positive (B1). May return `F::infinity()` for
    /// Cartan-Hadamard manifolds (e.g., Poincaré disk).
    fn injectivity_radius(&self) -> F;

    /// Riemannian exponential map: `out := exp_x(v)`.
    ///
    /// Writes `self.dim()` coordinates into `out`.
    ///
    /// # Errors
    /// Returns `DomainViolation` on length mismatch, NaN/Inf inputs, or
    /// `‖v‖_{g_x} ≥ injectivity_radius()`.
    fn exp_map(&self, x: &[F], v: &[F], out: &mut [F]) -> Result<(), SemiflowError>;

    /// Parallel transport along the geodesic from x to y: `out := P_{x→y}(v)`.
    ///
    /// # Errors
    /// Returns `DomainViolation` on length mismatch or NaN/Inf inputs.
    fn parallel_transport(
        &self,
        x: &[F],
        y: &[F],
        v: &[F],
        out: &mut [F],
    ) -> Result<(), SemiflowError>;

    /// Scalar curvature R(x). Used by `ManifoldChernoff` for the
    /// `[1 + (τ/12)·R(x)]` correction (MMRS 2023 Thm 1).
    fn scalar_curvature(&self, x: &[F]) -> F;

    /// Log of the volume element `log √det g(x)`. Log-space avoids underflow
    /// near coordinate singularities (Poincaré disk: metric diverges at `|x|→1`).
    fn volume_element_log(&self, x: &[F]) -> F;
}

// ─── Shared validation helpers ────────────────────────────────────────────────

/// Return `DomainViolation` for a dim-mismatch error.
#[inline]
fn dim_err(what: &'static str) -> SemiflowError {
    SemiflowError::DomainViolation {
        what,
        value: f64::NAN,
    }
}

/// Check that all elements of the slice are finite.
#[inline]
fn check_finite<F: SemiflowFloat>(s: &[F], what: &'static str) -> Result<(), SemiflowError> {
    for &v in s {
        if !v.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what,
                value: v.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

/// Validate that x, v, out have length == dim and all inputs are finite.
pub(crate) fn validate_exp_inputs<F: SemiflowFloat>(
    dim: usize,
    x: &[F],
    v: &[F],
    out: &[F],
) -> Result<(), SemiflowError> {
    if x.len() != dim {
        return Err(dim_err("exp_map: x.len() != dim"));
    }
    if v.len() != dim {
        return Err(dim_err("exp_map: v.len() != dim"));
    }
    if out.len() != dim {
        return Err(dim_err("exp_map: out.len() != dim"));
    }
    check_finite(x, "exp_map: x contains non-finite coordinate")?;
    check_finite(v, "exp_map: v contains non-finite tangent component")?;
    Ok(())
}

/// Validate that x, y, v, out have length == dim and all inputs are finite.
pub(crate) fn validate_pt_inputs<F: SemiflowFloat>(
    dim: usize,
    x: &[F],
    y: &[F],
    v: &[F],
    out: &[F],
) -> Result<(), SemiflowError> {
    if x.len() != dim || y.len() != dim || v.len() != dim || out.len() != dim {
        return Err(dim_err("parallel_transport: slice length != dim"));
    }
    check_finite(x, "parallel_transport: x contains non-finite")?;
    check_finite(y, "parallel_transport: y contains non-finite")?;
    check_finite(v, "parallel_transport: v contains non-finite")?;
    Ok(())
}

// ─── Backend 1: Torus<F, const D: usize> (flat, R ≡ 0) ──────────────────────

/// Flat *D*-torus `T^D = R^D / Z^D` (math.md §24.4).
///
/// `exp_x(v)` = x + v mod lattice; parallel transport = identity; R ≡ 0;
/// `injectivity_radius` = `min(lattice_period)` / 2.
/// The R/12 correction vanishes; `ManifoldChernoff` reduces to the standard heat kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Torus<F: SemiflowFloat = f64, const D: usize = 2> {
    /// Lattice period in each axis (all must be > 0).
    pub lattice_period: [F; D],
}

impl<F: SemiflowFloat, const D: usize> Copy for Torus<F, D> where F: Copy {}

impl<F: SemiflowFloat, const D: usize> Torus<F, D> {
    /// Unit torus: lattice period = 1.0 in each axis.
    #[must_use]
    pub fn unit() -> Self {
        Self {
            lattice_period: [F::one(); D],
        }
    }

    /// Torus with custom lattice periods (each must be > 0 and finite).
    ///
    /// # Errors
    /// Returns `DomainViolation` if any period is ≤ 0 or non-finite.
    pub fn with_period(period: [F; D]) -> Result<Self, SemiflowError> {
        for &p in &period {
            if !p.is_finite() || p <= F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what: "Torus: lattice period must be finite and > 0",
                    value: p.to_f64().unwrap_or(f64::NAN),
                });
            }
        }
        Ok(Self {
            lattice_period: period,
        })
    }
}

impl<F: SemiflowFloat, const D: usize> BoundedGeometryManifold<F> for Torus<F, D>
where
    F: Copy,
{
    fn dim(&self) -> usize {
        D
    }

    fn injectivity_radius(&self) -> F {
        // For T^D with periods p_i: inj = min_i(p_i) / 2.
        let two = F::one() + F::one();
        let mut min = self.lattice_period[0];
        for &p in &self.lattice_period[1..] {
            if p < min {
                min = p;
            }
        }
        min / two
    }

    fn exp_map(&self, x: &[F], v: &[F], out: &mut [F]) -> Result<(), SemiflowError> {
        validate_exp_inputs::<F>(D, x, v, out)?;
        for i in 0..D {
            out[i] = torus_wrap(x[i] + v[i], self.lattice_period[i]);
        }
        Ok(())
    }

    fn parallel_transport(
        &self,
        x: &[F],
        y: &[F],
        v: &[F],
        out: &mut [F],
    ) -> Result<(), SemiflowError> {
        validate_pt_inputs::<F>(D, x, y, v, out)?;
        // Flat torus: parallel transport is the identity in normal coordinates.
        out.copy_from_slice(v);
        Ok(())
    }

    fn scalar_curvature(&self, _x: &[F]) -> F {
        F::zero()
    }

    fn volume_element_log(&self, _x: &[F]) -> F {
        // √det g = 1 in normal coordinates on the flat torus.
        F::zero()
    }
}

/// Wrap a coordinate into [0, period) with care for floating-point edge cases.
#[inline]
pub(crate) fn torus_wrap<F: SemiflowFloat>(raw: F, period: F) -> F {
    // Equivalent to: raw - floor(raw / period) * period
    let quot = Float::floor(raw / period);
    let wrapped = raw - quot * period;
    // Guard negative zero or tiny negatives from rounding
    if wrapped < F::zero() {
        wrapped + period
    } else {
        wrapped
    }
}

// ─── Backend 2: Sphere2<F> (2-sphere, R ≡ 2 / radius²) ──────────────────────

/// 2-sphere `S²` with configurable radius (math.md §24.4, Sakai 1996 §III.4).
///
/// Chart: spherical `(θ, φ)`. `exp_x(v)` via 3D great-circle embedding.
/// R ≡ 2/radius²; `injectivity_radius` = π·radius; log √det g = 2·log r + log sin θ.
/// Poles θ = 0, π are coordinate singularities; `ManifoldChernoff` (Wave B) avoids them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere2<F: SemiflowFloat = f64> {
    /// Sphere radius (must be > 0 and finite).
    pub radius: F,
}

impl<F: SemiflowFloat> Sphere2<F> {
    /// Unit sphere (radius = 1).
    #[must_use]
    pub fn unit() -> Self {
        Self { radius: F::one() }
    }

    /// Sphere with given radius (must be > 0 and finite).
    ///
    /// # Errors
    /// Returns `DomainViolation` if radius ≤ 0 or non-finite.
    pub fn with_radius(radius: F) -> Result<Self, SemiflowError> {
        if !radius.is_finite() || radius <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "Sphere2: radius must be finite and > 0",
                value: radius.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self { radius })
    }
}

impl<F: SemiflowFloat> BoundedGeometryManifold<F> for Sphere2<F> {
    fn dim(&self) -> usize {
        2
    }

    fn injectivity_radius(&self) -> F {
        // For S² with radius r: inj = π·r (cut locus is the antipode).
        let pi = F::from(core::f64::consts::PI).unwrap_or_else(F::zero);
        pi * self.radius
    }

    /// `exp_x(v)` via 3D great-circle embedding (Sakai 1996 §III.4).
    ///
    /// # Errors
    /// Returns `DomainViolation` on length mismatch or non-finite inputs.
    fn exp_map(&self, x: &[F], v: &[F], out: &mut [F]) -> Result<(), SemiflowError> {
        validate_exp_inputs::<F>(2, x, v, out)?;
        let theta = x[0];
        let phi = x[1];
        // Convert (θ, φ) to 3D unit vector P
        let (p0, p1, p2) = spherical_to_cartesian::<F>(theta, phi);
        // Convert (v_θ, v_φ) to 3D tangent V at P (metric g = r²·diag(1, sin²θ))
        let st = Float::sin(theta);
        let ct = Float::cos(theta);
        let sp = Float::sin(phi);
        let cp = Float::cos(phi);
        // e_θ = (cosθ·cosφ, cosθ·sinφ, -sinθ) and e_φ = (-sinφ, cosφ, 0)
        // V = r · (v_θ · e_θ + v_φ · sin(θ) · e_φ)  [so ‖V‖=r·‖v‖_g]
        let r = self.radius;
        let vt = v[0];
        let vp = v[1];
        let v3x = r * (vt * ct * cp - vp * st * sp);
        let v3y = r * (vt * ct * sp + vp * st * cp);
        let v3z = r * (-vt * st);
        // ‖V‖ = Euclidean norm of 3D vector V
        let norm_v = Float::sqrt(v3x * v3x + v3y * v3y + v3z * v3z);
        if norm_v < F::epsilon() {
            // Zero tangent vector → identity
            out[0] = theta;
            out[1] = phi;
            return Ok(());
        }
        // α = ‖V‖ / r (the geodesic arc length, normalised to unit sphere angle)
        let alpha = norm_v / r;
        // Q = P·cos(α) + (V/‖V‖)·sin(α)
        let ca = Float::cos(alpha);
        let sa = Float::sin(alpha);
        let inv_norm = F::one() / norm_v;
        let q0 = p0 * ca + v3x * inv_norm * sa;
        let q1 = p1 * ca + v3y * inv_norm * sa;
        let q2 = p2 * ca + v3z * inv_norm * sa;
        // Convert Q back to spherical: θ' = arccos(q2), φ' = atan2(q1, q0)
        let theta_out = Float::acos(q2.min(F::one()).max(-F::one()));
        let phi_out = canonicalize_phi(Float::atan2(q1, q0));
        out[0] = theta_out;
        out[1] = phi_out;
        Ok(())
    }

    /// Parallel transport via Rodrigues rotation (SO(3) great-circle).
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
        // Work in 3D: lift x, y, v to R³ and apply SO(3) parallel transport.
        let (px, py_3, pz) = spherical_to_cartesian::<F>(x[0], x[1]);
        let (qx, qy_3, qz) = spherical_to_cartesian::<F>(y[0], y[1]);
        // Convert tangent v (2D) to 3D at point P
        let v3 = sphere_tangent_to_3d::<F>(x[0], x[1], v[0], v[1]);
        // Geodesic axis n = P × Q (normalised) and angle ψ = arccos(P·Q)
        let dot_pq = (px * qx + py_3 * qy_3 + pz * qz)
            .min(F::one())
            .max(-F::one());
        let psi = Float::acos(dot_pq);
        if Float::abs(psi) < F::epsilon() {
            // x ≈ y → identity transport
            out.copy_from_slice(v);
            return Ok(());
        }
        // n = (P × Q) / ‖P × Q‖ — rotation axis
        let nx = py_3 * qz - pz * qy_3;
        let ny = pz * qx - px * qz;
        let nz = px * qy_3 - py_3 * qx;
        let inv_sin = F::one() / Float::sin(psi);
        let nx = nx * inv_sin;
        let ny = ny * inv_sin;
        let nz = nz * inv_sin;
        // Rodrigues: rotate v3 by ψ around n
        let v3_rot = rodrigues_3d::<F>(v3, [nx, ny, nz], psi);
        // Project v3_rot onto T_y S² and convert back to 2D coords at y
        let v2 = sphere_3d_to_tangent::<F>(y[0], y[1], v3_rot);
        out[0] = v2[0];
        out[1] = v2[1];
        Ok(())
    }

    fn scalar_curvature(&self, _x: &[F]) -> F {
        // R = 2 / radius² (S² with unit radius has R = 2; see Sakai 1996 §IV.2)
        let two = F::one() + F::one();
        two / (self.radius * self.radius)
    }

    fn volume_element_log(&self, x: &[F]) -> F {
        // √det g = r²·sin(θ); log = 2·log(r) + log(sin(θ))
        // Caller is responsible for avoiding θ = 0, π (polar singularities).
        let two = F::one() + F::one();
        let log_r = Float::ln(self.radius);
        let sin_theta = Float::sin(x[0]);
        // sin(θ) = 0 at poles; return large negative (caller avoids poles).
        let log_sin = if sin_theta <= F::zero() {
            F::from(-1e30_f64).unwrap_or_else(F::zero)
        } else {
            Float::ln(sin_theta)
        };
        two * log_r + log_sin
    }
}

// ─── Sphere2 geometry helpers ─────────────────────────────────────────────────

/// Convert spherical (θ, φ) to unit 3D Cartesian (on unit sphere).
#[inline]
fn spherical_to_cartesian<F: SemiflowFloat>(theta: F, phi: F) -> (F, F, F) {
    let st = Float::sin(theta);
    let ct = Float::cos(theta);
    let sp = Float::sin(phi);
    let cp = Float::cos(phi);
    (st * cp, st * sp, ct)
}

/// Wrap φ into [0, 2π).
#[inline]
fn canonicalize_phi<F: SemiflowFloat>(phi: F) -> F {
    let two_pi = F::from(2.0 * core::f64::consts::PI).unwrap_or_else(F::zero);
    torus_wrap(phi, two_pi)
}

/// Convert 2D tangent (`v_θ`, `v_φ`) at spherical (θ, φ) to 3D vector.
///
/// Uses basis vectors:
/// - `e_θ` = (cos θ cos φ, cos θ sin φ, −sin θ)
/// - `e_φ` = (−sin φ, cos φ, 0)
fn sphere_tangent_to_3d<F: SemiflowFloat>(theta: F, phi: F, vt: F, vp: F) -> [F; 3] {
    let st = Float::sin(theta);
    let ct = Float::cos(theta);
    let sp = Float::sin(phi);
    let cp = Float::cos(phi);
    [vt * ct * cp - vp * sp, vt * ct * sp + vp * cp, -vt * st]
}

/// Project 3D vector onto T_{(θ,φ)} S² and express in (`v_θ`, `v_φ`) coordinates.
fn sphere_3d_to_tangent<F: SemiflowFloat>(theta: F, phi: F, w: [F; 3]) -> [F; 2] {
    // e_θ and e_φ are orthonormal at (θ, φ); dot with w to get components.
    let st = Float::sin(theta);
    let ct = Float::cos(theta);
    let sp = Float::sin(phi);
    let cp = Float::cos(phi);
    let vt = w[0] * ct * cp + w[1] * ct * sp - w[2] * st;
    let vp = -w[0] * sp + w[1] * cp;
    [vt, vp]
}

/// Rodrigues rotation: rotate vector `v` by angle `psi` around unit axis `n`.
pub(crate) fn rodrigues_3d<F: SemiflowFloat>(v: [F; 3], n: [F; 3], psi: F) -> [F; 3] {
    let cp = Float::cos(psi);
    let sp = Float::sin(psi);
    let dot = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
    // cross = n × v
    let cx = n[1] * v[2] - n[2] * v[1];
    let cy = n[2] * v[0] - n[0] * v[2];
    let cz = n[0] * v[1] - n[1] * v[0];
    // Rodrigues: v·cos(ψ) + (n×v)·sin(ψ) + n·(n·v)·(1−cos(ψ))
    let one_minus_cp = F::one() - cp;
    [
        v[0] * cp + cx * sp + n[0] * dot * one_minus_cp,
        v[1] * cp + cy * sp + n[1] * dot * one_minus_cp,
        v[2] * cp + cz * sp + n[2] * dot * one_minus_cp,
    ]
}

// ─── Unit tests (extracted to keep file under 500 lines) ──────────────────────

#[cfg(test)]
#[path = "manifold_tests.rs"]
mod tests;
