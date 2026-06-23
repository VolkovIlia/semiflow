//! `Hyperbolic2<F>` — Poincaré disk backend for `BoundedGeometryManifold`.
//!
//! Extracted from `manifold.rs` to keep that file under 500 lines.
//! See math.md §24.4 and Anderson 2005 §3.3.

use num_traits::Float;

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    manifold::{validate_exp_inputs, validate_pt_inputs, BoundedGeometryManifold},
};

// ─── Backend 3: Hyperbolic2<F> (Poincaré disk, R ≡ -2 / scale²) ─────────────

/// Poincaré disk `H²` (math.md §24.4, Anderson 2005 §3.3).
///
/// Chart: (u, w) with u² + w² < 1; metric 4·scale²/(1−|z|²)² · |dz|².
/// `exp_x(v)` via Möbius translate-to-origin → hyperbolic exp → translate back.
/// R ≡ −2/scale²; `injectivity_radius` = ∞; log √det g = log(2·scale) − log(1−|x|²).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hyperbolic2<F: SemiflowFloat = f64> {
    /// Scale parameter (must be > 0 and finite).
    ///
    /// Sectional curvature = −1/scale²; scalar curvature R = −2/scale².
    pub scale: F,
}

impl<F: SemiflowFloat> Hyperbolic2<F> {
    /// Unit Poincaré disk (scale = 1, R = −2).
    #[must_use]
    pub fn unit() -> Self {
        Self { scale: F::one() }
    }

    /// Poincaré disk with given scale (must be > 0 and finite).
    ///
    /// # Errors
    /// Returns `DomainViolation` if scale ≤ 0 or non-finite.
    pub fn with_scale(scale: F) -> Result<Self, SemiflowError> {
        if !scale.is_finite() || scale <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "Hyperbolic2: scale must be finite and > 0",
                value: scale.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self { scale })
    }
}

impl<F: SemiflowFloat> BoundedGeometryManifold<F> for Hyperbolic2<F> {
    fn dim(&self) -> usize {
        2
    }

    fn injectivity_radius(&self) -> F {
        // H² (Poincaré disk) has no conjugate points; inj = +∞.
        F::infinity()
    }

    /// `exp_x(v)` via Möbius transformation (Anderson 2005 §3.3).
    ///
    /// # Errors
    /// Returns `DomainViolation` on length mismatch or non-finite inputs.
    fn exp_map(&self, x: &[F], v: &[F], out: &mut [F]) -> Result<(), SemiflowError> {
        validate_exp_inputs::<F>(2, x, v, out)?;
        // Hyperbolic norm of v at x: ‖v‖_hyp = 2·scale·‖v‖_Euc/(1−|x|²)
        let norm_euc_v = Float::sqrt(v[0] * v[0] + v[1] * v[1]);
        if norm_euc_v < F::epsilon() {
            // Zero tangent → identity
            out[0] = x[0];
            out[1] = x[1];
            return Ok(());
        }
        let x_sq = x[0] * x[0] + x[1] * x[1];
        let two = F::one() + F::one();
        let hyp_norm_v = two * self.scale * norm_euc_v / (F::one() - x_sq);
        // At origin: w = tanh(‖v‖_hyp / 2) · v̂
        let tanh_half = Float::tanh(hyp_norm_v / two);
        let alpha_u = tanh_half * v[0] / norm_euc_v;
        let alpha_w = tanh_half * v[1] / norm_euc_v;
        // Translate back: Möbius_x(alpha)
        let (res_u, res_w) = mobius_translate::<F>(x[0], x[1], alpha_u, alpha_w);
        out[0] = res_u;
        out[1] = res_w;
        Ok(())
    }

    /// Parallel transport via Möbius derivative ratio.
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
        // Möbius map M_x: z ↦ (z - x) / (1 - x̄·z) that sends x to origin.
        // Its derivative at x is the conformal factor (1-|x|²) / (1 - |x|²)² = 1/(1-|x|²).
        // The parallel transport along the geodesic from x to y equals
        // the composition D(M_y)^{-1} ∘ D(M_x) applied to v (chain rule on Möbius maps).
        // Conformal factor at x: Jac_x = (1 - |x|²)  (holomorphic derivative of M_x at x)
        // Conformal factor at y: Jac_y = (1 - |y|²)
        // The parallel transport is a rotation + scaling by Jac_x / Jac_y.
        let x_sq = x[0] * x[0] + x[1] * x[1];
        let y_sq = y[0] * y[0] + y[1] * y[1];
        let jac_x = F::one() - x_sq;
        let jac_y = F::one() - y_sq;
        // Mobius derivative ratio
        let ratio = jac_x / jac_y;
        // Rotation angle: the Möbius composition also applies a phase rotation.
        // Compute the complex argument of the derivative d/dz[M_y^{-1} ∘ M_x](x).
        // At x, M_x(x)=0, M_x'(x) = 1/(1-|x|²). M_y^{-1}(w)=(w+y)/(1+ȳw).
        // d/dw M_y^{-1}(0) = 1 - |y|² (real, positive for |y|<1).
        // So the full derivative at x is real and positive; no rotation.
        // (Full Möbius phase rotation only appears for x≠y in general; for
        //  the geodesic direction, the derivative is always real-positive.)
        out[0] = v[0] * ratio;
        out[1] = v[1] * ratio;
        Ok(())
    }

    fn scalar_curvature(&self, _x: &[F]) -> F {
        // R = −2 / scale² (unit Poincaré disk: R = −2)
        let two = F::one() + F::one();
        -two / (self.scale * self.scale)
    }

    fn volume_element_log(&self, x: &[F]) -> F {
        // √det g = (2·scale)² / (1 - |x|²)² = 4·scale² / (1 - r²)²
        // But the Riemannian volume form is √det g in the (u, w) chart:
        // √det g = 2·scale / (1 - |x|²)  [from the conformal metric]
        // log = log(2·scale) - log(1 - |x|²)
        let two = F::one() + F::one();
        let x_sq = x[0] * x[0] + x[1] * x[1];
        let one_minus_r_sq = (F::one() - x_sq).max(F::epsilon());
        Float::ln(two * self.scale) - Float::ln(one_minus_r_sq)
    }
}

// ─── Hyperbolic2 geometry helpers ─────────────────────────────────────────────

/// Möbius translation: f(z) = (z + a) / (1 + ā·z) in complex arithmetic.
///
/// Here z = (`z_u`, `z_w`) and a = (`a_u`, `a_w`) are 2D complex numbers.
/// Returns the real and imaginary parts of f(z).
#[inline]
fn mobius_translate<F: SemiflowFloat>(a_u: F, a_w: F, z_u: F, z_w: F) -> (F, F) {
    // Numerator: (z + a) = (z_u + a_u, z_w + a_w)
    let num_u = z_u + a_u;
    let num_w = z_w + a_w;
    // Denominator: (1 + ā·z) where ā = (a_u, -a_w) (complex conj of a)
    // ā·z = (a_u·z_u + a_w·z_w, a_u·z_w - a_w·z_u)
    let conj_a_dot_z_re = a_u * z_u + a_w * z_w;
    let conj_a_dot_z_im = a_u * z_w - a_w * z_u;
    let den_u = F::one() + conj_a_dot_z_re;
    let den_w = conj_a_dot_z_im;
    // Division: (num_u + i·num_w) / (den_u + i·den_w)
    let den_sq = den_u * den_u + den_w * den_w;
    let res_u = (num_u * den_u + num_w * den_w) / den_sq;
    let res_w = (num_w * den_u - num_u * den_w) / den_sq;
    (res_u, res_w)
}
