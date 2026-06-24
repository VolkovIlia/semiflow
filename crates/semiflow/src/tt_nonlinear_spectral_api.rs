//! Public wrapper types for `tt_nonlinear_spectral` (ADR-0169, v9.2.0).
//!
//! Sibling module — contains [`S3BurgersColeHopf`] and [`S3ReactionDiffusion`]
//! which are the boundary-as-type public surface for the two S³ nonlinear evolvers.
//! Raw `pub(crate)` free functions stay in `tt_nonlinear_spectral`.
//!
//! Feature-gated: `s3-poc`.
#![cfg_attr(docsrs, doc(cfg(feature = "s3-poc")))]
#![allow(clippy::cast_possible_truncation)]

extern crate alloc;
use alloc::vec::Vec;

use crate::{
    float::SemiflowFloat,
    tt_nonlinear_spectral::{burgers_cole_hopf_evolve, strang_rd_evolve, Reaction, StrangConfig},
};

// ═══════════════════════════════════════════════════════════════════════════
// Seam A — Cole-Hopf Burgers
// ═══════════════════════════════════════════════════════════════════════════

/// Exact-in-time Cole-Hopf Burgers evolver (S³ POC, Seam A).
///
/// Solves `u_t = ν u_xx − u u_x` on a periodic 1-D grid via the exact Cole-Hopf
/// transform: linearise to heat, evolve exactly, invert.  Restricted to zero-mean
/// initial conditions (non-zero mean is the construction-time wall).
///
/// ## Proven boundary
/// Exact ONLY for Burgers via Cole-Hopf (`u_t = ν u_xx − u u_x`). Generalised
/// nonlinearities (e.g. `u_t = ν u_xx + g(u)`) require Seam B
/// [`S3ReactionDiffusion`]. `ν ≤ 0` is unconstructible (parabolicity wall). Proof:
/// `g_s3_nonlinear` (RELEASE-BLOCKING, `slow-tests`), Cole-Hopf sub-gate error
/// ≤1e-9. See ADR-0168, math.md §53.5.
pub struct S3BurgersColeHopf<F: SemiflowFloat> {
    /// Grid size (1-D).
    n: usize,
    /// Grid spacing.
    dx: F,
    /// Kinematic viscosity `ν > 0`.
    nu: F,
}

impl<F: SemiflowFloat> S3BurgersColeHopf<F> {
    /// Construct the evolver; fails if `n < 4`, `dx ≤ 0`, or `nu ≤ 0`.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] for out-of-class inputs.
    pub fn new(n: usize, dx: F, nu: F) -> Result<Self, crate::SemiflowError> {
        if n < 4 {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "n must be >= 4 for Burgers Cole-Hopf",
            });
        }
        if dx <= F::zero() {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "dx must be > 0",
            });
        }
        if nu <= F::zero() {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "nu must be > 0 (parabolicity wall)",
            });
        }
        Ok(Self { n, dx, nu })
    }

    /// Evolve `u0` (length `n`, periodic 1-D) to time `t_final`.
    ///
    /// `u0` may have any mean; the mean is preserved exactly.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if `u0.len() != n`.
    pub fn evolve(&self, u0: &[F], t_final: F) -> Result<Vec<F>, crate::SemiflowError> {
        if u0.len() != self.n {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "u0 length must equal n",
            });
        }
        Ok(burgers_cole_hopf_evolve(
            u0, self.n, self.dx, self.nu, t_final,
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Seam B — Strang-split reaction-diffusion
// ═══════════════════════════════════════════════════════════════════════════

/// Order-2 Strang-split reaction-diffusion evolver (S³ POC, Seam B).
///
/// Solves `u_t = ν Δu + f(u)` on a periodic `d`-D grid via the Strang sandwich
/// `react(τ/2) ∘ heat(τ) ∘ react(τ/2)`.  Reactions are restricted to the
/// closed-form-flow [`Reaction`] enum — generic/transcendental `f` is the
/// construction-time wall and is UNREPRESENTABLE.
///
/// ## Proven boundary
/// Order-2 ONLY for closed-form-flow reactions in [`Reaction`]. Generic/transcendental
/// `f` is the REACTION WALL: UNREPRESENTABLE by the enum (type enforcement).
/// Logistic IC must lie in `(0,1)` or construction fails (fail-loud). Proof:
/// `g_s3_nonlinear` (RELEASE-BLOCKING, `slow-tests`), Strang sub-gate slope ≤ −1.9.
/// See ADR-0168, math.md §53.5.
pub struct S3ReactionDiffusion<F: SemiflowFloat> {
    /// Grid size (same on every axis).
    n: usize,
    /// Number of spatial dimensions.
    d: usize,
    /// Grid spacing.
    dx: F,
    /// Diffusion coefficient `ν > 0`.
    nu: F,
    /// Reaction term (closed-form wall).
    reaction: Reaction<F>,
}

impl<F: SemiflowFloat> S3ReactionDiffusion<F> {
    /// Construct the evolver; validates domain and parabolicity fail-loud.
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if `n < 2`, `dx ≤ 0`,
    /// `nu ≤ 0`, or `d == 0`.
    pub fn new(
        n: usize,
        d: usize,
        dx: F,
        nu: F,
        reaction: Reaction<F>,
    ) -> Result<Self, crate::SemiflowError> {
        validate_rd(n, d, dx, nu)?;
        Ok(Self {
            n,
            d,
            dx,
            nu,
            reaction,
        })
    }

    /// Evolve `u0` (flat `n^d`) for `nsteps` steps of size `tau`.
    ///
    /// For `Reaction::Logistic`, each element must be in `(0, 1)` — the
    /// logistic domain wall is checked fail-loud here (promotes `debug_assert`).
    ///
    /// # Errors
    /// Returns [`crate::SemiflowError::S3OutOfClass`] if `u0.len() != n^d` or
    /// a Logistic IC is out of `(0,1)`.
    pub fn evolve(&self, u0: &[F], tau: F, nsteps: usize) -> Result<Vec<F>, crate::SemiflowError> {
        let nd = self.n.pow(self.d as u32);
        if u0.len() != nd {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "u0 length must equal n^d",
            });
        }
        check_logistic_domain(u0, &self.reaction)?;
        let cfg = StrangConfig {
            n: self.n,
            d: self.d,
            dx: self.dx,
            nu: self.nu,
            reaction: &self.reaction,
        };
        Ok(strang_rd_evolve(u0, &cfg, tau, nsteps))
    }
}

/// Validate common inputs for `S3ReactionDiffusion::new`.
fn validate_rd<F: SemiflowFloat>(
    n: usize,
    d: usize,
    dx: F,
    nu: F,
) -> Result<(), crate::SemiflowError> {
    if n < 2 {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "n must be >= 2",
        });
    }
    if d == 0 {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "d must be >= 1",
        });
    }
    if dx <= F::zero() {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "dx must be > 0",
        });
    }
    if nu <= F::zero() {
        return Err(crate::SemiflowError::S3OutOfClass {
            detail: "nu must be > 0 (parabolicity wall)",
        });
    }
    Ok(())
}

/// Release-mode domain check for Logistic: every IC must be in (0,1).
fn check_logistic_domain<F: SemiflowFloat>(
    u0: &[F],
    reaction: &Reaction<F>,
) -> Result<(), crate::SemiflowError> {
    if !matches!(reaction, Reaction::Logistic { .. }) {
        return Ok(());
    }
    for &ui in u0 {
        if ui <= F::zero() || ui >= F::one() {
            return Err(crate::SemiflowError::S3OutOfClass {
                detail: "Logistic IC must be in (0,1)",
            });
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "s3-poc"))]
mod tests {
    use super::*;

    // ── S3BurgersColeHopf ────────────────────────────────────────────────
    #[test]
    fn burgers_rejects_n_less_than_4() {
        assert!(S3BurgersColeHopf::<f64>::new(3, 0.1, 1.0).is_err());
    }

    #[test]
    fn burgers_rejects_dx_zero() {
        assert!(S3BurgersColeHopf::<f64>::new(8, 0.0, 1.0).is_err());
    }

    #[test]
    fn burgers_rejects_nu_zero() {
        assert!(S3BurgersColeHopf::<f64>::new(8, 0.1, 0.0).is_err());
    }

    #[test]
    fn burgers_accepts_valid() {
        assert!(S3BurgersColeHopf::<f64>::new(8, 0.1, 1.0).is_ok());
    }

    // ── S3ReactionDiffusion ──────────────────────────────────────────────
    #[test]
    fn rd_rejects_n_less_than_2() {
        let r = S3ReactionDiffusion::<f64>::new(1, 1, 0.1, 1.0, Reaction::Linear { c: 0.1 });
        assert!(r.is_err());
    }

    #[test]
    fn rd_rejects_nu_zero() {
        let r = S3ReactionDiffusion::<f64>::new(4, 1, 0.1, 0.0, Reaction::Linear { c: 0.1 });
        assert!(r.is_err());
    }

    #[test]
    fn rd_rejects_logistic_ic_out_of_domain() {
        let evolver =
            S3ReactionDiffusion::<f64>::new(4, 1, 0.1, 1.0, Reaction::Logistic { r: 1.0 }).unwrap();
        let bad_u0 = vec![1.5f64, 0.3, 0.4, 0.2]; // 1.5 ∉ (0,1)
        assert!(evolver.evolve(&bad_u0, 0.01, 1).is_err());
    }

    #[test]
    fn rd_rejects_u0_length_mismatch() {
        let evolver =
            S3ReactionDiffusion::<f64>::new(4, 1, 0.1, 1.0, Reaction::Linear { c: 0.1 }).unwrap();
        assert!(evolver.evolve(&[0.1f64; 3], 0.01, 1).is_err()); // 3 ≠ 4^1
    }

    #[test]
    fn rd_accepts_valid() {
        let evolver =
            S3ReactionDiffusion::<f64>::new(4, 1, 0.1, 1.0, Reaction::Linear { c: 0.1 }).unwrap();
        let u0 = vec![0.1f64; 4];
        assert!(evolver.evolve(&u0, 0.01, 1).is_ok());
    }
}
