//! Graph state-adjoint for the truncated Magnus K=4 map (Issue #2, ADR-0115).
//!
//! Implements the transpose of the *implemented finite Taylor map*
//!
//! ```text
//! S(τ) = Σ_{m=0..4} Ω₄(τ)^m / m!
//! ```
//!
//! via term-by-term transposition.  For symmetric node operators (combinatorial
//! Laplacian, `VarCoef` `L_a = √a · L · √a`) the transposed exponent equals Ω₄
//! with ONLY the commutator coefficient sign-flipped (math.md §42, Theorem 42.1):
//!
//! ```text
//! Ω₄ᵀ = (τ/2)(A₁+A₂) − (√3τ²/12)·[A₂,A₁]   =: Ω₄⋆
//! S⋆(τ) = Σ_{m=0..4} (Ω₄ᵀ)^m / m!
//! ```
//!
//! This is the state-adjoint of the IMPLEMENTED finite map, NOT of `exp(Ω₄)`.
//!
//! ## Cost
//! Same as the forward step (≤50 lines per function, R4 zero-alloc preserved).
//!
//! ## Kernel reuse
//! `apply_omega4` / `apply_omega4_la` are parameterised by `comm_sign: F`
//! (±1); adjoint uses `comm_sign = −1`.

use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    float::from_f64,
    graph::Laplacian,
    graph_signal::GraphSignal,
    magnus_graph::{apply_omega4, MagnusGraphHeatChernoff, GL4_C1_F64, GL4_C2_F64},
    scratch::ScratchPool,
    state::State,
    varcoef_magnus_graph::{apply_omega4_la, VarCoefMagnusGraphHeatChernoff},
    SemiflowFloat,
};

// ---------------------------------------------------------------------------
// Adjoint kernel — sign-flipped Ω₄
// ---------------------------------------------------------------------------

/// Accumulate one Taylor term: `dst += (1/fac) · Ω₄ᵀ^m · src`.
/// Mutates `omega_v` (next power) and `omega_pow` (scratch for current power).
/// Verbatim float ops from the original loop body; order unchanged.
#[allow(clippy::too_many_arguments)]
fn accumulate_adj_term<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    minus_one: F,
    fac: F,
    omega_v: &mut [F],
    omega_pow: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
    dst: &mut GraphSignal<F>,
) {
    omega_pow.copy_from_slice(omega_v);
    apply_omega4(
        lap1, lap2, tau, minus_one, omega_pow, omega_v, tmp_a, tmp_b, tmp_c,
    );
    dst.axpy_into_slice(F::one() / fac, omega_v);
}

/// Inner Taylor accumulation body for `apply_exp_omega4_adj_kernel` (all slices pre-allocated).
/// Verbatim float op order from the original implementation.
#[allow(clippy::too_many_arguments)]
fn adj_taylor_body<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    omega_v: &mut [F],
    omega_pow: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
) {
    let one = F::one();
    let minus_one = -one;
    let two = one + one;
    apply_omega4(
        lap1,
        lap2,
        tau,
        minus_one,
        src.values(),
        omega_v,
        tmp_a,
        tmp_b,
        tmp_c,
    );
    dst.copy_from(src);
    dst.axpy_into_slice(one, omega_v);
    let factorials = [two, two + two + two, (two + two) * (two + one) * two];
    for fac in factorials {
        accumulate_adj_term(
            lap1, lap2, tau, minus_one, fac, omega_v, omega_pow, tmp_a, tmp_b, tmp_c, dst,
        );
    }
}

/// Apply degree-4 Taylor of `S⋆(τ)`: `dst ← S⋆(τ) · src`.
/// R4 zero-alloc: 5 scratch buffers acquired and returned.
fn apply_exp_omega4_adj_kernel<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) {
    let n = src.len();
    let mut omega_v = scratch.take_vec(n);
    let mut omega_pow = scratch.take_vec(n);
    let mut tmp_a = scratch.take_vec(n);
    let mut tmp_b = scratch.take_vec(n);
    let mut tmp_c = scratch.take_vec(n);
    adj_taylor_body(
        lap1,
        lap2,
        tau,
        src,
        dst,
        &mut omega_v,
        &mut omega_pow,
        &mut tmp_a,
        &mut tmp_b,
        &mut tmp_c,
    );
    scratch.return_vec(tmp_c);
    scratch.return_vec(tmp_b);
    scratch.return_vec(tmp_a);
    scratch.return_vec(omega_pow);
    scratch.return_vec(omega_v);
}

// ---------------------------------------------------------------------------
// MagnusGraphHeatChernoff adjoint methods
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> MagnusGraphHeatChernoff<F> {
    /// One state-adjoint step at absolute time `t_start`:
    /// `dst := S⋆(t_start, τ) · src` where `S⋆ = Σ_{m=0..4} (Ω₄ᵀ)^m / m!`
    /// (math.md §42, Theorem 42.1).
    ///
    /// This is the transpose of the IMPLEMENTED finite Taylor map.
    /// Same cost and error conditions as `apply_into_at`.
    ///
    /// # Errors
    /// Same as `apply_into_at`: `DomainViolation`, `OutOfMagnusRadius`.
    pub fn apply_state_adjoint_into_at(
        &self,
        t_start: F,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        crate::magnus_graph::validate_tau(tau)?;
        if self.convergence_radius_check {
            crate::magnus_graph::validate_magnus_radius(self.rho_bar_max, tau)?;
        }
        let c1 = from_f64::<F>(GL4_C1_F64);
        let c2 = from_f64::<F>(GL4_C2_F64);
        let lap1 = (self.lap_at_t)(t_start + c1 * tau);
        let lap2 = (self.lap_at_t)(t_start + c2 * tau);
        apply_exp_omega4_adj_kernel(&lap1, &lap2, tau, src, dst, scratch);
        Ok(())
    }

    /// Convenience wrapper: `t_start = 0` (consistent with `apply_into`).
    ///
    /// # Errors
    /// Propagates any error from `apply_state_adjoint_into_at`.
    pub fn apply_state_adjoint_into(
        &self,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.apply_state_adjoint_into_at(F::zero(), tau, src, dst, scratch)
    }

    /// Backward costate sweep: `n_steps` steps of size `tau`.
    ///
    /// Terminal value `lambda_n` in `src`; result `λ_0` written to `dst`.
    /// Zero-alloc: no heap allocation beyond the two signal ping-pong buffers.
    ///
    /// # Errors
    /// Returns [`SemiflowError`] if any adjoint step fails.
    pub fn evolve_state_adjoint_into(
        &self,
        tau: F,
        n_steps: usize,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if n_steps == 0 {
            dst.copy_from(src);
            return Ok(());
        }
        let mut lam = src.clone();
        let mut lam_next = dst.clone();
        // Backward: step n → n-1 → … → 0.
        for k in 0..n_steps {
            // t_start for step k counts from the end: last k uses t_start=0.
            #[allow(clippy::cast_precision_loss)]
            let t_s = from_f64::<F>((n_steps - 1 - k) as f64) * tau;
            self.apply_state_adjoint_into_at(t_s, tau, &lam, &mut lam_next, scratch)?;
            core::mem::swap(&mut lam, &mut lam_next);
        }
        dst.copy_from(&lam);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VarCoefMagnusGraphHeatChernoff adjoint methods
// ---------------------------------------------------------------------------

/// Accumulate one Taylor term for the variable-coefficient adjoint.
/// Mirrors `accumulate_adj_term` but with the 7-arg `apply_omega4_la` call.
#[allow(clippy::too_many_arguments)]
fn accumulate_la_adj_term<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    sqrt_a1: &[F],
    lap2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    minus_one: F,
    fac: F,
    omega_v: &mut [F],
    omega_pow: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
    vt1: &mut [F],
    vt2: &mut [F],
    dst: &mut GraphSignal<F>,
) {
    omega_pow.copy_from_slice(omega_v);
    apply_omega4_la(
        lap1, sqrt_a1, lap2, sqrt_a2, tau, minus_one, omega_pow, omega_v, tmp_a, tmp_b, tmp_c, vt1,
        vt2,
    );
    dst.axpy_into_slice(F::one() / fac, omega_v);
}

/// Inner Taylor accumulation body for `apply_exp_omega4_la_adj_kernel` (slices pre-allocated).
/// Verbatim float op order; accumulates the degree-4 Taylor of Ω₄ᵀ (la variant) into `dst`.
#[allow(clippy::too_many_arguments)]
fn la_adj_taylor_body<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    sqrt_a1: &[F],
    lap2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    omega_v: &mut [F],
    omega_pow: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
    vt1: &mut [F],
    vt2: &mut [F],
) {
    let one = F::one();
    let minus_one = -one;
    let two = one + one;
    apply_omega4_la(
        lap1,
        sqrt_a1,
        lap2,
        sqrt_a2,
        tau,
        minus_one,
        src.values(),
        omega_v,
        tmp_a,
        tmp_b,
        tmp_c,
        vt1,
        vt2,
    );
    dst.copy_from(src);
    dst.axpy_into_slice(one, omega_v);
    let factorials = [two, two + two + two, (two + two) * (two + one) * two];
    for fac in factorials {
        accumulate_la_adj_term(
            lap1, sqrt_a1, lap2, sqrt_a2, tau, minus_one, fac, omega_v, omega_pow, tmp_a, tmp_b,
            tmp_c, vt1, vt2, dst,
        );
    }
}

/// Adjoint kernel for variable-coefficient operator. R4 zero-alloc: 7 scratch buffers.
#[allow(clippy::too_many_arguments)]
fn apply_exp_omega4_la_adj_kernel<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    sqrt_a1: &[F],
    lap2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) {
    let n = src.len();
    let mut omega_v = scratch.take_vec(n);
    let mut omega_pow = scratch.take_vec(n);
    let mut tmp_a = scratch.take_vec(n);
    let mut tmp_b = scratch.take_vec(n);
    let mut tmp_c = scratch.take_vec(n);
    let mut vt1 = scratch.take_vec(n);
    let mut vt2 = scratch.take_vec(n);
    la_adj_taylor_body(
        lap1,
        sqrt_a1,
        lap2,
        sqrt_a2,
        tau,
        src,
        dst,
        &mut omega_v,
        &mut omega_pow,
        &mut tmp_a,
        &mut tmp_b,
        &mut tmp_c,
        &mut vt1,
        &mut vt2,
    );
    scratch.return_vec(vt2);
    scratch.return_vec(vt1);
    scratch.return_vec(tmp_c);
    scratch.return_vec(tmp_b);
    scratch.return_vec(tmp_a);
    scratch.return_vec(omega_pow);
    scratch.return_vec(omega_v);
}

impl<F: SemiflowFloat> VarCoefMagnusGraphHeatChernoff<F> {
    /// One state-adjoint step at `t_start`:
    /// `dst := S⋆(t_start, τ) · src` (math.md §42, Theorem 42.1, `VarCoef` case).
    ///
    /// # Errors
    /// Same as `apply_into_at`.
    pub fn apply_state_adjoint_into_at(
        &self,
        t_start: F,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        crate::varcoef_magnus_graph::validate_tau(tau)?;
        if self.convergence_radius_check {
            crate::varcoef_magnus_graph::validate_magnus_radius(
                self.rho_bar_max,
                self.a_sup_max,
                tau,
            )?;
        }
        let n = self.n_nodes;
        let c1 = from_f64::<F>(GL4_C1_F64);
        let c2 = from_f64::<F>(GL4_C2_F64);
        let lap1 = (self.lap_at_t)(t_start + c1 * tau);
        let lap2 = (self.lap_at_t)(t_start + c2 * tau);
        let a1 = (self.a_at_t)(t_start + c1 * tau);
        let a2 = (self.a_at_t)(t_start + c2 * tau);
        let sqrt_a1: Vec<F> = a1.iter().map(|&x| x.sqrt()).collect();
        let sqrt_a2: Vec<F> = a2.iter().map(|&x| x.sqrt()).collect();
        validate_varcoef_sqrt_a(&sqrt_a1, &sqrt_a2)?;
        let mut sa1_buf = scratch.take_vec(n);
        let mut sa2_buf = scratch.take_vec(n);
        sa1_buf.copy_from_slice(&sqrt_a1);
        sa2_buf.copy_from_slice(&sqrt_a2);
        apply_exp_omega4_la_adj_kernel(&lap1, &sa1_buf, &lap2, &sa2_buf, tau, src, dst, scratch);
        scratch.return_vec(sa2_buf);
        scratch.return_vec(sa1_buf);
        Ok(())
    }

    /// Convenience wrapper: `t_start = 0`.
    ///
    /// # Errors
    /// Propagates any error from `apply_state_adjoint_into_at`.
    pub fn apply_state_adjoint_into(
        &self,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.apply_state_adjoint_into_at(F::zero(), tau, src, dst, scratch)
    }

    /// Backward costate sweep: `n_steps` steps of size `tau`.
    /// Terminal `lambda_n` in `src`; `λ_0` written to `dst`. Zero-alloc.
    ///
    /// # Errors
    /// Returns [`SemiflowError`] if any adjoint step fails.
    pub fn evolve_state_adjoint_into(
        &self,
        tau: F,
        n_steps: usize,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if n_steps == 0 {
            dst.copy_from(src);
            return Ok(());
        }
        let mut lam = src.clone();
        let mut lam_next = dst.clone();
        for k in 0..n_steps {
            #[allow(clippy::cast_precision_loss)]
            let t_s = from_f64::<F>((n_steps - 1 - k) as f64) * tau;
            self.apply_state_adjoint_into_at(t_s, tau, &lam, &mut lam_next, scratch)?;
            core::mem::swap(&mut lam, &mut lam_next);
        }
        dst.copy_from(&lam);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared validation helper
// ---------------------------------------------------------------------------

/// Validate `sqrt_a1`/`sqrt_a2`: all entries must be finite and positive.
fn validate_varcoef_sqrt_a<F: SemiflowFloat>(
    sqrt_a1: &[F],
    sqrt_a2: &[F],
) -> Result<(), SemiflowError> {
    for &ai in sqrt_a1.iter().chain(sqrt_a2.iter()) {
        if !ai.is_finite() || ai <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "VarCoef adjoint: a(t) not positive/finite",
                value: ai.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/magnus_graph_adjoint_tests.rs"
));
