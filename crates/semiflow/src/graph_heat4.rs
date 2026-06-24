//! Order-4 graph heat Chernoff via Padé\[0,4\] operator Taylor truncation.
//!
//! `S₄(τ) f = Σ_{k=0}^{4} (−τ L_G)^k / k! · f`
//!          = `f − τ L_G f + (τ²/2) L_G² f − (τ³/6) L_G³ f + (τ⁴/24) L_G⁴ f`.
//!
//! **Constant edge-coefficient `a ≡ 1` only** for v2.1. See math.md §12.7
//! (NORMATIVE) and Wave 2.1B contract §2.
//!
//! ## Citations
//! - Hochbruck-Ostermann 2010 *Acta Numerica* §3 (truncated-exponential families
//!   on bounded operators).
//! - Higham 2008 *Functions of Matrices* §10 (Taylor methods for `exp(A)`).
//!
//! ## Runtime cost
//! 4 `SpMV`s per `apply_into` call; 2 borrowed scratch buffers (ping-pong); 0
//! heap allocations in steady state.

use alloc::sync::Arc;

use crate::{
    adjoint::AdjointApply,
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    graph::Laplacian,
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// GraphHeat4thChernoff<F>
// ---------------------------------------------------------------------------

/// Order-4 Chernoff for `∂ₜu = −L_G u` via Padé\[0,4\] / 4-term operator
/// Taylor truncation of `exp(−τ L_G)`.
///
/// `S₄(τ) f = Σ_{k=0}^{4} (−τ L_G)^k / k! · f`.
///
/// **Constant edge-coefficient `a ≡ 1` only** for v2.1. See math.md §12.7
/// (NORMATIVE) and Hochbruck-Ostermann 2010 *Acta Numerica* §3.
///
/// Stores `Arc<Laplacian<F>>` (cheap clone for composition). Uses
/// [`ScratchPool`] for the four `SpMV` intermediates (0 heap allocations in
/// steady state).
pub struct GraphHeat4thChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
}

impl<F: SemiflowFloat> GraphHeat4thChernoff<F> {
    /// Construct from a shared Laplacian.
    pub fn new(laplacian: Arc<Laplacian<F>>) -> Self {
        Self { laplacian }
    }

    /// Convenience: wrap an owned Laplacian in `Arc` internally.
    pub fn from_owned(laplacian: Laplacian<F>) -> Self {
        Self {
            laplacian: Arc::new(laplacian),
        }
    }

    /// Borrow the underlying Laplacian.
    #[must_use]
    pub fn laplacian(&self) -> &Laplacian<F> {
        &self.laplacian
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> ChernoffFunction<F> for GraphHeat4thChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        apply_zeta4_into(&self.laplacian, tau, src, dst, scratch)
    }

    fn order(&self) -> u32 {
        4
    }

    fn growth(&self) -> Growth<F> {
        Growth {
            multiplier: F::one(),
            omega: self.laplacian.spectral_radius_bound(),
        }
    }
}

// ---------------------------------------------------------------------------
// AdjointApply for GraphHeat4thChernoff (genuinely self-adjoint — autonomous)
// ---------------------------------------------------------------------------

/// `AdjointApply` marker for `GraphHeat4thChernoff`.
///
/// `S₄*(τ) = S₄(τ)` because:
/// 1. The generator `−L_G` is **autonomous** (constant Laplacian, no time
///    variation) — there is no time-varying commutator term.
/// 2. `L_G` is the **symmetric combinatorial Laplacian** (`L_G = L_Gᵀ`), so
///    `(−L_G)ᵀ = −L_G` and `((−τ L_G)^k)ᵀ = (−τ L_G)^k`.
/// 3. Therefore each term of `S₄(τ) = Σ_{k=0}^{4} (−τ L_G)^k / k!` is
///    individually self-adjoint, and so is their sum.
///
/// This is the **honest** self-adjoint case: there is no antisymmetric
/// commutator correction (contrast `MagnusGraphHeatChernoff`, which has a
/// time-varying `Ω₄` whose `τ²·[L_i,L_j]` commutator is antisymmetric and
/// makes `exp(Ω₄)ᵀ ≠ exp(Ω₄)` at O(τ²)).
///
/// `apply_adjoint_into` therefore delegates to `apply_into` via the default
/// `ChernoffFunction` impl. See ADR-0114.
impl<F: SemiflowFloat> AdjointApply<F> for GraphHeat4thChernoff<F> {}

// ---------------------------------------------------------------------------
// Core kernel (extracted for 50-LoC function cap compliance)
// ---------------------------------------------------------------------------

/// Compute the four Taylor coefficients `(−τ)^k / k!` for `k = 1..=4`.
fn taylor_coefficients_k4<F: SemiflowFloat>(tau: F) -> (F, F, F, F) {
    let one = F::one();
    let two = one + one;
    let six = two + two + two;
    // 24 = 4! = (2+2) * (2+1) * 2
    let twenty_four = (two + two) * (two + one) * two;
    let tau2 = tau * tau;
    (
        -tau,                        // c1 = −τ
        tau2 / two,                  // c2 = τ²/2
        -(tau2 * tau) / six,         // c3 = −τ³/6
        (tau2 * tau2) / twenty_four, // c4 = τ⁴/24
    )
}

/// Apply Padé[0,4] operator Taylor truncation.
///
/// Uses a ping-pong pair of scratch buffers (`ping`, `pong`), reused
/// alternately for `L_G^1`, `L_G^2`, `L_G^3`, `L_G^4` · src.
fn apply_zeta4_into<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "GraphHeat4thChernoff: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    let n = src.len();
    debug_assert_eq!(dst.len(), n);
    debug_assert_eq!(lap.n_nodes(), n);

    let (c1, c2, c3, c4) = taylor_coefficients_k4(tau);

    // Two scratch buffers alternate: buf_odd holds L_G^(odd) · src,
    // buf_even holds L_G^(even) · src.  Both are live simultaneously.
    let mut buf_odd = scratch.take_vec(n);
    let mut buf_even = scratch.take_vec(n);

    // k=1: buf_odd ← L_G · src
    lap.apply_into_slice(src.values(), &mut buf_odd);
    // dst ← src + c1 * buf_odd
    dst.copy_from(src);
    dst.axpy_into_slice(c1, &buf_odd);

    // k=2: buf_even ← L_G · buf_odd = L_G² · src
    lap.apply_into_slice(&buf_odd, &mut buf_even);
    dst.axpy_into_slice(c2, &buf_even);

    // k=3: buf_odd ← L_G · buf_even = L_G³ · src
    lap.apply_into_slice(&buf_even, &mut buf_odd);
    dst.axpy_into_slice(c3, &buf_odd);

    // k=4: buf_even ← L_G · buf_odd = L_G⁴ · src
    lap.apply_into_slice(&buf_odd, &mut buf_even);
    dst.axpy_into_slice(c4, &buf_even);

    // Return both buffers to the pool.
    scratch.return_vec(buf_even);
    scratch.return_vec(buf_odd);

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit smoke test
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::*;
    use crate::{
        graph::{Graph, Laplacian},
        graph_signal::GraphSignal,
        state::State,
    };

    #[test]
    fn apply_at_zero_tau_returns_src() {
        let n = 8_usize;
        let g = Arc::new(Graph::<f64>::path(n));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let c = GraphHeat4thChernoff::new(Arc::clone(&lap));
        let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
        let mut dst = src.clone();
        let mut scratch = ScratchPool::<f64>::new();
        c.apply_into(0.0, &src, &mut dst, &mut scratch).unwrap();
        let mut diff = dst.clone();
        diff.axpy_into(-1.0, &src);
        assert!(
            diff.norm_sup() < 1e-14,
            "zero-tau should return src, got {}",
            diff.norm_sup()
        );
    }

    #[test]
    fn order_is_four() {
        let g = Arc::new(Graph::<f64>::path(4));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        assert_eq!(GraphHeat4thChernoff::new(lap).order(), 4);
    }

    #[test]
    fn negative_tau_returns_error() {
        let g = Arc::new(Graph::<f64>::path(4));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let c = GraphHeat4thChernoff::new(lap);
        let src = GraphSignal::zeros(Arc::clone(&g));
        let mut dst = src.clone();
        let mut scratch = ScratchPool::<f64>::new();
        assert!(c.apply_into(-0.1, &src, &mut dst, &mut scratch).is_err());
    }
}
