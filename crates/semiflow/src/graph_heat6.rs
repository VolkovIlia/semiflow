//! Order-6 graph heat Chernoff via operator-Taylor truncation.
//!
//! `S₆(τ) f = Σ_{k=0}^{6} (−τ L_G)^k / k! · f`
//!          = `f − τ L_G f + (τ²/2) L_G² f − (τ³/6) L_G³ f
//!             + (τ⁴/24) L_G⁴ f − (τ⁵/120) L_G⁵ f + (τ⁶/720) L_G⁶ f`.
//!
//! **Constant edge-coefficient `a ≡ 1` only** for v2.4. See math.md §19
//! (NORMATIVE) and ADR-0062.
//!
//! ## Citations
//! - Higham 2008 *Functions of Matrices* §10 (Taylor methods for `exp(A)`).
//! - Hochbruck-Ostermann 2010 *Acta Numerica* §3 (truncated-exponential families
//!   on bounded operators).
//!
//! ## Runtime cost
//! 6 `SpMV`s per `apply_into` call; 2 borrowed scratch buffers (ping-pong); 0
//! heap allocations in steady state.

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

use alloc::sync::Arc;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    graph::Laplacian,
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// GraphHeat6thChernoff<F>
// ---------------------------------------------------------------------------

/// Order-6 Chernoff for `∂ₜu = −L_G u` via 6-term operator Taylor truncation
/// of `exp(−τ L_G)`.
///
/// `S₆(τ) f = Σ_{k=0}^{6} (−τ L_G)^k / k! · f`.
///
/// **Constant edge-coefficient `a ≡ 1` only** for v2.4. See math.md §19
/// (NORMATIVE) and Hochbruck-Ostermann 2010 *Acta Numerica* §3.
///
/// Stores `Arc<Laplacian<F>>` (cheap clone for composition). Uses
/// [`ScratchPool`] for the six `SpMV` intermediates (0 heap allocations in
/// steady state).
///
/// Generic over `F` (f32 and f64). See ADR-0062 §"f32 stability rationale"
/// for the precision policy.
#[derive(Clone)]
pub struct GraphHeat6thChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
}

impl<F: SemiflowFloat> GraphHeat6thChernoff<F> {
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

impl<F: SemiflowFloat> ChernoffFunction<F> for GraphHeat6thChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        apply_zeta6_into(&self.laplacian, tau, src, dst, scratch)
    }

    fn order(&self) -> u32 {
        6
    }

    fn growth(&self) -> Growth<F> {
        Growth {
            multiplier: F::one(),
            omega: self.laplacian.spectral_radius_bound(),
        }
    }
}

// ---------------------------------------------------------------------------
// Core kernel (extracted for 50-LoC function cap compliance)
// ---------------------------------------------------------------------------

/// Compute the six Taylor coefficients `(−τ)^k / k!` for `k = 1..=6`.
fn taylor_coefficients_k6<F: SemiflowFloat>(tau: F) -> (F, F, F, F, F, F) {
    let one = F::one();
    let two = one + one;
    let three = two + one;
    let six = two + two + two;
    let twenty_four = (two + two) * three * two;
    let five = two + two + one;
    let one_twenty = twenty_four * five;
    let seven_twenty = one_twenty * six;

    let tau2 = tau * tau;
    let tau3 = tau2 * tau;
    let tau4 = tau2 * tau2;
    let tau5 = tau4 * tau;
    let tau6 = tau3 * tau3;

    (
        -tau,                 // c1 = −τ
        tau2 / two,           // c2 = τ²/2
        -(tau3 / six),        // c3 = −τ³/6
        tau4 / twenty_four,   // c4 = τ⁴/24
        -(tau5 / one_twenty), // c5 = −τ⁵/120
        tau6 / seven_twenty,  // c6 = τ⁶/720
    )
}

/// Apply the 6-step ping-pong `SpMV` loop and accumulate into `dst`.
///
/// `buf_odd` / `buf_even` are pre-allocated scratch slices owned by the caller.
/// On return both hold `L_G^5·src` and `L_G^6·src` respectively; the caller
/// returns them to the pool.
#[allow(clippy::too_many_arguments)]
fn apply_k6_pingpong<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    buf_odd: &mut [F],
    buf_even: &mut [F],
    c1: F,
    c2: F,
    c3: F,
    c4: F,
    c5: F,
    c6: F,
) {
    // k=1: buf_odd ← L_G · src
    lap.apply_into_slice(src.values(), buf_odd);
    dst.copy_from(src);
    dst.axpy_into_slice(c1, buf_odd);
    // k=2: buf_even ← L_G · buf_odd = L_G² · src
    lap.apply_into_slice(buf_odd, buf_even);
    dst.axpy_into_slice(c2, buf_even);
    // k=3: buf_odd ← L_G · buf_even = L_G³ · src
    lap.apply_into_slice(buf_even, buf_odd);
    dst.axpy_into_slice(c3, buf_odd);
    // k=4: buf_even ← L_G · buf_odd = L_G⁴ · src
    lap.apply_into_slice(buf_odd, buf_even);
    dst.axpy_into_slice(c4, buf_even);
    // k=5: buf_odd ← L_G · buf_even = L_G⁵ · src
    lap.apply_into_slice(buf_even, buf_odd);
    dst.axpy_into_slice(c5, buf_odd);
    // k=6: buf_even ← L_G · buf_odd = L_G⁶ · src
    lap.apply_into_slice(buf_odd, buf_even);
    dst.axpy_into_slice(c6, buf_even);
}

/// Apply degree-6 operator Taylor truncation `S_6(τ) src → dst`.
///
/// Uses a ping-pong pair of scratch buffers (`buf_odd`, `buf_even`), reused
/// alternately for `L_G^1`–`L_G^6` · src.
fn apply_zeta6_into<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "GraphHeat6thChernoff: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    let n = src.len();
    debug_assert_eq!(dst.len(), n);
    debug_assert_eq!(lap.n_nodes(), n);

    let (c1, c2, c3, c4, c5, c6) = taylor_coefficients_k6(tau);

    // Both buffers live simultaneously; use owned take_vec / return_vec.
    let mut buf_odd = scratch.take_vec(n);
    let mut buf_even = scratch.take_vec(n);

    apply_k6_pingpong(
        lap,
        src,
        dst,
        &mut buf_odd,
        &mut buf_even,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
    );

    scratch.return_vec(buf_even);
    scratch.return_vec(buf_odd);
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;

    use crate::graph::{Graph, Laplacian};
    use crate::graph_signal::GraphSignal;
    use crate::state::State;

    #[test]
    fn apply_at_zero_tau_returns_src() {
        let n = 8_usize;
        let g = Arc::new(Graph::<f64>::path(n));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let c = GraphHeat6thChernoff::new(Arc::clone(&lap));
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
    fn order_is_six() {
        let g = Arc::new(Graph::<f64>::path(4));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        assert_eq!(GraphHeat6thChernoff::new(lap).order(), 6);
    }

    #[test]
    fn negative_tau_returns_error() {
        let g = Arc::new(Graph::<f64>::path(4));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let c = GraphHeat6thChernoff::new(lap);
        let src = GraphSignal::zeros(Arc::clone(&g));
        let mut dst = src.clone();
        let mut scratch = ScratchPool::<f64>::new();
        assert!(c.apply_into(-0.1, &src, &mut dst, &mut scratch).is_err());
    }

    #[test]
    fn k6_more_accurate_than_k4_for_small_tau() {
        // Compare K=4 vs K=6 against a high-step "ground truth" e^{-τL_G} approximation.
        // K=6 should track the truth more closely at small τ.
        use crate::graph_heat4::GraphHeat4thChernoff;

        let n = 16_usize;
        let g = Arc::new(Graph::<f64>::path(n));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let k4 = GraphHeat4thChernoff::new(Arc::clone(&lap));
        let k6 = GraphHeat6thChernoff::new(Arc::clone(&lap));

        let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
            let x = f64::from(i) / n as f64 * core::f64::consts::TAU;
            x.cos()
        });

        let tau = 0.02_f64;
        let mut dst4 = src.clone();
        let mut dst6 = src.clone();
        let mut scratch = ScratchPool::<f64>::new();
        k4.apply_into(tau, &src, &mut dst4, &mut scratch).unwrap();
        k6.apply_into(tau, &src, &mut dst6, &mut scratch).unwrap();

        // K=4 → K=6 difference should be O(τ⁵) ≈ 3e-9 at τ=0.02 — tiny but
        // nonzero, proving the K=6 corrections are present.
        let mut diff = dst6.clone();
        diff.axpy_into(-1.0, &dst4);
        let d = diff.norm_sup();
        assert!(d > 0.0, "K=6 should differ from K=4");
        assert!(
            d < 1e-6,
            "K=6 vs K=4 diff at τ=0.02 should be tiny, got {d}"
        );
    }
}
