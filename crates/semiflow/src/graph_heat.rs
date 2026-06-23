//! Graph heat Chernoff kernels: order-1 (leading) and order-2 (ζ-A Taylor).
//!
//! `S_1(τ) f = f − τ · L_G · f` (order-1, Wave 2.1A).
//! `S_2(τ) f = f − τ · L_G · f + (τ²/2) · L_G² · f` (order-2, Wave 2.1B).
//!
//! See ADR-0047 and `contracts/v2.1/wave-b-higher-order-graph.md` §1.
//!
//! ## Mathematical context (CITATION only — no new theorems)
//!
//! Per Pazy 1983 §1.3 Thm 1.3 and Engel-Nagel 2000 §III.5 Thm 5.2, the family
//! `S(τ) = I − τ L_G` is a Chernoff function for `−L_G`. The order-2 variant
//! adds the `(τ²/2) L_G²` term (Hochbruck-Ostermann 2010 *Acta Numerica* §3);
//! it is the operator Taylor truncation of `exp(−τ L_G)` at degree 2.
//! See `contracts/semiflow-core.math.md` §12.2–§12.3, §12.6.

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
// GraphHeatOrder
// ---------------------------------------------------------------------------

/// Variant selector for [`GraphHeatChernoff`].
///
/// Controls which Taylor truncation is applied per step.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum GraphHeatOrder {
    /// Order-1: `S(τ) = I − τ L_G`. Default (Wave 2.1A).
    Leading,
    /// Order-2 (constant `a ≡ 1`): `S(τ) = I − τ L_G + (τ²/2) L_G²`.
    ///
    /// Operator Taylor truncation of `exp(−τ L_G)` at degree 2.
    /// See math.md §12.6 (NORMATIVE).
    ZetaATaylor2,
}

// ---------------------------------------------------------------------------
// GraphHeatChernoff<F>
// ---------------------------------------------------------------------------

/// Order-1 or order-2 Chernoff function for `∂ₜu = −L_G u`.
///
/// - **Order-1** (default, Wave 2.1A): `S(τ) f = f − τ · L_G · f`.
/// - **Order-2** (Wave 2.1B): `S(τ) f = f − τ L_G f + (τ²/2) L_G² f`.
///
/// Use [`Self::new`] for order-1 and [`Self::with_zeta_a`] for order-2.
///
/// Stores an `Arc<Laplacian<F>>` — cheap clone for composition. Uses
/// [`ScratchPool`] to compute `L_G · f` (and `L_G² · f` for order-2) in
/// borrowed buffers per step (0 heap allocations in steady state).
#[derive(Clone)]
pub struct GraphHeatChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
    order_variant: GraphHeatOrder,
}

impl<F: SemiflowFloat> GraphHeatChernoff<F> {
    /// Order-1 constructor (unchanged from Wave 2.1A).
    pub fn new(laplacian: Arc<Laplacian<F>>) -> Self {
        Self {
            laplacian,
            order_variant: GraphHeatOrder::Leading,
        }
    }

    /// Convenience: wrap an owned Laplacian in `Arc` internally (order-1).
    pub fn from_owned(laplacian: Laplacian<F>) -> Self {
        Self {
            laplacian: Arc::new(laplacian),
            order_variant: GraphHeatOrder::Leading,
        }
    }

    /// **Order-2 constructor (constant `a ≡ 1` only).** Wave 2.1B addition.
    ///
    /// Builds `S(τ) = I − τ L_G + (τ²/2) L_G²` — Taylor truncation of
    /// `exp(−τ L_G)`. See math.md §12.6 (NORMATIVE) and Hochbruck-Ostermann
    /// 2010 *Acta Numerica* §3.
    ///
    /// Variable `a(v)` (heterogeneous edge coefficients) is OUT OF SCOPE for
    /// v2.1 — deferred to v2.2 pending operator-product derivation.
    ///
    /// # Runtime cost (per `apply_into` call, steady state)
    /// - 2 `SpMV`s (`L_G · src`, then `L_G · (L_G · src)`).
    /// - 2 `borrow_vec(N)` from the `ScratchPool` (recycled after warmup).
    /// - 0 heap allocations.
    pub fn with_zeta_a(laplacian: Arc<Laplacian<F>>) -> Self {
        Self {
            laplacian,
            order_variant: GraphHeatOrder::ZetaATaylor2,
        }
    }

    /// Convenience: wrap an owned Laplacian in `Arc` (order-2).
    pub fn from_owned_with_zeta_a(laplacian: Laplacian<F>) -> Self {
        Self {
            laplacian: Arc::new(laplacian),
            order_variant: GraphHeatOrder::ZetaATaylor2,
        }
    }

    /// Borrow the underlying Laplacian (debug / composition).
    #[must_use]
    pub fn laplacian(&self) -> &Laplacian<F> {
        &self.laplacian
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> ChernoffFunction<F> for GraphHeatChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        // Validate tau.
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "GraphHeatChernoff: tau must be finite and >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let n = src.len();
        debug_assert_eq!(
            dst.len(),
            n,
            "GraphHeatChernoff: dst.len() must match src.len()"
        );
        debug_assert_eq!(
            self.laplacian.n_nodes(),
            n,
            "GraphHeatChernoff: Laplacian size mismatch"
        );

        match self.order_variant {
            GraphHeatOrder::Leading => apply_leading(&self.laplacian, tau, src, dst, scratch),
            GraphHeatOrder::ZetaATaylor2 => {
                apply_zeta_a_taylor2(&self.laplacian, tau, src, dst, scratch)
            }
        }
    }

    fn order(&self) -> u32 {
        match self.order_variant {
            GraphHeatOrder::Leading => 1,
            GraphHeatOrder::ZetaATaylor2 => 2,
        }
    }

    fn growth(&self) -> Growth<F> {
        Growth {
            multiplier: F::one(),
            omega: self.laplacian.spectral_radius_bound(),
        }
    }
}

// ---------------------------------------------------------------------------
// Order-1 body (Wave 2.1A — unchanged)
// ---------------------------------------------------------------------------

#[allow(clippy::unnecessary_wraps)] // must return Result<> to match apply_into signature
fn apply_leading<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = src.len();
    let mut buf = scratch.borrow_vec(n);
    lap.apply_into_slice(src.values(), &mut buf);
    dst.copy_from(src);
    dst.axpy_into_slice(-tau, &buf);
    Ok(())
}

// ---------------------------------------------------------------------------
// Order-2 body (Wave 2.1B — ζ-A Taylor truncation, constant a ≡ 1)
// ---------------------------------------------------------------------------

#[allow(clippy::unnecessary_wraps)] // must return Result<> to match apply_into signature
fn apply_zeta_a_taylor2<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = src.len();
    // Two owned take_vec calls — both alive simultaneously (two buffers needed
    // in steady state). Must return after use to preserve zero-alloc property.
    let mut lap1 = scratch.take_vec(n); // L_G · src
    lap.apply_into_slice(src.values(), &mut lap1);

    let mut lap2 = scratch.take_vec(n); // L_G² · src
    lap.apply_into_slice(&lap1, &mut lap2);

    // dst ← src − τ · lap1 + (τ²/2) · lap2
    let half = F::one() / (F::one() + F::one());
    dst.copy_from(src);
    dst.axpy_into_slice(-tau, &lap1);
    dst.axpy_into_slice(half * tau * tau, &lap2);

    scratch.return_vec(lap2);
    scratch.return_vec(lap1);
    Ok(())
}
