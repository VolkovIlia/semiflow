//! ADR-0122 — adaptive per-point Gauss-Hermite quadrature `q` for
//! `AnisotropicShiftChernoffND` (math.md §32.8).
//!
//! Ships as the ADDITIVE type `AnisotropicShiftAdaptiveQ` constructed via the
//! same interface as `AnisotropicShiftChernoffND` plus a `tol` parameter.
//! `order()` is UNCHANGED (1) — adaptivity is quadrature accuracy, not order lift.
//!
//! # Algorithm (math.md §32.8 / ADR-0122)
//!
//! For each grid point `x_k`, the per-step d-D GH integral is computed with an
//! isotropic tensor rule of size `q^D`. The selector picks the smallest
//! `q ∈ {3,5,7,9}` satisfying `|I_{q^D} − I_{(q-2)^D}| ≤ tol`, seeding with
//! q=1 (midpoint rule).
//!
//! # CRITICAL CAVEAT (ADR-0122)
//! The tolerance `tol` MUST equal the production target (default `1e-10`), NOT
//! machine-epsilon. With tol≈1e-16 the estimator over-refines transcendental
//! integrands to q=9 and the node-count saving evaporates (PRE-FLIGHT-confirmed).

use alloc::{boxed::Box, vec::Vec};
use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid_nd::{GridFnND, GridND},
    scratch::ScratchPool,
    shift_nd::{
        build_cholesky_cache, flat_to_x, AnisotropicShiftChernoffND, SquareMatrix, GH1_NODES_F64,
        GH1_WEIGHTS_F64, GH3_NODES_F64, GH3_WEIGHTS_F64, GH5_NODES_F64, GH5_WEIGHTS_F64,
        GH7_NODES_F64, GH7_WEIGHTS_F64, GH9_NODES_F64, GH9_WEIGHTS_F64,
    },
};

// ---------------------------------------------------------------------------
// GH level descriptor — (q, nodes slice, weights slice)
// ---------------------------------------------------------------------------

struct GhLevel {
    q: usize,
    nodes: &'static [f64],
    weights: &'static [f64],
}

const GH_LADDER: [GhLevel; 5] = [
    GhLevel {
        q: 1,
        nodes: &GH1_NODES_F64,
        weights: &GH1_WEIGHTS_F64,
    },
    GhLevel {
        q: 3,
        nodes: &GH3_NODES_F64,
        weights: &GH3_WEIGHTS_F64,
    },
    GhLevel {
        q: 5,
        nodes: &GH5_NODES_F64,
        weights: &GH5_WEIGHTS_F64,
    },
    GhLevel {
        q: 7,
        nodes: &GH7_NODES_F64,
        weights: &GH7_WEIGHTS_F64,
    },
    GhLevel {
        q: 9,
        nodes: &GH9_NODES_F64,
        weights: &GH9_WEIGHTS_F64,
    },
];

// ---------------------------------------------------------------------------
// AnisotropicShiftAdaptiveQ<F, D>
// ---------------------------------------------------------------------------

/// Adaptive per-point GH quadrature wrapper for the d-D anisotropic shift kernel.
///
/// `order()` = 1 (same as `AnisotropicShiftChernoffND`). The adaptive rule
/// reduces mean quadrature nodes per grid point vs the fixed q=5 baseline while
/// matching accuracy at the production tolerance.
///
/// **Default tol = 1e-10** — do NOT pass machine-epsilon (see ADR-0122).
pub struct AnisotropicShiftAdaptiveQ<F: SemiflowFloat = f64, const D: usize = 2> {
    /// Underlying kernel kept for `grid()` and `growth()`.
    inner: AnisotropicShiftChernoffND<F, D>,
    /// Cached Cholesky factors (same as inner, held separately for the apply loop).
    cholesky_cache: Vec<SquareMatrix<F, D>>,
    /// Drift vector closure (independent copy for the adaptive apply loop).
    #[allow(clippy::type_complexity)]
    b_i: Box<dyn Fn(&[F; D], &mut [F; D]) + Send + Sync>,
    /// Reaction coefficient closure.
    #[allow(clippy::type_complexity)]
    c_fn: Box<dyn Fn(&[F; D]) -> F + Send + Sync>,
    /// Adaptive estimator tolerance.
    tol: F,
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat, const D: usize> AnisotropicShiftAdaptiveQ<F, D> {
    /// Construct an adaptive-q kernel with the same interface as
    /// `AnisotropicShiftChernoffND::new` plus a `tol` parameter.
    ///
    /// **CRITICAL**: `tol` is the PRODUCTION tolerance (e.g. `1e-10`), NOT
    /// machine-epsilon. See ADR-0122 §CRITICAL CAVEAT.
    ///
    /// # Errors
    /// Same as `AnisotropicShiftChernoffND::new` (SPD check, grid min-nodes) plus
    /// `DomainViolation` if `tol <= 0`.
    pub fn new(
        a_ij: impl Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync + Clone + 'static,
        b_i: impl Fn(&[F; D], &mut [F; D]) + Send + Sync + Clone + 'static,
        c: impl Fn(&[F; D]) -> F + Send + Sync + Clone + 'static,
        grid: crate::grid_nd::GridND<F, D>,
        tol: F,
    ) -> Result<Self, SemiflowError> {
        if tol <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "AnisotropicShiftAdaptiveQ: tol must be > 0",
                value: tol.to_f64().unwrap_or(f64::NAN),
            });
        }
        // Build the Cholesky cache (also validates SPD + grid size).
        let cholesky_cache = build_cholesky_cache(&a_ij, &grid)?;
        // Build the inner kernel (passes the same validation).
        let b_i_copy = b_i.clone();
        let c_copy = c.clone();
        let inner = AnisotropicShiftChernoffND::new(a_ij, b_i, c, grid)?;
        Ok(Self {
            inner,
            cholesky_cache,
            b_i: Box::new(b_i_copy),
            c_fn: Box::new(c_copy),
            tol,
            _f: PhantomData,
        })
    }

    /// Return a shared reference to the kernel's grid.
    pub fn grid(&self) -> &GridND<F, D> {
        self.inner.grid()
    }

    /// Compute the d-D GH integral at `xk` for a given ladder level.
    #[allow(clippy::too_many_arguments)]
    fn gh_integral_at_level(
        &self,
        tau: F,
        src: &GridFnND<F, D>,
        xk: &[F; D],
        l_k: &SquareMatrix<F, D>,
        level: &GhLevel,
        two_sqrt_tau: F,
        inv_pi_dhalf: F,
    ) -> F {
        let q = level.q;
        let nodes: Vec<F> = level.nodes.iter().map(|&v| from_f64::<F>(v)).collect();
        let weights: Vec<F> = level.weights.iter().map(|&v| from_f64::<F>(v)).collect();
        let mut b_val = [F::zero(); D];
        (self.b_i)(xk, &mut b_val);
        let c_val = (self.c_fn)(xk);
        let exp_factor = (tau * c_val).exp();
        #[allow(clippy::cast_possible_truncation)]
        let total_q = q.pow(D as u32);
        let mut acc = F::zero();
        for qi in 0..total_q {
            let mut eta_q = [F::zero(); D];
            let mut w_prod = F::one();
            let mut rem = qi;
            for eta_slot in &mut eta_q {
                let digit = rem % q;
                rem /= q;
                *eta_slot = nodes[digit];
                w_prod *= weights[digit];
            }
            let mut l_eta = [F::zero(); D];
            l_k.mat_vec(&eta_q, &mut l_eta);
            let y_q: [F; D] =
                core::array::from_fn(|d| xk[d] + tau * b_val[d] + two_sqrt_tau * l_eta[d]);
            let fv = src.sample(&y_q).unwrap_or(F::zero());
            acc += w_prod * fv;
        }
        exp_factor * inv_pi_dhalf * acc
    }
}

impl<F: SemiflowFloat, const D: usize> AnisotropicShiftAdaptiveQ<F, D> {
    /// Compute the adaptive GH integral at one grid node.
    ///
    /// Ladder: q=1 seed, then upgrade q∈{3,5,7,9} until `|Iq − I_{q-2}| ≤ tol`.
    #[allow(clippy::too_many_arguments)]
    fn adaptive_gh_at_node(
        &self,
        tau: F,
        src: &GridFnND<F, D>,
        xk: &[F; D],
        l_k: &SquareMatrix<F, D>,
        two_sqrt_tau: F,
        inv_pi_dhalf: F,
    ) -> F {
        let mut prev =
            self.gh_integral_at_level(tau, src, xk, l_k, &GH_LADDER[0], two_sqrt_tau, inv_pi_dhalf);
        let mut result = prev;
        for level in GH_LADDER.iter().skip(1) {
            let iq =
                self.gh_integral_at_level(tau, src, xk, l_k, level, two_sqrt_tau, inv_pi_dhalf);
            result = iq;
            let diff = if iq > prev { iq - prev } else { prev - iq };
            if diff <= self.tol {
                break;
            }
            prev = iq;
        }
        result
    }
}

impl<F: SemiflowFloat, const D: usize> ChernoffFunction<F> for AnisotropicShiftAdaptiveQ<F, D> {
    type S = GridFnND<F, D>;

    /// Apply one step using per-point adaptive q selection.
    ///
    /// Selects `q* = min{q∈{3,5,7,9}: |I_{q^D}−I_{(q-2)^D}| ≤ tol}` per grid point,
    /// seeded with q=1.  Falls back to q=9 if no q satisfies the criterion.
    fn apply_into(
        &self,
        tau: F,
        src: &GridFnND<F, D>,
        dst: &mut GridFnND<F, D>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "AnisotropicShiftAdaptiveQ::apply_into: tau must be finite >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let grid = self.inner.grid();
        let ns: [usize; D] = core::array::from_fn(|d| grid.axes[d].n);
        let total = src.values.len();
        let two_sqrt_tau = from_f64::<F>(2.0) * tau.sqrt();
        #[allow(clippy::cast_precision_loss)]
        let inv_pi_dhalf =
            from_f64::<F>(core::f64::consts::PI).powf(from_f64::<F>(-(D as f64) / 2.0));

        for flat in 0..total {
            let xk = flat_to_x::<F, D>(flat, &ns, grid);
            let l_k = &self.cholesky_cache[flat];
            dst.values[flat] =
                self.adaptive_gh_at_node(tau, src, &xk, l_k, two_sqrt_tau, inv_pi_dhalf);
        }
        Ok(())
    }

    /// Order 1: adaptive quadrature is a quadrature-accuracy refinement, not an order lift.
    fn order(&self) -> u32 {
        1
    }

    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}
