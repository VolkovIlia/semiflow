//! Order-2 hard absorbing wall via the killed Dirichlet generator Cayley map.
//!
//! `KilledDirichletChernoff<F>` implements one Crank–Nicolson step of the
//! **killed Dirichlet generator** `L^R = ∂_x(a(x)∂_x) + b(x)∂_x` on a bounded
//! region `R` with the absorbing wall `u|_∂R = 0` baked into the generator's
//! **domain** — not as a post-multiply mask.
//!
//! ```text
//! U^{n+1} = (I − τ/2 · L^R)^{-1} (I + τ/2 · L^R) U^n   (math §44.ter eq. 44.ter.1)
//! ```
//!
//! ## Order-2 proof (math §44.ter, oracle `T_KILLED_GEN_CN`)
//!
//! The Cayley map of any generator is order-2 unconditionally (Hochbruck–Lubich
//! 2010 §3.4): `τ¹ = τ² = 0`, genuine `τ³` remainder. With `L^R` the hard wall
//! is a domain restriction (no mask, no rate ramp), so all three Amendment-1
//! obstructions (C3a idempotency, C3b factor accuracy, C3c boundary-layer blow-up)
//! are structurally absent. `scripts/killed_generator_cn_kit.py` (`T_KILLED_GEN_CN`,
//! CHECK A–D) proves this symbolically with exit 0.
//!
//! ## Solver reuse (ADR-0135 Am. 2, §2.2)
//!
//! Path A (preferred): calls `matrix_strang::block_thomas_solve::<f64, 1>` — the
//! verified block-Thomas solver — at `M = 1` with Dirichlet boundary rows. This
//! reuses the upstream solver unchanged (zero blast radius on shipped kernels; the
//! impact analysis confirms `block_thomas_solve` is HIGH risk only for modifiers, not
//! callers). `block_cn_diff_step` is NOT called; it bakes Neumann rows.
//!
//! ## References
//!
//! - ADR-0135 Amendment 2 (GO, 2026-06-07)
//! - `contracts/semiflow-core.math.md` §44.ter (NORMATIVE)
//! - `scripts/killed_generator_cn_kit.py` (`T_KILLED_GEN_CN`, oracle proof)

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    grid::Grid1D,
    grid_fn::GridFn1D,
    killing::KillingRegion,
    matrix_strang::block_thomas_solve,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// KilledDirichletChernoff<F>
// ---------------------------------------------------------------------------

/// Chernoff function for the hard absorbing wall via the killed Dirichlet
/// generator Cayley map (math §44.ter, ADR-0135 Amendment 2).
///
/// Implements `(I − τ/2 · L^R)^{-1} (I + τ/2 · L^R)` where `L^R` is the
/// discrete diffusion generator on region `R` with Dirichlet (absorbing) rows
/// baked into its domain. Order-2 (CHECK A: τ¹ = τ² = 0, τ³ ≠ 0). Growth
/// bound `ω = 0` (sub-Markov contraction).
///
/// ## Construction
///
/// - [`KilledDirichletChernoff::new`]: whole-grid endpoints as the absorbing
///   wall (nodes `0` and `n−1` are fixed at `u = 0`). Simplest and what the
///   gate test uses.
/// - [`KilledDirichletChernoff::with_region`]: absorbing on an explicit
///   `KillingRegion` (nodes outside `R` are the wall).
pub struct KilledDirichletChernoff {
    /// Diffusivity closure `a(x) > 0`.
    a: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Drift closure `b(x)` (use `|_| 0.0` for none).
    b: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Reference grid.
    pub grid: Grid1D<f64>,
    /// Indices of boundary/wall nodes (sorted, deduplicated).
    wall_nodes: Vec<usize>,
}

impl KilledDirichletChernoff {
    /// Whole-grid endpoints absorbing: nodes `0` and `n−1` are the absorbing
    /// wall (`u = 0`). Interior nodes are `1..n−2`.
    ///
    /// # Errors
    ///
    /// Returns `DomainViolation` if `n < 3` (need at least one interior node)
    /// or if `a` evaluates to `≤ 0` or non-finite at the grid centre.
    pub fn new(
        a: impl Fn(f64) -> f64 + Send + Sync + 'static,
        b: impl Fn(f64) -> f64 + Send + Sync + 'static,
        grid: Grid1D<f64>,
    ) -> Result<Self, SemiflowError> {
        if grid.n < 3 {
            return Err(SemiflowError::DomainViolation {
                what: "KilledDirichletChernoff requires grid.n >= 3 (one interior node)",
                value: grid.n as f64,
            });
        }
        let x_c = grid.x_at(grid.n / 2);
        let a_c = a(x_c);
        if !a_c.is_finite() || a_c <= 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "KilledDirichletChernoff: a(x) must be > 0 and finite at grid centre",
                value: a_c,
            });
        }
        Ok(Self {
            a: Box::new(a),
            b: Box::new(b),
            grid,
            wall_nodes: vec![0, grid.n - 1],
        })
    }

    /// Absorbing on an explicit `KillingRegion` `R`: nodes outside `R` are the
    /// absorbing wall. Nodes inside `R` are the interior (active) domain.
    ///
    /// # Errors
    ///
    /// Returns `DomainViolation` if fewer than 1 interior node exists, or if
    /// `a` evaluates non-positively at the grid centre.
    pub fn with_region<R: KillingRegion<f64>>(
        a: impl Fn(f64) -> f64 + Send + Sync + 'static,
        b: impl Fn(f64) -> f64 + Send + Sync + 'static,
        grid: Grid1D<f64>,
        region: &R,
    ) -> Result<Self, SemiflowError> {
        let n = grid.n;
        let wall: Vec<usize> = (0..n)
            .filter(|&k| !region.is_inside(&[grid.x_at(k)]))
            .collect();
        let interior_count = n - wall.len();
        if interior_count < 1 {
            return Err(SemiflowError::DomainViolation {
                what: "KilledDirichletChernoff: region leaves no interior nodes",
                value: interior_count as f64,
            });
        }
        let x_c = grid.x_at(n / 2);
        let a_c = a(x_c);
        if !a_c.is_finite() || a_c <= 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "KilledDirichletChernoff: a(x) must be > 0 and finite at grid centre",
                value: a_c,
            });
        }
        Ok(Self {
            a: Box::new(a),
            b: Box::new(b),
            grid,
            wall_nodes: wall,
        })
    }

    /// Test whether node `k` is a wall (Dirichlet) node.
    #[inline]
    fn is_wall(&self, k: usize) -> bool {
        // wall_nodes is sorted; binary search is O(log W).
        self.wall_nodes.binary_search(&k).is_ok()
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<f64> impl
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for KilledDirichletChernoff {
    type S = GridFn1D<f64>;

    /// Genuine order-2 (math §44.ter.2): Cayley map of L^R, τ¹=τ²=0, τ³≠0.
    fn order(&self) -> u32 {
        2
    }

    /// Sub-Markov contraction: `‖e^{τL^R}‖ ≤ 1` for the killed semigroup.
    fn growth(&self) -> Growth<f64> {
        Growth::new(1.0, 0.0)
    }

    /// One Crank–Nicolson Cayley step of the killed Dirichlet generator.
    ///
    /// Assembles the explicit half `RHS = (I + τ/2 · L^R) · src` and solves
    /// the implicit half `(I − τ/2 · L^R) · dst = RHS` via `block_thomas_solve`
    /// at `M = 1` with Dirichlet boundary rows.
    ///
    /// # Errors
    ///
    /// Returns `DomainViolation` if `tau` is non-finite or `< 0`.
    // sub_blk/sup_blk/main_blk are standard tridiagonal block-Thomas names.
    #[allow(clippy::similar_names)]
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "KilledDirichletChernoff: tau must be finite and >= 0",
                value: tau,
            });
        }
        let n = self.grid.n;
        let dx = self.grid.dx();
        let half_tau = tau * 0.5;
        let dx2 = dx * dx;
        let two_dx = 2.0 * dx;
        let mut sub_blk: Vec<[[f64; 1]; 1]> = vec![[[0.0]]; n];
        let mut main_blk: Vec<[[f64; 1]; 1]> = vec![[[1.0]]; n];
        let mut sup_blk: Vec<[[f64; 1]; 1]> = vec![[[0.0]]; n];
        let mut rhs: Vec<[f64; 1]> = vec![[0.0]; n];
        let mut sol: Vec<[f64; 1]> = vec![[0.0]; n];
        for k in 0..n {
            assemble_cn_row(
                self,
                k,
                n,
                half_tau,
                dx2,
                two_dx,
                src,
                &mut sub_blk,
                &mut main_blk,
                &mut sup_blk,
                &mut rhs,
            );
        }
        block_thomas_solve::<f64, 1>(&sub_blk, &main_blk, &sup_blk, &rhs, &mut sol, n)?;
        for (v, s) in dst.values.iter_mut().zip(sol[..n].iter()) {
            *v = s[0];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: per-row CN stencil assembly
// ---------------------------------------------------------------------------

/// Assemble one row `k` of the Crank–Nicolson system for the killed Dirichlet
/// generator.
///
/// Wall rows produce identity LHS + pass-through RHS (absorbing datum stays 0).
/// Interior rows produce the 3-pt diffusion stencil (§44.ter §2.1).
///
/// Extracted from `apply_into` to satisfy the 50-line function cap.
// sub_blk/sup_blk and sub_coef/sup_coef are standard tridiagonal stencil names.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn assemble_cn_row(
    kern: &KilledDirichletChernoff,
    k: usize,
    n: usize,
    half_tau: f64,
    dx2: f64,
    two_dx: f64,
    src: &GridFn1D<f64>,
    sub_blk: &mut [[[f64; 1]; 1]],
    main_blk: &mut [[[f64; 1]; 1]],
    sup_blk: &mut [[[f64; 1]; 1]],
    rhs: &mut [[f64; 1]],
) {
    if kern.is_wall(k) {
        // Dirichlet row: L^R[k,:] = 0 → identity in both LHS and RHS.
        main_blk[k] = [[1.0]];
        rhs[k] = [src.values[k]];
        return;
    }
    let x = kern.grid.x_at(k);
    let ak = (kern.a)(x);
    let bk = (kern.b)(x);
    // 3-point interior stencil of L^R (§44.ter §2.1):
    let sub_coef = if kern.is_wall(k.saturating_sub(1)) || k == 0 {
        0.0
    } else {
        ak / dx2 - bk / two_dx
    };
    let diag_coef = -2.0 * ak / dx2;
    let sup_coef = if kern.is_wall(k + 1) || k + 1 >= n {
        0.0
    } else {
        ak / dx2 + bk / two_dx
    };
    // LHS: (I − τ/2 · L^R) stencil.
    if k > 0 && !kern.is_wall(k.saturating_sub(1)) {
        sub_blk[k - 1] = [[-half_tau * sub_coef]];
    }
    main_blk[k] = [[1.0 - half_tau * diag_coef]];
    if k + 1 < n && !kern.is_wall(k + 1) {
        sup_blk[k] = [[-half_tau * sup_coef]];
    }
    // RHS: (I + τ/2 · L^R) · src.
    let u_prev = if k > 0 { src.values[k - 1] } else { 0.0 };
    let u_curr = src.values[k];
    let u_next = if k + 1 < n { src.values[k + 1] } else { 0.0 };
    let lhu = sub_coef * u_prev + diag_coef * u_curr + sup_coef * u_next;
    rhs[k] = [u_curr + half_tau * lhu];
}

// ---------------------------------------------------------------------------
// Unit tests (5.2 — mandatory items 1–3)
// ---------------------------------------------------------------------------
#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::KilledDirichletChernoff;
    use crate::{
        chernoff::ChernoffFunction, grid::Grid1D, grid_fn::GridFn1D, scratch::ScratchPool,
        state::State,
    };

    fn make_kernel(n: usize) -> KilledDirichletChernoff {
        let grid = Grid1D::new(0.0, 1.0, n).unwrap();
        KilledDirichletChernoff::new(|_| 0.5, |_| 0.0, grid).unwrap()
    }

    // Item 3: order() == 2 and F(0) = I on interior.
    #[test]
    fn order_is_2() {
        let kern = make_kernel(16);
        assert_eq!(kern.order(), 2);
    }

    // F(tau=0) acts as identity on interior nodes.
    #[test]
    fn zero_tau_is_identity() {
        let n = 16;
        let kern = make_kernel(n);
        let grid = Grid1D::new(0.0, 1.0, n).unwrap();
        // IC: linear ramp vanishing at endpoints.
        let src = GridFn1D::from_fn(grid, |x| x * (1.0 - x));
        let mut dst = GridFn1D::from_fn(grid, |_| 0.0);
        let mut scratch = ScratchPool::new();
        kern.apply_into(0.0, &src, &mut dst, &mut scratch).unwrap();
        // Interior nodes should be unchanged (wall nodes are 0 in both src and dst).
        for k in 1..n - 1 {
            let diff = (dst.values[k] - src.values[k]).abs();
            assert!(diff < 1e-13, "node {k}: F(0)u ≠ u, diff = {diff}");
        }
    }

    // Item 1: hard BC exact — after one step with a datum that is EXACTLY 0
    // on ∂R, boundary nodes remain EXACTLY 0.0 (structural identity, §2.2).
    #[test]
    fn hard_bc_exact() {
        let n = 32;
        let kern = make_kernel(n);
        let grid = Grid1D::new(0.0, 1.0, n).unwrap();
        // IC: parabola x(1-x) — exactly 0 at nodes 0 and n-1.
        let src = GridFn1D::from_fn(grid, |x| x * (1.0 - x));
        // Verify the src datum is exactly 0 at endpoints (x=0.0 and x=1.0).
        assert_eq!(src.values[0], 0.0);
        assert_eq!(src.values[n - 1], 0.0);
        let mut dst = GridFn1D::from_fn(grid, |_| 99.0); // sentinel
        let mut scratch = ScratchPool::new();
        kern.apply_into(1e-3, &src, &mut dst, &mut scratch).unwrap();
        assert_eq!(dst.values[0], 0.0, "left wall must be exactly 0.0");
        assert_eq!(dst.values[n - 1], 0.0, "right wall must be exactly 0.0");
    }

    // Item 2: contraction — ‖U^{n+1}‖_∞ ≤ ‖U^n‖_∞ (sub-Markov, growth ω=0).
    #[test]
    fn contraction() {
        let n = 64;
        let kern = make_kernel(n);
        let grid = Grid1D::new(0.0, 1.0, n).unwrap();
        let src = GridFn1D::from_fn(grid, |x| x * (1.0 - x));
        let mut dst = GridFn1D::from_fn(grid, |_| 0.0);
        let mut scratch = ScratchPool::new();
        kern.apply_into(1e-2, &src, &mut dst, &mut scratch).unwrap();
        let norm_in = src.norm_sup();
        let norm_out = dst.norm_sup();
        assert!(
            norm_out <= norm_in + 1e-14,
            "contraction violated: ‖out‖={norm_out} > ‖in‖={norm_in}"
        );
    }

    // Smoke: finite output for a multi-step iteration.
    #[test]
    fn smoke_multi_step_finite() {
        let n = 32;
        let kern = make_kernel(n);
        let grid = Grid1D::new(0.0, 1.0, n).unwrap();
        let mut state = GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin());
        let mut scratch = ScratchPool::new();
        let mut dst = GridFn1D::from_fn(grid, |_| 0.0);
        for _ in 0..10 {
            kern.apply_into(1e-3, &state, &mut dst, &mut scratch)
                .unwrap();
            core::mem::swap(&mut state, &mut dst);
        }
        assert!(
            state.values.iter().all(|v| v.is_finite()),
            "non-finite output"
        );
    }
}
