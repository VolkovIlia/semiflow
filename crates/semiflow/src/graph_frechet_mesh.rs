//! Two-sided graded Duhamel mesh (ADR-0203, math §63.3).
//!
//! `[0, t]` is split at `H = t/2`; each half is graded geometrically towards its
//! outer endpoint. Nodes are carried as DISTANCES to that endpoint (never as
//! `σ = t − r`, §63.3 "Distances, never positions"). The mesh is built by a
//! multiply loop (no `ln`/`powf`), so it is bit-identical in every build
//! (ADR-0200).

use alloc::vec::Vec;

use crate::{
    float::{from_f64, SemiflowFloat},
    graph_krylov::{graph_expmv_matvec_count, KrylovPath},
};

/// Geometric grading ratio `q` of §63.3.
pub(crate) const Q_RATIO: f64 = 1.5;
/// Innermost panel length in units of `1/ρ̄` (`δ` of §63.3).
pub(crate) const DELTA: f64 = 2.0;
/// Defensive cap on panels per half (`K ≤ 200`; unreachable for `ρ̄t < 1e35`).
const MAX_PANELS: usize = 201;

/// 8-point Gauss-Legendre rule on `[0, 1]`: `(node, weight)`, weights sum to 1.
/// Source: Abramowitz & Stegun §25.4.29.
pub(crate) const GL8: [(f64, f64); 8] = [
    (0.019_855_071_751_231_88, 0.050_614_268_145_188_29),
    (0.101_666_761_293_186_65, 0.111_190_517_226_687_24),
    (0.237_233_795_041_835_5, 0.156_853_322_938_943_64),
    (0.408_282_678_752_175_1, 0.181_341_891_689_180_6),
    (0.591_717_321_247_825, 0.181_341_891_689_180_6),
    (0.762_766_204_958_164_5, 0.156_853_322_938_943_64),
    (0.898_333_238_706_813_4, 0.111_190_517_226_687_24),
    (0.980_144_928_248_768_1, 0.050_614_268_145_188_29),
];

/// First panel end `d_0 = min(H, δ/ρ̄)`; `H` when `ρ̄` is zero or not finite.
fn first_edge<F: SemiflowFloat>(half: F, rho: F) -> F {
    if rho.is_nan() || rho <= F::zero() || !rho.is_finite() {
        return half;
    }
    let d0 = from_f64::<F>(DELTA) / rho;
    if d0 > F::zero() && d0 < half {
        d0
    } else {
        half
    }
}

/// Visit the panels `(lo, h)` of one half, inner panel first (§63.3.a).
fn walk_panels<F: SemiflowFloat>(half: F, rho: F, mut visit: impl FnMut(F, F)) {
    let q = from_f64::<F>(Q_RATIO);
    let mut lo = F::zero();
    let mut hi = first_edge(half, rho);
    let mut count = 1_usize;
    visit(lo, hi - lo);
    while hi < half {
        lo = hi;
        let grown = hi * q;
        hi = if count + 1 >= MAX_PANELS || grown >= half {
            half
        } else {
            grown
        };
        visit(lo, hi - lo);
        count += 1;
    }
}

/// Panels `(lo, h)` of ONE half, inner panel first, outer panel last.
pub(crate) fn half_panels<F: SemiflowFloat>(half: F, rho: F) -> Vec<(F, F)> {
    let mut panels = Vec::new();
    walk_panels(half, rho, |lo, h| panels.push((lo, h)));
    panels
}

/// Number of panels of one half (`K + 1`).
pub(crate) fn panels_per_half<F: SemiflowFloat>(half: F, rho: F) -> u32 {
    let mut count = 0_u32;
    walk_panels(half, rho, |_, _| count += 1);
    count
}

/// Distances `r_q = lo + h·x_q` of the 8 GL nodes of a panel, ascending.
pub(crate) fn panel_nodes<F: SemiflowFloat>(lo: F, h: F) -> [F; 8] {
    let mut r = [F::zero(); 8];
    for (slot, &(x, _)) in r.iter_mut().zip(GL8.iter()) {
        *slot = lo + h * from_f64::<F>(x);
    }
    r
}

/// `t/2`, exact in binary: the split point `H` of §63.3.
pub(crate) fn half_of<F: SemiflowFloat>(t: F) -> F {
    t * from_f64::<F>(0.5)
}

// ---------------------------------------------------------------------------
// Plan predictor (§63.8)
// ---------------------------------------------------------------------------

/// Pure cost/accuracy plan of one [`graph_expmv_frechet`](crate::graph_expmv_frechet) channel.
///
/// Produced by [`graph_expmv_frechet_plan`]; walks the SAME mesh and the SAME
/// step lengths as the sweep and calls the SAME degree selectors
/// (`graph_expmv_matvec_count`), so it equals what the sweep does (the decay
/// skip of §63.5.b can only shorten it). Used by the accuracy gates to derive
/// the error bound `τ_k` of §63.7.a without measuring anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrechetPlan {
    /// Panels per half, `K + 1`.
    pub panels_per_half: u32,
    /// Duhamel nodes per channel, `16·(K + 1)` (= calls of `accumulate_bilinear`).
    pub n_nodes: u32,
    /// Propagator actions per channel: `2·(1 + 16·(K + 1))`.
    pub propagator_calls: u32,
    /// Largest number of propagator EVALUATIONS (Chebyshev substeps, Lanczos
    /// outer steps, backward-Euler steps) along any single chain of §63.5.
    pub n_chain: u64,
    /// Largest polynomial degree / Krylov dimension among all calls.
    pub m_max: u32,
    /// Upper bound on sparse mat-vecs per channel, `Σ substeps × degree`.
    pub spmv_upper: u64,
}

/// Running totals of one half-sweep walk.
#[derive(Default)]
struct HalfCost {
    far_evals: u64,
    near_max: u64,
    m_max: u32,
    spmv: u64,
}

impl HalfCost {
    /// Record one propagator action of length `tau`; returns its substep count.
    fn call<F: SemiflowFloat>(&mut self, rho: F, tau: F, tol: F, path: &KrylovPath) -> u64 {
        let (s, m) = graph_expmv_matvec_count(rho, tau, tol, path);
        self.m_max = self.m_max.max(m);
        self.spmv = self.spmv.saturating_add(u64::from(s) * u64::from(m));
        u64::from(s)
    }
}

/// Walk one half exactly as the sweep does (outer to inner panel).
fn walk_half_cost<F: SemiflowFloat>(rho: F, half: F, tol: F, path: &KrylovPath) -> HalfCost {
    let mut cost = HalfCost::default();
    cost.far_evals = cost.call(rho, half, tol, path);
    let mut r_far = half;
    for &(lo, h) in half_panels(half, rho).iter().rev() {
        let r = panel_nodes(lo, h);
        let mut near = cost.call(rho, r[0], tol, path);
        for q in 1..8 {
            near += cost.call(rho, r[q] - r[q - 1], tol, path);
        }
        cost.near_max = cost.near_max.max(near);
        for q in (0..8).rev() {
            let step = r_far - r[q];
            r_far = r[q];
            if step > F::zero() {
                cost.far_evals += cost.call(rho, step, tol, path);
            }
        }
    }
    cost
}

/// Predict node count, chain length, max degree and `SpMV` bound (§63.8). Pure.
///
/// `rho_bar` is `GraphKrylovChernoff::lambda_max_bound`, `tol` is
/// `GraphKrylovChernoff::tol`, `path` is `GraphKrylovChernoff::path`.
/// Both halves of the symmetric mesh have identical cost, so the four chains
/// of §63.7 reduce to `max(far chain, longest near chain)`.
#[must_use]
pub fn graph_expmv_frechet_plan<F: SemiflowFloat>(
    rho_bar: F,
    t: F,
    tol: F,
    path: &KrylovPath,
) -> FrechetPlan {
    let half = half_of(t);
    let cost = walk_half_cost(rho_bar, half, tol, path);
    let panels = panels_per_half(half, rho_bar);
    let n_nodes = 16 * panels;
    FrechetPlan {
        panels_per_half: panels,
        n_nodes,
        propagator_calls: 2 * (1 + n_nodes),
        n_chain: cost.far_evals.max(cost.near_max),
        m_max: cost.m_max,
        spmv_upper: cost.spmv.saturating_mul(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Neumaier compensated sum (exact-summation stand-in for `math.fsum`).
    fn neumaier(xs: &[f64]) -> f64 {
        let (mut sum, mut comp) = (0.0_f64, 0.0_f64);
        for &x in xs {
            let t = sum + x;
            comp += if sum.abs() >= x.abs() {
                (sum - t) + x
            } else {
                (x - t) + sum
            };
            sum = t;
        }
        sum + comp
    }

    /// Relative error of the §63.3 rule on `e^{-mu σ}` + mirror, `t = 1`.
    fn rule_error(mu_t: f64, rho: f64) -> f64 {
        let mut terms = Vec::new();
        for (lo, h) in half_panels(0.5_f64, rho) {
            for &(x, w) in &GL8 {
                let r = lo + h * x;
                terms.push(h * w * (-mu_t * r).exp());
                terms.push(h * w * (-mu_t * (1.0 - r)).exp());
            }
        }
        let exact = -(-mu_t).exp_m1() / mu_t;
        ((neumaier(&terms) - exact) / exact).abs()
    }

    /// Legacy one-panel GL8 rule on `[0, 1]` for `e^{-mu σ}`.
    fn legacy_error(mu_t: f64) -> f64 {
        let terms: Vec<f64> = GL8.iter().map(|&(x, w)| w * (-mu_t * x).exp()).collect();
        let exact = -(-mu_t).exp_m1() / mu_t;
        ((neumaier(&terms) - exact) / exact).abs()
    }

    /// Supremum that propagates NaN (`f64::max` would silently drop it).
    fn sup_nan_propagating(errors: impl IntoIterator<Item = f64>) -> f64 {
        errors.into_iter().fold(0.0_f64, |acc, e| {
            if acc.is_nan() || e.is_nan() {
                f64::NAN
            } else {
                acc.max(e)
            }
        })
    }

    /// A NaN rule error must fail the gate comparison, not be skipped.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation IS the point: NaN must fail
    #[test]
    fn nan_rule_error_fails_the_gate() {
        let sup = sup_nan_propagating([1e-16, f64::NAN, 2e-16]);
        assert!(sup.is_nan());
        assert!(!(sup <= 1.1e-14), "NaN must not satisfy the bound");
        assert!(sup_nan_propagating([1e-16, 2e-16]) <= 1.1e-14);
    }

    /// `G_FRECHET_QUAD_CONSTANT` (ADR-0203, §63.4): sup rel. error `<= 1.1e-14`.
    #[test]
    fn g_frechet_quad_constant() {
        let mut errors = Vec::new();
        for i in 0..600 {
            let mu_t = 10.0_f64.powf(-3.0 + 11.0 * f64::from(i) / 599.0);
            for over in [1.0, 2.0] {
                errors.push(rule_error(mu_t, mu_t * over));
            }
        }
        assert!(
            errors.iter().all(|e| e.is_finite()),
            "non-finite rule error"
        );
        let worst = sup_nan_propagating(errors);
        std::eprintln!("G_FRECHET_QUAD_CONSTANT sup = {worst:.3e}");
        assert!(worst <= 1.1e-14, "sup rel err {worst:e} > 1.1e-14");
        // Non-vacuity: the pre-ADR-0203 one-panel rule fails the same bound.
        let legacy = legacy_error(10.0);
        std::eprintln!("legacy one-panel rel err at |mu|t = 10: {legacy:.3e}");
        assert!(
            legacy > 1.1e-14,
            "legacy rule unexpectedly accurate: {legacy:e}"
        );
    }

    /// `K` (= panels - 1) at `rho_bar t in {1, 10, 1e2, 1e4, 1e6}` (§63.6).
    #[test]
    fn panel_counts_match_contract() {
        let expected = [(1.0, 0_u32), (10.0, 3), (1e2, 8), (1e4, 20), (1e6, 31)];
        for (rho_t, k) in expected {
            let n = panels_per_half(0.5_f64, rho_t);
            assert_eq!(n, k + 1, "rho_bar t = {rho_t}");
            assert_eq!(u32::try_from(half_panels(0.5_f64, rho_t).len()), Ok(n));
        }
    }

    /// `n_nodes = 16(K+1)` and the plan agrees with the mesh (§63.6, §63.8).
    #[test]
    fn plan_node_table_and_consistency() {
        let path = KrylovPath::Chebyshev;
        let table = [
            (1.0, 16_u32),
            (10.0, 64),
            (1e2, 144),
            (1e4, 336),
            (1e6, 512),
        ];
        for (rho_t, nodes) in table {
            let plan = graph_expmv_frechet_plan(rho_t, 1.0_f64, 1e-12, &path);
            assert_eq!(plan.n_nodes, nodes, "rho_bar t = {rho_t}");
            assert_eq!(plan.panels_per_half, panels_per_half(0.5_f64, rho_t));
            assert_eq!(plan.propagator_calls, 2 * (1 + nodes));
            assert!(plan.n_chain >= 1 && plan.m_max >= 3 && plan.spmv_upper > 0);
        }
        let ratio = |a: f64, b: f64| {
            let (hi, lo) = (
                graph_expmv_frechet_plan(a, 1.0_f64, 1e-12, &path),
                graph_expmv_frechet_plan(b, 1.0_f64, 1e-12, &path),
            );
            f64::from(hi.n_nodes) / f64::from(lo.n_nodes)
        };
        assert!(
            ratio(1e6, 1e2) <= 4.0,
            "n_nodes(1e6)/n_nodes(1e2) = {}",
            ratio(1e6, 1e2)
        );
    }
}
