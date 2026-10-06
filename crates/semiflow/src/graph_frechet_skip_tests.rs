//! Part B of `G_FRECHET_SWEEP_EXACT_PROP` (ADR-0203 Amendment 3, math §63.7.c):
//! the decay skip of §63.5.b actually fires, stays within `τ_k^E + ε_skip·N_k`, and
//! two broken skips are caught.
//!
//! Fixture S1: F3 (per-point `W`, `t`) with `dj` projected onto the fast
//! intra-cluster eigenspace and `u0` keeping its slow content. The left half's
//! far vector `e^{−DL}dj` then decays below `ε_skip` and the skip fires there;
//! the right half's far vector keeps the slow content of `u0` and must not skip.

use std::{eprintln, format, string::String, vec::Vec};

use super::{
    common::{f3_matrix, f3_point, f3_signals, jacobi_eigh, nan_max, norm2, oracle, violates},
    edge_stencils, gershgorin, sweep_with, tau_e, Fixture, GRID, N,
};
use crate::{
    graph_expmv_frechet_plan, graph_frechet::decay_skip, graph_frechet_mesh::DELTA, KrylovPath,
};

const EPS_SKIP: f64 = 1e-12;

/// S1 at one point: the fixture, its time, and the onset facts of §63.7.c.
struct S1 {
    fx: Fixture,
    t: f64,
    /// Smallest eigenvalue of the projected (fast) eigenspace.
    lam_f: f64,
    /// `‖(I − Π_f)dj‖ / ‖dj‖` of the computed `dj` (onset premise).
    residual: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// S1 with both signals multiplied by `scale` (`1` or `1e-16`).
fn s1_fixture(lt: f64, scale: f64) -> S1 {
    let (w, t) = f3_point(lt);
    let (mat, edges) = f3_matrix(w);
    let eig = jacobi_eigh(&mat);
    let threshold = (eig.lam[1] * eig.lam[N - 1]).sqrt();
    let raw: Vec<f64> = (0..N)
        .map(|i| (-1.0_f64).powi(i32::try_from(i).expect("small")) + 0.1 * (2.3 * i as f64).cos())
        .collect();
    let (mut dj, mut lam_f) = (std::vec![0.0; N], f64::INFINITY);
    for (lam, phi) in eig.lam.iter().zip(&eig.vec) {
        if *lam >= threshold {
            lam_f = lam_f.min(*lam);
            let c = dot(phi, &raw);
            dj.iter_mut().zip(phi).for_each(|(o, p)| *o += c * p);
        }
    }
    let slow_sq: f64 = eig
        .lam
        .iter()
        .zip(&eig.vec)
        .filter(|(lam, _)| **lam < threshold)
        .map(|(_, phi)| dot(phi, &dj).powi(2))
        .sum();
    let residual = slow_sq.sqrt() / norm2(&dj);
    let v: Vec<f64> = f3_signals().0.iter().map(|x| x * scale).collect();
    let d: Vec<f64> = dj.iter().map(|x| x * scale).collect();
    let fx = Fixture {
        name: "S1-edge",
        rho: gershgorin(&mat),
        stencils: edge_stencils(edges.iter().map(|&(i, j, _)| (i, j))),
        entry_params: false,
        carrier: None,
        eig,
        d,
        v,
    };
    S1 {
        fx,
        t,
        lam_f,
        residual,
    }
}

/// B1: `max_k err/bound` and the number of violated parameters, bound `τ_k^E + ε_skip·N_k`.
fn b1(s1: &S1, grad: &[f64]) -> (f64, usize) {
    let fx = &s1.fx;
    let refs = oracle(&fx.eig, &fx.d, &fx.v, s1.t, &fx.stencils);
    let plan = graph_expmv_frechet_plan(fx.rho, s1.t, 1e-14, &KrylovPath::Chebyshev);
    let (nd, nv) = (norm2(&fx.d), norm2(&fx.v));
    let (mut worst, mut violations) = (0.0_f64, 0);
    for (k, st) in fx.stencils.iter().enumerate() {
        let (g_ref, big_g) = refs[k];
        let n_k = s1.t * st.norm2() * nd * nv;
        let bound = tau_e(big_g, n_k, plan.panels_per_half, plan.n_nodes, EPS_SKIP);
        let err = (grad[k] - g_ref).abs();
        worst = nan_max(worst, err / bound);
        violations += usize::from(violates(err, bound));
    }
    (worst, violations)
}

/// B2: call counts and the onset premise at one point.
fn b2(s1: &S1, calls: (usize, usize), identical: bool) -> Vec<String> {
    let (off, on) = calls;
    let mut fails = Vec::new();
    let onset = s1.lam_f * (s1.t - DELTA / s1.fx.rho) >= (2.0 / EPS_SKIP).ln();
    if s1.residual > EPS_SKIP / 2.0 {
        fails.push(format!("onset premise: residual {:e} > eps/2", s1.residual));
    }
    if onset && on >= off {
        fails.push(format!(
            "onset point fired no skip: calls on={on} off={off}"
        ));
    }
    if on > off {
        fails.push(format!("skip ON made MORE calls: on={on} off={off}"));
    }
    if !onset && on == off && !identical {
        fails.push("no skip fired but gradients differ".into());
    }
    fails
}

/// B4 inputs: signals ×1e-16 (bound and oracle scaled consistently), run with the
/// UNSCALED-norm skip N2 and with the correct relative skip. Returns `(N2, correct)`
/// each as `(max err/bound, violations)`.
fn scaled_runs(lt: f64) -> ((f64, usize), (f64, usize)) {
    let tiny = s1_fixture(lt, 1e-16);
    let n2 = |far: f64, _src: f64| far <= EPS_SKIP;
    let run = |skip: &dyn Fn(f64, f64) -> bool| b1(&tiny, &sweep_with(&tiny.fx, tiny.t, &skip).0);
    (run(&n2), run(&decay_skip(EPS_SKIP)))
}

/// Part B over the grid: B1, B2, and the negative tests B3 (N1) and B4 (N2).
pub(super) fn part_b() -> Vec<String> {
    let mut fails: Vec<String> = Vec::new();
    for lt in GRID {
        let s1 = s1_fixture(lt, 1.0);
        let (g_off, c_off) = sweep_with(&s1.fx, s1.t, &|_, _| false);
        let (g_on, c_on) = sweep_with(&s1.fx, s1.t, &decay_skip(EPS_SKIP));
        let (ratio, bad) = b1(&s1, &g_on);
        let identical = g_off
            .iter()
            .zip(&g_on)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let (n1_ratio, n1_bad) = b1(&s1, &sweep_with(&s1.fx, s1.t, &|_, _| true).0);
        let ((n2_ratio, n2_bad), (ok_ratio, ok_bad)) = scaled_runs(lt);
        eprintln!(
            "SKIP_B lt={lt:>6.0e} lam_f*t={:8.2e} correct err/bound={ratio:9.2e} calls off/on={c_off}/{c_on} \
             N1 err/bound={n1_ratio:9.2e} N2(x1e-16) err/bound={n2_ratio:9.2e} \
             correct(x1e-16)={ok_ratio:9.2e} residual={:8.2e}",
            s1.lam_f * s1.t,
            s1.residual
        );
        if bad > 0 {
            fails.push(format!(
                "B1 lt={lt:e}: {bad} parameters exceed tau_E + eps N_k"
            ));
        }
        if ok_bad > 0 {
            fails.push(format!(
                "B1 (signals x1e-16, relative skip) lt={lt:e}: {ok_bad} violations"
            ));
        }
        fails.extend(
            b2(&s1, (c_off, c_on), identical)
                .into_iter()
                .map(|f| format!("B2 lt={lt:e}: {f}")),
        );
        if lt >= 10.0 && n1_bad == 0 {
            fails.push(format!("B3 lt={lt:e}: always-fire skip (N1) not detected"));
        }
        if lt >= 10.0 && n2_bad == 0 {
            fails.push(format!(
                "B4 lt={lt:e}: unscaled-norm skip (N2) not detected"
            ));
        }
    }
    fails
}
