//! Regression (ADR-0203 review): the decay skip of §63.5.b compares `‖far‖₂` with
//! `tol·‖far_src‖₂`. A naive sum of squares overflows for entries `≳ 1e154` and
//! underflows for entries `≲ 1e-162`, which made the comparison `∞ ≤ ∞` /
//! `0 ≤ 0` true and silently dropped every inner panel.
//!
//! The gradient is bilinear in `(u0, dj)`, so `(u0·s⁻¹, dj·s)` must reproduce the
//! gradient of `(u0, dj)` (the products stay `O(1)`).

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // N = 8

use std::sync::Arc;

use semiflow::{
    graph_expmv_frechet, EdgeWeightSensitivity, Graph, GraphKrylovChernoff, KrylovPath, Laplacian,
    ScratchPool,
};

const N: usize = 8;

fn gradient(u0: &[f64], dj: &[f64]) -> Vec<f64> {
    let edges: Vec<(u32, u32, f64)> = (0..N as u32 - 1)
        .map(|i| (i, i + 1, 0.6 + 0.1 * f64::from(i)))
        .collect();
    let g = Arc::new(Graph::<f64>::from_edges(N, edges).expect("graph"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let gk = GraphKrylovChernoff::new(lap, KrylovPath::Chebyshev, 1e-12).expect("gk");
    let sens = EdgeWeightSensitivity {
        params: (0..N - 1).map(|i| (i, i + 1)).collect(),
        n_nodes: N,
    };
    let t = 6.0 / gk.lambda_max_bound();
    let mut grad = vec![0.0; N - 1];
    graph_expmv_frechet(&gk, u0, dj, 1, t, &sens, &mut grad, &mut ScratchPool::new())
        .expect("frechet");
    grad
}

fn signals() -> (Vec<f64>, Vec<f64>) {
    let u0 = (0..N).map(|i| 0.4 + (0.37 * i as f64).sin()).collect();
    let dj = (0..N).map(|i| 0.3 + (0.53 * i as f64).cos()).collect();
    (u0, dj)
}

#[test]
fn huge_and_tiny_signals_do_not_trigger_the_skip() {
    let (u0, dj) = signals();
    let base = gradient(&u0, &dj);
    let scale_max = base.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    assert!(scale_max > 1e-3, "base gradient is vacuous");
    for s in [1e160_f64, 1e200] {
        let u0s: Vec<f64> = u0.iter().map(|x| x / s).collect();
        let djs: Vec<f64> = dj.iter().map(|x| x * s).collect();
        let got = gradient(&u0s, &djs);
        for (k, (a, b)) in got.iter().zip(&base).enumerate() {
            assert!(
                (a - b).abs() <= 1e-9 * scale_max,
                "s={s:e} k={k}: {a:e} vs {b:e}"
            );
        }
    }
}

#[test]
fn zero_far_source_contributes_nothing() {
    let (u0, _) = signals();
    let zero = vec![0.0; N];
    assert!(gradient(&u0, &zero).iter().all(|g| *g == 0.0));
    assert!(gradient(&zero, &u0).iter().all(|g| *g == 0.0));
}
