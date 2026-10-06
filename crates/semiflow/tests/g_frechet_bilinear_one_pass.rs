//! `G_FRECHET_BILINEAR_ONE_PASS` (`RELEASE_BLOCKING`, ADR-0203, §63.6): the
//! all-parameter contraction is ONE `O(nnz)` call per Duhamel node.
//!
//! Counting wrappers around the three in-tree sensitivities check that
//! `accumulate_bilinear` is called exactly `n_nodes · n_cols` times and
//! `apply_param_deriv` never; a second wrapper that does NOT override
//! `accumulate_bilinear` (trait default = the pre-ADR-0203 loop) must give the
//! same gradient to `8·u·N_k`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)] // test arithmetic on small indices/counts (< 2^32), never on user data

use std::{cell::Cell, sync::Arc};

use semiflow::{
    graph_expmv_frechet, graph_expmv_frechet_plan, EdgeWeightSensitivity, EntrySensitivity,
    GeneratorSensitivity, Graph, GraphKrylovChernoff, KrylovPath, Laplacian,
    NodeTimescaleSensitivity, ScratchPool, SemiflowError,
};

const N: usize = 8;
const N_COLS: usize = 3;
const U: f64 = 1.110_223_024_625_156_5e-16;

/// Forwards everything, including `accumulate_bilinear`, and counts both.
struct Forwarding<'a, S> {
    inner: &'a S,
    bilinear: Cell<usize>,
    deriv: Cell<usize>,
}

/// Forwards ONLY `apply_param_deriv`: `accumulate_bilinear` is the trait default.
struct DefaultPath<'a, S> {
    inner: &'a S,
    deriv: Cell<usize>,
}

impl<S: GeneratorSensitivity<f64>> GeneratorSensitivity<f64> for Forwarding<'_, S> {
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    fn apply_param_deriv(
        &self,
        k: usize,
        t: f64,
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), SemiflowError> {
        self.deriv.set(self.deriv.get() + 1);
        self.inner.apply_param_deriv(k, t, v, out)
    }

    fn accumulate_bilinear(
        &self,
        t: f64,
        w: f64,
        a: &[f64],
        b: &[f64],
        grad: &mut [f64],
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        self.bilinear.set(self.bilinear.get() + 1);
        self.inner.accumulate_bilinear(t, w, a, b, grad, scratch)
    }
}

impl<S: GeneratorSensitivity<f64>> GeneratorSensitivity<f64> for DefaultPath<'_, S> {
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    fn apply_param_deriv(
        &self,
        k: usize,
        t: f64,
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), SemiflowError> {
        self.deriv.set(self.deriv.get() + 1);
        self.inner.apply_param_deriv(k, t, v, out)
    }
}

struct Fixture {
    gk: GraphKrylovChernoff<f64>,
    graph: Arc<Graph<f64>>,
    u0: Vec<f64>,
    dj: Vec<f64>,
    t: f64,
}

fn fixture() -> Fixture {
    let edges: Vec<(u32, u32, f64)> = (0..N as u32 - 1)
        .map(|i| (i, i + 1, 0.6 + 0.1 * f64::from(i)))
        .collect();
    let g = Arc::new(Graph::<f64>::from_edges(N, edges).expect("graph"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let gk = GraphKrylovChernoff::new(lap, KrylovPath::Chebyshev, 1e-12).expect("gk");
    let wave = |phase: f64| -> Vec<f64> {
        (0..N * N_COLS)
            .map(|i| 0.4 + (0.37 * i as f64 + phase).sin())
            .collect()
    };
    // t = 6 / ρ̄ → several panels, far chain never decays below `tol` (zero mode).
    let t = 6.0 / gk.lambda_max_bound();
    Fixture {
        gk,
        graph: g,
        u0: wave(0.3),
        dj: wave(1.1),
        t,
    }
}

fn run<P: GeneratorSensitivity<f64>>(fx: &Fixture, sens: &P) -> Vec<f64> {
    let mut grad = vec![0.0; sens.n_params()];
    let mut scratch = ScratchPool::new();
    graph_expmv_frechet(
        &fx.gk,
        &fx.u0,
        &fx.dj,
        N_COLS,
        fx.t,
        sens,
        &mut grad,
        &mut scratch,
    )
    .expect("graph_expmv_frechet");
    grad
}

/// `Σ_c ‖dj_c‖‖u0_c‖`, the channel-summed norm product of `N_k`.
fn norm_product(fx: &Fixture) -> f64 {
    let nrm = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>().sqrt();
    (0..N_COLS)
        .map(|c| nrm(&fx.dj[c * N..(c + 1) * N]) * nrm(&fx.u0[c * N..(c + 1) * N]))
        .sum()
}

/// `max_k |g_override − g_default| / (u·N_k)`, NaN-propagating; asserts `≤ 8` per `k`.
fn max_ratio(name: &str, (fast, slow): (&[f64], &[f64]), norm_m: &[f64], t_prod: f64) -> f64 {
    let mut worst = 0.0_f64;
    for (k, (a, b)) in fast.iter().zip(slow).enumerate() {
        let n_k = norm_m[k] * t_prod;
        let ratio = (a - b).abs() / (U * n_k);
        worst = if worst.is_nan() || ratio.is_nan() {
            f64::NAN
        } else {
            worst.max(ratio)
        };
        assert!(
            (a - b).abs() <= 8.0 * U * n_k,
            "{name} k={k}: {a:e} vs {b:e} (N_k={n_k:e})"
        );
    }
    worst
}

/// One sensitivity: counts, then override-vs-default agreement.
fn check<S: GeneratorSensitivity<f64>>(name: &str, fx: &Fixture, sens: &S, norm_m: &[f64]) {
    let plan = graph_expmv_frechet_plan(fx.gk.lambda_max_bound(), fx.t, fx.gk.tol(), &fx.gk.path());
    let fast = Forwarding {
        inner: sens,
        bilinear: Cell::new(0),
        deriv: Cell::new(0),
    };
    let g_fast = run(fx, &fast);
    let expect = plan.n_nodes as usize * N_COLS;
    assert_eq!(
        fast.bilinear.get(),
        expect,
        "{name}: accumulate_bilinear calls"
    );
    assert_eq!(
        fast.deriv.get(),
        0,
        "{name}: apply_param_deriv must not be called"
    );
    let slow = DefaultPath {
        inner: sens,
        deriv: Cell::new(0),
    };
    let g_slow = run(fx, &slow);
    assert_eq!(
        slow.deriv.get(),
        expect * g_slow.len(),
        "{name}: default path calls"
    );
    let worst = max_ratio(name, (&g_fast, &g_slow), norm_m, fx.t * norm_product(fx));
    assert!(
        worst.is_finite(),
        "{name}: non-finite override/default ratio"
    );
    eprintln!(
        "{name}: n_nodes*n_cols = {expect}, max |g_override - g_default| / (u N_k) = {worst:.3}"
    );
    assert!(
        g_fast.iter().any(|g| g.abs() > 1e-6),
        "{name}: gradient is vacuously zero"
    );
}

#[test]
fn g_frechet_bilinear_one_pass() {
    let fx = fixture();
    let params: Vec<(usize, usize)> = (0..N - 1).map(|i| (i, i + 1)).collect();
    let edge = EdgeWeightSensitivity {
        params: params.clone(),
        n_nodes: N,
    };
    check("edge", &fx, &edge, &[2.0; N - 1]);
    let mut entries = params;
    entries.extend([(0, 2), (0, 0), (N - 1, N - 1)]);
    let norms = vec![1.0; entries.len()];
    let entry = EntrySensitivity {
        entries,
        n_nodes: N,
    };
    check("entry", &fx, &entry, &norms);
    let sqrt_a: Vec<f64> = (0..N).map(|i| 0.8 + 0.05 * i as f64).collect();
    let rho = fx.gk.lambda_max_bound();
    let norm_a: Vec<f64> = sqrt_a.iter().map(|s| rho * sqrt_a[N - 1] / s).collect();
    let node = NodeTimescaleSensitivity {
        sqrt_a,
        bare_lap: Laplacian::assemble_combinatorial(&fx.graph),
    };
    check("node-timescale", &fx, &node, &norm_a);
}
