// Tests for tt_coupled.rs — moved from tt_coupled.rs (batch H6).
use super::*;

/// `CouplingTopology::None` on a rank-1 IC must preserve rank-1.
#[test]
fn none_topology_preserves_rank1() {
    let n = 16usize;
    let xmin = -3.0f64;
    let xmax = 3.0f64;
    let dx = (xmax - xmin) / (n as f64 - 1.0);
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let x = xmin + i as f64 * dx;
            (-x * x / 2.0).exp()
        })
        .collect();
    let evolver = CoupledTtChernoff::new(
        vec![0.5, 0.3],
        vec![0.0, 0.0],
        0.0,
        CouplingTopology::None,
        vec![(xmin, xmax), (xmin, xmax)],
        1e-10,
    );
    let mut state = TtState::rank1_separable(vec![f.clone(), f.clone()]);
    evolver.evolve(0.2, 10, &mut state);
    assert_eq!(state.peak_rank(), 1, "None topology should preserve rank-1");
}

/// ANTI-TRIVIALITY: Tridiagonal coupling on rank-1 IC → `peak_rank` > 1.
/// Rank-1 result = v9.0.0 separability bug. PASS = genuine coupling.
#[test]
fn coupled_step_grows_rank_from_rank1_ic() {
    let n = 8usize;
    let xmin = -3.0f64;
    let xmax = 3.0f64;
    let dx = (xmax - xmin) / (n as f64 - 1.0);
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let x = xmin + i as f64 * dx;
            (-x * x / 4.0).exp()
        })
        .collect();
    let rho = 0.6f64;
    let evolver = CoupledTtChernoff::new(
        vec![0.5, 0.4, 0.3],
        vec![0.0, 0.0, 0.0],
        0.0,
        CouplingTopology::Tridiagonal(rho),
        vec![(xmin, xmax), (xmin, xmax), (xmin, xmax)],
        1e-8,
    );
    let ic_state = TtState::rank1_separable(vec![f.clone(), f.clone(), f.clone()]);
    assert_eq!(
        ic_state.peak_rank(),
        1,
        "IC must be rank-1 for anti-triviality test"
    );

    let mut state = ic_state;
    evolver.step(0.05, &mut state);

    let peak = state.peak_rank();
    assert!(
        peak > 1,
        "ANTI-TRIVIALITY FAIL: coupled step produced rank-{peak} from rank-1 IC. \
         Coupling is a no-op (v9.0.0 separability bug). \
         pre-step rank=1, post-step peak_rank={peak}"
    );
}

/// Adjacent block-disjoint pairs (d=4) from rank-1 IC: rank must grow.
/// v9.1.0: non-adjacent pairs are rejected; this uses (0,1)+(2,3) only.
#[test]
fn adj_block_disjoint_4d_coupled_step_grows_rank() {
    let n = 6usize;
    let xmin = -2.0f64;
    let xmax = 2.0f64;
    let dx = (xmax - xmin) / (n as f64 - 1.0);
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let x = xmin + i as f64 * dx;
            (-x * x / 2.0).exp()
        })
        .collect();
    let d = 4;
    let slices = (0..d).map(|_| f.clone()).collect();
    let pairs: Vec<(usize, usize, f64)> = vec![(0, 1, 0.3), (2, 3, 0.3)]; // adjacent only
    let evolver = CoupledTtChernoff::new(
        vec![0.5; d],
        vec![0.0; d],
        0.0,
        CouplingTopology::Pairs(pairs),
        vec![(xmin, xmax); d],
        1e-8,
    );
    let mut state = TtState::rank1_separable(slices);
    assert_eq!(state.peak_rank(), 1);
    evolver.step(0.05, &mut state);
    assert!(
        state.peak_rank() > 1,
        "adj block-disjoint 4D coupled step must grow rank from 1, got {}",
        state.peak_rank()
    );
}

/// Coupling ON → `peak_rank` > no-coupling. Structural non-separability check.
#[test]
fn coupling_raises_peak_rank_vs_no_coupling() {
    let n = 8usize;
    let xmin = -2.0f64;
    let xmax = 2.0f64;
    let dx = (xmax - xmin) / (n as f64 - 1.0);
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let x = xmin + i as f64 * dx;
            (-x * x / 2.0).exp()
        })
        .collect();

    let no_couple = CoupledTtChernoff::new(
        vec![0.5, 0.4],
        vec![0.0, 0.0],
        0.0,
        CouplingTopology::None,
        vec![(xmin, xmax); 2],
        1e-10,
    );
    // ρ=0.6 is within SPD scope for d=2 block-disjoint (cj=aj, det=0.5*0.4-0.6²*0.5*0.4>0)
    let with_couple = CoupledTtChernoff::new(
        vec![0.5, 0.4],
        vec![0.0, 0.0],
        0.0,
        CouplingTopology::Tridiagonal(0.6f64),
        vec![(xmin, xmax); 2],
        1e-10,
    );

    let mut s1 = TtState::rank1_separable(vec![f.clone(), f.clone()]);
    let mut s2 = TtState::rank1_separable(vec![f.clone(), f.clone()]);
    no_couple.step(0.1, &mut s1);
    with_couple.step(0.1, &mut s2);

    let r1 = s1.peak_rank();
    let r2 = s2.peak_rank();
    assert!(
        r2 > r1,
        "coupling should raise peak_rank vs no-coupling: no-couple r={r1}, with-couple r={r2}"
    );
}

/// D=2 EXACTNESS SELF-CHECK (§10.13.2, P3' correctness proof).
///
/// Tests `CoupledTtChernoff` (rotated stable pair factor, scheme E) against a
/// dense `expm(T·L_h^{dx})·ic` reference built in this test. For the constant-
/// coefficient correlated-Gaussian class the result is machine-exact (~1e-14 for
/// the pair factor; the test allows ≤1e-10 for rounding at eps_round=1e-12).
///
/// Setup: n=13, T=0.05, `n_steps=15`, ρ=0.6, a=[0.8,0.6], domain=[0,1],
/// rank-1 separable Gaussian IC. Reference: dense n²×n² expm via 6-term Padé.
///
/// Anti-vacuity: h/dx=2√(a₀·τ)/dx ≈ 0.84 (non-integer, frac>0.05).
///
/// If rel-L2 > 1e-10: factor is built wrong — do NOT loosen the bound.
#[test]
fn d2_exactness_self_check() {
    use crate::tt_coupled_pair::{build_l_pair_pub, dense_expm_pub};
    const N: usize = 13;
    const T: f64 = 0.05;
    const NS: usize = 15;
    const RHO: f64 = 0.6;
    let a = [0.8f64, 0.6f64];
    let xmin = 0.0f64;
    let xmax = 1.0f64;
    let dx = (xmax - xmin) / (N as f64 - 1.0);
    let ic1d = make_gaussian_ic(N, xmin, xmax, dx);
    check_h_dx_not_integer(a[0], T, NS, dx);
    let tt_dense = d2_evolve_to_dense(N, T, NS, RHO, a, xmin, xmax, &ic1d);
    let ref_dense = d2_expm_ref(N, dx, &a, RHO, T, &ic1d, N * N, build_l_pair_pub, dense_expm_pub);
    let rel_err = compute_rel_l2_error(&tt_dense, &ref_dense);
    println!(
        "d=2 exactness: n={N} T={T} ns={NS} rho={RHO} rel_L2={rel_err:.3e}"
    );
    assert!(
        rel_err <= 1e-10,
        "D=2 EXACTNESS FAIL: {rel_err:.3e}>1e-10 (factor wrong; n={N} T={T} ns={NS} rho={RHO})"
    );
}

// ── Helpers for d2_exactness_self_check ──────────────────────────────────────

fn check_h_dx_not_integer(a0: f64, t: f64, n_steps: usize, dx: f64) {
    let h_over_dx = 2.0 * (a0 * t / n_steps as f64).sqrt() / dx;
    let frac = (h_over_dx - h_over_dx.round()).abs();
    assert!(frac > 0.05, "h/dx={h_over_dx:.4} near-integer");
}

#[allow(clippy::too_many_arguments)]
fn d2_evolve_to_dense(
    n: usize,
    t: f64,
    n_steps: usize,
    rho: f64,
    a: [f64; 2],
    xmin: f64,
    xmax: f64,
    ic1d: &[f64],
) -> Vec<f64> {
    let ev = CoupledTtChernoff::new(
        a.to_vec(),
        vec![0.0; 2],
        0.0,
        CouplingTopology::Tridiagonal(rho),
        vec![(xmin, xmax); 2],
        1e-12,
    );
    let mut tt = TtState::rank1_separable(vec![ic1d.to_vec(), ic1d.to_vec()]);
    ev.evolve(t, n_steps, &mut tt);
    d2_tt_to_dense(&tt, n, n * n)
}

fn compute_rel_l2_error(approx: &[f64], reference: &[f64]) -> f64 {
    let norm_ref: f64 = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
    let err: f64 = approx
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    err / norm_ref
}

fn make_gaussian_ic(n: usize, xmin: f64, xmax: f64, dx: f64) -> Vec<f64> {
    let cx = 0.5 * (xmin + xmax);
    (0..n)
        .map(|i| {
            let x = xmin + i as f64 * dx;
            (-(x - cx).powi(2) / 0.04).exp()
        })
        .collect()
}

fn d2_tt_to_dense(tt: &TtState<f64>, n: usize, n2: usize) -> Vec<f64> {
    let r0 = tt.cores[0].r_right;
    (0..n2)
        .map(|flat| {
            let ij = flat / n;
            let ik = flat % n;
            (0..r0)
                .map(|ir| tt.cores[0].get(0, ij, ir) * tt.cores[1].get(ir, ik, 0))
                .sum()
        })
        .collect()
}

// n, a, t, dx, rho: standard PDE/matrix-expm parameter names; single-char is idiomatic here.
// 9 args: n, dx, a, rho, t, ic1d, n2, build_fn, expm_fn — all required for the 2D reference.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments
)]
fn d2_expm_ref(
    n: usize,
    dx: f64,
    a: &[f64],
    rho: f64,
    t: f64,
    ic1d: &[f64],
    n2: usize,
    build_fn: impl Fn(usize, f64, f64, f64, f64, f64) -> Vec<f64>,
    expm_fn: impl Fn(&[f64], usize) -> Vec<f64>,
) -> Vec<f64> {
    let r = rho * (a[0] * a[1]).sqrt();
    let mut l = build_fn(n, dx, dx, a[0], a[1], r);
    for v in &mut l {
        *v *= t;
    }
    let e = expm_fn(&l, n2);
    let ic: Vec<f64> = (0..n2)
        .map(|flat| ic1d[flat / n] * ic1d[flat % n])
        .collect();
    (0..n2)
        .map(|i| (0..n2).map(|j| e[i * n2 + j] * ic[j]).sum())
        .collect()
}
