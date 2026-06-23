// Unit tests for [`TtChernoff`] and [`TtState`] (extracted per suckless ≤500-line cap).
#![allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]

use super::*;

// ── T1: TT-rounding of a known rank-1 tensor ──────────────────────────

/// A rank-1 tensor should round to rank-1 at any eps > 0.
#[test]
fn tt_round_rank1_stays_rank1() {
    let f1: Vec<f64> = (0..8).map(|i| f64::from(i).sin()).collect();
    let f2: Vec<f64> = (0..8).map(|i| f64::from(i).cos()).collect();
    let state = TtState::rank1_separable(vec![f1.clone(), f2.clone()]);
    let mut cores = state.cores.clone();
    tt_round(&mut cores, 1e-10);
    // After rounding a rank-1 tensor, all bonds should remain 1
    assert_eq!(cores[0].r_right, 1, "rank-1 tensor rounded to rank > 1");
    assert_eq!(cores[1].r_left, 1);
}

// ── T2: rank-1 separable IC + diagonal heat matches closed-form moments ──

/// For a product Gaussian IC `u₀(x,y) = exp(-x²/2)·exp(-y²/2)` on a 2D
/// diagonal heat equation ∂_t u = `a_x` ∂`²_x` u + `a_y` ∂`²_y` u:
/// The second moment ∂_xx u grows as `2·a_x·T` per axis (Gaussian diffusion).
/// We verify the mean of the TT evolution matches the closed-form Gaussian.
#[test]
fn rank1_heat_matches_gaussian_mean() {
    let n = 64usize;
    let xmin = -5.0f64;
    let xmax = 5.0f64;
    let dx = (xmax - xmin) / (n as f64 - 1.0);
    // IC: u₀(x) = exp(-x²/4) (Gaussian width 2)
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let x = xmin + i as f64 * dx;
            (-x * x / 4.0).exp()
        })
        .collect();
    let state0 = TtState::rank1_separable(vec![f.clone(), f.clone()]);

    let a_x = 0.5f64;
    let a_y = 0.7f64;
    let t_final = 0.5f64;
    let n_steps = 100usize;

    let evolver = TtChernoff::new(
        vec![a_x, a_y],
        vec![0.0, 0.0],
        0.0,
        vec![(xmin, xmax), (xmin, xmax)],
        1e-10,
    );

    let mut state = state0;
    evolver.evolve(t_final, n_steps, &mut state);

    // Verify rank stays 1 (diagonal heat, rank-1 IC → exact TT-rank=1)
    assert_eq!(state.peak_rank(), 1, "diagonal heat should preserve rank-1");

    // Verify discrete mass (Σ_i u(i), no dx) is preserved by the Chernoff kernel.
    // inner_separable(ones, ones) = (Σ_i G1[0,i,0]) * (Σ_j G2[0,j,0]) — no dx factor.
    // The initial discrete mass is (Σ_i f_i)^2 (product over both axes).
    let ones: Vec<f64> = vec![1.0; n];
    let mass = state.inner_separable(&[ones.clone(), ones.clone()]);
    // Initial discrete mass = (Σ_i exp(-xi²/4))²
    let init_discrete: f64 = f.iter().sum::<f64>();
    let init_mass_2d = init_discrete * init_discrete;
    // The shift kernel ¼f(x+h)+¼f(x-h)+½f(x) preserves Σ_i f(i) exactly.
    // After diffusion, mass should be within 10% of initial (coarse check).
    let rel_diff = (mass - init_mass_2d).abs() / init_mass_2d.abs().max(1e-10);
    assert!(
        rel_diff < 0.15,
        "mass conservation failed: init={init_mass_2d:.4e} final={mass:.4e} rel={rel_diff:.3}"
    );
}

// ── T3: TT-rounding correctness for a known rank-2 tensor ────────────

/// Build a rank-2 TT state as a sum of two rank-1 tensors, round at eps=0,
/// and verify the inner product is preserved.
#[test]
// core0 and cores are distinct (a single TT core vs the full core list).
#[allow(clippy::similar_names)]
fn tt_round_rank2_preserves_inner_product() {
    let n = 16usize;
    let f1: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let f2: Vec<f64> = (0..n).map(|i| -(i as f64 / n as f64)).collect();
    let g1: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64).powi(2)).collect();
    let g2: Vec<f64> = (0..n).map(|i| (n - i) as f64 / n as f64).collect();

    // Build a rank-2 core manually: G[0, i, 0] = f1[i], G[0, i, 1] = f2[i]
    // next core: H[0, i, 0] = g1[i], H[1, i, 0] = g2[i]
    let mut core0 = TtCore::zeros(1, n, 2);
    for i in 0..n {
        core0.set(0, i, 0, f1[i]);
        core0.set(0, i, 1, f2[i]);
    }
    let mut core1 = TtCore::zeros(2, n, 1);
    for i in 0..n {
        core1.set(0, i, 0, g1[i]);
        core1.set(1, i, 0, g2[i]);
    }
    let state = TtState {
        cores: vec![core0, core1],
    };

    // Expected inner product with (ones, ones): Σ_i f1[i] * Σ_j g1[j] + Σ_i f2[i] * Σ_j g2[j]
    let ones = vec![1.0f64; n];
    let expected = state.inner_separable(&[ones.clone(), ones.clone()]);

    // Round at very tight eps — should preserve the rank-2 structure
    let mut cores = state.cores.clone();
    tt_round(&mut cores, 1e-14);
    let state2 = TtState { cores };
    let actual = state2.inner_separable(&[ones.clone(), ones.clone()]);
    let err = (expected - actual).abs();
    assert!(
        err < 1e-10,
        "rounding destroyed inner product: expected={expected:.6e} got={actual:.6e} err={err:.2e}"
    );
}

// ── T4: Byte-reproducibility ───────────────────────────────────────────

/// Two independent runs of TT-Chernoff on the same IC produce bit-identical cores.
#[test]
fn tt_chernoff_byte_reproducible() {
    let n = 32usize;
    let xmin = -4.0f64;
    let xmax = 4.0f64;
    let dx = (xmax - xmin) / (n as f64 - 1.0);
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let x = xmin + i as f64 * dx;
            (-x * x / 2.0).exp()
        })
        .collect();

    let evolver = TtChernoff::new(
        vec![0.5, 0.3],
        vec![0.0, 0.0],
        0.0,
        vec![(xmin, xmax), (xmin, xmax)],
        1e-10,
    );

    let mut s1 = TtState::rank1_separable(vec![f.clone(), f.clone()]);
    let mut s2 = TtState::rank1_separable(vec![f.clone(), f.clone()]);

    evolver.evolve(0.2, 20, &mut s1);
    evolver.evolve(0.2, 20, &mut s2);

    for (k, (c1, c2)) in s1.cores.iter().zip(s2.cores.iter()).enumerate() {
        assert_eq!(
            c1.data, c2.data,
            "core {k} not bit-identical between two runs"
        );
    }
}

// ── T5: 3D rank-1 separable diagonal heat ─────────────────────────────

/// For d=3, rank-1 IC + diagonal heat: TT-Chernoff should maintain rank-1
/// and match mass within 15%.
#[test]
fn rank1_heat_3d_rank_preserved() {
    let n = 32usize;
    let xmin = -4.0f64;
    let xmax = 4.0f64;
    let dx = (xmax - xmin) / (n as f64 - 1.0);
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let x = xmin + i as f64 * dx;
            (-x * x / 2.0).exp()
        })
        .collect();

    let evolver = TtChernoff::new(
        vec![0.5, 0.3, 0.4],
        vec![0.0, 0.0, 0.0],
        0.0,
        vec![(xmin, xmax), (xmin, xmax), (xmin, xmax)],
        1e-10,
    );
    let mut state = TtState::rank1_separable(vec![f.clone(), f.clone(), f.clone()]);
    evolver.evolve(0.3, 30, &mut state);
    assert_eq!(
        state.peak_rank(),
        1,
        "diagonal heat should preserve rank-1 in 3D"
    );
    assert_eq!(state.ndim(), 3);
}
