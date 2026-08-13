//! `G_GRID1D_BATCH_ULP` — batched 1-D grid evolve is 0-ULP vs sequential (ADR-0193).
//!
//! Batching is a **throughput** device, not an accuracy one. Every channel runs
//! the identical `apply_into` sequence at the identical `tau`, and there is no
//! cross-channel reduction, so the batched result must be bit-identical to `C`
//! sequential solves — not merely close. The comparison is on raw `f64` bit
//! patterns, so a change that perturbed the arithmetic (reordering, fused
//! multiply-add, a shared scratch buffer leaking state between channels) cannot
//! hide inside a tolerance.
//!
//! The grid sizes straddle the channel-parallel dispatch boundary in both
//! directions, so the parallel build exercises both branches.

// Grid-index -> coordinate arithmetic; every index here is a small test
// constant, far below the f64 mantissa.
#![allow(clippy::cast_precision_loss)]

use semiflow::{
    grid_batched::evolve_batched_1d, ChernoffSemigroup, Grid1D, GridFn1D, ShiftChernoff1D,
};

const XMIN: f64 = -4.0;
const XMAX: f64 = 4.0;

/// Variable-coefficient shift kernel — the case the issue actually cares about.
fn kernel(grid: Grid1D<f64>) -> ShiftChernoff1D<f64> {
    ShiftChernoff1D::with_closure(
        |x: f64| 0.3 + 0.2 * (0.5 * x).tanh().abs(),
        |x: f64| 0.15 * x.sin(),
        |x: f64| -0.05 * (1.0 + x * x).ln(),
        0.7,
        grid,
    )
}

/// Channel `c` of the initial condition — deliberately distinct per channel.
fn channel_ic(n: usize, c: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((n - 1) as f64);
            let shift = 0.3 * (c as f64) - 0.6;
            (-(x - shift) * (x - shift)).exp() + 0.01 * (c as f64)
        })
        .collect()
}

/// `G_GRID1D_BATCH_ULP` — bit-identical to the per-channel sequential path.
#[test]
fn g_grid1d_batch_ulp() {
    for n in [64_usize, 512, 4096, 8192] {
        for n_cols in [1_usize, 2, 3, 5, 8] {
            for n_steps in [1_usize, 7] {
                let grid = Grid1D::new(XMIN, XMAX, n).unwrap();
                let func = kernel(grid);
                let t = 0.05;

                // Batched: [C, N] channel-major.
                let mut src = Vec::with_capacity(n * n_cols);
                for c in 0..n_cols {
                    src.extend_from_slice(&channel_ic(n, c));
                }
                let mut dst = vec![0.0_f64; n * n_cols];
                evolve_batched_1d(&func, grid, t, n_steps, &src, &mut dst).unwrap();

                // Sequential reference, one channel at a time.
                for c in 0..n_cols {
                    let sg = ChernoffSemigroup::new(kernel(grid), n_steps).unwrap();
                    let u0 = GridFn1D::new(grid, channel_ic(n, c)).unwrap();
                    let want = sg.evolve(t, &u0).unwrap();
                    let got = &dst[c * n..(c + 1) * n];
                    for (i, (g, w)) in got.iter().zip(want.values.iter()).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "n={n} n_cols={n_cols} n_steps={n_steps} c={c} i={i}: \
                             batched={g:.17e} sequential={w:.17e}"
                        );
                    }
                }
            }
        }
    }
}

/// Layout contract: divisibility, matching lengths, and the `n_steps == 0` copy.
#[test]
fn grid_batch_layout_contract() {
    let n = 32;
    let grid = Grid1D::new(XMIN, XMAX, n).unwrap();
    let func = kernel(grid);

    // n_steps == 0 copies src -> dst unchanged.
    let src: Vec<f64> = (0..3 * n).map(|i| i as f64).collect();
    let mut dst = vec![0.0_f64; 3 * n];
    evolve_batched_1d(&func, grid, 0.1, 0, &src, &mut dst).unwrap();
    assert_eq!(src, dst);

    // Length mismatch and non-divisible length are DomainViolations.
    let mut short = vec![0.0_f64; 2 * n];
    assert!(evolve_batched_1d(&func, grid, 0.1, 1, &src, &mut short).is_err());
    let ragged = vec![0.0_f64; 3 * n + 1];
    let mut ragged_dst = vec![0.0_f64; 3 * n + 1];
    assert!(evolve_batched_1d(&func, grid, 0.1, 1, &ragged, &mut ragged_dst).is_err());

    // Zero channels is a no-op, not an error.
    let empty: Vec<f64> = Vec::new();
    let mut empty_dst: Vec<f64> = Vec::new();
    assert!(evolve_batched_1d(&func, grid, 0.1, 1, &empty, &mut empty_dst).is_ok());
}

/// The channels really are independent — a change in one must not touch another.
///
/// Non-vacuity for the 0-ULP gate: if the workers shared a ping-pong buffer or a
/// `ScratchPool`, cross-talk would show up here even where bit-equality against
/// a *sequential* run might not (both could be wrong the same way).
#[test]
fn grid_batch_channels_are_independent() {
    let (n, n_cols) = (128_usize, 4_usize);
    let grid = Grid1D::new(XMIN, XMAX, n).unwrap();
    let func = kernel(grid);

    let mut src: Vec<f64> = Vec::with_capacity(n * n_cols);
    for c in 0..n_cols {
        src.extend_from_slice(&channel_ic(n, c));
    }
    let mut base = vec![0.0_f64; n * n_cols];
    evolve_batched_1d(&func, grid, 0.05, 5, &src, &mut base).unwrap();

    // Perturb channel 2 only.
    let mut perturbed = src.clone();
    perturbed[2 * n + n / 2] += 1.0;
    let mut out = vec![0.0_f64; n * n_cols];
    evolve_batched_1d(&func, grid, 0.05, 5, &perturbed, &mut out).unwrap();

    for c in 0..n_cols {
        let same = base[c * n..(c + 1) * n] == out[c * n..(c + 1) * n];
        if c == 2 {
            assert!(!same, "perturbed channel 2 should have changed");
        } else {
            assert!(same, "channel {c} changed when only channel 2 was perturbed");
        }
    }
}
