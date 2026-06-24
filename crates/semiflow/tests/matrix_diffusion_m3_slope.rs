//! `G_MATRIX` M=3 — coupled-diffusion convergence slope (`RELEASE_BLOCKING`).
//!
//! Gate: slope ≤ -1.95 on n ∈ {256, 512, 1024, 2048} at T = 0.5.
//! Reference: self-convergence against `n_ref` = 8192 (math §33.4, ADR-0082).
//! Coupling matrices: deterministic SPD via seed `0xC0FFEE_BABE_DEAD_BEEF`.
//!
//! Block-CN (ADR-0082 AMENDMENT 2): unconditionally A-stable; no CFL constraint.
//! Probe runs n∈{256..2048} valid for all step sizes.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_possible_truncation)] // PCG u128→u64 bitwise truncation; intentional bit extraction
#![allow(clippy::cast_precision_loss)] // u32/usize→f64 for arithmetic; values ≤ 8192 ≤ 2^52
#![allow(clippy::cast_lossless)] // u32→f64 widening, always exact for u32
#![allow(clippy::needless_range_loop)] // index loops do index-cross arithmetic (matrix ops)

use semiflow::{ChernoffFunction, Grid1D, MatrixDiffusionChernoff, MatrixGridFn1D, ScratchPool};

const M: usize = 3;
const N_GRID: usize = 128;
const T: f64 = 0.5;
const N_REF: u32 = 8192;

struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    fn new(seed: u128) -> Self {
        let inc = 0xda3e_39cb_94b9_5bdb_da3e_39cb_94b9_5bdb_u128 | 1;
        let mut rng = Self { state: 0, inc };
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u64();
        rng
    }

    fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.state = old
            .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645_u128)
            .wrapping_add(self.inc);
        (((old >> 64) ^ old) as u64).rotate_right(((old >> 122) as u32) & 63)
    }

    fn next_f64(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) | 0x3FF0_0000_0000_0000_u64;
        f64::from_bits(bits) - 1.0
    }
}

fn gen_spd<const M: usize>(rng: &mut Pcg64, eps: f64) -> [[f64; M]; M] {
    let mut l = [[0.0_f64; M]; M];
    for i in 0..M {
        for j in 0..=i {
            l[i][j] = rng.next_f64() * 0.8 + 0.1;
        }
    }
    let mut a = [[0.0_f64; M]; M];
    for i in 0..M {
        for j in 0..M {
            for k in 0..=i.min(j) {
                a[i][j] += l[i][k] * l[j][k];
            }
        }
        a[i][i] += eps;
    }
    a
}

fn gen_skew<const M: usize>(rng: &mut Pcg64) -> [[f64; M]; M] {
    let mut b = [[0.0_f64; M]; M];
    for i in 0..M {
        for j in (i + 1)..M {
            let v = (rng.next_f64() - 0.5) * 0.2;
            b[i][j] = v;
            b[j][i] = -v;
        }
    }
    b
}

fn gen_sym<const M: usize>(rng: &mut Pcg64) -> [[f64; M]; M] {
    let mut c = [[0.0_f64; M]; M];
    for i in 0..M {
        for j in i..M {
            let v = (rng.next_f64() - 0.5) * 0.1;
            c[i][j] = v;
            c[j][i] = v;
        }
    }
    c
}

fn make_kernel(n: usize) -> MatrixDiffusionChernoff<f64, M> {
    const SEED: u128 = 0x00C0_FFEE_BABE_DEAD_BEEF;
    let mut rng = Pcg64::new(SEED);
    // Skip M=2 draws to get M=3 specific matrices (advance past M=2 draws).
    for _ in 0..6 {
        rng.next_f64();
    } // skip 2x3 lower triangular + 1 skew pair + 1 sym pair
    let a_fixed: [[f64; M]; M] = gen_spd::<M>(&mut rng, 0.5);
    let b_fixed: [[f64; M]; M] = gen_skew::<M>(&mut rng);
    let c_fixed: [[f64; M]; M] = gen_sym::<M>(&mut rng);

    let grid = Grid1D::new(-5.0, 5.0, n).unwrap();
    MatrixDiffusionChernoff::<f64, M>::new(
        move |_, a| *a = a_fixed,
        move |_, b| *b = b_fixed,
        move |_, c| *c = c_fixed,
        grid,
    )
    .unwrap()
}

fn initial_fn(x: f64) -> [f64; M] {
    let mut comp = [0.0_f64; M];
    for i in 0..M {
        comp[i] = (-x * x / (1.0 + 0.2 * i as f64)).exp();
    }
    comp
}

fn run_sweep(n_steps: u32, reference: &MatrixGridFn1D<f64, M>) -> f64 {
    let tau = T / n_steps as f64;
    let kernel = make_kernel(N_GRID);
    let f0 = MatrixGridFn1D::<f64, M>::from_fn(kernel.grid, initial_fn);
    let mut pool = ScratchPool::<f64>::new();
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let n = N_GRID;
    let mut err = 0.0_f64;
    for k in 0..n {
        for i in 0..M {
            let diff = (cur.values[k * M + i] - reference.values[k * M + i]).abs();
            if diff > err {
                err = diff;
            }
        }
    }
    err
}

/// OLS slope of (ln `n_i`, ln `err_i`): negative for convergence (err ∝ n^slope → slope < 0).
fn ols_slope(ns: &[u32], errs: &[f64]) -> f64 {
    let x: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let y: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sxx: f64 = x.iter().map(|xi| xi * xi).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

#[test]
fn g_matrix_m3_slope() {
    let tau_ref = T / N_REF as f64;
    let kernel_ref = make_kernel(N_GRID);
    let f0 = MatrixGridFn1D::<f64, M>::from_fn(kernel_ref.grid, initial_fn);
    let mut pool = ScratchPool::<f64>::new();
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for _ in 0..N_REF {
        kernel_ref
            .apply_into(tau_ref, &cur, &mut nxt, &mut pool)
            .unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let reference = cur;

    // Sweep (stable regime: n ≥ 256, see stability note in file header).
    let ns = [256_u32, 512, 1024, 2048];
    let errs: Vec<f64> = ns.iter().map(|&n| run_sweep(n, &reference)).collect();

    for (&n, &e) in ns.iter().zip(errs.iter()) {
        println!("G_MATRIX M=3: n={n} err={e:.4e}");
    }

    let slope = ols_slope(&ns, &errs);
    println!("G_MATRIX M=3: OLS slope = {slope:.4}");
    assert!(
        slope <= -1.95,
        "G_MATRIX M=3: slope {slope:.4} > -1.95 (gate FAILED)"
    );
}
