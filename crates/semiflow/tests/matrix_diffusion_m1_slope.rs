//! `G_MATRIX` M=1 — coupled-diffusion convergence slope (`RELEASE_BLOCKING`).
//!
//! Gate: slope ≤ -1.95 on n ∈ {256, 512, 1024, 2048} at T = 0.5.
//! Reference: self-convergence against `n_ref` = 8192.
//! Replaces byte-identity test (AMENDMENT 2): block-CN Phase 2 gives order-2
//! convergence for M=1, not byte-equality with scalar `DiffusionChernoff`.
//! math.md §33.4 sub-test 1, ADR-0082 AMENDMENT 2.
//!
//! Coupling matrices: deterministic SPD via seed `0xC0FFEE_BABE_DEAD_BEEF`.
//! M=1 case: scalar a>0, b=0 (skew-symmetric = 0), c scalar.

#![cfg(feature = "slow-tests")]

use semiflow::{
    ChernoffFunction, Grid1D, MatrixDiffusionChernoff, MatrixGridFn1D, ScratchPool,
};

const M: usize = 1;
const N_GRID: usize = 128;
const T: f64 = 0.5;
const N_REF: u32 = 8192;

// ---------------------------------------------------------------------------
// Minimal PCG64 (O'Neill 2014, canonical implementation)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Matrix generation helpers (M=1 degenerate case)
// ---------------------------------------------------------------------------

/// Generate 1×1 "SPD" matrix: a = random positive scalar.
fn gen_spd<const M: usize>(rng: &mut Pcg64, eps: f64) -> [[f64; M]; M] {
    let mut a = [[0.0_f64; M]; M];
    // For M=1: scalar a = L·L^T + eps*I = l^2 + eps (l in (0.1, 0.9)).
    for i in 0..M {
        let l = rng.next_f64() * 0.8 + 0.1;
        a[i][i] = l * l + eps;
    }
    a
}

/// Generate 1×1 skew-symmetric matrix: always zero for M=1.
fn gen_skew<const M: usize>(_rng: &mut Pcg64) -> [[f64; M]; M] {
    [[0.0_f64; M]; M]
}

/// Generate 1×1 symmetric matrix (for reaction): small scalar.
fn gen_sym<const M: usize>(rng: &mut Pcg64) -> [[f64; M]; M] {
    let mut c = [[0.0_f64; M]; M];
    for i in 0..M {
        c[i][i] = (rng.next_f64() - 0.5) * 0.1;
    }
    c
}

// ---------------------------------------------------------------------------
// Kernel builder
// ---------------------------------------------------------------------------

fn make_kernel(n: usize) -> MatrixDiffusionChernoff<f64, M> {
    const SEED: u128 = 0xC0FFEE_BABE_DEAD_BEEF;
    let mut rng = Pcg64::new(SEED);
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
    [(-x * x).exp()]
}

// ---------------------------------------------------------------------------
// Convergence sweep helpers
// ---------------------------------------------------------------------------

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
    // sup-norm error.
    let n = N_GRID;
    let mut err = 0.0_f64;
    for k in 0..n {
        let diff = (cur.values[k] - reference.values[k]).abs();
        if diff > err {
            err = diff;
        }
    }
    err
}

/// OLS slope of (ln n_i, ln err_i): negative for convergence.
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

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

#[test]
fn g_matrix_m1_slope() {
    // Reference at N_REF steps.
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

    // Sweep (stable regime: n ≥ 256; block-CN is unconditionally stable).
    let ns = [256_u32, 512, 1024, 2048];
    let errs: Vec<f64> = ns.iter().map(|&n| run_sweep(n, &reference)).collect();

    for (&n, &e) in ns.iter().zip(errs.iter()) {
        println!("G_MATRIX M=1: n={n} err={e:.4e}");
    }

    let slope = ols_slope(&ns, &errs);
    println!("G_MATRIX M=1: OLS slope = {slope:.4}");
    assert!(
        slope <= -1.95,
        "G_MATRIX M=1: slope {slope:.4} > -1.95 (gate FAILED)"
    );
}
