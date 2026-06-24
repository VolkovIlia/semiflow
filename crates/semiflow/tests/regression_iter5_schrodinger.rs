//! Regression tests for iter-5 Schrödinger kernel fix (Phase B Round 1).
//!
//! These tests capture the bug where `cn_kinetic_step_f64` was called with
//! a positive `a_off` (sign error), causing `SchrodingerChernoff` to evolve
//! as `e^{+iτH}` instead of the correct `e^{-iτH}`.
//!
//! ## Bug summary
//!
//! In `apply_strang_step`, line:
//!   `cn_kinetic_step_f64(n, half_tau_d * a0_d / (dx_d * dx_d), w);`
//!
//! passed a **positive** `a_off`, meaning `A = +τ·a·D²/2` (negative semidefinite).
//! The Cayley map `[(I−A²)m − 2Ar, r + A(m+m_new)]` with this sign implements
//! `e^{+iτK}` (time reversal). Negating `a_off` restores the correct direction.
//!
//! ## Tests
//!
//! 1. `regression_kinetic_phase_sign_fast` — short-time (T=0.01) phase test on the
//!    ground state of the kinetic operator alone (V=0).  Phase must be NEGATIVE
//!    (forward time).  This is a FAST (non-ignored) regression gate.
//!
//! 2. `regression_schrodinger_short_time_phase` — short evolution (T=0.1) with the
//!    harmonic oscillator; the imaginary part of ⟨ψ₀, ψ(T)⟩ must be negative
//!    (energy phase accumulates as e^{−iET}, not e^{+iET}).

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
// Integration test/bench: allows for numerical patterns.
#![allow(clippy::too_many_lines)]

use semiflow::{
    diffusion4::Diffusion4thChernoff, ChernoffFunction, Grid1D, GridFn1D, SchrodingerChernoff,
    SchrodingerState, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helper: build SchrodingerChernoff for harmonic oscillator, V = ω²x²/2
// ---------------------------------------------------------------------------

fn make_schr(n: usize) -> (SchrodingerChernoff<f64>, Grid1D<f64>) {
    let grid = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    let kinetic = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, grid);
    let schr = SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x).unwrap();
    (schr, grid)
}

fn make_state(grid: Grid1D<f64>, x_center: f64, sigma: f64) -> SchrodingerState<f64> {
    let n = grid.n;
    let psi_re = GridFn1D::from_fn(grid, |x: f64| {
        let xc = x - x_center;
        (-(xc * xc) / (2.0 * sigma * sigma)).exp()
    });
    let psi_im = GridFn1D {
        values: vec![0.0_f64; n],
        grid,
    };
    SchrodingerState { psi_re, psi_im }
}

fn evolve(
    schr: &SchrodingerChernoff<f64>,
    psi0: &SchrodingerState<f64>,
    tau: f64,
    n_steps: usize,
) -> SchrodingerState<f64> {
    let mut cur = psi0.clone();
    let mut nxt = psi0.clone();
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        schr.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

// ---------------------------------------------------------------------------
// Regression 1 (FAST): kinetic phase direction
//
// Ground state of -∂²/2 on [-5,5]: ψ₀ ≈ exp(-x²/2).
// After time T = 0.01 (V=0), the phase of ⟨ψ₀, ψ(T)⟩ must be NEGATIVE
// (correct Schrödinger evolution: e^{-iE₀T} has phase ≈ -0.5 * 0.01 = -0.005).
// ---------------------------------------------------------------------------

#[test]
fn regression_kinetic_phase_sign_fast() {
    // Use V=0 (free particle) to isolate kinetic sign.
    let n = 64_usize;
    let grid = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    let kinetic = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, grid);
    // V=0 → purely kinetic
    let schr = SchrodingerChernoff::new(kinetic, |_: f64| 0.0).unwrap();

    // Approximate ground state ψ₀ = exp(-x²/2) (well within [-5,5])
    let n_nodes = grid.n;
    let psi0 = SchrodingerState {
        psi_re: GridFn1D::from_fn(grid, |x: f64| (-x * x / 2.0).exp()),
        psi_im: GridFn1D {
            values: vec![0.0_f64; n_nodes],
            grid,
        },
    };

    let tau = 0.01_f64;
    let n_steps = 1_usize;
    let psi_t = evolve(&schr, &psi0, tau, n_steps);

    // Im⟨ψ₀, ψ(T)⟩ = ψ₀_re · ψ_T_im − ψ₀_im · ψ_T_re
    // For e^{-iE₀τ}: Im = ψ₀_re · (-sin(E₀τ)·ψ₀_re) ≈ -E₀τ * ‖ψ₀‖²  (NEGATIVE)
    let im_inner: f64 = psi0
        .psi_re
        .values
        .iter()
        .zip(&psi_t.psi_im.values)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        - psi0
            .psi_im
            .values
            .iter()
            .zip(&psi_t.psi_re.values)
            .map(|(a, b)| a * b)
            .sum::<f64>();

    let re_inner: f64 = psi0
        .psi_re
        .values
        .iter()
        .zip(&psi_t.psi_re.values)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + psi0
            .psi_im
            .values
            .iter()
            .zip(&psi_t.psi_im.values)
            .map(|(a, b)| a * b)
            .sum::<f64>();

    let phase = im_inner.atan2(re_inner);
    println!("regression_kinetic_phase_sign: phase = {phase:.6}  (expected < 0 for e^{{-iE₀τ}})");

    assert!(
        phase < 0.0,
        "REGRESSION: kinetic phase {phase:.6} is non-negative — \
         SchrodingerChernoff evolves in WRONG direction (e^{{+iHt}} instead of e^{{-iHt}}). \
         Root cause: a_off sign error in cn_kinetic_step_f64 call."
    );
}

// ---------------------------------------------------------------------------
// Regression 2 (FAST): short-time harmonic oscillator phase
//
// Harmonic oscillator V=½x², Gaussian at x=0 (≈ ground state), T=0.1.
// Phase of ⟨ψ₀, ψ(T)⟩ must be negative (energy accumulates with negative sign).
// ---------------------------------------------------------------------------

#[test]
fn regression_schrodinger_short_time_phase() {
    let n = 128_usize;
    let (schr, grid) = make_schr(n);
    // Ground-state Gaussian (centred at 0, σ=1 = harmonic oscillator GS width)
    let psi0 = make_state(grid, 0.0, 1.0);

    let tau = 0.1_f64 / 100.0;
    let n_steps = 100_usize;
    let psi_t = evolve(&schr, &psi0, tau, n_steps);

    let im_inner: f64 = psi0
        .psi_re
        .values
        .iter()
        .zip(&psi_t.psi_im.values)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        - psi0
            .psi_im
            .values
            .iter()
            .zip(&psi_t.psi_re.values)
            .map(|(a, b)| a * b)
            .sum::<f64>();

    let re_inner: f64 = psi0
        .psi_re
        .values
        .iter()
        .zip(&psi_t.psi_re.values)
        .map(|(a, b)| a * b)
        .sum::<f64>();

    let phase = im_inner.atan2(re_inner);
    println!(
        "regression_short_time_phase: Im⟨ψ₀,ψ(T)⟩ = {im_inner:.4e}, phase = {phase:.6} \
         (expected < 0 for forward-time Schrödinger)"
    );

    assert!(
        im_inner < 0.0,
        "REGRESSION: Im⟨ψ₀,ψ(T)⟩ = {im_inner:.4e} is non-negative. \
         SchrodingerChernoff evolves in wrong direction. \
         Root cause: a_off sign error in cn_kinetic_step_f64."
    );
}
