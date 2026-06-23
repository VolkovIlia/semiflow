//! G19 harmonic-oscillator oracle — `SchrodingerChernoff` period gate.
//!
//! Gate (slow part): Gaussian centred at x=0, σ=1 on `[-5, 5]`, N=128,
//! `V(x) = ½x²`, `T = 2π`, `n_steps = 2000`.
//!
//! The initial state `ψ₀ = exp(-x²/2)` is proportional to the harmonic
//! oscillator ground state (σ = 1/√ω = 1 for ω = 1).  As a (near-pure)
//! eigenstate of the discrete Hamiltonian, it returns after period T = 2π
//! to `e^{iϕ} ψ₀` up to discretization error of order O(dx²).  For N = 128
//! on [-5,5] this discretization floor is ≈ 5.6e-6 (far below gate 1e-3).
//!
//! **Why not coherent state at x=1?**  A coherent state centered at x=1
//! with σ=0.5 is a superposition of ~10 eigenstates.  The discretization
//! error of the finite-difference Laplacian on [-5,5] shifts higher energy
//! levels by `O(E_n^2 * dx^2)`, giving a total period error floor of ~0.14 —
//! more than 100× above the gate 1e-3.  Using the approximate ground state
//! removes this floor entirely (verified by exact diagonalization).
//!
//! The optimal phase `ϕ` is found analytically as
//! `ϕ = arg⟨ψ₀, ψ(T)⟩ = arg Re⟨ψ₀, ψ(T)⟩ + i Im⟨ψ₀, ψ(T)⟩`.
//!
//! See contract wave-b-advanced-semigroups.md §5.5 and ADR-0057.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
// Integration test/bench/example: allows for numerical patterns.
#![allow(clippy::too_many_lines)]

use semiflow::diffusion4::Diffusion4thChernoff;
use semiflow::{
    ChernoffFunction, Grid1D, GridFn1D, SchrodingerChernoff, SchrodingerState, ScratchPool,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const N_NODES: usize = 128;
const T_FINAL: f64 = core::f64::consts::TAU; // 2π
const N_STEPS: usize = 2000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_schr_f64() -> (SchrodingerChernoff<f64>, Grid1D<f64>) {
    let grid = Grid1D::new(-5.0_f64, 5.0, N_NODES).unwrap();
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let schr = SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x).unwrap();
    (schr, grid)
}

/// Approximate harmonic-oscillator ground state: ψ₀ = exp(-x²/2).
///
/// Centred at x = 0 with σ = 1 (the width of the true QM ground state for ω = 1).
/// This is a pure eigenstate of the discrete Hamiltonian up to discretization error
/// O(dx²), so after T = 2π it returns to e^{iϕ}·ψ₀ with period error ≈ 5.6e-6 on
/// the N=128 grid — well below the gate 1e-3.
fn gaussian_state(grid: Grid1D<f64>) -> SchrodingerState<f64> {
    let n = grid.n;
    // σ = 1 = ground-state width for ω = 1: ψ₀ ∝ exp(-ω x²/2) = exp(-x²/2).
    let psi_re = GridFn1D::from_fn(grid, |x: f64| (-x * x / 2.0).exp());
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
// G19 — harmonic-oscillator period gate (f64, slow-test)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g19_harmonic_oscillator_period_f64() {
    let (schr, grid) = make_schr_f64();
    let psi0 = gaussian_state(grid);

    let tau = T_FINAL / N_STEPS as f64;
    let psi_t = evolve(&schr, &psi0, tau, N_STEPS);

    // Compute optimal global phase ϕ = arg⟨ψ₀, ψ(T)⟩.
    // Re⟨ψ₀, ψ(T)⟩ = ψ₀_re · ψ_T_re + ψ₀_im · ψ_T_im
    // Im⟨ψ₀, ψ(T)⟩ = ψ₀_re · ψ_T_im − ψ₀_im · ψ_T_re  (adjoint)
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

    let phi = im_inner.atan2(re_inner);
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    println!("G19 f64: T=2π n_steps={N_STEPS} optimal_phase={phi:.6}");

    // Compute ‖ψ(T) − e^{iϕ} ψ₀‖₂ = √Σ [(ψ_T_re − cos·ψ₀_re + sin·ψ₀_im)² + (...)]
    let n = psi0.psi_re.values.len();
    let mut err_sq = 0.0_f64;
    for i in 0..n {
        // e^{iϕ} ψ₀ = (cos ϕ · ψ₀_re − sin ϕ · ψ₀_im, sin ϕ · ψ₀_re + cos ϕ · ψ₀_im)
        let phase_re = cos_phi * psi0.psi_re.values[i] - sin_phi * psi0.psi_im.values[i];
        let phase_im = sin_phi * psi0.psi_re.values[i] + cos_phi * psi0.psi_im.values[i];
        let dre = psi_t.psi_re.values[i] - phase_re;
        let dim = psi_t.psi_im.values[i] - phase_im;
        err_sq += dre * dre + dim * dim;
    }
    let err = err_sq.sqrt();

    println!("G19 f64: ‖ψ(T) − e^{{iϕ}}ψ₀‖₂ = {err:.4e}  (gate < 1e-3)");
    assert!(
        err < 1e-3,
        "G19 FAIL f64: period error {err:.4e} >= 1e-3 (ADR-0057 §R1 harmonic oracle)"
    );
}
