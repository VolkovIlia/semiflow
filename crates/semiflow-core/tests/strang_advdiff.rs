//! G1, G2 — advection-diffusion accuracy tests for `StrangSplit` (v0.2.0, ADR-0006).
//!
//! PDE: `∂_t u = (1/2)∂_xx u + (1/2)∂_x u`, `u(0,x) = exp(-x²)`.
//!
//! Oracle (closed form, math.md §6.2 and §9.5, Galilean substitution):
//!
//! ```text
//! u(t,x) = (1 + 2t)^{-1/2} · exp(-(x + t/2)² / (1 + 2t))
//! ```
//!
//! At `t = 1`: `u(1,x) = 3^{-1/2} · exp(-(x + 0.5)² / 3)`.
//! Mass centre translates from `x = 0` at `t = 0` to `x = -0.5` at `t = 1`,
//! verifying that the `B = (1/2)∂_x` step is exercised non-trivially.
//!
//! G1 (n=100):  `‖u_chernoff − u_oracle‖_∞ < 1.0e-4` (gate: NON-NEGOTIABLE)
//! G2 (n=1000): `‖u_chernoff − u_oracle‖_∞ < 1.0e-6` (gate: NON-NEGOTIABLE)
//!
//! Adapter: `StrangSplit<DiffusionChernoff(α=0.5), DriftReactionChernoff(β=0.5, c≡0)>`
//! Grid: 100000 uniform nodes on [-10, 10], Reflect BC, `CubicHermite` interp.
//! Grid amended from 1000 → 8000 (Amendment 3) then 8000 → 100000 (Amendment 4, ADR-0006):
//! N=8000 G3-strang slope was -1.07 (empirical K·n·Δx^2.32 scaling);
//! N=100000 predicts slope ≈ -1.998, G1 ≈ 2.7e-7, G2 ≈ 2.7e-9.
//!
//! Reference: `contracts/semiflow-core.math.md` §6.2, §9.5, §9.7; ADR-0006 v2+Amendment 4.

use semiflow_core::{
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, StrangSplit,
};

// ---------------------------------------------------------------------------
// Gate constants (NON-NEGOTIABLE — do NOT relax; see acceptance-criteria.md)
// ---------------------------------------------------------------------------

/// G1 gate: sup-norm error at n=100 must be strictly below this value.
const TOL_G1: f64 = 1.0e-4;

/// G2 gate: sup-norm error at n=1000 must be strictly below this value.
const TOL_G2: f64 = 1.0e-6;

/// Advection-diffusion parameters (§6.2, §9.5).
const ALPHA: f64 = 0.5; // diffusion coefficient a(x) = α
const BETA: f64 = 0.5; // drift coefficient b(x) = β
const T_FINAL: f64 = 1.0;
const N_NODES: usize = 100_000; // amended 2026-04-29: was 8000 (Amendment 3), now 100000 (Amendment 4)

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// Closed-form solution of `∂_t u = (1/2)∂_xx u + (1/2)∂_x u`, `u(0,x) = exp(-x²)`.
///
/// Normative formula from `contracts/semiflow-core.math.md §6.2, §9.5`
/// (Galilean substitution `v(t,y) = u(t, y − t/2)` reduces to pure heat
/// with `a = 1/2`; classical Gaussian-initial-datum solution):
///
/// ```text
/// u(t,x) = (1 + 2t)^{-1/2} · exp(-(x + t/2)² / (1 + 2t))
/// ```
///
/// At `t=1`: `u(1,x) = 3^{-1/2} · exp(-(x+0.5)²/3)`.
/// Mass centre at `x = -0.5` confirms the B-step is non-trivial.
///
/// Note: the denominator is `(1 + 2t)` — a concrete result for `a = 1/2`;
/// it is NOT `(1 + 2·α·t)` for general α.
fn oracle_advdiff(t: f64, x: f64) -> f64 {
    // Normative per math.md §6.2 box: (1 + 2t)^{-1/2} exp(-(x + t/2)^2 / (1 + 2t))
    let denom = 1.0 + 2.0 * t;
    let arg = (x + BETA * t).powi(2) / denom;
    denom.sqrt().recip() * (-arg).exp()
}

// ---------------------------------------------------------------------------
// Strang runner
// ---------------------------------------------------------------------------

/// Run `n_steps` Strang iterations from `t=0` to `t=T_FINAL` and return the
/// sup-norm error vs. the advection-diffusion oracle.
///
/// Grid: `N_NODES = 100_000` nodes on `[-10, 10]`, Reflect BC, `CubicHermite` interp.
/// Operator: `StrangSplit<DiffusionChernoff(α), DriftReactionChernoff(β, c≡0)>`.
fn strang_advdiff_error(n_steps: usize) -> f64 {
    let grid = Grid1D::new(-10.0, 10.0, N_NODES).expect("grid params valid");
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    // Diffusion sub-operator: A = α·∂²_x (5-point order-2 Chernoff).
    // v0.3.0 (ADR-0008 Amendment 1, ζ-A): a_prime = a_double_prime = |_| 0.0 for constant α
    // (a' ≡ a'' ≡ 0 ⇒ S(s) = id AND τ²-correction = 0 ⇒ D_ζ = D_γ = K = v0.2.2 bit-equal).
    let diff = DiffusionChernoff::new(|_| ALPHA, |_| 0.0_f64, |_| 0.0_f64, ALPHA, grid);

    // Drift sub-operator: B = β·∂_x + 0 (exact characteristic, c ≡ 0).
    let drift = DriftReactionChernoff::new(|_| BETA, |_| 0.0_f64, 0.0, grid);

    // Strang sandwich: Φ(τ) = D(τ/2) ∘ R(τ) ∘ D(τ/2), global order 2.
    let strang = StrangSplit::new(diff, drift);
    let semi = ChernoffSemigroup::new(strang, n_steps).expect("n >= 1");
    let u_n = semi
        .evolve(T_FINAL, &u0)
        .expect("evolve succeeds for valid inputs");

    // Sup-norm error vs. oracle.
    let mut max_err: f64 = 0.0;
    for i in 0..N_NODES {
        let x = grid.x_at(i);
        let err = (u_n.values[i] - oracle_advdiff(T_FINAL, x)).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// G1 — n = 100
// ---------------------------------------------------------------------------

/// G1: sup-norm error at n=100 must satisfy `‖err‖_∞ < 1.0e-4`.
///
/// Gate from `acceptance-criteria.md §G1` (v0.2.0, ADR-0006 v2).
/// Non-negotiable: if this fails, report the empirical number and escalate.
#[test]
fn g1_strang_advdiff_n100() {
    let err = strang_advdiff_error(100);
    println!("G1: sup-norm error at n=100 = {err:.6e}  (gate: < {TOL_G1:.0e})");
    assert!(
        err < TOL_G1,
        "G1 FAIL: {err:.6e} >= {TOL_G1:.0e} — Gate FAILED, escalate to architect"
    );
}

// ---------------------------------------------------------------------------
// G2 — n = 1000
// ---------------------------------------------------------------------------

/// G2: sup-norm error at n=1000 must satisfy `‖err‖_∞ < 1.0e-6`.
///
/// Gate from `acceptance-criteria.md §G2` (v0.2.0, ADR-0006 v2).
/// Non-negotiable: if this fails, report the empirical number and escalate.
#[test]
fn g2_strang_advdiff_n1000() {
    let err = strang_advdiff_error(1000);
    println!("G2: sup-norm error at n=1000 = {err:.6e}  (gate: < {TOL_G2:.0e})");
    assert!(
        err < TOL_G2,
        "G2 FAIL: {err:.6e} >= {TOL_G2:.0e} — Gate FAILED, escalate to architect"
    );
}
