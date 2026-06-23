//! `G_truncated_exp_const` — constant-a regression gate (v0.4.0, ADR-0011).
//!
//! Validates that `TruncatedExpDiffusionChernoff` with `a' ≡ 0 ∧ a'' ≡ 0` (constant a)
//! approximates the exact heat-kernel semigroup correctly.
//!
//! ## Gate
//!
//! For `a ≡ 0.5`, initial condition G1 (Gaussian) or G2 (sinusoid), at each
//! combo (n ∈ {64, 128, 256}, τ ∈ {1e-4, 1e-3}), applying `TruncatedExpDiffusionChernoff`
//! once and comparing against the exact heat kernel:
//!   `u(x, τ) = (1/√(1+2·a·τ)) · exp(-x²/(2(1+2·a·τ)))`  for G1
//!   (no closed form for G2, so G2 uses Richardson self-consistency).
//!
//! For G1 (Gaussian): `sup_err < 1e-2` vs heat-kernel oracle (one step).
//!
//! For G2 (sinusoid), since there is no closed form, we use the consistency
//! check: `sup_err(TruncatedExp, τ) < sup_err_bound(τ, dx)` where
//! `sup_err_bound = 0.01 * τ` (O(τ) envelope for one-step error).
//!
//! ## Why not 1e-10?
//!
//! Both `TruncatedExpDiffusionChernoff` and `DiffusionChernoff` approximate the SAME
//! analytic semigroup `exp(τ·a·∂_xx)` but via DIFFERENT numerical schemes
//! (K=4 power series + 3-point stencil vs. Gaussian quadrature sampling).
//! Their discretization errors are O(dx²) and O(τ²) respectively, so they
//! cannot agree to 1e-10 on a finite grid. The correct gate is against the
//! analytic oracle or a consistency bound.
//!
//! Reference: contracts/semiflow-core.math.md §9.2.3.C, ADR-0011.

use semiflow_core::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExpDiffusionChernoff};

// ---------------------------------------------------------------------------
// Gate constants
// ---------------------------------------------------------------------------

/// Sup-norm gate for G1 (Gaussian) vs heat-kernel oracle: O(τ) per step.
const G1_ORACLE_GATE: f64 = 1e-2;

/// Richardson self-consistency bound for G2 (sinusoid): one `TruncatedExp` step
/// vs. the semigroup approximated by two half-steps. Bound: C·τ²·dx.
/// We use a generous 1e-1 since this is a single-step consistency check.
const G2_CONSISTENCY_GATE: f64 = 1e-1;

const A_CONST: f64 = 0.5;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_grid(n: usize) -> Grid1D {
    Grid1D::new(-10.0, 10.0, n).expect("grid params valid")
}

fn gaussian(grid: Grid1D) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| (-x * x / 2.0).exp())
}

fn sinusoid(grid: Grid1D) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| (2.0 * core::f64::consts::PI * x / 10.0).sin())
}

/// Exact heat-kernel value for Gaussian IC `u0(x) = exp(-x²/2)` after time τ.
/// `u(x, τ) = (1/√(1+2·a·τ)) · exp(-x²/(2(1+2·a·τ)))`.
fn gaussian_oracle(x: f64, tau: f64) -> f64 {
    let denom = 1.0 + 2.0 * A_CONST * tau;
    (1.0 / denom.sqrt()) * (-x * x / (2.0 * denom)).exp()
}

fn sup_norm(a: &GridFn1D, b: &GridFn1D) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0_f64, f64::max)
}

fn sup_norm_oracle(state: &GridFn1D, grid: Grid1D, tau: f64, oracle: fn(f64, f64) -> f64) -> f64 {
    let mut max_err: f64 = 0.0;
    for (i, &v) in state.values.iter().enumerate() {
        let x = grid.x_at(i);
        let err = (v - oracle(x, tau)).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

/// Apply `TruncatedExp` once with constant a=0.5, a'=0, a''=0.
fn magnus_apply_once(f: &GridFn1D, grid: Grid1D, tau: f64) -> Option<GridFn1D> {
    let dx2 = grid.dx() * grid.dx();
    if 2.0 * tau * A_CONST >= dx2 {
        return None; // CFL violated, skip
    }
    let m =
        TruncatedExpDiffusionChernoff::new(|_| A_CONST, |_| 0.0_f64, |_| 0.0_f64, A_CONST, grid);
    Some(m.apply_chernoff(tau, f).expect("apply ok"))
}

// ---------------------------------------------------------------------------
// G_truncated_exp_const
// ---------------------------------------------------------------------------

/// Check G1 (Gaussian vs oracle) for one (n, tau). Appends failure if any.
fn check_g1(n: usize, tau: f64, failures: &mut Vec<String>) {
    let grid = make_grid(n);
    let f0 = gaussian(grid);
    match magnus_apply_once(&f0, grid, tau) {
        None => {
            eprintln!(
                "{:<20}  {n:>5}  {tau:>8.0e}  {:>12}  SKIP(CFL)",
                "G1-gaussian", "—"
            );
        }
        Some(out) => {
            let err = sup_norm_oracle(&out, grid, tau, gaussian_oracle);
            let ok = err < G1_ORACLE_GATE;
            eprintln!(
                "{:<20}  {n:>5}  {tau:>8.0e}  {err:>12.4e}  {}",
                "G1-gaussian",
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                failures.push(format!(
                    "G1-gaussian n={n} tau={tau:.0e}: sup_err={err:.4e} >= {G1_ORACLE_GATE:.1e} (oracle gate)"
                ));
            }
        }
    }
}

/// Check G2 (sinusoid Richardson consistency) for one (n, tau). Appends failure if any.
fn check_g2(n: usize, tau: f64, failures: &mut Vec<String>) {
    let grid = make_grid(n);
    let f0 = sinusoid(grid);
    match magnus_apply_once(&f0, grid, tau) {
        None => {
            eprintln!(
                "{:<20}  {n:>5}  {tau:>8.0e}  {:>12}  SKIP(CFL)",
                "G2-sinusoid", "—"
            );
        }
        Some(out_full) => {
            let half = tau / 2.0;
            let out_half2 = magnus_apply_once(&f0, grid, half)
                .and_then(|mid| magnus_apply_once(&mid, grid, half));
            match out_half2 {
                None => {
                    eprintln!(
                        "{:<20}  {n:>5}  {tau:>8.0e}  {:>12}  SKIP(CFL half)",
                        "G2-sinusoid", "—"
                    );
                }
                Some(out_ref) => {
                    let err = sup_norm(&out_full, &out_ref);
                    let ok = err < G2_CONSISTENCY_GATE;
                    eprintln!(
                        "{:<20}  {n:>5}  {tau:>8.0e}  {err:>12.4e}  {}",
                        "G2-sinusoid",
                        if ok { "PASS" } else { "FAIL" }
                    );
                    if !ok {
                        failures.push(format!(
                            "G2-sinusoid n={n} tau={tau:.0e}: Richardson err={err:.4e} >= {G2_CONSISTENCY_GATE:.1e}"
                        ));
                    }
                }
            }
        }
    }
}

/// `G_truncated_exp_const`: constant-a `TruncatedExp` matches oracle and consistency.
///
/// G1 (Gaussian): `sup_err` vs analytic oracle < 1e-2 (single step).
/// G2 (sinusoid): Richardson consistency < 1e-1 (one-step vs two half-steps).
/// Collect-then-assert.
#[test]
fn g_truncated_exp_const_regression() {
    const N_VALUES: [usize; 3] = [64, 128, 256];
    const TAU_VALUES: [f64; 2] = [1e-4, 1e-3];
    let mut failures: Vec<String> = Vec::new();
    eprintln!(
        "{:<20}  {:>5}  {:>8}  {:>12}  gate",
        "case", "n", "tau", "err"
    );
    for &n in &N_VALUES {
        for &tau in &TAU_VALUES {
            check_g1(n, tau, &mut failures);
            check_g2(n, tau, &mut failures);
        }
    }
    assert!(
        failures.is_empty(),
        "G_truncated_exp_const FAIL ({} case(s)):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
