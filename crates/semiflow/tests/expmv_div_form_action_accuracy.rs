//! `g_expmv_div_form_action_accuracy` — ADR-0121 backward-error gate.
//!
//! ## Gate specification (§45.4 / ADR-0121)
//!
//! `expmv` is tolerance-driven, NOT fixed-order. This gate asserts a backward-error
//! bound, NOT a convergence slope. A slope assertion of "≤ −8" would be INAPPLICABLE.
//!
//! Setup:
//! - Divergence-form `A = ∂_x(a(x) ∂_x)` with `a(x) = 1 + 0.3·sin(2π·x/L)`.
//! - Grid: `N = 64` on `[0, L]`, `L = 20.0`. Neumann BCs (mirrors PRE-FLIGHT harness).
//! - τ chosen so `τ·‖A‖_est ≈ 40` (well into the blow-up regime that defeated the
//!   Padé kernel at `τ‖A‖ ≈ 62`; PRE-FLIGHT used 62).
//! - Reference: `expmv_action` at `(s_ref, m=18)` with per-step arg ≤ 1.0 (well inside
//!   the `θ_18=8.84` radius; `T_18` at arg≤1 is sub-round-off accurate per PRE-FLIGHT (a)).
//!   This is the self-convergence reference pattern (mirrors `G_zeta8` §27.tris).
//! - Tested kernel: `DiffusionExpmvChernoff::apply_into` with Algorithm-3.2
//!   auto-selected `(s, m)`.
//!
//! Gate: `sup_error ≤ 1e-11` (one order above the Chebyshev ζ⁸ floor `4.17e-12`
//! for discretisation headroom; ADR-0121 §Gate). PRE-FLIGHT measured `1.1e-15`.
//!
//! ## References
//!
//! - ADR-0121 — PRE-FLIGHT GO; engineer spec.
//! - math.md §45.4 — gate NORMATIVE.
//! - `scripts/verify_expmv_preflight.py` — PRE-FLIGHT harness (executed 2026-06-05).
//! - A. H. Al-Mohy, N. J. Higham (2011), SIAM J. Sci. Comput. 33(2):488–511.

// Integration test/example: allows for numerical patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

extern crate alloc;

use semiflow_core::{
    chernoff::ChernoffFunction, Diffusion4thChernoff, DiffusionExpmvChernoff, Grid1D, GridFn1D,
    ScratchPool,
};

const X_MIN: f64 = 0.0;
const X_MAX: f64 = 20.0;
const N_SPATIAL: usize = 64; // mirrors PRE-FLIGHT harness (N=64)
const L: f64 = X_MAX - X_MIN;

/// `a(x) = 1 + 0.3·sin(2π·x/L)` — matches the PRE-FLIGHT harness exactly.
fn a_fn(x: f64) -> f64 {
    1.0 + 0.3 * libm::sin(2.0 * core::f64::consts::PI * x / L)
}

/// Build the inner `Diffusion4thChernoff` for the gate.
fn make_inner(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(a_fn, |_| 0.0, |_| 0.0, 1.3, grid)
}

/// Apply divergence-form `A·f` in place (local duplicate to avoid pub(crate) use).
fn apply_av(inner: &Diffusion4thChernoff<f64>, f: &GridFn1D<f64>, out: &mut alloc::vec::Vec<f64>) {
    let n = f.values.len();
    let dx = inner.grid.dx();
    let dx2 = dx * dx;
    out.resize(n, 0.0);
    for i in 0..n {
        let xi = inner.grid.x_at(i);
        let ap = a_fn(xi + 0.5 * dx);
        let an = a_fn(xi - 0.5 * dx);
        let fp = if i + 1 < n {
            f.values[i + 1]
        } else {
            f.values[n - 1]
        };
        let fn_ = if i > 0 { f.values[i - 1] } else { f.values[0] };
        let fi = f.values[i];
        out[i] = (ap * (fp - fi) - an * (fi - fn_)) / dx2;
    }
}

/// High-accuracy reference: `T_18(τ_s` · A)^s with per-step arg ≤ 1.0.
///
/// At arg ≤ 1 ≤ `θ_8` = 1.44, `T_18` is sub-round-off accurate (PRE-FLIGHT (a)).
/// Cost: `s_ref` * 18 mat-vecs where `s_ref` = ceil(τ·‖A‖/1.0).
fn reference_expmv(
    inner: &Diffusion4thChernoff<f64>,
    tau: f64,
    norm_a_est: f64,
    src: &GridFn1D<f64>,
) -> GridFn1D<f64> {
    let theta_ref = 1.0_f64; // per-step arg target ≤ 1.0 (within θ_8=1.44, T_18 accurate)
    let s_ref = ((tau * norm_a_est / theta_ref).ceil() as u32).max(1);
    let m_ref = 18u32;
    let tau_s = tau / f64::from(s_ref);
    let n = src.values.len();
    let grid = inner.grid;

    let mut y_vals = src.values.clone();
    let mut w_vals = alloc::vec![0.0_f64; n];
    let mut av_vals = alloc::vec![0.0_f64; n];

    for _ in 0..s_ref {
        // One Horner step: T_m(τ_s · A) applied to y.
        w_vals.clone_from(&y_vals);
        let w_fn = GridFn1D {
            grid,
            values: w_vals,
        };
        let mut w_fn = w_fn; // make mutable
        for k in 1..=m_ref {
            apply_av(inner, &w_fn, &mut av_vals);
            let factor = tau_s / f64::from(k);
            for (wi, &avi) in w_fn.values.iter_mut().zip(av_vals.iter()) {
                *wi = factor * avi;
            }
            for (yi, &wi) in y_vals.iter_mut().zip(w_fn.values.iter()) {
                *yi += wi;
            }
        }
        w_vals = w_fn.values;
    }
    GridFn1D {
        grid,
        values: y_vals,
    }
}

/// `g_expmv_div_form_action_accuracy` (ADR-0121 §45.4).
///
/// Backward-error gate for `DiffusionExpmvChernoff`. Asserts `sup_error ≤ 1e-11`
/// against a high-`s` self-converged reference in the `τ‖A‖ ≈ 40` regime.
/// No slope assertion — `expmv` is tolerance-driven, not fixed-order.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_expmv_div_form_action_accuracy() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");
    let inner = make_inner(grid);

    // ‖A‖ estimate: 4 · a_norm_bound / dx²
    let dx = grid.dx();
    let norm_a_est = 4.0 * inner.a_norm_bound / (dx * dx);

    // Choose τ so τ·‖A‖_est ≈ 40 (well inside the blow-up regime).
    let tau = 40.0 / norm_a_est;

    eprintln!(
        "g_expmv_div_form_action_accuracy: N={N_SPATIAL}, τ={tau:.4e}, τ·‖A‖={:.2}",
        tau * norm_a_est
    );

    // IC: f₀(x) = exp(−(x − L/2)²) — smooth Gaussian matching PRE-FLIGHT harness.
    let mid = (X_MIN + X_MAX) / 2.0;
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-(x - mid) * (x - mid)));

    // High-accuracy self-converged reference (per-step arg ≤ 1.0, m=18).
    let u_ref = reference_expmv(&inner, tau, norm_a_est, &f0);

    // Tested kernel (Algorithm 3.2 auto-selector).
    let kernel = DiffusionExpmvChernoff::new(inner);
    let mut u_test = GridFn1D::from_fn(grid, |_| 0.0);
    let mut scratch = ScratchPool::new();
    kernel
        .apply_into(tau, &f0, &mut u_test, &mut scratch)
        .expect("expmv apply_into must not fail");

    // sup-norm error against the reference.
    let sup_error = u_test
        .values
        .iter()
        .zip(u_ref.values.iter())
        .map(|(&a_val, &b_val)| (a_val - b_val).abs())
        .fold(0.0_f64, f64::max);

    let ref_norm = u_ref
        .values
        .iter()
        .map(|&v| v.abs())
        .fold(0.0_f64, f64::max);

    eprintln!("  ref_norm    = {ref_norm:.4e}");
    eprintln!("  sup_error   = {sup_error:.4e}  (gate ≤ 1e-11)");
    eprintln!(
        "  rel_error   = {:.4e}",
        if ref_norm > 0.0 {
            sup_error / ref_norm
        } else {
            0.0
        }
    );

    assert!(
        sup_error <= 1e-11,
        "g_expmv_div_form_action_accuracy FAILED: sup_error={sup_error:.4e} > 1e-11 \
         (ADR-0121 backward-error gate; PRE-FLIGHT measured 1.1e-15)"
    );

    eprintln!("  PASS");
}
