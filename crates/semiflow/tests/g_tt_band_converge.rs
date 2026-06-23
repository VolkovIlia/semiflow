//! `G_TT_BAND_CONVERGE` — P2 acceptance gate: 1D self-convergence O(τ²).
//!
//! # Purpose (`RELEASE_BLOCKING`, P2 gate, §52.2 Amd 1 / §6.2, ADR-0162)
//!
//! Proves that the REAL `TtChernoff` evolve path (cubic-Lagrange band-split
//! shift, `apply_per_axis_shift` in `tt_chernoff.rs`) converges to the true
//! 1D heat semigroup at O(τ²)/O(dx²) under joint parabolic refinement, at a
//! NON-integer `h/dx` ratio where the OLD integer-rounded scheme PLATEAUS.
//!
//! This is the P2 acceptance gate required by §8.4 ("1D self-convergence O(τ²)").
//! Gate B in `g_tt_strang_identity.rs` CANNOT serve this purpose: it deliberately
//! sets `h/dx = 1` (an integer, see lines 48–58 of that file), at which the
//! cubic-Lagrange weights degenerate to `[0,1,0,0]` — plain integer shift —
//! making Gate B invariant to the band-split fix.
//!
//! # Anti-vacuity device (CRITICAL)
//!
//! The convergence reference is the analytic FFT heat truth:
//!   `u(·, T) = IFFT( exp(−a·k²·T) · FFT(u₀) )`
//! This is computed INDEPENDENTLY of the `TtChernoff` operator — it uses the
//! FFT-spectral formula, not any re-implementation of the band-split shift.
//! Using `TtChernoff` or a hand-rolled copy of the shift as the reference would
//! be vacuous self-comparison (the recurring defect caught 4× in this audit).
//!
//! # Regression witness (band-split vs integer-rounded)
//!
//! At the SAME grid/time parameters, the INTEGER-ROUNDED shift (a test-local
//! `integer_shift_heat_err` function) is verified to PLATEAU at error ≥ 0.10.
//! This proves that the band-split — not mere grid refinement — enables
//! convergence: integer error stays ≥ 0.10 at the finest grid where band-split
//! reaches < 1e-3.
//!
//! # Run
//! ```bash
//! cargo test -p semiflow-core --features slow-tests \
//!   --test g_tt_band_converge -- --ignored --nocapture
//! ```
//!
//! References:
//!   - math.md §52.2 Amendment 1 (band-split shift, normative)
//!   - .dev-docs/specs/v9.1.0-s3-triz-resolution.md §6.2 + §8.4
//!   - `scripts/tt_band_shift_kit.py` sub-check (c) (validated reference behaviour)
//!   - Kazeev & Khoromskij 2012 (QTT-op rank of banded Toeplitz operators)

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)] // f64→usize n_steps: rounded positive value
#![allow(clippy::cast_sign_loss)]           // f64→usize n_steps: .round().max(1.0) is positive
#![allow(clippy::cast_possible_wrap)]       // usize→isize n: n ≤ 512 on all supported targets
#![allow(clippy::needless_range_loop)]      // DFT inner loops use index arithmetic (k*j)
#![allow(clippy::many_single_char_names)]   // n, a, l, k, j are standard math variable names
#![allow(clippy::too_many_lines)]           // g_tt_band_converge is a single cohesive gate

extern crate alloc;
use alloc::vec::Vec;

use semiflow::{TtChernoff, TtState};

// ─── Pre-registered parameters ─────────────────────────────────────────────

/// Diffusion coefficient — matches the §6.2 probe.
const A: f64 = 0.7;

/// Target evolution time.
const T_FINAL: f64 = 0.05;

/// Domain (periodic unit interval).
const X_MIN: f64 = 0.0;
const X_MAX: f64 = 1.0;

/// Grid sizes for the joint parabolic refinement.
/// Each level doubles the grid; τ ∼ C·dx² is recomputed per level.
const N_GRIDS: [usize; 3] = [128, 256, 512];

/// Target h/dx ratio — held fixed across refinement.
/// 1.35 is deliberately NON-integer (matches the §6.2 convergence probe).
const H_OVER_DX_TARGET: f64 = 1.35;

/// Slope gate: log-log convergence rate ≥ this value (O(τ²) requires ≥ 2).
/// Relaxed slightly to 1.8 to tolerate leading-order pre-asymptotic terms.
const SLOPE_GATE: f64 = 1.8;

/// Band-split finest-grid error gate.
const BAND_FINEST_GATE: f64 = 1e-3;

/// Integer-rounded plateau gate: must stay ≥ this at finest grid.
const INT_PLATEAU_GATE: f64 = 0.10;

/// TT-rounding tolerance — zero so no rank truncation interferes.
const EPS_ROUND: f64 = 0.0;

// ─── Reference: analytic FFT heat truth ────────────────────────────────────

/// Compute the exact 1D heat semigroup action via FFT-spectral formula.
///
/// `u_exact(x, T) = IFFT( exp(-a·k²·T) · FFT(u₀)(k) )`
///
/// This is the INDEPENDENT reference — it uses no part of the `TtChernoff`
/// operator and no re-implementation of the band-split shift.
///
/// Parameters:
///   - `u0`:   initial condition on `n` periodic grid points
///   - `a`:    diffusion coefficient
///   - `t_end`: evolution time
///   - `l`:    domain length for the periodic DFT (`L = n * dx_periodic`)
fn heat_fft_reference(u0: &[f64], a: f64, t_end: f64, l: f64) -> Vec<f64> {
    let n = u0.len();
    // Compute DFT manually — O(n²), sufficient for n ≤ 512.
    // Forward DFT: û_k = Σ_j u_j · exp(-2πi·k·j/n)
    let pi2 = core::f64::consts::PI * 2.0;
    let mut u_hat_re = vec![0.0f64; n];
    let mut u_hat_im = vec![0.0f64; n];
    for k in 0..n {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (j, &uj) in u0.iter().enumerate() {
            let angle = pi2 * (k * j) as f64 / n as f64;
            re += uj * angle.cos();
            im -= uj * angle.sin();
        }
        u_hat_re[k] = re;
        u_hat_im[k] = im;
    }
    // Multiply by heat kernel: exp(-a · ω_k² · t_end)
    // ω_k = 2π/L · k_fq, k_fq centered: k if k≤n/2 else k-n.
    for k in 0..n {
        let k_fq = if k <= n / 2 {
            k as f64
        } else {
            k as f64 - n as f64
        };
        let omega = pi2 / l * k_fq;
        let damp = (-a * omega * omega * t_end).exp();
        u_hat_re[k] *= damp;
        u_hat_im[k] *= damp;
    }
    // Inverse DFT: u_j = (1/n) · Σ_k û_k · exp(+2πi·k·j/n)
    let mut result = vec![0.0f64; n];
    for j in 0..n {
        let mut re = 0.0f64;
        for k in 0..n {
            let angle = pi2 * (k * j) as f64 / n as f64;
            re += u_hat_re[k] * angle.cos() - u_hat_im[k] * angle.sin();
        }
        result[j] = re / n as f64;
    }
    result
}

// ─── Integer-shift reference (test-local, regression witness only) ──────────

/// Apply one Chernoff 3-branch heat step with INTEGER-ROUNDED shift.
///
/// `u_new[i] = ¼·u[i + s] + ¼·u[i − s] + ½·u[i]`  (periodic wrap)
///
/// `s = round(h/dx)`. This is the OLD pre-P2 scheme that plateaus at h/dx ≈ 1.35.
/// It is intentionally test-local (no production path) to serve as the anti-vacuity
/// regression witness: at the same grid, this should stay at error ≥ `INT_PLATEAU_GATE`.
///
/// Uses the same `dx = (X_MAX - X_MIN)/(n-1)` as `TtChernoff`, and passes
/// `L = n * dx` to the FFT reference so the wavenumber grid is consistent.
fn integer_shift_heat_err(n: usize, a: f64, t_end: f64) -> f64 {
    // TtChernoff's grid spacing: (xmax-xmin)/(n-1)
    let dx = (X_MAX - X_MIN) / (n as f64 - 1.0);
    // Periodic domain length for DFT: n points at spacing dx → L = n*dx
    let l = n as f64 * dx;
    let ratio = H_OVER_DX_TARGET;
    // Compute τ from the target ratio: h = ratio·dx = 2√(a·τ) → τ = (ratio·dx/2)²/a
    let h_target = ratio * dx;
    let tau_target = (h_target / 2.0).powi(2) / a;
    let n_steps = (t_end / tau_target).round().max(1.0) as usize;
    let tau = t_end / n_steps as f64;
    let h = 2.0 * (a * tau).sqrt();
    // Integer-rounded shift index
    let s_raw = (h / dx).round() as isize;
    let s = (s_raw.rem_euclid(n as isize)) as usize;
    let u0: Vec<f64> = (0..n)
        .map(|i| {
            let x = X_MIN + i as f64 * dx;
            let cx = 0.5f64 * (X_MIN + X_MAX);
            (-(x - cx).powi(2) / (2.0 * 0.02)).exp()
        })
        .collect();
    let truth = heat_fft_reference(&u0, a, t_end, l);
    let mut u = u0;
    if s == 0 {
        let norm_t: f64 = truth.iter().map(|v| v * v).sum::<f64>().sqrt();
        let err: f64 = u
            .iter()
            .zip(truth.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        return if norm_t > 1e-300 { err / norm_t } else { 0.0 };
    }
    for _ in 0..n_steps {
        let mut u_new = vec![0.0f64; n];
        for i in 0..n {
            let fwd = (i + s) % n;
            let bwd = (i + n - s) % n;
            u_new[i] = 0.25 * u[fwd] + 0.25 * u[bwd] + 0.5 * u[i];
        }
        u = u_new;
    }
    let norm_t: f64 = truth.iter().map(|v| v * v).sum::<f64>().sqrt();
    let err: f64 = u
        .iter()
        .zip(truth.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm_t > 1e-300 {
        err / norm_t
    } else {
        0.0
    }
}

// ─── Band-split error via REAL TtChernoff path ──────────────────────────────

/// Run the REAL `TtChernoff` evolver (1D, the actual band-split shift) and
/// compute the relative L2 error vs the analytic FFT heat truth.
///
/// The 1D `TtChernoff` with `d=1` exercises exactly `apply_per_axis_shift`
/// in `tt_chernoff.rs` — the P2 production path, NOT a local copy.
///
/// `dx` used here matches `TtChernoff`'s internal `dx = (xmax-xmin)/(n-1)`.
/// The FFT reference uses domain length `L = n * dx` (periodic convention).
fn band_split_heat_err(n: usize, a: f64, t_end: f64) -> (f64, f64) {
    // TtChernoff's grid spacing: (xmax-xmin)/(n-1)
    let dx = (X_MAX - X_MIN) / (n as f64 - 1.0);
    // Periodic domain length for DFT
    let l = n as f64 * dx;
    let h_target = H_OVER_DX_TARGET * dx;
    let tau_target = (h_target / 2.0).powi(2) / a;
    let n_steps = (t_end / tau_target).round().max(1.0) as usize;
    // Actual h/dx after rounding n_steps (so we can report it)
    let tau_actual = t_end / n_steps as f64;
    let h_actual = 2.0 * (a * tau_actual).sqrt();
    let h_over_dx = h_actual / dx;
    // Build 1D IC: Gaussian, using TtChernoff's grid spacing
    let cx = 0.5f64 * (X_MIN + X_MAX);
    let u0: Vec<f64> = (0..n)
        .map(|i| {
            let x = X_MIN + i as f64 * dx;
            (-(x - cx).powi(2) / (2.0 * 0.02)).exp()
        })
        .collect();
    // Analytic truth via FFT (INDEPENDENT reference — uses the same IC)
    let truth = heat_fft_reference(&u0, a, t_end, l);
    // Band-split evolution via REAL TtChernoff (d=1)
    let evolver = TtChernoff::new(
        vec![a],
        vec![0.0f64],
        0.0f64,
        vec![(X_MIN, X_MAX)],
        EPS_ROUND,
    );
    let mut state = TtState::rank1_separable(vec![u0]);
    evolver.evolve(t_end, n_steps, &mut state);
    // Extract the 1D result from the rank-1 TT state (core 0, slice)
    let u_tt: Vec<f64> = (0..n).map(|i| state.cores[0].get(0, i, 0)).collect();
    let norm_t: f64 = truth.iter().map(|v| v * v).sum::<f64>().sqrt();
    let err: f64 = u_tt
        .iter()
        .zip(truth.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let rel_err = if norm_t > 1e-300 { err / norm_t } else { 0.0 };
    (rel_err, h_over_dx)
}

// ─── Slope helper ──────────────────────────────────────────────────────────

/// Compute the 2-point log-log convergence slope of `errors` vs `dxs`.
///
/// `slope = log2(errors[0] / errors[1]) / log2(dxs[0] / dxs[1])`
/// Uses the first and last data points for the widest baseline.
fn two_point_log_log_slope(dxs: &[f64], errors: &[f64]) -> f64 {
    let n = dxs.len();
    assert!(n >= 2);
    let dx0 = dxs[0];
    let dx1 = dxs[n - 1];
    let e0 = errors[0];
    let e1 = errors[n - 1];
    if e0 <= 0.0 || e1 <= 0.0 || dx0 <= 0.0 || dx1 <= 0.0 {
        return 0.0;
    }
    (e0 / e1).ln() / (dx0 / dx1).ln()
}

// ─── Main gate ─────────────────────────────────────────────────────────────

#[test]
#[ignore = "slow P2 gate; run with: cargo run -p xtask -- test-flagship"]
fn g_tt_band_converge() {
    println!();
    println!("{}", "═".repeat(72));
    println!("G_TT_BAND_CONVERGE — P2 acceptance gate: 1D self-convergence O(τ²)");
    println!("  math.md §52.2 Amd 1 / .dev-docs/specs/v9.1.0-s3-triz-resolution §6.2/§8.4");
    println!("{}", "═".repeat(72));
    println!();
    println!("Reference: analytic FFT heat truth (INDEPENDENT of TtChernoff).");
    println!("  u_exact = IFFT( exp(-a·ω²·T) · FFT(u₀) )");
    println!("  a={A}, T={T_FINAL}, target h/dx≈{H_OVER_DX_TARGET} (NON-integer)");
    println!("  Joint parabolic refinement: τ ∼ C·dx², grids {N_GRIDS:?}");
    println!();

    // ── Band-split errors via REAL TtChernoff ─────────────────────────────
    println!("Band-split TtChernoff (cubic-Lagrange, P2 production path):");
    println!("  n     | τ       | h/dx    | rel-L2 err  | int-rnd err");
    println!("  {}", "-".repeat(58));

    let mut dxs: Vec<f64> = Vec::new();
    let mut band_errs: Vec<f64> = Vec::new();
    let mut int_errs: Vec<f64> = Vec::new();
    let mut h_over_dx_vals: Vec<f64> = Vec::new();

    for &n in &N_GRIDS {
        // TtChernoff's dx = (xmax-xmin)/(n-1)
        let dx = (X_MAX - X_MIN) / (n as f64 - 1.0);
        let (band_err, h_over_dx) = band_split_heat_err(n, A, T_FINAL);
        let int_err = integer_shift_heat_err(n, A, T_FINAL);
        // Compute actual τ for display
        let h_target = H_OVER_DX_TARGET * dx;
        let tau_target = (h_target / 2.0).powi(2) / A;
        let n_steps = (T_FINAL / tau_target).round().max(1.0) as usize;
        let tau_actual = T_FINAL / n_steps as f64;
        println!("  {n:>5} | {tau_actual:.4e} | {h_over_dx:.4}  | {band_err:.4e}  | {int_err:.4e}");
        dxs.push(dx);
        band_errs.push(band_err);
        int_errs.push(int_err);
        h_over_dx_vals.push(h_over_dx);
    }

    // ── Non-integer h/dx assertion ─────────────────────────────────────────
    println!();
    println!("Ratio h/dx validation (must be NON-integer for anti-vacuity):");
    for (i, &hd) in h_over_dx_vals.iter().enumerate() {
        let frac = (hd - hd.round()).abs();
        println!(
            "  n={} : h/dx={:.4}  fractional-part={:.4}  {}",
            N_GRIDS[i],
            hd,
            frac,
            if frac > 0.05 {
                "NON-INTEGER ✓"
            } else {
                "WARNING: near-integer!"
            }
        );
        assert!(
            frac > 0.05,
            "h/dx={hd:.4} is too close to an integer (frac={frac:.4} ≤ 0.05). \
             Anti-vacuity requires a strictly non-integer ratio to prevent degeneration \
             of cubic-Lagrange weights to plain integer shift."
        );
    }

    // ── Convergence slope assertion ───────────────────────────────────────
    let slope = two_point_log_log_slope(&dxs, &band_errs);
    println!();
    println!("Convergence slope (log-log, first→last grid):");
    println!("  slope = {slope:.4}  (gate: ≥ {SLOPE_GATE})");
    if slope >= SLOPE_GATE {
        println!("  O(τ²) convergence CONFIRMED ✓");
    } else {
        println!("  FAIL: slope below O(τ²) gate");
    }

    // ── Finest-grid accuracy assertion ────────────────────────────────────
    let band_finest = *band_errs.last().unwrap();
    println!();
    println!("Finest-grid band-split error:");
    println!("  err = {band_finest:.4e}  (gate: < {BAND_FINEST_GATE})");
    if band_finest < BAND_FINEST_GATE {
        println!("  Band-split accuracy CONFIRMED ✓");
    } else {
        println!("  FAIL: finest-grid error above gate");
    }

    // ── Integer-plateau regression witness ────────────────────────────────
    let int_finest = *int_errs.last().unwrap();
    println!();
    println!("Integer-rounded scheme plateau (regression witness):");
    println!("  finest-grid int error = {int_finest:.4e}  (gate: ≥ {INT_PLATEAU_GATE})");
    if int_finest >= INT_PLATEAU_GATE {
        println!("  Integer plateau CONFIRMED ✓  — band-split fix is essential");
    } else {
        println!("  UNEXPECTED: integer error below plateau (check h/dx ratio)");
    }

    // ── Verdict ────────────────────────────────────────────────────────────
    println!();
    println!("{}", "═".repeat(72));
    let all_pass =
        slope >= SLOPE_GATE && band_finest < BAND_FINEST_GATE && int_finest >= INT_PLATEAU_GATE;
    if all_pass {
        println!("G_TT_BAND_CONVERGE PASS");
        println!("  Band-split TtChernoff converges O(τ²) to the analytic heat truth.");
        println!("  Integer-rounded scheme plateaus at {int_finest:.4e} ≥ {INT_PLATEAU_GATE}.");
        println!("  The band-split fix — not grid refinement — enables convergence.");
    } else {
        println!("G_TT_BAND_CONVERGE FAIL");
    }
    println!("{}", "═".repeat(72));

    // ── Hard asserts ───────────────────────────────────────────────────────
    assert!(
        slope >= SLOPE_GATE,
        "G_TT_BAND_CONVERGE FAIL: convergence slope {slope:.4} < {SLOPE_GATE}. \
         Expected O(τ²) = slope ≥ 1.8 for cubic-Lagrange band-split shift."
    );
    assert!(
        band_finest < BAND_FINEST_GATE,
        "G_TT_BAND_CONVERGE FAIL: finest-grid band-split error {band_finest:.4e} \
         ≥ {BAND_FINEST_GATE}. Band-split TtChernoff not converging to analytic truth."
    );
    assert!(
        int_finest >= INT_PLATEAU_GATE,
        "G_TT_BAND_CONVERGE FAIL: integer-rounded error {int_finest:.4e} < {INT_PLATEAU_GATE} \
         at finest grid. Expected plateau ≥ 0.10 at h/dx≈{H_OVER_DX_TARGET} \
         (quantization floor, documented in §6.2 of the TRIZ spec)."
    );
}
