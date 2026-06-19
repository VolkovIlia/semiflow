//! G3⁶-2D — FLAGSHIP 2D 6th-order spatial convergence gate (v0.8.0, ADR-0020).
//!
//! PDE: `∂_t u = ½(∂_xx + ∂_yy)u`, `u_0(x,y) = exp(-(x²+y²))`.
//!
//! # Closed-form oracle
//! 2D heat kernel (math.md §10.5(a), eq. 10.7):
//! ```text
//! u(t, x, y) = (1+2t)^{-1} · exp(-(x²+y²)/(1+2t))
//! ```
//! At `t = 0.5`: `u = ½ · exp(-(x²+y²)/2)`.
//!
//! # Operator
//! `Strang2D<Diffusion6thChernoff(0.5), Diffusion6thChernoff(0.5)>` with
//! `InterpKind::SepticHermite` on both axes (v6.0+ default; `QuinticHermite` removed v7.0).
//! Theorem 7 separable-commutator identity (`[L_x ⊗ I, I ⊗ L_y] = 0`) makes
//! palindromic Strang exact at the BCH level; per-axis 6th-order spatial floor
//! (K7 + 9-pt Fornberg + `SepticHermite`) lifts directly to 2D.
//!
//! # Gate (G3⁶-2D — FLAGSHIP, `RELEASE_BLOCKING`, schema 0.7.6; recalibrated ADR-0163)
//! `N_CHERNOFF=1000`, T=0.5, N ∈ {191, 251, 331} (3-point prime-N, floor-safe asymptotic
//! band for the `SepticHermite` era), domain [-15,15]².
//! OLS log-log slope of ‖err‖_∞ vs N ∈ [−6.30, −5.85] (empirical OLS ≈ -6.07).
//! Wallclock ≤ 600 s under `RUSTFLAGS="-C target-cpu=native"
//! --features parallel,simd,slow-tests --release`.
//! The legacy basket {503,997,1999} (ADR-0020 Amendment 3) saturated the new
//! `SepticHermite` floor (φ≈1.5e-12) at N≥997 and was recalibrated under ADR-0163,
//! mirroring the 1D recalibration ADR-0120. NO method/sampler change.
//!
//! Reference: `contracts/semiflow-core.math.md §10.3 Theorem 7`, §9.2.6,
//! `contracts/semiflow-core.properties.yaml` gate `G3_6_2D`,
//! `docs/adr/0020-g3-6th-2d-flagship.md`.

#![cfg(feature = "slow-tests")]
// v7.0: QuinticHermite removed (ADR-0109 removal clock fulfilled); using SepticHermite default.

use semiflow_core::{
    ChernoffSemigroup, Diffusion6thChernoff, Grid1D, Grid2D, GridFn2D, InterpKind, Strang2D,
};

// ---------------------------------------------------------------------------
// Mandatory constants (DO NOT MODIFY — source of truth: properties.yaml::G3_6_2D)
// ---------------------------------------------------------------------------

const T_FINAL: f64 = 0.5;
const A_CONST: f64 = 0.5;
const X_MIN: f64 = -15.0;
const X_MAX: f64 = 15.0;
const N_CHERNOFF: usize = 1000; // temporal steps; τ = 5e-4
// v9.1+ (ADR-0163): floor-safe asymptotic prime basket for the SepticHermite era.
// Empirical regime map (g3_6_2d_regime_map_diagnostic): {191,251,331} sit in the clean
// 6th-order truncation band (seg slopes -5.86/-5.98/-6.18 straddle -6.0; finest err
// 2.25e-8 ≈ 5000× the ~5e-12 SepticHermite floor). Coarser N=127 is pre-asymptotic
// (-5.86); finer N≥419 steepens pre-floor (-6.48,…); N≥997 floored. OLS over basket -6.07.
// Supersedes ADR-0020 Amendment 3's {503,997,1999} (QuinticHermite-floor-era). See ADR-0163.
const N_SWEEP: [usize; 3] = [191, 251, 331];
const SLOPE_LO: f64 = -6.30; // upper-bound on |slope| (pre-floor super-convergence catch)
const SLOPE_HI: f64 = -5.85; // lower-bound on |slope| (order-degradation catch)
const RUNTIME_BUDGET_SEC: u64 = 600; // basket N≤331 ≪ old N≤1999; wallclock ≪ 1 min

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// 2D heat-kernel oracle: `(1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`.
///
/// Normative formula from math.md §10.5(a) eq. (10.7).
/// Initial datum: `u_0(x,y) = exp(-(x²+y²))`.
/// PDE: `∂_t u = ½(∂_xx + ∂_yy)u`.
#[inline]
fn oracle_heat_2d(t: f64, x: f64, y: f64) -> f64 {
    let denom = 1.0 + 2.0 * t;
    (1.0 / denom) * (-(x * x + y * y) / denom).exp()
}

// ---------------------------------------------------------------------------
// OLS slope helper (mirror of convergence_rate_6th.rs::log_log_slope)
// ---------------------------------------------------------------------------

/// OLS slope: `log(err) ~ slope * log(N) + const`.
///
/// Sign convention: slope is NEGATIVE for a converging method (err drops as N grows).
/// Asymptote for 6th-order: ≈ -5.95.
// m is a slice length ≤ 10, well within f64 mantissa.
// sum_y and sum_xy are standard OLS notation; allowing similar_names is intentional.
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn log_log_slope(ns: &[f64], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let lx: Vec<f64> = ns.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = lx.iter().sum();
    let sum_y: f64 = ly.iter().sum();
    let sum_xx: f64 = lx.iter().map(|&v| v * v).sum();
    let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// Per-N error helper
// ---------------------------------------------------------------------------

/// Build `Strang2D<Diffusion6thChernoff>` at resolution `n`, evolve to `T_FINAL`,
/// return the sup-norm error vs the 2D heat-kernel oracle.
///
/// Uses `ChernoffSemigroup::evolve` — `Strang2D` dispatches to the parallel
/// kernel automatically when compiled with `--features parallel`.
// n ≤ 1999, N_CHERNOFF = 1000 — both well within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
fn run_one_grid(n: usize) -> f64 {
    let gx = Grid1D::new(X_MIN, X_MAX, n)
        .expect("grid x valid")
        .with_interp(InterpKind::SepticHermite);
    let gy = Grid1D::new(X_MIN, X_MAX, n)
        .expect("grid y valid")
        .with_interp(InterpKind::SepticHermite);
    let grid2d = Grid2D::new(gx, gy);
    let f0 = GridFn2D::from_fn(grid2d, |x, y| (-(x * x + y * y)).exp());
    let cx = Diffusion6thChernoff::new(|_| A_CONST, |_| 0.0_f64, |_| 0.0_f64, A_CONST, gx);
    let cy = Diffusion6thChernoff::new(|_| A_CONST, |_| 0.0_f64, |_| 0.0_f64, A_CONST, gy);
    let phi2d = Strang2D::new(cx, cy);
    let semi = ChernoffSemigroup::new(phi2d, N_CHERNOFF).expect("N_CHERNOFF >= 1");
    let u_n = semi.evolve(T_FINAL, &f0).expect("evolve succeeds");
    sup_norm_error_2d(&u_n.values, grid2d.nx(), grid2d.ny(), gx, gy)
}

/// Compute sup-norm error between a 2D state buffer and the heat oracle at `T_FINAL`.
fn sup_norm_error_2d(vals: &[f64], nx: usize, ny: usize, gx: Grid1D, gy: Grid1D) -> f64 {
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let exact = oracle_heat_2d(T_FINAL, xi, yj);
            let err = (vals[j * nx + i] - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Sweep loop helper
// ---------------------------------------------------------------------------

/// Run the full `N_SWEEP`, printing per-N diagnostics.
/// Returns `(ns_f, errs)` for OLS slope computation.
///
/// CFL guard is asserted inside the loop (per-N tripwire).
// n ≤ 1999 — within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
fn run_sweep() -> (Vec<f64>, Vec<f64>) {
    let tau = T_FINAL / N_CHERNOFF as f64;
    let j_shift = 2.0 * (5.0 * A_CONST * tau).sqrt();
    let half_domain = 0.5 * (X_MAX - X_MIN);

    eprintln!(
        "G3⁶-2D spatial sweep: a={A_CONST}, T={T_FINAL}, \
         N_CHERNOFF={N_CHERNOFF}, tau={tau:.4e}"
    );
    eprintln!("Domain [{X_MIN}, {X_MAX}]²; InterpKind::SepticHermite per axis (v7.0+)");
    eprintln!(
        "{:>6}  {:>10}  {:>14}  {:>10}  {:>8}",
        "N", "dx", "err_sup", "ratio", "time(s)"
    );

    let mut ns_f: Vec<f64> = Vec::new();
    let mut errs: Vec<f64> = Vec::new();
    let mut prev_err: Option<f64> = None;

    for &n in &N_SWEEP {
        let dx = (X_MAX - X_MIN) / n as f64;
        let stencil_reach = j_shift + 3.0 * dx;
        assert!(
            stencil_reach < half_domain,
            "CFL guard FAIL at N={n}: J + 3·dx = {stencil_reach:.6} >= half-domain {half_domain:.1}",
        );
        let t_n = std::time::Instant::now();
        let err = run_one_grid(n);
        let elapsed_s = t_n.elapsed().as_secs();
        let ratio_str =
            prev_err.map_or_else(|| "         -".into(), |p| format!("{:>10.2}", p / err));
        eprintln!("{n:>6}  {dx:>10.4e}  {err:>14.4e}  {ratio_str}  ({elapsed_s:>6}s)");
        ns_f.push(n as f64);
        errs.push(err);
        prev_err = Some(err);
    }
    (ns_f, errs)
}

// ---------------------------------------------------------------------------
// G3⁶-2D FLAGSHIP GATE
// ---------------------------------------------------------------------------

/// Print the slope-out-of-window diagnostic (ADR-0163 interpretation hint).
fn print_slope_failure_diag(slope: f64, errs: &[f64]) {
    eprintln!("DIAGNOSTIC: G3⁶-2D slope {slope:.4} outside [{SLOPE_LO}, {SLOPE_HI}].");
    eprintln!("Per-N error table (re-printed for failure analysis):");
    for (n, e) in N_SWEEP.iter().zip(errs.iter()) {
        eprintln!("  N={n:>4}  err_sup={e:.4e}");
    }
    eprintln!(
        "Expected OLS ≈ -6.07 over {{191,251,331}} (in-band seg -5.98/-6.18); \
         if slope > -5.85 the order degraded; if slope < -6.30 the SepticHermite \
         floor returned (re-run g3_6_2d_regime_map_diagnostic). \
         See properties.yaml::G3_6_2D failure_mode and ADR-0163."
    );
}

/// Print the wallclock-over-budget diagnostic (parallel/SIMD engagement hints).
fn print_runtime_failure_diag(total_secs: u64) {
    eprintln!(
        "DIAGNOSTIC: G3⁶-2D wallclock {total_secs} s > {RUNTIME_BUDGET_SEC} s under \
         parallel,simd,slow-tests. Likely cause: Block B parallel kernel or Block C \
         SIMD kernel not engaged. Re-run STRANG2D_PARALLEL_BIT_EQUAL and \
         SIMD_BIT_EQUAL_PARALLEL first; if those pass, verify SIMD_COMBINED_SPEEDUP \
         >= 5x at N=1600² baseline. Basket {N_SWEEP:?} should run in well under 1 min."
    );
}

/// G3⁶-2D — Empirical 6th-order 2D spatial slope gate (`RELEASE_BLOCKING`, ADR-0020,
/// recalibrated ADR-0163).
///
/// Validates BOTH:
/// 1. OLS log-log slope of ‖err‖_∞ vs N lies in [-6.30, -5.85].
/// 2. Total wallclock ≤ 600 s under `RUSTFLAGS="-C target-cpu=native"
///    --features parallel,simd,slow-tests --release`.
///
/// Gate uses the floor-safe 3-point prime-N basket {191, 251, 331} (ADR-0163,
/// superseding ADR-0020 Amendment 3's {503,997,1999} which saturated the
/// SepticHermite floor at N≥997). The regime-map diagnostic
/// (`g3_6_2d_regime_map_diagnostic`) measured in-band segment slopes -5.98 / -6.18
/// (straddling -6.0) and basket OLS -6.07, with the finest point (err 2.25e-8) ≈
/// 5000× the ~5e-12 floor — fully floor-safe. Mirrors 1D recalibration ADR-0120.
///
/// Run via: `RUSTFLAGS="-C target-cpu=native" cargo test --release
///           --features parallel,simd,slow-tests --test convergence_rate_6th_2d`
#[test]
fn g3_6_2d_flagship_slope_and_runtime_gate() {
    let total_start = std::time::Instant::now();
    let (ns_f, errs) = run_sweep();
    let total_secs = total_start.elapsed().as_secs();
    let slope = log_log_slope(&ns_f, &errs);

    eprintln!(
        "\nG3⁶-2D FLAGSHIP: slope = {slope:.4} (window [{SLOPE_LO}, {SLOPE_HI}]); \
         wallclock = {total_secs} s (budget {RUNTIME_BUDGET_SEC} s)"
    );
    eprintln!(
        "Expected OLS ≈ -6.07 over floor-safe basket {{191,251,331}} \
         (O(dx⁶) K7+FD9+SepticHermite per axis, Theorem 7, ADR-0163)"
    );

    if !(SLOPE_LO..=SLOPE_HI).contains(&slope) {
        print_slope_failure_diag(slope, &errs);
    }
    if total_secs > RUNTIME_BUDGET_SEC {
        print_runtime_failure_diag(total_secs);
    }

    assert!(
        (SLOPE_LO..=SLOPE_HI).contains(&slope),
        "G3⁶-2D FAIL: slope={slope:.4} outside window [{SLOPE_LO}, {SLOPE_HI}]; \
         BLOCKS release. See ADR-0163 (basket {N_SWEEP:?}) and \
         properties.yaml::G3_6_2D failure_mode for diagnosis.",
    );
    assert!(
        total_secs <= RUNTIME_BUDGET_SEC,
        "G3⁶-2D FAIL: wallclock {total_secs} s > budget {RUNTIME_BUDGET_SEC} s under \
         RUSTFLAGS=-C target-cpu=native --features parallel,simd,slow-tests; \
         basket {N_SWEEP:?} should run ≪ 1 min — likely thermal throttling or \
         extra machine load. BLOCKS release. See ADR-0163.",
    );
}

// ---------------------------------------------------------------------------
// G3⁶-2D REGIME-MAP DIAGNOSTIC (non-asserting; #[ignore]) — ADR-0163
// ---------------------------------------------------------------------------
//
// PURPOSE: map the asymptotic-vs-f64-floor boundary for the SepticHermite (v6.0+,
// ADR-0109) 2D spatial convergence BEFORE recalibrating the FLAGSHIP grid basket.
//
// WHY: the headline basket {503, 997, 1999} was calibrated in ADR-0020 Amendment 3
// against the QuinticHermite interpolation floor (~1e-10). ADR-0109 replaced
// QuinticHermite with SepticHermite (O(dx⁸) virtual-node sampler), lowering the 1D
// floor to φ ≈ 1.5e-12 (gate G_SEPTIC_HERMITE_FLOOR: ≤5e-12 at N=512). Consequently
// N=997 (err 3.84e-12) and N=1999 (err 8.62e-13) now SATURATE at the new floor: the
// 997→1999 segment collapses to ≈ -2.15 and drags the OLS slope to -5.34, OUTSIDE
// the [-6.15, -5.85] window. The method is CORRECT (6th order genuinely holds in the
// truncation regime); only the grid basket is stale for the more-accurate interpolant.
// This mirrors the 1D recalibration already shipped under ADR-0120.
//
// TEMPORAL-FLOOR ISOLATION: N_CHERNOFF is held at the headline 1000 (τ = 5e-4,
// τ² ≈ 2.5e-7 temporal floor) — IDENTICAL to the FLAGSHIP gate — so the diagnostic
// measures the SAME error surface and any flattening is unambiguously the SPATIAL
// f64 floor, never a temporal artifact. Domain, a, T are also held identical: the
// regime map must reflect the real gate, so NO parameter is altered for speed.
//
// COST: 8 grids, the two largest (691, 997) near the current headline cost. Run ONCE
// to read the regime map, then set the recalibrated basket + window from the data.

/// Wider non-asserting N basket for the regime map. Coarse end probes the
/// pre-asymptotic onset; fine end (691, 997) probes the floor saturation.
const DIAG_N_SWEEP: [usize; 8] = [127, 191, 251, 331, 419, 503, 691, 997];

/// Run the diagnostic sweep, returning `(ns_f, errs)` and printing per-N rows.
///
/// Holds N_CHERNOFF/T/a/domain identical to the FLAGSHIP gate (see module note):
/// the regime map must describe the real error surface, not a cheapened proxy.
// n ≤ 997 — within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
fn run_diag_sweep() -> (Vec<f64>, Vec<f64>) {
    eprintln!(
        "G3⁶-2D REGIME MAP (ADR-0163): a={A_CONST}, T={T_FINAL}, \
         N_CHERNOFF={N_CHERNOFF} (τ=5e-4, IDENTICAL to FLAGSHIP gate)"
    );
    eprintln!("Domain [{X_MIN}, {X_MAX}]²; SepticHermite per axis (v6.0+, ADR-0109)");
    eprintln!("Floor reference: G_SEPTIC_HERMITE_FLOOR φ ≤ 5e-12 (1D); expect 2D saturation near it");
    eprintln!("{:>6}  {:>10}  {:>14}  {:>12}  {:>8}", "N", "dx", "err_sup", "seg_slope", "time(s)");

    let mut ns_f: Vec<f64> = Vec::new();
    let mut errs: Vec<f64> = Vec::new();
    let mut prev: Option<(f64, f64)> = None;

    for &n in &DIAG_N_SWEEP {
        let dx = (X_MAX - X_MIN) / n as f64;
        let t_n = std::time::Instant::now();
        let err = run_one_grid(n);
        let secs = t_n.elapsed().as_secs();
        let seg = prev.map_or_else(
            || "           -".into(),
            |(pn, pe): (f64, f64)| {
                format!("{:>12.4}", (err.ln() - pe.ln()) / ((n as f64).ln() - pn.ln()))
            },
        );
        eprintln!("{n:>6}  {dx:>10.4e}  {err:>14.4e}  {seg}  ({secs:>6}s)");
        prev = Some((n as f64, err));
        ns_f.push(n as f64);
        errs.push(err);
    }
    (ns_f, errs)
}

/// Print the full-basket OLS slope plus a floor-safe sub-basket slope candidate.
///
/// Floor-safe heuristic: keep grids whose err_sup ≥ 100 × the SepticHermite floor
/// (≥ 5e-10), which the dx⁶ model places at N ≲ 503; the printed sub-slope over the
/// floor-safe points is the predicted recalibrated-gate slope to seed the window.
fn print_diag_summary(ns_f: &[f64], errs: &[f64]) {
    const FLOOR_SAFE_MIN_ERR: f64 = 5e-10; // 100× the ~5e-12 SepticHermite floor
    let full = log_log_slope(ns_f, errs);
    let safe: Vec<(f64, f64)> = ns_f
        .iter()
        .zip(errs.iter())
        .filter(|(_, &e)| e >= FLOOR_SAFE_MIN_ERR)
        .map(|(&n, &e)| (n, e))
        .collect();
    eprintln!("\nFULL-basket OLS slope (incl. floored points) = {full:.4} (EXPECTED flat, ~-5.3)");
    if safe.len() >= 2 {
        let sn: Vec<f64> = safe.iter().map(|p| p.0).collect();
        let se: Vec<f64> = safe.iter().map(|p| p.1).collect();
        let safe_slope = log_log_slope(&sn, &se);
        let ns: Vec<usize> = sn.iter().map(|&v| v as usize).collect();
        eprintln!(
            "FLOOR-SAFE sub-basket {ns:?} (err ≥ {FLOOR_SAFE_MIN_ERR:.0e}) OLS slope = {safe_slope:.4}"
        );
        eprintln!("→ Predicted asymptotic order ≈ -6 (recalibrate FLAGSHIP window from this value).");
    } else {
        eprintln!("FLOOR-SAFE sub-basket too small — extend DIAG_N_SWEEP coarser (add N < 127).");
    }
    eprintln!("Read the seg_slope column: clean ≈ -6 rows are asymptotic; rows trending to 0 are floored.");
}

/// G3⁶-2D REGIME-MAP DIAGNOSTIC (NON-ASSERTING; ADR-0163).
///
/// Maps the asymptotic-vs-floor boundary for the SepticHermite 2D spatial sweep so
/// the FLAGSHIP basket {503, 997, 1999} can be honestly recalibrated to floor-safe
/// grids. Prints per-N err_sup, consecutive segment slopes, the full-basket OLS slope
/// (expected ~-5.3, demonstrating the floor flattening) and a floor-safe sub-basket
/// OLS slope (expected ≈ -6, the true asymptotic order). Asserts NOTHING.
///
/// Run ONCE via:
/// `RUSTFLAGS="-C target-cpu=native" cargo test -p semiflow-core \
///   --features parallel,simd,slow-tests --release \
///   --test convergence_rate_6th_2d -- --ignored \
///   g3_6_2d_regime_map_diagnostic --nocapture`
#[test]
#[ignore = "diagnostic regime map (slow, ~headline cost): run with --ignored --nocapture; ADR-0163"]
fn g3_6_2d_regime_map_diagnostic() {
    let (ns_f, errs) = run_diag_sweep();
    print_diag_summary(&ns_f, &errs);
}
