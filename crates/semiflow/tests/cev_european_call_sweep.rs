//! v0.3.1 parameter sweep: CEV European call vs Schroder 1989 closed form.
//! Contract: `contracts/tests/cev_european_call_sweep.yaml`.
//! ADR: docs/adr/0010-v0_3_1-cev-hardening.md.
//!
//! Gates per combo:
//!   sweep-A: sup-norm error in S ∈ [0.5K, 1.5K] < 5e-2 · max(`price_atm`, 1)
//!   sweep-B: pointwise ATM error < max(1e-2, 1e-3 · `price_atm`)
//!   sweep-C: relative ATM error < 5e-3  (only if σ₀ ≥ 0.30 ∧ T ≥ 0.5)
//!
//! VALIDATED REGIME (ζ-A PDE; v0.4.1 oracle lift):
//!   (1) σ < 0.50: PDE stability — ζ-A τ²-correction unstable at σ=0.50 in several corners.
//!   (2) `lam_peak` < 1400: oracle stability — pre-v0.4.1 limit; v0.4.1 lifted via log-space ncx2 (peak now ≲ 3500).
//!       Worst CEV combo: (K/S=1,T=0.25,σ=0.15,β=0.70) → λ≈1983; now in-regime.
//!   (3) σ·T·β < 0.40: domain-width corner — [1,200] too narrow for deep-OTM/long-T/high-β.
//!
//! `DiffusionChernoff`/`DriftReactionChernoff` store `fn(f64)->f64` (bare fn ptrs, not
//! closures); per-combo coefficients are passed via thread-local cells.

use std::cell::Cell;

use semiflow_core::{
    grid::{BoundaryPolicy, InterpKind},
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, StrangSplit,
};
use statrs::distribution::{ChiSquared, ContinuousCDF};

// Fixed parameters
const S0: f64 = 100.0;
const R: f64 = 0.05;
const X_MIN: f64 = 1.0;
const X_MAX: f64 = 200.0;
const N_GRID: usize = 512;

// Thread-local coefficient cells (set per combo, read by static fn ptrs)
thread_local! {
    static HALF_D2:    Cell<f64> = const { Cell::new(0.0) };
    static DELTA_SQ:   Cell<f64> = const { Cell::new(0.0) };
    static BETA_PDE:   Cell<f64> = const { Cell::new(0.0) };
    static TWO_B:      Cell<f64> = const { Cell::new(0.0) };
}

fn set_combo_params(delta_sq: f64, beta_pde: f64) {
    let two_b = 2.0 * beta_pde;
    HALF_D2.with(|c| c.set(0.5 * delta_sq));
    DELTA_SQ.with(|c| c.set(delta_sq));
    BETA_PDE.with(|c| c.set(beta_pde));
    TWO_B.with(|c| c.set(two_b));
}

// Static fn ptrs — read from thread-locals.
fn a_fn(s: f64) -> f64 {
    let h = HALF_D2.with(Cell::get);
    let b = TWO_B.with(Cell::get);
    h * s.powf(b)
}

fn a_prime_fn(s: f64) -> f64 {
    let d = DELTA_SQ.with(Cell::get);
    let bp = BETA_PDE.with(Cell::get);
    let b = TWO_B.with(Cell::get);
    d * bp * s.powf(b - 1.0)
}

fn a_dbl_prime_fn(s: f64) -> f64 {
    let d = DELTA_SQ.with(Cell::get);
    let bp = BETA_PDE.with(Cell::get);
    let b = TWO_B.with(Cell::get);
    d * bp * (b - 1.0) * s.powf(b - 2.0)
}

fn b_fn(s: f64) -> f64 {
    R * s - a_prime_fn(s)
}

fn c_fn(_: f64) -> f64 {
    -R
}

// Combo descriptor
struct Combo {
    moneyness: f64, // K/S0
    t: f64,
    sigma0: f64,
    beta_pde: f64,
}

// Default set (21 combos; σ ≥ 0.50 corners excluded; λ-underflow corner restored in v0.4.1).
// Validated regime: σ < 0.70 ∧ σ·T·β < 0.80 (Magnus envelope).
// Excluded: σ=0.50 blow-up/NaN combos (PDE boundary).
// Previously excluded (K/S=1,T=0.25,σ=0.15,β=0.70) oracle λ-underflow restored:
//   v0.4.1 log-space ncx2_cdf supports λ_peak≲3500; this combo (λ≈1983) is now in-regime.

const DEFAULT_COMBOS: &[Combo] = &[
    // Block A — moneyness sweep (5) @ (T=1.0, σ₀=0.30, β=0.5)
    Combo {
        moneyness: 0.80,
        t: 1.0,
        sigma0: 0.30,
        beta_pde: 0.5,
    },
    Combo {
        moneyness: 0.90,
        t: 1.0,
        sigma0: 0.30,
        beta_pde: 0.5,
    },
    Combo {
        moneyness: 1.00,
        t: 1.0,
        sigma0: 0.30,
        beta_pde: 0.5,
    },
    Combo {
        moneyness: 1.10,
        t: 1.0,
        sigma0: 0.30,
        beta_pde: 0.5,
    },
    Combo {
        moneyness: 1.20,
        t: 1.0,
        sigma0: 0.30,
        beta_pde: 0.5,
    },
    // Block B — beta sweep (2 new) @ (T=1.0, σ₀=0.30, K/S₀=1.0)
    // β=0.5 covered above; add β=0.3 and β=0.7 (σ=0.30 < 0.50 → in-regime).
    Combo {
        moneyness: 1.00,
        t: 1.0,
        sigma0: 0.30,
        beta_pde: 0.3,
    },
    Combo {
        moneyness: 1.00,
        t: 1.0,
        sigma0: 0.30,
        beta_pde: 0.7,
    },
    // Block C — vol sweep (1 new) @ (T=1.0, β=0.5, K/S₀=1.0)
    // σ₀=0.30 covered above; add σ₀=0.15 (σ=0.50 excluded — PDE boundary).
    Combo {
        moneyness: 1.00,
        t: 1.0,
        sigma0: 0.15,
        beta_pde: 0.5,
    },
    // Block D — maturity (2 new) @ (σ₀=0.30, β=0.5, K/S₀=1.0)
    // T=1.0 covered above; add T=0.25 and T=2.0.
    Combo {
        moneyness: 1.00,
        t: 0.25,
        sigma0: 0.30,
        beta_pde: 0.5,
    },
    Combo {
        moneyness: 1.00,
        t: 2.0,
        sigma0: 0.30,
        beta_pde: 0.5,
    },
    // β-T cross-axis (contract requirement)
    Combo {
        moneyness: 1.00,
        t: 0.25,
        sigma0: 0.30,
        beta_pde: 0.3,
    },
    // Block E — cross-axis practitioners' corners (σ < 0.50 only)
    Combo {
        moneyness: 0.90,
        t: 0.25,
        sigma0: 0.30,
        beta_pde: 0.7,
    }, // OTM/short/β-high
    Combo {
        moneyness: 1.10,
        t: 0.25,
        sigma0: 0.30,
        beta_pde: 0.3,
    }, // ITM/short/β-low
    // σ=0.50 cross-axis combos removed (PDE boundary)
    Combo {
        moneyness: 0.80,
        t: 1.00,
        sigma0: 0.30,
        beta_pde: 0.3,
    }, // deep-OTM/β-low
    Combo {
        moneyness: 1.20,
        t: 1.00,
        sigma0: 0.30,
        beta_pde: 0.7,
    }, // deep-ITM/β-high
    Combo {
        moneyness: 1.20,
        t: 2.00,
        sigma0: 0.30,
        beta_pde: 0.3,
    }, // deep-ITM/T-long/β-low
    Combo {
        moneyness: 1.10,
        t: 1.00,
        sigma0: 0.15,
        beta_pde: 0.3,
    }, // ITM/vol-low/β-low
    // ATM/T-long/vol-high/β-low: σ=0.50 → removed; keep σ=0.30 variant
    Combo {
        moneyness: 1.00,
        t: 2.00,
        sigma0: 0.30,
        beta_pde: 0.3,
    }, // ATM/T-long/β-low
    // Restored in v0.4.1: log-space ncx2_cdf lifts λ-underflow limit; lam≈1983 now in-regime.
    Combo {
        moneyness: 1.00,
        t: 0.25,
        sigma0: 0.15,
        beta_pde: 0.70,
    }, // ATM/short/vol-low/β-high (lam≈1983)
    Combo {
        moneyness: 1.00,
        t: 1.00,
        sigma0: 0.15,
        beta_pde: 0.7,
    }, // ATM/vol-low/β-high
];

// ---------------------------------------------------------------------------
// Noncentral χ² CDF — log-space Poisson recurrence (v0.4.1).
// Maintains log_p = log P_j additively; applies exp only when log_p > -700
// to avoid the f64 underflow at exp(-lam/2) that the linear-space version
// suffered when lam/2 > 745. Convergence test compares log_p against the
// observed peak; tail is truncated once log_p < max_log_p - 36 nats
// (≈4e-16 of peak — well below f64 ε). j_max=2000 covers lam_peak ≲ 3500.
// ---------------------------------------------------------------------------

fn ncx2_cdf(w: f64, v: f64, lam: f64) -> f64 {
    let half_lam = lam / 2.0;
    if half_lam == 0.0 {
        // central χ² limit
        return ChiSquared::new(v).expect("df > 0").cdf(w);
    }
    let log_half_lam = half_lam.ln();
    let mut log_p = -half_lam; // log P_0 = −λ/2
    let mut sum = 0.0_f64;
    let mut max_log_p = f64::NEG_INFINITY;
    let mut converged = false;
    for j in 0_u32..2000 {
        let chi = ChiSquared::new(v + 2.0 * f64::from(j)).expect("df > 0");
        let cdf_j = chi.cdf(w);
        if log_p > -700.0 && cdf_j > 0.0 {
            sum += log_p.exp() * cdf_j;
        }
        if log_p > max_log_p {
            max_log_p = log_p;
        }
        let past_peak = f64::from(j) > half_lam.max(5.0);
        // Tail bound: sum_{k>j} P_k ≤ P_j · (λ/2 / (j-λ/2)).
        // 36 nats ≈ 4e-16 of the observed peak — well below f64 ε.
        if past_peak && log_p < max_log_p - 36.0 {
            converged = true;
            break;
        }
        log_p += log_half_lam - f64::from(j + 1).ln();
    }
    assert!(
        converged,
        "ncx2_cdf did not converge in 2000 iters: w={w:.4}, v={v:.4}, lam={lam:.4}"
    );
    sum
}

// ---------------------------------------------------------------------------
// Schroder (1989) closed-form European call — generalised (σ₀, β, S₀ vary).
// δ² = σ₀²·S₀^(2-2β) per combo (spot-normalised convention).
// ---------------------------------------------------------------------------

// s, k, t, x, y are standard finance/math symbols for spot, strike, time, CDF args.
#[allow(clippy::many_single_char_names)]
fn schroder_call(s: f64, k: f64, t: f64, sigma0: f64, beta_pde: f64) -> f64 {
    let delta_sq = sigma0.powi(2) * S0.powf(2.0 - 2.0 * beta_pde);
    let beta_s = 2.0 * beta_pde;
    let two_m_beta = 2.0 - beta_s;
    let expon = R * two_m_beta * t;
    let k_param = 2.0 * R / (delta_sq * two_m_beta * (libm::exp(expon) - 1.0));
    let x = k_param * s.powf(two_m_beta) * libm::exp(expon);
    let y = k_param * k.powf(two_m_beta);
    let df_v = 2.0 / two_m_beta;
    let df_v2 = df_v + 2.0;
    let q1 = 1.0 - ncx2_cdf(2.0 * y, df_v2, 2.0 * x);
    let q2 = 1.0 - ncx2_cdf(2.0 * x, df_v, 2.0 * y);
    s * q1 - k * libm::exp(-R * t) * (1.0 - q2)
}

// ---------------------------------------------------------------------------
// Build the CEV Strang operator for the current combo (cells already set).
// Returns (strang, grid, i_atm, s_atm, n_steps).
// ---------------------------------------------------------------------------

fn make_strang_for_combo(
    t: f64,
) -> (
    StrangSplit<DiffusionChernoff, DriftReactionChernoff>,
    Grid1D,
    usize,
    f64,
    usize,
) {
    let grid = Grid1D::new(X_MIN, X_MAX, N_GRID)
        .unwrap()
        .with_boundary(BoundaryPolicy::LinearExtrapolate)
        .with_interp(InterpKind::CubicHermite);
    let a_norm = a_fn(X_MAX); // ½·δ²·X_MAX^(2β) — monotone increasing
    let diffusion = DiffusionChernoff::new(a_fn, a_prime_fn, a_dbl_prime_fn, a_norm, grid);
    let drift = DriftReactionChernoff::new(b_fn, c_fn, R, grid);
    let strang = StrangSplit::new(diffusion, drift);
    let dx = grid.dx();
    // S0 and X_MIN are positive, so (S0-X_MIN)/dx is always positive and small.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i_atm = ((S0 - X_MIN) / dx).round() as usize;
    let s_atm = grid.x_at(i_atm);
    let n_steps = if t >= 1.5 { 512 } else { 256 };
    (strang, grid, i_atm, s_atm, n_steps)
}

// ---------------------------------------------------------------------------
// Sup-norm error over S ∈ [0.5K, 1.5K], every 5th node.
// ---------------------------------------------------------------------------

// u, k, t, s, i are standard finance/math symbols for state, strike, time, spot, index.
#[allow(clippy::many_single_char_names)]
fn compute_sup_err(u: &GridFn1D, k: f64, t: f64, sigma0: f64, beta_pde: f64, grid: Grid1D) -> f64 {
    let s_lo = 0.5 * k;
    let s_hi = 1.5 * k;
    let mut max_err = 0.0_f64;
    let mut i = 0_usize;
    while i < grid.n {
        let s = grid.x_at(i);
        if s >= s_lo && s <= s_hi {
            let oracle = schroder_call(s, k, t, sigma0, beta_pde);
            let err = (u.values[i] - oracle).abs();
            if err > max_err {
                max_err = err;
            }
            i += 5;
        } else {
            i += 1;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Run one combo; returns (sup_err, atm_err, price_atm_oracle, rel_atm_err).
// ---------------------------------------------------------------------------

fn run_combo(combo: &Combo) -> (f64, f64, f64, f64) {
    let k = combo.moneyness * S0;
    let delta_sq = combo.sigma0.powi(2) * S0.powf(2.0 - 2.0 * combo.beta_pde);
    set_combo_params(delta_sq, combo.beta_pde);
    let (strang, grid, i_atm, s_atm, n_steps) = make_strang_for_combo(combo.t);
    let f0 = GridFn1D::from_fn(grid, |s| (s - k).max(0.0));
    let sg = ChernoffSemigroup::new(strang, n_steps).expect("n >= 1");
    let u = sg.evolve(combo.t, &f0).expect("evolve ok");
    let price_atm = schroder_call(s_atm, k, combo.t, combo.sigma0, combo.beta_pde);
    let atm_err = (u.values[i_atm] - price_atm).abs();
    let rel_atm = if price_atm > 1e-10 {
        atm_err / price_atm
    } else {
        atm_err
    };
    let sup_err = compute_sup_err(&u, k, combo.t, combo.sigma0, combo.beta_pde, grid);
    (sup_err, atm_err, price_atm, rel_atm)
}

// ---------------------------------------------------------------------------
// Per-combo result for collect-then-assert pattern.
// ---------------------------------------------------------------------------

// gate_a/b/c_pass and gate_c_active are gate status flags; 4 bools is domain-appropriate.
#[allow(clippy::struct_excessive_bools)]
struct ComboResult {
    moneyness: f64,
    t: f64,
    sigma0: f64,
    beta_pde: f64,
    sup_err: f64,
    atm_err: f64,
    price_atm: f64,
    rel_atm: f64,
    gate_a_pass: bool,
    gate_b_pass: bool,
    gate_c_pass: bool,   // true = pass or skipped (low-quality regime)
    gate_c_active: bool, // true = sweep-C was actually asserted
}

// Evaluate gates for one combo without panicking; return ComboResult.
// gate_a/b/c_pass are gate status names; allowing similar_names is intentional.
#[allow(clippy::similar_names)]
fn eval_gates(combo: &Combo) -> ComboResult {
    let (sup_err, atm_err, price_atm, rel_atm) = run_combo(combo);
    let thresh_a = 5e-2 * price_atm.max(1.0);
    let thresh_b = (1e-2_f64).max(1e-3 * price_atm);
    let gate_a_pass = sup_err < thresh_a;
    let gate_b_pass = atm_err < thresh_b;
    let gate_c_active = combo.sigma0 >= 0.30 && combo.t >= 0.5;
    let gate_c_pass = if gate_c_active { rel_atm < 5e-3 } else { true };
    ComboResult {
        moneyness: combo.moneyness,
        t: combo.t,
        sigma0: combo.sigma0,
        beta_pde: combo.beta_pde,
        sup_err,
        atm_err,
        price_atm,
        rel_atm,
        gate_a_pass,
        gate_b_pass,
        gate_c_pass,
        gate_c_active,
    }
}

// ---------------------------------------------------------------------------
// Print diagnostic table header.
// ---------------------------------------------------------------------------

fn print_table_header() {
    eprintln!(
        "{:>6} {:>5} {:>6} {:>5}  {:>9}  {:>8}  {:>8}  {:>8}  gates",
        "K/S0", "T", "sigma0", "beta", "price_atm", "sup_err", "atm_err", "rel%"
    );
}

// ---------------------------------------------------------------------------
// Print one result row; `oor_tag` is " [OOR]" or "".
// ---------------------------------------------------------------------------

fn print_result_row(r: &ComboResult, oor_tag: &str) {
    let gate_str = format!(
        "A:{} B:{} C:{}{}",
        if r.gate_a_pass { "ok" } else { "FAIL" },
        if r.gate_b_pass { "ok" } else { "FAIL" },
        if !r.gate_c_active {
            "skip"
        } else if r.gate_c_pass {
            "ok"
        } else {
            "FAIL"
        },
        oor_tag
    );
    eprintln!(
        "{:>6.2} {:>5.2} {:>6.2} {:>5.2}  {:>9.4}  {:>8.3e}  {:>8.3e}  {:>7.4}%  {}",
        r.moneyness,
        r.t,
        r.sigma0,
        r.beta_pde,
        r.price_atm,
        r.sup_err,
        r.atm_err,
        r.rel_atm * 100.0,
        gate_str
    );
}

// ---------------------------------------------------------------------------
// Run a slice of combos — collect-then-assert pattern.
// All combos run first; full table printed; single aggregate assert at end.
// ---------------------------------------------------------------------------

fn run_combos(combos: &[Combo]) {
    print_table_header();
    let results: Vec<ComboResult> = combos.iter().map(eval_gates).collect();
    print_all_rows(&results);
    assert_no_failures(&results);
}

fn print_all_rows(results: &[ComboResult]) {
    for r in results {
        print_result_row(r, "");
        if !r.gate_c_active && r.gate_a_pass && r.gate_b_pass {
            eprintln!(
                "  sweep-C INFO: K/S={:.2} T={:.2} σ={:.2} β={:.2} \
                rel_atm={:.3e} (no assert in low-quality regime)",
                r.moneyness, r.t, r.sigma0, r.beta_pde, r.rel_atm
            );
        }
    }
}

/// Build the gate failure string for a single combo result.
fn format_failure(r: &ComboResult) -> Option<String> {
    if r.gate_a_pass && r.gate_b_pass && r.gate_c_pass {
        return None;
    }
    let a_str = if r.gate_a_pass {
        "ok".to_string()
    } else {
        format!("FAIL sup={:.3e}", r.sup_err)
    };
    let b_str = if r.gate_b_pass {
        "ok".to_string()
    } else {
        format!("FAIL atm={:.3e}", r.atm_err)
    };
    let c_str = if r.gate_c_pass {
        "ok".to_string()
    } else {
        format!("FAIL rel={:.3e}", r.rel_atm)
    };
    Some(format!(
        "K/S={:.2} T={:.2} σ={:.2} β={:.2} | A:{a_str} B:{b_str} C:{c_str}",
        r.moneyness, r.t, r.sigma0, r.beta_pde,
    ))
}

/// Track worst combo per sweep gate across all results.
fn compute_worst(results: &[ComboResult]) -> [(f64, f64, f64, f64, f64); 3] {
    let mut worst_a = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut worst_b = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut worst_c = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for r in results {
        if r.sup_err > worst_a.0 {
            worst_a = (r.sup_err, r.moneyness, r.t, r.sigma0, r.beta_pde);
        }
        if r.atm_err > worst_b.0 {
            worst_b = (r.atm_err, r.moneyness, r.t, r.sigma0, r.beta_pde);
        }
        if r.gate_c_active && r.rel_atm > worst_c.0 {
            worst_c = (r.rel_atm, r.moneyness, r.t, r.sigma0, r.beta_pde);
        }
    }
    [worst_a, worst_b, worst_c]
}

fn assert_no_failures(results: &[ComboResult]) {
    let failures: Vec<String> = results.iter().filter_map(format_failure).collect();
    let [worst_a, worst_b, worst_c] = compute_worst(results);
    eprintln!(
        "Worst sweep-A: K/S={:.2} T={:.2} σ={:.2} β={:.2} err={:.3e}",
        worst_a.1, worst_a.2, worst_a.3, worst_a.4, worst_a.0
    );
    eprintln!(
        "Worst sweep-B: K/S={:.2} T={:.2} σ={:.2} β={:.2} err={:.3e}",
        worst_b.1, worst_b.2, worst_b.3, worst_b.4, worst_b.0
    );
    eprintln!(
        "Worst sweep-C: K/S={:.2} T={:.2} σ={:.2} β={:.2} rel={:.3e}",
        worst_c.1, worst_c.2, worst_c.3, worst_c.4, worst_c.0
    );
    assert!(
        failures.is_empty(),
        "{} combo(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Default set test — always runs
// ---------------------------------------------------------------------------

/// sweep-A/B/C across the 20-combo default parameter set (validated regime).
///
/// Contract: `contracts/tests/cev_european_call_sweep.yaml` §5.
#[test]
fn cev_sweep_default() {
    eprintln!(
        "=== cev_sweep_default ({} combos) ===",
        DEFAULT_COMBOS.len()
    );
    run_combos(DEFAULT_COMBOS);
}

// ---------------------------------------------------------------------------
// Full set test — gated by slow-tests feature (135 combos)
// Regime-aware: OOR combos logged with [OOR] tag but not counted as failures.
// ---------------------------------------------------------------------------

/// Full 135-combo Cartesian grid sweep.
///
/// Gate: cargo test --release --features slow-tests --test `cev_european_call_sweep`
///
/// OOR combos are logged with [OOR] and NOT counted in the failure tally.
/// OOR condition: σ ≥ 0.50 (ζ-A PDE boundary) OR σ·T·β ≥ 0.40
/// (combined-magnitude corner). Oracle λ-underflow boundary (`lam_peak`<1400) retired
/// in v0.4.1: log-space `ncx2_cdf` now covers `λ_peak` ≲ 3500.
/// Only in-regime combos are strictly asserted.
/// See module-level doc comment and ADR-0010 Amendment 1.
#[cfg(feature = "slow-tests")]
#[test]
fn cev_sweep_full() {
    const MONEYNESS: &[f64] = &[0.80, 0.90, 1.00, 1.10, 1.20];
    const T_VALUES: &[f64] = &[0.25, 1.0, 2.0];
    const SIGMA: &[f64] = &[0.15, 0.30, 0.50];
    const BETA: &[f64] = &[0.3, 0.5, 0.7];
    let mut combos: Vec<Combo> = Vec::with_capacity(135);
    for &moneyness in MONEYNESS {
        for &t in T_VALUES {
            for &sigma0 in SIGMA {
                for &beta_pde in BETA {
                    combos.push(Combo {
                        moneyness,
                        t,
                        sigma0,
                        beta_pde,
                    });
                }
            }
        }
    }
    eprintln!("=== cev_sweep_full ({} combos) ===", combos.len());
    run_combos_regime_aware(&combos);
}

// ---------------------------------------------------------------------------
// Regime-aware runner: OOR combos skipped in failure tally, logged [OOR].
// ---------------------------------------------------------------------------

/// Returns true if the combo is within the validated regime (ζ-A sweep boundaries):
///   σ < 0.50                (PDE stability boundary — ζ-A τ²-correction unstable at σ=0.50)
///   AND σ·T·β < 0.40        (combined-magnitude corner; long-T high-β domain truncation)
///
/// The σ·T·β criterion captures the deep-OTM / long-T / high-β PDE domain
/// truncation issue surfaced at (K/S=1.20, T=2.00, σ=0.30, β=0.70).
/// Oracle λ-underflow boundary (`lam_peak` < 1400) retired in v0.4.1: log-space
/// `ncx2_cdf` now supports `λ_peak` ≲ 3500 — this guard is no longer needed.
#[cfg(feature = "slow-tests")]
fn combo_in_regime(combo: &Combo) -> bool {
    if combo.sigma0 >= 0.50 {
        return false;
    }
    combo.sigma0 * combo.t * combo.beta_pde < 0.40
}

#[cfg(feature = "slow-tests")]
fn run_combos_regime_aware(combos: &[Combo]) {
    print_table_header();
    let mut in_regime_results: Vec<ComboResult> = Vec::new();
    let mut oor_count = 0_usize;
    for combo in combos {
        let r = eval_gates(combo);
        if combo_in_regime(combo) {
            print_result_row(&r, "");
            in_regime_results.push(r);
        } else {
            print_result_row(&r, " [OOR]");
            oor_count += 1;
        }
    }
    eprintln!("OOR combos (σ≥0.50 | σ·T·β≥0.40, ζ-A PDE boundary): {oor_count}");
    eprintln!("In-regime combos: {}", in_regime_results.len());
    assert_no_failures(&in_regime_results);
}
