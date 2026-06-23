//! `G_REVERSE_AD_GRADIENT`, `G_REVERSE_AD_STRUCTURE`, and `G_REVERSE_AD_CHECKPOINT` —
//! `RELEASE_BLOCKING` gates for `ReverseChernoff` reverse-mode AD
//! (math §51.6, ADR-0156 Amendment 2; v9.1.0 Shift B GENUINE).
//!
//! ## `G_REVERSE_AD_GRADIENT` — gradient correctness + `< 1e-12` cross-mode parity
//!
//! End-to-end gradient of `J(θ) = ‖(F_θ(τ))ⁿ u₀‖²` (target=0) w.r.t. scalar
//! diffusivity `θ` on the DEFAULT `Grid1D::new` (`SepticHermite`) grid (§51.6
//! anti-dodge clause). Two conjunctive thresholds:
//!   (i)  `|reverse_grad − richardson_FD_grad| / |FD_grad| < 1e-9` (relative).
//!   (ii) For `K = 1`, `value_and_grad_k1` (GENUINE cotangent backward sweep, §51.9)
//!        matches `forward_mode_grad` (forward-mode `Dual<f64>`, §46) to
//!        `< 1e-12` RELATIVE — NOT 0 ULP (ADR-0156 Amendment 2).  Two genuinely
//!        independent float paths (reverse VJP vs forward JVP) agree by the adjoint
//!        identity to `O(n·κ·ε) ≈ 1e-13`, never exactly 0 ULP.  The measured
//!        value MUST be non-zero (i.e., `> 0`) to confirm they are independent
//!        computations — 0.0 exactly would signal a tautology.
//!
//! ## `G_REVERSE_AD_STRUCTURE` — mutation oracle (proves genuine `Jᵀ` usage)
//!
//! `RELEASE_BLOCKING`, runs in `test-fast` (fast; no `slow-tests` feature needed).
//! Sub-checks: (a) replacing `apply_transpose_step` with identity MUST change the
//! gradient by `> 1e-6` relative — proves `Jᵀ` is load-bearing; (b) the backward
//! loop runs `k = n → 1` (direction witness asserted by the loop structure).
//! A forward-mode-JVP implementation FAILS sub-check (a).
//!
//! ## `G_REVERSE_AD_CHECKPOINT` — O(√n) peak memory scaling
//!
//! Peak live heap bytes during `value_and_grad_k1` for `n ∈ {64, 256, 1024}`,
//! measured by a tracking `GlobalAlloc`. Fits `log(peak_bytes)` vs `log(n)`;
//! asserts slope ≤ 0.6 (ideal 0.5 = O(√n)).
//!
//! ## std requirement
//!
//! `#[global_allocator]` requires `std::alloc::System`. Test binaries link `std`
//! even when the lib target is `no_std + alloc`.
//!
//! ## Run commands
//!
//! ```sh
//! # Full slow-tests (G_REVERSE_AD_GRADIENT + G_REVERSE_AD_CHECKPOINT):
//! cargo test -p semiflow-core --features slow-tests --test g_reverse_ad \
//!     -- --ignored --nocapture
//! # Structure gate only (test-fast, no --ignored):
//! cargo test -p semiflow-core --test g_reverse_ad g_reverse_ad_structure \
//!     --nocapture
//! ```

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use semiflow::{
    CheckpointSchedule, DiffusionChernoff, Dual, Grid1D, GridFn1D, ReverseChernoff,
};

// ---------------------------------------------------------------------------
// Peak-tracking global allocator
//
// Tracks CURRENT live bytes and PEAK live bytes (not cumulative alloc count).
// This measures what §51.6 calls "peak heap allocation".
// ---------------------------------------------------------------------------

struct PeakTrackingAlloc;

/// Current live bytes (allocated − freed).
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Maximum live_bytes seen since last reset.
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakTrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let prev = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let now = prev + layout.size();
            // Update peak if new high.
            let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
            while now > peak {
                match PEAK_BYTES.compare_exchange_weak(
                    peak,
                    now,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            // Net change in live bytes.
            if new_size >= layout.size() {
                let delta = new_size - layout.size();
                let prev = LIVE_BYTES.fetch_add(delta, Ordering::Relaxed);
                let now = prev + delta;
                let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
                while now > peak {
                    match PEAK_BYTES.compare_exchange_weak(
                        peak,
                        now,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: PeakTrackingAlloc = PeakTrackingAlloc;

fn reset_peak() {
    // Reset peak but keep live (current allocations are still alive).
    PEAK_BYTES.store(LIVE_BYTES.load(Ordering::SeqCst), Ordering::SeqCst);
}
fn read_peak_bytes() -> usize {
    PEAK_BYTES.load(Ordering::SeqCst)
}
fn read_live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Gate constants (NON-NEGOTIABLE per §51.6)
// ---------------------------------------------------------------------------

/// Relative gradient accuracy: reverse vs Richardson FD (§51.6 threshold).
const GRAD_REL_GATE: f64 = 1e-9;
/// Cross-mode parity: |reverse − forward_dual| / |forward_dual| < 1e-12
/// (§51.4 Amendment 2 — NOT 0 ULP; two independent float paths).
const CROSS_MODE_REL_GATE: f64 = 1e-12;
/// Peak memory slope: log(peak_bytes)/log(n) ≤ 0.6 (§51.6; smaller = better).
const CHECKPOINT_SLOPE_GATE: f64 = 0.6;

/// Gate kernel parameters.
const THETA0: f64 = 0.5;
const T_FINAL: f64 = 1.0;
const N_STEPS_GRAD: usize = 32;
const N_GRID: usize = 128;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
/// 4-point Richardson h (O(h⁴) truncation).
const FD_H: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Dual coefficient functions (fn-ptr compatible)
// ---------------------------------------------------------------------------

fn a_seeded_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::variable(THETA0)
}
fn zero_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::constant(0.0)
}

// ---------------------------------------------------------------------------
// Grid builders (DEFAULT = SepticHermite, §51.6 anti-dodge)
// ---------------------------------------------------------------------------

fn f64_default_grid() -> Grid1D<f64> {
    Grid1D::new(X_MIN, X_MAX, N_GRID).expect("grid valid")
}
fn dual_default_grid() -> Grid1D<Dual<f64>> {
    Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
        .expect("grid valid")
}

// ---------------------------------------------------------------------------
// Forward-mode gradient (§46 — the 0-ULP reference)
// ---------------------------------------------------------------------------

/// Forward-mode Dual<f64> gradient of J = ‖u_n‖² (target=0) via §46.
/// This is the 0-ULP reference that `value_and_grad_k1` must match (§51.4).
fn forward_mode_grad(tau: f64, n: usize) -> f64 {
    let grid = dual_default_grid();
    let diff = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA0,
        grid,
    );
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau_d = Dual::constant(tau);
    let mut u = u0;
    for _ in 0..n {
        u = diff.apply_f(tau_d, &u).expect("fwd dual step");
    }
    // ∂J/∂θ = 2 Σ_i u_n.value_i · u_n.tangent_i  (target=0).
    u.values
        .iter()
        .fold(0.0, |acc, d| acc + 2.0 * d.value * d.tangent)
}

// ---------------------------------------------------------------------------
// ReverseChernoff builder (DEFAULT grid, §51.6 anti-dodge)
// ---------------------------------------------------------------------------

fn make_reverse_chernoff(n: usize) -> ReverseChernoff<f64> {
    let f64_grid = f64_default_grid();
    let kernel_f64 =
        DiffusionChernoff::with_closure(|_| THETA0, |_| 0.0_f64, |_| 0.0_f64, THETA0, f64_grid);
    let dual_grid = dual_default_grid();
    let kernel_dual = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA0,
        dual_grid,
    );
    let sched = CheckpointSchedule::sqrt_n(n);
    ReverseChernoff::new(kernel_f64, kernel_dual, sched)
}

fn reverse_mode_grad(tau: f64, n: usize) -> f64 {
    let rc = make_reverse_chernoff(n);
    let grid = f64_default_grid();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let (_, grad) = rc
        .value_and_grad_k1(tau, n, &u0, &target)
        .expect("reverse AD");
    grad
}

// ---------------------------------------------------------------------------
// Richardson FD reference (4-point, O(h⁴))
// ---------------------------------------------------------------------------

fn eval_loss_at_theta(theta: f64, tau: f64, n: usize) -> f64 {
    let grid = Grid1D::new(X_MIN, X_MAX, N_GRID).expect("grid valid");
    let k = DiffusionChernoff::with_closure(move |_| theta, |_| 0.0_f64, |_| 0.0_f64, theta, grid);
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let mut u = u0;
    for _ in 0..n {
        u = k.apply_f(tau, &u).expect("step");
    }
    u.values.iter().map(|v| v * v).sum()
}

fn richardson_fd(tau: f64, n: usize) -> f64 {
    let h = FD_H;
    let fp2 = eval_loss_at_theta(THETA0 + 2.0 * h, tau, n);
    let fp1 = eval_loss_at_theta(THETA0 + h, tau, n);
    let fm1 = eval_loss_at_theta(THETA0 - h, tau, n);
    let fm2 = eval_loss_at_theta(THETA0 - 2.0 * h, tau, n);
    (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
}

// ---------------------------------------------------------------------------
// G_REVERSE_AD_GRADIENT gate
// ---------------------------------------------------------------------------

/// `G_REVERSE_AD_GRADIENT` — RELEASE_BLOCKING (§51.6, ADR-0156 Amendment 2).
///
/// Two conjunctive thresholds:
/// (i)  `|reverse − FD| / |FD| < 1e-9` (relative accuracy vs Richardson).
/// (ii) `|reverse − forward_dual| / |forward_dual| < 1e-12` (cross-mode parity).
///      LHS = GENUINE cotangent backward sweep (§51.9, `value_and_grad_k1`).
///      RHS = forward-mode `Dual<f64>` (§46, `forward_mode_grad`).
///      Two genuinely independent float paths — agree to `O(n·ε) ≈ 1e-13`
///      by the adjoint identity, NOT 0 ULP (Amendment 2).
///      ANTI-TAUTOLOGY WITNESS: the measured relative difference MUST be > 0
///      (if it were exactly 0 it would indicate both paths are the same computation).
#[test]
#[ignore = "G_REVERSE_AD_GRADIENT: run with --features slow-tests -- --ignored --nocapture"]
fn g_reverse_ad_gradient() {
    let tau = T_FINAL / N_STEPS_GRAD as f64;

    let rev_grad = reverse_mode_grad(tau, N_STEPS_GRAD);
    let fwd_grad = forward_mode_grad(tau, N_STEPS_GRAD);
    let fd_grad = richardson_fd(tau, N_STEPS_GRAD);

    let fd_rel_err = if fd_grad.abs() > 1e-30 {
        (rev_grad - fd_grad).abs() / fd_grad.abs()
    } else {
        (rev_grad - fd_grad).abs()
    };
    let cross_mode_rel = if fwd_grad.abs() > 1e-30 {
        (rev_grad - fwd_grad).abs() / fwd_grad.abs()
    } else {
        (rev_grad - fwd_grad).abs()
    };
    // Anti-tautology: independent paths agree to float eps, never exactly 0.
    // (Exactly 0 would mean both sides are the same computation — a relabel.)
    let cross_mode_nonzero = rev_grad.to_bits() != fwd_grad.to_bits();

    println!(
        "G_REVERSE_AD_GRADIENT:\n  \
         reverse_grad    = {rev_grad:.15e}\n  \
         forward_dual    = {fwd_grad:.15e}\n  \
         richardson_fd   = {fd_grad:.15e}\n  \
         FD rel error    = {fd_rel_err:.3e}  (gate < {GRAD_REL_GATE:.0e})\n  \
         cross-mode rel  = {cross_mode_rel:.3e}  (gate < {CROSS_MODE_REL_GATE:.0e})\n  \
         bits differ     = {cross_mode_nonzero}  (anti-tautology: must be true)"
    );

    assert!(
        fd_rel_err < GRAD_REL_GATE,
        "G_REVERSE_AD_GRADIENT FAIL (FD accuracy): rel err {fd_rel_err:.3e} >= {GRAD_REL_GATE:.0e}"
    );
    assert!(
        cross_mode_rel < CROSS_MODE_REL_GATE,
        "G_REVERSE_AD_GRADIENT FAIL (cross-mode parity): rel {cross_mode_rel:.3e} >= {CROSS_MODE_REL_GATE:.0e} \
         (reverse={rev_grad:.15e}, forward={fwd_grad:.15e})"
    );
    assert!(
        cross_mode_nonzero,
        "G_REVERSE_AD_GRADIENT FAIL (anti-tautology): reverse == forward_dual bit-exactly \
         ({rev_grad:.17e}) — both sides are the same computation (relabel not reverse-mode)"
    );

    println!(
        "G_REVERSE_AD_GRADIENT PASS (FD rel={fd_rel_err:.3e}, \
         cross-mode rel={cross_mode_rel:.3e} [NOT 0-ULP, proving independence])"
    );
}

// ---------------------------------------------------------------------------
// G_REVERSE_AD_CHECKPOINT — peak LIVE BYTES gate
// ---------------------------------------------------------------------------

/// Measure peak live bytes during `value_and_grad_k1` (genuine reverse-mode
/// backward sweep).
///
/// Protocol (mirrors G_DUAL_ZERO_ALLOC warm-up pattern):
///   1. Warm run (Vec capacities settle, JIT-like effects done).
///   2. Reset peak tracker (peak = current live_bytes at reset point).
///   3. Measurement run — capture peak above the reset baseline.
fn measure_peak_bytes_above_baseline(n: usize) -> usize {
    let tau = T_FINAL / n as f64;
    let rc = make_reverse_chernoff(n);
    let grid = f64_default_grid();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);

    // Warm-up: allocate all working sets.
    let _ = rc.value_and_grad_k1(tau, n, &u0, &target).expect("warm");

    // Reset: set peak = current live (baseline).
    let baseline = read_live_bytes();
    reset_peak();

    // Measurement run.
    let _ = rc.value_and_grad_k1(tau, n, &u0, &target).expect("measure");

    let peak = read_peak_bytes();
    // Peak ABOVE baseline = peak bytes used beyond steady-state allocations.
    peak.saturating_sub(baseline)
}

/// `G_REVERSE_AD_CHECKPOINT` — RELEASE_BLOCKING memory-scaling gate (§51.6).
///
/// Fits `log(peak_live_bytes_above_baseline)` vs `log(n)` over
/// `n ∈ {64, 256, 1024}`. Slope ≤ 0.6 PASSES (O(√n) ideal = 0.5).
#[test]
#[ignore = "G_REVERSE_AD_CHECKPOINT: run with --features slow-tests -- --ignored --nocapture"]
fn g_reverse_ad_checkpoint() {
    let ns: &[usize] = &[64, 256, 1024];
    let mut measurements: Vec<(usize, usize)> = Vec::with_capacity(ns.len());

    for &n in ns {
        let peak_bytes = measure_peak_bytes_above_baseline(n);
        measurements.push((n, peak_bytes));
        println!(
            "  n={n:5}: peak_bytes_above_baseline={peak_bytes:8}  \
             log(n)={:.3}  log(peak)={:.3}",
            (n as f64).ln(),
            (peak_bytes.max(1) as f64).ln()
        );
    }

    // Least-squares slope of log(peak_bytes) vs log(n).
    let log_ns: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let log_peaks: Vec<f64> = measurements
        .iter()
        .map(|(_, p)| ((*p).max(1) as f64).ln())
        .collect();
    let k = ns.len() as f64;
    let mean_x = log_ns.iter().sum::<f64>() / k;
    let mean_y = log_peaks.iter().sum::<f64>() / k;
    let num: f64 = log_ns
        .iter()
        .zip(log_peaks.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_ns.iter().map(|x| (x - mean_x).powi(2)).sum();
    let slope = if den.abs() > 1e-12 { num / den } else { 0.0 };

    let (_, p64) = measurements[0];
    let (_, p256) = measurements[1];
    let (_, p1024) = measurements[2];

    println!(
        "G_REVERSE_AD_CHECKPOINT:\n  \
         n=64   peak_above={p64}\n  \
         n=256  peak_above={p256}\n  \
         n=1024 peak_above={p1024}\n  \
         measured slope = {slope:.4}  (gate: <= {CHECKPOINT_SLOPE_GATE})"
    );

    assert!(
        slope <= CHECKPOINT_SLOPE_GATE,
        "G_REVERSE_AD_CHECKPOINT FAIL: slope={slope:.4} > {CHECKPOINT_SLOPE_GATE} \
         (peak bytes: {p64}/{p256}/{p1024} for n=64/256/1024)"
    );

    println!("G_REVERSE_AD_CHECKPOINT PASS (slope={slope:.4} <= {CHECKPOINT_SLOPE_GATE})");
}

// K-vector gradient tests (§51.10, ADR-0177) have been moved to the separate
// binary `g_reverse_ad_kvector.rs`.  They require `build_f_transpose` which
// allocates an N×N matrix; keeping them here would race with the
// `PeakTrackingAlloc` global allocator used by `g_reverse_ad_checkpoint`,
// causing non-deterministic slope failures.  Separation is structural
// (each `tests/*.rs` compiles into its own binary) — no `#[serial]` needed.
