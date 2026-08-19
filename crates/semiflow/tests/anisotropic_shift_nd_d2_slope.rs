//! `G_DDIM` D=2 — d-D anisotropic shift self-convergence slope (`RELEASE_BLOCKING`).
//!
//! Gate: successive-difference OLS slope in `[-0.75, -0.45]` — order ½, not 1
//! (ADR-0112 §Decision 2+3, RE-BASED by ADR-0191 AMENDMENT 3).
//!
//! Method: reference-free convergence ladder calling the REAL
//! `AnisotropicShiftChernoffND::apply_into`. Fixed spatial grid `N_AXIS = 8` per axis (8² = 64 nodes);
//! ladder `n ∈ {32, 64, 128, 256, 512}`, each run `apply_into` `n` times with `tau = T/n`.
//! The fitted quantity is `sup|u_2n − u_n|` over consecutive ladder entries,
//! which needs no reference run and so cannot be contaminated by one.
//!
//! <details><summary>Superseded pre-ADR-0191 description (kept for provenance)</summary>
//!
//! > Gate: slope ≤ -0.95 (order-1, ADR-0112 §Decision 2+3).
//! >
//! > Method: temporal self-convergence test calling the REAL `AnisotropicShiftChernoffND::apply_into`.
//! > Fixed spatial grid `N_AXIS=8` per axis (8²=64 nodes); reference at `n_ref=512` steps.
//! > Sweep n ∈ {32,64,128,256}: iterate `apply_into` n times with tau=T/n.
//! > Error = sup-norm vs reference on the SAME grid (spatial error is common-mode).
//! > OLS slope of log(err) vs log(n); gate `assert!(slope.is_finite()` && slope <= -0.95).
//! >
//! > `N_AXIS=8` chosen so grid spacing dx≈1.43 is comparable to the 5-pt GH node displacement
//! > `2√τ·σ_max·η_max` ≈ 0.35–1.4, ensuring the spatial interpolation floor does not dominate
//! > the temporal convergence signal.  Sweeping n∈{32,64,128,256} skips the pre-asymptotic
//! > n=16 region where the per-step τ² curvature bends the OLS slope above −0.95.
//! >
//! > ADR-0112 §Decision 3 specifies `N_AXIS=128` for D=2 in the normative N(D) ladder, but
//! > empirical validation (QA run 2026-05-30) shows `N_AXIS=128` gives slope ≈ −0.05 (spatially
//! > floor-dominated, non-monotone) while `N_AXIS=8` gives slope ≈ −1.03 (clean).  The ADR
//! > "floor cancels common-mode" argument fails for this parameter range because `u_n` and `u_ref`
//! > accumulate interpolation error at different rates (O(n·dx^p) each), so the floor does NOT
//! > fully cancel in the difference |`u_n` − `u_ref`|.  The ADR N(D) ladder needs correction.
//! > FLAG for ai-solutions-architect: ADR-0112 §Decision 3 `N_AXIS(D=2)=128` is empirically wrong;
//! > should be `N_AXIS(D=2)=8`.  See adversarial QA probe 2026-05-30.
//! >
//!
//! </details>
//!
//! ## Estimator re-based (ADR-0191 AMENDMENT 3, 2026-08-14)
//!
//! This gate used to compare each swept `n` against a single reference run at
//! `n_ref = 512`, which is only twice the largest swept `n`, so the last point
//! measured the *reference's* remaining error rather than its own. Holding the
//! sweep fixed and raising `n_ref` walked the reported slope from −0.92 to
//! −0.54 — a converged estimator would have been flat.
//!
//! It now fits the OLS slope of the **successive differences**
//! `sup|u_2n − u_n|`, which need no reference at all and therefore cannot be
//! contaminated. Their ratio settles on √2 across `n ∈ [32, 16384]` and
//! `N_AXIS ∈ {8, 16, 32}`: the kernel's global temporal order on its own
//! normative variable-`A` datum is **½**, not 1 — the `√τ` inside the
//! Gauss–Hermite shift makes the frozen-coefficient mismatch local `O(τ^{3/2})`.
//! The threshold is `≤ −0.45` accordingly.
//!
//! This is not a relaxation of what the gate could previously detect: the old
//! reading of ≈ −1.0 was an artefact, the new estimator is strictly stronger,
//! and the band is two-sided so a kernel that becomes genuinely order-1 fails
//! here rather than silently passing.
//!
//! Sub-tests:
//!   1. F(0)=I smoke: ‖F(τ)·1 − 1‖_∞ < 1e-12 at τ ∈ {0, T/16, T/128}.
//!   2. Successive-difference slope in [-0.75, -0.45] (calls real `apply_into`).
//!
//! Feature: slow-tests.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS; values ≤ 512 ≤ 2^52
#![allow(clippy::cast_lossless)] // u32→f64 for n_steps: infallible, project idiom
#![allow(clippy::too_many_lines)] // one linear scenario, kept inline to read top-down

use semiflow::{
    approximation::ApproximationSubspace,
    grid_nd::{GridFnND, GridND},
    AnisotropicShiftChernoffND, ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};

const T: f64 = 0.5;
const N_AXIS: usize = 8;
/// Reference-free ladder: the OLS slope of `sup|u_2n - u_n|` over these `n`.
const N_LADDER: [u32; 5] = [32, 64, 128, 256, 512];
const SLOPE_GATE: f64 = -0.45;
/// Upper guard: a genuinely order-1 kernel must fail here too, not pass quietly.
const SLOPE_CEILING: f64 = -0.75;

fn make_grid_d2(n: usize) -> GridND<f64, 2> {
    let ax = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    GridND::new([ax, ax]).unwrap()
}

/// Build anisotropic kernel per math §32.5 spec: a = I + 0.25·tanh(xᵢ+xⱼ) off-diag.
fn make_kernel_d2(n: usize) -> AnisotropicShiftChernoffND<f64, 2> {
    let grid = make_grid_d2(n);
    AnisotropicShiftChernoffND::new(
        |x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            let off = 0.25 * (x[0] + x[1]).tanh();
            a.set(0, 1, off);
            a.set(1, 0, off);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x: &[f64; 2]| 0.0_f64,
        grid,
    )
    .unwrap()
}

fn initial_fn(x: &[f64; 2]) -> f64 {
    (-x[0] * x[0] - x[1] * x[1]).exp()
}

/// Iterate `kernel.apply_into` `n_steps` times with step `tau=T/n_steps`.
fn run_steps(kernel: &AnisotropicShiftChernoffND<f64, 2>, n_steps: u32) -> GridFnND<f64, 2> {
    let tau = T / n_steps as f64;
    let f0 = GridFnND::from_fn(kernel.grid().clone(), initial_fn);
    let mut src = f0;
    let mut dst = GridFnND::from_fn(kernel.grid().clone(), |_| 0.0_f64);
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut pool).unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src
}

/// NaN-propagating sup-norm of (a - b).
fn sup_diff(a: &GridFnND<f64, 2>, b: &GridFnND<f64, 2>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(&ai, &bi)| (ai - bi).abs())
        .fold(0.0_f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) })
}

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

/// `G_DDIM` D=2 — anisotropic shift Chernoff self-convergence (calls real `apply_into`).
#[test]
fn g_ddim_d2_slope() {
    // --- F(0)=I smoke check (ADR-0112 §Decision 5) ---
    {
        let kernel_smoke = make_kernel_d2(N_AXIS);
        let one_fn = GridFnND::from_fn(kernel_smoke.grid().clone(), |_| 1.0_f64);
        let mut pool = ScratchPool::<f64>::new();
        let mut out = one_fn.clone();
        for &tau in &[0.0_f64, T / 16.0, T / 128.0] {
            kernel_smoke
                .apply_into(tau, &one_fn, &mut out, &mut pool)
                .unwrap();
            let sup_err = out
                .values
                .iter()
                .map(|&v| (v - 1.0_f64).abs())
                .fold(0.0_f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) });
            assert!(
                sup_err < 1e-12,
                "G_DDIM D=2 F(0)=I smoke: tau={tau} ‖out−1‖_∞={sup_err:.3e} ≥ 1e-12"
            );
        }
    }

    // --- Self-convergence slope (calls real apply_into iterated n times) ---
    // Reference run at n_ref=512; sweep n ∈ {32,64,128,256}.
    // Spatial grid is shared (N_AXIS=8): spatial error cancels common-mode.
    // Sweep starts at n=32 to skip the pre-asymptotic τ² curvature region (n=16).
    let kernel = make_kernel_d2(N_AXIS);

    assert!(
        kernel.in_subspace(&GridFnND::from_fn(kernel.grid().clone(), initial_fn)),
        "G_DDIM D=2: initial fn not in ApproximationSubspace<2>"
    );

    // Reference-free: successive differences over the ladder. `d_k = sup|u_2n - u_n|`
    // scales as `C * n^-p` for a scheme of order `p`, with no reference run to
    // contaminate the fit (ADR-0191 AMENDMENT 3).
    // Progress is printed per ladder entry: at D >= 4 this gate runs for hours,
    // and a run that emits nothing until it is finished cannot be distinguished
    // from a hung one (ADR-0192 AMENDMENT 4).
    let t_start = std::time::Instant::now();
    let us: Vec<_> = N_LADDER
        .iter()
        .map(|&n| {
            let u = run_steps(&kernel, n);
            println!(
                "G_DDIM D=2: ladder n={n} done  (+{:.0} s cumulative)",
                t_start.elapsed().as_secs_f64()
            );
            u
        })
        .collect();
    let diffs: Vec<f64> = (0..N_LADDER.len() - 1)
        .map(|k| sup_diff(&us[k], &us[k + 1]))
        .collect();
    let ns: Vec<u32> = N_LADDER[..N_LADDER.len() - 1].to_vec();

    for (&n, &d) in ns.iter().zip(diffs.iter()) {
        println!("G_DDIM D=2: n={n} -> {}  sup|u_2n - u_n|={d:.4e}", 2 * n);
    }

    let slope = ols_slope(&ns, &diffs);
    println!(
        "G_DDIM D=2: successive-difference OLS slope = {slope:.4}  \
         (gate: {SLOPE_CEILING} <= slope <= {SLOPE_GATE}; order 1/2 expected)"
    );
    assert!(
        slope.is_finite() && slope <= SLOPE_GATE,
        "G_DDIM D=2: slope {slope:.4} not finite-and-≤{SLOPE_GATE}"
    );
    assert!(
        slope >= SLOPE_CEILING,
        "G_DDIM D=2: slope {slope:.4} is steeper than {SLOPE_CEILING} — the kernel \
         appears to have gained an order. That is good news, but it means \
         `AnisotropicShiftChernoffND::order()` and this gate both need revisiting \
         rather than silently passing (ADR-0191 AMENDMENT 3)."
    );
}

#[test]
fn g_ddim_d2_in_subspace_witness() {
    use semiflow::approximation::ApproximationSubspace;
    let kernel = make_kernel_d2(8);
    let f0 = GridFnND::from_fn(kernel.grid().clone(), initial_fn);
    assert!(
        kernel.in_subspace(&f0),
        "D=2 Gaussian IC must be in ApproximationSubspace<2>"
    );
}
