//! Issue #11 gate tests — conservative (divergence-form) diffusion (ADR-0187, §56).
//!
//! Five `RELEASE_BLOCKING` gates (all require `--features slow-tests --release -- --ignored`):
//!
//! | Gate | What it verifies |
//! |---|---|
//! | `G_CONS_SERIES`       | Sharp k=[1,100,1] steady-state vs analytic series-resistance |
//! | `G_CONS_NONCONS_FAILS`| Same stack via non-conservative `DiffusionChernoff` fails ≥50% |
//! | `G_CONS_SYMOP`        | Assembled `A=−L_k` → `SymmetricOperator` + Krylov vs Padé13 |
//! | `G_CONS_ORDER`        | Const k, CN vs Padé-13 exact eigenmode: slope ∈ [1.9, 2.3] |
//! | `G_CONS_CONTACT`      | Contact resistance: interface jump ≈ q·R_c < 1% |

use semiflow::{
    assemble_conservative_csr_1d,
    boundary::BoundaryPolicy,
    chernoff::ChernoffFunction,
    dense_csr_expmv_ref,
    graph_krylov::{graph_expmv_krylov, KrylovPath},
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
    steady_state_dirichlet_1d,
    ConservativeDiffusionChernoff, DiffusionChernoff, SymmetricOperator,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Three-layer k field: nodes `[0..lo)` and `[hi..n)` get `k_outer`; `[lo..hi)` gets `k_inner`.
fn k_three_layer(n: usize, lo: usize, hi: usize, k_outer: f64, k_inner: f64) -> Vec<f64> {
    (0..n)
        .map(|i| if i < lo || i >= hi { k_outer } else { k_inner })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// G_CONS_SERIES
// ─────────────────────────────────────────────────────────────────────────────

/// `G_CONS_SERIES` (`RELEASE_BLOCKING`, §56.4): sharp k=[1,100,1] steady-state
/// via harmonic-mean FV vs analytic series-resistance network.
///
/// Non-vacuity:
/// 1. `max_k/min_k ≥ 50` asserted.
/// 2. Jump at each interface asserted.
/// 3. Per-face ΔT relative error < 1 %.
/// 4. Flux spread across faces < 1 % (conservation).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::cast_precision_loss)]
fn g_cons_series() {
    let n = 12_usize;
    let dx = 1.0_f64 / (n - 1) as f64;
    let k_nodes = k_three_layer(n, 4, 8, 1.0, 100.0);

    let k_max = k_nodes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let k_min = k_nodes.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        k_max / k_min >= 50.0,
        "G_CONS_SERIES: max_k/min_k={:.1} < 50 (non-vacuity fail)",
        k_max / k_min
    );
    assert!(
        (k_nodes[3] - k_nodes[4]).abs() > 50.0,
        "G_CONS_SERIES: k does not jump at interface 3→4"
    );
    assert!(
        (k_nodes[7] - k_nodes[8]).abs() > 50.0,
        "G_CONS_SERIES: k does not jump at interface 7→8"
    );

    let t_left = 1.0_f64;
    let t_right = 0.0_f64;
    let u = steady_state_dirichlet_1d(&k_nodes, None, dx, t_left, t_right)
        .expect("G_CONS_SERIES: steady_state_dirichlet_1d failed");
    assert_eq!(u.len(), n);

    // Analytic oracle: face resistance = dx / k_harm.
    let faces_r: Vec<f64> = (0..n - 1)
        .map(|i| {
            let k_h = 2.0 * k_nodes[i] * k_nodes[i + 1] / (k_nodes[i] + k_nodes[i + 1]);
            dx / k_h
        })
        .collect();
    let r_tot: f64 = faces_r.iter().sum();
    let q = (t_left - t_right) / r_tot;

    // Per-face ΔT error.
    let mut max_rel_err = 0.0_f64;
    for i in 0..n - 1 {
        let dt_analytic = q * faces_r[i];
        let dt_numeric = u[i] - u[i + 1]; // positive since T_l > T_r
        let rel_err = ((dt_numeric - dt_analytic) / dt_analytic).abs();
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
    }

    // Flux conservation: q_face = ΔT_face / R_face must equal q everywhere.
    let fluxes: Vec<f64> = (0..n - 1).map(|i| (u[i] - u[i + 1]) / faces_r[i]).collect();
    let flux_min = fluxes.iter().copied().fold(f64::INFINITY, f64::min);
    let flux_max = fluxes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let flux_spread = (flux_max - flux_min).abs() / q.abs();

    eprintln!(
        "G_CONS_SERIES  n={n}  q={q:.6}  max_rel_err={max_rel_err:.3e}  \
         flux_spread={flux_spread:.3e}"
    );
    assert!(max_rel_err < 0.01, "G_CONS_SERIES: max_rel_err={max_rel_err:.3e} ≥ 1%");
    assert!(flux_spread < 0.01, "G_CONS_SERIES: flux_spread={flux_spread:.3e} ≥ 1%");
}

// ─────────────────────────────────────────────────────────────────────────────
// G_CONS_NONCONS_FAILS
// ─────────────────────────────────────────────────────────────────────────────

/// `G_CONS_NONCONS_FAILS` — TEETH gate: same sharp stack via non-conservative
/// `DiffusionChernoff` evolved to steady state must fail the series-resistance
/// test by ≥ 50%.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(
    clippy::too_many_lines,   // 55 lines: evolved steady-state + oracle verification
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
)]
fn g_cons_noncons_fails() {
    let n = 12_usize;
    let dx = 1.0_f64 / (n - 1) as f64;
    let k_nodes = k_three_layer(n, 4, 8, 1.0, 100.0);

    let k_max = k_nodes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let k_min = k_nodes.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(k_max / k_min >= 50.0, "G_CONS_NONCONS_FAILS: max_k/min_k < 50");

    // Non-conservative DiffusionChernoff with a'=0 (ignores interface jumps).
    let k_nc = k_nodes.clone();
    let grid = Grid1D::new(0.0_f64, 1.0_f64, n).expect("G_CONS_NONCONS_FAILS: grid");
    let dc = DiffusionChernoff::with_closure(
        move |x: f64| {
            let i = ((x / dx).round() as usize).min(n - 1);
            k_nc[i]
        },
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        100.0_f64,
        grid,
    );

    // Evolve 1000 steps of τ=0.01 from linear IC; clamp Dirichlet BCs each step.
    let t_left = 1.0_f64;
    let t_right = 0.0_f64;
    let mut cur = GridFn1D::from_fn(grid, |x| t_left + (t_right - t_left) * x);
    let mut nxt = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();
    for _ in 0..1000_usize {
        dc.apply_into(0.01_f64, &cur, &mut nxt, &mut scratch)
            .expect("G_CONS_NONCONS_FAILS: apply_into failed");
        nxt.values[0] = t_left;
        nxt.values[n - 1] = t_right;
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_nc = &cur.values;

    // Analytic oracle (harmonic-mean faces, same k stack).
    let faces_r: Vec<f64> = (0..n - 1)
        .map(|i| {
            let k_h = 2.0 * k_nodes[i] * k_nodes[i + 1] / (k_nodes[i] + k_nodes[i + 1]);
            dx / k_h
        })
        .collect();
    let r_tot: f64 = faces_r.iter().sum();
    let q_analytic = (t_left - t_right) / r_tot;

    let mut max_rel_err = 0.0_f64;
    for i in 0..n - 1 {
        let dt_analytic = q_analytic * faces_r[i];
        if dt_analytic.abs() > 1e-15 {
            let dt_numeric = u_nc[i] - u_nc[i + 1];
            let rel_err = ((dt_numeric - dt_analytic) / dt_analytic).abs();
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        }
    }

    eprintln!("G_CONS_NONCONS_FAILS  max_rel_err={max_rel_err:.3e}  (expect ≥ 0.50)");
    assert!(
        max_rel_err >= 0.50,
        "G_CONS_NONCONS_FAILS: non-conservative error {max_rel_err:.3e} < 50%"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// G_CONS_SYMOP
// ─────────────────────────────────────────────────────────────────────────────

/// `G_CONS_SYMOP` (`RELEASE_BLOCKING`, §56.2): assemble `A=−L_k` → `SymmetricOperator`,
/// verify `graph_expmv_krylov` (Chebyshev) vs `dense_csr_expmv_ref` (Padé-13) agree
/// within a real, non-trivial error band.
///
/// Root-cause investigation (original bug): τ=0.5 → `z=τ·λ_max_bound/2≈800` →
/// `em_z=e^{−800}=0.0` (f64 underflow) → all Chebyshev coefficients `c_k=0·I_k(800)=0·∞=NaN`
/// → `dst_krylov` filled with NaN → `f64::max(acc, NaN) = acc` swallowed every NaN diff
/// → `sup_error` appeared as 0.0 though the methods had never been compared at all.
///
/// Fix: τ=0.03 → z=48 → `em_z≈7.7e−21` (representable) → genuine Chebyshev convergence
/// (~52 terms, ~1e−13 truncation error) → real algorithmic difference vs Padé-13.
///
/// Non-vacuity:
/// 1. NaN guard on both outputs (catches em_z-underflow regression).
/// 2. `dst_norm` > 1e-14 (non-trivial result from the PSD operator).
/// 3. Two-sided bound: 1e-16 < `sup_error` ≤ 1e-10 (genuinely independent algorithms
///    cannot agree BIT-IDENTICALLY; strict lower bound catches shared-path or NaN masking).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn g_cons_symop() {
    let n = 9_usize;
    let k_nodes = k_three_layer(n, 3, 6, 1.0, 100.0);

    let k_max = k_nodes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let k_min = k_nodes.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(k_max / k_min >= 50.0, "G_CONS_SYMOP: max_k/min_k < 50");

    let grid = Grid1D::new(0.0_f64, 1.0_f64, n).expect("G_CONS_SYMOP: grid");
    let op: SymmetricOperator<f64> =
        assemble_conservative_csr_1d(grid, &k_nodes, None, BoundaryPolicy::Neumann)
            .expect("G_CONS_SYMOP: assemble failed");

    // Gaussian test vector.
    let sigma = 2.0_f64;
    let center = (n as f64) / 2.0;
    let src: Vec<f64> = (0..n)
        .map(|i| (-0.5 * (i as f64 - center).powi(2) / sigma.powi(2)).exp())
        .collect();

    // τ=0.5 → z=τ·λ_max_bound/2 = 0.5·25600/2 = 6400 → e^{−6400} underflows to 0
    // → NaN (0·∞) WITHOUT substepping.  With substep scaling s=32 each substep has
    // z_sub=200 ≤ Z_SAFE → e^{−200}≈1.4e−87 (representable) → finite output.
    let tau = 0.5_f64;
    let mut dst_krylov = vec![0.0_f64; n];
    let mut dst_dense = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();

    graph_expmv_krylov(
        &op, tau, &src, &mut dst_krylov, KrylovPath::Chebyshev, 1e-12, &mut scratch,
    )
    .expect("G_CONS_SYMOP: krylov failed");
    dense_csr_expmv_ref(&op, tau, &src, &mut dst_dense)
        .expect("G_CONS_SYMOP: dense ref failed");

    // NaN guard: detect silently-broken Chebyshev (NaN from 0·∞ when em_z underflows).
    assert!(
        !dst_krylov.iter().any(|v| v.is_nan()),
        "G_CONS_SYMOP: dst_krylov contains NaN — Chebyshev path broken (em_z underflow?)"
    );
    assert!(
        !dst_dense.iter().any(|v| v.is_nan()),
        "G_CONS_SYMOP: dst_dense contains NaN — Padé path broken"
    );

    let sup_error = dst_krylov
        .iter()
        .zip(dst_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let dst_norm = dst_dense.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    eprintln!(
        "G_CONS_SYMOP  n={n}  tau={tau}  dst_norm={dst_norm:.3e}  sup_error={sup_error:.3e}"
    );
    assert!(dst_norm > 1e-14, "G_CONS_SYMOP: dst_norm={dst_norm:.3e} trivially zero");
    // Lower bound: two genuinely independent algorithms cannot agree BIT-IDENTICALLY.
    // sup_error ≤ 1e-16 indicates shared code path or NaN-masking re-regression.
    assert!(
        sup_error > 1e-16,
        "G_CONS_SYMOP: sup_error={sup_error:.3e} ≤ 1e-16 — shared code path or NaN masking"
    );
    assert!(sup_error <= 1e-10, "G_CONS_SYMOP: sup_error={sup_error:.3e} > 1e-10");
}

// ─────────────────────────────────────────────────────────────────────────────
// G_CONS_ORDER
// ─────────────────────────────────────────────────────────────────────────────

/// `G_CONS_ORDER` (`RELEASE_BLOCKING`, §56.3): temporal order-2 gate for
/// `ConservativeDiffusionChernoff` (Crank–Nicolson / Padé(1,1)).
///
/// **Methodology — clean eigenmode-vs-exact (no self-reference)**:
/// * Constant k=1, Dirichlet BC=0 on [0,1], n=12 grid (`n_int`=10 interior nodes).
/// * Initial condition u₀(x)=sin(πx) — satisfies zero Dirichlet BCs exactly.
/// * **Exact discrete reference**: `e^{−t·A_int}·u₀` via `dense_csr_expmv_ref`
///   (Padé-13 on the (n−2)×(n−2) interior Dirichlet tridiagonal `A_int`).
/// * **Temporal error** = sup‖CN(τ, steps) − `u_ref`‖ measured at three τ levels
///   (τ₀, τ₀/2, τ₀/4) with FIXED grid and FIXED `t_end`=0.1.
/// * Both adjacent-pair slopes must lie in `[1.9, 2.3]`.
///
/// Why non-degenerate:
/// * No self-reference — comparing CN against an independent Padé-13 matrix exponential.
/// * No spatial-floor — spatial error is absorbed into the exact discrete reference.
/// * `t_end`=0.1 → decay factor exp(−π²·0.1) ≈ 0.37; solution actively evolving.
/// * CN is unconditionally stable (no CFL constraint on τ).
///
/// Original anomaly (slope=3.1765): self-convergence with smooth k and Neumann BCs
/// near-equilibrium — solution variance ≈ machine-eps, differences are noise-dominated,
/// and the reference level introduces log₂ bias from the ratio not being exactly 4.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::too_many_lines)]
fn g_cons_order() {
    use core::f64::consts::PI;

    // n=12 (n_int=10 interior nodes ≤ MAX_DENSE_N=12). Const k=1, Dirichlet BC=0.
    let n     = 12_usize;
    let n_int = n - 2; // 10 interior nodes
    let grid  = Grid1D::new(0.0_f64, 1.0_f64, n).expect("G_CONS_ORDER: grid");
    let dx    = grid.dx();
    // t_end=0.1 → exp(−π²·0.1) ≈ 0.37: actively decaying, not near-equilibrium.
    let t_end = 0.1_f64;

    // Interior initial condition u₀[i] = sin(π·x_{i+1}), i = 0..n_int.
    let u0_int: Vec<f64> = (0..n_int)
        .map(|i| (PI * (i + 1) as f64 * dx).sin())
        .collect();

    // ── Exact discrete reference: e^{-t_end · A_int} · u₀ via Padé-13 ──────
    // Interior Dirichlet system for const k=1: tridiag(−1/dx², 2/dx², −1/dx²).
    // A_int is the (n_int × n_int) interior block of the full Dirichlet operator.
    let a_diag    =  2.0_f64 / (dx * dx); //  2/dx²
    let a_offdiag = -1.0_f64 / (dx * dx); // -1/dx²
    let mut row_ptr = vec![0_usize; n_int + 1];
    let mut col_idx: Vec<u32>  = Vec::with_capacity(3 * n_int);
    let mut vals:    Vec<f64>  = Vec::with_capacity(3 * n_int);
    for i in 0..n_int {
        if i > 0 {
            col_idx.push((i - 1) as u32);
            vals.push(a_offdiag);
        }
        col_idx.push(i as u32);
        vals.push(a_diag);
        if i + 1 < n_int {
            col_idx.push((i + 1) as u32);
            vals.push(a_offdiag);
        }
        row_ptr[i + 1] = col_idx.len();
    }
    let int_op = SymmetricOperator::from_csr(n_int, &row_ptr, &col_idx, &vals, 1e-10_f64)
        .expect("G_CONS_ORDER: interior SymmetricOperator failed");
    let mut u_ref = vec![0.0_f64; n_int];
    dense_csr_expmv_ref(&int_op, t_end, &u0_int, &mut u_ref)
        .expect("G_CONS_ORDER: dense_csr_expmv_ref failed");

    // Non-vacuity: reference must be strictly smaller than initial (mode is decaying).
    let ref_norm = u_ref.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let u0_norm  = u0_int.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        ref_norm < u0_norm * 0.9,
        "G_CONS_ORDER: reference has not decayed (ref_norm={ref_norm:.3e}, u0_norm={u0_norm:.3e})"
    );

    // ── CN solver (const k=1, Dirichlet BC=0) ────────────────────────────────
    let k_nodes = vec![1.0_f64; n];
    let cd = ConservativeDiffusionChernoff::from_k_array(
        grid,
        &k_nodes,
        None,
        BoundaryPolicy::Dirichlet { value: 0.0_f64 },
    )
    .expect("G_CONS_ORDER: from_k_array failed");

    let mut scratch = ScratchPool::new();
    // Returns interior values (indices 1..n-1) after `steps` CN steps of size `tau`.
    let mut cn_run = |tau: f64, steps: usize| -> Vec<f64> {
        let mut cur = GridFn1D::from_fn(grid, |x| (PI * x).sin());
        let mut nxt = GridFn1D::from_fn(grid, |_| 0.0_f64);
        for _ in 0..steps {
            cd.apply_into(tau, &cur, &mut nxt, &mut scratch)
                .expect("G_CONS_ORDER: apply_into failed");
            core::mem::swap(&mut cur, &mut nxt);
        }
        cur.values[1..n - 1].to_vec()
    };

    // Three τ-levels at ratio 2: τ₀ = t_end/20 = 0.005.
    let n_base   = 20_usize;
    let tau_base = t_end / n_base as f64;
    let u_c = cn_run(tau_base,       n_base    ); // 20 steps
    let u_f = cn_run(tau_base / 2.0, n_base * 2); // 40 steps
    let u_x = cn_run(tau_base / 4.0, n_base * 4); // 80 steps

    let sup_err = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    };
    let err_c = sup_err(&u_c, &u_ref);
    let err_f = sup_err(&u_f, &u_ref);
    let err_x = sup_err(&u_x, &u_ref);

    let ratio2slope = |hi: f64, lo: f64| -> f64 {
        if hi > 0.0 && lo > 0.0 { (hi / lo).log2() } else { 0.0_f64 }
    };
    let slope_cf = ratio2slope(err_c, err_f);
    let slope_fx = ratio2slope(err_f, err_x);

    eprintln!(
        "G_CONS_ORDER  n={n}  n_int={n_int}  dx={dx:.5}  t_end={t_end}  tau_base={tau_base:.5}\n  \
         err_coarse={err_c:.3e}  err_fine={err_f:.3e}  err_xfine={err_x:.3e}\n  \
         slope(c→f)={slope_cf:.4}  slope(f→x)={slope_fx:.4}"
    );

    assert!(err_f < err_c,
        "G_CONS_ORDER: err_fine={err_f:.3e} ≥ err_coarse={err_c:.3e} — wrong direction");
    assert!(err_x < err_f,
        "G_CONS_ORDER: err_xfine={err_x:.3e} ≥ err_fine={err_f:.3e} — wrong direction");
    // Two-sided gate [1.9, 2.3]: CN is order-2 → slope must be ≈ 2.
    assert!(
        (1.9_f64..=2.3_f64).contains(&slope_cf),
        "G_CONS_ORDER: coarse→fine slope={slope_cf:.4} ∉ [1.9, 2.3]"
    );
    assert!(
        (1.9_f64..=2.3_f64).contains(&slope_fx),
        "G_CONS_ORDER: fine→xfine slope={slope_fx:.4} ∉ [1.9, 2.3]"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// G_CONS_CONTACT
// ─────────────────────────────────────────────────────────────────────────────

/// `G_CONS_CONTACT` (`RELEASE_BLOCKING`, §56.4.b): two-layer k=[1,10] with contact
/// resistance `R_c=0.05` at the interface.
///
/// Root-cause of original anomaly (`rel_err=0.000e0`, always): the numeric formula
/// `dt_contact = t_face * (u[4]−u[5]) * R_c` is an algebraic identity.  In steady
/// state `u[4]−u[5] = q*(R_mat+R_c)` and `t_face = 1/(R_mat+R_c)`, so the
/// expression reduces to `q*R_c = dt_contact_analytic` by substitution — no
/// matter what the solver computes, the formula returns the analytic answer.
///
/// Fix: two independent checks that genuinely exercise the contact-resistance solver:
///
/// 1. Full analytic profile comparison: `T[j] = t_left − q·Σ_{i<j} R_i` vs `u[j]`
///    (independent derivation from heat-flux conservation; max relative error < 1%).
///
/// 2. `R_c=0` comparison: solving without contact resistance must give a STRICTLY SMALLER
///    jump at the interface face.  Exact extra-jump formula verifies contact contribution.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn g_cons_contact() {
    let n = 10_usize;
    let dx = 1.0_f64 / (n - 1) as f64;
    let k_nodes: Vec<f64> = (0..n).map(|i| if i < 5 { 1.0_f64 } else { 10.0_f64 }).collect();
    let r_c = 0.05_f64;

    // Contact resistance at face 4 (between nodes 4 and 5).
    let face_idx = 4_usize;
    let mut r_contact = vec![0.0_f64; n - 1];
    r_contact[face_idx] = r_c;

    // Non-vacuity: R_c must be significant vs material face resistance.
    let k_harm_face =
        2.0 * k_nodes[face_idx] * k_nodes[face_idx + 1] / (k_nodes[face_idx] + k_nodes[face_idx + 1]);
    let r_mat_face = dx / k_harm_face;
    let r_ratio = r_c / r_mat_face;
    assert!(
        r_ratio >= 0.1,
        "G_CONS_CONTACT: R_c/R_mat={r_ratio:.3} < 0.1 — contact too small"
    );

    let t_left = 1.0_f64;
    let t_right = 0.0_f64;
    let u = steady_state_dirichlet_1d(&k_nodes, Some(&r_contact), dx, t_left, t_right)
        .expect("G_CONS_CONTACT: steady_state_dirichlet_1d failed");

    // ── Check 1: independent full analytic profile ────────────────────────────
    // T[j] = t_left − q · Σ_{i<j} R_i  where R_i = dx/k_harm_i + r_contact[i].
    // Derived from heat-flux conservation; independent of the Thomas-algorithm path.
    let all_faces_r: Vec<f64> = (0..n - 1)
        .map(|i| {
            let k_h = 2.0 * k_nodes[i] * k_nodes[i + 1] / (k_nodes[i] + k_nodes[i + 1]);
            dx / k_h + r_contact[i]
        })
        .collect();
    let total_r: f64 = all_faces_r.iter().sum();
    let q_analytic = (t_left - t_right) / total_r;

    let t_analytic: Vec<f64> = (0..n)
        .map(|j| t_left - q_analytic * all_faces_r[..j].iter().sum::<f64>())
        .collect();

    let max_abs_err = u
        .iter()
        .zip(t_analytic.iter())
        .map(|(&un, &ta)| (un - ta).abs())
        .fold(0.0_f64, f64::max);
    let max_rel_err = max_abs_err / (t_left - t_right);

    eprintln!(
        "G_CONS_CONTACT  n={n}  R_c={r_c}  q={q_analytic:.6}  \
         max_abs_err={max_abs_err:.3e}  max_rel_err={max_rel_err:.3e}"
    );
    assert!(
        max_rel_err < 0.01,
        "G_CONS_CONTACT: profile max_rel_err={max_rel_err:.3e} ≥ 1% — solver or contact error"
    );

    // ── Check 2: R_c=0 comparison — contact adds a measurable extra jump ─────
    // Without contact, total resistance drops by R_c and flux increases.
    let u_no_rc = steady_state_dirichlet_1d(&k_nodes, None, dx, t_left, t_right)
        .expect("G_CONS_CONTACT: no-contact solve failed");

    let jump_with_rc = u[face_idx] - u[face_idx + 1];
    let jump_no_rc   = u_no_rc[face_idx] - u_no_rc[face_idx + 1];
    let extra_jump   = jump_with_rc - jump_no_rc;

    // Exact analytic extra jump accounting for flux change: q_with*(R_mat+R_c) - q_without*R_mat.
    let total_r_no_rc: f64 = all_faces_r
        .iter()
        .enumerate()
        .map(|(i, &r)| if i == face_idx { r - r_c } else { r })
        .sum();
    let q_no_rc_val  = (t_left - t_right) / total_r_no_rc;
    let expected_extra = q_analytic * (r_mat_face + r_c) - q_no_rc_val * r_mat_face;
    let extra_rel_err  = ((extra_jump - expected_extra) / expected_extra).abs();

    eprintln!(
        "G_CONS_CONTACT  jump_with={jump_with_rc:.6}  jump_no={jump_no_rc:.6}  \
         extra={extra_jump:.6}  expected={expected_extra:.6}  \
         extra_rel_err={extra_rel_err:.3e}"
    );
    assert!(
        jump_with_rc > jump_no_rc,
        "G_CONS_CONTACT: contact resistance must increase interface jump \
         (with={jump_with_rc:.6} ≤ without={jump_no_rc:.6})"
    );
    assert!(
        extra_rel_err < 0.01,
        "G_CONS_CONTACT: extra_jump={extra_jump:.6} vs expected={expected_extra:.6} \
         — extra_rel_err={extra_rel_err:.3e} ≥ 1%"
    );
}
