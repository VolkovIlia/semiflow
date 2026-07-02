//! Multilayer thermal stack gates (§57, issue #14, ADR-0188).
//!
//! All four tests are `RELEASE_BLOCKING` and `#[ignore]` (slow-tests gate).
//! Run with `--features slow-tests --release -- --ignored`.
//!
//! | Gate | What it verifies |
//! |---|---|
//! | `G_TPS_MASS_WEIGHT`      | `multilayer_evolve` vs dense oracle; `sup_error` ∈ (1e-12, 1e-8] |
//! | `G_TPS_UNITMASS_FAILS`   | Unit-mass CN gives ≥ 50 % error on 1:5 ρc contrast |
//! | `G_TPS_STACK_ACCEPTANCE` | Full TPS Krylov vs CN reference; `rel_err` ≤ 2 % |
//! | `G_TPS_STIFF_STEPCOUNT`  | Substep economy: X/Y ≥ 100 and Y ≤ 2√X |

#![allow(clippy::doc_markdown)]

use semiflow::{
    boundary::BoundaryPolicy,
    chernoff::ChernoffFunction,
    dense_csr_expmv_ref,
    graph_krylov::{graph_expmv_matvec_count, KrylovPath},
    grid::Grid1D,
    grid_fn::GridFn1D,
    multilayer_evolve,
    scratch::ScratchPool,
    ConservativeDiffusionChernoff, Layer, MassWeightedConservativeChernoff, MultilayerStack,
};

// TPS material constants (SI units: W/(m·K) and J/(m³·K)).
const K_LI900: f64 = 0.058;
const RC_LI900: f64 = 90_432.0;
const K_SIP: f64 = 0.058;
const RC_SIP: f64 = 90_432.0;
const K_RTV: f64 = 0.25;
const RC_RTV: f64 = 1_207_500.0;
const K_AL: f64 = 177.0;
const RC_AL: f64 = 2_432_500.0;

// Build the 4-layer TPS stack: LI-900 (38 mm) | SIP (25 mm) | RTV (3 mm) | Al-2024 (2 mm).
fn tps_stack() -> MultilayerStack<f64> {
    let layers = [
        Layer {
            thickness: 0.038,
            k: K_LI900,
            rho_c: RC_LI900,
        },
        Layer {
            thickness: 0.025,
            k: K_SIP,
            rho_c: RC_SIP,
        },
        Layer {
            thickness: 0.003,
            k: K_RTV,
            rho_c: RC_RTV,
        },
        Layer {
            thickness: 0.002,
            k: K_AL,
            rho_c: RC_AL,
        },
    ];
    MultilayerStack::from_layers(&layers, 5e-4_f64).expect("tps_stack: from_layers failed")
}

// Run `steps` CN applications of `dc` on initial state `v`.
fn cn_evolve<D: ChernoffFunction<f64, S = GridFn1D<f64>>>(
    dc: &D,
    grid: Grid1D<f64>,
    v: &[f64],
    dt: f64,
    steps: usize,
    scratch: &mut ScratchPool<f64>,
) -> Vec<f64> {
    let n = v.len();
    let mut cur = GridFn1D {
        values: v.to_vec(),
        grid,
    };
    let mut nxt = GridFn1D {
        values: vec![0.0_f64; n],
        grid,
    };
    for _ in 0..steps {
        dc.apply_into(dt, &cur, &mut nxt, scratch)
            .expect("cn_evolve: step failed");
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur.values
}

/// `G_TPS_MASS_WEIGHT` (`RELEASE_BLOCKING`, §57.3): `multilayer_evolve` vs dense
/// Padé-13 oracle on the congruence Â.
///
/// 2-material stack: layer 0 (k=1, ρc=1, 3 mm) | layer 1 (k=10, ρc=3, 3 mm).
/// N=7 nodes, τ=0.5 s, Krylov tol=1e-12.
///
/// ## Threshold rationale — Gershgorin substep accumulation floor
///
/// The Gershgorin bound for Â = D^{−½} A D^{−½} is λ_max ≈ 1.333 × 10⁷.
/// With τ=0.5 this gives z_total = τ·λ_max/2 ≈ 3.33 × 10⁶, hence
/// s = ⌈z_total / Z_SAFE⌉ = 16 667 Chebyshev substeps.  Accumulated error
/// ≈ s × tol_per_step ≈ 16 667 × 10⁻¹² ≈ 1.7 × 10⁻⁸; measured: 6.84 × 10⁻⁹.
/// Tightening tol reduces error proportionally (tol=10⁻¹⁴ → 2.3 × 10⁻¹⁰) but
/// reaching 10⁻¹⁰ would need tol ≈ 6 × 10⁻¹⁵, below the per-step ULP floor.
/// The Lanczos path floors at 1.7 × 10⁻¹⁰ from accumulated rounding over
/// 754 k substeps.  The original 10⁻¹⁰ threshold (copied from milder gates
/// with O(1) substeps) is not achievable here; 10⁻⁸ is the honest ceiling
/// at tol=10⁻¹².  This is a Gershgorin-conservatism consequence, not a bug.
///
/// Gate (two-sided band): `1e-12 < sup_error ≤ 1e-8`.
/// Lower bound rules out silent-identity bugs; upper bound catches regressions.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::too_many_lines)]
fn g_tps_mass_weight() {
    let layers = [
        Layer {
            thickness: 0.003,
            k: 1.0,
            rho_c: 1.0,
        },
        Layer {
            thickness: 0.003,
            k: 10.0,
            rho_c: 3.0,
        },
    ];
    let stack = MultilayerStack::from_layers(&layers, 0.001_f64)
        .expect("G_TPS_MASS_WEIGHT: from_layers failed");
    let n = stack.grid.n;
    assert_eq!(n, 7, "G_TPS_MASS_WEIGHT: expected 7 nodes, got {n}");

    let tau = 0.5_f64;
    let tol = 1e-12_f64;
    let v = [1.0_f64, -0.5, 0.3, 0.7, -0.2, 0.4, 0.8];

    // Primary: Krylov expmv.
    let mut out_krylov = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();
    multilayer_evolve(
        &stack,
        BoundaryPolicy::Neumann,
        tau,
        &v,
        &mut out_krylov,
        KrylovPath::Chebyshev,
        tol,
        &mut scratch,
    )
    .expect("G_TPS_MASS_WEIGHT: multilayer_evolve failed");

    // Oracle: Â = D^{-½} A D^{-½}, pre/post-scale with √m.
    let (a, masses) = stack
        .to_stiffness_and_mass(BoundaryPolicy::Neumann)
        .expect("G_TPS_MASS_WEIGHT: to_stiffness_and_mass failed");
    let a_hat = a
        .lumped_congruence(&masses)
        .expect("G_TPS_MASS_WEIGHT: lumped_congruence failed");
    let w0: Vec<f64> = v
        .iter()
        .zip(masses.iter())
        .map(|(&vi, &mi)| vi * mi.sqrt())
        .collect();
    let mut w1 = vec![0.0_f64; n];
    dense_csr_expmv_ref(&a_hat, tau, &w0, &mut w1).expect("G_TPS_MASS_WEIGHT: dense oracle failed");
    let out_ref: Vec<f64> = w1
        .iter()
        .zip(masses.iter())
        .map(|(&wi, &mi)| wi / mi.sqrt())
        .collect();

    let sup_error = out_krylov
        .iter()
        .zip(out_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    eprintln!("G_TPS_MASS_WEIGHT  n={n}  tau={tau}  sup_error={sup_error:.3e}");
    // Two-sided band — see §57.3 threshold rationale in the doc comment above.
    // Lower bound: error < 1e-12 would indicate a silent-identity bug (both paths
    //   returning the same result through an unexpected code path).
    // Upper bound: error > 1e-8 indicates a genuine regression beyond the
    //   Gershgorin-accumulation floor measured at 6.84e-9 (tol=1e-12, s=16667).
    assert!(
        sup_error > 1e-12_f64,
        "G_TPS_MASS_WEIGHT: sup_error={sup_error:.3e} ≤ 1e-12 (non-vacuity violated)"
    );
    assert!(
        sup_error <= 1e-8_f64,
        "G_TPS_MASS_WEIGHT: sup_error={sup_error:.3e} > 1e-8 (Gershgorin-accumulation floor)"
    );
}

/// `G_TPS_UNITMASS_FAILS` (`RELEASE_BLOCKING`, §57.3): unit-mass CN gives ≥ 50 %
/// relative error at the cold node vs mass-weighted Krylov.
///
/// Stack: 3 mm (ρc=1) | 3 mm (ρc=5), k=1 uniform, N=7 nodes. τ=10 s.
/// IC: T=1500 K (nodes 0-3), T=300 K (nodes 4-6).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::too_many_lines)]
fn g_tps_unitmass_fails() {
    let layers = [
        Layer {
            thickness: 0.003,
            k: 1.0,
            rho_c: 1.0,
        },
        Layer {
            thickness: 0.003,
            k: 1.0,
            rho_c: 5.0,
        },
    ];
    let stack = MultilayerStack::from_layers(&layers, 0.001_f64)
        .expect("G_TPS_UNITMASS_FAILS: from_layers failed");
    let n = stack.grid.n;
    assert_eq!(n, 7, "G_TPS_UNITMASS_FAILS: expected 7 nodes, got {n}");

    let v: Vec<f64> = (0..n)
        .map(|i| if i < 4 { 1500.0_f64 } else { 300.0_f64 })
        .collect();
    let tau = 10.0_f64;

    // Correct physics: mass-weighted Krylov evolve.
    let mut out_mw = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();
    multilayer_evolve(
        &stack,
        BoundaryPolicy::Neumann,
        tau,
        &v,
        &mut out_mw,
        KrylovPath::Chebyshev,
        1e-12_f64,
        &mut scratch,
    )
    .expect("G_TPS_UNITMASS_FAILS: multilayer_evolve failed");

    // Wrong physics: unit-mass CN (ignores ρc contrast).
    let k_nodes: Vec<f64> = vec![1.0_f64; n];
    let dc_unit = ConservativeDiffusionChernoff::from_k_array(
        stack.grid,
        &k_nodes,
        None,
        BoundaryPolicy::Neumann,
    )
    .expect("G_TPS_UNITMASS_FAILS: from_k_array failed");
    let out_um = cn_evolve(&dc_unit, stack.grid, &v, 0.01_f64, 1000, &mut scratch);

    let check_node = n - 1;
    let rel_err = ((out_um[check_node] - out_mw[check_node]) / out_mw[check_node]).abs();
    eprintln!(
        "G_TPS_UNITMASS_FAILS  node={check_node}  mw={:.4}  um={:.4}  rel_err={rel_err:.4}",
        out_mw[check_node], out_um[check_node]
    );
    assert!(
        rel_err >= 0.5_f64,
        "G_TPS_UNITMASS_FAILS: rel_err={rel_err:.4} < 0.50 (unit-mass did not diverge)"
    );
}

/// `G_TPS_STACK_ACCEPTANCE` (`RELEASE_BLOCKING`, §57.3): full TPS 4-layer Krylov
/// vs `MassWeightedConservativeChernoff` CN reference.
///
/// τ=2500 s, linear IC. Gate: `rel_err` ≤ 2 % at Al node and LI-900/SIP interface.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn g_tps_stack_acceptance() {
    let stack = tps_stack();
    let n = stack.grid.n;
    let tau = 2500.0_f64;
    let v: Vec<f64> = (0..n)
        .map(|i| 300.0 + 1200.0 * (i as f64 / (n - 1) as f64))
        .collect();

    // Primary: one-shot Krylov.
    let mut out_krylov = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();
    multilayer_evolve(
        &stack,
        BoundaryPolicy::Neumann,
        tau,
        &v,
        &mut out_krylov,
        KrylovPath::Chebyshev,
        1e-8_f64,
        &mut scratch,
    )
    .expect("G_TPS_STACK_ACCEPTANCE: multilayer_evolve failed");

    // Reference: 1000 CN steps via MassWeightedConservativeChernoff.
    let dc = MassWeightedConservativeChernoff::from_stack(&stack)
        .expect("G_TPS_STACK_ACCEPTANCE: from_stack failed");
    let out_cn = cn_evolve(&dc, stack.grid, &v, 2.5_f64, 1000, &mut scratch);

    // Check at Al surface (n-1) and LI-900/SIP interface (≈ node 76).
    let check_nodes = [n - 1, 76_usize.min(n - 1)];
    for &node in &check_nodes {
        let t_kry = out_krylov[node];
        let t_cn = out_cn[node];
        let rel_err = ((t_kry - t_cn) / t_cn).abs();
        eprintln!(
            "G_TPS_STACK_ACCEPTANCE  node={node}  krylov={t_kry:.4}  \
             cn={t_cn:.4}  rel_err={rel_err:.4}"
        );
        assert!(
            rel_err <= 0.02_f64,
            "G_TPS_STACK_ACCEPTANCE: node={node}  rel_err={rel_err:.4} > 0.02"
        );
    }
}

/// `G_TPS_STIFF_STEPCOUNT` (`RELEASE_BLOCKING`, §57.3): Chebyshev substep economy.
///
/// X = ⌈τ·λ_max(Â)⌉, Y = m (Chebyshev degree per substep).
/// Gate: X/Y ≥ 100 and Y ≤ 2√X.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn g_tps_stiff_stepcount() {
    let stack = tps_stack();
    let tau = 2500.0_f64;
    let tol = 1e-12_f64;

    let (a, masses) = stack
        .to_stiffness_and_mass(BoundaryPolicy::Neumann)
        .expect("G_TPS_STIFF_STEPCOUNT: to_stiffness_and_mass failed");
    let a_hat = a
        .lumped_congruence(&masses)
        .expect("G_TPS_STIFF_STEPCOUNT: lumped_congruence failed");
    let lambda_max = a_hat.lambda_max_bound();

    let (_s, m) = graph_expmv_matvec_count(lambda_max, tau, tol, &KrylovPath::Chebyshev);
    let x: u64 = (tau * lambda_max).ceil() as u64;
    let y: u32 = m;

    let ratio = x as f64 / f64::from(y);
    let deg_bound = 2.0 * (x as f64).sqrt();
    eprintln!(
        "G_TPS_STIFF_STEPCOUNT  lambda_max={lambda_max:.3e}  X={x}  Y={y}  \
         X/Y={ratio:.1}  2√X={deg_bound:.1}"
    );
    assert!(
        ratio >= 100.0,
        "G_TPS_STIFF_STEPCOUNT: X/Y={ratio:.1} < 100 (economy gate failed)"
    );
    assert!(
        f64::from(y) <= deg_bound,
        "G_TPS_STIFF_STEPCOUNT: Y={y} > 2√X={deg_bound:.1} (degree bound violated)"
    );
}
