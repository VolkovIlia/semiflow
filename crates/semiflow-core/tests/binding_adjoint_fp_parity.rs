//! `G_BINDING_ADJOINT_FP_PARITY` — sub-test 1 (core golden + Lemma A.1 analytic anchor).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0138, ADR-0107 Amdt 1, slow-tests):
//!   1. Compute `AdjointFokkerPlanckChernoff::apply_into` at the canonical smoke params
//!      (`V8_1_TIER3_BINDING_DESIGN.md` §1.2, contracts/semiflow-core.properties.yaml
//!      §`G_BINDING_ADJOINT_FP_PARITY)`:
//!        a=0.5, b=0.0, c=0.0 (Brownian, mass-conserving),
//!        ρ₀ = `δ_0` (positions=[0.0], weights=[1.0]),
//!        tau=0.1, `n_steps=1` → exactly 4 Diracs.
//!   2. Assert Lemma A.1 closed-form anchor:
//!      h = 2√(aτ) = 2√(0.05), positions {+h, -h, 0, 0}, weights {¼,¼,½,τc=0},
//!      ‖pos − analytic‖∞ ≤ 1e-14, mass = 1.0.
//!   3. PRINT the golden (positions, weights) for embedding in binding sub-tests 2/3/4
//!      (each binding independently re-computes and asserts 0-ULP against this golden).
//!
//! ## Why this is GENUINE (not tautological)
//!
//! The analytic Lemma A.1 formula uses `h = 2√(aτ)`, `k = 2bτ`, `τc`.
//! The Rust core executes the same arithmetic in `lemma_a1_push`. Any sign
//! error or coefficient bug would shift positions or weights away from the
//! analytic anchor. Sub-tests 2/3/4 re-compute via FFI/PyO3/WASM and assert
//! byte-identical (0 ULP) output — any marshalling bug would show non-zero ULP.

// Integration test/example: allows for numerical patterns.
#![allow(clippy::doc_overindented_list_items, clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]

use semiflow_core::{
    AdjointFokkerPlanckChernoff, ChernoffFunction, DiffusionChernoff, Grid1D, MeasureState,
    ScratchPool,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters (§1.2, V8_1_TIER3_BINDING_DESIGN.md)
// ---------------------------------------------------------------------------

const A: f64 = 0.5;
const B: f64 = 0.0;
const C_COEF: f64 = 0.0;
const TAU: f64 = 0.1;

// ---------------------------------------------------------------------------
// Build helper
// ---------------------------------------------------------------------------

fn brownian_adjoint() -> AdjointFokkerPlanckChernoff<DiffusionChernoff<f64>, f64, 1> {
    let grid = Grid1D::new(-4.0_f64, 4.0, 32).unwrap();
    let fwd = DiffusionChernoff::new(|_| A, |_| B, |_| C_COEF, A, grid);
    AdjointFokkerPlanckChernoff::new(fwd, A, B, C_COEF)
}

// ---------------------------------------------------------------------------
// Core golden: one adjoint step from δ_0
// ---------------------------------------------------------------------------

/// Compute the core golden `(positions, weights)` after one step from `δ_0`.
///
/// Public so that FFI/WASM Rust binding tests can import and compare directly.
#[must_use]
pub fn canonical_adjoint_fp_core() -> (Vec<f64>, Vec<f64>) {
    let adj = brownian_adjoint();
    let rho0 = MeasureState::<f64, 1>::dirac([0.0_f64], 1.0);
    let mut rho1 = MeasureState::<f64, 1>::dirac([0.0_f64], 0.0);
    let mut pool = ScratchPool::<f64>::new();
    adj.apply_into(TAU, &rho0, &mut rho1, &mut pool)
        .expect("apply_into valid at canonical params");
    rho1.to_flat_buffers_d1()
}

// ---------------------------------------------------------------------------
// G_BINDING_ADJOINT_FP_PARITY sub-test 1: core golden + Lemma A.1 anchor
// ---------------------------------------------------------------------------

/// `G_BINDING_ADJOINT_FP_PARITY` sub-test 1 (core golden + analytic anchor).
///
/// Marked slow-tests; fast in practice (pure arithmetic, no PDE solve).
#[cfg(feature = "slow-tests")]
#[test]
fn g_binding_adjoint_fp_parity_core_golden() {
    let adj = brownian_adjoint();
    let rho0 = MeasureState::<f64, 1>::dirac([0.0_f64], 1.0);
    let mut rho1 = MeasureState::<f64, 1>::dirac([0.0_f64], 0.0);
    let mut pool = ScratchPool::<f64>::new();
    adj.apply_into(TAU, &rho0, &mut rho1, &mut pool)
        .expect("apply_into valid");

    // -----------------------------------------------------------------------
    // Structural check
    // -----------------------------------------------------------------------
    assert_eq!(rho1.n_diracs(), 4, "expected 4 children from δ_0");

    // -----------------------------------------------------------------------
    // Mass conservation (c = 0 → exact mass = 1.0)
    // -----------------------------------------------------------------------
    let mass = rho1.total_variation();
    // c=0 → weight[3] = τc = 0.0, weights sum = 0.25+0.25+0.50+0.0 = 1.0 exactly.
    assert!(
        (mass - 1.0_f64).abs() < 1e-14,
        "mass not conserved: got {mass}, expected 1.0"
    );

    // -----------------------------------------------------------------------
    // Lemma A.1 analytic anchor
    // -----------------------------------------------------------------------
    let h = 2.0_f64 * (A * TAU).sqrt(); // 2√(0.05)
    let k = 2.0_f64 * B * TAU; // 0.0
    let tc = TAU * C_COEF; // 0.0

    // Analytic positions/weights for δ_0 pushed by Lemma A.1:
    //   child 0: x+h, weight ¼
    //   child 1: x-h, weight ¼
    //   child 2: x+k, weight ½
    //   child 3: x,   weight τc = 0
    let analytic_pos = [h, -h, k, 0.0_f64];
    let analytic_wt = [0.25_f64, 0.25, 0.5, tc];

    // Second moment: ∑ p_j² w_j = h²·0.25 + h²·0.25 + 0²·0.5 + 0²·0 = 0.5·h²
    let sm = rho1.second_moment();
    let sm_analytic = 0.5_f64 * h * h;
    assert!(
        (sm - sm_analytic).abs() < 1e-14,
        "second_moment mismatch: got {sm:.16e}, expected {sm_analytic:.16e}"
    );

    // First moment: ∑ p_j w_j = h·0.25 + (-h)·0.25 + 0·0.5 + 0·0 = 0
    let fm = rho1.pair(|p| p[0]);
    assert!(fm.abs() < 1e-14, "first moment nonzero: {fm:.16e}");

    // -----------------------------------------------------------------------
    // Extract golden and compare to analytic (1e-14 tolerance)
    // -----------------------------------------------------------------------
    let (golden_pos, golden_wt) = canonical_adjoint_fp_core();
    assert_eq!(golden_pos.len(), 4);
    assert_eq!(golden_wt.len(), 4);

    for (i, (&gp, &ap)) in golden_pos.iter().zip(analytic_pos.iter()).enumerate() {
        assert!(
            (gp - ap).abs() < 1e-14,
            "golden_pos[{i}]: got {gp:.16e}, analytic {ap:.16e}"
        );
    }
    for (i, (&gw, &aw)) in golden_wt.iter().zip(analytic_wt.iter()).enumerate() {
        assert!(
            (gw - aw).abs() < 1e-14,
            "golden_wt[{i}]: got {gw:.16e}, analytic {aw:.16e}"
        );
    }

    // -----------------------------------------------------------------------
    // Print golden for embedding in binding integration tests
    // -----------------------------------------------------------------------
    println!(
        "\nG_BINDING_ADJOINT_FP_PARITY sub-test 1 (core golden + Lemma A.1 anchor):\n\
         a={A}, b={B}, c={C_COEF}, tau={TAU}\n\
         h = 2√(aτ) = {h:.16e}\n\
         Golden positions: {golden_pos:?}\n\
         Golden weights:   {golden_wt:?}\n\
         mass = {mass:.16e}  (expected 1.0)\n\
         second_moment = {sm:.16e}  (analytic {sm_analytic:.16e})\n\
         Lemma A.1 anchor: PASS (‖·‖∞ within 1e-14)",
    );
}
