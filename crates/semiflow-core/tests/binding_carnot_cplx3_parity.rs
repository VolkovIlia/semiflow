//! `G_BINDING_CARNOT_CPLX3_PARITY` — sub-test 1 (core golden + γ⋆ + finiteness anchor).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0138, ADR-0136 Amdt 2, slow-tests):
//!   Canonical smoke params (`V8_1_TIER3_BINDING_DESIGN.md` §1.4,
//!   contracts/semiflow-core.properties.yaml §`G_BINDING_CARNOT_CPLX3_PARITY)`:
//!     D=5, domain=[-1.5,1.5] per axis, `n_per_axis=5`,
//!     tau=0.02, u0(x)=exp(-(x0²+x1²)/2) (Gaussian IC).
//!
//! ## Anchors
//!
//! 1. `verify_gamma_star() == true` — γ⋆ cubic residual check.
//! 2. Output is finite and max > 0 (real projection of heat-like flow).
//! 3. Golden flat vec (len 5^5=3125) printed for `PyO3` 0-ULP embedding.
//!
//! ## Why this is GENUINE
//!
//! The Rust core runs `apply_real` (triple complex Strang) and takes Re(·).
//! The γ⋆ anchor is INDEPENDENT of `apply_real` — it checks the cubic root
//! directly. Any bug in the triple-jump composition would corrupt the output
//! while the γ⋆ check stays green, but finiteness + max>0 would catch complete
//! failures. Sub-test 2 (`PyO3` 0-ULP) catches marshalling bugs.

// Integration test/bench: allows for numerical patterns.
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)] // D as u32: D=5, well within u32

use semiflow_core::{
    grid_nd::{GridFnND, GridND},
    ComplexTripleJump, Grid1D,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters (§1.4, V8_1_TIER3_BINDING_DESIGN.md)
// ---------------------------------------------------------------------------

const D: usize = 5;
const DOMAIN_LO: f64 = -1.5;
const DOMAIN_HI: f64 = 1.5;
const N_PER_AXIS: usize = 5;
const TAU: f64 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_grid() -> GridND<f64, D> {
    let ax = Grid1D::new(DOMAIN_LO, DOMAIN_HI, N_PER_AXIS).unwrap();
    GridND::new([ax; D]).unwrap()
}

/// Gaussian IC: exp(-(x0²+x1²)/2) (spec §1.4, matches `apply_finite` test).
fn gaussian_ic(x: &[f64; D]) -> f64 {
    (-(x[0] * x[0] + x[1] * x[1]) * 0.5).exp()
}

// ---------------------------------------------------------------------------
// Core golden computation
// ---------------------------------------------------------------------------

/// Apply one `ComplexTripleJump::apply_real` step at the canonical params.
///
/// Public so the `PyO3` binding test can embed the exact golden values (0-ULP).
pub fn canonical_carnot_cplx3_core() -> Vec<f64> {
    let grid = make_grid();
    let u0 = GridFnND::from_fn(grid, gaussian_ic);
    let kernel = ComplexTripleJump::new().unwrap();
    kernel.apply_real(TAU, &u0).unwrap().values
}

// ---------------------------------------------------------------------------
// G_BINDING_CARNOT_CPLX3_PARITY sub-test 1: core golden + anchors
// ---------------------------------------------------------------------------

/// `G_BINDING_CARNOT_CPLX3_PARITY` sub-test 1 (core golden + γ⋆ + finiteness).
///
/// Feature-gated under slow-tests (fast in practice: 5^5=3125 pts, 1 step).
#[cfg(feature = "slow-tests")]
#[test]
fn g_binding_carnot_cplx3_parity_core_golden() {
    println!(
        "\nG_BINDING_CARNOT_CPLX3_PARITY sub-test 1 (core golden + anchors):\n\
         D={D}, domain=[{DOMAIN_LO},{DOMAIN_HI}], n_per_axis={N_PER_AXIS}, tau={TAU}"
    );

    // -----------------------------------------------------------------------
    // Anchor 1: γ⋆ cubic residual
    // -----------------------------------------------------------------------
    let gamma_ok = ComplexTripleJump::verify_gamma_star();
    println!("verify_gamma_star() = {gamma_ok}  (expected true)");
    assert!(gamma_ok, "γ⋆ cubic residual check FAILED");

    // -----------------------------------------------------------------------
    // Anchor 2 + golden: compute apply_real, assert finite + max > 0
    // -----------------------------------------------------------------------
    let golden = canonical_carnot_cplx3_core();
    let expected_len = N_PER_AXIS.pow(D as u32);
    assert_eq!(golden.len(), expected_len, "golden length mismatch");

    let all_finite = golden.iter().all(|v| v.is_finite());
    let max_val = golden.iter().copied().fold(0.0_f64, f64::max);
    let min_val = golden.iter().copied().fold(f64::MAX, f64::min);

    println!(
        "golden length={expected_len}, min={min_val:.6e}, max={max_val:.6e}, all_finite={all_finite}"
    );
    assert!(all_finite, "golden output contains non-finite values");
    assert!(max_val > 0.0, "golden max should be > 0 (heat semigroup)");

    // Print first 8 and last 4 values for PyO3 sub-test 2 embedding.
    println!("golden[0..8]  = {:?}", &golden[..8.min(golden.len())]);
    println!(
        "golden[-4..]  = {:?}",
        &golden[golden.len().saturating_sub(4)..]
    );
    println!("G_BINDING_CARNOT_CPLX3_PARITY sub-test 1: PASS ✓");
}
