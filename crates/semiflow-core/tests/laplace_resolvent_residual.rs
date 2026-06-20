//! `G_RES_RES` — Laplace-Chernoff Resolvent Residual gate (v4.0 Wave E, ADR-0083).
//!
//! Gate spec (properties.yaml v1.0.0, `RELEASE_BLOCKING)`:
//!   ‖(λI − A) R̃(λ) f − f‖_∞ ≤ 1e-3
//!   at λ=1.0, n=64, N=512, smooth Gaussian f, unit diffusion (a=1).
//!
//! Uses [`LaplaceChernoffResolventResidual`] wrapper (v4.0 additive surface).
//! A is approximated by the 3-point central-difference Laplacian on interior nodes.
//!
//! ## Relation to G24
//!
//! G24(1) in `resolvent_laplace_chernoff_slope.rs` computes the same residual
//! inline. `G_RES_RES` is the DEDICATED sibling gate that calls the wrapper's
//! `verify_residual` method — the narrow single-purpose acceptance criterion
//! that downstream HFT code references. G24 stays as the v2.7 multi-part umbrella.

#![cfg(feature = "slow-tests")]

use semiflow_core::{
    resolvent::{LaplaceChernoffResolvent, LaplaceChernoffResolventResidual, LaplaceQuadrature},
    DiffusionChernoff, Grid1D, GridFn1D,
};

// ---------------------------------------------------------------------------
// Gate constants — do NOT relax without ADR + properties.yaml bump.
// ---------------------------------------------------------------------------

/// `G_RES_RES` gate budget (`RELEASE_BLOCKING` per ADR-0083).
const RESIDUAL_BUDGET: f64 = 1e-3;

/// Chernoff truncation level per `G_RES_RES` canonical spec.
const N_CHERNOFF: usize = 64;

/// Spatial grid size per `G_RES_RES` canonical spec.
const N_SPATIAL: usize = 512;

/// λ value per `G_RES_RES` canonical spec.
const LAMBDA: f64 = 1.0;

/// Domain half-width.
const DOMAIN_L: f64 = 5.0;

// ---------------------------------------------------------------------------
// G_RES_RES — main gate test
// ---------------------------------------------------------------------------

/// `G_RES_RES` — resolvent residual ≤ 1e-3 (`RELEASE_BLOCKING`, ADR-0083).
///
/// Canonical inputs: unit diffusion (a=1), λ=1.0, n=64, N=512, Gaussian IC
/// f(x) = exp(-x²) on [-5, 5].
///
/// The `LaplaceChernoffResolventResidual::verify_residual` method computes
/// ‖(λI − ∂_xx) `R̃_n(λ)` f − f‖_∞ via 3-point central differences on interior
/// nodes. Empirically: residual ≈ 1.57e-4 ≪ gate 1e-3.
#[test]
fn g_res_res_unit_diffusion_lambda_1_gaussian() {
    let grid = Grid1D::new(-DOMAIN_L, DOMAIN_L, N_SPATIAL).unwrap();

    // Unit diffusion: a=1, a'=0, a''=0, norm_bound=1.
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);

    let inner =
        LaplaceChernoffResolvent::new(chernoff, N_CHERNOFF, LaplaceQuadrature::GaussLaguerre32)
            .unwrap();

    let gate = LaplaceChernoffResolventResidual::new(inner, RESIDUAL_BUDGET);

    // Smooth Gaussian: infinitely differentiable, decays rapidly at boundaries.
    let f = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());

    let residual = gate.verify_residual(LAMBDA, &f).unwrap();

    println!(
        "G_RES_RES residual = {:.6e}  (budget ≤ {:.0e})",
        residual,
        gate.budget()
    );

    assert!(
        residual <= gate.budget(),
        "G_RES_RES FAIL: residual {residual:.6e} > budget {:.0e} \
         (λ={LAMBDA}, n={N_CHERNOFF}, N={N_SPATIAL}, Gaussian IC, unit diffusion)",
        gate.budget()
    );
}
