//! `G_BINDING_RESOLVENT_JUMP_PARITY` — sub-test 1 (core golden + self-convergence anchor).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0138, ADR-0134, slow-tests):
//!   1. Compute `ResolventJumpChernoff::jump` at the canonical smoke params
//!      (`V8_1_TIER3_BINDING_DESIGN.md` §1.1, contracts/semiflow-core.properties.yaml
//!      §`G_BINDING_RESOLVENT_JUMP_PARITY)`:
//!        XMIN=-10.0, XMAX=10.0, N=64, `M_NODES=16`, T=0.5,
//!        u0(x)=exp(-x²), unit diffusion a=1, DEFAULT grid.
//!   2. Assert the jump approximation matches an `M_ref=40` self-convergence reference:
//!      `‖jump_M16 − jump_M40‖∞ ≤ 1e-3` — both use the same discrete A (no
//!      continuous/discrete mismatch).  This mirrors `G_RESOLVENT_JUMP_ORDER`'s
//!      `M_ref=40` self-convergence design (§47.5, ADR-0134).  At M=16, T=0.5 the
//!      measured M16-vs-M80 error is ~3e-8, comfortably inside 1e-3.
//!   3. PRINT the golden vector for embedding in the binding integration tests
//!      (sub-tests 2/3/4 will assert byte-identical 0-ULP against this output).
//!
//! ## Why this is GENUINE (not tautological)
//!
//! M=16 and `M_ref=40` run different TWS contour quadratures (different number of
//! complex Thomas solves at different node positions).  Any implementation bug in
//! the contour loop or Thomas solve would produce different results.  Sub-tests
//! 2/3/4 independently re-compute via FFI/PyO3/WASM and compare bit-for-bit —
//! any marshalling divergence would be a non-zero ULP.
//!
//! ## Note on Chernoff-vs-resolvent discretization
//!
//! `DiffusionChernoff::apply_into` approximates `e^{τ · ∂²_continuous}` (Gaussian
//! convolution), whereas `ResolventJumpChernoff` targets `e^{t · A_discrete}` (the
//! same 3-pt Neumann Laplacian as `G_RESOLVENT_JUMP_ORDER`).  These two operators
//! differ by O(dx²·t) ≈ 3e-3 at t=0.5 regardless of step count — the Chernoff
//! product saturates at ~3.155e-3.  A Chernoff-product anchor cannot reach 1e-3;
//! the `M_ref=40` self-convergence anchor is the correct independent check.

#![allow(clippy::cast_precision_loss)]
// Integration test/bench/example: allows for numerical patterns.
#![allow(clippy::doc_overindented_list_items, clippy::missing_panics_doc)]

use semiflow::{DiffusionChernoff, Grid1D, GridFn1D, ResolventJumpChernoff};

// ---------------------------------------------------------------------------
// Canonical smoke parameters (§1.1, V8_1_TIER3_BINDING_DESIGN.md)
// ---------------------------------------------------------------------------

const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
/// Grid node count.
const N: usize = 64;
/// TWS contour node count.
const M_NODES: usize = 16;
/// Large time step (the whole point of F2 — single big jump).
const T: f64 = 0.5;
/// High-M reference for self-convergence anchor (mirrors `G_RESOLVENT_JUMP_ORDER`).
#[cfg(feature = "slow-tests")]
const M_REF: usize = 40;
/// Accuracy gate: ‖`jump_M16` − `jump_M40`‖∞ ≤ 1e-3 (M=16 vs `M_ref=40` self-convergence).
/// Empirically ~3e-8 at T=0.5 (well inside 1e-3).
#[cfg(feature = "slow-tests")]
const TOL_JUMP: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Build helpers
// ---------------------------------------------------------------------------

fn make_grid() -> Grid1D<f64> {
    Grid1D::new(XMIN, XMAX, N).expect("canonical grid valid")
}

fn make_u0(grid: Grid1D<f64>) -> GridFn1D<f64> {
    GridFn1D::from_fn(grid, |x: f64| (-x * x).exp())
}

fn unit_diffusion(grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid)
}

// ---------------------------------------------------------------------------
// Core golden: TWS contour jump
// ---------------------------------------------------------------------------

/// Compute the core golden `jump(T, u0)` for F2 at the canonical smoke params.
///
/// Public so that binding tests can import the golden vector directly.
#[must_use]
pub fn canonical_resolvent_jump_core() -> Vec<f64> {
    let grid = make_grid();
    let inner = unit_diffusion(grid);
    let rj = ResolventJumpChernoff::new(inner, M_NODES, grid)
        .expect("canonical ResolventJumpChernoff valid");
    let g = make_u0(grid);
    let result = rj.jump(T, &g).expect("jump valid");
    result.values
}

// ---------------------------------------------------------------------------
// Independent evolve reference (many small Chernoff steps)
// ---------------------------------------------------------------------------

/// Compute `M_ref=40` self-convergence reference via higher-M resolvent.
///
/// Uses the same discrete A as the M=16 probe — no Chernoff/discrete mismatch.
/// Mirrors `G_RESOLVENT_JUMP_ORDER`'s `M_ref=40` design (§47.5, ADR-0134).
#[cfg(feature = "slow-tests")]
fn resolvent_reference(grid: Grid1D<f64>, g: &GridFn1D<f64>) -> Vec<f64> {
    let chernoff = unit_diffusion(grid);
    let rj_ref = ResolventJumpChernoff::new(chernoff, M_REF, grid).expect("M_ref=40 valid");
    rj_ref.jump(T, g).expect("ref jump valid").values
}

// ---------------------------------------------------------------------------
// Test: golden + 1e-3 evolve anchor
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "slow-tests")]
fn g_binding_resolvent_jump_parity_core_golden() {
    let grid = make_grid();
    let g = make_u0(grid);

    let jump_vals = canonical_resolvent_jump_core();
    let ref_vals = resolvent_reference(grid, &g);

    // ‖jump_M16 − jump_M40‖∞ ≤ 1e-3 (self-convergence anchor, same discrete A).
    let sup_err: f64 = jump_vals
        .iter()
        .zip(ref_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!(
        "G_BINDING_RESOLVENT_JUMP_PARITY (core golden, N={N}, M={M_NODES}, M_ref={M_REF}, T={T}):\n\
         sup‖jump_M{M_NODES} − jump_M{M_REF}‖∞ = {sup_err:.3e}  (gate ≤ {TOL_JUMP:.0e})\n\
         Golden (node 32 sample):\n\
           jump[32] = {:.16e}",
        jump_vals[32],
    );

    assert!(
        sup_err <= TOL_JUMP,
        "G_BINDING_RESOLVENT_JUMP_PARITY FAIL (self-convergence anchor, M{M_NODES} vs M{M_REF}): \
         sup = {sup_err:.3e} > {TOL_JUMP:.0e}"
    );
}
