//! G20 — `NonSeparable2DChernoff` / `NonSeparable2DAnisotropicChernoff` vs
//! `NonSeparableMixedChernoff::with_scalar_c` / `with_beta` alias-identity gate.
//!
//! Threshold: **0 ULP (byte-equal)** for all three τ values.
//!
//! ## What this tests
//!
//! After the v2.2 ADR-0058 unification, `NonSeparable2DChernoff` and
//! `NonSeparable2DAnisotropicChernoff` are type aliases for
//! `NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>`. The `new` constructor
//! delegates to `with_scalar_c`; the numerics must be byte-identical.
//!
//! Gate (§4 of `contracts/v2.2/wave-c-refactor-bindings.md`):
//! ```text
//! for τ ∈ {0.001, 0.01, 0.1}:
//!   NonSeparable2DChernoff::new(x, y, c, c_bound, grid).apply_into(τ, f)
//!   ==
//!   NonSeparableMixedChernoff::with_scalar_c(x, y, c, c_bound, grid).apply_into(τ, f)
//!   (byte-equal, 0 ULP)
//! ```
//!
//! See ADR-0058 §"Acceptance gates", math.md §10.7-ter Theorem 7-bis, §18.

use semiflow_core::{
    chernoff::ChernoffFunction, scratch::ScratchPool, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    NonSeparable2DAnisotropicChernoff, NonSeparable2DChernoff, NonSeparableMixedChernoff,
};

// Grid size chosen so CFL is satisfied at all three tau values.
// dx = 2/(N-1), dx² = (2/(N-1))².
// CFL: 4*tau*c_norm < dx² (square grid, dx=dy).
// At tau=0.1: 4*0.1*c_norm < (2/127)² ≈ 2.47e-4 → c_norm < 6.2e-4.
// Use c_norm = 1e-4 with N=128 for safe headroom.
const N: usize = 128;
const C_CONST: f64 = 1e-4; // small constant coupling, CFL-safe at tau=0.1
const C_NORM: f64 = 1e-4;
const TAU_VALUES: [f64; 3] = [0.001, 0.01, 0.1];

fn make_grid() -> Grid2D<f64> {
    let g = Grid1D::new(-1.0, 1.0, N).unwrap();
    Grid2D::new(g, g)
}

fn diffusion_inner() -> DiffusionChernoff {
    let gx = Grid1D::new(-1.0, 1.0, N).unwrap();
    DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx)
}

fn make_f0() -> GridFn2D<f64> {
    let grid = make_grid();
    GridFn2D::from_fn(grid, |x, y| (-x * x - y * y).exp())
}

/// Apply `op.apply_into(tau, f)` and return result values.
fn apply_values<C>(op: &C, tau: f64, f: &GridFn2D<f64>) -> Vec<f64>
where
    C: ChernoffFunction<f64, S = GridFn2D<f64>>,
{
    let mut dst = f.clone();
    let mut scratch = ScratchPool::new();
    op.apply_into(tau, f, &mut dst, &mut scratch).unwrap();
    dst.values
}

// ---------------------------------------------------------------------------
// G20a: NonSeparable2DChernoff::new ≡ NonSeparableMixedChernoff::with_scalar_c
// ---------------------------------------------------------------------------

#[test]
fn g20a_scalar_alias_identity_tau_0_001() {
    let grid = make_grid();
    let alias_op = NonSeparable2DChernoff::new(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_CONST,
        C_NORM,
        grid,
    )
    .unwrap();
    let unified_op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_CONST,
        C_NORM,
        grid,
    )
    .unwrap();
    let f = make_f0();
    let alias_vals = apply_values(&alias_op, TAU_VALUES[0], &f);
    let unified_vals = apply_values(&unified_op, TAU_VALUES[0], &f);
    assert_eq!(alias_vals, unified_vals, "G20a: 0 ULP at τ=0.001 (scalar)");
}

#[test]
fn g20a_scalar_alias_identity_tau_0_01() {
    let grid = make_grid();
    let alias_op = NonSeparable2DChernoff::new(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_CONST,
        C_NORM,
        grid,
    )
    .unwrap();
    let unified_op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_CONST,
        C_NORM,
        grid,
    )
    .unwrap();
    let f = make_f0();
    let alias_vals = apply_values(&alias_op, TAU_VALUES[1], &f);
    let unified_vals = apply_values(&unified_op, TAU_VALUES[1], &f);
    assert_eq!(alias_vals, unified_vals, "G20a: 0 ULP at τ=0.01 (scalar)");
}

#[test]
fn g20a_scalar_alias_identity_tau_0_1() {
    let grid = make_grid();
    let alias_op = NonSeparable2DChernoff::new(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_CONST,
        C_NORM,
        grid,
    )
    .unwrap();
    let unified_op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_CONST,
        C_NORM,
        grid,
    )
    .unwrap();
    let f = make_f0();
    let alias_vals = apply_values(&alias_op, TAU_VALUES[2], &f);
    let unified_vals = apply_values(&unified_op, TAU_VALUES[2], &f);
    assert_eq!(alias_vals, unified_vals, "G20a: 0 ULP at τ=0.1 (scalar)");
}

// ---------------------------------------------------------------------------
// G20b: NonSeparable2DAnisotropicChernoff::new ≡ NonSeparableMixedChernoff::with_beta
// ---------------------------------------------------------------------------

// Beta norm also CFL-safe: at tau=0.1, need 4*0.1*beta_norm < 2.47e-4 → beta_norm < 6.2e-4.
const BETA_NORM: f64 = 5e-5;

fn beta_fn(x: f64, _y: f64) -> f64 {
    // Position-dependent coupling; sup-norm = 5e-5 on [-1,1]².
    5e-5 * (-x * x).exp()
}

#[test]
fn g20b_aniso_alias_identity_tau_0_001() {
    let grid = make_grid();
    let alias_op = NonSeparable2DAnisotropicChernoff::new(
        diffusion_inner(),
        diffusion_inner(),
        beta_fn,
        BETA_NORM,
        grid,
    )
    .unwrap();
    let unified_op = NonSeparableMixedChernoff::with_beta(
        diffusion_inner(),
        diffusion_inner(),
        beta_fn,
        BETA_NORM,
        grid,
    )
    .unwrap();
    let f = make_f0();
    let alias_vals = apply_values(&alias_op, TAU_VALUES[0], &f);
    let unified_vals = apply_values(&unified_op, TAU_VALUES[0], &f);
    assert_eq!(alias_vals, unified_vals, "G20b: 0 ULP at τ=0.001 (aniso)");
}

#[test]
fn g20b_aniso_alias_identity_tau_0_01() {
    let grid = make_grid();
    let alias_op = NonSeparable2DAnisotropicChernoff::new(
        diffusion_inner(),
        diffusion_inner(),
        beta_fn,
        BETA_NORM,
        grid,
    )
    .unwrap();
    let unified_op = NonSeparableMixedChernoff::with_beta(
        diffusion_inner(),
        diffusion_inner(),
        beta_fn,
        BETA_NORM,
        grid,
    )
    .unwrap();
    let f = make_f0();
    let alias_vals = apply_values(&alias_op, TAU_VALUES[1], &f);
    let unified_vals = apply_values(&unified_op, TAU_VALUES[1], &f);
    assert_eq!(alias_vals, unified_vals, "G20b: 0 ULP at τ=0.01 (aniso)");
}

#[test]
fn g20b_aniso_alias_identity_tau_0_1() {
    let grid = make_grid();
    let alias_op = NonSeparable2DAnisotropicChernoff::new(
        diffusion_inner(),
        diffusion_inner(),
        beta_fn,
        BETA_NORM,
        grid,
    )
    .unwrap();
    let unified_op = NonSeparableMixedChernoff::with_beta(
        diffusion_inner(),
        diffusion_inner(),
        beta_fn,
        BETA_NORM,
        grid,
    )
    .unwrap();
    let f = make_f0();
    let alias_vals = apply_values(&alias_op, TAU_VALUES[2], &f);
    let unified_vals = apply_values(&unified_op, TAU_VALUES[2], &f);
    assert_eq!(alias_vals, unified_vals, "G20b: 0 ULP at τ=0.1 (aniso)");
}

// ---------------------------------------------------------------------------
// G20c: cross-alias identity: scalar alias ≡ aniso alias on same function ptr
// ---------------------------------------------------------------------------

#[test]
fn g20c_cross_alias_scalar_vs_aniso_tau_0_01() {
    // When both aliases receive the same function pointer (constant fn),
    // they must produce bit-identical output.
    let grid = make_grid();
    let scalar_op = NonSeparable2DChernoff::new(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_CONST,
        C_NORM,
        grid,
    )
    .unwrap();
    let aniso_op = NonSeparable2DAnisotropicChernoff::new(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_CONST,
        C_NORM,
        grid,
    )
    .unwrap();
    let f = make_f0();
    let scalar_vals = apply_values(&scalar_op, TAU_VALUES[1], &f);
    let aniso_vals = apply_values(&aniso_op, TAU_VALUES[1], &f);
    assert_eq!(scalar_vals, aniso_vals, "G20c: cross-alias 0 ULP");
}
