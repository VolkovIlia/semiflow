//! Smoke tests for Generic-over-Float refactor (ADR-0025, v0.9.0 Block D).
//!
//! Verifies that `Grid1D<f32>`, `GridFn1D<f32>`, and all Wave-1/Wave-2/Wave-3
//! types compile and produce plausible results via the scalar `*_generic` code
//! paths.
//!
//! Wave 1 new types: `Diffusion4thChernoff`, `Diffusion6thChernoff`,
//! `TruncatedExpDiffusionChernoff`, `TruncatedExp4thDiffusionChernoff`,
//! `ShiftChernoff1D`, `DriftReactionChernoff`.
//!
//! Wave 2 new types: `Grid2D<f32>`, `GridFn2D<f32>`.
//!
//! Wave 3 new types: `Grid3D<f32>`, `GridFn3D<f32>`.
//!
//! Note: f32 accuracy is looser than f64; tolerances are set accordingly.
//! SIMD stays f64-concrete (ADR-0025 §SIMD carve-out) so f32 paths exercise
//! the pure-scalar fallback only.

#[cfg(not(feature = "parallel"))]
use semiflow_core::nonseparable2d::NonSeparable2DChernoff;
#[cfg(not(feature = "parallel"))]
use semiflow_core::nonseparable2d_aniso::NonSeparable2DAnisotropicChernoff;
#[cfg(not(feature = "parallel"))]
use semiflow_core::strang2d::Strang2D;
#[cfg(not(feature = "parallel"))]
use semiflow_core::strang3d::Strang3D;
use semiflow_core::{
    adaptive::AdaptivePI,
    axis::{Axis, AxisLift},
    boundary::InterpKind,
    diffusion::DiffusionChernoff,
    diffusion4::Diffusion4thChernoff,
    diffusion6::Diffusion6thChernoff,
    drift_reaction::DriftReactionChernoff,
    grid2d::Grid2D,
    grid3d::Grid3D,
    grid_fn2d::GridFn2D,
    grid_fn3d::GridFn3D,
    shift1d::ShiftChernoff1D,
    strang::StrangSplit,
    truncated_exp::TruncatedExpDiffusionChernoff,
    truncated_exp4::TruncatedExp4thDiffusionChernoff,
    BoundaryPolicy, Grid1D, GridFn1D, SemiflowFloat, State,
};

// ---------------------------------------------------------------------------
// Helper — absolute tolerance check for f32
// ---------------------------------------------------------------------------

fn assert_abs_f32(actual: f32, expected: f32, tol: f32, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{label}: |{actual} - {expected}| = {diff} > tol {tol}"
    );
}

// ---------------------------------------------------------------------------
// Wave-4 helper: Wrap<C, F> — thin ChernoffFunction<F> shim for smoke tests.
//
// Leaf types (DiffusionChernoff<F>, DriftReactionChernoff<F>) provide
// `apply_f` / `apply_generic` for non-f64 scalar paths but do NOT expose
// `ChernoffFunction<F>` for F ≠ f64 (to preserve bit-equality on the f64
// SIMD code path per ADR-0025 §SIMD carve-out).
//
// This wrapper bridges the gap for tests ONLY: it satisfies
// `Wrap<C, F>: ChernoffFunction<F>` by delegating to `apply_f` or
// `apply_generic` without touching any production impl.
// ---------------------------------------------------------------------------

use semiflow_core::{
    chernoff::{ApplyChernoffExt, ChernoffFunction, Growth},
    error::SemiflowError,
    scratch::ScratchPool,
};

/// Test-only shim so `DiffusionChernoff<F>` and `DriftReactionChernoff<F>`
/// can be used as `ChernoffFunction<F>` for non-f64 smoke tests.
/// `Copy` removed in v0.12.0 (ADR-0034): `DiffusionChernoff` holds Arc closures.
#[derive(Clone)]
struct WrapDiff<F: SemiflowFloat>(semiflow_core::diffusion::DiffusionChernoff<F>);

impl<F: SemiflowFloat> ChernoffFunction<F> for WrapDiff<F> {
    type S = GridFn1D<F>;
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn1D<F>,
        dst: &mut GridFn1D<F>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let out = self.0.apply_f(tau, src)?;
        dst.values.clear();
        dst.values.extend_from_slice(&out.values);
        Ok(())
    }
    fn order(&self) -> u32 {
        self.0.order_val()
    }
    fn growth(&self) -> Growth<F> {
        Growth::contraction()
    }
}

/// `Copy` removed (v2.3 ADR-0034 ext): `DriftReactionChernoff` now holds
/// `Option<Storage2<F>>` which contains `Arc<dyn Fn>` in the closure path.
#[derive(Clone)]
struct WrapDrift<F: SemiflowFloat>(semiflow_core::drift_reaction::DriftReactionChernoff<F>);

impl<F: SemiflowFloat> ChernoffFunction<F> for WrapDrift<F> {
    type S = GridFn1D<F>;
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn1D<F>,
        dst: &mut GridFn1D<F>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let out = self.0.apply_f(tau, src)?;
        dst.values.clear();
        dst.values.extend_from_slice(&out.values);
        Ok(())
    }
    fn order(&self) -> u32 {
        self.0.order_val()
    }
    fn growth(&self) -> Growth<F> {
        Growth::contraction()
    }
}

// ---------------------------------------------------------------------------
// Grid1D<f32> smoke
// ---------------------------------------------------------------------------

#[test]
fn grid1d_f32_new_generic() {
    // Construct a 1-D grid over [-5, 5] with 100 nodes using f32 scalars.
    let grid =
        Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 100).expect("Grid1D::<f32>::new_generic");

    assert_eq!(grid.n, 100);
    assert_abs_f32(grid.xmin, -5.0_f32, 1e-7, "xmin");
    assert_abs_f32(grid.xmax, 5.0_f32, 1e-7, "xmax");

    let dx = grid.dx();
    // dx = (xmax - xmin) / (n - 1) = 10 / 99 ≈ 0.10101…
    let expected_dx = 10.0_f32 / 99.0_f32;
    assert_abs_f32(dx, expected_dx, 1e-6, "dx");

    // x_at(0) == xmin, x_at(n-1) == xmax.
    assert_abs_f32(grid.x_at(0), -5.0_f32, 1e-7, "x_at(0)");
    assert_abs_f32(grid.x_at(99), 5.0_f32, 1e-6, "x_at(n-1)");
}

#[test]
fn grid1d_f32_builder_methods() {
    let grid = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 50)
        .expect("grid")
        .with_boundary(BoundaryPolicy::ZeroExtend);

    assert_eq!(grid.n, 50);
    assert!(matches!(grid.boundary, BoundaryPolicy::ZeroExtend));
}

// ---------------------------------------------------------------------------
// GridFn1D<f32> smoke
// ---------------------------------------------------------------------------

#[test]
fn grid_fn1d_f32_smoke() {
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 101).expect("grid");

    // Sample the Gaussian g(x) = exp(-x²).
    let gfn = GridFn1D::from_fn_generic(grid, |x: f32| (-x * x).exp());

    // At x = 0 (midpoint of [-5, 5] for odd n), g(0) = 1.0.
    // The midpoint index is 50.
    let mid = gfn.values[50];
    assert_abs_f32(mid, 1.0_f32, 1e-5, "g(0) ≈ 1");

    // At x = ±5, g(x) = exp(-25) ≈ 1.4e-11 — essentially 0 in f32.
    assert!(gfn.values[0] < 1e-9_f32, "g(-5) near zero");
    assert!(gfn.values[100] < 1e-9_f32, "g(5) near zero");
}

#[test]
fn grid_fn1d_f32_sample_generic() {
    // sample_generic routes through interp_generic (SepticHermite default, v8.0+
    // §46.5.bis); checks at interior points with generous f32 tolerance.
    let grid = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 101).expect("grid");

    // f(x) = x² → smooth, polynomial-representable.
    let gfn = GridFn1D::from_fn_generic(grid, |x: f32| x * x);

    // Sample at x = 0.0 — should be ≈ 0.
    let v = gfn.sample_generic(0.0_f32).expect("sample_generic");
    assert_abs_f32(v, 0.0_f32, 1e-4, "x²(0)=0");

    // Sample at x = 0.5 — should be ≈ 0.25.
    let v2 = gfn.sample_generic(0.5_f32).expect("sample_generic");
    assert_abs_f32(v2, 0.25_f32, 1e-4, "x²(0.5)=0.25");
}

// ---------------------------------------------------------------------------
// DiffusionChernoff<f32> smoke
// ---------------------------------------------------------------------------

#[test]
fn diffusion_f32_apply_f_runs() {
    // Build a simple constant-a diffusion on f32 grid.
    // Constant a(x) = 1.0, a'(x) = 0, a''(x) = 0.
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 51).expect("grid");

    let dc = DiffusionChernoff::<f32>::new(
        |_: f32| 1.0_f32,
        |_: f32| 0.0_f32,
        |_: f32| 0.0_f32,
        1.0_f64, // a_norm_bound stays f64 (growth_bound interface)
        grid,
    );

    // Initial condition: Gaussian exp(-x²).
    let f0 = GridFn1D::from_fn_generic(grid, |x: f32| (-x * x).exp());

    let tau = 0.01_f32;
    let f1 = dc.apply_f(tau, &f0).expect("apply_f");

    // After one step the norm should be ≤ initial norm (contractivity for
    // constant a=1 and small τ).
    let norm0: f32 = f0.values.iter().copied().fold(0.0_f32, f32::max);
    let norm1: f32 = f1
        .values
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    assert!(
        norm1 <= norm0 * 1.05_f32,
        "contractivity violated: norm1={norm1} > norm0={norm0}*1.05"
    );

    // The grid geometry must be preserved.
    assert_eq!(f1.values.len(), f0.values.len());
}

#[test]
fn diffusion_f32_matches_f64_within_tol() {
    // Run DiffusionChernoff<f32> and DiffusionChernoff<f64> on the same
    // problem and verify that f32 results are within a generous f32-level
    // tolerance of the f64 reference.
    //
    // Tolerance is set to 5e-3 (relative) — f32 has ~6–7 decimal digits;
    // one step of diffusion introduces O(τ²) truncation, so single-step
    // errors at τ=0.01 are negligible compared to f32 precision.

    // Both grids pinned to CubicHermite so the f32/f64 comparison isolates
    // floating-point precision, not interpolation-order differences.
    // v8.0+ §46.5.bis changed the new_generic default to SepticHermite; pin
    // here explicitly so this cross-precision smoke test stays stable.
    let grid64 = Grid1D::new(-5.0_f64, 5.0_f64, 51)
        .expect("grid64")
        .with_interp(InterpKind::CubicHermite);
    let dc64 = DiffusionChernoff::<f64>::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        1.0_f64,
        grid64,
    );
    let f0_64 = GridFn1D::from_fn(grid64, |x: f64| (-x * x).exp());
    let f1_64 = dc64.apply_f(0.01_f64, &f0_64).expect("apply_f f64");

    // f32 run: pin to CubicHermite to match grid64 interpolant.
    let grid32 = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 51)
        .expect("grid32")
        .with_interp(InterpKind::CubicHermite);
    let dc32 = DiffusionChernoff::<f32>::new(
        |_: f32| 1.0_f32,
        |_: f32| 0.0_f32,
        |_: f32| 0.0_f32,
        1.0_f64,
        grid32,
    );
    let f0_32 = GridFn1D::from_fn_generic(grid32, |x: f32| (-x * x).exp());
    let f1_32 = dc32.apply_f(0.01_f32, &f0_32).expect("apply_f f32");

    assert_eq!(f1_64.values.len(), f1_32.values.len());

    // Compare element-wise.
    let tol = 5e-3_f64;
    for (i, (&v64, &v32)) in f1_64.values.iter().zip(f1_32.values.iter()).enumerate() {
        let v32_as_f64 = f64::from(v32);
        let diff = (v64 - v32_as_f64).abs();
        let scale = v64.abs().max(1e-10); // relative scale
        assert!(
            diff / scale <= tol,
            "index {i}: f64={v64:.8e} f32={v32:.8e} rel_diff={:.2e} > tol {tol}",
            diff / scale
        );
    }
}

// ---------------------------------------------------------------------------
// Trait-object smoke — verify SemiflowFloat is usable as a bound
// ---------------------------------------------------------------------------

fn compute_sup_norm<F: SemiflowFloat>(values: &[F]) -> F {
    values.iter().fold(F::zero(), |acc, &v| {
        let abs_v = <F as num_traits::Float>::abs(v);
        if abs_v > acc {
            abs_v
        } else {
            acc
        }
    })
}

#[test]
fn remote_float_bound_f32() {
    let v: Vec<f32> = vec![0.1, -0.5, 0.3, -1.2_f32];
    let norm = compute_sup_norm(&v);
    assert_abs_f32(norm, 1.2_f32, 1e-7, "sup norm f32");
}

#[test]
fn remote_float_bound_f64() {
    let v: Vec<f64> = vec![0.1, -0.5, 0.3, -1.2_f64];
    let norm = compute_sup_norm(&v);
    let diff = (norm - 1.2_f64).abs();
    assert!(diff <= 1e-14_f64, "sup norm f64: |{norm} - 1.2| = {diff}");
}

// ---------------------------------------------------------------------------
// Wave-1 smoke: Diffusion4thChernoff<f32>
// ---------------------------------------------------------------------------

#[test]
fn diffusion4_f32_smoke() {
    // Constant-a 4th-order diffusion on f32.
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 51).expect("grid");
    let dc = Diffusion4thChernoff::<f32>::new_generic(
        |_: f32| 1.0_f32, // a(x) = 1
        |_: f32| 0.0_f32, // a'(x) = 0
        |_: f32| 0.0_f32, // a''(x) = 0
        1.0_f64,          // a_norm_bound
        grid,
    );
    let f0 = GridFn1D::from_fn_generic(grid, |x: f32| (-x * x).exp());

    // CFL for 4th-order: 8·τ·a_norm < 3·dx²  =>  τ < 3·dx²/8
    let dx = grid.dx();
    let tau = 0.3_f32 * dx * dx;

    let f1 = dc.apply_f(tau, &f0).expect("apply_f D4 f32");

    // Output must have same shape.
    assert_eq!(f1.values.len(), f0.values.len());

    // Sup-norm should not blow up for small τ on constant-a diffusion.
    let norm0 = compute_sup_norm(&f0.values);
    let norm1 = compute_sup_norm(&f1.values);
    assert!(
        norm1 <= norm0 * 1.1_f32,
        "D4 f32: norm1={norm1} > 1.1 * norm0={norm0}"
    );
}

// ---------------------------------------------------------------------------
// Wave-1 smoke: Diffusion6thChernoff<f32>
// ---------------------------------------------------------------------------

#[test]
fn diffusion6_f32_smoke() {
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 51).expect("grid");
    let dc = Diffusion6thChernoff::<f32>::new_generic(
        |_: f32| 1.0_f32,
        |_: f32| 0.0_f32,
        |_: f32| 0.0_f32,
        1.0_f64,
        grid,
    );
    let f0 = GridFn1D::from_fn_generic(grid, |x: f32| (-x * x).exp());

    let dx = grid.dx();
    let tau = 0.1_f32 * dx * dx;

    let f1 = dc.apply_f(tau, &f0).expect("apply_f D6 f32");

    assert_eq!(f1.values.len(), f0.values.len());

    let norm0 = compute_sup_norm(&f0.values);
    let norm1 = compute_sup_norm(&f1.values);
    assert!(
        norm1 <= norm0 * 1.1_f32,
        "D6 f32: norm1={norm1} > 1.1 * norm0={norm0}"
    );
}

// ---------------------------------------------------------------------------
// Wave-1 smoke: TruncatedExpDiffusionChernoff<f32>
// ---------------------------------------------------------------------------

#[test]
fn truncated_exp_f32_smoke() {
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 51).expect("grid");
    let dc = TruncatedExpDiffusionChernoff::<f32>::new_generic(
        |_: f32| 1.0_f32,
        |_: f32| 0.0_f32,
        |_: f32| 0.0_f32,
        1.0_f64,
        grid,
    );
    let f0 = GridFn1D::from_fn_generic(grid, |x: f32| (-x * x).exp());

    // CFL for TruncatedExp: 2·τ·a_norm < dx²  =>  τ < dx²/2
    let dx = grid.dx();
    let tau = 0.4_f32 * dx * dx;

    let f1 = dc.apply_f(tau, &f0).expect("apply_f TExp f32");

    assert_eq!(f1.values.len(), f0.values.len());

    let norm0 = compute_sup_norm(&f0.values);
    let norm1 = compute_sup_norm(&f1.values);
    assert!(
        norm1 <= norm0 * 1.1_f32,
        "TExp f32: norm1={norm1} > 1.1 * norm0={norm0}"
    );
}

// ---------------------------------------------------------------------------
// Wave-1 smoke: TruncatedExp4thDiffusionChernoff<f32>
// ---------------------------------------------------------------------------

#[test]
fn truncated_exp4_f32_smoke() {
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 51).expect("grid");
    let dc = TruncatedExp4thDiffusionChernoff::<f32>::new_generic(
        |_: f32| 1.0_f32,
        |_: f32| 0.0_f32,
        |_: f32| 0.0_f32,
        1.0_f64,
        grid,
    );
    let f0 = GridFn1D::from_fn_generic(grid, |x: f32| (-x * x).exp());

    // CFL for TruncatedExp4th: 8·τ·a_norm < 3·dx²  =>  τ < 3·dx²/8
    let dx = grid.dx();
    let tau = 0.3_f32 * dx * dx;

    let f1 = dc.apply_f(tau, &f0).expect("apply_f TExp4 f32");

    assert_eq!(f1.values.len(), f0.values.len());

    let norm0 = compute_sup_norm(&f0.values);
    let norm1 = compute_sup_norm(&f1.values);
    assert!(
        norm1 <= norm0 * 1.1_f32,
        "TExp4 f32: norm1={norm1} > 1.1 * norm0={norm0}"
    );
}

// ---------------------------------------------------------------------------
// Wave-1 smoke: ShiftChernoff1D<f32>
// ---------------------------------------------------------------------------

#[test]
fn shift_f32_smoke() {
    // Pure transport: a(x) = 0.01, b(x) = 0, c(x) = 0.
    // With a > 0 (Theorem 6 strict ellipticity) and b = c = 0,
    // the Shift formula reduces to a symmetric three-point average.
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, 101).expect("grid");
    let sc = ShiftChernoff1D::<f32>::new_generic(
        |_: f32| 0.01_f32, // a(x) > 0
        |_: f32| 0.0_f32,  // b(x) = 0
        |_: f32| 0.0_f32,  // c(x) = 0
        0.0_f64,           // c_norm_bound = 0 (no reaction)
        grid,
    );
    let f0 = GridFn1D::from_fn_generic(grid, |x: f32| (-x * x).exp());

    let tau = 0.1_f32;
    let f1 = sc.apply_f(tau, &f0).expect("apply_f Shift f32");

    assert_eq!(f1.values.len(), f0.values.len());

    // Output should be non-negative (Gaussian initial data, positive coefficients).
    for (i, &v) in f1.values.iter().enumerate() {
        assert!(v >= -1e-5_f32, "Shift f32: negative at i={i}: v={v}");
    }
}

// ---------------------------------------------------------------------------
// Wave-1 smoke: DriftReactionChernoff<f32>
// ---------------------------------------------------------------------------

#[test]
fn drift_reaction_f32_smoke() {
    // Pure reaction (b = 0, c = 0): should return f unchanged at τ → 0.
    let grid = Grid1D::<f32>::new_generic(-3.0_f32, 3.0_f32, 61).expect("grid");
    let dr = DriftReactionChernoff::<f32>::new_generic(
        |_: f32| 0.0_f32, // b(x) = 0 (no drift)
        |_: f32| 0.0_f32, // c(x) = 0 (no reaction)
        0.0_f64,          // c_norm_bound = 0
        grid,
    );
    let f0 = GridFn1D::from_fn_generic(grid, |x: f32| (-x * x).exp());

    let tau = 1e-4_f32;
    let f1 = dr.apply_f(tau, &f0).expect("apply_f DriftReaction f32");

    assert_eq!(f1.values.len(), f0.values.len());

    // With b = c = 0, apply should return f almost unchanged.
    let max_diff = f0
        .values
        .iter()
        .zip(f1.values.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-3_f32,
        "DriftReaction f32: max diff with b=c=0 is {max_diff} at tau={tau}"
    );
}

// ---------------------------------------------------------------------------
// Wave-1 smoke: AdaptivePI<DiffusionChernoff<f32>> (composition smoke)
// ---------------------------------------------------------------------------

#[test]
fn adaptive_pi_f32_smoke() {
    // Compose AdaptivePI around DiffusionChernoff<f32> to verify that the
    // generic bound C: ChernoffFunction compiles through the f32 path.
    //
    // Note: AdaptivePI::evolve_adaptive takes t: f64 and C::S = GridFn1D<f64>
    // because AdaptivePI is bounded by ChernoffFunction which is f64-monomorphic.
    // The "f32" here refers only to DiffusionChernoff's float type F = f32;
    // the trait interface still works through the f64 ChernoffFunction impl.
    //
    // For types without a ChernoffFunction impl (non-f64 F), AdaptivePI cannot
    // wrap them directly — that would require extending ChernoffFunction to be
    // generic itself (Wave 2 scope).
    //
    // This test therefore uses DiffusionChernoff<f64> (pilot type) inside
    // AdaptivePI, purely to verify the composition compiles and runs.
    let grid = Grid1D::new(-3.0_f64, 3.0_f64, 41).expect("grid");
    let dc = DiffusionChernoff::<f64>::new(
        |_: f64| 0.5_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        0.5_f64,
        grid,
    );
    let mut pi = AdaptivePI::new(dc);
    let f0 = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());

    let outcome = pi.evolve_adaptive(0.05_f64, &f0).expect("evolve_adaptive");
    let f1 = outcome.final_state;

    assert_eq!(f1.values.len(), f0.values.len());

    // Sup-norm should not blow up.
    let norm0: f64 = f0.values.iter().copied().fold(0.0_f64, f64::max);
    let norm1: f64 = f1
        .values
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    assert!(
        norm1 <= norm0 * 1.05_f64,
        "AdaptivePI f64: norm1={norm1} > norm0={norm0}*1.05"
    );
}

// ---------------------------------------------------------------------------
// Wave-2 smoke: Grid2D<f32> and GridFn2D<f32>
// ---------------------------------------------------------------------------

#[test]
fn grid2d_f32_smoke() {
    // Verify that Grid2D<f32> constructs correctly and geometry methods work.
    let gx = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).expect("gx");
    let gy = Grid1D::<f32>::new_generic(-2.0_f32, 2.0_f32, 6).expect("gy");
    let g = Grid2D::<f32>::new(gx, gy);

    assert_eq!(g.nx(), 8);
    assert_eq!(g.ny(), 6);
    assert_eq!(g.len(), 48);

    // PartialEq round-trip.
    let ax = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).expect("ax");
    let ay = Grid1D::<f32>::new_generic(-2.0_f32, 2.0_f32, 6).expect("ay");
    let clone = Grid2D::<f32>::new(ax, ay);
    assert_eq!(g, clone, "Grid2D<f32> PartialEq");
}

#[test]
fn grid_fn2d_f32_smoke() {
    // Verify that GridFn2D<f32> constructs and State<f32> ops work correctly.
    let gx = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).expect("gx");
    let gy = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 6).expect("gy");
    let g = Grid2D::<f32>::new(gx, gy);

    // from_fn_generic: sample Gaussian.
    let f = GridFn2D::<f32>::from_fn_generic(g, |x, y| (-x * x - y * y).exp());
    assert_eq!(f.values.len(), 48);
    // All values in (0, 1].
    for &v in &f.values {
        assert!(v > 0.0_f32 && v <= 1.0_f32, "value out of range: {v}");
    }

    // State ops: axpy, scale, norm_sup, zeroed_like.
    let ones = GridFn2D::<f32>::from_fn_generic(g, |_, _| 1.0_f32);
    let mut out = GridFn2D::<f32>::from_fn_generic(g, |_, _| 2.0_f32);
    out.axpy(3.0_f32, &ones);
    // All values are 5.0; norm_sup = 5.0.
    assert_abs_f32(out.norm_sup(), 5.0_f32, 1e-6, "axpy norm_sup");

    out.scale(2.0_f32);
    assert_abs_f32(out.norm_sup(), 10.0_f32, 1e-5, "scale norm_sup");

    let z = out.zeroed_like();
    assert_abs_f32(z.norm_sup(), 0.0_f32, f32::EPSILON, "zeroed_like");

    // row_generic / col_generic / write_row_generic / write_col_generic.
    let row0 = f.row_generic(0);
    assert_eq!(row0.values.len(), 8);

    let col0 = f.col_generic(0);
    assert_eq!(col0.values.len(), 6);

    // new_generic with wrong length returns error.
    let err = GridFn2D::<f32>::new_generic(g, vec![0.0_f32; 3]).unwrap_err();
    assert!(
        matches!(err, semiflow_core::SemiflowError::DomainViolation { .. }),
        "expected DomainViolation, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Wave-3 smoke: Grid3D<f32>
// ---------------------------------------------------------------------------

#[test]
fn grid3d_f32_smoke() {
    // Verify Grid3D<f32> geometry methods and PartialEq.
    let gx = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).expect("gx");
    let gy = Grid1D::<f32>::new_generic(-2.0_f32, 2.0_f32, 6).expect("gy");
    let gz = Grid1D::<f32>::new_generic(-3.0_f32, 3.0_f32, 4).expect("gz");
    let g = Grid3D::<f32>::new_generic(gx, gy, gz).expect("Grid3D f32");

    assert_eq!(g.nx(), 8);
    assert_eq!(g.ny(), 6);
    assert_eq!(g.nz(), 4);
    assert_eq!(g.len(), 8 * 6 * 4);

    // PartialEq round-trip.
    let ax = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).expect("ax");
    let ay = Grid1D::<f32>::new_generic(-2.0_f32, 2.0_f32, 6).expect("ay");
    let az = Grid1D::<f32>::new_generic(-3.0_f32, 3.0_f32, 4).expect("az");
    let clone = Grid3D::<f32>::new_generic(ax, ay, az).expect("clone");
    assert_eq!(g, clone, "Grid3D<f32> PartialEq");
}

// ---------------------------------------------------------------------------
// Wave-3 smoke: GridFn3D<f32>
// ---------------------------------------------------------------------------

#[test]
fn grid_fn3d_f32_smoke() {
    // Verify GridFn3D<f32> construction and State<f32> ops.
    let gx = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).expect("gx");
    let gy = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 6).expect("gy");
    let gz = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 4).expect("gz");
    let g = Grid3D::<f32>::new_generic(gx, gy, gz).expect("Grid3D f32");

    // from_fn_generic: sample Gaussian exp(-x²-y²-z²).
    let f = GridFn3D::<f32>::from_fn_generic(g, |x, y, z| (-x * x - y * y - z * z).exp());
    assert_eq!(f.values.len(), 8 * 6 * 4);
    // All values in (0, 1].
    for &v in &f.values {
        assert!(v > 0.0_f32 && v <= 1.0_f32, "value out of range: {v}");
    }

    // State<f32> ops: axpy, scale, norm_sup, zeroed_like.
    let ones = GridFn3D::<f32>::from_fn_generic(g, |_, _, _| 1.0_f32);
    let mut out = GridFn3D::<f32>::from_fn_generic(g, |_, _, _| 2.0_f32);
    out.axpy(3.0_f32, &ones);
    assert_abs_f32(out.norm_sup(), 5.0_f32, 1e-6, "axpy norm_sup 3D");

    out.scale(2.0_f32);
    assert_abs_f32(out.norm_sup(), 10.0_f32, 1e-5, "scale norm_sup 3D");

    let z = out.zeroed_like();
    assert_abs_f32(z.norm_sup(), 0.0_f32, f32::EPSILON, "zeroed_like 3D");

    // new_generic with wrong length returns DomainViolation.
    let err = GridFn3D::<f32>::new_generic(g, vec![0.0_f32; 3]).unwrap_err();
    assert!(
        matches!(err, semiflow_core::SemiflowError::DomainViolation { .. }),
        "expected DomainViolation, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Wave 4 — composition types over f32
// ---------------------------------------------------------------------------

/// `AxisLift<WrapDiff<f32>, f32>` compiles and produces finite values.
#[test]
fn axis_lift_f32_smoke() {
    let gx = Grid1D::<f32>::new_generic(-3.0_f32, 3.0_f32, 16).unwrap();
    let gy = Grid1D::<f32>::new_generic(-3.0_f32, 3.0_f32, 12).unwrap();
    let g2 = Grid2D::<f32>::new(gx, gy);
    let f = GridFn2D::<f32>::from_fn_generic(g2, |x, _y| (-x * x).exp());
    let diff = DiffusionChernoff::<f32>::new(|_| 0.5_f32, |_| 0.0_f32, |_| 0.0_f32, 0.5, gx);
    let lift = AxisLift::<_, f32>::new(WrapDiff(diff), Axis::X);
    let out = lift.apply_chernoff(0.01_f32, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
    assert!(out.values.iter().all(|v| v.is_finite()));
}

/// `Strang2D<WrapDiff<f32>, WrapDiff<f32>, f32>` compiles and runs (serial path only).
///
/// Under `parallel` feature, `Strang2D` is f64-only (SIMD path, ADR-0018).
/// This test is gated on `not(feature = "parallel")` accordingly.
#[cfg(not(feature = "parallel"))]
#[test]
fn strang2d_f32_smoke() {
    let gx = Grid1D::<f32>::new_generic(-3.0_f32, 3.0_f32, 12).unwrap();
    let gy = Grid1D::<f32>::new_generic(-3.0_f32, 3.0_f32, 10).unwrap();
    let g2 = Grid2D::<f32>::new(gx, gy);
    let f = GridFn2D::<f32>::from_fn_generic(g2, |x, y| (-x * x - y * y).exp());
    let dx = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.25_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.25,
        gx,
    ));
    let dy = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.25_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.25,
        gy,
    ));
    let s = Strang2D::<_, _, f32>::new(dx, dy);
    let out = s.apply_chernoff(0.01_f32, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
    assert!(out.values.iter().all(|v| v.is_finite()));
    assert_eq!(s.order(), 2);
}

/// `NonSeparable2DChernoff<WrapDiff<f32>, .., f32>` zero-coupling fast path compiles and runs.
///
/// Serial path only — under `parallel` feature, `NonSeparable2DChernoff` is f64-only.
#[cfg(not(feature = "parallel"))]
#[test]
fn nonseparable2d_f32_smoke() {
    let gx = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).unwrap();
    let gy = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).unwrap();
    let g2 = Grid2D::<f32>::new(gx, gy);
    let f = GridFn2D::<f32>::from_fn_generic(g2, |x, _y| x * 0.5_f32);
    let ix = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.5_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.5,
        gx,
    ));
    let iy = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.5_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.5,
        gy,
    ));
    let op = NonSeparable2DChernoff::<_, _, f32>::new(ix, iy, |_, _| 0.0_f32, 0.0, g2).unwrap();
    let out = op.apply_chernoff(0.005_f32, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
    assert!(out.values.iter().all(|v| v.is_finite()));
    assert_eq!(op.order(), 2);
}

/// `NonSeparable2DAnisotropicChernoff<WrapDiff<f32>, .., f32>` zero-beta fast path compiles and runs.
///
/// Serial path only — under `parallel` feature, `NonSeparable2DAnisotropicChernoff` is f64-only.
#[cfg(not(feature = "parallel"))]
#[test]
fn nonseparable2d_aniso_f32_smoke() {
    let gx = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).unwrap();
    let gy = Grid1D::<f32>::new_generic(-1.0_f32, 1.0_f32, 8).unwrap();
    let g2 = Grid2D::<f32>::new(gx, gy);
    let f = GridFn2D::<f32>::from_fn_generic(g2, |x, _y| x * 0.5_f32);
    let ix = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.5_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.5,
        gx,
    ));
    let iy = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.5_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.5,
        gy,
    ));
    let op = NonSeparable2DAnisotropicChernoff::<_, _, f32>::new(ix, iy, |_, _| 0.0_f32, 0.0, g2)
        .unwrap();
    let out = op.apply_chernoff(0.005_f32, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
    assert!(out.values.iter().all(|v| v.is_finite()));
    assert_eq!(op.order(), 2);
}

/// `Strang3D<WrapDiff<f32>, .., f32>` compiles and runs (serial path only).
///
/// Under `parallel` feature, `Strang3D` is f64-only (parallel path, mirrors ADR-0018).
/// This test is gated on `not(feature = "parallel")` accordingly.
#[cfg(not(feature = "parallel"))]
#[test]
fn strang3d_f32_smoke() {
    let gx = Grid1D::<f32>::new_generic(-2.0_f32, 2.0_f32, 8).unwrap();
    let gy = Grid1D::<f32>::new_generic(-2.0_f32, 2.0_f32, 6).unwrap();
    let gz = Grid1D::<f32>::new_generic(-2.0_f32, 2.0_f32, 6).unwrap();
    let g3 = Grid3D::<f32>::new_generic(gx, gy, gz).unwrap();
    let f = GridFn3D::<f32>::from_fn_generic(g3, |x, y, z| (-x * x - y * y - z * z).exp());
    let dx = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.25_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.25,
        gx,
    ));
    let dy = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.25_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.25,
        gy,
    ));
    let dz = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.25_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.25,
        gz,
    ));
    let s = Strang3D::<_, _, _, f32>::new(dx, dy, dz);
    let out = s.apply_chernoff(0.005_f32, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
    assert!(out.values.iter().all(|v| v.is_finite()));
    assert_eq!(s.order(), 2);
}

/// `StrangSplit<WrapDiff<f32>, WrapDrift<f32>, f32>` compiles and runs.
#[test]
fn strang_split_f32_smoke() {
    let g = Grid1D::<f32>::new_generic(-4.0_f32, 4.0_f32, 20).unwrap();
    let f = GridFn1D::<f32>::from_fn_generic(g, |x| (-x * x).exp());
    let diff = WrapDiff(DiffusionChernoff::<f32>::new(
        |_| 0.5_f32,
        |_| 0.0_f32,
        |_| 0.0_f32,
        0.5,
        g,
    ));
    let drift = WrapDrift(DriftReactionChernoff::<f32>::new_generic(
        |_| 0.1_f32,
        |_| 0.0_f32,
        0.0,
        g,
    ));
    let s = StrangSplit::<_, _, f32>::new(diff, drift);
    let out = s.apply_chernoff(0.01_f32, &f).unwrap();
    assert_eq!(out.values.len(), f.values.len());
    assert!(out.values.iter().all(|v| v.is_finite()));
    assert_eq!(s.order(), 2);
}

/// `AdaptivePI<StrangSplit<DiffusionChernoff<f64>, DriftReactionChernoff<f64>>>` f64 path
/// (`AdaptivePI` stays f64-only per ADR-0025 pragmatic decision).
#[test]
fn adaptive_pi_strang_split_f64_smoke() {
    // AdaptivePI is f64-only. This test verifies the f64 path still works
    // after the Wave 4 generic refactor.
    use semiflow_core::grid::Grid1D as G1;
    let g = G1::new(-4.0_f64, 4.0_f64, 32).unwrap();
    let f = GridFn1D::<f64>::from_fn(g, |x| (-x * x).exp());
    let diff = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, g);
    let drift = DriftReactionChernoff::new(|_| 0.0_f64, |_| 0.0_f64, 0.0, g);
    let s = StrangSplit::new(diff, drift);
    let mut pi = AdaptivePI::new(s);
    let outcome = pi.evolve_adaptive(0.1, &f).unwrap();
    assert!(outcome.final_state.values.iter().all(|v| v.is_finite()));
    assert!(outcome.steps_accepted > 0);
}
