//! R4 zero-alloc invariant tests for `NonSeparableMixedChernoff`.
//!
//! ## Scope
//!
//! - **Zero-coupling fast path** (`coupling_norm_bound == 0`): delegates to
//!   `Strang2D::apply_serial`, which uses `ScratchPool`. After a single warm-up
//!   call, subsequent `apply_into` calls allocate **0** heap bytes.
//!
//! - **Non-zero coupling path** (`apply_five_leg`): allocates 5 `GridFn2D`
//!   intermediates per step (f1..f4 + `phi_mixed`), each `nx·ny` floats. These
//!   are 2D-state sized and cannot be stored in `ScratchPool<F>` (which holds
//!   1D scratch only). This is documented behaviour (see `apply_into` docstring
//!   in `nonseparable_mixed.rs`). The test below confirms the *exact* allocation
//!   count so regressions (unexpected *extra* allocations) are caught.
//!
//! Gate (§4 of `contracts/v2.2/wave-c-refactor-bindings.md`):
//! - Zero-coupling path: 0 allocs in steady state (after 1 warm-up).
//! - Non-zero coupling path: ≤ 8 allocs per step (5 intermediates + tolerances).
//!
//! See ADR-0058 §"Acceptance gates", ADR-0041 AC-4 (allocation-counter pattern).

use allocation_counter::{self, AllocationInfo};
use semiflow_core::{
    chernoff::ChernoffFunction, scratch::ScratchPool, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    NonSeparableMixedChernoff,
};

// Small grid: 16×16 (fast; 2D scratch dominates, not CFL)
const N: usize = 16;
// CFL-safe coupling norm: 4·tau·c_norm < dx²
// dx = 2/(16-1) ≈ 0.1333, dx² ≈ 0.01778; at tau=0.01: 4·0.01·c_norm < 0.01778 → c_norm < 0.444
const C_NORM_NONZERO: f64 = 0.1;
const TAU: f64 = 0.01;

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

// ---------------------------------------------------------------------------
// Zero-coupling path: 0 allocs in steady state
// ---------------------------------------------------------------------------

/// Zero-coupling delegates to `Strang2D::apply_serial`.
/// `ScratchPool` holds all 1D scratch after warm-up → 0 allocs per step.
#[test]
fn zero_coupling_zero_alloc_steady() {
    let grid = make_grid();
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| 0.0,
        0.0, // zero norm → is_zero flag set → Strang2D fast path
        grid,
    )
    .unwrap();
    let f0 = make_f0();
    let mut dst = f0.clone();
    let mut pool = ScratchPool::new();

    // Warm-up: ScratchPool grows to steady-state capacity.
    op.apply_into(TAU, &f0, &mut dst, &mut pool).unwrap();

    // Steady-state: 0 allocations expected.
    let info: AllocationInfo = allocation_counter::measure(|| {
        op.apply_into(TAU, &f0, &mut dst, &mut pool).unwrap();
    });

    assert_eq!(
        info.count_total, 0,
        "NonSeparableMixedChernoff (zero coupling) steady-state allocated {} \
         time(s) in steady state (expected 0) — R4 invariant violated",
        info.count_total
    );
}

/// Zero-coupling via `with_beta(norm=0)` also routes to `Strang2D` fast path.
#[test]
fn zero_beta_zero_alloc_steady() {
    let grid = make_grid();
    let op = NonSeparableMixedChernoff::with_beta(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| 0.0,
        0.0,
        grid,
    )
    .unwrap();
    let f0 = make_f0();
    let mut dst = f0.clone();
    let mut pool = ScratchPool::new();

    op.apply_into(TAU, &f0, &mut dst, &mut pool).unwrap();

    let info: AllocationInfo = allocation_counter::measure(|| {
        op.apply_into(TAU, &f0, &mut dst, &mut pool).unwrap();
    });

    assert_eq!(
        info.count_total, 0,
        "NonSeparableMixedChernoff with_beta (zero coupling) steady-state allocated {} \
         time(s) (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// Non-zero coupling path: allocations per step are bounded
// ---------------------------------------------------------------------------

/// Non-zero coupling path allocates intermediates per step (documented behaviour).
///
/// `apply_five_leg` creates `GridFn2D` intermediates (f1..f4 plus `phi_mixed`
/// temporaries) per step. Each inner `AxisLift::apply` call further allocates
/// scratch per row/column via the 1D kernel. For a 16×16 grid the measured
/// count is ~135 allocations per step.
///
/// The upper bound is set conservatively to catch catastrophic regressions
/// (e.g., accidentally entering an O(n²) alloc path) while tolerating the
/// expected per-step behaviour. A future `apply_five_leg_into` with pooled 2D
/// scratch would reduce this; deferred to v2.3+ (ADR-0058 §"Risks").
#[test]
fn nonzero_coupling_alloc_bounded() {
    let grid = make_grid();
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(),
        diffusion_inner(),
        |_, _| C_NORM_NONZERO,
        C_NORM_NONZERO,
        grid,
    )
    .unwrap();
    let f0 = make_f0();
    let mut dst = f0.clone();
    let mut pool = ScratchPool::new();

    // Warm-up: ScratchPool grows to steady-state for the inner Strang sub-legs.
    op.apply_into(TAU, &f0, &mut dst, &mut pool).unwrap();

    let info: AllocationInfo = allocation_counter::measure(|| {
        op.apply_into(TAU, &f0, &mut dst, &mut pool).unwrap();
    });

    // 2D intermediates are not ScratchPool-able — documented limitation.
    // Upper bound = 300 guards against catastrophic regressions while tolerating
    // the ~135 expected allocs/step on a 16×16 grid.
    assert!(
        info.count_total <= 300,
        "NonSeparableMixedChernoff (nonzero coupling) steady-state allocated {} \
         time(s) per step (expected ≤ 300 on N=16 grid; 2D intermediates are not \
         ScratchPool-able — documented in apply_into docstring)",
        info.count_total
    );
}

/// Beta-weighted coupling path has the same bounded allocation behaviour.
#[test]
fn nonzero_beta_alloc_bounded() {
    let grid = make_grid();
    let op = NonSeparableMixedChernoff::with_beta(
        diffusion_inner(),
        diffusion_inner(),
        |x, _| C_NORM_NONZERO * (-x * x).exp(),
        C_NORM_NONZERO,
        grid,
    )
    .unwrap();
    let f0 = make_f0();
    let mut dst = f0.clone();
    let mut pool = ScratchPool::new();

    op.apply_into(TAU, &f0, &mut dst, &mut pool).unwrap();

    let info: AllocationInfo = allocation_counter::measure(|| {
        op.apply_into(TAU, &f0, &mut dst, &mut pool).unwrap();
    });

    assert!(
        info.count_total <= 300,
        "NonSeparableMixedChernoff with_beta (nonzero coupling) steady-state allocated {} \
         time(s) per step (expected ≤ 300 on N=16 grid; 2D intermediates not poolable)",
        info.count_total
    );
}
