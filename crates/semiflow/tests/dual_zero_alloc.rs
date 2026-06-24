//! `G_DUAL_ZERO_ALLOC` — steady-state heap allocation parity gate.
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0133, math.md §46.3):
//!   alloc count during one warm `apply_f` evolution loop at F = Dual<f64>
//!   MUST equal the count at F = f64 (both SHOULD be 0 in steady state).
//!
//! # Measurement protocol
//!   1. Build both kernels outside the measurement window.
//!   2. Pre-warm: run one full N-step loop for each type (first call allocates
//!      the output Vec; any one-time scratch growth settles here).
//!   3. RESET the counter AFTER the warm-up pass.
//!   4. Measure a SECOND full N-step loop.
//!   5. Assert `dual_count` == `baseline_count`.
//!
//! # Why `apply_f` instead of Evolver?
//!   `DiffusionChernoff<Dual<f64>>` does NOT implement `ChernoffFunction<Dual<f64>>`
//!   (the trait impl is f64-concrete for SIMD; ADR-0018). The generic path is
//!   `DiffusionChernoff::apply_f(tau, &u)`. We iterate `apply_f` in a manual
//!   loop, which is exactly what the generic Evolver would do if the trait bound
//!   were satisfied.
//!
//! # std requirement
//!   `#[global_allocator]` requires `std::alloc::System`.
//!   Test binaries link std even when the lib target is `no_std + alloc`.

#![cfg(feature = "slow-tests")]
#![allow(unsafe_code)]
#![allow(clippy::cast_precision_loss)] // usize→f64 for N_STEPS: values ≤ 1000 ≤ 2^52

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use semiflow::{DiffusionChernoff, Dual, Grid1D, GridFn1D, InterpKind};

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

struct CountingAlloc;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn reset_counter() {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
}
fn read_counter() -> usize {
    ALLOC_COUNT.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Gate constants
// ---------------------------------------------------------------------------

const N_STEPS: usize = 64;
const N_GRID: usize = 128;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const T_FINAL: f64 = 1.0;

fn a_const_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::constant(0.5)
}
fn zero_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::constant(0.0)
}

// ---------------------------------------------------------------------------
// f64 baseline — apply_chernoff loop (allocates a fresh Vec per step).
// After warm-up the Vec capacity is stable; we measure a second pass.
// ---------------------------------------------------------------------------

fn run_f64_warmup_then_count() -> usize {
    // CubicHermite to match new_generic default (SepticHermite unsupported in apply_f).
    let grid = Grid1D::new(X_MIN, X_MAX, N_GRID)
        .expect("grid valid")
        .with_interp(InterpKind::CubicHermite);
    let diff = DiffusionChernoff::with_closure(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, grid);
    let u0 = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());
    let tau = T_FINAL / N_STEPS as f64;
    let mut u = u0.clone();

    // Pre-warm.
    for _ in 0..N_STEPS {
        u = diff.apply_chernoff(tau, &u).expect("f64 warmup");
    }
    reset_counter();

    // Measurement.
    for _ in 0..N_STEPS {
        u = diff.apply_chernoff(tau, &u).expect("f64 measure");
    }
    let _ = u;
    read_counter()
}

// ---------------------------------------------------------------------------
// Dual<f64> measurement — apply_f loop, same structure.
// ---------------------------------------------------------------------------

fn run_dual_warmup_then_count() -> usize {
    let grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("grid valid");
    let diff = DiffusionChernoff::<Dual<f64>>::new(
        a_const_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        0.5,
        grid,
    );
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau = Dual::constant(T_FINAL / N_STEPS as f64);
    let mut u = u0.clone();

    // Pre-warm.
    for _ in 0..N_STEPS {
        u = diff.apply_f(tau, &u).expect("dual warmup");
    }
    reset_counter();

    // Measurement.
    for _ in 0..N_STEPS {
        u = diff.apply_f(tau, &u).expect("dual measure");
    }
    let _ = u;
    read_counter()
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

/// `G_DUAL_ZERO_ALLOC`: warm `apply_f` alloc count equal at F=`Dual<f64>` vs F=f64.
///
/// Both SHOULD be 0 in steady state (ADR-0041 zero-alloc ping-pong invariant).
/// The load-bearing contract is COUNT equality: a wider element (Dual = 2× bytes)
/// changes first-allocation SIZE, not per-step COUNT. Any positive delta means
/// the dual field leaks heap allocations on the hot path — escalate to Architect.
#[test]
#[ignore = "G_DUAL_ZERO_ALLOC: run with --features slow-tests --release -- --ignored"]
fn g_dual_zero_alloc() {
    let baseline = run_f64_warmup_then_count();
    let dual_allocs = run_dual_warmup_then_count();

    println!(
        "G_DUAL_ZERO_ALLOC: f64 baseline={baseline} allocs, \
         Dual<f64>={dual_allocs} allocs  (gate: dual == baseline)"
    );

    assert!(
        dual_allocs == baseline,
        "G_DUAL_ZERO_ALLOC FAIL: Dual<f64> steady-state allocs ({dual_allocs}) \
         != f64 baseline ({baseline}). Dual field leaks per-step heap allocation. \
         Escalate to Architect."
    );
}
