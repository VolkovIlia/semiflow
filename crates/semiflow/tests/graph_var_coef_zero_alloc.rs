//! R4 zero-alloc test for `VarCoefGraphHeatChernoff` via `allocation-counter`.
//!
//! Verifies that the steady-state hot path (`apply_into` after warmup) performs
//! zero heap allocations.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use allocation_counter::{self, AllocationInfo};

use semiflow::{ChernoffFunction, Graph, GraphSignal, ScratchPool, VarCoefGraphHeatChernoff};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_vc(n: usize) -> (VarCoefGraphHeatChernoff<f64>, Arc<Graph<f64>>) {
    let g = Arc::new(Graph::<f64>::path(n));
    let a: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.4 * ((i as f64 * 0.3).cos()))
        .collect();
    let rho_bar = 4.0_f64;
    let vc = VarCoefGraphHeatChernoff::new(Arc::clone(&g), a, rho_bar).expect("valid inputs");
    (vc, g)
}

// ---------------------------------------------------------------------------
// R4 zero-alloc test
// ---------------------------------------------------------------------------

#[test]
fn var_coef_apply_into_zero_alloc_steady_state() {
    let n = 32usize;
    let (vc, g) = make_vc(n);

    let src = GraphSignal::from_fn(Arc::clone(&g), |i| (f64::from(i) * 0.1).sin());
    let mut dst = src.clone();
    let mut scratch = ScratchPool::<f64>::new();

    // Warmup: first call may allocate into the scratch pool.
    vc.apply_into(0.01, &src, &mut dst, &mut scratch)
        .expect("warmup ok");

    // Steady state: measure allocations.
    let info: AllocationInfo = allocation_counter::measure(|| {
        vc.apply_into(0.01, &src, &mut dst, &mut scratch)
            .expect("apply_into ok");
    });

    assert_eq!(
        info.count_total, 0,
        "VarCoefGraphHeatChernoff::apply_into must be zero-alloc in steady state, got {} allocs",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// Test that multiple warmup calls remain zero-alloc after first warmup
// ---------------------------------------------------------------------------

#[test]
fn var_coef_apply_into_zero_alloc_repeated() {
    let n = 16usize;
    let (vc, g) = make_vc(n);

    let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
    let mut dst = src.clone();
    let mut scratch = ScratchPool::<f64>::new();

    // Two warmup calls.
    for _ in 0..2 {
        vc.apply_into(0.005, &src, &mut dst, &mut scratch)
            .expect("warmup ok");
    }

    // Third call: must be zero-alloc.
    let info: AllocationInfo = allocation_counter::measure(|| {
        vc.apply_into(0.005, &src, &mut dst, &mut scratch)
            .expect("ok");
    });

    assert_eq!(
        info.count_total, 0,
        "repeated apply_into after warmup must be zero-alloc, got {} allocs",
        info.count_total
    );
}
