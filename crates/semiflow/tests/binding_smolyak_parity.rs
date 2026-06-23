//! `G_BINDING_SMOLYAK_PARITY` — sub-test 1 (core golden + F(0)=I anchor).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0138, ADR-0123 Amdt 1, slow-tests):
//!   Canonical smoke params (`V8_1_TIER3_BINDING_DESIGN.md` §1.3,
//!   contracts/semiflow-core.properties.yaml §`G_BINDING_SMOLYAK_PARITY)`:
//!     D=6, domain=[-2.0,2.0] per axis (see rationale below), `n_per_axis=4`,
//!     `n_chernoff=1`, tau=0.01, u0(x)=exp(-Σx²).
//!
//! ## Domain choice rationale
//!
//! The design doc §1.3 originally specifies domain=[-5,5].  The `g_smolyak_d6`
//! gate (`tests/g_smolyak_d6.rs`) established that `N_AXIS=4` with [-5,5] produces
//! near-zero IC values (inner grid points at ±5/3 give IC≈5.6e-8 — essentially
//! machine-zero).  The parity gate uses domain=[-2,2] (same correction as
//! `g_smolyak_d6`) so the Gaussian IC is non-trivial on the grid.
//!
//! ## Anchor: F(0)=I
//!
//! Apply at tau=0 to a ones-function: result should equal input,
//! `sup_err` ≤ 1e-10 (mirrors `g_smolyak_d6` sub-test 2).
//!
//! ## Golden output
//!
//! The golden flat vec (length 4^6=4096) from `apply_into` at tau=0.01
//! is printed and exposed via `canonical_smolyak_core()` for embedding
//! in the `PyO3` binding sub-test (0-ULP check).

// Integration test/bench/example: allows for numerical patterns.
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)] // D as u32: D=6, well within u32

use semiflow::{
    grid_nd::{GridFnND, GridND},
    smolyak::SmolyakGridND,
    ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters (§1.3, V8_1_TIER3_BINDING_DESIGN.md)
// ---------------------------------------------------------------------------

const D: usize = 6;
const DOMAIN_LO: f64 = -2.0;
const DOMAIN_HI: f64 = 2.0;
const N_PER_AXIS: usize = 4;
const N_CHERNOFF: usize = 1;
const TAU: f64 = 0.01;
const LEVEL: usize = D + 3; // 9 → 533 nodes

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_grid() -> GridND<f64, D> {
    let ax = Grid1D::new(DOMAIN_LO, DOMAIN_HI, N_PER_AXIS).unwrap();
    GridND::new([ax; D]).unwrap()
}

/// Build unit-diffusion D=6 Smolyak kernel (a=I, b=0, c=0).
fn make_kernel() -> SmolyakGridND<f64, D> {
    let grid = make_grid();
    SmolyakGridND::with_level(
        |_x: &[f64; D], a: &mut SquareMatrix<f64, D>| {
            for i in 0..D {
                a.set(i, i, 1.0);
            }
        },
        |_x: &[f64; D], b: &mut [f64; D]| {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        },
        |_x: &[f64; D]| 0.0_f64,
        grid,
        LEVEL,
    )
    .unwrap()
}

/// Gaussian IC: exp(-Σx²).
fn gaussian(x: &[f64; D]) -> f64 {
    (-x.iter().map(|xi| xi * xi).sum::<f64>()).exp()
}

// ---------------------------------------------------------------------------
// Core golden computation
// ---------------------------------------------------------------------------

/// Apply `n_chernoff` Smolyak steps with `tau` to the Gaussian IC.
///
/// Public so the `PyO3` binding test can embed the exact golden values.
pub fn canonical_smolyak_core() -> Vec<f64> {
    let kernel = make_kernel();
    let grid = make_grid();
    let u0 = GridFnND::from_fn(grid.clone(), gaussian);
    let mut src = u0;
    let mut dst = GridFnND::from_fn(grid, |_| 0.0_f64);
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..N_CHERNOFF {
        kernel.apply_into(TAU, &src, &mut dst, &mut pool).unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src.values
}

// ---------------------------------------------------------------------------
// G_BINDING_SMOLYAK_PARITY sub-test 1: core golden + F(0)=I anchor
// ---------------------------------------------------------------------------

/// `G_BINDING_SMOLYAK_PARITY` sub-test 1: core golden + F(0)=I anchor.
///
/// Feature-gated under slow-tests (fast in practice: 533 Smolyak nodes on
/// 4^6=4096 grid pts, 1 step).
#[cfg(feature = "slow-tests")]
#[test]
fn g_binding_smolyak_parity_core_golden() {
    let kernel = make_kernel();
    let n_nodes = kernel.n_nodes();
    println!(
        "\nG_BINDING_SMOLYAK_PARITY sub-test 1 (core golden + F(0)=I):\n\
         D={D}, domain=[{DOMAIN_LO},{DOMAIN_HI}], n_per_axis={N_PER_AXIS}, \
         n_chernoff={N_CHERNOFF}, tau={TAU}\n\
         Smolyak level={LEVEL}, n_nodes={n_nodes}"
    );

    // -----------------------------------------------------------------------
    // Sub-test 1a: node-count sanity (D=6, ℓ=9 → 533 nodes)
    // -----------------------------------------------------------------------
    assert!(
        n_nodes < 46656,
        "n_nodes={n_nodes} should be < tensor 6^6=46656"
    );

    // -----------------------------------------------------------------------
    // Sub-test 1b: F(0)=I anchor
    // -----------------------------------------------------------------------
    {
        let grid = make_grid();
        let ones = GridFnND::from_fn(grid.clone(), |_| 1.0_f64);
        let mut out = GridFnND::from_fn(grid, |_| 0.0_f64);
        let mut pool = ScratchPool::<f64>::new();
        kernel.apply_into(0.0, &ones, &mut out, &mut pool).unwrap();
        let sup_err = out
            .values
            .iter()
            .map(|&v| (v - 1.0).abs())
            .fold(0.0_f64, f64::max);
        println!("F(0)=I sup_err = {sup_err:.3e}  (gate < 1e-10)");
        assert!(
            sup_err < 1e-10,
            "F(0)=I gate FAILED: sup_err={sup_err:.3e} >= 1e-10"
        );
    }

    // -----------------------------------------------------------------------
    // Sub-test 1c: compute and print golden for embedding in PyO3 sub-test
    // -----------------------------------------------------------------------
    let golden = canonical_smolyak_core();
    assert_eq!(golden.len(), N_PER_AXIS.pow(D as u32));

    let max_val = golden.iter().copied().fold(0.0_f64, f64::max);
    let min_val = golden.iter().copied().fold(f64::MAX, f64::min);
    let all_finite = golden.iter().all(|v| v.is_finite());

    println!(
        "Golden length={}, min={min_val:.6e}, max={max_val:.6e}, all_finite={all_finite}",
        golden.len()
    );
    assert!(all_finite, "golden output contains non-finite values");
    assert!(max_val > 0.0, "golden output is all-zero or negative");

    // Print first 8 and last 4 values for embedding.
    println!("golden[0..8]  = {:?}", &golden[..8.min(golden.len())]);
    println!(
        "golden[-4..]  = {:?}",
        &golden[golden.len().saturating_sub(4)..]
    );
    println!("G_BINDING_SMOLYAK_PARITY sub-test 1: PASS ✓");
}
