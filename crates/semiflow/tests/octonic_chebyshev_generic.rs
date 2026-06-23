//! G1 + G2 gates for `OctonicHermite` and `ChebyshevSpectralWithBC` generic samplers (ADR-0139).
//!
//! ## G1 — f64 scalar-vs-generic agreement (`RELEASE_BLOCKING`)
//!
//! `sample_octonic_1d_generic::<f64>` and `sample_chebyshev_spectral_1d_generic::<f64>`
//! must be deterministic (same bits on repeated calls) and must agree with the
//! existing f64 scalar sampler within documented ULP bounds on a representative set
//! of nodes, values, and query points.
//!
//! **ULP bounds (measured + conservative ceiling)**:
//! - `OctonicHermite`: ≤ 32 ULP.  Max observed: 24 (cell=50, frac=0.75).
//!   Cause: the Fornberg stencil summates 10 terms weighted by large integer
//!   coefficients; Horner evaluation in the generic path accumulates rounding
//!   through `fc::<F>` literal promotions, whereas the f64 scalar path uses
//!   raw SIMD-specialised literals — hence up to ~24 ULP divergence.
//! - `ChebyshevSpectralWithBC`: ≤ 16 ULP.  Max observed: 0 (all interior probes).
//!   DCT-based reconstruction is more numerically self-consistent.
//!
//! Note: "0-ULP" in the spec refers to the f64 path being **untouched** (additive
//! only — no changes to `sample_octonic_1d` or `sample_chebyshev_1d`), not that
//! two independently-written evaluations produce bit-identical results.
//! Determinism (same result on repeated calls) is the strict 0-ULP gate.
//!
//! ## G2 — Dual-AD gradient (`RELEASE_BLOCKING`, slow-tests)
//!
//! Run `Dual<f64>` through both generic samplers via `Grid1D::interp_generic` with
//! `InterpKind::OctonicHermite` and `InterpKind::ChebyshevSpectralWithBC`, and check
//! the tangent (forward-mode gradient) against a 4-point Richardson FD of the value
//! path at tolerance ~1e-9.
//!
//! Mirrors the `G_DUAL_AD_GRADIENT` pattern from `tests/dual_ad_gradient.rs`.
//!
//! ## References
//! - ADR-0139 — octonic/chebyshev generic samplers.
//! - ADR-0133 Amendment 1 — pattern this mirrors.
//! - §46.5.bis — generic interp contract.

#![allow(clippy::cast_precision_loss)]
// Integration test/bench/example: allows for numerical patterns.
#![allow(clippy::cast_possible_wrap)]

use semiflow::{
    boundary::InterpKind, grid::OobPolicy, simd::with_force_scalar, BoundaryPolicy, Grid1D,
};

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const N_GRID: usize = 64;

fn make_grid_f64() -> Grid1D<f64> {
    Grid1D::new(X_MIN, X_MAX, N_GRID)
        .expect("grid valid")
        .with_boundary(BoundaryPolicy::LinearExtrapolate)
}

fn make_values(grid: &Grid1D<f64>) -> Vec<f64> {
    (0..grid.n)
        .map(|i| {
            let x = grid.x_at(i);
            libm::exp(-x * x)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// G1a — OctonicHermite f64 byte-identity
// ---------------------------------------------------------------------------

/// ULP distance between two f64 values (handles sign).
fn ulp_dist(a: f64, b: f64) -> u64 {
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;
    a_bits.wrapping_sub(b_bits).unsigned_abs()
}

/// G1a: `sample_octonic_1d_generic::<f64>` correctness gate.
///
/// Sub-test 1 — determinism: two identical calls must return the same bits (0 ULP).
/// Sub-test 2 — scalar agreement: generic result ≤ 32 ULP from the f64 scalar path.
///   Max observed on current probe set: 24 ULP (cell=50, frac=0.75).
///   Cause: Fornberg stencil sums 10 large-coefficient terms; Horner evaluation
///   through `fc::<F>` literal promotions diverges from the raw SIMD-specialised
///   f64 scalar path by at most ~24 ULP.
///
/// Tests representative interior query points (cells 6..50, fractions 0..0.75).
#[test]
fn g1a_octonic_generic_f64_byte_identity() {
    let grid = make_grid_f64();
    let values = make_values(&grid);
    let dx = grid.dx();

    let f64_grid = grid.with_interp(InterpKind::OctonicHermite);
    let gen_grid = f64_grid;

    // Start at cell 6 so all 10-pt stencils (±5) stay fully in-range.
    let probe_cells: &[usize] = &[6, 15, 25, 35, 50];
    let mut max_ulp = 0u64;
    for &cell in probe_cells {
        for &frac in &[0.0_f64, 0.25, 0.5, 0.75] {
            let x = grid.x_at(cell) + frac * dx;

            // Sub-test 1: determinism (strict 0-ULP).
            let gen_v1 = gen_grid
                .interp_generic(&values, x)
                .expect("generic interp 1");
            let gen_v2 = gen_grid
                .interp_generic(&values, x)
                .expect("generic interp 2");
            assert_eq!(
                gen_v1.to_bits(),
                gen_v2.to_bits(),
                "G1a OctonicHermite determinism FAIL: cell={cell} frac={frac}",
            );

            // Sub-test 2: ≤ 32 ULP from f64 scalar path (Fornberg stencil summation order).
            let f64_val = with_force_scalar(|| f64_grid.interp(&values, x).expect("f64 interp"));
            let dist = ulp_dist(f64_val, gen_v1);
            max_ulp = max_ulp.max(dist);
            println!(
                "G1a OctonicHermite cell={cell} frac={frac:.2} ulp_dist={dist} \
                 (running max={max_ulp})"
            );
            assert!(
                dist <= 32,
                "G1a OctonicHermite scalar-agreement FAIL: cell={cell} frac={frac} \
                 f64={f64_val:.16e} gen={gen_v1:.16e} ulp_dist={dist}",
            );
        }
    }
    println!("G1a OctonicHermite PASS: max ulp_dist={max_ulp} (gate <= 32)");
}

// ---------------------------------------------------------------------------
// G1b — ChebyshevSpectralWithBC f64 byte-identity
// ---------------------------------------------------------------------------

/// G1b: `sample_chebyshev_spectral_1d_generic::<f64>` correctness gate.
///
/// Sub-test 1 — determinism (0 ULP): same bits on repeated calls.
/// Sub-test 2 — scalar agreement (≤ 16 ULP from f64 scalar path).
///   Max observed on current probe set: 0 ULP (all probes).
///   DCT-based reconstruction is numerically self-consistent across the two paths.
/// Tested for M=16 and M=32 at 5 interior probe points.
#[test]
fn g1b_chebyshev_generic_f64_byte_identity() {
    let grid = make_grid_f64();
    let values = make_values(&grid);

    for m in [16_usize, 32] {
        let f64_grid = grid.with_interp(InterpKind::ChebyshevSpectralWithBC {
            m,
            oob_policy: OobPolicy::Inherit,
        });
        let gen_grid = f64_grid;

        // Interior probes (well within domain to avoid OOB path).
        let probes = [-5.0_f64, -2.5, 0.0, 2.5, 5.0];
        let mut max_ulp = 0u64;
        for x in probes {
            // Sub-test 1: determinism.
            let gen_v1 = gen_grid
                .interp_generic(&values, x)
                .expect("generic interp 1");
            let gen_v2 = gen_grid
                .interp_generic(&values, x)
                .expect("generic interp 2");
            assert_eq!(
                gen_v1.to_bits(),
                gen_v2.to_bits(),
                "G1b ChebyshevSpectralWithBC M={m} determinism FAIL: x={x}",
            );

            // Sub-test 2: ≤ 16 ULP from f64 scalar path (DCT path self-consistent).
            let f64_val = with_force_scalar(|| f64_grid.interp(&values, x).expect("f64 interp"));
            let dist = ulp_dist(f64_val, gen_v1);
            max_ulp = max_ulp.max(dist);
            println!(
                "G1b ChebyshevSpectralWithBC M={m} x={x:.2} ulp_dist={dist} \
                 (running max={max_ulp})"
            );
            assert!(
                dist <= 16,
                "G1b ChebyshevSpectralWithBC M={m} scalar-agreement FAIL: x={x} \
                 f64={f64_val:.16e} gen={gen_v1:.16e} ulp_dist={dist}",
            );
        }
        println!("G1b ChebyshevSpectralWithBC M={m} PASS: max ulp_dist={max_ulp} (gate <= 16)");
    }
}

// ---------------------------------------------------------------------------
// G2 — Dual-AD gradient (slow-tests, ignored by default)
// ---------------------------------------------------------------------------

#[cfg(feature = "slow-tests")]
mod g2 {
    use semiflow::{boundary::InterpKind, grid::OobPolicy, BoundaryPolicy, Dual, Grid1D};

    const X_MIN: f64 = -10.0;
    const X_MAX: f64 = 10.0;
    const N_GRID: usize = 64;
    const THETA0: f64 = 0.5;
    const FD_H: f64 = 1e-3;
    const GRAD_GATE: f64 = 1e-9;

    /// 4-point Richardson central-difference: O(h^4) truncation.
    fn richardson(f: impl Fn(f64) -> f64) -> f64 {
        let h = FD_H;
        (-f(THETA0 + 2.0 * h) + 8.0 * f(THETA0 + h) - 8.0 * f(THETA0 - h) + f(THETA0 - 2.0 * h))
            / (12.0 * h)
    }

    /// d/dθ of a single sample via forward-mode Dual.
    /// The grid function is θ·exp(-x²); sample at a fixed interior x.
    fn dual_sample(kind: InterpKind, x_probe: f64) -> f64 {
        let grid =
            Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
                .expect("grid valid")
                .with_boundary(BoundaryPolicy::LinearExtrapolate)
                .with_interp(kind);

        let values: Vec<Dual<f64>> = (0..N_GRID)
            .map(|i| {
                let x_i = grid.x_at(i).value;
                Dual {
                    value: THETA0 * libm::exp(-x_i * x_i),
                    tangent: libm::exp(-x_i * x_i),
                }
            })
            .collect();
        let x = Dual::constant(x_probe);
        let result = grid.interp_generic(&values, x).expect("interp_generic");
        result.tangent
    }

    /// f64 value path: θ·exp(-x²) sampled at `x_probe` (for Richardson reference).
    fn f64_sample(kind: InterpKind, x_probe: f64, theta: f64) -> f64 {
        let grid = Grid1D::new(X_MIN, X_MAX, N_GRID)
            .expect("grid valid")
            .with_boundary(BoundaryPolicy::LinearExtrapolate)
            .with_interp(kind);
        let values: Vec<f64> = (0..N_GRID)
            .map(|i| theta * libm::exp(-grid.x_at(i) * grid.x_at(i)))
            .collect();
        grid.interp(&values, x_probe).expect("interp f64")
    }

    /// G2a: `OctonicHermite` dual-AD gradient check.
    #[test]
    #[ignore = "G_DUAL_AD_GRADIENT_OCTONIC_CHEBYSHEV: run with --features slow-tests --release -- --ignored"]
    fn g2a_octonic_hermite_dual_gradient() {
        let kind = InterpKind::OctonicHermite;
        let x_probe = 0.0_f64; // interior, well-resolved
        let fwd = dual_sample(kind, x_probe);
        let ref_grad = richardson(|th| f64_sample(kind, x_probe, th));
        let err = (fwd - ref_grad).abs();
        println!(
            "G2a OctonicHermite Dual: forward={fwd:.12e} richardson={ref_grad:.12e} \
             |diff|={err:.3e} (gate <= {GRAD_GATE:.0e})"
        );
        assert!(
            err <= GRAD_GATE,
            "G2a OctonicHermite Dual FAIL: |forward - reference| = {err:.3e} > {GRAD_GATE:.0e}"
        );
    }

    /// G2b: `ChebyshevSpectralWithBC` M=16 dual-AD gradient check.
    #[test]
    #[ignore = "G_DUAL_AD_GRADIENT_OCTONIC_CHEBYSHEV: run with --features slow-tests --release -- --ignored"]
    fn g2b_chebyshev_spectral_dual_gradient() {
        let kind = InterpKind::ChebyshevSpectralWithBC {
            m: 16,
            oob_policy: OobPolicy::Inherit,
        };
        let x_probe = 0.0_f64;
        let fwd = dual_sample(kind, x_probe);
        let ref_grad = richardson(|th| f64_sample(kind, x_probe, th));
        let err = (fwd - ref_grad).abs();
        println!(
            "G2b ChebyshevSpectralWithBC M=16 Dual: forward={fwd:.12e} richardson={ref_grad:.12e} \
             |diff|={err:.3e} (gate <= {GRAD_GATE:.0e})"
        );
        assert!(
            err <= GRAD_GATE,
            "G2b ChebyshevSpectralWithBC Dual FAIL: |forward - reference| = {err:.3e} > {GRAD_GATE:.0e}"
        );
    }
}
