//! `G_ADAPTIVE_Q` — adaptive per-point Gauss-Hermite quadrature gate
//! (`RELEASE_BLOCKING`, ADR-0122, math.md §32.8).
//!
//! Gate: `|I_adaptive − I_fixed32| ≤ 1e-10` on the kernel-envelope class (analytic f).
//! Also reports mean q* achieved and verifies it is ≤ 9 (saving vs accuracy-equivalent fixed-9).
//!
//! # Test methodology
//! The PRE-FLIGHT tests analytic f, NOT grid-interpolated f (a coarse grid introduces
//! O(dx²) interpolation error that dominates the quadrature comparison).  This gate
//! directly evaluates ∫ e^{-‖η‖²} f(x+2√τ σ η) dη for representative kernel-envelope
//! integrands in 1-D, which is the setting the PRE-FLIGHT verifies.
//!
//! Tolerance: `tol=1e-10` (production target, NOT machine-epsilon — see ADR-0122).

// ─── GH node/weight tables for the adaptive ladder and the fixed-32 reference ──

// Integration test/example: allows for numerical patterns.
#![allow(clippy::cast_precision_loss, clippy::items_after_statements)]

const GH1_NODES: [f64; 1] = [0.0];
const GH1_WEIGHTS: [f64; 1] = [1.772_453_850_905_516];

const GH3_NODES: [f64; 3] = [-1.224_744_871_391_589, 0.0, 1.224_744_871_391_589];
const GH3_WEIGHTS: [f64; 3] = [
    0.295_408_975_150_919,
    1.181_635_900_603_677,
    0.295_408_975_150_919,
];

const GH5_NODES: [f64; 5] = [
    -2.020_182_870_456_086,
    -0.958_572_464_613_819,
    0.0,
    0.958_572_464_613_819,
    2.020_182_870_456_086,
];
const GH5_WEIGHTS: [f64; 5] = [
    0.019_953_242_059_046,
    0.393_619_323_152_241,
    0.945_308_720_482_942,
    0.393_619_323_152_241,
    0.019_953_242_059_046,
];

const GH7_NODES: [f64; 7] = [
    -2.651_961_356_835_233,
    -1.673_551_628_767_471,
    -0.816_287_882_858_965,
    0.0,
    0.816_287_882_858_965,
    1.673_551_628_767_471,
    2.651_961_356_835_233,
];
const GH7_WEIGHTS: [f64; 7] = [
    0.000_971_781_245_099_520,
    0.054_515_582_819_127_05,
    0.425_607_252_610_127_8,
    0.810_264_617_556_807_2,
    0.425_607_252_610_127_8,
    0.054_515_582_819_127_05,
    0.000_971_781_245_099_520,
];

const GH9_NODES: [f64; 9] = [
    -3.190_993_201_781_528,
    -2.266_580_584_531_843,
    -1.468_553_289_216_668,
    -0.723_551_018_752_838,
    0.0,
    0.723_551_018_752_838,
    1.468_553_289_216_668,
    2.266_580_584_531_843,
    3.190_993_201_781_528,
];
const GH9_WEIGHTS: [f64; 9] = [
    3.960_697_726_326_437e-5,
    0.004_943_624_275_536_941,
    0.088_474_527_394_376_64,
    0.432_651_559_002_555_64,
    0.720_235_215_606_051,
    0.432_651_559_002_555_64,
    0.088_474_527_394_376_64,
    0.004_943_624_275_536_941,
    3.960_697_726_326_437e-5,
];

const GH32_NODES: [f64; 32] = [
    -7.125_813_909_830_728,
    -6.409_498_149_269_661,
    -5.812_225_949_515_914,
    -5.275_550_986_515_881,
    -4.777_164_503_502_596,
    -4.305_547_953_351_199,
    -3.853_755_485_471_444,
    -3.417_167_492_818_571,
    -2.992_490_825_002_374,
    -2.577_249_537_732_317,
    -2.169_499_183_606_112,
    -1.767_654_109_463_201,
    -1.370_376_410_952_872,
    -0.976_500_463_589_683,
    -0.584_978_765_435_932,
    -0.194_840_741_569_399,
    0.194_840_741_569_399,
    0.584_978_765_435_932,
    0.976_500_463_589_683,
    1.370_376_410_952_872,
    1.767_654_109_463_201,
    2.169_499_183_606_112,
    2.577_249_537_732_317,
    2.992_490_825_002_374,
    3.417_167_492_818_571,
    3.853_755_485_471_444,
    4.305_547_953_351_199,
    4.777_164_503_502_596,
    5.275_550_986_515_881,
    5.812_225_949_515_914,
    6.409_498_149_269_661,
    7.125_813_909_830_728,
];
const GH32_WEIGHTS: [f64; 32] = [
    7.310_676_427_384_096e-23,
    9.231_736_536_518_258e-19,
    1.197_344_017_092_85e-15,
    4.215_010_211_326_416e-13,
    5.933_291_463_396_676e-11,
    4.098_832_164_770_879e-9,
    1.574_167_792_545_588e-7,
    3.650_585_129_562_378e-6,
    5.416_584_061_819_991e-5,
    5.362_683_655_279_72e-4,
    3.654_890_326_654_426e-3,
    1.755_342_883_157_344e-2,
    6.045_813_095_591_269e-2,
    1.512_697_340_766_423e-1,
    2.774_581_423_025_3e-1,
    3.752_383_525_928_025e-1,
    3.752_383_525_928_025e-1,
    2.774_581_423_025_3e-1,
    1.512_697_340_766_423e-1,
    6.045_813_095_591_269e-2,
    1.755_342_883_157_344e-2,
    3.654_890_326_654_426e-3,
    5.362_683_655_279_72e-4,
    5.416_584_061_819_991e-5,
    3.650_585_129_562_378e-6,
    1.574_167_792_545_588e-7,
    4.098_832_164_770_879e-9,
    5.933_291_463_396_676e-11,
    4.215_010_211_326_416e-13,
    1.197_344_017_092_85e-15,
    9.231_736_536_518_258e-19,
    7.310_676_427_384_096e-23,
];

const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3; // 1/√π — normalisation for GH

// ─── Analytic 1-D integrands (kernel-envelope class) ─────────────────────────
// g(η) = f(x_k + s*η) where s = 2√τ·σ is the effective shift (small in product limit).

fn gh_1d(nodes: &[f64], weights: &[f64], g: impl Fn(f64) -> f64) -> f64 {
    nodes
        .iter()
        .zip(weights.iter())
        .map(|(&n, &w)| w * g(n))
        .sum::<f64>()
        * INV_SQRT_PI
}

/// Adaptive selector: q* = min{q∈{3,5,7,9}: |`I_q` − I_{q-2}| ≤ tol}, seeded with q=1.
fn adaptive_1d(g: impl Fn(f64) -> f64, tol: f64) -> (usize, f64) {
    let mut prev = gh_1d(&GH1_NODES, &GH1_WEIGHTS, &g);
    let all = [
        (3_usize, GH3_NODES.as_slice(), GH3_WEIGHTS.as_slice()),
        (5, &GH5_NODES, &GH5_WEIGHTS),
        (7, &GH7_NODES, &GH7_WEIGHTS),
        (9, &GH9_NODES, &GH9_WEIGHTS),
    ];
    for &(q, ns, ws) in &all {
        let iq = gh_1d(ns, ws, &g);
        let diff = (iq - prev).abs();
        if diff <= tol {
            return (q, iq);
        }
        prev = iq;
    }
    (9, gh_1d(&GH9_NODES, &GH9_WEIGHTS, g))
}

fn fixed_32_1d(g: impl Fn(f64) -> f64) -> f64 {
    gh_1d(&GH32_NODES, &GH32_WEIGHTS, g)
}

/// `G_ADAPTIVE_Q` — adaptive matches fixed-32 within 1e-10 on kernel-envelope integrands.
///
/// Tests the 1-D adaptive quadrature rule in isolation (analytic f, no grid).
/// The PRE-FLIGHT (`scripts/verify_adaptive_quad.py`) proves this property.
#[test]
fn g_adaptive_q_vs_fixed32() {
    const TOL_GATE: f64 = 1e-10;
    let s = 0.3_f64; // small effective shift 2√τ σ (representative product regime)

    // Kernel-envelope integrand class (PRE-FLIGHT §methodology).
    let integrands: &[(&str, &dyn Fn(f64) -> f64)] = &[
        ("const", &|_eta| 1.0_f64),
        ("linear(s)", &|eta| 1.0 + 0.4 * (s * eta)),
        ("quadratic(s)", &|eta| {
            1.0 + 0.3 * (s * eta) + 0.2 * (s * eta) * (s * eta)
        }),
        ("cubic(s)", &|eta| {
            1.0 + 0.2 * (s * eta) + 0.1 * (s * eta).powi(2) + 0.05 * (s * eta).powi(3)
        }),
        ("exp(0.5 s)", &|eta| (0.5 * s * eta).exp()),
        ("gaussian IC(s)", &|eta| (-(s * eta) * (s * eta)).exp()),
    ];

    let mut max_err = 0.0_f64;
    let mut q_sum = 0_usize;
    let mut count = 0_usize;
    println!(
        "  {:<36}  {:>6}  {:>14}  {:>6}",
        "integrand", "q*", "|adapt-ref32|", "gate"
    );
    for &(name, g) in integrands {
        let (q_star, i_adapt) = adaptive_1d(g, TOL_GATE);
        let i_ref = fixed_32_1d(g);
        let err = (i_adapt - i_ref).abs();
        let gate_ok = err <= TOL_GATE;
        max_err = max_err.max(err);
        q_sum += q_star;
        count += 1;
        println!(
            "  {:<36}  {:>6}  {:>14.3e}  {:>6}",
            name,
            q_star,
            err,
            if gate_ok { "PASS" } else { "FAIL" }
        );
    }
    let mean_q = q_sum as f64 / count as f64;
    println!("  mean q* = {mean_q:.2}  (must be < 9, the accuracy-equivalent ceiling)");
    println!("G_ADAPTIVE_Q: max |adaptive−fixed_32| = {max_err:.3e}  (gate: ≤ {TOL_GATE:.1e})");
    assert!(
        max_err.is_finite() && max_err <= TOL_GATE,
        "G_ADAPTIVE_Q: max_err {max_err:.3e} exceeds gate {TOL_GATE:.1e}"
    );
    assert!(
        mean_q < 9.0,
        "G_ADAPTIVE_Q: mean q* {mean_q:.2} should be < 9 (adaptive must save nodes)"
    );
}

/// Smoke: `order()` == 1 for the adaptive wrapper.
#[test]
fn g_adaptive_q_order_is_1() {
    use semiflow::{
        grid_nd::GridND, shift_nd_adaptive::AnisotropicShiftAdaptiveQ, Grid1D, SquareMatrix,
    };
    let ax = Grid1D::new(-5.0_f64, 5.0, 8).unwrap();
    let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
    let k = AnisotropicShiftAdaptiveQ::new(
        |_x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            a.set(0, 1, 0.0);
            a.set(1, 0, 0.0);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x: &[f64; 2]| 0.0_f64,
        grid,
        1e-10_f64,
    )
    .unwrap();
    use semiflow::ChernoffFunction;
    assert_eq!(k.order(), 1, "AnisotropicShiftAdaptiveQ::order() must be 1");
}

/// Smoke: adaptive kernel produces finite output on Gaussian IC.
#[test]
fn g_adaptive_q_apply_smoke() {
    use semiflow::{
        grid_nd::{GridFnND, GridND},
        shift_nd_adaptive::AnisotropicShiftAdaptiveQ,
        ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
    };
    let ax = Grid1D::new(-5.0_f64, 5.0, 8).unwrap();
    let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
    let kernel = AnisotropicShiftAdaptiveQ::new(
        |_x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            a.set(0, 1, 0.0);
            a.set(1, 0, 0.0);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x: &[f64; 2]| 0.0_f64,
        grid,
        1e-10_f64,
    )
    .unwrap();
    let f0 = GridFnND::from_fn(kernel.grid().clone(), |x: &[f64; 2]| {
        (-x[0] * x[0] - x[1] * x[1]).exp()
    });
    let mut dst = f0.clone();
    let mut pool = ScratchPool::<f64>::new();
    kernel.apply_into(0.01, &f0, &mut dst, &mut pool).unwrap();
    assert!(
        dst.values.iter().all(|&v| v.is_finite()),
        "adaptive apply_into must produce finite output"
    );
}

/// Smoke: tol=0 must be rejected.
#[test]
fn g_adaptive_q_tol_zero_rejected() {
    use semiflow::{
        grid_nd::GridND, shift_nd_adaptive::AnisotropicShiftAdaptiveQ, Grid1D, SquareMatrix,
    };
    let ax = Grid1D::new(-5.0_f64, 5.0, 8).unwrap();
    let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
    let result = AnisotropicShiftAdaptiveQ::new(
        |_x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            a.set(0, 1, 0.0);
            a.set(1, 0, 0.0);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x: &[f64; 2]| 0.0_f64,
        grid,
        0.0_f64,
    );
    assert!(result.is_err(), "tol=0 must be rejected");
}
