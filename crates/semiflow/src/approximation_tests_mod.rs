// Tests for `approximation.rs` (ApproximationSubspace, LadderRung, assert_in_subspace).
//
// Properties asserted:
//   1. DiffusionChernoff K=2: in_subspace=true for >=5 pts, false for <5 or NaN.
//   2. DiffusionChernoff K=2: jet returns [f, Af, A²f] with correct lengths.
//   3. DiffusionChernoff K=2: jet DomainViolation if out.len() != 3.
//   4. Diffusion4thChernoff K=2: in_subspace=true for >=9 pts; jet smoke.
//   5. Diffusion4thChernoff K=4: in_subspace=true for >=9 pts; jet 5 slots.
//   6. assert_in_subspace: returns Ok when in subspace, Err when not.
//   7. LadderRung::PREDECESSOR_K values for K=2,4,6,8.
//   8. KolmogorovHypoelliptic K=2: jet returns Unsupported.

use super::*;
use crate::{
    diffusion::DiffusionChernoff,
    diffusion4::Diffusion4thChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    truncated_exp4::TruncatedExp4thDiffusionChernoff,
};

fn make_diffusion(n: usize) -> DiffusionChernoff<f64> {
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid)
}

fn make_d4(n: usize) -> Diffusion4thChernoff<f64> {
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid)
}

fn gridfn(n: usize, f: impl Fn(f64) -> f64) -> GridFn1D<f64> {
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    GridFn1D::from_fn(grid, f)
}

// ── DiffusionChernoff K=2: in_subspace ───────────────────────────────────────

#[test]
fn diffusion_in_subspace_true_for_5pts() {
    let dc = make_diffusion(16);
    let f = gridfn(16, |x| x.sin());
    assert!(
        ApproximationSubspace::<2, f64>::in_subspace(&dc, &f),
        "expected true for 16-pt grid"
    );
}

#[test]
fn diffusion_in_subspace_false_for_4pts() {
    let dc = make_diffusion(4);
    let f = gridfn(4, |x| x);
    assert!(
        !ApproximationSubspace::<2, f64>::in_subspace(&dc, &f),
        "expected false for 4-pt grid"
    );
}

#[test]
fn diffusion_in_subspace_false_for_nan_value() {
    let dc = make_diffusion(16);
    let mut f = gridfn(16, |x| x);
    f.values[3] = f64::NAN;
    assert!(
        !ApproximationSubspace::<2, f64>::in_subspace(&dc, &f),
        "expected false for NaN value"
    );
}

// ── DiffusionChernoff K=2: jet ────────────────────────────────────────────────

#[test]
fn diffusion_jet_k2_returns_3_slots() {
    let dc = make_diffusion(16);
    let f = gridfn(16, |x| x.sin());
    let zero = gridfn(16, |_| 0.0);
    let mut out = [zero.clone(), zero.clone(), zero];
    ApproximationSubspace::<2, f64>::jet(&dc, &f, &mut out).unwrap();
    // out[0] = f
    for (v, e) in out[0].values.iter().zip(f.values.iter()) {
        assert!((v - e).abs() < 1e-14, "out[0] != f");
    }
    // out[1] = Af should be finite
    assert!(out[1].values.iter().all(|v| v.is_finite()), "out[1] non-finite");
}

#[test]
fn diffusion_jet_k2_wrong_out_len_errors() {
    let dc = make_diffusion(16);
    let f = gridfn(16, |x| x);
    let zero = gridfn(16, |_| 0.0);
    let mut out = [zero.clone(), zero];
    let result = ApproximationSubspace::<2, f64>::jet(&dc, &f, &mut out);
    assert!(result.is_err(), "expected DomainViolation for wrong out.len()");
}

// ── Diffusion4thChernoff K=2 and K=4 ─────────────────────────────────────────

#[test]
fn d4_in_subspace_k2_true_for_9pts() {
    let dc = make_d4(16);
    let f = gridfn(16, |x| x.sin());
    assert!(ApproximationSubspace::<2, f64>::in_subspace(&dc, &f));
}

#[test]
fn d4_in_subspace_k4_true_for_9pts() {
    let dc = make_d4(16);
    let f = gridfn(16, |x| x.sin());
    assert!(ApproximationSubspace::<4, f64>::in_subspace(&dc, &f));
}

#[test]
fn d4_jet_k4_writes_5_slots() {
    let dc = make_d4(16);
    let f = gridfn(16, |x| x.sin());
    let zero = gridfn(16, |_| 0.0);
    let mut out = vec![zero; 5];
    ApproximationSubspace::<4, f64>::jet(&dc, &f, &mut out).unwrap();
    assert!(out[0].values.iter().all(|v| v.is_finite()));
    assert!(out[4].values.iter().all(|v| v.is_finite()));
}

// ── assert_in_subspace helper ─────────────────────────────────────────────────

#[test]
fn assert_in_subspace_ok_for_valid_datum() {
    let dc = make_diffusion(16);
    let f = gridfn(16, |x| x.sin());
    assert_in_subspace::<_, f64, 2>(&dc, &f).unwrap();
}

#[test]
fn assert_in_subspace_err_for_invalid_datum() {
    let dc = make_diffusion(4); // 4 pts: in_subspace=false
    let f = gridfn(4, |x| x);
    assert!(assert_in_subspace::<_, f64, 2>(&dc, &f).is_err());
}

// ── LadderRung PREDECESSOR_K constants ───────────────────────────────────────

#[test]
fn ladder_rung_k2_has_no_predecessor() {
    use crate::LadderRung;
    assert_eq!(
        <Diffusion4thChernoff<f64> as LadderRung<2, f64>>::PREDECESSOR_K,
        None
    );
}

#[test]
fn ladder_rung_k4_predecessor_is_2() {
    use crate::{diffusion4_zeta4::Diffusion4thZeta4Chernoff, LadderRung};
    assert_eq!(
        <Diffusion4thZeta4Chernoff<f64> as LadderRung<4, f64>>::PREDECESSOR_K,
        Some(2)
    );
}

// ── KolmogorovHypoelliptic K=2: jet returns Unsupported ──────────────────────

#[test]
fn kolmogorov_jet_returns_unsupported() {
    use crate::{
        grid2d::Grid2D,
        grid_fn2d::GridFn2D,
        hormander::{KolmogorovHypoelliptic, KolmogorovPhaseSpace},
    };
    let x0 = alloc::boxed::Box::new(KolmogorovPhaseSpace::<f64>::x0_drift());
    let x1 = alloc::boxed::Box::new(KolmogorovPhaseSpace::<f64>::x1_diffusion());
    let kernel = KolmogorovHypoelliptic::<f64>::new(x0, [x1]).unwrap();
    let gx = Grid1D::new(0.0_f64, 1.0, 6).unwrap();
    let gy = Grid1D::new(0.0_f64, 1.0, 6).unwrap();
    let grid = Grid2D::new(gx, gy);
    let f = GridFn2D::from_fn(grid, |x, _y| x);
    let zero = GridFn2D::from_fn(grid, |_, _| 0.0);
    let mut out = [zero.clone(), zero.clone(), zero];
    let result = ApproximationSubspace::<2, f64>::jet(&kernel, &f, &mut out);
    assert!(result.is_err(), "expected Unsupported from Kolmogorov jet");
}

// ── Diffusion4thChernoff K=2 jet writes 3 slots ───────────────────────────────

#[test]
fn d4_k2_jet_writes_3_slots() {
    let kernel = make_d4(16);
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let f = GridFn1D::from_fn(grid, |x| x * (1.0 - x));
    let zero = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut out = [zero.clone(), zero.clone(), zero];
    ApproximationSubspace::<2, f64>::jet(&kernel, &f, &mut out).unwrap();
    assert_eq!(out[0].values, f.values, "out[0] must equal f");
    for v in &out[1].values {
        assert!(v.is_finite(), "out[1] non-finite: {v}");
    }
}

// ── Diffusion4thChernoff K=2 jet DomainViolation for wrong out len ────────────

#[test]
fn d4_k2_jet_wrong_out_len_errors() {
    let kernel = make_d4(16);
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let f = GridFn1D::from_fn(grid, |x| x);
    let zero = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut out = [zero.clone(), zero];
    let result = ApproximationSubspace::<2, f64>::jet(&kernel, &f, &mut out);
    assert!(result.is_err(), "expected Err for out.len() != 3");
}

// ── TruncatedExp4thDiffusionChernoff K=6 in_subspace ────────────────────────

#[test]
fn te4_k6_in_subspace_true_for_16pts() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let kernel = TruncatedExp4thDiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let f = GridFn1D::from_fn(grid, |x| x.sin());
    assert!(
        ApproximationSubspace::<6, f64>::in_subspace(&kernel, &f),
        "in_subspace should be true for 16 pts"
    );
}

#[test]
fn te4_k6_in_subspace_false_for_10pts() {
    // K=6 requires >= 13 pts.
    let grid = Grid1D::new(0.0_f64, 1.0, 10).unwrap();
    let kernel = TruncatedExp4thDiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let f = GridFn1D::from_fn(grid, |x| x.sin());
    assert!(
        !ApproximationSubspace::<6, f64>::in_subspace(&kernel, &f),
        "in_subspace should be false for 10 pts (need >= 13)"
    );
}

// ── TruncatedExp4thDiffusionChernoff K=6 jet writes 7 slots ─────────────────

#[test]
fn te4_k6_jet_writes_7_slots() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let kernel = TruncatedExp4thDiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let f = GridFn1D::from_fn(grid, |x| x.sin());
    let zero = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut out: alloc::vec::Vec<_> = (0..7).map(|_| zero.clone()).collect();
    ApproximationSubspace::<6, f64>::jet(&kernel, &f, &mut out).unwrap();
    assert_eq!(out[0].values, f.values, "out[0] must equal f");
    for (i, slot) in out.iter().enumerate() {
        for v in &slot.values {
            assert!(v.is_finite(), "out[{i}] has non-finite value: {v}");
        }
    }
}

// ── TruncatedExp4thDiffusionChernoff K=6 jet DomainViolation ─────────────────

#[test]
fn te4_k6_jet_wrong_out_len_errors() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let kernel = TruncatedExp4thDiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let f = GridFn1D::from_fn(grid, |x| x);
    let zero = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut out = [zero.clone(), zero.clone(), zero];
    let result = ApproximationSubspace::<6, f64>::jet(&kernel, &f, &mut out);
    assert!(result.is_err(), "expected Err for out.len() != 7");
}

// ── LadderRung K=6, K=8 PREDECESSOR_K ────────────────────────────────────────

#[test]
fn ladder_rung_k6_predecessor_is_4() {
    use crate::diffusion6_zeta6::Diffusion6thZeta6Chernoff;
    assert_eq!(
        <Diffusion6thZeta6Chernoff<f64> as LadderRung<6, f64>>::PREDECESSOR_K,
        Some(4)
    );
}

#[test]
fn ladder_rung_k8_predecessor_is_6() {
    use crate::diffusion8_zeta8::Diffusion8thZeta8Chernoff;
    assert_eq!(
        <Diffusion8thZeta8Chernoff<f64> as LadderRung<8, f64>>::PREDECESSOR_K,
        Some(6)
    );
}
