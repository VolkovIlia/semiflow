// Inline unit tests for [`SmolyakGridND`] (extracted per suckless ≤500-line cap).
use super::*;
use crate::grid::Grid1D;

#[test]
fn binom_small() {
    assert_eq!(binom(0, 0), 1);
    assert_eq!(binom(4, 0), 1);
    assert_eq!(binom(4, 2), 6);
    assert_eq!(binom(4, 4), 1);
    assert_eq!(binom(3, 5), 0); // k > n
}

#[test]
fn smolyak_d2_l3_node_count() {
    // D=2, ell=3: standard Smolyak, known node count = 5
    let nw = build_smolyak::<2>(3).unwrap();
    assert_eq!(
        nw.len(),
        5,
        "D=2 ell=3 should have 5 nodes, got {}",
        nw.len()
    );
}

#[test]
fn smolyak_d5_l8_node_count() {
    // D=5, ell=8: the pre-flight verified 341 nodes.
    let nw = build_smolyak::<5>(8).unwrap();
    assert_eq!(
        nw.len(),
        341,
        "D=5 ell=8 should have 341 nodes, got {}",
        nw.len()
    );
}

#[test]
fn smolyak_d5_l8_weight_sum() {
    // Σ weights = π^{5/2} (F(0)=I witness).
    let nw = build_smolyak::<5>(8).unwrap();
    let wsum: f64 = nw.iter().map(|(_, w)| *w).sum();
    let expected = core::f64::consts::PI.powf(2.5);
    let rel = (wsum - expected).abs() / expected;
    assert!(rel < 1e-10, "weight sum rel err {rel:.2e} ≥ 1e-10");
}

#[test]
fn smolyak_d5_l8_has_negative_weights() {
    // Smolyak weights MUST include negative values (combination signs).
    let nw = build_smolyak::<5>(8).unwrap();
    let has_neg = nw.iter().any(|(_, w)| *w < 0.0);
    assert!(has_neg, "D=5 ell=8 must have negative combination weights");
}

fn make_grid_d5(n: usize) -> GridND<f64, 5> {
    let ax = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    GridND::new([ax; 5]).unwrap()
}

/// Per-axis node count for the two D=5 plumbing smoke tests.
///
/// These assert finiteness and `F(0) = I` — invariants that do not depend on the
/// grid size — so the smallest legal grid serves them exactly as well as a large
/// one. The size matters because ADR-0190 made `GridFnND::sample` a genuine
/// tensor-product interpolant: at `D = 5` the default `CubicHermite` reads
/// `4⁵ = 1024` nodes per sample against multilinear's `2⁵ = 32`, so an `n = 8`
/// grid (`8⁵ = 32768` points × 341 sparse quadrature nodes) turned these two
/// smoke tests into a ~198 s pair and `xtask test-fast` into a 14-minute run.
/// Shrinking a *smoke* test is the right lever; the D=5 accuracy claim lives in
/// the `slow-tests`-gated `G_SMOLYAK_D5` slope gate, which is untouched.
const SMOKE_N_D5: usize = 5;

fn unit_kernel_d5(n: usize) -> SmolyakGridND<f64, 5> {
    let grid = make_grid_d5(n);
    SmolyakGridND::new(
        |_x, a| {
            for i in 0..5 {
                a.set(i, i, 1.0);
            }
        },
        |_x, b| {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        },
        |_x| 0.0_f64,
        grid,
    )
    .unwrap()
}

#[test]
fn constructor_ok_d5() {
    let k = unit_kernel_d5(8);
    assert_eq!(k.n_nodes(), 341);
    assert_eq!(k.level(), 8);
}

#[test]
fn apply_into_smoke_d5() {
    use crate::scratch::ScratchPool;
    let k = unit_kernel_d5(SMOKE_N_D5);
    let f0 = GridFnND::from_fn(k.grid().clone(), |x: &[f64; 5]| {
        (-x.iter().map(|xi| xi * xi).sum::<f64>()).exp()
    });
    let mut dst = f0.clone();
    let mut pool = ScratchPool::<f64>::new();
    k.apply_into(0.01, &f0, &mut dst, &mut pool).unwrap();
    assert!(
        dst.values.iter().all(|&v| v.is_finite()),
        "apply_into smoke: non-finite output"
    );
}

#[test]
fn f0_equals_identity_smoke() {
    use crate::scratch::ScratchPool;
    // F(0)=I: applying at tau=0 must return the identity.
    let k = unit_kernel_d5(SMOKE_N_D5);
    let one_fn = GridFnND::from_fn(k.grid().clone(), |_| 1.0_f64);
    let mut out = one_fn.clone();
    let mut pool = ScratchPool::<f64>::new();
    k.apply_into(0.0, &one_fn, &mut out, &mut pool).unwrap();
    let sup_err = out
        .values
        .iter()
        .map(|&v| (v - 1.0).abs())
        .fold(0.0_f64, f64::max);
    assert!(sup_err < 1e-10, "F(0)=I: sup_err={sup_err:.3e} ≥ 1e-10");
}
