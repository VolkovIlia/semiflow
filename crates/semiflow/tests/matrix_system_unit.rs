//! Unit tests for `MatrixDiffusionChernoff` and `MatrixGridFn1D` public API.
//! Tests that previously lived in `matrix_system.rs` inline `mod tests`.

use semiflow::{
    ApproximationSubspace, ChernoffFunction, Grid1D, MatrixDiffusionChernoff, MatrixGridFn1D,
    ScratchPool, State,
};

fn make_grid(n: usize) -> Grid1D<f64> {
    Grid1D::new(-5.0, 5.0, n).unwrap()
}

// --- MatrixGridFn1D construction + point accessors ---

#[test]
fn matrix_grid_fn_new_zeroed() {
    let g = make_grid(8);
    let state = MatrixGridFn1D::<f64, 2>::new(g);
    assert!(state.values.iter().all(|&v| v == 0.0));
    assert_eq!(state.len(), 8);
}

#[test]
fn matrix_grid_fn_from_fn_point_view() {
    let g = make_grid(4);
    let state = MatrixGridFn1D::<f64, 2>::from_fn(g, |x| [x, x * x]);
    let v = state.point_view(1);
    let x1 = make_grid(4).x_at(1);
    assert!((v[0] - x1).abs() < 1e-12);
    assert!((v[1] - x1 * x1).abs() < 1e-12);
}

#[test]
fn state_axpy_into_scale_norm() {
    let g = make_grid(4);
    let mut u = MatrixGridFn1D::<f64, 2>::from_fn(g, |_| [1.0, 2.0]);
    let v = MatrixGridFn1D::<f64, 2>::from_fn(g, |_| [3.0, 4.0]);
    u.axpy_into(2.0, &v);
    assert!((u.values[0] - 7.0).abs() < 1e-12); // 1+2*3
    assert!((u.values[1] - 10.0).abs() < 1e-12); // 2+2*4
    let norm = u.norm_sup();
    assert!((norm - 10.0).abs() < 1e-12);
}

// --- matrix_exp_m2: identity and diagonal verified via apply_into (zero C, I=identity) ---

#[test]
fn matrix_exp_identity_via_zero_c() {
    // With C=0, Phase 2 applies exp(0)=I, so apply_into with A=0,B=0,C=0 returns src unchanged.
    let g = make_grid(8);
    let kernel =
        MatrixDiffusionChernoff::<f64, 2>::new(|_, _| {}, |_, _| {}, |_, _| {}, g).unwrap();
    let u0 = MatrixGridFn1D::<f64, 2>::from_fn(g, |x| [x.sin(), x.cos()]);
    let mut u1 = MatrixGridFn1D::<f64, 2>::new(g);
    let mut pool = ScratchPool::<f64>::new();
    // With tau=0, F(0)u = u.
    kernel.apply_into(0.0, &u0, &mut u1, &mut pool).unwrap();
    for k in 0..g.n {
        let v0 = u0.point_view(k);
        let v1 = u1.point_view(k);
        assert!(
            (v0[0] - v1[0]).abs() < 1e-12,
            "component 0 mismatch at k={k}"
        );
        assert!(
            (v0[1] - v1[1]).abs() < 1e-12,
            "component 1 mismatch at k={k}"
        );
    }
}

// --- apply_into M=2: smoke (does not crash, values finite) ---

#[test]
fn apply_into_m2_smoke_finite() {
    let g = make_grid(8);
    let kernel = MatrixDiffusionChernoff::<f64, 2>::new(
        |_, a| {
            a[0][0] = 1.0;
            a[1][1] = 0.5;
        },
        |_, _| {},
        |_, _| {},
        g,
    )
    .unwrap();
    let u0 = MatrixGridFn1D::<f64, 2>::from_fn(g, |x| [(-x * x).exp(), (-x * x / 2.0).exp()]);
    let mut u1 = MatrixGridFn1D::<f64, 2>::new(g);
    let mut pool = ScratchPool::<f64>::new();
    kernel.apply_into(0.01, &u0, &mut u1, &mut pool).unwrap();
    assert!(
        u1.values.iter().all(|v| v.is_finite()),
        "apply_into M=2 non-finite"
    );
}

// --- in_subspace witness ---

#[test]
fn in_subspace_gaussian_ic() {
    let g = make_grid(8);
    let kernel = MatrixDiffusionChernoff::<f64, 2>::new(
        |_, a| {
            a[0][0] = 1.0;
            a[1][1] = 1.0;
        },
        |_, _| {},
        |_, _| {},
        g,
    )
    .unwrap();
    let u0 = MatrixGridFn1D::<f64, 2>::from_fn(g, |x| [(-x * x).exp(); 2]);
    assert!(kernel.in_subspace(&u0));
}

// --- M=5 now works via Padé[13/13] (ADR-0125) ---

#[test]
fn apply_into_m5_pade_smoke() {
    // M=5: Padé[13/13] path. Zero initial condition → zero output (exp(0)·0 = 0).
    let g = make_grid(8);
    let kernel =
        MatrixDiffusionChernoff::<f64, 5>::new(|_, _| {}, |_, _| {}, |_, _| {}, g).unwrap();
    let u0 = MatrixGridFn1D::<f64, 5>::new(g);
    let mut u1 = MatrixGridFn1D::<f64, 5>::new(g);
    let mut pool = ScratchPool::<f64>::new();
    let result = kernel.apply_into(0.01, &u0, &mut u1, &mut pool);
    assert!(result.is_ok(), "M=5 Pade path must not error: {result:?}");
    assert!(
        u1.values.iter().all(|v| v.is_finite()),
        "M=5 Pade output must be finite"
    );
}

// --- M=1: apply_into matches scalar exp for constant solution ---

#[test]
fn apply_into_m1_scalar_matches_scalar_exp() {
    // With A=1, B=0, C=c (constant reaction), u(x)=1 everywhere:
    // Phase 1 yields 1 + tau*(1*0) = 1 (Laplacian of constant = 0).
    // Phase 2 yields exp(tau*c)*1 = exp(tau*c).
    let g = make_grid(8);
    let c_val = 0.5_f64;
    let tau = 0.1_f64;
    let kernel = MatrixDiffusionChernoff::<f64, 1>::new(
        |_, a| {
            a[0][0] = 1.0;
        },
        |_, _| {},
        move |_, c| {
            c[0][0] = c_val;
        },
        g,
    )
    .unwrap();
    let u0 = MatrixGridFn1D::<f64, 1>::from_fn(g, |_| [1.0]);
    let mut u1 = MatrixGridFn1D::<f64, 1>::new(g);
    let mut pool = ScratchPool::<f64>::new();
    kernel.apply_into(tau, &u0, &mut u1, &mut pool).unwrap();
    let expected = (tau * c_val).exp();
    for k in 0..g.n {
        let v = u1.point_view(k);
        assert!(
            (v[0] - expected).abs() < 1e-10,
            "k={k}: got {}, expected {expected}",
            v[0]
        );
    }
}
