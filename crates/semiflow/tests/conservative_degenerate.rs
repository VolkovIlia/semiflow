//! `G_CONS_DEGENERATE` — conservative diffusion admits `k = 0` (ADR-0191, §56.8).
//!
//! Degenerate-at-the-boundary diffusions are the norm in finance rather than an
//! edge case: CEV `k(S) = ½σ²S^{2β}` vanishes at `S = 0`, Feller/CIR
//! `k(v) = ½ξ²v` vanishes at `v = 0`, and Wright–Fisher vanishes at both ends.
//! The harmonic-mean face conductivity already gives the right answer at such a
//! node — the harmonic mean of a conductor and an insulator is an insulator — so
//! the previous `k > 0` guard was rejecting inputs the scheme handles natively,
//! forcing callers to floor `k` at an artificial positive value that moves
//! quantiles when mass sits near the degenerate end.

// `assert_eq!(k[0], 0.0)` is a *bit-exact* claim on purpose — the point of
// each gate is that the datum is genuinely degenerate, not merely small.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use semiflow::{
    conservative::ConservativeDiffusionChernoff,
    conservative_assemble::assemble_conservative_csr_1d,
    symmetric_operator::SymmetricLinearOp,
    BoundaryPolicy, ChernoffFunction, Grid1D, GridFn1D, ScratchPool,
};

fn grid(n: usize, lo: f64, hi: f64) -> Grid1D<f64> {
    Grid1D::new(lo, hi, n).unwrap().with_boundary(BoundaryPolicy::Neumann)
}

/// Feller/CIR variance conductivity `k(v) = ½ξ²v`, degenerate at `v = 0`.
fn cir_k(n: usize, v_max: f64, xi: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let v = v_max * (i as f64) / ((n - 1) as f64);
            0.5 * xi * xi * v
        })
        .collect()
}

/// `⟨u, L_k u⟩` via the assembled CSR carrier `A = −L_k`; must be `≥ 0` (PSD).
fn quad_form(op: &semiflow::symmetric_operator::SymmetricOperator<f64>, u: &[f64]) -> f64 {
    let mut au = vec![0.0; u.len()];
    op.apply_into_slice(u, &mut au);
    u.iter().zip(au.iter()).map(|(a, b)| a * b).sum()
}

/// CIR/Feller: `k(0) = 0` is accepted and the assembled operator stays finite + PSD.
#[test]
fn g_cons_degenerate_cir_accepts_zero_at_the_origin() {
    let n = 64;
    let g = grid(n, 0.0, 1.0);
    let k = cir_k(n, 1.0, 0.7);
    assert_eq!(k[0], 0.0, "datum must actually be degenerate");

    let op = assemble_conservative_csr_1d(g, &k, None, BoundaryPolicy::Neumann)
        .expect("k >= 0 must be accepted");
    let u: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
    let q = quad_form(&op, &u);
    assert!(q.is_finite(), "quadratic form must be finite, got {q}");
    assert!(q >= -1e-12, "A = -L_k must stay PSD, got <u,Au> = {q:.3e}");
}

/// CEV in price space: `k(S) = ½σ²S^{2β}` vanishes at `S = 0`.
#[test]
fn g_cons_degenerate_cev_accepts_zero_at_the_origin() {
    let n = 48;
    let (sigma, beta) = (0.3_f64, 0.7_f64);
    let g = grid(n, 0.0, 200.0);
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let s = 200.0 * (i as f64) / ((n - 1) as f64);
            0.5 * sigma * sigma * s.powf(2.0 * beta)
        })
        .collect();
    assert_eq!(k[0], 0.0);
    assert!(assemble_conservative_csr_1d(g, &k, None, BoundaryPolicy::Neumann).is_ok());
}

/// Wright–Fisher: degenerate at BOTH ends, including two adjacent zeros.
///
/// This is the case that used to be genuinely dangerous. `harmonic_mean(0, 0)`
/// is `0·0/(0+0) = NaN` under bare IEEE arithmetic, and on the
/// `ConservativeDiffusionChernoff → cn_step → thomas_solve` path there was no
/// finiteness backstop to catch it — the state vector would silently go NaN.
#[test]
fn g_cons_degenerate_adjacent_zeros_do_not_produce_nan() {
    let n = 32;
    let g = grid(n, 0.0, 1.0);
    let mut k: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64) / ((n - 1) as f64);
            x * (1.0 - x)
        })
        .collect();
    // Force a genuine adjacent pair of zeros at each end.
    k[0] = 0.0;
    k[1] = 0.0;
    k[n - 1] = 0.0;
    k[n - 2] = 0.0;

    let cd = ConservativeDiffusionChernoff::from_k_array(g, &k, None, BoundaryPolicy::Neumann)
        .expect("adjacent zeros must be accepted");
    let u0: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64) / ((n - 1) as f64);
            (-((x - 0.5) * (x - 0.5)) / 0.02).exp()
        })
        .collect();
    let src = GridFn1D::new(g, u0).unwrap();
    let mut dst = GridFn1D::new(g, vec![0.0; n]).unwrap();
    let mut pool = ScratchPool::<f64>::new();
    cd.apply_into(1e-3, &src, &mut dst, &mut pool).unwrap();
    assert!(
        dst.values.iter().all(|v| v.is_finite()),
        "adjacent zero conductivities produced a non-finite state"
    );
}

/// A zero-conductivity face is an insulator: nothing crosses it.
///
/// Non-vacuity: the same configuration with `k` floored at a small positive
/// value DOES leak across, so the assertion cannot pass by accident.
#[test]
fn g_cons_degenerate_zero_face_blocks_flux() {
    let n = 33;
    let mid = n / 2;
    let g = grid(n, 0.0, 1.0);

    let mut k = vec![1.0_f64; n];
    k[mid] = 0.0; // single insulating node splits the domain in two

    let cd = ConservativeDiffusionChernoff::from_k_array(g, &k, None, BoundaryPolicy::Neumann)
        .unwrap();
    // All mass on the left half.
    let mut u0 = vec![0.0_f64; n];
    for u in u0.iter_mut().take(mid) {
        *u = 1.0;
    }
    let src = GridFn1D::new(g, u0.clone()).unwrap();
    let mut dst = GridFn1D::new(g, vec![0.0; n]).unwrap();
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..200 {
        cd.apply_into(1e-3, &src.clone(), &mut dst, &mut pool).unwrap();
    }
    let right_mass: f64 = dst.values[mid + 1..].iter().sum();
    assert!(
        right_mass.abs() < 1e-12,
        "flux leaked across a zero-conductivity face: right mass = {right_mass:.3e}"
    );

    // Teeth: with k floored positive, the same run DOES transport mass.
    let mut k_floored = vec![1.0_f64; n];
    k_floored[mid] = 0.15;
    let cd2 =
        ConservativeDiffusionChernoff::from_k_array(g, &k_floored, None, BoundaryPolicy::Neumann)
            .unwrap();
    let src2 = GridFn1D::new(g, u0).unwrap();
    let mut dst2 = GridFn1D::new(g, vec![0.0; n]).unwrap();
    cd2.apply_into(0.2, &src2, &mut dst2, &mut pool).unwrap();
    let leaked: f64 = dst2.values[mid + 1..].iter().sum();
    assert!(
        leaked.abs() > 1e-9,
        "teeth check failed: the floored-k control should transport mass, got {leaked:.3e}"
    );
}

/// `k < 0` stays rejected — it breaks the §56.2 PSD energy argument and no
/// `from_csr` check would catch the resulting negative transmissibility.
#[test]
fn g_cons_degenerate_negative_k_still_rejected() {
    let n = 16;
    let g = grid(n, 0.0, 1.0);
    let mut k = vec![1.0_f64; n];
    k[5] = -0.5;
    assert!(
        ConservativeDiffusionChernoff::from_k_array(g, &k, None, BoundaryPolicy::Neumann).is_err(),
        "negative conductivity must remain a DomainViolation"
    );
    assert!(assemble_conservative_csr_1d(g, &k, None, BoundaryPolicy::Neumann).is_err());

    let mut k_nan = vec![1.0_f64; n];
    k_nan[3] = f64::NAN;
    assert!(
        ConservativeDiffusionChernoff::from_k_array(g, &k_nan, None, BoundaryPolicy::Neumann)
            .is_err(),
        "non-finite conductivity must remain a DomainViolation"
    );
}
