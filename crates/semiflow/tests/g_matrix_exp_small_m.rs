//! `G_MATRIX_EXP_SMALL_M` (`RELEASE_BLOCKING`): the per-point matrix exponential
//! of `MatrixDiffusionChernoff<F, M>` is exact for `M ∈ {2, 3, 4}`.
//!
//! With `A = B = 0` the block Crank–Nicolson diffusion phase is the identity, so
//! one Strang step is `e^{τC/2}·e^{τC/2}` and `n` steps give `e^{tC}·u₀` exactly
//! (up to rounding). The oracle is closed form:
//!
//! - `M = 2`, rotation `C = [[0, ω], [−ω, 0]]`: `e^{tC} = [[cos ωt, sin ωt], [−sin ωt, cos ωt]]`.
//!   Complex eigenvalues — the old Putzer branch took `√|disc|` and returned
//!   `cosh/sinh` instead of `cos/sin`.
//! - `M = 2`, nearly repeated eigenvalues `C = [[λ, 1], [ε², λ]]`: `δ = ε²`,
//!   `e^{tC} = e^{λt}[cosh(εt)·I + sinh(εt)/ε·(C − λI)]` evaluated in closed form.
//! - `M = 3, 4`: `C = Q·diag(μ)·Qᵀ` with an explicit orthogonal `Q`, so
//!   `e^{tC} = Q·diag(e^{μt})·Qᵀ`. The old Taylor helper summed `Bᵈ/d`, not `Bᵈ/d!`.
//!
//! Threshold: relative sup error `≤ 1e-13` over every grid point.

use semiflow::{ChernoffSemigroup, Grid1D, MatrixDiffusionChernoff, MatrixGridFn1D};

const T: f64 = 1.3;
const N_STEPS: usize = 7;
const TOL: f64 = 1e-13;

fn evolve_const_c<const M: usize>(c: [[f64; M]; M], u0: [f64; M]) -> Vec<[f64; M]> {
    let grid = Grid1D::new(0.0, 1.0, 9).expect("grid");
    let eng =
        MatrixDiffusionChernoff::<f64, M>::new(|_, _| {}, |_, _| {}, move |_, out| *out = c, grid)
            .expect("engine");
    let state = MatrixGridFn1D::<f64, M>::from_fn(grid, |_| u0);
    let semi = ChernoffSemigroup::new(eng, N_STEPS).expect("n >= 1");
    let out = semi.evolve(T, &state).expect("evolve");
    (0..grid.n).map(|k| out.point_view(k)).collect()
}

fn assert_close<const M: usize>(label: &str, got: &[[f64; M]], want: [f64; M]) {
    let scale = want.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    for (k, g) in got.iter().enumerate() {
        for i in 0..M {
            let err = (g[i] - want[i]).abs();
            assert!(
                err <= TOL * scale,
                "{label}: point {k} comp {i}: got {} want {} (err {err:e})",
                g[i],
                want[i]
            );
        }
    }
}

/// `Q·diag(e^{μt})·Qᵀ·u₀` for a row-major orthogonal `Q`.
fn spectral_apply<const M: usize>(q: [[f64; M]; M], mu: [f64; M], u0: [f64; M]) -> [f64; M] {
    let mut w = [0.0; M];
    for j in 0..M {
        let qt_u: f64 = (0..M).map(|i| q[i][j] * u0[i]).sum();
        w[j] = (mu[j] * T).exp() * qt_u;
    }
    let mut out = [0.0; M];
    for i in 0..M {
        out[i] = (0..M).map(|j| q[i][j] * w[j]).sum();
    }
    out
}

/// `Q·diag(μ)·Qᵀ`.
fn spectral_matrix<const M: usize>(q: [[f64; M]; M], mu: [f64; M]) -> [[f64; M]; M] {
    let mut c = [[0.0; M]; M];
    for i in 0..M {
        for j in 0..M {
            c[i][j] = (0..M).map(|k| q[i][k] * mu[k] * q[j][k]).sum();
        }
    }
    c
}

#[test]
fn g_matrix_exp_small_m_rotation_m2() {
    let w = 0.9;
    let u0 = [0.7, -0.4];
    let got = evolve_const_c::<2>([[0.0, w], [-w, 0.0]], u0);
    let (s, c) = (w * T).sin_cos();
    let want = [c * u0[0] + s * u0[1], -s * u0[0] + c * u0[1]];
    assert_close("rotation M=2", &got, want);
}

#[test]
fn g_matrix_exp_small_m_near_repeated_m2() {
    let (lam, eps) = (-0.6, 1e-5);
    let u0 = [1.1, 0.3];
    let got = evolve_const_c::<2>([[lam, 1.0], [eps * eps, lam]], u0);
    let (ch, sh) = ((eps * T).cosh(), (eps * T).sinh() / eps);
    let e = (lam * T).exp();
    // (C − λI) = [[0, 1], [ε², 0]]
    let want = [
        e * (ch * u0[0] + sh * u0[1]),
        e * (sh * eps * eps * u0[0] + ch * u0[1]),
    ];
    assert_close("near-repeated M=2", &got, want);
}

#[test]
fn g_matrix_exp_small_m_m3() {
    let s3 = 3.0_f64.sqrt();
    let s2 = 2.0_f64.sqrt();
    let s6 = 6.0_f64.sqrt();
    let q = [
        [1.0 / s3, 1.0 / s2, 1.0 / s6],
        [1.0 / s3, -1.0 / s2, 1.0 / s6],
        [1.0 / s3, 0.0, -2.0 / s6],
    ];
    let mu = [-0.2, 0.5, -1.4];
    let u0 = [0.3, -1.0, 0.8];
    let got = evolve_const_c::<3>(spectral_matrix(q, mu), u0);
    assert_close("orthogonal M=3", &got, spectral_apply(q, mu, u0));
}

#[test]
fn g_matrix_exp_small_m_m4() {
    let h = 0.5;
    let q = [[h, h, h, h], [h, -h, h, -h], [h, h, -h, -h], [h, -h, -h, h]];
    let mu = [0.4, -0.9, -2.5, 0.05];
    let u0 = [1.0, 0.2, -0.6, 0.9];
    let got = evolve_const_c::<4>(spectral_matrix(q, mu), u0);
    assert_close("Hadamard M=4", &got, spectral_apply(q, mu, u0));
}

// ---------------------------------------------------------------------------
// Complex coefficients (`MatrixDiffusionChernoffComplex`)
// ---------------------------------------------------------------------------

use num_complex::Complex64 as C64;
use semiflow::{MatrixDiffusionChernoffComplex, MatrixGridFnComplex1D};

fn evolve_const_c_complex<const M: usize>(c: [[C64; M]; M], u0: [C64; M]) -> Vec<[C64; M]> {
    evolve_const_c_complex_n(c, u0, N_STEPS)
}

fn evolve_const_c_complex_n<const M: usize>(
    c: [[C64; M]; M],
    u0: [C64; M],
    n_steps: usize,
) -> Vec<[C64; M]> {
    let grid = Grid1D::new(0.0, 1.0, 9).expect("grid");
    let eng = MatrixDiffusionChernoffComplex::<C64, M>::new(
        |_, _| {},
        |_, _| {},
        move |_, out| *out = c,
        grid,
    )
    .expect("engine");
    let state = MatrixGridFnComplex1D::<C64, M>::from_fn(grid, |_| u0);
    let semi = ChernoffSemigroup::new(eng, n_steps).expect("n >= 1");
    let out = semi.evolve(T, &state).expect("evolve");
    (0..grid.n).map(|k| out.point_view(k)).collect()
}

fn assert_close_c<const M: usize>(label: &str, got: &[[C64; M]], want: [C64; M]) {
    let scale = want.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
    for (k, g) in got.iter().enumerate() {
        for i in 0..M {
            let err = (g[i] - want[i]).norm();
            assert!(
                err <= TOL * scale,
                "{label}: point {k} comp {i}: got {} want {} (err {err:e})",
                g[i],
                want[i]
            );
        }
    }
}

/// `C = −iH`, `H = h₀I + h·σ` Hermitian: `e^{tC} = e^{−ih₀t}[cos(|h|t)·I − i·sin(|h|t)·(h·σ)/|h|]`.
#[test]
fn g_matrix_exp_small_m_complex_unitary_m2() {
    let (h0, hx, hy, hz) = (0.3, 0.5, -0.3, 0.8);
    let i = C64::new(0.0, 1.0);
    // h·σ = [[hz, hx − i·hy], [hx + i·hy, −hz]]
    let hs = [
        [C64::new(hz, 0.0), C64::new(hx, -hy)],
        [C64::new(hx, hy), C64::new(-hz, 0.0)],
    ];
    let mut c = [[C64::new(0.0, 0.0); 2]; 2];
    for r in 0..2 {
        for q in 0..2 {
            let h = hs[r][q]
                + if r == q {
                    C64::new(h0, 0.0)
                } else {
                    C64::new(0.0, 0.0)
                };
            c[r][q] = -i * h;
        }
    }
    let u0 = [C64::new(0.6, 0.1), C64::new(-0.2, 0.7)];
    let got = evolve_const_c_complex::<2>(c, u0);
    let norm_h = (hx * hx + hy * hy + hz * hz).sqrt();
    let phase = (-i * h0 * T).exp();
    let (s, co) = (norm_h * T).sin_cos();
    let mut want = [C64::new(0.0, 0.0); 2];
    for r in 0..2 {
        want[r] = phase * (co * u0[r] - i * s / norm_h * (hs[r][0] * u0[0] + hs[r][1] * u0[1]));
    }
    assert_close_c("complex unitary M=2", &got, want);
}

#[test]
fn g_matrix_exp_small_m_complex_m4() {
    complex_m4_case(1.0, N_STEPS);
}

/// One step with `‖τC/2‖ ≈ 3`: after scaling the largest ENTRY to `≤ 1` the old
/// degree-12 Taylor helper still saw a row sum near 4 and truncated at `≈ 4¹³/13!`.
#[test]
fn g_matrix_exp_small_m_complex_m4_large_argument() {
    complex_m4_case(3.0, 1);
}

fn complex_m4_case(scale: f64, n_steps: usize) {
    let h = 0.5;
    let q = [[h, h, h, h], [h, -h, h, -h], [h, h, -h, -h], [h, -h, -h, h]];
    let mu = [
        C64::new(-0.4, 1.1) * scale,
        C64::new(0.2, -0.7) * scale,
        C64::new(-1.5, 0.3) * scale,
        C64::new(0.05, 2.0) * scale,
    ];
    let mut c = [[C64::new(0.0, 0.0); 4]; 4];
    for r in 0..4 {
        for s in 0..4 {
            c[r][s] = (0..4).map(|k| q[r][k] * mu[k] * q[s][k]).sum();
        }
    }
    let u0 = [
        C64::new(1.0, 0.0),
        C64::new(0.2, -0.3),
        C64::new(-0.6, 0.1),
        C64::new(0.0, 0.9),
    ];
    let got = evolve_const_c_complex_n::<4>(c, u0, n_steps);
    let mut w = [C64::new(0.0, 0.0); 4];
    for j in 0..4 {
        let qt_u: C64 = (0..4).map(|r| q[r][j] * u0[r]).sum();
        w[j] = (mu[j] * T).exp() * qt_u;
    }
    let mut want = [C64::new(0.0, 0.0); 4];
    for r in 0..4 {
        want[r] = (0..4).map(|j| q[r][j] * w[j]).sum();
    }
    assert_close_c("complex spectral M=4", &got, want);
}
