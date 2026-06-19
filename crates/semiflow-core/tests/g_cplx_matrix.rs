//! `G_CPLX_MATRIX` — complex matrix-exponential accuracy gate (ADR-0128).
//!
//! Gate: relative Frobenius error ≤ 1e-12 for the complex Padé[13/13] path,
//! tested via the Phase 1/3 reaction-half-step of `MatrixDiffusionChernoffComplex`.
//!
//! Test strategy:
//! - Set `a_ij` = 0, `b_ij` = 0 (zero diffusion): the kernel reduces to
//!   `u^{n+1}_k = exp(τ C(x_k)) · u^n_k` at each grid point k.
//! - Compare against a high-degree Taylor series reference (degree 60, at
//!   small ‖τC‖_∞ ≤ 0.01 where Taylor converges to machine precision).
//! - Unitarity: set `c_ij` = i·H(x) for Hermitian H; check ‖`UᴴU` − I‖_F ≤ 1e-12.
//!
//! M ∈ {5, 6, 8} (Padé[13/13] path, ADR-0128).
//!
//! ADR-0128; contracts/semiflow-core.math.md §33.8 Para 3.

#![cfg(feature = "slow-tests")]

use num_complex::Complex;
use semiflow_core::{
    ChernoffFunction, Grid1D, MatrixDiffusionChernoffComplex, MatrixGridFnComplex1D, ScratchPool,
};

type C64 = Complex<f64>;

// ---------------------------------------------------------------------------
// Minimal PCG-64 (no external deps)
// ---------------------------------------------------------------------------

struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    fn new(seed: u128) -> Self {
        let inc = 0xda3e_39cb_94b9_5bdb_da3e_39cb_94b9_5bdb_u128 | 1;
        let mut rng = Self { state: 0, inc };
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u64();
        rng
    }

    fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.state = old
            .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645_u128)
            .wrapping_add(self.inc);
        (((old >> 64) ^ old) as u64).rotate_right(((old >> 122) as u32) & 63)
    }

    fn next_f64(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) | 0x3FF0_0000_0000_0000_u64;
        f64::from_bits(bits) - 1.0
    }
}

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

fn frob_norm_c<const M: usize>(a: &[[C64; M]; M]) -> f64 {
    a.iter()
        .flat_map(|r| r.iter())
        .map(|z| z.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn frob_diff_c<const M: usize>(a: &[[C64; M]; M], b: &[[C64; M]; M]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..M {
        for j in 0..M {
            let d = a[i][j] - b[i][j];
            s += d.norm_sqr();
        }
    }
    s.sqrt()
}

fn inf_norm_c<const M: usize>(a: &[[C64; M]; M]) -> f64 {
    a.iter()
        .map(|row| row.iter().map(|z| z.norm()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Scale to ‖result‖_∞ = target.
fn scale_c<const M: usize>(a: &[[C64; M]; M], target: f64) -> [[C64; M]; M] {
    let n = inf_norm_c::<M>(a);
    let mut out = *a;
    if n > 1e-300 {
        let s = target / n;
        for row in &mut out {
            for v in row.iter_mut() {
                *v = C64::new(v.re * s, v.im * s);
            }
        }
    }
    out
}

/// Degree-60 Taylor (no squaring) — reference for small ‖Z‖ ≤ 0.01.
fn cmat_exp_taylor60<const M: usize>(a: &[[C64; M]; M]) -> [[C64; M]; M] {
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    let mut result = [[zero; M]; M];
    for i in 0..M {
        result[i][i] = one;
    }
    let mut term = result;
    for d in 1u32..=60 {
        let mut t2 = [[zero; M]; M];
        for i in 0..M {
            for k in 0..M {
                for j in 0..M {
                    t2[i][j] += term[i][k] * a[k][j];
                }
            }
        }
        let inv_d = one / C64::new(f64::from(d), 0.0);
        for i in 0..M {
            for j in 0..M {
                t2[i][j] *= inv_d;
            }
        }
        term = t2;
        for i in 0..M {
            for j in 0..M {
                result[i][j] += term[i][j];
            }
        }
    }
    result
}

fn cmat_conj_transpose<const M: usize>(a: &[[C64; M]; M]) -> [[C64; M]; M] {
    let zero = C64::new(0.0, 0.0);
    let mut out = [[zero; M]; M];
    for i in 0..M {
        for j in 0..M {
            out[i][j] = a[j][i].conj();
        }
    }
    out
}

fn cmat_mul<const M: usize>(a: &[[C64; M]; M], b: &[[C64; M]; M]) -> [[C64; M]; M] {
    let zero = C64::new(0.0, 0.0);
    let mut c = [[zero; M]; M];
    for i in 0..M {
        for k in 0..M {
            for j in 0..M {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Core accuracy probe:
// Zero diffusion + reaction C → exp(τC) acts pointwise on each grid point.
// Extract the resulting M×M matrix by feeding unit basis vectors.
// ---------------------------------------------------------------------------

/// Compute exp(tau * C_fixed) using MatrixDiffusionChernoffComplex apply_into
/// (zero diffusion, constant reaction). Returns the implied M×M matrix.
fn extract_exp_via_kernel<const M: usize>(c_fixed: [[C64; M]; M], tau: f64) -> [[C64; M]; M] {
    let n = 8usize; // small grid; all points see same constant C_fixed.
    let grid = Grid1D::<f64>::new(-1.0, 1.0, n).expect("grid");

    let kernel = MatrixDiffusionChernoffComplex::<C64, M>::new(
        |_, a| {
            for row in a.iter_mut() {
                for v in row.iter_mut() {
                    *v = C64::new(0.0, 0.0);
                }
            }
        },
        |_, b| {
            for row in b.iter_mut() {
                for v in row.iter_mut() {
                    *v = C64::new(0.0, 0.0);
                }
            }
        },
        move |_, c| *c = c_fixed,
        grid,
    )
    .expect("kernel construction");

    // To extract exp(τC) column j: feed e_j (unit basis vector) at all grid points.
    let mut result = [[C64::new(0.0, 0.0); M]; M];
    let mut scratch = ScratchPool::<f64>::new();

    for col in 0..M {
        let mut src = MatrixGridFnComplex1D::<C64, M>::new(grid);
        let mut dst = MatrixGridFnComplex1D::<C64, M>::new(grid);
        // Load e_col at each grid point.
        for k in 0..n {
            let mut v = [C64::new(0.0, 0.0); M];
            v[col] = C64::new(1.0, 0.0);
            src.set_point(k, &v);
        }
        kernel
            .apply_into(tau, &src, &mut dst, &mut scratch)
            .unwrap();
        // Extract column: dst at grid point 0 (constant C, same at all k).
        let v0 = dst.point_view(0);
        for row in 0..M {
            result[row][col] = v0[row];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Test: accuracy vs high-degree Taylor (small norm)
// ---------------------------------------------------------------------------

fn run_accuracy_check<const M: usize>(seed: u128) -> f64 {
    let mut rng = Pcg64::new(seed);
    let mut worst = 0.0f64;
    for _ in 0..20 {
        // Random non-Hermitian complex matrix, scaled to small ∞-norm.
        let mut raw = [[C64::new(0.0, 0.0); M]; M];
        for i in 0..M {
            for j in 0..M {
                let re = rng.next_f64() * 2.0 - 1.0;
                let im = rng.next_f64() * 2.0 - 1.0;
                raw[i][j] = C64::new(re, im);
            }
        }
        // Scale so ‖C_fixed‖_∞ = 0.01, tau = 1.0 → ‖τC‖_∞ = 0.01
        let c_fixed = scale_c::<M>(&raw, 0.01);
        let tau = 1.0f64;

        let ref_val = cmat_exp_taylor60::<M>(&c_fixed);
        let pade_val = extract_exp_via_kernel::<M>(c_fixed, tau);

        let fref = frob_norm_c::<M>(&ref_val);
        let rel = frob_diff_c::<M>(&pade_val, &ref_val) / fref.max(1e-300);
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Test: unitarity exp(iH) for Hermitian H
// ---------------------------------------------------------------------------

fn run_unitarity_check<const M: usize>(seed: u128) -> f64 {
    let mut rng = Pcg64::new(seed);
    let zero = C64::new(0.0, 0.0);
    let mut worst = 0.0f64;
    for _ in 0..20 {
        // Random Hermitian matrix scaled to ‖H‖_∞ = 5.0.
        let mut h = [[zero; M]; M];
        for i in 0..M {
            h[i][i] = C64::new(rng.next_f64() * 2.0 - 1.0, 0.0);
            for j in (i + 1)..M {
                let re = rng.next_f64() * 2.0 - 1.0;
                let im = rng.next_f64() * 2.0 - 1.0;
                h[i][j] = C64::new(re, im);
                h[j][i] = C64::new(re, -im);
            }
        }
        let h = scale_c::<M>(&h, 5.0);
        // iH: c_ij = i * h_ij
        let ih: [[C64; M]; M] = {
            let mut m = [[zero; M]; M];
            for i in 0..M {
                for j in 0..M {
                    m[i][j] = C64::new(-h[i][j].im, h[i][j].re);
                }
            }
            m
        };
        let u = extract_exp_via_kernel::<M>(ih, 1.0);
        let uh = cmat_conj_transpose::<M>(&u);
        let prod = cmat_mul::<M>(&uh, &u);
        // ‖prod - I‖_F
        let mut diff = [[zero; M]; M];
        for i in 0..M {
            for j in 0..M {
                diff[i][j] = prod[i][j];
            }
            diff[i][i] -= C64::new(1.0, 0.0);
        }
        let err = frob_norm_c::<M>(&diff);
        if err > worst {
            worst = err;
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Gate tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn g_cplx_matrix_accuracy_m5() {
    const M: usize = 5;
    let err = run_accuracy_check::<M>(0xC0FF_EE_01);
    println!("G_CPLX_MATRIX M={M} rel-err (Taylor60 ref): {err:.3e}");
    assert!(err <= 1e-12, "M={M}: rel-err {err:.3e} > 1e-12");
}

#[test]
#[ignore]
fn g_cplx_matrix_accuracy_m6() {
    const M: usize = 6;
    let err = run_accuracy_check::<M>(0xC0FF_EE_02);
    println!("G_CPLX_MATRIX M={M} rel-err (Taylor60 ref): {err:.3e}");
    assert!(err <= 1e-12, "M={M}: rel-err {err:.3e} > 1e-12");
}

#[test]
#[ignore]
fn g_cplx_matrix_accuracy_m8() {
    const M: usize = 8;
    let err = run_accuracy_check::<M>(0xC0FF_EE_03);
    println!("G_CPLX_MATRIX M={M} rel-err (Taylor60 ref): {err:.3e}");
    assert!(err <= 1e-12, "M={M}: rel-err {err:.3e} > 1e-12");
}

#[test]
#[ignore]
fn g_cplx_matrix_unitarity_m5() {
    const M: usize = 5;
    let err = run_unitarity_check::<M>(0xDEAD_01);
    println!("G_CPLX_MATRIX M={M} unitarity ‖UᴴU−I‖_F: {err:.3e}");
    assert!(err <= 1e-12, "M={M}: unitarity drift {err:.3e} > 1e-12");
}

#[test]
#[ignore]
fn g_cplx_matrix_unitarity_m6() {
    const M: usize = 6;
    let err = run_unitarity_check::<M>(0xDEAD_02);
    println!("G_CPLX_MATRIX M={M} unitarity ‖UᴴU−I‖_F: {err:.3e}");
    assert!(err <= 1e-12, "M={M}: unitarity drift {err:.3e} > 1e-12");
}

#[test]
#[ignore]
fn g_cplx_matrix_unitarity_m8() {
    const M: usize = 8;
    let err = run_unitarity_check::<M>(0xDEAD_03);
    println!("G_CPLX_MATRIX M={M} unitarity ‖UᴴU−I‖_F: {err:.3e}");
    assert!(err <= 1e-12, "M={M}: unitarity drift {err:.3e} > 1e-12");
}
