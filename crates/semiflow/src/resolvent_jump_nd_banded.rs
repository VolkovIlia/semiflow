//! Banded LU factorisation helpers for `resolvent_jump_nd` (§47.8).
//!
//! Provides `set_band`, `get_band`, `banded_lu_solve`, `banded_lu_forward`,
//! and `back_substitute_band`. Called by `build_2d_system` / `build_3d_system`
//! and the 2D/3D LHP solvers.

// Grid dimensions (usize) cast to f64/isize/usize for contour/stencil/index computations.
// All values are grid sizes ≪ 2^52 (precision) and ≪ isize::MAX (wrap); sign is checked
// by pre-conditions in the banded LU solver.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;
use alloc::{vec, vec::Vec};

use num_complex::Complex;

use crate::error::SemiflowError;

// ---------------------------------------------------------------------------
// Banded storage helpers
// ---------------------------------------------------------------------------

/// Banded storage: `mat[row * (2*bw + 1) + (bw + col - row)]`.
#[inline]
pub(super) fn set_band(
    mat: &mut [Complex<f64>],
    _n: usize,
    bw: usize,
    row: usize,
    col: usize,
    val: Complex<f64>,
) {
    let width = 2 * bw + 1;
    let off = (bw as isize + col as isize - row as isize) as usize;
    mat[row * width + off] = val;
}

#[inline]
pub(super) fn get_band(mat: &[Complex<f64>], bw: usize, row: usize, col: usize) -> Complex<f64> {
    let width = 2 * bw + 1;
    let off_i = bw as isize + col as isize - row as isize;
    if off_i < 0 || off_i as usize >= width {
        return Complex::new(0.0, 0.0);
    }
    mat[row * width + off_i as usize]
}

// ---------------------------------------------------------------------------
// Banded Gaussian elimination (no partial pivoting; valid off-spectrum)
// ---------------------------------------------------------------------------

/// Banded Gaussian elimination solve `M·x = rhs`, half-bandwidth `bw`.
///
/// Factorisation is `O(N·bw²)`. No pivoting — valid when `λ ∉ σ(A)` (all
/// diagonal entries are non-zero after elimination, as guaranteed by the
/// parabolic contour staying off the negative-real axis).
///
/// # Errors
/// [`SemiflowError::DomainViolation`] if a pivot is near-zero.
// a, l, u, i, j: standard LU factorization names.
#[allow(clippy::many_single_char_names)]
pub(super) fn banded_lu_solve(
    mat: &[Complex<f64>],
    bw: usize,
    n: usize,
    rhs: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, SemiflowError> {
    let width = 2 * bw + 1;
    let mut a = mat.to_vec();
    let mut b = rhs.to_vec();
    for k in 0..n {
        let pivot = a[k * width + bw];
        if pivot.norm() < 1e-300 {
            return Err(SemiflowError::DomainViolation {
                what: "banded_lu_solve: pivot near zero (λ on spectrum)",
                value: pivot.norm(),
            });
        }
        banded_lu_forward(&mut a, &mut b, bw, width, n, k, pivot);
    }
    back_substitute_band(&a, &b, bw, n, width)
}

/// Forward elimination step for row `k` of the banded LU factorisation.
///
/// Eliminates column `k` from all sub-diagonal rows `[k+1, k+bw]`.
/// Caller guarantees `pivot.norm() >= 1e-300`.
// a, b, k, row, col: standard LU names; pivot passed in to avoid recomputing.
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
fn banded_lu_forward(
    a: &mut [Complex<f64>],
    b: &mut [Complex<f64>],
    bw: usize,
    width: usize,
    n: usize,
    k: usize,
    pivot: Complex<f64>,
) {
    let pivot_inv = Complex::new(1.0, 0.0) / pivot;
    let k_end = (k + bw + 1).min(n);
    for row in (k + 1)..k_end {
        let off_i = bw as isize + k as isize - row as isize;
        if off_i < 0 {
            break;
        }
        let off = off_i as usize;
        let factor = a[row * width + off] * pivot_inv;
        if factor.norm() < 1e-300 {
            continue;
        }
        a[row * width + off] = Complex::new(0.0, 0.0);
        for col in (k + 1)..k_end {
            let src_off_i = bw as isize + col as isize - k as isize;
            if src_off_i < 0 || src_off_i as usize >= width {
                continue;
            }
            let src = a[k * width + src_off_i as usize];
            let dst_off_i = bw as isize + col as isize - row as isize;
            if dst_off_i < 0 || dst_off_i as usize >= width {
                continue;
            }
            a[row * width + dst_off_i as usize] -= factor * src;
        }
        let bk = b[k];
        b[row] -= factor * bk;
    }
}

/// Banded back substitution after in-place LU factorisation.
// a, b, k, s, x: standard banded LU back-substitution names.
#[allow(clippy::needless_range_loop, clippy::many_single_char_names)]
pub(super) fn back_substitute_band(
    a: &[Complex<f64>],
    b: &[Complex<f64>],
    bw: usize,
    n: usize,
    width: usize,
) -> Result<Vec<Complex<f64>>, SemiflowError> {
    let mut x = vec![Complex::new(0.0, 0.0); n];
    for k in (0..n).rev() {
        let mut s = b[k];
        for col in (k + 1)..(k + bw + 1).min(n) {
            s -= get_band(a, bw, k, col) * x[col];
        }
        let diag = a[k * width + bw];
        if diag.norm() < 1e-300 {
            return Err(SemiflowError::DomainViolation {
                what: "banded_lu_solve: back-sub diagonal near zero",
                value: diag.norm(),
            });
        }
        x[k] = s / diag;
    }
    Ok(x)
}
