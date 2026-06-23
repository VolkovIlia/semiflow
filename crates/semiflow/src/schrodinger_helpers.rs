// Helpers for schrodinger.rs — moved here in batch H5 to keep functions ≤50 lines.
// All float operations are verbatim moves; bit-identity is preserved exactly.

use crate::float::SemiflowFloat;
use alloc::vec::Vec;

/// First half-step V-rotation: `src → w[0] (r_d), w[1] (m_d)` (f64).
///
/// Per-node 2×2 rotation by angle `α = v_at_node[i] · half_tau`:
/// ```text
/// w[0][i] =  cos(α)·re_in + sin(α)·im_in
/// w[1][i] = -sin(α)·re_in + cos(α)·im_in
/// ```
/// Writes into pre-zeroed slots `w[0]` and `w[1]` (caller must have resized them to `n`).
#[allow(clippy::needless_range_loop)]
pub(crate) fn strang_first_v_rotation<F: SemiflowFloat>(
    n: usize,
    psi_re: &[F],
    psi_im: &[F],
    v_at_node: &[F],
    half_tau_d: f64,
    w: &mut [Vec<f64>; 12],
) {
    for i in 0..n {
        let re_in = psi_re[i].to_f64().unwrap_or(0.0);
        let im_in = psi_im[i].to_f64().unwrap_or(0.0);
        let alpha = v_at_node[i].to_f64().unwrap_or(0.0) * half_tau_d;
        let c = alpha.cos();
        let s = alpha.sin();
        w[0][i] = c * re_in + s * im_in;
        w[1][i] = -s * re_in + c * im_in;
    }
}

/// Second half-step V-rotation in-place (f64 → F cast).
///
/// Reads `w[0][i]` (`r_i`) and `w[1][i]` (`m_i`), applies the rotation, and
/// writes the result to `dst_re[i]` and `dst_im[i]` cast to `F`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn strang_last_v_rotation_cast<F: SemiflowFloat>(
    n: usize,
    v_at_node: &[F],
    half_tau_d: f64,
    w: &[Vec<f64>; 12],
    dst_re: &mut [F],
    dst_im: &mut [F],
) {
    for i in 0..n {
        let alpha = v_at_node[i].to_f64().unwrap_or(0.0) * half_tau_d;
        let c = alpha.cos();
        let s = alpha.sin();
        let r_i = w[0][i];
        let m_i = w[1][i];
        dst_re[i] = F::from(c * r_i + s * m_i).unwrap_or_else(F::zero);
        dst_im[i] = F::from(-s * r_i + c * m_i).unwrap_or_else(F::zero);
    }
}

/// Initialize band arrays for `pentadiag_solve_f64`.
///
/// Writes main diagonal (w[8]), first super-diagonal (w[9]), second
/// super-diagonal (w[10]), and RHS copy (w[11]).  Boundary rows use
/// `d0_boundary`; interior rows use `d0_interior`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn pentadiag_init_bands(
    n: usize,
    d0_interior: f64,
    d0_boundary: f64,
    d1: f64,
    d2: f64,
    w: &mut [Vec<f64>; 12],
) {
    for i in 0..n {
        w[8][i] = if i == 0 || i == n - 1 {
            d0_boundary
        } else {
            d0_interior
        };
        w[9][i] = d1;
        w[10][i] = d2;
        w[11][i] = w[5][i]; // copy rhs
    }
}

/// Banded forward elimination for the pentadiagonal system.
///
/// Reads and modifies `w[8]` (diag), `w[9]` (sup1), `w[10]` (sup2),
/// `w[11]` (b).  Band width = 2.
///
/// **Correctness note on fill-in**: fill from the k+2 step falls in
/// position (k+2, k+1) — the LOWER triangle — and does NOT update
/// `sup1[k+1]`.  Only the k+1 step updates `sup1[k+1]`.
pub(crate) fn pentadiag_forward_elim(n: usize, w: &mut [Vec<f64>; 12]) {
    for k in 0..n {
        let pivot = w[8][k];
        let u1k = w[9][k];
        let u2k = w[10][k];

        if k + 1 < n {
            let m1 = u1k / pivot;
            w[8][k + 1] -= m1 * u1k;
            if k + 2 < n {
                w[9][k + 1] -= m1 * u2k;
            }
            w[11][k + 1] -= m1 * w[11][k];
        }
        if k + 2 < n {
            let m2 = u2k / pivot;
            w[8][k + 2] -= m2 * u2k;
            w[11][k + 2] -= m2 * w[11][k];
            // Fill (k+2, k+1) is lower-triangle only — sup1[k+1] NOT updated here.
        }
    }
}
