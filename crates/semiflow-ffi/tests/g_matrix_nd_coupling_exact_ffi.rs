//! `G_MATRIX_ND_COUPLING_EXACT` (FFI) — ADR-0124 Amendment 1.
//!
//! `smf_matrix2d/3d` solve `∂ₜu = a∂²u + C·u`, `C = [[0, c], [c, 0]]`. On a uniform
//! field with Neumann ends the diffusion part is the identity (up to Thomas rounding),
//! so the result must be `e^{tC}u₀` at every node. Before the fix every axis kernel
//! carried the full `c`, so the 2-D/3-D compositions applied `e^{2tC}` / `e^{3tC}`.

#![allow(unsafe_code)]

use semiflow_ffi::{
    smf_matrix2d_evolve, smf_matrix2d_free, smf_matrix2d_new, smf_matrix3d_evolve,
    smf_matrix3d_free, smf_matrix3d_new, SemiflowStatus, SmfMatrix2D, SmfMatrix3D,
};

const C: f64 = 1.0;
const T: f64 = 0.5;
const U: [f64; 2] = [1.0, 0.5];
const TOL: f64 = 1e-12;

/// `e^{tC}u₀` for `C = [[0, c], [c, 0]]`.
fn exact() -> [f64; 2] {
    let (ch, sh) = ((C * T).cosh(), (C * T).sinh());
    [ch * U[0] + sh * U[1], sh * U[0] + ch * U[1]]
}

fn check(values: &[f64]) {
    let want = exact();
    let scale = U[0].abs().max(U[1].abs());
    for (k, pair) in values.chunks_exact(2).enumerate() {
        for c in 0..2 {
            let err = (pair[c] - want[c]).abs();
            assert!(
                err <= TOL * scale,
                "node {k} comp {c}: {} vs {} (err {err:e})",
                pair[c],
                want[c]
            );
        }
    }
}

fn uniform(nodes: usize) -> Vec<f64> {
    (0..nodes).flat_map(|_| U).collect()
}

#[test]
fn g_matrix_nd_coupling_exact_2d_ffi() {
    let (nx, ny) = (9, 7);
    let u0 = uniform(nx * ny);
    let mut ev: *mut SmfMatrix2D = core::ptr::null_mut();
    let st = unsafe {
        smf_matrix2d_new(
            -1.0,
            1.0,
            nx,
            -1.0,
            1.0,
            ny,
            0.3,
            C,
            u0.as_ptr(),
            u0.len(),
            8,
            &mut ev,
        )
    };
    assert_eq!(st, SemiflowStatus::Ok);
    let mut out = vec![0.0; u0.len()];
    let st = unsafe { smf_matrix2d_evolve(ev, T, out.as_mut_ptr(), out.len()) };
    assert_eq!(st, SemiflowStatus::Ok);
    unsafe { smf_matrix2d_free(ev) };
    check(&out);
}

#[test]
fn g_matrix_nd_coupling_exact_3d_ffi() {
    let (nx, ny, nz) = (7, 6, 5);
    let u0 = uniform(nx * ny * nz);
    let mut ev: *mut SmfMatrix3D = core::ptr::null_mut();
    let st = unsafe {
        smf_matrix3d_new(
            -1.0,
            1.0,
            nx,
            -1.0,
            1.0,
            ny,
            -1.0,
            1.0,
            nz,
            0.3,
            C,
            u0.as_ptr(),
            u0.len(),
            8,
            &mut ev,
        )
    };
    assert_eq!(st, SemiflowStatus::Ok);
    let mut out = vec![0.0; u0.len()];
    let st = unsafe { smf_matrix3d_evolve(ev, T, out.as_mut_ptr(), out.len()) };
    assert_eq!(st, SemiflowStatus::Ok);
    unsafe { smf_matrix3d_free(ev) };
    check(&out);
}
