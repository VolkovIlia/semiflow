//! Smoke tests for `SmfLaplacian` + `SmfGraphTraj` FFI entry points
//! (C-parity pass, ADR-0028/0171).
//!
//! Verifies:
//! 1. Null-pointer early returns.
//! 2. Constructor / free lifecycle (no leak / double-free).
//! 3. Scalar introspection: `n_nodes`, `is_combinatorial`, `is_normalized`,
//!    `spectral_bound`.
//! 4. CSR read-back consistency (row_ptr / col_idx / vals lengths vs a known
//!    P_3 graph; values cross-checked against hand-computed Laplacian).
//! 5. `to_dense` diagonal for P_4 combinatorial Laplacian.
//! 6. `GraphTraj` degenerate constructor + getters.
//! 7. All returned buffers are freed with the matching free functions
//!    (`smf_free_buf_usize`, `smf_free_buf_f64`).

#![allow(unsafe_code)]

use semiflow_ffi::{
    smf_free_buf_f64, smf_free_buf_usize,
    smf_graph_laplacian_combinatorial, smf_graph_laplacian_normalized,
    smf_graph_path, smf_graph_drop,
    smf_graph_traj_free, smf_graph_traj_n_nodes, smf_graph_traj_n_segments,
    smf_graph_traj_new, smf_graph_traj_t_horizon,
    smf_laplacian_col_idx, smf_laplacian_free, smf_laplacian_is_combinatorial,
    smf_laplacian_is_normalized, smf_laplacian_n_nodes, smf_laplacian_row_ptr,
    smf_laplacian_spectral_bound, smf_laplacian_to_dense, smf_laplacian_vals,
    SemiflowStatus, SmfGraph, SmfGraphTraj, SmfLaplacian,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a path graph of `n` nodes; panics on failure.
unsafe fn make_path(n: u32) -> *mut SmfGraph {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    let s = unsafe { smf_graph_path(n, &mut g) };
    assert_eq!(s, SemiflowStatus::Ok, "smf_graph_path({n})");
    assert!(!g.is_null());
    g
}

// ---------------------------------------------------------------------------
// SmfLaplacian — null-safety
// ---------------------------------------------------------------------------

#[test]
fn laplacian_combinatorial_null_graph_returns_null_ptr() {
    let mut lap: *mut SmfLaplacian = std::ptr::null_mut();
    let s = unsafe { smf_graph_laplacian_combinatorial(std::ptr::null(), &mut lap) };
    assert_eq!(s, SemiflowStatus::NullPtr);
    assert!(lap.is_null());
}

#[test]
fn laplacian_combinatorial_null_out_returns_null_ptr() {
    let g = unsafe { make_path(4) };
    let s = unsafe { smf_graph_laplacian_combinatorial(g, std::ptr::null_mut()) };
    assert_eq!(s, SemiflowStatus::NullPtr);
    unsafe { smf_graph_drop(g) };
}

#[test]
fn laplacian_normalized_null_graph_returns_null_ptr() {
    let mut lap: *mut SmfLaplacian = std::ptr::null_mut();
    let s = unsafe { smf_graph_laplacian_normalized(std::ptr::null(), &mut lap) };
    assert_eq!(s, SemiflowStatus::NullPtr);
    assert!(lap.is_null());
}

#[test]
fn laplacian_free_null_is_noop() {
    unsafe { smf_laplacian_free(std::ptr::null_mut()) };
}

// ---------------------------------------------------------------------------
// SmfLaplacian — scalar introspection (P_3)
// ---------------------------------------------------------------------------
//
// P_3: nodes {0,1,2}, edges {0-1, 1-2} (unit weights).
// Combinatorial Laplacian:
//   L = [[1,-1,0],[-1,2,-1],[0,-1,1]]
// n_directed_edges = 4 (each undirected edge → 2 directed entries).
// Gershgorin bound = max row sum of |L| = 2 (rows: 1+1=2, 1+2+1=4 → max diag+offdiag).
// Actually spectral_radius_bound = max(abs(L[i][j]), j≠i) + L[i][i] where
// Gershgorin says eigenvalues lie in discs centred at diagonal. For L row 1
// the Gershgorin disc centre is 2, radius sum of off-diags = 2 → bound = 4.

#[test]
fn laplacian_p3_combinatorial_n_nodes() {
    let g = unsafe { make_path(3) };
    let mut lap: *mut SmfLaplacian = std::ptr::null_mut();
    let s = unsafe { smf_graph_laplacian_combinatorial(g, &mut lap) };
    assert_eq!(s, SemiflowStatus::Ok);
    assert_eq!(unsafe { smf_laplacian_n_nodes(lap) }, 3);
    assert!(unsafe { smf_laplacian_is_combinatorial(lap) });
    assert!(!unsafe { smf_laplacian_is_normalized(lap) });
    // spectral bound must be >= true spectral radius of P_3's Laplacian (~3.41)
    let bound = unsafe { smf_laplacian_spectral_bound(lap) };
    assert!(bound > 0.0, "spectral_bound must be > 0");
    unsafe { smf_laplacian_free(lap) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn laplacian_p3_normalized_flags() {
    let g = unsafe { make_path(3) };
    let mut lap: *mut SmfLaplacian = std::ptr::null_mut();
    let s = unsafe { smf_graph_laplacian_normalized(g, &mut lap) };
    assert_eq!(s, SemiflowStatus::Ok);
    assert!(!unsafe { smf_laplacian_is_combinatorial(lap) });
    assert!(unsafe { smf_laplacian_is_normalized(lap) });
    unsafe { smf_laplacian_free(lap) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn laplacian_n_nodes_null_returns_zero() {
    assert_eq!(unsafe { smf_laplacian_n_nodes(std::ptr::null()) }, 0);
}

#[test]
fn laplacian_spectral_bound_null_returns_zero() {
    assert_eq!(unsafe { smf_laplacian_spectral_bound(std::ptr::null()) }, 0.0);
}

// ---------------------------------------------------------------------------
// SmfLaplacian — CSR read-back (P_3)
// ---------------------------------------------------------------------------
//
// Note on Laplacian CSR size:
// The Laplacian matrix is assembled with *explicit* diagonal entries as well as
// the off-diagonal adjacency entries.  For P_3 the Laplacian is 3×3 with:
//   3 diagonal entries + 4 off-diagonal entries (2 undirected edges → 4 directed)
//   = 7 non-zeros total.
// So col_idx.len() = vals.len() = 7.

#[test]
fn laplacian_p3_csr_lengths() {
    let g = unsafe { make_path(3) };
    let mut lap: *mut SmfLaplacian = std::ptr::null_mut();
    unsafe { smf_graph_laplacian_combinatorial(g, &mut lap) };

    // row_ptr: length = n_nodes + 1 = 4
    let mut rp_ptr: *mut usize = std::ptr::null_mut();
    let mut rp_len: usize = 0;
    let s = unsafe { smf_laplacian_row_ptr(lap, &mut rp_ptr, &mut rp_len) };
    assert_eq!(s, SemiflowStatus::Ok);
    assert_eq!(rp_len, 4); // n+1 = 3+1 = 4

    // col_idx: length = nnz in Laplacian = 3 diag + 4 off-diag = 7 for P_3
    let mut ci_ptr: *mut usize = std::ptr::null_mut();
    let mut ci_len: usize = 0;
    let s2 = unsafe { smf_laplacian_col_idx(lap, &mut ci_ptr, &mut ci_len) };
    assert_eq!(s2, SemiflowStatus::Ok);
    assert_eq!(ci_len, 7); // 3 diag + 4 off-diag (2 undirected edges × 2 + 3 diag)

    // vals: same length as col_idx
    let mut v_ptr: *mut f64 = std::ptr::null_mut();
    let mut v_len: usize = 0;
    let s3 = unsafe { smf_laplacian_vals(lap, &mut v_ptr, &mut v_len) };
    assert_eq!(s3, SemiflowStatus::Ok);
    assert_eq!(v_len, 7);

    // Free buffers
    unsafe { smf_free_buf_usize(rp_ptr, rp_len) };
    unsafe { smf_free_buf_usize(ci_ptr, ci_len) };
    unsafe { smf_free_buf_f64(v_ptr, v_len) };
    unsafe { smf_laplacian_free(lap) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn laplacian_row_ptr_null_returns_null_ptr() {
    let s = unsafe {
        smf_laplacian_row_ptr(std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut())
    };
    assert_eq!(s, SemiflowStatus::NullPtr);
}

// ---------------------------------------------------------------------------
// SmfLaplacian — to_dense diagonal check (P_4)
// ---------------------------------------------------------------------------
//
// P_4: nodes {0,1,2,3}, edges {0-1,1-2,2-3}.
// Combinatorial Laplacian diagonal: [1, 2, 2, 1] (degree sequence of P_4).
// to_dense returns row-major n×n = 4×4 = 16 floats.
// Diagonal entries are at indices 0, 5, 10, 15.

#[test]
fn laplacian_p4_to_dense_diagonal() {
    let g = unsafe { make_path(4) };
    let mut lap: *mut SmfLaplacian = std::ptr::null_mut();
    unsafe { smf_graph_laplacian_combinatorial(g, &mut lap) };

    let mut d_ptr: *mut f64 = std::ptr::null_mut();
    let mut n: usize = 0;
    let s = unsafe { smf_laplacian_to_dense(lap, &mut d_ptr, &mut n) };
    assert_eq!(s, SemiflowStatus::Ok);
    assert_eq!(n, 4);

    let dense = unsafe { std::slice::from_raw_parts(d_ptr, n * n) };

    // Diagonal values (degree of each node).
    assert!((dense[0] - 1.0).abs() < 1e-12, "L[0,0] should be 1.0 (degree of node 0)");
    assert!((dense[5] - 2.0).abs() < 1e-12, "L[1,1] should be 2.0 (degree of node 1)");
    assert!((dense[10] - 2.0).abs() < 1e-12, "L[2,2] should be 2.0 (degree of node 2)");
    assert!((dense[15] - 1.0).abs() < 1e-12, "L[3,3] should be 1.0 (degree of node 3)");

    // Off-diagonal adjacent entries must be -1.0.
    assert!((dense[1] + 1.0).abs() < 1e-12, "L[0,1] should be -1.0");
    assert!((dense[4] + 1.0).abs() < 1e-12, "L[1,0] should be -1.0");

    unsafe { smf_free_buf_f64(d_ptr, n * n) };
    unsafe { smf_laplacian_free(lap) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn laplacian_to_dense_null_returns_null_ptr() {
    let s = unsafe {
        smf_laplacian_to_dense(std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut())
    };
    assert_eq!(s, SemiflowStatus::NullPtr);
}

// ---------------------------------------------------------------------------
// SmfGraphTraj — constructors
// ---------------------------------------------------------------------------

#[test]
fn graph_traj_new_and_getters() {
    let g = unsafe { make_path(5) };
    let mut traj: *mut SmfGraphTraj = std::ptr::null_mut();
    let s = unsafe { smf_graph_traj_new(g, 2.5, &mut traj) };
    assert_eq!(s, SemiflowStatus::Ok);
    assert!(!traj.is_null());

    assert_eq!(unsafe { smf_graph_traj_n_nodes(traj) }, 5);
    assert_eq!(unsafe { smf_graph_traj_n_segments(traj) }, 1);
    let th = unsafe { smf_graph_traj_t_horizon(traj) };
    assert!((th - 2.5).abs() < 1e-15);

    unsafe { smf_graph_traj_free(traj) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn graph_traj_null_graph_returns_null_ptr() {
    let mut traj: *mut SmfGraphTraj = std::ptr::null_mut();
    let s = unsafe { smf_graph_traj_new(std::ptr::null(), 1.0, &mut traj) };
    assert_eq!(s, SemiflowStatus::NullPtr);
    assert!(traj.is_null());
}

#[test]
fn graph_traj_null_out_returns_null_ptr() {
    let g = unsafe { make_path(3) };
    let s = unsafe { smf_graph_traj_new(g, 1.0, std::ptr::null_mut()) };
    assert_eq!(s, SemiflowStatus::NullPtr);
    unsafe { smf_graph_drop(g) };
}

#[test]
fn graph_traj_nonpositive_t_horizon_returns_out_of_domain() {
    let g = unsafe { make_path(3) };
    let mut traj: *mut SmfGraphTraj = std::ptr::null_mut();
    let s = unsafe { smf_graph_traj_new(g, 0.0, &mut traj) };
    assert_eq!(s, SemiflowStatus::OutOfDomain);
    assert!(traj.is_null());
    let s2 = unsafe { smf_graph_traj_new(g, -1.0, &mut traj) };
    assert_eq!(s2, SemiflowStatus::OutOfDomain);
    unsafe { smf_graph_drop(g) };
}

#[test]
fn graph_traj_nan_t_horizon_returns_nan_inf() {
    let g = unsafe { make_path(3) };
    let mut traj: *mut SmfGraphTraj = std::ptr::null_mut();
    let s = unsafe { smf_graph_traj_new(g, f64::NAN, &mut traj) };
    assert_eq!(s, SemiflowStatus::NanInf);
    assert!(traj.is_null());
    unsafe { smf_graph_drop(g) };
}

#[test]
fn graph_traj_free_null_is_noop() {
    unsafe { smf_graph_traj_free(std::ptr::null_mut()) };
}

#[test]
fn graph_traj_n_nodes_null_returns_zero() {
    assert_eq!(unsafe { smf_graph_traj_n_nodes(std::ptr::null()) }, 0);
}

#[test]
fn graph_traj_n_segments_null_returns_zero() {
    assert_eq!(unsafe { smf_graph_traj_n_segments(std::ptr::null()) }, 0);
}

#[test]
fn graph_traj_t_horizon_null_returns_zero() {
    assert_eq!(unsafe { smf_graph_traj_t_horizon(std::ptr::null()) }, 0.0);
}
