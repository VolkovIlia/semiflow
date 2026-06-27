//! Harmonic-mean face assembler → `SymmetricOperator<F>` (ADR-0187 D1, §56.1–56.2).
//!
//! Provides:
//! - [`assemble_conservative_csr_1d`] — 1-D tridiagonal PSD carrier `A = −L_k`.
//! - [`assemble_conservative_csr_nd`] — separable 2-D/3-D Kronecker-sum PSD carrier.
//!
//! Both functions produce CSR matrices consumable by `SymmetricOperator::from_csr`,
//! bridging Issue #11 assembly to Issue #13 Krylov propagation (§56.2).

use alloc::vec::Vec;

use crate::{
    boundary::BoundaryPolicy,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_nd::GridND,
    symmetric_operator::SymmetricOperator,
};

// Helpers: harmonic_mean, face_transmissibility, build_faces (pub(crate)).
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/conservative_helpers.rs"));

// ── 1-D assembler ─────────────────────────────────────────────────────────────

/// Assemble symmetric PSD carrier `A = −L_k` (CSR, diag ≥ 0) for the conservative
/// 1-D operator `L_k u = ∂_x(k(x) ∂_x u)` with harmonic-mean faces (§56.1–56.2).
///
/// Uses Neumann (zero-flux) BCs at both endpoints, which is the natural choice for
/// time evolution via Krylov (Issue #13/14 bridge). For Dirichlet time evolution
/// use [`crate::conservative::ConservativeDiffusionChernoff`] with its Thomas solver.
///
/// `k_nodes` must have length `grid.n`; `r_contact`, if provided, must have length
/// `grid.n - 1`.
///
/// # Errors
///
/// [`SemiflowError::DomainViolation`] on length mismatch, `k ≤ 0`, non-finite `k`
/// or `R_c`, or `from_csr` validation failure (should not occur by construction).
///
/// # Panics
///
/// Never.
pub fn assemble_conservative_csr_1d<F: SemiflowFloat>(
    grid: Grid1D<F>,
    k_nodes: &[F],
    r_contact: Option<&[F]>,
    _boundary: BoundaryPolicy<F>,
) -> Result<SymmetricOperator<F>, SemiflowError> {
    let n = grid.n;
    if k_nodes.len() != n {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "assemble_conservative_csr_1d: k_nodes.len() != grid.n",
            value: k_nodes.len() as f64,
        });
    }
    let dx = grid.dx();
    let faces = build_faces(k_nodes, dx, r_contact)?;
    let (row_ptr, col_idx, vals) = build_1d_csr(n, &faces, dx);
    SymmetricOperator::from_csr(n, &row_ptr, &col_idx, &vals, F::from(1e-10_f64).unwrap_or(F::zero()))
}

/// Build the raw CSR triples for `A = −L_k` with Neumann BCs (sorted columns, symmetric).
///
/// Row 0:         [A[0,0]=T[0]/dx, A[0,1]=-T[0]/dx]
/// Row i interior:[A[i,i-1]=-T[i-1]/dx, A[i,i]=(T[i-1]+T[i])/dx, A[i,i+1]=-T[i]/dx]
/// Row n-1:       [A[n-1,n-2]=-T[n-2]/dx, A[n-1,n-1]=T[n-2]/dx]
fn build_1d_csr<F: SemiflowFloat>(
    n: usize,
    faces: &[F],
    dx: F,
) -> (Vec<usize>, Vec<u32>, Vec<F>) {
    // NNZ: endpoints have 2 entries each, interior have 3 each.
    let nnz = if n == 2 { 4 } else { 4 + 3 * (n - 2) };
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    row_ptr.push(0usize);
    for i in 0..n {
        emit_1d_row(i, n, faces, dx, &mut col_idx, &mut vals);
        row_ptr.push(col_idx.len());
    }
    (row_ptr, col_idx, vals)
}

/// Emit the CSR entries for row `i` of `A = −L_k` (Neumann BCs, sorted columns).
fn emit_1d_row<F: SemiflowFloat>(
    i: usize,
    n: usize,
    faces: &[F],
    dx: F,
    col_idx: &mut Vec<u32>,
    vals: &mut Vec<F>,
) {
    // left face T_{i-½}: exists when i > 0 (Neumann: no left face at i=0)
    let t_left = if i > 0 { faces[i - 1] } else { F::zero() };
    // right face T_{i+½}: exists when i < n-1 (Neumann: no right face at i=n-1)
    let t_right = if i + 1 < n { faces[i] } else { F::zero() };

    if i > 0 {
        #[allow(clippy::cast_possible_truncation)]
        col_idx.push((i - 1) as u32);
        vals.push(-t_left / dx);
    }
    #[allow(clippy::cast_possible_truncation)]
    col_idx.push(i as u32);
    vals.push((t_left + t_right) / dx);
    if i + 1 < n {
        #[allow(clippy::cast_possible_truncation)]
        col_idx.push((i + 1) as u32);
        vals.push(-t_right / dx);
    }
}

// ── N-D separable assembler ────────────────────────────────────────────────────

/// Assemble separable N-D PSD carrier `A = −Σ_d L_{k_d}` (§56.5).
///
/// Supports `D ∈ {2, 3}`. Each axis uses Neumann BCs. Columns emitted sorted
/// per row (invariant I3 of `SymmetricOperator::from_csr`).
///
/// `k_nodes_per_axis[d]` must have length `grid.axes[d].n`.
///
/// # Errors
///
/// [`SemiflowError::DomainViolation`] on length mismatch or bad conductivities.
/// [`SemiflowError::Unsupported`] if `D ∉ {2, 3}`.
pub fn assemble_conservative_csr_nd<F: SemiflowFloat, const D: usize>(
    grid: &GridND<F, D>,
    k_nodes_per_axis: &[&[F]],
    _boundary: BoundaryPolicy<F>,
) -> Result<SymmetricOperator<F>, SemiflowError> {
    if D == 2 {
        assemble_2d(grid, k_nodes_per_axis)
    } else if D == 3 {
        assemble_3d(grid, k_nodes_per_axis)
    } else {
        Err(SemiflowError::Unsupported {
            feature: "assemble_conservative_csr_nd: only D=2 and D=3 are supported",
        })
    }
}

/// Build 2-D Kronecker-sum CSR for `A_2D = A_0 ⊗ I_1 + I_0 ⊗ A_1` (5-pt stencil).
fn assemble_2d<F: SemiflowFloat, const D: usize>(
    grid: &GridND<F, D>,
    k_per_axis: &[&[F]],
) -> Result<SymmetricOperator<F>, SemiflowError> {
    validate_nd_axes(grid, k_per_axis, 2)?;
    let n0 = grid.axes[0].n;
    let n1 = grid.axes[1].n;
    let dx0 = grid.axes[0].dx();
    let dx1 = grid.axes[1].dx();
    let faces0 = build_faces(k_per_axis[0], dx0, None)?;
    let faces1 = build_faces(k_per_axis[1], dx1, None)?;
    let n_total = n0 * n1;
    let mut row_ptr = Vec::with_capacity(n_total + 1);
    let mut col_idx: Vec<u32> = Vec::new();
    let mut vals: Vec<F> = Vec::new();
    row_ptr.push(0usize);
    for k1 in 0..n1 {
        for k0 in 0..n0 {
            emit_2d_row(k0, k1, n0, n1, &faces0, &faces1, dx0, dx1, &mut col_idx, &mut vals);
            row_ptr.push(col_idx.len());
        }
    }
    SymmetricOperator::from_csr(
        n_total,
        &row_ptr,
        &col_idx,
        &vals,
        F::from(1e-10_f64).unwrap_or(F::zero()),
    )
}

/// Emit the 5-pt stencil row for node `(k0, k1)` in the 2-D Kronecker-sum operator.
///
/// Flat index: `k = k1 * n0 + k0`. Neighbors emitted in sorted column order:
/// `k - n0, k - 1, k, k + 1, k + n0`.
#[allow(clippy::too_many_arguments)]
fn emit_2d_row<F: SemiflowFloat>(
    k0: usize,
    k1: usize,
    n0: usize,
    n1: usize,
    faces0: &[F],
    faces1: &[F],
    dx0: F,
    dx1: F,
    col_idx: &mut Vec<u32>,
    vals: &mut Vec<F>,
) {
    let k = k1 * n0 + k0;
    // Axis-1 left neighbor (k - n0): exists when k1 > 0.
    let t1_left = if k1 > 0 { faces1[k1 - 1] } else { F::zero() };
    let t1_right = if k1 + 1 < n1 { faces1[k1] } else { F::zero() };
    let t0_left = if k0 > 0 { faces0[k0 - 1] } else { F::zero() };
    let t0_right = if k0 + 1 < n0 { faces0[k0] } else { F::zero() };

    if k1 > 0 {
        #[allow(clippy::cast_possible_truncation)]
        col_idx.push((k - n0) as u32);
        vals.push(-t1_left / dx1);
    }
    if k0 > 0 {
        #[allow(clippy::cast_possible_truncation)]
        col_idx.push((k - 1) as u32);
        vals.push(-t0_left / dx0);
    }
    #[allow(clippy::cast_possible_truncation)]
    col_idx.push(k as u32);
    vals.push((t0_left + t0_right) / dx0 + (t1_left + t1_right) / dx1);
    if k0 + 1 < n0 {
        #[allow(clippy::cast_possible_truncation)]
        col_idx.push((k + 1) as u32);
        vals.push(-t0_right / dx0);
    }
    if k1 + 1 < n1 {
        #[allow(clippy::cast_possible_truncation)]
        col_idx.push((k + n0) as u32);
        vals.push(-t1_right / dx1);
    }
}

/// Build 3-D Kronecker-sum CSR (7-pt stencil). Flat index: `k = k2*n1*n0 + k1*n0 + k0`.
fn assemble_3d<F: SemiflowFloat, const D: usize>(
    grid: &GridND<F, D>,
    k_per_axis: &[&[F]],
) -> Result<SymmetricOperator<F>, SemiflowError> {
    validate_nd_axes(grid, k_per_axis, 3)?;
    let n0 = grid.axes[0].n;
    let n1 = grid.axes[1].n;
    let n2 = grid.axes[2].n;
    let faces0 = build_faces(k_per_axis[0], grid.axes[0].dx(), None)?;
    let faces1 = build_faces(k_per_axis[1], grid.axes[1].dx(), None)?;
    let faces2 = build_faces(k_per_axis[2], grid.axes[2].dx(), None)?;
    let n_total = n0 * n1 * n2;
    let mut row_ptr = Vec::with_capacity(n_total + 1);
    let mut col_idx: Vec<u32> = Vec::new();
    let mut vals: Vec<F> = Vec::new();
    row_ptr.push(0usize);
    for k2 in 0..n2 {
        for k1 in 0..n1 {
            for k0 in 0..n0 {
                emit_3d_row(
                    k0, k1, k2, n0, n1, n2,
                    &faces0, &faces1, &faces2,
                    grid.axes[0].dx(), grid.axes[1].dx(), grid.axes[2].dx(),
                    &mut col_idx, &mut vals,
                );
                row_ptr.push(col_idx.len());
            }
        }
    }
    SymmetricOperator::from_csr(
        n_total,
        &row_ptr,
        &col_idx,
        &vals,
        F::from(1e-10_f64).unwrap_or(F::zero()),
    )
}

/// Emit the 7-pt stencil row for node `(k0, k1, k2)` in the 3-D operator.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn emit_3d_row<F: SemiflowFloat>(
    k0: usize, k1: usize, k2: usize,
    n0: usize, n1: usize, n2: usize,
    faces0: &[F], faces1: &[F], faces2: &[F],
    dx0: F, dx1: F, dx2: F,
    col_idx: &mut Vec<u32>,
    vals: &mut Vec<F>,
) {
    let k = k2 * n1 * n0 + k1 * n0 + k0;
    let t0l = if k0 > 0 { faces0[k0 - 1] } else { F::zero() };
    let t0r = if k0 + 1 < n0 { faces0[k0] } else { F::zero() };
    let t1l = if k1 > 0 { faces1[k1 - 1] } else { F::zero() };
    let t1r = if k1 + 1 < n1 { faces1[k1] } else { F::zero() };
    let t2l = if k2 > 0 { faces2[k2 - 1] } else { F::zero() };
    let t2r = if k2 + 1 < n2 { faces2[k2] } else { F::zero() };
    // Emit off-diagonal entries in CSR column order (sorted).
    if k2 > 0 { col_idx.push((k - n1 * n0) as u32); vals.push(-t2l / dx2); }
    if k1 > 0 { col_idx.push((k - n0) as u32);       vals.push(-t1l / dx1); }
    if k0 > 0 { col_idx.push((k - 1) as u32);         vals.push(-t0l / dx0); }
    col_idx.push(k as u32);
    vals.push((t0l + t0r) / dx0 + (t1l + t1r) / dx1 + (t2l + t2r) / dx2);
    if k0 + 1 < n0 { col_idx.push((k + 1) as u32);        vals.push(-t0r / dx0); }
    if k1 + 1 < n1 { col_idx.push((k + n0) as u32);       vals.push(-t1r / dx1); }
    if k2 + 1 < n2 { col_idx.push((k + n1 * n0) as u32);  vals.push(-t2r / dx2); }
}

/// Validate that `k_nodes_per_axis` matches the grid axis lengths.
fn validate_nd_axes<F: SemiflowFloat, const D: usize>(
    grid: &GridND<F, D>,
    k_per_axis: &[&[F]],
    expected_d: usize,
) -> Result<(), SemiflowError> {
    if D != expected_d {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "assemble_conservative_csr_nd: D mismatch",
            value: D as f64,
        });
    }
    if k_per_axis.len() != D {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "assemble_conservative_csr_nd: k_nodes_per_axis.len() must equal D",
            value: k_per_axis.len() as f64,
        });
    }
    for (d, &k_d) in k_per_axis.iter().enumerate() {
        if k_d.len() != grid.axes[d].n {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "assemble_conservative_csr_nd: k_nodes length mismatch for axis",
                value: d as f64,
            });
        }
    }
    Ok(())
}
