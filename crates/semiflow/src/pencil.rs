//! Stride-aware pencil-slice utilities for `GridFnXD::values`.
//!
//! Provides slice-level access to individual pencils (rows, columns, slabs)
//! inside the flat `values` buffer of `GridFn2D` and `GridFn3D`, without
//! allocating a `GridFn1D`.  Used by `AxisLift::apply_into` and
//! `AxisLift3D::apply_into` (Wave 2, ADR-0042).
//!
//! All functions are `pub(crate)` — not part of the v1.0.0 stable API
//! (ADR-0035).  Future SIMD-on-strided-pack rewrites may replace internals
//! without semver impact.
//!
//! **Stride math (NORMATIVE)** — must match `GridFn3D::idx = k*nx*ny + j*nx + i`:
//!
//! | Pencil       | Stride   | Base           | Length |
//! |--------------|----------|----------------|--------|
//! | X-row 2D     | 1        | `j*nx`         | `nx`   |
//! | Y-col 2D     | `nx`     | `i`            | `ny`   |
//! | X-pencil 3D  | 1        | `k*nx*ny+j*nx` | `nx`   |
//! | Y-pencil 3D  | `nx`     | `k*nx*ny+i`    | `ny`   |
//! | Z-pencil 3D  | `nx*ny`  | `j*nx+i`       | `nz`   |

use crate::float::SemiflowFloat;

// ---------------------------------------------------------------------------
// 2D — X-rows (contiguous, stride = 1)
// ---------------------------------------------------------------------------

/// Shared slice over the j-th X-row: `values[j*nx .. (j+1)*nx]`.
#[inline]
pub(crate) fn row_2d<F: SemiflowFloat>(values: &[F], nx: usize, j: usize) -> &[F] {
    &values[j * nx..(j + 1) * nx]
}

/// Mutable slice over the j-th X-row.
#[inline]
pub(crate) fn row_2d_mut<F: SemiflowFloat>(values: &mut [F], nx: usize, j: usize) -> &mut [F] {
    &mut values[j * nx..(j + 1) * nx]
}

// ---------------------------------------------------------------------------
// 2D — Y-columns (strided, stride = nx)
// ---------------------------------------------------------------------------

/// Gather Y-column `i` (stride `nx`) into a pre-allocated `slot` of length `ny`.
///
/// `slot` is reused across columns to avoid per-pencil allocation.
#[inline]
pub(crate) fn gather_y_2d_into<F: SemiflowFloat>(
    values: &[F],
    nx: usize,
    ny: usize,
    i: usize,
    slot: &mut [F],
) {
    debug_assert_eq!(slot.len(), ny, "gather_y_2d_into: slot length mismatch");
    for j in 0..ny {
        slot[j] = values[j * nx + i];
    }
}

/// Scatter `slot` back into Y-column `i` (stride `nx`).
#[inline]
pub(crate) fn scatter_y_2d_from<F: SemiflowFloat>(
    values: &mut [F],
    nx: usize,
    ny: usize,
    i: usize,
    slot: &[F],
) {
    debug_assert_eq!(slot.len(), ny, "scatter_y_2d_from: slot length mismatch");
    for j in 0..ny {
        values[j * nx + i] = slot[j];
    }
}

// ---------------------------------------------------------------------------
// 3D — X-pencils (contiguous, stride = 1)
// ---------------------------------------------------------------------------

/// Shared slice over the X-pencil `(j, k)`: `values[k*nx*ny + j*nx .. + nx]`.
#[inline]
pub(crate) fn pencil_x_3d<F: SemiflowFloat>(
    values: &[F],
    nx: usize,
    ny: usize,
    j: usize,
    k: usize,
) -> &[F] {
    let start = k * nx * ny + j * nx;
    &values[start..start + nx]
}

/// Mutable slice over the X-pencil `(j, k)`.
#[inline]
pub(crate) fn pencil_x_3d_mut<F: SemiflowFloat>(
    values: &mut [F],
    nx: usize,
    ny: usize,
    j: usize,
    k: usize,
) -> &mut [F] {
    let start = k * nx * ny + j * nx;
    &mut values[start..start + nx]
}

// ---------------------------------------------------------------------------
// 3D — Y-pencils (strided, stride = nx)
// ---------------------------------------------------------------------------

/// Gather Y-pencil `(i, k)` (stride `nx`) into `slot` of length `ny`.
///
/// `base = k*nx*ny + i`; `values[base + j*nx]` for `j` in `0..ny`.
#[inline]
pub(crate) fn gather_y_3d_into<F: SemiflowFloat>(
    values: &[F],
    nx: usize,
    ny: usize,
    i: usize,
    k: usize,
    slot: &mut [F],
) {
    debug_assert_eq!(slot.len(), ny, "gather_y_3d_into: slot length mismatch");
    let base = k * nx * ny + i;
    for j in 0..ny {
        slot[j] = values[base + j * nx];
    }
}

/// Scatter `slot` back into Y-pencil `(i, k)`.
#[inline]
pub(crate) fn scatter_y_3d_from<F: SemiflowFloat>(
    values: &mut [F],
    nx: usize,
    ny: usize,
    i: usize,
    k: usize,
    slot: &[F],
) {
    debug_assert_eq!(slot.len(), ny, "scatter_y_3d_from: slot length mismatch");
    let base = k * nx * ny + i;
    for j in 0..ny {
        values[base + j * nx] = slot[j];
    }
}

// ---------------------------------------------------------------------------
// 3D — Z-pencils (strided, stride = nx*ny)
// ---------------------------------------------------------------------------

/// Gather Z-pencil `(i, j)` (stride `nx*ny`) into `slot` of length `nz`.
///
/// `base = j*nx + i`; `values[base + k*nx*ny]` for `k` in `0..nz`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_z_3d_into<F: SemiflowFloat>(
    values: &[F],
    nx: usize,
    ny: usize,
    nz: usize,
    i: usize,
    j: usize,
    slot: &mut [F],
) {
    debug_assert_eq!(slot.len(), nz, "gather_z_3d_into: slot length mismatch");
    let base = j * nx + i;
    let stride = nx * ny;
    for k in 0..nz {
        slot[k] = values[base + k * stride];
    }
}

/// Scatter `slot` back into Z-pencil `(i, j)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_z_3d_from<F: SemiflowFloat>(
    values: &mut [F],
    nx: usize,
    ny: usize,
    nz: usize,
    i: usize,
    j: usize,
    slot: &[F],
) {
    debug_assert_eq!(slot.len(), nz, "scatter_z_3d_from: slot length mismatch");
    let base = j * nx + i;
    let stride = nx * ny;
    for k in 0..nz {
        values[base + k * stride] = slot[k];
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vals(n: usize) -> Vec<f64> {
        (0..n)
            .map(|x| f64::from(u32::try_from(x).unwrap()))
            .collect()
    }

    /// Row round-trip: gather a row, modify, scatter, verify.
    #[test]
    fn row_2d_roundtrip() {
        // 3x4 grid (nx=3, ny=4)
        let mut vals = make_vals(12);
        // Row 1 = indices [3,4,5]
        let row = row_2d(&vals, 3, 1);
        assert_eq!(row, &[3.0, 4.0, 5.0]);
        row_2d_mut(&mut vals, 3, 1)[0] = 99.0;
        // Integer index 3; comparison is exact (integer-valued f64).
        assert!((vals[3] - 99.0).abs() < f64::EPSILON);
    }

    /// Y-column gather/scatter round-trip.
    #[test]
    fn y_2d_gather_scatter_roundtrip() {
        // nx=3, ny=4; column i=1 is indices [1,4,7,10]
        let vals = make_vals(12);
        let mut slot = vec![0.0_f64; 4];
        gather_y_2d_into(&vals, 3, 4, 1, &mut slot);
        assert_eq!(slot, vec![1.0, 4.0, 7.0, 10.0]);

        let mut out = vec![0.0_f64; 12];
        scatter_y_2d_from(&mut out, 3, 4, 1, &slot);
        // Integer-valued f64 comparisons — exact.
        assert!((out[1] - 1.0).abs() < f64::EPSILON);
        assert!((out[4] - 4.0).abs() < f64::EPSILON);
        assert!((out[7] - 7.0).abs() < f64::EPSILON);
        assert!((out[10] - 10.0).abs() < f64::EPSILON);
    }

    /// 3D Z-pencil gather/scatter round-trip.
    #[test]
    fn z_3d_gather_scatter_roundtrip() {
        // nx=2, ny=2, nz=3; z-pencil (0,0) = indices [0, 4, 8]
        let vals = make_vals(12);
        let mut slot = vec![0.0_f64; 3];
        gather_z_3d_into(&vals, 2, 2, 3, 0, 0, &mut slot);
        assert_eq!(slot, vec![0.0, 4.0, 8.0]);

        let mut out = vec![0.0_f64; 12];
        scatter_z_3d_from(&mut out, 2, 2, 3, 0, 0, &slot);
        // Integer-valued f64 comparisons — exact.
        assert!((out[0] - 0.0).abs() < f64::EPSILON);
        assert!((out[4] - 4.0).abs() < f64::EPSILON);
        assert!((out[8] - 8.0).abs() < f64::EPSILON);
    }
}
