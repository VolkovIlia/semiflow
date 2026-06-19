//! Cross-stencil and boundary helpers for `NonSeparableMixedChernoff`.
//!
//! Extracted from `nonseparable_mixed.rs` to keep that file under 500 lines.

extern crate alloc;

use crate::{
    float::{from_f64, half, SemiflowFloat},
    grid::BoundaryPolicy,
    grid2d::Grid2D,
    grid_fn2d::GridFn2D,
};

// ---------------------------------------------------------------------------
// Cross-stencil helpers (scalar and beta-weighted)
// ---------------------------------------------------------------------------

/// Scalar-weighted 4-point centred cross-stencil `c(x,y)·∂_x∂_y` (v2.3 `pub(crate)`).
///
/// `c(x_i,y_j)·(f[i+1,j+1]-f[i+1,j-1]-f[i-1,j+1]+f[i-1,j-1])/(4·dx·dy)`.
/// Accepts any `C: Fn(F,F)->F` (bare fn pointers satisfy `Fn`).
#[allow(clippy::cast_possible_wrap, clippy::similar_names)]
pub(crate) fn cross_stencil_scalar<F: SemiflowFloat, C: Fn(F, F) -> F>(
    src: &GridFn2D<F>,
    dst: &mut GridFn2D<F>,
    grid: &Grid2D<F>,
    c: C,
) {
    let nx = grid.nx();
    let ny = grid.ny();
    let denom: F = from_f64(
        4.0 * grid.x.dx().to_f64().unwrap_or(f64::NAN) * grid.y.dx().to_f64().unwrap_or(f64::NAN),
    );
    let bc_x = grid.x.boundary;
    let bc_y = grid.y.boundary;
    for j in 0..ny {
        let yj = grid.y.x_at(j);
        for i in 0..nx {
            let xi = grid.x.x_at(i);
            let ii = i as i64;
            let jj = j as i64;
            let v_pp = bc2d_val(&src.values, grid, bc_x, bc_y, ii + 1, jj + 1);
            let v_pm = bc2d_val(&src.values, grid, bc_x, bc_y, ii + 1, jj - 1);
            let v_mp = bc2d_val(&src.values, grid, bc_x, bc_y, ii - 1, jj + 1);
            let v_mm = bc2d_val(&src.values, grid, bc_x, bc_y, ii - 1, jj - 1);
            let raw = (v_pp - v_pm - v_mp + v_mm) / denom;
            dst.values[grid.idx(i, j)] = c(xi, yj) * raw;
        }
    }
}

/// Beta-weighted 4-point centred cross-stencil for `β(x,y)·∂_x∂_y`.
///
/// `(M_β f)[i,j] = β(x_i,y_j)·(f[i+1,j+1]-f[i+1,j-1]-f[i-1,j+1]+f[i-1,j-1])/(4·dx·dy)`
#[allow(clippy::cast_possible_wrap, clippy::similar_names)]
pub(crate) fn cross_stencil_beta<F: SemiflowFloat>(
    src: &GridFn2D<F>,
    dst: &mut GridFn2D<F>,
    grid: &Grid2D<F>,
    beta: fn(F, F) -> F,
) {
    // beta path is numerically identical to scalar path when beta == c;
    // dispatch here is the only difference (function pointer vs closure).
    cross_stencil_scalar(src, dst, grid, beta);
}

// ---------------------------------------------------------------------------
// 2D boundary value helper (generic F)
// ---------------------------------------------------------------------------

/// 2D boundary value via row-then-column 1D composition.
pub(crate) fn bc2d_val<F: SemiflowFloat>(
    values: &[F],
    grid: &Grid2D<F>,
    bc_x: BoundaryPolicy<F>,
    bc_y: BoundaryPolicy<F>,
    ii: i64,
    jj: i64,
) -> F {
    let nx = grid.nx();
    let ny = grid.ny();
    match resolve_axis(bc_y, ny, jj) {
        AxisHit::Inside(j_in) => bc1d_val(bc_x, &values[j_in * nx..(j_in + 1) * nx], nx, ii),
        AxisHit::Zero => F::zero(),
        AxisHit::Extrap(ja, jb, jc, sign_f, d_f) => {
            let sign: F = from_f64(sign_f);
            let d: F = from_f64(d_f);
            let h = half::<F>();
            let three: F = from_f64(3.0);
            let four: F = from_f64(4.0);
            let f0 = bc1d_val(bc_x, &values[ja * nx..(ja + 1) * nx], nx, ii);
            let f1 = bc1d_val(bc_x, &values[jb * nx..(jb + 1) * nx], nx, ii);
            let f2 = bc1d_val(bc_x, &values[jc * nx..(jc + 1) * nx], nx, ii);
            f0 + d * h * sign * (F::zero() - three * f0 + four * f1 - f2)
        }
    }
}

/// 1D boundary value for a row/column slice.
pub(crate) fn bc1d_val<F: SemiflowFloat>(bc: BoundaryPolicy<F>, row: &[F], n: usize, idx: i64) -> F {
    match resolve_axis(bc, n, idx) {
        AxisHit::Inside(i) => row[i],
        AxisHit::Zero => F::zero(),
        AxisHit::Extrap(ja, jb, jc, sign_f, d_f) => {
            let sign: F = from_f64(sign_f);
            let d: F = from_f64(d_f);
            let h = half::<F>();
            let three: F = from_f64(3.0);
            let four: F = from_f64(4.0);
            let f0 = row[ja];
            let f1 = row[jb];
            let f2 = row[jc];
            f0 + d * h * sign * (F::zero() - three * f0 + four * f1 - f2)
        }
    }
}

// ---------------------------------------------------------------------------
// Axis resolution (mirrors nonseparable2d.rs resolve_axis)
// ---------------------------------------------------------------------------

/// Resolved index for boundary-extended axis lookup.
pub(crate) enum AxisHit {
    Inside(usize),
    Zero,
    Extrap(usize, usize, usize, f64, f64),
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
pub(crate) fn resolve_axis<F: SemiflowFloat>(bc: BoundaryPolicy<F>, n: usize, idx: i64) -> AxisHit {
    if idx >= 0 && (idx as usize) < n {
        return AxisHit::Inside(idx as usize);
    }
    match bc {
        BoundaryPolicy::Reflect => {
            let nn = n as i64;
            let period = 2 * (nn - 1);
            let mut k = ((idx % period) + period) % period;
            if k >= nn {
                k = period - k;
            }
            AxisHit::Inside(k as usize)
        }
        // ZeroExtend and Dirichlet: both extend with zero outside the domain.
        BoundaryPolicy::ZeroExtend | BoundaryPolicy::Dirichlet { .. } => AxisHit::Zero,
        BoundaryPolicy::Periodic => {
            let nn = n as i64;
            let k = ((idx % nn) + nn) % nn;
            AxisHit::Inside(k as usize)
        }
        BoundaryPolicy::LinearExtrapolate => {
            if idx < 0 {
                let d = (-idx) as f64;
                AxisHit::Extrap(0, 1, 2, -1.0, d)
            } else {
                let d = (idx - (n as i64 - 1)) as f64;
                AxisHit::Extrap(n - 1, n - 2, n - 3, 1.0, d)
            }
        }
        // Neumann and Robin: clamp to nearest interior node (Robin character lives in operator).
        BoundaryPolicy::Neumann | BoundaryPolicy::Robin { .. } => {
            if idx < 0 {
                AxisHit::Inside(0)
            } else {
                AxisHit::Inside(n - 1)
            }
        }
    }
}
