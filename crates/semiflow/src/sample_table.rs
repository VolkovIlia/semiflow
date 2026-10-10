//! Per-step prepared sampling of an `f64` grid function (ADR-0204).
//!
//! `GridFn1D::sample` rebuilds, for EVERY sample, the ghost-extended nodal data
//! its interpolant needs: the default septic-Hermite sampler evaluates
//! `dx·f′, dx²·f″, dx³·f‴` at both cell ends from finite-difference stencils
//! (44 `bc_value` calls per sample), the octonic one adds a fourth derivative,
//! and the Chebyshev sampler resamples all `M + 1` virtual nodes per sample.
//! Within one Chernoff step every sample reads the same state, so
//! [`PreparedGridFn`] computes that data ONCE per step into a ghost-extended
//! table and evaluates each sample from it.
//!
//! The table entries are produced by the very functions the direct sampler
//! calls (`bc_value`, the FD stencils, the virtual-node sampler) and combined in
//! the same order, so a prepared sample is BIT-IDENTICAL to `GridFn1D::sample`
//! (gate `G_PLAN_BIT_EQUAL`). Samples whose cell lies outside the table margin
//! fall back to the direct formula.

use crate::{
    error::SemiflowError,
    grid::{bc_value, Grid1D, InterpKind},
    grid_chebyshev::{barycentric_lobatto_eval, chebyshev_eval_with, virtual_node_value},
    grid_chebyshev_nodes::{chebyshev_nodes, chebyshev_weights, is_supported_m},
    grid_chebyshev_octonic::{
        octonic_combine, octonic_node_data, octonic_weights, OCTONIC_FD_RADIUS,
    },
    grid_chebyshev_septic::{
        cell_of, sample_septic_1d, septic_combine, septic_node_data, septic_weights,
        SEPTIC_FD_RADIUS,
    },
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

/// Cells beyond each end of the grid covered by the ghost table. Samples whose
/// cell lies further out use the direct formula (same bits, just slower).
pub(crate) const MARGIN: i64 = 24;

/// A grid function that can be sampled at off-grid points (crate-internal).
///
/// Implemented by `GridFn1D<f64>` (direct) and [`PreparedGridFn`] (table); the
/// Chernoff node kernels are generic over it so both paths share one formula.
pub(crate) trait Sample1D: Sync {
    /// Interpolated value at `x` (boundary policy applied outside the grid).
    fn sample(&self, x: f64) -> Result<f64, SemiflowError>;
    /// Nodal value `f[i]`.
    fn node(&self, i: usize) -> f64;
}

impl Sample1D for GridFn1D<f64> {
    #[inline]
    fn sample(&self, x: f64) -> Result<f64, SemiflowError> {
        GridFn1D::sample(self, x)
    }

    #[inline]
    fn node(&self, i: usize) -> f64 {
        self.values[i]
    }
}

/// Table layout per interpolant (`stride` doubles per ghost-extended node).
#[derive(Clone, Copy)]
enum Layout {
    /// No table (linear interpolation, unsupported Chebyshev `M`): direct sampling.
    Direct,
    /// Catmull–Rom: nodal values.
    Cubic,
    /// Septic Hermite: `[f, dx·f′, dx²·f″, dx³·f‴]` per node.
    Septic,
    /// Octonic Hermite: `[f, dx·f′, …, dx⁴·f⁗]` per node.
    Octonic,
    /// Chebyshev spectral: the `M + 1` virtual-node values.
    Chebyshev { m: usize },
}

fn layout_of(grid: &Grid1D) -> Layout {
    match grid.interp {
        InterpKind::CubicHermite => Layout::Cubic,
        InterpKind::SepticHermite => Layout::Septic,
        InterpKind::OctonicHermite => Layout::Octonic,
        InterpKind::ChebyshevSpectralWithBC { m, .. } if is_supported_m(m) => {
            Layout::Chebyshev { m }
        }
        _ => Layout::Direct,
    }
}

/// Number of ghost-extended nodes `[−MARGIN, n − 1 + MARGIN]`.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // MARGIN > 0
fn span(n: usize) -> usize {
    n + 2 * MARGIN as usize
}

/// Per-step prepared view of `(values, grid)`: bit-identical samples from a table.
pub(crate) struct PreparedGridFn<'a> {
    values: &'a [f64],
    grid: Grid1D,
    layout: Layout,
    table: &'a [f64],
}

impl<'a> PreparedGridFn<'a> {
    /// Work-buffer length [`fill`](Self::fill) needs for `grid`.
    pub(crate) fn work_len(grid: &Grid1D) -> usize {
        match layout_of(grid) {
            Layout::Direct => 0,
            Layout::Cubic => span(grid.n),
            Layout::Septic => 4 * span(grid.n),
            Layout::Octonic => 5 * span(grid.n),
            Layout::Chebyshev { m } => m + 1,
        }
    }

    /// Fill `work` (length [`work_len`](Self::work_len)) for the state `values` on `grid`.
    pub(crate) fn fill(values: &[f64], grid: &Grid1D, work: &mut [f64]) {
        match layout_of(grid) {
            Layout::Direct => {}
            Layout::Cubic => fill_nodes(values, grid, work, &CubicEntry),
            Layout::Septic => fill_nodes(values, grid, work, &SepticEntry),
            Layout::Octonic => fill_nodes(values, grid, work, &OctonicEntry),
            Layout::Chebyshev { m } => {
                let eff = grid.chebyshev_effective_grid();
                let nodes = chebyshev_nodes(m).unwrap_or(&[]);
                for (w, &node) in work.iter_mut().zip(nodes) {
                    *w = virtual_node_value(values, &eff, node);
                }
            }
        }
    }

    /// Immutable view over a filled work buffer.
    pub(crate) fn view(values: &'a [f64], grid: Grid1D, table: &'a [f64]) -> Self {
        Self {
            values,
            grid,
            layout: layout_of(&grid),
            table,
        }
    }

    /// Table offset of ghost node `j` when cells `j0..=j1` are all covered.
    #[inline]
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn covered(&self, j0: i64, j1: i64) -> Option<usize> {
        let hi = self.grid.n as i64 - 1 + MARGIN;
        (j0 >= -MARGIN && j1 <= hi).then(|| (j0 + MARGIN) as usize)
    }

    fn sample_chebyshev(&self, m: usize, x: f64) -> f64 {
        let eff = self.grid.chebyshev_effective_grid();
        let nodes = chebyshev_nodes(m).unwrap_or(&[]);
        let weights = chebyshev_weights(m).unwrap_or(&[]);
        let vn = self.table;
        let bary = |xx: f64| barycentric_lobatto_eval(&eff, xx, m, nodes, weights, &|k, _| vn[k]);
        chebyshev_eval_with(self.values, &eff, x, m, &bary)
    }
}

impl Sample1D for PreparedGridFn<'_> {
    #[inline]
    fn sample(&self, x: f64) -> Result<f64, SemiflowError> {
        match self.layout {
            Layout::Direct => self.grid.interp(self.values, x),
            Layout::Cubic => {
                let (idx, s) = cell_of(&self.grid, x);
                Ok(match self.covered(idx - 1, idx + 2) {
                    Some(o) => {
                        let t = &self.table[o..o + 4];
                        crate::grid_cubic::catmull_rom(t[0], t[1], t[2], t[3], s)
                    }
                    None => self.grid.interp(self.values, x)?,
                })
            }
            Layout::Septic => {
                let (idx, s) = cell_of(&self.grid, x);
                Ok(match self.covered(idx, idx + 1) {
                    Some(o) => {
                        let mut d = [0.0; 8];
                        d.copy_from_slice(&self.table[4 * o..4 * o + 8]);
                        septic_combine(&septic_weights(s), &d)
                    }
                    None => sample_septic_1d(self.values, &self.grid, x),
                })
            }
            Layout::Octonic => {
                let (idx, s) = cell_of(&self.grid, x);
                Ok(match self.covered(idx, idx + 1) {
                    Some(o) => {
                        let mut d = [0.0; 10];
                        d.copy_from_slice(&self.table[5 * o..5 * o + 10]);
                        octonic_combine(&octonic_weights(s), &d)
                    }
                    None => self.grid.interp(self.values, x)?,
                })
            }
            Layout::Chebyshev { m } => Ok(self.sample_chebyshev(m, x)),
        }
    }

    #[inline]
    fn node(&self, i: usize) -> f64 {
        self.values[i]
    }
}

/// What one ghost node of a table holds (`STRIDE` doubles).
trait NodeEntry<const STRIDE: usize> {
    /// Stencil radius: nodes `j ± radius` are read.
    const RADIUS: i64;
    fn at<G: Fn(i64) -> f64>(&self, get: &G, j: i64) -> [f64; STRIDE];
}

struct CubicEntry;
impl NodeEntry<1> for CubicEntry {
    const RADIUS: i64 = 0;
    #[inline]
    fn at<G: Fn(i64) -> f64>(&self, get: &G, j: i64) -> [f64; 1] {
        [get(j)]
    }
}

struct SepticEntry;
impl NodeEntry<4> for SepticEntry {
    const RADIUS: i64 = SEPTIC_FD_RADIUS;
    #[inline]
    fn at<G: Fn(i64) -> f64>(&self, get: &G, j: i64) -> [f64; 4] {
        septic_node_data(get, j)
    }
}

struct OctonicEntry;
impl NodeEntry<5> for OctonicEntry {
    const RADIUS: i64 = OCTONIC_FD_RADIUS;
    #[inline]
    fn at<G: Fn(i64) -> f64>(&self, get: &G, j: i64) -> [f64; 5] {
        octonic_node_data(get, j)
    }
}

/// Fill `STRIDE` doubles per ghost node `j ∈ [−MARGIN, n−1+MARGIN]`.
///
/// Where the whole stencil is interior the nodal values are read straight from
/// the slice — the same numbers `bc_value` returns there — so only
/// boundary-adjacent entries pay for the policy dispatch.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn fill_nodes<const STRIDE: usize, E: NodeEntry<STRIDE>>(
    values: &[f64],
    grid: &Grid1D,
    work: &mut [f64],
    entry: &E,
) {
    let (bnd, n, dx) = (grid.boundary, grid.n, grid.dx());
    let n_i = n as i64;
    let direct = |k: i64| values[k as usize];
    let ghost = |k: i64| bc_value(bnd, values, n, k, dx);
    for (slot, j) in work.chunks_exact_mut(STRIDE).zip(-MARGIN..) {
        let e = if j - E::RADIUS >= 0 && j + E::RADIUS < n_i {
            entry.at(&direct, j)
        } else {
            entry.at(&ghost, j)
        };
        slot.copy_from_slice(&e);
    }
}

/// Run `body` with a prepared view of `src` re-viewed through `grid`.
///
/// The table is borrowed from `scratch` and returned afterwards, so steady-state
/// steps allocate nothing.
pub(crate) fn with_prepared<R>(
    values: &[f64],
    grid: Grid1D,
    scratch: &mut ScratchPool<f64>,
    body: impl FnOnce(&PreparedGridFn<'_>) -> R,
) -> R {
    let mut work = scratch.take_vec(PreparedGridFn::work_len(&grid));
    PreparedGridFn::fill(values, &grid, &mut work);
    let out = body(&PreparedGridFn::view(values, grid, &work));
    scratch.return_vec(work);
    out
}

#[cfg(test)]
#[path = "sample_table_tests.rs"]
mod tests;
