//! [`GridND`] and [`GridFnND`] — d-dimensional tensor-product grid (v4.0 Wave C).
//!
//! Generic over `F: SemiflowFloat` and `const D: usize` (dimension). Each axis
//! is an independent [`Grid1D<F>`] with its own boundary policy and interp kind.
//!
//! Storage convention (axis 0 fastest — Fortran order for a `(n₀, …, n_{D-1})` view, NOT numpy's C order):
//! `idx(k₀, k₁, ..., k_{D-1}) = k_{D-1}·n_{D-2}·…·n₀ + … + k₁·n₀ + k₀`.
//! This is the `GridFnND<F, D>` state type used by
//! [`crate::shift_nd::AnisotropicShiftChernoffND<F, D>`] (math.md §32,
//! ADR-0081) and [`crate::hormander::HypoellipticChernoff`] (future generic).
//!
//! ## Sub-grid sampling (ADR-0191)
//!
//! [`GridFnND::sample`] evaluates the tensor-product interpolant selected by
//! [`GridND::interp`] (default [`InterpKind::CubicHermite`]) and honours each
//! axis's own [`crate::grid::BoundaryPolicy`]. Until ADR-0191 it hard-coded
//! multilinear interpolation with an index clamp; because every Chernoff step
//! resamples at off-grid quadrature feet, that injected ≈ `dx²/6` of spurious
//! second moment **per step**, growing linearly in the step count (math §32.7).

use alloc::vec::Vec;

use crate::{
    boundary::{bc_index, bc_value_from_hit, BoundaryHit},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::{Grid1D, InterpKind},
    interp_stencil::{interp_stencil, K_MAX},
    state::State,
};

// ---------------------------------------------------------------------------
// GridND<F, D>
// ---------------------------------------------------------------------------

/// d-dimensional tensor-product grid with uniform axes.
///
/// Each axis is a [`Grid1D<F>`] with independent geometry, boundary policy,
/// and interpolation kind. Dimension `D` is a const generic parameter.
/// Values are stored axis-0-fastest:
/// `flat_idx = k_{D-1}·n_{D-2}·…·n₀ + … + k₁·n₀ + k₀`.
///
/// # Example
///
/// ```rust
/// use semiflow::{Grid1D, grid_nd::GridND};
/// let axes = core::array::from_fn(|_| Grid1D::new(-5.0_f64, 5.0, 16).unwrap());
/// let grid = GridND::<f64, 2>::new(axes).unwrap();
/// assert_eq!(grid.len(), 256); // 16*16
/// ```
#[derive(Clone)]
pub struct GridND<F: SemiflowFloat = f64, const D: usize = 2> {
    /// Per-axis grids. `axes[0]` is the fastest-varying axis.
    ///
    /// Each axis contributes its own [`crate::grid::BoundaryPolicy`] to
    /// [`GridFnND::sample`]. The axes' individual [`InterpKind`]s are NOT
    /// consulted — see [`GridND::interp`].
    pub axes: [Grid1D<F>; D],
    /// Tensor-product interpolation kind used by [`GridFnND::sample`].
    ///
    /// Default [`InterpKind::CubicHermite`] (ADR-0191). A single grid-level
    /// knob rather than a per-axis one because `Grid1D::new` already stamps
    /// every axis with `SepticHermite`, leaving no way to tell a deliberate
    /// choice from an inherited default. `CubicHermite` removes the accumulated
    /// per-step interpolation variance at `4^D` nodes per sample; `SepticHermite`
    /// would cost `8^D` for no gain on that failure mode, and is not implemented
    /// for `D > 1`.
    pub interp: InterpKind,
}

impl<F: SemiflowFloat, const D: usize> GridND<F, D> {
    /// Construct from an array of `D` axes.
    ///
    /// # Errors
    /// - `DomainViolation` if `D == 0`.
    /// - `DomainViolation` if any axis has `n < 4`.
    pub fn new(axes: [Grid1D<F>; D]) -> Result<Self, SemiflowError> {
        if D == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "GridND: D must be >= 1",
                value: 0.0,
            });
        }
        for (i, ax) in axes.iter().enumerate() {
            if ax.n < 4 {
                #[allow(clippy::cast_precision_loss)]
                return Err(SemiflowError::DomainViolation {
                    what: "GridND: each axis must have n >= 4",
                    value: i as f64,
                });
            }
        }
        Ok(Self {
            axes,
            interp: InterpKind::CubicHermite,
        })
    }

    /// Override the tensor-product interpolation kind (builder, ADR-0191).
    ///
    /// Only [`InterpKind::CubicHermite`] and [`InterpKind::Linear`] are
    /// implemented for `D > 1`; the others make [`GridFnND::sample`] return
    /// [`SemiflowError::Unsupported`]. `Linear` needs the `linear-interp`
    /// feature, matching the 1-D contract.
    #[must_use]
    pub fn with_interp(mut self, interp: InterpKind) -> Self {
        self.interp = interp;
        self
    }

    /// Total number of grid points: `n₀ · n₁ · … · n_{D-1}`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.axes.iter().map(|ax| ax.n).product()
    }

    /// Returns `false` for any valid `GridND` (all axes have `n >= 4`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Number of nodes along axis `d`.
    ///
    /// # Panics
    /// Panics if `d >= D` (debug only).
    #[must_use]
    pub fn n_along(&self, d: usize) -> usize {
        self.axes[d].n
    }

    /// Convert a multi-index `[k₀, k₁, …, k_{D-1}]` to a flat index.
    ///
    /// Axis 0 is fastest-varying (row-major in axis-0 direction).
    ///
    /// # Panics
    /// Panics in debug if `kd >= n_d` for any `d`.
    #[must_use]
    pub fn flat_idx(&self, idx: &[usize; D]) -> usize {
        let mut flat = 0_usize;
        let mut stride = 1_usize;
        // d indexes both idx[] and self.axes[] simultaneously; range loop is needed.
        #[allow(clippy::needless_range_loop)]
        for d in 0..D {
            debug_assert!(idx[d] < self.axes[d].n, "index out of range on axis {d}");
            flat += idx[d] * stride;
            stride *= self.axes[d].n;
        }
        flat
    }

    /// Physical coordinate of multi-index `[k₀, …, k_{D-1}]` on axis `d`.
    #[must_use]
    pub fn x_at(&self, d: usize, k: usize) -> F {
        self.axes[d].x_at(k)
    }

    /// Physical coordinates of the grid CENTRE (used for SPD validation).
    #[must_use]
    pub fn centre(&self) -> [F; D] {
        core::array::from_fn(|d| {
            let ax = &self.axes[d];
            let half = from_f64::<F>(0.5_f64);
            ax.xmin + half * (ax.xmax - ax.xmin)
        })
    }
}

// ---------------------------------------------------------------------------
// AxisStencil — one axis's contribution to a tensor-product sample (ADR-0191)
// ---------------------------------------------------------------------------

/// Per-axis interpolation stencil, fully resolved once per sample.
///
/// `hits`/`stride` are hoisted out of the collapse recursion (ADR-0191 AM 4):
/// the recursion re-visits axis `d` once per combination of the axes above it,
/// so resolving the policy inside it cost 1364 [`bc_index`] calls per sample at
/// `D = 5` against the 20 that are distinct.
#[derive(Clone, Copy)]
struct AxisStencil<F: SemiflowFloat> {
    /// Number of meaningful entries in `weights` / `hits`.
    k: usize,
    /// Nodal weights; `Σ weights[..k] == 1`.
    weights: [F; K_MAX],
    /// Boundary resolution of each stencil node, computed once.
    hits: [BoundaryHit<F>; K_MAX],
    /// Flat-index stride of this axis: `Π_{e<d} n_e`.
    stride: usize,
}

impl<F: SemiflowFloat> AxisStencil<F> {
    /// Placeholder used to initialise the per-axis plan array before filling it.
    fn zeroed() -> Self {
        Self {
            k: 1,
            weights: [F::zero(); K_MAX],
            hits: [BoundaryHit::Inside(0); K_MAX],
            stride: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// GridFnND<F, D>
// ---------------------------------------------------------------------------

/// d-dimensional function sampled on a [`GridND<F, D>`].
///
/// Flat row-major storage with axis 0 fastest. Implements [`State<F>`] for use
/// in the Chernoff iteration loop.
///
/// # Example
///
/// ```rust
/// use semiflow::{Grid1D, grid_nd::{GridND, GridFnND}};
/// let axes = core::array::from_fn(|_| Grid1D::new(-5.0_f64, 5.0, 16).unwrap());
/// let grid = GridND::<f64, 2>::new(axes).unwrap();
/// let f = GridFnND::from_fn(grid.clone(), |x: &[f64; 2]| (-x[0]*x[0] - x[1]*x[1]).exp());
/// assert_eq!(f.values.len(), 256);
/// ```
#[derive(Clone)]
pub struct GridFnND<F: SemiflowFloat = f64, const D: usize = 2> {
    /// Flat sample values. Length `grid.len()`.
    pub values: Vec<F>,
    /// Grid geometry.
    pub grid: GridND<F, D>,
}

impl<F: SemiflowFloat, const D: usize> GridFnND<F, D> {
    /// Construct from grid + pre-computed values.
    ///
    /// # Errors
    /// - `DomainViolation` if `values.len() != grid.len()`.
    /// - `Unsupported` if `grid.interp` has no `D > 1` stencil (ADR-0191). The
    ///   check lives here, once per state, so that [`GridFnND::sample`] cannot
    ///   fail inside the per-node kernel loops for an in-shape coordinate.
    pub fn new(grid: GridND<F, D>, values: Vec<F>) -> Result<Self, SemiflowError> {
        if values.len() != grid.len() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "GridFnND::new: values.len() must equal grid.len()",
                value: values.len() as f64,
            });
        }
        if !crate::interp_stencil::supports_nd(grid.interp) {
            return Err(SemiflowError::Unsupported {
                feature: "GridND::interp for D > 1 (use InterpKind::CubicHermite or Linear)",
            });
        }
        Ok(Self { values, grid })
    }

    /// Sample a closure at every grid node.
    ///
    /// Iterates nodes in row-major order (axis 0 fastest).
    pub fn from_fn<C: Fn(&[F; D]) -> F>(grid: GridND<F, D>, f: C) -> Self {
        let total = grid.len();
        let mut values = Vec::with_capacity(total);
        enumerate_nd(&grid, |_flat, x| {
            values.push(f(x));
        });
        Self { values, grid }
    }

    /// Tensor-product interpolation at an arbitrary d-D point `x` (ADR-0191).
    ///
    /// Evaluates `Σ_{k∈K^D} (Π_d w^{(d)}_{k_d}) · f[idx + o_k]`, where the
    /// per-axis nodal weights come from [`GridND::interp`] and out-of-range
    /// node indices are resolved through each axis's own
    /// [`crate::grid::BoundaryPolicy`] — the same
    /// `crate::boundary::bc_value_by` resolver the 1-D samplers use, so the
    /// two paths cannot disagree about boundary handling.
    ///
    /// Cost is `K^D` node reads per call (`K = 4` for the default
    /// `CubicHermite`, `K = 2` for `Linear`).
    ///
    /// # Errors
    /// - `DomainViolation` if `x.len() != D`.
    /// - `Unsupported` if [`GridND::interp`] has no `D > 1` stencil
    ///   (`SepticHermite`, `OctonicHermite`, `ChebyshevSpectralWithBC`), or for
    ///   `Linear` without the `linear-interp` feature.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn sample(&self, x: &[F]) -> Result<F, SemiflowError> {
        if x.len() != D {
            return Err(SemiflowError::DomainViolation {
                what: "GridFnND::sample: x.len() must equal D",
                value: x.len() as f64,
            });
        }
        let mut plan = [AxisStencil::<F>::zeroed(); D];
        for (d, st) in plan.iter_mut().enumerate() {
            *st = self.axis_stencil(d, x[d])?;
        }
        // Collapse from the slowest axis down; `flat_hi` accumulates the
        // contribution of every axis already pinned.
        Ok(self.collapse(D, &plan, 0))
    }

    /// Weights, resolved boundary hits and stride for axis `d` at coordinate `xd`.
    fn axis_stencil(&self, d: usize, xd: F) -> Result<AxisStencil<F>, SemiflowError> {
        let ax = &self.grid.axes[d];
        let t_frac = (xd - ax.xmin) / ax.dx();
        let t_floor = num_traits::Float::floor(t_frac);
        #[allow(clippy::cast_possible_truncation)]
        let idx = t_floor.to_f64().unwrap_or(0.0) as i64;
        let (k, offsets, weights) = interp_stencil::<F>(self.grid.interp, t_frac - t_floor)?;
        let mut hits = [BoundaryHit::Inside(0); K_MAX];
        for (j, hit) in hits.iter_mut().enumerate().take(k) {
            *hit = bc_index(ax.boundary, ax.n, idx + offsets[j]);
        }
        Ok(AxisStencil {
            k,
            weights,
            hits,
            stride: self.axis_stride(d),
        })
    }

    /// Interpolate the `remaining` fastest axes (`0..remaining`), with every
    /// axis at or above `remaining` already pinned into `flat_hi`; `remaining
    /// == 0` is the base case. The depth counts *axes still to resolve* so the
    /// recursion terminates on `usize`.
    ///
    /// Out-of-range stencil nodes fold through each axis's own
    /// [`crate::grid::BoundaryPolicy`] — including the affine ones
    /// (`LinearExtrapolate`, `Dirichlet`, `Robin`), which a flat index map could
    /// not express — via the hit resolved in [`Self::axis_stencil`].
    fn collapse(&self, remaining: usize, plan: &[AxisStencil<F>; D], flat_hi: usize) -> F {
        let Some(d) = remaining.checked_sub(1) else {
            return self.values[flat_hi];
        };
        let ax = &self.grid.axes[d];
        let st = &plan[d];
        let mut acc = F::zero();
        for j in 0..st.k {
            let v = bc_value_from_hit(
                st.hits[j],
                ax.boundary,
                |i| self.collapse(d, plan, flat_hi + i * st.stride),
                ax.n,
                ax.dx(),
            );
            acc += st.weights[j] * v;
        }
        acc
    }

    /// Flat-index stride of axis `d`: `Π_{e<d} n_e` (axis 0 is fastest).
    fn axis_stride(&self, d: usize) -> usize {
        self.grid.axes[..d].iter().map(|ax| ax.n).product()
    }
}

// ---------------------------------------------------------------------------
// Row-major node enumeration helper
// ---------------------------------------------------------------------------

/// Enumerate all `D`-dimensional nodes in row-major order, calling `f` with
/// `(flat_index, [x₀, x₁, …, x_{D-1}])`.
pub(crate) fn enumerate_nd<F, const D: usize, C>(grid: &GridND<F, D>, mut callback: C)
where
    F: SemiflowFloat,
    C: FnMut(usize, &[F; D]),
{
    let ns: [usize; D] = core::array::from_fn(|d| grid.axes[d].n);
    let total: usize = ns.iter().product();
    for flat in 0..total {
        let mut remaining = flat;
        let mut x = [F::zero(); D];
        for d in 0..D {
            let k = remaining % ns[d];
            x[d] = grid.x_at(d, k);
            remaining /= ns[d];
        }
        callback(flat, &x);
    }
}

// ---------------------------------------------------------------------------
// State<F> impl for GridFnND<F, D>
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat, const D: usize> State<F> for GridFnND<F, D> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn axpy_into(&mut self, alpha: F, src: &Self) {
        debug_assert_eq!(self.values.len(), src.values.len());
        for (s, &x) in self.values.iter_mut().zip(src.values.iter()) {
            *s += alpha * x;
        }
    }

    #[inline]
    fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.values.len(), src.values.len());
        self.values.copy_from_slice(&src.values);
    }

    #[inline]
    fn zero_into(&mut self) {
        for v in &mut self.values {
            *v = F::zero();
        }
    }

    #[inline]
    fn norm_sup(&self) -> F {
        self.values.iter().copied().fold(F::zero(), |acc, v| {
            let av = v.abs();
            if av > acc {
                av
            } else {
                acc
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Inline unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::grid::Grid1D;

    fn make_2d_grid() -> GridND<f64, 2> {
        let ax = Grid1D::new(-5.0_f64, 5.0, 16).unwrap();
        GridND::new([ax, ax]).unwrap()
    }

    #[test]
    fn grid_nd_len_2d() {
        let g = make_2d_grid();
        assert_eq!(g.len(), 256); // 16*16
    }

    #[test]
    fn flat_idx_round_trip_2d() {
        let g = make_2d_grid();
        let idx = [3_usize, 7_usize];
        let flat = g.flat_idx(&idx);
        // axis 0 fastest: flat = 7*16 + 3 = 115
        assert_eq!(flat, 7 * 16 + 3);
    }

    #[test]
    fn gridfn_nd_from_fn_len() {
        let g = make_2d_grid();
        let f = GridFnND::from_fn(g, |x: &[f64; 2]| x[0] * x[1]);
        assert_eq!(f.values.len(), 256);
    }

    #[test]
    fn gridfn_nd_sample_at_node() {
        let ax = Grid1D::new(0.0_f64, 1.0, 4).unwrap();
        let g = GridND::<f64, 2>::new([ax, ax]).unwrap();
        // f(x, y) = x + y
        let f = GridFnND::from_fn(g, |x: &[f64; 2]| x[0] + x[1]);
        // Sample at (1/3, 2/3) — grid nodes at k=1 and k=2
        let x_at_1 = ax.x_at(1); // 1/3
        let x_at_2 = ax.x_at(2); // 2/3
        let sampled = f.sample(&[x_at_1, x_at_2]).unwrap();
        let expected = x_at_1 + x_at_2;
        assert!(
            (sampled - expected).abs() < 1e-12,
            "sample at node: {sampled} != {expected}"
        );
    }

    #[test]
    fn gridfn_nd_state_zero_into() {
        let g = make_2d_grid();
        let mut f = GridFnND::from_fn(g, |_: &[f64; 2]| 1.0_f64);
        f.zero_into();
        assert_eq!(f.norm_sup(), 0.0);
    }

    #[test]
    fn gridfn_nd_state_axpy() {
        let ax = Grid1D::new(-1.0_f64, 1.0, 4).unwrap();
        let g = GridND::<f64, 2>::new([ax, ax]).unwrap();
        let mut f = GridFnND::from_fn(g.clone(), |_: &[f64; 2]| 1.0_f64);
        let src = GridFnND::from_fn(g, |_: &[f64; 2]| 2.0_f64);
        f.axpy_into(3.0, &src); // f = 1 + 3*2 = 7
        let all_seven = f.values.iter().all(|&v| (v - 7.0).abs() < 1e-12);
        assert!(all_seven, "axpy_into: expected all 7.0");
    }
}
