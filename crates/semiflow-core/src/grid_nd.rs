//! [`GridND`] and [`GridFnND`] — d-dimensional tensor-product grid (v4.0 Wave C).
//!
//! Generic over `F: SemiflowFloat` and `const D: usize` (dimension). Each axis
//! is an independent [`Grid1D<F>`] with its own boundary policy and interp kind.
//!
//! Storage convention (row-major, fastest axis = axis 0):
//! `idx(k₀, k₁, ..., k_{D-1}) = k_{D-1}·n_{D-2}·…·n₀ + … + k₁·n₀ + k₀`.
//!
//! This is the `GridFnND<F, D>` state type used by
//! [`crate::shift_nd::AnisotropicShiftChernoffND<F, D>`] (math.md §32,
//! ADR-0081) and [`crate::hormander::HypoellipticChernoff`] (future generic).
//!
//! v4.0 scope: construction + sampling (linear interp) + `State<F>` impl.
//! Cubic Hermite per-axis interp deferred to v4.x (same priority as
//! `GridFn1D` cubic hermite extension).

use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::Grid1D,
    state::State,
};

// ---------------------------------------------------------------------------
// GridND<F, D>
// ---------------------------------------------------------------------------

/// d-dimensional tensor-product grid with uniform axes.
///
/// Each axis is a [`Grid1D<F>`] with independent geometry, boundary policy,
/// and interpolation kind. Dimension `D` is a const generic parameter.
///
/// ## Storage convention
///
/// Values are stored row-major with axis 0 fastest:
/// `flat_idx = k_{D-1}·n_{D-2}·…·n₀ + … + k₁·n₀ + k₀`.
///
/// # Example
///
/// ```rust
/// use semiflow_core::{Grid1D, grid_nd::GridND};
/// let axes = core::array::from_fn(|_| Grid1D::new(-5.0_f64, 5.0, 16).unwrap());
/// let grid = GridND::<f64, 2>::new(axes).unwrap();
/// assert_eq!(grid.len(), 256); // 16*16
/// ```
#[derive(Clone)]
pub struct GridND<F: SemiflowFloat = f64, const D: usize = 2> {
    /// Per-axis grids. `axes[0]` is the fastest-varying axis.
    pub axes: [Grid1D<F>; D],
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
        Ok(Self { axes })
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
/// use semiflow_core::{Grid1D, grid_nd::{GridND, GridFnND}};
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
    pub fn new(grid: GridND<F, D>, values: Vec<F>) -> Result<Self, SemiflowError> {
        if values.len() != grid.len() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "GridFnND::new: values.len() must equal grid.len()",
                value: values.len() as f64,
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

    /// Linear interpolation at an arbitrary d-D point `x`.
    ///
    /// Uses multilinear (tensor-product linear) interpolation. Out-of-range
    /// coordinates are clamped to the grid boundary (`ZeroExtend` → 0; Reflect
    /// → reflected index; default Reflect clamps to nearest node if beyond
    /// the stencil). For v4.0 we use CLAMPED linear interp for simplicity.
    ///
    /// # Errors
    /// - `DomainViolation` if `x.len() != D`.
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
        // Per-axis fractional indices (clamped to [0, n-2]).
        let mut frac = [F::zero(); D];
        let mut lo = [0_usize; D];
        for d in 0..D {
            let ax = &self.grid.axes[d];
            let xi = (x[d] - ax.xmin) / (ax.xmax - ax.xmin) * from_f64::<F>((ax.n - 1) as f64);
            let xi = xi.max(F::zero()).min(from_f64::<F>((ax.n - 2) as f64));
            lo[d] = xi.to_f64().unwrap_or(0.0) as usize;
            frac[d] = xi - from_f64::<F>(lo[d] as f64);
        }
        // Multilinear interp over 2^D corners.
        let n_corners = 1_usize << D;
        let mut result = F::zero();
        for corner in 0..n_corners {
            let mut weight = F::one();
            let mut idx = [0_usize; D];
            for d in 0..D {
                let hi_bit = (corner >> d) & 1;
                if hi_bit == 1 {
                    weight *= frac[d];
                    idx[d] = lo[d] + 1;
                } else {
                    weight *= F::one() - frac[d];
                    idx[d] = lo[d];
                }
            }
            let flat = self.grid.flat_idx(&idx);
            result += weight * self.values[flat];
        }
        Ok(result)
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
