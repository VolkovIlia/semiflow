//! v4.0 A2 — `AnisotropicShiftChernoffND<F, const D: usize>` flagship (ADR-0081, math.md §32).
//!
//! General d-dimensional anisotropic shift Chernoff for parabolic
//! `L = Σᵢⱼ aᵢⱼ(x) ∂²ᵢⱼ + Σᵢ bᵢ(x) ∂ᵢ + c(x)`.
//!
//! Algorithm (math §32.4 eq 32.3): at each grid point `x_k`, compute
//! `F(τ) f(x_k) ≈ exp(τ·c(x_k)) · π^{-D/2} · Σ_q w_q · f(x_k + τ·b(x_k) + 2√τ·σ_k·η_q)`
//! where `η_q` are the 5-pt Gauss-Hermite tensor nodes and `σ_k` is the
//! Cholesky factor of `A(x_k)` (pre-computed at construction, math §32.4 step 1).
//! Node scale `2√τ` (NOT `√(2τ)`) and normalization `π^{-D/2}` are both required
//! for `F(0)=I` (ADR-0112).
//!
//! CITATION: Remizov 2025 *Vladikavkaz Math. J.* 27:4 Theorem 3 cousin formula
//! (d-D anisotropic shift Chernoff product). Friedman 1964 §1.4 (anisotropic
//! Gaussian fundamental solutions).
//!
//! ## Cohort 7 — constitution v1.8.0 Override #1
//!
//! HARD LIMIT: 700 `LoC`. This file co-locates:
//! - `AnisotropicShiftChernoffND<F, D>` + `GaussHermiteTensor<F, D>` structs
//! - `ChernoffFunction<F>` impl (`apply_into` per math §32.4)
//! - `ApproximationSubspace<2, F>` impl (FIFTH v3.x K=2 opt-in)
//! - `SquareMatrix<F, D>` helper (in-crate `DxD` matrix; no nalgebra dep)
//! - Cholesky cache management
//! - Inline unit tests

// D is a small const generic (PDE dimension, typically 1–5): D as u32 and D as f64 are safe.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use alloc::{boxed::Box, vec::Vec};
use core::marker::PhantomData;

use crate::{
    approximation::ApproximationSubspace,
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid_nd::{GridFnND, GridND},
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// SquareMatrix<F, D> — minimal D×D column-major matrix (no new dep)
// ---------------------------------------------------------------------------

/// Column capacity for `SquareMatrix`: 6×6 = 36 entries, supporting D ≤ 6.
///
/// Single-sourced here so that the field type, `zero()` literal, and the
/// `debug_assert` all reference the same constant rather than a repeated `36`.
/// TODO: remove when `generic_const_exprs` stabilises (migrate to `[F; D*D]`).
const SQMAT_CAP: usize = 36;

/// Minimal D×D matrix for Cholesky factor and diffusion tensor storage.
///
/// Column-major storage: `m[i + j*D]` is row `i`, column `j`.
/// Used exclusively inside this module; not part of the public API.
///
/// Storage is `[F; SQMAT_CAP]` (6×6 = 36), supporting D ≤ 6.
/// TODO: migrate to `[F; D*D]` when `generic_const_exprs` stabilises —
/// that removes the 11-slot waste for D≤5 and lifts the D≤6 ceiling
/// without any logic change (slots 0..D*D-1 are the only ones touched).
#[derive(Clone)]
pub struct SquareMatrix<F, const D: usize> {
    /// Column-major flat storage. Length D*D; only first D*D slots used.
    /// Sized for D≤6 (6×6=36). D=5 uses 25 slots; D=6 uses all 36.
    pub data: [F; SQMAT_CAP],
    _phantom: PhantomData<F>,
}

impl<F: SemiflowFloat, const D: usize> SquareMatrix<F, D> {
    /// Zero matrix.
    ///
    /// # Panics (debug only)
    /// Panics if `D > 6` — `SquareMatrix` supports D≤6; D≥7 requires
    /// `generic_const_exprs` to use `[F; D*D]`.
    #[must_use]
    pub fn zero() -> Self {
        debug_assert!(
            D * D <= SQMAT_CAP,
            "SquareMatrix supports D≤6; D={D} needs generic_const_exprs"
        );
        Self {
            data: [F::zero(); SQMAT_CAP],
            _phantom: PhantomData,
        }
    }

    /// `DxD` identity matrix.
    ///
    /// # Panics (debug only)
    /// Panics if `D > 6` — same limit as `zero()`.
    #[must_use]
    pub fn identity() -> Self {
        let mut m = Self::zero(); // debug_assert fires inside zero()
        for i in 0..D {
            m.data[i + i * D] = F::one();
        }
        m
    }

    /// Element access (row i, col j).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> F {
        self.data[i + j * D]
    }

    /// Mutable element access (row i, col j).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: F) {
        self.data[i + j * D] = v;
    }

    /// Multiply matrix `self` (`DxD`) by vector `v` (D), writing result into `out`.
    pub fn mat_vec(&self, v: &[F; D], out: &mut [F; D]) {
        // i and j index different dimensions simultaneously; range loop is necessary.
        #[allow(clippy::needless_range_loop)]
        for i in 0..D {
            let mut s = F::zero();
            for j in 0..D {
                s += self.get(i, j) * v[j];
            }
            out[i] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Cholesky factorization of A = L·Lᵀ (lower-triangular, in-place on L)
// ---------------------------------------------------------------------------

/// Cholesky factor L of a symmetric-positive-definite D×D matrix A.
///
/// Computes lower-triangular L such that A = L·Lᵀ via the standard
/// Cholesky-Banachiewicz algorithm (O(D³/3)).
///
/// Returns `Err` if A is not strictly positive-definite (diagonal pivot ≤ 0).
fn cholesky_factor<F: SemiflowFloat, const D: usize>(
    a: &SquareMatrix<F, D>,
    l: &mut SquareMatrix<F, D>,
) -> Result<(), SemiflowError> {
    *l = SquareMatrix::zero();
    for j in 0..D {
        let mut diag = a.get(j, j);
        for k in 0..j {
            diag -= l.get(j, k) * l.get(j, k);
        }
        if diag <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "AnisotropicShiftChernoffND: diffusion tensor not SPD at grid point",
                value: diag.to_f64().unwrap_or(f64::NAN),
            });
        }
        l.set(j, j, diag.sqrt());
        let ljj = l.get(j, j);
        for i in (j + 1)..D {
            let mut s = a.get(i, j);
            for k in 0..j {
                s -= l.get(i, k) * l.get(j, k);
            }
            l.set(i, j, s / ljj);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GaussHermiteTensor<F, D>
// ---------------------------------------------------------------------------

/// Pre-computed 5-pt tensor-product Gauss-Hermite quadrature for D dimensions.
///
/// The 1-D 5-pt rule integrates `∫ exp(-t²) p(t) dt` exactly for degree ≤ 9
/// (Abramowitz-Stegun §25.4.46). Total nodes: `5^D`. Weights satisfy
/// `Σ wᵢ = √π` (1-D sum) so the D-D product weight sum is `π^(D/2)`.
///
/// Nodes stored in increasing order; weights are positive.
pub struct GaussHermiteTensor<F: SemiflowFloat = f64, const D: usize = 2> {
    /// 5-pt 1-D nodes (shared across all axes).
    pub nodes: [F; 5],
    /// 5-pt 1-D weights (shared across all axes).
    pub weights: [F; 5],
}

// Gauss-Hermite constants extracted to shift_nd_gauss.rs to keep this file ≤500 lines.
// Re-exported here so existing `crate::shift_nd::GH*` paths still resolve.
pub(crate) use crate::shift_nd_gauss::{
    GH1_NODES_F64, GH1_WEIGHTS_F64, GH3_NODES_F64, GH3_WEIGHTS_F64, GH5_NODES_F64, GH5_WEIGHTS_F64,
    GH7_NODES_F64, GH7_WEIGHTS_F64, GH9_NODES_F64, GH9_WEIGHTS_F64,
};

impl<F: SemiflowFloat, const D: usize> GaussHermiteTensor<F, D> {
    /// Construct from canonical 5-pt Gauss-Hermite rule (Abramowitz-Stegun).
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: core::array::from_fn(|i| from_f64::<F>(GH5_NODES_F64[i])),
            weights: core::array::from_fn(|i| from_f64::<F>(GH5_WEIGHTS_F64[i])),
        }
    }

    /// Total number of tensor nodes: `5^D`.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        5_usize.pow(D as u32)
    }

    /// Tensor node `q_index` as a D-vector of 1-D Gauss-Hermite nodes.
    ///
    /// `q_index` runs `0..5^D`; axis 0 is the fastest-varying digit in base-5.
    #[must_use]
    pub fn node(&self, q_index: usize) -> [F; D] {
        let mut q = q_index;
        core::array::from_fn(|_d| {
            let digit = q % 5;
            q /= 5;
            self.nodes[digit]
        })
    }

    /// Product weight for tensor node `q_index`.
    #[must_use]
    pub fn weight(&self, q_index: usize) -> F {
        let mut q = q_index;
        let mut w = F::one();
        for _d in 0..D {
            let digit = q % 5;
            q /= 5;
            w *= self.weights[digit];
        }
        w
    }
}

impl<F: SemiflowFloat, const D: usize> Default for GaussHermiteTensor<F, D> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AnisotropicShiftChernoffND<F, D>
// ---------------------------------------------------------------------------

/// d-D anisotropic shift Chernoff kernel (ADR-0081, math.md §32).
///
/// Generic over floating-point type `F` and dimension `D`. Reference
/// implementations gated for `D ∈ {2, 3, 4, 5}` (`G_DDIM` gate); `D ≥ 6`
/// compiles but is not gated in v4.0 (Smolyak sparse grid deferred to v4.1+).
///
/// # Construction
///
/// ```rust,ignore
/// # use semiflow::{Grid1D, AnisotropicShiftChernoffND, grid_nd::GridND};
/// # let axes = core::array::from_fn(|_| Grid1D::new(-5.0_f64, 5.0, 16).unwrap());
/// # let grid = GridND::<f64, 2>::new(axes).unwrap();
/// let kernel = AnisotropicShiftChernoffND::<f64, 2>::new(
///     |x, a_out| {
///         a_out.set(0,0, 1.0); a_out.set(1,1, 1.0);
///         a_out.set(0,1, 0.25*(x[0]+x[1]).tanh()); a_out.set(1,0, a_out.get(0,1));
///     },
///     |_x, b_out| { b_out[0] = 0.0; b_out[1] = 0.0; },
///     |_x| 0.0_f64,
///     grid,
/// ).unwrap();
/// ```
// Type aliases for the coefficient field closures used in AnisotropicShiftChernoffND.
type AijField<F, const D: usize> = Box<dyn Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync>;
type BiField<F, const D: usize> = Box<dyn Fn(&[F; D], &mut [F; D]) + Send + Sync>;
type CField<F, const D: usize> = Box<dyn Fn(&[F; D]) -> F + Send + Sync>;

/// Anisotropic D-dimensional shift Chernoff function (see module doc for full description).
pub struct AnisotropicShiftChernoffND<F: SemiflowFloat = f64, const D: usize = 2> {
    /// Diffusion tensor `a_ij(x)` (SPD at every grid point). Stored for potential
    /// runtime re-evaluation (e.g. adaptive tau).
    // accessed from shift_nd_zeta2.rs via op.a_ij; rustc dead_code fires in non-zeta2 builds
    #[allow(dead_code)]
    a_ij: AijField<F, D>,
    /// Drift vector `b_i(x)`.
    b_i: BiField<F, D>,
    /// Reaction coefficient `c(x)`.
    c: CField<F, D>,
    /// Cached Cholesky factor `L(x_k)` at each grid point (math §32.4 step 1).
    cholesky_cache: Vec<SquareMatrix<F, D>>,
    /// Pre-computed 5-pt tensor-product Gauss-Hermite quadrature.
    quadrature: GaussHermiteTensor<F, D>,
    /// Reference grid geometry.
    grid: GridND<F, D>,
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat, const D: usize> AnisotropicShiftChernoffND<F, D> {
    /// Construct the kernel, pre-computing Cholesky factors at each grid point.
    ///
    /// # Errors
    /// - `DomainViolation` if `D == 0`.
    /// - `DomainViolation` if the grid has fewer than `5^D` points total.
    /// - `DomainViolation` if `a_ij` is not SPD at any grid point.
    /// - `DomainViolation` if `a_ij` produces any non-finite values.
    pub fn new(
        a_ij: impl Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync + 'static,
        b_i: impl Fn(&[F; D], &mut [F; D]) + Send + Sync + 'static,
        c: impl Fn(&[F; D]) -> F + Send + Sync + 'static,
        grid: GridND<F, D>,
    ) -> Result<Self, SemiflowError> {
        if D == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "AnisotropicShiftChernoffND: D must be >= 1",
                value: 0.0,
            });
        }
        let min_pts = 5_usize.pow(D as u32);
        if grid.len() < min_pts {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "AnisotropicShiftChernoffND: grid.len() must be >= 5^D",
                value: grid.len() as f64,
            });
        }
        let cache = build_cholesky_cache(&a_ij, &grid)?;
        Ok(Self {
            a_ij: Box::new(a_ij),
            b_i: Box::new(b_i),
            c: Box::new(c),
            cholesky_cache: cache,
            quadrature: GaussHermiteTensor::new(),
            grid,
            _f: PhantomData,
        })
    }

    /// Return a shared reference to the kernel's grid geometry.
    ///
    /// Tests and downstream code use this to construct initial [`GridFnND`] values
    /// without needing to clone the grid independently.
    pub fn grid(&self) -> &GridND<F, D> {
        &self.grid
    }
}

/// Build the Cholesky cache: `L(x_k)` for every grid point `k`.
pub(crate) fn build_cholesky_cache<F: SemiflowFloat, const D: usize>(
    a_ij: &dyn Fn(&[F; D], &mut SquareMatrix<F, D>),
    grid: &GridND<F, D>,
) -> Result<Vec<SquareMatrix<F, D>>, SemiflowError> {
    let total = grid.len();
    let mut cache = Vec::with_capacity(total);
    let ns: [usize; D] = core::array::from_fn(|d| grid.axes[d].n);
    for flat in 0..total {
        let x = flat_to_x::<F, D>(flat, &ns, grid);
        let mut a = SquareMatrix::zero();
        a_ij(&x, &mut a);
        let mut l = SquareMatrix::zero();
        cholesky_factor(&a, &mut l)?;
        cache.push(l);
    }
    Ok(cache)
}

/// Recover physical coordinates from flat index.
#[inline]
pub(crate) fn flat_to_x<F: SemiflowFloat, const D: usize>(
    flat: usize,
    ns: &[usize; D],
    grid: &GridND<F, D>,
) -> [F; D] {
    let mut remaining = flat;
    core::array::from_fn(|d| {
        let k = remaining % ns[d];
        remaining /= ns[d];
        grid.x_at(d, k)
    })
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl — math §32.4 algorithm
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat, const D: usize> ChernoffFunction<F> for AnisotropicShiftChernoffND<F, D> {
    type S = GridFnND<F, D>;

    /// Apply one step: `f_dst[k] = exp(τ·c(x_k)) · Σ_q w_q · f_src(x_k + τ·b(x_k) + √(2τ)·L_k·η_q)`.
    ///
    /// Per math §32.4 equation 32.3. Cost: O(N^D · 5^D) per step.
    #[allow(clippy::too_many_lines)]
    fn apply_into(
        &self,
        tau: F,
        src: &GridFnND<F, D>,
        dst: &mut GridFnND<F, D>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "AnisotropicShiftChernoffND::apply_into: tau must be finite >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let ns: [usize; D] = core::array::from_fn(|d| self.grid.axes[d].n);
        let total = src.values.len();
        // Node scale: 2√τ (NOT √(2τ)) — required for correct heat variance 2τA (ADR-0112).
        let two_sqrt_tau = from_f64::<F>(2.0_f64) * tau.sqrt();
        // Normalization: π^{-D/2} — required for F(0)=I (ADR-0112 §Decision 1).
        let inv_pi_dhalf =
            from_f64::<F>(core::f64::consts::PI).powf(from_f64::<F>(-(D as f64) / 2.0_f64));
        let n_q = self.quadrature.n_nodes();
        for flat in 0..total {
            let xk = flat_to_x::<F, D>(flat, &ns, &self.grid);
            let mut b_val = [F::zero(); D];
            (self.b_i)(&xk, &mut b_val);
            let c_val = (self.c)(&xk);
            let exp_factor = (tau * c_val).exp();
            let l_k = &self.cholesky_cache[flat];
            let mut acc = F::zero();
            for q_idx in 0..n_q {
                let eta_q = self.quadrature.node(q_idx);
                let w_q = self.quadrature.weight(q_idx);
                let mut l_eta = [F::zero(); D];
                l_k.mat_vec(&eta_q, &mut l_eta);
                // y_q = x_k + tau * b(x_k) + 2√τ * L_k * eta_q  (ADR-0112 eq 32.3)
                let y_q: [F; D] =
                    core::array::from_fn(|d| xk[d] + tau * b_val[d] + two_sqrt_tau * l_eta[d]);
                let f_y = src.sample(&y_q).unwrap_or(F::zero());
                acc += w_q * f_y;
            }
            dst.values[flat] = exp_factor * inv_pi_dhalf * acc;
        }
        Ok(())
    }

    /// Order 1: frozen-coefficient kernel, math §32.5 (ADR-0112).
    ///
    /// Exact for constant A; O(1/n) global error for variable A (empirical 1D
    /// variable-a self-convergence slope −1.02, sympy-proven τ² residual).
    /// A genuine order-2 d-D lift requires explicit ∂A gradient/Hessian closures
    /// plus a τ² correction polynomial; deferred to math §32.6.
    fn order(&self) -> u32 {
        1
    }

    /// Growth bound: multiplier 1.5, omega 0 (per ADR-0081 §"Rationale").
    fn growth(&self) -> Growth<F> {
        Growth::new(from_f64::<F>(1.5_f64), F::zero())
    }
}

// ---------------------------------------------------------------------------
// ApproximationSubspace<2, F> — FIFTH v3.x K=2 opt-in (ADR-0073, ADR-0081)
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat, const D: usize> ApproximationSubspace<2, F>
    for AnisotropicShiftChernoffND<F, D>
{
    /// Returns `true` iff the grid has at least 5 nodes per axis AND all values
    /// are finite (proxy for `f ∈ C²_b` core per Theorem 32.1 preconditions).
    fn in_subspace(&self, f: &GridFnND<F, D>) -> bool {
        self.grid.axes.iter().all(|ax| ax.n >= 5) && f.values.iter().all(|v| v.is_finite())
    }

    /// Jet stub: returns `Unsupported` (same pattern as v3.0 K=2 trio).
    fn jet(&self, _f: &GridFnND<F, D>, _out: &mut [GridFnND<F, D>]) -> Result<(), SemiflowError> {
        Err(SemiflowError::Unsupported {
            feature: "AnisotropicShiftChernoffND::jet (v4.x opportunity)",
        })
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Unit tests (extracted to keep file under 500 lines)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "shift_nd_tests.rs"]
mod tests;
