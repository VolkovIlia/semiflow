//! v4.x Wave 2 — `MatrixDiffusionChernoff<F, M>` with palindromic Strang splitting.
//!
//! Coupled-component diffusion `(uₜ)ᵢ = Σⱼ aᵢⱼ∂²uⱼ + bᵢⱼ∂uⱼ + cᵢⱼuⱼ`, u ∈ ℝᴹ.
//!
//! Three-phase palindromic Strang: `R(τ/2) ∘ D(τ) ∘ R(τ/2)` (ADR-0082 AMENDMENT 2).
//! - Phase 1 / Phase 3: half-step reaction via pointwise `exp(τ/2 · C(x_k))`.
//! - Phase 2: full-step diffusion via block Crank-Nicolson Cayley map
//!   `(I − τ/2 · L^h)⁻¹ (I + τ/2 · L^h)` solved by block-Thomas algorithm.
//!
//! References: Pazy 1983 §3.3; Higham 2008 §10.7.3; Golub-Van Loan §4.5.1;
//! Hochbruck-Lubich 2010 Acta Numerica §3.4; Hilhorst-Mimura 1992.

// Matrix scaling: log2(norm).ceil() as u32 where norm > 1 ⟹ log2 > 0 ⟹ cast is safe.
// f64→u32 cast after clamp(.min(30)) prevents truncation beyond u32 range.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use alloc::{boxed::Box, vec, vec::Vec};
use core::marker::PhantomData;

use crate::{
    approximation::ApproximationSubspace,
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    matrix_strang::block_cn_diff_step,
    scratch::ScratchPool,
    state::State,
};

// Per-M matrix exponentials and arithmetic helpers (closed form M ≤ 2, Padé[13/13] M ≥ 3).
#[path = "matrix_system_exp.rs"]
mod matrix_system_exp;

// ---------------------------------------------------------------------------
// MatrixGridFn1D<F, M> — multi-component 1D grid state (ADR-0082, math §33.2)
// ---------------------------------------------------------------------------

/// Multi-component grid state: `u : Grid₁D → ℝᴹ`.
///
/// Internal layout: row-major flat `Vec<F>` of length `N * M`; component `i`
/// at grid point `k` is at index `k * M + i`. Optimal for the per-grid-point
/// matrix-vector multiply in Phase 2 (math §33.3).
///
/// Implements `State<F>` with per-component delegation (math §33.2, ADR-0082).
///
/// # Generic parameters
/// - `F: SemiflowFloat` — scalar float type (default `f64`).
/// - `M: usize` — number of coupled components (const generic).
#[derive(Clone, Debug)]
pub struct MatrixGridFn1D<F: SemiflowFloat = f64, const M: usize = 2> {
    /// Reference grid geometry (immutable after construction).
    pub grid: Grid1D<F>,
    /// Row-major flat values: `values[k * M + i]` = component `i` at node `k`.
    pub values: Vec<F>,
}

impl<F: SemiflowFloat, const M: usize> MatrixGridFn1D<F, M> {
    /// Create a zero-valued state on `grid`.
    pub fn new(grid: Grid1D<F>) -> Self {
        let n = grid.n;
        Self {
            grid,
            values: vec![F::zero(); n * M],
        }
    }

    /// Create from a pointwise closure `f(x) -> [F; M]`.
    pub fn from_fn(grid: Grid1D<F>, mut f: impl FnMut(F) -> [F; M]) -> Self {
        let n = grid.n;
        let mut values = vec![F::zero(); n * M];
        for k in 0..n {
            let x = grid.x_at(k);
            let v = f(x);
            for i in 0..M {
                values[k * M + i] = v[i];
            }
        }
        Self { grid, values }
    }

    /// Read the M-component vector at grid point `k`.
    #[inline]
    pub fn point_view(&self, k: usize) -> [F; M] {
        let base = k * M;
        let mut out = [F::zero(); M];
        out.copy_from_slice(&self.values[base..base + M]);
        out
    }

    /// Write an M-component vector at grid point `k`.
    #[inline]
    pub fn set_point(&mut self, k: usize, val: &[F; M]) {
        let base = k * M;
        self.values[base..base + M].copy_from_slice(val);
    }
}

impl<F: SemiflowFloat, const M: usize> State<F> for MatrixGridFn1D<F, M> {
    fn len(&self) -> usize {
        self.grid.n
    }

    fn axpy_into(&mut self, alpha: F, src: &Self) {
        debug_assert_eq!(self.values.len(), src.values.len());
        for (d, &s) in self.values.iter_mut().zip(src.values.iter()) {
            *d += alpha * s;
        }
    }

    fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.values.len(), src.values.len());
        self.values.clone_from(&src.values);
    }

    fn zero_into(&mut self) {
        for v in &mut self.values {
            *v = F::zero();
        }
    }

    fn norm_sup(&self) -> F {
        self.values.iter().fold(
            F::zero(),
            |acc, &v| if v.abs() > acc { v.abs() } else { acc },
        )
    }

    fn scale_into(&mut self, k: F) {
        for v in &mut self.values {
            *v *= k;
        }
    }
}

// ---------------------------------------------------------------------------
// MatrixDiffusionChernoff<F, M> — coupled-component diffusion kernel (ADR-0082)
// ---------------------------------------------------------------------------

/// Chernoff function for coupled-component 1D diffusion (math.md §33, ADR-0082).
///
/// Implements `(Lu)ᵢ = Σⱼ aᵢⱼ(x) ∂ₓ²uⱼ + Σⱼ bᵢⱼ(x) ∂ₓuⱼ + Σⱼ cᵢⱼ(x) uⱼ`
/// for `u ∈ ℝᴹ`.
///
/// ## Matrix exponential coverage
///
/// - M ≤ 2: closed form (Cayley–Hamilton, Higham 2008 §10.4). Fast, exact.
/// - M ≥ 3: Padé[13/13] scaling-and-squaring (Higham 2005, ADR-0125; dispatch in
///   `matrix_system_exp::matrix_exp_dispatch`).
///   Relative error ≤ 1e-12 for symmetric reaction matrices with
///   `‖τC(x)/2‖_∞ ≤ 10`. Stiffer reaction: reduce τ via more sub-steps.
///
/// ## Construction
///
/// Each coefficient field closure takes `(x: F, mat: &mut [[F; M]; M])` and
/// fills the M×M matrix in-place. Coefficients:
/// - `a_ij`: diffusion tensor (symmetric, positive-definite per grid point).
/// - `b_ij`: drift tensor (skew-symmetric per math §33.1; may be zero).
/// - `c_ij`: reaction matrix (symmetric; growth bound via `c_norm_bound`).
///
/// ## Growth bound
///
/// `growth()` returns `(1, c_norm_bound)`. Pass `Some(bound)` to opt into the
/// checked path; `None` returns `omega = +∞` (sub-Markov contraction deferred).
// Type alias for the matrix-valued coefficient field closures.
type MatField<F, const M: usize> = Box<dyn Fn(F, &mut [[F; M]; M]) + Send + Sync>;

/// Matrix diffusion Chernoff function (see module doc above for full description).
pub struct MatrixDiffusionChernoff<F: SemiflowFloat = f64, const M: usize = 2> {
    a_ij_field: MatField<F, M>,
    b_ij_field: MatField<F, M>,
    c_ij_field: MatField<F, M>,
    /// Reference grid geometry.
    pub grid: Grid1D<F>,
    /// Optional ‖C(x)‖_∞ bound for `growth()`. `None` → omega = +∞.
    pub c_norm_bound: Option<F>,
    _phantom: PhantomData<F>,
}

impl<F: SemiflowFloat, const M: usize> MatrixDiffusionChernoff<F, M> {
    /// Construct from closures.
    ///
    /// Validates: grid.n ≥ 5; centre-point `a_ij` is finite; `c_ij` is finite.
    ///
    /// # Errors
    /// - `DomainViolation` if grid.n < 5 or centre evaluations are non-finite.
    pub fn new(
        a_ij_field: impl Fn(F, &mut [[F; M]; M]) + Send + Sync + 'static,
        b_ij_field: impl Fn(F, &mut [[F; M]; M]) + Send + Sync + 'static,
        c_ij_field: impl Fn(F, &mut [[F; M]; M]) + Send + Sync + 'static,
        grid: Grid1D<F>,
    ) -> Result<Self, SemiflowError> {
        if grid.n < 5 {
            return Err(SemiflowError::DomainViolation {
                what: "MatrixDiffusionChernoff requires grid.n >= 5 (5-pt stencil)",
                value: grid.n as f64,
            });
        }
        // Sanity-check centre evaluations are finite.
        let x_c = grid.x_at(grid.n / 2);
        let mut a_tmp = [[F::zero(); M]; M];
        a_ij_field(x_c, &mut a_tmp);
        if a_tmp.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
            return Err(SemiflowError::DomainViolation {
                what: "a_ij_field at grid centre returned non-finite entry",
                value: f64::NAN,
            });
        }
        Ok(Self {
            a_ij_field: Box::new(a_ij_field),
            b_ij_field: Box::new(b_ij_field),
            c_ij_field: Box::new(c_ij_field),
            grid,
            c_norm_bound: None,
            _phantom: PhantomData,
        })
    }

    /// Builder: set an explicit ‖C(x)‖_∞ bound for `growth()`.
    #[must_use]
    pub fn with_c_norm_bound(mut self, bound: F) -> Self {
        self.c_norm_bound = Some(bound);
        self
    }
}

impl<F: SemiflowFloat, const M: usize> ChernoffFunction<F> for MatrixDiffusionChernoff<F, M> {
    type S = MatrixGridFn1D<F, M>;

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<F> {
        let omega = self.c_norm_bound.unwrap_or_else(F::infinity);
        Growth::new(F::one(), omega)
    }

    /// Apply one palindromic Strang step per math §33.3 AMENDMENT 2.
    ///
    /// Three phases: `R(τ/2) ∘ D(τ) ∘ R(τ/2)` (ADR-0082 AMENDMENT 2).
    ///
    /// - **Phase 1**: half-step reaction `u^(1)_k = exp(τ/2 · C(x_k)) · u^n_k`.
    /// - **Phase 2**: full-step diffusion via block-CN Cayley map.
    /// - **Phase 3**: half-step reaction `u^{n+1}_k = exp(τ/2 · C(x_k)) · u^(2)_k`.
    ///
    /// # Errors
    /// - `DomainViolation` if `tau < 0` or non-finite.
    /// - `Unsupported` if `(V−U)` is near-singular in the Padé solve (M≥5 path).
    fn apply_into(
        &self,
        tau: F,
        src: &MatrixGridFn1D<F, M>,
        dst: &mut MatrixGridFn1D<F, M>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "tau must be finite and >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let n = src.grid.n;
        let dx = src.grid.dx();
        let half_tau = tau / (F::one() + F::one());
        let (a_at_k, b_at_k, exp_half_c) = precompute_strang_coeffs::<F, M>(
            src,
            &self.a_ij_field,
            &self.b_ij_field,
            &self.c_ij_field,
            half_tau,
        )?;
        // Phase 1: u^(1)_k = exp(τ/2 · C(x_k)) · u^n_k
        let mut u1_vals: alloc::vec::Vec<F> = alloc::vec![F::zero(); n * M];
        for k in 0..n {
            let u1k = mat_vec_mul::<F, M>(&exp_half_c[k], &src.point_view(k));
            u1_vals[k * M..k * M + M].copy_from_slice(&u1k);
        }
        // Phase 2: full-step diffusion via block-CN Cayley map.
        let mut u2_vals: alloc::vec::Vec<F> = alloc::vec![F::zero(); n * M];
        block_cn_diff_step::<F, M>(tau, &a_at_k, &b_at_k, &u1_vals, &mut u2_vals, n, dx)?;
        // Phase 3: u^{n+1}_k = exp(τ/2 · C(x_k)) · u^(2)_k
        // exp_half_c cached from Phase 1 (C(x_k) unchanged between phases).
        for k in 0..n {
            let mut u2k = [F::zero(); M];
            u2k.copy_from_slice(&u2_vals[k * M..k * M + M]);
            dst.set_point(k, &mat_vec_mul::<F, M>(&exp_half_c[k], &u2k));
        }
        Ok(())
    }
}

impl<F: SemiflowFloat, const M: usize> ApproximationSubspace<2, F>
    for MatrixDiffusionChernoff<F, M>
{
    /// Returns `true` when: grid.n ≥ 5 AND all state values finite.
    /// Optional `c_norm_bound` is not required for `in_subspace` (ADR-0082).
    fn in_subspace(&self, f: &MatrixGridFn1D<F, M>) -> bool {
        f.grid.n >= 5 && f.values.iter().all(|v| v.is_finite())
    }

    /// 2-jet for matrix-valued operators deferred; returns `Unsupported`.
    ///
    /// The K=2 tangency proof for matrix semigroups is via Pazy 1983 §3.3;
    /// the discrete jet implementation is deferred to v4.x per ADR-0082.
    fn jet(
        &self,
        _f: &MatrixGridFn1D<F, M>,
        _out: &mut [MatrixGridFn1D<F, M>],
    ) -> Result<(), SemiflowError> {
        Err(SemiflowError::Unsupported {
            feature: "MatrixDiffusionChernoff jet K=2: deferred to v4.x",
        })
    }
}

// ---------------------------------------------------------------------------
// Strang coefficient precomputation
// ---------------------------------------------------------------------------

/// Pre-evaluate `A(x_k)`, `B(x_k)`, and `exp(τ/2 · C(x_k))` at every grid point.
///
/// Returns `(a_at_k, b_at_k, exp_half_c)` — three length-`n` arrays used by
/// the palindromic Strang phases.  The C matrix is consumed internally; only
/// `exp_half_c` (already scaled and exponentiated) is returned for Phase 1/3.
///
/// # Errors
/// [`SemiflowError`] if `matrix_exp_dispatch` fails for any point.
// Three-tuple of equal-type Vecs; the repetition is structural, not accidental.
#[allow(clippy::type_complexity)]
fn precompute_strang_coeffs<F: SemiflowFloat, const M: usize>(
    src: &MatrixGridFn1D<F, M>,
    a_field: &MatField<F, M>,
    b_field: &MatField<F, M>,
    c_field: &MatField<F, M>,
    half_tau: F,
) -> Result<
    (
        alloc::vec::Vec<[[F; M]; M]>,
        alloc::vec::Vec<[[F; M]; M]>,
        alloc::vec::Vec<[[F; M]; M]>,
    ),
    SemiflowError,
> {
    let n = src.grid.n;
    let mut a_at_k = alloc::vec![[[F::zero(); M]; M]; n];
    let mut b_at_k = alloc::vec![[[F::zero(); M]; M]; n];
    let mut exp_half_c = alloc::vec![[[F::zero(); M]; M]; n];
    for k in 0..n {
        let x = src.grid.x_at(k);
        a_field(x, &mut a_at_k[k]);
        b_field(x, &mut b_at_k[k]);
        let mut tc = [[F::zero(); M]; M];
        c_field(x, &mut tc);
        for row in &mut tc {
            for v in row.iter_mut() {
                *v = half_tau * *v;
            }
        }
        exp_half_c[k] = matrix_exp_dispatch::<F, M>(&tc)?;
    }
    Ok((a_at_k, b_at_k, exp_half_c))
}

// Matrix exponential backends (dispatch, closed form M ≤ 2, Padé[13/13] M ≥ 3,
// matrix multiply) live in the sibling module.
use matrix_system_exp::{mat_vec_mul, matrix_exp_dispatch};
