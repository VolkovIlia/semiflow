//! Complex matrix-valued Chernoff kernel (ADR-0128, §33.8 Para 3).
//!
//! `MatrixDiffusionChernoffComplex<C, M>`: palindromic Strang splitting for
//! coupled M-component diffusion with **complex** coefficient matrices A, B, C.
//!
//! Three-phase structure is identical to `MatrixDiffusionChernoff<F, M>` (real,
//! ADR-0082 AMENDMENT 2); the only change is the scalar type:
//! `F: SemiflowFloat` → `C: SemiflowComplex`. The Padé[13/13] complex path
//! (`matrix_pade_complex.rs`) is used for M ≥ 5.
//!
//! ## State type
//!
//! `MatrixGridFnComplex1D<C, M>` — multi-component 1D grid state `u: Grid₁D → ℂᴹ`.
//! Layout: row-major flat `Vec<C>` of length `N·M`; component `i` at point `k`
//! at index `k·M + i`.
//!
//! ## Phase 2 (complex block-CN Cayley map)
//!
//! For complex `a_ij`, `b_ij`, the block-tridiagonal system is solved via block-Thomas
//! with complex arithmetic. The pivot threshold uses complex modulus.
//!
//! ADR-0128; §33.7 AMENDMENT 2 (structure); §33.8 Para 3 (extension rationale).

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use num_traits::{Float, One, ToPrimitive, Zero};

use crate::{
    approximation::ApproximationSubspace,
    chernoff::{ChernoffFunction, Growth},
    complex::SemiflowComplex,
    error::SemiflowError,
    grid::Grid1D,
    matrix_pade_complex::{cmat_vec_mul, mat_exp_pade13_complex, real_from_f64_cplx},
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// MatrixGridFnComplex1D<C, M>
// ---------------------------------------------------------------------------

/// Multi-component complex grid state: `u : Grid₁D → ℂᴹ`.
///
/// Layout: row-major `Vec<C>`, length `N·M`; component `i` at grid point `k`
/// at index `k·M + i`.
#[derive(Clone, Debug)]
pub struct MatrixGridFnComplex1D<C: SemiflowComplex = num_complex::Complex<f64>, const M: usize = 2>
{
    /// Reference grid geometry.
    pub grid: Grid1D<C::Real>,
    /// Flat complex values.
    pub values: Vec<C>,
}

impl<C: SemiflowComplex, const M: usize> MatrixGridFnComplex1D<C, M> {
    /// Zero-valued state on `grid`.
    pub fn new(grid: Grid1D<C::Real>) -> Self {
        let n = grid.n;
        Self {
            grid,
            values: vec![C::zero(); n * M],
        }
    }

    /// Construct from pointwise closure `f(x) -> [C; M]`.
    pub fn from_fn(grid: Grid1D<C::Real>, mut f: impl FnMut(C::Real) -> [C; M]) -> Self {
        let n = grid.n;
        let mut values = vec![C::zero(); n * M];
        for k in 0..n {
            let x = grid.x_at(k);
            let v = f(x);
            for i in 0..M {
                values[k * M + i] = v[i];
            }
        }
        Self { grid, values }
    }

    /// M-component vector at grid point `k`.
    #[inline]
    pub fn point_view(&self, k: usize) -> [C; M] {
        let base = k * M;
        let mut out = [C::zero(); M];
        out.copy_from_slice(&self.values[base..base + M]);
        out
    }

    /// Set M-component vector at grid point `k`.
    #[inline]
    pub fn set_point(&mut self, k: usize, val: &[C; M]) {
        let base = k * M;
        self.values[base..base + M].copy_from_slice(val);
    }
}

impl<C: SemiflowComplex, const M: usize> State<C::Real> for MatrixGridFnComplex1D<C, M> {
    fn len(&self) -> usize {
        self.grid.n
    }

    fn axpy_into(&mut self, alpha: C::Real, src: &Self) {
        let a = C::from_real(alpha);
        for (d, &s) in self.values.iter_mut().zip(src.values.iter()) {
            *d += a * s;
        }
    }

    fn copy_from(&mut self, src: &Self) {
        self.values.clone_from(&src.values);
    }

    fn zero_into(&mut self) {
        for v in &mut self.values {
            *v = C::zero();
        }
    }

    fn norm_sup(&self) -> C::Real {
        self.values.iter().fold(C::Real::zero(), |acc, &v| {
            let av = v.abs();
            if av > acc {
                av
            } else {
                acc
            }
        })
    }

    fn scale_into(&mut self, k: C::Real) {
        let ck = C::from_real(k);
        for v in &mut self.values {
            *v = ck * *v;
        }
    }
}

// ---------------------------------------------------------------------------
// MatrixDiffusionChernoffComplex<C, M>
// ---------------------------------------------------------------------------

/// Chernoff function for coupled M-component 1D diffusion with **complex**
/// coefficient matrices (ADR-0128).
///
/// `(Lu)ᵢ = Σⱼ aᵢⱼ(x)∂ₓ²uⱼ + Σⱼ bᵢⱼ(x)∂ₓuⱼ + Σⱼ cᵢⱼ(x)uⱼ`, `u ∈ ℂᴹ`.
///
/// ## Matrix exponential (Phase 1/3)
///
/// - M ≤ 4: two-component complex Cayley-Hamilton (via Taylor degree-12
///   scaling-and-squaring — reused from the real helper `mat_exp_taylor_complex`).
/// - M ≥ 5: Padé[13/13] scaling-and-squaring (ADR-0128).
///
/// ## Phase 2 (complex block-CN Cayley map)
///
/// For complex `a_ij`, `b_ij` the block-tridiagonal system uses the same
/// block-Thomas structure as `matrix_strang::block_cn_diff_step` but with
/// complex arithmetic throughout.
#[allow(clippy::type_complexity)]
pub struct MatrixDiffusionChernoffComplex<
    C: SemiflowComplex = num_complex::Complex<f64>,
    const M: usize = 2,
> {
    a_ij_field: Box<dyn Fn(C::Real, &mut [[C; M]; M]) + Send + Sync>,
    b_ij_field: Box<dyn Fn(C::Real, &mut [[C; M]; M]) + Send + Sync>,
    c_ij_field: Box<dyn Fn(C::Real, &mut [[C; M]; M]) + Send + Sync>,
    /// Reference grid geometry.
    pub grid: Grid1D<C::Real>,
    /// Optional ‖C(x)‖_∞ bound for `growth()`.
    pub c_norm_bound: Option<C::Real>,
    _phantom: PhantomData<C>,
}

impl<C: SemiflowComplex, const M: usize> MatrixDiffusionChernoffComplex<C, M> {
    /// Construct from complex coefficient closures.
    ///
    /// # Errors
    /// `DomainViolation` if grid.n < 5 or centre evaluations non-finite.
    pub fn new(
        a_ij_field: impl Fn(C::Real, &mut [[C; M]; M]) + Send + Sync + 'static,
        b_ij_field: impl Fn(C::Real, &mut [[C; M]; M]) + Send + Sync + 'static,
        c_ij_field: impl Fn(C::Real, &mut [[C; M]; M]) + Send + Sync + 'static,
        grid: Grid1D<C::Real>,
    ) -> Result<Self, SemiflowError> {
        if grid.n < 5 {
            return Err(SemiflowError::DomainViolation {
                what: "MatrixDiffusionChernoffComplex requires grid.n >= 5",
                #[allow(clippy::cast_precision_loss)]
                value: grid.n as f64,
            });
        }
        let x_c = grid.x_at(grid.n / 2);
        let mut a_tmp = [[C::zero(); M]; M];
        a_ij_field(x_c, &mut a_tmp);
        if a_tmp.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
            return Err(SemiflowError::DomainViolation {
                what: "a_ij_field: non-finite entry at grid centre",
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

    /// Set ‖C(x)‖_∞ bound for `growth()`.
    #[must_use]
    pub fn with_c_norm_bound(mut self, bound: C::Real) -> Self {
        self.c_norm_bound = Some(bound);
        self
    }
}

impl<C: SemiflowComplex, const M: usize> ChernoffFunction<C::Real>
    for MatrixDiffusionChernoffComplex<C, M>
{
    type S = MatrixGridFnComplex1D<C, M>;

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<C::Real> {
        let omega = self.c_norm_bound.unwrap_or_else(C::Real::infinity);
        Growth::new(C::Real::one(), omega)
    }

    /// Palindromic Strang step: `R(τ/2) ∘ D(τ) ∘ R(τ/2)` (§33.7 AMENDMENT 2).
    fn apply_into(
        &self,
        tau: C::Real,
        src: &MatrixGridFnComplex1D<C, M>,
        dst: &mut MatrixGridFnComplex1D<C, M>,
        _scratch: &mut ScratchPool<C::Real>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < C::Real::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "tau must be finite and >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let n = src.grid.n;
        let (a_at_k, b_at_k, exp_half_c) = self.eval_coeff(n, &src.grid, tau)?;
        let u1 = phase_reaction::<C, M>(&exp_half_c, src, n);
        let mut u2 = vec![C::zero(); n * M];
        complex_block_cn_diff_step::<C, M>(tau, &a_at_k, &b_at_k, &u1, &mut u2, n, src.grid.dx())?;
        for (k, exp_k) in exp_half_c.iter().enumerate() {
            let u2k = read_point::<C, M>(&u2, k);
            dst.set_point(k, &cmat_vec_mul::<C, M>(exp_k, &u2k));
        }
        Ok(())
    }
}

impl<C: SemiflowComplex, const M: usize> MatrixDiffusionChernoffComplex<C, M> {
    /// Pre-evaluate `a`, `b`, and `exp(τ/2 · C)` at each grid point.
    #[allow(clippy::type_complexity)]
    fn eval_coeff(
        &self,
        n: usize,
        grid: &Grid1D<C::Real>,
        tau: C::Real,
    ) -> Result<(Vec<[[C; M]; M]>, Vec<[[C; M]; M]>, Vec<[[C; M]; M]>), SemiflowError> {
        let two = C::Real::one() + C::Real::one();
        let ht = C::from_real(tau / two);
        let mut a_at_k: Vec<[[C; M]; M]> = vec![[[C::zero(); M]; M]; n];
        let mut b_at_k: Vec<[[C; M]; M]> = vec![[[C::zero(); M]; M]; n];
        let mut exp_c: Vec<[[C; M]; M]> = vec![[[C::zero(); M]; M]; n];
        for k in 0..n {
            let x = grid.x_at(k);
            (self.a_ij_field)(x, &mut a_at_k[k]);
            (self.b_ij_field)(x, &mut b_at_k[k]);
            let mut ck = [[C::zero(); M]; M];
            (self.c_ij_field)(x, &mut ck);
            let mut tc = [[C::zero(); M]; M];
            for i in 0..M {
                for j in 0..M {
                    tc[i][j] = ht * ck[i][j];
                }
            }
            exp_c[k] = cmatrix_exp_dispatch::<C, M>(&tc)?;
        }
        Ok((a_at_k, b_at_k, exp_c))
    }
}

impl<C: SemiflowComplex, const M: usize> ApproximationSubspace<2, C::Real>
    for MatrixDiffusionChernoffComplex<C, M>
{
    fn in_subspace(&self, f: &MatrixGridFnComplex1D<C, M>) -> bool {
        f.grid.n >= 5 && f.values.iter().all(|v| v.is_finite())
    }

    fn jet(
        &self,
        _f: &MatrixGridFnComplex1D<C, M>,
        _out: &mut [MatrixGridFnComplex1D<C, M>],
    ) -> Result<(), SemiflowError> {
        Err(SemiflowError::Unsupported {
            feature: "MatrixDiffusionChernoffComplex jet K=2: deferred",
        })
    }
}

// ---------------------------------------------------------------------------
// Phase helper
// ---------------------------------------------------------------------------

/// Apply exp(τ/2 C) to each grid point: Phase 1 reaction half-step.
fn phase_reaction<C: SemiflowComplex, const M: usize>(
    exp_half_c: &[[[C; M]; M]],
    src: &MatrixGridFnComplex1D<C, M>,
    n: usize,
) -> Vec<C> {
    let mut u1 = vec![C::zero(); n * M];
    for k in 0..n {
        let uk = src.point_view(k);
        let u1k = cmat_vec_mul::<C, M>(&exp_half_c[k], &uk);
        for i in 0..M {
            u1[k * M + i] = u1k[i];
        }
    }
    u1
}

// ---------------------------------------------------------------------------
// Complex matrix-exp dispatch (Cayley-Hamilton for M≤4; Padé for M≥5)
// ---------------------------------------------------------------------------

fn cmatrix_exp_dispatch<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
) -> Result<[[C; M]; M], SemiflowError> {
    match M {
        0 => Ok([[C::zero(); M]; M]),
        1..=4 => Ok(cmat_exp_taylor::<C, M>(a, M)),
        _ => mat_exp_pade13_complex::<C, M>(a),
    }
}

/// Scaling-and-squaring Taylor degree-12 for complex matrices (M ≤ 4).
fn cmat_exp_taylor<C: SemiflowComplex, const M: usize>(a: &[[C; M]; M], dim: usize) -> [[C; M]; M] {
    let (k, b) = taylor_scale::<C, M>(a, dim);
    let result = taylor_series::<C, M>(&b, dim);
    mat_square_k::<C, M>(result, k, dim)
}

/// Compute scaling count k and scaled matrix B = A / 2^k.
fn taylor_scale<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
    dim: usize,
) -> (u32, [[C; M]; M]) {
    let mut norm = C::Real::zero();
    #[allow(clippy::needless_range_loop)]
    for i in 0..dim {
        let rs = (0..dim).fold(C::Real::zero(), |acc, j| {
            let av = a[i][j].abs();
            if av > acc {
                av
            } else {
                acc
            }
        });
        if rs > norm {
            norm = rs;
        }
    }
    let k = {
        let nf = norm.to_f64().unwrap_or(0.0);
        if nf <= 1.0 {
            0u32
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                (nf.log2().ceil() as u32).min(30)
            }
        }
    };
    let inv_s = C::from_real(real_from_f64_cplx::<C::Real>(
        1.0 / <f64 as From<u32>>::from(1u32 << k),
    ));
    let mut b = [[C::zero(); M]; M];
    for i in 0..dim {
        for j in 0..dim {
            b[i][j] = a[i][j] * inv_s;
        }
    }
    (k, b)
}

/// Degree-12 Taylor sum starting from identity.
fn taylor_series<C: SemiflowComplex, const M: usize>(b: &[[C; M]; M], dim: usize) -> [[C; M]; M] {
    let mut result = [[C::zero(); M]; M];
    #[allow(clippy::needless_range_loop)]
    for i in 0..dim {
        result[i][i] = C::one();
    }
    let mut term = result;
    for d in 1u32..=12 {
        let mut t2 = [[C::zero(); M]; M];
        for i in 0..dim {
            for mid in 0..dim {
                for j in 0..dim {
                    t2[i][j] += term[i][mid] * b[mid][j];
                }
            }
        }
        let inv_d = C::from_real(real_from_f64_cplx::<C::Real>(
            1.0 / <f64 as From<u32>>::from(d),
        ));
        for i in 0..dim {
            for j in 0..dim {
                t2[i][j] *= inv_d;
                result[i][j] += t2[i][j];
            }
        }
        term = t2;
    }
    result
}

/// Square a matrix k times: result^(2^k).
fn mat_square_k<C: SemiflowComplex, const M: usize>(
    mut r: [[C; M]; M],
    k: u32,
    dim: usize,
) -> [[C; M]; M] {
    for _ in 0..k {
        let mut sq = [[C::zero(); M]; M];
        #[allow(clippy::needless_range_loop)]
        for i in 0..dim {
            for mid in 0..dim {
                for j in 0..dim {
                    sq[i][j] += r[i][mid] * r[mid][j];
                }
            }
        }
        r = sq;
    }
    r
}

// ---------------------------------------------------------------------------
// Complex block-CN diffusion step (Phase 2): included via include! to stay
// within the 500-line suckless limit.
// ---------------------------------------------------------------------------
include!("matrix_system_complex_cn.rs");
