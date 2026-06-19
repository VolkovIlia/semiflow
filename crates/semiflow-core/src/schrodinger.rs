//! Schrödinger Chernoff `i ψ_t = (−Δ + V(x)) ψ` via palindromic Strang splitting.
//!
//! **Option A — real-only representation**: `ψ = ψ_re + i ψ_im` is stored as
//! two real `GridFn1D<F>` arrays (a 2N-dimensional real state).
//!
//! The palindromic Strang splitting is:
//!
//! ```text
//! S(τ) = V(τ/2) · K(τ) · V(τ/2)
//! ```
//!
//! where:
//!
//! - `V(α)` is the potential half-step: per-node 2×2 real rotation by angle
//!   `α = V(x_i) · τ/2`:
//!   ```text
//!   [ cos α,  sin α ] [ψ_re]
//!   [-sin α,  cos α ] [ψ_im]
//!   ```
//! - `K(τ)` is the kinetic full-step: **Crank-Nicolson (implicit midpoint / Cayley map)**
//!   applied to the real block system `[r, m]' = [[0, L], [-L, 0]] [r, m]`
//!   (L = a·Δ, Dirichlet BCs). This is exactly unitary in exact arithmetic and
//!   norm-preserving to machine precision in floating point.
//!
//! **Order 2 globally** (palindromic Strang theorem). **Unitary by construction**
//! (each V-rotation is exactly unitary; Cayley map preserves norms exactly).
//!
//! **Mixed-precision implementation**: the entire Strang step is computed in
//! `f64` regardless of `F`, then cast back. This ensures the unitarity error
//! satisfies both `< 1e-12` (f64, G18a) and `< 1e-6` (f32, G18b) after 100
//! steps of T=1.0 evolution.
//!
//! # References
//!
//! - I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025); ADR-0057 (Option A).
//! - Iserles et al., *Acta Numerica* **9** (2000) §3 (Strang splitting for Schrödinger).
//!
//! See `contracts/semiflow-core.math.md` §17 (NORMATIVE) and ADR-0057.
//!
//! # Zero-alloc steady state (R4 invariant)
//!
//! `apply_into` uses ephemeral local `Vec`s (not `ScratchPool`) for the f64
//! working arrays. `ScratchPool` is accepted for API compatibility but not used
//! in steady state (zero `ScratchPool` allocations after warm-up).

use alloc::vec::Vec;
use core::cell::RefCell;

#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion4::Diffusion4thChernoff,
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
    state::{HilbertState, State},
};

#[path = "schrodinger_helpers.rs"]
mod schrodinger_helpers;
use schrodinger_helpers::{
    pentadiag_forward_elim, pentadiag_init_bands, strang_first_v_rotation,
    strang_last_v_rotation_cast,
};

// ---------------------------------------------------------------------------
// SchrodingerState<F>
// ---------------------------------------------------------------------------

/// Real representation of a complex wavefunction `ψ = ψ_re + i ψ_im`.
///
/// Total dimensionality is `2N` (N real components + N imaginary components).
/// Implements [`State<F>`] and [`HilbertState<F>`] for use in the Chernoff
/// iteration via `ChernoffSemigroup`.
///
/// ## Norm convention
///
/// `norm_sup` returns `sup_i √(ψ_re[i]² + ψ_im[i]²)` (amplitude norm).
///
/// ## Hilbert inner product
///
/// `dot` implements `Re⟨ψ, φ⟩ = ψ_re · φ_re + ψ_im · φ_im` (the real part
/// of the standard complex inner product).
#[derive(Clone, Debug)]
pub struct SchrodingerState<F: SemiflowFloat = f64> {
    /// Real part of the wavefunction, sampled at grid nodes.
    pub psi_re: GridFn1D<F>,
    /// Imaginary part of the wavefunction, sampled at grid nodes.
    pub psi_im: GridFn1D<F>,
}

impl<F: SemiflowFloat> SchrodingerState<F> {
    /// Construct from two real `GridFn1D<F>` arrays.
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] if `psi_re` and `psi_im` grids differ.
    pub fn new(psi_re: GridFn1D<F>, psi_im: GridFn1D<F>) -> Result<Self, SemiflowError> {
        if psi_re.values.len() != psi_im.values.len() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "SchrodingerState: psi_re and psi_im must have the same length",
                value: psi_im.values.len() as f64,
            });
        }
        Ok(Self { psi_re, psi_im })
    }

    /// L2-squared norm `‖ψ‖² = ψ_re·ψ_re + ψ_im·ψ_im` (grid-spacing-weighted).
    ///
    /// Returns the discrete squared norm (sum over grid nodes of `ψ_re[i]² + ψ_im[i]²`).
    pub fn norm_l2_sq(&self) -> F {
        let re_sq = self
            .psi_re
            .values
            .iter()
            .fold(F::zero(), |acc, &v| acc + v * v);
        let im_sq = self
            .psi_im
            .values
            .iter()
            .fold(F::zero(), |acc, &v| acc + v * v);
        re_sq + im_sq
    }
}

impl<F: SemiflowFloat> State<F> for SchrodingerState<F> {
    /// Total size: `2 * N` (real + imaginary components).
    #[inline]
    fn len(&self) -> usize {
        2 * self.psi_re.len()
    }

    /// `self ← self + alpha * src` (applied independently to re and im).
    #[inline]
    fn axpy_into(&mut self, alpha: F, src: &Self) {
        self.psi_re.axpy_into(alpha, &src.psi_re);
        self.psi_im.axpy_into(alpha, &src.psi_im);
    }

    /// Copy both components from `src`.
    #[inline]
    fn copy_from(&mut self, src: &Self) {
        self.psi_re.copy_from(&src.psi_re);
        self.psi_im.copy_from(&src.psi_im);
    }

    /// Zero both components.
    #[inline]
    fn zero_into(&mut self) {
        self.psi_re.zero_into();
        self.psi_im.zero_into();
    }

    /// Amplitude supremum norm: `sup_i √(ψ_re[i]² + ψ_im[i]²)`.
    fn norm_sup(&self) -> F {
        let n = self.psi_re.values.len();
        let mut max = F::zero();
        for i in 0..n {
            let r = self.psi_re.values[i];
            let im = self.psi_im.values[i];
            let amp = (r * r + im * im).sqrt();
            if amp > max {
                max = amp;
            }
        }
        max
    }

    /// `self ← k * self` (applied independently to re and im).
    #[inline]
    fn scale_into(&mut self, k: F) {
        self.psi_re.scale_into(k);
        self.psi_im.scale_into(k);
    }
}

impl<F: SemiflowFloat> HilbertState<F> for SchrodingerState<F> {
    /// Real part of complex inner product: `Re⟨ψ, φ⟩ = ψ_re·φ_re + ψ_im·φ_im`.
    fn dot(&self, other: &Self) -> F {
        self.psi_re.dot(&other.psi_re) + self.psi_im.dot(&other.psi_im)
    }
}

// ---------------------------------------------------------------------------
// SchrodingerChernoff<F>
// ---------------------------------------------------------------------------

/// Schrödinger Chernoff `i ψ_t = (−Δ + V(x)) ψ` via palindromic Strang splitting.
///
/// **Order 2 globally** (palindromic Strang theorem).
/// **Unitary by construction** (V-rotations are exactly `SO(2)`;
/// kinetic op is self-adjoint positive semi-definite).
///
/// ## Usage
///
/// ```rust
/// use semiflow_core::{Grid1D, GridFn1D, SchrodingerChernoff, SchrodingerState};
/// use semiflow_core::diffusion4::Diffusion4thChernoff;
/// use semiflow_core::ChernoffSemigroup;
///
/// let grid = Grid1D::new(-5.0_f64, 5.0, 64).unwrap();
/// // a=0.5, a'=0, a''=0, a_norm_bound=0.5.
/// let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
/// let schr = SchrodingerChernoff::new(kinetic, |x| 0.5 * x * x).unwrap();
/// ```
///
/// ## Zero-alloc steady state (R4 invariant)
///
/// `apply_into` uses ephemeral local f64 `Vec`s for the mixed-precision kinetic
/// step. `ScratchPool` is accepted for API compatibility but not used.
/// Zero heap allocations in steady state (R4 invariant).
///
/// 12 f64 working buffers are pre-allocated in `new()` and reused on every
/// `apply_into` call via a `RefCell`. The `ScratchPool<F>` argument is accepted
/// for API compatibility but not used.
///
/// See `contracts/semiflow-core.math.md` §17 (NORMATIVE) and ADR-0057.
#[derive(Clone)]
pub struct SchrodingerChernoff<F: SemiflowFloat = f64> {
    /// Kinetic operator `K = −Δ` (real, acts identically on `ψ_re` and `ψ_im`).
    kinetic: Diffusion4thChernoff<F>,
    /// Potential values at each grid node: `v_at_node[i] = V(x_i)`.
    v_at_node: Vec<F>,
    /// Pre-allocated f64 working buffers for the Crank-Nicolson step (R4 zero-alloc).
    ///
    /// Slots: `[0]=r_d` `[1]=m_d` `[2]=am` `[3]=ar` `[4]=a_sq_m` `[5]=rhs`
    /// `[6]=m_new` `[7]=am_new` `[8]=diag` `[9]=sup1` `[10]=sup2` `[11]=b`
    f64_work: RefCell<[Vec<f64>; 12]>,
}

impl<F: SemiflowFloat> SchrodingerChernoff<F> {
    /// Construct from a kinetic operator and a potential function `V: ℝ → ℝ`.
    ///
    /// # Parameters
    ///
    /// - `kinetic`: a `Diffusion4thChernoff<F>` representing `−Δ`.
    /// - `v`: closure `x ↦ V(x)`, sampled once at each grid node during
    ///   construction.
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] if any `V(x_i)` is not finite.
    pub fn new(kinetic: Diffusion4thChernoff<F>, v: impl Fn(F) -> F) -> Result<Self, SemiflowError> {
        let grid = kinetic.grid;
        let n = grid.n;
        let mut v_at_node = Vec::with_capacity(n);
        for i in 0..n {
            let xi = grid.x_at(i);
            let vi = v(xi);
            if !vi.is_finite() {
                return Err(SemiflowError::DomainViolation {
                    what: "SchrodingerChernoff: V(x_i) must be finite at all grid nodes",
                    value: vi.to_f64().unwrap_or(f64::NAN),
                });
            }
            v_at_node.push(vi);
        }
        let f64_work = RefCell::new(core::array::from_fn(|_| Vec::with_capacity(n)));
        Ok(Self {
            kinetic,
            v_at_node,
            f64_work,
        })
    }

    /// Borrow the kinetic operator.
    #[must_use]
    pub fn kinetic(&self) -> &Diffusion4thChernoff<F> {
        &self.kinetic
    }

    /// Return the pre-sampled potential values `V(x_0), …, V(x_{N-1})`.
    #[must_use]
    pub fn v_at_node(&self) -> &[F] {
        &self.v_at_node
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> for SchrodingerChernoff<F>
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> ChernoffFunction<F> for SchrodingerChernoff<F> {
    type S = SchrodingerState<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &SchrodingerState<F>,
        dst: &mut SchrodingerState<F>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let mut w = self.f64_work.borrow_mut();
        apply_strang_step(self, tau, src, dst, &mut w)
    }

    /// Global order 2 (palindromic Strang, Theorem 7).
    fn order(&self) -> u32 {
        2
    }

    /// Unitary: growth bound = 1, no spectral shift.
    fn growth(&self) -> Growth<F> {
        Growth::contraction()
    }
}

// ---------------------------------------------------------------------------
// Strang-splitting kernel (NORMATIVE — math.md §17.3)
// ---------------------------------------------------------------------------

/// Apply one palindromic Strang step `V(τ/2) · K(τ) · V(τ/2)` to `src → dst`.
///
/// The entire step is performed in `f64` regardless of `F` (mixed-precision
/// strategy). This guarantees that both f64 (G18a: gate 1e-12) and f32
/// (G18b: gate 1e-6) pass their unitarity gates, because the per-step
/// floating-point deviation from exact unitarity is near f64 machine epsilon
/// (~1e-16), well below either gate threshold after 100 steps.
///
/// Algorithm — palindromic Strang: `S(τ) = V(τ/2) · K(τ) · V(τ/2)`
///
/// V(α): per-node 2×2 rotation `[[cos α, sin α], [−sin α, cos α]]`.
///
/// K(τ): Crank-Nicolson kinetic step (exactly unitary in exact arithmetic).
/// Block system `[[I,−A],[A,I]][r_new,m_new]^T = [r+Am, m−Ar]^T` (A = τ/2·L)
/// solved as `(I+A²)·m_new = (I−A²)·m − 2A·r`, then `r_new = r + Am + Am_new`.
/// The pentadiagonal `(I+A²)` is band-5 SPD, factored by banded LU.
///
/// Zero heap allocations in steady state (R4 invariant).
#[allow(clippy::unnecessary_wraps)]
fn apply_strang_step<F: SemiflowFloat>(
    sc: &SchrodingerChernoff<F>,
    tau: F,
    src: &SchrodingerState<F>,
    dst: &mut SchrodingerState<F>,
    w: &mut [Vec<f64>; 12],
) -> Result<(), SemiflowError> {
    let n = src.psi_re.values.len();
    let half_tau_d = 0.5 * tau.to_f64().unwrap_or(0.0);

    // Slots 0,1: r_d, m_d (outer V-rotation scratch).
    w[0].clear();
    w[0].resize(n, 0.0);
    w[1].clear();
    w[1].resize(n, 0.0);

    // --- Step 1: first half-step V-rotation (src → w[0]/w[1]) ---
    strang_first_v_rotation(
        n,
        &src.psi_re.values,
        &src.psi_im.values,
        &sc.v_at_node,
        half_tau_d,
        w,
    );

    // --- Step 2: Crank-Nicolson kinetic step (f64) ---
    //
    // `a_off` must be negative so that `a_diag = -2·a_off > 0` (positive-definite),
    // ensuring we evolve as e^{−iτK} (not e^{+iτK}).
    let a0_d = (sc.kinetic.a)(sc.kinetic.grid.x_at(0))
        .to_f64()
        .unwrap_or(0.0);
    let dx_d = sc.kinetic.grid.dx().to_f64().unwrap_or(1.0);
    let a_off = -(half_tau_d * a0_d / (dx_d * dx_d)); // negated: A = +τK/2
    cn_kinetic_step_f64(n, a_off, w);

    // --- Step 3: second half-step V-rotation in-place (f64) → cast to F ---
    strang_last_v_rotation_cast(
        n,
        &sc.v_at_node,
        half_tau_d,
        w,
        &mut dst.psi_re.values,
        &mut dst.psi_im.values,
    );

    Ok(())
}

/// Crank-Nicolson kinetic step `[r, m] ← Cayley(A) · [r, m]` in-place (f64).
///
/// `a_off` = off-diagonal coefficient of `A = (τ/2)·L` (positive; Dirichlet BCs).
/// `a_diag = -2·a_off` (main diagonal of A, negative).
///
/// `w` = pre-allocated working buffers (slots 0..12 from `SchrodingerChernoff::f64_work`).
/// Slot layout: `[0]=r_d` `[1]=m_d` `[2]=am` `[3]=ar` `[4]=a_sq_m` `[5]=rhs`
/// `[6]=m_new` `[7]=am_new` `[8]=diag` `[9]=sup1` `[10]=sup2` `[11]=b`
#[allow(clippy::needless_range_loop)]
fn cn_kinetic_step_f64(n: usize, a_off: f64, w: &mut [Vec<f64>; 12]) {
    let a_diag = -2.0 * a_off;

    // Slots 2..8: am, ar, a_sq_m, rhs, m_new, am_new.
    for v in &mut w[2..8] {
        v.clear();
        v.resize(n, 0.0);
    }

    // am = A·m_d (input=w[1], output=w[2]).
    {
        let (left, right) = w.split_at_mut(2);
        tridiag_matvec_f64(n, a_diag, a_off, &left[1], &mut right[0]);
    }
    // ar = A·r_d (input=w[0], output=w[3]).
    {
        let (left, right) = w.split_at_mut(3);
        tridiag_matvec_f64(n, a_diag, a_off, &left[0], &mut right[0]);
    }
    // a_sq_m = A·am (input=w[2], output=w[4]).
    {
        let (left, right) = w.split_at_mut(4);
        tridiag_matvec_f64(n, a_diag, a_off, &left[2], &mut right[0]);
    }
    // rhs = m_d − A²·m_d − 2·A·r_d.
    for i in 0..n {
        w[5][i] = w[1][i] - w[4][i] - 2.0 * w[3][i];
    }

    let pd_int = 1.0 + a_diag * a_diag + 2.0 * a_off * a_off;
    let pd_bnd = 1.0 + a_diag * a_diag + a_off * a_off;
    let pd_d1 = 2.0 * a_diag * a_off;
    let pd_d2 = a_off * a_off;

    // m_new = (I + A²)⁻¹ · rhs (reads w[5], writes w[6]).
    pentadiag_solve_f64(n, pd_int, pd_bnd, pd_d1, pd_d2, w);

    // am_new = A·m_new (input=w[6], output=w[7]).
    {
        let (left, right) = w.split_at_mut(7);
        tridiag_matvec_f64(n, a_diag, a_off, &left[6], &mut right[0]);
    }
    // r_d += am + am_new.
    for i in 0..n {
        w[0][i] += w[2][i] + w[7][i];
    }
    // m_d ← m_new: swap slots 1 and 6 to avoid allocation.
    w.swap(1, 6);
}

/// Tridiagonal matrix-vector product (f64, Dirichlet BCs).
///
/// A[i,i] = `diag`, A[i,i±1] = `off`; boundary off-diags are 0.
#[inline]
fn tridiag_matvec_f64(n: usize, diag: f64, off: f64, v: &[f64], out: &mut Vec<f64>) {
    debug_assert_eq!(v.len(), n);
    out.resize(n, 0.0);
    for i in 0..n {
        let left = if i == 0 { 0.0 } else { off * v[i - 1] };
        let right = if i == n - 1 { 0.0 } else { off * v[i + 1] };
        out[i] = diag * v[i] + left + right;
    }
}

/// Solve the pentadiagonal system `(I + A²)·x = rhs` (f64, band-5 LU).
///
/// Uses pre-allocated buffers from `w`:
/// - `w[5]` = rhs (read-only input)
/// - `w[6]` = output (`m_new`, written by this function)
/// - `w[8..=11]` = [diag, sup1, sup2, b] — temporary LU scratch
fn pentadiag_solve_f64(
    n: usize,
    d0_interior: f64,
    d0_boundary: f64,
    d1: f64,
    d2: f64,
    w: &mut [Vec<f64>; 12],
) {
    // Slots 6=out, 8=diag, 9=sup1, 10=sup2, 11=b.
    w[6].clear();
    w[6].resize(n, 0.0);
    for v in &mut w[8..12] {
        v.clear();
        v.resize(n, 0.0);
    }
    if n == 0 {
        return;
    }

    pentadiag_init_bands(n, d0_interior, d0_boundary, d1, d2, w);
    pentadiag_forward_elim(n, w);

    // Back-substitution into slot 6 (m_new).
    w[6][n - 1] = w[11][n - 1] / w[8][n - 1];
    if n >= 2 {
        w[6][n - 2] = (w[11][n - 2] - w[9][n - 2] * w[6][n - 1]) / w[8][n - 2];
    }
    for i in (0..n.saturating_sub(2)).rev() {
        let v = w[11][i] - w[9][i] * w[6][i + 1] - w[10][i] * w[6][i + 2];
        w[6][i] = v / w[8][i];
    }
}

#[cfg(test)]
mod tests {
    include!("schrodinger_tests.rs");
}
