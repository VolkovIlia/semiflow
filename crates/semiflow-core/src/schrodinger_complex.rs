//! v4.0 B6 — Schrödinger Option B native-complex implementation (ADR-0079, math.md §30.3).
//!
//! Solves `i ψ_t = H ψ`, `H = −½ ∂²_x + V(x)`, with `ψ : ℝ → ℂ` complex-valued.
//!
//! ## Algorithm (palindromic Strang, math.md §30.3)
//!
//! `U(τ) ≈ exp(−i τ V/2) ∘ Cayley(τ) ∘ exp(−i τ V/2)   +  O(τ³)`
//!
//! where `Cayley(τ)` is the unitary Crank-Nicolson map for the free kinetic
//! operator `(−½ ∂²_x)`:
//!
//! ```text
//! (I − (iτ/4) Δ_h) ψ_new = (I + (iτ/4) Δ_h) ψ_old
//! ```
//!
//! solved via complex Thomas algorithm (O(N) per step).
//!
//! ## State type
//!
//! `GridFn1D<F: SemiflowFloat>` has a hard `SemiflowFloat` bound. Option B requires
//! complex state; we introduce [`GridFnComplex1D<C>`] — a thin wrapper holding
//! `Vec<C>` plus a `Grid1D<C::Real>`. It implements [`State<C::Real>`] so it
//! plugs into the standard Chernoff iteration machinery.
//!
//! ## Citations
//!
//! - Pazy 1983 §6.4 — operator-splitting decomposition of unitary semigroups.
//! - Cheng 2008 §3.2 — Crank-Nicolson Cayley map as unitary rational approximation.
//! - Engel-Nagel 2000 §IV.6 — unitary semigroups on Hilbert space.
//!
//! ## Residual v2.2 Option A
//!
//! The real-pair `SchrodingerChernoff<F>` (ADR-0057) is PRESERVED verbatim.
//! This module is ADDITIVE per ADR-0079.

extern crate alloc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use num_traits::{Float, One, Zero};

use crate::{
    approximation::ApproximationSubspace,
    chernoff::{ChernoffFunction, Growth},
    complex::SemiflowComplex,
    error::SemiflowError,
    grid::Grid1D,
    scratch::ScratchPool,
};

pub use crate::schrodinger_complex_state::GridFnComplex1D;

// ---------------------------------------------------------------------------
// SchrödingerChernoffComplex<C>
// ---------------------------------------------------------------------------

/// Schrödinger Chernoff kernel with native complex state (ADR-0079 Option B).
///
/// State: [`GridFnComplex1D<C>`].  Algorithm: palindromic Strang per math §30.3.
/// Order 2.  Growth: unitary (`multiplier=1, omega=0`).
///
/// ## Construction
///
/// ```rust,no_run
/// # use semiflow_core::{Grid1D, schrodinger_complex::{SchrödingerChernoffComplex, GridFnComplex1D}};
/// # use num_complex::Complex;
/// let grid  = Grid1D::<f64>::new(-10.0, 10.0, 512).unwrap();
/// let v_fn  = |x: f64| 0.5 * x * x;                      // harmonic oscillator
/// let kernel = SchrödingerChernoffComplex::<Complex<f64>>::new(grid, v_fn).unwrap();
/// ```
pub struct SchrödingerChernoffComplex<C: SemiflowComplex> {
    grid: Grid1D<C::Real>,
    /// Sampled potential values `V(x_0), …, V(x_{N−1})`.
    v_at_node: Vec<C::Real>,
    _c: PhantomData<C>,
}

impl<C: SemiflowComplex> SchrödingerChernoffComplex<C> {
    /// Construct from a grid and potential closure `V : ℝ → ℝ`.
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] if any `V(x_i)` is non-finite
    /// or if `grid.n < 3` (minimum for a meaningful tridiagonal solve).
    pub fn new(
        grid: Grid1D<C::Real>,
        v: impl Fn(C::Real) -> C::Real,
    ) -> Result<Self, SemiflowError> {
        // Grid1D enforces n >= 4 (Catmull-Rom stencil); no additional check needed.
        let mut v_at_node = Vec::with_capacity(grid.n);
        for i in 0..grid.n {
            let vi = v(grid.x_at(i));
            if !vi.is_finite() {
                return Err(SemiflowError::DomainViolation {
                    what: "SchrödingerChernoffComplex: V(x_i) not finite",
                    value: 0.0,
                });
            }
            v_at_node.push(vi);
        }
        Ok(Self {
            grid,
            v_at_node,
            _c: PhantomData,
        })
    }

    /// Return a reference to the pre-sampled potential values.
    #[must_use]
    pub fn v_at_node(&self) -> &[C::Real] {
        &self.v_at_node
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<C::Real> impl
// ---------------------------------------------------------------------------

impl<C: SemiflowComplex> ChernoffFunction<C::Real> for SchrödingerChernoffComplex<C> {
    type S = GridFnComplex1D<C>;

    fn apply_into(
        &self,
        tau: C::Real,
        src: &GridFnComplex1D<C>,
        dst: &mut GridFnComplex1D<C>,
        _scratch: &mut ScratchPool<C::Real>,
    ) -> Result<(), SemiflowError> {
        if !Float::is_finite(tau) || tau < <C::Real as Zero>::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "SchrödingerChernoffComplex::apply_into: tau must be finite and non-negative",
                value: 0.0,
            });
        }
        strang_step(self, tau, src, dst)
    }

    /// Global order 2 (palindromic Strang, Pazy 1983 §6.4).
    fn order(&self) -> u32 {
        2
    }

    /// Unitary: `‖U(τ) ψ‖₂ = ‖ψ‖₂`; multiplier=1, omega=0.
    fn growth(&self) -> Growth<C::Real> {
        Growth::new(<C::Real as One>::one(), <C::Real as Zero>::zero())
    }
}

// ---------------------------------------------------------------------------
// ApproximationSubspace<2, C::Real> (ADR-0073 opt-in, K=2 witness)
// ---------------------------------------------------------------------------

impl<C: SemiflowComplex> ApproximationSubspace<2, C::Real> for SchrödingerChernoffComplex<C> {
    /// Returns `true` if all values are finite and grid has ≥ 5 nodes.
    fn in_subspace(&self, f: &GridFnComplex1D<C>) -> bool {
        f.grid.n >= 5 && f.values.iter().all(|z| z.is_finite())
    }

    /// Jet stub — complex jet not yet implemented (math.md §30.5 scope note).
    fn jet(
        &self,
        _f: &GridFnComplex1D<C>,
        _out: &mut [GridFnComplex1D<C>],
    ) -> Result<(), SemiflowError> {
        Err(SemiflowError::Unsupported {
            feature: "SchrödingerChernoffComplex::jet (K-jet not implemented in v4.0; deferred)",
        })
    }
}

// ---------------------------------------------------------------------------
// Palindromic Strang step (NORMATIVE — math.md §30.3)
// ---------------------------------------------------------------------------

/// One palindromic Strang step: `V(τ/2) ∘ Cayley(τ) ∘ V(τ/2)`.
///
/// Each sub-step is exact-unitary in complex arithmetic:
/// - `V(δ)`: pointwise multiplication by `exp(−i δ V(x_k))` (unit-modulus).
/// - `Cayley(τ)`: complex Crank-Nicolson via Thomas algorithm (O(N)).
fn strang_step<C: SemiflowComplex>(
    sc: &SchrödingerChernoffComplex<C>,
    tau: C::Real,
    src: &GridFnComplex1D<C>,
    dst: &mut GridFnComplex1D<C>,
) -> Result<(), SemiflowError> {
    let n = src.values.len();
    let half_tau = tau / (<C::Real as One>::one() + <C::Real as One>::one()); // τ/2

    // tmp buffer: potential half-step applied to src
    let mut tmp: Vec<C> = (0..n).map(|_| C::zero()).collect();
    potential_half_step(sc, half_tau, &src.values, &mut tmp);

    // Cayley kinetic full step in-place on tmp → dst
    let mut dst_vals: Vec<C> = (0..n).map(|_| C::zero()).collect();
    cayley_step(sc, tau, &tmp, &mut dst_vals)?;

    // Second potential half-step: dst_vals → dst.values
    potential_half_step(sc, half_tau, &dst_vals, &mut dst.values);
    Ok(())
}

/// Potential half-step: `out[k] = exp(−i · delta · V(x_k)) · src[k]`.
fn potential_half_step<C: SemiflowComplex>(
    sc: &SchrödingerChernoffComplex<C>,
    delta: C::Real,
    src: &[C],
    out: &mut [C],
) {
    let neg_delta = <C::Real as Zero>::zero() - delta;
    for (k, (o, &z)) in out.iter_mut().zip(src.iter()).enumerate() {
        // phase = −delta · V[k]
        let phase = neg_delta * sc.v_at_node[k];
        // exp(i · phase) in polar form: r=1, theta=phase
        let rotation = C::from_polar(<C::Real as One>::one(), phase);
        *o = rotation * z;
    }
}

/// Complex Crank-Nicolson kinetic step via Thomas algorithm (O(N)).
///
/// Solves: `(I − (iτ/4) Δ_h) ψ_new = (I + (iτ/4) Δ_h) ψ_old`
///
/// where `Δ_h` is the 3-point Laplacian with reflecting BCs:
/// `Δ_h ψ[k] = (ψ[k−1] − 2ψ[k] + ψ[k+1]) / dx²`.
///
/// The coefficient `α = iτ / (4 dx²)` parameterises both the RHS
/// tridiagonal matvec and the LHS tridiagonal system.
fn cayley_step<C: SemiflowComplex>(
    sc: &SchrödingerChernoffComplex<C>,
    tau: C::Real,
    src: &[C],
    dst: &mut [C],
) -> Result<(), SemiflowError> {
    cayley_step_dx(sc.grid.dx(), tau, src, dst)
}

/// Cayley kinetic step parameterised by grid spacing `dx` (V=0, free Schrödinger).
///
/// Extracted from `cayley_step` so `QuantumSchrödingerChernoff` can reuse it
/// per-edge without holding a `SchrödingerChernoffComplex` instance.
pub(crate) fn cayley_step_dx<C: SemiflowComplex>(
    dx: C::Real,
    tau: C::Real,
    src: &[C],
    dst: &mut [C],
) -> Result<(), SemiflowError> {
    let n = src.len();
    let dx2 = dx * dx;

    // α = i · τ / (4 · dx²).
    // 4·dx² computed from first principles: one + one + one + one.
    let f_one = <C::Real as One>::one();
    let four = f_one + f_one + f_one + f_one;
    let two = f_one + f_one;
    let coeff = tau / (four * dx2);
    let alpha = C::i() * C::from_real(coeff); // iτ/(4dx²)

    // Build RHS: b[k] = (I + α Δ_h) src[k]
    //   b[0]     = (1 − 2α) src[0] + α src[1]
    //   b[k]     = α src[k−1] + (1 − 2α) src[k] + α src[k+1]   (interior)
    //   b[n−1]   = α src[n−2] + (1 − 2α) src[n−1]
    let diag_rhs = C::one() - C::from_real(two) * alpha; // 1 − 2α
    let mut rhs: Vec<C> = (0..n).map(|_| C::zero()).collect();
    rhs[0] = diag_rhs * src[0] + alpha * src[1];
    for k in 1..n - 1 {
        rhs[k] = alpha * src[k - 1] + diag_rhs * src[k] + alpha * src[k + 1];
    }
    rhs[n - 1] = alpha * src[n - 2] + diag_rhs * src[n - 1];

    // Solve tridiagonal system (I − α Δ_h) x = rhs via Thomas algorithm.
    // LHS: main diagonal = 1 + 2α, off-diagonal = −α.
    let diag_lhs = C::one() + C::from_real(two) * alpha; // 1 + 2α
    let off_lhs = C::zero() - alpha; // −α
    thomas_solve(n, diag_lhs, off_lhs, &rhs, dst)
}

/// Thomas algorithm for tridiagonal system with uniform off-diagonal `off`.
///
/// System: `off·x[k−1] + diag·x[k] + off·x[k+1] = rhs[k]`.
/// (Both sub- and super-diagonal are `off`.)
/// Reflecting BCs: boundary rows have only the main diagonal and one off-diag.
pub(crate) fn thomas_solve<C: SemiflowComplex>(
    n: usize,
    diag: C,
    off: C,
    rhs: &[C],
    out: &mut [C],
) -> Result<(), SemiflowError> {
    let mut c_prime: Vec<C> = (0..n).map(|_| C::zero()).collect(); // modified super-diagonal
    let mut d_prime: Vec<C> = (0..n).map(|_| C::zero()).collect(); // modified RHS

    // Forward sweep
    c_prime[0] = off / diag;
    d_prime[0] = rhs[0] / diag;
    for k in 1..n {
        let denom = diag - off * c_prime[k - 1];
        if !denom.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "SchrödingerChernoffComplex::thomas_solve: zero pivot",
                value: 0.0,
            });
        }
        c_prime[k] = if k < n - 1 { off / denom } else { C::zero() };
        d_prime[k] = (rhs[k] - off * d_prime[k - 1]) / denom;
    }

    // Back substitution
    out[n - 1] = d_prime[n - 1];
    for k in (0..n - 1).rev() {
        out[k] = d_prime[k] - c_prime[k] * out[k + 1];
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    type C64 = Complex<f64>;
    type Kern = SchrödingerChernoffComplex<C64>;

    fn make_kernel(n: usize) -> Kern {
        let grid = Grid1D::<f64>::new(-5.0, 5.0, n).unwrap();
        SchrödingerChernoffComplex::new(grid, |x: f64| 0.5 * x * x).unwrap()
    }

    fn make_state(kernel: &Kern) -> GridFnComplex1D<C64> {
        GridFnComplex1D::from_fn(kernel.grid, |x: f64| C64::new((-x * x / 2.0).exp(), 0.0))
    }

    #[test]
    fn ctor_valid() {
        let k = make_kernel(32);
        assert_eq!(k.v_at_node().len(), 32);
        assert_eq!(k.order(), 2);
    }

    #[test]
    fn ctor_rejects_non_finite_potential() {
        let grid = Grid1D::<f64>::new(-1.0, 1.0, 16).unwrap();
        let res = SchrödingerChernoffComplex::<C64>::new(grid, |_| f64::INFINITY);
        assert!(res.is_err());
    }

    #[test]
    fn growth_is_unitary() {
        let k = make_kernel(16);
        let g = k.growth();
        assert!((g.multiplier - 1.0).abs() < 1e-15);
        assert!(g.omega.abs() < 1e-15);
    }

    #[test]
    fn apply_into_zero_potential_norm_preserved() {
        // V = 0: free kinetic evolution. Unitarity must hold.
        let grid = Grid1D::<f64>::new(-5.0, 5.0, 64).unwrap();
        let kernel = SchrödingerChernoffComplex::<C64>::new(grid, |_| 0.0).unwrap();
        let src = GridFnComplex1D::from_fn(grid, |x: f64| C64::new((-x * x).exp(), 0.0));
        let norm_0 = src.norm_l2();

        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        kernel
            .apply_into(0.01, &src, &mut dst, &mut scratch)
            .unwrap();
        let norm_1 = dst.norm_l2();

        // Unitarity: ‖ψ₁‖ ≈ ‖ψ₀‖ to 1e-12 per step
        assert!(
            (norm_1 - norm_0).abs() < 1e-12,
            "unitarity deviation = {}",
            (norm_1 - norm_0).abs()
        );
    }

    #[test]
    fn apply_into_harmonic_oscillator_smoke() {
        // V = ½x²: does not panic, output is finite
        let kernel = make_kernel(64);
        let src = make_state(&kernel);
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        kernel
            .apply_into(0.01, &src, &mut dst, &mut scratch)
            .unwrap();
        assert!(dst.values.iter().all(|z| z.is_finite()));
    }

    #[test]
    fn in_subspace_checks() {
        let k = make_kernel(32);
        let s = make_state(&k);
        assert!(k.in_subspace(&s));

        // Too-small grid: construct directly (bypassing ctor guard)
        let tiny_grid = Grid1D::<f64>::new(-1.0, 1.0, 4).unwrap();
        let tiny = GridFnComplex1D::<C64>::from_fn(tiny_grid, |_| C64::new(1.0, 0.0));
        // A 4-node state: not in subspace (needs >= 5)
        assert!(!k.in_subspace(&tiny));
    }

    #[test]
    fn jet_is_unsupported() {
        let k = make_kernel(16);
        let s = make_state(&k);
        let mut out = [s.clone(), s.clone(), s.clone()];
        assert!(k.jet(&s, &mut out).is_err());
    }
}
