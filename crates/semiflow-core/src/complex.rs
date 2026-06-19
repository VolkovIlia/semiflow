//! v4.0 B6 — `SemiflowComplex` trait (ADR-0079, math.md §30.2).
//!
//! Generic complex-number arithmetic abstraction for v4.0 native Schrödinger
//! Option B (lifts the v2.2 ADR-0057 real-pair workaround to a first-class
//! complex state type).
//!
//! ## Mathematical foundation
//!
//! Definition 30.1 (math.md §30.2, NORMATIVE): a type `C` satisfies
//! `SemiflowComplex` iff it carries real/imaginary components of a
//! `SemiflowFloat`-valued associated type `C::Real`, supports the four field
//! operations + negation, and provides modulus, conjugate, exponential,
//! principal square root, and polar/rectangular constructors.
//!
//! ## Citations
//!
//! - Pazy 1983 §2.1 — Banach-space C₀-semigroup foundations.
//! - Cheng 2008 §4 — Schrödinger semigroups.
//! - Engel-Nagel 2000 §IV — unitary semigroups on Hilbert space.
//!
//! ## Reference implementations (v4.0)
//!
//! - [`num_complex::Complex<f64>`] — standard f64 complex.
//! - [`num_complex::Complex<f32>`] — reduced-precision f32 complex.
//!
//! Downstream users may plug in alternative complex types (GMP-precision,
//! SIMD-batch, etc.) by implementing this trait without modifying any kernel.

extern crate alloc;

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::{Float, One, Zero};

use crate::float::SemiflowFloat;

// ---------------------------------------------------------------------------
// SemiflowComplex trait
// ---------------------------------------------------------------------------

/// Complex-number arithmetic abstraction for generic Chernoff kernels.
///
/// Parallels [`SemiflowFloat`] for real scalars (ADR-0025). An implementor
/// carries two `Self::Real` components and the full `ℂ`-algebra.
///
/// ## Required bounds
///
/// The super-trait list mirrors exactly what `SchrödingerChernoffComplex`
/// (and future complex kernels) need:
/// - Arithmetic ops `+, −, ×, ÷` with output `Self`.
/// - In-place variants (`AddAssign`, etc.) for hot-path loops.
/// - `Neg<Output = Self>` for Cayley-map sign flips.
/// - `Copy + Send + Sync + 'static` — stored in kernel structs, shared
///   across thread boundaries.
///
/// ## v4.0 reference impls
///
/// Shipped for `num_complex::Complex<f64>` and `num_complex::Complex<f32>`.
/// The `num-complex = "0.4"` dependency is promoted from "reserved" to
/// "direct" per ADR-0079; total `semiflow-core` deps = 3/3 (constitution
/// v1.8.0 cap).
pub trait SemiflowComplex:
    Copy
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Real scalar type for re/im components and the modulus.
    type Real: SemiflowFloat;

    /// Real part: `Re(z)`.
    fn re(self) -> Self::Real;

    /// Imaginary part: `Im(z)`.
    fn im(self) -> Self::Real;

    /// Modulus: `|z| = √(Re²+Im²)`.
    fn abs(self) -> Self::Real;

    /// Complex conjugate: `Re(z̄) = Re(z)`, `Im(z̄) = -Im(z)`.
    #[must_use]
    fn conj(self) -> Self;

    /// Embed real scalar: `from_real(r) = r + 0·i`.
    fn from_real(r: Self::Real) -> Self;

    /// Construct from rectangular components: `re + im·i`.
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;

    /// Construct from polar form: `r·(cos θ + i·sin θ)`.
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;

    /// Complex exponential: `exp(re + im·i) = exp(re)·(cos(im) + i·sin(im))`.
    #[must_use]
    fn exp(self) -> Self;

    /// Principal square root: `(√z)² = z`, branch `arg(√z) ∈ (−π/2, π/2]`.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Additive identity: `0 + 0·i`.
    #[must_use]
    fn zero() -> Self {
        Self::from_real(<Self::Real as Zero>::zero())
    }

    /// Multiplicative identity: `1 + 0·i`.
    #[must_use]
    fn one() -> Self {
        Self::from_real(<Self::Real as One>::one())
    }

    /// Imaginary unit: `0 + 1·i`.
    #[must_use]
    fn i() -> Self {
        Self::from_parts(<Self::Real as Zero>::zero(), <Self::Real as One>::one())
    }

    /// Returns `true` if both components are finite.
    fn is_finite(self) -> bool {
        Float::is_finite(self.re()) && Float::is_finite(self.im())
    }
}

// ---------------------------------------------------------------------------
// impl for num_complex::Complex<f64>
// ---------------------------------------------------------------------------

impl SemiflowComplex for num_complex::Complex<f64> {
    type Real = f64;

    #[inline]
    fn re(self) -> f64 {
        self.re
    }

    #[inline]
    fn im(self) -> f64 {
        self.im
    }

    #[inline]
    fn abs(self) -> f64 {
        num_complex::Complex::norm(self)
    }

    #[inline]
    fn conj(self) -> Self {
        num_complex::Complex::conj(&self)
    }

    #[inline]
    fn from_real(r: f64) -> Self {
        Self::new(r, 0.0)
    }

    #[inline]
    fn from_parts(re: f64, im: f64) -> Self {
        Self::new(re, im)
    }

    #[inline]
    fn from_polar(r: f64, theta: f64) -> Self {
        num_complex::Complex::from_polar(r, theta)
    }

    #[inline]
    fn exp(self) -> Self {
        num_complex::Complex::exp(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        num_complex::Complex::sqrt(self)
    }
}

// ---------------------------------------------------------------------------
// impl for num_complex::Complex<f32>
// ---------------------------------------------------------------------------

impl SemiflowComplex for num_complex::Complex<f32> {
    type Real = f32;

    #[inline]
    fn re(self) -> f32 {
        self.re
    }

    #[inline]
    fn im(self) -> f32 {
        self.im
    }

    #[inline]
    fn abs(self) -> f32 {
        num_complex::Complex::norm(self)
    }

    #[inline]
    fn conj(self) -> Self {
        num_complex::Complex::conj(&self)
    }

    #[inline]
    fn from_real(r: f32) -> Self {
        Self::new(r, 0.0)
    }

    #[inline]
    fn from_parts(re: f32, im: f32) -> Self {
        Self::new(re, im)
    }

    #[inline]
    fn from_polar(r: f32, theta: f32) -> Self {
        num_complex::Complex::from_polar(r, theta)
    }

    #[inline]
    fn exp(self) -> Self {
        num_complex::Complex::exp(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        num_complex::Complex::sqrt(self)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::SemiflowComplex;
    use num_complex::Complex;

    type C = Complex<f64>;
    type C32 = Complex<f32>;

    // Helpers
    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    #[test]
    fn re_im_roundtrip() {
        let z = c(3.0, -4.0);
        assert_eq!(z.re(), 3.0_f64);
        assert_eq!(z.im(), -4.0_f64);
    }

    #[test]
    fn abs_canonical() {
        // |3 + 4i| = 5
        assert!((c(3.0, 4.0).abs() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn conj_signs() {
        let z = c(2.0, -5.0);
        let zc = z.conj();
        assert_eq!(zc.re(), 2.0);
        assert_eq!(zc.im(), 5.0);
    }

    #[test]
    fn from_polar_i() {
        // r=1, θ=π/2  →  i (within ULP)
        let z = C::from_polar(1.0, core::f64::consts::FRAC_PI_2);
        assert!(z.re().abs() < 1e-15);
        assert!((z.im() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn euler_identity() {
        // exp(iπ) + 1 ≈ 0
        let z = C::from_parts(0.0, core::f64::consts::PI).exp();
        assert!((z.re() + 1.0).abs() < 1e-15);
        assert!(z.im().abs() < 1e-15);
    }

    #[test]
    fn sqrt_minus_one() {
        // √(−1) = i (principal branch)
        let z = c(-1.0, 0.0).sqrt();
        assert!(z.re().abs() < 1e-15);
        assert!((z.im() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn zero_one_i_arithmetic() {
        let zero = C::zero();
        let one = C::one();
        let i = C::i();
        // i * i = -1
        let ii = i * i;
        assert!((ii.re() + 1.0).abs() < 1e-15);
        assert!(ii.im().abs() < 1e-15);
        // 0 + 1 = 1
        assert_eq!((zero + one).re(), 1.0);
    }

    #[test]
    fn is_finite_checks() {
        assert!(c(1.0, -1.0).is_finite());
        assert!(!c(f64::NAN, 0.0).is_finite());
        assert!(!c(0.0, f64::INFINITY).is_finite());
    }

    #[test]
    fn f32_impl_smoke() {
        let z = C32::from_parts(0.0, core::f32::consts::PI).exp();
        assert!((z.re() + 1.0).abs() < 1e-6);
    }
}
