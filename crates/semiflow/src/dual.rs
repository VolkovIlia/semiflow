//! [`Dual<F>`] — forward-mode dual-number scalar field for automatic differentiation.
//!
//! `Dual(a, b) = a + ε·b`, `ε² = 0` (nilpotent; NOT complex).
//! Seeding a parameter tangent with 1.0 propagates its exact derivative
//! through every Chernoff kernel at zero new heap allocation (§46, ADR-0133).
//!
//! Arithmetic rules (§46.2, NORMATIVE):
//! ```text
//! U + V = (u+v, u'+v')       (linearity)
//! U · V = (uv,  u'v + uv')   (product rule)
//! U / V = (u/v, (u'v − uv')/v²)  (quotient rule)
//! g(U)  = (g(u), g'(u)·u')   (chain rule — all transcendentals)
//! ```
//! `PartialOrd` compares value only; tangent is carried, never compared,
//! so `tau < 0` / `a(x) < 0` validation guards behave identically to scalar.
//!
//! `num_traits::Float` chain-rule ops are in `dual_helpers.rs` (included below).

use core::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};

use crate::float::SemiflowFloat;

// ── Struct ───────────────────────────────────────────────────────────────────

/// Dual number `value + ε·tangent` over base float field `F` (§46.1, ADR-0133).
///
/// Implements [`SemiflowFloat`]: every generic kernel at `F = Dual<f64>` gains
/// forward-mode AD at zero allocation. `Dual<Dual<f64>>` gives second
/// derivatives (§46.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<F: SemiflowFloat = f64> {
    /// Primal value component.
    pub value: F,
    /// Tangent (derivative) component.
    pub tangent: F,
}

impl<F: SemiflowFloat> Dual<F> {
    /// Explicit `(value, tangent)` pair.
    #[inline]
    pub const fn new(value: F, tangent: F) -> Self {
        Self { value, tangent }
    }
    /// θ-independent constant: tangent = 0.
    #[inline]
    pub fn constant(value: F) -> Self {
        Self {
            value,
            tangent: F::zero(),
        }
    }
    /// Seeded parameter θ: tangent = 1 (forward sweep seeds here).
    #[inline]
    pub fn variable(value: F) -> Self {
        Self {
            value,
            tangent: F::one(),
        }
    }
}

impl<F: SemiflowFloat + fmt::Display> fmt::Display for Dual<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+ε·{}", self.value, self.tangent)
    }
}

// ── Arithmetic — §46.2 (NORMATIVE) ──────────────────────────────────────────

impl<F: SemiflowFloat> Add for Dual<F> {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        Self::new(self.value + r.value, self.tangent + r.tangent)
    }
}
impl<F: SemiflowFloat> Sub for Dual<F> {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        Self::new(self.value - r.value, self.tangent - r.tangent)
    }
}
impl<F: SemiflowFloat> Mul for Dual<F> {
    type Output = Self;
    /// §46.2 product rule: `(uv, u'v + uv')`.
    #[inline]
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.value * r.value,
            self.tangent * r.value + self.value * r.tangent,
        )
    }
}
impl<F: SemiflowFloat> Div for Dual<F> {
    type Output = Self;
    /// §46.2 quotient rule: `(u/v, (u'v − uv')/v²)`.
    #[inline]
    fn div(self, r: Self) -> Self {
        Self::new(
            self.value / r.value,
            (self.tangent * r.value - self.value * r.tangent) / (r.value * r.value),
        )
    }
}
impl<F: SemiflowFloat> Rem for Dual<F> {
    type Output = Self;
    /// Remainder: value only; tangent = 0 (piecewise-constant, §46.2).
    #[inline]
    fn rem(self, r: Self) -> Self {
        Self::new(self.value % r.value, F::zero())
    }
}
impl<F: SemiflowFloat> Neg for Dual<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.value, -self.tangent)
    }
}

impl<F: SemiflowFloat> AddAssign for Dual<F> {
    #[inline]
    fn add_assign(&mut self, r: Self) {
        *self = *self + r;
    }
}
impl<F: SemiflowFloat> SubAssign for Dual<F> {
    #[inline]
    fn sub_assign(&mut self, r: Self) {
        *self = *self - r;
    }
}
impl<F: SemiflowFloat> MulAssign for Dual<F> {
    #[inline]
    fn mul_assign(&mut self, r: Self) {
        *self = *self * r;
    }
}
impl<F: SemiflowFloat> DivAssign for Dual<F> {
    #[inline]
    fn div_assign(&mut self, r: Self) {
        *self = *self / r;
    }
}
impl<F: SemiflowFloat> RemAssign for Dual<F> {
    #[inline]
    fn rem_assign(&mut self, r: Self) {
        *self = *self % r;
    }
}

// ── PartialOrd — value only (§46.2) ─────────────────────────────────────────

impl<F: SemiflowFloat> PartialOrd for Dual<F> {
    /// §46.2: ordering compares **value only**; tangent is never compared.
    #[inline]
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&o.value)
    }
}

// ── num_traits: Zero / One / ToPrimitive / NumCast / Num ────────────────────

impl<F: SemiflowFloat> Zero for Dual<F> {
    #[inline]
    fn zero() -> Self {
        Self::constant(F::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}
impl<F: SemiflowFloat> One for Dual<F> {
    #[inline]
    fn one() -> Self {
        Self::constant(F::one())
    }
}

impl<F: SemiflowFloat> ToPrimitive for Dual<F> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.value.to_f64()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.value.to_f32()
    }
}

impl<F: SemiflowFloat> NumCast for Dual<F> {
    #[inline]
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        F::from(n).map(Self::constant)
    }
}

impl<F: SemiflowFloat> Num for Dual<F> {
    type FromStrRadixErr = <F as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        F::from_str_radix(s, radix).map(Self::constant)
    }
}

// ── num_traits::Float — §46.2 chain-rule ops (NORMATIVE) ────────────────────
// Moved to dual_helpers.rs (batch H4 suckless split) and included below.

include!("dual_helpers.rs");

// ── SemiflowFloat impl (§46, ADR-0133) ────────────────────────────────────────

/// Registers `Dual<F>` as a first-class member of the `SemiflowFloat` family.
///
/// This single `impl` makes every generic kernel accept `F = Dual<f64>`;
/// no per-kernel `ChernoffFunction` impl is needed (§46, ADR-0133).
impl<F: SemiflowFloat> SemiflowFloat for Dual<F> {}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    include!("dual_tests.rs");
}
