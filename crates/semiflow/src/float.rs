//! Sealed scalar-float trait used throughout `semiflow-core`.
//!
//! [`SemiflowFloat`] bundles the exact bounds required by every generic Chernoff
//! implementation. Only `f32` and `f64` implement the trait (explicit `impl`
//! blocks; no blanket impl) so the set of accepted scalars cannot grow without
//! a deliberate ADR. [`Dual<F>`](crate::dual::Dual) is the deliberate third
//! member, authorized by ADR-0133.
//!
//! See [`docs/adr/0025-generic-over-float.md`](../docs/adr/) and
//! `contracts/semiflow-core.math.md` for motivation and bound derivation.

use core::{
    fmt::{Debug, Display},
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num_traits::Float;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Sealed scalar-float trait: the set of types accepted by generic `semiflow-core` types.
///
/// ## Sealed design (ADR-0025, v0.9.0 Block D pilot)
///
/// Only `f32` and `f64` implement `SemiflowFloat` — explicit `impl` blocks,
/// no blanket impl. This prevents accidental instantiation with `Complex<f64>`
/// or other numeric types before the relevant ADR lands.
///
/// ## Bound rationale
///
/// The supertrait list mirrors what every Chernoff kernel actually needs:
/// - [`num_traits::Float`] — `sqrt`, `abs`, `is_finite`, `floor`, `exp`, `ln`, …
/// - `AddAssign + SubAssign + MulAssign + DivAssign` — in-place BLAS operations in [`crate::State`].
/// - `Send + Sync + Copy + 'static` — grid and function types stored in structs,
///   shared across thread boundaries when `parallel` is enabled.
/// - `Debug + Display` — error messages include the offending value.
/// - `PartialOrd` — comparisons in validation helpers (`tau < 0`, `a(x) < 0`).
///
/// ## SIMD note
///
/// `f64` uses AVX2/NEON SIMD paths (Catmull-Rom, K-kernel) when the `simd`
/// feature is enabled. `f32` uses scalar-only paths; a dedicated `f32x8`
/// intrinsic path is deferred to a future ADR.
///
/// ## Example
///
/// ```rust
/// use semiflow_core::float::SemiflowFloat;
/// fn sum_two<F: SemiflowFloat>(a: F, b: F) -> F { a + b }
///
/// // Both concrete float types work:
/// assert_eq!(sum_two(1.0_f64, 2.0_f64), 3.0_f64);
/// assert_eq!(sum_two(1.0_f32, 2.0_f32), 3.0_f32);
/// ```
#[allow(clippy::module_name_repetitions)]
pub trait SemiflowFloat:
    Float
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Send
    + Sync
    + Copy
    + Debug
    + Display
    + PartialOrd
    + 'static
{
}

impl SemiflowFloat for f32 {}
impl SemiflowFloat for f64 {}

// ---------------------------------------------------------------------------
// Small numeric helpers used by generic kernels
// ---------------------------------------------------------------------------

/// Return the additive identity (`0.0`) for `F`.
// future generic helper; companion to `one` and `two` which are actively used
#[allow(dead_code)]
#[inline]
pub(crate) fn zero<F: SemiflowFloat>() -> F {
    F::zero()
}

/// Return the multiplicative identity (`1.0`) for `F`.
#[inline]
pub(crate) fn one<F: SemiflowFloat>() -> F {
    F::one()
}

/// Return `2.0` as `F`.
#[inline]
pub(crate) fn two<F: SemiflowFloat>() -> F {
    let o = one::<F>();
    o + o
}

/// Return `0.5` as `F`.
#[inline]
pub(crate) fn half<F: SemiflowFloat>() -> F {
    one::<F>() / two::<F>()
}

/// Convert an `f64` literal to `F`.
///
/// Uses `num_traits::cast::ToPrimitive` + `from_f64`.  Panics in debug if
/// the conversion fails; in release the unwrap degrades to zero (the
/// `num_traits` contract for out-of-range).
#[inline]
pub(crate) fn from_f64<F: SemiflowFloat>(v: f64) -> F {
    F::from(v).unwrap_or_else(F::zero)
}
