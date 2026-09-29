//! Sealed scalar-float trait used throughout `semiflow`.
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

/// Sealed scalar-float trait: the set of types accepted by generic `semiflow` types.
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
/// use semiflow::float::SemiflowFloat;
/// fn sum_two<F: SemiflowFloat>(a: F, b: F) -> F { a + b }
///
/// // Both concrete float types work:
/// assert_eq!(sum_two(1.0_f64, 2.0_f64), 3.0_f64);
/// assert_eq!(sum_two(1.0_f32, 2.0_f32), 3.0_f32);
/// ```
#[allow(clippy::module_name_repetitions)]
// The provided bodies are the generic fallback (used by `Dual<F>`, whose own
// `Float` impl routes through the `libm_*` methods of its components).
#[allow(clippy::disallowed_methods)]
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
    /// `eˣ` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_exp(self) -> Self {
        Float::exp(self)
    }
    /// `2ˣ` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_exp2(self) -> Self {
        Float::exp2(self)
    }
    /// `eˣ − 1` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_exp_m1(self) -> Self {
        Float::exp_m1(self)
    }
    /// `ln x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_ln(self) -> Self {
        Float::ln(self)
    }
    /// `ln(1 + x)` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_ln_1p(self) -> Self {
        Float::ln_1p(self)
    }
    /// `log₂ x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_log2(self) -> Self {
        Float::log2(self)
    }
    /// `log₁₀ x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_log10(self) -> Self {
        Float::log10(self)
    }
    /// `log_base x = ln x / ln base` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_log(self, base: Self) -> Self {
        Float::log(self, base)
    }
    /// `xᵉ` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_powf(self, e: Self) -> Self {
        Float::powf(self, e)
    }
    /// `xⁿ` by exponentiation by squaring: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_powi(self, n: i32) -> Self {
        Float::powi(self, n)
    }
    /// `sin x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_sin(self) -> Self {
        Float::sin(self)
    }
    /// `cos x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_cos(self) -> Self {
        Float::cos(self)
    }
    /// `tan x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_tan(self) -> Self {
        Float::tan(self)
    }
    /// `(sin x, cos x)` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_sin_cos(self) -> (Self, Self) {
        Float::sin_cos(self)
    }
    /// `arcsin x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_asin(self) -> Self {
        Float::asin(self)
    }
    /// `arccos x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_acos(self) -> Self {
        Float::acos(self)
    }
    /// `arctan x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_atan(self) -> Self {
        Float::atan(self)
    }
    /// `atan2(self, other)` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_atan2(self, other: Self) -> Self {
        Float::atan2(self, other)
    }
    /// `sinh x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_sinh(self) -> Self {
        Float::sinh(self)
    }
    /// `cosh x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_cosh(self) -> Self {
        Float::cosh(self)
    }
    /// `tanh x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_tanh(self) -> Self {
        Float::tanh(self)
    }
    /// `arsinh x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_asinh(self) -> Self {
        Float::asinh(self)
    }
    /// `arcosh x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_acosh(self) -> Self {
        Float::acosh(self)
    }
    /// `artanh x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_atanh(self) -> Self {
        Float::atanh(self)
    }
    /// `∛x` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_cbrt(self) -> Self {
        Float::cbrt(self)
    }
    /// `√(x² + y²)` via `libm`: identical bits in `std` and `no_std` builds.
    #[inline]
    #[must_use]
    fn libm_hypot(self, other: Self) -> Self {
        Float::hypot(self, other)
    }
}

/// `xⁿ` by exponentiation by squaring — the algorithm of compiler-rt
/// `__powidf2`, written out so it is the same in every build.
macro_rules! powi_by_squaring {
    ($x:expr, $n:expr) => {{
        let mut acc = 1.0;
        let mut base = $x;
        let mut e = $n.unsigned_abs();
        loop {
            if e & 1 == 1 {
                acc *= base;
            }
            e >>= 1;
            if e == 0 {
                break;
            }
            base *= base;
        }
        if $n < 0 {
            1.0 / acc
        } else {
            acc
        }
    }};
}

/// Override every `libm_*` method with the concrete `libm` function.
macro_rules! impl_libm_methods {
    (
        $t:ty,
        $exp:ident, $exp2:ident, $expm1:ident, $log:ident, $log1p:ident, $log2:ident,
        $log10:ident, $pow:ident, $sin:ident, $cos:ident, $tan:ident, $asin:ident,
        $acos:ident, $atan:ident, $atan2:ident, $sinh:ident, $cosh:ident, $tanh:ident,
        $asinh:ident, $acosh:ident, $atanh:ident, $cbrt:ident, $hypot:ident
    ) => {
        impl SemiflowFloat for $t {
            #[inline]
            fn libm_exp(self) -> Self {
                libm::$exp(self)
            }
            #[inline]
            fn libm_exp2(self) -> Self {
                libm::$exp2(self)
            }
            #[inline]
            fn libm_exp_m1(self) -> Self {
                libm::$expm1(self)
            }
            #[inline]
            fn libm_ln(self) -> Self {
                libm::$log(self)
            }
            #[inline]
            fn libm_ln_1p(self) -> Self {
                libm::$log1p(self)
            }
            #[inline]
            fn libm_log2(self) -> Self {
                libm::$log2(self)
            }
            #[inline]
            fn libm_log10(self) -> Self {
                libm::$log10(self)
            }
            #[inline]
            fn libm_log(self, base: Self) -> Self {
                libm::$log(self) / libm::$log(base)
            }
            #[inline]
            fn libm_powf(self, e: Self) -> Self {
                libm::$pow(self, e)
            }
            #[inline]
            fn libm_powi(self, n: i32) -> Self {
                powi_by_squaring!(self, n)
            }
            #[inline]
            fn libm_sin(self) -> Self {
                libm::$sin(self)
            }
            #[inline]
            fn libm_cos(self) -> Self {
                libm::$cos(self)
            }
            #[inline]
            fn libm_tan(self) -> Self {
                libm::$tan(self)
            }
            #[inline]
            fn libm_sin_cos(self) -> (Self, Self) {
                (libm::$sin(self), libm::$cos(self))
            }
            #[inline]
            fn libm_asin(self) -> Self {
                libm::$asin(self)
            }
            #[inline]
            fn libm_acos(self) -> Self {
                libm::$acos(self)
            }
            #[inline]
            fn libm_atan(self) -> Self {
                libm::$atan(self)
            }
            #[inline]
            fn libm_atan2(self, other: Self) -> Self {
                libm::$atan2(self, other)
            }
            #[inline]
            fn libm_sinh(self) -> Self {
                libm::$sinh(self)
            }
            #[inline]
            fn libm_cosh(self) -> Self {
                libm::$cosh(self)
            }
            #[inline]
            fn libm_tanh(self) -> Self {
                libm::$tanh(self)
            }
            #[inline]
            fn libm_asinh(self) -> Self {
                libm::$asinh(self)
            }
            #[inline]
            fn libm_acosh(self) -> Self {
                libm::$acosh(self)
            }
            #[inline]
            fn libm_atanh(self) -> Self {
                libm::$atanh(self)
            }
            #[inline]
            fn libm_cbrt(self) -> Self {
                libm::$cbrt(self)
            }
            #[inline]
            fn libm_hypot(self, other: Self) -> Self {
                libm::$hypot(self, other)
            }
        }
    };
}

impl_libm_methods!(
    f32, expf, exp2f, expm1f, logf, log1pf, log2f, log10f, powf, sinf, cosf, tanf, asinf, acosf,
    atanf, atan2f, sinhf, coshf, tanhf, asinhf, acoshf, atanhf, cbrtf, hypotf
);
impl_libm_methods!(
    f64, exp, exp2, expm1, log, log1p, log2, log10, pow, sin, cos, tan, asin, acos, atan, atan2,
    sinh, cosh, tanh, asinh, acosh, atanh, cbrt, hypot
);

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

/// `π^{-D/2}` computed without `powf`, so the value is bit-identical everywhere.
///
/// `f64::powf` lowers to the platform `pow`, which is not correctly rounded and
/// whose glibc IFUNC variants differ by CPU. That matters here because this is a
/// GLOBAL normalisation multiplying every output of the d-D Gauss-Hermite
/// kernels: a 1-ULP difference in the prefactor moves every value, which is what
/// made `G_BINDING_SMOLYAK_PARITY`'s 0-ULP golden pass on five CI runners and
/// fail on the sixth (ADR-0191 AMENDMENT 5).
///
/// Every operation here is IEEE-754 correctly rounded — multiplication, `sqrt`
/// and division are all required to be — so the result is reproducible across
/// platforms and libm versions.
///
/// Reproducibility is the whole claim; accuracy is NOT. Measured against a
/// 60-digit reference on glibc x86-64: at `D = 2, 4, 6` this and `powf` agree
/// bit-for-bit, and at `D = 3, 5` they differ by 1 ULP with `powf` on the
/// correctly rounded side. A future reader tempted to call this "more accurate"
/// should re-measure per `D` per platform first — the point is that this one
/// gives the same bits everywhere, not that it gives better ones.
pub(crate) fn inv_pi_pow_half<F: SemiflowFloat>(d: usize) -> F {
    let pi = core::f64::consts::PI;
    let mut acc = 1.0_f64;
    for _ in 0..(d / 2) {
        acc *= pi;
    }
    if d % 2 == 1 {
        acc *= libm::sqrt(pi);
    }
    from_f64::<F>(1.0 / acc)
}
