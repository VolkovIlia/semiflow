//! `num_complex::Complex` transcendentals evaluated through `SemiflowFloat::libm_*`.
//!
//! `num-complex` routes its `Float`-based methods through `num-traits`, whose
//! backend is the platform math library in `std` builds and `libm` in `no_std`
//! builds. These ports keep num-complex 0.4.6's formulas and special cases
//! verbatim but always use `libm`, so `std` and `no_std` produce identical bits
//! (ADR-0200). `clippy::disallowed_methods` forbids the num-complex originals.

use num_complex::Complex;

use crate::float::SemiflowFloat;

/// `|z|` (num-complex `Complex::norm`).
#[inline]
pub(crate) fn norm<T: SemiflowFloat>(z: Complex<T>) -> T {
    z.re.libm_hypot(z.im)
}

/// Principal argument `arg z ∈ (−π, π]` (num-complex `Complex::arg`).
#[inline]
pub(crate) fn arg<T: SemiflowFloat>(z: Complex<T>) -> T {
    z.im.libm_atan2(z.re)
}

/// `(|z|, arg z)` (num-complex `Complex::to_polar`).
#[inline]
pub(crate) fn to_polar<T: SemiflowFloat>(z: Complex<T>) -> (T, T) {
    (norm(z), arg(z))
}

/// `r·e^{iθ}` (num-complex `Complex::from_polar`).
#[inline]
pub(crate) fn from_polar<T: SemiflowFloat>(r: T, theta: T) -> Complex<T> {
    Complex::new(r * theta.libm_cos(), r * theta.libm_sin())
}

/// `e^z` (num-complex `Complex::exp`, including its ±∞/NaN corner cases).
#[inline]
pub(crate) fn exp<T: SemiflowFloat>(z: Complex<T>) -> Complex<T> {
    let Complex { re, mut im } = z;
    if re.is_infinite() {
        if re < T::zero() {
            if !im.is_finite() {
                return Complex::new(T::zero(), T::zero());
            }
        } else if im == T::zero() || !im.is_finite() {
            if im.is_infinite() {
                im = T::nan();
            }
            return Complex::new(re, im);
        }
    } else if re.is_nan() && im == T::zero() {
        return z;
    }
    from_polar(re.libm_exp(), im)
}

/// Principal square root, branch cut `(−∞, 0)` continuous from above
/// (num-complex `Complex::sqrt`).
#[inline]
pub(crate) fn sqrt<T: SemiflowFloat>(z: Complex<T>) -> Complex<T> {
    let two = T::one() + T::one();
    if z.im.is_zero() {
        if z.re.is_sign_positive() {
            Complex::new(z.re.sqrt(), z.im)
        } else {
            let re = T::zero();
            let im = (-z.re).sqrt();
            if z.im.is_sign_positive() {
                Complex::new(re, im)
            } else {
                Complex::new(re, -im)
            }
        }
    } else if z.re.is_zero() {
        let x = (z.im.abs() / two).sqrt();
        if z.im.is_sign_positive() {
            Complex::new(x, x)
        } else {
            Complex::new(x, -x)
        }
    } else {
        let (r, theta) = to_polar(z);
        from_polar(r.sqrt(), theta / two)
    }
}
