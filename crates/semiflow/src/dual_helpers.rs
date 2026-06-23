// `num_traits::Float` impl for `Dual<F>` — chain-rule ops (§46.2 NORMATIVE).
// Included into `dual.rs` via `include!` so that `Dual<F>`, `SemiflowFloat`,
// and arithmetic ops defined there are in scope without re-import.

impl<F: SemiflowFloat> Float for Dual<F> {
    // --- constants (tangent = 0) ---
    #[inline]
    fn nan() -> Self {
        Self::constant(F::nan())
    }
    #[inline]
    fn infinity() -> Self {
        Self::constant(F::infinity())
    }
    #[inline]
    fn neg_infinity() -> Self {
        Self::constant(F::neg_infinity())
    }
    #[inline]
    fn neg_zero() -> Self {
        Self::constant(F::neg_zero())
    }
    #[inline]
    fn min_value() -> Self {
        Self::constant(F::min_value())
    }
    #[inline]
    fn min_positive_value() -> Self {
        Self::constant(F::min_positive_value())
    }
    #[inline]
    fn epsilon() -> Self {
        Self::constant(F::epsilon())
    }
    #[inline]
    fn max_value() -> Self {
        Self::constant(F::max_value())
    }

    // --- predicates: value component ---
    #[inline]
    fn is_nan(self) -> bool {
        self.value.is_nan()
    }
    #[inline]
    fn is_infinite(self) -> bool {
        self.value.is_infinite()
    }
    #[inline]
    fn is_finite(self) -> bool {
        self.value.is_finite()
    }
    #[inline]
    fn is_normal(self) -> bool {
        self.value.is_normal()
    }
    #[inline]
    fn is_subnormal(self) -> bool {
        self.value.is_subnormal()
    }
    #[inline]
    fn is_sign_positive(self) -> bool {
        self.value.is_sign_positive()
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        self.value.is_sign_negative()
    }
    #[inline]
    fn classify(self) -> core::num::FpCategory {
        self.value.classify()
    }

    // --- rounding: tangent = 0 (piecewise-constant) ---
    #[inline]
    fn floor(self) -> Self {
        Self::constant(self.value.floor())
    }
    #[inline]
    fn ceil(self) -> Self {
        Self::constant(self.value.ceil())
    }
    #[inline]
    fn round(self) -> Self {
        Self::constant(self.value.round())
    }
    #[inline]
    fn trunc(self) -> Self {
        Self::constant(self.value.trunc())
    }
    #[inline]
    fn fract(self) -> Self {
        Self::constant(self.value.fract())
    }
    #[inline]
    fn signum(self) -> Self {
        Self::constant(self.value.signum())
    }

    /// §46.2 abs: `(|u|, sgn(u)·u')`.
    #[inline]
    fn abs(self) -> Self {
        Self::new(self.value.abs(), self.value.signum() * self.tangent)
    }

    // --- comparison (value-only, §46.2) ---
    #[inline]
    fn min(self, o: Self) -> Self {
        if self.value <= o.value {
            self
        } else {
            o
        }
    }
    #[inline]
    fn max(self, o: Self) -> Self {
        if self.value >= o.value {
            self
        } else {
            o
        }
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        if self.value < lo.value {
            lo
        } else if self.value > hi.value {
            hi
        } else {
            self
        }
    }

    /// §46.2 recip: `(1/u, -u'/u²)`.
    #[inline]
    fn recip(self) -> Self {
        Self::new(
            self.value.recip(),
            -self.tangent / (self.value * self.value),
        )
    }

    /// §46.2 powi: `(uⁿ, n·u^(n-1)·u')`.
    #[inline]
    fn powi(self, n: i32) -> Self {
        Self::new(
            self.value.powi(n),
            F::from(n).unwrap_or_else(F::zero) * self.value.powi(n - 1) * self.tangent,
        )
    }

    /// powf: chain rule `p·u^(p-1)·u'`.
    #[inline]
    fn powf(self, p: Self) -> Self {
        let v = self.value.powf(p.value);
        Self::new(
            v,
            p.value * self.value.powf(p.value - F::one()) * self.tangent,
        )
    }

    /// §46.2 sqrt: `(√u, u'/(2√u))`.
    #[inline]
    fn sqrt(self) -> Self {
        let v = self.value.sqrt();
        Self::new(v, self.tangent / (v + v))
    }

    /// §46.2 exp: `(eᵘ, eᵘ·u')`.
    #[inline]
    fn exp(self) -> Self {
        let v = self.value.exp();
        Self::new(v, v * self.tangent)
    }

    /// exp2: `2^u·ln2·u'`.
    #[inline]
    fn exp2(self) -> Self {
        let v = self.value.exp2();
        let ln2 = F::from(2.0_f64.ln()).unwrap_or_else(F::zero);
        Self::new(v, v * ln2 * self.tangent)
    }

    /// §46.2 ln: `(ln u, u'/u)`.
    #[inline]
    fn ln(self) -> Self {
        Self::new(self.value.ln(), self.tangent / self.value)
    }

    /// log base b = ln/ln(base).
    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        let ln2 = F::from(2.0_f64.ln()).unwrap_or_else(F::zero);
        Self::new(self.value.log2(), self.tangent / (self.value * ln2))
    }
    #[inline]
    fn log10(self) -> Self {
        let ln10 = F::from(10.0_f64.ln()).unwrap_or_else(F::zero);
        Self::new(self.value.log10(), self.tangent / (self.value * ln10))
    }

    #[allow(deprecated)]
    #[inline]
    fn abs_sub(self, o: Self) -> Self {
        Self::constant(self.value.abs_sub(o.value))
    }

    /// cbrt: `u'/(3·u^(2/3))`.
    #[inline]
    fn cbrt(self) -> Self {
        let v = self.value.cbrt();
        let three = F::from(3.0_f64).unwrap_or_else(F::one);
        Self::new(v, self.tangent / (three * v * v))
    }

    /// hypot: `(x·x' + y·y')/hypot`.
    #[inline]
    fn hypot(self, o: Self) -> Self {
        let v = self.value.hypot(o.value);
        Self::new(v, (self.value * self.tangent + o.value * o.tangent) / v)
    }

    /// §46.2 sin: `(sin u, cos(u)·u')`.
    #[inline]
    fn sin(self) -> Self {
        Self::new(self.value.sin(), self.value.cos() * self.tangent)
    }
    /// §46.2 cos: `(cos u, -sin(u)·u')`.
    #[inline]
    fn cos(self) -> Self {
        Self::new(self.value.cos(), -self.value.sin() * self.tangent)
    }
    /// tan: `u'/cos²(u)`.
    #[inline]
    fn tan(self) -> Self {
        let c = self.value.cos();
        Self::new(self.value.tan(), self.tangent / (c * c))
    }
    #[inline]
    fn asin(self) -> Self {
        Self::new(
            self.value.asin(),
            self.tangent / (F::one() - self.value * self.value).sqrt(),
        )
    }
    #[inline]
    fn acos(self) -> Self {
        Self::new(
            self.value.acos(),
            -self.tangent / (F::one() - self.value * self.value).sqrt(),
        )
    }
    #[inline]
    fn atan(self) -> Self {
        Self::new(
            self.value.atan(),
            self.tangent / (F::one() + self.value * self.value),
        )
    }
    /// atan2: `(x·y' − y·x') / (x²+y²)`.
    #[inline]
    fn atan2(self, o: Self) -> Self {
        let d = self.value * self.value + o.value * o.value;
        Self::new(
            self.value.atan2(o.value),
            (o.value * self.tangent - self.value * o.tangent) / d,
        )
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self::new(self.value.exp_m1(), self.value.exp() * self.tangent)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        Self::new(self.value.ln_1p(), self.tangent / (F::one() + self.value))
    }
    #[inline]
    fn sinh(self) -> Self {
        Self::new(self.value.sinh(), self.value.cosh() * self.tangent)
    }
    #[inline]
    fn cosh(self) -> Self {
        Self::new(self.value.cosh(), self.value.sinh() * self.tangent)
    }
    /// tanh: `(1 - tanh²(u))·u'`.
    #[inline]
    fn tanh(self) -> Self {
        let v = self.value.tanh();
        Self::new(v, (F::one() - v * v) * self.tangent)
    }
    #[inline]
    fn asinh(self) -> Self {
        Self::new(
            self.value.asinh(),
            self.tangent / (self.value * self.value + F::one()).sqrt(),
        )
    }
    #[inline]
    fn acosh(self) -> Self {
        Self::new(
            self.value.acosh(),
            self.tangent / (self.value * self.value - F::one()).sqrt(),
        )
    }
    #[inline]
    fn atanh(self) -> Self {
        Self::new(
            self.value.atanh(),
            self.tangent / (F::one() - self.value * self.value),
        )
    }

    #[inline]
    fn to_degrees(self) -> Self {
        let k = F::from(180.0_f64 / core::f64::consts::PI).unwrap_or_else(F::one);
        Self::new(self.value.to_degrees(), self.tangent * k)
    }
    #[inline]
    fn to_radians(self) -> Self {
        let k = F::from(core::f64::consts::PI / 180.0_f64).unwrap_or_else(F::one);
        Self::new(self.value.to_radians(), self.tangent * k)
    }

    /// `integer_decode` delegates to value (bitwise structure of value).
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.value.integer_decode()
    }

    /// `mul_add`: chain rule `self'·a + self·a' + b'`.
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(
            self.value.mul_add(a.value, b.value),
            self.tangent * a.value + self.value * a.tangent + b.tangent,
        )
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        Self::new(
            self.value.copysign(sign.value),
            self.tangent.copysign(sign.value),
        )
    }
}
