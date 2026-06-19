// Private helpers for DiffusionChernoff — included into diffusion.rs via include!() (batch H8).
// Float ops are verbatim; sum/product order unchanged. All items visible as if defined inline.

// ---------------------------------------------------------------------------
// Private helpers — f64-specific (SIMD path, bit-identical to v0.8.x)
// ---------------------------------------------------------------------------

/// Validate `tau`: must be finite and non-negative (f64).
#[inline]
fn validate_tau_f64(tau: f64) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau,
        });
    }
    Ok(())
}

/// Validate `a(x_pre) >= 0` and finite (f64).
#[inline]
fn validate_a_x_f64(a_x: f64, x: f64) -> Result<(), SemiflowError> {
    if !a_x.is_finite() || a_x < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "a(x) must be finite and >= 0 (strict ellipticity required for sqrt)",
            value: x,
        });
    }
    Ok(())
}

/// γ-A inner-Strang baseline (f64, uses `f.sample()` = SIMD `catmull_rom` path).
///
/// IDENTICAL to the pre-v0.9.0 implementation to preserve bit-equality.
#[inline]
fn gamma_a_baseline_f64(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    x: f64,
) -> Result<f64, SemiflowError> {
    let s_half = 0.5 * tau;

    let x_pre = x + s_half * dc.call_a_prime(x);
    let a_at_pre = dc.call_a(x_pre);
    validate_a_x_f64(a_at_pre, x_pre)?;

    let h0 = 2.0 * libm::sqrt(a_at_pre * tau);
    let h0_3 = 2.0 * libm::sqrt(3.0 * a_at_pre * tau);

    let center_pos = x_pre + s_half * dc.call_a_prime(x_pre);

    let near_p_raw = x_pre + h0;
    let near_p_pos = near_p_raw + s_half * dc.call_a_prime(near_p_raw);

    let near_neg_raw = x_pre - h0;
    let near_neg_pos = near_neg_raw + s_half * dc.call_a_prime(near_neg_raw);

    let far_p_raw = x_pre + h0_3;
    let far_p_pos = far_p_raw + s_half * dc.call_a_prime(far_p_raw);

    let far_neg_raw = x_pre - h0_3;
    let far_neg_pos = far_neg_raw + s_half * dc.call_a_prime(far_neg_raw);

    let center = W0 * f.sample(center_pos)?;
    let near = W1 * (f.sample(near_p_pos)? + f.sample(near_neg_pos)?);
    let far = W2 * (f.sample(far_p_pos)? + f.sample(far_neg_pos)?);

    Ok(center + near + far)
}

/// ζ-A τ²-correction (f64).
///
/// IDENTICAL to the pre-v0.9.0 implementation to preserve bit-equality.
#[inline]
fn zeta_correction_f64(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    x: f64,
) -> Result<f64, SemiflowError> {
    let delta = libm::sqrt(tau).max(2.0 * dc.grid.dx());

    let f0 = f.sample(x)?;
    let f_pos1 = f.sample(x + delta)?;
    let f_neg1 = f.sample(x - delta)?;
    let f_pos2 = f.sample(x + 2.0 * delta)?;
    let f_neg2 = f.sample(x - 2.0 * delta)?;

    let f1 = (f_pos1 - f_neg1) / (2.0 * delta);
    let f2 = (f_pos1 - 2.0 * f0 + f_neg1) / (delta * delta);
    let f3 = (f_pos2 - 2.0 * f_pos1 + 2.0 * f_neg1 - f_neg2) / (2.0 * delta * delta * delta);

    let a_x = dc.call_a(x);
    let a_prime_x = dc.call_a_prime(x);
    let app_x = dc.call_a_double_prime(x);

    Ok(tau
        * tau
        * (a_x * a_prime_x * f3 + (a_x * app_x / 2.0) * f2 + (a_prime_x * app_x / 4.0) * f1))
}

/// γ-A baseline for constant `a` (f64, D1 fast path, v0.13.0).
///
/// When `a'(x) ≡ 0`, the inner Strang shift `x_pre = x + τ/2·a'(x)` reduces
/// to `x_pre = x`, and all five sampling positions simplify to fixed offsets:
/// `center = x`, `near_± = x ± 2√(a·τ)`, `far_± = x ± 2√(3a·τ)`.
///
/// No `a'` calls are made — only one `call_a` per node to fetch the constant value.
#[inline]
fn gamma_a_const_f64(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    x: f64,
) -> Result<f64, SemiflowError> {
    let a_val = dc.call_a(x);
    validate_a_x_f64(a_val, x)?;

    let h0 = 2.0 * libm::sqrt(a_val * tau);
    let h0_3 = 2.0 * libm::sqrt(3.0 * a_val * tau);

    let center = W0 * f.sample(x)?;
    let near = W1 * (f.sample(x + h0)? + f.sample(x - h0)?);
    let far = W2 * (f.sample(x + h0_3)? + f.sample(x - h0_3)?);

    Ok(center + near + far)
}

/// Apply ζ-A at a single grid node `i` (f64 path).
///
/// For the `ConstA` variant (D1, v0.13.0): skips the S-shift and the ζ-A
/// τ²-correction entirely, as both are identically zero when `a'(x)=0, a''(x)=0`.
#[inline]
fn apply_at_node_f64(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    i: usize,
) -> Result<f64, SemiflowError> {
    let x = dc.grid.x_at(i);
    if dc.is_const_a() {
        // ConstA fast path: no S-shift, no ζ correction.
        gamma_a_const_f64(dc, tau, f, x)
    } else {
        Ok(gamma_a_baseline_f64(dc, tau, f, x)? + zeta_correction_f64(dc, tau, f, x)?)
    }
}

// ---------------------------------------------------------------------------
// Private helpers — generic (scalar path for non-f64 SemiflowFloat types)
// ---------------------------------------------------------------------------

/// Validate `tau`: must be finite and non-negative (generic).
#[inline]
fn validate_tau_generic<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// Validate `a(x_pre) >= 0` and finite (generic).
#[inline]
fn validate_a_x_generic<F: SemiflowFloat>(a_x: F, x: F) -> Result<(), SemiflowError> {
    if !a_x.is_finite() || a_x < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "a(x) must be finite and >= 0 (strict ellipticity required for sqrt)",
            value: x.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// γ-A inner-Strang baseline (generic, uses `sample_generic` = scalar path).
#[inline]
fn gamma_a_baseline_generic<F: SemiflowFloat>(
    dc: &DiffusionChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    x: F,
) -> Result<F, SemiflowError> {
    let s_half = half::<F>() * tau;
    let two = from_f64::<F>(2.0);
    let three = from_f64::<F>(3.0);
    let w0 = from_f64::<F>(W0);
    let w1 = from_f64::<F>(W1);
    let w2 = from_f64::<F>(W2);

    let x_pre = x + s_half * dc.call_a_prime(x);
    let a_at_pre = dc.call_a(x_pre);
    validate_a_x_generic(a_at_pre, x_pre)?;

    let h0 = two * Float::sqrt(a_at_pre * tau);
    let h0_3 = two * Float::sqrt(three * a_at_pre * tau);

    let center_pos = x_pre + s_half * dc.call_a_prime(x_pre);

    let near_p_raw = x_pre + h0;
    let near_p_pos = near_p_raw + s_half * dc.call_a_prime(near_p_raw);

    let near_neg_raw = x_pre - h0;
    let near_neg_pos = near_neg_raw + s_half * dc.call_a_prime(near_neg_raw);

    let far_p_raw = x_pre + h0_3;
    let far_p_pos = far_p_raw + s_half * dc.call_a_prime(far_p_raw);

    let far_neg_raw = x_pre - h0_3;
    let far_neg_pos = far_neg_raw + s_half * dc.call_a_prime(far_neg_raw);

    let center = w0 * f.sample_generic(center_pos)?;
    let near = w1 * (f.sample_generic(near_p_pos)? + f.sample_generic(near_neg_pos)?);
    let far = w2 * (f.sample_generic(far_p_pos)? + f.sample_generic(far_neg_pos)?);

    Ok(center + near + far)
}

/// ζ-A τ²-correction (generic).
#[inline]
fn zeta_correction_generic<F: SemiflowFloat>(
    dc: &DiffusionChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    x: F,
) -> Result<F, SemiflowError> {
    let two = from_f64::<F>(2.0);
    let half_val = half::<F>();
    let quarter = half_val * half_val;
    let delta = Float::max(Float::sqrt(tau), two * dc.grid.dx());

    let f0 = f.sample_generic(x)?;
    let f_pos1 = f.sample_generic(x + delta)?;
    let f_neg1 = f.sample_generic(x - delta)?;
    let f_pos2 = f.sample_generic(x + two * delta)?;
    let f_neg2 = f.sample_generic(x - two * delta)?;

    let f1 = (f_pos1 - f_neg1) / (two * delta);
    let f2 = (f_pos1 - two * f0 + f_neg1) / (delta * delta);
    let f3 = (f_pos2 - two * f_pos1 + two * f_neg1 - f_neg2) / (two * delta * delta * delta);

    let a_x = dc.call_a(x);
    let a_prime_x = dc.call_a_prime(x);
    let app_x = dc.call_a_double_prime(x);

    Ok(tau
        * tau
        * (a_x * a_prime_x * f3
            + (a_x * app_x * half_val) * f2
            + (a_prime_x * app_x * quarter) * f1))
}

/// Apply ζ-A at a single grid node `i` (generic path).
#[inline]
fn apply_at_node_generic<F: SemiflowFloat>(
    dc: &DiffusionChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    i: usize,
) -> Result<F, SemiflowError> {
    let x = dc.grid.x_at(i);
    Ok(gamma_a_baseline_generic(dc, tau, f, x)? + zeta_correction_generic(dc, tau, f, x)?)
}
