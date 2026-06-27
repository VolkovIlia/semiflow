// conservative_helpers.rs — included into conservative_assemble.rs (ADR-0187 D1, §56.1).
//
// Provides `harmonic_mean`, `face_transmissibility`, `build_faces`.
// All imports are supplied by the including file; no `use` here.

/// Harmonic mean of two conductivities: `2·k_l·k_r / (k_l + k_r)` (§56.1.a).
///
/// Strictly between `k_l` and `k_r` (both positive). Denominator is positive
/// because `k_l, k_r > 0` (validated by `build_faces` before calling this).
#[inline]
fn harmonic_mean<F: SemiflowFloat>(k_l: F, k_r: F) -> F {
    let two = F::from(2.0_f64).unwrap_or_else(|| F::one() + F::one());
    two * k_l * k_r / (k_l + k_r)
}

/// Face transmissibility: `T = 1 / (dx/k_harm + R_c)` (§56.1.b).
///
/// `R_c ≥ 0` — perfect contact if `R_c = 0`. Both `dx` and `k_harm` are > 0.
#[inline]
fn face_transmissibility<F: SemiflowFloat>(k_harm: F, dx: F, r_c: F) -> F {
    F::one() / (dx / k_harm + r_c)
}

/// Validate `k_nodes` (positive finite) and build face transmissibilities `T_{i+½}`.
///
/// # Errors
///
/// [`SemiflowError::DomainViolation`] if:
/// - `k_nodes.len() < 2`
/// - any `k_i ≤ 0` or non-finite
/// - `r_contact` supplied but `r_contact.len() != k_nodes.len() - 1`
/// - any `R_c < 0` or non-finite
///
/// # Panics
///
/// Never panics (all branches return `Err` on bad inputs).
pub(crate) fn build_faces<F: SemiflowFloat>(
    k_nodes: &[F],
    dx: F,
    r_contact: Option<&[F]>,
) -> Result<alloc::vec::Vec<F>, SemiflowError> {
    let n = k_nodes.len();
    if n < 2 {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "conservative: k_nodes.len() must be >= 2",
            value: n as f64,
        });
    }
    validate_k_positive(k_nodes)?;
    validate_r_contact(r_contact, n)?;
    let faces = (0..n - 1)
        .map(|i| {
            let k_harm = harmonic_mean(k_nodes[i], k_nodes[i + 1]);
            let r_c = r_contact.map_or(F::zero(), |rc| rc[i]);
            face_transmissibility(k_harm, dx, r_c)
        })
        .collect();
    Ok(faces)
}

/// Validate all `k_i > 0` and finite.
fn validate_k_positive<F: SemiflowFloat>(k_nodes: &[F]) -> Result<(), SemiflowError> {
    for &k in k_nodes {
        if !k.is_finite() || k <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "conservative: k_nodes must be strictly positive and finite",
                value: k.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

/// Validate optional contact-resistance slice (finite, non-negative, correct length).
fn validate_r_contact<F: SemiflowFloat>(
    r_contact: Option<&[F]>,
    n_nodes: usize,
) -> Result<(), SemiflowError> {
    let Some(rc) = r_contact else { return Ok(()) };
    if rc.len() != n_nodes - 1 {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "conservative: r_contact.len() must equal k_nodes.len() - 1",
            value: rc.len() as f64,
        });
    }
    for &r in rc {
        if !r.is_finite() || r < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "conservative: r_contact entries must be finite and >= 0",
                value: r.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}
