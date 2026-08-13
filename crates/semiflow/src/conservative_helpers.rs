// conservative_helpers.rs — included into conservative_assemble.rs (ADR-0187 D1, §56.1).
//
// Provides `harmonic_mean`, `face_transmissibility`, `build_faces`.
// All imports are supplied by the including file; no `use` here.

/// Harmonic mean of two conductivities: `2·k_l·k_r / (k_l + k_r)` (§56.1.a).
///
/// Strictly between `k_l` and `k_r` when both are positive. A face touching a
/// **degenerate** node (`k = 0`) returns `0`: the harmonic mean of a conductor
/// and an insulator is an insulator, which is the physically right answer and
/// the reason degenerate diffusions need no special casing downstream (§56.8,
/// ADR-0191).
///
/// The zero test is explicit rather than left to IEEE arithmetic. For a single
/// zero, `0·k_r/(0+k_r)` would indeed give `0`; but for two adjacent zeros
/// `0·0/(0+0)` is `0/0 = NaN`, and on the `cn_step`/`thomas_solve` path there is
/// no finiteness backstop to catch it. Branching costs one comparison and makes
/// adjacent zeros — the Wright–Fisher case, degenerate at both ends — as safe as
/// isolated ones.
#[inline]
fn harmonic_mean<F: SemiflowFloat>(k_l: F, k_r: F) -> F {
    if k_l <= F::zero() || k_r <= F::zero() {
        return F::zero();
    }
    let two = F::from(2.0_f64).unwrap_or_else(|| F::one() + F::one());
    two * k_l * k_r / (k_l + k_r)
}

/// Face transmissibility: `T = 1 / (dx/k_harm + R_c)` (§56.1.b).
///
/// `R_c ≥ 0` — perfect contact if `R_c = 0`. A degenerate face (`k_harm = 0`)
/// carries **zero** flux: the resistance `dx/k_harm` is infinite, so `T = 0`
/// regardless of `R_c` (§56.8, ADR-0191). Returned directly rather than via
/// `1/∞` so the result is exact and independent of the IEEE division mode.
#[inline]
fn face_transmissibility<F: SemiflowFloat>(k_harm: F, dx: F, r_c: F) -> F {
    if k_harm <= F::zero() {
        return F::zero();
    }
    F::one() / (dx / k_harm + r_c)
}

/// Validate `k_nodes` (non-negative finite) and build face transmissibilities `T_{i+½}`.
///
/// # Errors
///
/// [`SemiflowError::DomainViolation`] if:
/// - `k_nodes.len() < 2`
/// - any `k_i < 0` or non-finite
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
    validate_k_nonneg(k_nodes)?;
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

/// Validate all `k_i ≥ 0` and finite (§56.8, ADR-0191).
///
/// `k = 0` is admitted: degenerate-at-the-boundary diffusions are the norm, not
/// an edge case — CEV `k(S) = ½σ²S^{2β}` vanishes at `S = 0`, Feller/CIR
/// `k(v) = ½ξ²v` vanishes at `v = 0`, Wright–Fisher vanishes at both ends. The
/// harmonic-mean face gives such a node exactly zero conductivity, so the
/// boundary classifies itself and no flux crosses it.
///
/// `k < 0` stays rejected, and that is not symmetry with the old rule: a
/// negative conductivity produces negative transmissibility, which breaks the
/// §56.2 energy argument `⟨u, L_k u⟩ = −Σ T(u_{i+1}−u_i)² ≤ 0` and silently
/// destroys the PSD property the Krylov contraction bound depends on. None of
/// `from_csr`'s three checks would catch it.
fn validate_k_nonneg<F: SemiflowFloat>(k_nodes: &[F]) -> Result<(), SemiflowError> {
    for &k in k_nodes {
        if !k.is_finite() || k < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "conservative: k_nodes must be non-negative and finite",
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
