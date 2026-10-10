//! ETDRK4 step assembly (ADR-0189 §D2, Cox–Matthews 2002; ADR-0205).
//!
//! ## Stage formula
//!
//! ```text
//! a       = e^{hL/2} u + (h/2) φ₁(hL/2) N(u)
//! b       = e^{hL/2} u + (h/2) φ₁(hL/2) N(a)
//! c       = e^{hL/2} a + (h/2) φ₁(hL/2) (2 N(b) − N(u))
//! u_{n+1} = e^{hL}   u
//!         + h (φ₁ − 3φ₂ + 4φ₃)(hL) N(u)
//!         + h (2φ₂  − 4φ₃)(hL)      (N(a) + N(b))
//!         + h (4φ₃  −  φ₂)(hL)      N(c)
//! ```
//!
//! Equivalently (linearity of `φ_k`), with
//! `combo2 = −3N(u) + 2N(a) + 2N(b) − N(c)`, `combo3 = 4N(u) − 4N(a) − 4N(b) + 4N(c)`:
//! `u_{n+1} = φ₀(hL)u + h(φ₁(hL)N(u) + φ₂(hL)combo2 + φ₃(hL)combo3)`.
//!
//! ## One sweep per stage
//!
//! Every line is one combination `Σ_k τᵏ φ_k(τL) w_k`, which
//! [`phi_combination`] evaluates in ONE augmented Horner sweep: `a`, `b`, `c` with
//! `τ = h/2` and `w = [·, ·]`, the update with `τ = h` and
//! `w = [u, N(u), combo2/h, combo3/h²]`. Four sweeps per step; the former
//! one-φ-per-call assembly needed nine (ADR-0205, gate `G_ETDRK4_SWEEPS`).

use crate::{
    error::SemiflowError,
    float::{from_f64, half, two, SemiflowFloat},
    generator_action::GeneratorAction,
    nonlinearity::Nonlinearity,
    phi_action::phi_combination,
    scratch::ScratchPool,
};

/// `combo2/h` and `combo3/h²` of the update (see the module notes).
#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
fn update_columns<F: SemiflowFloat>(
    h: F,
    n_u: &[F],
    n_a: &[F],
    n_b: &[F],
    n_c: &[F],
    combo2: &mut [F],
    combo3: &mut [F],
) {
    let (tw, three, four) = (two::<F>(), from_f64::<F>(3.0), from_f64::<F>(4.0));
    let (inv_h, inv_h2) = (F::one() / h, F::one() / (h * h));
    for i in 0..n_u.len() {
        let nab = n_a[i] + n_b[i];
        combo2[i] = (-three * n_u[i] + tw * nab - n_c[i]) * inv_h;
        combo3[i] = (four * (n_u[i] + n_c[i]) - four * nab) * inv_h2;
    }
}

/// Execute one ETDRK4 step: `u → u_next` (Cox–Matthews 2002), four sweeps.
///
/// All temporary buffers are taken from `scratch` and returned before this
/// function exits — no allocation occurs if the pool already has capacity.
#[allow(clippy::many_single_char_names)]
pub(crate) fn etdrk4_step<F, Op, Nl>(
    op: &Op,
    nl: &Nl,
    h: F,
    u: &[F],
    u_next: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Op: GeneratorAction<F>,
    Nl: Nonlinearity<F>,
{
    let n = u.len();
    let hh = h * half::<F>();
    let mut bufs: [_; 8] = core::array::from_fn(|_| scratch.take_vec(n));
    let result = (|| {
        let [n_u, a, n_a, b, n_b, c, n_c, w] = &mut bufs;
        nl.eval(u, n_u)?;
        phi_combination(op, hh, &[u, n_u], a, scratch)?;
        nl.eval(a, n_a)?;
        phi_combination(op, hh, &[u, n_a], b, scratch)?;
        nl.eval(b, n_b)?;
        for i in 0..n {
            w[i] = two::<F>() * n_b[i] - n_u[i];
        }
        phi_combination(op, hh, &[a, w], c, scratch)?;
        nl.eval(c, n_c)?;
        // `a` and `b` are no longer needed: reuse them for the update columns.
        update_columns(h, n_u, n_a, n_b, n_c, a, b);
        phi_combination(op, h, &[u, n_u, a, b], u_next, scratch)
    })();
    for buf in bufs {
        scratch.return_vec(buf);
    }
    result
}
