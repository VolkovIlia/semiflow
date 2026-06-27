//! ETDRK4 stage-assembly helpers (ADR-0189 §D2, Cox–Matthews 2002).
//!
//! Each helper computes one stage of the four-stage method; scratch buffers
//! are borrowed from the caller's [`ScratchPool`] and returned before the
//! function exits (take-return discipline, no allocation on the fast path).
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
//! Equivalently (using linearity of `φ_k`):
//!
//! - combo2 = −3 N(u) + 2 N(a) + 2 N(b) − N(c)
//! - combo3 =  4 N(u) − 4 N(a) − 4 N(b) + 4 N(c)
//! - u_{n+1} = φ₀(hL) u + h (φ₁(hL) N(u) + φ₂(hL) combo2 + φ₃(hL) combo3)

use crate::{
    error::SemiflowError,
    float::{from_f64, half, SemiflowFloat, two},
    generator_action::GeneratorAction,
    nonlinearity::Nonlinearity,
    phi_action::phi_action,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Stage A
// ---------------------------------------------------------------------------

/// Compute stage A: `a = e^{hL/2}·u + (h/2)·φ₁(hL/2)·N(u)`.
///
/// Side effects: fills `n_u` with `N(u)` and `e_half_u` with `e^{hL/2}·u`.
/// Both are reused in subsequent stages.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_a<F, Op, Nl>(
    op: &Op,
    nl: &Nl,
    h: F,
    u: &[F],
    a: &mut [F],
    n_u: &mut [F],
    e_half_u: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Op: GeneratorAction<F>,
    Nl: Nonlinearity<F>,
{
    let n = u.len();
    let hh = h * half::<F>();

    nl.eval(u, n_u)?;
    phi_action(op, 0, hh, u, e_half_u, scratch)?;

    let mut phi1_n_u = scratch.take_vec(n);
    phi_action(op, 1, hh, n_u, &mut phi1_n_u, scratch)?;
    let half_h = hh; // h/2 already in hh
    for i in 0..n {
        a[i] = e_half_u[i] + half_h * phi1_n_u[i];
    }
    scratch.return_vec(phi1_n_u);
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage B
// ---------------------------------------------------------------------------

/// Compute stage B: `b = e^{hL/2}·u + (h/2)·φ₁(hL/2)·N(a)`.
///
/// Uses pre-computed `e_half_u` (from stage A). Fills `n_a` with `N(a)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_b<F, Op, Nl>(
    op: &Op,
    nl: &Nl,
    h: F,
    e_half_u: &[F],
    a: &[F],
    b: &mut [F],
    n_a: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Op: GeneratorAction<F>,
    Nl: Nonlinearity<F>,
{
    let n = a.len();
    let hh = h * half::<F>();

    nl.eval(a, n_a)?;

    let mut phi1_n_a = scratch.take_vec(n);
    phi_action(op, 1, hh, n_a, &mut phi1_n_a, scratch)?;
    for i in 0..n {
        b[i] = e_half_u[i] + hh * phi1_n_a[i];
    }
    scratch.return_vec(phi1_n_a);
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage C
// ---------------------------------------------------------------------------

/// Compute stage C: `c = e^{hL/2}·a + (h/2)·φ₁(hL/2)·(2·N(b) − N(u))`.
///
/// Fills `n_b` with `N(b)`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
pub(crate) fn stage_c<F, Op, Nl>(
    op: &Op,
    nl: &Nl,
    h: F,
    n_u: &[F],
    a: &[F],
    b: &[F],
    c: &mut [F],
    n_b: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Op: GeneratorAction<F>,
    Nl: Nonlinearity<F>,
{
    let n = a.len();
    let hh = h * half::<F>();
    let tw = two::<F>();

    nl.eval(b, n_b)?;

    let mut e_half_a = scratch.take_vec(n);
    phi_action(op, 0, hh, a, &mut e_half_a, scratch)?;

    let mut combo = scratch.take_vec(n);
    for i in 0..n { combo[i] = tw * n_b[i] - n_u[i]; }

    let mut phi1_combo = scratch.take_vec(n);
    phi_action(op, 1, hh, &combo, &mut phi1_combo, scratch)?;

    for i in 0..n { c[i] = e_half_a[i] + hh * phi1_combo[i]; }

    scratch.return_vec(phi1_combo);
    scratch.return_vec(combo);
    scratch.return_vec(e_half_a);
    Ok(())
}

// ---------------------------------------------------------------------------
// Final update
// ---------------------------------------------------------------------------

// combo2 = −3 N(u) + 2(N(a)+N(b)) − N(c)
// combo3 =  4(N(u) + N(c)) − 4(N(a)+N(b))
fn build_combos<F: SemiflowFloat>(
    n_u: &[F], n_a: &[F], n_b: &[F], n_c: &[F],
    combo2: &mut [F], combo3: &mut [F],
) {
    let tw = two::<F>();
    let three = from_f64::<F>(3.0);
    let four = from_f64::<F>(4.0);
    for i in 0..n_u.len() {
        let nab = n_a[i] + n_b[i];
        combo2[i] = -three * n_u[i] + tw * nab - n_c[i];
        combo3[i] = four * (n_u[i] + n_c[i]) - four * nab;
    }
}

/// Assemble `u_{n+1}` from the four stage evaluations.
///
/// Requires N evaluated at all four stages (u, a, b, c); calls `nl.eval(c, …)` internally.
#[allow(clippy::too_many_arguments)]
pub(crate) fn final_update<F, Op, Nl>(
    op: &Op,
    nl: &Nl,
    h: F,
    u: &[F],
    n_u: &[F],
    n_a: &[F],
    n_b: &[F],
    c: &[F],
    u_next: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Op: GeneratorAction<F>,
    Nl: Nonlinearity<F>,
{
    let n = u.len();
    let mut n_c = scratch.take_vec(n);
    nl.eval(c, &mut n_c)?;

    let mut e_un = scratch.take_vec(n);
    phi_action(op, 0, h, u, &mut e_un, scratch)?;

    let mut combo2 = scratch.take_vec(n);
    let mut combo3 = scratch.take_vec(n);
    build_combos(n_u, n_a, n_b, &n_c, &mut combo2, &mut combo3);

    let mut phi1_nu = scratch.take_vec(n);
    let mut phi2_c2 = scratch.take_vec(n);
    let mut phi3_c3 = scratch.take_vec(n);
    phi_action(op, 1, h, n_u, &mut phi1_nu, scratch)?;
    phi_action(op, 2, h, &combo2, &mut phi2_c2, scratch)?;
    phi_action(op, 3, h, &combo3, &mut phi3_c3, scratch)?;

    for i in 0..n {
        u_next[i] = e_un[i] + h * (phi1_nu[i] + phi2_c2[i] + phi3_c3[i]);
    }

    scratch.return_vec(phi3_c3); scratch.return_vec(phi2_c2); scratch.return_vec(phi1_nu);
    scratch.return_vec(combo3); scratch.return_vec(combo2);
    scratch.return_vec(e_un); scratch.return_vec(n_c);
    Ok(())
}

// ---------------------------------------------------------------------------
// One ETDRK4 step (entry point called by Etdrk4::step)
// ---------------------------------------------------------------------------

/// Execute one ETDRK4 step: `u → u_next` (Cox–Matthews 2002).
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

    // Persistent stage buffers: survive across multiple helper calls.
    let mut n_u = scratch.take_vec(n);
    let mut e_half_u = scratch.take_vec(n);
    let mut a = scratch.take_vec(n);
    let mut n_a = scratch.take_vec(n);
    let mut b = scratch.take_vec(n);
    let mut n_b = scratch.take_vec(n);
    let mut c = scratch.take_vec(n);

    stage_a(op, nl, h, u, &mut a, &mut n_u, &mut e_half_u, scratch)?;
    stage_b(op, nl, h, &e_half_u, &a, &mut b, &mut n_a, scratch)?;
    stage_c(op, nl, h, &n_u, &a, &b, &mut c, &mut n_b, scratch)?;
    final_update(op, nl, h, u, &n_u, &n_a, &n_b, &c, u_next, scratch)?;

    // Return in reverse take order (LIFO is correct but any order is safe).
    scratch.return_vec(c);
    scratch.return_vec(n_b);
    scratch.return_vec(b);
    scratch.return_vec(n_a);
    scratch.return_vec(a);
    scratch.return_vec(e_half_u);
    scratch.return_vec(n_u);
    Ok(())
}
