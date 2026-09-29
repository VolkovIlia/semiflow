//! Chebyshev boundary dispatch, Schrödinger unitarity and `expmv` accuracy.

use alloc::vec::Vec;

use semiflow::{BoundaryPolicy, InterpKind, OobPolicy};
use semiflow::{
    ChernoffFunction, Diffusion4thChernoff, DiffusionExpmvChernoff, Grid1D, GridFn1D,
    SchrodingerChernoff, SchrodingerState, ScratchPool,
};

use crate::{check, digest_of, m, sup_diff, Digest, Failure, ScenarioResult};

/// Chebyshev grid of `tests/grid_chebyshev_bc_dispatch.rs`: 64 nodes on
/// `[-1, 1]`, `m = 16`, `OobPolicy::Inherit`, datum `exp(-x²)`.
fn cheb_grid(bc: BoundaryPolicy) -> Result<(Grid1D, Vec<f64>), Failure> {
    let grid = Grid1D::new(-1.0, 1.0, 64)?.with_boundary(bc).with_interp(
        InterpKind::ChebyshevSpectralWithBC {
            m: 16,
            oob_policy: OobPolicy::Inherit,
        },
    );
    let values = (0..grid.n)
        .map(|i| m::exp(-grid.x_at(i) * grid.x_at(i)))
        .collect();
    Ok((grid, values))
}

/// Compare out-of-domain queries with their in-domain images.
///
/// Returns the largest `|interp(x_out) − interp(x_in)|` over the pairs; the
/// in-domain value must also be within `1e-3` of `exp(-x_in²)` (the `m = 16`
/// interpolant on this grid is accurate to about `3.4e-4`), which rules out
/// a degenerate interpolant that would make the mirror identity vacuous.
fn mirror_err(bc: BoundaryPolicy, pairs: &[(f64, f64)]) -> ScenarioResult {
    let (grid, values) = cheb_grid(bc)?;
    let (mut mirror, mut accuracy): (f64, f64) = (0.0, 0.0);
    let mut digest = Digest::new();
    for &(x_out, x_in) in pairs {
        let out = grid.interp(&values, x_out)?;
        let inside = grid.interp(&values, x_in)?;
        digest.extend(&[out, inside]);
        if !out.is_finite() || !inside.is_finite() {
            return check(f64::NAN, 0.0, digest.finish());
        }
        mirror = mirror.max(m::abs(out - inside));
        accuracy = accuracy.max(m::abs(inside - m::exp(-x_in * x_in)));
    }
    let digest = digest.finish();
    check(accuracy, 1e-3, digest)?;
    check(mirror, 1e-12, digest)
}

/// Out-of-domain Chebyshev query with `Reflect` equals the mirrored query
/// (exercises the `no_std` `rem_euclid` replacement in `grid_chebyshev.rs`).
///
/// `tests/grid_chebyshev_bc_dispatch.rs` only checks finiteness and range at
/// `±1.5`; the oracle here is the mirror image itself: `x ↦ 2 − x` past
/// `x = 1`, `x ↦ −2 − x` past `x = −1`, period 4. The fold is exact for these
/// points up to rounding of `x − xmin`, hence the `1e-12` tolerance.
pub(crate) fn chebyshev_reflect() -> ScenarioResult {
    let pairs = [
        (1.5, 0.5),
        (-1.5, -0.5),
        (3.3, -0.7),
        (-2.7, 0.7),
        (1.25, 0.75),
    ];
    mirror_err(BoundaryPolicy::Reflect, &pairs)
}

/// Out-of-domain Chebyshev query with `Periodic` equals the wrapped query
/// (period 2); same source test and tolerances as [`chebyshev_reflect`].
pub(crate) fn chebyshev_periodic() -> ScenarioResult {
    let pairs = [
        (1.5, -0.5),
        (-1.5, 0.5),
        (3.3, -0.7),
        (-2.7, -0.7),
        (1.25, -0.75),
    ];
    mirror_err(BoundaryPolicy::Periodic, &pairs)
}

/// `SchrodingerChernoff` norm conservation (exercises `sin`/`cos` in
/// `schrodinger_helpers.rs`).
///
/// `tests/g18_schrodinger_unitarity.rs`, G18a: 64 nodes on `[-5, 5]`,
/// `V = ½x²`, wavepacket `exp(-(x−1)²/(2·0.5²))`, `τ = 0.01`, 100 steps,
/// `|‖ψ(1)‖² − ‖ψ₀‖²| < 1e-12`.
pub(crate) fn schrodinger_unitarity() -> ScenarioResult {
    let grid = Grid1D::new(-5.0, 5.0, 64)?;
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let schr = SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x)?;
    let psi_re = GridFn1D::from_fn(grid, |x| m::exp(-(x - 1.0) * (x - 1.0) / 0.5));
    let psi_im = GridFn1D::from_fn(grid, |_| 0.0);
    let psi0 = SchrodingerState::new(psi_re, psi_im)?;
    let (mut cur, mut nxt) = (psi0.clone(), psi0.clone());
    let mut pool = ScratchPool::new();
    for _ in 0..100 {
        schr.apply_into(0.01, &cur, &mut nxt, &mut pool)?;
        core::mem::swap(&mut cur, &mut nxt);
    }
    // Guard against a trivial pass: the phase must actually have rotated.
    let im_max = cur
        .psi_im
        .values
        .iter()
        .fold(0.0_f64, |a, &v| a.max(m::abs(v)));
    if im_max.is_nan() || im_max < 1e-3 {
        return Err(Failure::Invariant("Schrödinger evolution left ψ real"));
    }
    let mut digest = Digest::new();
    for (&re, &im) in cur.psi_re.values.iter().zip(&cur.psi_im.values) {
        digest.extend(&[re, im]);
    }
    check(
        m::abs(cur.norm_l2_sq() - psi0.norm_l2_sq()),
        1e-12,
        digest.finish(),
    )
}

/// `a(x) = 1 + 0.3 sin(2πx/20)` — `tests/expmv_div_form_action_accuracy.rs`.
fn a_fn(x: f64) -> f64 {
    1.0 + 0.3 * m::sin(2.0 * core::f64::consts::PI * x / 20.0)
}

/// Divergence-form `A f = ∂_x(a ∂_x f)` with Neumann ends (copied from the
/// source test, which duplicates the crate-private stencil).
fn apply_av(grid: Grid1D, f: &[f64], out: &mut [f64]) {
    let n = f.len();
    let dx = grid.dx();
    for i in 0..n {
        let xi = grid.x_at(i);
        let fp = if i + 1 < n { f[i + 1] } else { f[n - 1] };
        let fm = if i > 0 { f[i - 1] } else { f[0] };
        let (ap, am) = (a_fn(xi + 0.5 * dx), a_fn(xi - 0.5 * dx));
        out[i] = (ap * (fp - f[i]) - am * (f[i] - fm)) / (dx * dx);
    }
}

/// Reference `exp(τA) f`: `s` steps of the degree-18 Taylor polynomial with
/// per-step argument `≤ 1` (the source test's self-converged reference).
fn reference_expmv(grid: Grid1D, tau: f64, norm_a: f64, f: &[f64]) -> Vec<f64> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // ≈ 40
    let s = (tau * norm_a).max(1.0) as u32 + 1;
    let tau_s = tau / f64::from(s);
    let mut y = f.to_vec();
    let mut w = alloc::vec![0.0; f.len()];
    let mut av = alloc::vec![0.0; f.len()];
    for _ in 0..s {
        w.copy_from_slice(&y);
        for k in 1..=18_u32 {
            apply_av(grid, &w, &mut av);
            for ((wi, yi), &a) in w.iter_mut().zip(y.iter_mut()).zip(&av) {
                *wi = tau_s / f64::from(k) * a;
                *yi += *wi;
            }
        }
    }
    y
}

/// `DiffusionExpmvChernoff` (Al-Mohy–Higham `expmv`) at `τ‖A‖ ≈ 40`.
///
/// `tests/expmv_div_form_action_accuracy.rs` (ADR-0121 gate): 64 nodes on
/// `[0, 20]`, datum `exp(-(x−10)²)`, reference Taylor-18 with per-step
/// argument ≤ 1, `sup_error ≤ 1e-11`.
pub(crate) fn expmv_div_form() -> ScenarioResult {
    let grid = Grid1D::new(0.0, 20.0, 64)?;
    let inner = Diffusion4thChernoff::new(a_fn, |_| 0.0, |_| 0.0, 1.3, grid);
    let dx = grid.dx();
    let norm_a = 4.0 * inner.a_norm_bound / (dx * dx);
    let tau = 40.0 / norm_a;
    let f0 = GridFn1D::from_fn(grid, |x| m::exp(-(x - 10.0) * (x - 10.0)));
    let reference = reference_expmv(grid, tau, norm_a, &f0.values);
    let kernel = DiffusionExpmvChernoff::new(inner);
    let mut u = GridFn1D::from_fn(grid, |_| 0.0);
    kernel.apply_into(tau, &f0, &mut u, &mut ScratchPool::new())?;
    check(sup_diff(&u.values, &reference), 1e-11, digest_of(&u.values))
}
