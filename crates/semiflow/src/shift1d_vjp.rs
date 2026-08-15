//! Gradients w.r.t. `ShiftChernoff1D`'s coefficient **fields** (ADR-0197, Issue #25).
//!
//! `EvolverHeat1DGreeksV3` differentiates w.r.t. a single global diffusion scale
//! θ of the unit-heat kernel — one model (Bachelier, via `θ = σ_N²/2`). What
//! calibration workflows need is `∂J/∂a_i` for the per-node coefficient arrays
//! of `Shift1D::with_arrays`: local-vol Vega surfaces, `∂V/∂σ(S_i)`.
//!
//! Design mirrors `edge_weight_grad` (ADR-0115, §43): the caller supplies the
//! cotangent `∂J/∂u_n`, the loss stays **outside** the library, and the gradient
//! buffer is caller-owned, zeroed and length-checked here.
//!
//! ## The two structural facts (§61.2)
//!
//! **Output coupling is diagonal.** The coefficients are evaluated only at
//! `x = grid.x_at(i)`, so `∂(S u)_p/∂θ_i = 0` for `p ≠ i`. All `n` parameter
//! derivatives for a field therefore come from **one** `O(n)` pass, not `n`
//! passes. This is why `GeneratorSensitivity` (which computes one parameter at a
//! time, `graph_sensitivity.rs:38-54`) is deliberately **not** implemented here:
//! it would advertise an `O(n_steps · n²)` path when an `O(n_steps · n)` one
//! exists — `10⁹` stencil evaluations against `10⁶` at `n = 1024`.
//!
//! **Input coupling is wide, and the coefficient sits inside the foot.** Writing
//! `h_i = 2√(a_i τ)` and `g_i = 2 b_i τ`,
//!
//! ```text
//!   (S(τ)u)_i = ¼·ũ(x_i + h_i) + ¼·ũ(x_i − h_i) + ½·ũ(x_i + g_i) + τ·c_i·u_i
//! ```
//!
//! where `ũ` is the *interpolant* of `u`. With `∂h_i/∂a_i = √(τ/a_i)`:
//!
//! ```text
//!   ∂(Su)_i/∂a_i = ¼·√(τ/a_i)·[ ũ′(x_i + h_i) − ũ′(x_i − h_i) ]
//!   ∂(Su)_i/∂b_i = τ · ũ′(x_i + g_i)
//!   ∂(Su)_i/∂c_i = τ · u_i
//! ```
//!
//! The `√(τ/a_i)` factor diverges as `a_i → 0⁺`: the `a` gradient is **undefined**
//! at a degenerate node, so this module requires `a_i > 0` strictly even though
//! the forward kernel admits `a_i ≥ 0`. The gradient's domain is strictly smaller
//! than the forward one.
//!
//! ## Weight rows by probing, not by hand-transposition (§61.3)
//!
//! The adjoint needs the interpolation weight row `w(y)` with
//! `ũ(y) = Σ_j w_j(y)·u_j`. Deriving those rows by hand for `SepticHermite` —
//! eight Birkhoff–Garabedian–Lorentz polynomials composed with three central-FD
//! stencils, folded through seven boundary policies — is the single most
//! error-prone thing this feature could contain, and a sign slip in one policy
//! would produce plausible-but-wrong gradients.
//!
//! Instead the rows are **measured**: since the interpolant is linear in the node
//! values, `w_j(y) = sample(e_j, y)`. Probing the real sampler on its compact
//! support costs `O(1)` calls per foot and is correct by construction for every
//! `InterpKind` and every `BoundaryPolicy`, including ones added later. The
//! support assumption is not taken on faith — `weight_row` is gated against
//! `Σ_j w_j·u_j == sample(u, y)` on random data.
//!
//! The probing uses `Grid1D::interp` — the **f64 SIMD** entry point the forward
//! kernel itself uses — not the scalar `interp_generic`. That is a correctness
//! requirement (the gradient must differentiate the sampler the solve actually
//! runs, and the two can differ at ULP level) and it is also why this module is
//! f64-monomorphic: instantiating `interp_generic` at `f64` from a new call site
//! made every septic/octonic/Chebyshev generic sampler newly reachable there,
//! and under `codegen-units = 1` + LTO that perturbed inlining enough to slow
//! the 1-D pre-sampled path by ~70% (`test_path2_faster_than_path1`, 2.5x -> 1.9x).

extern crate alloc;

use alloc::{vec, vec::Vec};

use crate::{error::SemiflowError, float::SemiflowFloat, grid::Grid1D};

/// Candidate support half-width probed around the containing cell.
///
/// Septic-Hermite reaches `idx−3 … idx+4` through its FD derivative stencils;
/// Octonic reaches one further. 6 covers both with margin. The three nodes at
/// each end are probed as well, because `LinearExtrapolate` folds out-of-range
/// requests onto them.
const PROBE_HALF_WIDTH: i64 = 6;

/// Maximum number of nodes a single weight row can touch.
///
/// `PROBE_HALF_WIDTH` is a positive literal, so the cast is exact on every
/// target — the lint fires on the `i64 → usize` *form*, not on this value.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) const MAX_ROW: usize = 2 * PROBE_HALF_WIDTH as usize + 2 + 6;

/// Nodes the sampler can touch when evaluating at `y`.
///
/// Each raw stencil index is resolved through the SAME boundary policy the
/// sampler uses, rather than assuming the touched nodes sit near the containing
/// cell. Under `Reflect` or `Periodic` a query several cells outside the domain
/// folds to interior nodes arbitrarily far from the raw index — a raw-index
/// window silently misses them, which is exactly what
/// `G_SHIFT1D_WEIGHTS_ORACLE` falsified on the first attempt.
fn candidate_nodes(grid: &Grid1D<f64>, y: f64, idx_out: &mut [usize; MAX_ROW]) -> usize {
    let n = grid.n;
    #[allow(clippy::cast_possible_truncation)]
    let centre = ((y - grid.xmin) / grid.dx()).floor() as i64;
    let mut count = 0_usize;
    let push = |j: usize, count: &mut usize, idx_out: &mut [usize; MAX_ROW]| {
        if !idx_out[..*count].contains(&j) && *count < MAX_ROW {
            idx_out[*count] = j;
            *count += 1;
        }
    };
    for d in -PROBE_HALF_WIDTH..=PROBE_HALF_WIDTH + 1 {
        match crate::boundary::bc_index::<f64>(grid.boundary, n, centre + d) {
            crate::boundary::BoundaryHit::Inside(j) => push(j, &mut count, idx_out),
            crate::boundary::BoundaryHit::RobinSkew { reflected, .. }
            | crate::boundary::BoundaryHit::OddReflected { reflected } => {
                push(reflected, &mut count, idx_out);
            }
            // `Zero`/`Dirichlet` contribute no node; `OutsideLeft/Right` are
            // affine in the three end nodes, pushed below.
            _ => {}
        }
    }
    // `LinearExtrapolate` folds onto the three nodes at each end.
    for j in [0_usize, 1, 2] {
        if j < n {
            push(j, &mut count, idx_out);
        }
    }
    for k in 1..=3_usize {
        if n >= k {
            push(n - k, &mut count, idx_out);
        }
    }
    count
}

/// Interpolation weight row at `y`: `ũ(y) = Σ w[k]·values[idx[k]]`.
///
/// Returns the number of entries written. Correct for any interpolant linear in
/// the node values, because it measures the sampler rather than reproducing it.
///
/// `probe` is a caller-owned scratch buffer of length `grid.n`, left all-zero on
/// return so it can be reused across calls without reallocation.
///
/// # Errors
/// Propagates `Unsupported` from the grid's interpolation kind.
pub(crate) fn weight_row(
    grid: &Grid1D<f64>,
    y: f64,
    probe: &mut [f64],
    idx_out: &mut [usize; MAX_ROW],
    w_out: &mut [f64; MAX_ROW],
) -> Result<usize, SemiflowError> {
    let count = candidate_nodes(grid, y, idx_out);
    for slot in 0..count {
        let j = idx_out[slot];
        probe[j] = 1.0;
        w_out[slot] = grid.interp(probe, y)?;
        probe[j] = 0.0;
    }
    Ok(count)
}

/// `ũ′(y)` — derivative of the interpolant with respect to position.
///
/// Central difference with `δ = dx·10⁻⁴`. The interpolant is piecewise
/// polynomial and smooth inside a cell, so the truncation term is
/// `O(δ²·ũ‴) ≈ 10⁻⁸·dx²·ũ‴` while the roundoff term is `O(ε·|u|/δ) ≈ 10⁻¹²`;
/// both sit an order or more below the `10⁻⁶` band the FD gate uses. A δ that
/// straddles a cell boundary sees a `C¹` (Catmull-Rom) or `C³` (septic) join, so
/// the one-sided limits agree to at least first order and the estimate degrades
/// gracefully rather than jumping.
///
/// # Errors
/// Propagates `Unsupported` from the grid's interpolation kind.
pub(crate) fn sample_deriv(
    values: &[f64],
    grid: &Grid1D<f64>,
    y: f64,
) -> Result<f64, SemiflowError> {
    let delta = grid.dx() * 1.0e-4_f64;
    let plus = grid.interp(values, y + delta)?;
    let minus = grid.interp(values, y - delta)?;
    Ok((plus - minus) / (delta + delta))
}

/// Which coefficient field a gradient is taken with respect to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShiftCoeffField {
    /// Diffusion `a(x)`. Requires `a_i > 0` strictly (see the module note).
    A,
    /// Drift `b(x)`.
    B,
    /// Reaction `c(x)`.
    C,
}

/// Per-node coefficient arrays for a `ShiftChernoff1D`-shaped generator.
///
/// Carried directly rather than re-read through the kernel's closures, so the
/// diagonality of `∂a(x_i)/∂a_k = δ_ik` is exact by construction instead of
/// resting on the coefficient interpolant reproducing node values to the last
/// ULP.
pub struct ShiftCoeffs<'a, F: SemiflowFloat = f64> {
    /// Diffusion at each node.
    pub a: &'a [F],
    /// Drift at each node.
    pub b: &'a [F],
    /// Reaction at each node.
    pub c: &'a [F],
}

/// The primal problem a gradient is taken *about*: kernel, coefficient fields,
/// and the uniform stepping it is iterated with.
///
/// Grouped into one value rather than passed as four positional arguments
/// because every internal helper threads exactly this tuple, and because the
/// gradient entry point otherwise reaches eight parameters — over the suckless
/// argument budget, and long enough that positional call sites become hard to
/// read.
pub struct Shift1DProblem<'a> {
    /// Spatial grid — carries the `InterpKind` and `BoundaryPolicy` the
    /// gradient differentiates through.
    pub grid: &'a Grid1D<f64>,
    /// Per-node coefficient fields.
    pub coeffs: ShiftCoeffs<'a, f64>,
    /// Uniform time step. Must be finite and `> 0`.
    pub tau: f64,
    /// Number of Chernoff steps composed.
    pub n_steps: usize,
}

// ---------------------------------------------------------------------------
// Forward step, transpose, and the adjoint-state driver
// ---------------------------------------------------------------------------

/// The three feet of node `i`: `x_i ± 2√(a_i τ)` and `x_i + 2 b_i τ`.
///
/// Mirrors `shift1d::apply_at_node_f64` exactly — the gradient must be of the
/// kernel as implemented, not of an idealised version of it.
fn feet(grid: &Grid1D<f64>, co: &ShiftCoeffs<'_, f64>, tau: f64, i: usize) -> (f64, f64, f64) {
    let x = grid.x_at(i);
    let s_diff = 2.0 * libm::sqrt(co.a[i] * tau);
    let s_drift = 2.0 * co.b[i] * tau;
    (x + s_diff, x - s_diff, x + s_drift)
}

/// One forward step `dst ← S(τ)·src`, reproducing the shipped node formula.
fn forward_step(
    grid: &Grid1D<f64>,
    co: &ShiftCoeffs<'_, f64>,
    tau: f64,
    src: &[f64],
    dst: &mut [f64],
) -> Result<(), SemiflowError> {
    for i in 0..grid.n {
        let (fp, fm, fd) = feet(grid, co, tau, i);
        dst[i] = 0.25 * grid.interp(src, fp)?
            + 0.25 * grid.interp(src, fm)?
            + 0.50 * grid.interp(src, fd)?
            + tau * co.c[i] * src[i];
    }
    Ok(())
}

/// One adjoint step `out ← S(τ)ᵀ·lam`.
///
/// `S` is a shift-and-interpolate **gather**; its transpose is the corresponding
/// **scatter** of the same interpolation weights, plus the diagonal reaction
/// term. `S` is not self-adjoint for variable coefficients — the drift foot is
/// one-sided — so the transpose is computed, never assumed.
fn adjoint_step(
    grid: &Grid1D<f64>,
    co: &ShiftCoeffs<'_, f64>,
    tau: f64,
    lam: &[f64],
    out: &mut [f64],
    probe: &mut [f64],
) -> Result<(), SemiflowError> {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let mut idx = [0_usize; MAX_ROW];
    let mut w = [0.0_f64; MAX_ROW];
    for i in 0..grid.n {
        let (fp, fm, fd) = feet(grid, co, tau, i);
        for (foot, weight) in [(fp, 0.25), (fm, 0.25), (fd, 0.50)] {
            let k = weight_row(grid, foot, probe, &mut idx, &mut w)?;
            let scale = weight * lam[i];
            for slot in 0..k {
                out[idx[slot]] += scale * w[slot];
            }
        }
        out[i] += tau * co.c[i] * lam[i];
    }
    Ok(())
}

/// Diagonal `∂(S(τ)u)_i/∂θ_i` for every `i`, in one `O(n)` pass (§61.2).
///
/// All off-diagonal entries are structurally zero, which is what collapses the
/// driver from `O(n_steps · n_params)` to `O(n_steps · n)`.
///
/// # Errors
/// `DomainViolation` if `field == A` and any `a_i <= 0` — the `√(τ/a_i)` chain
/// factor is undefined there.
fn param_deriv_diag(
    grid: &Grid1D<f64>,
    co: &ShiftCoeffs<'_, f64>,
    field: ShiftCoeffField,
    tau: f64,
    u: &[f64],
    out: &mut [f64],
) -> Result<(), SemiflowError> {
    for i in 0..grid.n {
        let (fp, fm, fd) = feet(grid, co, tau, i);
        out[i] = match field {
            ShiftCoeffField::A => {
                if co.a[i] <= 0.0 {
                    return Err(SemiflowError::DomainViolation {
                        what: "shift1d_coeff_gradient: d/da undefined at a_i <= 0",
                        value: co.a[i],
                    });
                }
                let chain = libm::sqrt(tau / co.a[i]);
                0.25 * chain * (sample_deriv(u, grid, fp)? - sample_deriv(u, grid, fm)?)
            }
            ShiftCoeffField::B => tau * sample_deriv(u, grid, fd)?,
            ShiftCoeffField::C => tau * u[i],
        };
    }
    Ok(())
}

/// Length and domain checks shared by the gradient entry point.
fn validate_gradient_inputs(
    problem: &Shift1DProblem<'_>,
    u0: &[f64],
    dj_du_n: &[f64],
    out: &[f64],
) -> Result<(), SemiflowError> {
    let n = problem.grid.n;
    let co = &problem.coeffs;
    for len in [
        u0.len(),
        dj_du_n.len(),
        out.len(),
        co.a.len(),
        co.b.len(),
        co.c.len(),
    ] {
        if len != n {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "shift1d_coeff_gradient: array length != grid.n",
                value: len as f64,
            });
        }
    }
    if !problem.tau.is_finite() || problem.tau <= 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "shift1d_coeff_gradient: tau must be finite and > 0",
            value: problem.tau,
        });
    }
    Ok(())
}

/// Adjoint-state gradient `∂J/∂θ` for one coefficient field (§61.4).
///
/// `dj_du_n` is the caller-supplied cotangent at the final time; the loss `J`
/// itself lives **outside** the library (the ADR-0115 boundary — no autograd
/// hook, no torch/JAX types in core). `out` is the caller-owned gradient
/// buffer; it is zeroed and length-checked here.
///
/// Reverse in time, forward-JVP in parameter, accumulating
/// `∂J/∂θ_i += ⟨λ_{k+1}, (∂S_k/∂θ)u_k⟩` — which the diagonality of §61.2 reduces
/// to an elementwise product.
///
/// # Errors
/// `DomainViolation` on a length mismatch or `a_i <= 0` (field `A`);
/// `Unsupported` from the grid's interpolation kind.
pub fn shift1d_coeff_gradient(
    problem: &Shift1DProblem<'_>,
    field: ShiftCoeffField,
    u0: &[f64],
    dj_du_n: &[f64],
    out: &mut [f64],
) -> Result<(), SemiflowError> {
    validate_gradient_inputs(problem, u0, dj_du_n, out)?;
    let (grid, co, tau) = (problem.grid, &problem.coeffs, problem.tau);
    let n = grid.n;
    for g in out.iter_mut() {
        *g = 0.0;
    }

    // Forward trajectory: u_0 … u_{n_steps}.
    let mut traj: Vec<Vec<f64>> = Vec::with_capacity(problem.n_steps + 1);
    traj.push(u0.to_vec());
    for k in 0..problem.n_steps {
        let mut next = vec![0.0_f64; n];
        forward_step(grid, co, tau, &traj[k], &mut next)?;
        traj.push(next);
    }

    // Backward sweep.
    let mut lam = dj_du_n.to_vec();
    let mut lam_next = vec![0.0_f64; n];
    let mut diag = vec![0.0_f64; n];
    let mut probe = vec![0.0_f64; n];
    for k in (0..problem.n_steps).rev() {
        param_deriv_diag(grid, co, field, tau, &traj[k], &mut diag)?;
        for i in 0..n {
            out[i] += lam[i] * diag[i];
        }
        adjoint_step(grid, co, tau, &lam, &mut lam_next, &mut probe)?;
        core::mem::swap(&mut lam, &mut lam_next);
    }
    Ok(())
}

/// Evolve `u0` for `problem.n_steps` with the shipped node formula (for callers
/// that need the primal alongside the gradient).
///
/// # Errors
/// Propagates from the interpolation kind.
pub fn shift1d_forward(
    problem: &Shift1DProblem<'_>,
    u0: &[f64],
) -> Result<Vec<f64>, SemiflowError> {
    let mut cur = u0.to_vec();
    let mut next = vec![0.0_f64; problem.grid.n];
    for _ in 0..problem.n_steps {
        forward_step(problem.grid, &problem.coeffs, problem.tau, &cur, &mut next)?;
        core::mem::swap(&mut cur, &mut next);
    }
    Ok(cur)
}

// ---------------------------------------------------------------------------
// Gate-facing wrappers
// ---------------------------------------------------------------------------

/// `Σ_j w_j(y)·values[j]` from the measured weight row.
///
/// Exists so `G_SHIFT1D_WEIGHTS_ORACLE` can compare the row against the sampler
/// it was measured from, which is what verifies the compact-support assumption.
///
/// # Errors
/// Propagates from the grid's interpolation kind.
pub fn weight_row_dot(grid: &Grid1D<f64>, y: f64, values: &[f64]) -> Result<f64, SemiflowError> {
    let mut probe = vec![0.0_f64; grid.n];
    let mut idx = [0_usize; MAX_ROW];
    let mut w = [0.0_f64; MAX_ROW];
    let k = weight_row(grid, y, &mut probe, &mut idx, &mut w)?;
    let mut acc = 0.0_f64;
    for slot in 0..k {
        acc += w[slot] * values[idx[slot]];
    }
    Ok(acc)
}

/// One forward step, exposed for the adjoint-identity gate.
///
/// # Errors
/// Propagates from the grid's interpolation kind.
pub fn forward_once(
    grid: &Grid1D<f64>,
    co: &ShiftCoeffs<'_, f64>,
    tau: f64,
    u: &[f64],
) -> Result<Vec<f64>, SemiflowError> {
    let mut out = vec![0.0_f64; grid.n];
    forward_step(grid, co, tau, u, &mut out)?;
    Ok(out)
}

/// One adjoint step, exposed for the adjoint-identity gate.
///
/// # Errors
/// Propagates from the grid's interpolation kind.
pub fn adjoint_once(
    grid: &Grid1D<f64>,
    co: &ShiftCoeffs<'_, f64>,
    tau: f64,
    lam: &[f64],
) -> Result<Vec<f64>, SemiflowError> {
    let mut out = vec![0.0_f64; grid.n];
    let mut probe = vec![0.0_f64; grid.n];
    adjoint_step(grid, co, tau, lam, &mut out, &mut probe)?;
    Ok(out)
}
