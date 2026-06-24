//! v3.1 A3 — Hörmander hypoelliptic Chernoff approximation (math.md §28, ADR-0077).
//!
//! For `L = X₀ + ½Σ Xₖ²` with smooth vector fields `{Xᵢ}` satisfying
//! Hörmander's bracket-generating condition (Hörmander 1967 *Acta Math.* §1,
//! Theorem 1.1), the Chernoff approximation uses the palindromic
//! Strang-Hörmander decomposition (math.md §28.3 eq 28.4):
//!
//! ```text
//! F(τ) = exp(τX₀/2) ∘ ∏_{k=1..M-1} exp(τXₖ²/2) ∘ exp(τX_M²/2)
//!                   ∘ ∏_{rev} exp(τXₖ²/2) ∘ exp(τX₀/2)
//! ```
//!
//! Order-2 tangency on `f ∈ D(L²)` per Galkin-Remizov 2025 *IJM* Theorem 3.1
//! (K=2), riding on v3.0 `ApproximationSubspace<2, F>` (ADR-0073).
//! Higher orders are OPEN per Festschrift §3.
//!
//! Restricted to step-2 Carnot groups (Kolmogorov, Heisenberg) in v3.1.0.
//!
//! ## References
//!
//! - Hörmander 1967 *Acta Math.* 119:1, pp. 147-171 (bracket-generating condition)
//! - Kolmogorov 1934 *Math. Annalen* 108, pp. 149-160 (fundamental solution, G28 oracle)
//! - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 (K=2 tangency)
//! - Folland 1975 *Ark. Mat.* 13, pp. 161-207 (Heisenberg sub-Laplacian, §2.3)

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use core::marker::PhantomData;

use crate::{error::SemiflowError, float::SemiflowFloat};

// ─── Heisenberg group helpers (batch H6 split) ───────────────────────────────
#[path = "hormander_helpers.rs"]
mod hormander_helpers;
pub use hormander_helpers::{HeisenbergGroup, HeisenbergX, HeisenbergY};

// ─── VectorField<F, D> trait ──────────────────────────────────────────────────

/// Smooth vector field on `ℝ^D` (math.md §28.6, ADR-0077).
///
/// Required by `HypoellipticChernoff<F, D, M>` for the palindromic
/// Strang-Hörmander decomposition (math.md §28.3). The Lie bracket
/// `bracket_with` default uses central-difference Jacobians; closed-form
/// Carnot backends override for sympy-traceable accuracy.
///
/// **Step-2 Carnot backends shipped in v3.1**: `KolmogorovDriftX0<F>`,
/// `KolmogorovDiffusionX1<F>` (d=2), `HeisenbergX<F>`, `HeisenbergY<F>` (d=3).
pub trait VectorField<F: SemiflowFloat, const D: usize>: Send + Sync + 'static {
    /// Write the vector-field value `X(x)` into `out`.
    ///
    /// # Errors
    /// Returns `DomainViolation` on NaN/Inf in `x`, or if any component of
    /// `out` is set to a non-finite value by the implementation.
    fn evaluate(&self, x: &[F; D], out: &mut [F; D]) -> Result<(), SemiflowError>;

    /// Write the Lie bracket `[self, other](x)` into `out`.
    ///
    /// Default: central-difference Jacobians with `h ≈ ε^(1/3) ≈ 6e-6`.
    /// Closed-form Carnot backends MAY override for sympy-traceable accuracy.
    ///
    /// Used by `HypoellipticChernoff::new` for step-2 verification.
    ///
    /// # Errors
    /// Returns `DomainViolation` on NaN/Inf in inputs or computed outputs.
    fn bracket_with(
        &self,
        other: &dyn VectorField<F, D>,
        x: &[F; D],
        out: &mut [F; D],
    ) -> Result<(), SemiflowError>
    where
        Self: Sized,
    {
        bracket_central_diff(self, other, x, out)
    }
}

/// Central-difference Lie bracket `[X, Y]^i = DY·X(x) − DX·Y(x)`.
///
/// Uses `h ≈ 6e-6` (cube-root of f64 machine epsilon).
/// Caller provides concrete `x_field` and `y_field` (both `Sized`-to-dyn).
// y_xp/y_xm and x_xp/x_xm are directional perturbation buffers; names are intentional pairs.
#[allow(clippy::similar_names)]
pub(crate) fn bracket_central_diff<F: SemiflowFloat, const D: usize>(
    x_field: &dyn VectorField<F, D>,
    y_field: &dyn VectorField<F, D>,
    x: &[F; D],
    out: &mut [F; D],
) -> Result<(), SemiflowError> {
    let h = crate::float::from_f64::<F>(6e-6_f64);
    let two_h = h + h;
    let mut xp = *x;
    let mut xm = *x;
    let mut y_xp = [F::zero(); D];
    let mut y_xm = [F::zero(); D];
    let mut x_xp = [F::zero(); D];
    let mut x_xm = [F::zero(); D];
    let mut x_at = [F::zero(); D];
    let mut y_at = [F::zero(); D];
    x_field.evaluate(x, &mut x_at)?;
    y_field.evaluate(x, &mut y_at)?;
    out.fill(F::zero());
    for j in 0..D {
        xp[j] = x[j] + h;
        xm[j] = x[j] - h;
        y_field.evaluate(&xp, &mut y_xp)?;
        y_field.evaluate(&xm, &mut y_xm)?;
        x_field.evaluate(&xp, &mut x_xp)?;
        x_field.evaluate(&xm, &mut x_xm)?;
        xp[j] = x[j];
        xm[j] = x[j];
        for i in 0..D {
            let dy_i = (y_xp[i] - y_xm[i]) / two_h;
            let dx_i = (x_xp[i] - x_xm[i]) / two_h;
            out[i] = out[i] + dy_i * x_at[j] - dx_i * y_at[j];
        }
    }
    Ok(())
}

// ─── Kolmogorov phase space d=2 (math.md §28.4.A) ────────────────────────────

/// Marker for the Kolmogorov phase space `ℝ² = {(x, v)}` (math.md §28.4.A).
///
/// Vector fields: `X₀ = v·∂_x` (drift in x), `X₁ = ∂_v` (diffusion in v).
/// Bracket: `[X₁, X₀] = ∂_x` — step-2 Carnot (generates missing x-direction).
///
/// Reference: Kolmogorov 1934 *Math. Annalen* 108.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KolmogorovPhaseSpace<F: SemiflowFloat = f64> {
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat> KolmogorovPhaseSpace<F> {
    /// Construct the Kolmogorov phase-space marker (zero-sized, infallible).
    #[must_use]
    pub fn new() -> Self {
        Self { _f: PhantomData }
    }

    /// Construct the drift field `X₀ = v·∂_x` (D=2, `x[0]`=`x_pos`, `x[1]`=v).
    #[must_use]
    pub fn x0_drift() -> KolmogorovDriftX0<F> {
        KolmogorovDriftX0 { _f: PhantomData }
    }

    /// Construct the diffusion field `X₁ = ∂_v` (D=2, `x[0]`=`x_pos`, `x[1]`=v).
    #[must_use]
    pub fn x1_diffusion() -> KolmogorovDiffusionX1<F> {
        KolmogorovDiffusionX1 { _f: PhantomData }
    }
}

impl<F: SemiflowFloat> Default for KolmogorovPhaseSpace<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Drift field `X₀ = v·∂_x` for the Kolmogorov phase space.
///
/// Returns `(v, 0)` at `(x, v)` — the drift flows position along velocity.
/// Closed-form exponential: `e^{τX₀}(x, v) = (x + τ·v, v)` (exact, linear).
///
/// Reference: Kolmogorov 1934 *Math. Annalen* 108; math.md §28.4.A.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KolmogorovDriftX0<F: SemiflowFloat = f64> {
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat> VectorField<F, 2> for KolmogorovDriftX0<F> {
    /// `X₀(x, v) = (v, 0)`: velocity is the drift rate of position.
    fn evaluate(&self, x: &[F; 2], out: &mut [F; 2]) -> Result<(), SemiflowError> {
        out[0] = x[1]; // v
        out[1] = F::zero();
        Ok(())
    }
}

/// Diffusion field `X₁ = ∂_v` for the Kolmogorov phase space.
///
/// Returns `(0, 1)` at any point — constant unit vector in v-direction.
/// Closed-form exp: `e^{τX₁²/2}` is 1D heat kernel in v (x frozen).
///
/// Reference: Kolmogorov 1934 *Math. Annalen* 108; math.md §28.4.A.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KolmogorovDiffusionX1<F: SemiflowFloat = f64> {
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat> VectorField<F, 2> for KolmogorovDiffusionX1<F> {
    /// `X₁(x, v) = (0, 1)`: constant unit diffusion in v.
    fn evaluate(&self, _x: &[F; 2], out: &mut [F; 2]) -> Result<(), SemiflowError> {
        out[0] = F::zero();
        out[1] = F::one();
        Ok(())
    }
}

// ─── HypoellipticChernoff stub (Wave B fills apply_into body) ─────────────────

/// Hörmander hypoelliptic Chernoff approximation scaffold (v3.1 Wave A).
///
/// Wave A: trait + Carnot backends + struct stub.
/// Wave B will add:
/// - `HypoellipticChernoff::new` with step-2 bracket-generating checker
/// - `impl ChernoffFunction<F>` with palindromic Strang-Hörmander `apply_into`
/// - `impl ApproximationSubspace<2, F>` opt-in marker
/// - G28 / G29 acceptance gates in `tests/hormander_kolmogorov_slope.rs`
///
/// ## Fields
///
/// - `x0_drift`: drift field `X₀` (may be zero for Heisenberg `X₀ = 0`).
/// - `x_diff`: diffusive fields `X₁, …, X_M`.
///
/// ## References
///
/// - math.md §28.3 (palindromic Strang-Hörmander, eq 28.4)
/// - ADR-0077 §"Decision" (struct layout + Wave A/B split)
/// - Galkin-Remizov 2025 *IJM* Theorem 3.1 (order-2 tangency, K=2)
pub struct HypoellipticChernoff<F: SemiflowFloat = f64, const D: usize = 2, const M: usize = 1> {
    /// Drift field `X₀`. May be the zero field (`HeisenbergGroup` case).
    pub x0_drift: alloc::boxed::Box<dyn VectorField<F, D>>,
    /// Diffusive fields `X₁, …, X_M`.
    pub x_diff: alloc::vec::Vec<alloc::boxed::Box<dyn VectorField<F, D>>>,
    pub(crate) _f: PhantomData<F>,
}

// ─── Wave B: HypoellipticChernoff impl ───────────────────────────────────────

use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion::DiffusionChernoff,
    float::from_f64,
    grid::InterpKind,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    scratch::ScratchPool,
};

/// Kolmogorov hypoelliptic Chernoff: `L = v·∂_x + ½∂²_v` on `ℝ²` (math.md §28.4.A).
///
/// Type alias for `HypoellipticChernoff<F, 2, 1>`.
/// Accepts only Kolmogorov backends; Hörmander rank checked in `new`.
pub type KolmogorovHypoelliptic<F = f64> = HypoellipticChernoff<F, 2, 1>;

impl<F: SemiflowFloat> HypoellipticChernoff<F, 2, 1> {
    /// Construct from Kolmogorov drift + diffusion fields.
    ///
    /// Verifies step-2 Carnot bracket-generating condition at origin:
    /// `{X₁, [X₁, X₀]}` must span `T_{(0,0)}ℝ²` (rank 2).
    ///
    /// # Errors
    /// - `DomainViolation` if fields do not satisfy the Hörmander condition.
    pub fn new(
        x0_drift: alloc::boxed::Box<dyn VectorField<F, 2>>,
        x_diff: [alloc::boxed::Box<dyn VectorField<F, 2>>; 1],
    ) -> Result<Self, SemiflowError> {
        let origin = [F::zero(); 2];
        let rank = check_rank_d2m1(&*x0_drift, &*x_diff[0], &origin)?;
        if rank < 2 {
            return Err(SemiflowError::DomainViolation {
                what: "Hörmander condition violated: brackets fail to span T_xM at origin",
                value: rank as f64,
            });
        }
        let mut fields: alloc::vec::Vec<alloc::boxed::Box<dyn VectorField<F, 2>>> =
            alloc::vec::Vec::with_capacity(1);
        for field in x_diff {
            fields.push(field);
        }
        Ok(Self {
            x0_drift,
            x_diff: fields,
            _f: PhantomData,
        })
    }
}

/// Rank checker: columns are [X₁(x), [X₁, X₀](x)] — must span ℝ².
///
/// Returns the column rank (0..=2) via Gaussian elimination on the 2×2 matrix.
fn check_rank_d2m1<F: SemiflowFloat>(
    x0: &dyn VectorField<F, 2>,
    x1: &dyn VectorField<F, 2>,
    x: &[F; 2],
) -> Result<usize, SemiflowError> {
    let mut col1 = [F::zero(); 2];
    let mut bracket = [F::zero(); 2];
    x1.evaluate(x, &mut col1)?;
    bracket_central_diff(x1, x0, x, &mut bracket)?;
    let eps = from_f64::<F>(1e-8_f64);
    let a11 = col1[0];
    let a21 = col1[1];
    let a12 = bracket[0];
    let a22 = bracket[1];
    // Gaussian elimination: det = a11*a22 - a12*a21
    let det_abs = (a11 * a22 - a12 * a21).abs();
    if det_abs > eps {
        Ok(2)
    } else if col1[0].abs() > eps || col1[1].abs() > eps {
        Ok(1)
    } else {
        Ok(0)
    }
}

// ─── Heisenberg backend (extracted to keep hormander.rs ≤ 800 LoC) ───────────

// `hormander_heisenberg` lives alongside this file; it uses `pub(super)` items
// defined here (VectorField, HypoellipticChernoff, bracket_central_diff).
// The `ChernoffFunction<f64> for HypoellipticChernoff<f64, 3, 2>` impl lives
// in that sibling module.

impl ChernoffFunction<f64> for KolmogorovHypoelliptic<f64> {
    type S = GridFn2D<f64>;

    /// Palindromic Strang-Hörmander: `exp(τX₀/2) ∘ exp(τ·½∂²_v) ∘ exp(τX₀/2)`.
    ///
    /// Step 1 + 3: shift x by `±(τ/2)·v` at each (x,v) node (closed-form advection).
    /// Step 2: 1D heat in v-direction at each x-column, time `τ`, diffusivity `½`.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn2D<f64>,
        dst: &mut GridFn2D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "tau must be finite and non-negative",
                value: tau,
            });
        }
        let half_tau = tau * 0.5;
        let nx = src.grid.nx();
        let ny = src.grid.ny();
        let n = nx * ny;
        let mut mid = GridFn2D {
            values: alloc::vec![0.0_f64; n],
            grid: src.grid,
        };
        shift_x_pass(src, &mut mid, half_tau, nx, ny)?;
        let v_grid = src.grid.y;
        let heat_v = DiffusionChernoff::new_const_a(0.5_f64, 0.5_f64, v_grid);
        heat_v_pass(&mid, dst, tau, &heat_v, nx, ny, scratch)?;
        let mut out2 = GridFn2D {
            values: alloc::vec![0.0_f64; n],
            grid: src.grid,
        };
        shift_x_pass(dst, &mut out2, half_tau, nx, ny)?;
        dst.values.copy_from_slice(&out2.values);
        Ok(())
    }

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

/// Shift-x pass: `u_out(x_i, v_j) = u_src(x_i − shift * v_j, v_j)`.
///
/// For each row j (fixed `v_j`), shift the 1D function u(·, `v_j`) by `shift * v_j`
/// using `SepticHermite` interpolation (8th-order in space, ADR-0109). Higher order than
/// `CubicHermite` reduces the spatial-discretization accumulation: O(dx^8/tau)
/// vs O(dx^4/tau), keeping the method in the O(tau^2) convergence regime over
/// the `N_SWEEP` range used by G28.
fn shift_x_pass(
    src: &GridFn2D<f64>,
    dst: &mut GridFn2D<f64>,
    shift: f64,
    nx: usize,
    ny: usize,
) -> Result<(), SemiflowError> {
    // Use SepticHermite for the x-grid shift to suppress per-step spatial truncation (ADR-0109).
    let x_grid_septic = src.grid.x.with_interp(InterpKind::SepticHermite);
    for j in 0..ny {
        let v_j = src.grid.y.x_at(j);
        let delta = shift * v_j;
        let row = src.row(j);
        for i in 0..nx {
            let x_i = src.grid.x.x_at(i);
            let x_src = x_i - delta;
            dst.values[j * nx + i] = x_grid_septic.interp(&row.values, x_src)?;
        }
    }
    Ok(())
}

/// Heat-v pass: apply 1D heat in v at each x-column.
///
/// For each column i (fixed `x_i`), apply `DiffusionChernoff.apply_into` in v.
/// Uses `scratch` for zero-alloc hot path.
// Tight per-axis heat kernel: all args are grid/kernel state required simultaneously; struct adds noise.
#[allow(clippy::too_many_arguments)]
fn heat_v_pass(
    src: &GridFn2D<f64>,
    dst: &mut GridFn2D<f64>,
    tau: f64,
    heat: &DiffusionChernoff<f64>,
    nx: usize,
    ny: usize,
    scratch: &mut ScratchPool<f64>,
) -> Result<(), SemiflowError> {
    let v_grid = src.grid.y;
    for i in 0..nx {
        let col = src.col(i);
        let mut col_dst = GridFn1D {
            values: alloc::vec![0.0_f64; ny],
            grid: v_grid,
        };
        heat.apply_into(tau, &col, &mut col_dst, scratch)?;
        for j in 0..ny {
            dst.values[j * nx + i] = col_dst.values[j];
        }
    }
    Ok(())
}

// ─── Inline unit tests (batch H6: moved to hormander_tests.rs) ───────────────

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    include!("hormander_tests.rs");
}
