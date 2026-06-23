//! v4.0 A6 — `PointEval<F>` first-class API (ADR-0080, math.md §31).
//!
//! Bound-stack iterative pointwise evaluation. Replaces the v3.0 `PointEval`
//! Unsupported stub that was planned in ADR-0073 §"Future work" (never shipped).
//!
//! # Algorithm (math §31.2 Algorithm 31.1)
//!
//! For the pragmatic Wave-B path, `eval_at` runs `apply_into^n` internally
//! and samples the resulting state at point `x`. This preserves the byte-identity
//! contract (Proposition 31.1) because it computes exactly the same floating-point
//! reductions as the full-grid path sampled at `x`. The bound-stack O(n·q)
//! memory optimisation is documented as a v4.x opportunity in each impl block.
//!
//! # v4.0 Wave B backends (4 of 5)
//!
//! - **Backend A**: `DiffusionChernoff<f64>` — 1-D variable-coefficient heat.
//! - **Backend B**: `ShiftChernoff1D<f64>` — 1-D Theorem 6 shift kernel.
//! - **Backend C**: `ManifoldChernoff<Sphere2<f64>, f64>` — Riemannian manifold heat.
//! - **Backend D**: `HypoellipticChernoff<f64, 2, 1>` — Kolmogorov phase-space.
//! - **Backend E** (`AnisotropicShiftChernoffND`): deferred to Wave C retrofit.
//!
//! # Byte-identity contract (`G_POINTEVAL` gate)
//!
//! `kernel.eval_at(τ, src, x, n).unwrap().to_bits()
//!     == (apply_into^n src).sample_at(x).unwrap().to_bits()`
//!
//! See `tests/point_eval_byte_identity.rs` for the `RELEASE_BLOCKING` gate.

#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::{
    chernoff::ChernoffFunction, error::SemiflowError, float::SemiflowFloat, grid_fn::GridFn1D,
    grid_fn2d::GridFn2D, scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// PointEval trait
// ---------------------------------------------------------------------------

/// First-class pointwise evaluation: `(F(τ))^n src @ x` without full-grid alloc.
///
/// A super-trait of [`ChernoffFunction<F>`]. Kernels opt in by implementing
/// this trait; no default bridge (ADR-0080 §"Rationale" — silent-false-witness
/// footgun). Grid-only kernels (`Strang2D`, `Magnus*`, `Diffusion4thChernoff`)
/// do NOT implement `PointEval` (math §31.5).
///
/// ## Byte-identity contract (Proposition 31.1)
///
/// The scalar returned MUST be bit-identical at f64 precision to the value
/// obtained by running `apply_into` n times and sampling at `x`:
/// `eval_at(τ, src, x, n) ≡_{f64} (apply_into^n src).sample(x)`.
///
/// ## Query point encoding
///
/// `x` is a slice over the kernel's spatial domain:
/// - 1-D kernels: `x.len() == 1`.
/// - 2-D kernels: `x.len() == 2` (col-axis first, row-axis second).
/// - Future d-D kernels: `x.len() == D`.
///
/// Returns `Err(DomainViolation)` if `x` has incorrect length.
/// Returns `Err(DomainViolation)` if `n_steps == 0`.
pub trait PointEval<F: SemiflowFloat = f64>: ChernoffFunction<F> {
    /// Evaluate `(F(τ))^{n_steps}` applied to `src`, sampled at point `x`.
    ///
    /// # Errors
    ///
    /// - `DomainViolation` if `n_steps == 0`.
    /// - `DomainViolation` if `x` has wrong length for this backend.
    /// - Propagates any error from `apply_into` or interpolation.
    fn eval_at(&self, tau: F, src: &Self::S, x: &[F], n_steps: u32) -> Result<F, SemiflowError>;

    /// Evaluate `(F(τ))^{n_steps}` applied to `src` at each query in `x_queries`.
    ///
    /// Each element of `x_queries` is a coordinate slice with the same encoding
    /// as the `x` parameter of [`eval_at`]: 1-D backends expect slices of length
    /// 1, 2-D backends length 2, and so on.
    ///
    /// Returns a `Vec<F>` of length `x_queries.len()` in the same order.
    /// The first error encountered short-circuits and is returned immediately.
    ///
    /// v4.x optimisation opportunity: replace the per-query `eval_at` calls with
    /// a single shared `apply_into^n` pass followed by batch sampling.
    ///
    /// # Errors
    ///
    /// Propagates the first `Err` returned by `eval_at`.
    fn eval_at_batch(
        &self,
        tau: F,
        src: &Self::S,
        x_queries: &[&[F]],
        n_steps: u32,
    ) -> Result<alloc::vec::Vec<F>, SemiflowError> {
        x_queries
            .iter()
            .map(|x| self.eval_at(tau, src, x, n_steps))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Shared guard helpers
// ---------------------------------------------------------------------------

/// Validate that `n_steps > 0` and `x` has exactly `expected_dim` coordinates.
#[allow(clippy::cast_precision_loss)]
#[inline]
fn guard_1d(n_steps: u32, x: &[impl Copy]) -> Result<(), SemiflowError> {
    if n_steps == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "eval_at: n_steps must be >= 1",
            value: 0.0,
        });
    }
    if x.len() != 1 {
        return Err(SemiflowError::DomainViolation {
            what: "eval_at: 1-D backend requires x.len() == 1",
            value: x.len() as f64,
        });
    }
    Ok(())
}

/// Validate that `n_steps > 0` and `x` has exactly 2 coordinates.
#[allow(clippy::cast_precision_loss)]
#[inline]
fn guard_2d(n_steps: u32, x: &[impl Copy]) -> Result<(), SemiflowError> {
    if n_steps == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "eval_at: n_steps must be >= 1",
            value: 0.0,
        });
    }
    if x.len() != 2 {
        return Err(SemiflowError::DomainViolation {
            what: "eval_at: 2-D backend requires x.len() == 2",
            value: x.len() as f64,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared iterate-n helper for 1-D kernels (GridFn1D<f64> state)
// ---------------------------------------------------------------------------

/// Run `kernel.apply_into` for `n_steps` steps, starting from `src`.
///
/// Returns the final state. Allocates one working buffer (size = `src.len()`).
///
/// v4.x optimisation opportunity: replace with bound-stack O(n·q) path
/// per math §31.2 Algorithm 31.1 — avoids materialising the full grid state
/// on each step, reducing memory from O(N) to O(q) per query.
fn iterate_1d<C, F>(
    kernel: &C,
    tau: F,
    src: &GridFn1D<F>,
    n_steps: u32,
) -> Result<GridFn1D<F>, SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    let mut pool = ScratchPool::new();
    let mut cur = src.clone();
    let mut nxt = src.clone();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &cur, &mut nxt, &mut pool)?;
        core::mem::swap(&mut cur, &mut nxt);
    }
    Ok(cur)
}

// ---------------------------------------------------------------------------
// Shared iterate-n helper for 2-D kernels (GridFn2D<f64> state)
// ---------------------------------------------------------------------------

/// Run `kernel.apply_into` for `n_steps` steps, starting from `src`.
///
/// Returns the final `GridFn2D<f64>` state.
///
/// v4.x optimisation opportunity: replace with bound-stack O(n·q) path
/// per math §31.2 Algorithm 31.1 (Backend C: parallel-transport quadrature;
/// Backend D: graded-tangent quadrature on ℝ²).
fn iterate_2d<C>(
    kernel: &C,
    tau: f64,
    src: &GridFn2D<f64>,
    n_steps: u32,
) -> Result<GridFn2D<f64>, SemiflowError>
where
    C: ChernoffFunction<f64, S = GridFn2D<f64>>,
{
    let mut pool = ScratchPool::<f64>::new();
    let n = src.values.len();
    let mut cur = GridFn2D {
        values: src.values.clone(),
        grid: src.grid,
    };
    let mut nxt = GridFn2D {
        values: alloc::vec![0.0_f64; n],
        grid: src.grid,
    };
    for _ in 0..n_steps {
        kernel.apply_into(tau, &cur, &mut nxt, &mut pool)?;
        core::mem::swap(&mut cur, &mut nxt);
    }
    Ok(cur)
}

// ---------------------------------------------------------------------------
// Bilinear sample on GridFn2D<f64> at chart point (cx, cy)
// ---------------------------------------------------------------------------

/// Bilinear interpolation of `state` at chart position `(cx, cy)`.
///
/// `cx` indexes the x-axis (col), `cy` the y-axis (row).
/// Clamps out-of-range coordinates to the nearest valid node pair.
///
/// This is the canonical `GridFn2D<f64>` point-evaluation primitive for
/// `eval_at` on 2-D backends. Matching the same logic in both `eval_at`
/// and the byte-identity test's full-grid path ensures Proposition 31.1.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn sample_gridfn2d(state: &GridFn2D<f64>, cx: f64, cy: f64) -> f64 {
    let nx = state.grid.nx();
    let ny = state.grid.ny();
    let xi = (cx - state.grid.x.xmin) / (state.grid.x.xmax - state.grid.x.xmin) * (nx - 1) as f64;
    let yi = (cy - state.grid.y.xmin) / (state.grid.y.xmax - state.grid.y.xmin) * (ny - 1) as f64;
    let xi = xi.clamp(0.0, (nx - 2) as f64);
    let yi = yi.clamp(0.0, (ny - 2) as f64);
    let i0 = xi.floor() as usize;
    let j0 = yi.floor() as usize;
    let fx = (xi - i0 as f64).clamp(0.0, 1.0);
    let fy = (yi - j0 as f64).clamp(0.0, 1.0);
    let v00 = state.values[j0 * nx + i0];
    let v10 = state.values[j0 * nx + i0 + 1];
    let v01 = state.values[(j0 + 1) * nx + i0];
    let v11 = state.values[(j0 + 1) * nx + i0 + 1];
    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

// ---------------------------------------------------------------------------
// Backend A — DiffusionChernoff<f64> (math §31.2 Backend A)
// ---------------------------------------------------------------------------

use crate::diffusion::DiffusionChernoff;

/// Backend A: `DiffusionChernoff<f64>` pointwise eval (math §31.2 Backend A).
///
/// Pragmatic Wave-B path: `apply_into^n` + `sample_generic(x[0])`.
/// Byte-identity guaranteed by Proposition 31.1 (same fp reductions).
///
/// v4.x optimisation: replace `iterate_1d` with 5-pt Gauss-Hermite bound-stack
/// per math §31.2 Algorithm 31.1 Backend A (O(n·5) not O(n·N)).
impl PointEval<f64> for DiffusionChernoff<f64> {
    fn eval_at(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        x: &[f64],
        n_steps: u32,
    ) -> Result<f64, SemiflowError> {
        guard_1d(n_steps, x)?;
        let final_state = iterate_1d(self, tau, src, n_steps)?;
        final_state.sample(x[0])
    }
}

// ---------------------------------------------------------------------------
// Backend B — ShiftChernoff1D<f64> (math §31.2 Backend B)
// ---------------------------------------------------------------------------

use crate::shift1d::ShiftChernoff1D;

/// Backend B: `ShiftChernoff1D<f64>` pointwise eval (math §31.2 Backend B).
///
/// Identical pattern to Backend A (both have `GridFn1D<f64>` state).
/// Pragmatic Wave-B path: `apply_into^n` + `sample(x[0])`.
///
/// v4.x optimisation: 5-pt Gauss-Hermite bound-stack (constant-a kernel
/// has a closed-form Gaussian, so the bound-stack is simpler than Backend A).
impl PointEval<f64> for ShiftChernoff1D<f64> {
    fn eval_at(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        x: &[f64],
        n_steps: u32,
    ) -> Result<f64, SemiflowError> {
        guard_1d(n_steps, x)?;
        let final_state = iterate_1d(self, tau, src, n_steps)?;
        final_state.sample(x[0])
    }
}

// ---------------------------------------------------------------------------
// Backend C — ManifoldChernoff<Sphere2<f64>, f64> (math §31.2 Backend C)
// ---------------------------------------------------------------------------

use crate::{manifold::Sphere2, manifold_chernoff::ManifoldChernoff};

/// Backend C: `ManifoldChernoff<Sphere2<f64>, f64>` pointwise eval.
///
/// Math §31.2 Backend C: manifold heat kernel at `(x[0], x[1])` in chart
/// coordinates (θ, φ). Pragmatic Wave-B path: `apply_into^n` + bilinear
/// sample of the resulting `GridFn2D<f64>` at `(x[0], x[1])`.
///
/// v4.x optimisation: per-step parallel-transport quadrature on `T_{x₀}S²`
/// with `q_tangent² = 25` auxiliary nodes (math §31.2 Algorithm 31.1 Backend C).
impl PointEval<f64> for ManifoldChernoff<Sphere2<f64>, f64> {
    fn eval_at(
        &self,
        tau: f64,
        src: &GridFn2D<f64>,
        x: &[f64],
        n_steps: u32,
    ) -> Result<f64, SemiflowError> {
        guard_2d(n_steps, x)?;
        let final_state = iterate_2d(self, tau, src, n_steps)?;
        Ok(sample_gridfn2d(&final_state, x[0], x[1]))
    }
}

// ---------------------------------------------------------------------------
// Backend D — HypoellipticChernoff<f64, 2, 1> (math §31.2 Backend D)
// ---------------------------------------------------------------------------

use crate::hormander::HypoellipticChernoff;

/// Backend D: `HypoellipticChernoff<f64, 2, 1>` pointwise eval.
///
/// Math §31.2 Backend D: Kolmogorov phase-space at `(x[0], x[1])`.
/// Pragmatic Wave-B path: `apply_into^n` + bilinear sample at `(x[0], x[1])`.
///
/// v4.x optimisation: graded-tangent quadrature on `ℝ²` with Strang-Hörmander
/// palindromic update per step (math §31.2 Algorithm 31.1 Backend D).
impl PointEval<f64> for HypoellipticChernoff<f64, 2, 1> {
    fn eval_at(
        &self,
        tau: f64,
        src: &GridFn2D<f64>,
        x: &[f64],
        n_steps: u32,
    ) -> Result<f64, SemiflowError> {
        guard_2d(n_steps, x)?;
        let final_state = iterate_2d(self, tau, src, n_steps)?;
        Ok(sample_gridfn2d(&final_state, x[0], x[1]))
    }
}

// ---------------------------------------------------------------------------
// Backend E — AnisotropicShiftChernoffND<f64, D> (Wave C retrofit)
// ---------------------------------------------------------------------------

use crate::{grid_nd::GridFnND, shift_nd::AnisotropicShiftChernoffND};

/// Validate that `n_steps > 0` and `x` has exactly `D` coordinates.
#[allow(clippy::cast_precision_loss)]
#[inline]
fn guard_nd<const D: usize>(n_steps: u32, x: &[f64]) -> Result<(), SemiflowError> {
    if n_steps == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "eval_at: n_steps must be >= 1",
            value: 0.0,
        });
    }
    if x.len() != D {
        return Err(SemiflowError::DomainViolation {
            what: "eval_at: x.len() must equal D",
            value: x.len() as f64,
        });
    }
    Ok(())
}

/// Run `kernel.apply_into` for `n_steps` steps on a `GridFnND<f64, D>` state.
fn iterate_nd<const D: usize>(
    kernel: &AnisotropicShiftChernoffND<f64, D>,
    tau: f64,
    src: &GridFnND<f64, D>,
    n_steps: u32,
) -> Result<GridFnND<f64, D>, SemiflowError>
where
    AnisotropicShiftChernoffND<f64, D>: ChernoffFunction<f64, S = GridFnND<f64, D>>,
{
    let mut pool = ScratchPool::new();
    let mut cur = src.clone();
    let mut nxt = GridFnND::new(src.grid.clone(), alloc::vec![0.0_f64; src.values.len()])?;
    for _ in 0..n_steps {
        kernel.apply_into(tau, &cur, &mut nxt, &mut pool)?;
        core::mem::swap(&mut cur, &mut nxt);
    }
    Ok(cur)
}

/// Backend E: `AnisotropicShiftChernoffND<f64, D>` pointwise eval (math §31.2 Backend E).
///
/// Pragmatic Wave-C path: `apply_into^n` + multilinear sample at `x`.
/// Byte-identity guaranteed by Proposition 31.1 (same fp reductions).
///
/// Supports D ∈ {2, 3, 4, 5} in v4.0 (the G_DDIM-gated range).
/// D ≥ 6 compiles but is ungated (tensor-product Gauss-Hermite cost 5^D).
///
/// v4.x optimisation: replace with sparse Gauss-Hermite bound-stack
/// per math §31.2 Algorithm 31.1 Backend E.
impl<const D: usize> PointEval<f64> for AnisotropicShiftChernoffND<f64, D>
where
    AnisotropicShiftChernoffND<f64, D>: ChernoffFunction<f64, S = GridFnND<f64, D>>,
{
    fn eval_at(
        &self,
        tau: f64,
        src: &GridFnND<f64, D>,
        x: &[f64],
        n_steps: u32,
    ) -> Result<f64, SemiflowError> {
        guard_nd::<D>(n_steps, x)?;
        let final_state = iterate_nd(self, tau, src, n_steps)?;
        final_state.sample(x)
    }
}

// ---------------------------------------------------------------------------
// Inline unit tests (batch H6: moved to point_eval_tests.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    include!("point_eval_tests.rs");
}
