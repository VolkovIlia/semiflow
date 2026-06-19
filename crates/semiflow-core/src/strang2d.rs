//! [`Strang2D`] — palindromic Strang composition for 2D tensor-product generators.
//!
//! Implements `Φ²ᴰ(τ) = AxisLift<X>(τ/2) ∘ AxisLift<Y>(τ) ∘ AxisLift<X>(τ/2)`
//! for separable generators `L = L_x ⊗ I + I ⊗ L_y`.
//!
//! Global order 2 (math.md §10 Theorem 7): the commutator identity
//! `[L_x ⊗ I, I ⊗ L_y] = 0` makes palindromic Strang exact at the BCH level;
//! per-axis order-2 lifts are therefore sufficient.
//!
//! `Strang2D` is a **dedicated type** (closed ADR-0012 decision) — it does NOT
//! generalise the existing `StrangSplit<D, R>` to avoid touching stable v0.2.0+ code.
//!
//! When the `parallel` feature is enabled, `apply` dispatches to an 8-thread
//! palindromic Strang kernel via `std::thread::scope` (ADR-0018).  The parallel
//! and serial paths return **bit-identical** results (see
//! `tests/strang2d_parallel_bit_equal.rs`); callers see no API change, but
//! the `ChernoffFunction` impl gains `Send + Sync` bounds on `X` and `Y`.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 4)
//!
//! `Strang2D<X, Y, F: SemiflowFloat = f64>` — the `= f64` default keeps all
//! existing call-sites compiling unchanged. The parallel path remains f64-only
//! (the parallel feature requires `X, Y: ChernoffFunction<f64>`).
//!
//! See `contracts/semiflow-core.tensor.yaml` §3 `Strang2D`,
//! `math.md` §10 Theorem 7 and Lemma 10.2,
//! `docs/adr/0012-tensor-product-2d.md`,
//! and `docs/adr/0018-parallel-strang2d.md`.

#[cfg(feature = "parallel")]
use crate::parallel_pool::ParallelPool2D;
#[cfg(feature = "parallel")]
use crate::strang2d_parallel::{
    parallel_x_pass, parallel_y_pass, resolve_threads, MIN_ROWS_PER_THREAD,
};
use alloc::vec::Vec;

use crate::{
    axis::{Axis, AxisLift},
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{half, SemiflowFloat},
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Strang2D
// ---------------------------------------------------------------------------

/// Palindromic [`Strang2D`] composition for 2D tensor-product generators.
///
/// Composition: `Φ²ᴰ(τ) = AxisLift<X>(τ/2) ∘ AxisLift<Y>(τ) ∘ AxisLift<X>(τ/2)`.
///
/// Both `X` and `Y` type parameters must implement
/// `ChernoffFunction<F, S = GridFn1D<F>> + Clone`.
///
/// Implements `ChernoffFunction<F, S = GridFn2D<F>>` with `order = 2` (canonical
/// Strang global order for separable `L`).
///
/// When the `parallel` feature is enabled, `apply` dispatches to an 8-thread
/// palindromic Strang kernel via `std::thread::scope` (ADR-0018); the serial and
/// parallel paths return **bit-identical** results.
///
/// ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 4)
///
/// `Strang2D<X, Y, F: SemiflowFloat = f64>` — the `= f64` default keeps all
/// existing call-sites compiling unchanged.
///
/// # Example
///
/// ```rust
/// use semiflow_core::{
///     chernoff::ApplyChernoffExt,
///     Grid1D, Grid2D, GridFn2D, DiffusionChernoff, Strang2D, AxisLift, Axis,
/// };
/// let gx = Grid1D::new(-4.0, 4.0, 32).unwrap();
/// let gy = Grid1D::new(-4.0, 4.0, 32).unwrap();
/// let grid2 = Grid2D::new(gx, gy);
/// let dx = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx);
/// let dy = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gy);
/// let strang2d = Strang2D::new(dx, dy);
/// let u0 = GridFn2D::from_fn(grid2, |x, y| (-(x * x + y * y)).exp());
/// let u1 = strang2d.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 32 * 32);
/// ```
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `Strang2D`.
#[derive(Clone, Debug)]
pub struct Strang2D<X, Y, F: SemiflowFloat = f64> {
    /// X-axis lift (`AxisLift::new(_, Axis::X)`). Applied at half-step twice.
    pub x: AxisLift<X, F>,
    /// Y-axis lift (`AxisLift::new(_, Axis::Y)`). Applied at full step once.
    pub y: AxisLift<Y, F>,
}

impl<X, Y, F: SemiflowFloat> Strang2D<X, Y, F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// Construct a `Strang2D` from the two inner 1D Chernoff functions.
    ///
    /// No validation — both `x_inner` and `y_inner` are already valid
    /// `ChernoffFunction<F>`s.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Strang2D::new`.
    #[must_use]
    pub fn new(x_inner: X, y_inner: Y) -> Self {
        Self {
            x: AxisLift::new(x_inner, Axis::X),
            y: AxisLift::new(y_inner, Axis::Y),
        }
    }
}

// ---------------------------------------------------------------------------
// Serial ChernoffFunction impl (no `parallel` feature)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "parallel"))]
impl<X, Y, F: SemiflowFloat> ChernoffFunction<F> for Strang2D<X, Y, F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    type S = GridFn2D<F>;

    /// Wave 2 (ADR-0042): 3-leg palindromic ping-pong, 0 allocs in steady state.
    // (serial impl — no feature guard needed)
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn2D<F>,
        dst: &mut GridFn2D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        apply_strang2d_into(tau, src, dst, &self.x, &self.y, scratch)
    }

    fn order(&self) -> u32 {
        serial_order(&self.x, &self.y)
    }

    fn growth(&self) -> Growth<F> {
        serial_growth(&self.x, &self.y)
    }
}

// ---------------------------------------------------------------------------
// Parallel ChernoffFunction impl (`parallel` feature enabled)
// ---------------------------------------------------------------------------

// Parallel impl — generic over F: SemiflowFloat + Send + Sync + ParallelPool2D
// (ADR-0045 Wave 5). SIMD stays f64-specialised inside strang2d_parallel.rs;
// the type signature change here is zero-cost (monomorphisation).
// f64 codegen path is unchanged; f64 bit-equality contract (ADR-0018) preserved.
// `private_bounds`: ParallelPool2D is a sealed pub(crate) trait — downstream
// crates cannot name it; the bound is merely an implementation detail.
#[cfg(feature = "parallel")]
#[allow(private_bounds)]
impl<X, Y, F> ChernoffFunction<F> for Strang2D<X, Y, F>
where
    F: SemiflowFloat + Send + Sync + ParallelPool2D,
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    type S = GridFn2D<F>;

    /// Parallel palindromic ping-pong — parallel X/Y passes wire into evolve.
    ///
    /// Delegates to `apply_parallel_into` so that `ChernoffSemigroup::evolve`
    /// benefits from multi-threading. Small grids fall back to the serial
    /// scratch-pool path (zero allocation in steady state).
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn2D<F>,
        dst: &mut GridFn2D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.apply_parallel_into(tau, src, dst, scratch)
    }

    fn order(&self) -> u32 {
        serial_order(&self.x, &self.y)
    }

    fn growth(&self) -> Growth<F> {
        serial_growth(&self.x, &self.y)
    }
}

// ---------------------------------------------------------------------------
// Shared order/growth helpers (both impls use the same logic)
// ---------------------------------------------------------------------------

/// Canonical Strang τ-axis order: `min(X.order(), Y.order(), 4)`.
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `Strang2D::order`
/// and math.md §11.1.bis.
fn serial_order<X, Y, F: SemiflowFloat>(x: &AxisLift<X, F>, y: &AxisLift<Y, F>) -> u32
where
    X: ChernoffFunction<F, S = GridFn1D<F>>,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
{
    x.order().min(y.order()).min(4)
}

/// Composed growth bound `(M_x² · M_y, ω_x + ω_y)`.
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `Strang2D::growth`.
fn serial_growth<X, Y, F: SemiflowFloat>(x: &AxisLift<X, F>, y: &AxisLift<Y, F>) -> Growth<F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>>,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let gx = x.growth();
    let gy = y.growth();
    Growth {
        multiplier: gx.multiplier * gy.multiplier * gx.multiplier,
        omega: gx.omega + gy.omega,
    }
}

// ---------------------------------------------------------------------------
// Serial apply helper (v0.7.0 body — no content change)
// ---------------------------------------------------------------------------

impl<X, Y, F: SemiflowFloat> Strang2D<X, Y, F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// Serial palindromic Strang sandwich:
    /// `f3 = X(τ/2) ( Y(τ) ( X(τ/2) f ) )`.
    ///
    /// Identical to the v0.7.0 `apply` body. Called by both the serial impl
    /// (all builds) and the parallel impl (small-grid fallback).
    /// `pub(crate)` so `nonseparable2d` can call it directly without the
    /// `Send + Sync` bounds required by the parallel `ChernoffFunction` impl.
    // called from strang3d_helpers.rs apply_parallel fallback and nonseparable_mixed
    #[allow(dead_code)]
    pub(crate) fn apply_serial(
        &self,
        tau: F,
        f: &GridFn2D<F>,
    ) -> Result<GridFn2D<F>, SemiflowError> {
        apply_strang2d_full(tau, f, &self.x, &self.y)
    }

    /// Fused-axis path: `Y(τ) ∘ X(τ)` — 2 passes instead of 3.
    ///
    /// **ABORTED in ADR-0039 v0.13.0 Wave C**: experimental evidence shows that
    /// `‖fused(τ) − palindromic(τ)‖` scales as O(τ¹·¹), NOT O(τ²).  The
    /// palindromic structure `X(τ/2) ∘ Y(τ) ∘ X(τ/2)` is essential for
    /// achieving second-order accuracy with `DiffusionChernoff` even though
    /// `[L_x, L_y] = 0`; the half-step composition changes the FP error
    /// structure in a way that the theoretical BCH cancellation does not capture
    /// numerically.  This method is retained here solely to allow the
    /// `strang_fused_order_confirmation_2d` test to document the finding.
    ///
    /// **Do NOT wire into `apply_serial` or any production path** without
    /// re-running the slope gate `STRANG_FUSED_TAU2_PRESERVATION` (ADR-0039).
    // used by strang_fused_order_confirmation_2d test (documents aborted experiment)
    #[allow(dead_code)]
    fn apply_fused(&self, tau: F, f: &GridFn2D<F>) -> Result<GridFn2D<F>, SemiflowError> {
        use crate::chernoff::ApplyChernoffExt;
        let f1 = self.x.apply_chernoff(tau, f)?;
        let f2 = self.y.apply_chernoff(tau, &f1)?;
        Ok(f2)
    }
}

/// 3-pass palindromic Strang helper (shared between serial and fused dispatch).
// called from apply_serial and tests; lint fires when feature = "parallel" is on
#[allow(dead_code)]
fn apply_strang2d_full<X, Y, F: SemiflowFloat>(
    tau: F,
    f: &GridFn2D<F>,
    x: &crate::axis::AxisLift<X, F>,
    y: &crate::axis::AxisLift<Y, F>,
) -> Result<GridFn2D<F>, SemiflowError>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    use crate::chernoff::ApplyChernoffExt;
    let half_tau = half::<F>() * tau;
    let f1 = x.apply_chernoff(half_tau, f)?;
    let f2 = y.apply_chernoff(tau, &f1)?;
    let f3 = x.apply_chernoff(half_tau, &f2)?;
    Ok(f3)
}

// ---------------------------------------------------------------------------
// Wave 2 (ADR-0042): 3-leg ping-pong apply_into helper
// ---------------------------------------------------------------------------

/// 3-leg palindromic ping-pong via `ScratchPool`.
///
/// Buffer parity rule (ADR-0042 §2):
///
/// - `buf_a` = copy of `src.values`; `buf_b` = empty scratch
/// - Leg 1 X(τ/2): A→B, swap → A=leg1
/// - Leg 2 Y(τ):   A→B, swap → A=leg2
/// - Leg 3 X(τ/2): A→B (no swap) → final in B
///
/// Copy B → `dst.values`.
#[inline]
pub(crate) fn apply_strang2d_into<X, Y, F: SemiflowFloat>(
    tau: F,
    src: &GridFn2D<F>,
    dst: &mut GridFn2D<F>,
    x: &AxisLift<X, F>,
    y: &AxisLift<Y, F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    let n = src.values.len();
    let half_tau = half::<F>() * tau;
    if dst.values.len() != n {
        dst.values.resize(n, F::zero());
    }
    dst.grid = src.grid;

    let mut buf_a = scratch.take_vec(n);
    buf_a.copy_from_slice(&src.values);
    let mut buf_b = scratch.take_vec(n);

    run_axislift_into_2d(x, half_tau, &buf_a, &mut buf_b, src.grid, scratch)?;
    core::mem::swap(&mut buf_a, &mut buf_b);

    run_axislift_into_2d(y, tau, &buf_a, &mut buf_b, src.grid, scratch)?;
    core::mem::swap(&mut buf_a, &mut buf_b);

    // Leg 3: A→B, no swap — final result in buf_b.
    run_axislift_into_2d(x, half_tau, &buf_a, &mut buf_b, src.grid, scratch)?;

    dst.values.copy_from_slice(&buf_b);
    scratch.return_vec(buf_a);
    scratch.return_vec(buf_b);
    Ok(())
}

/// Wrap flat buffers in transient `GridFn2D` views and call `lift.apply_into`.
#[inline]
fn run_axislift_into_2d<C, F: SemiflowFloat>(
    lift: &AxisLift<C, F>,
    tau: F,
    src_buf: &[F],
    dst_buf: &mut Vec<F>,
    grid: crate::grid2d::Grid2D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    let mut src_vals = scratch.take_vec(src_buf.len());
    src_vals.copy_from_slice(src_buf);
    let src_gf = GridFn2D {
        values: src_vals,
        grid,
    };
    let dst_vals = core::mem::take(dst_buf);
    let mut dst_gf = GridFn2D {
        values: dst_vals,
        grid,
    };
    lift.apply_into(tau, &src_gf, &mut dst_gf, scratch)?;
    *dst_buf = dst_gf.values;
    scratch.return_vec(src_gf.values);
    Ok(())
}

// ---------------------------------------------------------------------------
// Parallel apply helper — only compiled with `parallel` feature
// ---------------------------------------------------------------------------

// Parallel apply — generic over F: SemiflowFloat + Send + Sync + ParallelPool2D.
// SIMD specialisation lives inside strang2d_parallel.rs (parallel_x_pass /
// parallel_y_pass dispatch on F at the call site). ADR-0045 Wave 5.
#[cfg(feature = "parallel")]
#[allow(private_bounds)]
impl<X, Y, F> Strang2D<X, Y, F>
where
    F: SemiflowFloat + Send + Sync + ParallelPool2D,
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    /// Parallel palindromic Strang — allocation-free ping-pong path.
    ///
    /// Used by `ChernoffSemigroup::evolve` so that multi-step
    /// integration benefits from the parallel X/Y passes.
    ///
    /// For grids smaller than `2 * MIN_ROWS_PER_THREAD` rows, falls back to
    /// the serial scratch-pool path (zero allocation in steady state).
    /// For larger grids, allocates `y_scratch` locally (one Vec per call).
    fn apply_parallel_into(
        &self,
        tau: F,
        src: &GridFn2D<F>,
        dst: &mut GridFn2D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let ny = src.grid.ny();

        if ny < 2 * MIN_ROWS_PER_THREAD {
            // Serial fallback — zero-alloc steady state via scratch pool.
            return apply_strang2d_into(tau, src, dst, &self.x, &self.y, scratch);
        }

        let n_threads = resolve_threads(ny);
        let half_tau = half::<F>() * tau;
        let nx = src.grid.nx();
        let mut state = src.values.clone();
        let mut y_scratch: Vec<F> = Vec::with_capacity(nx * ny);

        parallel_x_pass(&mut state, src.grid.x, n_threads, &self.x.inner, half_tau)?;
        parallel_y_pass(
            &mut state,
            src.grid.y,
            n_threads,
            &self.y.inner,
            tau,
            &mut y_scratch,
        )?;
        parallel_x_pass(&mut state, src.grid.x, n_threads, &self.x.inner, half_tau)?;

        dst.values.resize(state.len(), F::zero());
        dst.values.copy_from_slice(&state);
        dst.grid = src.grid;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests — extracted to sibling file per ≤500-line cap.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "strang2d_tests.rs"]
mod tests;
