//! [`Strang3D`] palindromic composition for 3D tensor-product generators.
//!
//! [`AxisLift3D`] was split to `strang3d_axislift.rs` in Wave 2 (ADR-0042)
//! to keep this file within the 700-LoC constitution carve-out.
//!
//! # `Strang3D<X, Y, Z, F>`
//!
//! Palindromic 5-leg composition for separable `L = A + B + C` where
//! A, B, C act on disjoint tensor factors (Lemma 10.1 of math.md §10.8.1):
//!
//! ```text
//! Strang3D(τ) = X(τ/2) ∘ Y(τ/2) ∘ Z(τ) ∘ Y(τ/2) ∘ X(τ/2)
//! ```
//!
//! Because `[A,B] = [A,C] = [B,C] = 0`, the BCH residue is exactly zero and
//! the composition reduces to `e^{τ(A+B+C)}` (math.md §10.8.3, Theorem 7').
//! Global order = `min(order(X), order(Y), order(Z))`.
//!
//! ## Wave 2 (ADR-0042): `apply_into` override
//!
//! `Strang3D::apply_into` runs the 5-leg palindromic ping-pong on two
//! `Vec<F>` buffers borrowed from `ScratchPool<F>`.  In steady state,
//! 0 allocations per step.  Palindromic order is preserved verbatim.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 4)
//!
//! `Strang3D<X, Y, Z, F: SemiflowFloat = f64>` — the `= f64` default keeps
//! all existing call-sites compiling unchanged.
//!
//! See `contracts/semiflow-core.tensor.yaml` §3 `Strang3D`,
//! `docs/adr/0024-tensor-3d.md`, `docs/adr/0042-inplace-strang-pencil-pingpong.md`,
//! and `contracts/semiflow-core.math.md` §10.8.
//!
//! When the `parallel` feature is enabled, `Strang3D::apply` (f64 only)
//! dispatches to `strang3d_parallel`, mirroring the `Strang2D` parallel path
//! (ADR-0018).  The serial and parallel paths return **bit-identical** results
//! (see `tests/strang3d_parallel_bit_equal.rs`).

#[cfg(feature = "parallel")]
use crate::parallel_pool::ParallelPool3D;
#[cfg(feature = "parallel")]
use crate::strang3d_parallel::{
    parallel_x_pass_3d, parallel_y_pass_3d, parallel_z_pass_3d, resolve_threads_3d,
    MIN_PENCILS_PER_THREAD,
};
use alloc::vec::Vec;

use crate::{
    axis::Axis,
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{half, SemiflowFloat},
    grid_fn::GridFn1D,
    grid_fn3d::GridFn3D,
    scratch::ScratchPool,
    strang3d_axislift::AxisLift3D,
};

// ---------------------------------------------------------------------------
// Strang3D
// ---------------------------------------------------------------------------

/// Palindromic 5-leg [`Strang3D`] composition for 3D tensor-product generators.
///
/// Composition:
/// `Strang3D(τ) = X(τ/2) ∘ Y(τ/2) ∘ Z(τ) ∘ Y(τ/2) ∘ X(τ/2)`.
///
/// Because `[A, B] = [A, C] = [B, C] = 0` for separable generators, the BCH
/// residue is exactly zero (math.md §10.8.3, Theorem 7'): the composition
/// reduces to `e^{τ(A+B+C)}` exactly. Global order equals
/// `min(order(X), order(Y), order(Z))`.
///
/// All three type parameters `X`, `Y`, `Z` must implement
/// `ChernoffFunction<F, S = GridFn1D<F>> + Clone`.
///
/// Implements `ChernoffFunction<F, S = GridFn3D<F>>`.
///
/// ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 4)
///
/// `Strang3D<X, Y, Z, F: SemiflowFloat = f64>` — the `= f64` default keeps all
/// existing call-sites compiling unchanged.
///
/// # Example
///
/// ```rust
/// use semiflow_core::{
///     chernoff::ApplyChernoffExt,
///     Grid1D, Grid3D, GridFn3D, DiffusionChernoff, Strang3D,
/// };
/// let g = Grid1D::new(-2.0, 2.0, 16).unwrap();
/// let d = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, g);
/// let s3d = Strang3D::new(d.clone(), d.clone(), d);
/// let grid3 = Grid3D::new(g, g, g).unwrap();
/// let u0 = GridFn3D::from_fn(grid3, |x, y, z| (-(x*x + y*y + z*z)).exp());
/// let u1 = s3d.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 16 * 16 * 16);
/// ```
///
/// See `contracts/semiflow-core.tensor.yaml` §3 `Strang3D`.
#[derive(Clone, Debug)]
pub struct Strang3D<X, Y, Z, F: SemiflowFloat = f64> {
    /// X-axis lift. Applied at half-step twice (outermost).
    pub x: AxisLift3D<X, F>,
    /// Y-axis lift. Applied at half-step twice (middle).
    pub y: AxisLift3D<Y, F>,
    /// Z-axis lift. Applied at full step once (innermost).
    pub z: AxisLift3D<Z, F>,
}

impl<X, Y, Z, F: SemiflowFloat> Strang3D<X, Y, Z, F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Z: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// Construct a `Strang3D<X, Y, Z, F>` from three inner 1D Chernoff functions.
    ///
    /// No validation — all three must already be valid `ChernoffFunction<F>`s.
    ///
    /// See `contracts/semiflow-core.tensor.yaml` §3 `Strang3D::new`.
    #[must_use]
    pub fn new(x_inner: X, y_inner: Y, z_inner: Z) -> Self {
        Self {
            x: AxisLift3D::new(x_inner, Axis::X),
            y: AxisLift3D::new(y_inner, Axis::Y),
            z: AxisLift3D::new(z_inner, Axis::Z),
        }
    }
}

// ---------------------------------------------------------------------------
// Serial ChernoffFunction impl (no `parallel` feature)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "parallel"))]
impl<X, Y, Z, F: SemiflowFloat> ChernoffFunction<F> for Strang3D<X, Y, Z, F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Z: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    type S = GridFn3D<F>;

    /// Wave 2 (ADR-0042): 5-leg palindromic ping-pong, 0 allocs in steady state.
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn3D<F>,
        dst: &mut GridFn3D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        apply_strang3d_into(tau, src, dst, &self.x, &self.y, &self.z, scratch)
    }

    fn order(&self) -> u32 {
        strang3d_order(&self.x, &self.y, &self.z)
    }

    fn growth(&self) -> Growth<F> {
        strang3d_growth(&self.x, &self.y, &self.z)
    }
}

// ---------------------------------------------------------------------------
// Parallel ChernoffFunction impl (`parallel` feature, generic over F)
// ---------------------------------------------------------------------------

// Generic over F: SemiflowFloat + Send + Sync + ParallelPool3D (ADR-0045 Wave 5).
// f64 codegen path is unchanged; f64 bit-equality contract (ADR-0018) preserved.
// `private_bounds`: ParallelPool3D is sealed pub(crate); downstream cannot name it.
#[cfg(feature = "parallel")]
#[allow(private_bounds)]
impl<X, Y, Z, F> ChernoffFunction<F> for Strang3D<X, Y, Z, F>
where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Z: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    type S = GridFn3D<F>;

    /// Parallel palindromic ping-pong — parallel X/Y/Z passes wire into evolve.
    ///
    /// Delegates to `apply_parallel_into` so that `ChernoffSemigroup::evolve`
    /// benefits from multi-threading. Small grids fall back to the serial
    /// scratch-pool path (zero allocation in steady state).
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn3D<F>,
        dst: &mut GridFn3D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.apply_parallel_into(tau, src, dst, scratch)
    }

    fn order(&self) -> u32 {
        strang3d_order(&self.x, &self.y, &self.z)
    }

    fn growth(&self) -> Growth<F> {
        strang3d_growth(&self.x, &self.y, &self.z)
    }
}

// ---------------------------------------------------------------------------
// Shared order/growth helpers
// ---------------------------------------------------------------------------

fn strang3d_order<X, Y, Z, F: SemiflowFloat>(
    x: &AxisLift3D<X, F>,
    y: &AxisLift3D<Y, F>,
    z: &AxisLift3D<Z, F>,
) -> u32
where
    X: ChernoffFunction<F, S = GridFn1D<F>>,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
    Z: ChernoffFunction<F, S = GridFn1D<F>>,
{
    x.order().min(y.order()).min(z.order())
}

fn strang3d_growth<X, Y, Z, F: SemiflowFloat>(
    x: &AxisLift3D<X, F>,
    y: &AxisLift3D<Y, F>,
    z: &AxisLift3D<Z, F>,
) -> Growth<F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>>,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
    Z: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let gx = x.growth();
    let gy = y.growth();
    let gz = z.growth();
    Growth {
        multiplier: gx.multiplier * gx.multiplier * gy.multiplier * gy.multiplier * gz.multiplier,
        omega: gx.omega + gy.omega + gz.omega,
    }
}

// ---------------------------------------------------------------------------
// Serial apply helper (ADR-0022 Amendment 1 scratch-pool path)
// ---------------------------------------------------------------------------

impl<X, Y, Z, F: SemiflowFloat> Strang3D<X, Y, Z, F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Z: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// Serial palindromic 5-leg Strang sandwich (scratch-pool path).
    ///
    /// `f5 = X(τ/2) ( Y(τ/2) ( Z(τ) ( Y(τ/2) ( X(τ/2) f ) ) ) )`.
    ///
    /// `pub(crate)` to allow tests to call it directly.
    // called from strang3d_helpers.rs apply_parallel_into fallback and tests
    #[allow(dead_code)]
    pub(crate) fn apply_serial(
        &self,
        tau: F,
        f: &GridFn3D<F>,
    ) -> Result<GridFn3D<F>, SemiflowError> {
        apply_strang3d_full(tau, f, &self.x.inner, &self.y.inner, &self.z.inner)
    }

    /// Fused-axis path: `Z(τ) ∘ Y(τ) ∘ X(τ)` — 3 passes instead of 5.
    ///
    /// **ABORTED in ADR-0039**: O(τ¹) difference vs palindromic, NOT O(τ²).
    /// Retained solely for the `strang_fused_order_confirmation_3d` test.
    // used by strang3d_tests.rs (documents aborted experiment, ADR-0039)
    #[allow(dead_code)]
    fn apply_fused(&self, tau: F, f: &GridFn3D<F>) -> Result<GridFn3D<F>, SemiflowError> {
        let mut buf = f.clone();
        serial_x_pass_inplace(&self.x.inner, tau, &mut buf)?;
        serial_y_pass_inplace(&self.y.inner, tau, &mut buf)?;
        serial_z_pass_inplace(&self.z.inner, tau, &mut buf)?;
        Ok(buf)
    }
}

/// 5-pass palindromic Strang helper.
// called from apply_serial and strang3d_tests.rs
#[allow(dead_code)]
pub(crate) fn apply_strang3d_full<X, Y, Z, F: SemiflowFloat>(
    tau: F,
    f: &GridFn3D<F>,
    x: &X,
    y: &Y,
    z: &Z,
) -> Result<GridFn3D<F>, SemiflowError>
where
    X: ChernoffFunction<F, S = GridFn1D<F>>,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
    Z: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let half_tau = half::<F>() * tau;
    let mut buf = f.clone();
    serial_x_pass_inplace(x, half_tau, &mut buf)?;
    serial_y_pass_inplace(y, half_tau, &mut buf)?;
    serial_z_pass_inplace(z, tau, &mut buf)?;
    serial_y_pass_inplace(y, half_tau, &mut buf)?;
    serial_x_pass_inplace(x, half_tau, &mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Wave 2 (ADR-0042): 5-leg ping-pong apply_into helper
// ---------------------------------------------------------------------------

/// 5-leg palindromic ping-pong via `ScratchPool`.
///
/// Buffer parity rule (ADR-0042 §3):
///
/// - Start: `buf_a` = `src.values` copy, `buf_b` = empty scratch
/// - Leg 1 X(τ/2): A→B, swap → A=leg1, B=stale
/// - Leg 2 Y(τ/2): A→B, swap → A=leg2, B=stale
/// - Leg 3 Z(τ):   A→B, swap → A=leg3, B=stale
/// - Leg 4 Y(τ/2): A→B, swap → A=leg4, B=stale
/// - Leg 5 X(τ/2): A→B (no swap) → final result in B
///
/// Copy B → `dst.values`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_strang3d_into<X, Y, Z, F: SemiflowFloat>(
    tau: F,
    src: &GridFn3D<F>,
    dst: &mut GridFn3D<F>,
    x: &AxisLift3D<X, F>,
    y: &AxisLift3D<Y, F>,
    z: &AxisLift3D<Z, F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    X: ChernoffFunction<F, S = GridFn1D<F>>,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
    Z: ChernoffFunction<F, S = GridFn1D<F>>,
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

    // Legs 1-4: A→B then swap.
    run_lift_into(x, half_tau, &buf_a, &mut buf_b, src.grid, scratch)?;
    core::mem::swap(&mut buf_a, &mut buf_b);

    run_lift_into(y, half_tau, &buf_a, &mut buf_b, src.grid, scratch)?;
    core::mem::swap(&mut buf_a, &mut buf_b);

    run_lift_into(z, tau, &buf_a, &mut buf_b, src.grid, scratch)?;
    core::mem::swap(&mut buf_a, &mut buf_b);

    run_lift_into(y, half_tau, &buf_a, &mut buf_b, src.grid, scratch)?;
    core::mem::swap(&mut buf_a, &mut buf_b);

    // Leg 5: A→B, no swap — final result in buf_b.
    run_lift_into(x, half_tau, &buf_a, &mut buf_b, src.grid, scratch)?;

    dst.values.copy_from_slice(&buf_b);
    scratch.return_vec(buf_a);
    scratch.return_vec(buf_b);
    Ok(())
}

/// Helper: wrap flat buffers in transient `GridFn3D` views and call `lift.apply_into`.
///
/// `src_buf` is the read-only source. `dst_buf` is an owned `Vec<F>` that will
/// receive the output (reused across legs to avoid per-leg allocation).
///
/// We borrow `src_vals` from the scratch pool, copy `src_buf` into it, then
/// call `apply_into`. After the call we return `src_vals` to the pool. The
/// `dst_buf` is moved into/out of the `GridFn3D` wrapper via `mem::take` —
/// no extra allocation.
fn run_lift_into<C, F: SemiflowFloat>(
    lift: &AxisLift3D<C, F>,
    tau: F,
    src_buf: &[F],
    dst_buf: &mut Vec<F>,
    grid: crate::grid3d::Grid3D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    // Take a pool buffer for src, copy contents, then release after the call.
    let mut src_vals = scratch.take_vec(src_buf.len());
    src_vals.copy_from_slice(src_buf);
    // Build src_gf owning src_vals (NOT in pool — safe to pass scratch below).
    let src_gf = GridFn3D {
        values: src_vals,
        grid,
    };
    // Take ownership of dst_buf to avoid a second pool borrow.
    let dst_vals = core::mem::take(dst_buf);
    let mut dst_gf = GridFn3D {
        values: dst_vals,
        grid,
    };
    lift.apply_into(tau, &src_gf, &mut dst_gf, scratch)?;
    // Restore ownership.
    *dst_buf = dst_gf.values;
    scratch.return_vec(src_gf.values);
    Ok(())
}

// ---------------------------------------------------------------------------
// In-place serial pass helpers (ADR-0022 Amendment 1)
// ---------------------------------------------------------------------------

fn serial_x_pass_inplace<C, F: SemiflowFloat>(
    inner: &C,
    tau: F,
    buf: &mut GridFn3D<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    use crate::chernoff::ApplyChernoffExt;
    let ny = buf.grid.ny();
    let nz = buf.grid.nz();
    for k in 0..nz {
        for j in 0..ny {
            let pencil = buf.pencil_x_generic(j, k);
            let evolved = inner.apply_chernoff(tau, &pencil)?;
            buf.write_pencil_x_generic(j, k, &evolved)?;
        }
    }
    Ok(())
}

fn serial_y_pass_inplace<C, F: SemiflowFloat>(
    inner: &C,
    tau: F,
    buf: &mut GridFn3D<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    use crate::chernoff::ApplyChernoffExt;
    let nx = buf.grid.nx();
    let nz = buf.grid.nz();
    for k in 0..nz {
        for i in 0..nx {
            let pencil = buf.pencil_y_generic(i, k);
            let evolved = inner.apply_chernoff(tau, &pencil)?;
            buf.write_pencil_y_generic(i, k, &evolved)?;
        }
    }
    Ok(())
}

fn serial_z_pass_inplace<C, F: SemiflowFloat>(
    inner: &C,
    tau: F,
    buf: &mut GridFn3D<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
{
    use crate::chernoff::ApplyChernoffExt;
    let nx = buf.grid.nx();
    let ny = buf.grid.ny();
    for j in 0..ny {
        for i in 0..nx {
            let pencil = buf.pencil_z_generic(i, j);
            let evolved = inner.apply_chernoff(tau, &pencil)?;
            buf.write_pencil_z_generic(i, j, &evolved)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parallel apply helpers — only compiled with `parallel` feature
// ---------------------------------------------------------------------------

include!("strang3d_helpers.rs");

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    include!("strang3d_tests.rs");
}
