//! τ-keyed step plans (ADR-0215): the header every plan carries, and the consumer
//! helper that applies one engine with one `τ` to many states.
//!
//! A plan (see `ChernoffFunction::plan_len`) caches what an engine recomputes on
//! every step although it depends only on `(τ, grid, coefficients)`. Its header
//! records the engine's address, `τ`, the state length and the sampling grid
//! (bit patterns); a plan presented with anything else is not used — the engine
//! falls back to `apply_into`, so a plan can only make a step faster, never
//! different.

use alloc::vec::Vec;

use crate::{
    boundary::{BoundaryPolicy, InterpKind, OobPolicy},
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

/// Header length of an `f64` grid plan.
pub(crate) const HEADER: usize = 12;

/// `[tag, a, b]` of a boundary policy.
fn boundary_code(b: BoundaryPolicy<f64>) -> [f64; 3] {
    match b {
        BoundaryPolicy::Reflect => [0.0, 0.0, 0.0],
        BoundaryPolicy::ZeroExtend => [1.0, 0.0, 0.0],
        BoundaryPolicy::Periodic => [2.0, 0.0, 0.0],
        BoundaryPolicy::LinearExtrapolate => [3.0, 0.0, 0.0],
        BoundaryPolicy::Dirichlet { value } => [4.0, value, 0.0],
        BoundaryPolicy::Neumann => [5.0, 0.0, 0.0],
        BoundaryPolicy::Robin { alpha, beta } => [6.0, alpha, beta],
        BoundaryPolicy::OddReflect => [7.0, 0.0, 0.0],
    }
}

/// `[tag, m·4 + oob]` of an interpolant.
#[allow(clippy::cast_precision_loss)] // m ≤ 512
fn interp_code(k: InterpKind) -> [f64; 2] {
    match k {
        InterpKind::SepticHermite => [0.0, 0.0],
        InterpKind::OctonicHermite => [1.0, 0.0],
        InterpKind::CubicHermite => [2.0, 0.0],
        InterpKind::Linear => [3.0, 0.0],
        InterpKind::ChebyshevSpectralWithBC { m, oob_policy } => {
            let oob = match oob_policy {
                OobPolicy::Inherit => 0.0,
                OobPolicy::ForceReflect => 1.0,
                OobPolicy::ForcePeriodic => 2.0,
                OobPolicy::ForceZero => 3.0,
            };
            [4.0, m as f64 * 4.0 + oob]
        }
    }
}

/// The header of a plan by `engine` for `τ` and states of length `n` on `grid`.
#[allow(clippy::cast_precision_loss)] // addresses and lengths below 2^53
pub(crate) fn header<E: ?Sized>(
    engine: &E,
    tag: f64,
    tau: f64,
    n: usize,
    grid: &Grid1D,
) -> [f64; HEADER] {
    let addr = core::ptr::from_ref(engine).cast::<u8>() as usize as f64;
    let [bt, ba, bb] = boundary_code(grid.boundary);
    let [it, im] = interp_code(grid.interp);
    [
        tag,
        tau,
        n as f64,
        addr,
        grid.xmin,
        grid.xmax,
        grid.n as f64,
        bt,
        ba,
        bb,
        it,
        im,
    ]
}

/// `true` when `data` starts with exactly `want` (bit patterns).
pub(crate) fn header_matches(data: &[f64], want: &[f64; HEADER]) -> bool {
    data.len() >= HEADER
        && data[..HEADER]
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Applies one engine with one `τ` to a sequence of states (steps, pencils),
/// building the engine's plan on the first state (buffers from the scratch pool:
/// zero allocations in steady state). Without a plan it is plain `apply_into`.
pub(crate) struct PencilPlan<F> {
    buf: Vec<F>,
    split: usize,
    state: Built,
}

#[derive(Clone, Copy, PartialEq)]
enum Built {
    Not,
    Absent,
    Ready,
}

impl<F: SemiflowFloat> PencilPlan<F> {
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::new(),
            split: 0,
            state: Built::Not,
        }
    }

    /// `dst := S(τ) src`, planned when the engine supports it.
    ///
    /// # Errors
    /// As `inner.apply_into` (a plan-building error is the error `apply_into`
    /// raises for the same `τ` and grid).
    pub(crate) fn apply<C: ChernoffFunction<F> + ?Sized>(
        &mut self,
        inner: &C,
        tau: F,
        src: &C::S,
        dst: &mut C::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if self.state == Built::Not {
            self.state = Built::Absent;
            if let Some((data, work)) = inner.plan_len(tau, src) {
                self.buf = scratch.take_vec(data + work);
                self.split = data;
                self.state = Built::Ready;
                inner.fill_plan(tau, src, &mut self.buf[..data])?;
            }
        }
        if self.state == Built::Ready {
            let (data, work) = self.buf.split_at_mut(self.split);
            inner.apply_planned_into(tau, data, work, src, dst, scratch)
        } else {
            inner.apply_into(tau, src, dst, scratch)
        }
    }

    /// [`apply`](Self::apply) on a pencil held in slices (pool buffers; mirrors
    /// `grid_fn::apply_into_via_view`).
    ///
    /// # Errors
    /// As [`apply`](Self::apply).
    pub(crate) fn apply_slices<C>(
        &mut self,
        inner: &C,
        tau: F,
        (src_slot, dst_slot): (&[F], &mut [F]),
        grid: Grid1D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>
    where
        C: ChernoffFunction<F, S = GridFn1D<F>> + ?Sized,
    {
        let n = src_slot.len();
        let mut values = scratch.take_vec(n);
        values.copy_from_slice(src_slot);
        let src_gf = GridFn1D { values, grid };
        let mut dst_gf = GridFn1D {
            values: scratch.take_vec(n),
            grid,
        };
        self.apply(inner, tau, &src_gf, &mut dst_gf, scratch)?;
        dst_slot.copy_from_slice(&dst_gf.values);
        scratch.return_vec(src_gf.values);
        scratch.return_vec(dst_gf.values);
        Ok(())
    }

    /// Return the plan buffer to `scratch`.
    pub(crate) fn release(self, scratch: &mut ScratchPool<F>) {
        if self.state == Built::Ready {
            scratch.return_vec(self.buf);
        }
    }
}

/// The shared plan of one parallel axis sweep: `inner` with `τ` on pencils of
/// `grid` (a zero proto pencil fixes the shape).
///
/// # Errors
/// As `inner.fill_plan`.
#[cfg(feature = "parallel")]
pub(crate) fn axis_plan<F: SemiflowFloat, C>(
    inner: &C,
    tau: F,
    grid: Grid1D<F>,
) -> Result<Option<(Vec<F>, usize)>, SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>> + ?Sized,
{
    let proto = GridFn1D {
        values: alloc::vec![F::zero(); grid.n],
        grid,
    };
    shared_plan(inner, tau, &proto)
}

#[cfg(feature = "parallel")]
/// A plan shared read-only by worker threads: `(data, work length)`, or `None`.
///
/// # Errors
/// As `inner.fill_plan`.
pub(crate) fn shared_plan<F: SemiflowFloat, C: ChernoffFunction<F> + ?Sized>(
    inner: &C,
    tau: F,
    proto: &C::S,
) -> Result<Option<(Vec<F>, usize)>, SemiflowError> {
    let Some((data, work)) = inner.plan_len(tau, proto) else {
        return Ok(None);
    };
    let mut buf = alloc::vec![F::zero(); data];
    inner.fill_plan(tau, proto, &mut buf)?;
    Ok(Some((buf, work)))
}

/// Per-thread pencil applier of the parallel sweeps: the shared plan with a
/// thread-local work area and output buffer, or (no plan) today's allocating
/// `apply_chernoff` path.
#[cfg(feature = "parallel")]
pub(crate) struct ThreadPencil<'p, F: SemiflowFloat> {
    plan: Option<(&'p [F], Vec<F>)>,
    pool: ScratchPool<F>,
    out: Vec<F>,
}

#[cfg(feature = "parallel")]
impl<'p, F: SemiflowFloat> ThreadPencil<'p, F> {
    /// From the pass's shared plan `(data, work length)`, if any.
    pub(crate) fn new(plan: Option<&'p (Vec<F>, usize)>) -> Self {
        Self {
            plan: plan.map(|(data, work)| (data.as_slice(), alloc::vec![F::zero(); *work])),
            pool: ScratchPool::new(),
            out: Vec::new(),
        }
    }

    /// `S(τ) src`, borrowed until the next call.
    ///
    /// # Errors
    /// As `op.apply_into`.
    pub(crate) fn apply<C>(
        &mut self,
        op: &C,
        tau: F,
        src: &GridFn1D<F>,
    ) -> Result<&[F], SemiflowError>
    where
        C: ChernoffFunction<F, S = GridFn1D<F>> + ?Sized,
    {
        if let Some((data, work)) = &mut self.plan {
            let mut dst = GridFn1D {
                values: core::mem::take(&mut self.out),
                grid: src.grid,
            };
            let r = op.apply_planned_into(tau, data, work, src, &mut dst, &mut self.pool);
            self.out = dst.values;
            r?;
        } else {
            let mut dst = src.clone();
            op.apply_into(tau, src, &mut dst, &mut self.pool)?;
            self.out = dst.values;
        }
        Ok(&self.out)
    }
}
