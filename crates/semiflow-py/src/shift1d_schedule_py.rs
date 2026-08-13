//! Piecewise-constant coefficient schedules for `Shift1D` (#23, ADR-0192).
//!
//! `Shift1D.evolve_with_time_schedule` takes a **scalar** schedule for `a` only,
//! with `b` and `c` fixed constants for the whole run. Two gaps bite in practice:
//!
//! 1. **No `b`/`c` schedules.** Time-dependent Feynman–Kac killing `c(x, t)` is
//!    the core of optimal-execution policy evaluation — e.g.
//!    `∂_τ u = ½σ²·u_pp − γν(t)(p − ην(t))·u`, where the killing term is linear
//!    in `p` with a time-varying slope.
//! 2. **No space-varying coefficients inside a schedule.** Schedules and
//!    `with_arrays` were mutually exclusive, so a piecewise-constant-in-time,
//!    variable-in-space generator had to be rebuilt from scratch per segment:
//!    a fresh `Shift1D` object plus re-sampled coefficient arrays for every
//!    macro-segment, with the state round-tripping through numpy in between.
//!
//! [`evolve_with_coefficient_schedule`] closes both: each of `a`, `b`, `c` takes
//! a per-segment list whose entries are independently either a scalar or a
//! length-`n` array, and the whole walk runs inside one `py.detach` with the
//! state staying in Rust.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_precision_loss, clippy::needless_pass_by_value)]

use std::sync::Arc;

use pyo3::prelude::*;
use semiflow::{ChernoffFunction, GridFn1D, ScratchPool, ShiftChernoff1D};

use crate::{coeff::interp_catmull_rom_pub, error::new_pyerr};

/// One segment's value for one coefficient: constant in space, or tabulated.
#[derive(Clone)]
pub(crate) enum SegmentCoeff {
    /// Spatially constant over this segment.
    Scalar(f64),
    /// Tabulated on the grid; interpolated with Catmull-Rom, as `with_arrays` does.
    Array(Arc<Vec<f64>>),
}

impl SegmentCoeff {
    /// Sup-norm over the segment, for the growth bound.
    fn norm(&self) -> f64 {
        match self {
            Self::Scalar(v) => v.abs(),
            Self::Array(a) => a.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
        }
    }

    /// Build the `Send + Sync + 'static` coefficient closure for this segment.
    fn closure(&self, xmin: f64, dx: f64) -> impl Fn(f64) -> f64 + Send + Sync + 'static {
        let me = self.clone();
        move |x: f64| match &me {
            SegmentCoeff::Scalar(v) => *v,
            SegmentCoeff::Array(a) => interp_catmull_rom_pub(a, xmin, dx, x),
        }
    }
}

/// Parse one schedule entry: a float, or an array of length `n`.
fn parse_entry(obj: &Bound<'_, PyAny>, n: usize, name: &str) -> PyResult<SegmentCoeff> {
    if let Ok(v) = obj.extract::<f64>() {
        if !v.is_finite() {
            return Err(new_pyerr("NanInf", &format!("{name} entry is NaN or Inf")));
        }
        return Ok(SegmentCoeff::Scalar(v));
    }
    let vals: Vec<f64> = obj.extract::<Vec<f64>>().map_err(|_| {
        new_pyerr(
            "GridMismatch",
            &format!("{name} entries must be float or numpy.ndarray[float64]"),
        )
    })?;
    if vals.len() != n {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("{name} array entry has length {} != n={n}", vals.len()),
        ));
    }
    if vals.iter().any(|v| !v.is_finite()) {
        return Err(new_pyerr("NanInf", &format!("{name} entry contains NaN or Inf")));
    }
    Ok(SegmentCoeff::Array(Arc::new(vals)))
}

/// Parse a whole schedule; `None` yields `n_segments` copies of `default`.
pub(crate) fn parse_schedule(
    obj: Option<&Bound<'_, PyAny>>,
    n_segments: usize,
    n: usize,
    default: f64,
    name: &str,
) -> PyResult<Vec<SegmentCoeff>> {
    let Some(seq) = obj else {
        return Ok(vec![SegmentCoeff::Scalar(default); n_segments]);
    };
    let items: Vec<Bound<'_, PyAny>> = seq
        .try_iter()
        .map_err(|_| new_pyerr("GridMismatch", &format!("{name} must be a sequence")))?
        .collect::<PyResult<_>>()?;
    if items.len() != n_segments {
        return Err(new_pyerr(
            "GridMismatch",
            &format!(
                "{name} has {} segments, expected {n_segments} (must match a_schedule)",
                items.len()
            ),
        ));
    }
    items
        .iter()
        .map(|it| parse_entry(it, n, name))
        .collect::<PyResult<Vec<_>>>()
}

/// Walk the segments, rebuilding only the coefficient closures per segment.
///
/// Captures no Python objects; safe inside `py.detach`. Unlike the scalar-`a`
/// path this ping-pongs two buffers instead of cloning the whole state on every
/// step, and it keeps the state in Rust across segment boundaries.
pub(crate) fn run_coefficient_schedule(
    grid: semiflow::Grid1D<f64>,
    input: Vec<f64>,
    t_final: f64,
    n_steps_per_segment: usize,
    sched: (Vec<SegmentCoeff>, Vec<SegmentCoeff>, Vec<SegmentCoeff>),
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let (a_s, b_s, c_s) = sched;
    let n_segments = a_s.len();
    let dt = t_final / n_segments as f64;
    let tau = dt / n_steps_per_segment as f64;
    let (xmin, dx) = (grid.xmin, grid.dx());

    let n = input.len();
    let mut state = GridFn1D::new(grid, input)?;
    let mut next = GridFn1D::new(grid, vec![0.0_f64; n])?;
    let mut scratch = ScratchPool::new();

    for k in 0..n_segments {
        let norm = a_s[k].norm() + b_s[k].norm() + c_s[k].norm();
        let kernel = ShiftChernoff1D::with_closure(
            a_s[k].closure(xmin, dx),
            b_s[k].closure(xmin, dx),
            c_s[k].closure(xmin, dx),
            norm,
            grid,
        );
        for _ in 0..n_steps_per_segment {
            kernel.apply_into(tau, &state, &mut next, &mut scratch)?;
            core::mem::swap(&mut state, &mut next);
        }
    }
    Ok(state.values)
}

/// Rebuild a `ChernoffSemigroup` carrying one segment's coefficients.
///
/// Called after a schedule walk so the object's own kernel matches where the
/// walk ended. The pre-existing `evolve_with_time_schedule` does NOT do this:
/// it updates only the state, leaving `semigroup.func` on the construction-time
/// coefficients, so a later `evolve()` silently reverts. That behaviour is left
/// alone for backward compatibility; the new entry point does the consistent
/// thing instead.
pub(crate) fn semigroup_from_segment(
    grid: semiflow::Grid1D<f64>,
    tail: &(SegmentCoeff, SegmentCoeff, SegmentCoeff),
) -> Result<
    semiflow::ChernoffSemigroup<ShiftChernoff1D<f64>, GridFn1D<f64>>,
    semiflow::SemiflowError,
> {
    let (xmin, dx) = (grid.xmin, grid.dx());
    let norm = tail.0.norm() + tail.1.norm() + tail.2.norm();
    let kernel = ShiftChernoff1D::with_closure(
        tail.0.closure(xmin, dx),
        tail.1.closure(xmin, dx),
        tail.2.closure(xmin, dx),
        norm,
        grid,
    );
    semiflow::ChernoffSemigroup::new(kernel, 100)
}
