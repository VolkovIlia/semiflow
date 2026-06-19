//! Builder helpers for `ObstacleChernoff` Python binding.
//!
//! Extracted from `obstacle_py.rs` per suckless file-cap: `obstacle_py.rs` was at
//! 500 lines; adding b,c support required extraction (mirrors ADR-0115 pattern).
//!
//! Public surface: [`build_inner`] only.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use pyo3::prelude::*;
use semiflow_core::{
    ConstantObstacle, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, ObstacleChernoff,
    StrangSplit,
};

use crate::obstacle_py::{
    ArrayKernel, ArrayObstacle, ConstKernel, DiffUnit, ObstaclePyInner, ObstacleVariant,
    StrangArrayKernel, StrangConstKernel, StrangUnit,
};

// ---------------------------------------------------------------------------
// Top-level builder
// ---------------------------------------------------------------------------

/// Build the fully initialised `ObstaclePyInner`.
///
/// Dispatch:
/// - `obstacle_array` present → array obstacle variant
/// - otherwise → constant-level obstacle variant
/// - `b == 0.0 && c == 0.0` → pure-diffusion fast path
/// - otherwise → `StrangSplit<DiffusionChernoff, DriftReactionChernoff>`
pub(crate) fn build_inner(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    a: f64,
    b: f64,
    c: f64,
    level: f64,
    obstacle_array: Option<&Bound<'_, PyAny>>,
) -> Result<ObstaclePyInner, semiflow_core::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    let use_strang = b != 0.0 || c != 0.0;
    match obstacle_array {
        Some(arr) => build_array(a, b, c, arr, current, use_strang),
        None => build_const(a, b, c, level, current, use_strang),
    }
}

// ---------------------------------------------------------------------------
// Constant-obstacle paths
// ---------------------------------------------------------------------------

fn build_const(
    a: f64,
    b: f64,
    c: f64,
    level: f64,
    current: GridFn1D<f64>,
    use_strang: bool,
) -> Result<ObstaclePyInner, semiflow_core::SemiflowError> {
    let obs = ConstantObstacle::new(level)?;
    if use_strang {
        let inner = build_strang(a, b, c, current.grid);
        let kernel: StrangConstKernel = ObstacleChernoff::new(inner, obs)?;
        Ok(ObstaclePyInner {
            kernel: ObstacleVariant::Strang(kernel),
            current,
        })
    } else {
        let diff = build_diffusion(a, current.grid);
        let kernel: ConstKernel = ObstacleChernoff::new(diff, obs)?;
        Ok(ObstaclePyInner {
            kernel: ObstacleVariant::Const(kernel),
            current,
        })
    }
}

// ---------------------------------------------------------------------------
// Array-obstacle paths
// ---------------------------------------------------------------------------

fn build_array(
    a: f64,
    b: f64,
    c: f64,
    arr: &Bound<'_, PyAny>,
    current: GridFn1D<f64>,
    use_strang: bool,
) -> Result<ObstaclePyInner, semiflow_core::SemiflowError> {
    let vals: Vec<f64> =
        arr.extract()
            .map_err(|_| semiflow_core::SemiflowError::DomainViolation {
                what: "obstacle_array must be a sequence of floats",
                value: 0.0,
            })?;
    if vals.len() != current.grid.n {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "obstacle_array length must equal n",
            value: vals.len() as f64,
        });
    }
    let obs = ArrayObstacle::new(vals)?;
    if use_strang {
        let inner = build_strang(a, b, c, current.grid);
        let kernel: StrangArrayKernel = ObstacleChernoff::new(inner, obs)?;
        Ok(ObstaclePyInner {
            kernel: ObstacleVariant::StrangArray(kernel),
            current,
        })
    } else {
        let diff = build_diffusion(a, current.grid);
        let kernel: ArrayKernel = ObstacleChernoff::new(diff, obs)?;
        Ok(ObstaclePyInner {
            kernel: ObstacleVariant::Array(kernel),
            current,
        })
    }
}

// ---------------------------------------------------------------------------
// Inner-kernel constructors
// ---------------------------------------------------------------------------

/// Pure-diffusion fast path: `a · ∂_xx`.
fn build_diffusion(a: f64, grid: Grid1D<f64>) -> DiffUnit {
    DiffusionChernoff::new_const_a(a, a, grid)
}

/// Strang split `a·∂_xx + b·∂_x + c·` for non-zero drift/reaction.
///
/// Uses constant closures. The `c_norm_bound` for the reaction is `|c|`.
fn build_strang(a: f64, b: f64, c: f64, grid: Grid1D<f64>) -> StrangUnit {
    let diff = DiffusionChernoff::new_const_a(a, a, grid);
    let c_bound = c.abs();
    let drift = DriftReactionChernoff::with_closure(move |_| b, move |_| c, c_bound, grid);
    StrangSplit::new(diff, drift)
}
