//! v9.0.0-beta WASM bindings for `MeasureState` and `GridlessChernoff` (ADR-0171).
//!
//! Exposes two JS classes:
//! - `MeasureState`     — weighted-Dirac particle carrier (D=1 monomorphic).
//! - `GridlessEvolver`  — gridless particle-ensemble Chernoff evolver.
//!
//! ## D=1 monomorphism
//!
//! This build fixes `D = 1`.  Passing `dim != 1` to any constructor returns
//! an `Unsupported` error, matching the FFI v9.2.0 `COMPILED_D = 1` contract.
//!
//! ## Curse-escape (C-1 invariant)
//!
//! The dense 3^D particle tree is NEVER materialised across the JS boundary.
//! `marginal()` returns a sparse 1-D projection (positions + weights arrays).
//!
//! ## Particle ABI
//!
//! `n_part` Diracs cross as two parallel `Float64Array`s:
//!   `positions[n_part]` (D=1: one position per particle)
//!   `weights[n_part]`
//!
//! ## Panic boundary (ADR-0028 Amendment 1)
//!
//! Workspace `[profile.release]` uses `panic = "abort"`.  No `catch_unwind`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

use wasm_bindgen::prelude::*;

use semiflow_core::{
    chernoff::ChernoffFunction, GridlessChernoff, MeasureState as CoreMeasureState,
    ParticleReduction, ScratchPool,
};

use crate::error::make_js_error;

/// Compiled dimension for this build (matches FFI `COMPILED_D` = 1).
const COMPILED_D: usize = 1;

// ---------------------------------------------------------------------------
// MeasureState JS class
// ---------------------------------------------------------------------------

/// Weighted-Dirac particle carrier for adjoint FP Chernoff (D=1, v9.0.0-beta).
///
/// Built from two parallel `Float64Array`s (positions, weights).
/// Only D=1 is supported in this build; pass `dim = 1`.
///
/// ## Error model (`.kind`)
///
/// - `"Unsupported"` — `dim != 1`.
/// - `"GridMismatch"` — `n_part == 0` or buffer length mismatch.
/// - `"NanInf"`       — non-finite position or weight.
/// - `"OutOfDomain"`  — axis out of range.
#[wasm_bindgen]
pub struct MeasureState {
    inner: CoreMeasureState<f64, COMPILED_D>,
}

#[wasm_bindgen]
impl MeasureState {
    /// Construct a `MeasureState` from particle buffers.
    ///
    /// `positions` — `Float64Array` of length `n_part` (D=1: one per particle).
    /// `weights`   — `Float64Array` of length `n_part`.
    /// `dim`       — must be 1 for this build.
    #[wasm_bindgen(constructor)]
    pub fn new(
        positions: &js_sys::Float64Array,
        weights: &js_sys::Float64Array,
        dim: usize,
    ) -> Result<MeasureState, JsValue> {
        if dim != COMPILED_D {
            return Err(make_js_error(
                "Unsupported",
                "MeasureState: only dim=1 is supported in this build",
            ));
        }
        let n = positions.length() as usize;
        if n == 0 {
            return Err(make_js_error("GridMismatch", "MeasureState: n_part must be >= 1"));
        }
        if weights.length() as usize != n {
            return Err(make_js_error(
                "GridMismatch",
                "MeasureState: weights.length must equal positions.length",
            ));
        }
        let mut pos_v = vec![0.0f64; n];
        let mut wt_v = vec![0.0f64; n];
        positions.copy_to(&mut pos_v);
        weights.copy_to(&mut wt_v);
        let ms = build_measure_state_wasm(&pos_v, &wt_v, n)?;
        Ok(MeasureState { inner: ms })
    }

    /// Number of Dirac atoms (curse-escape diagnostic: bounded by reduction cap).
    #[wasm_bindgen(js_name = "nDiracs")]
    #[must_use]
    pub fn n_diracs(&self) -> usize {
        self.inner.n_diracs()
    }

    /// Total-variation norm `‖ρ‖_TV` (total mass observable).
    #[wasm_bindgen(js_name = "totalVariation")]
    #[must_use]
    pub fn total_variation(&self) -> f64 {
        self.inner.total_variation()
    }

    /// Second moment `⟨x², ρ⟩` (spread observable).
    #[wasm_bindgen(js_name = "secondMoment")]
    #[must_use]
    pub fn second_moment(&self) -> f64 {
        self.inner.second_moment()
    }

    /// 1-D marginal: positions and weights projected onto `axis` (= 0 for D=1).
    ///
    /// Returns a plain JS object `{ positions: Float64Array, weights: Float64Array }`.
    /// Curse-escape preserved: sparse marginal only, never a dense grid.
    ///
    /// ## Errors
    /// - `"OutOfDomain"` — axis >= 1.
    pub fn marginal(&self, axis: usize) -> Result<js_sys::Object, JsValue> {
        if axis >= COMPILED_D {
            return Err(make_js_error(
                "OutOfDomain",
                "MeasureState.marginal: axis must be < 1 for D=1 build",
            ));
        }
        let diracs = self.inner.diracs();
        let n = diracs.len();
        let pos_out = js_sys::Float64Array::new_with_length(n as u32);
        let wt_out = js_sys::Float64Array::new_with_length(n as u32);
        for (i, (pos, w)) in diracs.iter().enumerate() {
            pos_out.set_index(i as u32, pos[axis]);
            wt_out.set_index(i as u32, *w);
        }
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"positions".into(), &pos_out)?;
        js_sys::Reflect::set(&obj, &"weights".into(), &wt_out)?;
        Ok(obj)
    }
}

impl MeasureState {
    /// Crate-internal read access to the core state.
    pub(crate) fn inner(&self) -> &CoreMeasureState<f64, COMPILED_D> {
        &self.inner
    }

    /// Crate-internal mutable access to the core state.
    pub(crate) fn inner_mut(&mut self) -> &mut CoreMeasureState<f64, COMPILED_D> {
        &mut self.inner
    }
}

// ---------------------------------------------------------------------------
// GridlessEvolver JS class
// ---------------------------------------------------------------------------

/// Gridless particle-ensemble Chernoff evolver (D=1, v9.0.0-beta, math §50).
///
/// Applies `apply(tau, src, dst)` for one Chernoff step or
/// `evolve(state, t_final, n_steps)` for multiple steps in-place.
///
/// ## Error model (`.kind`)
///
/// - `"Unsupported"` — `dim != 1`.
/// - `"NanInf"`      — non-finite a, b, or c.
/// - `"OutOfDomain"` — invalid reduction tag, `voronoi_cap == 0`, or
///                     `n_steps == 0` / `t_final < 0` / `tau < 0`.
#[wasm_bindgen]
pub struct GridlessEvolver {
    inner: GridlessChernoff<f64, COMPILED_D>,
}

#[wasm_bindgen]
impl GridlessEvolver {
    /// Construct a `GridlessEvolver`.
    ///
    /// `a`            — `Float64Array` of length `dim`: per-axis diffusion.
    /// `b`            — `Float64Array` of length `dim`: per-axis drift.
    /// `c`            — scalar reaction.
    /// `dim`          — must be 1 for this build.
    /// `reductionTag` — `0` = `WeightedVoronoi`, `1` = `GaussianBackground`.
    /// `voronoiCap`   — max particle count (>= 1, used iff tag == 0).
    #[wasm_bindgen(constructor)]
    pub fn new(
        a: &js_sys::Float64Array,
        b: &js_sys::Float64Array,
        c: f64,
        dim: usize,
        reduction_tag: u32,
        voronoi_cap: usize,
    ) -> Result<GridlessEvolver, JsValue> {
        if dim != COMPILED_D {
            return Err(make_js_error(
                "Unsupported",
                "GridlessEvolver: only dim=1 is supported in this build",
            ));
        }
        if a.length() as usize != COMPILED_D || b.length() as usize != COMPILED_D {
            return Err(make_js_error(
                "GridMismatch",
                "GridlessEvolver: a and b must have length == dim",
            ));
        }
        if !c.is_finite() {
            return Err(make_js_error("NanInf", "GridlessEvolver: c must be finite"));
        }
        let mut a_v = [0.0f64; COMPILED_D];
        let mut b_v = [0.0f64; COMPILED_D];
        a.copy_to(&mut a_v);
        b.copy_to(&mut b_v);
        for &v in a_v.iter().chain(b_v.iter()) {
            if !v.is_finite() {
                return Err(make_js_error("NanInf", "GridlessEvolver: a or b is NaN/Inf"));
            }
        }
        if a_v[0] < 0.0 {
            return Err(make_js_error("NanInf", "GridlessEvolver: diffusion a[0] must be >= 0"));
        }
        let reduction = decode_reduction_wasm(reduction_tag, voronoi_cap)?;
        Ok(GridlessEvolver {
            inner: GridlessChernoff::<f64, COMPILED_D>::new(a_v, b_v, c, reduction),
        })
    }

    /// Apply one Chernoff step of size `tau`: write push-forward of `src` into `dst`.
    ///
    /// `src` is read-only; `dst` is overwritten entirely.
    /// A fresh `ScratchPool` is created per call (matches FFI `smf_gridless_apply`).
    pub fn apply(
        &self,
        tau: f64,
        src: &MeasureState,
        dst: &mut MeasureState,
    ) -> Result<(), JsValue> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "GridlessEvolver.apply: tau must be finite and >= 0",
            ));
        }
        let mut pool = ScratchPool::<f64>::new();
        self.inner
            .apply_into(tau, src.inner(), dst.inner_mut(), &mut pool)
            .map_err(|e| make_js_error("OutOfDomain", &e.to_string()))
    }

    /// Evolve `state` in-place for time `t_final` using `n_steps` Chernoff steps.
    ///
    /// Ping-pongs between two scratch buffers; the result lands in `state`.
    pub fn evolve(
        &self,
        state: &mut MeasureState,
        t_final: f64,
        n_steps: usize,
    ) -> Result<(), JsValue> {
        if n_steps == 0 {
            return Err(make_js_error(
                "OutOfDomain",
                "GridlessEvolver.evolve: n_steps must be >= 1",
            ));
        }
        if !t_final.is_finite() || t_final < 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "GridlessEvolver.evolve: t_final must be finite and >= 0",
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let tau = t_final / n_steps as f64;
        let mut buf_a = state.inner().clone();
        let mut buf_b = state.inner().clone();
        let mut pool = ScratchPool::<f64>::new();
        let mut a_is_src = true;
        for _ in 0..n_steps {
            let res = if a_is_src {
                self.inner.apply_into(tau, &buf_a, &mut buf_b, &mut pool)
            } else {
                self.inner.apply_into(tau, &buf_b, &mut buf_a, &mut pool)
            };
            res.map_err(|e| make_js_error("OutOfDomain", &e.to_string()))?;
            a_is_src = !a_is_src;
        }
        *state.inner_mut() = if a_is_src { buf_a } else { buf_b };
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn decode_reduction_wasm(tag: u32, voronoi_cap: usize) -> Result<ParticleReduction, JsValue> {
    match tag {
        0 => {
            if voronoi_cap == 0 {
                return Err(make_js_error(
                    "OutOfDomain",
                    "GridlessEvolver: voronoiCap must be >= 1",
                ));
            }
            Ok(ParticleReduction::WeightedVoronoi { cap: voronoi_cap })
        }
        1 => Ok(ParticleReduction::GaussianBackground),
        _ => Err(make_js_error("OutOfDomain", "GridlessEvolver: unknown reductionTag")),
    }
}

fn build_measure_state_wasm(
    pos: &[f64],
    wts: &[f64],
    n: usize,
) -> Result<CoreMeasureState<f64, COMPILED_D>, JsValue> {
    let mut particles: Vec<([f64; COMPILED_D], f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let p = pos[i];
        let w = wts[i];
        if !p.is_finite() || !w.is_finite() {
            return Err(make_js_error("NanInf", "MeasureState: position or weight is NaN/Inf"));
        }
        particles.push(([p], w));
    }
    Ok(CoreMeasureState::<f64, COMPILED_D>::from_particles(&particles))
}
