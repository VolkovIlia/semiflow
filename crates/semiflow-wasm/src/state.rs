//! `#[wasm_bindgen] Heat1D` — 1-D heat equation state exposed to JavaScript.
//!
//! Wraps `SemiflowStateInner` (heap-allocated semigroup + current grid function)
//! and exposes a JS-idiomatic API:
//!   `new Heat1D(xmin, xmax, n, u0)` — constructor
//!   `evolve(t, n_steps)` — advance by time `t`
//!   `values()` — return current state as `Float64Array`
//!   `len()` — number of grid nodes
//!
//! `#![allow(unsafe_code)]` is required because the `wasm-bindgen` `#[wasm_bindgen]`
//! proc-macro expands `unsafe` blocks inside this file, identical to `#[pyclass]`
//! expansion in Wave B.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

use crate::{
    error::{err_to_js, make_js_error},
    handle::{
        build_heat_unit, build_heat_with_closure, ClosureParams, JsCallback, SemiflowStateInner,
    },
};

// ---------------------------------------------------------------------------
// Heat1D wasm class
// ---------------------------------------------------------------------------

/// 1-D heat equation state for JavaScript/WebAssembly.
///
/// Solves `∂_t u = ∂_{xx} u` (unit diffusion coefficient `a = 1`) on the
/// closed interval `[xmin, xmax]` with `n` uniformly-spaced grid nodes.
///
/// All fallible operations throw a JS `Error` whose `.kind` property
/// identifies the error category.  See the crate-level documentation for the
/// full `.kind` → meaning table.
///
/// # Lifecycle
/// ```js
/// import init, { Heat1D, panic_hook_init } from "./semiflow_wasm.js";
/// await init();
/// panic_hook_init();             // optional, improves dev-time diagnostics
/// const u0 = new Float64Array([0, 1, 0, 0]);
/// const state = new Heat1D(0.0, 1.0, 4, u0);
/// state.evolve(0.01, 100);
/// const vals = state.values();   // Float64Array copy
/// ```
#[wasm_bindgen]
pub struct Heat1D {
    inner: SemiflowStateInner,
}

#[wasm_bindgen]
impl Heat1D {
    /// Create a new 1-D heat equation state.
    ///
    /// ## Parameters
    /// - `xmin` — left boundary; must be finite.
    /// - `xmax` — right boundary; must be finite and > `xmin`.
    /// - `n` — number of uniformly-spaced grid nodes; must be ≥ 4.
    /// - `u0` — `Float64Array` of length exactly `n` holding the initial
    ///   condition; all elements must be finite.
    ///
    /// # Errors
    /// - `.kind = "GridMismatch"` — `u0.length != n`, `n < 4`, or
    ///   `xmin >= xmax`.
    /// - `.kind = "NanInf"` — `u0` contains NaN or Inf.
    /// - `.kind = "OutOfDomain"` — other domain precondition violated.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, u0: &Float64Array) -> Result<Heat1D, JsValue> {
        if u0.length() as usize != n {
            return Err(make_js_error("GridMismatch", "u0.length() must equal n"));
        }
        let mut buf = vec![0.0f64; n];
        u0.copy_to(&mut buf);
        let inner = build_heat_unit(xmin, xmax, n, 100, &buf).map_err(|e| err_to_js(&e))?;
        Ok(Heat1D { inner })
    }

    /// Advance the state by time `t` using `n_steps` Chernoff iterations.
    ///
    /// Mutates the state in place.  Call `values()` afterwards to read the
    /// updated grid function.
    ///
    /// ## Parameters
    /// - `t` — time to advance; must be ≥ 0 and finite.
    /// - `n_steps` — number of Chernoff steps; must be ≥ 1.
    ///
    /// ## Note on `t = 0`
    /// `evolve(0, n)` is accepted but does **not** guarantee identity.
    /// Applying the Chernoff kernel `n` times with `tau = 0` numerically
    /// underflows to near-zero rather than recovering the initial condition.
    /// Callers who need an identity step should skip the call.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `t < 0`, `t` is NaN/Inf, or
    ///   `n_steps == 0`.
    /// - `.kind = "CflViolated"` — CFL constraint violated for the chosen
    ///   `t`/`n_steps` combination.
    /// - `.kind = "BoundaryFailure"` — grid resolution too coarse.
    /// - `.kind = "ConvergenceFailed"` — iterative solver did not converge.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        if n_steps == 0 {
            return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
        }
        if !t.is_finite() || t < 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
        }
        let chernoff = self.inner.semigroup.func.clone();
        let sg =
            semiflow::ChernoffSemigroup::new(chernoff, n_steps).map_err(|e| err_to_js(&e))?;
        let next = sg
            .evolve(t, &self.inner.current)
            .map_err(|e| err_to_js(&e))?;
        self.inner.current = next;
        self.inner.semigroup = sg;
        Ok(())
    }

    /// Return the current grid values as a new `Float64Array` (copy).
    ///
    /// Returns a freshly allocated `Float64Array` of length `len()`.
    /// Mutations to the returned array do not affect the `Heat1D` state.
    #[must_use]
    pub fn values(&self) -> Float64Array {
        let v = &self.inner.current.values;
        // wasm32 has 32-bit address space; n <= 2^32 is a safe assumption.
        #[allow(clippy::cast_possible_truncation)]
        let arr = Float64Array::new_with_length(v.len() as u32);
        arr.copy_from(v);
        arr
    }

    /// Return the number of grid nodes (same as `u0.length` passed to the
    /// constructor).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.inner.current.values.len()
    }

    /// Create a 1-D heat equation state with a variable diffusion coefficient
    /// `a(x)` supplied as three JS functions: `a`, `a_prime`, `a_double_prime`.
    ///
    /// This is the ADR-0034 WASM binding for `DiffusionChernoff::with_closure`.
    /// The three JS functions are called once per grid node per Chernoff step;
    /// expect ~0.5-2 µs per call (JS↔WASM crossing overhead).
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds; must satisfy `xmin < xmax`.
    /// - `n` — number of grid nodes; must be ≥ 4.
    /// - `a` — JS function `(x: number) => number` returning the diffusion
    ///   coefficient. Must be positive everywhere on `[xmin, xmax]`.
    /// - `a_prime` — JS function returning `a'(x)` (first derivative).
    /// - `a_double_prime` — JS function returning `a''(x)` (second derivative).
    /// - `a_norm_bound` — an upper bound for `‖a‖∞` (used for diagnostics;
    ///   not validated against the actual function values).
    /// - `u0` — `Float64Array` of length `n`, initial condition.
    /// - `n_steps` — number of Chernoff steps per `evolve(t, n_steps)` call.
    ///
    /// ## JS-side invariants
    /// - `a(x) > 0` everywhere (strict ellipticity; the integrator does not
    ///   validate this — violating it produces incorrect results silently).
    /// - All three functions must return a finite `number`. If a function throws
    ///   or returns a non-numeric value, the grid function becomes NaN.
    ///
    /// # Errors
    /// - `.kind = "GridMismatch"` — `u0.length != n`, `n < 4`, or `xmin >= xmax`.
    /// - `.kind = "NanInf"` — `u0` contains NaN or Inf.
    /// - `.kind = "OutOfDomain"` — other domain precondition violated.
    ///
    /// # Example (JS)
    /// ```js
    /// const a = x => 1 + 0.1 * x * x;
    /// const ap = x => 0.2 * x;
    /// const app = _ => 0.2;
    /// const u0 = new Float64Array(new Array(64).fill(0).map((_, i) => Math.exp(-i)));
    /// const state = Heat1D.withAFunction(0.0, 1.0, 64, a, ap, app, 2.0, u0, 100);
    /// state.evolve(0.01, 50);
    /// ```
    #[allow(clippy::too_many_arguments)]
    // wasm-bindgen public API: JS callers pass all 9 args at once; grouping into
    // a JS object would break the ergonomic `withAFunction(xmin, xmax, n, a, ap, app, ...)` call.
    #[wasm_bindgen(js_name = "withAFunction")]
    pub fn with_a_function(
        xmin: f64,
        xmax: f64,
        n: usize,
        a: js_sys::Function,
        a_prime: js_sys::Function,
        a_double_prime: js_sys::Function,
        a_norm_bound: f64,
        u0: &Float64Array,
        n_steps: usize,
    ) -> Result<Heat1D, JsValue> {
        if u0.length() as usize != n {
            return Err(make_js_error("GridMismatch", "u0.length() must equal n"));
        }
        let mut buf = vec![0.0f64; n];
        u0.copy_to(&mut buf);
        let params = ClosureParams {
            a: JsCallback(a),
            a_prime: JsCallback(a_prime),
            a_double_prime: JsCallback(a_double_prime),
            a_norm_bound,
        };
        let inner = build_heat_with_closure(xmin, xmax, n, n_steps, &buf, params)
            .map_err(|e| err_to_js(&e))?;
        Ok(Heat1D { inner })
    }
}
