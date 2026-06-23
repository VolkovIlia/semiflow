//! v8.1.0 WASM binding for `ResolventJumpChernoff` (F2, ADR-0138, ADR-0134).
//!
//! Implements `ResolventJumpV8` — a stateless-per-call JS class that evaluates
//! `e^{tA}g` for the 1D unit-diffusion heat kernel via the TWS parabolic-
//! contour inverse Laplace quadrature (math.md §47).
//!
//! ## NARROW scope (§47.4, ADR-0134 NORMATIVE)
//!
//! Self-adjoint / sectorial generators only (diffusion family).
//! Non-self-adjoint / advection-dominated generators are OUT of scope
//! (math.md §47.4). `m_nodes >= 6` is enforced at construction.
//!
//! ## ABI-safety invariant (ADR-0138 hard constraint)
//!
//! `Complex<f64>` / TWS contour arithmetic stays sealed inside core.
//! `.jump(t, g)` receives/returns `Float64Array` only; the complex math
//! never crosses the WASM boundary.
//!
//! Profile: `[profile.release]` (`panic = "abort"`) per ADR-0028 Amendment 1.
//! All error paths return `Err(JsValue)`.
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/py.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_truncation, clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use semiflow::{DiffusionChernoff, Grid1D, GridFn1D, ResolventJumpChernoff};

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Inner Rust state
// ---------------------------------------------------------------------------

/// Heap-owned kernel (stored in the WASM class JS object).
struct ResolventJumpInnerWasm {
    kernel: ResolventJumpChernoff<DiffusionChernoff<f64>, f64>,
}

// ---------------------------------------------------------------------------
// ResolventJumpV8 WASM class
// ---------------------------------------------------------------------------

/// v8.1.0 Resolvent time-jump evaluator for 1D unit-diffusion heat (ADR-0134).
///
/// Evaluates `e^{tA}g` for a LARGE step `t` via the TWS parabolic-contour
/// inverse Laplace quadrature (math.md §47).  Suitable for large `t` where
/// a many-step Chernoff product would be expensive.
///
/// **NARROW scope**: self-adjoint / sectorial generators only (diffusion family).
/// Non-self-adjoint / advection-dominated generators are OUT of scope
/// (math.md §47.4 NORMATIVE).  `mNodes >= 6` required.
///
/// ## JS Example
///
/// ```js
/// import init, { ResolventJumpV8 } from "@semiflow/wasm";
/// await init();
/// const N = 64;
/// const g = new Float64Array(N).map((_, i) => {
///   const x = -10 + 20*i/(N-1);
///   return Math.exp(-x*x);
/// });
/// const rj = new ResolventJumpV8(-10, 10, N, 16);
/// const result = rj.jump(0.5, g);  // Float64Array of length N
/// ```
///
/// ## Error model (`.kind` discriminator)
///
/// - `"GridMismatch"` — invalid geometry or `g.length != n_grid`.
/// - `"OutOfDomain"`  — `m_nodes < 6` or `t <= 0`.
#[wasm_bindgen]
pub struct ResolventJumpV8 {
    inner: ResolventJumpInnerWasm,
}

#[wasm_bindgen]
impl ResolventJumpV8 {
    /// Construct a resolvent-jump evaluator for unit-diffusion heat.
    ///
    /// ## Parameters
    /// - `domainLo`  — left boundary (finite).
    /// - `domainHi`  — right boundary (finite, > `domainLo`).
    /// - `nGrid`     — number of grid nodes (>= 4).
    /// - `mNodes`    — TWS contour node count (>= 6).
    #[wasm_bindgen(constructor)]
    pub fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        m_nodes: usize,
    ) -> Result<ResolventJumpV8, JsValue> {
        let inner =
            build_inner(domain_lo, domain_hi, n_grid, m_nodes).map_err(|e| err_to_js(&e))?;
        Ok(ResolventJumpV8 { inner })
    }

    /// Evaluate `e^{tA}g`; return result as `Float64Array`.
    ///
    /// The complex contour arithmetic stays sealed in core (ADR-0138).
    ///
    /// ## Parameters
    /// - `t` — time step (`> 0`, finite).
    /// - `g` — `Float64Array` of length `size()`.
    ///
    /// ## Errors
    /// - `.kind = "GridMismatch"` — `g.length != n_grid`.
    /// - `.kind = "OutOfDomain"`  — `t <= 0` or non-finite.
    pub fn jump(&self, t: f64, g: &js_sys::Float64Array) -> Result<js_sys::Float64Array, JsValue> {
        if !t.is_finite() || t <= 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and > 0"));
        }
        let n = self.inner.kernel.grid.n;
        if g.length() as usize != n {
            return Err(make_js_error("GridMismatch", "g.length must equal n_grid"));
        }
        let mut g_buf = vec![0.0f64; n];
        g.copy_to(&mut g_buf);
        let result_vals = run_jump(self.inner.kernel.grid, &g_buf, self.inner.kernel.m_nodes, t)
            .map_err(|e| err_to_js(&e))?;
        let out = js_sys::Float64Array::new_with_length(n as u32);
        out.copy_from(&result_vals);
        Ok(out)
    }

    /// Return the number of grid nodes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.kernel.grid.n
    }

    /// Return the number of TWS contour nodes.
    #[wasm_bindgen(js_name = "mNodes")]
    #[must_use]
    pub fn m_nodes(&self) -> usize {
        self.inner.kernel.m_nodes
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust contour solve
// ---------------------------------------------------------------------------

/// Rebuild the kernel and evaluate jump (per-crate dup, ADR-0028 Amdt 2).
fn run_jump(
    grid: Grid1D<f64>,
    g_vals: &[f64],
    m_nodes: usize,
    t: f64,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    let kernel = ResolventJumpChernoff::new(chernoff, m_nodes, grid)?;
    let g = GridFn1D::new(grid, g_vals.to_vec())?;
    let result = kernel.jump(t, &g)?;
    Ok(result.values)
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_inner(
    lo: f64,
    hi: f64,
    n_grid: usize,
    m_nodes: usize,
) -> Result<ResolventJumpInnerWasm, semiflow::SemiflowError> {
    let grid = Grid1D::new(lo, hi, n_grid)?;
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    let kernel = ResolventJumpChernoff::new(chernoff, m_nodes, grid)?;
    Ok(ResolventJumpInnerWasm { kernel })
}
