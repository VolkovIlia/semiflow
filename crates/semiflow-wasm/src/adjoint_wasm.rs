//! Round-9 adjoint semigroup engine for WebAssembly (`full` feature).
//!
//! | JS class      | Core type                                      | Python mirror |
//! |---------------|------------------------------------------------|---------------|
//! | `Adjoint1D`   | `AdjointChernoff<DiffusionChernoff<f64>, f64>` | `Adjoint`     |
//!
//! ## Design
//!
//! Mirrors `semiflow-py` `Adjoint` (`adjoint.rs`), exposing 5 kernel variants
//! via a `kernel` string selector: `"heat2"` (default), `"heat4"`, `"heat6"`,
//! `"drift"`, `"shift"`.  Uses the same `KernelVariant` enum pattern as Python.
//!
//! Self-adjoint kernels (heat2/4/6, shift) use `new_self_adjoint`; `"drift"`
//! defaults to general (`new_general`).  `self_adjoint` boolean override mirrors
//! Python's keyword argument.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    AdjointChernoff, ChernoffFunction, Diffusion4thChernoff, Diffusion6thChernoff,
    DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, ScratchPool, ShiftChernoff1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Coefficient fn-pointers (unit / zero) — fn-pointer variant (Copy-friendly)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_adj(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_adj(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// 5-kernel enum (avoids Box<dyn ChernoffFunction>)
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
enum AdjKernel {
    Diff2(AdjointChernoff<DiffusionChernoff<f64>>),
    Diff4(AdjointChernoff<Diffusion4thChernoff<f64>>),
    Diff6(AdjointChernoff<Diffusion6thChernoff<f64>>),
    DriftReaction(AdjointChernoff<DriftReactionChernoff<f64>>),
    Shift(AdjointChernoff<ShiftChernoff1D<f64>>),
}

impl AdjKernel {
    fn apply_step(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow::SemiflowError> {
        match self {
            Self::Diff2(k) => k.apply_into(tau, src, dst, scratch),
            Self::Diff4(k) => k.apply_into(tau, src, dst, scratch),
            Self::Diff6(k) => k.apply_into(tau, src, dst, scratch),
            Self::DriftReaction(k) => k.apply_into(tau, src, dst, scratch),
            Self::Shift(k) => k.apply_into(tau, src, dst, scratch),
        }
    }

    fn order(&self) -> u32 {
        match self {
            Self::Diff2(k) => k.order(),
            Self::Diff4(k) => k.order(),
            Self::Diff6(k) => k.order(),
            Self::DriftReaction(k) => k.order(),
            Self::Shift(k) => k.order(),
        }
    }

    fn is_self_adjoint(&self) -> bool {
        match self {
            Self::Diff2(k) => k.is_self_adjoint(),
            Self::Diff4(k) => k.is_self_adjoint(),
            Self::Diff6(k) => k.is_self_adjoint(),
            Self::DriftReaction(k) => k.is_self_adjoint(),
            Self::Shift(k) => k.is_self_adjoint(),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn extract_u0_adj(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
    if u0.length() as usize != n {
        return Err(make_js_error("GridMismatch", "u0.length() must equal n"));
    }
    let mut buf = vec![0.0f64; n];
    u0.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(buf)
}

fn fn_to_js_adj(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

// ---------------------------------------------------------------------------
// Kernel builders (each ≤50 lines)
// ---------------------------------------------------------------------------

fn build_adj_kernel(
    xmin: f64,
    xmax: f64,
    n: usize,
    kernel: &str,
    self_adjoint: bool,
) -> Result<AdjKernel, JsValue> {
    let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
    match kernel {
        "heat2" => {
            let inner = DiffusionChernoff::new(unit_a_adj, zero_adj, zero_adj, 1.0, grid);
            Ok(AdjKernel::Diff2(AdjointChernoff::new_self_adjoint(inner)))
        }
        "heat4" => {
            let inner = Diffusion4thChernoff::new(unit_a_adj, zero_adj, zero_adj, 1.0, grid);
            Ok(AdjKernel::Diff4(AdjointChernoff::new_self_adjoint(inner)))
        }
        "heat6" => {
            let inner = Diffusion6thChernoff::new(unit_a_adj, zero_adj, zero_adj, 1.0, grid);
            Ok(AdjKernel::Diff6(AdjointChernoff::new_self_adjoint(inner)))
        }
        "drift" => Ok(build_adj_drift(grid, self_adjoint)),
        "shift" => {
            let inner = ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
            Ok(AdjKernel::Shift(AdjointChernoff::new_self_adjoint(inner)))
        }
        other => Err(make_js_error(
            "OutOfDomain",
            &format!("unknown kernel '{other}'; expected heat2|heat4|heat6|drift|shift"),
        )),
    }
}

fn build_adj_drift(grid: Grid1D<f64>, self_adjoint: bool) -> AdjKernel {
    let inner = DriftReactionChernoff::with_closure(|_| 0.5_f64, |_| 0.0, 0.0, grid);
    let adj = if self_adjoint {
        AdjointChernoff::new_self_adjoint(inner)
    } else {
        AdjointChernoff::new_general(inner)
    };
    AdjKernel::DriftReaction(adj)
}

// ---------------------------------------------------------------------------
// Pure-Rust evolve helper
// ---------------------------------------------------------------------------

fn run_adj_evolve(
    kv: &AdjKernel,
    grid: Grid1D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let mut src = GridFn1D::new(grid, input)?;
    let mut dst = GridFn1D::new(grid, vec![0.0; src.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kv.apply_step(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Adjoint1D — JS class
// ---------------------------------------------------------------------------

/// Adjoint semigroup wrapper for any supported 1-D Chernoff kernel.
///
/// Mirrors `Adjoint` (Python, `adjoint.rs`). Select kernel via string:
/// `"heat2"` (default), `"heat4"`, `"heat6"`, `"drift"`, `"shift"`.
///
/// Self-adjoint kernels (heat2/4/6, shift) use the zero-overhead path.
/// For `"drift"` the `self_adjoint` flag controls whether the dual
/// correction is applied (default: `false` = general dual semigroup).
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Adjoint1D {
    kernel: AdjKernel,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Adjoint1D {
    /// Construct `Adjoint1D`.
    ///
    /// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — `Float64Array` of length `n`, all finite.
    /// - `kernel` — `"heat2"` (default), `"heat4"`, `"heat6"`, `"drift"`, `"shift"`.
    /// - `self_adjoint` — if `true`, skip dual correction for non-symmetric kernels.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        kernel: &str,
        self_adjoint: bool,
    ) -> Result<Adjoint1D, JsValue> {
        let buf = extract_u0_adj(u0, n)?;
        let kv = build_adj_kernel(xmin, xmax, n, kernel, self_adjoint)?;
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Adjoint1D {
            kernel: kv,
            current,
        })
    }

    /// Advance the adjoint state by time `t` using `n_steps` steps.
    ///
    /// Returns updated `Float64Array` of length `n` (copy).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<Float64Array, JsValue> {
        if n_steps == 0 {
            return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
        }
        if !t.is_finite() || t <= 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and > 0"));
        }
        #[allow(clippy::cast_precision_loss)]
        let tau = t / n_steps as f64;
        let grid = self.current.grid;
        let input = self.current.values.clone();
        let out =
            run_adj_evolve(&self.kernel, grid, input, tau, n_steps).map_err(|e| err_to_js(&e))?;
        self.current.values.clone_from(&out);
        Ok(fn_to_js_adj(&out))
    }

    /// Return current grid values as `Float64Array` of length `n` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js_adj(&self.current.values)
    }

    /// Approximation order of the wrapped adjoint kernel.
    #[must_use]
    pub fn order(&self) -> u32 {
        self.kernel.order()
    }

    /// Whether the inner kernel is declared self-adjoint.
    #[must_use]
    pub fn is_self_adjoint(&self) -> bool {
        self.kernel.is_self_adjoint()
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}
