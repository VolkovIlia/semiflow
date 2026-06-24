//! `PyO3` wrappers for Hörmander/Heisenberg types (v4.1 Phase D).
//!
//! Exports:
//! - `HypoellipticChernoffHeisenberg` — `HypoellipticChernoff<f64, 3, 2>` via
//!   `new_heisenberg()`.
//! - `heisenberg_heat_kernel(h, x, y, t)` — module-level oracle function.
//!
//! ADR-0028: f64-only bindings; no generic `<F>` surface.
//! ADR-0087: Heisenberg step-2 Carnot bracket verified at construction.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::unused_self)]

use pyo3::prelude::*;
use semiflow::{heisenberg_heat_kernel, HypoellipticChernoff};

use crate::{error::from_core, panic::catch_panic_py};

// ---------------------------------------------------------------------------
// HypoellipticChernoffHeisenberg pyclass
// ---------------------------------------------------------------------------

/// Heisenberg group H₁ Chernoff approximation (palindromic Strang-Hörmander).
///
/// Implements the `exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)` step with
/// `X₁ = HeisenbergGroup::x1()`, `X₂ = HeisenbergGroup::x2()` on ℝ³.
///
/// Per ADR-0087 / math.md §28 AMENDMENT 2.  The step-2 Carnot bracket
/// condition `[X₁, X₂] ≈ (0, 0, 1)` is verified at construction.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`OutOfDomain`' if the bracket check fails (should never occur with
///     the canonical HeisenbergX/HeisenbergY fields).
#[pyclass(name = "HypoellipticChernoffHeisenberg")]
pub struct PyHypoellipticChernoffHeisenberg {
    inner: HypoellipticChernoff<f64, 3, 2>,
}

#[pymethods]
impl PyHypoellipticChernoffHeisenberg {
    /// Construct the Heisenberg group Chernoff approximation.
    ///
    /// Verifies the step-2 Carnot bracket `[X₁, X₂] = ∂_t` at the origin.
    ///
    /// Returns
    /// -------
    /// `HypoellipticChernoffHeisenberg`
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if bracket verification fails.
    #[new]
    fn new() -> PyResult<Self> {
        catch_panic_py!({
            let inner =
                HypoellipticChernoff::<f64, 3, 2>::new_heisenberg().map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 2 for Strang-Hörmander).
    ///
    /// Returns
    /// -------
    /// int
    ///     Approximation order = 2.
    fn order(&self) -> u32 {
        use semiflow::ChernoffFunction;
        self.inner.order()
    }

    /// Evaluate the Heisenberg heat kernel oracle `p_h(x, y, tc)`.
    ///
    /// This is a convenience method that delegates to the module-level
    /// `heisenberg_heat_kernel(h, x, y, tc)` function.
    ///
    /// Parameters
    /// ----------
    /// h : float
    ///     Step parameter h > 0.
    /// x : float
    ///     First horizontal coordinate.
    /// y : float
    ///     Second horizontal coordinate.
    /// tc : float
    ///     Vertical (centre) coordinate t.
    ///
    /// Returns
    /// -------
    /// float
    ///     Kernel value `p_h(x, y, tc)`.  Returns 0.0 if `h <= 0`.
    fn kernel(&self, h: f64, x: f64, y: f64, tc: f64) -> f64 {
        heisenberg_heat_kernel(h, x, y, tc)
    }

    fn __repr__(&self) -> &'static str {
        "HypoellipticChernoffHeisenberg(order=2, D=3, M=2)"
    }
}

// ---------------------------------------------------------------------------
// Module-level heisenberg_heat_kernel pyfunction
// ---------------------------------------------------------------------------

/// Heisenberg group heat kernel oracle `p_h(x, y, tc)` (math.md §28 AMENDMENT 2).
///
/// Computes the Gaveau-Hulanicki integral via 32-pt Gauss-Legendre quadrature:
///
/// ``p_h(x, y, tc) = (1/(2π)²) · ∫_{-Λ}^{+Λ} (λ/sinh(λh/2))
///   · exp(−(λ/4)·coth(λh/2)·(x²+y²)) · cos(λ·tc) dλ``
///
/// Parameters
/// ----------
/// h : float
///     Step parameter h > 0.  Returns 0.0 for h ≤ 0.
/// x : float
///     First horizontal coordinate.
/// y : float
///     Second horizontal coordinate.
/// tc : float
///     Vertical (centre) coordinate.
///
/// Returns
/// -------
/// float
///     Kernel value.  At the origin (`x=y=tc=0`) equals ``(2/(π²h²))``.
///
/// Examples
/// --------
/// >>> import semiflow as rp
/// >>> # At origin the kernel equals 2/(π²h²)
/// >>> import math
/// >>> h = 1.0
/// >>> kernel = rp.heisenberg_heat_kernel(h, 0.0, 0.0, 0.0)
/// >>> abs(kernel - 2.0 / (math.pi**2 * h**2)) < 1e-6
/// True
#[pyfunction]
#[pyo3(name = "heisenberg_heat_kernel")]
pub fn py_heisenberg_heat_kernel(h: f64, x: f64, y: f64, tc: f64) -> f64 {
    heisenberg_heat_kernel(h, x, y, tc)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register Hörmander/Heisenberg pyclasses into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyHypoellipticChernoffHeisenberg>()?;
    m.add_function(wrap_pyfunction!(py_heisenberg_heat_kernel, m)?)?;
    Ok(())
}
