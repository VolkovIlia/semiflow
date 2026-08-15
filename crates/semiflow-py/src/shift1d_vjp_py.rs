//! `shift1d_coeff_grad` — gradients w.r.t. `Shift1D` coefficient fields (#25).
//!
//! Free function rather than a method, mirroring `edge_weight_grad`
//! (`graph_sensitivity_py.rs`): the loss lives outside the library and the
//! caller supplies the cotangent `∂J/∂u_n`, so there is nothing to attach to an
//! object's state.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    // `a`/`b`/`c`/`n`/`t` are the Python-facing keyword names; renaming them to
    // satisfy a lint would break the documented call signature.
    clippy::many_single_char_names,
    clippy::similar_names
)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::shift1d_vjp::{shift1d_coeff_gradient, Shift1DProblem, ShiftCoeffField, ShiftCoeffs};

use crate::{
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

/// `∂J/∂θ` for one coefficient field of a `Shift1D`-shaped generator.
///
/// Parameters
/// ----------
/// xmin, xmax, n : grid, as for :meth:`Shift1D.with_arrays`.
/// a, b, c : array-like[float64]
///     Per-node coefficients, length ``n`` each.
/// u0 : array-like[float64]
///     Initial condition, length ``n``.
/// `dj_du_n` : array-like[float64]
///     **Cotangent** ``∂J/∂u_n`` at the final time, length ``n``. The loss ``J``
///     is yours; for ``J = ½‖u_n − target‖²`` pass ``u_n − target``.
/// t : float
///     Total time. ``tau = t / n_steps``.
/// `n_steps` : int
/// wrt : str
///     ``"a"``, ``"b"`` or ``"c"``.
/// boundary : str, optional
///
/// Returns
/// -------
/// numpy.ndarray[float64]
///     ``∂J/∂θ_i`` for every node, length ``n``.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` on a length mismatch; ``kind='OutOfDomain'`` for
///     an unknown ``wrt``, non-positive ``t``, or ``a_i <= 0`` when
///     ``wrt='a'`` — the ``√(τ/a)`` chain factor is undefined there, so the
///     gradient's domain is strictly smaller than the forward kernel's.
#[pyfunction]
#[pyo3(signature = (xmin, xmax, n, a, b, c, u0, dj_du_n, t, n_steps, *,
                    wrt = "a", boundary = "reflect"))]
pub fn shift1d_coeff_grad<'py>(
    py: Python<'py>,
    xmin: f64,
    xmax: f64,
    n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    u0: Vec<f64>,
    dj_du_n: Vec<f64>,
    t: f64,
    n_steps: usize,
    wrt: &str,
    boundary: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    catch_panic_py!({
        let field = parse_field(wrt)?;
        let named = [
            ("a", &a),
            ("b", &b),
            ("c", &c),
            ("u0", &u0),
            ("dj_du_n", &dj_du_n),
        ];
        validate_inputs(n, n_steps, t, &named)?;
        let grid = build_grid(xmin, xmax, n, boundary)?;
        #[allow(clippy::cast_precision_loss)]
        let tau = t / n_steps as f64;
        let mut grad = vec![0.0_f64; n];
        let result = py.detach(|| {
            let problem = Shift1DProblem {
                grid: &grid,
                coeffs: ShiftCoeffs {
                    a: &a,
                    b: &b,
                    c: &c,
                },
                tau,
                n_steps,
            };
            shift1d_coeff_gradient(&problem, field, &u0, &dj_du_n, &mut grad)
        });
        result.map_err(|e| from_core(&e))?;
        Ok(grad.as_slice().to_pyarray(py))
    })
}

/// Build the grid with the requested boundary policy.
fn build_grid(xmin: f64, xmax: f64, n: usize, boundary: &str) -> PyResult<semiflow::Grid1D<f64>> {
    let policy = parse_boundary(boundary)?;
    Ok(semiflow::Grid1D::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy))
}

/// Map the `wrt=` string to a field selector.
fn parse_field(wrt: &str) -> PyResult<ShiftCoeffField> {
    match wrt {
        "a" => Ok(ShiftCoeffField::A),
        "b" => Ok(ShiftCoeffField::B),
        "c" => Ok(ShiftCoeffField::C),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!("wrt must be 'a', 'b' or 'c', got '{other}'"),
        )),
    }
}

/// Shared length / finiteness / domain checks.
fn validate_inputs(n: usize, n_steps: usize, t: f64, arrays: &[(&str, &Vec<f64>)]) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and > 0"));
    }
    for (name, v) in arrays {
        if v.len() != n {
            return Err(new_pyerr(
                "GridMismatch",
                &format!("{name} has length {}, expected n={n}", v.len()),
            ));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(new_pyerr("NanInf", &format!("{name} contains NaN or Inf")));
        }
    }
    Ok(())
}

/// Register the free function on the module.
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shift1d_coeff_grad, m)?)?;
    Ok(())
}
