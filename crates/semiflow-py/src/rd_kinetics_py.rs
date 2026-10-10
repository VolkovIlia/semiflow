//! Kinetics for the Python reaction–diffusion classes (ADR-0208/0209).
//!
//! Built-in models are selected by name with a `params` dict; any Python
//! callable `f(t, x, u) -> du` is wrapped by [`PyKinetics`], which is called
//! ONCE per Runge–Kutta stage with all nodes (`x`: `(dim, N)`, `u`: `(K, N)`
//! numpy arrays), so a NumPy-vectorised reaction costs `O(steps)` Python calls.
//! The GIL is released by the caller around the whole evolution and
//! re-acquired here only for the callback; a Python exception is kept and
//! re-raised verbatim after the evolution stops.

// Binding layer: PyO3 wrapper patterns.
#![allow(clippy::cast_precision_loss, clippy::needless_pass_by_value)]

use std::sync::{Arc, Mutex};

use numpy::{ndarray::Array2, PyReadonlyArrayDyn, ToPyArray};
use pyo3::{
    prelude::*,
    types::{PyDict, PyDictMethods},
};
use semiflow::{
    AllenCahnReaction, Brusselator, FisherKpp, FitzHughNagumo, GrayScott, Kinetics, LinearReaction,
    Nagumo, SemiflowError as CoreError,
};

use crate::error::{from_core, new_pyerr};

/// Shared kinetics (`Box`ed into each `ReactionDiffusion` without cloning state).
pub(crate) struct ArcKinetics(pub(crate) Arc<dyn Kinetics<f64>>);

#[allow(clippy::many_single_char_names)] // the trait's own parameter names
impl Kinetics<f64> for ArcKinetics {
    fn species(&self) -> usize {
        self.0.species()
    }
    fn eval_batch(
        &self,
        t: f64,
        n: usize,
        x: &[f64],
        u: &[f64],
        du: &mut [f64],
    ) -> Result<(), CoreError> {
        self.0.eval_batch(t, n, x, u, du)
    }
    fn exact_flow(
        &self,
        t: f64,
        h: f64,
        n: usize,
        x: &[f64],
        u: &mut [f64],
    ) -> Option<Result<(), CoreError>> {
        self.0.exact_flow(t, h, n, x, u)
    }
}

/// A vectorised Python reaction `f(t, x, u) -> du`.
pub(crate) struct PyKinetics {
    k: usize,
    dim: usize,
    func: Py<PyAny>,
    error: Mutex<Option<PyErr>>,
}

impl PyKinetics {
    pub(crate) fn new(k: usize, dim: usize, func: Py<PyAny>) -> Self {
        Self {
            k,
            dim,
            func,
            error: Mutex::new(None),
        }
    }

    /// The Python exception raised by the last failed callback, if any.
    pub(crate) fn take_error(&self) -> Option<PyErr> {
        self.error.lock().ok().and_then(|mut e| e.take())
    }

    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)] // Kinetics::eval_batch + py
    fn call(
        &self,
        py: Python<'_>,
        t: f64,
        n: usize,
        x: &[f64],
        u: &[f64],
        du: &mut [f64],
    ) -> PyResult<()> {
        let shape = |rows: usize, v: &[f64]| {
            Array2::from_shape_vec((rows, n), v.to_vec())
                .map_err(|_| new_pyerr("GridMismatch", "internal reshape failure"))
        };
        let (xa, ua) = (shape(self.dim, x)?, shape(self.k, u)?);
        let out = self
            .func
            .call1(py, (t, xa.to_pyarray(py), ua.to_pyarray(py)))?;
        let np = py.import("numpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "float64")?;
        let arr = np.call_method("ascontiguousarray", (out,), Some(&kwargs))?;
        let ro: PyReadonlyArrayDyn<'_, f64> = arr.extract()?;
        let flat = ro.as_slice()?;
        if flat.len() != self.k * n {
            return Err(new_pyerr(
                "GridMismatch",
                &format!(
                    "reaction returned {} values, expected K·N = {}·{}",
                    flat.len(),
                    self.k,
                    n
                ),
            ));
        }
        du[..flat.len()].copy_from_slice(flat);
        Ok(())
    }
}

#[allow(clippy::many_single_char_names)] // the trait's own parameter names
impl Kinetics<f64> for PyKinetics {
    fn species(&self) -> usize {
        self.k
    }
    fn eval_batch(
        &self,
        t: f64,
        n: usize,
        x: &[f64],
        u: &[f64],
        du: &mut [f64],
    ) -> Result<(), CoreError> {
        Python::attach(|py| self.call(py, t, n, x, u, du)).map_err(|e| {
            if let Ok(mut slot) = self.error.lock() {
                *slot = Some(e);
            }
            CoreError::DomainViolation {
                what: "ReactionDiffusion: the Python reaction callback raised",
                value: t,
            }
        })
    }
}

/// `params[key]` as `f64`, or `default`.
fn param(params: Option<&Bound<'_, PyDict>>, key: &str, default: f64) -> PyResult<f64> {
    match params.map(|p| p.get_item(key)).transpose()?.flatten() {
        Some(v) => v.extract::<f64>(),
        None => Ok(default),
    }
}

/// Reject keys the chosen model does not know (catches typos like `"feeed"`).
fn check_keys(params: Option<&Bound<'_, PyDict>>, model: &str, allowed: &[&str]) -> PyResult<()> {
    let Some(p) = params else { return Ok(()) };
    for key in p.keys() {
        let key: String = key.extract()?;
        if !allowed.contains(&key.as_str()) {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("reaction {model:?}: unknown parameter {key:?} (allowed: {allowed:?})"),
            ));
        }
    }
    Ok(())
}

/// One catalogue entry: allowed parameter names and the model.
type Entry = (&'static [&'static str], Arc<dyn Kinetics<f64>>);

/// The catalogue (see the class docstrings); `None` for an unknown name.
fn catalogue(name: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<Option<Entry>> {
    let p = |k: &str, d: f64| param(params, k, d);
    Ok(Some(match name {
        "fisher_kpp" => (
            &["rate", "capacity"],
            Arc::new(FisherKpp {
                rate: p("rate", 1.0)?,
                capacity: p("capacity", 1.0)?,
            }),
        ),
        "allen_cahn" => (
            &["kappa"],
            Arc::new(AllenCahnReaction {
                kappa: p("kappa", 1.0)?,
            }),
        ),
        "nagumo" => (&["a"], Arc::new(Nagumo { a: p("a", 0.25)? })),
        "gray_scott" => (
            &["feed", "kill"],
            Arc::new(GrayScott {
                feed: p("feed", 0.04)?,
                kill: p("kill", 0.06)?,
            }),
        ),
        "fitzhugh_nagumo" => (
            &["a", "b", "eps", "current"],
            Arc::new(FitzHughNagumo {
                a: p("a", 0.7)?,
                b: p("b", 0.8)?,
                eps: p("eps", 0.08)?,
                current: p("current", 0.5)?,
            }),
        ),
        "brusselator" => (
            &["a", "b"],
            Arc::new(Brusselator {
                a: p("a", 1.0)?,
                b: p("b", 3.0)?,
            }),
        ),
        "linear" => (&["matrix"], Arc::new(linear_kinetics(params)?)),
        _ => return Ok(None),
    }))
}

/// Built-in kinetics by name, parameters checked against the model's names.
pub(crate) fn builtin_kinetics(
    name: &str,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Arc<dyn Kinetics<f64>>> {
    let (allowed, kin) = catalogue(name, params)?.ok_or_else(|| {
        new_pyerr(
            "Unsupported",
            &format!(
                "unknown reaction {name:?}: expected one of fisher_kpp, allen_cahn, nagumo, \
                 gray_scott, fitzhugh_nagumo, brusselator, linear, or a callable f(t, x, u)"
            ),
        )
    })?;
    check_keys(params, name, allowed)?;
    Ok(kin)
}

/// `"linear"`: `params["matrix"]` is a `K×K` nested sequence.
fn linear_kinetics(params: Option<&Bound<'_, PyDict>>) -> PyResult<LinearReaction<f64>> {
    let rows: Vec<Vec<f64>> = params
        .map(|p| p.get_item("matrix"))
        .transpose()?
        .flatten()
        .ok_or_else(|| {
            new_pyerr(
                "OutOfDomain",
                "reaction \"linear\" needs params={\"matrix\": K×K}",
            )
        })?
        .extract()?;
    let k = rows.len();
    if rows.iter().any(|r| r.len() != k) {
        return Err(new_pyerr(
            "GridMismatch",
            "linear reaction matrix must be square (K×K)",
        ));
    }
    LinearReaction::new(rows.concat()).map_err(|e| from_core(&e))
}
