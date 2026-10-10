//! Construction helpers of the Python reaction–diffusion classes (ADR-0209):
//! grid geometry, `u0` extraction, reaction parsing.

// Binding layer: PyO3 wrapper patterns.
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};
use semiflow::{BoundaryPolicy, Grid1D, Grid2D, Grid3D, Kinetics};

use crate::{
    error::{from_core, new_pyerr},
    rd_kinetics_py::{builtin_kinetics, PyKinetics},
};

/// Grid geometry: axes in storage order (x fastest).
pub(crate) enum Geometry {
    D1(Grid1D<f64>),
    D2(Grid2D<f64>),
    D3(Grid3D<f64>),
}

impl Geometry {
    pub(crate) fn axes(&self) -> Vec<Grid1D<f64>> {
        match self {
            Self::D1(g) => vec![*g],
            Self::D2(g) => vec![g.x, g.y],
            Self::D3(g) => vec![g.x, g.y, g.z],
        }
    }

    /// numpy shape of one species field (`(n,)`, `(ny, nx)`, `(nz, ny, nx)`).
    pub(crate) fn field_shape(&self) -> Vec<usize> {
        self.axes().iter().rev().map(|g| g.n).collect()
    }
}

/// Grid axes `[(min, max, n)]` with a common boundary policy.
pub(crate) fn build_geometry(
    axes: &[(f64, f64, usize)],
    policy: BoundaryPolicy,
) -> PyResult<Geometry> {
    let g: Vec<Grid1D<f64>> = axes
        .iter()
        .map(|&(lo, hi, n)| Grid1D::new(lo, hi, n).map(|g| g.with_boundary(policy)))
        .collect::<Result<_, _>>()
        .map_err(|e| from_core(&e))?;
    Ok(match g.len() {
        1 => Geometry::D1(g[0]),
        2 => Geometry::D2(Grid2D::new(g[0], g[1])),
        _ => Geometry::D3(Grid3D::new(g[0], g[1], g[2]).map_err(|e| from_core(&e))?),
    })
}

/// `u0` as `K` flat fields, checked against the grid shape.
pub(crate) fn extract_fields(
    py: Python<'_>,
    u0: &Bound<'_, PyAny>,
    shape: &[usize],
) -> PyResult<Vec<Vec<f64>>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    let arr = np.call_method("ascontiguousarray", (u0,), Some(&kwargs))?;
    let ro: PyReadonlyArrayDyn<'_, f64> = arr.extract()?;
    let dims = ro.shape().to_vec();
    if dims.len() != shape.len() + 1 || dims[1..] != *shape || dims[0] == 0 {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("u0 must have shape (K, {shape:?}) with K >= 1, got {dims:?}"),
        ));
    }
    let flat = ro.as_slice()?;
    if flat.iter().any(|v| !v.is_finite()) {
        return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
    }
    let n: usize = shape.iter().product();
    Ok(flat.chunks_exact(n).map(<[f64]>::to_vec).collect())
}

/// Kinetics as a trait object, plus the Python callable's handle when there is one.
pub(crate) type ParsedKinetics = (Arc<dyn Kinetics<f64>>, Option<Arc<PyKinetics>>);

/// Kinetics from `reaction` (name or callable) for `k` species in `dim` dimensions.
pub(crate) fn parse_reaction(
    reaction: &Bound<'_, PyAny>,
    params: Option<&Bound<'_, PyDict>>,
    k: usize,
    dim: usize,
) -> PyResult<ParsedKinetics> {
    if let Ok(name) = reaction.extract::<String>() {
        let kin = builtin_kinetics(&name, params)?;
        if kin.species() != k {
            return Err(new_pyerr(
                "GridMismatch",
                &format!(
                    "reaction {name:?} has {} species, u0 has K = {k}",
                    kin.species()
                ),
            ));
        }
        return Ok((kin, None));
    }
    if !reaction.is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "reaction must be a model name (str) or a callable f(t, x, u) -> du",
        ));
    }
    if params.is_some() {
        return Err(new_pyerr(
            "OutOfDomain",
            "params= applies to built-in reactions only",
        ));
    }
    let pk = Arc::new(PyKinetics::new(k, dim, reaction.clone().unbind()));
    let kin: Arc<dyn Kinetics<f64>> = pk.clone();
    Ok((kin, Some(pk)))
}
