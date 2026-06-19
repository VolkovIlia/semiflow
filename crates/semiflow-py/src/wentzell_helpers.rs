//! Private helpers for `wentzell_py` — split for suckless file-size compliance.
//!
//! Contains:
//! - `ScheduledWentzellRegion` struct + trait impls
//! - Scalar validators (`validate_u0_finite`, `validate_c_reaction`,
//!   `validate_schedule`, `validate_t`)
//! - `extract_f64_vec`

use pyo3::prelude::*;

use semiflow_core::{
    error::SemiflowError,
    reflection::{HalfSpaceRegion, ReflectingRegion},
    robin::RobinRegion,
    wentzell::WentzellRegion,
    GridFn1D,
};

use crate::error::new_pyerr;

// ---------------------------------------------------------------------------
// Schedule-backed WentzellRegion (per-crate duplicate, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

pub(crate) struct ScheduledWentzellRegion {
    pub(crate) gamma_val: f64,
    pub(crate) c: f64,
    pub(crate) half_space: HalfSpaceRegion<f64, 1>,
}

impl ScheduledWentzellRegion {
    pub(crate) fn new(gamma_val: f64, c: f64) -> Result<Self, SemiflowError> {
        Ok(Self {
            gamma_val,
            c,
            half_space: HalfSpaceRegion::<f64, 1>::new([0.0], [1.0])?,
        })
    }
}

impl ReflectingRegion<f64> for ScheduledWentzellRegion {
    fn dim(&self) -> usize {
        self.half_space.dim()
    }
    fn is_inside(&self, point: &[f64]) -> bool {
        self.half_space.is_inside(point)
    }
    fn reflect_in_place(
        &self,
        dst: &mut GridFn1D<f64>,
        src: &GridFn1D<f64>,
    ) -> Result<(), SemiflowError> {
        self.half_space.reflect_in_place(dst, src)
    }
}

impl RobinRegion<f64> for ScheduledWentzellRegion {
    fn robin_coeffs(&self) -> (f64, f64) {
        (self.c, self.gamma_val)
    }
}

impl WentzellRegion<f64> for ScheduledWentzellRegion {
    fn gamma_at(&self, _t: f64) -> f64 {
        self.gamma_val
    }
    fn reaction(&self) -> f64 {
        self.c
    }
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

pub(crate) fn validate_u0_finite(u0: &[f64]) -> Result<(), SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_c_reaction(c: f64) -> Result<(), SemiflowError> {
    if !c.is_finite() || c < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "c_reaction must be finite and >= 0",
            value: c,
        });
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn validate_schedule(sched: &[f64], n_steps: usize) -> Result<(), SemiflowError> {
    if sched.len() != n_steps {
        return Err(SemiflowError::DomainViolation {
            what: "gamma_schedule length must equal n_steps",
            value: sched.len() as f64,
        });
    }
    for &g in sched {
        if !g.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "gamma_schedule contains NaN or Inf",
                value: g,
            });
        }
        if g < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "gamma_schedule values must be >= 0",
                value: g,
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_t(t: f64) -> PyResult<()> {
    if !t.is_finite() || t <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and > 0"));
    }
    Ok(())
}

pub(crate) fn extract_f64_vec(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected a numpy.ndarray[float64] or list of floats",
        )
    })
}
