//! `no_std` + `libm` execution check for [`semiflow`].
//!
//! `semiflow` builds `#![no_std]` (with `alloc`) when its default features are
//! off; `f64` math then comes from `num_traits::Float`, backed by `libm`. The
//! crate's own test suite cannot exercise that path: its dev-dependencies turn
//! on `num-traits/std`, which links `std` and routes `f64` math to the
//! platform library. This crate depends on `semiflow` alone, with default
//! features off, so every scenario below computes through `libm` — on the host
//! (`cargo test -p semiflow-nostd-check`) and on bare-metal Cortex-M
//! (`nostd-qemu/` at the repository root, run under QEMU).
//!
//! Each [`Scenario`] runs a small problem, compares it with a closed-form or
//! independent oracle, and returns the measured error with its tolerance. The
//! oracles and tolerances come from existing `semiflow` integration tests,
//! cited on each scenario. Nothing here panics on a numerical failure: a
//! scenario returns a [`Failure`] and [`run_all`] reports it.

#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

use core::fmt;

use semiflow::{SemiflowError, SemiflowFloat};

mod graph;
mod heat;
mod particles;
mod spectral;

/// Result of a scenario that met its tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    /// Measured error against the oracle (the norm is scenario-specific).
    pub err: f64,
    /// Tolerance the error had to stay within (`err <= tol`).
    pub tol: f64,
}

/// Why a scenario failed.
#[derive(Clone, Debug)]
pub enum Failure {
    /// A `semiflow` call returned an error.
    Api(SemiflowError),
    /// The measured error is above the tolerance, or not finite.
    Tolerance(Outcome),
    /// A structural invariant did not hold (message says which).
    Invariant(&'static str),
}

impl From<SemiflowError> for Failure {
    fn from(e: SemiflowError) -> Self {
        Self::Api(e)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Api(e) => write!(f, "semiflow error: {e}"),
            Self::Tolerance(o) => write!(f, "err={:.3e} tol={:.3e}", o.err, o.tol),
            Self::Invariant(what) => write!(f, "invariant violated: {what}"),
        }
    }
}

/// Outcome of one scenario run.
pub type ScenarioResult = Result<Outcome, Failure>;

/// One named check: a problem, its oracle, and a tolerance.
#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    /// Short identifier printed in logs.
    pub name: &'static str,
    /// Runs the scenario.
    pub run: fn() -> ScenarioResult,
}

static SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "heat_shift1d",
        run: heat::shift1d_heat,
    },
    Scenario {
        name: "heat_diffusion",
        run: heat::diffusion_heat,
    },
    Scenario {
        name: "strang_heat_decay",
        run: heat::strang_heat_decay,
    },
    Scenario {
        name: "truncated_exp_heat",
        run: heat::truncated_exp_heat,
    },
    Scenario {
        name: "chebyshev_bc_reflect",
        run: spectral::chebyshev_reflect,
    },
    Scenario {
        name: "chebyshev_bc_periodic",
        run: spectral::chebyshev_periodic,
    },
    Scenario {
        name: "schrodinger_unitarity",
        run: spectral::schrodinger_unitarity,
    },
    Scenario {
        name: "expmv_div_form",
        run: spectral::expmv_div_form,
    },
    Scenario {
        name: "graph_krylov_chebyshev",
        run: graph::krylov_chebyshev,
    },
    Scenario {
        name: "symop_implicit_euler_pcg",
        run: graph::implicit_euler_pcg,
    },
    Scenario {
        name: "subordinated_gamma",
        run: particles::subordinated_gamma,
    },
    Scenario {
        name: "gridless_moments",
        run: particles::gridless_moments,
    },
    Scenario {
        name: "smolyak_heat_d2",
        run: particles::smolyak_heat_d2,
    },
    Scenario {
        name: "reverse_ad_sqrt_n",
        run: particles::reverse_ad_sqrt_n,
    },
];

/// All scenarios, in run order.
#[must_use]
pub fn scenarios() -> &'static [Scenario] {
    SCENARIOS
}

/// Pass/fail counts from [`run_all`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Scenarios that met their tolerance.
    pub passed: usize,
    /// Scenarios that failed.
    pub failed: usize,
}

impl Summary {
    /// `true` when at least one scenario ran and none failed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.passed > 0
    }
}

/// Run every scenario in order, calling `report` after each one.
pub fn run_all(mut report: impl FnMut(&Scenario, &ScenarioResult)) -> Summary {
    let mut summary = Summary::default();
    for scenario in scenarios() {
        let result = (scenario.run)();
        if result.is_ok() {
            summary.passed += 1;
        } else {
            summary.failed += 1;
        }
        report(scenario, &result);
    }
    summary
}

/// Accept `err` when it is finite and within `tol`.
fn check(err: f64, tol: f64) -> ScenarioResult {
    let outcome = Outcome { err, tol };
    if err.is_finite() && err <= tol {
        Ok(outcome)
    } else {
        Err(Failure::Tolerance(outcome))
    }
}

/// Largest absolute difference between two equal-length slices.
///
/// A NaN anywhere makes the result NaN, so [`check`] rejects it.
fn sup_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    a.iter().zip(b).fold(0.0, |m, (x, y)| {
        let d = m::abs(x - y);
        if d.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(d)
        }
    })
}

/// Scalar `f64` math through `num_traits::Float` — `libm` in this build.
///
/// `f64::exp` and friends are inherent methods only when `std` is linked, so
/// the oracles use these wrappers; they run the same `libm` code as `semiflow`.
mod m {
    use super::SemiflowFloat;

    fn exp_g<F: SemiflowFloat>(x: F) -> F {
        x.exp()
    }
    fn sqrt_g<F: SemiflowFloat>(x: F) -> F {
        x.sqrt()
    }
    fn cos_g<F: SemiflowFloat>(x: F) -> F {
        x.cos()
    }
    fn sin_g<F: SemiflowFloat>(x: F) -> F {
        x.sin()
    }
    fn abs_g<F: SemiflowFloat>(x: F) -> F {
        x.abs()
    }

    pub(crate) fn exp(x: f64) -> f64 {
        exp_g(x)
    }
    pub(crate) fn sqrt(x: f64) -> f64 {
        sqrt_g(x)
    }
    pub(crate) fn cos(x: f64) -> f64 {
        cos_g(x)
    }
    pub(crate) fn sin(x: f64) -> f64 {
        sin_g(x)
    }
    pub(crate) fn abs(x: f64) -> f64 {
        abs_g(x)
    }
}

#[cfg(test)]
mod tests;
