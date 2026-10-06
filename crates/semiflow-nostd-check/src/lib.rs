//! `no_std` execution and bit-identity check for [`semiflow`].
//!
//! `semiflow`'s default build is `#![no_std]` (with `alloc`, ADR-0201). This
//! crate depends on `semiflow` alone, with its default features, and runs every
//! scenario below in that build — on the host
//! (`cargo test -p semiflow-nostd-check`) and on bare-metal Cortex-M
//! (`nostd-qemu/` at the repository root, run under QEMU).
//!
//! Each [`Scenario`] runs a small problem, compares it with a closed-form or
//! independent oracle, and returns the measured error with its tolerance. The
//! oracles and tolerances come from existing `semiflow` integration tests,
//! cited on each scenario. Nothing here panics on a numerical failure: a
//! scenario returns a [`Failure`] and [`run_all`] reports it.
//!
//! Each scenario also hashes its primary output bit-for-bit
//! ([`Outcome::digest`]); [`run_scenario`] rejects a digest that differs from
//! the committed table in `expected.rs`. `semiflow` evaluates every
//! transcendental through `libm` and runs the same lane arithmetic in every
//! build (ADR-0200), so the table must be reproduced unchanged by the `no_std`
//! build, by the `std-ref` feature (semiflow with `std` + `simd`,
//! `num-traits/std` on), with AVX2 or NEON intrinsics, and on Cortex-M.

#![no_std]
// ADR-0200: oracles must not change with `std-ref`, so every transcendental
// here goes through `SemiflowFloat::libm_*` (see `m`).
#![deny(clippy::disallowed_methods)]

extern crate alloc;

#[cfg(test)]
extern crate std;

use core::fmt;

use semiflow::{SemiflowError, SemiflowFloat};

mod expected;
mod graph;
mod heat;
mod particles;
mod resolvent;
mod spectral;

/// Result of a scenario that met its tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    /// Measured error against the oracle (the norm is scenario-specific).
    pub err: f64,
    /// Tolerance the error had to stay within (`err <= tol`).
    pub tol: f64,
    /// FNV-1a 64 over `f64::to_bits` of the scenario's primary output.
    pub digest: u64,
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
    /// The output bits differ from the committed digest, or none is committed.
    Digest {
        /// Scenario name.
        name: &'static str,
        /// Digest this run produced.
        got: u64,
        /// Committed digest (`None`: the scenario has no entry yet).
        want: Option<u64>,
    },
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
            Self::Digest {
                name,
                got,
                want: Some(want),
            } => write!(
                f,
                "digest mismatch for {name}: got 0x{got:016x} want 0x{want:016x}"
            ),
            Self::Digest {
                name,
                got,
                want: None,
            } => write!(
                f,
                "digest mismatch for {name}: got 0x{got:016x} want missing"
            ),
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
        name: "graph_frechet_large_t",
        run: graph::frechet_large_t,
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
    Scenario {
        name: "texp4_cached_heat",
        run: heat::texp4_cached_heat,
    },
    Scenario {
        name: "diffusion6_heat_f64",
        run: heat::diffusion6_heat_f64,
    },
    Scenario {
        name: "diffusion6_catmull_f32",
        run: heat::diffusion6_catmull_f32,
    },
    Scenario {
        name: "cubic_hermite_heat",
        run: heat::cubic_hermite_heat,
    },
    Scenario {
        name: "octonic_heat",
        run: heat::octonic_heat,
    },
    Scenario {
        name: "spdr_tridiag",
        run: resolvent::spdr_tridiag,
    },
    Scenario {
        name: "spdr_pcg_ic0",
        run: resolvent::spdr_pcg_ic0,
    },
    Scenario {
        name: "phi_combination",
        run: resolvent::phi_combination_p3,
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
        let result = run_scenario(scenario);
        if result.is_ok() {
            summary.passed += 1;
        } else {
            summary.failed += 1;
        }
        report(scenario, &result);
    }
    summary
}

/// Run one scenario: its tolerance check, then its digest against the table.
///
/// # Errors
/// The scenario's own [`Failure`], or [`Failure::Digest`] when the output
/// bits differ from (or are missing in) the committed table.
pub fn run_scenario(scenario: &Scenario) -> ScenarioResult {
    let outcome = (scenario.run)()?;
    match expected::lookup(scenario.name) {
        Some(want) if want == outcome.digest => Ok(outcome),
        want => Err(Failure::Digest {
            name: scenario.name,
            got: outcome.digest,
            want,
        }),
    }
}

/// Accept `err` when it is finite and within `tol`; carry the output digest.
fn check(err: f64, tol: f64, digest: u64) -> ScenarioResult {
    let outcome = Outcome { err, tol, digest };
    if err.is_finite() && err <= tol {
        Ok(outcome)
    } else {
        Err(Failure::Tolerance(outcome))
    }
}

/// FNV-1a 64 over the little-endian bytes of `f64::to_bits`.
///
/// Bit-exact by construction: two runs agree only if every hashed value has
/// identical bits. `f32` outputs are pushed as `f64::from(x)` (exact).
pub(crate) struct Digest(u64);

impl Digest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub(crate) fn new() -> Self {
        Self(Self::OFFSET)
    }

    pub(crate) fn push(&mut self, x: f64) {
        for byte in x.to_bits().to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    pub(crate) fn extend(&mut self, xs: &[f64]) {
        for &x in xs {
            self.push(x);
        }
    }

    pub(crate) fn finish(self) -> u64 {
        self.0
    }
}

/// Digest of one slice.
fn digest_of(xs: &[f64]) -> u64 {
    let mut d = Digest::new();
    d.extend(xs);
    d.finish()
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

/// Scalar `f64` math for the oracles, identical in every build.
///
/// Transcendentals use `SemiflowFloat::libm_*`, so enabling `std-ref` (which
/// turns on `num-traits/std`) changes neither the oracles nor the closures
/// handed to `semiflow`. `sqrt`/`abs` are exact in every backend; they go
/// through `num_traits::Float` because `f64::sqrt` is not in `core`.
mod m {
    use super::SemiflowFloat;

    fn sqrt_g<F: SemiflowFloat>(x: F) -> F {
        x.sqrt()
    }
    fn abs_g<F: SemiflowFloat>(x: F) -> F {
        x.abs()
    }

    pub(crate) fn exp(x: f64) -> f64 {
        x.libm_exp()
    }
    pub(crate) fn sqrt(x: f64) -> f64 {
        sqrt_g(x)
    }
    pub(crate) fn cos(x: f64) -> f64 {
        x.libm_cos()
    }
    pub(crate) fn sin(x: f64) -> f64 {
        x.libm_sin()
    }
    pub(crate) fn abs(x: f64) -> f64 {
        abs_g(x)
    }
}

#[cfg(test)]
mod tests;
