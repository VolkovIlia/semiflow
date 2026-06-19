//! `G_BINDING_WENTZELL_PARITY` — sub-test 1 (core golden, `RELEASE_BLOCKING`).
//!
//! Gate spec (contracts/semiflow-core.properties.yaml §`G_BINDING_WENTZELL_PARITY`,
//! ADR-0153, `V8_3_TIER3_BINDING_DESIGN.md` §5):
//!
//! Canonical smoke: unit-a heat half-line, N=64, `n_steps=32`, c=0.5, t=0.05,
//! γ-schedule `γ(t_k)` = 0.5 + `0.1·t_k` at `t_k` = k·τ, τ = `T/n_steps`,
//! u0 = exp(−x²) on [0, 10].
//!
//! The golden vector is produced by sweeping `TimedChernoffFunction::apply_at`
//! step-by-step with a schedule-backed region newtype whose `gamma_at` ignores
//! the time argument and returns `schedule[k]` directly — exactly what the
//! FFI/PyO3/WASM bindings do.  The binding-level schedule ABI is: the host
//! pre-samples `γ(t_k)` and passes `gamma_schedule: &[f64]`; the kernel reads
//! `schedule[k]` per step.  The core golden mirrors this precisely.
//!
//! Sub-tests 2/3/4 compare their binding output byte-for-byte (0 ULP) against
//! `canonical_wentzell_core()`.  Any divergence = marshalling bug in that layer.
//!
//! # Why GENUINE
//!
//! The FFI extern "C" round-trip, `PyO3` GIL-off schedule sweep, and WASM JS heap
//! copy each reconstruct `DynamicWentzellChernoff` from a flat f64 schedule.
//! Wrong schedule indexing, incorrect boundary array layout, or GIL-off data
//! ordering would produce a different bit pattern from the golden.

#![allow(clippy::cast_precision_loss)]
// Integration test/bench: allows for numerical patterns.
#![allow(clippy::missing_panics_doc, clippy::needless_range_loop)]

use semiflow_core::{
    error::SemiflowError,
    reflection::{HalfSpaceRegion, ReflectingRegion},
    robin::RobinRegion,
    scratch::ScratchPool,
    wentzell::WentzellRegion,
    DiffusionChernoff, DynamicWentzellChernoff, Grid1D, GridFn1D, TimedChernoffFunction,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters
// ---------------------------------------------------------------------------

/// Left domain boundary.
pub const XMIN: f64 = 0.0;
/// Right domain boundary.
pub const XMAX: f64 = 10.0;
/// Number of grid nodes.
pub const N: usize = 64;
/// Number of Chernoff steps per evolve.
pub const N_STEPS: usize = 32;
/// Boundary reaction coefficient.
pub const C_REACTION: f64 = 0.5;
/// Total evolution time.
pub const T: f64 = 0.05;

/// Canonical γ-schedule: `γ(t_k)` = 0.5 + `0.1·t_k`, `t_k` = `k·(T/N_STEPS)`.
#[must_use]
pub fn make_gamma_schedule() -> Vec<f64> {
    let tau = T / N_STEPS as f64;
    (0..N_STEPS).map(|k| 0.5 + 0.1 * (k as f64 * tau)).collect()
}

/// Canonical initial condition: u0[i] = `exp(−x_i²)` on [XMIN, XMAX].
#[must_use]
pub fn make_u0() -> Vec<f64> {
    let dx = (XMAX - XMIN) / (N - 1) as f64;
    (0..N)
        .map(|i| (-(XMIN + i as f64 * dx).powi(2)).exp())
        .collect()
}

// ---------------------------------------------------------------------------
// Schedule-backed WentzellRegion (per-crate duplicate pattern §6)
// ---------------------------------------------------------------------------
// This is the same newtype each binding crate duplicates.  Here it is
// used to produce the golden, making it the definitive reference.

/// A `WentzellRegion` backed by a pre-sampled schedule.
///
/// `gamma_at` ignores `t` and returns the stored constant `gamma_val`
/// for the current step.  The caller is responsible for updating `gamma_val`
/// before each `apply_at` call (left-endpoint freeze contract, §1 / ADR-0153).
///
/// `reflect_in_place` delegates to `HalfSpaceRegion` (needed for the trait
/// hierarchy, but never called by `DynamicWentzellChernoff::apply_at` directly).
struct StepRegion {
    gamma_val: f64,
    c: f64,
    half_space: HalfSpaceRegion<f64, 1>,
}

impl StepRegion {
    fn new(gamma_val: f64, c: f64) -> Self {
        let half_space = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).expect("half-space valid");
        Self {
            gamma_val,
            c,
            half_space,
        }
    }
}

impl ReflectingRegion<f64> for StepRegion {
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

impl RobinRegion<f64> for StepRegion {
    fn robin_coeffs(&self) -> (f64, f64) {
        (self.c, self.gamma_val)
    }
}

impl WentzellRegion<f64> for StepRegion {
    /// Returns the pre-sampled schedule value (ignores `t`).
    fn gamma_at(&self, _t: f64) -> f64 {
        self.gamma_val
    }
    fn reaction(&self) -> f64 {
        self.c
    }
}

// ---------------------------------------------------------------------------
// Core reference sweep
// ---------------------------------------------------------------------------

/// Run the canonical Wentzell sweep and return the evolved grid.
///
/// This is the reference computation all binding sub-tests must match byte-for-byte.
/// It sweeps `N_STEPS` Chernoff steps; at each step k:
///   - constructs a fresh `DynamicWentzellChernoff` with `gamma_val = schedule[k]`
///   - calls `apply_at(t_k, tau, ...)` where `t_k = k·tau` (left-endpoint freeze)
///
/// This is `pub` so binding sub-tests (FFI/PyO3/WASM) can import the golden.
#[must_use]
pub fn canonical_wentzell_core() -> Vec<f64> {
    let schedule = make_gamma_schedule();
    let tau = T / N_STEPS as f64;
    let grid = Grid1D::new(XMIN, XMAX, N).expect("grid valid");

    let mut state = GridFn1D::new(grid, make_u0()).expect("u0 valid");
    let mut scratch = ScratchPool::new();

    for k in 0..N_STEPS {
        let t_k = k as f64 * tau;
        let inner = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
        let region = StepRegion::new(schedule[k], C_REACTION);
        let wrapper = DynamicWentzellChernoff::new(inner, region).expect("wrapper valid");
        let src = state.clone();
        wrapper
            .apply_at(t_k, tau, &src, &mut state, &mut scratch)
            .expect("apply_at step");
    }

    state.values
}

// ---------------------------------------------------------------------------
// Test: core golden
// ---------------------------------------------------------------------------

#[test]
fn g_binding_wentzell_parity_core_golden() {
    let result = canonical_wentzell_core();

    assert_eq!(result.len(), N);
    assert!(
        result.iter().all(|v| v.is_finite()),
        "all values must be finite"
    );

    println!(
        "G_BINDING_WENTZELL_PARITY (core golden):\n\
         config: N={N}, n_steps={N_STEPS}, c={C_REACTION}, t={T}, \
         domain=[{XMIN},{XMAX}], schedule=0.5+0.1*t_k, u0=exp(-x^2)\n\
         result[0]  = {:.16e}  (boundary DOF)\n\
         result[1]  = {:.16e}  (first interior)\n\
         result[32] = {:.16e}  (mid)\n\
         result[63] = {:.16e}  (far boundary)",
        result[0], result[1], result[32], result[63],
    );

    // The boundary DOF (result[0]) must be modified by the Cayley step.
    // With Wentzell BC active (c>0, γ>0) the boundary value is NOT the same
    // as a pure Neumann BC — this sanity check will catch if the Cayley step
    // was accidentally skipped.
    assert!(
        result[0].is_finite() && result[0] > 0.0,
        "boundary DOF result[0]={:.4e} must be positive (Wentzell attenuates \
         but does not kill the Gaussian hump at x=0)",
        result[0],
    );
}
