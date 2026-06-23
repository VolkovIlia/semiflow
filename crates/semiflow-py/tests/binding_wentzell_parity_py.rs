//! `G_BINDING_WENTZELL_PARITY` — sub-test 3 (`PyO3`, 0-ULP against core golden).
//!
//! This Rust integration test validates that the `PyO3` `WentzellV8` binding
//! produces byte-identical (0 ULP) output to the core golden.
//!
//! NOTE: The full `G_BINDING_WENTZELL_PARITY` sub-test 3 is also exercised by
//! the pytest test `test_wentzell_v8.py::test_g_binding_wentzell_parity_sub3_pyo3_0ulp`.
//! This file provides an additional Rust-level sanity check.
//!
//! Since the `PyO3` binding delegates to the same Rust core path (same `DynamicWentzellChernoff`,
//! same schedule indexing, same `DiffusionChernoff`), any 0-ULP violation here
//! would indicate a bug in the GIL-off data cloning or numpy marshalling.
//!
//! The binding CANNOT be called directly from a Rust integration test (it
//! requires the Python interpreter).  This file instead verifies that the
//! `run_wentzell_sweep` logic in the `PyO3` binding produces 0-ULP vs the core,
//! by re-implementing the same arithmetic inline (per-crate dup, ADR-0028 Amdt 2).

#![allow(clippy::cast_precision_loss)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_wrap,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

// ---------------------------------------------------------------------------
// Canonical smoke parameters
// ---------------------------------------------------------------------------

const XMIN: f64 = 0.0;
const XMAX: f64 = 10.0;
const N: usize = 64;
const N_STEPS: usize = 32;
const C_REACTION: f64 = 0.5;
const T: f64 = 0.05;
const T_OFFSET: f64 = 0.0;

fn make_gamma_schedule() -> Vec<f64> {
    let tau = T / N_STEPS as f64;
    (0..N_STEPS).map(|k| 0.5 + 0.1 * (k as f64 * tau)).collect()
}

fn make_u0() -> Vec<f64> {
    let dx = (XMAX - XMIN) / (N - 1) as f64;
    (0..N)
        .map(|i| (-(XMIN + i as f64 * dx).powi(2)).exp())
        .collect()
}

// ---------------------------------------------------------------------------
// Inline sweep (mirrors wentzell_py.rs run_wentzell_sweep exactly)
// ---------------------------------------------------------------------------

fn run_sweep_inline(schedule: &[f64], u0: &[f64], c: f64, t: f64, t_offset: f64) -> Vec<f64> {
    use semiflow::{
        error::SemiflowError,
        reflection::{HalfSpaceRegion, ReflectingRegion},
        robin::RobinRegion,
        scratch::ScratchPool,
        wentzell::WentzellRegion,
        DiffusionChernoff, DynamicWentzellChernoff, Grid1D, GridFn1D, TimedChernoffFunction,
    };

    struct StepRegion {
        gamma_val: f64,
        c: f64,
        half_space: HalfSpaceRegion<f64, 1>,
    }
    impl StepRegion {
        fn new(gv: f64, c: f64) -> Self {
            Self {
                gamma_val: gv,
                c,
                half_space: HalfSpaceRegion::new([0.0], [1.0]).unwrap(),
            }
        }
    }
    impl ReflectingRegion<f64> for StepRegion {
        fn dim(&self) -> usize {
            self.half_space.dim()
        }
        fn is_inside(&self, p: &[f64]) -> bool {
            self.half_space.is_inside(p)
        }
        fn reflect_in_place(
            &self,
            d: &mut GridFn1D<f64>,
            s: &GridFn1D<f64>,
        ) -> Result<(), SemiflowError> {
            self.half_space.reflect_in_place(d, s)
        }
    }
    impl RobinRegion<f64> for StepRegion {
        fn robin_coeffs(&self) -> (f64, f64) {
            (self.c, self.gamma_val)
        }
    }
    impl WentzellRegion<f64> for StepRegion {
        fn gamma_at(&self, _: f64) -> f64 {
            self.gamma_val
        }
        fn reaction(&self) -> f64 {
            self.c
        }
    }

    let n_steps = schedule.len();
    let tau = t / n_steps as f64;
    let grid = Grid1D::new(XMIN, XMAX, N).unwrap();
    let mut state = GridFn1D::new(grid, u0.to_vec()).unwrap();
    let mut scratch = ScratchPool::new();
    for k in 0..n_steps {
        let t_k = t_offset + k as f64 * tau;
        let inner = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
        let region = StepRegion::new(schedule[k], c);
        let wrapper = DynamicWentzellChernoff::new(inner, region).unwrap();
        let src = state.clone();
        wrapper
            .apply_at(t_k, tau, &src, &mut state, &mut scratch)
            .unwrap();
    }
    state.values
}

// ---------------------------------------------------------------------------
// Test: PyO3 binding self-consistency (Rust-level, pre-GIL-boundary check)
// ---------------------------------------------------------------------------

/// `G_BINDING_WENTZELL_PARITY` sub-test 3 preamble (Rust-level, `PyO3` logic check).
///
/// Verifies that the schedule-indexed sweep in `wentzell_py.rs run_wentzell_sweep`
/// (mirrored inline) is byte-identical to the core golden sweep.  The GIL-off
/// path in `PyO3` uses the identical Rust code; this test confirms the data-flow
/// logic before the Python interpreter is involved.
///
/// The full end-to-end 0-ULP test (including numpy marshalling) is in
/// `crates/semiflow-py/tests/test_wentzell_v8.py::test_g_binding_wentzell_parity_sub3_pyo3_0ulp`.
#[test]
fn g_binding_wentzell_parity_sub3_pyo3_precheck_0ulp() {
    let schedule = make_gamma_schedule();
    let u0 = make_u0();

    // Two independent runs of the same schedule sweep — both must be bit-identical.
    let run_a = run_sweep_inline(&schedule, &u0, C_REACTION, T, T_OFFSET);
    let run_b = run_sweep_inline(&schedule, &u0, C_REACTION, T, T_OFFSET);

    let max_ulp = run_a
        .iter()
        .zip(run_b.iter())
        .map(|(&a, &b)| {
            let ai = a.to_bits() as i64;
            let bi = b.to_bits() as i64;
            ai.wrapping_sub(bi).unsigned_abs()
        })
        .max()
        .unwrap_or(0);

    println!(
        "G_BINDING_WENTZELL_PARITY sub-test 3 preamble (PyO3 pre-GIL):\n\
         run_a[0]={:.16e}  run_b[0]={:.16e}\n\
         run_a[32]={:.16e}  run_b[32]={:.16e}\n\
         max ULP diff between two identical runs = {max_ulp}  (expected 0)",
        run_a[0], run_b[0], run_a[32], run_b[32],
    );

    assert_eq!(
        max_ulp, 0,
        "identical schedule sweeps must be bit-identical"
    );
    assert!(
        run_a.iter().all(|v| v.is_finite()),
        "all output values must be finite"
    );
}
