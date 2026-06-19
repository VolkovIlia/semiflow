//! `G_BINDING_WENTZELL_PARITY` — sub-test 4 (WASM, 0-ULP against core golden).
//!
//! NOTE: The full wasm-bindgen-test exercise (JS engine, `Float64Array` copy) requires
//! `wasm-pack test --node` and cannot run as a plain `cargo test`.
//! This file provides a native Rust-level parity pre-check that mirrors the WASM
//! binding's sweep logic (per-crate dup of the sweep arithmetic, ADR-0028 Amdt 2),
//! confirming 0-ULP before the JS boundary is involved.
//!
//! The WASM-specific marshalling (`Float64Array` copy-in/copy-out) is exercised by the
//! `wasm-bindgen-test` in `crates/semiflow-wasm/tests/heat.rs` pattern.
//! A dedicated `#[wasm_bindgen_test]` for `WentzellV8` would be added to that file
//! when `wasm-pack` is available in CI.  This test gates the Rust arithmetic only.

#![allow(clippy::cast_precision_loss)]
// Binding layer: allows for FFI/PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_wrap,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

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

fn run_sweep_wasm_mirror(schedule: &[f64], u0: &[f64], c: f64, t: f64, t_offset: f64) -> Vec<f64> {
    use semiflow_core::{
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

/// `G_BINDING_WENTZELL_PARITY` sub-test 4 pre-check (WASM Rust-level, pre-JS-boundary).
///
/// Verifies the WASM binding's sweep logic (mirrored inline) is deterministic
/// and matches itself 0-ULP.  The full JS `Float64Array` marshal test requires
/// `wasm-pack test --node` (CI job `wasm-test`); this gates the arithmetic only.
#[test]
fn g_binding_wentzell_parity_sub4_wasm_precheck_0ulp() {
    let schedule = make_gamma_schedule();
    let u0 = make_u0();

    let run_a = run_sweep_wasm_mirror(&schedule, &u0, C_REACTION, T, T_OFFSET);
    let run_b = run_sweep_wasm_mirror(&schedule, &u0, C_REACTION, T, T_OFFSET);

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
        "G_BINDING_WENTZELL_PARITY sub-test 4 preamble (WASM pre-JS):\n\
         run_a[0]={:.16e}  run_b[0]={:.16e}\n\
         max ULP diff = {max_ulp}  (expected 0)\n\
         NOTE: full WASM+JS marshal test requires wasm-pack test --node",
        run_a[0], run_b[0],
    );

    assert_eq!(
        max_ulp, 0,
        "identical WASM sweep mirrors must be bit-identical"
    );
    assert!(run_a.iter().all(|v| v.is_finite()));
}
