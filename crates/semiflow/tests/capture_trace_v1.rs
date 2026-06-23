//! Capture v1.0.0 accepted-τ trajectory fixtures for the bit-equality gate.
//!
//! Run ONCE before Wave 4 lands to record baseline. Output goes to
//! `tests/fixtures/adaptive_classical_trace_v1.json`.
//!
//! Three fixtures:
//!   1. `heat_smooth`   — 1D heat, Gaussian IC, t=0.5, tol_rel=1e-5
//!   2. `heat_strict`   — 1D heat, Gaussian IC, t=0.5, tol_rel=1e-8
//!   3. `stiff_heat`    — 1D heat, sharper IC, t=0.5, tol_rel=1e-6
//!
//! JSON format (per entry):
//!   `{ "name": <name>, "taus": [<hex f64 bits>, ...] }`

// Integration test/bench/example: allows for numerical patterns.
#![allow(clippy::too_many_lines)]

use semiflow_core::{
    boundary::InterpKind, grid::BoundaryPolicy, DiffusionChernoff, Grid1D, GridFn1D,
};

/// Capture the v1.0.0 accepted-τ trajectory by re-implementing the substep loop.
/// Since the current code cannot be instrumented directly we replay the formulas.
fn capture_taus_v1(
    diff: &DiffusionChernoff,
    u0: &GridFn1D,
    t: f64,
    tol_abs: f64,
    tol_rel: f64,
) -> Vec<f64> {
    use libm::pow;
    use semiflow_core::{chernoff::ChernoffFunction, State};

    let p_u32 = diff.order();
    let p = f64::from(p_u32);
    let alpha = 0.7 / p;
    let beta = 0.4 / p;
    let safety = 0.9_f64;
    let min_ratio = 0.2_f64;
    let max_ratio = 5.0_f64;
    let max_substeps = 100_000_usize;

    let mut u_curr = u0.clone();
    let mut t_curr = 0.0_f64;
    let mut tau = t * 1e-2;
    let mut err_prev = 1.0_f64;
    let mut taus = Vec::new();
    let mut total = 0_usize;

    loop {
        if total >= max_substeps {
            break;
        }
        let tau_step = tau.min(t - t_curr);

        // apply full + 2 half steps
        let u_full = diff.apply_chernoff(tau_step, &u_curr).unwrap();
        let u_half_a = diff.apply_chernoff(tau_step / 2.0, &u_curr).unwrap();
        let u_half = diff.apply_chernoff(tau_step / 2.0, &u_half_a).unwrap();

        // richardson error
        let mut diff_s = u_half.clone();
        diff_s.axpy_into(-1.0, &u_full);
        let divisor = f64::from((1u32 << p_u32) - 1);
        let err_norm = diff_s.norm_sup() / divisor;

        // tolerance
        let u_curr_norm = u_curr.norm_sup();
        let u_full_norm = u_full.norm_sup();
        let tol = tol_abs + tol_rel * u_curr_norm.max(u_full_norm);

        total += 1;
        if err_norm <= tol {
            // accepted
            let safe_err = err_norm.max(1e-300);
            let e = pow(tol / safe_err, alpha);
            let e_prev = pow(err_prev / safe_err, beta);
            let factor = safety * e * e_prev; // NORMATIVE: left-to-right
            tau = (tau_step * factor).clamp(min_ratio * tau_step, max_ratio * tau_step);

            taus.push(tau_step);
            u_curr = u_half;
            t_curr += tau_step;
            err_prev = err_norm;

            if t_curr >= t {
                break;
            }
        } else {
            // rejected
            let safe_err = err_norm.max(1e-300);
            let factor = safety * pow(tol / safe_err, alpha);
            tau = (tau_step * factor).clamp(min_ratio * tau_step, max_ratio * tau_step);
        }
    }
    taus
}

/// Format a slice of f64 as hex-bit strings for exact round-trip.
fn to_hex(taus: &[f64]) -> Vec<String> {
    taus.iter()
        .map(|&v| format!("{:016x}", v.to_bits()))
        .collect()
}

fn heat_grid(n: usize) -> Grid1D {
    // Pin to CubicHermite: the v1.0.0 fixture was captured with CubicHermite.
    // v6.0 changed Grid1D::new default to SepticHermite (ADR-0109); pin here
    // so that re-capturing the fixture preserves bit-exact v1.0.0 tau values.
    Grid1D::new(-10.0, 10.0, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect)
        .with_interp(InterpKind::CubicHermite)
}

fn diff_const(grid: Grid1D) -> DiffusionChernoff {
    DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid)
}

fn gaussian_ic(grid: Grid1D) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| libm::exp(-x * x / 2.0))
}

fn sharp_ic(grid: Grid1D) -> GridFn1D {
    // steeper Gaussian to exercise more substeps
    GridFn1D::from_fn(grid, |x| libm::exp(-x * x * 4.0))
}

fn build_json(name: &str, taus: &[f64]) -> String {
    let hex: Vec<String> = to_hex(taus);
    let items: Vec<String> = hex.iter().map(|h| format!("\"{h}\"")).collect();
    let joined = items.join(", ");
    let n = taus.len();
    format!("  {{\"name\": \"{name}\", \"len\": {n}, \"taus\": [{joined}]}}")
}

/// Capture the v1.0.0 fixture and write to disk.
///
/// This test writes to `tests/fixtures/adaptive_classical_trace_v1.json`.
/// It is IGNORED by default — run manually ONLY when the fixture needs
/// to be re-generated (e.g., after a deliberate API-breaking change to
/// the `AdaptivePI` τ-stepping algorithm).
///
/// Do NOT remove the `#[ignore]`: running this test automatically in CI
/// would overwrite the golden fixture and break the bit-equality gate.
///
/// Run with: `cargo test -p semiflow-core --test capture_trace_v1 -- --ignored`
#[test]
#[ignore = "one-shot fixture capture: run manually, never in CI (overwrites golden fixture)"]
fn capture_and_write_fixtures() {
    let grid = heat_grid(400);

    let taus1 = capture_taus_v1(&diff_const(grid), &gaussian_ic(grid), 0.5, 0.0, 1e-5);
    let taus2 = capture_taus_v1(&diff_const(grid), &gaussian_ic(grid), 0.5, 0.0, 1e-8);
    let taus3 = capture_taus_v1(&diff_const(grid), &sharp_ic(grid), 0.5, 0.0, 1e-6);

    println!("heat_smooth : {} accepted steps", taus1.len());
    println!("heat_strict : {} accepted steps", taus2.len());
    println!("stiff_heat  : {} accepted steps", taus3.len());

    let json = format!(
        "[\n{},\n{},\n{}\n]\n",
        build_json("heat_smooth", &taus1),
        build_json("heat_strict", &taus2),
        build_json("stiff_heat", &taus3),
    );

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/adaptive_classical_trace_v1.json"
    );
    std::fs::write(path, &json).expect("write fixture");
    println!("Wrote fixture to {path}");

    assert!(!taus1.is_empty(), "heat_smooth: no accepted steps");
    assert!(!taus2.is_empty(), "heat_strict: no accepted steps");
    assert!(!taus3.is_empty(), "stiff_heat: no accepted steps");
}
