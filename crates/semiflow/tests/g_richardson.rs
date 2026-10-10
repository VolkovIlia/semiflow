//! Richardson extrapolation gates (`RELEASE_BLOCKING`, ADR-0207).
//!
//! * `G_RICHARDSON_WEIGHTS`: the weights of `richardson_weights` are the closed
//!   forms for small cases and cancel the leading error terms to rounding.
//! * `G_RICHARDSON_SHIFT1D`: `ShiftChernoff1D` (order 1) on `u_t = ½u_xx`,
//!   `u₀ = e^{−x²}`, exact solution `(1+2t)^{−½}e^{−x²/(1+2t)}`: the observed
//!   order is 1 / 2 / 3 with 1 / 2 / 3 levels.
//! * `G_RICHARDSON_COST_GAIN`: at an equal number of steps, three levels beat
//!   the plain run by at least 1000× (measured ≈ 4600×).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)] // small levels, orders and indices

use semiflow::{richardson_weights, Evolver, Grid1D, GridFn1D, ScratchPool, ShiftChernoff1D};

#[test]
fn g_richardson_weights() {
    let close = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-12 * y.abs().max(1.0))
    };
    assert!(close(&richardson_weights(1, 1).unwrap(), &[1.0]));
    assert!(close(&richardson_weights(1, 2).unwrap(), &[-1.0, 2.0]));
    assert!(close(&richardson_weights(1, 3).unwrap(), &[0.5, -4.0, 4.5]));
    assert!(close(
        &richardson_weights(2, 2).unwrap(),
        &[-1.0 / 3.0, 4.0 / 3.0]
    ));
    for p in 1..=4_u32 {
        for levels in 1..=6 {
            let w = richardson_weights(p, levels).unwrap();
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() <= 1e-12, "p={p} L={levels}: Σw = {sum}");
            for q in p..p + levels as u32 - 1 {
                let r: f64 = w
                    .iter()
                    .enumerate()
                    .map(|(j, &wj)| wj * ((j + 1) as f64).powi(-(q as i32)))
                    .sum();
                assert!(r.abs() <= 1e-11, "p={p} L={levels} q={q}: residual {r:e}");
            }
        }
    }
    assert!(richardson_weights(0, 2).is_err());
    assert!(richardson_weights(u32::MAX, 2).is_err());
    assert!(richardson_weights(1, 0).is_err() && richardson_weights(1, 7).is_err());
}

const T: f64 = 1.0;

fn grid() -> Grid1D<f64> {
    Grid1D::new(-12.0, 12.0, 2401).expect("grid")
}

/// `sup |U − u(T)|` for `levels` levels on base step count `n`.
fn error(n: usize, levels: usize) -> f64 {
    let g = grid();
    let engine = ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.0, g);
    let u0 = GridFn1D::from_fn(g, |x| (-x * x).exp());
    let mut out = u0.clone();
    Evolver::new(engine, n)
        .expect("evolver")
        .evolve_extrapolated_into(T, levels, &u0, &mut out, &mut ScratchPool::new())
        .expect("extrapolate");
    let exact = |x: f64| (-x * x / (1.0 + 2.0 * T)).exp() / (1.0 + 2.0 * T).sqrt();
    out.values
        .iter()
        .enumerate()
        .map(|(i, v)| (v - exact(g.x_at(i))).abs())
        .fold(0.0_f64, |a, e| if e.is_nan() { f64::NAN } else { a.max(e) })
}

#[test]
fn g_richardson_shift1d() {
    let mut fails = Vec::new();
    for (levels, min_slope) in [(1, 0.9), (2, 1.9), (3, 2.8)] {
        let (e1, e2, e3) = (error(8, levels), error(16, levels), error(32, levels));
        let (s1, s2) = ((e1 / e2).log2(), (e2 / e3).log2());
        eprintln!("G_RICHARDSON_SHIFT1D L={levels}: err {e1:.3e} {e2:.3e} {e3:.3e} slopes {s1:.3} {s2:.3}");
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must fail
        if !(s1 >= min_slope && s2 >= min_slope) {
            fails.push(format!("L={levels}: slopes {s1:.3}, {s2:.3} < {min_slope}"));
        }
    }
    assert!(fails.is_empty(), "{fails:?}");
}

#[test]
fn g_richardson_cost_gain() {
    // Three levels on n = 16 cost (1+2+3)·16 = 96 steps.
    let (plain, extrapolated) = (error(96, 1), error(16, 3));
    eprintln!(
        "G_RICHARDSON_COST_GAIN plain(96)={plain:.3e} extrapolated(16, L=3)={extrapolated:.3e}"
    );
    assert!(
        extrapolated * 10.0 <= plain,
        "gain {:.2} < 10",
        plain / extrapolated
    );
}
