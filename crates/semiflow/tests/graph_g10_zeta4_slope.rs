//! G10 slope gate: `GraphHeat4thChernoff` (order-4) convergence.
//!
//! Gate: log-log slope ≤ −3.95 (f64), ≤ −3.50 (f32) per ADR-0046.
//! f32 band wider because 4-SpMV round-off saturates at finer grids.
//!
//! See Wave 2.1B contract §4.3 and math.md §12.7.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use semiflow_core::{
    graph::{Graph, Laplacian},
    graph_heat4::GraphHeat4thChernoff,
    graph_signal::GraphSignal,
    ChernoffFunction, ChernoffSemigroup, ScratchPool,
};

/// f64 floor ~ 1e-12 arrives at n ≥ 200 for order-4 (6.4e-10 → 3.9e-11 → 2.1e-12
/// at n=25/50/100; floor flat at n≥200).  Cap at 3 values to stay in clean
/// order-4 window.
const N_VALUES: [usize; 3] = [25, 50, 100];
/// f32 round-off floor kicks in at n ≥ 8 for order-4 (4-SpMV / step, eps ~1e-7).
/// 3 coarse values: confirm τ⁴ drop before floor appears.
const N_VALUES_F32: [usize; 3] = [3, 4, 6];
const N_NODES: usize = 64;
const T: f64 = 0.5;

// ---------------------------------------------------------------------------
// Jacobi oracle (f64)
// ---------------------------------------------------------------------------

struct EigDecomp {
    eigenvalues: Vec<f64>,
    eigenvectors_col_major: Vec<f64>,
}

fn jacobi_eig(lap: &Laplacian<f64>) -> EigDecomp {
    let n = lap.n_nodes();
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in lap.row_ptr()[i]..lap.row_ptr()[i + 1] {
            a[i * n + lap.col_idx()[k] as usize] = lap.vals()[k];
        }
    }
    let mut q = vec![0.0_f64; n * n];
    for k in 0..n {
        q[k + k * n] = 1.0;
    }
    let tol = 1e-24_f64;
    for _ in 0..(200 * n) {
        let mut off2 = 0.0_f64;
        for p in 0..n {
            for qi in (p + 1)..n {
                let v = a[p * n + qi];
                off2 += v * v;
            }
        }
        off2 *= 2.0;
        if off2 <= 0.0 {
            break;
        }
        let diag2: f64 = (0..n)
            .map(|k| {
                let v = a[k * n + k];
                v * v
            })
            .sum();
        if off2 < tol * diag2 {
            break;
        }
        let (mut p, mut qi) = (0usize, 1usize);
        let mut mx = 0.0_f64;
        for pi in 0..n {
            for qj in (pi + 1)..n {
                let av = a[pi * n + qj].abs();
                if av > mx {
                    mx = av;
                    p = pi;
                    qi = qj;
                }
            }
        }
        if a[p * n + qi] == 0.0 {
            break;
        }
        let tau_r = (a[qi * n + qi] - a[p * n + p]) / (2.0 * a[p * n + qi]);
        let t_r = if tau_r >= 0.0 {
            1.0 / (tau_r + (1.0 + tau_r * tau_r).sqrt())
        } else {
            -1.0 / (-tau_r + (1.0 + tau_r * tau_r).sqrt())
        };
        let c = 1.0 / (1.0 + t_r * t_r).sqrt();
        let s = t_r * c;
        for r in 0..n {
            let rp = a[r * n + p];
            let rq = a[r * n + qi];
            a[r * n + p] = c * rp - s * rq;
            a[r * n + qi] = s * rp + c * rq;
        }
        for r in 0..n {
            let pr = a[p * n + r];
            let qr = a[qi * n + r];
            a[p * n + r] = c * pr - s * qr;
            a[qi * n + r] = s * pr + c * qr;
        }
        for r in 0..n {
            let rp = q[r + p * n];
            let rq = q[r + qi * n];
            q[r + p * n] = c * rp - s * rq;
            q[r + qi * n] = s * rp + c * rq;
        }
    }
    let mut pairs: Vec<(f64, usize)> = (0..n).map(|k| (a[k * n + k], k)).collect();
    pairs.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let eigenvalues: Vec<f64> = pairs.iter().map(|&(v, _)| v).collect();
    let mut evecs = vec![0.0_f64; n * n];
    for (new_k, &(_, old_k)) in pairs.iter().enumerate() {
        for j in 0..n {
            evecs[j + new_k * n] = q[j + old_k * n];
        }
    }
    for k in 0..n {
        if evecs[k * n] < 0.0 {
            for j in 0..n {
                evecs[j + k * n] = -evecs[j + k * n];
            }
        }
    }
    EigDecomp {
        eigenvalues,
        eigenvectors_col_major: evecs,
    }
}

fn heat_oracle(decomp: &EigDecomp, f0: &[f64], g: Arc<Graph<f64>>, t: f64) -> GraphSignal<f64> {
    let n = f0.len();
    let mut alpha = vec![0.0_f64; n];
    for k in 0..n {
        let mut d = 0.0;
        for j in 0..n {
            d += decomp.eigenvectors_col_major[j + k * n] * f0[j];
        }
        alpha[k] = d;
    }
    let mut u = vec![0.0_f64; n];
    for k in 0..n {
        let coeff = alpha[k] * (-t * decomp.eigenvalues[k]).exp();
        for j in 0..n {
            u[j] += coeff * decomp.eigenvectors_col_major[j + k * n];
        }
    }
    GraphSignal::from_fn(g, |i| u[i as usize])
}

fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let lx: Vec<f64> = ns.iter().map(|&x| (x as f64).ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&y| y.ln()).collect();
    let sx: f64 = lx.iter().sum();
    let sy: f64 = ly.iter().sum();
    let sxx: f64 = lx.iter().map(|&x| x * x).sum();
    let sxy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sxy - sx * sy) / (m * sxx - sx * sx)
}

// ---------------------------------------------------------------------------
// G10 f64 slope gate
// ---------------------------------------------------------------------------

#[test]
fn g10_graph_heat4_convergence_slope_f64() {
    let g = Arc::new(Graph::<f64>::path(N_NODES));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let decomp = jacobi_eig(&lap);
    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| ((i as f64 * 0.31).sin() + 1.0) * 0.5);
    let oracle = heat_oracle(&decomp, f0.values(), Arc::clone(&g), T);

    let errs: Vec<f64> = N_VALUES
        .iter()
        .map(|&n_steps| {
            let semi = ChernoffSemigroup::new(GraphHeat4thChernoff::new(Arc::clone(&lap)), n_steps)
                .expect("n >= 1");
            let u_t = semi.evolve(T, &f0).expect("evolve");
            u_t.values()
                .iter()
                .zip(oracle.values().iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();

    for (&n_steps, &err) in N_VALUES.iter().zip(errs.iter()) {
        println!("G10 f64 n_steps={n_steps:4}, err={err:.4e}");
    }
    let slope = log_log_slope(&N_VALUES, &errs);
    println!("G10 f64 slope = {slope:.4}");
    assert!(
        slope <= -3.95,
        "G10 FAIL f64: slope {slope:.4} > -3.95 (order-4 gate)"
    );
}

// ---------------------------------------------------------------------------
// G10 f32 slope gate (relaxed threshold per ADR-0046)
// ---------------------------------------------------------------------------

#[test]
fn g10_graph_heat4_convergence_slope_f32() {
    // f32 uses 3 coarse N-values (cap upper end to avoid round-off floor).
    let g32 = Arc::new(Graph::<f32>::path(N_NODES));
    let lap32 = Arc::new(Laplacian::assemble_combinatorial(&g32));

    let g64 = Arc::new(Graph::<f64>::path(N_NODES));
    let lap64 = Arc::new(Laplacian::assemble_combinatorial(&g64));
    let decomp = jacobi_eig(&lap64);
    let f0_f32 = GraphSignal::from_fn(Arc::clone(&g32), |i| ((i as f32 * 0.31).sin() + 1.0) * 0.5);
    let f0_vals64: Vec<f64> = f0_f32.values().iter().map(|&v| v as f64).collect();
    let oracle64 = heat_oracle(&decomp, &f0_vals64, Arc::clone(&g64), T);

    let errs: Vec<f64> = N_VALUES_F32
        .iter()
        .map(|&n_steps| {
            let tau = T as f32 / n_steps as f32;
            let chernoff = GraphHeat4thChernoff::new(Arc::clone(&lap32));
            let mut pool = ScratchPool::<f32>::new();
            let mut cur = f0_f32.clone();
            let mut nxt = f0_f32.clone();
            for _ in 0..n_steps {
                chernoff.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
                core::mem::swap(&mut cur, &mut nxt);
            }
            cur.values()
                .iter()
                .zip(oracle64.values().iter())
                .map(|(&a, &b)| (a as f64 - b).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();

    for (&n_steps, &err) in N_VALUES_F32.iter().zip(errs.iter()) {
        println!("G10 f32 n_steps={n_steps:4}, err={err:.4e}");
    }
    let slope = log_log_slope(&N_VALUES_F32, &errs);
    println!("G10 f32 slope = {slope:.4}");
    assert!(
        slope <= -3.50,
        "G10 FAIL f32: slope {slope:.4} > -3.50 (order-4 f32 gate per ADR-0046)"
    );
}
