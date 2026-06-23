//! G7 slope gate: `GraphHeatChernoff` convergence rate is order 1 in `N_steps`.
//!
//! Uses `erdos_renyi(64, 0.15, 0xDEAD_BEEF)` and compares Chernoff approximation
//! against the closed-form spectral oracle `e^{−t L_G} f₀` (Jacobi eigdecomp).
//!
//! Gate: log-log slope of sup-error vs `N_steps` must be ≤ −0.95 (f64)
//! and ≤ −0.90 (f32).
//!
//! The oracle is inlined as a self-contained test helper.
//! See ADR-0047 AC-5 and ADR-0050.

// The inlined Jacobi oracle uses compact numeric single-char names (n, a, q, p …)
// and index-based loops that are idiomatic for dense linear algebra.
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)] // usize/u32 → f64/f32 in test initializers
#![allow(clippy::cast_lossless)] // f32 → f64, u32 → f64 widening casts
#![allow(clippy::cast_possible_truncation)] // usize → u32 node index (n_nodes ≤ 64)
#![allow(clippy::needless_range_loop)] // Jacobi sweep uses index-based indexing
#![allow(clippy::too_many_lines)] // jacobi_eig is a dense algorithm, kept intact

use std::sync::Arc;

use semiflow::{
    graph::{Graph, Laplacian},
    graph_heat::GraphHeatChernoff,
    graph_signal::GraphSignal,
    ChernoffFunction, ChernoffSemigroup, ScratchPool,
};

// N_steps sweep values.
const N_VALUES: [usize; 5] = [25, 50, 100, 200, 400];

// ---------------------------------------------------------------------------
// Inline Jacobi eigendecomposition oracle
// ---------------------------------------------------------------------------

struct EigDecomp {
    eigenvalues: Vec<f64>,
    /// Column-major: column k in [k*n .. (k+1)*n].
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
        let (c, s) = jacobi_cs(a[p * n + p], a[p * n + qi], a[qi * n + qi]);
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

fn jacobi_cs(app: f64, apq: f64, aqq: f64) -> (f64, f64) {
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    (c, t * c)
}

fn heat_oracle(
    decomp: &EigDecomp,
    f0_vals: &[f64],
    g: Arc<Graph<f64>>,
    t: f64,
) -> GraphSignal<f64> {
    let n = f0_vals.len();
    let mut alpha = vec![0.0_f64; n];
    for k in 0..n {
        let mut dot = 0.0_f64;
        for j in 0..n {
            dot += decomp.eigenvectors_col_major[j + k * n] * f0_vals[j];
        }
        alpha[k] = dot;
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

// ---------------------------------------------------------------------------
// Log-log OLS slope helper
// ---------------------------------------------------------------------------

fn log_log_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let lxs: Vec<f64> = xs.iter().map(|&x| x.ln()).collect();
    let lys: Vec<f64> = ys.iter().map(|&y| y.ln()).collect();
    let sx: f64 = lxs.iter().sum();
    let sy: f64 = lys.iter().sum();
    let sxx: f64 = lxs.iter().map(|&x| x * x).sum();
    let sxy: f64 = lxs.iter().zip(lys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sxy - sx * sy) / (m * sxx - sx * sx)
}

// ---------------------------------------------------------------------------
// G7 f64 slope gate
// ---------------------------------------------------------------------------

#[test]
fn g7_graph_heat_convergence_slope_f64() {
    let n_nodes = 64_usize;
    let t = 0.5_f64;
    let g = Arc::new(Graph::<f64>::erdos_renyi(n_nodes, 0.15, 0xDEAD_BEEF));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));

    let decomp = jacobi_eig(&lap);

    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - (n_nodes as f64) / 2.0;
        (-x * x / 16.0).exp()
    });
    let f0_vals: Vec<f64> = f0.values().to_vec();
    let u_oracle = heat_oracle(&decomp, &f0_vals, Arc::clone(&g), t);

    let errs: Vec<f64> = N_VALUES
        .iter()
        .map(|&n_steps| {
            let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));
            let semi = ChernoffSemigroup::new(chernoff, n_steps).expect("n >= 1");
            let u = semi.evolve(t, &f0).expect("evolve must succeed");
            u.values()
                .iter()
                .zip(u_oracle.values().iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();

    for (&n_steps, &err) in N_VALUES.iter().zip(errs.iter()) {
        println!("n_steps={n_steps:4}, err={err:.4e}");
    }

    let n_vals_f64: Vec<f64> = N_VALUES.iter().map(|&n| n as f64).collect();
    let slope = log_log_slope(&n_vals_f64, &errs);
    println!("G7 f64 slope = {slope:.4}");

    assert!(
        slope <= -0.95,
        "G7 f64 slope {slope:.4} > -0.95 (order-1 gate)"
    );
}

// ---------------------------------------------------------------------------
// G7 f32 slope gate (manual loop — ChernoffSemigroup bounded to f64)
// ---------------------------------------------------------------------------

#[test]
fn g7_graph_heat_convergence_slope_f32() {
    let n_nodes = 64_usize;
    let t = 0.5_f32;
    let g = Arc::new(Graph::<f32>::erdos_renyi(n_nodes, 0.15, 0xDEAD_BEEF));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));

    // Build f64 oracle on the same graph topology (same seed → same edges).
    let g64 = Arc::new(Graph::<f64>::erdos_renyi(n_nodes, 0.15, 0xDEAD_BEEF));
    let lap64 = Arc::new(Laplacian::assemble_combinatorial(&g64));
    let decomp = jacobi_eig(&lap64);

    let f0_f32 = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f32 - (n_nodes as f32) / 2.0;
        (-x * x / 16.0).exp()
    });
    let f0_vals64: Vec<f64> = f0_f32.values().iter().map(|&v| v as f64).collect();
    let u_oracle64 = heat_oracle(&decomp, &f0_vals64, Arc::clone(&g64), t as f64);

    let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));

    let errs: Vec<f64> = N_VALUES
        .iter()
        .map(|&n_steps| {
            let tau = t / n_steps as f32;
            let mut pool = ScratchPool::<f32>::new();
            let mut current = f0_f32.clone();
            let mut next = f0_f32.clone();
            for _ in 0..n_steps {
                chernoff
                    .apply_into(tau, &current, &mut next, &mut pool)
                    .expect("apply_into f32 must succeed");
                core::mem::swap(&mut current, &mut next);
            }
            current
                .values()
                .iter()
                .zip(u_oracle64.values().iter())
                .map(|(&a, &b)| (a as f64 - b).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();

    for (&n_steps, &err) in N_VALUES.iter().zip(errs.iter()) {
        println!("n_steps={n_steps:4}, err={err:.4e}");
    }

    let n_vals_f64: Vec<f64> = N_VALUES.iter().map(|&n| n as f64).collect();
    let slope = log_log_slope(&n_vals_f64, &errs);
    println!("G7 f32 slope = {slope:.4}");

    assert!(
        slope <= -0.90,
        "G7 f32 slope {slope:.4} > -0.90 (order-1 gate)"
    );
}
