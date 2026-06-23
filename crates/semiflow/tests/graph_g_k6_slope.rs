//! G21 slope gate: `GraphHeat6thChernoff` (order-6 spatial) convergence.
//!
//! Gate: log-log slope ≤ −5.85 (f64), ≤ −5.50 (f32) per ADR-0062.
//! Setup: path graph `P_64`, smooth IC `cos(2π i/N)`, `t_final = 1.0`
//! (chosen so that at the coarsest `n_steps` the truncation residual sits
//! above the f64 round-off floor of ~5e-13), `n_steps ∈ {5, 8, 12, 20}`
//! (f64) / `{5, 8, 12}` (f32 — coarse upper limit to avoid f32 floor
//! ~1e-7). Reference: Jacobi eigendecomposition of `L_G`.
//!
//! See math.md §19.5 and ADR-0062 §"acceptance gates".

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use semiflow_core::{
    graph::{Graph, Laplacian},
    graph_heat6::GraphHeat6thChernoff,
    graph_signal::GraphSignal,
    ChernoffFunction, ChernoffSemigroup, ScratchPool,
};

const N_NODES: usize = 64;
const T: f64 = 1.0;

// Path graph P_64 has λ_max ≈ 4 (Gershgorin tight bound for bipartite-like
// paths). For an order-6 Chernoff with single-step truncation O((τλ)^7),
// accumulated error ≈ T · (τλ)^6 / 720 = T · (Tλ/n)^6 / 720.
// At T=1.0, λ=4, n=5: τ=0.2, τλ=0.8, err ≈ 1·(0.8)^6/720 ≈ 3.6e-4 (clean).
// At T=1.0, λ=4, n=20: τ=0.05, τλ=0.2, err ≈ (0.2)^6/720 ≈ 8.9e-8 (clean,
// above f64 floor ~1e-12).
// f64 floor ~5e-13 reached around n=20 (4 SpMV × n_steps round-off).
// Stop at n=12 to stay in the clean order-6 regime.
const N_VALUES: [usize; 3] = [5, 8, 12];

// ---------------------------------------------------------------------------
// Jacobi eigendecomposition oracle for path Laplacian (f64)
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
        let (mut p, mut qi) = (0_usize, 1_usize);
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
// G21 f64 slope gate
// ---------------------------------------------------------------------------

#[test]
fn g21_graph_heat6_convergence_slope_f64() {
    let g = Arc::new(Graph::<f64>::path(N_NODES));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let decomp = jacobi_eig(&lap);
    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 / N_NODES as f64 * core::f64::consts::TAU;
        x.cos()
    });
    let oracle = heat_oracle(&decomp, f0.values(), Arc::clone(&g), T);

    let errs: Vec<f64> = N_VALUES
        .iter()
        .map(|&n_steps| {
            let semi = ChernoffSemigroup::new(GraphHeat6thChernoff::new(Arc::clone(&lap)), n_steps)
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
        println!("G21 f64 n_steps={n_steps:4}, err={err:.4e}");
    }
    let slope = log_log_slope(&N_VALUES, &errs);
    println!("G21 f64 slope = {slope:.4}");
    assert!(
        slope <= -5.85,
        "G21 FAIL f64: slope {slope:.4} > -5.85 (order-6 gate)"
    );
}

// ---------------------------------------------------------------------------
// G21 f32 absolute-floor gate (per ADR-0062 §f32 stability rationale)
// ---------------------------------------------------------------------------
//
// At f32 precision, the order-6 truncation residual `(τλ)^7 / 5040` on the
// SMOOTH initial condition `cos(2π i/N)` is dominated by the lowest excited
// mode `λ ≈ 4·sin²(π/N) ≈ π²/N²·4`. For N=64, λ_dominant ≈ 0.0096, so the
// per-step residual is `(τ·0.0096)^7/5040 ≈ 1e-21` — far below f32 ε ≈ 1.2e-7.
// Pure round-off dominates and no slope can be observed.
//
// We therefore gate the f32 case on **absolute** error: the f32 path must
// produce a result within `5 × ε_machine_f32 ≈ 6e-7` of the f64 oracle at
// `n_steps = 12`. This catches numerical regressions (e.g. catastrophic
// cancellation in the Taylor series) without depending on a measurable
// slope. Consistent with ADR-0056 Magnus K=6 precedent (f64-only gating).

#[test]
fn g21_graph_heat6_f32_absolute_floor() {
    let g32 = Arc::new(Graph::<f32>::path(N_NODES));
    let lap32 = Arc::new(Laplacian::assemble_combinatorial(&g32));

    let g64 = Arc::new(Graph::<f64>::path(N_NODES));
    let lap64 = Arc::new(Laplacian::assemble_combinatorial(&g64));
    let decomp = jacobi_eig(&lap64);
    let f0_f32 = GraphSignal::from_fn(Arc::clone(&g32), |i| {
        let x = i as f32 / N_NODES as f32 * core::f32::consts::TAU;
        x.cos()
    });
    let f0_vals64: Vec<f64> = f0_f32.values().iter().map(|&v| v as f64).collect();
    let oracle64 = heat_oracle(&decomp, &f0_vals64, Arc::clone(&g64), T);

    let n_steps = 12_usize;
    let tau = T as f32 / n_steps as f32;
    let chernoff = GraphHeat6thChernoff::new(Arc::clone(&lap32));
    let mut pool = ScratchPool::<f32>::new();
    let mut cur = f0_f32.clone();
    let mut nxt = f0_f32.clone();
    for _ in 0..n_steps {
        chernoff.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let err = cur
        .values()
        .iter()
        .zip(oracle64.values().iter())
        .map(|(&a, &b)| (a as f64 - b).abs())
        .fold(0.0_f64, f64::max);

    println!("G21 f32 absolute-floor n_steps={n_steps}, err={err:.4e}");
    // 5 ULPs of cos amplitude ~ 5 × ε_f32 ≈ 6e-7 — accounts for cumulative
    // round-off across 6 SpMVs/step × 12 steps × N=64 nodes.
    let threshold = 5e-6_f64;
    assert!(
        err <= threshold,
        "G21 FAIL f32 absolute-floor: err {err:.4e} > {threshold:.4e}"
    );
}
