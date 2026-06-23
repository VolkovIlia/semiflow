//! Eigenmode parity tests for `GraphHeatChernoff<F>` against the Jacobi oracle.
//!
//! For an eigenvector `φ_k` of `L_G` with eigenvalue `λ_k`, the order-1 Chernoff step
//! satisfies `S(τ) φ_k = φ_k − τ L_G φ_k = (1 − τ λ_k) φ_k` **exactly**.
//! These tests verify this to near-machine-precision thresholds.
//!
//! The Jacobi oracle is inlined here as a self-contained test helper.
//!
//! See ADR-0047 AC-3 and ADR-0050.

// Jacobi oracle uses compact numeric single-char names (n, a, q, p, …)
// and index-based dense-matrix loops — standard for inline numerical code.
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)] // usize → f64/f32 in initializers
#![allow(clippy::cast_lossless)] // f32/u32 → f64 widening casts
#![allow(clippy::cast_possible_truncation)] // usize → u32 node index
#![allow(clippy::needless_range_loop)] // Jacobi sweep: index-based loops required
#![allow(clippy::too_many_lines)] // jacobi_eig is a dense O(n³) routine

use std::sync::Arc;

use semiflow::{
    graph::{Graph, Laplacian},
    graph_heat::GraphHeatChernoff,
    graph_signal::GraphSignal,
    ChernoffFunction, Discrete, ScratchPool, State,
};

// ---------------------------------------------------------------------------
// Inline symmetric Jacobi eigendecomposition (test helper)
// ---------------------------------------------------------------------------

struct EigDecomp<F: Copy> {
    eigenvalues: Vec<F>,
    /// Column-major: column k occupies indices k*n .. (k+1)*n.
    eigenvectors_col_major: Vec<F>,
    #[allow(dead_code)]
    n: usize,
}

// --- f64 ---

fn jacobi_eig_f64(lap: &Laplacian<f64>) -> EigDecomp<f64> {
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
        // Off-diagonal Frobenius^2 (explicit nested loops to avoid closure borrow issues).
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

        let (c, s) = jacobi_cs_f64(a[p * n + p], a[p * n + qi], a[qi * n + qi]);
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
        n,
    }
}

fn jacobi_cs_f64(app: f64, apq: f64, aqq: f64) -> (f64, f64) {
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    (c, t * c)
}

// --- f32 (mirrors f64) ---

fn jacobi_eig_f32(lap: &Laplacian<f32>) -> EigDecomp<f32> {
    let n = lap.n_nodes();
    let mut a = vec![0.0_f32; n * n];
    for i in 0..n {
        for k in lap.row_ptr()[i]..lap.row_ptr()[i + 1] {
            a[i * n + lap.col_idx()[k] as usize] = lap.vals()[k];
        }
    }
    let mut q = vec![0.0_f32; n * n];
    for k in 0..n {
        q[k + k * n] = 1.0;
    }

    let tol = 1e-10_f32;
    for _ in 0..(200 * n) {
        let mut off2 = 0.0_f32;
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
        let diag2: f32 = (0..n)
            .map(|k| {
                let v = a[k * n + k];
                v * v
            })
            .sum();
        if off2 < tol * diag2 {
            break;
        }

        let (mut p, mut qi) = (0usize, 1usize);
        let mut mx = 0.0_f32;
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

        let (c, s) = jacobi_cs_f32(a[p * n + p], a[p * n + qi], a[qi * n + qi]);
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

    let mut pairs: Vec<(f32, usize)> = (0..n).map(|k| (a[k * n + k], k)).collect();
    pairs.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let eigenvalues: Vec<f32> = pairs.iter().map(|&(v, _)| v).collect();
    let mut evecs = vec![0.0_f32; n * n];
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
        n,
    }
}

fn jacobi_cs_f32(app: f32, apq: f32, aqq: f32) -> (f32, f32) {
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    (c, t * c)
}

// ---------------------------------------------------------------------------
// Tests: single-step eigenmode parity
//
// S(τ) φ_k = (1 − τ λ_k) φ_k  (exact for order-1 Chernoff applied to eigenvec)
// ---------------------------------------------------------------------------

#[test]
fn graph_heat_apply_into_eigenmode_parity_f64() {
    let n = 16_usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let decomp = jacobi_eig_f64(&lap);
    let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));

    let k = 3;
    let lambda_k = decomp.eigenvalues[k];
    let mut phi = GraphSignal::zeros(Arc::clone(&g));
    for i in 0..n {
        phi.set(i as u32, decomp.eigenvectors_col_major[i + k * n]);
    }

    let tau = 0.01_f64;
    let mut dst = phi.clone();
    let mut scratch = ScratchPool::<f64>::new();
    chernoff
        .apply_into(tau, &phi, &mut dst, &mut scratch)
        .expect("apply_into f64 must succeed");

    let mut expected = phi.clone();
    expected.scale_into(1.0 - tau * lambda_k);

    let mut diff = dst.clone();
    diff.axpy_into(-1.0, &expected);
    let err = diff.norm_sup();
    assert!(err < 1e-12, "f64 parity drift {err:.3e} >= 1e-12");
}

#[test]
fn graph_heat_apply_into_eigenmode_parity_f32() {
    let n = 16_usize;
    let g = Arc::new(Graph::<f32>::path(n));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let decomp = jacobi_eig_f32(&lap);
    let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));

    let k = 3;
    let lambda_k = decomp.eigenvalues[k];
    let mut phi = GraphSignal::zeros(Arc::clone(&g));
    for i in 0..n {
        phi.set(i as u32, decomp.eigenvectors_col_major[i + k * n]);
    }

    let tau = 0.01_f32;
    let mut dst = phi.clone();
    let mut scratch = ScratchPool::<f32>::new();
    chernoff
        .apply_into(tau, &phi, &mut dst, &mut scratch)
        .expect("apply_into f32 must succeed");

    let mut expected = phi.clone();
    expected.scale_into(1.0 - tau * lambda_k);

    let mut diff = dst.clone();
    diff.axpy_into(-1.0, &expected);
    let err = diff.norm_sup();
    assert!(err < 1e-5, "f32 parity drift {err:.3e} >= 1e-5");
}
