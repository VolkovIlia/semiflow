//! `G_TT_COUPLED_EXACT` — P4' exactness gate + `no_lu_in_coupling` audit
//! (`RELEASE_BLOCKING`; `#[ignore]` + `slow-tests` gated for the dense-expm levels).
//!
//! # Gate spec (§10.13.2 / §10.13.2.bis, ADR-0162)
//!
//! Reference: `u_ref = expm(T·L_h^{dx})·u₀` where
//! `L_h^{dx} = Σ_j a_j·D2_j/dx² + Σ_{(j,j+1)} 2ρ√(aⱼaₖ)·(D1_j/dx)⊗(D1_k/dx)`
//! assembled as a dense `n^d × n^d` matrix, computed via Padé[6/6] scaling-squaring
//! implemented entirely in this test file (ZERO reuse of `tt_coupled_pair` or
//! `tt_spectral` production code — genuine independent reference).
//!
//! Assertion (HARD, NEVER weakened): `rel_l2(coupled_tt_dense, u_ref) ≤ 1e-12`.
//!
//! Grid selection (tractability):
//! - d=3 ρ=0.6 n∈{9,11,13}: 13³=2197 → dense matrix 2197²≈4.8M entries, feasible.
//! - d=4 ρ=0.4 n∈{5,7}:     7⁴=2401 → dense matrix 2401²≈5.8M entries, feasible.
//!   n=13 at d=4 is 13⁴=28561 → 28561²≈8.2e8 entries (~6.5 GB), INFEASIBLE — excluded.
//!
//! Anti-vacuity (HARD asserts): ρ≠0; h/dx non-integer (frac>0.05);
//! generator assembled in this file, NOT via `tt_spectral::pair_expsym_real`.
//!
//! # Run
//! ```bash
//! cargo test -p semiflow-core --features slow-tests \
//!   --test g_tt_coupled_converge -- --ignored --nocapture
//! ```
//!
//! # `no_lu_in_coupling` audit (non-ignored, fast)
//! Separate test `no_lu_in_coupling` scans production source files for
//! `lu_solve_inplace` / `dense_expm` outside `#[cfg(test)]` blocks.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)] // usize→u32 for .pow(): d ≤ 4, n ≤ 13 in test
#![allow(clippy::cast_possible_wrap)]       // u32→i32 for .powi(): scaling factor ≤ 30
#![allow(clippy::many_single_char_names)]   // n, d, r, a, s, etc. are standard math variable names
#![allow(clippy::needless_range_loop)]      // index loops use cross-index arithmetic (Kronecker)
#![allow(clippy::unreadable_literal)]       // LCG/expm coefficients are mathematical identifiers

extern crate alloc;
use alloc::vec::Vec;

use semiflow::{CoupledTtChernoff, CouplingTopology, TtState};

// ═══════════════════════════════════════════════════════════════════════════
// §A — Pre-registered parameters (NORMATIVE, frozen before run)
// ═══════════════════════════════════════════════════════════════════════════

const T: f64 = 0.05;
const X_MIN: f64 = 0.0;
const X_MAX: f64 = 1.0;
const EPS_ROUND: f64 = 1e-13;
const EXACTNESS_GATE: f64 = 1e-12;

/// d=3 parameters: ρ=0.6 (SPD: |ρ|<1/√2≈0.707 ✓; interior c=a/2, |ρ|<0.707 ✓)
const RHO3: f64 = 0.6;
const A3: [f64; 3] = [0.8, 0.6, 0.4];
/// n∈{9,11,13}, `n_steps` so τ≈0.35·dx² (τ-only sweep NOT included — §10.3 forbidden)
const N3: [usize; 3] = [9, 11, 13];
const STEPS3: [usize; 3] = [9, 14, 20]; // n=13: 20 steps → h/dx≈1.073, frac≈0.073 >0.05 ✓

/// d=4 parameters: ρ=0.4 (interior c=a/2, |ρ|<0.5 ✓; full-tensor |ρ|<1/√3≈0.577 ✓)
const RHO4: f64 = 0.4;
const A4: [f64; 4] = [0.8, 0.6, 0.4, 0.5];
/// n∈{5,7}; n=13 at d=4 is 13⁴=28561 → ~6.5 GB matrix, infeasible — excluded
const N4: [usize; 2] = [5, 7];
const STEPS4: [usize; 2] = [3, 5];

// ═══════════════════════════════════════════════════════════════════════════
// §B — Local dense matrix helpers (independent of production code)
// ═══════════════════════════════════════════════════════════════════════════

fn mat_vec_local(a: &[f64], v: &[f64], m: usize) -> Vec<f64> {
    (0..m)
        .map(|i| (0..m).map(|j| a[i * m + j] * v[j]).sum())
        .collect()
}

fn mat_mat_local(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * m];
    for i in 0..m {
        for k in 0..m {
            let aik = a[i * m + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..m {
                c[i * m + j] += aik * b[k * m + j];
            }
        }
    }
    c
}

fn mat_eye_local(m: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; m * m];
    for i in 0..m {
        e[i * m + i] = 1.0;
    }
    e
}

fn mat_norm_inf_local(a: &[f64], m: usize) -> f64 {
    (0..m)
        .map(|i| (0..m).map(|j| a[i * m + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// LU factorize `a_lu` in-place; returns pivot vector. Does NOT apply to RHS.
fn lu_factor(a_lu: &mut [f64], m: usize) -> Vec<usize> {
    let mut piv: Vec<usize> = (0..m).collect();
    for col in 0..m {
        let mut mx = a_lu[col * m + col].abs();
        let mut mr = col;
        for row in (col + 1)..m {
            let v = a_lu[row * m + col].abs();
            if v > mx {
                mx = v;
                mr = row;
            }
        }
        if mr != col {
            for j in 0..m {
                a_lu.swap(col * m + j, mr * m + j);
            }
            piv.swap(col, mr);
        }
        let pv = a_lu[col * m + col];
        if pv.abs() < 1e-300 {
            continue;
        }
        let inv = 1.0 / pv;
        for row in (col + 1)..m {
            let f = a_lu[row * m + col] * inv;
            a_lu[row * m + col] = f;
            for j in (col + 1)..m {
                let acj = a_lu[col * m + j];
                a_lu[row * m + j] -= f * acj;
            }
        }
    }
    piv
}

/// Solve L·U·x = P·b using precomputed LU factors. Result in b.
fn lu_solve_factored(a_lu: &[f64], piv: &[usize], b: &mut [f64], m: usize) {
    // Permute b according to piv
    let tmp = b.to_vec();
    for i in 0..m {
        b[i] = tmp[piv[i]];
    }
    // Forward substitution L·y = b (unit lower triangular)
    for col in 0..m {
        for row in (col + 1)..m {
            b[row] -= a_lu[row * m + col] * b[col];
        }
    }
    // Backward substitution U·x = y
    for row in (0..m).rev() {
        for col in (row + 1)..m {
            b[row] -= a_lu[row * m + col] * b[col];
        }
        let d = a_lu[row * m + row];
        if d.abs() > 1e-300 {
            b[row] /= d;
        }
    }
}

/// Local Padé[6/6] + scaling-squaring expm (independent of `tt_dense_expm`).
fn expm_local(a: &[f64], m: usize) -> Vec<f64> {
    let norm = mat_norm_inf_local(a, m);
    let mut s = 0u32;
    let mut thr = 0.5_f64;
    while norm > thr && s < 30 {
        s += 1;
        thr *= 2.0;
    }
    let sc = 0.5_f64.powi(s as i32);
    let mut as_ = a.to_vec();
    for v in &mut as_ {
        *v *= sc;
    }
    let c: [f64; 7] = [
        1.0,
        0.5,
        5.0 / 44.0,
        1.0 / 66.0,
        1.0 / 792.0,
        1.0 / 15840.0,
        1.0 / 665280.0,
    ];
    let a2 = mat_mat_local(&as_, &as_, m);
    let a4 = mat_mat_local(&a2, &a2, m);
    let a6 = mat_mat_local(&a2, &a4, m);
    let blend = |coeffs: &[(usize, f64)]| -> Vec<f64> {
        let mut acc = vec![0.0f64; m * m];
        for &(k, ck) in coeffs {
            let src = match k {
                0 => mat_eye_local(m),
                2 => a2.clone(),
                4 => a4.clone(),
                _ => a6.clone(),
            };
            for (a, &s) in acc.iter_mut().zip(&src) {
                *a += ck * s;
            }
        }
        acc
    };
    let v = blend(&[(0, c[0]), (2, c[2]), (4, c[4]), (6, c[6])]);
    let inner = blend(&[(0, c[1]), (2, c[3]), (4, c[5])]);
    let u = mat_mat_local(&as_, &inner, m);
    let mut p = v.clone();
    for (pi, &ui) in p.iter_mut().zip(&u) {
        *pi += ui;
    }
    let mut q = v;
    for (qi, &ui) in q.iter_mut().zip(&u) {
        *qi -= ui;
    }
    let piv = lu_factor(&mut q, m); // factorize q ONCE
    let mut exp_s = vec![0.0f64; m * m];
    for col in 0..m {
        let mut rhs: Vec<f64> = (0..m).map(|row| p[row * m + col]).collect();
        lu_solve_factored(&q, &piv, &mut rhs, m); // reuse factored q
        for row in 0..m {
            exp_s[row * m + col] = rhs[row];
        }
    }
    for _ in 0..s {
        exp_s = mat_mat_local(&exp_s, &exp_s, m);
    }
    exp_s
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Dense full-dimensional generator L_h^{dx}
//
// Independent of tt_spectral::pair_expsym_real (assembles its own dense L_h).
// L_h = Σ_j a_j·D2_j/dx² + Σ_{(j,j+1)} 2ρ√(aⱼaₖ)·(D1_j/dx)⊗(D1_k/dx)
// Periodic BCs. Row/column index: flat = Σ_j i_j · n^(d-1-j).
// ═══════════════════════════════════════════════════════════════════════════

fn stride(d: usize, j: usize, n: usize) -> usize {
    n.pow((d - 1 - j) as u32)
}

fn add_lapl(l: &mut [f64], d: usize, j: usize, n: usize, coeff: f64) {
    let tot = n.pow(d as u32);
    let s = stride(d, j, n);
    for idx in 0..tot {
        let ij = (idx / s) % n;
        let base = idx - ij * s;
        let f = base + ((ij + 1) % n) * s;
        let b = base + ((ij + n - 1) % n) * s;
        l[idx * tot + f] += coeff;
        l[idx * tot + b] += coeff;
        l[idx * tot + idx] -= 2.0 * coeff;
    }
}

fn add_cross(l: &mut [f64], d: usize, j: usize, k: usize, n: usize, coeff: f64) {
    let tot = n.pow(d as u32);
    let sj = stride(d, j, n);
    let sk = stride(d, k, n);
    for idx in 0..tot {
        let ij = (idx / sj) % n;
        let ik = (idx / sk) % n;
        let base = idx - ij * sj - ik * sk;
        let ff = base + ((ij + 1) % n) * sj + ((ik + 1) % n) * sk;
        let fb = base + ((ij + 1) % n) * sj + ((ik + n - 1) % n) * sk;
        let bf = base + ((ij + n - 1) % n) * sj + ((ik + 1) % n) * sk;
        let bb = base + ((ij + n - 1) % n) * sj + ((ik + n - 1) % n) * sk;
        l[idx * tot + ff] += coeff;
        l[idx * tot + fb] -= coeff;
        l[idx * tot + bf] -= coeff;
        l[idx * tot + bb] += coeff;
    }
}

/// Assemble dense n^d × n^d generator `L_h` (independent reference).
fn build_l_full(d: usize, n: usize, dx: f64, a: &[f64], rho: f64) -> Vec<f64> {
    let tot = n.pow(d as u32);
    let mut l = vec![0.0f64; tot * tot];
    for j in 0..d {
        add_lapl(&mut l, d, j, n, a[j] / (dx * dx));
    }
    for j in 0..(d - 1) {
        let k = j + 1;
        let coeff = 2.0 * rho * (a[j] * a[k]).sqrt() * 0.25 / (dx * dx);
        add_cross(&mut l, d, j, k, n, coeff);
    }
    l
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — TT-to-dense + IC + metric
// ═══════════════════════════════════════════════════════════════════════════

fn tt_to_dense(state: &TtState<f64>, d: usize, n: usize) -> Vec<f64> {
    let tot = n.pow(d as u32);
    (0..tot)
        .map(|flat| {
            let mut rem = flat;
            let mut mi = [0usize; 6];
            for j in (0..d).rev() {
                mi[j] = rem % n;
                rem /= n;
            }
            let mut v = vec![1.0f64; 1];
            for j in 0..d {
                let core = &state.cores[j];
                let rl = core.r_left;
                let rr = core.r_right;
                let mut w = vec![0.0f64; rr];
                for il in 0..rl {
                    for ir in 0..rr {
                        w[ir] += v[il] * core.get(il, mi[j], ir);
                    }
                }
                v = w;
            }
            v[0]
        })
        .collect()
}

fn ic_1d(n: usize) -> Vec<f64> {
    let cx = 0.5 * (X_MIN + X_MAX);
    (0..n)
        .map(|i| {
            let x = X_MIN + i as f64 * (X_MAX - X_MIN) / (n as f64 - 1.0);
            (-(x - cx).powi(2) / 0.04).exp()
        })
        .collect()
}

fn ic_dense(d: usize, n: usize) -> Vec<f64> {
    let s = ic_1d(n);
    (0..n.pow(d as u32))
        .map(|idx| {
            let mut rem = idx;
            let mut p = 1.0f64;
            for _ in 0..d {
                p *= s[rem % n];
                rem /= n;
            }
            p
        })
        .collect()
}

fn ic_tt(d: usize, n: usize) -> TtState<f64> {
    TtState::rank1_separable((0..d).map(|_| ic_1d(n)).collect())
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let nb: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if nb < 1e-300 {
        return 0.0;
    }
    let e: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    e / nb
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — Run one (d, n, n_steps, a, rho) level; return (rel-L2, h/dx, peak_r)
//
// ANTI-VACUITY:
//   1. ρ≠0 (genuine coupling asserted in gate below)
//   2. h/dx non-integer (frac>0.05 asserted per level)
//   3. Reference = expm(T·L_h) assembled here — ZERO TT/spectral code reuse
// ═══════════════════════════════════════════════════════════════════════════

fn run_level(d: usize, n: usize, ns: usize, a: &[f64], rho: f64) -> (f64, f64, usize) {
    let dx = (X_MAX - X_MIN) / (n as f64 - 1.0);
    let tau = T / ns as f64;
    let h_over_dx = 2.0 * (a[0] * tau).sqrt() / dx;

    // CoupledTtChernoff (spectral P3'' solver-free path)
    let ev = CoupledTtChernoff::new(
        a.to_vec(),
        vec![0.0f64; d],
        0.0,
        CouplingTopology::Tridiagonal(rho),
        vec![(X_MIN, X_MAX); d],
        EPS_ROUND,
    );
    let mut tt = ic_tt(d, n);
    ev.evolve(T, ns, &mut tt);
    let peak_r = tt.peak_rank();
    let tt_dense = tt_to_dense(&tt, d, n);

    // Independent reference: expm(T·L_h^{dx})·u₀ (no TT/spectral code)
    let tot = n.pow(d as u32);
    let mut l = build_l_full(d, n, dx, a, rho);
    for v in &mut l {
        *v *= T;
    }
    let e = expm_local(&l, tot);
    let ic = ic_dense(d, n);
    let u_ref = mat_vec_local(&e, &ic, tot);

    (rel_l2(&tt_dense, &u_ref), h_over_dx, peak_r)
}

// ═══════════════════════════════════════════════════════════════════════════
// §F — Run all levels for a given d; print table; assert exactness per level
// ═══════════════════════════════════════════════════════════════════════════

fn run_exactness(d: usize, grids: &[usize], steps: &[usize], a: &[f64], rho: f64) {
    println!();
    println!("{}", "─".repeat(72));
    println!("d={d} ρ={rho:.2} — exactness gate (§10.13.2 / §10.13.2.bis, ADR-0162)");
    println!("  Reference: expm(T·L_h^{{dx}})·u₀ — dense n^{d} matrix (LOCAL build)");
    println!("  Anti-vacuity: ρ={rho:.2}≠0; h/dx non-integer; independent generator.");
    println!("  n     | steps | τ         | h/dx   | peak_r | rel-L2 err   | gate ≤1e-12");
    println!("  {}", "─".repeat(68));

    // Anti-vacuity: ρ≠0
    assert!(
        rho.abs() > 0.05,
        "ρ={rho:.4} near zero — genuine coupling required"
    );

    for i in 0..grids.len() {
        let n = grids[i];
        let ns = steps[i];
        let (err, hd, pk) = run_level(d, n, ns, a, rho);
        let ok = if err <= EXACTNESS_GATE { "✓" } else { "FAIL" };
        let dx = (X_MAX - X_MIN) / (n as f64 - 1.0);
        let tau = T / ns as f64;
        println!(
            "  {n:>5} | {ns:>5} | {tau:.4e} | {hd:.4} | {pk:>6} | {err:.4e}   | {ok}"
        );

        // Anti-vacuity: h/dx non-integer (frac>0.05)
        let frac = (hd - hd.round()).abs();
        assert!(
            frac > 0.05,
            "d={d} n={n}: h/dx={hd:.4} near-integer (frac={frac:.4} ≤ 0.05). \
             Shift scheme must not degenerate to integer lattice step."
        );

        // HARD EXACTNESS ASSERT (RELEASE_BLOCKING, NEVER weakened)
        assert!(
            err <= EXACTNESS_GATE,
            "G_TT_COUPLED_EXACT FAIL: d={d} n={n} rel-L2={err:.3e} > {EXACTNESS_GATE:.0e}. \
             Spectral pair factor is NOT exact for constant-coef correlated-Gaussian class. \
             Do NOT loosen this bound. Diagnosis: check tt_coupled_pair.rs pair_sweep_strang \
             and tt_spectral.rs apply_spectral_pair_to_panel."
        );
        let _ = dx; // silence unused warning
    }
    println!("  All levels: rel-L2 ≤ {EXACTNESS_GATE:.0e} ✓");
}

// ═══════════════════════════════════════════════════════════════════════════
// §G — G_TT_COUPLED_EXACT (RELEASE_BLOCKING exactness gate)
// ═══════════════════════════════════════════════════════════════════════════

/// `G_TT_COUPLED_EXACT` — P4' exactness gate (`RELEASE_BLOCKING`).
///
/// Proves `CoupledTtChernoff` (spectral pair factor P3'') is machine-exact
/// for the constant-coefficient correlated-Gaussian class at d∈{3,4}.
///
/// # Spec (§10.13.2 / §10.13.2.bis, ADR-0162)
/// Reference: `expm(T·L_h^{dx})·u₀`, dense `n^d × n^d` matrix assembled in this
/// test (INDEPENDENT — no `tt_spectral` or `tt_coupled_pair` code reused).
/// Assert: `rel_L2 ≤ 1e-12` at EVERY grid level for d∈{3,4}.
///
/// # Grid selection (tractability)
/// - d=3 n∈{9,11,13}: 13³=2197 dense, feasible.
/// - d=4 n∈{5,7}:     7⁴=2401 dense, feasible.
///   n=13 at d=4 → 28561² ≈ 6.5 GB, INFEASIBLE — excluded by design.
///
/// # No slope sub-check
/// For the exact constant-coef class the error is at machine floor (~1e-13–1e-14);
/// a slope test is degenerate. The τ-only fixed-dx sweep is RETIRED (§10.3:
/// shift-Laplacian diverges as h/dx→0 at fixed dx).
///
/// # Previously DEFERRED
/// This gate was `#[ignore]`-DEFERRED pending the rotated pair factor (P3'').
/// P3'' (spectral FFT-diagonal apply, solver-free) is now implemented
/// (`tt_spectral.rs`, `tt_coupled_pair.rs`). The d=2 self-check confirms
/// rel-L2 ≤ 1e-10 (`tt_coupled::tests::d2_exactness_self_check`). This gate
/// is the full d∈{3,4} confirmation. It is `RELEASE_BLOCKING`.
#[test]
#[ignore = "RELEASE_BLOCKING slow dense-expm exactness gate; run with: cargo run -p xtask -- test-flagship"]
fn g_tt_coupled_converge() {
    let bar = "═".repeat(72);
    println!("\n{bar}");
    println!("G_TT_COUPLED_EXACT — P4' exactness gate (RELEASE_BLOCKING)");
    println!("  §10.13.2 / §10.13.2.bis / ADR-0162 Amendment");
    println!("  Reference: expm(T·L_h^{{dx}})·u₀ (dense LOCAL build, zero TT code reuse)");
    println!("  Assertion: rel_L2 ≤ {EXACTNESS_GATE:.0e} at every level (HARD, never weakened)");
    println!("{bar}");

    run_exactness(3, &N3, &STEPS3, &A3, RHO3);
    run_exactness(4, &N4, &STEPS4, &A4, RHO4);

    println!("\n{bar}");
    println!("G_TT_COUPLED_EXACT PASS");
    println!("  d=3 ρ={RHO3} n∈{N3:?}: all levels rel-L2 ≤ {EXACTNESS_GATE:.0e}");
    println!("  d=4 ρ={RHO4} n∈{N4:?}: all levels rel-L2 ≤ {EXACTNESS_GATE:.0e}");
    println!("  Spectral pair factor (P3'') is EXACT for constant-coef correlated-Gaussian.");
    println!("  v9.1.0 strong claim (§10.13.3 form i) CERTIFIED by this gate.");
    println!("{bar}\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// §H — no_lu_in_coupling audit (non-ignored, fast, non-slow-tests)
//
// Statically proves the production coupling path (tt_coupled_pair.rs,
// tt_spectral.rs, tt_coupled.rs) contains no `lu_solve_inplace` or
// `dense_expm` call outside `#[cfg(test)]` blocks.
//
// Approach: scan source bytes; strip all content inside `#[cfg(test)] mod`
// blocks by tracking brace depth after the `#[cfg(test)]` marker.
// Then assert no occurrence of the forbidden substrings.
// ═══════════════════════════════════════════════════════════════════════════

/// Strip all items marked `#[cfg(test)]` from source text.
///
/// Handles both `#[cfg(test)] mod tests { ... }` and
/// `#[cfg(test)] fn foo() { ... }` by finding the opening `{` and
/// tracking brace depth to find the closing `}`.
fn strip_test_items(src: &str) -> String {
    let marker = "#[cfg(test)]";
    let mk = marker.as_bytes();
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        if i + mk.len() <= n && &bytes[i..i + mk.len()] == mk {
            // Find the first '{' after the marker (the item body opener).
            let mut j = i + mk.len();
            while j < n && bytes[j] != b'{' {
                j += 1;
            }
            if j < n {
                // Skip everything from the marker through the matching '}'.
                let mut depth = 0usize;
                while j < n {
                    if bytes[j] == b'{' {
                        depth += 1;
                    } else if bytes[j] == b'}' {
                        depth -= 1;
                        if depth == 0 {
                            j += 1;
                            break;
                        }
                    }
                    j += 1;
                }
                i = j;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Audit: coupling production path must NOT call `lu_solve_inplace` or `dense_expm`.
///
/// Reads `tt_coupled_pair.rs`, `tt_spectral.rs`, `tt_coupled.rs` source files,
/// strips `#[cfg(test)]` blocks, and asserts no forbidden solver call remains.
///
/// This test requires `--features slow-tests` (whole file gated) but is NOT `#[ignore]`,
/// so it runs whenever `slow-tests` is active (e.g. `cargo test --features slow-tests`).
#[test]
fn no_lu_in_coupling() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let src_dir = std::path::Path::new(manifest_dir).join("src");
    let targets = ["tt_coupled_pair.rs", "tt_spectral.rs", "tt_coupled.rs"];
    // Check for CALL patterns (with opening paren), not mere mentions in comments.
    // This avoids false positives from doc-comment lines like `// NO lu_solve_inplace`.
    let forbidden_calls = ["lu_solve_inplace(", "dense_expm("];
    for fname in &targets {
        let path = src_dir.join(fname);
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("no_lu_in_coupling: cannot read {fname}: {e}"));
        let production = strip_test_items(&raw);
        // Additionally strip line comments to avoid doc-comment hits.
        let prod_no_comments: String = production
            .lines()
            .map(|line| {
                if let Some(pos) = line.find("//") {
                    &line[..pos]
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        for &kw in &forbidden_calls {
            assert!(
                !prod_no_comments.contains(kw),
                "no_lu_in_coupling FAIL: `{kw}` found as a CALL in production code of {fname}. \
                 Coupling path must be solver-free (R2, §11.3, ADR-0162). \
                 A solver call outside #[cfg(test)] violates Theorem-6 R2."
            );
        }
    }
    println!("no_lu_in_coupling PASS");
    println!("  Checked: {targets:?}");
    println!("  Forbidden calls (outside #[cfg(test)]): {forbidden_calls:?}");
    println!("  All clean — coupling path is solver-free (R2 honoured).");
}
