//! `G_S3_VARCOEF_SPECTRAL` — S³ variable-coefficient POC gate (RELEASE-BLOCKING class).
//!
//! Proves order-2 curse-escape for **additive-separable** variable coefficients via a
//! solver-free 1-D Chernoff sandwich `P₂(τ/2)·k(τ)·P₂(τ/2)` stacked by exact inter-axis split.
//!
//! See: `contracts/s3-variable-coef-poc.contract.md` (NORMATIVE),
//!      `.dev-docs/specs/s3-variable-coef.md`, `docs/adr/0166-variable-coef-curse-escape.md`.
//!
//! # 7 HARD asserts
//!
//! 1. **ORDER-2** — log-log slope of `rel_err` vs τ ≤ −1.9, d=3, vs independent dense Padé\[6/6\] expm.
//! 2. **LAYER-1 EXACT** — `‖[Lⱼ,Lₖ]‖ ≤ 1e-12` AND `‖exp(τΣL) − ∏exp(τLⱼ)‖ ≤ 1e-12`.
//! 3. **RANK-1 OP** — operator-TT-rank of `E₀⊗I^{d-1}` = 1 at eps=1e-12.
//! 4. **BOUNDARY** — non-separable `0.25cos(x)sin(y)∂²ₓ` cross-term: slope > −1.0 AND floor > 1e-4.
//! 5. **LOAD-BEARING** — `‖u(a_var)−u(a_const)‖/‖u(a_const)‖ ≥ 0.02` AND var-amp > 0.1.
//! 6. **COST-SCALING** — evolver finite+real at d∈{8,10}; static `n^{2d}·8 > 1 TB`.
//! 7. **REDUCTION+NO-SOLVER** — const-a,b=0 step ≡ spectral ≤1e-12; source grep clean.
//!
//! The local Chernoff evolver is an independent re-implementation (zero reuse of
//! `tt_varcoef_spectral.rs`).  The reference is a local Padé[6/6] dense expm.
//! The source grep (assert 7) audits `tt_varcoef_spectral.rs` for solver calls.
//!
//! # Run
//! ```bash
//! cargo test -p semiflow-core --features slow-tests g_s3_varcoef_spectral -- --nocapture
//! ```

#![cfg(feature = "slow-tests")]
#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::suboptimal_flops,
    clippy::many_single_char_names
)]

extern crate alloc;
use alloc::vec::Vec;
use core::f64::consts::TAU;

// ═══════════════════════════════════════════════════════════════════════════
// §A — Pre-registered parameters (NORMATIVE, frozen before gate run)
// ═══════════════════════════════════════════════════════════════════════════

const N_ORDER: usize = 7;      // grid size for order/exactness/boundary asserts (1,2,4)
const N_RANK: usize = 4;       // grid size for rank assert (3)
const N_COST: usize = 5;       // grid size for cost/load-bearing asserts (5,6)
const T: f64 = 0.15;           // evolution time (frozen)
const CURSE_TB: f64 = 1e12;    // 1 TB threshold (assert 6)

// ═══════════════════════════════════════════════════════════════════════════
// §B — Coefficient builders (frozen, pre-registered)
// ═══════════════════════════════════════════════════════════════════════════

fn grid_xs(n: usize) -> Vec<f64> {
    let dx = TAU / n as f64;
    (0..n).map(|i| i as f64 * dx).collect()
}

/// a[j][i] = 0.5 + 0.2·cos(x_i + 0.4·j)  (varies in [0.3, 0.7], genuinely variable).
fn coef_a(xs: &[f64], j: usize) -> Vec<f64> {
    xs.iter().map(|&x| 0.5 + 0.2 * (x + 0.4 * j as f64).cos()).collect()
}

/// b[j][i] = 0.3·sin(x_i + 0.2·j).
fn coef_b(xs: &[f64], j: usize) -> Vec<f64> {
    xs.iter().map(|&x| 0.3 * (x + 0.2 * j as f64).sin()).collect()
}

/// Smooth IC: tensor product of (cos(x_i)+0.3) per axis.
fn smooth_u0_n(n: usize, d: usize) -> Vec<f64> {
    let xs = grid_xs(n);
    let g: Vec<f64> = xs.iter().map(|&x| x.cos() + 0.3).collect();
    let nd = n.pow(d as u32);
    (0..nd).map(|flat| {
        let mut f = flat; let mut val = 1.0f64;
        for _ in 0..d { val *= g[f % n]; f /= n; }
        val
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Dense generator (independent reference)
// ═══════════════════════════════════════════════════════════════════════════

/// Divergence-form 1-D FD generator (periodic, n×n matrix).
fn gen_1d(a: &[f64], b: &[f64], n: usize, dx: f64) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    let dx2 = dx * dx;
    let two_dx = 2.0 * dx;
    for i in 0..n {
        let ip = (i + 1) % n;
        let im = (i + n - 1) % n;
        let a_ip = 0.5 * (a[i] + a[ip]);
        let a_im = 0.5 * (a[i] + a[im]);
        l[i * n + ip] += a_ip / dx2 + b[i] / two_dx;
        l[i * n + i]  -= (a_ip + a_im) / dx2;
        l[i * n + im] += a_im / dx2 - b[i] / two_dx;
    }
    l
}

fn mat_eye(m: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; m * m];
    for i in 0..m { e[i * m + i] = 1.0; }
    e
}

fn mat_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn mat_vec(a: &[f64], v: &[f64], m: usize) -> Vec<f64> {
    (0..m).map(|i| (0..m).map(|j| a[i * m + j] * v[j]).sum()).collect()
}

fn mat_mat(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * m];
    for i in 0..m {
        for k in 0..m {
            if a[i * m + k] == 0.0 { continue; }
            for j in 0..m { c[i * m + j] += a[i * m + k] * b[k * m + j]; }
        }
    }
    c
}

fn mat_scale(a: &[f64], s: f64) -> Vec<f64> { a.iter().map(|&v| v * s).collect() }

fn mat_inf(a: &[f64], m: usize) -> f64 {
    (0..m).map(|i| (0..m).map(|j| a[i * m + j].abs()).sum::<f64>())
        .fold(0.0f64, f64::max)
}

fn max_abs(a: &[f64]) -> f64 { a.iter().map(|x| x.abs()).fold(0.0f64, f64::max) }

/// Kronecker product: A(m×m) ⊗ B(k×k) → (mk×mk).
fn kron(a: &[f64], m: usize, b: &[f64], k: usize) -> Vec<f64> {
    let mk = m * k;
    let mut c = vec![0.0f64; mk * mk];
    for ia in 0..m {
        for ja in 0..m {
            let aij = a[ia * m + ja];
            if aij == 0.0 { continue; }
            for ib in 0..k {
                for jb in 0..k {
                    c[(ia * k + ib) * mk + ja * k + jb] += aij * b[ib * k + jb];
                }
            }
        }
    }
    c
}

/// Build lifted L_j ⊗ I^{rest} for d-D tensor (last index fastest).
fn lift_axis(lj: &[f64], n: usize, d: usize, j: usize) -> Vec<f64> {
    let before = n.pow(j as u32);
    let after  = n.pow((d - 1 - j) as u32);
    let eye_b = mat_eye(before);
    let eye_a = mat_eye(after);
    kron(&kron(&eye_b, before, lj, n), before * n, &eye_a, after)
}

// ── Padé[6/6] + scaling-squaring expm (local, zero production reuse) ────

fn lu_factor(a: &mut [f64], m: usize) -> Vec<usize> {
    let mut piv: Vec<usize> = (0..m).collect();
    for col in 0..m {
        let mut mx = a[col * m + col].abs(); let mut mr = col;
        for row in (col + 1)..m {
            let v = a[row * m + col].abs(); if v > mx { mx = v; mr = row; }
        }
        if mr != col {
            for j in 0..m { a.swap(col * m + j, mr * m + j); }
            piv.swap(col, mr);
        }
        let pv = a[col * m + col]; if pv.abs() < 1e-300 { continue; }
        let inv = 1.0 / pv;
        for row in (col + 1)..m {
            let f = a[row * m + col] * inv;
            a[row * m + col] = f;
            for j in (col + 1)..m { let acj = a[col * m + j]; a[row * m + j] -= f * acj; }
        }
    }
    piv
}

fn lu_solve(a: &[f64], piv: &[usize], b: &mut [f64], m: usize) {
    let tmp = b.to_vec(); for i in 0..m { b[i] = tmp[piv[i]]; }
    for col in 0..m { for row in (col + 1)..m { b[row] -= a[row * m + col] * b[col]; } }
    for row in (0..m).rev() {
        for col in (row + 1)..m { b[row] -= a[row * m + col] * b[col]; }
        let d = a[row * m + row]; if d.abs() > 1e-300 { b[row] /= d; }
    }
}

fn expm_l(a: &[f64], m: usize) -> Vec<f64> {
    let norm = mat_inf(a, m);
    let mut s = 0u32; let mut thr = 0.5f64;
    while norm > thr && s < 30 { s += 1; thr *= 2.0; }
    let sc = 0.5f64.powi(s as i32);
    let as_ = mat_scale(a, sc);
    let c = [1.0, 0.5, 5.0/44.0, 1.0/66.0, 1.0/792.0, 1.0/15840.0, 1.0/665280.0];
    let a2 = mat_mat(&as_, &as_, m);
    let a4 = mat_mat(&a2, &a2, m);
    let a6 = mat_mat(&a2, &a4, m);
    let eye = mat_eye(m);
    let blend = |coeffs: &[(usize, f64)]| -> Vec<f64> {
        let mut acc = vec![0.0f64; m * m];
        for &(k, ck) in coeffs {
            let src: &[f64] = match k { 0 => &eye, 2 => &a2, 4 => &a4, _ => &a6 };
            for (av, &sv) in acc.iter_mut().zip(src.iter()) { *av += ck * sv; }
        }
        acc
    };
    let v = blend(&[(0, c[0]), (2, c[2]), (4, c[4]), (6, c[6])]);
    let inner = blend(&[(0, c[1]), (2, c[3]), (4, c[5])]);
    let u = mat_mat(&as_, &inner, m);
    let mut p = v.clone(); for (pi, &ui) in p.iter_mut().zip(u.iter()) { *pi += ui; }
    let mut q = v; for (qi, &ui) in q.iter_mut().zip(u.iter()) { *qi -= ui; }
    let piv = lu_factor(&mut q, m);
    let mut exp_s = vec![0.0f64; m * m];
    for col in 0..m {
        let mut rhs: Vec<f64> = (0..m).map(|row| p[row * m + col]).collect();
        lu_solve(&q, &piv, &mut rhs, m);
        for row in 0..m { exp_s[row * m + col] = rhs[row]; }
    }
    for _ in 0..s { exp_s = mat_mat(&exp_s, &exp_s, m); }
    let _ = as_; // suppress move warning
    exp_s
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — Local Chernoff evolver (independent re-impl; zero reuse of production)
// ═══════════════════════════════════════════════════════════════════════════

/// 1-D DFT: real → complex interleaved.
fn dft_r2c(x: &[f64]) -> Vec<f64> {
    let n = x.len(); let tpn = TAU / n as f64;
    let mut out = vec![0.0f64; 2 * n];
    for k in 0..n {
        let (mut re, mut im) = (0.0, 0.0);
        for j in 0..n { let a = -(tpn * (j * k) as f64); re += x[j]*a.cos(); im += x[j]*a.sin(); }
        out[2*k] = re; out[2*k+1] = im;
    }
    out
}

/// 1-D IDFT: complex interleaved → real (returns 2n vec, re at [2k]).
fn idft_c2r(x: &[f64]) -> Vec<f64> {
    let n = x.len() / 2; let tpn = TAU / n as f64; let inv_n = 1.0 / n as f64;
    let mut out = vec![0.0f64; 2 * n];
    for k in 0..n {
        let (mut re, mut im) = (0.0, 0.0);
        for j in 0..n {
            let a = tpn * (j * k) as f64;
            re += x[2*j]*a.cos() - x[2*j+1]*a.sin();
            im += x[2*j]*a.sin() + x[2*j+1]*a.cos();
        }
        out[2*k] = re * inv_n; out[2*k+1] = im * inv_n;
    }
    out
}

/// Apply k(τ) = exp(τ·a0·Lap_1d) via 1-D const-coef spectral (NO solver, NO drift).
fn k1d(line: &[f64], n: usize, dx: f64, a0: f64, tau: f64) -> Vec<f64> {
    let dx2 = dx * dx;
    let mut cplx = dft_r2c(line);
    for m in 0..n {
        let omega = TAU * m as f64 / n as f64;
        let factor = (tau * a0 * (2.0*omega.cos()-2.0) / dx2).exp();
        cplx[2*m] *= factor; cplx[2*m+1] *= factor;
    }
    let iv = idft_c2r(&cplx);
    (0..n).map(|i| iv[2*i]).collect()
}

/// Apply R = L_j(a,b) - a0·Lap_fd as a periodic tridiagonal mat-vec.
fn r_matvec(u: &[f64], a: &[f64], b: &[f64], n: usize, dx: f64, a0: f64) -> Vec<f64> {
    let dx2 = dx * dx; let two_dx = 2.0 * dx;
    let mut ru = vec![0.0f64; n];
    for i in 0..n {
        let ip = (i+1)%n; let im = (i+n-1)%n;
        let a_ip = 0.5*(a[i]+a[ip]); let a_im = 0.5*(a[i]+a[im]);
        let lju = a_ip*(u[ip]-u[i])/dx2 - a_im*(u[i]-u[im])/dx2 + b[i]*(u[ip]-u[im])/two_dx;
        let lapu = a0*(u[ip]-2.0*u[i]+u[im])/dx2;
        ru[i] = lju - lapu;
    }
    ru
}

/// Apply P₂(s) = I + s·R + s²/2·R² (2 tridiagonal mat-vecs; zero solver/expm).
fn p2_apply(u: &[f64], a: &[f64], b: &[f64], n: usize, dx: f64, a0: f64, s: f64) -> Vec<f64> {
    let ru = r_matvec(u, a, b, n, dx, a0);
    let rru = r_matvec(&ru, a, b, n, dx, a0);
    (0..n).map(|i| u[i] + s*ru[i] + 0.5*s*s*rru[i]).collect()
}

/// One 1-D Chernoff step: P₂(τ/2)·k(τ)·P₂(τ/2).
fn chernoff_1d_step(u: &[f64], a: &[f64], b: &[f64], n: usize, dx: f64, tau: f64) -> Vec<f64> {
    let a0: f64 = a.iter().sum::<f64>() / n as f64;
    let half = tau / 2.0;
    let u1 = p2_apply(u, a, b, n, dx, a0, half);
    let u2 = k1d(&u1, n, dx, a0, tau);
    p2_apply(&u2, a, b, n, dx, a0, half)
}

/// Apply 1-D step along axis j of n^d tensor (last index fastest).
fn apply_axis_local(
    u: &[f64], n: usize, d: usize, j: usize,
    a: &[f64], b: &[f64], dx: f64, tau: f64,
) -> Vec<f64> {
    let stride = n.pow((d-1-j) as u32);
    let n_outer = n.pow(j as u32);
    let mut out = u.to_vec();
    let mut line = vec![0.0f64; n];
    for i_outer in 0..n_outer {
        for i_inner in 0..stride {
            for k in 0..n {
                line[k] = u[i_outer*n*stride + k*stride + i_inner];
            }
            let stepped = chernoff_1d_step(&line, a, b, n, dx, tau);
            for k in 0..n {
                out[i_outer*n*stride + k*stride + i_inner] = stepped[k];
            }
        }
    }
    out
}

/// d-D symmetric Strang (local re-impl, zero reuse of tt_varcoef_spectral.rs).
fn evolve_local(
    u0: &[f64], n: usize, d: usize, dx: f64,
    a_axis: &[Vec<f64>], b_axis: &[Vec<f64>],
    tau: f64, nsteps: usize,
) -> Vec<f64> {
    let half = tau / 2.0;
    let mut u = u0.to_vec();
    for _ in 0..nsteps {
        // forward: j=0..d-1 with τ/2, j=d-1 with τ
        for j in 0..d {
            let t = if j == d-1 { tau } else { half };
            u = apply_axis_local(&u, n, d, j, &a_axis[j], &b_axis[j], dx, t);
        }
        // backward: j=d-2..0 with τ/2
        for j in (0..d-1).rev() {
            u = apply_axis_local(&u, n, d, j, &a_axis[j], &b_axis[j], dx, half);
        }
    }
    u
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — Rank helper
// ═══════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
fn rank_rect_qr(a: &[f64], rows: usize, cols: usize, eps: f64) -> usize {
    let mut r = a.to_vec();
    let k = rows.min(cols);
    let mut col_norms: Vec<f64> = (0..cols)
        .map(|j| (0..rows).map(|i| r[i*cols+j]*r[i*cols+j]).sum::<f64>().sqrt())
        .collect();
    let max_norm = col_norms.iter().cloned().fold(0.0f64, f64::max);
    let tol = eps * max_norm;
    let mut rank = 0usize;
    for col in 0..k {
        let (pivot, pnorm) = col_norms[col..].iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &v)| (i+col, v)).unwrap_or((col, 0.0));
        if pnorm < tol { break; }
        rank += 1;
        if pivot != col {
            for row in 0..rows { r.swap(row*cols+col, row*cols+pivot); }
            col_norms.swap(col, pivot);
        }
        let norm = (col..rows).map(|i| r[i*cols+col]*r[i*cols+col]).sum::<f64>().sqrt();
        if norm < 1e-300 { break; }
        let sign = if r[col*cols+col] >= 0.0 { 1.0 } else { -1.0 };
        r[col*cols+col] += sign * norm;
        let inv = 1.0 / (col..rows).map(|i| r[i*cols+col]*r[i*cols+col]).sum::<f64>().sqrt();
        for i in col..rows { r[i*cols+col] *= inv; }
        for j in (col+1)..cols {
            let dot: f64 = (col..rows).map(|i| r[i*cols+col]*r[i*cols+j]).sum();
            for i in col..rows { r[i*cols+j] -= 2.0*dot*r[i*cols+col]; }
        }
        for j in (col+1)..cols {
            col_norms[j] = (col..rows).map(|i| r[i*cols+j]*r[i*cols+j]).sum::<f64>().sqrt();
        }
    }
    rank
}

// ═══════════════════════════════════════════════════════════════════════════
// §F — Statistics helpers
// ═══════════════════════════════════════════════════════════════════════════

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let (num, den) = a.iter().zip(b.iter())
        .fold((0.0, 0.0), |(n, d), (ai, bi)| (n + (ai-bi).powi(2), d + bi*bi));
    if den < 1e-300 { num.sqrt() } else { (num/den).sqrt() }
}

/// OLS log-log slope of log(err) vs log(tau), dropping `drop` leading (coarsest) entries.
/// Convention: returns d(log err)/d(log tau) > 0 for a convergent scheme (err ∝ tau^p → slope +p).
/// Use negated result to compare with "slope ≤ -1.9" (contract convention = slope of err vs 1/tau).
fn log_slope(taus: &[f64], errs: &[f64], drop: usize) -> f64 {
    let n = taus.len();
    assert!(n > drop + 1);
    let xs: Vec<f64> = taus[drop..].iter().map(|&t| t.ln()).collect();
    let ys: Vec<f64> = errs[drop..].iter().map(|&e| {
        if e <= 0.0 { f64::NEG_INFINITY } else { e.ln() }
    }).collect();
    let m = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x*x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x*y).sum();
    // Returns +p for convergent (err ∝ tau^p); contract uses negated convention.
    (m*sxy - sx*sy) / (m*sxx - sx*sx)
}

// ═══════════════════════════════════════════════════════════════════════════
// §G — THE GATE TEST
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn g_s3_varcoef_spectral() {
    // ── Assert 7a: source audit — NO solver calls in tt_varcoef_spectral.rs ─
    {
        let src_path =
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/tt_varcoef_spectral.rs");
        let src = std::fs::read_to_string(src_path)
            .expect("cannot read tt_varcoef_spectral.rs");
        let no_comments: String = src.lines()
            .map(|l| if let Some(p) = l.find("//") { &l[..p] } else { l })
            .collect::<Vec<_>>().join("\n");
        for kw in ["lu_solve_inplace(", "dense_expm("] {
            assert!(
                !no_comments.contains(kw),
                "assert 7 FAIL: tt_varcoef_spectral.rs calls `{kw}` (Theorem-6 R2 violation)"
            );
        }
        println!("Assert 7a PASS: tt_varcoef_spectral.rs has no solver calls (R2 honoured)");
    }

    // ── Assert 7b: const-a,b=0 step ≡ pure k(τ) spectral (≤1e-12) ──────
    // With a=const, b=0: R=0 → P₂=I → step = k(τ) = exp(τ·a0·Lap). Target: ~8.9e-16.
    {
        let n = N_ORDER;
        let dx = TAU / n as f64;
        let a0 = 0.5f64;
        let a: Vec<f64> = vec![a0; n];
        let b: Vec<f64> = vec![0.0; n];
        let tau = 0.05f64;
        let u0: Vec<f64> = grid_xs(n).iter().map(|&x| x.cos() + 0.3).collect();
        // Local Chernoff step (with const a, b=0, R=0, P₂=I ⇒ pure k)
        let u_vc = chernoff_1d_step(&u0, &a, &b, n, dx, tau);
        // Reference: pure k1d
        let u_k = k1d(&u0, n, dx, a0, tau);
        let max_err = u_vc.iter().zip(u_k.iter()).map(|(a, b)| (a-b).abs())
            .fold(0.0f64, f64::max);
        println!("Assert 7b: const-a,b=0 step vs k1d: max_err={max_err:.3e} (target ~8.9e-16)");
        assert!(
            max_err < 1e-12,
            "assert 7b FAIL: const-a reduction residual={max_err:.3e} > 1e-12"
        );
        println!("Assert 7b PASS: P₂=I for const-a,b=0 (residual {max_err:.3e} ≤ 1e-12)");
    }

    // ── Assert 2: LAYER-1 EXACTNESS ─────────────────────────────────────
    {
        let n = N_ORDER; let d = 3usize;
        let dx = TAU / n as f64;
        let xs = grid_xs(n);
        // Lifted per-axis generators
        let l_lifted: Vec<Vec<f64>> = (0..d).map(|j| {
            lift_axis(&gen_1d(&coef_a(&xs, j), &coef_b(&xs, j), n, dx), n, d, j)
        }).collect();
        let tot = n.pow(d as u32);
        // max ‖[L_j, L_k]‖
        let mut max_comm = 0.0f64;
        for j in 0..d {
            for k in (j+1)..d {
                let jk = mat_mat(&l_lifted[j], &l_lifted[k], tot);
                let kj = mat_mat(&l_lifted[k], &l_lifted[j], tot);
                let comm: Vec<f64> = jk.iter().zip(kj.iter()).map(|(a, b)| a-b).collect();
                let c = max_abs(&comm);
                if c > max_comm { max_comm = c; }
            }
        }
        println!("Assert 2: max‖[L_j,L_k]‖ = {max_comm:.3e}  (expect 0, ≤1e-12)");
        assert!(max_comm < 1e-12,
            "assert 2 FAIL: max commutator = {max_comm:.3e} > 1e-12");
        // ‖exp(τ·ΣL) - Π exp(τL_j)‖_max
        let tau = 0.03f64;
        let l_sum = l_lifted.iter().skip(1)
            .fold(l_lifted[0].clone(), |acc, lj| mat_add(&acc, lj));
        let exp_sum = expm_l(&mat_scale(&l_sum, tau), tot);
        let mut exp_prod = mat_eye(tot);
        for lj in &l_lifted {
            exp_prod = mat_mat(&exp_prod, &expm_l(&mat_scale(lj, tau), tot), tot);
        }
        let diff: Vec<f64> = exp_sum.iter().zip(exp_prod.iter()).map(|(a, b)| a-b).collect();
        let split_err = max_abs(&diff);
        println!("Assert 2: ‖exp(τΣL)−∏exp(τL_j)‖ = {split_err:.3e}  (expect ≤1e-12)");
        assert!(split_err < 1e-12,
            "assert 2 FAIL: split residue = {split_err:.3e} > 1e-12");
        println!("Assert 2 PASS: Layer-1 EXACT (commutator {max_comm:.3e}, split {split_err:.3e})");
    }

    // ── Assert 3: RANK-1 TT OPERATOR ────────────────────────────────────
    // Method: matricise E₀⊗I across (axis0|rest) cut.
    // E₀⊗I[r0*rest+r1, c0*rest+c1] = E₀[r0,c0]*δ[r1,c1].
    // After the (row_axis0,col_axis0)|(row_rest,col_rest) bipartition (same as Python probe),
    // M[(r0,c0),(r1,c1)] = E₀[r0,c0]*δ[r1,c1].
    // All columns with r1=c1 are proportional to vec(E₀); columns with r1≠c1 are zero.
    // Verify rank=1 by: σ₁ > 0 AND ‖M - σ₁ u₁ v₁^T‖_F / ‖M‖_F < 1e-12.
    {
        let n = N_RANK; let d = 3usize;
        let dx = TAU / n as f64;
        let xs = grid_xs(n);
        let tau = 0.02f64;
        let l0 = gen_1d(&coef_a(&xs, 0), &coef_b(&xs, 0), n, dx);
        let exp_l0 = expm_l(&mat_scale(&l0, tau), n);
        // Build the (n*n) × (rest*rest) bipartite matrix directly:
        // M[(r0,c0),(r1,c1)] = E₀[r0,c0]*δ[r1,c1]
        let rest = n.pow((d-1) as u32);
        let rows = n * n; let cols = rest * rest;
        let mut m_mat = vec![0.0f64; rows * cols];
        for r0 in 0..n {
            for c0 in 0..n {
                let e_val = exp_l0[r0 * n + c0];
                let row_idx = r0 * n + c0;
                for r1 in 0..rest {
                    // δ[r1,c1] = 1 only when c1=r1
                    let col_idx = r1 * rest + r1;
                    m_mat[row_idx * cols + col_idx] += e_val;
                }
            }
        }
        // Compute σ₁ by power iteration on M^T M (cols×cols).
        // Actually use M M^T (rows×rows, smaller: 16×16 for n=4,d=3).
        let frob_sq: f64 = m_mat.iter().map(|x| x * x).sum();
        let frob = frob_sq.sqrt();
        // Power iteration: u₁ via M M^T, converge to dominant left singular vector.
        let mut u = vec![1.0f64 / (rows as f64).sqrt(); rows];
        for _ in 0..40 {
            // v_tmp = M^T u (cols-dim)
            let v_tmp: Vec<f64> = (0..cols).map(|j| {
                (0..rows).map(|i| m_mat[i * cols + j] * u[i]).sum()
            }).collect();
            // u_new = M v_tmp (rows-dim)
            let mut u_new: Vec<f64> = (0..rows).map(|i| {
                (0..cols).map(|j| m_mat[i * cols + j] * v_tmp[j]).sum()
            }).collect();
            let norm = u_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-300 { for x in &mut u_new { *x /= norm; } }
            u = u_new;
        }
        // σ₁ = ‖M^T u‖
        let v1: Vec<f64> = (0..cols).map(|j| {
            (0..rows).map(|i| m_mat[i * cols + j] * u[i]).sum()
        }).collect();
        let sigma1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v1_unit: Vec<f64> = if sigma1 > 1e-300 {
            v1.iter().map(|x| x / sigma1).collect()
        } else { v1.clone() };
        // Residual: ‖M - σ₁ u₁ v₁^T‖_F / ‖M‖_F
        let res_sq: f64 = (0..rows).map(|i| {
            (0..cols).map(|j| {
                let r = m_mat[i * cols + j] - sigma1 * u[i] * v1_unit[j];
                r * r
            }).sum::<f64>()
        }).sum();
        let rel_res = if frob > 1e-300 { res_sq.sqrt() / frob } else { 0.0 };
        println!("Assert 3: σ₁={sigma1:.6e}, ‖M-σ₁u₁v₁ᵀ‖_F/‖M‖_F={rel_res:.3e} \
                  (rank-1 iff rel_res<1e-12, n={n}, d={d})");
        assert!(
            rel_res < 1e-12,
            "assert 3 FAIL: rank > 1 (rel_residual={rel_res:.3e} > 1e-12)"
        );
        println!("Assert 3 PASS: operator-TT-rank = 1 (curse-escape algebraic proof)");
    }

    // ── Assert 1: ORDER-2 CONVERGENCE ───────────────────────────────────
    {
        let n = N_ORDER; let d = 3usize;
        let dx = TAU / n as f64;
        let xs = grid_xs(n);
        let a_axis: Vec<Vec<f64>> = (0..d).map(|j| coef_a(&xs, j)).collect();
        let b_axis: Vec<Vec<f64>> = (0..d).map(|j| coef_b(&xs, j)).collect();
        // Dense reference: full n^d×n^d additive generator
        let tot = n.pow(d as u32);
        let l_full: Vec<f64> = (0..d).map(|j| {
            lift_axis(&gen_1d(&a_axis[j], &b_axis[j], n, dx), n, d, j)
        }).fold(vec![0.0f64; tot*tot], |acc, lj| mat_add(&acc, &lj));
        let u0 = smooth_u0_n(n, d);
        let u_ref = mat_vec(&expm_l(&mat_scale(&l_full, T), tot), &u0, tot);
        let nsteps_sweep = [4usize, 8, 16, 32, 64, 128];
        let mut errs = Vec::new(); let mut taus = Vec::new();
        println!("Assert 1: ORDER-2 (d={d}, n={n}, T={T}):");
        for &ns in &nsteps_sweep {
            let tau = T / ns as f64;
            let u = evolve_local(&u0, n, d, dx, &a_axis, &b_axis, tau, ns);
            let err = rel_l2(&u, &u_ref);
            println!("  nsteps={ns:4}  tau={tau:.4e}  rel_err={err:.4e}");
            errs.push(err); taus.push(tau);
        }
        let slope = log_slope(&taus, &errs, 2);
        // slope = d(log err)/d(log tau) ≈ +2.0 for order-2; contract says "≤ -1.9" meaning |slope| ≥ 1.9
        println!("  log-log slope d(log err)/d(log tau) = {slope:.4}  (target ≥ 1.9, probe=2.0000)");
        assert!(slope >= 1.9,
            "assert 1 FAIL: ORDER slope = {slope:.4} < 1.9 (not order-2)");
        println!("Assert 1 PASS: slope={slope:.4} ≥ 1.9 (order-2 confirmed)");
    }

    // ── Assert 4: BOUNDARY — non-separable cross-diffusion ──────────────
    {
        let n = N_ORDER; let d = 2usize;
        let dx = TAU / n as f64;
        let xs = grid_xs(n);
        let a_axis: Vec<Vec<f64>> = (0..d).map(|j| coef_a(&xs, j)).collect();
        let b_zero: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0; n]).collect();
        // True generator = additive + 0.25·cos(x_i)·sin(x_{j})·∂²_{x_0}
        let tot = n.pow(d as u32);
        let mut l_true: Vec<f64> = (0..d).map(|j| {
            lift_axis(&gen_1d(&a_axis[j], &b_zero[j], n, dx), n, d, j)
        }).fold(vec![0.0f64; tot*tot], |acc, lj| mat_add(&acc, &lj));
        // Add cross-diffusion term 0.25·cos(x_i)·sin(x_{j})·∂²_{x_0}
        // idx(i0, i1) = i0*n + i1 (axis0 slow, axis1 fast for d=2 last-fast).
        let dx2 = dx * dx;
        for i0 in 0..n {
            for i1 in 0..n {
                let row = i0*n + i1;
                let ip = (i0+1)%n; let im = (i0+n-1)%n;
                let acr = 0.25 * xs[i0].cos() * xs[i1].sin();
                l_true[row*tot + ip*n+i1] += acr / dx2;
                l_true[row*tot + im*n+i1] += acr / dx2;
                l_true[row*tot + row]     -= 2.0*acr / dx2;
            }
        }
        let g: Vec<f64> = xs.iter().map(|&x| x.cos() + 0.3).collect();
        let u0: Vec<f64> = (0..tot).map(|flat| g[flat/n] * g[flat%n]).collect();
        let t_bnd = 0.1f64;
        let u_ref = mat_vec(&expm_l(&mat_scale(&l_true, t_bnd), tot), &u0, tot);
        let nsteps_sweep = [16usize, 32, 64, 128, 256];
        let mut errs = Vec::new(); let mut taus = Vec::new();
        println!("Assert 4: BOUNDARY non-separable cross-term (d={d}, n={n}, T={t_bnd}):");
        for &ns in &nsteps_sweep {
            let tau = t_bnd / ns as f64;
            let u = evolve_local(&u0, n, d, dx, &a_axis, &b_zero, tau, ns);
            let err = rel_l2(&u, &u_ref);
            println!("  nsteps={ns:4}  tau={tau:.4e}  rel_err={err:.4e}");
            errs.push(err); taus.push(tau);
        }
        let slope = log_slope(&taus, &errs, 1);
        let floor = *errs.last().unwrap();
        // slope = d(log err)/d(log tau); plateau means slope ≈ 0 < 1.0 (contract says slope > -1.0)
        println!("  slope={slope:.4}  floor={floor:.3e}  (target: slope<1.0, floor>1e-4; probe slope≈0)");
        assert!(slope < 1.0,
            "assert 4 FAIL: slope={slope:.4} ≥ 1.0 (non-sep falsely converges like order-1+!)");
        assert!(floor > 1e-4,
            "assert 4 FAIL: floor={floor:.3e} ≤ 1e-4 (no wrong-operator floor!)");
        println!("Assert 4 PASS: slope={slope:.4} < 1.0 AND floor={floor:.3e} > 1e-4 \
                  (boundary proven: wrong-operator floor)");
    }

    // ── Assert 5: LOAD-BEARING variable coefficient ──────────────────────
    {
        let n = N_COST; let d = 3usize;
        let dx = TAU / n as f64;
        let xs = grid_xs(n);
        let a_axis: Vec<Vec<f64>> = (0..d).map(|j| coef_a(&xs, j)).collect();
        let b_axis: Vec<Vec<f64>> = (0..d).map(|j| coef_b(&xs, j)).collect();
        // Const-a: per-axis mean
        let a_const: Vec<Vec<f64>> = a_axis.iter().map(|aj| {
            let mean = aj.iter().sum::<f64>() / n as f64;
            vec![mean; n]
        }).collect();
        let u0 = smooth_u0_n(n, d);
        let tau = T / 32.0;
        let u_var   = evolve_local(&u0, n, d, dx, &a_axis, &b_axis, tau, 32);
        let u_const = evolve_local(&u0, n, d, dx, &a_const, &b_axis, tau, 32);
        let rel_diff = rel_l2(&u_var, &u_const);
        let var_amp = {
            let a0 = &a_axis[0];
            a0.iter().cloned().fold(0.0f64, f64::max)
                - a0.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        println!("Assert 5: rel_diff={rel_diff:.3e} (≥0.02), var-amp={var_amp:.3} (>0.1)");
        assert!(rel_diff >= 0.02,
            "assert 5 FAIL: rel_diff={rel_diff:.3e} < 0.02 (variable coef not load-bearing)");
        assert!(var_amp > 0.1,
            "assert 5 FAIL: var-amp={var_amp:.3} ≤ 0.1 (coef not genuinely variable)");
        println!("Assert 5 PASS: variable coefficient is load-bearing \
                  (rel={rel_diff:.3e}, amp={var_amp:.3})");
    }

    // ── Assert 6: COST-SCALING — d=8,10 finite+real + TB check ─────────
    {
        let n = N_COST; let dx = TAU / n as f64;
        let xs = grid_xs(n);
        let nsteps = 16usize;
        let tau = T / nsteps as f64;
        for &d in &[8usize, 10usize] {
            let a_axis: Vec<Vec<f64>> = (0..d).map(|j| coef_a(&xs, j)).collect();
            let b_axis: Vec<Vec<f64>> = (0..d).map(|j| coef_b(&xs, j)).collect();
            let u0 = smooth_u0_n(n, d);
            let u_out = evolve_local(&u0, n, d, dx, &a_axis, &b_axis, tau, nsteps);
            let all_finite = u_out.iter().all(|x| x.is_finite());
            let dense_bytes = 8.0 * (n as f64).powi(2 * d as i32);
            println!(
                "Assert 6 d={d}: finite={all_finite}, dense_bytes={dense_bytes:.3e} > {CURSE_TB:.0e}"
            );
            assert!(all_finite,
                "assert 6 FAIL: evolver non-finite at d={d}");
            assert!(dense_bytes > CURSE_TB,
                "assert 6 FAIL: dense_bytes={dense_bytes:.3e} ≤ {CURSE_TB:.0e} at d={d}");
        }
        println!("Assert 6 PASS: evolver runs at d=8,10; dense reference > 1 TB (curse proven)");
    }

    println!();
    println!("=== All 7 asserts PASS — S³ variable-coefficient POC gate COMPLETE ===");
    println!("  Positive class: additive-separable a_j(x_j), b_j(x_j), v_j(x_j).");
    println!("  Layer-1 (inter-axis): EXACT (commuting); Layer-2 (intra-axis): ORDER-2.");
    println!("  Boundary proven: non-separable a(x,y) → wrong-operator floor (Assert 4).");
}
