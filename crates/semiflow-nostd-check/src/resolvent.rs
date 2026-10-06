//! ADR-0202 scenarios: SPD resolvent (tridiagonal `LDLᵀ`, PCG with IC(0)) and `phi_combination`.
//!
//! The resolvent scenarios use manufactured solutions: pick `x★`, form
//! `b = (λM + A)x★` with a plain CSR matvec, solve, compare with `x★`. The
//! `phi_combination` scenario uses the eigen-exact DST-I oracle of
//! `tests/g_phi_combination_dense.rs` (`G_PHI_COMBINATION_DENSE`). Every
//! transcendental goes through `m` (libm), so the digests are build-independent.

use alloc::vec::Vec;

use semiflow::{
    generator_action::{CsrGenerator, GeneratorAction},
    phi_combination, Precond, ResolventMethod, ScratchPool, SpdSolver, SymmetricOperator,
};

use crate::{check, m, sup_diff, Digest, Failure, ScenarioResult};

/// `y = A x` for a CSR operator.
fn csr_matvec(op: &SymmetricOperator<f64>, x: &[f64]) -> Vec<f64> {
    let (row_ptr, cols, vals) = op.csr();
    (0..op.n())
        .map(|i| {
            (row_ptr[i]..row_ptr[i + 1])
                .map(|k| vals[k] * x[cols[k] as usize])
                .sum()
        })
        .collect()
}

/// `b = (λM + A) x`.
fn shifted_rhs(op: &SymmetricOperator<f64>, lam: f64, mass: &[f64], x: &[f64]) -> Vec<f64> {
    let mut b = csr_matvec(op, x);
    for ((bi, mi), xi) in b.iter_mut().zip(mass).zip(x) {
        *bi += lam * mi * xi;
    }
    b
}

/// Smooth manufactured solution with non-trivial bits.
fn manufactured(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = f64::from(u32::try_from(i).unwrap_or(0));
            m::sin(0.3 * t) + 0.5 * m::cos(0.11 * t) + 1.5
        })
        .collect()
}

/// Path-graph Neumann Laplacian (conductances in `[0.5, 1.5]`) plus reaction `c > 0`.
fn tridiag_operator(n: usize) -> Result<SymmetricOperator<f64>, Failure> {
    let ni = u32::try_from(n).unwrap_or(0);
    let cond = |i: u32| 1.0 + 0.5 * m::sin(f64::from(i) * 1.7);
    let (mut row_ptr, mut cols, mut vals) = (alloc::vec![0_usize], Vec::new(), Vec::new());
    for i in 0..ni {
        let left = if i > 0 { cond(i - 1) } else { 0.0 };
        let right = if i + 1 < ni { cond(i) } else { 0.0 };
        if i > 0 {
            cols.push(i - 1);
            vals.push(-left);
        }
        cols.push(i);
        vals.push(left + right);
        if i + 1 < ni {
            cols.push(i + 1);
            vals.push(-right);
        }
        row_ptr.push(cols.len());
    }
    let base = SymmetricOperator::from_csr(n, &row_ptr, &cols, &vals, 1e-10)?;
    let c: Vec<f64> = (0..n)
        .map(|i| 0.1 + 0.05 * f64::from(u32::try_from(i % 7).unwrap_or(0)))
        .collect();
    Ok(base.with_diagonal(&c)?)
}

/// `n = 64` tridiagonal `LDLᵀ` at `λ = 0` (reaction + mass) and `λ = 0.75`.
///
/// Gate analogue: `G_SPDR_TRIDIAG_DENSE`. Both solves are digested.
pub(crate) fn spdr_tridiag() -> ScenarioResult {
    let n = 64;
    let op = tridiag_operator(n)?;
    let mass: Vec<f64> = (0..n)
        .map(|i| 1.0 + f64::from(u32::try_from(i).unwrap_or(0)) / 32.0)
        .collect();
    let x_true = manufactured(n);
    let mut digest = Digest::new();
    let mut err = 0.0_f64;
    for lam in [0.0, 0.75] {
        let res = op.resolvent(lam, Some(&mass), SpdSolver::Auto, 1e-12)?;
        if res.method() != ResolventMethod::Tridiagonal {
            return Err(Failure::Invariant("spdr_tridiag: Auto must pick LDL^T"));
        }
        let b = shifted_rhs(&op, lam, &mass, &x_true);
        let mut x = alloc::vec![0.0; n];
        res.solve_into(&b, &mut x, &mut ScratchPool::new())?;
        err = err.max(sup_diff(&x, &x_true));
        digest.extend(&x);
    }
    check(err, 1e-9, digest.finish())
}

/// 2-D 8×8 five-point Neumann Laplacian plus reaction `c = 0.2`, row-major, `n = 64`.
fn grid_operator() -> Result<SymmetricOperator<f64>, Failure> {
    const SIDE: u32 = 8;
    let (mut row_ptr, mut cols, mut vals) = (alloc::vec![0_usize], Vec::new(), Vec::new());
    for r in 0..SIDE {
        for c in 0..SIDE {
            let i = r * SIDE + c;
            let mut row: Vec<(u32, f64)> = Vec::new();
            if r > 0 {
                row.push((i - SIDE, -1.0));
            }
            if c > 0 {
                row.push((i - 1, -1.0));
            }
            if c + 1 < SIDE {
                row.push((i + 1, -1.0));
            }
            if r + 1 < SIDE {
                row.push((i + SIDE, -1.0));
            }
            // Neumann diagonal = number of neighbours.
            let degree = f64::from(u32::try_from(row.len()).unwrap_or(0));
            row.push((i, degree));
            row.sort_by_key(|&(j, _)| j);
            for (j, v) in row {
                cols.push(j);
                vals.push(v);
            }
            row_ptr.push(cols.len());
        }
    }
    let base = SymmetricOperator::from_csr(64, &row_ptr, &cols, &vals, 1e-10)?;
    Ok(base.with_diagonal(&[0.2; 64])?)
}

/// PCG + IC(0) on the 8×8 grid at `λ = 0`, `tol = 1e-12`.
///
/// Gate analogue: `G_SPDR_PCG_DENSE`. The iteration count is digested too.
pub(crate) fn spdr_pcg_ic0() -> ScenarioResult {
    let op = grid_operator()?;
    let solver = SpdSolver::Pcg {
        precond: Precond::Ic0,
        max_iter: None,
    };
    let res = op.resolvent(0.0, None, solver, 1e-12)?;
    if res.method() != ResolventMethod::PcgIc0 {
        return Err(Failure::Invariant(
            "spdr_pcg_ic0: IC(0) fell back to Jacobi",
        ));
    }
    let x_true = manufactured(64);
    let b = shifted_rhs(&op, 0.0, &[1.0; 64], &x_true);
    let mut x = alloc::vec![0.0; 64];
    let report = res.solve_into(&b, &mut x, &mut ScratchPool::new())?;
    let mut digest = Digest::new();
    digest.extend(&x);
    digest.push(f64::from(
        u32::try_from(report.iterations).unwrap_or(u32::MAX),
    ));
    check(sup_diff(&x, &x_true), 1e-8, digest.finish())
}

const PHI_N: usize = 6;
const PHI_TAU: f64 = 0.5;

/// `G = tridiag(1, −2, 1)` on 6 nodes (Dirichlet ends).
struct TriDiagGen;

impl GeneratorAction<f64> for TriDiagGen {
    fn dim(&self) -> usize {
        PHI_N
    }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        for i in 0..PHI_N {
            let left = if i > 0 { src[i - 1] } else { 0.0 };
            let right = if i + 1 < PHI_N { src[i + 1] } else { 0.0 };
            dst[i] = left - 2.0 * src[i] + right;
        }
    }
    fn norm_bound(&self) -> f64 {
        4.0
    }
}

/// `φ_k(z)` by its convergent Taylor series (`z ≤ 0`, moderate size).
fn scalar_phi_k(k: usize, z: f64) -> f64 {
    let fact: f64 = (1..=k)
        .map(|i| f64::from(u32::try_from(i).unwrap_or(1)))
        .product();
    let mut term = 1.0 / fact;
    let mut sum = term;
    for n in 1_u32..120 {
        term *= z / f64::from(n + u32::try_from(k).unwrap_or(0));
        sum += term;
        if m::abs(term) < 1e-18 * (m::abs(sum) + 1e-300) {
            break;
        }
    }
    sum
}

/// Eigen-exact `φ_k(τG)·v` via the DST-I spectrum of `tridiag(1, −2, 1)`.
fn eigen_phi_k(k: usize, v: &[f64]) -> Vec<f64> {
    let np1 = f64::from(u32::try_from(PHI_N + 1).unwrap_or(1));
    let norm = m::sqrt(2.0 / np1);
    let node = |i: usize| f64::from(u32::try_from(i).unwrap_or(0));
    let mut out = alloc::vec![0.0; PHI_N];
    for j in 1..=PHI_N {
        let ang = node(j) * core::f64::consts::PI / np1;
        let phi = scalar_phi_k(k, PHI_TAU * (-2.0 + 2.0 * m::cos(ang)));
        let inner: f64 = (1..=PHI_N)
            .map(|i| norm * m::sin(node(i) * ang) * v[i - 1])
            .sum();
        for i in 1..=PHI_N {
            out[i - 1] += phi * inner * norm * m::sin(node(i) * ang);
        }
    }
    out
}

/// Deterministic weights in `[-1, 1)` (LCG), one vector per `φ_k` slot.
fn weight(seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..PHI_N)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = f64::from(u32::try_from(s >> 40).unwrap_or(0)) / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

/// `phi_combination` (p = 3, τ = 0.5) vs the DST-I eigen-exact `Σ τ^k φ_k(τG) w_k`.
///
/// Gate analogue: `G_PHI_COMBINATION_DENSE` (≤ 1e-12). The same weights go through
/// a `CsrGenerator` on the equivalent CSR operator; both outputs are digested.
pub(crate) fn phi_combination_p3() -> ScenarioResult {
    let w: Vec<Vec<f64>> = (0..4_u64).map(|k| weight(100 + k)).collect();
    let refs: Vec<&[f64]> = w.iter().map(Vec::as_slice).collect();
    let mut got = alloc::vec![0.0; PHI_N];
    phi_combination(
        &TriDiagGen,
        PHI_TAU,
        &refs,
        &mut got,
        &mut ScratchPool::new(),
    )?;
    let mut want = alloc::vec![0.0; PHI_N];
    let mut tk = 1.0;
    for (k, wk) in w.iter().enumerate() {
        for (acc, p) in want.iter_mut().zip(eigen_phi_k(k, wk)) {
            *acc += tk * p;
        }
        tk *= PHI_TAU;
    }
    let csr = csr_generator_output(&refs)?;
    let mut digest = Digest::new();
    digest.extend(&got);
    digest.extend(&csr);
    check(
        sup_diff(&got, &want).max(sup_diff(&csr, &want)),
        1e-12,
        digest.finish(),
    )
}

/// The same combination through `CsrGenerator::from_symmetric` (`A = −tridiag(1, −2, 1)`).
fn csr_generator_output(w: &[&[f64]]) -> Result<Vec<f64>, Failure> {
    let (mut row_ptr, mut cols, mut vals) = (alloc::vec![0_usize], Vec::new(), Vec::new());
    for i in 0..u32::try_from(PHI_N).unwrap_or(0) {
        if i > 0 {
            cols.push(i - 1);
            vals.push(-1.0);
        }
        cols.push(i);
        vals.push(2.0);
        if usize::try_from(i + 1).unwrap_or(0) < PHI_N {
            cols.push(i + 1);
            vals.push(-1.0);
        }
        row_ptr.push(cols.len());
    }
    let op = SymmetricOperator::from_csr(PHI_N, &row_ptr, &cols, &vals, 1e-10)?;
    let gen = CsrGenerator::from_symmetric(&op, None)?;
    let mut out = alloc::vec![0.0; PHI_N];
    phi_combination(&gen, PHI_TAU, w, &mut out, &mut ScratchPool::new())?;
    Ok(out)
}
