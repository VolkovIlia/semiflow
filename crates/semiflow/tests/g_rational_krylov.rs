//! Rational (shift-and-invert) Krylov gates (`RELEASE_BLOCKING`, ADR-0206, math §66).
//!
//! * `G_RATKRYLOV_NEUMANN_ORACLE`: the Neumann path-graph Laplacian against its
//!   closed-form cosine eigenbasis (the oracle of `G_CHEB_STIFF_ORACLE`), `n = 400`
//!   at `λ_max t ∈ {1e2, 1e4, 1e6, 1e8, 1e10}` and `n = 4000` at `1e6, 1e8` (slow
//!   modes alive): `‖e^{−tL}v − exact‖_∞ ≤ (tol + 1e−11)·‖v‖_∞`, `tol = 1e−10`, the
//!   floor of `G_CHEB_STIFF_ORACLE` — at every stiffness: the cancellation-free
//!   tridiagonal factor resolves the slow eigenvalues to relative accuracy (the
//!   textbook `LDLᵀ` loses `≈ log₁₀(λ_max t)` digits: `4.9e−9` at `1e10`). The slow
//!   modes must be alive at ≥ 100× the allowance where marked. Plus the adversarial
//!   start vector `cos(0.97πi) + 10⁻⁶` at `λ_max t ∈ {1e4, 1e6}`: every early Ritz
//!   value is stiff, so a stop on small a-posteriori differences returns after 4
//!   solves with the `10⁻⁶` constant lost; the a-priori dimension must keep it.
//! * `G_RATKRYLOV_SOLVE_COUNT`: at every point above, at most 30 resolvent solves
//!   (deterministic: the a-priori dimension is 27 at `tol = 1e−10`, `γ = t/10`); at
//!   `λ_max t ≥ 1e6` the Chebyshev degree for the same `tol` is at least 100× the
//!   solve count.
//! * `G_RATKRYLOV_MASS_DENSE`: a lumped mass `M ≠ I` on the tridiagonal path and a
//!   non-tridiagonal 4×3 grid operator on the PCG path (`n = 12`) against the dense
//!   congruence reference `M^{−½}e^{−tÂ}M^{½}v`: `≤ 1e−10·‖v‖_∞`.
//! * `G_RATKRYLOV_ISSUE16`: the singular Neumann Laplacian ×1e7 of
//!   `G_SYMOP_IMPLICIT_STIFF` (`n = 400`, `t = 0.7`, so `1/γ` is not a power of
//!   two) with an `O(1)` constant component: every non-constant mode underflows,
//!   the exact result is `mean(v)·1`, to `1e−12·‖v‖_∞`.

#![allow(clippy::cast_precision_loss)] // indices < 2^20

use std::f64::consts::PI;

use semiflow::{
    dense_csr_expmv_ref, graph_expmv_matvec_count, rational_krylov_expmv, KrylovPath,
    RationalOptions, ScratchPool, SpdSolver, SymmetricOperator,
};

const TOL: f64 = 1e-10;
/// Rounding allowance on top of `tol`, the same floor as `G_CHEB_STIFF_ORACLE`.
const ERR_FLOOR: f64 = 1e-11;
/// `tol + 1e−11` (module docs).
const ALLOWANCE: f64 = TOL + ERR_FLOOR;

/// Neumann path Laplacian `n×n` scaled by `s`: CSR triple.
fn path_laplacian(n: usize, s: f64) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let (mut rp, mut ci, mut va) = (vec![0], Vec::new(), Vec::new());
    let idx = |i: usize| u32::try_from(i).expect("n < 2^32");
    for i in 0..n {
        let deg = f64::from(u8::from(i > 0) + u8::from(i + 1 < n));
        if i > 0 {
            ci.push(idx(i - 1));
            va.push(-s);
        }
        ci.push(idx(i));
        va.push(deg * s);
        if i + 1 < n {
            ci.push(idx(i + 1));
            va.push(-s);
        }
        rp.push(ci.len());
    }
    (rp, ci, va)
}

/// `e^{−tL}v` by the cosine eigenbasis (exact phase reduction mod `4n`, compensated sums).
fn exact(v: &[f64], t: f64) -> Vec<f64> {
    let n = v.len();
    let nf = n as f64;
    let table: Vec<f64> = (0..4 * n)
        .map(|k| (k as f64 * PI / (2.0 * nf)).cos())
        .collect();
    let phi = |i: usize, j: usize| table[((2 * i + 1) * j) % (4 * n)];
    let mut out = vec![Neumaier::default(); n];
    for j in 0..n {
        let decay = (-t * (2.0 - 2.0 * table[(2 * j) % (4 * n)])).exp();
        if decay == 0.0 {
            continue;
        }
        let norm = if j == 0 { nf } else { nf / 2.0 };
        let mut dot = Neumaier::default();
        for (i, &x) in v.iter().enumerate() {
            dot.add(x * phi(i, j));
        }
        let coef = dot.value() * decay / norm;
        for (i, o) in out.iter_mut().enumerate() {
            o.add(coef * phi(i, j));
        }
    }
    out.iter().map(Neumaier::value).collect()
}

#[derive(Clone, Copy, Default)]
struct Neumaier {
    sum: f64,
    comp: f64,
}

impl Neumaier {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        self.comp += if self.sum.abs() >= x.abs() {
            (self.sum - t) + x
        } else {
            (x - t) + self.sum
        };
        self.sum = t;
    }
    fn value(&self) -> f64 {
        self.sum + self.comp
    }
}

/// `max |x|`, NaN-propagating.
fn sup(x: impl Iterator<Item = f64>) -> f64 {
    x.fold(0.0_f64, |a, e| {
        if a.is_nan() || e.is_nan() {
            f64::NAN
        } else {
            a.max(e.abs())
        }
    })
}

fn test_vector(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 / n as f64;
            (7.0 * x).sin() + 0.3 * (61.0 * x).cos() + if i % 3 == 0 { 0.5 } else { -0.25 }
        })
        .collect()
}

/// Failures of the oracle and solve-count checks on the `n`-node carrier.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x <= y)`: NaN must fail
fn check_carrier(n: usize, points: &[f64], informative: bool) -> Vec<String> {
    let (rp, ci, va) = path_laplacian(n, 1.0);
    let op = SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).expect("path Laplacian");
    let rho = op.lambda_max_bound();
    let v = test_vector(n);
    let v_inf = sup(v.iter().copied());
    let mean = v.iter().sum::<f64>() / n as f64;
    let mut scratch = ScratchPool::new();
    let mut fails = Vec::new();
    for &lt in points {
        let t = lt / rho;
        let want = exact(&v, t);
        let mut got = vec![0.0; n];
        let opts = RationalOptions::default().with_tol(TOL);
        let rep = rational_krylov_expmv(&op, None, t, &v, &mut got, opts, &mut scratch)
            .expect("rational");
        let err = sup(got.iter().zip(&want).map(|(g, w)| g - w)) / v_inf;
        let dynamics = sup(want.iter().map(|w| w - mean)) / v_inf;
        let (_, cheb) = graph_expmv_matvec_count(1.0, lt, TOL, &KrylovPath::Chebyshev);
        eprintln!(
            "G_RATKRYLOV n={n} lt={lt:e} err/|v|={err:.3e} solves={} est={:.2e} cheb={cheb} |exact-mean|/|v|={dynamics:.3e}",
            rep.solves, rep.est_error
        );
        if !(err <= ALLOWANCE) {
            fails.push(format!("ORACLE n={n} lt={lt:e}: err {err:e} > {ALLOWANCE}"));
        }
        if rep.solves > 30 {
            fails.push(format!(
                "SOLVE_COUNT n={n} lt={lt:e}: {} solves > 30",
                rep.solves
            ));
        }
        if lt >= 1e6 && u64::from(cheb) < 100 * rep.solves as u64 {
            fails.push(format!(
                "SOLVE_COUNT n={n} lt={lt:e}: chebyshev {cheb} < 100 x {}",
                rep.solves
            ));
        }
        if informative && !(dynamics >= 1e2 * ALLOWANCE) {
            fails.push(format!(
                "n={n} lt={lt:e}: slow modes decayed ({dynamics:e}), vacuous"
            ));
        }
    }
    fails
}

#[test]
fn g_ratkrylov_neumann_oracle_and_solve_count() {
    let mut fails = check_carrier(400, &[1e2, 1e4, 1e6, 1e8, 1e10], false);
    fails.extend(check_carrier(4000, &[1e6, 1e8], true));
    assert!(fails.is_empty(), "{fails:#?}");
}

/// The adversarial start vector of the module docs: a high-frequency mode (stiff
/// Ritz values only) plus a `1e−6` constant that must survive.
#[test]
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x <= y)`: NaN must fail
fn g_ratkrylov_neumann_oracle_slow_component() {
    let n = 400;
    let (rp, ci, va) = path_laplacian(n, 1.0);
    let op = SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).expect("path Laplacian");
    let v: Vec<f64> = (0..n)
        .map(|i| (0.97 * PI * i as f64).cos() + 1e-6)
        .collect();
    let v_inf = sup(v.iter().copied());
    let mut scratch = ScratchPool::new();
    for lt in [1e4, 1e6] {
        let t = lt / op.lambda_max_bound();
        let want = exact(&v, t);
        let mut got = vec![0.0; n];
        let opts = RationalOptions::default().with_tol(TOL);
        let rep = rational_krylov_expmv(&op, None, t, &v, &mut got, opts, &mut scratch)
            .expect("rational");
        let err = sup(got.iter().zip(&want).map(|(g, w)| g - w)) / v_inf;
        let mean = v.iter().sum::<f64>() / n as f64;
        eprintln!(
            "G_RATKRYLOV slow component lt={lt:e}: err/|v|={err:.3e} solves={} mean={mean:.3e}",
            rep.solves
        );
        assert!(
            err <= ALLOWANCE && rep.solves <= 30,
            "lt={lt:e}: err {err:e}, {rep:?}"
        );
        assert!(
            mean >= 1e3 * ALLOWANCE,
            "the slow component must be informative"
        );
    }
}

/// Dense reference `M^{−½}·e^{−tÂ}·M^{½}v`, `Â = M^{−½}AM^{−½}` (`n ≤ 12`).
fn dense_mass_ref(op: &SymmetricOperator<f64>, mass: &[f64], t: f64, v: &[f64]) -> Vec<f64> {
    let a_hat = op.lumped_congruence(mass).expect("congruence");
    let w: Vec<f64> = v.iter().zip(mass).map(|(x, m)| x * m.sqrt()).collect();
    let mut out = vec![0.0; v.len()];
    dense_csr_expmv_ref(&a_hat, t, &w, &mut out).expect("dense ref");
    out.iter().zip(mass).map(|(y, m)| y / m.sqrt()).collect()
}

/// 4×3 grid 5-point Laplacian with Neumann ends (non-tridiagonal: PCG path).
fn grid_laplacian() -> SymmetricOperator<f64> {
    let (nx, ny) = (4, 3);
    let n = nx * ny;
    let mut dense = vec![0.0; n * n];
    let mut link = |a: usize, b: usize, w: f64| {
        dense[a * n + b] -= w;
        dense[b * n + a] -= w;
        dense[a * n + a] += w;
        dense[b * n + b] += w;
    };
    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            if i + 1 < nx {
                link(k, k + 1, 1.0 + 0.1 * i as f64);
            }
            if j + 1 < ny {
                link(k, k + nx, 0.7 + 0.2 * j as f64);
            }
        }
    }
    let (mut rp, mut ci, mut va) = (vec![0], Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..n {
            if dense[i * n + j] != 0.0 {
                ci.push(u32::try_from(j).expect("small"));
                va.push(dense[i * n + j]);
            }
        }
        rp.push(ci.len());
    }
    SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).expect("grid Laplacian")
}

#[test]
fn g_ratkrylov_mass_dense() {
    let n = 12;
    let mass: Vec<f64> = (0..n).map(|i| 0.5 + 0.13 * i as f64).collect();
    let v = test_vector(n);
    let v_inf = sup(v.iter().copied());
    let (rp, ci, va) = path_laplacian(n, 3.0);
    let tri = SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).expect("path");
    let grid = grid_laplacian();
    let mut scratch = ScratchPool::new();
    for (name, op) in [("tridiagonal", &tri), ("pcg", &grid)] {
        for t in [0.01, 0.3, 2.0, 40.0] {
            let want = dense_mass_ref(op, &mass, t, &v);
            let mut got = vec![0.0; n];
            let opts = RationalOptions::default()
                .with_tol(1e-12)
                .with_solver(SpdSolver::Auto);
            rational_krylov_expmv(op, Some(&mass), t, &v, &mut got, opts, &mut scratch)
                .expect("rational");
            let err = sup(got.iter().zip(&want).map(|(g, w)| g - w)) / v_inf;
            eprintln!("G_RATKRYLOV_MASS_DENSE {name} t={t}: err/|v|={err:.3e}");
            assert!(err <= 1e-10, "{name} t={t}: err {err:e}");
        }
    }
}

#[test]
fn g_ratkrylov_issue16() {
    let n = 400;
    let (rp, ci, va) = path_laplacian(n, 1e7);
    let op = SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).expect("scaled Neumann");
    let v: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
    let mean = v.iter().sum::<f64>() / n as f64;
    let mut got = vec![0.0; n];
    let mut scratch = ScratchPool::new();
    let rep = rational_krylov_expmv(
        &op,
        None,
        0.7,
        &v,
        &mut got,
        RationalOptions::default(),
        &mut scratch,
    )
    .expect("rational");
    let err = sup(got.iter().map(|g| g - mean)) / sup(v.iter().copied());
    eprintln!(
        "G_RATKRYLOV_ISSUE16: err/|v|={err:.3e} solves={}",
        rep.solves
    );
    assert!(err <= 1e-12, "err {err:e}");
}
