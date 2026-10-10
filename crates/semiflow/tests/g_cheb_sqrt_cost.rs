//! Chebyshev single-expansion gates (`RELEASE_BLOCKING`, ADR-0205, math §63.6.d).
//!
//! * `G_CHEB_SQRT_COST`: one action `e^{−τL}v` with `z = τλ_max/2` is ONE
//!   expansion whose degree obeys the Bennett bound of the Skellam tail,
//!   `m(z) ≤ ⌈L/3 + √(L²/9 + 2zL)⌉ − 1`, `L = ln(8/tol)`, for `z ∈ [1e−3, 1e7]`.
//!   Deterministic counter (`graph_expmv_matvec_count`); the substep kernel it
//!   replaced cost `⌈z/200⌉·101` mat-vecs, linear in `z`.
//! * `G_CHEB_STIFF_ORACLE`: the Neumann path-graph Laplacian (the issue #16
//!   operator, `n = 400`) against its closed-form cosine eigenbasis at
//!   `λ_max t ∈ {1e2, 1e4, 1e6, 1e8}`, and at `n = 4000` where the slow modes
//!   are still alive at `λ_max t = 1e6, 1e8`:
//!   `‖e^{−tL}v − exact‖_∞ ≤ (tol + 1e−11)·‖v‖_∞`. The substep kernel it
//!   replaced lost `6.8e−6` at `λ_max t = 4e7`, `tol = 1e−10` (ADR-0205).

#![allow(clippy::cast_precision_loss)] // indices < 2^20

use std::f64::consts::PI;

use semiflow::{
    graph_expmv_krylov, graph_expmv_matvec_count, KrylovPath, ScratchPool, SymmetricOperator,
};

/// `⌈L/3 + √(L²/9 + 2zL)⌉ − 1`, `L = ln(8/tol)` (math §63.6.d).
fn bennett_degree(z: f64, tol: f64) -> f64 {
    let l = (8.0 / tol).ln();
    (l / 3.0 + (l * l / 9.0 + 2.0 * z * l).sqrt()).ceil() - 1.0
}

#[test]
fn g_cheb_sqrt_cost() {
    let path = KrylovPath::Chebyshev;
    let mut worst = 0.0_f64;
    for tol in [1e-6, 1e-10, 1e-12, 1e-14] {
        for e in -30..=70 {
            let z = 10f64.powf(f64::from(e) / 10.0);
            let (s, m) = graph_expmv_matvec_count(1.0, 2.0 * z, tol, &path);
            let bound = bennett_degree(z, tol).max(3.0);
            assert_eq!(
                s, 1,
                "z={z:e} tol={tol:e}: one expansion expected, got {s} substeps"
            );
            assert!(
                f64::from(m) <= bound,
                "z={z:e} tol={tol:e}: degree {m} > Bennett bound {bound}"
            );
            worst = worst.max(f64::from(m) / bound);
        }
    }
    eprintln!("G_CHEB_SQRT_COST max m/bound = {worst:.3}");
    // Non-vacuity: at z = 1e6 the substep kernel's ⌈z/200⌉·101 is ≥ 50× the new cost.
    let (_, m) = graph_expmv_matvec_count(1.0, 2e6, 1e-12, &path);
    assert!(50 * u64::from(m) <= 5000 * 101, "degree {m} at z = 1e6");
}

/// Neumann path Laplacian `n×n` (unit weights): CSR triple.
fn path_laplacian(n: usize) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let (mut rp, mut ci, mut va) = (vec![0], Vec::new(), Vec::new());
    for i in 0..n {
        let deg = f64::from(u8::from(i > 0) + u8::from(i + 1 < n));
        if i > 0 {
            ci.push(u32::try_from(i - 1).expect("n < 2^32"));
            va.push(-1.0);
        }
        ci.push(u32::try_from(i).expect("n < 2^32"));
        va.push(deg);
        if i + 1 < n {
            ci.push(u32::try_from(i + 1).expect("n < 2^32"));
            va.push(-1.0);
        }
        rp.push(ci.len());
    }
    (rp, ci, va)
}

/// `e^{−tL}v` by the cosine eigenbasis: `λ_j = 2 − 2cos(jπ/n)`,
/// `φ_j(i) = cos((2i + 1)jπ/(2n))`, `‖φ_0‖² = n`, `‖φ_j‖² = n/2`. The phase is
/// reduced exactly modulo `4n` into a table of `4n` cosines.
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

/// Compensated (Neumaier) running sum: the oracle's sums carry `O(u)` error,
/// not `O(n·u)`.
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

/// Worst `‖got − exact‖_∞/‖v‖_∞` minus the allowance `tol + ERR_FLOOR` (≤ 0 passes).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x <= y)`: NaN must fail
fn check_carrier(n: usize, points: &[(f64, f64)], informative_from: f64) -> Vec<String> {
    /// Rounding allowance on top of `tol`. Measured ≤ 3.5e−12 (n = 4000,
    /// `λ_max t = 1e6`); the derived bound `(r+3)·u·(z/2 + m)·‖v‖₂` (math §63.7.a)
    /// is ≈ 50× larger there, so this floor is the stricter, empirical one.
    const ERR_FLOOR: f64 = 1e-11;
    let (rp, ci, va) = path_laplacian(n);
    let op = SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).expect("path Laplacian");
    let rho = op.lambda_max_bound();
    let v: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 / n as f64;
            (7.0 * x).sin() + 0.3 * (61.0 * x).cos() + if i % 3 == 0 { 0.5 } else { -0.25 }
        })
        .collect();
    let v_inf = sup(v.iter().copied());
    let mean = v.iter().sum::<f64>() / n as f64;
    let mut scratch = ScratchPool::new();
    let mut fails = Vec::new();
    for &(lt, tol) in points {
        let t = lt / rho;
        let want = exact(&v, t);
        let mut got = vec![0.0; n];
        graph_expmv_krylov(
            &op,
            t,
            &v,
            &mut got,
            KrylovPath::Chebyshev,
            tol,
            &mut scratch,
        )
        .expect("chebyshev");
        let err = sup(got.iter().zip(&want).map(|(g, w)| g - w)) / v_inf;
        let dynamics = sup(want.iter().map(|w| w - mean)) / v_inf;
        eprintln!(
            "G_CHEB_STIFF_ORACLE n={n} lt={lt:e} tol={tol:e} err/|v|={err:.3e} \
             |exact-mean|/|v|={dynamics:.3e}"
        );
        if !(err <= tol + ERR_FLOOR) {
            fails.push(format!(
                "n={n} lt={lt:e} tol={tol:e}: err {err:e} > {}",
                tol + ERR_FLOOR
            ));
        }
        // Informative: the non-mean part exceeds the allowance by 1e3.
        if lt >= informative_from && !(dynamics >= 1e3 * (tol + ERR_FLOOR)) {
            fails.push(format!(
                "n={n} lt={lt:e}: slow modes decayed ({dynamics:e}), vacuous"
            ));
        }
    }
    fails
}

#[test]
fn g_cheb_stiff_oracle() {
    let mut fails = check_carrier(
        400,
        &[
            (1e2, 1e-12),
            (1e4, 1e-12),
            (1e6, 1e-12),
            (1e8, 1e-12),
            (1e6, 1e-8),
        ],
        f64::INFINITY,
    );
    // λ₁t = 0.15 / 15.4 at λ_max t = 1e6 / 1e8: the slow modes are alive
    // (non-mean part 0.33 / 7.1e−8 of ‖v‖_∞, ≥ 1e3× the allowance).
    fails.extend(check_carrier(4000, &[(1e6, 1e-12), (1e8, 1e-12)], 0.0));
    assert!(fails.is_empty(), "{fails:#?}");
}
