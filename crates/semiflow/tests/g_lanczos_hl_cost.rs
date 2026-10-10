//! Lanczos schedule gates (`RELEASE_BLOCKING`, ADR-0205).
//!
//! * `G_LANCZOS_M_MAX_CAP`: the schedule `(s, m)` returned by
//!   `graph_expmv_matvec_count` respects `m ≤ m_max`, and the substep count
//!   belongs to THAT degree: the Hochbruck–Lubich bound of one step of length
//!   `τ/s`, recomputed here independently, is `≤ tol/s` (the old kernel chose `s`
//!   for the uncapped degree, then capped `m`).
//! * `G_LANCZOS_HL_COST`: the schedule costs at most half the mat-vecs of the
//!   Taylor θ-table rule it replaced (`⌈τλ/θ₁₈⌉·18`, `θ₁₈ = 1.091`) for
//!   `λτ ≥ 10`, and stays accurate: the Neumann path Laplacian (`n = 400`)
//!   against its cosine eigenbasis, `‖err‖_∞ ≤ (tol + 1e−11)·‖v‖_∞`.

#![allow(clippy::cast_precision_loss)] // small counts and indices

use std::f64::consts::PI;

use semiflow::{
    graph_expmv_krylov, graph_expmv_matvec_count, KrylovPath, ScratchPool, SymmetricOperator,
};

/// `ln` of the Hochbruck–Lubich bound (SIAM J. Numer. Anal. 34, 1997, Thm 2) for
/// `m` steps at `x = ρh`, `ρ = λ_max/4`; `+∞` outside both regimes.
fn ln_hl(m: f64, x: f64) -> f64 {
    if m >= 2.0 * x {
        (10.0 / x).ln() - x + m * (1.0 + (x / m).ln())
    } else if m * m >= 4.0 * x {
        10f64.ln() - m * m / (5.0 * x)
    } else {
        f64::INFINITY
    }
}

/// `ln` of the Chebyshev-interpolation bound `4xᵐ/m!` (Saad's lemma).
fn ln_interp(m: u32, x: f64) -> f64 {
    let ln_fact: f64 = (2..=m).map(|k| f64::from(k).ln()).sum();
    4f64.ln() + f64::from(m) * x.ln() - ln_fact
}

/// Smallest proven `ln` bound for a step `x`: both bounds bound `2·E_{m−1}`, which
/// is nondecreasing in the step, so a bound at any `x'' ≥ x` also counts. The HL
/// regime-1 form `10e^{−m²/(5x'')}` holds on `[m/2, m²/4]` (both conditions hold
/// at `x'' = m/2`) and increases in `x''`, so its best `x''` is `max(x, m/2)`.
fn ln_step_bound(m: u32, x: f64) -> f64 {
    let mf = f64::from(m);
    let x1 = x.max(0.5 * mf);
    let regime1 = if mf * mf >= 4.0 * x1 {
        10f64.ln() - mf * mf / (5.0 * x1)
    } else {
        f64::INFINITY
    };
    ln_interp(m, x).min(ln_hl(mf, x)).min(regime1)
}

#[test]
fn g_lanczos_m_max_cap() {
    let tol = 1e-10;
    let mut scheduled = 0;
    for m_max in [2_usize, 4, 8, 18, 30, 64] {
        for e in -10..=40 {
            let lt = 10f64.powf(f64::from(e) / 10.0);
            let path = KrylovPath::Lanczos { m_max };
            let (s, m) = graph_expmv_matvec_count(lt, 1.0_f64, tol, &path);
            assert!(m as usize <= m_max, "m={m} > m_max={m_max} at λτ={lt:e}");
            if s == u32::MAX {
                // s exceeds u32 (m_max = 2 needs s ≈ 2(ρτ)²/tol): the
                // kernel reports it (checked below), never a truncated schedule.
                assert_eq!(m_max, 2, "λτ={lt:e}: unexpectedly unschedulable");
                continue;
            }
            scheduled += 1;
            let ln_step = ln_step_bound(m, 0.25 * lt / f64::from(s));
            assert!(
                ln_step <= (tol / f64::from(s)).ln() + 1e-9,
                "m_max={m_max} λτ={lt:e}: (s={s}, m={m}) step bound e^{ln_step} > tol/s"
            );
        }
    }
    assert!(scheduled > 5 * 51, "only {scheduled} schedulable points");
    let (rp, ci, va) = path_laplacian(8);
    let op = SymmetricOperator::from_csr(8, &rp, &ci, &va, 0.0).expect("path");
    let mut out = vec![0.0; 8];
    let r = graph_expmv_krylov(
        &op,
        1e4,
        &[1.0; 8],
        &mut out,
        KrylovPath::Lanczos { m_max: 1 },
        tol,
        &mut ScratchPool::new(),
    );
    assert!(
        r.is_err(),
        "m_max = 1 cannot reach tol = 1e-10: must be an error"
    );
}

/// Neumann path Laplacian `n×n` (unit weights): CSR triple.
fn path_laplacian(n: usize) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let (mut rp, mut ci, mut va) = (vec![0], Vec::new(), Vec::new());
    let idx = |i: usize| u32::try_from(i).expect("n < 2^32");
    for i in 0..n {
        if i > 0 {
            ci.push(idx(i - 1));
            va.push(-1.0);
        }
        ci.push(idx(i));
        va.push(f64::from(u8::from(i > 0) + u8::from(i + 1 < n)));
        if i + 1 < n {
            ci.push(idx(i + 1));
            va.push(-1.0);
        }
        rp.push(ci.len());
    }
    (rp, ci, va)
}

/// `e^{−tL}v` by the cosine eigenbasis (`λ_j = 2 − 2cos(jπ/n)`).
fn exact(v: &[f64], t: f64) -> Vec<f64> {
    let n = v.len();
    let nf = n as f64;
    let table: Vec<f64> = (0..4 * n)
        .map(|k| (k as f64 * PI / (2.0 * nf)).cos())
        .collect();
    let phi = |i: usize, j: usize| table[((2 * i + 1) * j) % (4 * n)];
    let mut out = vec![0.0; n];
    for j in 0..n {
        let decay = (-t * (2.0 - 2.0 * table[(2 * j) % (4 * n)])).exp();
        let norm = if j == 0 { nf } else { nf / 2.0 };
        let coef = v
            .iter()
            .enumerate()
            .map(|(i, &x)| x * phi(i, j))
            .sum::<f64>()
            * decay
            / norm;
        for (i, o) in out.iter_mut().enumerate() {
            *o += coef * phi(i, j);
        }
    }
    out
}

/// `(err/‖v‖∞, s, m)` of one Lanczos action on the path Laplacian against the oracle.
fn lanczos_point(
    op: &SymmetricOperator<f64>,
    v: &[f64],
    lt: f64,
    tol: f64,
    m_max: usize,
) -> (f64, u32, u32) {
    let rho = op.lambda_max_bound();
    let path = KrylovPath::Lanczos { m_max };
    let (s, m) = graph_expmv_matvec_count(rho, lt / rho, tol, &path);
    let mut got = vec![0.0; v.len()];
    graph_expmv_krylov(
        op,
        lt / rho,
        v,
        &mut got,
        path,
        tol,
        &mut ScratchPool::new(),
    )
    .expect("lanczos");
    let v_inf = v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
    let err = got
        .iter()
        .zip(exact(v, lt / rho))
        .fold(0.0_f64, |a, (g, w)| {
            if (g - w).is_nan() {
                f64::NAN
            } else {
                a.max((g - w).abs())
            }
        });
    (err / v_inf, s, m)
}

#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // ceil of a small positive count
fn g_lanczos_hl_cost() {
    const N: usize = 400;
    const ERR_FLOOR: f64 = 1e-11;
    let (rp, ci, va) = path_laplacian(N);
    let op = SymmetricOperator::from_csr(N, &rp, &ci, &va, 0.0).expect("path Laplacian");
    let v: Vec<f64> = (0..N)
        .map(|i| (0.07 * i as f64).sin() + 0.2 * (i % 5) as f64)
        .collect();
    let mut fails = Vec::new();
    for m_max in [18_usize, 30] {
        for (lt, tol) in [
            (1.0, 1e-10),
            (10.0, 1e-10),
            (1e2, 1e-10),
            (1e3, 1e-12),
            (1e3, 1e-6),
        ] {
            let (err, s, m) = lanczos_point(&op, &v, lt, tol, m_max);
            let cost = u64::from(s) * u64::from(m);
            let legacy = (lt / 1.091).ceil() as u64 * 18;
            eprintln!(
                "G_LANCZOS_HL_COST m_max={m_max} λτ={lt:e} tol={tol:e} (s,m)=({s},{m}) \
                 cost={cost} legacy={legacy} err/|v|={err:.3e}"
            );
            if lt >= 10.0 && 2 * cost > legacy {
                fails.push(format!(
                    "m_max={m_max} λτ={lt:e}: {cost} mat-vecs > legacy {legacy}/2"
                ));
            }
            #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must fail
            let bad = !(err <= tol + ERR_FLOOR);
            if bad {
                fails.push(format!("m_max={m_max} λτ={lt:e} tol={tol:e}: err {err:e}"));
            }
        }
    }
    assert!(fails.is_empty(), "{fails:#?}");
}
