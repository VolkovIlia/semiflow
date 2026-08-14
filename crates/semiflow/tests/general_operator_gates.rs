//! `G_GENOP_*` — non-symmetric operator action (ADR-0195, Issue #24).
//!
//! `SymmetricOperator::from_csr` validates symmetry, closing the whole Krylov
//! surface to non-self-adjoint generators. `GeneralOperator` opens it via the
//! scaled-truncated-Taylor engine, which needs only a matvec.
//!
//! Reference throughout is a dense series expansion computed at high enough
//! order to be converged for the small `n` used here — an oracle independent of
//! the scaled-and-squared Taylor path under test (different scaling, different
//! truncation, different arithmetic ordering).

// Row/column index -> matrix entry arithmetic on small test operators.
#![allow(clippy::cast_precision_loss)]

use semiflow::general_operator::{expmv_cost_probe, GeneralOperator};

/// Dense `e^{−tA}·v` by direct series summation — independent of the kernel.
///
/// Converged by construction: terms are summed until they stop contributing at
/// f64 precision, on matrices small and mild enough that the series is not
/// hump-dominated.
fn dense_expmv_ref(n: usize, dense: &[f64], t: f64, v: &[f64]) -> Vec<f64> {
    let mut term: Vec<f64> = v.to_vec();
    let mut acc: Vec<f64> = v.to_vec();
    let mut tmp = vec![0.0_f64; n];
    for k in 1..=400_usize {
        // tmp = (-t) * A * term / k
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += dense[i * n + j] * term[j];
            }
            tmp[i] = -t * s / (k as f64);
        }
        term.copy_from_slice(&tmp);
        let mut mag = 0.0_f64;
        for i in 0..n {
            acc[i] += term[i];
            mag = mag.max(term[i].abs());
        }
        if mag < 1e-18 {
            break;
        }
    }
    acc
}

/// Dense → CSR.
fn to_csr(n: usize, dense: &[f64]) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..n {
            let a = dense[i * n + j];
            if a != 0.0 {
                #[allow(clippy::cast_possible_truncation)]
                ci.push(j as u32);
                va.push(a);
            }
        }
        rp.push(va.len());
    }
    (rp, ci, va)
}

/// Drifted Fokker–Planck `∂_t p = ∂_x(D ∂_x p) − ∂_x(μ p)`, upwinded.
///
/// The drift term makes the off-diagonals genuinely unequal — this matrix is
/// non-symmetric by construction, not by rounding.
fn drifted_fokker_planck(n: usize) -> Vec<f64> {
    let dx = 1.0 / (n as f64 - 1.0);
    let mut a = vec![0.0_f64; n * n];
    for i in 1..n - 1 {
        let d = 0.05;
        let mu = 0.4; // constant positive drift → upwind from the left
        let diff = d / (dx * dx);
        let adv = mu / dx;
        a[i * n + (i - 1)] = -diff - adv;
        a[i * n + i] = 2.0 * diff + adv;
        a[i * n + (i + 1)] = -diff;
    }
    a[0] = 1.0;
    a[(n - 1) * n + (n - 1)] = 1.0;
    a
}

/// Cartea–Jaimungal inventory ladder: upper-bidiagonal, nilpotent + diagonal.
fn inventory_ladder(n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        a[i * n + i] = 0.1 * (i as f64) - 0.3;
        if i + 1 < n {
            a[i * n + (i + 1)] = -0.7;
        }
    }
    a
}

fn run_case(n: usize, dense: &[f64], t: f64, tol: f64, label: &str) {
    let (rp, ci, va) = to_csr(n, dense);
    let op = GeneralOperator::<f64>::from_csr(n, &rp, &ci, &va)
        .unwrap_or_else(|e| panic!("{label}: from_csr rejected a valid CSR: {e:?}"));
    let kernel = op.expmv();

    let v: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.7).sin() + 1.0).collect();
    let mut got = vec![0.0_f64; n];
    kernel.action_into_slice(t, &v, &mut got).unwrap();
    let want = dense_expmv_ref(n, dense, t, &v);

    let err = got
        .iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err <= tol,
        "{label}: sup_error={err:.3e} > {tol:.1e}\n got={got:?}\nwant={want:?}"
    );
    assert!(err > 0.0 || want.iter().all(|w| w.is_finite()));
}

/// `G_GENOP_DENSE` — drifted Fokker–Planck matches the dense reference.
#[test]
fn g_genop_dense() {
    let n = 10;
    run_case(n, &drifted_fokker_planck(n), 0.3, 1e-10, "drifted Fokker-Planck");
}

/// `G_GENOP_NONNORMAL` — the inventory ladder (nilpotent + diagonal) matches.
///
/// Deliberately a looser tolerance than `G_GENOP_DENSE`. This matrix has an
/// ill-conditioned eigenvector basis, so only the *backward* error is certified;
/// the forward error can exceed the backward radius by `κ(V)`, which is not
/// estimated. The gate measures one instance and does not generalise — that is
/// the honest-limits statement in ADR-0195, made falsifiable.
#[test]
fn g_genop_nonnormal() {
    let n = 12;
    run_case(n, &inventory_ladder(n), 0.5, 1e-9, "inventory ladder");
}

/// `G_GENOP_ASYM_ACCEPTED` — teeth.
///
/// The SAME CSR must be rejected by `SymmetricOperator::from_csr` and accepted
/// by `GeneralOperator::from_csr`. Without this the capability could be an
/// accidental relaxation rather than a genuinely new one.
#[test]
fn g_genop_asym_accepted() {
    let n = 10;
    let dense = drifted_fokker_planck(n);
    let (rp, ci, va) = to_csr(n, &dense);

    assert!(
        semiflow::symmetric_operator::SymmetricOperator::<f64>::from_csr(
            n, &rp, &ci, &va, 1e-10
        )
        .is_err(),
        "the datum is not actually asymmetric — the teeth check is vacuous"
    );
    assert!(GeneralOperator::<f64>::from_csr(n, &rp, &ci, &va).is_ok());
}

/// The transpose is a real transpose, not the self-adjoint default.
///
/// `⟨Aᵀx, y⟩ == ⟨x, Ay⟩` must hold for a NON-symmetric `A`; if
/// `apply_transpose_into_slice` fell back to `apply_into_slice`, it would not.
#[test]
fn genop_transpose_is_exact() {
    let n = 10;
    let dense = drifted_fokker_planck(n);
    let (rp, ci, va) = to_csr(n, &dense);
    let op = GeneralOperator::<f64>::from_csr(n, &rp, &ci, &va).unwrap();

    let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.31).cos()).collect();
    let y: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.77).sin()).collect();
    let (mut atx, mut ay) = (vec![0.0; n], vec![0.0; n]);
    op.apply_transpose_into_slice(&x, &mut atx);
    op.apply_into_slice(&y, &mut ay);

    let lhs: f64 = atx.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let rhs: f64 = x.iter().zip(ay.iter()).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-12 * lhs.abs().max(1.0),
        "adjoint identity failed: {lhs:.17e} vs {rhs:.17e}"
    );

    // Non-vacuity: A is asymmetric, so A^T x differs from A x.
    let mut ax = vec![0.0; n];
    op.apply_into_slice(&x, &mut ax);
    assert!(
        atx.iter().zip(ax.iter()).any(|(a, b)| (a - b).abs() > 1e-12),
        "datum is symmetric — the transpose check proves nothing"
    );
}

/// `G_GENOP_COST_LINEAR` (ADVISORY) — cost is linear in `t‖A‖`, not flat.
///
/// This pins the *anti*-claim. `G_GRAPH_EXPMV_DEPTH_FLAT` must never be extended
/// to this path, and stating that in prose is weaker than measuring it.
#[test]
fn g_genop_cost_linear_advisory() {
    let dense = drifted_fokker_planck(10);
    let (rp, _, va) = to_csr(10, &dense);
    let norm = (0..10)
        .map(|i| (rp[i]..rp[i + 1]).map(|k| va[k].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    let cost = |t: f64| -> f64 {
        let (s, m) = expmv_cost_probe(norm, t);
        f64::from(s) * f64::from(m)
    };
    let (c1, c2, c4) = (cost(0.1), cost(0.2), cost(0.4));
    for (lo, hi, label) in [(c1, c2, "0.1->0.2"), (c2, c4, "0.2->0.4")] {
        let ratio = hi / lo;
        assert!(
            (1.5..=2.6).contains(&ratio),
            "cost ratio {label} = {ratio:.3} is not ~2x — depth-flatness is NOT claimed \
             for this path, and this test exists to keep that measurable"
        );
    }
}
