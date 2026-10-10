//! `G_NONFINITE_BOUND_REJECT` (`RELEASE_BLOCKING`, issue #44): a non-finite operator
//! norm bound is an error wherever a schedule is derived from it — never a
//! silently wrong schedule.
//!
//! Every Krylov / Taylor / φ path derives its degree and substep count from a norm
//! bound (Gershgorin `ρ̄` or `‖A‖_∞`). Finite entries can still overflow that bound
//! (a 2×2 block of `1e308`). Such an operator is legal — with a diagonal mass its
//! generator `M⁻¹A` has a finite norm (`G_SPDR_OVERFLOW_ROBUST`) — so construction
//! succeeds and every CONSUMER of the bound rejects it. Before the fix an `∞`
//! reached `select_s_m`, skipped every θ row and returned one substep: an
//! inaccurate result with no error.
//!
//! The Lanczos `m_max` cap (`G_LANCZOS_M_MAX_CAP`) lives in `g_lanczos_hl_cost.rs`.

use semiflow::{
    general_operator::{expmv_cost_probe, GeneralOperator},
    graph_expmv_krylov, phi_action, phi_cost_probe, CsrGenerator, Graph, KrylovPath, Laplacian,
    ScratchPool, SemiflowError, SymmetricLinearOp, SymmetricOperator,
};

const BIG: f64 = 1e308;

fn dense2_csr(a: [[f64; 2]; 2]) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    (
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![a[0][0], a[0][1], a[1][0], a[1][1]],
    )
}

fn is_domain<T>(r: &Result<T, SemiflowError>) -> bool {
    matches!(r, Err(SemiflowError::DomainViolation { .. }))
}

/// Every symmetric consumer of the bound rejects an overflowed one.
fn assert_symmetric_consumers_reject(op: &SymmetricOperator<f64>, label: &str) {
    assert!(
        op.lambda_max_bound().is_infinite(),
        "{label}: bound should overflow"
    );
    assert!(
        is_domain(&CsrGenerator::from_symmetric(op, None)),
        "{label}: CsrGenerator"
    );
    for path in [KrylovPath::Chebyshev, KrylovPath::Lanczos { m_max: 18 }] {
        assert!(
            is_domain(&op.krylov(path, 1e-10)),
            "{label}: krylov({path:?})"
        );
        let mut out = vec![0.0; op.n()];
        let v = vec![1.0; op.n()];
        let r = graph_expmv_krylov(op, 0.5, &v, &mut out, path, 1e-10, &mut ScratchPool::new());
        assert!(is_domain(&r), "{label}: graph_expmv_krylov({path:?})");
    }
}

#[test]
fn g_nonfinite_bound_symmetric_from_csr() {
    let (rp, ci, va) = dense2_csr([[BIG, BIG], [BIG, BIG]]);
    let op = SymmetricOperator::from_csr(2, &rp, &ci, &va, 0.0).expect("legal operator");
    assert_symmetric_consumers_reject(&op, "2×2 all-1e308");
    // With a mass the generator is finite (‖M⁻¹A‖ = 2): construction must succeed.
    let g = CsrGenerator::from_symmetric(&op, Some(&[BIG, BIG])).expect("mass path");
    assert!(semiflow::GeneratorAction::norm_bound(&g).is_finite());
}

#[test]
fn g_nonfinite_bound_with_diagonal() {
    let h = 0.6 * BIG;
    let (rp, ci, va) = dense2_csr([[h, h], [h, h]]);
    let op = SymmetricOperator::from_csr(2, &rp, &ci, &va, 0.0).expect("row sum 1.2e308 is finite");
    assert!(op.lambda_max_bound().is_finite());
    // Entries stay finite (1.2e308) but the row sum 1.8e308 overflows.
    let d = op.with_diagonal(&[h, 0.0]).expect("entries finite");
    assert_symmetric_consumers_reject(&d, "with_diagonal");
}

#[test]
fn g_nonfinite_bound_lumped_congruence() {
    let (rp, ci, va) = dense2_csr([[1.0, 0.0], [0.0, 1.0]]);
    let op = SymmetricOperator::from_csr(2, &rp, &ci, &va, 0.0).expect("identity");
    // 1 / 1e-310 overflows the ENTRY: this used to build a Laplacian with an `inf` entry.
    assert!(is_domain(&op.lumped_congruence(&[1e-310, 1.0])));
}

#[test]
fn g_nonfinite_bound_laplacian_entries() {
    let r = Laplacian::from_csr_parts(
        1,
        vec![0, 1],
        vec![0],
        vec![f64::INFINITY],
        semiflow::LaplacianKind::GeneralSymmetric,
    );
    assert!(is_domain(&r), "non-finite entry");
}

#[test]
fn g_nonfinite_bound_general_operator_and_generator() {
    let (rp, ci, va) = dense2_csr([[BIG, -BIG], [0.0, 1.0]]);
    let g = GeneralOperator::from_csr(2, &rp, &ci, &va).expect("legal operator");
    assert!(g.norm_inf_bound().is_infinite());
    assert!(is_domain(&CsrGenerator::from_general(&g, None)));
    let mut out = [0.0; 2];
    assert!(is_domain(&g.expmv().action_into_slice(
        0.5,
        &[1.0, 1.0],
        &mut out
    )));
    // A legal operator still yields a finite generator norm.
    let (rp, ci, va) = dense2_csr([[2.0, -1.0], [-1.0, 2.0]]);
    let g = GeneralOperator::from_csr(2, &rp, &ci, &va).expect("finite");
    assert!(CsrGenerator::from_general(&g, None).is_ok());
}

#[test]
fn g_nonfinite_bound_graph_degree_overflow() {
    let r = Graph::<f64>::from_edges(3, [(0, 1, BIG), (0, 2, BIG)]);
    assert!(
        is_domain(&r),
        "node 0 has weighted degree 2e308 (an ∞ diagonal entry)"
    );
    // Degree 1e308 is a finite entry; the combinatorial row sum 2e308 overflows.
    let g = Graph::<f64>::from_edges(2, [(0, 1, BIG)]).expect("finite degrees");
    let lap = std::sync::Arc::new(Laplacian::assemble_combinatorial(&g));
    assert!(lap.spectral_radius_bound().is_infinite());
    let r = semiflow::GraphKrylovChernoff::new(lap, KrylovPath::Chebyshev, 1e-10);
    assert!(is_domain(&r), "GraphKrylovChernoff must reject an ∞ bound");
}

#[test]
fn g_nonfinite_bound_schedule_selectors() {
    for bad in [f64::INFINITY, f64::NAN, -1.0] {
        assert_eq!(
            expmv_cost_probe(bad, 1.0),
            (0, 0),
            "expmv probe, norm={bad}"
        );
        assert_eq!(phi_cost_probe(bad, 1.0), (0, 0), "phi probe, norm={bad}");
    }
    // `s` would exceed u32::MAX: no schedule, not a saturated one.
    assert_eq!(expmv_cost_probe(1e300, 1.0), (0, 0));
    let (s, m) = expmv_cost_probe(10.0, 1.0);
    assert!(s >= 1 && m >= 1);
}

#[test]
fn g_nonfinite_bound_phi_rejects_bad_tau() {
    let (rp, ci, va) = dense2_csr([[2.0, -1.0], [-1.0, 2.0]]);
    let op = SymmetricOperator::from_csr(2, &rp, &ci, &va, 1e-12).expect("op");
    let gen = CsrGenerator::from_symmetric(&op, None).expect("gen");
    let mut scratch = ScratchPool::new();
    let mut out = [0.0; 2];
    for tau in [f64::NAN, f64::INFINITY, -0.5] {
        let r = phi_action(&gen, 1, tau, &[1.0, 0.0], &mut out, &mut scratch);
        assert!(is_domain(&r), "phi_action tau={tau}");
    }
}

/// External operator whose bound overflowed: Krylov must refuse it.
struct InfBound;
impl SymmetricLinearOp<f64> for InfBound {
    fn n(&self) -> usize {
        2
    }
    fn lambda_max_bound(&self) -> f64 {
        f64::INFINITY
    }
    fn apply_into_slice(&self, src: &[f64], dst: &mut [f64]) {
        dst.copy_from_slice(src);
    }
}

#[test]
fn g_nonfinite_bound_krylov_external_operator() {
    let mut scratch = ScratchPool::new();
    let mut out = [0.0; 2];
    for path in [KrylovPath::Chebyshev, KrylovPath::Lanczos { m_max: 18 }] {
        let r = graph_expmv_krylov(
            &InfBound,
            1.0,
            &[1.0, 2.0],
            &mut out,
            path,
            1e-10,
            &mut scratch,
        );
        assert!(is_domain(&r), "{path:?}");
    }
}
