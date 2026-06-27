// Tests for `graph_krylov.rs` — `GraphKrylovChernoff` smoke tests (ADR-0185).
//
// Properties asserted:
//   1. Chebyshev path: contraction ‖e^{-τL}v‖ ≤ ‖v‖ on a 5-node path graph.
//   2. Lanczos path: contraction on a 6-node path graph.
//   3. Both Laplacian kinds accepted by constructor (D5 boundary: symmetric).
//   4. Chebyshev and Lanczos agree within tolerance (parity test).
//   5. Lanczos at large τ (n ≥ 18): no OOB panic, Chebyshev-oracle ≤ 1e-9 (fix/lanczos-stiff-oob).

use alloc::sync::Arc;

use super::*;
use crate::{
    chernoff::ChernoffFunction,
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    scratch::ScratchPool,
};

#[test]
fn chebyshev_smoke_small() {
    // Chebyshev path on a 5-node path: result should have ‖e^{-τL}v‖ ≤ ‖v‖.
    let g = Arc::new(Graph::<f64>::path(5));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let krylov = GraphKrylovChernoff::default_cheb(Arc::clone(&lap));
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| if i == 2 { 1.0 } else { 0.0 });
    let mut dst = GraphSignal::zeros(Arc::clone(&g));
    let mut scratch = ScratchPool::new();
    krylov.apply_into(0.1, &src, &mut dst, &mut scratch).unwrap();
    let norm_in: f64 = src.values().iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_out: f64 = dst.values().iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm_out <= norm_in + 1e-12, "contraction violated: {norm_out} > {norm_in}");
}

#[test]
fn lanczos_smoke_small() {
    let g = Arc::new(Graph::<f64>::path(6));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let krylov = GraphKrylovChernoff::new(
        Arc::clone(&lap),
        KrylovPath::Lanczos { m_max: 6 },
        1e-10,
    )
    .unwrap();
    let src = GraphSignal::from_fn(Arc::clone(&g), |_| 1.0);
    let mut dst = GraphSignal::zeros(Arc::clone(&g));
    let mut scratch = ScratchPool::new();
    krylov.apply_into(0.05, &src, &mut dst, &mut scratch).unwrap();
    // Semigroup property: ‖e^{-τL}v‖ ≤ ‖v‖ for PSD L.
    let norm_src: f64 = src.values().iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_dst: f64 = dst.values().iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm_dst <= norm_src + 1e-10);
}

#[test]
fn both_laplacian_kinds_accepted() {
    // D5 boundary: both existing LaplacianKind variants are symmetric — constructor succeeds.
    let g = Arc::new(Graph::<f64>::path(4));
    let lap_comb = Arc::new(Laplacian::assemble_combinatorial(&g));
    let lap_norm = Arc::new(Laplacian::assemble_normalized(&g));
    assert!(GraphKrylovChernoff::new(lap_comb, KrylovPath::Chebyshev, 1e-10).is_ok());
    assert!(GraphKrylovChernoff::new(lap_norm, KrylovPath::Chebyshev, 1e-10).is_ok());
}

#[test]
fn chebyshev_lanczos_parity() {
    // Both paths should agree to within tolerance on a small graph.
    let g = Arc::new(Graph::<f64>::path(8));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let cheb = GraphKrylovChernoff::default_cheb(Arc::clone(&lap));
    let lanc = GraphKrylovChernoff::new(Arc::clone(&lap), KrylovPath::Lanczos { m_max: 18 }, 1e-10).unwrap();
    // i is u32 (from_fn index type) — f64::from(i) is exact and infallible.
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
    let mut dst_c = GraphSignal::zeros(Arc::clone(&g));
    let mut dst_l = GraphSignal::zeros(Arc::clone(&g));
    let mut scratch = ScratchPool::new();
    cheb.apply_into(0.2, &src, &mut dst_c, &mut scratch).unwrap();
    lanc.apply_into(0.2, &src, &mut dst_l, &mut scratch).unwrap();
    let sup_err: f64 = dst_c.values().iter().zip(dst_l.values().iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(sup_err < 1e-8, "chebyshev/lanczos parity error = {sup_err}");
}

/// Regression gate for fix/lanczos-stiff-oob (`graph_krylov_helpers.rs:129`).
///
/// Calls Lanczos with `n=20 ≥ MAX_LANCZOS_DIM=18` at τ ∈ {4, 8, 16} where
/// `lanczos_select_s_m` returns `m=18`.  Before the fix, `lanczos_iterate` wrote
/// `beta[18]` into `[F; 18]`, panicking with "index out of bounds: len is 18 but index is 18".
///
/// Independent oracle: Chebyshev path at tol=1e-12 (separate algorithm, validated
/// against `mat_exp_pade13` in `G_GRAPH_EXPMV_DENSE` for small n).
/// Gate threshold: `sup_error` ≤ 1e-9 at each τ.
#[test]
fn lanczos_large_t_no_oob() {
    // n=20 > MAX_LANCZOS_DIM=18 so the Krylov dim is not capped to n — it stays 18.
    let n = 20usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    // Oracle: Chebyshev at tight tolerance — independent of Lanczos.
    let cheb = GraphKrylovChernoff::default_cheb(Arc::clone(&lap));
    let lanc = GraphKrylovChernoff::new(
        Arc::clone(&lap),
        KrylovPath::Lanczos { m_max: 18 },
        1e-10,
    )
    .unwrap();
    // Unit impulse at node 10 (u32 index).
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| if i == 10u32 { 1.0_f64 } else { 0.0 });
    let mut scratch = ScratchPool::new();
    // τ=4: arg=4·λ_max≈16→m=18 selected with s=2 substeps (cost=36 < 52).
    // τ=8: arg=32→m=18 with s=4.  τ=16: arg=64→m=18 with s=8.
    for &tau in &[4.0_f64, 8.0, 16.0] {
        let mut dst_cheb = GraphSignal::zeros(Arc::clone(&g));
        let mut dst_lanc = GraphSignal::zeros(Arc::clone(&g));
        cheb.apply_into(tau, &src, &mut dst_cheb, &mut scratch).unwrap();
        // Before fix: panics "index out of bounds: len is 18 but index is 18".
        lanc.apply_into(tau, &src, &mut dst_lanc, &mut scratch).unwrap();
        let sup_err: f64 = dst_cheb
            .values()
            .iter()
            .zip(dst_lanc.values().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            sup_err <= 1e-9,
            "lanczos_large_t_no_oob: tau={tau}  sup_err={sup_err:.3e} > 1e-9"
        );
    }
}
