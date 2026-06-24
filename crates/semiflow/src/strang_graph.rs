//! Palindromic Strang split for commuting graph Chernoff kernels.
//!
//! `S(τ) f = A(τ/2) ∘ B(τ) ∘ A(τ/2) · f` on `GraphSignal<F>`.
//!
//! **Commutativity is a precondition, not a verified property.** Two safe
//! constructors (`new_bipartite_path`, `new_bipartite_cycle`) build
//! guaranteed-commuting decompositions by edge-parity 2-coloring. The generic
//! constructor `new(a, b, commutes_axiomatically: bool)` requires the caller
//! to opt in via the boolean flag.
//!
//! See math.md §12.8 (NORMATIVE), ADR-0012 (palindromic Strang pattern), and
//! Wave 2.1B contract §3.
//!
//! ## Zero-alloc R4 mitigation
//!
//! The naive `src.clone()` pattern allocates a `Vec<F>` per leg and violates
//! the zero-alloc steady-state invariant. Instead, `apply_into` uses
//! `ScratchPool::take_graph_buf` / `return_graph_buf` to borrow and return
//! pool-owned buffers wrapped in transient `GraphSignal`s.

use alloc::{sync::Arc, vec::Vec};

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{half, SemiflowFloat},
    graph::{Graph, Laplacian},
    graph_heat::GraphHeatChernoff,
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// StrangSplitGraph<A, B, F>
// ---------------------------------------------------------------------------

/// Palindromic Strang split for two commuting graph Chernoff kernels.
///
/// `S(τ) f = A(τ/2) ∘ B(τ) ∘ A(τ/2) · f` on `GraphSignal<F>`.
///
/// **Commutativity is a precondition.** See [`Self::new_bipartite_path`] and
/// [`Self::new_bipartite_cycle`] for safe constructors. The generic
/// [`Self::new`] constructor requires `commutes_axiomatically = true` for the
/// palindromic order-2 guarantee.
///
/// See math.md §12.8 (NORMATIVE) and ADR-0012.
#[derive(Clone, Debug)]
pub struct StrangSplitGraph<A, B, F: SemiflowFloat = f64>
where
    A: ChernoffFunction<F, S = GraphSignal<F>> + Clone,
    B: ChernoffFunction<F, S = GraphSignal<F>> + Clone,
{
    a: A,
    b: B,
    /// Caller-attested commutativity. `true` ⇒ order-2; `false` ⇒ order-1.
    commutes: bool,
    _phantom: core::marker::PhantomData<F>,
}

impl<A, B, F: SemiflowFloat> StrangSplitGraph<A, B, F>
where
    A: ChernoffFunction<F, S = GraphSignal<F>> + Clone,
    B: ChernoffFunction<F, S = GraphSignal<F>> + Clone,
{
    /// Generic constructor with explicit commutativity attestation.
    ///
    /// `commutes_axiomatically = true` ⇒ caller asserts `[L_A, L_B] = 0` and
    /// the palindromic product achieves order-2.
    /// `commutes_axiomatically = false` ⇒ Lie-Trotter-style splitting at
    /// order-1 (BCH residue not cancelled).
    pub fn new(a: A, b: B, commutes_axiomatically: bool) -> Self {
        Self {
            a,
            b,
            commutes: commutes_axiomatically,
            _phantom: core::marker::PhantomData,
        }
    }

    /// `[doc(hidden)]` accessor for testing the inner kernels' Laplacians.
    #[doc(hidden)]
    pub fn test_only_kernels(&self) -> (&A, &B) {
        (&self.a, &self.b)
    }
}

// ---------------------------------------------------------------------------
// Safe constructors for GraphHeatChernoff<F> specialisation
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> StrangSplitGraph<GraphHeatChernoff<F>, GraphHeatChernoff<F>, F> {
    /// **Safe constructor**: build a guaranteed-commuting Strang split for a
    /// path graph `P_n` by edge-parity 2-coloring.
    ///
    /// - A-edges (red): `(0,1), (2,3), (4,5), …` (even-indexed undirected edges).
    /// - B-edges (black): `(1,2), (3,4), (5,6), …` (odd-indexed).
    ///
    /// Both Laplacians are block-diagonal on DISJOINT node pairs ⇒ commute.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `graph.n_nodes() < 2`.
    #[allow(clippy::cast_precision_loss)] // n < 2^53 in all practical graph sizes
    pub fn new_bipartite_path(graph: &Arc<Graph<F>>) -> Result<Self, SemiflowError> {
        let n = graph.n_nodes();
        if n < 2 {
            return Err(SemiflowError::DomainViolation {
                what: "StrangSplitGraph::new_bipartite_path: n_nodes >= 2 required",
                value: n as f64,
            });
        }
        let (lap_a, lap_b) = split_path_laplacians(graph);
        // Use order-2 (with_zeta_a) sub-kernels so the palindromic product
        // achieves global order-2 convergence.  With order-1 sub-kernels the
        // local error is O(τ²) giving only global order-1; order-2 sub-kernels
        // give local error O(τ³) so the palindromic cancellation yields
        // global order-2.  See math.md §12.8 (NORMATIVE).
        let a = GraphHeatChernoff::with_zeta_a(Arc::new(lap_a));
        let b = GraphHeatChernoff::with_zeta_a(Arc::new(lap_b));
        Ok(Self {
            a,
            b,
            commutes: true,
            _phantom: core::marker::PhantomData,
        })
    }

    /// **Safe constructor**: build a guaranteed-commuting Strang split for an
    /// **even-length** cycle graph `C_n` by edge-parity 2-coloring.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `graph.n_nodes() < 4` or
    /// `graph.n_nodes() % 2 != 0` (odd cycles are not bipartite).
    #[allow(clippy::cast_precision_loss)] // n < 2^53 in all practical graph sizes
    pub fn new_bipartite_cycle(graph: &Arc<Graph<F>>) -> Result<Self, SemiflowError> {
        let n = graph.n_nodes();
        if n < 4 {
            return Err(SemiflowError::DomainViolation {
                what: "StrangSplitGraph::new_bipartite_cycle: n_nodes >= 4 required",
                value: n as f64,
            });
        }
        if n % 2 != 0 {
            return Err(SemiflowError::DomainViolation {
                what: "StrangSplitGraph::new_bipartite_cycle: n_nodes must be even (odd cycles not bipartite)",
                value: n as f64,
            });
        }
        // For cycle: edges are (0,1),(1,2),...,(n-2,n-1),(n-1,0).
        // Even-indexed undirected edges (0-based): (0,1),(2,3),(4,5),...
        // Plus the wrap edge (n-1,0) has global index n-1.
        // For even n: wrap edge has odd index (n-1 is odd) → goes to B.
        let (lap_a, lap_b) = split_cycle_laplacians(graph);
        // Order-2 sub-kernels for global order-2 convergence (see path comment).
        let a = GraphHeatChernoff::with_zeta_a(Arc::new(lap_a));
        let b = GraphHeatChernoff::with_zeta_a(Arc::new(lap_b));
        Ok(Self {
            a,
            b,
            commutes: true,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Expose sub-Laplacians for numerical commutativity tests.
    #[doc(hidden)]
    #[must_use]
    pub fn test_only_laplacians(&self) -> (&Laplacian<F>, &Laplacian<F>) {
        (self.a.laplacian(), self.b.laplacian())
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<A, B, F: SemiflowFloat> ChernoffFunction<F> for StrangSplitGraph<A, B, F>
where
    A: ChernoffFunction<F, S = GraphSignal<F>> + Clone,
    B: ChernoffFunction<F, S = GraphSignal<F>> + Clone,
{
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "StrangSplitGraph: tau must be finite and >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let half_tau = half::<F>() * tau;
        let n = src.len();
        let graph_arc = src.graph_arc();

        // R4 mitigation: pool-owned buffers, not src.clone().
        // Leg 1: A(τ/2) src → tmp_a  (pool-borrowed)
        let buf_a: Vec<F> = scratch.take_graph_buf(n);
        let mut tmp_a = GraphSignal::from_pool_buf(Arc::clone(&graph_arc), buf_a);
        self.a.apply_into(half_tau, src, &mut tmp_a, scratch)?;

        // Leg 2: B(τ) tmp_a → tmp_b  (pool-borrowed)
        let buf_b: Vec<F> = scratch.take_graph_buf(n);
        let mut tmp_b = GraphSignal::from_pool_buf(Arc::clone(&graph_arc), buf_b);
        self.b.apply_into(tau, &tmp_a, &mut tmp_b, scratch)?;

        // Leg 3: A(τ/2) tmp_b → dst
        self.a.apply_into(half_tau, &tmp_b, dst, scratch)?;

        // Return pool buffers (reclaim the owned vecs from the transient signals).
        let (reclaim_b, _) = tmp_b.into_pool_buf();
        scratch.return_graph_buf(reclaim_b);
        let (reclaim_a, _) = tmp_a.into_pool_buf();
        scratch.return_graph_buf(reclaim_a);

        Ok(())
    }

    fn order(&self) -> u32 {
        if self.commutes {
            // Commuting ⇒ Strang order-2 (capped at min of inner orders, 2).
            core::cmp::min(self.a.order(), self.b.order()).min(2)
        } else {
            1
        }
    }

    fn growth(&self) -> Growth<F> {
        let ga = self.a.growth();
        let gb = self.b.growth();
        // Sub-multiplicative bound for palindromic product.
        Growth {
            multiplier: ga.multiplier * gb.multiplier * ga.multiplier,
            omega: ga.omega + gb.omega + ga.omega,
        }
    }
}

// ---------------------------------------------------------------------------
// Bipartite edge-split helpers
// ---------------------------------------------------------------------------

/// Assemble two combinatorial Laplacians from disjoint edge sets of a path
/// graph by even/odd edge-parity 2-coloring.
///
/// Edge index `k` (0-based among undirected edges `(0,1),(1,2),...`):
/// - Even → `L_A`.
/// - Odd  → `L_B`.
fn split_path_laplacians<F: SemiflowFloat>(graph: &Graph<F>) -> (Laplacian<F>, Laplacian<F>) {
    split_laplacians_by_edge(graph, |edge_idx| edge_idx % 2 == 0)
}

/// Assemble two combinatorial Laplacians from disjoint edge sets of a cycle
/// graph by even/odd edge-parity 2-coloring.
///
/// For an even-length cycle graph `C_n`, the undirected edges are listed in
/// CSR traversal order. Even-indexed edges → A; odd-indexed → B.
fn split_cycle_laplacians<F: SemiflowFloat>(graph: &Graph<F>) -> (Laplacian<F>, Laplacian<F>) {
    split_laplacians_by_edge(graph, |edge_idx| edge_idx % 2 == 0)
}

/// Generic edge-split: iterate the CSR representation and assign each
/// undirected edge to sub-graph A or B by `keep_in_a(undirected_edge_idx)`.
///
/// Builds two new [`Laplacian`]s (combinatorial) from the disjoint edge sets.
#[allow(clippy::cast_possible_truncation)] // node indices < 2^32 in all practical graphs
fn split_laplacians_by_edge<F: SemiflowFloat>(
    graph: &Graph<F>,
    keep_in_a: impl Fn(usize) -> bool,
) -> (Laplacian<F>, Laplacian<F>) {
    let n = graph.n_nodes();
    let mut edges_a: Vec<(u32, u32)> = Vec::new();
    let mut edges_b: Vec<(u32, u32)> = Vec::new();
    let mut edge_idx = 0_usize;

    for u in 0..n {
        let lo = graph.row_ptr()[u];
        let hi = graph.row_ptr()[u + 1];
        for k in lo..hi {
            let v = graph.col_idx()[k] as usize;
            // Only visit each undirected edge once (canonical u < v direction).
            if u < v {
                if keep_in_a(edge_idx) {
                    edges_a.push((u as u32, v as u32));
                } else {
                    edges_b.push((u as u32, v as u32));
                }
                edge_idx += 1;
            }
        }
    }

    let lap_a = build_combinatorial_lap_from_edges(n, &edges_a);
    let lap_b = build_combinatorial_lap_from_edges(n, &edges_b);
    (lap_a, lap_b)
}

/// Build a combinatorial Laplacian from a list of undirected unit-weight edges.
fn build_combinatorial_lap_from_edges<F: SemiflowFloat>(
    n: usize,
    edges: &[(u32, u32)],
) -> Laplacian<F> {
    // Reconstruct a sub-Graph then assemble via the standard path.
    let edge_iter = edges.iter().map(|&(u, v)| (u, v, F::one()));
    let sub_graph = Graph::from_edges(n, edge_iter)
        .expect("split_laplacians: sub-graph build failed (should never happen)");
    Laplacian::assemble_combinatorial(&sub_graph)
}

// ---------------------------------------------------------------------------
// Unit smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::*;
    use crate::{
        graph::{Graph, Laplacian},
        graph_signal::GraphSignal,
        state::State,
    };

    #[test]
    fn new_bipartite_path_rejects_small_graph() {
        let g = Arc::new(Graph::<f64>::path(1));
        assert!(StrangSplitGraph::new_bipartite_path(&g).is_err());
    }

    #[test]
    fn new_bipartite_cycle_rejects_odd_n() {
        let g = Arc::new(Graph::<f64>::cycle(5));
        assert!(StrangSplitGraph::new_bipartite_cycle(&g).is_err());
    }

    #[test]
    fn new_bipartite_cycle_rejects_small_n() {
        let g = Arc::new(Graph::<f64>::cycle(3));
        assert!(StrangSplitGraph::new_bipartite_cycle(&g).is_err());
    }

    #[test]
    fn apply_at_zero_tau_returns_src() {
        let g = Arc::new(Graph::<f64>::path(8));
        let strang = StrangSplitGraph::new_bipartite_path(&g).unwrap();
        let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
        let mut dst = src.clone();
        let mut scratch = ScratchPool::<f64>::new();
        strang
            .apply_into(0.0, &src, &mut dst, &mut scratch)
            .unwrap();
        let mut diff = dst.clone();
        diff.axpy_into(-1.0, &src);
        assert!(diff.norm_sup() < 1e-14);
    }

    #[test]
    fn order_commuting_is_two() {
        let g = Arc::new(Graph::<f64>::path(8));
        let strang = StrangSplitGraph::new_bipartite_path(&g).unwrap();
        assert_eq!(strang.order(), 2);
    }

    #[test]
    fn order_non_commuting_is_one() {
        let g = Arc::new(Graph::<f64>::path(8));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let a = GraphHeatChernoff::new(Arc::clone(&lap));
        let b = GraphHeatChernoff::new(Arc::clone(&lap));
        let strang = StrangSplitGraph::new(a, b, false);
        assert_eq!(strang.order(), 1);
    }
}
