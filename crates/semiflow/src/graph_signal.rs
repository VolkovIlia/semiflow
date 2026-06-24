//! [`GraphSignal<F>`] — dense `Vec<F>`-backed function on a graph's node set.
//!
//! Implements [`State<F>`], [`HilbertState<F>`], and [`Discrete<F>`] for use as
//! the state space of [`crate::GraphHeatChernoff`].
//!
//! See `contracts/v2.1/wave-a-graph-foundations.md` §2 and ADR-0047.

use alloc::{sync::Arc, vec, vec::Vec};

use num_traits::Float;

use crate::{
    float::SemiflowFloat,
    graph::Graph,
    state::{Discrete, HilbertState, State},
};

// ---------------------------------------------------------------------------
// GraphSignal<F>
// ---------------------------------------------------------------------------

/// Dense signal on a graph's node set: one `F`-value per node.
///
/// Holds an `Arc<Graph<F>>` so [`Discrete<F>::neighbours`] can iterate the
/// backing CSR without copying. Multiple signals may share the same `Arc`.
///
/// `Clone` is derived — clones the `Vec<F>` and bumps the `Arc` ref-count.
#[derive(Clone)]
pub struct GraphSignal<F: SemiflowFloat = f64> {
    pub(crate) values: Vec<F>,
    graph: Arc<Graph<F>>,
}

impl<F: SemiflowFloat> GraphSignal<F> {
    /// Construct from a closure `init(node_index) -> F`.
    #[must_use]
    pub fn from_fn(graph: Arc<Graph<F>>, init: impl Fn(u32) -> F) -> Self {
        let n = graph.n_nodes();
        #[allow(clippy::cast_possible_truncation)]
        let values = (0..n as u32).map(init).collect();
        Self { values, graph }
    }

    /// Zero signal on `graph`.
    #[must_use]
    pub fn zeros(graph: Arc<Graph<F>>) -> Self {
        let n = graph.n_nodes();
        Self {
            values: vec![F::zero(); n],
            graph,
        }
    }

    /// Construct a `GraphSignal` wrapping a pool-owned `Vec<F>`.
    ///
    /// Used by `StrangSplitGraph::apply_into` to avoid heap allocation in the
    /// steady-state hot path (R4 mitigation, Wave 2.1B). The caller takes a
    /// buffer via [`crate::scratch::ScratchPool::take_graph_buf`], wraps it
    /// here, uses the signal, then destructs it back into its parts and returns
    /// the buffer via `return_graph_buf`.
    ///
    /// # Safety contract (no `unsafe` required)
    ///
    /// `buf.len() == graph.n_nodes()` is asserted in debug builds.
    pub(crate) fn from_pool_buf(graph: Arc<Graph<F>>, buf: Vec<F>) -> Self {
        debug_assert_eq!(
            buf.len(),
            graph.n_nodes(),
            "GraphSignal::from_pool_buf: buf.len() != graph.n_nodes()"
        );
        Self { values: buf, graph }
    }

    /// Decompose into the underlying `(Vec<F>, Arc<Graph<F>>)` without cloning.
    ///
    /// Used by `StrangSplitGraph::apply_into` to return the pool buffer after use.
    pub(crate) fn into_pool_buf(self) -> (Vec<F>, Arc<Graph<F>>) {
        (self.values, self.graph)
    }

    /// Borrow the `Arc<Graph<F>>` (for R4 pool operations).
    #[must_use]
    pub(crate) fn graph_arc(&self) -> Arc<Graph<F>> {
        Arc::clone(&self.graph)
    }

    /// Borrow the value slice.
    ///
    /// Public to allow integration tests and bindings to read signal values.
    #[must_use]
    pub fn values(&self) -> &[F] {
        &self.values
    }

    /// Reference to the backing graph (Wave 2.1B composition + tests).
    #[must_use]
    pub fn graph(&self) -> &Graph<F> {
        &self.graph
    }

    /// In-place `dst += alpha * src_slice` without a `GraphSignal` wrapper on RHS.
    ///
    /// Used by [`crate::GraphHeatChernoff::apply_into`] after `L_G · f`.
    pub(crate) fn axpy_into_slice(&mut self, alpha: F, src: &[F]) {
        debug_assert_eq!(
            self.values.len(),
            src.len(),
            "axpy_into_slice: shape mismatch"
        );
        for (s, &x) in self.values.iter_mut().zip(src.iter()) {
            *s += alpha * x;
        }
    }
}

// ---------------------------------------------------------------------------
// State<F> impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> State<F> for GraphSignal<F> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn axpy_into(&mut self, alpha: F, src: &Self) {
        debug_assert_eq!(
            self.values.len(),
            src.values.len(),
            "axpy_into: shape mismatch"
        );
        for (s, &x) in self.values.iter_mut().zip(src.values.iter()) {
            *s += alpha * x;
        }
    }

    fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(
            self.values.len(),
            src.values.len(),
            "copy_from: shape mismatch"
        );
        self.values.copy_from_slice(&src.values);
    }

    fn zero_into(&mut self) {
        for v in &mut self.values {
            *v = F::zero();
        }
    }

    fn norm_sup(&self) -> F {
        self.values.iter().fold(F::zero(), |acc, &v| {
            let av = <F as Float>::abs(v);
            if av > acc {
                av
            } else {
                acc
            }
        })
    }

    fn scale_into(&mut self, k: F) {
        for v in &mut self.values {
            *v *= k;
        }
    }
}

// ---------------------------------------------------------------------------
// HilbertState<F> impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> HilbertState<F> for GraphSignal<F> {
    fn dot(&self, other: &Self) -> F {
        debug_assert_eq!(self.values.len(), other.values.len(), "dot: shape mismatch");
        self.values
            .iter()
            .zip(other.values.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
    }
    // `norm_sq` and `norm_l2` use default trait impls.
}

// ---------------------------------------------------------------------------
// CsrRowIter<'a, F> — GAT neighbour iterator
// ---------------------------------------------------------------------------

/// Zero-allocation neighbour iterator: walks one CSR row of a [`Graph<F>`].
///
/// Returned by [`GraphSignal::neighbours`] via the [`Discrete<F>`] GAT
/// `type Neighbours<'a> = CsrRowIter<'a, F>`. Holds borrowed slices + a cursor.
pub struct CsrRowIter<'a, F: SemiflowFloat> {
    col_idx: &'a [u32],
    vals: &'a [F],
    cursor: usize,
}

impl<F: SemiflowFloat> Iterator for CsrRowIter<'_, F> {
    type Item = (u32, F);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.col_idx.len() {
            let nb = self.col_idx[self.cursor];
            let w = self.vals[self.cursor];
            self.cursor += 1;
            Some((nb, w))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Discrete<F> impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> Discrete<F> for GraphSignal<F> {
    type Idx = u32;
    type Neighbours<'a>
        = CsrRowIter<'a, F>
    where
        Self: 'a;

    fn get(&self, idx: u32) -> F {
        self.values[idx as usize]
    }

    fn set(&mut self, idx: u32, val: F) {
        self.values[idx as usize] = val;
    }

    fn indices(&self) -> impl Iterator<Item = u32> + '_ {
        #[allow(clippy::cast_possible_truncation)]
        let n = self.values.len() as u32;
        0..n
    }

    fn neighbours(&self, idx: u32) -> CsrRowIter<'_, F> {
        let i = idx as usize;
        let lo = self.graph.row_ptr()[i];
        let hi = self.graph.row_ptr()[i + 1];
        CsrRowIter {
            col_idx: &self.graph.col_idx()[lo..hi],
            vals: &self.graph.vals()[lo..hi],
            cursor: 0,
        }
    }
}
