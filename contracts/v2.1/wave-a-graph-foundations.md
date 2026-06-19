# Wave 2.1A Contract — Graph PDE Foundations

**Status**: NORMATIVE — engineer implements verbatim against this contract.
**ADRs**: ADR-0047 (`GraphHeatChernoff` design) · ADR-0048 (CSR storage) · ADR-0049 (math.md §12 introduction) · ADR-0050 (test-only Jacobi oracle)
**Scope**: `semiflow-core` v2.1 Wave A — first-class graph PDE foundations.
**Author**: ai-solutions-architect · **Date**: 2026-05-20 · **Reviewers**: reviewer-suckless, agentic-engineer

Wave 2.1A ships ONLY the order-1 leading Chernoff function `S(τ) f = f − τ L_G f`.
ζ-A const-`a` τ²-correction, Strang splittings, and Magnus K=4 graph variants are
DEFERRED to Wave 2.1B / Wave 2.1C (per AskUserQuestion #1).

---

## §1 — `Graph<F>` API (NORMATIVE)

### 1.1 Type definition

```rust
//! crates/semiflow-core/src/graph.rs

use alloc::vec::Vec;
use alloc::sync::Arc;

use crate::{error::SemiflowError, float::SemiflowFloat};

/// Frozen sparse weighted graph in CSR layout.
///
/// **Immutable post-construction** — no mutator methods. See ADR-0048 §"Invariants".
pub struct Graph<F: SemiflowFloat = f64> {
    n_nodes: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    vals:    Vec<F>,
}

impl<F: SemiflowFloat> Graph<F> {
    /// `n_nodes` accessor. O(1).
    pub fn n_nodes(&self) -> usize { self.n_nodes }

    /// `n_edges` accessor (undirected; symmetric pair stored once internally
    /// for cache, exposed as `col_idx.len()` which is `2E` — see §1.4).
    pub fn n_directed_edges(&self) -> usize { self.col_idx.len() }

    /// Borrow CSR slices (engine-internal use; not for downstream rebinding).
    pub(crate) fn row_ptr(&self) -> &[usize] { &self.row_ptr }
    pub(crate) fn col_idx(&self) -> &[u32]   { &self.col_idx }
    pub(crate) fn vals(&self)    -> &[F]     { &self.vals }
}
```

### 1.2 Builders

```rust
impl<F: SemiflowFloat> Graph<F> {
    /// Path graph `0 — 1 — 2 — … — (n − 1)` with unit edge weights.
    ///
    /// Panics if `n == 0`.
    pub fn path(n: usize) -> Self;

    /// Cycle graph: path plus edge `(n − 1) — 0`, all unit weights.
    ///
    /// Panics if `n < 3` (cycle on 1 or 2 nodes is degenerate).
    pub fn cycle(n: usize) -> Self;

    /// Build from an arbitrary iterator of undirected edges `(u, v, w)`.
    ///
    /// # Errors
    /// - `SemiflowError::DomainViolation` on `u == v` (self-loop), `u/v >= n_nodes`,
    ///   `w <= 0` or non-finite, or duplicate edges.
    pub fn from_edges(
        n_nodes: usize,
        edges: impl IntoIterator<Item = (u32, u32, F)>,
    ) -> Result<Self, SemiflowError>;

    /// Erdős–Rényi `G(n_nodes, p)` with seeded splitmix64 RNG.
    /// All edge weights = `F::one()`.
    pub fn erdos_renyi(n_nodes: usize, p: f64, seed: u64) -> Self;
}
```

### 1.3 Invariants (verified by `debug_assert!` in builders, by `assert_invariants` in tests)

- I1: `row_ptr.len() == n_nodes + 1`; `row_ptr[0] == 0`; `row_ptr[n_nodes] == col_idx.len() == vals.len()`.
- I2: `row_ptr` is non-strictly monotonic.
- I3: For every row `i`, `col_idx[row_ptr[i]..row_ptr[i+1]]` is strictly sorted ascending; no duplicates.
- I4: No self-loops: for every `k`, no `i` with `col_idx[k] == i AND row_ptr[i] <= k < row_ptr[i+1]`.
- I5: All `vals[k]` are finite (no NaN, no Inf); all `vals[k] > F::zero()`.
- I6: All `col_idx[k] < n_nodes`.
- I7: Symmetric storage: edge `(u, v)` appears as `(u → v)` in row `u` AND `(v → u)` in row `v`, with `vals` matching to bit-equal precision.

### 1.4 Storage convention

Undirected edges are stored TWICE — once per endpoint — so that `Laplacian::apply_into_slice`
needs only a single forward sweep. This is the standard symmetric-CSR convention
(Chung 1997 §1.2; cf. `nalgebra::CsCholesky` documentation). Memory cost is `2× E`
edges; for the G7 gate `N = 64`, `p = 0.15` → `E ≈ 300` → `col_idx + vals = 2 · 300 ·
(4 + 8) = 7.2 KB`, well within L1.

---

## §2 — `GraphSignal<F>` API (NORMATIVE)

### 2.1 Type definition

```rust
//! crates/semiflow-core/src/graph_signal.rs

use crate::{float::SemiflowFloat, state::{State, HilbertState, Discrete}, graph::Graph};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Dense `Vec<F>`-backed function on a graph's node set.
///
/// One `Arc<Graph<F>>` per signal so neighbour iteration via `Discrete<F>` can
/// reach the graph's CSR. Multiple signals MAY share the same `Arc` (cheap
/// clone).
#[derive(Clone)]
pub struct GraphSignal<F: SemiflowFloat = f64> {
    pub(crate) values: Vec<F>,
    graph: Arc<Graph<F>>,
}

impl<F: SemiflowFloat> GraphSignal<F> {
    /// Construct from a closure `init(node) -> F`.
    pub fn from_fn(graph: Arc<Graph<F>>, init: impl Fn(u32) -> F) -> Self;

    /// Zero signal on `graph`.
    pub fn zeros(graph: Arc<Graph<F>>) -> Self;

    /// Borrow the value slice (engine-internal).
    pub(crate) fn values(&self) -> &[F] { &self.values }

    /// Mutable slice view (engine-internal, used by GraphHeatChernoff).
    pub(crate) fn values_mut(&mut self) -> &mut [F] { &mut self.values }

    /// In-place `dst += alpha * src_slice` without a `GraphSignal` wrapper on
    /// the RHS. Used by `GraphHeatChernoff::apply_into` after `L_G · f`.
    pub(crate) fn axpy_into_slice(&mut self, alpha: F, src: &[F]) {
        debug_assert_eq!(self.values.len(), src.len(), "axpy_into_slice: shape mismatch");
        for (s, &x) in self.values.iter_mut().zip(src.iter()) {
            *s += alpha * x;
        }
    }

    /// Reference to the backing graph (Wave 2.1B composition + tests).
    pub fn graph(&self) -> &Graph<F> { &self.graph }
}
```

### 2.2 `State<F>` impl

```rust
impl<F: SemiflowFloat> State<F> for GraphSignal<F> {
    fn len(&self) -> usize { self.values.len() }
    fn axpy_into(&mut self, alpha: F, src: &Self) {
        debug_assert_eq!(self.values.len(), src.values.len(), "axpy_into: shape mismatch");
        for (s, &x) in self.values.iter_mut().zip(src.values.iter()) { *s += alpha * x; }
    }
    fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.values.len(), src.values.len(), "copy_from: shape mismatch");
        self.values.copy_from_slice(&src.values);
    }
    fn zero_into(&mut self) {
        for v in &mut self.values { *v = F::zero(); }
    }
    fn norm_sup(&self) -> F {
        self.values.iter().fold(F::zero(), |acc, &v| {
            let av = <F as num_traits::Float>::abs(v);
            if av > acc { av } else { acc }
        })
    }
    fn scale_into(&mut self, k: F) {
        for v in &mut self.values { *v *= k; }
    }
}
```

### 2.3 `HilbertState<F>` impl

```rust
impl<F: SemiflowFloat> HilbertState<F> for GraphSignal<F> {
    fn dot(&self, other: &Self) -> F {
        debug_assert_eq!(self.values.len(), other.values.len(), "dot: shape mismatch");
        self.values.iter().zip(other.values.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
    }
    // norm_sq and norm_l2 inherit default impls from trait.
}
```

### 2.4 `Discrete<F>` impl with GAT `CsrRowIter<'a, F>`

```rust
/// GAT neighbour iterator: walks one row of the backing `Graph<F>` CSR.
/// Zero-allocation: holds `&'a [u32]` + `&'a [F]` slices + a single `usize` cursor.
pub struct CsrRowIter<'a, F: SemiflowFloat> {
    col_idx: &'a [u32],
    vals:    &'a [F],
    cursor:  usize,
}

impl<'a, F: SemiflowFloat> Iterator for CsrRowIter<'a, F> {
    type Item = (u32, F);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.col_idx.len() {
            let nb = self.col_idx[self.cursor];
            let w  = self.vals[self.cursor];
            self.cursor += 1;
            Some((nb, w))
        } else {
            None
        }
    }
}

impl<F: SemiflowFloat> Discrete<F> for GraphSignal<F> {
    type Idx = u32;
    type Neighbours<'a> = CsrRowIter<'a, F> where Self: 'a;

    fn get(&self, idx: u32) -> F { self.values[idx as usize] }
    fn set(&mut self, idx: u32, val: F) { self.values[idx as usize] = val; }

    fn indices(&self) -> impl Iterator<Item = u32> + '_ {
        #[allow(clippy::cast_possible_truncation)]
        let n = self.values.len() as u32;
        0..n
    }

    fn neighbours(&self, idx: u32) -> CsrRowIter<'_, F> {
        let i  = idx as usize;
        let lo = self.graph.row_ptr()[i];
        let hi = self.graph.row_ptr()[i + 1];
        CsrRowIter {
            col_idx: &self.graph.col_idx()[lo..hi],
            vals:    &self.graph.vals()[lo..hi],
            cursor:  0,
        }
    }
}
```

**GAT lifetime check (Phase 1 verified)**: matches `state.rs:180-182` exactly —
`type Neighbours<'a>: Iterator<Item = (Self::Idx, F)> where Self: 'a`. The
`CsrRowIter<'a, F>` returns `(u32, F)` ↔ `(Self::Idx, F) = (u32, F)`. ✓

---

## §3 — `Laplacian<F>` API (NORMATIVE)

### 3.1 Type definition

```rust
//! crates/semiflow-core/src/graph.rs (same file as Graph<F>)

/// Normalization kind for the assembled Laplacian. Tracked for documentation
/// only — `apply_into_slice` is identical for both (the CSR carries the values).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LaplacianKind { Combinatorial, SymNormalized }

/// Frozen sparse Laplacian operator in CSR layout. See ADR-0048.
pub struct Laplacian<F: SemiflowFloat = f64> {
    n_nodes: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    vals:    Vec<F>,
    spectral_radius_bound: F,
    kind: LaplacianKind,
}

impl<F: SemiflowFloat> Laplacian<F> {
    /// Combinatorial Laplacian `L = D − W` from a graph.
    pub fn assemble_combinatorial(g: &Graph<F>) -> Self;

    /// Symmetric normalized Laplacian `L_sym = I − D^{−½} W D^{−½}` (Chung 1997).
    /// Isolated nodes (`deg = 0`) get `L_sym[i, i] = 0` per Chung 1997 §1.2.
    pub fn assemble_normalized(g: &Graph<F>) -> Self;

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize { self.n_nodes }

    /// Gershgorin upper bound on `ρ(L_G)`. Cached at assembly time.
    pub fn spectral_radius_bound(&self) -> F { self.spectral_radius_bound }

    /// `dst ← L_G · src`. Zero-alloc. `dst.len() == src.len() == self.n_nodes`.
    pub fn apply_into_slice(&self, src: &[F], dst: &mut [F]);

    /// Normalization kind (documentation only).
    pub fn kind(&self) -> LaplacianKind { self.kind }
}
```

### 3.2 Assembly invariants

Beyond the §1.3 invariants inherited from `Graph<F>`, `Laplacian<F>` adds:

- L1: Diagonal entry `L[i, i]` is the LAST entry of each row's CSR slice (i.e. at
  index `row_ptr[i+1] − 1`). Off-diagonal entries are in ascending `col_idx`
  order, then the diagonal. This invariant accelerates degree access in
  Wave 2.1B and is part of the contract.
- L2: All off-diagonal `vals < F::zero()` (since `L[i,j] = −w(i,j)`); the
  diagonal `vals > F::zero()`.
- L3: `vals` are finite. NaN inputs from `Graph<F>` are impossible (graph
  invariant I5).
- L4: `spectral_radius_bound` computed once at assembly: `max_i Σ_j |L[i, j]|`.

### 3.3 `apply_into_slice` body (NORMATIVE pseudocode)

```rust
pub fn apply_into_slice(&self, src: &[F], dst: &mut [F]) {
    debug_assert_eq!(src.len(), self.n_nodes);
    debug_assert_eq!(dst.len(), self.n_nodes);
    for i in 0..self.n_nodes {
        let mut acc = F::zero();
        for k in self.row_ptr[i]..self.row_ptr[i + 1] {
            acc += self.vals[k] * src[self.col_idx[k] as usize];
        }
        dst[i] = acc;
    }
}
```

≤ 12 lines (well under 50-LoC function cap). No allocation. Bounds checks
remain — `unsafe_code = "deny"` per constitution. Profiling-driven `unsafe`
revisit deferred to Wave 2.1B if benchmark shows >10% bounds-check overhead.

### 3.4 Gershgorin bound

For combinatorial `L = D − W` with non-negative weights:
```
ρ̄ = max_i Σ_j |L[i, j]| = max_i ( deg(i) + Σ_{j ≠ i} w(i, j) ) = 2 · max_i deg(i)
```
For normalized `L_sym`: `ρ̄ = 2` (Chung 1997 §1.3 — eigenvalues lie in `[0, 2]`).

Engineer implements both as a single sum over CSR `|vals|` per row, which keeps
the assembler agnostic and naturally covers the weighted-graph case.

---

## §4 — `GraphHeatChernoff<F>` API (NORMATIVE)

### 4.1 Type definition

```rust
//! crates/semiflow-core/src/graph_heat.rs

use alloc::sync::Arc;
use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::SemiflowFloat,
    graph::Laplacian,
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

/// Order-1 leading Chernoff for the discrete heat semigroup `e^{−t L_G}`.
///
/// `S(τ) f = f − τ · L_G · f`. See ADR-0047.
pub struct GraphHeatChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
}

impl<F: SemiflowFloat> GraphHeatChernoff<F> {
    pub fn new(laplacian: Arc<Laplacian<F>>) -> Self { Self { laplacian } }
    pub fn from_owned(laplacian: Laplacian<F>) -> Self { Self { laplacian: Arc::new(laplacian) } }
    pub fn laplacian(&self) -> &Laplacian<F> { &self.laplacian }
}
```

### 4.2 `ChernoffFunction<F>` impl

```rust
impl<F: SemiflowFloat> ChernoffFunction<F> for GraphHeatChernoff<F> {
    type S = GraphSignal<F>;

    fn apply(&self, tau: F, f: &Self::S) -> Result<Self::S, SemiflowError>
    where Self::S: Clone
    {
        let mut dst = f.clone();
        let mut scratch = ScratchPool::<F>::new();
        self.apply_into(tau, f, &mut dst, &mut scratch)?;
        Ok(dst)
    }

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "GraphHeatChernoff: tau must be finite and >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let n = src.len();
        debug_assert_eq!(dst.len(), n, "GraphHeatChernoff: dst.len() must match src.len()");
        debug_assert_eq!(self.laplacian.n_nodes(), n, "Laplacian/state size mismatch");

        let mut lap = scratch.borrow_vec(n);                 // borrow N-length scratch
        self.laplacian.apply_into_slice(src.values(), &mut lap);   // lap ← L_G · src
        dst.copy_from(src);                                  // dst ← src
        dst.axpy_into_slice(-tau, &lap);                     // dst -= τ * lap
        Ok(())
    }

    fn order(&self) -> u32 { 1 }

    fn growth(&self) -> (f64, f64) {
        (1.0, self.laplacian.spectral_radius_bound().to_f64().unwrap_or(f64::INFINITY))
    }
}
```

Allocation accounting per `apply_into` call **in steady state** (after first
warmup call, when `ScratchPool` has at least one `N`-capacity buffer in its
free list): **0 bytes** — `borrow_vec` recycles, `copy_from` is `slice::copy_from_slice`,
`axpy_into_slice` is a single-pass mutation. Verified by AC-2 zero-alloc gate.

---

## §5 — Composition with `ChernoffSemigroup` (NORMATIVE)

**`chernoff.rs` is UNCHANGED.** The existing executor signature

```rust
impl<C, S> ChernoffSemigroup<C, S>
where C: ChernoffFunction<f64, S = S>,
      S: State<f64> + Clone,
```

(chernoff.rs:194-197) accepts `(GraphHeatChernoff<f64>, GraphSignal<f64>)`
directly because:

- `GraphHeatChernoff<f64>: ChernoffFunction<f64, S = GraphSignal<f64>>` ✓ (§4.2)
- `GraphSignal<f64>: State<f64>` ✓ (§2.2)
- `GraphSignal<f64>: Clone` ✓ (`#[derive(Clone)]` in §2.1)

For `F = f32`, callers use `ChernoffSemigroup::<GraphHeatChernoff<f32>,
GraphSignal<f32>>` once the existing chernoff.rs generalization (still bounded
to `f64` per ADR-0026) is extended. **Wave 2.1A does NOT alter chernoff.rs**;
the f32 G7 gate uses a small inline harness that manually loops `apply_into`
(mirroring `chernoff_signal_evolve` helper in `tests/generic_float_smoke.rs`,
~30 LoC). Generalizing `ChernoffSemigroup` to `F: SemiflowFloat` is a Wave 2.1B
task (Wave 2.1A explicitly out of scope to keep the regression set unchanged).

---

## §6 — `graph_oracle` API (test-only, NORMATIVE)

```rust
//! crates/semiflow-core/src/graph_oracle.rs
#![cfg(test)]

use crate::{float::SemiflowFloat, graph::Laplacian, graph_signal::GraphSignal};
use alloc::vec::Vec;

#[doc(hidden)]
pub(crate) struct EigDecomp<F: SemiflowFloat> {
    pub eigenvalues: Vec<F>,                  // sorted ascending
    pub eigenvectors_col_major: Vec<F>,       // length n²; column k is φ_k
    pub n: usize,
}

#[doc(hidden)]
pub(crate) fn jacobi_eig<F: SemiflowFloat>(lap: &Laplacian<F>) -> EigDecomp<F>;

#[doc(hidden)]
pub(crate) fn heat_oracle<F: SemiflowFloat>(
    decomp: &EigDecomp<F>,
    f0: &GraphSignal<F>,
    t: F,
) -> GraphSignal<F>;
```

Algorithm (symmetric Jacobi, off-diagonal Frobenius minimisation):

1. Initialise `A ← L_G` (dense `Vec<F>` of length `n²`, row-major).
2. Initialise `Q ← I_n` (dense, col-major; `Q[j, k]` stored at `j + k·n`).
3. Loop:
   a. Find `(p, q)` (with `p < q`) maximising `|A[p, q]|`.
   b. Terminate when `Σ_{i < j} A[i, j]² < tol² · ‖A‖_F²` (tol = `1e-12` f64,
      `1e-5` f32) OR after `200 · n` rotations.
   c. Compute rotation `(c, s)` from `A[p, p]`, `A[p, q]`, `A[q, q]`.
   d. Apply rotation: `A ← G^T A G`, `Q ← Q G`.
4. Extract eigenvalues from diagonal of `A`; eigenvectors are columns of `Q`.
5. Sort eigenvalues ascending; permute columns of `Q` accordingly.
6. Enforce deterministic sign: for each `k`, flip column `k` of `Q` if
   `Q[0, k] < 0`.

Total `O(n³)` work, `O(n²)` memory. Acceptable for `n ≤ 1024` (`assert!`). All
on the stack-free heap via `Vec<F>` — no `unsafe`.

`heat_oracle` is a 3-line application: basis change via `Q^T`, exponential of
diagonal, back-transform via `Q`. O(`n²`) work.

---

## §7 — Test plan

### 7.1 G7 slope gate — `crates/semiflow-core/tests/convergence_graph.rs`

```rust
const N_VALUES: [usize; 5] = [25, 50, 100, 200, 400];

#[test]
fn g7_graph_heat_convergence_slope_f64() {
    let g = Arc::new(Graph::<f64>::erdos_renyi(64, 0.15, 0xDEAD_BEEF));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));

    let decomp = jacobi_eig(&lap);
    let f0 = GraphSignal::from_fn(Arc::clone(&g),
        |i| ((i as f64 * 0.31).sin() + 1.0) * 0.5);
    let oracle = heat_oracle(&decomp, &f0, 0.5);

    let errs: Vec<f64> = N_VALUES.iter().map(|&n| {
        let semi = ChernoffSemigroup::new(GraphHeatChernoff::new(Arc::clone(&lap)), n).unwrap();
        let u_t = semi.evolve(0.5, &f0).unwrap();
        let mut diff = u_t.clone();
        diff.axpy_into(-1.0, &oracle);
        diff.norm_sup()
    }).collect();

    let slope = log_log_slope(&N_VALUES, &errs);
    assert!(slope <= -0.95, "G7 FAIL: slope {slope:.4} > -0.95");
}

#[test]
fn g7_graph_heat_convergence_slope_f32() {
    // Same as above, F = f32, threshold ≤ -0.90 per ADR-0046 precision policy.
}
```

`log_log_slope` is the same OLS helper copied from `tests/convergence_rate.rs`
(lines 36-51 of that file — exact reuse, no abstraction).

### 7.2 Zero-alloc steady-state gate — `tests/graph_apply_into_zero_alloc.rs`

```rust
use allocation_counter::measure;

#[test]
fn graph_heat_apply_into_zero_alloc_steady_state() {
    let g = Arc::new(Graph::<f64>::path(128));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));
    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.1).sin());
    let mut dst = f0.clone();
    let mut scratch = ScratchPool::<f64>::new();
    // Warmup — first call may allocate the scratch buffer.
    chernoff.apply_into(0.001, &f0, &mut dst, &mut scratch).unwrap();
    // Now measure: subsequent calls must NOT allocate.
    let info = measure(|| {
        for _ in 0..1000 {
            chernoff.apply_into(0.001, &f0, &mut dst, &mut scratch).unwrap();
        }
    });
    assert_eq!(info.count_total, 0, "expected 0 allocs, got {}", info.count_total);
    assert_eq!(info.bytes_total, 0, "expected 0 bytes, got {}", info.bytes_total);
}
```

Mirrors the Wave 1 `tests/apply_into_byte_equal.rs` allocation-counter template
(see `default_bridge_compat.rs` and `apply_into_byte_equal.rs` for the
in-repo allocation-counter pattern).

### 7.3 Oracle eigenmode parity — `tests/graph_heat_oracle.rs` (repurposed)

Pre-Wave-2.1A `tests/graph_heat_oracle.rs` is the spike-trait smoke test. The
engineer **deletes** the spike's `PathGraphFn` and replaces with:

```rust
#[test]
fn graph_heat_apply_into_eigenmode_parity_f64() {
    let n = 16;
    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let decomp = jacobi_eig(&lap);
    let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));

    // Pick eigenvector φ_3 (4th smallest eigenvalue).
    let k = 3;
    let lambda_k = decomp.eigenvalues[k];
    let mut phi = GraphSignal::zeros(Arc::clone(&g));
    for i in 0..n {
        phi.set(i as u32, decomp.eigenvectors_col_major[i + k * n]);
    }

    let tau = 0.01_f64;
    let mut dst = phi.clone();
    let mut scratch = ScratchPool::<f64>::new();
    chernoff.apply_into(tau, &phi, &mut dst, &mut scratch).unwrap();

    // Expected: (1 - τ λ_k) φ_k
    let mut expected = phi.clone();
    expected.scale_into(1.0 - tau * lambda_k);
    let mut diff = dst.clone();
    diff.axpy_into(-1.0, &expected);
    assert!(diff.norm_sup() < 1e-12, "f64 parity drift {}", diff.norm_sup());
}

#[test]
fn graph_heat_apply_into_eigenmode_parity_f32() {
    // Same with F = f32, threshold 1e-5.
}
```

### 7.4 Graph builder invariants — `tests/graph_invariants.rs`

`proptest` 1.4.0 (already in dev-deps):

```rust
proptest! {
    #[test]
    fn from_edges_preserves_invariants(
        n in 2usize..64,
        edges in prop::collection::vec((0u32..63, 0u32..63, 0.1f64..10.0), 0..200)
    ) {
        let valid: Vec<_> = edges.into_iter()
            .filter(|(u, v, w)| u != v && (*u as usize) < n && (*v as usize) < n && w.is_finite() && *w > 0.0)
            .collect();
        let g = Graph::<f64>::from_edges(n, valid)?;
        assert_graph_invariants(&g);  // I1..I7
        let lap = Laplacian::assemble_combinatorial(&g);
        assert_laplacian_invariants(&lap);  // L1..L4
    }
}
```

`assert_graph_invariants` / `assert_laplacian_invariants` live in a
`tests/common/mod.rs` helper (≤ 100 LoC, separate from the 500 file cap).

### 7.5 Spike retirement — `tests/spike_parity_removal.rs`

Not a separate test file. Acceptance criterion: after AC-3 (eigenmode parity)
passes, the engineer:

1. Removes `crates/remizov-graph-spike` from workspace `Cargo.toml`.
2. Deletes `crates/remizov-graph-spike/` directory.
3. Verifies `cargo run -p xtask -- test-fast` still passes.

### 7.6 v2.0 regression set re-pass

After Wave 2.1A lands, all of:

- `apply_into_byte_equal` (6 tests)
- `strang_inplace` / `STRANG3D_SERIAL_SCRATCH_BIT_EQUAL` (7 tests)
- `state_trait` / `path_graph_*` (10 tests — they STAY in
  `tests/graph_heat_oracle.rs`, just augmented per §7.3)
- `adaptive_classical_bit_equal` (4 tests)
- `cev_european_call` (2 tests)
- 18 NORMATIVE sympy gates
- 6 v2.0 slope gates (G1..G6 spread across `convergence_rate*.rs`)

MUST all still pass with no edits required to their bodies. Verified by
`cargo run -p xtask -- test-fast` and `test-flagship`.

---

## §8 — `contracts/semiflow-core.math.md` §12 content outline

(Detailed in ADR-0049 §"Sub-section layout"; reproduced here for engineer
convenience.)

| §     | Content (NORMATIVE = N, CITATION = C)                                 |
|-------|----------------------------------------------------------------------|
| §12   | Header + scope note (no theorems, citation + library choices only) |
| §12.1 | Setting: `G = (V, E, w)`, combinatorial Laplacian definition — N    |
| §12.2 | Pazy 1983 §1.3 Thm 1.3 + Engel-Nagel 2000 §III.5 Thm 5.2 verbatim — C |
| §12.3 | Hypothesis check for `S(τ) = I − τ L_G` — C + N                      |
| §12.4 | Gershgorin bound + ½ stability scaling — N (cites Varga 2000, Chung 1997) |
| §12.5 | CSR storage layout invariants — N (cross-refs ADR-0048)              |

Existing §10.9 renumbers to §13. Net diff ≈ +250 LoC math.md, −0 LoC for
renumber (just `## §10.9` → `## §13`). 3 ADR cross-refs to update
(`docs/adr/0022,0034,0019`).

---

## §9 — Risk table (top 5)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **R1: GAT `Neighbours<'a>` lifetime conflict with `&Self::S` in `apply_into`** | LOW | HIGH (compile fail) | Spike `PathNeighboursIter` proves the pattern compiles; `CsrRowIter<'a, F>` mirrors it exactly. Engineer compiles `tests/graph_heat_oracle.rs` first (smallest reproducer) before wiring `GraphHeatChernoff`. |
| **R2: CSR builder edge ordering / sorting** | MEDIUM | MEDIUM | I3 enforced by sorting each row's `col_idx + vals` pair-wise at builder finalisation. Property test §7.4 catches violations. Engineer uses `Vec::sort_unstable_by_key` (no new dep). |
| **R3: Jacobi eigendecomposition numerical drift on `N ≤ 1024`** | LOW | LOW (test-only) | Termination on `Σ off-diag² < 1e-24 · ‖L‖²` for f64; cap at `200 N` rotations. Path-graph closed-form check (§7.3 fallback path-graph eigenvalues `2(1 − cos(kπ/N))`) detects drift > 1e-12 immediately. |
| **R4: f32 stability margin for Gershgorin bound** | MEDIUM | MEDIUM | f32 G7 gate uses `tau` already a factor of 4 below the ½/ρ̄ envelope (engineer computes `tau_max = 0.5 / ρ̄`, then uses `t / n` with `n` chosen so `t/n ≤ tau_max / 4`). f32 slope threshold relaxed to ≤ −0.90 per ADR-0046. |
| **R5: Migration regression vs spike `PathGraphFn` smoke tests** | LOW | LOW | Acceptance criterion §7.3 explicitly migrates the spike's behavioural assertions (indices coverage, boundary Dirichlet zero, interior 2-neighbour, get/set roundtrip, norm_sup, state axioms) onto the production `GraphSignal<f64> + Graph::<f64>::path`. Spike deletion (§7.5) blocked until these all pass. |

---

## §10 — Spike retirement (NORMATIVE plan)

Pre-condition: AC-3 (`graph_heat_apply_into_eigenmode_parity_{f64,f32}`) green
locally and in CI.

Steps (engineer, single commit, but only AFTER all other Wave 2.1A work has
landed and the regression set re-passes):

1. Remove workspace member: edit `Cargo.toml` line `members = [...,
   "crates/remizov-graph-spike", ...]` → drop the entry. Confirm
   `cargo run -p xtask -- test-fast` still passes (spike-internal tests vanish;
   no production code depended on the spike).
2. `git rm -r crates/remizov-graph-spike/`.
3. Verify `cargo metadata --no-deps` lists 5 workspace members (was 6): `semiflow-core`,
   `semiflow-ffi`, `semiflow-py`, `semiflow-wasm`, `xtask`.
4. Update `ROADMAP.md` v2.1.0 entry: cross out "remizov-graph-spike (deletion)"
   line item.

Spike findings preserved by ADR-0047 §"Why one scratch borrow per step", §"Why
`Arc<Laplacian<F>>`", and the migrated `tests/graph_heat_oracle.rs`.

---

## §11 — LoC budget

| File | Target LoC | Cap | Carve-out? |
|------|-----------:|----:|:----------:|
| `crates/semiflow-core/src/graph.rs` (Graph + Laplacian + LaplacianKind) | 450 | 500 | NO |
| `crates/semiflow-core/src/graph_signal.rs` (GraphSignal + CsrRowIter + State/HilbertState/Discrete impls) | 280 | 500 | NO |
| `crates/semiflow-core/src/graph_heat.rs` (GraphHeatChernoff) | 130 | 500 | NO |
| `crates/semiflow-core/src/graph_oracle.rs` (#[cfg(test)] Jacobi + heat_oracle) | 270 | 500 | NO |
| `tests/convergence_graph.rs` (G7 gate, f64 + f32) | 100 | 500 | NO |
| `tests/graph_apply_into_zero_alloc.rs` | 70 | 500 | NO |
| `tests/graph_heat_oracle.rs` (repurposed) | 180 | 500 | NO |
| `tests/graph_invariants.rs` (proptest builder invariants) | 120 | 500 | NO |
| `contracts/semiflow-core.math.md` §12 | +250 | 4900 (existing 4617 + 250 + renumber) | NO |
| ADR 0047-0050 (this design wave) | 200 each | n/a | n/a |

**Total estimated new LoC**: ≈ 1300 LoC of Rust + 250 LoC math + 800 LoC ADRs.
All files stay below the 500 cap; no Override #1 expansion requested.

Function cap (50 LoC) check: `Laplacian::apply_into_slice` (~12 LoC),
`GraphHeatChernoff::apply_into` (~25 LoC), `jacobi_eig` rotation kernel
(broken into `eig_loop` ≤ 40 LoC + `rotation_apply` ≤ 30 LoC). All comfortably
within.

---

## §12 — Build / run path (unchanged)

The single build path stays:

```bash
cargo run -p xtask -- test-fast      # 5–10× faster than debug; covers Wave 2.1A green path
cargo run -p xtask -- test-full      # parallel + simd + slow-tests; covers G7 f64 + f32
cargo run -p xtask -- test-flagship  # ignored / heavy tests; not relevant for Wave 2.1A
```

No new xtask command, no new feature flag, no new env var. The new tests slot
into `cargo test --workspace` automatically.

---

## §13 — Engineer handoff checklist

Tick in order; do not start step N+1 until step N is green.

- [ ] Read ADR-0047, ADR-0048, ADR-0049, ADR-0050 in full.
- [ ] Read this contract in full.
- [ ] Run `mcp__gitnexus__context({name: "ChernoffFunction"})` and confirm trait
      surface unchanged (the engineer must NOT modify `chernoff.rs`).
- [ ] Implement `crates/semiflow-core/src/graph.rs` (Graph + builders + Laplacian
      + assemblers + `apply_into_slice`).
- [ ] Add `pub mod graph;` to `crates/semiflow-core/src/lib.rs`.
- [ ] Implement `crates/semiflow-core/src/graph_signal.rs` (GraphSignal + State +
      HilbertState + Discrete + GAT `CsrRowIter`).
- [ ] Add `pub mod graph_signal;` to `lib.rs`.
- [ ] Implement `crates/semiflow-core/src/graph_heat.rs` (GraphHeatChernoff).
- [ ] Add `pub mod graph_heat;` to `lib.rs`.
- [ ] Implement `crates/semiflow-core/src/graph_oracle.rs` (Jacobi +
      heat_oracle, behind `#[cfg(test)]`).
- [ ] Add `#[cfg(test)] pub(crate) mod graph_oracle;` to `lib.rs`.
- [ ] Write `tests/convergence_graph.rs` (G7 slope, f64 + f32).
- [ ] Write `tests/graph_apply_into_zero_alloc.rs`.
- [ ] Repurpose `tests/graph_heat_oracle.rs` (delete spike-style helpers; add
      eigenmode parity tests).
- [ ] Write `tests/graph_invariants.rs` (proptest).
- [ ] Append §12 to `contracts/semiflow-core.math.md`; renumber existing §10.9 →
      §13; update 3 ADR cross-refs.
- [ ] Run `cargo run -p xtask -- test-fast` — all 198 v2.0 tests + new Wave
      2.1A tests green.
- [ ] Run `cargo run -p xtask -- test-full` — G7 f64 + f32 green; slope
      printed to stdout.
- [ ] Delete `crates/remizov-graph-spike/` and update workspace `Cargo.toml`.
- [ ] Run `cargo run -p xtask -- test-fast` AGAIN — confirm no spike-dependent
      tests broke.
- [ ] Verify `git grep -nE 'remizov-graph-spike'` returns zero hits.
- [ ] Hand off to git-workflow for a single Wave-2.1A commit (Anchor
      delegates; do NOT commit yourself).
