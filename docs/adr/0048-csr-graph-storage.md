# ADR-0048 — Hand-rolled CSR storage for `Graph<F>` and `Laplacian<F>`

- **Status**: PROPOSED
- **Date**: 2026-05-20
- **Wave**: v2.1 Wave A (Graph PDE Foundations)
- **Companion**: ADR-0047 (`GraphHeatChernoff` design), ADR-0049, ADR-0050.

## Decision

Both `Graph<F>` (adjacency + edge weights) and `Laplacian<F>` (assembled operator)
store their non-zero structure in **hand-rolled Compressed Sparse Row (CSR)**:

```rust
pub struct Graph<F: SemiflowFloat = f64> {
    /// Row pointers — `row_ptr[i..i+1]` gives the slice into `col_idx` / `vals`
    /// holding the edges out of node `i`. `row_ptr.len() == n_nodes + 1`.
    row_ptr: Vec<usize>,
    /// Column indices (neighbour node ids), sorted ascending within each row.
    col_idx: Vec<u32>,
    /// Edge weights `w(i, j)`. Same length as `col_idx`.
    vals:    Vec<F>,
}

pub struct Laplacian<F: SemiflowFloat = f64> {
    /// CSR row pointers — invariant: diagonal entry is the LAST entry of each row
    /// (so `vals[row_ptr[i+1] − 1]` is `L[i, i] = deg(i)` or `1` for normalized).
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    vals:    Vec<F>,
    /// Pre-computed Gershgorin spectral radius bound `ρ̄ ≥ ρ(L_G)`.
    spectral_radius_bound: F,
    /// Normalization choice — recorded for documentation / sanity checks only.
    kind: LaplacianKind,
}
```

Where `LaplacianKind` is `Combinatorial` (default, `L = D − W`) or
`SymNormalized` (`L_sym = I − D^{−½} W D^{−½}` per Chung 1997 §1.2).

**Invariants (post-assembly, frozen)**:

1. `row_ptr.len() == n_nodes + 1`; `row_ptr[0] == 0`; `row_ptr[n_nodes] ==
   col_idx.len() == vals.len()`.
2. `row_ptr` is non-strictly monotonic.
3. Within each row, `col_idx[row_ptr[i]..row_ptr[i+1]]` is **strictly sorted
   ascending** with **no duplicates**.
4. **No self-loops** in `Graph<F>` (combinatorial Laplacian); `Laplacian<F>` MAY
   contain a diagonal entry, which by invariant (5) is the LAST entry of each row.
5. `Laplacian<F>` row layout: for row `i`, all off-diagonal entries come first
   in ascending `col_idx` order, then the diagonal entry. This places `L[i, i]`
   at position `row_ptr[i+1] − 1` for O(1) degree access.
6. All `col_idx[k] < n_nodes`. All `vals` are finite (no NaN, no Inf).
7. **Immutable post-assembly** — no mutator methods on `Graph<F>` or
   `Laplacian<F>` after the builder returns.

## Alternatives rejected

| Option | Rejected because |
|--------|-----------------|
| **`sprs` crate** | Adds 1 direct dep; constitution caps `semiflow-core` at 3 direct deps (currently 2: `num-traits`, `libm`). `sprs` brings `ndarray` transitively. CSR is a 200-LoC implementation; not worth a dep. |
| **COO (Coordinate)** | Cache-unfriendly for `L_G · f` (random `col_idx` accesses per row). CSR groups all neighbours of node `i` into a contiguous slice — one cache line typically covers a degree-≤8 row. |
| **Adjacency list of `Vec<(u32, F)>`** | Pointer chase per node + double indirection. Allocates O(N) sub-vectors. CSR is two flat `Vec`s. |
| **`Vec<HashMap<u32, F>>`** | Hash overhead per edge access; non-deterministic iteration order breaks bit-equal regression gates (ADR-0026 invariant). |
| **Dense `Vec<F>` of size `N²`** | OK for `N ≤ 256` (oracle path, ADR-0050), but the production path targets sparse `N ∈ [10³, 10⁵]` where `N²` quadratic blow-up is unacceptable. |

## Edge weight typing

Edge weights are `F: SemiflowFloat` — uniform with the rest of the v0.9.0
generic-over-float work (ADR-0025, ADR-0026). Path graphs and cycle graphs
populate `vals` with `F::one()`; weighted graphs may populate arbitrary positive
finite values. **Negative weights are NOT supported in Wave 2.1A** — the
Gershgorin bound and Chernoff quasi-contractivity proof both assume `w_ij ≥ 0`.
The builder validates this and returns `SemiflowError::DomainViolation` on negative
or non-finite weights.

## Builder API

The builder is **not** a separate type — small fluent constructors return
`Result<Graph<F>, SemiflowError>`:

```rust
impl<F: SemiflowFloat> Graph<F> {
    /// Path graph `0 — 1 — 2 — … — (n − 1)` with unit edge weights.
    pub fn path(n: usize) -> Self;

    /// Cycle graph: path + edge `(n − 1) — 0`, all unit weights.
    pub fn cycle(n: usize) -> Self;

    /// Generic builder from an unordered iterator of `(u, v, w)` triples.
    ///
    /// Edges are de-duplicated (`(u, v)` and `(v, u)` collapse into a single
    /// undirected edge with summed weight only if explicitly passed twice).
    /// Self-loops `u == v` are REJECTED — combinatorial Laplacian convention.
    pub fn from_edges(
        n_nodes: usize,
        edges: impl IntoIterator<Item = (u32, u32, F)>,
    ) -> Result<Self, SemiflowError>;

    /// Erdős–Rényi `G(n, p)` with seeded RNG (deterministic for tests).
    ///
    /// Edge weights all = `F::one()`. Used by the G7 slope gate.
    pub fn erdos_renyi(n_nodes: usize, p: f64, seed: u64) -> Self;
}
```

For `erdos_renyi`, the seeded RNG is a tiny hand-rolled splitmix64 (≤ 20 LoC,
no new dep). Determinism is required for the G7 slope gate to be reproducible.

## Laplacian assembly

```rust
impl<F: SemiflowFloat> Laplacian<F> {
    /// Combinatorial Laplacian `L = D − W`.
    pub fn assemble_combinatorial(g: &Graph<F>) -> Self;

    /// Symmetric normalized Laplacian `L_sym = I − D^{−½} W D^{−½}` (Chung 1997).
    ///
    /// Isolated nodes (`deg = 0`) are folded as `L_sym[i, i] = 0` per Chung
    /// 1997 §1.2 convention.
    pub fn assemble_normalized(g: &Graph<F>) -> Self;

    /// Gershgorin spectral-radius upper bound:
    /// `ρ̄ = max_i Σ_j |L_G[i, j]|` — computed at assembly time, cached.
    pub fn spectral_radius_bound(&self) -> F;

    /// `dst ← L_G · src` (zero-alloc; `dst.len() == self.n_nodes()`).
    pub fn apply_into_slice(&self, src: &[F], dst: &mut [F]);
}
```

`apply_into_slice` is the hot path: one outer loop over rows, one inner loop over
the row's CSR slice. Fully predictable branches, no allocation. The loop body is
≤ 10 LoC (well under the 50-LoC function cap).

For combinatorial `L = D − W`, the Gershgorin bound simplifies to `ρ̄ = 2 ·
max_i deg(i)` (deterministic; never under-estimates `ρ`). For normalized, `ρ̄ =
2` (Chung 1997 §1.3, eigenvalues lie in `[0, 2]`).

## Memory budget

For a graph with `N` nodes and `E` edges (Erdős–Rényi `p = 0.15`, `N = 64` gives
`E ≈ 300`):

| Field                | Size (bytes, f64) |
|----------------------|-------------------|
| `row_ptr`            | `8 · (N + 1) ≈ 520` |
| `col_idx` (Laplacian)| `4 · (E + N) ≈ 1456` |
| `vals` (Laplacian)   | `8 · (E + N) ≈ 2912` |
| `spectral_radius_bound` | 8 |
| **Total**            | ~4.9 KB |

Cache-friendly: `apply_into_slice` touches `col_idx` (~1.4 KB) and `vals` (~2.9 KB)
once per call, plus `src` (~0.5 KB) and `dst` (~0.5 KB) — comfortably in L1 for
the slope-gate `N ≤ 400` test, in L2 for production `N ≤ 10⁵`.

## Acceptance criteria

1. **Invariants hold post-builder** — assembled via property test
   (`proptest = 1.4.0`) checking 1000 random graphs from `from_edges`.
2. **`Laplacian::apply_into_slice` is zero-alloc** — verified by an
   `allocation-counter::measure` test.
3. **Gershgorin bound is correct** — compared to the Jacobi eigendecomposition
   oracle on `N = 64` graphs: `ρ̄ ≥ max(λ_i) − 1e-12`.
4. **Determinism of `erdos_renyi(N, p, seed)`** — same seed produces byte-identical
   `Graph<F>` across runs. Verified by snapshot test.
5. **`unsafe_code = "deny"`** preserved — no raw pointer arithmetic in the CSR
   path; all `Vec` indexing is bounds-checked. Hot-path benchmarks (deferred to
   Wave 2.1B) may revisit if profiling shows bounds-check overhead.

## File budget

Single file `crates/semiflow-core/src/graph.rs`, target ≤ 450 LoC (under 500
cap). Contains `Graph<F>`, `Laplacian<F>`, `LaplacianKind` enum, all builder
methods, all assembly methods, and `apply_into_slice`. If LoC threatens the cap,
split off `Laplacian<F>` into `graph_laplacian.rs` (no carve-out requested).
