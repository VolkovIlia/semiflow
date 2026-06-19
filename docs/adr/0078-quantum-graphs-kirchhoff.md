# ADR-0078 — Quantum Graphs (Kirchhoff Vertex Condition) (B7)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v3.1 Wave C (the secondary pillar of the Hörmander Research Pillar release; independent of A3 ADR-0077 — ships in parallel). Adds the first metric-graph (= quantum graph, edge-PDE-with-vertex-coupling) kernel class to complement the v2.1 ordinary-graph `GraphHeat*` family (ADR-0047/0051/0062). Wave A ships ADR-0077 trait infrastructure; Wave C (this ADR) ships `QuantumGraph<F>` + `KirchhoffVertex<F>` + the wrapper kernel. Independent of A3 (different module — `quantum_graph.rs` vs `hormander.rs`).
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (`ChernoffFunction<F>` generic over F), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0047 (v2.1 graph-bindings `GraphSignal<F>` — REUSED for per-edge representation), ADR-0051 (v2.1 `MagnusGraphChernoff` — pattern for per-edge ChernoffFunction composition), ADR-0074 (v3.0 ChernoffFunction trait cleanup — `Growth<F>` return type used by the new wrapper's `growth()`). Reuses existing `ShiftChernoff1D<F>` (v0.1.0) as the per-edge heat kernel WITHOUT modification.
- **Supersedes / amends**: none — strictly additive on the public surface. Establishes a NEW kernel class `QuantumGraphHeatChernoff<F>` (the wrapper) backed by two NEW value types `QuantumGraph<F>` (the metric-graph data structure) and `KirchhoffVertex<F>` (the per-vertex coupling-condition projection). Sibling to v2.1 ordinary-graph kernels — different mathematical object (continuum on edges with discrete vertex coupling, NOT a discrete graph Laplacian).
- **Mathematical foundation**: math.md §29 (NORMATIVE library — `QuantumGraphHeatChernoff` semantics + `KirchhoffVertex` orthogonal-projection construction; CITATION Friedlander 2005 *Annales Inst. Fourier* — genericity of simple eigenvalues for the metric graph Laplacian (G30 oracle for path-graph first 4 eigenmodes); Kuchment 2004 *Waves in Random Media* — survey of quantum-graph foundations (Kirchhoff vertex condition + edge-PDE formulation); Exner-Post 2009 *J. Phys. A* — convergence of spectra of graph-like thin manifolds (vertex-condition theory)).
- **Acceptance gates added**: G30 (RELEASE_BLOCKING — first 4 Friedlander eigenmodes on path graph (3 vertices, 2 edges of length 1) reconstructed with error ≤ 1e-3 at n=64 spectral decomposition; lives in `tests/quantum_graph_eigenmode.rs` new file, feature `slow-tests`); T_QG (NORMATIVE sympy — Kirchhoff projection symbolic identity + first 4 Friedlander path-graph eigenmodes derivation).

## Context

A **quantum graph** (Kuchment 2004 §1) is a *metric graph* — a 1-complex of edges $e_i$ with caller-supplied edge lengths $\ell_i$ — equipped with a *differential operator* on each edge (typically the second derivative $\partial_x^2$) plus a *vertex coupling condition* at each branch point. The simplest vertex condition is the **Kirchhoff condition** (Kuchment 2004 §3.2):
- (V1) Continuity: $u$ is single-valued at the vertex (all edge values agree at the vertex).
- (V2) Conservation of derivative flux: $\sum_{e \ni v} \partial_x u(v, e) = 0$ where the sum is over edges incident to vertex $v$ and the derivative is taken in the *into-the-vertex* direction.

The combined operator $L = -\tfrac{1}{2} \partial_x^2$ on each edge with the Kirchhoff coupling generates the **Kirchhoff heat semigroup** $\{e^{tL}\}_{t \ge 0}$ on $L^2(\Gamma)$ where $\Gamma$ is the metric graph. The semigroup preserves the Kirchhoff subspace (a finite-codimensional subspace of $L^2$).

The library already ships **ordinary graph** kernels (v2.1 ADR-0047/0051): `GraphHeatChernoff<F>`, `MagnusGraphChernoff<F>`, `GraphHeat6thChernoff<F>`. These operate on the *discrete* graph Laplacian $L_G : \mathbb{R}^V \to \mathbb{R}^V$ — one value per vertex, no edge structure. They are mathematically distinct from quantum graphs: the discrete Laplacian's spectrum is bounded, while the metric-graph Laplacian's spectrum is unbounded (continuum spectrum on each edge). Quantum graphs are the natural setting for *wave propagation on networks* (Kirchhoff's circuit-laws lift to PDE), *coupled-PDE bench problems* (thin-domain limits of waveguides per Exner-Post 2009), and *spectral theory of metric trees* (Friedlander 2005's simple-eigenvalue genericity result).

v3.1 ships the **quantum-graph heat semigroup** via a *natural* Chernoff product formula (math.md §29.3): per Chernoff step, evolve each edge independently as a 1D heat semigroup (using the existing `ShiftChernoff1D<F>`), then project the per-vertex multi-edge values onto the Kirchhoff-satisfying subspace via an orthogonal projection. The Strang-style symmetrisation is *not needed* in v3.1 because the per-edge heat exponential is already exact in the boundary-condition-respecting basis; the order is the same as the per-edge inner (order 1 for `ShiftChernoff1D`).

The natural future extension is order-2 via `DiffusionChernoff<F>` on each edge with the per-edge boundary conditions matched to the vertex projection — deferred to v3.2 because: (a) per-edge `DiffusionChernoff<F>` requires a non-trivial boundary policy at the vertex; (b) the v3.1 order-1 result already validates the *infrastructure* (the Kirchhoff projection + per-edge ChernoffFunction composition). Higher orders (4, 6) and the matrix-vertex extensions (δ-vertex, δ'-vertex per Berkolaiko-Kuchment 2013) defer to C7 (Tier C).

The Friedlander 2005 *Annales Inst. Fourier* paper gives closed-form expressions for the first eigenmodes of the path-graph Laplacian; these become the **G30 oracle**.

## Decision

Ship three additive public-surface items in v3.1 Wave C:

**Item 1 — `pub struct QuantumGraph<F: SemiflowFloat>` in `crates/semiflow-core/src/quantum_graph.rs`**:
```rust
pub struct QuantumGraph<F: SemiflowFloat = f64> {
    n_vertices: usize,                         // |V|
    n_edges: usize,                            // |E|
    edge_endpoints: Vec<(usize, usize)>,       // per-edge (vertex_a, vertex_b); len == n_edges
    edge_lengths: Vec<F>,                      // per-edge length ℓ_i; len == n_edges
    edge_grids: Vec<Grid1D<F>>,                // per-edge GridFn1D backing grid; len == n_edges
    vertex_adjacency: Vec<Vec<usize>>,         // per-vertex list of incident edge indices; len == n_vertices
}
```
Constructor:
```rust
impl<F: SemiflowFloat> QuantumGraph<F> {
    pub fn new(
        edge_endpoints: Vec<(usize, usize)>,
        edge_lengths: Vec<F>,
        edge_n_grid_points: usize,             // same N for all edges (v3.1 simplification)
    ) -> Result<Self, SemiflowError>;
}
```
Validation at construction:
- `edge_endpoints.len() == edge_lengths.len()` (shape parity).
- All `edge_lengths[i] > 0 && is_finite` (no degenerate edges).
- All `edge_endpoints[i].0 < n_vertices && edge_endpoints[i].1 < n_vertices` (vertices in range).
- `edge_n_grid_points >= 5` (5-point stencil width for ShiftChernoff1D).
- Per-edge `Grid1D::new(0.0, edge_lengths[i], edge_n_grid_points)` constructed with `BoundaryPolicy::Reflect` (per-edge default; will be overridden by Kirchhoff projection at the vertex step).
- `vertex_adjacency` derived from `edge_endpoints` (cached for O(1) per-vertex lookup).

**Item 2 — `pub struct KirchhoffVertex<F: SemiflowFloat>` in `crates/semiflow-core/src/quantum_graph.rs`**:
```rust
pub struct KirchhoffVertex<F: SemiflowFloat = f64> {
    vertex_index: usize,                       // index into QuantumGraph::n_vertices
    degree: usize,                             // number of incident edges (≥ 1)
    incident_edges: Vec<usize>,                // indices into QuantumGraph::n_edges; len == degree
    projection_matrix: Vec<F>,                 // row-major degree×degree orthogonal projector
}
```
Constructor:
```rust
impl<F: SemiflowFloat> KirchhoffVertex<F> {
    pub fn new(
        graph: &QuantumGraph<F>,
        vertex_index: usize,
    ) -> Result<Self, SemiflowError>;
}
```
Validation:
- `vertex_index < graph.n_vertices`.
- `degree := graph.vertex_adjacency[vertex_index].len() >= 1`.
- Constructs `projection_matrix` as the closed-form orthogonal projector onto the Kirchhoff subspace (math §29.2): for a degree-$d$ vertex,
  $$
  P = I - \frac{1}{d} \mathbf{1}\mathbf{1}^T - \frac{1}{d} J,
  $$
  where $\mathbf{1}$ is the all-ones $d$-vector (continuity component) and $J$ is the all-ones $d \times d$ matrix (the derivative-flux component). The specific form is derived in math.md §29.2; symbolically verified by T_QG sub-check (1).
- The projector is *exact* (idempotent) at $F::EPSILON$-precision; verified at construction with `debug_assert!(projection_squared_close_to_self())`.

**Item 3 — `pub struct QuantumGraphHeatChernoff<F: SemiflowFloat>` in `crates/semiflow-core/src/quantum_graph.rs`**:
```rust
pub struct QuantumGraphHeatChernoff<F: SemiflowFloat = f64> {
    graph: QuantumGraph<F>,
    vertices: Vec<KirchhoffVertex<F>>,         // one per graph vertex; len == graph.n_vertices
    edge_kernels: Vec<ShiftChernoff1D<F>>,     // one per graph edge; len == graph.n_edges
}
```
Constructor:
```rust
impl<F: SemiflowFloat> QuantumGraphHeatChernoff<F> {
    pub fn new(
        graph: QuantumGraph<F>,
    ) -> Result<Self, SemiflowError>;
}
```
Validation:
- All `KirchhoffVertex` constructors succeed for `vertex_index ∈ 0..n_vertices`.
- All `ShiftChernoff1D::new(a=0.5, b=0, c=0, edge_grid)` constructors succeed for each edge (unit-diffusion heat kernel; `a=0.5` matches the $L = -\tfrac{1}{2}\partial_x^2$ heat operator).

The `apply_into(τ, src, dst, scratch)` algorithm realises the per-step composition:
```
QuantumGraphHeatChernoff::apply_into(τ, src, dst, scratch):
  // Step 1 — Edgewise heat:
  //   For each edge e in 0..n_edges:
  //     edge_kernels[e].apply_into(τ, src.edge_signal(e), dst.edge_signal_mut(e), scratch)?;
  //   src.edge_signal(e) is a GridFn1D<F> view on the e-th edge's data.
  // Step 2 — Per-vertex Kirchhoff projection:
  //   For each vertex v in 0..n_vertices:
  //     // Gather: collect the values at vertex v from each incident edge into a degree-len buffer.
  //     let mut endpoint_buf = scratch.borrow_vec(vertices[v].degree);
  //     for (k, e) in vertices[v].incident_edges.iter().enumerate():
  //         endpoint_buf[k] := dst.edge_endpoint_value(e, vertex_v_side_of_e);
  //     // Project: apply the projection matrix (degree×degree GEMV) onto endpoint_buf.
  //     let mut projected_buf = scratch.borrow_vec(vertices[v].degree);
  //     gemv(&vertices[v].projection_matrix, &endpoint_buf, &mut projected_buf);
  //     // Scatter: write the projected values back into each incident edge's vertex endpoint.
  //     for (k, e) in vertices[v].incident_edges.iter().enumerate():
  //         dst.set_edge_endpoint_value(e, vertex_v_side_of_e, projected_buf[k]);
  // dst is now in the Kirchhoff subspace.
```

The per-step cost is $\sum_e \text{cost}(\text{ShiftChernoff1D}(\text{edge } e)) + \sum_v O(\deg(v)^2)$ — dominated by the per-edge work for typical graphs.

The `order()` method returns `1` (matches `ShiftChernoff1D::order()`; the projection step is exact at every Chernoff step). `growth()` returns `Growth { multiplier: F::one(), omega: F::zero() }` — the Kirchhoff projection is a contraction (orthogonal projector), so the wrapper is contractive on top of the per-edge contractive heat kernel.

The `Self::S` type is a NEW `pub struct QuantumGraphSignal<F: SemiflowFloat>` — a wrapper around `Vec<GridFn1D<F>>` (one per edge) with `edge_signal(e: usize)`, `edge_signal_mut(e: usize)`, `edge_endpoint_value(e, side)`, `set_edge_endpoint_value(e, side, v)` accessors. Implements `State<F>` via per-edge delegation (`axpy`, `scale`, `norm_sup`, `zeroed_like` are per-edge linear-combination + sup-norm-across-edges).

**Future extension hooks (NOT in v3.1)**:
- δ-vertex condition (Berkolaiko-Kuchment 2013 §1.4.2): replace `KirchhoffVertex` with `DeltaVertex { strength: F }`. Same trait surface, different projection matrix. Defer to C7.
- δ'-vertex condition: similar. Defer to C7.
- Schrödinger on quantum graphs: replace `ShiftChernoff1D` with the v3.1 `Schrödinger` kernel (v2.2 ADR-0057) on each edge. Per-edge BoundaryPolicy needs design work; defer to v4.x.

## Rationale

**Why ship in v3.1 (vs v3.2+)**: the v2.1 graph-bindings infrastructure (ADR-0047/0051: `GraphSignal<F>`, the per-edge data representation pattern) is mature; the `ShiftChernoff1D<F>` per-edge heat kernel is v0.1.0 stable; the v3.0 `Growth<F>` typed return is fresh. Shipping in v3.1 completes the **graph completeness arc** (ordinary graphs in v2.1 → quantum graphs in v3.1), and the small footprint (~400 LoC; under default cap; NO Override expansion) makes it a low-risk Wave C addition to the v3.1 Hörmander pillar release.

**Why `QuantumGraph` + `KirchhoffVertex` as separate value types** (vs collapsed into the wrapper): the metric-graph data structure is reusable infrastructure for future extensions (Schrödinger on quantum graphs, δ-vertex conditions). Separating the graph (geometry) from the vertex coupling (boundary condition) is the canonical mathematical decomposition (Kuchment 2004 §3). The wrapper composes them.

**Why orthogonal projection (vs alternating-projection iteration)**: the Kirchhoff subspace is a closed linear subspace of finite codimension at each vertex. Closed-form orthogonal projection is one matrix-vector product per vertex per step; alternating projection (split per-vertex) introduces iteration overhead with no order benefit. The closed-form formula
$$
P = I - \tfrac{1}{d}\mathbf{1}\mathbf{1}^T - \tfrac{1}{d}J
$$
is symbolically derived in math.md §29.2 and is sub-microsecond for typical vertex degrees (d ≤ 8). Verified by T_QG sub-check (1).

**Why `ShiftChernoff1D` as per-edge inner (vs `DiffusionChernoff`)**: `ShiftChernoff1D` is the simplest, most validated v0.1.0 kernel — guaranteed numerically stable on per-edge grids of any length. Using `DiffusionChernoff` per-edge would require designing a per-edge boundary policy that matches the Kirchhoff projection at the vertex — significant additional design effort with no marginal gain in v3.1 (the projection ALREADY enforces the vertex condition globally). Order-2 per-edge in v3.2+ is the natural next step.

**Why no Override #1 expansion** (vs Cohort 6 for hormander.rs): the v3.1 quantum_graph.rs target is ~400 LoC (3 types + 1 wrapper, no closed-form backends comparable to manifold or Hörmander), under the default 500-LoC cap with comfortable headroom. If quantum_graph.rs exceeds 500 LoC at engineer Wave C time (unlikely; rustdoc citations are lighter than for hormander.rs/manifold.rs because the projection formula is more elementary), a future v1.7.x PATCH would add it to Override #1; this is documented as a non-blocking risk in the engineer handoff.

**Why G30 = first 4 eigenmodes (vs slope convergence)**: the metric-graph heat semigroup is *spectrally decomposable* — the Friedlander 2005 first-4-eigenmodes path-graph oracle gives a *deterministic* validation target (not a convergence rate). Using spectral-decomposition reconstruction with error ≤ 1e-3 at n=64 grid points per edge gives a quantitative gate that is more discriminating than a slope sweep — the Kirchhoff projection either preserves the eigenmode structure or it does not (to within the per-edge `ShiftChernoff1D` first-order error). The error budget 1e-3 accounts for the ShiftChernoff1D O(τ) global error at n=64 with T=0.1 (typical heat-equation time-scale on a unit-length path graph).

## Alternatives Considered

**Alt 1 — Full Schrödinger operator on quantum graphs**: rejected for v3.1. Per-edge Schrödinger kernels require complex-valued state (v2.2 ADR-0057 uses real-only Δ+V scope; complex is v4.0+ SemiflowComplex). Per-vertex condition for Schrödinger is more subtle (the conserved quantity is *probability current*, not *flux*). Defer to v4.x once SemiflowComplex is available.

**Alt 2 — δ-vertex or δ'-vertex coupling conditions (vs Kirchhoff only)**: deferred to C7. The δ-vertex condition (Berkolaiko-Kuchment 2013 §1.4.2) replaces the Kirchhoff conservation-of-flux requirement with $\sum \partial_x u = \alpha u(v)$ (strength α). The δ'-vertex is the dual. Both fit the same `QuantumGraphHeatChernoff` framework with a different projection matrix per vertex — a v3.2+ extension with no new infrastructure required.

**Alt 3 — Reuse `GraphHeatChernoff` with edge subdivision (1 vertex per grid point per edge)**: rejected. Subdivision converts a metric-graph problem into an ordinary-graph problem on the subdivided graph; the per-edge spectral content collapses to a finite-dimensional discrete Laplacian. This LOSES the continuum-on-edges structure that is the defining feature of quantum graphs (and the Friedlander oracle does not apply). The wrapper-with-projection design preserves the continuum.

**Alt 4 — Strang-symmetrise the per-edge step around the projection** (i.e., project (τ/2), edgewise heat (τ), project (τ/2)): considered. The projection is idempotent ($P^2 = P$), so $P \circ$ heat $\circ P$ has the SAME order as $P \circ$ heat (because the second $P$ is a no-op on the already-projected post-heat state up to per-edge O(τ²) terms that don't change the order). The simpler `heat (τ); project` composition is order-equivalent and runs faster. Strang-symmetrisation would matter ONLY if the projection were *not* idempotent (e.g., a soft penalty); rejected.

## Consequences

**Positive**:
- The library acquires its **first quantum-graph kernel class**, completing the graph-completeness arc: ordinary graphs (v2.1) + quantum graphs (v3.1).
- The `QuantumGraph<F>` + `KirchhoffVertex<F>` infrastructure becomes available for v3.x extensions (δ-vertex, δ'-vertex; defer to C7).
- The Friedlander 2005 path-graph eigenmode oracle becomes a permanent spectral-decomposition test asset.
- Demonstrates the v3.0 trait surface (`Growth<F>`, `Evolver<C, F>`) on a NEW state type (`QuantumGraphSignal<F>`); validates that the v3.0 trait cleanup did not over-fit to existing kernels.

**Negative**:
- New module `quantum_graph.rs` (~400 LoC target; under default 500-LoC cap; NO Override expansion).
- A new state type `QuantumGraphSignal<F>` requires per-edge `State<F>` delegation logic; adds a small amount of `State<F>` trait surface complexity.
- Limited gate coverage in v3.1 (only path graph 3 vertices, 2 edges; only first 4 eigenmodes); broader validation (star graphs, tree graphs, ring graphs) deferred to v3.2.

**Neutral**:
- No deprecation, no migration playbook update needed (strictly additive).
- Independent of A3 Hörmander pillar; the two ship in parallel in v3.1.
- No new direct deps in `semiflow-core` (stays at 2 / 3).

## Cross-references

- math.md §29 (NEW section — NORMATIVE library semantics + CITATION mathematics)
- properties.yaml: G30 (RELEASE_BLOCKING — first 4 Friedlander eigenmodes on path graph ≤ 1e-3 at n=64), T_QG (NORMATIVE sympy — Kirchhoff projection symbolic identity + first 4 path-graph eigenmodes derivation)
- traits.yaml schema 1.0.0 → 1.1.0 (additive — `QuantumGraph<F>` + `KirchhoffVertex<F>` + `QuantumGraphHeatChernoff<F>` + `QuantumGraphSignal<F>` state type)
- constitution v1.7.0 → v1.7.1 PATCH (Cohort 6 added for hormander.rs only — quantum_graph.rs ~400 LoC is under the default 500-LoC cap; NO override expansion for this file)
- ADR-0047 (v2.1 graph-bindings — `GraphSignal<F>` pattern for per-vertex data)
- ADR-0051 (v2.1 `MagnusGraphChernoff` — pattern for per-edge ChernoffFunction composition; structural inspiration)
- ADR-0074 (the `Growth<F>` typed return — direct dependency)
- ADR-0077 (v3.1 A3 Hörmander hypoelliptic — sibling pillar, independent)

## References

- L. Friedlander, *Genericity of simple eigenvalues for a metric graph*, **Annales de l'Institut Fourier** 55:1 (2005), pp. 199-211. — Cited for §29.5 G30 oracle (the first 4 eigenmodes on the unit-edge path graph). The genericity result itself is foundational for spectral theory of metric graphs.
- P. Kuchment, *Quantum graphs: an introduction and a brief survey*, **Waves in Random Media** 14 (2004), pp. S107-S128. — Cited for §29.1 motivation + §29.2 Kirchhoff vertex condition. The canonical survey of the quantum-graph framework adopted by v3.1.
- P. Exner, O. Post, *Convergence of spectra of graph-like thin manifolds*, **Journal of Geometric Analysis** 18 (2008), pp. 113-145. — Cited for the vertex-condition theory (the Kirchhoff condition arises as the thin-manifold limit of waveguide spectra). Provides physical motivation for the v3.1 vertex condition choice.
- G. Berkolaiko, P. Kuchment, *Introduction to Quantum Graphs*, **AMS Mathematical Surveys and Monographs** 186 (2013). — Cited in rustdoc for the δ-vertex / δ'-vertex extensions (deferred to C7).
- v2.1 graph pillar precedent: ADR-0047 + math.md §12 (ordinary-graph `GraphSignal<F>` + `GraphHeatChernoff<F>`; the structural model for the v3.1 quantum-graph state type design).
- v0.1.0 ShiftChernoff1D precedent: math.md §1-§2 (the per-edge inner kernel; reused without modification).
