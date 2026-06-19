# ADR-0052 — Variable-topology graph trajectories `GraphTraj<F>`

- **Status**: ACCEPTED (v2.2 Wave A — implementation shipped; G12 slope gate pending HW run)
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave A (graph time-dependence)
- **Authors**: ai-solutions-architect
- **Reviewers**: reviewer-suckless (pending), agentic-engineer (pending)
- **Depends on**: ADR-0047 (`GraphHeatChernoff`), ADR-0048 (CSR storage),
  ADR-0049 (math.md §12), ADR-0051 (Magnus K=4 graph)
- **Supersedes / amends**: nothing (additive); answers v2.1 "Out of scope"
  bullet "Variable topology" in `docs/adr/0051-graph-magnus.md` §"Out of scope".
- **Mathematical foundation**: math.md §14.1 (NORMATIVE — piecewise-constant
  topology with smooth edge-weight segments; CITATION: Chung 1997 §1.2 on
  weighted graph Laplacian variation).

## Context

`MagnusGraphHeatChernoff<F>` (v2.1 Wave C, ADR-0051) is restricted to
**fixed-topology** generators — the `row_ptr` / `col_idx` of every
sampled `L_G(t)` MUST be byte-identical to those of the topology graph
passed to its constructor (math.md §12.9 contract clause 2). Real
applications — adaptive network rewiring, mesh adaptation events,
catalytic-network dynamics — need topology that changes at discrete
times while the vertex set typically stays fixed.

A naive "pass `Box<dyn Fn(F) -> Graph<F>>`" approach has three failure
modes: (a) it invites callers to mutate `row_ptr`/`col_idx`
arbitrarily — but the Magnus quadrature is mathematically nonsense
across discontinuities of `L_G(·)`; (b) it provides no metadata about
WHERE the discontinuities sit, so the library cannot subdivide steps
correctly; (c) it makes the `Graph::row_ptr` immutability invariant
(ADR-0048) callable-policy rather than type-enforced.

## Decision

Introduce a new public type `GraphTraj<F: SemiflowFloat = f64>` —
**piecewise-smooth graph trajectory** — as a sequence of frozen graph
snapshots with explicit transition times:

```rust
//! crates/semiflow-core/src/graph_traj.rs (NEW FILE, ~280 LoC)

#[derive(Clone)]
pub struct GraphTraj<F: SemiflowFloat = f64> {
    /// Strictly increasing transition times `t_0 = 0 < t_1 < … < t_K = T_horizon`.
    /// Length K+1; K segments.
    breakpoints: Vec<F>,
    /// One `Arc<Graph<F>>` per piecewise segment `[t_k, t_{k+1})`.
    /// Length K.
    snapshots: Vec<Arc<Graph<F>>>,
    /// One edge-weight closure per segment, smooth in `t` within segment.
    /// Returns a `Laplacian<F>` whose CSR layout MUST match `snapshots[k]`.
    /// Length K.
    weight_fns: Vec<Box<dyn Fn(F) -> Arc<Laplacian<F>> + Send + Sync + 'static>>,
}

impl<F: SemiflowFloat> GraphTraj<F> {
    /// Construct from explicit segments. K = `snapshots.len()`.
    ///
    /// # Errors
    /// - `DomainViolation` if `breakpoints` is not strictly increasing.
    /// - `DomainViolation` if `breakpoints.len() != snapshots.len() + 1`.
    /// - `DomainViolation` if `snapshots.len() != weight_fns.len()`.
    /// - `DomainViolation` if `snapshots.len() == 0` or `> 65_535`.
    /// - `DomainViolation` if for any `k`, `weight_fns[k](breakpoints[k])`
    ///   returns a Laplacian whose `row_ptr` / `col_idx` does not match
    ///   `snapshots[k]` (debug-asserted at construction; release retains
    ///   only the count check).
    pub fn new(
        breakpoints: Vec<F>,
        snapshots: Vec<Arc<Graph<F>>>,
        weight_fns: Vec<Box<dyn Fn(F) -> Arc<Laplacian<F>> + Send + Sync + 'static>>,
    ) -> Result<Self, SemiflowError>;

    /// Construct from a *single* (Graph, weight_fn) pair — degenerate
    /// fixed-topology case. Equivalent to v2.1 Magnus contract input.
    pub fn fixed_topology(
        graph: Arc<Graph<F>>,
        weight_fn: Box<dyn Fn(F) -> Arc<Laplacian<F>> + Send + Sync + 'static>,
        t_horizon: F,
    ) -> Result<Self, SemiflowError>;

    /// Borrow the breakpoints slice. Read-only.
    pub fn breakpoints(&self) -> &[F];

    /// Borrow the k-th snapshot. Returns `None` if `k >= n_segments()`.
    pub fn snapshot(&self, k: usize) -> Option<&Arc<Graph<F>>>;

    /// Sample the Laplacian at time `t`. Returns `Err(DomainViolation)`
    /// if `t < 0` or `t > breakpoints.last().unwrap()`.
    /// Boundary policy at `t == breakpoints[k]` for `k >= 1`: belongs
    /// to segment `k` (right-continuous, matches `weight_fns[k]`).
    pub fn laplacian_at(&self, t: F) -> Result<Arc<Laplacian<F>>, SemiflowError>;

    /// Number of piecewise segments K. Equivalent to `snapshots.len()`.
    pub fn n_segments(&self) -> usize;

    /// Locate the segment index `k` such that
    /// `t ∈ [breakpoints[k], breakpoints[k+1])`. Returns `None` if out of range.
    pub fn segment_index(&self, t: F) -> Option<usize>;
}
```

`GraphTraj<F>` is consumed by:

- `MagnusGraphHeatChernoff::evolve_with_traj(traj, n_steps_per_segment, f0)`
  (ADR-0054) — splits the time axis at breakpoints, runs Magnus K=4
  within each segment, copies state across the break (zero-allocation:
  scratch ping-pong reuses across segments).
- `GraphHeatChernoff::evolve_with_traj(traj, n_steps, f0)` — frozen-Laplacian
  fallback (order-1) for users who don't need order-4.
- (Future v2.3+) `KrylovGraphHeatChernoff` — out of v2.2 scope.

## Rationale

- **Snapshot semantics matches physics.** Catalytic-network dynamics,
  rewiring graphs, and mesh-adaptation events all have discrete topology
  jumps at known times. Smooth edge-weight evolution between jumps is
  the same regime that v2.1 Wave C Magnus K=4 already handles.
- **Type-enforces CSR immutability.** The constructor accepts
  `Vec<Arc<Graph<F>>>` — each `Graph<F>` is immutable post-construction
  (ADR-0048 invariant I-immut), so callers physically cannot mutate
  `row_ptr`/`col_idx` mid-stream.
- **Library detects out-of-range queries** but DOES NOT detect physical
  topology jumps inside `weight_fns[k]`. The contract `weight_fns[k]`
  preserves `snapshots[k]`'s CSR layout is debug-asserted; release
  behaviour is best-effort (matches `MagnusGraphHeatChernoff` precedent).
- **Zero new dependencies.** Reuses `Vec`, `Box`, `Arc`, `Graph`,
  `Laplacian` — all `core`/`alloc` types already in `semiflow-core`.

## Consequences

- New module `src/graph_traj.rs` (~280 LoC); under file cap.
- `lib.rs` re-export adds `GraphTraj`. Public surface +1 type, +1
  module — additive minor bump.
- Future v2.3+ option: `GraphTraj<F>` becomes the input type for an
  `evolve_with_traj` method on a future `KrylovGraphHeatChernoff` —
  good forward-compat shape.

## Acceptance gates

- **G12 slope gate** (NORMATIVE — see Wave-A contract §3). Use a
  3-segment trajectory on `P_64`: segment 1 `t ∈ [0, 0.1]`, edges
  `{(0,1), (1,2), …, (62,63)}`; segment 2 `t ∈ [0.1, 0.2]`, add edge
  `(0, 31)` (creates shortcut); segment 3 `t ∈ [0.2, 0.3]`, remove edge
  `(31, 32)`. Smooth `w(t) = 1 + 0.2·sin(πt)` within each segment.
  Run `MagnusGraphHeatChernoff::evolve_with_traj`, self-convergence at
  2× refinement, OLS slope on `(log n_steps, log err_sup)` over
  `n_steps ∈ {10, 20, 40, 80, 160}` per segment. Threshold: slope ≤ −3.95
  (f64, matches single-segment G11 since Magnus quadrature is unchanged
  within segments).
- **T11N_graph_traj sympy gate** (NORMATIVE) — verify that on a 3-segment
  toy `P_4` trajectory, the `evolve_with_traj` time-stepper computes the
  same matrix exponential product `exp(τ·Ω₄^(3)) · exp(τ·Ω₄^(2)) ·
  exp(τ·Ω₄^(1)) · u₀` (verified to τ⁴ per segment) as a direct sympy
  matrix-exponential evaluation. Pass/fail purely symbolic on
  4×4 Laplacians; no library runtime dependency.

## Out of scope (v2.2)

- **Continuous topology drift** (vertex add/remove with smooth `t`
  parametrisation). Requires a vertex-edge commutator class; deferred
  to v2.3+.
- **Auto-detection of breakpoints** (sniffing Lipschitz constant of
  `lap_at_t`). Heuristic-only; brittle. Out of scope.
- **Time-discontinuous edge weights without topology change.** This is
  a degenerate `GraphTraj` with `snapshots[k]` all `Arc::clone`-equal —
  works through the same API but ADR-0054 carries the explicit story
  for jump-aware step subdivision.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Caller's `weight_fns[k]` returns a Laplacian with wrong CSR layout, jumping through `debug_assert` in release | Documented in rustdoc + ADR-0052 §"Decision"; future v2.3 may add a release-mode opt-in `--features graph-traj-strict` check. |
| R2 | `GraphTraj` clone is O(K · (nnz + closure_size)) — large for K → ∞ | Bound K ≤ 65_535 (constructor check); 64K segments at average 1024-edge graphs = ~16 MB of `Arc<Laplacian>` storage — acceptable. |
| R3 | `segment_index` linear scan O(K) makes per-step `laplacian_at` O(K) | At K ≤ 64K and 5 steps per segment, O(K)·O(5K) = O(K²) total = O(64K²) ≈ 4 GFLOP — acceptable for v2.2; can optimise with binary search in v2.3+ if needed. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/graph_traj.rs` | ~280 |
| `tests/g12_graph_traj_slope.rs` | ~160 |
| `.dev-docs/verification/scripts/verify_v2_2_variable_topology_residual.py` | ~130 |
| math.md §14.1 (CITATION + library policy) | ~90 |
| ADR-0052 (this) | ~210 |
| **Total** | **~870** |

## References

- I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) — Theorem 6.
- F. R. K. Chung, *Spectral Graph Theory*, AMS (1997), §1.2.
- A. Iserles, H. Z. Munthe-Kaas, S. P. Nørsett, A. Zanna,
  *Lie-group methods*, **Acta Numerica** **9** (2000) — applies on each
  segment (§5.5 Theorem 5.2).
- ADR-0051 (Magnus K=4 graph) — v2.2 extends this to time-discontinuous
  generators via `evolve_with_traj`.
