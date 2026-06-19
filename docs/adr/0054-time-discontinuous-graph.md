# ADR-0054 — Time-discontinuous `L_G(t)` via `evolve_with_traj`

- **Status**: ACCEPTED (v2.2 Wave A — implementation shipped; G12+G14 slope gates pending HW run)
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave A
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0051 (Magnus K=4), ADR-0052 (`GraphTraj`).
- **Mathematical foundation**: math.md §14.3 (NORMATIVE; CITATION:
  Caglioti-Pulvirenti-Rousset 2016 "Operator splitting for discontinuous
  generators" survey on piecewise-smooth Magnus methods).

## Context

GL₄ Gauss-Legendre quadrature (ADR-0051 §"Decision" step 1) assumes
`L_G(·) ∈ C²([0, τ])`. At a topology breakpoint `t_*`, the second derivative
of `L_G(·)` is a Dirac measure — the quadrature error blows up. The
naive remedy ("subdivide the step until each sub-step lies inside a smooth
segment") works but requires the library to KNOW where the breakpoints
are. ADR-0052's `GraphTraj<F>` provides exactly that knowledge.

Without explicit jump-handling, `MagnusGraphHeatChernoff::apply` across
a breakpoint converges only at order 1 in `n_steps` (the Magnus
truncation error inherits the worst case from the rough segment).
Customers needing piecewise-smooth `L_G(t)` (catalytic networks,
rewiring graphs) cannot achieve order-4 without library support.

## Decision

Add a new method `evolve_with_traj` on `MagnusGraphHeatChernoff<F>` that
consumes a `GraphTraj<F>` (ADR-0052) and walks it segment-by-segment:

```rust
//! crates/semiflow-core/src/magnus_graph.rs (extends existing file by ~140 LoC)

impl<F: SemiflowFloat> MagnusGraphHeatChernoff<F> {
    /// Evolve `f0 ↦ u(T_horizon)` with explicit jump-handling.
    ///
    /// For each segment `[t_k, t_{k+1})` in `traj.breakpoints()`:
    ///   1. Compute segment τ = `(t_{k+1} - t_k) / n_steps_per_segment`.
    ///   2. Run `n_steps_per_segment` Magnus K=4 sub-steps within segment.
    ///   3. At segment boundary, copy state into the next segment's
    ///      working buffer (zero-allocation: `scratch.copy_from`).
    ///
    /// The result `u(T_horizon)` is accurate to order 4 GLOBALLY in
    /// `n_steps_per_segment` PROVIDED `lap_at_t` is `C²` within each segment.
    ///
    /// # Errors
    /// - `DomainViolation` if `n_steps_per_segment == 0`.
    /// - `DomainViolation` if `traj.breakpoints().first() != 0`.
    /// - `DomainViolation` if `f0.n_nodes() != traj.snapshot(0).n_nodes()`.
    /// - `OutOfMagnusRadius` if `traj.rho_bar_max() · τ ≥ π/2` for any segment.
    ///
    /// # Topology preservation
    /// The library assumes `traj.snapshot(0).n_nodes() == traj.snapshot(k).n_nodes()`
    /// for all `k` (fixed vertex set, variable edges only). Otherwise → `DomainViolation`.
    pub fn evolve_with_traj(
        &mut self,
        traj: &GraphTraj<F>,
        n_steps_per_segment: usize,
        f0: &GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<GraphSignal<F>, SemiflowError>;

    /// Alternative: write into caller-supplied buffer (zero-alloc).
    pub fn evolve_with_traj_into(
        &mut self,
        traj: &GraphTraj<F>,
        n_steps_per_segment: usize,
        f0: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
}
```

The same closure-of-Laplacian is shared across segments — the
`weight_fns[k]` for each segment is invoked inside the Magnus K=4 GL₄
nodes via `traj.laplacian_at(t_start + c_i · τ)`. The library's
`MagnusGraphHeatChernoff` instance is mutated to re-bind its internal
`lap_at_t` pointer to `traj.weight_fns[k]` at each segment boundary
(borrow-checker note: `evolve_with_traj` takes `&mut self`).

## Rationale

- **Order-4 preserved within each segment.** GL₄ quadrature error is
  `O(τ⁵)` on `C²` data — within each segment, `lap_at_t` is `C²` by
  the `GraphTraj<F>` contract (ADR-0052 §"Decision" clause "smooth
  within segment").
- **Order-1 at jumps is correct.** Strict topology jump means
  `L_G(t_k^−) ≠ L_G(t_k^+)`; no Magnus quadrature can give better than
  the order of the worst sub-step. By splitting at breakpoints, the
  library avoids spreading the error across smooth regions.
- **Zero alloc.** All Magnus scratch buffers (allocated once in the
  `MagnusGraphHeatChernoff` constructor via `ScratchPool`) are reused
  across segments. Only the segment-boundary state copy happens
  (single `GraphSignal::copy_from`, allocation-free per ADR-0043).
- **No new types.** Reuses `GraphTraj<F>` (ADR-0052) as the sole new
  data structure; `evolve_with_traj` is a method on the existing
  `MagnusGraphHeatChernoff`.

## Consequences

- `magnus_graph.rs` grows from 675 LoC (v2.1.0-rc.1) to ~820 LoC at
  v2.2. **Joins Override #1 file-list at v2.2** (already in carve-out
  per constitution v1.4.0; this is an EXPANSION of existing
  Override #1 capacity, not a new override).
- Public surface +2 methods (`evolve_with_traj`, `evolve_with_traj_into`).
- Backwards-compatible: existing `apply`, `apply_into`, `apply_into_at`
  methods unchanged.

## Acceptance gates

- **G12 slope gate** (extension of ADR-0052's G12). Pure topology jump
  case: identical edge-weight closure across two segments, one with
  added edge. Slope: ≤ −3.95 (f64) (since both segments are smooth, the
  jump is the only break, but the library handles it correctly).
- **G14 jump-resolution gate** (NORMATIVE). Star graph S_4 with 4
  segments, edge `(0, 1)` weight `w(t)` switching `1 → 0.5 → 2 → 0.5`
  at `t ∈ {0.1, 0.2, 0.3}`. Without `evolve_with_traj`: pure Magnus
  with `lap_at_t` capturing the discontinuities gives order ~1 slope.
  With `evolve_with_traj`: slope ≤ −3.95. CI gate ratio:
  `slope_with_traj / slope_without_traj ≤ 0.5` (with-traj is at least
  twice as steep).

## Out of scope (v2.2)

- **Auto-detect breakpoints.** Heuristic (sniff Lipschitz constant of
  `lap_at_t`). Brittle. Deferred.
- **Continuous topology evolution.** Smooth `Graph<F>` evolution (vertex
  add/remove via commutator class). Deferred to v2.3+.
- **Higher-order quadrature within segment.** ADR-0056's K=6 Magnus
  applies as-is via `MagnusGraphHeat6thChernoff::evolve_with_traj`
  (Wave B). Out of Wave A scope.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | `evolve_with_traj` callsite ergonomically harder than `apply_into` (caller must build `GraphTraj` first) | Document `GraphTraj::fixed_topology` constructor (ADR-0052) for the degenerate single-segment case — gives v2.1 compatibility. |
| R2 | Mid-segment topology drift (caller mutates `traj` between calls) | `&GraphTraj<F>` shared-ref enforces no-mutation; `Graph<F>` is immutable per ADR-0048. |
| R3 | Segment boundary state copy adds an O(N) memcpy per breakpoint | Negligible (cache-friendly memcpy, N ≤ 1M typical) vs cost of K=4 sub-steps (each ~4 sparse mat-vec). |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/magnus_graph.rs` (extension) | +140 |
| `tests/g14_jump_resolution.rs` | ~180 |
| math.md §14.3 | ~70 |
| ADR-0054 (this) | ~180 |
| **Total** | **~570** |

## References

- E. Caglioti, M. Pulvirenti, F. Rousset, "Operator splitting for
  discontinuous generators" — review pattern of segment-wise Magnus
  for piecewise-smooth generators.
- A. Iserles, H. Z. Munthe-Kaas, S. P. Nørsett, A. Zanna,
  *Acta Numerica* **9** (2000) §5.5 Theorem 5.2 — GL₄ Magnus error
  bound on `C²` data.
