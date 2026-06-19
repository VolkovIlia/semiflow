# Wave 2.2A Contract — Time-Varying Graph Laplacians

**Status**: NORMATIVE — engineer implements verbatim against this contract.
**ADRs**: 0052 (`GraphTraj<F>`), 0053 (`VarCoefGraphHeatChernoff`), 0054
(`evolve_with_traj`).
**Depends on**: Wave 2.1A/B/C — `Graph<F>`, `Laplacian<F>`, `GraphSignal<F>`,
`GraphHeatChernoff<F>`, `MagnusGraphHeatChernoff<F>` (all shipped at HEAD =
`1dcca26`).
**Math**: contracts/semiflow-core.math.md §14.
**Sympy gates**: T11N (variable topology), T12N (variable-a graph ζ-A) —
both NEW NORMATIVE.
**Slope gates**: G12 (graph trajectory), G13 (var-a graph), G14
(jump-resolution).
**Author**: ai-solutions-architect · **Date**: 2026-05-21.

This wave ships THREE new public types:

1. `GraphTraj<F>` — piecewise-smooth graph trajectory data structure
   (ADR-0052).
2. `VarCoefGraphHeatChernoff<F>` — variable-coefficient `L_a` Chernoff
   with ζ-A τ²-correction (ADR-0053).
3. `MagnusGraphHeatChernoff::evolve_with_traj` — trajectory-aware
   evolution method on the existing v2.1 Magnus type (ADR-0054).

Wave 2.2A is the only Wave producing math.md changes (§14 is new). Waves
2.2B and 2.2C add §15-§18 but those are CITATION-only (Wave B) or refactor-
pointer (Wave C).

---

## §1 — `GraphTraj<F>` (NORMATIVE — ADR-0052)

### 1.1 File location & module name

```text
crates/semiflow-core/src/graph_traj.rs    (NEW FILE, ≤ 500 LoC)
```

`lib.rs` adds `pub mod graph_traj;` and re-exports `pub use graph_traj::GraphTraj;`.

### 1.2 Public API (verbatim — implement EXACTLY this surface)

```rust
//! crates/semiflow-core/src/graph_traj.rs
//!
//! Piecewise-smooth graph trajectory data structure.
//!
//! See math.md §14.1 (NORMATIVE) and ADR-0052 (design).

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use crate::{Graph, Laplacian, SemiflowError};
use crate::float::SemiflowFloat;

/// Maximum number of segments (resource bound per §14.1 NORMATIVE).
pub const MAX_GRAPH_TRAJ_SEGMENTS: usize = 65_535;

/// Closure type for per-segment Laplacian sampling.
/// Mirrors v2.1C `LaplacianAtTime<F>` (defined in magnus_graph.rs).
pub type SegmentWeightFn<F> =
    Box<dyn Fn(F) -> Arc<Laplacian<F>> + Send + Sync + 'static>;

/// Piecewise-smooth graph trajectory.
///
/// Constructed from K segments `[t_k, t_{k+1})`, each with a snapshot
/// graph (fixed topology within segment) and a smooth edge-weight
/// closure.
///
/// Right-continuous convention: at `t = breakpoints[k]` for `k >= 1`,
/// `laplacian_at(t)` invokes `weight_fns[k]` (matches `snapshots[k]`).
pub struct GraphTraj<F: SemiflowFloat = f64> {
    breakpoints: Vec<F>,
    snapshots: Vec<Arc<Graph<F>>>,
    weight_fns: Vec<SegmentWeightFn<F>>,
}

impl<F: SemiflowFloat> GraphTraj<F> {
    /// Construct from explicit segments.
    ///
    /// # Errors
    /// - `DomainViolation { what: "breakpoints must be strictly increasing", ... }`
    /// - `DomainViolation { what: "breakpoints.len() == snapshots.len() + 1", ... }`
    /// - `DomainViolation { what: "snapshots.len() != weight_fns.len()", ... }`
    /// - `DomainViolation { what: "snapshots.len() must be in [1, MAX_GRAPH_TRAJ_SEGMENTS]", ... }`
    ///
    /// Debug-only: asserts each `weight_fns[k](breakpoints[k])` returns a
    /// Laplacian whose `row_ptr` / `col_idx` matches `snapshots[k]`.
    pub fn new(
        breakpoints: Vec<F>,
        snapshots: Vec<Arc<Graph<F>>>,
        weight_fns: Vec<SegmentWeightFn<F>>,
    ) -> Result<Self, SemiflowError>;

    /// Degenerate single-segment constructor (fixed-topology, full horizon).
    /// Equivalent to v2.1 Magnus contract input.
    pub fn fixed_topology(
        graph: Arc<Graph<F>>,
        weight_fn: SegmentWeightFn<F>,
        t_horizon: F,
    ) -> Result<Self, SemiflowError>;

    /// Read-only borrow of breakpoints. Length `n_segments() + 1`.
    pub fn breakpoints(&self) -> &[F];

    /// Read-only borrow of snapshot at segment `k`. `None` if `k >= n_segments()`.
    pub fn snapshot(&self, k: usize) -> Option<&Arc<Graph<F>>>;

    /// Number of segments K.
    pub fn n_segments(&self) -> usize;

    /// Total horizon `breakpoints[K] - breakpoints[0]`.
    pub fn t_horizon(&self) -> F;

    /// Locate segment containing time `t`. `None` if out of range.
    /// Right-continuous at internal breakpoints.
    pub fn segment_index(&self, t: F) -> Option<usize>;

    /// Sample Laplacian at time `t`. Errors if `t` out of range.
    pub fn laplacian_at(&self, t: F) -> Result<Arc<Laplacian<F>>, SemiflowError>;
}
```

### 1.3 R4 zero-alloc invariants

- `breakpoints()`, `snapshot()`, `n_segments()`, `t_horizon()`,
  `segment_index()` MUST be allocation-free (return slices/refs/values
  only).
- `laplacian_at()` MAY allocate inside the user's `weight_fns[k]` closure;
  library does NOT control this.
- `new()` is one-shot allocating; no allocation budget during use.

### 1.4 Generic-over-F coverage

`F: SemiflowFloat` — both `f32` and `f64` supported. Default `f64`.

---

## §2 — `VarCoefGraphHeatChernoff<F>` (NORMATIVE — ADR-0053)

### 2.1 File location

```text
crates/semiflow-core/src/graph_var_coef.rs    (NEW FILE, ≤ 500 LoC)
```

### 2.2 Public API

```rust
//! crates/semiflow-core/src/graph_var_coef.rs

use alloc::{sync::Arc, vec::Vec};
use crate::{Graph, GraphSignal, Laplacian, SemiflowError, ScratchPool};
use crate::chernoff::ChernoffFunction;
use crate::float::SemiflowFloat;

/// Variable-coefficient graph heat Chernoff with ζ-A τ²-correction.
///
/// Generator: `L_a = A^{1/2} L_G A^{1/2}` where `A = diag(a)`.
/// Order: 2 on smooth `a` (`a_sup/a_inf <= 5` AND `diameter <= 50`); 1 on rough `a`.
///
/// See math.md §14.2 (NORMATIVE) and ADR-0053 (design).
pub struct VarCoefGraphHeatChernoff<F: SemiflowFloat = f64> {
    graph: Arc<Graph<F>>,
    laplacian: Arc<Laplacian<F>>,
    a: Vec<F>,
    sqrt_a: Vec<F>,
    rho_bar: F,
}

impl<F: SemiflowFloat> VarCoefGraphHeatChernoff<F> {
    /// Construct from topology, base Laplacian, conductivity `a`, and Gershgorin
    /// spectral bound `rho_bar`.
    ///
    /// # Errors
    /// - `DomainViolation` if `a.len() != graph.n_nodes()`.
    /// - `DomainViolation` if any `a[i] < 1e-12 * a.iter().copied().fold(F::zero(), F::max)`.
    /// - `DomainViolation` if `rho_bar <= 0` or non-finite.
    pub fn new(
        graph: Arc<Graph<F>>,
        a: Vec<F>,
        rho_bar: F,
    ) -> Result<Self, SemiflowError>;

    pub fn graph(&self) -> &Graph<F>;
    pub fn a(&self) -> &[F];
    pub fn rho_bar(&self) -> F;
}

impl<F: SemiflowFloat> ChernoffFunction<F> for VarCoefGraphHeatChernoff<F> {
    type S = GraphSignal<F>;

    fn apply(
        &self,
        tau: F,
        f: &GraphSignal<F>,
    ) -> Result<GraphSignal<F>, SemiflowError>
    where
        GraphSignal<F>: Clone;

    fn apply_into(
        &self,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;

    fn order(&self) -> u32 { 2 }

    fn growth(&self) -> (f64, f64);
}
```

### 2.3 Algorithm (NORMATIVE — corrected v2.2.0)

`apply_into` computes `dst ← f − τ L_a f + (τ²/2) L_a² f`:

1. Compute `L_a f` via the formula `L_a f = sqrt_a ⊙ (L_G (sqrt_a ⊙ f))`.
   - Step 1a: `tmp[i] = sqrt_a[i] · f[i]` (one vec_mul).
   - Step 1b: `tmp2 = L_G · tmp` (one sparse mat-vec via `Laplacian::apply_into_slice`).
   - Step 1c: `out[i] = sqrt_a[i] · tmp2[i]` (one vec_mul).
2. Compute `L_a² f` = `L_a (L_a f)` via repeating step 1 on the result of step 1.
3. Combine: `dst[i] = f[i] − τ · L_a_f[i] + (τ²/2) · L_a2_f[i]`.

**Correction note**: The original draft included a `−(τ²/12) D_a^{(2)} f` term
(1D Stratonovich correction). This term is NOT applied for the graph operator
`L_a = A^{1/2} L_G A^{1/2}`: since `L_a` is an exact linear map, the Taylor
truncation `I − τL_a + (τ²/2)L_a²` already achieves O(τ³) local error (verified
by T12N gate). Adding `D_a^{(2)}` introduces a spurious O(τ²) offset that degrades
global convergence from order 2 to order 1 (G13 gate regression, fixed v2.2.0).

**CFL check** (NORMATIVE): if `tau * rho_bar * a.iter().fold(F::zero(), F::max).powi(2) > 0.5`,
return `SemiflowError::CflViolated { dx_squared: ..., a_norm_bound: ..., tau: ... }`.

### 2.4 R4 zero-alloc invariant

All scratch buffers (`tmp`, `tmp2`, `L_a_f`, `L_a2_f`) MUST be allocated
inside `scratch` (the `ScratchPool<F>` passed to `apply_into`). Inspect
`crates/semiflow-core/src/scratch.rs` for the allocation pattern.

### 2.5 Generic-over-F coverage

`F: SemiflowFloat` — both `f32` and `f64`. f32 slope band ≤ −1.50 (ADR-0046,
verified at n_steps ∈ {1, 2, 3} — coarser sweep needed since f32 noise floor
dominates at n_steps ≥ 5 for this test configuration). f64 slope band ≤ −1.95.

---

## §3 — `MagnusGraphHeatChernoff::evolve_with_traj` (NORMATIVE — ADR-0054)

### 3.1 File location

Extends existing `crates/semiflow-core/src/magnus_graph.rs` by ~150 LoC.
The file MAY grow to ~820 LoC at v2.2 (within Override #1 carve-out).

### 3.2 Public API

```rust
//! crates/semiflow-core/src/magnus_graph.rs (extension)

use crate::graph_traj::GraphTraj;

impl<F: SemiflowFloat> MagnusGraphHeatChernoff<F> {
    /// Evolve `f0 ↦ u(T_horizon)` along a piecewise-smooth trajectory.
    ///
    /// Within each segment, runs `n_steps_per_segment` Magnus K=4 sub-steps.
    /// At each breakpoint, re-binds the internal `lap_at_t` and resets the
    /// time-axis local origin to the breakpoint.
    ///
    /// # Errors
    /// - `DomainViolation` if `n_steps_per_segment == 0`.
    /// - `DomainViolation` if `traj.breakpoints().first()` != `F::zero()`.
    /// - `DomainViolation` if `f0.n_nodes() != traj.snapshot(0).n_nodes()`.
    /// - `OutOfMagnusRadius` if `self.rho_bar_max * tau >= π/2` for any segment.
    ///
    /// # Note
    /// Per ADR-0052 §"Out of scope": all segments MUST have the same vertex
    /// count. Different snapshots MAY have different topologies but identical N.
    // NOTE: receiver relaxed from `&mut self` to `&self` at implementation time.
    // `MagnusGraphHeatChernoff` does not mutate internal state during evolution;
    // `&self` is caller-favorable (permits concurrent evolutions of the same
    // instance with different trajectories or initial conditions).
    pub fn evolve_with_traj(
        &self,
        traj: &GraphTraj<F>,
        n_steps_per_segment: usize,
        f0: &GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<GraphSignal<F>, SemiflowError>;

    /// Same as `evolve_with_traj` but writes into caller-supplied `dst`.
    /// Zero allocation in the steady-state loop.
    pub fn evolve_with_traj_into(
        &self,
        traj: &GraphTraj<F>,
        n_steps_per_segment: usize,
        f0: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
}
```

### 3.3 Algorithm (NORMATIVE — verbatim)

```text
fn evolve_with_traj_into(traj, n_steps_per_segment, f0, dst, scratch) {
    validate inputs
    dst.copy_from(f0)
    for k in 0..traj.n_segments() {
        let segment_start = traj.breakpoints()[k]
        let segment_tau = (traj.breakpoints()[k+1] - segment_start) / F::from(n_steps_per_segment).unwrap()
        // Re-bind to segment k's weight closure.
        // The closure is owned by `traj`; `self` must temporarily hold a clone of the
        // function pointer or use a different internal mechanism (see Risk R2 below).
        for step in 0..n_steps_per_segment {
            let t_start = segment_start + F::from(step).unwrap() * segment_tau
            self.apply_into_at_using_traj(traj, k, t_start, segment_tau, dst, scratch)?
        }
    }
    Ok(())
}
```

**Implementation note for engineer**: the existing `apply_into_at` takes a closure
from `self.lap_at_t`. The trajectory version needs to access `traj.weight_fns[k]`
without taking ownership. Two options:

- **Option A (recommended)**: add a private method
  `apply_into_at_with_lap_fn` that takes a `&SegmentWeightFn<F>` instead of
  using `self.lap_at_t`. The public `evolve_with_traj` and existing
  `apply_into_at` both delegate to this.
- **Option B**: temporarily swap `self.lap_at_t` with the segment's closure.
  Borrow-checker-unfriendly; avoid.

### 3.4 R4 zero-alloc invariant

`evolve_with_traj_into` MUST allocate zero bytes in the steady-state loop
(after segment 0 is set up). All Magnus scratch buffers reused across segments.
A single `GraphSignal::copy_from` per segment boundary (already zero-alloc per
ADR-0043).

### 3.5 Generic-over-F coverage

Inherits `MagnusGraphHeatChernoff<F>`'s coverage: `F: SemiflowFloat` — both
`f32` and `f64`.

---

## §4 — Sympy gates (NORMATIVE)

### 4.1 T11N_variable_topology_residual (NEW)

Path: `.dev-docs/verification/scripts/verify_v2_2_variable_topology_residual.py`

Logic (verbatim — engineer implements):

```python
# Symbolic 4-node path graph trajectory:
# - Segment 0: t ∈ [0, τ], P_4 with edge weights all = 1
# - Segment 1: t ∈ [τ, 2τ], same P_4 with edge weights all = 1 + α  (smooth, α=symbol)
# - Segment 2: t ∈ [2τ, 3τ], P_4 + extra edge (0, 3) creating a cycle
#
# Verify: matrix product exp(τ·Ω₄^{(2)}) · exp(τ·Ω₄^{(1)}) · exp(τ·Ω₄^{(0)}) · u_0
# matches sympy series expansion of u(3τ) to order τ⁴ within each segment.
#
# Pass criterion: residual norm in Frobenius < 1e-12 for symbolic τ → 0 limit
# (i.e., the τ⁵ remainder is the only nonzero term).
```

Expected gate exit code: 0 PASS / 1 FAIL.

### 4.2 T12N_variable_a_graph_residual (NEW)

Path: `.dev-docs/verification/scripts/verify_v2_2_variable_a_graph_residual.py`

Logic:

```python
# Symbolic 4-node path with a = [a_0, a_1, a_2, a_3], edge weights all = 1.
# Compute L_a = A^{1/2} L_G A^{1/2} symbolically.
# Expand exp(-τ L_a) through τ² via sp.series.
# Expand the library's formula `f - τ L_a f + (τ²/2) L_a² f - (τ²/12) D_a^{(2)} f` symbolically.
# Verify both expansions match through τ²; the τ³ residual is documented but non-zero.
```

Expected gate exit code: 0 PASS / 1 FAIL.

---

## §5 — Slope gates (NORMATIVE — tests/g{12,13,14}_*.rs)

### 5.1 G12 slope gate (`tests/g12_graph_traj_slope.rs`)

Setup:
- Path `P_64` with 3 segments per ADR-0052 §"Acceptance gates" §G12.
- `n_steps_per_segment ∈ {10, 20, 40, 80, 160}`.
- Self-convergence at 2× refinement.

Pass criterion: `slope ≤ -3.95` (f64) / `slope ≤ -3.50` (f32).

### 5.2 G13 slope gate (`tests/g13_var_a_graph_slope.rs`)

Setup:
- `P_n` with `n ∈ {32, 64, 128, 256}`.
- `a(i) = 1 + 0.5·cos(2π · i/n)`.
- `t_final = 0.05`, `n_steps ∈ {25, 50, 100, 200, 400}`.

Pass criterion: `slope ≤ -1.95` (f64) / `slope ≤ -1.50` (f32).

### 5.3 G14 jump-resolution gate (`tests/g14_jump_resolution.rs`)

Setup:
- Star graph `S_4`, edge `(0, 1)` weight switches `1 → 0.5 → 2 → 0.5` at fixed breakpoints `[0.1, 0.2, 0.3]`.
- Compare `evolve_with_traj` (with-trajectory) vs the naive `apply_into_at` (no trajectory) at `t_final = 0.4`.

Pass criterion: `slope_with_traj / slope_without_traj ≤ 0.5` (with-traj is ≥ 2× steeper slope).

---

## §6 — Capability check (NORMATIVE — security-by-design guardrail #7)

No new capability boundaries. v2.2 Wave A is an additive Rust API surface;
caller-supplied closures (`SegmentWeightFn`) are `Send + Sync + 'static` which
matches the existing `LaplacianAtTime<F>` v2.1 contract. STRIDE:

- **S/T**: closures are pure, called within scope; no spoofing/tampering surface.
- **I**: `Graph<F>`, `Laplacian<F>`, `GraphTraj<F>` are read-only post-construction.
- **D**: `MAX_GRAPH_TRAJ_SEGMENTS = 65_535` cap prevents resource exhaustion.

---

## §7 — Build/run path (NORMATIVE — same as v2.1)

```bash
cargo run -p xtask -- test-fast       # default
cargo run -p xtask -- test-full       # +parallel +simd +slow-tests
cargo run -p xtask -- test-flagship   # ignored tests only (slope gates)
```

No new build targets, no new feature flags.

---

## §8 — Engineer pickup ordering (NORMATIVE)

Step 1: Read ADR-0052, ADR-0053, ADR-0054 + math.md §14.

Step 2: Implement `graph_traj.rs` (ADR-0052). Re-run `cargo test` to ensure
no regressions in existing v2.1 tests.

Step 3: Implement `graph_var_coef.rs` (ADR-0053). Add `tests/g13_var_a_graph_slope.rs`.
Run `test-flagship` to confirm slope.

Step 4: Implement `magnus_graph.rs::evolve_with_traj` (ADR-0054). Add
`tests/g12_graph_traj_slope.rs` and `tests/g14_jump_resolution.rs`. Confirm
slopes.

Step 5: Implement sympy gates T11N + T12N. Add to CI.

Step 6: Update math.md cross-references (§12.9 OutOfScope → "answered in §14").

Step 7: Update CHANGELOG.md with Wave 2.2A entry; re-cite ADRs.

Step 8: Handoff to git-workflow for Wave 2.2A commit. Trailer: `Agent: agentic-engineer`,
`Task-ID: v2.2-wave-a-time-varying-graph`.
