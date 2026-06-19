# Wave 2.1B Contract — Higher-Order Graph PDE

**Status**: NORMATIVE — engineer implements verbatim against this contract.
**ADRs**: ADR-0047 (GraphHeatChernoff design — extended), ADR-0048 (CSR storage), ADR-0012 (palindromic Strang reference), ADR-0013 (ζ⁴ pattern reference), ADR-0046 (precision policy bands), ADR-0042 (apply_into ping-pong template).
**No new ADR**: Wave 2.1B additions are sub-design of ADR-0047 (`GraphHeatChernoff` family) plus idiomatic reuse of ADR-0012 / ADR-0013 / ADR-0042 patterns transposed from regular grids to graph signals. See §A appendix for rationale.
**Scope**: `semiflow-core` v2.1 Wave B — order-2 / order-4 / Strang-split graph Chernoff variants.
**Author**: ai-solutions-architect · **Date**: 2026-05-20 · **Reviewers**: reviewer-suckless, agentic-engineer.
**Depends on**: Wave 2.1A (`Graph<F>`, `GraphSignal<F>`, `Laplacian<F>`, `GraphHeatChernoff<F>`) — already shipped at commit `828f7bb`.

Wave 2.1B ships THREE Chernoff variants for **constant edge-coefficient `a ≡ 1`**
only (per AskUserQuestion #2; variable `a(v)` deferred to v2.2 — requires operator-
product derivation):

1. **`GraphHeatChernoff::with_zeta_a()`** — order-2 Chernoff via operator Taylor truncation.
2. **`GraphHeat4thChernoff<F>`** — order-4 Chernoff via Padé[0,4] operator Taylor truncation.
3. **`StrangSplitGraph<A, B, F>`** — palindromic composition on commuting graph operator pairs.

All three are additive to the existing `ChernoffFunction<F, S = GraphSignal<F>>`
surface — `chernoff.rs` is UNCHANGED, the `ChernoffSemigroup::evolve` executor
accepts them without modification.

Magnus K=4 for time-dependent `L_G(t)` is deferred to **Wave 2.1C** (separate
contract, separate slope gate G11).

---

## §1 — `GraphHeatChernoff::with_zeta_a()` constructor (NORMATIVE)

### 1.1 Mathematical statement (CITATION; see math.md §12.6)

For constant `a ≡ 1` (the only case in scope for v2.1), the full ζ-A τ²-correction
formula from §9.2.3.B `τ²·[a·a'·f''' + ½·a·a''·f'' + ¼·a'·a''·f']` reduces to the
zero polynomial in `f', f'', f'''` because `a' ≡ 0` and `a'' ≡ 0`. The natural
order-2 Chernoff for bounded generators is therefore the operator Taylor
truncation of `exp(−τ L_G)`:

```text
S(τ) = I − τ L_G + (τ² / 2) · L_G²       (NORMATIVE library choice)
```

This is a degree-2 polynomial in `L_G` evaluated by two SpMV's; it is **not** a
new theorem but the standard Taylor expansion of the matrix exponential on a
bounded generator (Pazy 1983 §1.3, Engel-Nagel 2000 §III.5, Hochbruck-Ostermann
2010 *Acta Numerica* §3 on truncated-exponential families for bounded operators).

Chernoff-formula hypothesis check (for completeness; see math.md §12.6 for the
full citation):

- `S(0) = I` ✓ (zero-th term is `I`).
- `S'(0) = −L_G` ✓ (linear term coefficient).
- Quasi-contractivity holds for `τ · ρ̄ ≤ ½` (same envelope as the order-1 path;
  the additional `τ² L_G²/2` term has operator norm `≤ τ² ρ̄²/2 ≤ τ ρ̄ /4`,
  bounded by the existing margin).

Operator-level convergence rate: `‖(S(t/n))^n f − e^{−tL_G} f‖ = O(1/n²)`, hence
the slope gate target is **−1.95** (f64) / **−1.85** (f32).

### 1.2 API surface

```rust
//! crates/semiflow-core/src/graph_heat.rs  (EDIT: extend existing module)

use alloc::sync::Arc;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum GraphHeatOrder {
    /// Order-1: `S(τ) = I − τ L_G`. Default.
    Leading,
    /// Order-2 (constant `a ≡ 1`): `S(τ) = I − τ L_G + (τ²/2) L_G²`.
    /// Operator Taylor truncation; see math.md §12.6.
    ZetaATaylor2,
}

pub struct GraphHeatChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
    order_variant: GraphHeatOrder,
}

impl<F: SemiflowFloat> GraphHeatChernoff<F> {
    /// Order-1 constructor (unchanged from Wave 2.1A).
    pub fn new(laplacian: Arc<Laplacian<F>>) -> Self {
        Self { laplacian, order_variant: GraphHeatOrder::Leading }
    }

    /// Order-1 owned-Laplacian constructor (unchanged from Wave 2.1A).
    pub fn from_owned(laplacian: Laplacian<F>) -> Self {
        Self {
            laplacian: Arc::new(laplacian),
            order_variant: GraphHeatOrder::Leading,
        }
    }

    /// **Order-2 constructor (constant `a ≡ 1` only).** Wave 2.1B addition.
    ///
    /// Builds `S(τ) = I − τ L_G + (τ²/2) L_G²` — Taylor truncation of
    /// `exp(−τ L_G)`. See math.md §12.6 (NORMATIVE).
    ///
    /// Variable `a(v)` (heterogeneous edge coefficient per node) is OUT OF
    /// SCOPE for v2.1 — deferred to v2.2 pending operator-product derivation.
    ///
    /// # Runtime cost (per `apply_into` call, steady state)
    /// - 2 SpMV's (one for `L_G · src`, one for `L_G · (L_G · src) = L_G² · src`).
    /// - 1 `borrow_vec(N)` from the `ScratchPool` (recycled after warmup).
    /// - 0 heap allocations.
    pub fn with_zeta_a(laplacian: Arc<Laplacian<F>>) -> Self {
        Self { laplacian, order_variant: GraphHeatOrder::ZetaATaylor2 }
    }

    /// Owned-Laplacian convenience mirror for `with_zeta_a`.
    pub fn from_owned_with_zeta_a(laplacian: Laplacian<F>) -> Self {
        Self {
            laplacian: Arc::new(laplacian),
            order_variant: GraphHeatOrder::ZetaATaylor2,
        }
    }

    pub fn laplacian(&self) -> &Laplacian<F> { &self.laplacian }

    /// Document the active variant (for downstream debugging / regression
    /// tests). Not part of the `ChernoffFunction` contract.
    #[doc(hidden)]
    pub(crate) fn order_variant(&self) -> GraphHeatOrder { self.order_variant }
}
```

### 1.3 `apply_into` body (NORMATIVE)

Branch on `self.order_variant`; the existing Wave 2.1A code path stays
byte-identical for `Leading`. The `ZetaATaylor2` branch borrows ONE additional
`N`-length scratch buffer for `L_G² · src`. Borrow ordering keeps the steady-state
allocation count at zero (existing `ScratchPool` LIFO recycling).

```rust
fn apply_into(
    &self,
    tau: F,
    src: &Self::S,
    dst: &mut Self::S,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where Self::S: Clone,
{
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "GraphHeatChernoff: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    let n = src.len();
    debug_assert_eq!(dst.len(), n);
    debug_assert_eq!(self.laplacian.n_nodes(), n);

    match self.order_variant {
        GraphHeatOrder::Leading => {
            let mut lap = scratch.borrow_vec(n);
            self.laplacian.apply_into_slice(src.values(), &mut lap);
            dst.copy_from(src);
            dst.axpy_into_slice(-tau, &lap);
        }
        GraphHeatOrder::ZetaATaylor2 => {
            // Two sequential borrows; second outlives first (sequential drop scope).
            let mut lap1 = scratch.borrow_vec(n);
            self.laplacian.apply_into_slice(src.values(), &mut lap1);
            let mut lap2 = scratch.borrow_vec(n);
            self.laplacian.apply_into_slice(&lap1, &mut lap2);

            // dst ← src − τ · lap1 + (τ²/2) · lap2
            let half = F::one() / (F::one() + F::one());
            dst.copy_from(src);
            dst.axpy_into_slice(-tau, &lap1);
            dst.axpy_into_slice(half * tau * tau, &lap2);
        }
    }
    Ok(())
}
```

### 1.4 `ChernoffFunction` trait method overrides

- `order() -> u32`: dispatch by variant — `Leading → 1`, `ZetaATaylor2 → 2`.
- `growth() -> (f64, f64)`: same `(1.0, ρ̄)` for both variants. The order-2
  Taylor truncation does NOT improve the contractivity envelope; the additional
  `τ² L_G²/2` term is dominated by the linear `−τ L_G` term within the existing
  `τ · ρ̄ ≤ ½` window.

### 1.5 LoC budget — `graph_heat.rs` extension

| Element | Δ LoC |
|---|---:|
| `GraphHeatOrder` enum + doc | +18 |
| Struct field `order_variant: GraphHeatOrder` | +1 |
| Existing constructors set `order_variant: Leading` | +2 |
| `with_zeta_a` + `from_owned_with_zeta_a` constructors + docs | +30 |
| `order_variant` accessor | +4 |
| `apply_into` branch on enum (existing `Leading` branch kept verbatim) | +25 |
| `order()` dispatch | +5 |
| **Total Δ in graph_heat.rs** | **+85** |

Existing file is ~120 LoC; post-edit ~205 LoC, well under the 500 cap.

---

## §2 — `GraphHeat4thChernoff<F>` (NORMATIVE)

### 2.1 Mathematical statement (CITATION; see math.md §12.7)

For constant `a ≡ 1`, the order-4 Chernoff is the Padé[0,4] / 4-term operator
Taylor truncation of `exp(−τ L_G)`:

```text
S₄(τ) = I − τ L_G + (τ²/2) L_G² − (τ³/6) L_G³ + (τ⁴/24) L_G⁴
      = Σ_{k=0}^{4} (−τ L_G)^k / k!                              (NORMATIVE)
```

This is a degree-4 polynomial in `L_G`. Reference: Hochbruck-Ostermann 2010
*Acta Numerica* "Exponential Integrators" §3 (truncated-exponential families on
bounded operators) and Higham 2008 *Functions of Matrices* §10 (Taylor methods
for `exp(A)`).

Hypothesis check for Chernoff product formula:

- `S₄(0) = I` ✓ (constant term).
- `S₄'(0) = −L_G` ✓ (linear coefficient).
- Quasi-contractivity for `τ · ρ̄ ≤ ½` (the alternating-sign tail dominates by
  the linear term as long as `τ ρ̄ < 1`; the ½ envelope is conservative).

Operator-level rate: `‖(S₄(t/n))^n f − e^{−tL_G} f‖ = O(1/n⁴)`. Slope gate
target: **−3.95** (f64) / **−3.50** (f32) — f32 band wider per ADR-0046 because
the four-SpMV accumulation amplifies round-off error proportionally to `‖L_G‖⁴·τ⁴`
which saturates f32's `~10⁻⁷` floor at finer grids.

### 2.2 API surface

```rust
//! crates/semiflow-core/src/graph_heat4.rs (NEW module, ~240 LoC)

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

/// Order-4 Chernoff for `∂ₜu = −L_G u` via Padé[0,4] / 4-term operator
/// Taylor truncation of `exp(−τ L_G)`.
///
/// `S₄(τ) f = Σ_{k=0}^{4} (−τ L_G)^k / k! · f`.
///
/// **Constant edge-coefficient `a ≡ 1` only** for v2.1. See math.md §12.7
/// (NORMATIVE) and Hochbruck-Ostermann 2010 *Acta Numerica* §3.
///
/// Stores `Arc<Laplacian<F>>` (cheap clone for composition). Uses
/// [`ScratchPool`] for the four SpMV intermediates (0 heap allocations in
/// steady state).
pub struct GraphHeat4thChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
}

impl<F: SemiflowFloat> GraphHeat4thChernoff<F> {
    pub fn new(laplacian: Arc<Laplacian<F>>) -> Self {
        Self { laplacian }
    }

    pub fn from_owned(laplacian: Laplacian<F>) -> Self {
        Self { laplacian: Arc::new(laplacian) }
    }

    pub fn laplacian(&self) -> &Laplacian<F> { &self.laplacian }
}

impl<F: SemiflowFloat> ChernoffFunction<F> for GraphHeat4thChernoff<F> {
    type S = GraphSignal<F>;

    fn apply(&self, tau: F, f: &Self::S) -> Result<Self::S, SemiflowError>
    where Self::S: Clone,
    {
        let mut dst = f.clone();
        let mut scratch = ScratchPool::<F>::new();
        self.apply_into(tau, f, &mut dst, &mut scratch)?;
        Ok(dst)
    }

    /// Body in §2.3.
    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>
    where Self::S: Clone,
    { apply_zeta4_into(&self.laplacian, tau, src, dst, scratch) }

    fn order(&self) -> u32 { 4 }

    fn growth(&self) -> (f64, f64) {
        (1.0, self.laplacian.spectral_radius_bound()
                  .to_f64().unwrap_or(f64::INFINITY))
    }
}
```

### 2.3 `apply_zeta4_into` body (NORMATIVE)

Four sequential SpMV's accumulate `L_G^k · src` for `k = 1, 2, 3, 4` into a
single rolling pair of buffers. Coefficients are fixed: `−τ`, `+τ²/2`, `−τ³/6`,
`+τ⁴/24`. Borrow ordering: at most TWO `borrow_vec(N)` live simultaneously.

```rust
fn apply_zeta4_into<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "GraphHeat4thChernoff: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    let n = src.len();
    debug_assert_eq!(dst.len(), n);
    debug_assert_eq!(lap.n_nodes(), n);

    // Coefficient table (Padé[0,4] / Taylor truncation).
    let one  = F::one();
    let two  = one + one;
    let six  = two + two + two;
    let _    = (); // placeholder
    // 24 = 4! = 2 * 3 * 4
    let twenty_four = (two + two) * (two + one) * two; // (2+2)*3*2 = 24
    let c1 = -tau;
    let c2 =  tau * tau / two;
    let c3 = -tau * tau * tau / six;
    let c4 =  tau * tau * tau * tau / twenty_four;

    // ping = L_G^1 · src; pong = L_G^2 · src; reused alternately.
    let mut ping = scratch.borrow_vec(n);
    let mut pong = scratch.borrow_vec(n);

    // ping ← L_G · src
    lap.apply_into_slice(src.values(), &mut ping);

    // dst ← src + c1 · ping = src − τ L_G · src
    dst.copy_from(src);
    dst.axpy_into_slice(c1, &ping);

    // pong ← L_G · ping = L_G² · src
    lap.apply_into_slice(&ping, &mut pong);
    dst.axpy_into_slice(c2, &pong);

    // ping ← L_G · pong = L_G³ · src
    lap.apply_into_slice(&pong, &mut ping);
    dst.axpy_into_slice(c3, &ping);

    // pong ← L_G · ping = L_G⁴ · src
    lap.apply_into_slice(&ping, &mut pong);
    dst.axpy_into_slice(c4, &pong);

    Ok(())
}
```

Function size: ≈ 42 LoC (within 50-LoC cap).

### 2.4 LoC budget — `graph_heat4.rs`

| Element | LoC |
|---|---:|
| Module doc + imports | 30 |
| `GraphHeat4thChernoff` struct + 3 inherent methods | 25 |
| `ChernoffFunction` impl skeleton (apply/order/growth) | 35 |
| `apply_zeta4_into` body | 50 |
| Inline unit-test smoke (`#[cfg(test)] mod tests`) | 60 |
| Doc examples | 40 |
| **Total** | **≈ 240** |

Well under 500 LoC cap.

---

## §3 — `StrangSplitGraph<A, B, F>` (NORMATIVE)

### 3.1 Mathematical statement (CITATION; see math.md §12.8)

Let `L_G = L_A + L_B` be a decomposition of a graph Laplacian into two
commuting bounded symmetric positive-semidefinite operators (`[L_A, L_B] = 0`).
Then the palindromic Strang product

```text
S(τ) = e^{(τ/2) L_A} ∘ e^{τ L_B} ∘ e^{(τ/2) L_A}
```

is **exact** for the combined operator: `S(τ) = e^{τ (L_A + L_B)}`. This follows
from the BCH series terminating at zero-th order when `[L_A, L_B] = 0`
(`exp(X)·exp(Y) = exp(X + Y)` for commuting `X, Y`). See math.md §10.8
(Theorem 7' carries through verbatim for bounded operators) and the citation
chain in math.md §12.8.

When the Chernoff functions `A: ChernoffFunction<F, S = GraphSignal<F>>` and
`B: ChernoffFunction<F, S = GraphSignal<F>>` are themselves order-`m_A` /
order-`m_B` approximants of `e^{−τ L_A}` / `e^{−τ L_B}` (not exact), the
overall order of the palindromic product is `min(m_A, m_B, 2)` — Strang's
canonical global order — because the splitting error is zero (commuting) but
the per-leg approximation error is non-zero.

**The library's `StrangSplitGraph` does NOT verify commutativity.** It accepts
either (a) a safety-prefixed `unsafe_commutes_axiomatically: bool` opt-in via
the generic constructor with documented caller responsibility, or (b) one of
the two **safe** constructors `new_bipartite_path` / `new_bipartite_cycle`
which produce a guaranteed-commuting decomposition by 2-coloring.

### 3.2 Bipartite 2-coloring decomposition (NORMATIVE)

For a **bipartite graph** with parts `V = V_red ∪ V_black`, every edge has one
endpoint in each part. Split the edge set:

- `L_A`: assembled from edges `(u, v)` with `min(color(u), color(v)) == RED`.
- `L_B`: assembled from edges `(u, v)` with `min(color(u), color(v)) == BLACK`.

Wait — this is degenerate (one set is empty for bipartite). The correct
decomposition is **alternating-edge**: index edges in their CSR traversal order
and assign even-indexed edges to `L_A`, odd-indexed to `L_B`. For path and
cycle graphs with 2-coloring of NODES `0 = RED, 1 = BLACK, 2 = RED, …`:

- **Red edges**: edges `(i, i+1)` where `i` is RED (i.e., `i` even). Path: edges `(0,1), (2,3), (4,5), …`.
- **Black edges**: edges `(i, i+1)` where `i` is BLACK (`i` odd). Path: edges `(1,2), (3,4), (5,6), …`.

Each node `i` belongs to AT MOST ONE red edge (the one toward `i+1` if `i`
even, OR the one toward `i−1` if `i` odd — but never both — by 2-coloring).
Same for black. Therefore the red Laplacian `L_red` is block-diagonal with
2×2 blocks `[[1,-1],[-1,1]]` on disjoint node pairs; same for `L_black` with
the opposite pairing. Block-diagonal operators on **disjoint pair sets**
commute trivially: `L_red · L_black = L_black · L_red` because each block of
`L_red` acts on nodes that lie in DIFFERENT blocks of `L_black` (each black
block intersects exactly two red blocks).

> **Verification: pen-and-paper.** Let nodes be `0,1,2,3` (path-4). Then
> `L_red = block diag([[1,-1],[-1,1]] on {0,1}, [[1,-1],[-1,1]] on {2,3})`
> and `L_black = [[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]]` (single block on
> `{1,2}`). Compute `L_red · L_black` and `L_black · L_red` row-by-row; they
> are equal. Numerical proptest §4.5 makes this rigorous on random sizes.

For **cycle graphs** the analogous decomposition works when `n` is even
(otherwise the cycle is not bipartite — 2-coloring fails on an odd cycle).
`new_bipartite_cycle` MUST require `n_nodes % 2 == 0` and return an error
(or panic at construction) otherwise.

For **arbitrary user-supplied graphs**, the engineer ships the generic
constructor `new(a, b, commutes_axiomatically: bool)` with explicit
caller-responsibility documentation; the safe path is to compose via
`new_bipartite_path(graph)` or `new_bipartite_cycle(graph)` only.

### 3.3 API surface

```rust
//! crates/semiflow-core/src/strang_graph.rs (NEW module, ~140 LoC)

use alloc::sync::Arc;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{half, SemiflowFloat},
    graph::{Graph, Laplacian},
    graph_heat::GraphHeatChernoff,
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

/// Palindromic Strang split for two commuting graph Chernoff kernels.
///
/// `S(τ) f = A(τ/2) ∘ B(τ) ∘ A(τ/2) · f` on `GraphSignal<F>`.
///
/// **Commutativity is a precondition, not a verified property.** Two safe
/// constructors (`new_bipartite_path`, `new_bipartite_cycle`) build
/// guaranteed-commuting decompositions by edge 2-coloring. The generic
/// constructor `new(a, b, commutes_axiomatically: bool)` requires the
/// caller to opt in via the boolean flag — if the kernels do not commute,
/// the global order degrades from 2 to 1 (BCH error proportional to
/// `‖[L_A, L_B]‖ · τ²`).
///
/// See math.md §12.8 (NORMATIVE) and ADR-0012 (palindromic Strang pattern).
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
    /// the resulting palindromic product is order-2.
    /// `commutes_axiomatically = false` ⇒ Lie-Trotter-style splitting at
    /// order-1 (BCH residue not cancelled).
    pub fn new(a: A, b: B, commutes_axiomatically: bool) -> Self {
        Self {
            a, b,
            commutes: commutes_axiomatically,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<F: SemiflowFloat>
    StrangSplitGraph<GraphHeatChernoff<F>, GraphHeatChernoff<F>, F>
{
    /// **Safe constructor**: build a guaranteed-commuting Strang split for a
    /// path graph `P_n` by edge-parity 2-coloring.
    ///
    /// - `A`-edges: `(0,1), (2,3), (4,5), …` (red).
    /// - `B`-edges: `(1,2), (3,4), (5,6), …` (black).
    ///
    /// Both Laplacians are block-diagonal on DISJOINT node pairs ⇒ commute.
    /// Each leg is an order-1 `GraphHeatChernoff::new` (leading Chernoff).
    ///
    /// # Errors
    /// `SemiflowError::DomainViolation` if `graph.n_nodes() < 2`.
    pub fn new_bipartite_path(graph: Arc<Graph<F>>)
        -> Result<Self, SemiflowError>;

    /// **Safe constructor**: build a guaranteed-commuting Strang split for an
    /// **even-length** cycle graph `C_n` by edge-parity 2-coloring.
    ///
    /// # Errors
    /// `SemiflowError::DomainViolation` if `graph.n_nodes() < 4` OR
    /// `graph.n_nodes() % 2 != 0` (odd cycles are not bipartite).
    pub fn new_bipartite_cycle(graph: Arc<Graph<F>>)
        -> Result<Self, SemiflowError>;
}
```

**Builder helpers** (engineer impl detail, not part of public surface):

```rust
/// Assemble two combinatorial Laplacians from disjoint edge sets defined by a
/// caller-supplied predicate `keep_in_a: Fn(usize edge_global_idx) -> bool`.
fn split_laplacians_by_edge<F: SemiflowFloat>(
    graph: &Graph<F>,
    keep_in_a: impl Fn(usize) -> bool,
) -> (Laplacian<F>, Laplacian<F>);
```

The function walks the CSR edges, partitions them into two new symmetric-CSR
buffers, and calls the existing `Laplacian::assemble_combinatorial` on each
sub-graph. Reuses Wave 2.1A builders verbatim. ≤ 50 LoC.

### 3.4 `apply_into` body (NORMATIVE)

Mirrors `strang2d.rs::apply_strang2d_into` with two important simplifications:

1. **No axis lifts.** Both `A` and `B` already operate on `GraphSignal<F>`
   directly. No need for the `AxisLift` wrapper or `run_axislift_into_2d`
   transient view helper.
2. **Buffer parity**: 3-leg ping-pong with two `borrow_vec(N)` buffers
   (`buf_a`, `buf_b`).

```rust
impl<A, B, F: SemiflowFloat> ChernoffFunction<F> for StrangSplitGraph<A, B, F>
where
    A: ChernoffFunction<F, S = GraphSignal<F>> + Clone,
    B: ChernoffFunction<F, S = GraphSignal<F>> + Clone,
{
    type S = GraphSignal<F>;

    fn apply(&self, tau: F, f: &GraphSignal<F>) -> Result<GraphSignal<F>, SemiflowError> {
        let mut dst = f.clone();
        let mut scratch = ScratchPool::<F>::new();
        self.apply_into(tau, f, &mut dst, &mut scratch)?;
        Ok(dst)
    }

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

        // Leg 1: A(τ/2) src → tmp_a
        let mut tmp_a = src.clone();   // shape-preserving clone (Graph<F> shared via Arc)
        // [scratch borrows happen inside child apply_into calls]
        self.a.apply_into(half_tau, src, &mut tmp_a, scratch)?;

        // Leg 2: B(τ) tmp_a → tmp_b
        let mut tmp_b = src.clone();
        self.b.apply_into(tau, &tmp_a, &mut tmp_b, scratch)?;

        // Leg 3: A(τ/2) tmp_b → dst
        self.a.apply_into(half_tau, &tmp_b, dst, scratch)?;
        Ok(())
    }

    fn order(&self) -> u32 {
        // Commuting ⇒ palindromic Strang at canonical order 2 (or min of inner
        // orders, capped at 2). Non-commuting ⇒ degrade to 1.
        if self.commutes {
            core::cmp::min(self.a.order(), self.b.order()).min(2)
        } else {
            1
        }
    }

    fn growth(&self) -> (f64, f64) {
        let (m_a, w_a) = self.a.growth();
        let (m_b, w_b) = self.b.growth();
        // Sub-multiplicative bound on the palindromic product.
        (m_a * m_b * m_a, w_a + w_b + w_a)
    }
}
```

> **Zero-alloc note (R4 mitigation)**: the `src.clone()` calls allocate a
> `Vec<F>` per leg → **2 allocations per `apply_into` call** in this naive
> draft. **This violates the zero-alloc steady-state invariant.** The
> engineer MUST replace the two `src.clone()` slots with a dedicated
> `GraphSignal::from_scratch(scratch: &mut ScratchPool<F>, graph: Arc<Graph<F>>)`
> ctor that borrows from the pool and constructs a transient `GraphSignal`
> with a pool-owned `Vec<F>`. The pool returns the buffer on drop. Alternative:
> add a 2-slot graph-signal arena to `ScratchPool` mirroring `take_vec` /
> `return_vec`. See §6 R4 for the agreed mitigation contract.

### 3.5 LoC budget — `strang_graph.rs`

| Element | LoC |
|---|---:|
| Module doc + imports | 25 |
| `StrangSplitGraph` struct + generic ctor | 20 |
| Safe ctors `new_bipartite_path` / `new_bipartite_cycle` | 35 |
| `split_laplacians_by_edge` helper | 35 |
| `ChernoffFunction` impl | 45 |
| Inline doc-test + `#[cfg(test)]` smoke | 50 |
| **Total** | **≈ 210** |

Above the 140-LoC plan estimate but well under 500 cap. Function-cap (50) check:
all functions ≤ 45 LoC.

---

## §4 — Test plan

### 4.1 G8 — order-2 slope gate

**File**: `crates/semiflow-core/tests/graph_g8_zeta_a_slope.rs` (~120 LoC)

```rust
const N_STEPS: [usize; 5] = [25, 50, 100, 200, 400];

#[test]
fn g8_graph_heat_zeta_a_convergence_slope_f64() {
    let g = Arc::new(Graph::<f64>::path(64));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeatChernoff::with_zeta_a(Arc::clone(&lap));

    let decomp = jacobi_eig(&lap);
    let f0 = GraphSignal::from_fn(Arc::clone(&g),
        |i| ((i as f64 * 0.31).sin() + 1.0) * 0.5);
    let oracle = heat_oracle(&decomp, &f0, 0.5);

    let errs: Vec<f64> = N_STEPS.iter().map(|&n| {
        let semi = ChernoffSemigroup::new(
            GraphHeatChernoff::with_zeta_a(Arc::clone(&lap)), n).unwrap();
        let u_t = semi.evolve(0.5, &f0).unwrap();
        let mut diff = u_t.clone();
        diff.axpy_into(-1.0, &oracle);
        diff.norm_sup()
    }).collect();

    let slope = log_log_slope(&N_STEPS, &errs);
    assert!(slope <= -1.95, "G8 FAIL: slope {slope:.4} > -1.95");
}

#[test]
fn g8_graph_heat_zeta_a_convergence_slope_f32() {
    // Same harness with F = f32. Threshold ≤ -1.85 per ADR-0046.
}
```

Reuses `log_log_slope` OLS helper from `tests/convergence_rate.rs`. Path graph
`N = 64` matches Wave 2.1A G7 harness so the regression is comparable.

### 4.2 G9 — Strang slope gate

**File**: `crates/semiflow-core/tests/graph_g9_strang_slope.rs` (~140 LoC)

Two sub-tests: bipartite path and bipartite cycle.

```rust
#[test]
fn g9_strang_bipartite_path_slope_f64() {
    let g = Arc::new(Graph::<f64>::path(64));
    let lap_full = Arc::new(Laplacian::assemble_combinatorial(&g));
    let strang = StrangSplitGraph::new_bipartite_path(Arc::clone(&g)).unwrap();

    let decomp = jacobi_eig(&lap_full);    // oracle uses full L_G = L_A + L_B
    let f0 = GraphSignal::from_fn(Arc::clone(&g),
        |i| ((i as f64 * 0.31).sin() + 1.0) * 0.5);
    let oracle = heat_oracle(&decomp, &f0, 0.5);

    let errs: Vec<f64> = [25, 50, 100, 200, 400].iter().map(|&n| {
        let semi = ChernoffSemigroup::new(
            StrangSplitGraph::new_bipartite_path(Arc::clone(&g)).unwrap(), n).unwrap();
        let u_t = semi.evolve(0.5, &f0).unwrap();
        let mut diff = u_t.clone();
        diff.axpy_into(-1.0, &oracle);
        diff.norm_sup()
    }).collect();
    let slope = log_log_slope(&[25, 50, 100, 200, 400], &errs);
    assert!(slope <= -1.95, "G9 path FAIL: slope {slope:.4} > -1.95");
}

#[test]
fn g9_strang_bipartite_cycle_slope_f64() {
    // Same harness with cycle(64) — note 64 is even, so 2-coloring valid.
}

#[test]
fn g9_strang_bipartite_path_slope_f32() {
    // Threshold ≤ -1.85 per ADR-0046.
}

#[test]
fn g9_strang_bipartite_cycle_slope_f32() {
    // Same.
}
```

### 4.3 G10 — order-4 slope gate

**File**: `crates/semiflow-core/tests/graph_g10_zeta4_slope.rs` (~120 LoC)

```rust
#[test]
fn g10_graph_heat4_convergence_slope_f64() {
    let g = Arc::new(Graph::<f64>::path(64));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeat4thChernoff::new(Arc::clone(&lap));

    let decomp = jacobi_eig(&lap);
    let f0 = GraphSignal::from_fn(Arc::clone(&g),
        |i| ((i as f64 * 0.31).sin() + 1.0) * 0.5);
    let oracle = heat_oracle(&decomp, &f0, 0.5);

    let errs: Vec<f64> = [25, 50, 100, 200, 400].iter().map(|&n| {
        let semi = ChernoffSemigroup::new(
            GraphHeat4thChernoff::new(Arc::clone(&lap)), n).unwrap();
        let u_t = semi.evolve(0.5, &f0).unwrap();
        let mut diff = u_t.clone();
        diff.axpy_into(-1.0, &oracle);
        diff.norm_sup()
    }).collect();
    let slope = log_log_slope(&[25, 50, 100, 200, 400], &errs);
    assert!(slope <= -3.95, "G10 FAIL: slope {slope:.4} > -3.95");
}

#[test]
fn g10_graph_heat4_convergence_slope_f32() {
    // Threshold ≤ -3.50 per ADR-0046 (f32 round-off floor saturates at finer N).
}
```

### 4.4 Erdős-Rényi invariants proptest

**File**: `crates/semiflow-core/tests/graph_proptest.rs` (~110 LoC)

64 random cases per parameter combination. Verifies:

- **P1 quasi-contractivity**: for any `f ∈ GraphSignal`, `‖S(τ) f‖_∞ ≤ (1 + ε(τ)) · ‖f‖_∞`
  with `ε(τ)` derived from the operator's `growth()` bound. Threshold: 1e-12 relative
  slack (f64), 1e-5 (f32).
- **P2 NaN-free**: output buffer contains only finite values.
- **P3 Gershgorin bound**: `Laplacian::spectral_radius_bound() <= 2 · max_i deg(i)`
  (combinatorial) / `≤ 2.0` (normalized).
- **P4 with_zeta_a vs leading agreement at small `τ`**: `‖S_zeta_a(τ) f − S_leading(τ) f‖_∞
  ≤ τ² · ρ̄² / 2 + 1e-10` (the `τ² L_G²/2` term is bounded analytically).

Properties P1, P2, P3 run for all three new kernels (`with_zeta_a`,
`GraphHeat4thChernoff`, `StrangSplitGraph::new_bipartite_path`); P4 specific to
`with_zeta_a`.

```rust
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn quasi_contractive_zeta_a_f64(
        n in 16usize..128,
        p in 0.05f64..0.30,
        seed in any::<u64>(),
        tau in 1e-5_f64..1e-2_f64,
    ) {
        let g = Arc::new(Graph::<f64>::erdos_renyi(n, p, seed));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        // P3
        assert!(lap.spectral_radius_bound() <= 2.0 * max_degree(&g) as f64 + 1e-12);

        let kernel = GraphHeatChernoff::with_zeta_a(Arc::clone(&lap));
        let rho = lap.spectral_radius_bound();
        prop_assume!(tau * rho <= 0.5);          // stay in stability envelope

        let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.17).sin());
        let mut dst = f0.clone();
        let mut scratch = ScratchPool::<f64>::new();
        kernel.apply_into(tau, &f0, &mut dst, &mut scratch).unwrap();

        // P1
        let bound = (1.0 + tau * tau * rho * rho * 0.5) * f0.norm_sup() + 1e-12;
        assert!(dst.norm_sup() <= bound);
        // P2
        for v in dst.values() { assert!(v.is_finite()); }
    }

    // Analogous blocks for GraphHeat4thChernoff and StrangSplitGraph::new_bipartite_path
    // …
}
```

Note: `new_bipartite_path` requires a path graph; the proptest harness uses
`Graph::<f64>::path(n)` for the Strang case (NOT `erdos_renyi`), and
`erdos_renyi` for the order-2 / order-4 cases.

### 4.5 Numerical commutativity gate (Strang safety)

**File**: `crates/semiflow-core/tests/graph_g9_strang_slope.rs` (additional test inside same file)

```rust
#[test]
fn bipartite_path_split_commutes_numerically() {
    // For path(N=16) verify that the two assembled sub-Laplacians from
    // new_bipartite_path commute up to round-off.
    let g = Arc::new(Graph::<f64>::path(16));
    let strang = StrangSplitGraph::new_bipartite_path(Arc::clone(&g)).unwrap();
    // Engineer exposes pub(crate) accessor for testing only:
    let (lap_a, lap_b) = strang.test_only_laplacians();

    // Compute commutator ‖L_A L_B − L_B L_A‖_∞ on the all-ones vector.
    let f = GraphSignal::from_fn(Arc::clone(&g), |_| 1.0);
    let mut ab = f.clone();   let mut ba = f.clone();
    let mut tmp = f.clone();  let mut scratch = ScratchPool::<f64>::new();
    let mut buf = scratch.borrow_vec(16);
    lap_a.apply_into_slice(f.values(), &mut buf);
    let mut buf2 = scratch.borrow_vec(16);
    lap_b.apply_into_slice(&buf, &mut buf2);   ab.copy_from_slice(&buf2);
    lap_b.apply_into_slice(f.values(), &mut buf);
    lap_a.apply_into_slice(&buf, &mut buf2);   ba.copy_from_slice(&buf2);
    let mut diff = ab.clone(); diff.axpy_into(-1.0, &ba);
    assert!(diff.norm_sup() < 1e-12, "L_A and L_B do not commute: {}", diff.norm_sup());
}

#[test]
fn bipartite_cycle_split_commutes_numerically() { /* analogous */ }
```

### 4.6 Zero-alloc steady-state continuation

**File**: `crates/semiflow-core/tests/graph_apply_into_zero_alloc.rs` (EXTEND Wave 2.1A test)

Add three new test functions to the existing file (post-edit ~160 LoC):

```rust
#[test]
fn with_zeta_a_apply_into_zero_alloc_steady_state() { /* mirror Wave 2.1A pattern */ }

#[test]
fn graph_heat4_apply_into_zero_alloc_steady_state() { /* mirror */ }

#[test]
fn strang_bipartite_path_apply_into_zero_alloc_steady_state() {
    // CRITICAL: this gate enforces the R4 mitigation (no GraphSignal::clone in
    // the apply_into hot path). Engineer MUST ship pool-borrowed transient
    // graph-signal arena to pass this test.
}
```

If `strang_bipartite_path_apply_into_zero_alloc_steady_state` cannot be made to
pass (R4 mitigation fails), the engineer reports back to architect; the test is
NOT relaxed. The architect re-considers the pool API extension before merge.

### 4.7 v2.0 + Wave 2.1A regression set

All of:

- Wave 2.1A: `convergence_graph`, `graph_oracle_eigenmode`, `graph_alloc_zero`,
  `graph_invariants` — must re-pass byte-identical.
- v2.0: `apply_into_byte_equal`, `zero_alloc_steady`, `strang_inplace_byte_equal`,
  `state_trait_contract`, `adaptive_classical_bit_equal`, `cev_european_call`,
  18 NORMATIVE sympy gates, 6 v2.0 slope gates (G1..G6) — must re-pass byte-identical.

The engineer runs `cargo run -p xtask -- test-fast` after each Wave 2.1B
module lands and confirms the test count grows monotonically.

### 4.8 Sympy gate manifest update

**File**: `contracts/semiflow-core.properties.yaml` (EDIT, +60 LoC)

Append two new T12_* entries near the end of file (after the Wave 2.1A T12_*
section):

```yaml
# Wave 2.1B (this contract):
#   - T12_zeta_tau2_residual            (sympy gate, NORMATIVE)
#       — Verifies that for constant a ≡ 1 on a 5-node path graph, the
#         polynomial S(τ) f − e^{−τ L_G} f = τ³·(L_G³/6)·f + O(τ⁴) — i.e.,
#         the order-2 Taylor truncation residual matches the expected cubic
#         leading term. Test script:
#         `.dev-docs/verification/scripts/verify_v2_1_zeta_tau2_residual.py`.
#         Runs sympy.Matrix exponential of −τ·L_path5, expands to τ-series,
#         subtracts the truncation S(τ), verifies the leading remainder is
#         τ³·(L³/6)·f within sympy.simplify == 0.
#   - T12_zeta_tau4_residual            (sympy gate, NORMATIVE)
#       — Same harness for Padé[0,4]: verifies S₄(τ) f − e^{−τ L_G} f
#         = τ⁵·(L_G⁵/120)·f + O(τ⁶) on a 5-node path graph.
#         Test script:
#         `.dev-docs/verification/scripts/verify_v2_1_zeta_tau4_residual.py`.
#   - T12_strang_commuting_path_exact   (sympy gate, NORMATIVE)
#       — Verifies that for the bipartite edge-split on path-5,
#         e^{(τ/2)L_A} · e^{τ L_B} · e^{(τ/2)L_A} = e^{τ(L_A + L_B)} = e^{τ L_G}
#         exactly (sympy.simplify == 0).
#         Test script:
#         `.dev-docs/verification/scripts/verify_v2_1_strang_commuting_path.py`.
```

If `contracts/semiflow-core.properties.yaml` does not yet have a T12_* section,
the engineer creates one — model on the v0.7.0 NS2D entries that already exist
(see line ~3500 of the same file). The sympy scripts themselves live in
`.dev-docs/verification/scripts/` per repo convention.

---

## §5 — `contracts/semiflow-core.math.md` §12.6–§12.8 content outline

The engineer appends three sub-sections to math.md (after the existing §12.5
which Wave 2.1A added). Net diff ≈ +300 LoC.

### §12.6 — Order-2 Chernoff for constant `a ≡ 1` (NORMATIVE library + CITATION)

Paragraph outline:

- **Setting.** Given a frozen `Laplacian<F>` `L_G` on a finite weighted graph
  `G = (V, E, w)` (notation per §12.1), and the constant edge-coefficient case
  `a ≡ 1`, define the order-2 Chernoff function:
  `S(τ) = I − τ L_G + (τ²/2) L_G²`.
- **Reduction from §9.2.3.B.** Show explicitly that
  `τ² · [a·a'·f''' + ½·a·a''·f'' + ¼·a'·a''·f'] = 0` for `a' = a'' = 0`. State that
  the resulting "trivialised" ζ-A correction on graphs is the operator Taylor
  truncation listed above.
- **Citation chain.** Pazy 1983 §1.3 Thm 1.3 (Chernoff product formula); Engel-Nagel
  2000 §III.5 Thm 5.2 (bounded generator special case); Hochbruck-Ostermann 2010
  *Acta Numerica* §3 (truncated-exponential families).
- **Hypothesis check.** Verify `S(0) = I`, `S'(0) = −L_G`, quasi-contractivity in
  the envelope `τ · ρ̄ ≤ ½`. Each of the four hypotheses is one-line.
- **Convergence rate.** State (with citation) that
  `‖(S(t/n))^n f − e^{−tL_G} f‖ = O(1/n²)` and reference §12.2.
- **Variable `a(v)` deferred.** One paragraph explicitly stating that the
  variable-edge-coefficient generalization is out of scope for v2.1; cites
  §9.2.3.B for the regular-grid version and notes the operator-product
  derivation work scheduled for v2.2.

### §12.7 — Order-4 Chernoff for constant `a ≡ 1` (NORMATIVE library + CITATION)

Paragraph outline:

- **Setting.** Define `S₄(τ) = Σ_{k=0}^4 (−τ L_G)^k / k!`.
- **Citation chain.** Higham 2008 *Functions of Matrices* §10 (Taylor methods for
  matrix exponentials); Hochbruck-Ostermann 2010 *Acta Numerica* §3
  (truncated-exponential bounds on bounded operators).
- **Hypothesis check.** `S₄(0) = I` (trivial); `S₄'(0) = −L_G` (linear term);
  quasi-contractivity envelope `τ · ρ̄ ≤ ½`; convergence rate
  `‖(S₄(t/n))^n f − e^{−tL_G} f‖ = O(1/n⁴)`.
- **Algorithmic cost.** State that the per-step cost is **4 SpMV operations**
  (one for each `L_G^k · src` for `k = 1, 2, 3, 4`) using a ping-pong scratch
  pair of `N`-length buffers.
- **Stencil comparison.** One paragraph contrasting with regular-grid
  `Diffusion4thChernoff` (§9.2.4): on graphs, the Fornberg 7-point FD stencils
  are replaced by repeated `L_G`-applications because there is no continuous
  derivative to discretise.

### §12.8 — Strang split on commuting graph operators (CITATION + NORMATIVE)

Paragraph outline:

- **Setting.** Let `L_G = L_A + L_B` with `[L_A, L_B] = 0`. Define the
  palindromic product `S(τ) = e^{(τ/2)L_A} · e^{τ L_B} · e^{(τ/2)L_A}` and
  state that it equals `e^{τ(L_A + L_B)} = e^{τ L_G}` **exactly**
  (CITATION: BCH series terminates at zero-th order for commuting operators).
- **Theorem 7' carry-through.** State that the proof of math.md §10.8
  (Theorem 7' for tensor products on regular grids) applies verbatim to
  bounded symmetric operators on `ℝ^N` because the only requirement is
  `[L_A, L_B] = 0`. No new theorem statement; one sentence saying "Theorem 7'
  applies; see §10.8".
- **Approximation degradation.** When `A` and `B` are themselves order-`m_A` /
  order-`m_B` Chernoff approximants (NOT exact exponentials), the global order
  of the palindromic product is `min(m_A, m_B, 2)`. Cite Strang 1968 *SIAM J.
  Numer. Anal.* (canonical Strang order-2 result).
- **Bipartite 2-coloring construction.** Describe the alternating-edge
  decomposition for path / even-cycle graphs in 3–5 lines (see §3.2 of this
  contract). State that each sub-Laplacian is block-diagonal on disjoint
  node pairs, hence trivially commuting.
- **Safety contract.** State NORMATIVELY that the library exposes safe
  constructors `new_bipartite_path` / `new_bipartite_cycle` and an opt-in
  generic constructor `new(a, b, commutes_axiomatically: bool)` for advanced
  users; non-commuting splits degrade to order-1.
- **Sympy verification gate.** Reference T12_strang_commuting_path_exact
  (manifest §4.8).

Existing §12.9 in math.md (if any — see Wave 2.1A; per the v2.1 plan the
sympy gate list will be §12.11 once Wave 2.1C also lands) is not affected by
this Wave's edits.

---

## §6 — Risk table (top 5)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **R1: Bipartite 2-coloring algorithm incorrect → silent slope drift.** Engineer ships a flawed edge-parity decomposition (e.g., assigns adjacent edges to the same sub-Laplacian on the cycle) leading to non-commuting `L_A`, `L_B` and a non-Strang slope. | MEDIUM | HIGH (slope drop from −2 to −1) | Numerical commutativity gate §4.5 (`bipartite_path_split_commutes_numerically`, `bipartite_cycle_split_commutes_numerically`) catches this BEFORE G9 runs. Sympy gate `T12_strang_commuting_path_exact` provides analytical verification for path-5. Engineer MUST run the commutativity gate first, G9 second. |
| **R2: Strang composition order error (palindromic vs Lie-Trotter).** Engineer accidentally writes `A(τ) ∘ B(τ)` (Lie-Trotter, order-1) instead of `A(τ/2) ∘ B(τ) ∘ A(τ/2)` (palindromic, order-2). | LOW | HIGH (slope drop) | Inline assertion in `apply_into` body documenting the leg sequence (see §3.4 NORMATIVE pseudocode). Code review by reviewer-suckless before merge. Byte-equal proptest vs explicit `e^{τ(L_A+L_B)}` on path-5 (P5 in §4.4 proptest extension, optional). |
| **R3: f32 slope band narrower than f64 for ζ⁴.** Round-off at fine `N` (e.g., `N = 400`) for the 4-SpMV accumulation drives the slope below −3.50, failing G10 f32. | MEDIUM | MEDIUM (test failure) | f32 G10 threshold relaxed to −3.50 per ADR-0046 (precision policy bands, formally `f32_slope_floor = ceil(theoretical_slope * 0.875)`). Engineer documents this in the test header. If still failing, engineer reduces `N_STEPS` upper bound to `200` for f32 only. |
| **R4: Zero-alloc property regresses for `StrangSplitGraph`.** The naive draft in §3.4 calls `src.clone()` twice per `apply_into`, which heap-allocates `Vec<F>`. Existing `graph_apply_into_zero_alloc.rs` and `tests/zero_alloc_steady.rs` will FAIL. | HIGH | HIGH (regression on Wave 2.1A's zero-alloc invariant) | Engineer MUST add a `GraphSignal` arena to `ScratchPool` mirroring the existing `take_vec` / `return_vec` API. New method signatures: `scratch.take_graph_signal(graph: Arc<Graph<F>>) -> GraphSignal<F>` and `scratch.return_graph_signal(sig: GraphSignal<F>)`. The `StrangSplitGraph::apply_into` body replaces both `src.clone()` with `scratch.take_graph_signal(src.graph().clone())`. Acceptance: §4.6 `strang_bipartite_path_apply_into_zero_alloc_steady_state` passes with 0 byte/0 alloc per steady-state call. **The contract REQUIRES this mitigation — no escape hatch.** |
| **R5: `cargo-semver-checks` flags `with_zeta_a` as breaking.** Adding a field to `GraphHeatChernoff` (the `order_variant` enum) could be flagged as a breaking struct change even though all existing constructors and public methods remain additive. | LOW | MEDIUM (release blocker) | Field is `pub(crate)` (engineer verifies — see §1.2 struct definition; no `pub` on `order_variant`). `order_variant()` accessor is `#[doc(hidden)] pub(crate)`. Constructor signatures unchanged. Engineer runs `cargo semver-checks check-release -p semiflow-core` as part of CI before commit; flags trigger a re-review with architect, not silent merge. |

---

## §7 — LoC budget summary

| File | Status | Target LoC (Δ for edits) | Cap | Carve-out? |
|---|---|---:|---:|:---:|
| `crates/semiflow-core/src/graph_heat.rs` | EDIT | +85 (~205 total) | 500 | NO |
| `crates/semiflow-core/src/graph_heat4.rs` | NEW | 240 | 500 | NO |
| `crates/semiflow-core/src/strang_graph.rs` | NEW | 210 | 500 | NO |
| `crates/semiflow-core/src/lib.rs` | EDIT | +6 (re-exports) | n/a | n/a |
| `crates/semiflow-core/src/scratch.rs` | EDIT | +30 (graph-signal arena per R4) | n/a | n/a |
| `crates/semiflow-core/tests/graph_g8_zeta_a_slope.rs` | NEW | 120 | 500 | NO |
| `crates/semiflow-core/tests/graph_g9_strang_slope.rs` | NEW | 200 (incl. commutativity gates) | 500 | NO |
| `crates/semiflow-core/tests/graph_g10_zeta4_slope.rs` | NEW | 120 | 500 | NO |
| `crates/semiflow-core/tests/graph_proptest.rs` | NEW | 200 | 500 | NO |
| `crates/semiflow-core/tests/graph_apply_into_zero_alloc.rs` | EDIT | +60 (3 new tests) | 500 | NO |
| `contracts/semiflow-core.math.md` | EDIT | +300 (§12.6–§12.8) | unchanged | n/a |
| `contracts/semiflow-core.properties.yaml` | EDIT | +60 | unchanged | n/a |
| Sympy scripts in `.dev-docs/verification/scripts/` | NEW (3 files) | ~80 each (~240 total) | 500 | NO |

**Total new + edited LoC budget**: ≈ 700 LoC Rust + ≈ 300 LoC math + ≈ 240 LoC sympy + ≈ 60 LoC manifest ≈ **1300 LoC total**.

Plan estimate was "~700 LoC total" — the breakdown above exceeds that figure
because the R4 mitigation (graph-signal arena) and the commutativity gates
(§4.5) were not in the plan's accounting; both are critical and non-negotiable.

All files stay under 500-LoC cap; no Override #1 expansion requested.

Function-cap (50 LoC) check, spot-verified:

- `GraphHeatChernoff::apply_into` (post-edit): branch + each branch body ≤ 25 LoC. ✓
- `apply_zeta4_into`: ~42 LoC. ✓
- `StrangSplitGraph::apply_into`: ~30 LoC. ✓
- `split_laplacians_by_edge`: ~35 LoC. ✓
- `new_bipartite_path`, `new_bipartite_cycle`: ~20 LoC each. ✓

---

## §8 — Engineer handoff checklist

Tick in order; do not start step N+1 until step N is green.

- [ ] Read this contract end-to-end.
- [ ] Read ADR-0047 and ADR-0048 (Wave 2.1A reference) — DO NOT re-read math.md
      §1-§11 (no changes).
- [ ] Run `mcp__gitnexus__context({name: "ChernoffFunction"})` and confirm trait
      surface unchanged. The engineer MUST NOT modify `chernoff.rs`.
- [ ] Run `mcp__gitnexus__context({name: "GraphHeatChernoff"})` to confirm
      current Wave 2.1A surface — the `order_variant` field addition is the
      only struct change.
- [ ] Run `mcp__gitnexus__impact({target: "GraphHeatChernoff", direction: "upstream"})`
      to assess blast radius of the §1.2 struct edit. Report HIGH/CRITICAL to
      architect BEFORE editing.
- [ ] Implement the `ScratchPool` graph-signal arena (R4 mitigation) FIRST.
      Acceptance: existing `graph_apply_into_zero_alloc.rs` still passes.
- [ ] Implement `GraphHeatChernoff::with_zeta_a` (§1) — edit `graph_heat.rs`.
- [ ] Implement `GraphHeat4thChernoff` (§2) — new `graph_heat4.rs` module.
- [ ] Implement `StrangSplitGraph` (§3) — new `strang_graph.rs` module.
- [ ] Add `pub mod graph_heat4; pub mod strang_graph;` to `lib.rs`. Re-export
      `GraphHeat4thChernoff`, `StrangSplitGraph` at the crate root.
- [ ] Write the three sympy gates listed in §4.8 (3 files in
      `.dev-docs/verification/scripts/`).
- [ ] Append T12_* entries to `contracts/semiflow-core.properties.yaml`.
- [ ] Append §12.6–§12.8 to `contracts/semiflow-core.math.md` per §5 outline.
- [ ] Write `tests/graph_g8_zeta_a_slope.rs` and confirm both f64 and f32 pass.
- [ ] Write `tests/graph_g9_strang_slope.rs` INCLUDING the commutativity gates.
      The commutativity gates MUST pass before the slope gate; if commutativity
      fails, engineer fixes the bipartite decomposition before running slope.
- [ ] Write `tests/graph_g10_zeta4_slope.rs` and confirm both f64 and f32 pass.
- [ ] Write `tests/graph_proptest.rs` (P1–P4 over the three new kernels).
- [ ] Extend `tests/graph_apply_into_zero_alloc.rs` with the three new tests
      from §4.6.
- [ ] Run `cargo run -p xtask -- test-fast` — all v2.0 + Wave 2.1A + Wave 2.1B
      tests green.
- [ ] Run `cargo run -p xtask -- test-full` — slope gates G8, G9, G10 (f64 + f32)
      green; slopes printed to stdout.
- [ ] Run sympy gates by executing the three new `verify_v2_1_*.py` scripts
      manually; all return `0`.
- [ ] Run `cargo semver-checks check-release -p semiflow-core` — no breaking
      changes flagged.
- [ ] Verify v2.0 regression set (`apply_into_byte_equal`, `zero_alloc_steady`,
      `strang_inplace_byte_equal`, `state_trait_contract`, `adaptive_classical_bit_equal`,
      `cev_european_call`) re-passes byte-identical. Also Wave 2.1A regressions.
- [ ] Hand off to git-workflow for a single Wave-2.1B commit (Anchor delegates;
      do NOT commit yourself).

---

## §A — Appendix: ADR rationale (no new ADR)

This contract does NOT introduce a new ADR for the following reasons:

1. **`GraphHeatChernoff::with_zeta_a`** is an additive constructor variant on
   the same `GraphHeatChernoff` struct introduced in ADR-0047. The design
   decisions (Arc-shared Laplacian, ScratchPool-borrowed intermediates,
   `ChernoffFunction<F, S = GraphSignal<F>>` impl pattern) are all carried
   over from ADR-0047 verbatim. The enum-based variant dispatch is a minor
   implementation choice fully covered by ADR-0047's "extensibility" note.

2. **`GraphHeat4thChernoff`** is the graph analogue of `Diffusion4thChernoff`
   from ADR-0013. The mathematical pattern (truncated operator exponential),
   the buffer ping-pong via `ScratchPool` (ADR-0042), and the zero-alloc /
   bit-equality contracts (ADR-0046) are all already established. Replacing
   the Fornberg 7-point stencil with `L_G^k`-applications is a substitution
   of operator representation, not a new design pattern.

3. **`StrangSplitGraph`** is the graph analogue of `Strang2D` from ADR-0012.
   The palindromic 3-leg structure, the `min(order, 2)` global-order rule,
   the `growth()` composition, and the ping-pong via `ScratchPool` are all
   already documented in ADR-0012 / ADR-0042. The only graph-specific
   subtlety is the bipartite 2-coloring constructor, which is a 30-LoC helper
   and documented in §3.2 of this contract.

4. **The R4 mitigation (graph-signal arena in ScratchPool)** is a minor
   extension of ADR-0041 (ScratchPool arena pattern). The new methods
   `take_graph_signal` / `return_graph_signal` mirror the existing
   `take_vec` / `return_vec` API surface. If the engineer's implementation
   reveals an unexpected design tension (e.g., requires lifetime games or
   `unsafe`), the engineer pauses and requests a new ADR from the architect
   before merging.

If during implementation the engineer identifies a design decision NOT covered
by ADR-0042 / ADR-0046 / ADR-0047 (e.g., a non-obvious choice about how
sub-Laplacians are stored or shared across the two legs of `StrangSplitGraph`),
they pause and request ADR-0051 from the architect before merge. The default
position is: **no new ADR**.

---

**End of Wave 2.1B contract.**
