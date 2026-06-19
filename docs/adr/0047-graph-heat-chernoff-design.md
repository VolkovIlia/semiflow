# ADR-0047 — `GraphHeatChernoff<F>`: order-1 leading Chernoff for `∂ₜu = −L_G u`

- **Status**: PROPOSED
- **Date**: 2026-05-20
- **Wave**: v2.1 Wave A (Graph PDE Foundations)
- **Supersedes / amends**: spike crate `remizov-graph-spike` (deleted upon ship — see §"Spike retirement")
- **Companion ADRs**: ADR-0048 (CSR storage), ADR-0049 (math.md §12 introduction), ADR-0050 (test-only Jacobi oracle)

## Decision

Introduce `GraphHeatChernoff<F: SemiflowFloat>` as the first first-class graph-PDE Chernoff
function in `semiflow-core`, implementing the **leading-order** Chernoff approximation

```
S(τ) f = f − τ · L_G · f      (1)
```

for the discrete heat semigroup `e^{−tL_G}` on a finite weighted graph `G = (V, E, w)`
with combinatorial Laplacian `L_G = D − W` (or normalized `L_sym = I − D^{−½} W D^{−½}`).

The constructor stores an `Arc<Laplacian<F>>` (cheap clone, Send+Sync) and implements
`ChernoffFunction<F, S = GraphSignal<F>>` reusing the Wave 1 `ScratchPool<F>::borrow_vec`
machinery: every `apply_into` call borrows ONE `len = N` scratch vector to compute
`L_G · f`, then writes `dst ← src − τ · (L_G · f)` via `dst.copy_from(src);
dst.axpy_into(−τ, &tmp_as_GraphSignal)`. Zero heap allocations in steady state.

The `ChernoffFunction` trait surface is **UNCHANGED** — `GraphHeatChernoff<F>` slots
into the existing `apply` / `apply_into` / `order` / `growth` quartet. No new associated
items, no new bounds, no new dispatch path. Wave 1/2/3/4/5 byte-equal regression set
re-passes verbatim.

## Rationale

### Why order-1 leading Chernoff in Wave 2.1A?

Per Theorem 6 (Remizov 2025), an order-1 Chernoff function `S(τ)` with `S(0) = I`,
`S'(0) = −L_G`, and quasi-contractivity `‖S(τ)‖ ≤ M e^{ωτ}` guarantees

```
‖(S(t/n))^n f − e^{−tL_G} f‖  ≤  C(t) / n      (Theorem 6, inequality (9))
```

Equation (1) satisfies all three hypotheses on the finite-dimensional Hilbert space
`ℝ^N`:

- `S(0) = I` — trivially.
- `S'(0) = −L_G` — by construction.
- `‖I − τL_G‖ ≤ 1` for `0 ≤ τ ≤ ½ · ρ(L_G)^{−1}` where `ρ(L_G)` is the spectral
  radius (Engel-Nagel 2000 §III.5 Thm 5.2; Chung 1997 §1.3 for combinatorial /
  normalized variant). Per ADR-0048, the constructor computes a Gershgorin upper
  bound `ρ̄ ≥ ρ(L_G)`; callers MUST respect `τ ≤ ½ ρ̄^{−1}` or the leading-order
  Chernoff is not quasi-contractive. This is documented in `growth()` as
  `(1.0, ρ̄)` — `M = 1`, `ω = ρ̄` (gives the linear bound `‖S(τ)‖ ≤ 1 + τω` which
  exponentially-bounds to `e^{ωτ}` for τ in the stability window).

Order-1 suffices to ship a clean Wave 2.1A foundation: the slope gate G7 (≤ −0.95)
verifies global O(1/n) convergence empirically against the Jacobi eigendecomposition
oracle (ADR-0050). Higher-order variants (ζ-A diffusion-style τ²-correction with
const `a ≡ 1`, Strang splittings on bipartite graphs, Magnus K=4) are deliberately
DEFERRED to Wave 2.1B / Wave 2.1C — they introduce neighbour-of-neighbour stencils
and second-difference operators that need their own scratch-buffer policy and
contract documentation.

### Why `Arc<Laplacian<F>>`?

The Laplacian is **frozen post-assembly** (immutable CSR — see ADR-0048). Cheap clone
through `Arc::clone` lets `GraphHeatChernoff<F>` be embedded in future composition
types (`GraphStrang<…>` in Wave 2.1B) without copying the CSR arrays. `Arc<T>` is
already in the `alloc` crate so the `no_std + alloc` posture is preserved (no new
direct dep).

### Why one scratch borrow per step?

`L_G · f` requires one full pass over the CSR (read `f[col_idx[k]]`, accumulate into
`tmp[row]`). One `borrow_vec(N)` covers it. Steady-state allocation count: **0
bytes/step** after warmup (verified by an `allocation-counter`-style gate mirroring
the Wave 1 `apply_into_byte_equal` template).

### Math fidelity (no new theorems)

§12 in `contracts/semiflow-core.math.md` is **CITATION + NORMATIVE library choices
only**:

- **CITATION**: Pazy 1983 §1.3 Thm 1.3 (Chernoff product formula); Engel-Nagel 2000
  §III.5 Thm 5.2 (semigroup approximation on Banach spaces; restated for `ℝ^N`);
  Chung 1997 §1.2–1.3 (combinatorial Laplacian eigenvalue bounds).
- **NORMATIVE library choices**: CSR row-major layout, ½ stability bound,
  normalization choice (combinatorial default; symmetric-normalized opt-in).

ADR-0049 makes this classification policy explicit.

## API surface

### Public type

```rust
//! `crates/semiflow-core/src/graph_heat.rs`
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

/// Order-1 leading Chernoff function for `∂ₜu = −L_G u`.
///
/// `S(τ) f = f − τ · L_G · f`. See `contracts/v2.1/wave-a-graph-foundations.md` §4
/// and `contracts/semiflow-core.math.md` §12.2 (CITATION: Pazy 1983 §1.3 Thm 1.3;
/// Engel-Nagel 2000 §III.5 Thm 5.2; Chung 1997 §1.2–1.3).
pub struct GraphHeatChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
}

impl<F: SemiflowFloat> GraphHeatChernoff<F> {
    /// Construct from a frozen Laplacian.
    ///
    /// `Arc` clone is cheap; the same `Laplacian<F>` instance may back multiple
    /// `GraphHeatChernoff<F>` (e.g. when wrapped by a future `GraphStrang` in
    /// Wave 2.1B).
    pub fn new(laplacian: Arc<Laplacian<F>>) -> Self {
        Self { laplacian }
    }

    /// Owned-Laplacian convenience: wraps in `Arc` internally.
    pub fn from_owned(laplacian: Laplacian<F>) -> Self {
        Self { laplacian: Arc::new(laplacian) }
    }

    /// Borrow the underlying Laplacian (debug + composition only).
    pub fn laplacian(&self) -> &Laplacian<F> {
        &self.laplacian
    }
}

impl<F: SemiflowFloat> ChernoffFunction<F> for GraphHeatChernoff<F> {
    type S = GraphSignal<F>;

    fn apply(&self, tau: F, f: &Self::S) -> Result<Self::S, SemiflowError>
    where Self::S: Clone {
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
        // 1. Validate τ.
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "GraphHeatChernoff: tau must be finite and >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        // 2. Borrow scratch for L_G · src.
        let n = src.len();
        let mut lap = scratch.borrow_vec(n);
        self.laplacian.apply_into_slice(src.values(), &mut lap);
        // 3. dst ← src; dst -= τ * lap  (via in-place axpy on a wrapped view).
        dst.copy_from(src);
        dst.axpy_into_slice(-tau, &lap);   // helper on GraphSignal — see §2
        Ok(())
    }

    fn order(&self) -> u32 { 1 }

    fn growth(&self) -> (f64, f64) {
        // M = 1, ω = ρ̄ (Gershgorin spectral-radius upper bound).
        (1.0, self.laplacian.spectral_radius_bound().to_f64().unwrap_or(f64::INFINITY))
    }
}
```

### Composition with `ChernoffSemigroup`

No change required to `chernoff.rs`. The existing executor

```rust
ChernoffSemigroup::<GraphHeatChernoff<f64>, GraphSignal<f64>>::new(func, n)?;
let u_t = semi.evolve(t, &u0)?;
```

just works because `ChernoffSemigroup<C, S>` is already generic over `C:
ChernoffFunction<f64, S = S>`, `S: State<f64> + Clone`. Wave 2.1A adds `Clone` to
`GraphSignal<F>` via `#[derive(Clone)]` (one `Vec<F>` allocation per clone —
explicit, allowed by ADR-0043 since `State<F>` no longer requires `Clone`).

## Out of scope (deferred)

- **ζ-A constant-`a` τ²-correction on graphs** — Wave 2.1B. Per the AskUserQuestion
  resolution recorded in this Wave's planning, Wave 2.1A ships ONLY the order-1
  leading Chernoff. Wave 2.1B will introduce `GraphDiffusionChernoffConstA<F>` with
  `a ≡ 1` and order 2.
- **Strang splittings on bipartite graphs** — Wave 2.1B (needs the order-2 building
  block from above).
- **Magnus K=4 graph Chernoff** — Wave 2.1C. Needs commutator analysis on graphs and
  a longer scratch chain (3–4 buffers/step).
- **Anisotropic / time-dependent edge weights** — Wave 2.2 (requires `with_closure`-
  style API, ADR-0034).
- **Variable Laplacian re-assembly mid-evolution** — never (Laplacian is frozen by
  ADR-0048).

## Acceptance criteria

1. **G7 slope gate** ≤ −0.95 over `N_VALUES = {25, 50, 100, 200, 400}` time-steps
   on a 64-node random graph (Erdős–Rényi `p = 0.15`, seed pinned), comparing
   `GraphHeatChernoff<f64>` iterated by `ChernoffSemigroup` against the Jacobi
   eigendecomposition oracle (ADR-0050) at `t = 0.5`. Test file:
   `crates/semiflow-core/tests/convergence_graph.rs`.
2. **Zero-alloc steady-state gate** — `allocation-counter::measure` around the
   inner Chernoff loop reports `bytes_total_acc == 0` after the first warmup step.
   Test file: `crates/semiflow-core/tests/graph_apply_into_zero_alloc.rs`.
3. **Oracle eigenmode parity** — applying `GraphHeatChernoff` to a Laplacian
   eigenvector `φ_k` for `n = 1` Chernoff steps yields `(1 − τ·λ_k) φ_k` to
   `1e-12` (f64) / `1e-5` (f32). Test file: `tests/graph_heat_oracle.rs`
   (migrated from the spike crate's smoke tests, expanded).
4. **f32 + f64 compile-and-run** — the slope gate (relaxed to ≤ −0.90 for f32 per
   ADR-0046 precision policy) and oracle eigenmode parity both pass for `F = f32`.
5. **v2.0 byte-equal regression set re-passes** unchanged:
   `apply_into_byte_equal` 6/6, `strang_inplace` 7/7, `state_trait` 10/10,
   `adaptive_classical` 4/4, `cev_european_call` 2/2.
6. **All 18 NORMATIVE sympy + 6 v2.0 slope gates** re-pass.
7. **File cap**: every new file ≤ 500 LoC. No carve-outs.
8. **Function cap**: every new function ≤ 50 LoC.
9. **`unsafe_code = "deny"`** preserved workspace-wide.
10. **No new direct deps in `semiflow-core`** — still 2 direct (num-traits, libm).
    Test-only `#[cfg(test)]` Jacobi solver is a hand-rolled rotation routine, no
    new dev-dep (ADR-0050).

## Spike retirement

After AC-3 (oracle eigenmode parity) passes on the production types, delete:

- `crates/remizov-graph-spike/` (entire crate, 333 LoC + Cargo.toml + tests/)
- Workspace `Cargo.toml` member list entry `"crates/remizov-graph-spike"`

The spike's findings are captured in this ADR and ADR-0048/0049/0050; no
information lost. See contract §10 for the deletion checklist.

## Open questions (none — all resolved pre-Wave 2.1A)

| Question | Resolution |
|----------|------------|
| `Arc` vs owned? | `Arc<Laplacian<F>>` for cheap composition (above). |
| `Vec`- vs `BTreeMap`-backed signal? | `Vec` (spike finding: O(1) access beats O(log N) BTreeMap for dense graphs). |
| Boundary conditions? | Implicit Dirichlet zero via empty `neighbours()` iter (inherited from `Discrete<F>` axiom, `state.rs:174`). Explicit BC enum deferred to v2.x. |
| ζ-A correction in Wave 2.1A? | NO — Wave 2.1B per AskUserQuestion #1. |
| `growth()` for non-quasi-contractive τ? | Returns `(1.0, ρ̄)`; caller responsible for `τ ≤ ½ ρ̄^{−1}`. Documented in rustdoc + math.md §12.4. |
