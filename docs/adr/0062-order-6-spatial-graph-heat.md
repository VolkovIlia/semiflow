# ADR-0062 — Order-6 spatial graph heat `GraphHeat6thChernoff`

- **Status**: ACCEPTED (v2.4 Wave A)
- **Date**: 2026-05-22
- **Wave**: v2.4 Wave A (Graph Completeness)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0048 (CSR storage), ADR-0026 (Generic-over-Float),
  ADR-0042 (ScratchPool ping-pong), ADR-0046 (precision-policy bands).
- **Mathematical foundation**: math.md §19 (CITATION: Higham 2008
  *Functions of Matrices* §10, Hochbruck-Ostermann 2010 *Acta Numerica* §3;
  NORMATIVE: degree-6 coefficient table `−τ, τ²/2, −τ³/6, τ⁴/24, −τ⁵/120, τ⁶/720`).

## Context

ADR-0048 ships order-1 and order-2 graph heat (`GraphHeatChernoff`,
ζ-A variant). ADR-0049 (v2.1) ships order-4 spatial
(`GraphHeat4thChernoff`) via degree-4 operator-Taylor. ADR-0056 (v2.2)
ships order-6 **time-dependent** Magnus (`MagnusGraphHeat6thChernoff`)
but **f64-only** because of `τ⁵ · √15 / 1080` underflow at f32.

The static-graph order-6 rung is missing — there is no
`GraphHeat6thChernoff` for the case `L_G` is constant in `t`. Use case:
HPC validation suites where graph topology is fixed but accuracy must
reach ε ≲ 1e-10; long-time evolution on Erdős-Rényi or expander
graphs where K=4 accumulates round-off.

The fix: extend the K=4 operator-Taylor pattern to K=6 — `S_6(τ) f = Σ_{k=0}^{6} (−τ L_G)^k / k! · f`. 6 SpMV per step, 2 ping-pong scratch buffers, same shape as K=4.

## Decision

Ship `GraphHeat6thChernoff<F: SemiflowFloat = f64>` in a NEW module
`crates/semiflow-core/src/graph_heat6.rs` (parallel to `graph_heat4.rs`).
Implements `ChernoffFunction<F, S = GraphSignal<F>>` with `order() == 6`.

```rust
pub struct GraphHeat6thChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
}

impl<F: SemiflowFloat> GraphHeat6thChernoff<F> {
    pub fn new(laplacian: Arc<Laplacian<F>>) -> Self;
    pub fn from_owned(laplacian: Laplacian<F>) -> Self;
    pub fn laplacian(&self) -> &Laplacian<F>;
}

impl<F: SemiflowFloat> ChernoffFunction<F> for GraphHeat6thChernoff<F> {
    type S = GraphSignal<F>;
    fn apply_into(&self, tau, src, dst, scratch) -> Result<(), SemiflowError>;
    fn order(&self) -> u32 { 6 }
    fn growth(&self) -> (f64, f64) { (1.0, rho_bar_f64) }
}
```

The `apply_into` kernel uses two ScratchPool buffers `buf_odd`,
`buf_even` ping-pong-style for `L_G^k · src`, `k ∈ {1..6}` (zero heap
allocation in steady state — same pattern as K=4).

**Generic over `F`**, **NOT** f64-only. See "f32 stability rationale"
below.

### f32 stability rationale (NORMATIVE)

The smallest coefficient is `τ⁶ / 720`. On a SMOOTH initial condition
like `cos(2π i/N)` on path `P_64`, the dominant excited mode has
eigenvalue `λ ≈ 4 sin²(π/N) ≈ 4π²/N² ≈ 9.6e-3` for `N=64`. The per-step
order-6 truncation residual is `(τλ)⁷ / 5040 ≈ (τ · 9.6e-3)⁷ / 5040` —
below f32 ε ≈ 1.2e-7 for any reasonable `τ`. Thus an **f32 slope gate
on smooth IC is not observable** — the residual sits beneath round-off
noise, and what would be measured is pure mat-vec round-off, not
convergence.

This is structurally similar to Magnus K=6 (ADR-0056, f64-only) — but
the **type** stays generic over `F`. f32 users get a numerically
correct order-6 evaluation up to round-off; what they LOSE is the
ability to *measure* the order-6 slope on smooth ICs. Sharp ICs that
excite high-frequency modes (`λ ~ λ_max`) would in principle expose
the K=6 convergence at f32, but path-graph + smooth IC is the standard
slope-gate workload and we honour that.

Compare with Magnus K=6 (ADR-0056), where the coefficient `τ⁴ · √15 /
12 · ‖[A_3, A_2]‖ · ‖[A_2, A_1]‖` involves a **product of two
commutators**, each O(1), making the f32 round-off catastrophic — that
is why Magnus K=6 is f64-only **at the type level**. Static K=6 has no
commutators; only powers of `L_G` and pure rational coefficients —
which is why the type stays generic.

**Per-step cost**: 6 SpMV + 6 axpy. Compare K=4: 4 SpMV + 4 axpy
(~1.5× cost per step, but at 10× fewer steps for same accuracy at
ε ≲ 1e-8 → net ~6.7× faster on cool-down/asymptotic regimes).

## Rationale

- **Order-6 with bounded generator is well-understood.** Higham 2008
  §10 Theorem 10.1 gives the truncation-error bound
  `‖exp(τA) − S_K(τ)‖ ≤ (‖τA‖^{K+1} / (K+1)!) · exp(‖τA‖)`. For `A = −L_G`
  this is `(τλ_max)^7 / 5040 · exp(τλ_max)` — explicit, no estimation
  needed.
- **Reuses sparse mat-vec hot path.** Same `Laplacian::apply_into_slice`
  + `axpy_into_slice` building blocks as K=4.
- **Zero new dependencies.** Just `Arc<Laplacian<F>>` + `ScratchPool<F>`.
- **Cross-binding parity from day 1.** Per ADR-0059 graph-bindings
  policy, Python + FFI + WASM exposed in same milestone (P3/P4 of v2.4).

## Consequences

- `src/graph_heat6.rs` projected ~250 LoC (well under 500-LoC cap).
- Public surface +1 type. Additive minor bump.
- `lib.rs` re-exports `GraphHeat6thChernoff`.
- New tests: `tests/graph_heat6_basics.rs`, `graph_heat6_zero_alloc.rs`,
  `graph_g_k6_slope.rs` (~150 LoC total).
- New sympy gate: `scripts/verify_graph_heat6_sympy.py`.

## Acceptance gates

- **G21 slope gate (f64 only)** (NORMATIVE). Static `P_64`, smooth IC
  `f_i = cos(2π i / N)`, `t_final = 1.0`, `n_steps ∈ {5, 8, 12}`
  (carefully chosen: at `n=5` the per-step residual ≈ 1.2e-9 — well
  above f64 floor ~5e-13; at `n=12` ≈ 5.3e-12 — last point still in
  clean regime). Threshold: slope ≤ −5.85.
- **G21 f32 absolute-floor gate** (NORMATIVE). Same workload, `n_steps
  = 12`. Threshold: `|err|_∞ ≤ 5 × 10⁻⁶` (5 ULPs of `cos` amplitude;
  accounts for round-off across 6 SpMVs × 12 steps × 64 nodes). f32
  slope on smooth IC is NOT gated — see §"f32 stability rationale".
- **T16N sympy gate** (NORMATIVE). On a symbolic 4×4 path Laplacian,
  verify `S_6(τ) f − exp(−τL_G) f = O(τ⁷)` via sympy series expansion.
  Also verify the order-5 truncation S_5 disagrees at τ⁶ (sanity check
  that our truncation order is genuinely 6, not accidentally higher).

## Out of scope (v2.4)

- **Padé[3,3] rational approximation.** Costs 3 LU solves per step
  (~12× a single SpMV); benefits only kick in at `‖τA‖ ≳ 6` (well
  beyond the stability window the caller should keep). Deferred
  indefinitely.
- **Variable-coefficient K=6.** ADR-0063 ships VarCoef × Magnus K=4
  this milestone; the K=6 variable-coefficient extension is a v2.5+
  deferral.
- **Order-8 spatial Taylor.** Same rationale as ADR-0056 §"out of
  scope" for Magnus K=8: 8 SpMV per step, ε ~ 1e-12 unrealistic on
  f64 round-off. Deferred indefinitely.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Caller uses `τ > 6 / λ_max(L_G)` — outside Taylor stability domain | rustdoc on `GraphHeat6thChernoff` explicit-bound section; `spectral_radius_bound()` exposed; G21 slope gate fails loudly outside domain. |
| R2 | f32 user sees order-5 effective convergence on coarse grids | rustdoc notes f32 band 0.35 looser; G21 f32 threshold tracks the band. |
| R3 | Per-step buffer reuse incorrect under composition (StrangSplitGraph) | Same `ScratchPool` contract as K=4; tested via `tests/graph_heat6_zero_alloc.rs` with `allocation_counter`. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/graph_heat6.rs` | ~250 |
| `tests/graph_heat6_basics.rs` | ~80 |
| `tests/graph_heat6_zero_alloc.rs` | ~60 |
| `tests/graph_g_k6_slope.rs` | ~110 |
| `scripts/verify_graph_heat6_sympy.py` | ~140 |
| math.md §19 | ~80 |
| ADR-0062 (this) | ~180 |
| **Total** | **~900** |

## References

- N. J. Higham, *Functions of Matrices: Theory and Computation* (SIAM
  2008), §10 — Taylor methods for `exp(A)` on bounded operators.
- M. Hochbruck, A. Ostermann, *Acta Numerica* **19** (2010) §3 —
  truncated-exponential families.
- ADR-0049 (Graph Heat order-4) — predecessor; same operator-form
  pattern.
- ADR-0056 (Magnus K=6) — companion for time-dependent K=6;
  precision-policy precedent.
