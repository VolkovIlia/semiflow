# ADR-0051 — Magnus K=4 expansion for time-dependent graph Laplacian `L_G(t)`

- **Status**: ACCEPTED (Wave 2.1C shipped 2026-05-20)
- **Date**: 2026-05-20
- **Wave**: v2.1 Wave 2.1C (final wave of v2.1)
- **Authors**: ai-solutions-architect
- **Reviewers**: reviewer-suckless, agentic-engineer
- **Depends on**: ADR-0047 (`GraphHeatChernoff` design), ADR-0048 (CSR storage), ADR-0049 (math.md §12), ADR-0046 (precision-policy bands), ADR-0042 (`apply_into` ping-pong template)
- **Supersedes / amends**: nothing (additive sibling of `GraphHeatChernoff` / `GraphHeat4thChernoff`)
- **Mathematical foundation**: math.md §12.9 (NORMATIVE library policy + CITATION; **no new theorem**)

---

## Context

`v2.1` Waves A and B ship time-**independent** graph heat semigroups
(`GraphHeatChernoff`, `GraphHeat4thChernoff`, `StrangSplitGraph`). The
classical PDE family `∂_t u = −L_G(t) u`, where the edge-weight map varies
smoothly in time (vertex set fixed; topology fixed), is **not** covered: a
frozen-generator Chernoff variant achieves only order 1 in this regime,
silently dropping two orders of accuracy.

The standard remedy on bounded matrix generators is the **Magnus
expansion** truncated to fourth-order matching, using Gauss-Legendre
quadrature at two nodes per step (Iserles, Munthe-Kaas, Nørsett, Zanna
2000 *Acta Numerica*; Blanes, Casas, Oteo, Ros 2009 *Phys. Rep.*). For
bounded `L_G(t)` the Magnus convergence radius `∫₀ᵗ ‖L_G(s)‖₂ ds < π`
is satisfied automatically on physically reasonable graphs and step
sizes — a runtime Gershgorin estimate is the only check required.

Note on naming. The pre-existing
`TruncatedExpDiffusionChernoff` / `TruncatedExp4thDiffusionChernoff` types
in `src/truncated_exp.rs` and `src/truncated_exp4.rs` were renamed in
v0.7.0 (audit finding D2) precisely because they are **not** genuine
Magnus expansions — they truncate `exp(τG)` for a time-frozen `G`. Wave
2.1C is therefore the **first** genuine Magnus expansion in semiflow-core,
and reuses none of those types' code paths. The shared structure is
purely conceptual (factorial-table coefficients and ping-pong scratch).

## Decision

Ship a new public type `MagnusGraphHeatChernoff<F: SemiflowFloat = f64>`
in a new module `crates/semiflow-core/src/magnus_graph.rs`. It implements
`ChernoffFunction<F, S = GraphSignal<F>>` with `order() == 4`. Its
`apply_into` performs one Magnus step `f → exp(Ω(τ)) f` via:

1. **Sample** `L_G(t_*)` at the two GL₄ Gauss-Legendre quadrature points
   `t₁* = τ · c₁` and `t₂* = τ · c₂` with `c₁ = (1 − √3⁻¹)/2`,
   `c₂ = (1 + √3⁻¹)/2`, by invoking a caller-supplied closure
   `Box<dyn Fn(F) -> Arc<Laplacian<F>>>`.
2. **Assemble** the fourth-order Magnus operator
   `Ω₄(τ) = τ·(A₁ + A₂)/2 + (√3·τ²/12) · [A₂, A₁]`
   where `A_i = −L_G(t_i*)` (sign matches §12 convention `∂_t u = −L_G u`).
3. **Apply** `exp(Ω₄)` to `src` via the same degree-4 Taylor truncation
   `Σ_{k=0}^{4} Ω₄ᵏ / k!` already used by `GraphHeat4thChernoff` (the
   classical fourth-order Magnus method; Iserles+ 2000 §5.5).
4. **Check** `‖L_G(τ/2)‖_∞ · τ < π/2` via Gershgorin estimate; on
   violation return `SemiflowError::OutOfMagnusRadius { tau, rho_estimate }`.

The constructor takes the topology graph `Arc<Graph<F>>` (vertex set and
edge set fixed) plus the time-to-Laplacian callback. Wave 2.1C is
**edge-weight-varying only** — `Graph::row_ptr`/`col_idx` MUST be
identical across all sampled `L_G(t)`; only `Laplacian::vals` may vary.
The runtime check is `debug_assert` on row pointer equality at each
call (cheap, catches caller errors during development).

## Rationale

- **Order-4 in τ on time-dep generators.** Order-1 (frozen-generator
  Chernoff applied at midpoint) collapses to global order 1 because the
  τ²-correction is non-zero whenever `L_G(t₁) ≠ L_G(t₂)`. Magnus K=4
  matches `exp(∫₀ᵗ −L_G(s) ds)` through `τ⁴` locally → global order 4 in
  the substep count.
- **Bounded generator simplifies everything.** Unlike the unbounded
  differential-operator Magnus (`∂_x(a(x,t)∂_x ·)`), the matrix Magnus
  expansion has a closed-form convergence radius
  `∫₀ᵗ ‖L_G(s)‖₂ ds < π`, and the bound holds for free as long as the
  Gershgorin estimate is respected (`L_G(t)` is symmetric PSD by §12.1).
- **Reuses the existing operator infrastructure.** `Laplacian::apply_into_slice`
  is the matrix-vector hot path for both `A_i · v` evaluations and the
  Taylor truncation `(Ω₄)ᵏ · v`. The commutator `[A₂, A₁] · v` is two
  nested `apply_into_slice` calls — no dense `N×N` matrix is ever
  materialised. Sparse-graph cost: `O(N · nnz)` per commutator-vector
  product.
- **Zero new dependencies.** `dependencies` count stays at 2/3 (under
  the 3-dep budget set by Wave 2.1A). The GL₄ abscissa/weight constants
  are six `f64` literals, no Newton-Cotes solver.
- **Math fidelity.** §12.9 is CITATION + NORMATIVE only. Classical
  Magnus on bounded matrix generators is established in Iserles+ 2000
  *Acta Numerica* §5 and Blanes+ 2009 *Phys. Rep.* §3-§5. The library
  contribution is the API surface and the runtime-check policy — not the
  mathematics.

## Acceptance gates

- **G11 slope gate** (NORMATIVE — see Wave 2.1C contract §7).
  Time-dependent path graph `P_n`, edge weight `w(t) = 1 + 0.3·sin(πt)`,
  `t_final = 0.5`, OLS slope on `(log n_steps, log err_sup)` over
  `n_steps ∈ {25, 50, 100, 200, 400}`. Threshold:
  `slope ≤ −3.95` (f64), `slope ≤ −3.50` (f32) — see ADR-0046 precision
  bands.
- **T12_magnus_consistency sympy gate** — verify the GL₄ abscissae
  `c₁, c₂ ∈ {(3 − √3)/6, (3 + √3)/6}` and weights `b₁ = b₂ = 1/2`
  match the standard 2-point Gauss-Legendre rule on `[0,1]`, and that
  the Ω₄ coefficient table `(τ/2, τ/2, √3·τ²/12)` matches the
  fourth-order Magnus identity `Ω₄(τ) = ½·(τA₁+τA₂) + (√3/12)·τ²[A₂,A₁]`
  as published in Iserles+ 2000 eq. (5.10). The gate is purely symbolic
  on a 4×4 path Laplacian; no library runtime dependency.
- **All prior gates re-pass byte-identical.** 6 v2.0 + G7 + G8 + G9 +
  G10 slope gates from Waves 2.1A/B re-pass with no numerical drift.

## Out of scope (v2.1)

- **Time-varying topology.** `Graph::row_ptr` and `Graph::col_idx`
  MUST be fixed across the caller's `t ↦ Arc<Laplacian<F>>` callback.
  Variable vertex/edge sets require a new abstraction (vertex-add /
  edge-add commute classes) — deferred to v2.2+.
- **Time-discontinuous `L_G(t)`.** GL₄ quadrature assumes
  `L_G(·) ∈ C²([0, τ])`. The caller is responsible for refining the
  step subdivision around discontinuities; the library does not detect
  them.
- **Order-6 / order-8 Magnus.** Higher-order Magnus variants are well
  understood (Blanes+ 2009 Table 5) but require nested commutators
  `[A_i, [A_j, A_k]]` that scale O(N · nnz · degree) per step. Cost
  vs. benefit deferred to v2.2.
- **Self-adjoint preservation.** Ω₄ is anti-symmetric in `(A₁, A₂)`
  for the commutator term; since `A_i = −L_G(t_i)` are symmetric, `Ω₄`
  is **not** symmetric in general (the commutator is anti-symmetric).
  The Magnus map `exp(Ω₄)` is therefore not symmetric either —
  acceptable for the heat semigroup (which is not a unitary group), but
  noted explicitly because some downstream consumers (e.g. Schrödinger
  on graphs in v2.3) would require a symmetric variant. Deferred.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Caller closure captures non-`'static` references | Closure typed `Box<dyn Fn(F) -> Arc<Laplacian<F>> + Send + Sync + 'static>`; documented in rustdoc |
| R2 | Gershgorin estimate too loose; Magnus runs outside radius silently | Estimate `ρ̄(τ/2) · τ < π/2` (50% safety margin vs. theoretical `< π`) |
| R3 | f32 GL₄ underflow on `τ² · √3 / 12 ≈ 1.44·10⁻¹·τ²` for τ ≈ 10⁻⁴ | f32 slope threshold relaxed to −3.50 (ADR-0046); audit-findings if f32 G11 fails |
| R4 | Topology drift inside caller's callback (caller mutates Graph) | `debug_assert_eq!(self.graph.row_ptr(), sampled_lap.row_ptr())` at each call; release build is best-effort |
| R5 | Commutator-vector product order matters: `[A₂, A₁]v = A₂(A₁v) − A₁(A₂v)` (sign) | Explicit unit test `magnus_commutator_sign_check` on the 5-node star graph oracle |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/magnus_graph.rs` | ~280 |
| `tests/g11_magnus_graph_slope.rs` | ~140 |
| `sympy/verify_v2_1c_magnus_consistency.py` | ~120 |
| math.md §12.9 (citation + library policy) | ~80 |
| ADR-0051 (this) | ~190 |
| **Total** | **~810** |

`src/magnus_graph.rs` stays under the 500-LoC file cap and uses
helper-function decomposition to honour the 50-LoC per-function cap
(see Wave 2.1C contract §11).

## References

- I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) — Theorem 6.
- A. Iserles, H. Z. Munthe-Kaas, S. P. Nørsett, A. Zanna,
  *Lie-group methods*, **Acta Numerica** **9** (2000) 215–365.
  DOI 10.1017/S0962492900002154. (Sections §5–§6 establish the
  fourth-order Magnus method with two-point Gauss-Legendre.)
- S. Blanes, F. Casas, J. A. Oteo, J. Ros,
  *The Magnus expansion and some of its applications*,
  **Physics Reports** **470** (2009) 151–238.
  DOI 10.1016/j.physrep.2008.11.001. (Tables 5–6 list the fourth-order
  Magnus weights and the convergence-radius condition.)
- M. Hochbruck, A. Ostermann, *Exponential integrators*,
  **Acta Numerica** **19** (2010) 209–286.
  DOI 10.1017/S0962492910000048. (§3 reviews truncated-exponential
  application of `exp(Ω)·v` on bounded operators.)
- Hairer, Lubich, Wanner, *Geometric Numerical Integration* (Springer
  2006) §III.4 — pedagogical introduction to Magnus methods.
