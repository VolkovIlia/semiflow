# ADR-0176 — Higher-Order Dirichlet via Odd (Antisymmetric) Image Method

- **Status**: Accepted
- **Date**: 2026-06-23
- **Wave**: issue-6 (additive; order-2 Dirichlet sibling to the §21 order-1 killing wrapper)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0026 (`ChernoffFunction` generic over `F`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0068 (`BoundaryPolicy` widening — the `OddReflect` variant is additive on this enum), ADR-0072 (Neumann **even**-image method, §25 — this ADR is its minus-sign mirror).
- **Supersedes / amends**: none. Strictly additive. `KillingChernoff` (§21) is UNCHANGED and remains the order-1 hard-wall reference.
- **Mathematical foundation**: math.md §21.9 (NORMATIVE library — odd-image Dirichlet kernel `K^D`; mirror of §25 with `+ → −`).
- **Acceptance gates added**: `G_DIRICHLET_ORDER2` (RELEASE_BLOCKING — slope ≤ −1.95 against the Dirichlet eigenmode oracle), `T_DIRICHLET_ORDER2` (NORMATIVE sympy — `K^D` satisfies the heat PDE and vanishes on `∂R`).

## Context

The library has exactly one operator-level Dirichlet construction: `KillingChernoff<C, R>` (§21, Feynman–Kac killing). It is **order-1, full stop** — the indicator `𝟙_R` has an irreducible `O(τ)` commutator `[L, 𝟙_R]` at the boundary discontinuity (Butko 2018 §3.2; §21.8 proves the symmetric-split conjecture vacuous over the hard indicator). The contradiction: a hard absorbing wall `u|_{∂R} = 0` *demands* a discontinuous indicator, yet that same discontinuity *caps the order at 1*. We need the BC `u=0 on ∂R` **and** order-2 — two properties that the killing formulation cannot hold simultaneously.

The mirror construction in §25 — the **even**-image Neumann method `C(τ)f + C(τ)(f∘σ_R)` — escapes the order-1 cap precisely because the *symmetric* extension `𝟙_R + 𝟙_R∘σ_R` has a *vanishing* commutator with self-adjoint `L`. Nothing in that argument used the sign of the image term; it used only that `σ_R` is an isometry and `L` self-adjoint.

## Decision

Ship the **odd (antisymmetric) image method** — the minus-sign mirror of §25:
$$F_D(τ)\,f(x) := C(τ)f(x) - C(τ)(f\circ σ_R)(x), \qquad x \in R.$$
The odd extension `f̃(x) = −f(σ_R(x))` for `x ∉ R` makes the Dirichlet kernel `K^D(x,y;t) = K(x,y;t) − K(x,σ_R(y);t)`.

Additive surface (two items, both mirroring existing v2.8 patterns):

- **`BoundaryPolicy::OddReflect`** — new variant on the v2.6 enum (ADR-0068). Stencil-level ghost-fill: for an out-of-range index, the ghost value is **`−(mirrored interior value)`** (the existing `Reflect` variant returns `+(mirrored interior value)`; `OddReflect` negates it). This realises the odd extension at the discretisation level, exactly as `Reflect` realises the even (Neumann) extension consumed by `ReflectedHeatChernoff` on a `[0, L]` half-line.
- **`DirichletHeat2ndChernoff<C, R, F>`** — additive sibling wrapper (NEW module `killing_order2.rs`), mirroring `ReflectedHeatChernoff` structure. D=1 half-line single-step trick (§25.3 note): clone `src`, set its grid boundary to `BoundaryPolicy::OddReflect`, run one inner `apply_into`. `order()` returns `inner.order()` (order-2 for `DiffusionChernoff`).

## TRIZ resolution summary (the sign as a free resource → ИКР)

- **ТП**: instrument = the image-extension operator; izdelie = the discrete state. ТП-1 (killing `𝟙_R`): BC enforced, but `[L,𝟙_R] ≠ 0` ⇒ order capped at 1. ТП-2 (no wall): order-2, but `u ≠ 0` on `∂R`. Chosen half: **keep order-2** and make the BC appear by other means.
- **ФП**: the boundary ghost must be **present** (to enforce `u=0`) yet **commute** with `L` (to keep order-2). Resolution **in structure / by inversion**: use the ghost that is already symmetric — but make it *odd* instead of *even*. Принцип #13 (наоборот: `+ → −`) and #4 (асимметрия: odd vs even extension).
- **Resource (ВПР)**: the *sign* of the image term — a resource already present in §25's construction, costing nothing. At the symmetry point `σ_R(x₀)=x₀`, an odd function is automatically zero: `f̃(x₀) = −f(x₀) ⇒ f(x₀) = 0`. The BC is not *imposed*; it *falls out* of oddness.
- **ИКР**: the boundary condition enforces itself (the odd ghost vanishes at `∂R` by symmetry) while the order is preserved (the commutator vanishes by self-adjointness) — both properties at once, zero added machinery beyond a sign flip.

## Order-preservation argument (mirror of Proposition 25.1, `+ → −`)

`σ_R` is a Riemannian isometry on `R` and the identity on `∂R`; `L` is self-adjoint. The commutator
$$[L,\; 𝟙_R - 𝟙_R\circ σ_R] = 0 \text{ identically on the core of } L,$$
because the sign of the second term does not affect the commutator vanishing (the §25 Lemma 3.4.1 argument is `+`-agnostic). Hence `F_D` introduces no `O(τ)` commutator term and inherits the inner order exactly — the Prop 25.1 proof transfers verbatim with `+ → −`. The boundary condition `K^D(x₀,·;t) = K(x₀,·;t) − K(x₀,·;t) = 0` for `x₀ ∈ ∂R` (since `σ_R(x₀)=x₀`) is the odd kernel's defining identity.

## Consequences and limits

- **Self-adjoint `L` only.** The vanishing-commutator argument requires self-adjoint `L`. Non-self-adjoint drift-reaction operators stay order-1 — use `KillingChernoff` (§21). Documented in rustdoc and §21.9.
- **No non-negativity.** The odd ghost subtracts mass — `F_D` does **not** preserve non-negativity, and this is correct: an absorbing Dirichlet wall removes mass. Therefore `DirichletHeat2ndChernoff` ships **no** non-negativity test (cf. `ReflectedHeatChernoff`'s `nonneg_preserved`, which is valid only because reflection is mass-preserving).
- **Distinct from ADR-0126 soft-killing.** `Killing2ndChernoff` (§21.8, smooth killing *rate* `κ`) and this `DirichletHeat2ndChernoff` (hard wall `u|_{∂R}=0` at order 2) are DIFFERENT operators. ADR-0126 makes a continuous reaction term order-2; this ADR makes the *hard absorbing wall* order-2 via the odd image — a problem §21.8 Para 1 explicitly states the killing formulation cannot solve. The odd-image route sidesteps the cap entirely by never forming the discontinuous indicator.
- **`KillingChernoff` untouched.** Strictly additive; the order-1 §21 wrapper is the unchanged reference and continues to gate G23.
- **Gates**: `G_DIRICHLET_ORDER2` (RELEASE_BLOCKING, slope ≤ −1.95, eigenmode oracle `u(t,x)=Σ_{k=1}^{8} a_k sin(kπx) e^{-(kπ)² t/2}` on `(0,1)`) and oracle `scripts/verify_dirichlet_order2.py` (`T_DIRICHLET_ORDER2`).
