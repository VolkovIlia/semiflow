# ADR-0023 — v0.9.0 Anisotropic 2D non-separable operator (NonSeparable2DAnisotropicChernoff, Block A)

**Status**: Proposed
**Date**: 2026-05-07
**Authors**: ai-solutions-architect
**Supersedes/Amends**: ADR-0016 forward-compatibility clause (lines 156–181)
— the "**v0.8+** MAY add `NonSeparable2DAnisotropicChernoff<X, Y>`" bullet
is hereby ratified. v0.7.0 scalar-$c$ semantics unchanged.
**Cross-refs**: ADR-0016 (scalar-$c$ predecessor — proof inherited verbatim
with $M \to M_\beta$), ADR-0012 (`Strang2D<X, Y>` — degeneration target
when $\beta \equiv 0$), ADR-0011 (`CflViolated` reused, third semantic
field overload), `contracts/semiflow-core.math.md` §10.7-ter (NORMATIVE,
math companion) and §10.7-bis (parent scalar-$c$ section),
`verify_v0_9_0_nonseparable_aniso.py` (Block B; reuses `canonicalize_AB`
from v0.7.0 verifier via NC-symbol relabel). Cites HLW (2006) §III.5.2 /
§V.2.1, McLachlan & Quispel (2002), Blanes, Casas, Murua (2024).

## Context

v0.7.0 (ADR-0016, math.md §10.7-bis) shipped `NonSeparable2DChernoff<X, Y>`
covering the **scalar-coefficient** mixed-derivative case
$L = L_x \otimes I + I \otimes L_y + c(x, y)\,\partial_x \partial_y$
(Option γ — simplest API). The §10.7-bis preamble (lines 3119–3121) and
ADR-0016 forward-compat clause both flagged the **anisotropic full-tensor**
$\beta(x, y) \cdot \partial_x \partial_y$ generalisation as deferred to
"v0.8+", with the additive-sibling principle preserved.

The mathematical core insight (proven during planning, ratified here):
the per-axis legs $A := L_x \otimes I$ and $B := I \otimes L_y$ are
**unchanged** from v0.7.0, so $[A, B] = 0$ (Lemma 10.1) survives the
$c \to \beta$ substitution verbatim. The §10.7-bis 5-leg → 2-operator
collapse depends **only** on $[A, B] = 0$ (its §10.7-bis.2 proof nowhere
uses any property of $M$ beyond its identity as a single operator).
Hence the palindromic Strang BCH analysis, the K=2 truncated-Taylor
mandate, and the order-2 ceiling all carry over to $M_\beta$ without
new proof obligations beyond the relabel — Block A confirms
symbolically; Block B implements.

The v0.9.0 motivation: enable callers with directionally-dependent mixed
coupling (e.g. anisotropic option pricing with stochastic correlation)
without forcing an artificial isotropic $c$ that loses position-dependent
$\beta$ shape.

## Decision

Adopt **`NonSeparable2DAnisotropicChernoff<X, Y>`** as an additive
sibling of `NonSeparable2DChernoff<X, Y>` (no replacement; v0.7.0 /
v0.8.x callers using the scalar-$c$ type remain bit-equal) for the
generator $L = (L_x \otimes I) + (I \otimes L_y) + \beta(x, y) \cdot
\partial_x \partial_y$ with $\beta \in C^3(\Omega, \mathbb{R})$ and
$\|\beta\|_\infty < \infty$. Per-axis legs $A, B$ are unchanged from
v0.7.0, so $[A, B] = 0$ is preserved and the §10.7-bis 5-leg →
2-operator Strang collapse applies **verbatim** with $M_\beta := \beta(x, y)
\cdot \partial_x \partial_y$ substituted for $M$. The K=2 truncated-Taylor
mixed leg $\Phi_{M_\beta}(\tau) = I + \tau M_\beta + (\tau^2 / 2) M_\beta^2$
is preserved (K=1 still collapses global order to 1 by §10.7-bis.3,
relabelled); the 4-point centred cross-stencil for $\partial_x \partial_y$
is unchanged in **shape** but gains a pointwise $\beta(x_i, y_j)$
multiplier at each interior node (no boundary-policy work — the stencil
support and `Reflect` / `ZeroExtend` / `LinearExtrapolate` composition
rules are identical to v0.7.0). The CFL gate becomes
$4 \cdot \tau \cdot \|\beta\|_\infty < \mathrm{dx} \cdot \mathrm{dy}$
(direct $c \to \beta$ substitution into eq. 10.7-bis.12). The
ellipticity invariant generalises to $\beta^2(x, y) \le 4\,a(x)\,b(y)$
(doc-only caller invariant; caller supplies a single
`beta_norm_bound: f64` analogous to v0.7.0 `c_norm_bound` — no runtime
grid scan, suckless: explicit input over implicit estimator). `order()`
returns **2** (τ-axis Chernoff consistency per §11.1.bis); spatial slope
is $O(\mathrm{dx}^2 + \mathrm{dy}^2)$ (cross-stencil floor, identical to
v0.7.0). When $\beta \equiv 0$ the new type detects this at construction
(single $f64$ test on `beta_norm_bound == 0.0`) and branches to the
existing `Strang2D::apply` so the FP result is bit-equal to `Strang2D`
on the same per-axis legs. Verification: gates **T9N_τ²**, **T9N_τ³**,
**T9N_K2_local**, **T9N_oracle**, **T9N_zero-β**, **T9N_palindrome**
(math.md §10.7-ter.6) plus the slope gate **G4_NS2D_aniso** (§10.7-ter.7)
at threshold $\le -1.95$. Order 2 is provable (§10.7-ter.2 + §10.7-ter.3)
and is the mechanism G4_NS2D_aniso verifies.

## Considered alternatives

- **(a) Reuse `NonSeparable2DChernoff` with `c = β`** (silent
  relabel of the field): rejected. Silently changes v0.7.0 contract
  semantics; `c_norm_bound` carries $\|c\|_\infty$ for an isotropic term.
  Documentation-only broadening leaves callers no opt-out and breaks the
  additive-sibling principle of the ADR-0016 forward-compat clause.
- **(b) Krylov inner solver for $e^{\tau M_\beta}$**: rejected, same
  deferral as ADR-0016 alt (a). Lifts global order to 4 at 2–3× per-step
  cost; deferred pending profiling evidence.
- **(c) Per-node rotation to principal axes**: rejected. Defeats per-axis
  splitting, needs a per-step matrix decomposition, incompatible with
  `AxisLift::X` / `AxisLift::Y`.
- **(d) Drop K=2 → K=1 truncated Taylor**: rejected. §10.7-bis.3
  counterexample applies verbatim ($\tau^2 M^2 / 2$ leading term is
  non-zero for $M_\beta$); order regresses to 1.
- **(e) Full off-diagonal $2 \times 2$ diffusion tensor $D(x, y)$
  breaking per-axis product structure**: rejected, out of scope. Breaks
  additive-sibling; incompatible with v0.5.0+ per-axis catalogue. A
  future `Coupled2DChernoff` is a different architecture, not a sibling.

## Consequences

- **+1 public type**: `NonSeparable2DAnisotropicChernoff<X, Y>`
  (additive sibling — no v0.7.0 / v0.8.x API break).
- **+0 dependencies, +0 `SemiflowError` variants**: `CflViolated` reused
  with a third semantic field overload (rustdoc-amendment in Block B
  documents `a_norm_bound` as $\|\beta\|_\infty$ in this context).
- **+0 changes to existing source files**: a new module
  `nonseparable2d_aniso.rs` (~350 LoC, mirrors v0.7.0
  `nonseparable2d.rs` with a pointwise $\beta(x, y)$ multiplier in the
  cross-stencil) is added; existing `nonseparable2d.rs`, `strang2d.rs`,
  `axis.rs`, `grid_fn2d.rs`, `error.rs` untouched.
- **No new boundary-policy work**: cross-stencil shape unchanged; only
  the scalar multiplier per node changes. §10.7-bis.3 boundary rules
  apply verbatim.
- **No perf-baseline regression**: the cross-stencil gains one $f64$
  multiply per interior node (the $\beta(x_i, y_j)$ pointwise factor) —
  negligible on cached production grids; v0.8.1 baseline (heat_2d
  4.38× speedup, parallel tile-scratch) is unaffected.

## Forward compatibility

- **v1.0+** MAY add `NonSeparable2DAnisotropicKrylovChernoff` evaluating
  $e^{\tau M_\beta}$ via the matrix exponential to lift the global order
  to 4 — additive sibling.
- **v1.0+** MAY add 4th-cross-derivative legs $\partial_x^3 \partial_y$
  and $\partial_x \partial_y^3$ for higher-order spatial coupling
  (separate ADR; never in v0.7.0 / v0.8.x scope; symmetric across
  scalar-$c$ and anisotropic-$\beta$ cases).
- **v0.9+** MAY add `beta_norm_bound_auto(beta_fn, grid)` (sample
  $\beta$ on the grid, take the max), but this is NOT v0.9.0 scope
  (suckless: explicit input over implicit estimator; analogous to v0.7.0
  `c_norm_bound` and v0.4.0 `a_norm_bound`).

## Verification

The Block B engineer deliverable produces a passing run of all four
acceptance commands:

1. `python3 .dev-docs/verification/scripts/verify_v0_9_0_nonseparable_aniso.py`
   — exits 0 with all six **T9N_*** sympy gates passing (math.md §10.7-ter.6).
2. Regression guard: `verify_v0_7_0_nonseparable.py` continues to exit 0
   (v0.7.0 scalar-$c$ semantics unchanged).
3. `cargo run -p xtask -- test-fast` — green workspace.
4. `cargo build --workspace --release` — clean release build.

The empirical slope gate **G4_NS2D_aniso** (math.md §10.7-ter.7) at
threshold $\le -1.95$ over $N \in \{32, 64, 128, 256\}$ ratifies the
order-2 claim end-to-end on a non-trivial
$\beta(x, y) = 0.05 \cdot \exp(-(x^2 + y^2)/4)$, exercising the
$[A, [B, M_\beta]]$ commutator beyond the constant-coefficient case.
