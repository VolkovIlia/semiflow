# ADR-0072 — Neumann Boundary via Image Method (B4)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v2.8 (second math pillar of the Manifold Pillar release; additive minor)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (`ChernoffFunction` trait generic over `F`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0068 (v2.6 `KillingRegion<F>` trait + `BoxRegion<F, D>` + `BallRegion<F, D>` — `ReflectingRegion<F>` sibling design directly mirrors this pattern), ADR-0071 (v2.8 A4 manifold pillar — companion ADR; reflected boundary on the upper half-plane $\mathbb{H}^2$ requires the manifold trait).
- **Supersedes / amends**: none — strictly additive on the public surface. Establishes a NEW trait `ReflectingRegion<F>` as a SIBLING (not subtype) of v2.6's `KillingRegion<F>`; the two share the structural shape of the API but have *distinct semantics* (kill = "set to 0 outside", reflect = "mirror across boundary" — see Rationale).
- **Mathematical foundation**: math.md §25 (NORMATIVE library — `ReflectedHeatChernoff` semantics; CITATION Walsh 1986 *Markov Processes and Potential Theory* §3.4 for the image method on reflecting boundaries; Anderson 1988 *Reflected Brownian Motion: Theory and Computation* SIAM for the general reflecting BM framework and convergence).
- **Acceptance gates added**: G27 (RELEASE_BLOCKING — Reflected heat half-line $L^\infty$ residual + slope), T22N (NORMATIVE sympy — image method symbolic identity, 3 sub-checks: PDE residual + Neumann boundary + initial-condition delta limit).

## Context

For a self-adjoint heat-equation generator $A = \Delta$ (or any second-order divergence-form operator) on a bounded or semi-infinite domain $R \subset \mathbb{R}^d$ with *Neumann* boundary $\partial_\nu u|_{\partial R} = 0$, the reflected semigroup $\{S^N_t\}_{t \ge 0}$ is the heat semigroup *with mass-preserving reflection at $\partial R$* (no flux through the boundary). Walsh 1986 *Markov Processes and Potential Theory* §3.4 gives the **image-method kernel** formula:

```
K^N(x, y; t) = K(x, y; t) + K(x, σ_R(y); t)
```

where $K(x, y; t)$ is the free-space heat kernel of $A$ and $\sigma_R : M \to M$ is **reflection across $\partial R$** (the *image* of $y$ in $\partial R$). For half-space $R = \{x : x_d > 0\}$ with planar boundary, $\sigma_R$ flips the $d$-th coordinate ($y \mapsto (y_1, \ldots, y_{d-1}, -y_d)$); for axis-aligned boxes, $\sigma_R$ composes axis-flips; for balls, $\sigma_R$ is the spherical inversion.

Operationally, given a Chernoff function $C(\tau)$ approximating $\exp(\tau A)$, the *reflected Chernoff function* on $R$ is

```
F_refl(τ) f(x)  :=  C(τ)f(x) + C(τ)(f ∘ σ_R)(x)
```

This is the v2.8 B4 design — a generic wrapper `ReflectedHeatChernoff<C, R, F>` that lifts any `ChernoffFunction<F>` to its reflected counterpart on a region $R$.

**Why now (v2.8, not v2.6/v2.7)?** v2.6's `BoundaryPolicy::Neumann` (ADR-0068 Track 1) is a *grid-level* clamp-to-boundary policy — it handles the BC for individual `Grid1D::sample` calls but does NOT realise the *semigroup-level* Neumann condition. The semigroup-level Neumann requires the image-method wrapper (this ADR). The v2.6 grid-level Neumann is the substrate; this ADR closes the gap for the *Chernoff function* surface. The pillar pairs with A4 (ADR-0071) — reflected boundary on bounded subsets of curved manifolds (e.g., the upper half-plane $\mathbb{H}^2_+$ with reflecting real axis) requires both the manifold trait AND the reflecting region trait.

This is **scoped** for v2.8: order matches `C::order()` (reflection does NOT degrade the order of the underlying Chernoff for symmetric BCs, unlike killing in v2.6 which is order-1 globally per Butko 2018). No Robin BC ($\alpha u + \beta \partial_\nu u = 0$) in v2.8 (Engel 2003 active research; deferred to C5+). No mixed Dirichlet+Neumann BC (split-region; deferred to v2.9).

## Decision

Ship four additive public-surface items in v2.8 (plus two additive `impl ReflectingRegion<F> for ...` blocks on existing v2.6 types):

- **`pub trait ReflectingRegion<F: SemiflowFloat = f64>`** — new trait. *Sibling* to v2.6's `KillingRegion<F>` (NOT subtype, NOT supertype — see Rationale for the sibling design choice). Required methods:
  ```rust
  fn is_inside(&self, point: &[F]) -> bool;                   // open interior; ∂R EXCLUDED (open convention)
  fn reflect_in_place<S: State<F>>(&self, dst: &mut S)
      -> Result<(), SemiflowError>;                            // in-place compose with σ_R: for each cell at
                                                              // coord c not in R, replace with σ_R(c)'s value
                                                              // (the reflected ghost contribution).
  fn dim(&self) -> usize;
  ```
  The `reflect_in_place` method writes the *reflected ghost-contribution image* into `dst` — semantically distinct from `KillingRegion::mask_in_place` which writes zero outside `R`. The default impl loops per-cell via `is_inside` and the per-region $\sigma_R$ formula; concrete impls (`HalfSpaceRegion`, additive `BoxRegion` / `BallRegion`) override with batched per-axis flip / spherical inversion routines.

- **`pub struct ReflectedHeatChernoff<C, R, F>`** where `C: ChernoffFunction<F>`, `R: ReflectingRegion<F>` — generic wrapper that implements `ChernoffFunction<F>`. Single Chernoff step semantics (math §25.3):
  ```
  1. inner.apply_into(τ, src, dst, scratch)                    // unrestricted step → dst
  2. tmp := zeroed_like(dst); region.reflect_in_place(tmp);    // build reflected ghost (the σ_R image)
  3. inner.apply_into(τ, tmp_src, tmp, scratch);                // step the reflected ghost forward
  4. dst.axpy(F::one(), &tmp);                                  // add the reflected contribution
  ```
  This is the discrete realisation of the kernel identity $K^N(x, y; t) = K(x, y; t) + K(x, \sigma_R(y); t)$. `order()` returns `inner.order()` (reflection preserves the order of the underlying Chernoff for symmetric BCs — math §25.2 Proposition 25.1). `growth()` returns `(2 * inner_M, inner_ω)` (the doubled mass is the ghost contribution; the exponential rate is unchanged).

- **`pub struct HalfSpaceRegion<F, const D: usize>`** — first new `ReflectingRegion<F>` impl. Half-space $\{x : (x - \mathrm{origin}) \cdot \mathrm{normal} > 0\}$ with caller-supplied origin and unit normal. $\sigma_R(p) = p - 2 ((p - \mathrm{origin}) \cdot \mathrm{normal}) \cdot \mathrm{normal}$ (reflection across the hyperplane). Const-generic $D$ for zero-overhead axis dispatch.

- **Additive `impl ReflectingRegion<F> for BoxRegion<F, D>`** — reuses the v2.6 `BoxRegion<F, D>` struct *unchanged* (no new struct). The reflection $\sigma_R$ for an axis-aligned box composes per-axis flips: if $p$ is outside the box on axis $k$, reflect $p_k$ across the nearest box boundary (low or high). For multi-axis violations, apply the composition in canonical order (axis $0$ first, then axis $1$, etc.). The same struct now implements both `KillingRegion<F>` and `ReflectingRegion<F>`; the caller picks the wrapper type to choose the BC semantics.

- **Additive `impl ReflectingRegion<F> for BallRegion<F, D>`** — reuses the v2.6 `BallRegion<F, D>` struct *unchanged*. The reflection $\sigma_R$ for the ball is the **spherical inversion** $p \mapsto \mathrm{center} + r^2 \cdot (p - \mathrm{center}) / \|p - \mathrm{center}\|^2$ for $D \ge 2$; for $D = 1$ the ball is a closed interval $[c-r, c+r]$, and reflection across either endpoint is the axis-flip handled by `HalfSpaceRegion<F, 1>` semantics (the impl delegates to the appropriate per-axis flip when $D = 1$, or returns `DomainViolation` from `reflect_in_place` if the caller used $D=1$ via `BallRegion` rather than `HalfSpaceRegion` — `BallRegion<F, 1>` callers SHOULD use `HalfSpaceRegion<F, 1>` instead; documented in rustdoc).

File layout: `crates/semiflow-core/src/reflection.rs` (~300 LoC target, under the default 500-LoC cap — NO Override #1 expansion). Module added to `traits.yaml` `modules:` list with `budget_lines: 400` (50 LoC headroom).

Schema bumps: `properties.yaml` 0.9.0 → 0.10.0 (shared with ADR-0071 — combined v2.8 batch); `traits.yaml` 0.7.0 → 0.8.0 (shared with ADR-0071). math.md is append-only (§25 NEW).

## Rationale

- **Why a SIBLING trait `ReflectingRegion<F>` and not a subtype of `KillingRegion<F>` (or vice versa)?** The two traits have the *same structural shape* (`is_inside`, in-place region-aware mutator, `dim`) but *distinct semantics*: `KillingRegion::mask_in_place` writes ZERO outside the region (absorbing-boundary kernel); `ReflectingRegion::reflect_in_place` writes the SHIFTED IMAGE of the outside values into a buffer (Neumann image-method kernel). Subtyping in either direction is mathematically incorrect: a killing region is NOT a degenerate reflecting region (killing destroys mass; reflection preserves mass) — and vice versa. The sibling design keeps the two trait classes independent and prevents accidental confusion at the type level (the caller MUST pick `KillingChernoff<C, R>` vs `ReflectedHeatChernoff<C, R>` explicitly). The same physical region (e.g., a box) can implement BOTH traits (and `BoxRegion<F, D>` does), exposing both BC semantics under one struct; the caller chooses by selecting the wrapper.
- **Why post-add the reflected contribution (steps 3-4) rather than pre-compose with $\sigma_R$ in the inner Chernoff step?** The pre-compose variant `inner.apply((f + f ∘ σ_R) / 2)` is mathematically equivalent at the operator level but breaks the operator-additivity of $C(\tau)$ when $C$ is *nonlinear in the boundary condition* — e.g., for `KillingChernoff<C, R>` the inner already carries an absorbing BC, and pre-composing would double-count the boundary effect. Post-adding the reflected contribution (Walsh 1986 §3.4 step (3.4.2)) is the contract-honest path: it works for any inner `C: ChernoffFunction<F>` regardless of the inner's own BC semantics.
- **Why order matches `inner.order()` (vs the v2.6 `KillingChernoff` order-1 cap)?** Killing introduces the commutator $[L, \mathbf{1}_R]$ per Butko 2018 §3.2 — irreducibly $O(\tau)$, capping the global rate at 1. Reflection introduces $[L, \mathbf{1}_R + \mathbf{1}_R \circ \sigma_R]$ which *vanishes identically* for self-adjoint $L$ when $\sigma_R$ is a Riemannian isometry (Walsh 1986 §3.4 Lemma 3.4.1) — so no order cap. Symmetric BCs preserve the Chernoff order; asymmetric BCs (Robin, mixed) do not, hence the v2.8 scope to *Neumann* (the symmetric self-adjoint case) only.
- **Why reuse `BoxRegion<F, D>` and `BallRegion<F, D>` (additive impls) instead of new types `ReflectingBoxRegion` / `ReflectingBallRegion`?** The geometric region is the same; only the BC semantics differ. Creating duplicate types for each BC family (Killing*Region, Reflecting*Region, future Robin*Region) would explode the type-system surface combinatorially. The additive-impl approach (single region struct implementing multiple BC traits) is the suckless choice: minimal types, explicit BC choice at the wrapper level (`KillingChernoff` vs `ReflectedHeatChernoff`).
- **Why `HalfSpaceRegion<F, D>` as a new type (not additive impl on an existing region)?** No existing region type encodes a half-space — `BoxRegion` is closed (bounded by lo/hi pairs) and `BallRegion` is compact. A half-space is the natural geometric primitive for one of the most common reflecting-BC problems (reflected heat on $[0, \infty)$ — the G27 oracle problem) and for the Poincaré half-plane model (v2.8 A4 + B4 composition example). Adding it costs ~80 LoC and avoids a kludge (representing a half-space as a degenerate `BoxRegion` with $\mathrm{hi}[k] = +\infty$ — would propagate NaN through the per-axis bound checks).
- **Why scalar `apply_into` (sequential ghost build) rather than a fused single-pass kernel?** The ghost-build step (`tmp := zeroed_like(dst); region.reflect_in_place(tmp)`) is per-cell sequential; the inner Chernoff step on `tmp` is the same `inner.apply_into` already in the trait. Fusing into a single-pass kernel would require either (a) a new trait method on `ChernoffFunction<F>` like `apply_with_ghost(τ, src, ghost_src, dst, scratch)` — premature abstraction, or (b) bypassing the trait via `dyn ChernoffFunction` downcasts — anti-pattern. The sequential ghost-build path matches the v2.6 `KillingChernoff` post-multiply discipline and is suckless-minimal.
- **Why `BallRegion<F, 1>` callers SHOULD use `HalfSpaceRegion<F, 1>` instead?** A 1D ball is a closed interval $[c-r, c+r]$; the natural reflection for the *interior* of such an interval is across one of the two endpoints — i.e., the half-line reflection. The full 1D ball reflection (reflecting across the entire interval) is ambiguous (which endpoint?) and not a standard operation. Documenting "use HalfSpaceRegion<F, 1>" in rustdoc is the contract-honest path; the impl returns `DomainViolation` from `reflect_in_place` on $D = 1$ ball to make the contract enforce-at-runtime. For $D \ge 2$, ball reflection is the well-defined spherical inversion.
- **Why explicit `RELEASE_BLOCKING` $L^\infty$ residual gate (G27 sub-test 1) AND slope sub-test (G27 sub-test 2)?** The residual gate (`||err||_∞ ≤ 1e-6` at $n=64$) tests *correctness* of the image-method formula — that the wrapper produces the right answer at a single grid resolution. The slope sub-test ($\le -0.95$ on $n \in \{16, 32, 64, 128\}$) tests *order preservation* — that the wrapping does not degrade the inner Chernoff's order (which is the central claim of math §25.2 Proposition 25.1). Both are needed; the residual alone could miss a constant-error bug (correct formula evaluated at wrong scale), and the slope alone could miss an order-1 wrapping bug that happens to converge to a wrong limit. The two sub-tests are complementary — same pattern as G24 sub-tests (1) residual + (2) slope.
- **Why no Override #1 expansion for `reflection.rs`?** Target ~300 LoC, with 50 LoC headroom = 350 LoC budget. The default 500-LoC cap absorbs this with 150 LoC margin. The 3 backend types (`HalfSpaceRegion` + `BoxRegion`/`BallRegion` additive impls) are each ≤ 80 LoC of closed-form geometry — no sympy-derived coefficient tables, no Magnus quadrature, no $R/12$ curvature corrections to co-locate. The math.md §25 spec is small (image-method kernel formula + 1 proposition + 3 backends); the source maps directly. No carve-out justification needed.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Subtype `ReflectingRegion: KillingRegion` (or vice versa) | Mathematically incorrect: kill destroys mass, reflect preserves mass; neither is a degenerate version of the other. Subtyping would force type-system-level confusion. Sibling design is the only sound choice. |
| Single trait `BoundaryRegion<F>` with an enum-typed `bc: BoundaryCondition` field, dispatched at runtime | Erases the type-system distinction between killing and reflecting. The wrapper structs (`KillingChernoff`, `ReflectedHeatChernoff`) would need to runtime-check the BC kind on every call — performance hit AND type-safety regression. |
| Robin BC ($\alpha u + \beta \partial_\nu u = 0$) in v2.8 | Engel 2003 *Multiplicative Perturbations for Reflecting Brownian Motion* is active research; the image-method extension to Robin requires a multiplier formula in the kernel that depends on $\alpha/\beta$ and is operator-specific (not a uniform `ReflectingRegion::reflect_in_place` API). Defer to C5+. |
| Mixed Dirichlet + Neumann BCs (split-region: kill on part of $\partial R$, reflect on the rest) | Requires partitioning $\partial R$ into subsets and dispatching different BCs per subset — premature abstraction. v2.8 ships single-BC reflection; composition via `KillingChernoff<ReflectedHeatChernoff<...>>` is *possible* but the composition order matters and the gate coverage is not there. Defer to v2.9. |
| Penalty methods (replace $\partial_\nu u = 0$ with $-\epsilon^{-1} u$ outside $R$ and let $\epsilon \to 0$) | $\epsilon$-tunable, ill-conditioned (small $\epsilon$ stiffens the equation), and gives only an approximate Neumann (the limit $\epsilon \to 0$ is non-uniform). Image method is *exact* for symmetric BCs — strictly better. |
| Implement BoxRegion-with-reflection as a NEW struct `ReflectingBoxRegion<F, D>` | Type-system explosion: 3 BC kinds (Kill / Reflect / future Robin) × 3 geometric kinds (Box / Ball / HalfSpace) = 9 types in the worst case. Additive trait-impl approach: 3 BC kinds × 3 geometries = 3 + 3 = 6 types (the 3 region structs + 3 wrappers). Much cleaner. |
| Make `reflect_in_place` take an `&S` source and an `&mut S` destination (two-arg form) | Matches v2.6 `KillingRegion::mask_in_place` shape with `&mut dst` only. The in-place form keeps the per-call signature minimal; the wrapper (`ReflectedHeatChernoff`) handles the scratch-buffer ping-pong externally via `ScratchPool<F>`. |
| Fuse the ghost-build + inner-Chernoff into a single-pass kernel via new trait method `apply_with_ghost` | Premature abstraction; would force EVERY `ChernoffFunction<F>` impl to either implement or default-bridge a new method. The current sequential design (build ghost; step ghost; add) is suckless-minimal and uses only existing trait methods. |
| Ship Robin BC AND Neumann BC in v2.8 | Robin requires a multiplier-formula kernel; image method does NOT extend uniformly. Two distinct algorithms shipped in one minor release is scope creep. v2.8 = Neumann only; Robin = v3.x research. |
| `BallRegion<F, 1>` impl: silently delegate to `HalfSpaceRegion` semantics | Footgun: the caller constructed `BallRegion<F, 1>` thinking they wanted a 1D ball reflection (whatever that means), and the impl silently does something else. Better to return `DomainViolation` and document "use HalfSpaceRegion<F, 1> for 1D reflecting BCs". |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; the v2.6 `BoxRegion<F, D>` and `BallRegion<F, D>` structs are unchanged (only new `impl ReflectingRegion<F>` blocks are added).
- **New module `crates/semiflow-core/src/reflection.rs`** (~300 LoC budget; under default 500-LoC cap; NO Override #1 expansion needed).
- **New trait `ReflectingRegion<F>`** — sibling to v2.6's `KillingRegion<F>`. Independent trait class; no subtyping in either direction. The same region struct (`BoxRegion`, `BallRegion`) implements BOTH; the caller selects BC semantics at the wrapper.
- **Dependency count unchanged** at 2/3 budget (still `num-traits`, `libm`). The closed-form $\sigma_R$ formulas use only basic arithmetic.
- **Schema bumps** (combined with ADR-0071 — single v2.8 batch): `properties.yaml` 0.9.0 → 0.10.0; `traits.yaml` 0.7.0 → 0.8.0. math.md is append-only (§25 NEW).
- **New gates**: G27 (RELEASE_BLOCKING — two sub-tests: (1) reflected heat half-line $L^\infty$ residual ≤ 1e-6 at $n=64$ against the analytical kernel $K^N(x, y; t) = (4\pi t)^{-1/2} [e^{-(x-y)^2 / (4t)} + e^{-(x+y)^2 / (4t)}]$ on Gaussian initial datum; (2) slope $\le -0.95$ on $n \in \{16, 32, 64, 128\}$, confirming order preservation per math §25.2); T22N (NORMATIVE sympy — 3 sub-checks: (a) heat-PDE residual = 0 in $(0, \infty)$, (b) Neumann boundary $\partial_x K|_{x=0} = 0$ symbolically, (c) initial condition reduces to delta function as $t \to 0^+$).
- **No L-gate for `ReflectedHeatChernoff` in v2.8.** The per-call cost is `2 × inner_apply_into + reflect_in_place` — at most $2\times$ the bare-Chernoff latency. No HFT-priority use case; defer L-gate to v2.9 if benchmark evidence demands.
- **CITATIONs added to math.md §25**: Walsh 1986 *Markov Processes and Potential Theory* §3.4 (image-method kernel formula); Anderson 1988 *Reflected Brownian Motion: Theory and Computation* SIAM (general reflecting Brownian motion convergence theory); Butko 2018 cited only for the *contrast* with killing-order-1 cap (math §25.2 Proposition 25.1 derives the order-preservation for symmetric BCs).

## Migration

None for end-users. v2.7 binaries / crates link against v2.8 without recompilation. The new trait + types are additive; the v2.6 `BoxRegion` / `BallRegion` structs are unchanged and gain an additional trait impl that does not affect existing call-sites.

The v2.6 `BoundaryPolicy::Neumann` grid-level clamp policy (ADR-0068 Track 1) is UNCHANGED and continues to gate the per-cell BC; the new `ReflectedHeatChernoff` is the *operator-level* counterpart and may be used independently or composed with the grid-level policy depending on the caller's design.

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0003 — no_std + alloc; reflection uses only basic arithmetic, no allocation.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; reused for `ReflectingRegion<F>`, `ReflectedHeatChernoff<C, R, F>`, `HalfSpaceRegion<F, D>`.
- ADR-0026 — `ChernoffFunction<F>` super-trait; `ReflectedHeatChernoff<C, R, F>` implements it.
- ADR-0041 — `apply_into` + `ScratchPool`; the wrapper uses `inner.apply_into` twice (once for the original, once for the ghost) sharing the same `ScratchPool<F>`.
- ADR-0068 — v2.6 BC widening; `KillingRegion<F>` trait + `BoxRegion<F, D>` + `BallRegion<F, D>` are the *sibling* of the new `ReflectingRegion<F>` design. Same shape, different semantics. The two trait classes are independent; same region struct can implement both.
- ADR-0071 — Riemannian manifold Chernoff (A4, v2.8 companion ADR; shared release window); upper half-plane $\mathbb{H}^2_+$ with reflecting real axis requires both ADRs in composition.
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v2.8 (Manifold Pillar) — release-level roadmap.
- math.md §25 (NEW v2.8) — Neumann via image method normative spec.
- math.md §21 (v2.6) — killing-functional Dirichlet (sibling math; cited for the order-1 contrast).
- `.dev-docs/research/research1.md` — Walsh 1986 image method pillar selection rationale.

## Amendments

(none at acceptance time)
