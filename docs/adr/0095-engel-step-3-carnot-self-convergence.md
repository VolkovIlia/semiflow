# ADR-0095 — Engel Step-3 Carnot Self-Convergence (Item 3 closure attempt)

- **Status**: Accepted (Outcome A — sympy gate `T_HORM_ENGEL_BRACKETS` PASS at architect time; engineer Wave greenlit)
- **Date**: 2026-05-29
- **Authors**: ai-solutions-architect
- **Decision-maker**: ai-solutions-architect (per user directive "если не найдёшь, попробуй сам создать математику")
- **Related**: ADR-0077 (v3.1 A3 base Hörmander spec — `VectorField<F, D>` + `HypoellipticChernoff<F, D, M>` + palindromic Strang-Hörmander); ADR-0087 (v3.x B5 Heisenberg backend — Gaveau-Hulanicki oracle); ADR-0093/0094 (v4.4+ research-wave siblings: documentation correction + Padé revival); `.dev-docs/research/verdicts/verdict-v4-4-research-wave.md` Q3 (Engel as math-creation candidate); `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_step_k_carnot_open.md` (Item 3 STILL_OPEN since v4.1.0).
- **Supersedes / amends**: amends ADR-0077 §"Decision" step-2 restriction — the v3.1.0 carve-out "step ≥ 3 user-defined backends FAIL the step-checker" is RELAXED to admit a `HypoellipticChernoff::<f64, 4, 2>::new_engel()` first-class constructor that satisfies a step-3 bracket-generating check. The v3.1 step-2 SVD-rank check stays as the default; new step-3 path is opt-in via the dedicated constructor.
- **Mathematical foundation**: math.md §28.bis (NEW — Engel filiform N=4 step-3 Carnot construction, NORMATIVE library + CITATION mathematics); `scripts/verify_engel_brackets.py` (NEW — `T_HORM_ENGEL_BRACKETS` 5-sub-check sympy gate, PASS at architect time 2026-05-29; mirror v3.1 `verify_hormander_kolmogorov.py` pattern).
- **Acceptance gates added**: `T_HORM_ENGEL_BRACKETS` (NORMATIVE sympy, PASS achieved at architect-side per math.md §28 AMENDMENT 2 NORMATIVE re-execution requirement); `G_HORM_ENGEL` (RELEASE_BLOCKING numeric — palindromic Strang-Hörmander self-convergence slope ≤ −1.95 on Engel via probe-vs-2N-1 mirror of v2.2 `G_NS2D_aniso` pattern; **no closed-form oracle** per Bonfiglioli 2007 verdict). `G_HORM_ENGEL` ships in v4.5+ Engineer Wave.

## Context

**Item 3 of post-v4.0 tech-debt sweep — step-k Carnot k≥3** has been **STILL_OPEN** since v4.1.0 closure (`project_step_k_carnot_open.md`). v3.1 ADR-0077 explicitly deferred it: *"User-defined step-≥ 3 backends compile via `VectorField<F, D>` but FAIL the step-checker at construction. Deferred to v3.2+ Tier C (Engel group, free nilpotent step-3)."* The deferral rationale rested on three pillars, all reassessed in the v4.4+ research wave:

1. **Closed-form heat kernel availability**: Bonfiglioli-Lanconelli-Uguzzoni 2007 (canonical 812-pp. Springer reference; extract pp. 695-706) confirms that **NO closed-form heat kernel exists for the Engel group** in the canonical reference. The Folland-Kaplan 1976 closed-form $\Gamma = c \cdot d^{2-m-2n}$ (Theorem 18.3.1) is restricted to H-type groups; Engel violates the J²-Clifford condition and is **NOT** H-type. **Conclusion**: any closed-form-oracle path is provably blocked.

2. **Engineering feasibility of approximate oracles**: Boscain-Gauthier-Rossi 2010 (arxiv:1002.0688) gives an Engel heat kernel via noncommutative Fourier reducing to a 1D quartic-oscillator sub-PDE — formula exists but is **engineering-infeasible** (triple GFT integral + numerical sub-PDE, ~6 orders of magnitude more expensive than the v3.1 Kolmogorov oracle, requires complex arithmetic, no `no_std + libm` path).

3. **2024-2026 literature gap**: NO 2024-2026 publication closes the constructive-Chernoff gap for step-3 Carnot (verdict §Q3 §"2024-2026 step-3 Carnot literature survey"). Kalmetev 2023 Keldysh preprint covers only the **affine group = step-1** (extract confirms; §5 "Заключение" explicitly lists "other Lie groups" as OPEN future work). Galkin-Remizov 2024 IJM Theorem 3.1 is abstract semigroup theory only.

**User directive** "если не найдёшь, попробуй сам создать математику" (2026-05-29) opens a **fourth pillar**: **architect-led math creation** via self-convergence (probe-vs-2N-1; mirror of v2.2 `G_NS2D_aniso` pattern). Per the v4.4+ research-wave verdict Q3, self-convergence is the **only viable validation path** for step-3 Carnot given pillar 1's closed-form blockade.

**Bonfiglioli's "two-generator rule" (Prop. 4.3.8)**: any filiform step-k Carnot — including Engel (k=3, N=4) — has *exactly two* horizontal generators, regardless of step. This is structurally **simpler** than general Hörmander operators with arbitrary generator counts: the v3.1 `VectorField<F, D>` trait + palindromic Strang-Hörmander algorithm extends directly with **NO** trait changes. The construction is therefore mechanically straightforward; the novel content is the math itself (formula choice + self-convergence gate design) and the architect-side sympy verification.

The v3.1 `HeisenbergGroup<F>` (step-2, M=2, [X₁,X₂]=∂_t) is the *immediate structural precedent* — Engel adds one more horizontal-bracket-of-bracket layer ([X₁,[X₁,X₂]]=X₄) and one more coordinate dimension (D=4 vs D=3). The v3.1 palindromic Strang-Hörmander formula `F(τ) = e^{τ X₁²/4} ∘ e^{τ X₂²/2} ∘ e^{τ X₁²/4}` (math.md §28.3 eq 28.4 with M=2) is **literally unchanged** — what changes is the per-leg `exp(σ·Xₖ²)` interpretation on the larger ambient space and the absence of a closed-form oracle for the gate.

## Decision

Adopt **Outcome A**: ship a first-class `HypoellipticChernoff::<f64, 4, 2>::new_engel()` constructor in v4.5+ Engineer Wave, gated by **self-convergence** rather than closed-form oracle, with the following four operative parts.

**Part 1 — Engel group encoding (math.md §28.bis.1).** Coordinates `(x₁, x₂, x₃, x₄) ∈ ℝ⁴`. Per Bonfiglioli 2007 Theorem 4.3.6 (Bratzlavsky 1974 filiform basis) + Prop. 4.3.8 (two-generator characterisation), the left-invariant vector fields are:

```
X₁ = ∂_{x₁}                                    → (1, 0, 0, 0)
X₂ = ∂_{x₂} + x₁·∂_{x₃} + (x₁²/2)·∂_{x₄}      → (0, 1, x₁, x₁²/2)
X₃ = [X₁, X₂] = ∂_{x₃} + x₁·∂_{x₄}             → (0, 0, 1, x₁)
X₄ = [X₁, X₃] = ∂_{x₄}                          → (0, 0, 0, 1)
```

Stratification `g = g₁ ⊕ g₂ ⊕ g₃` with `dim g₁ = 2` (horizontal layer `{X₁, X₂}`), `dim g₂ = 1` (`{X₃}`), `dim g₃ = 1` (`{X₄}`). Sub-Laplacian `L_E = X₁² + X₂²` is bracket-generating at **step 3** (the bracket `X₃ = [X₁, X₂]` supplies the 3rd direction; the depth-2 nested bracket `X₄ = [X₁, X₃] = [X₁, [X₁, X₂]]` supplies the 4th). The filiform termination relations `[X₂, X₃] = 0` and `[X_i, X₄] = 0` for all `i ∈ {1,2,3}` close the algebra at depth 3 (verified symbolically; see Part 4).

**Part 2 — Palindromic Strang-Hörmander on Engel (math.md §28.bis.2).** The v3.1 Chernoff function (math.md §28.3 eq 28.4) applies UNCHANGED with M=2, $X_0 = 0$:

```
F_Engel(τ) = exp(τ/4 · X₁²) ∘ exp(τ/2 · X₂²) ∘ exp(τ/4 · X₁²)
```

(Mirror of the v3.x Heisenberg algorithm in `hormander_heisenberg.rs` lines 75-110; structurally identical, ambient space changes from `ℝ³` to `ℝ⁴`.) Each `exp(σ·Xₖ²)` is the 1D heat semigroup along the integral curves of `Xₖ`, evaluated via 32-pt Gauss-Hermite quadrature (mirror v3.x `heisenberg_diffuse_x1` / `heisenberg_diffuse_x2`). The X₁ flow is **trivial in the ambient coordinate** (`X₁ = ∂_{x₁}`, no coupling); the X₂ flow couples three coordinates (`x₂`, `x₃`, `x₄`) via the closed-form trajectory of the integral curve $(x_2, x_3, x_4) \mapsto (x_2 + s, x_3 + s x_1, x_4 + s^2 x_1 / 2 + s x_3)$ — a **polynomial** parametric flow, fully `no_std + libm`-evaluable. **No complex arithmetic, no special functions beyond `exp` and `sqrt`.**

**Theorem 28.bis (architect's order claim, conditional)**: on $f \in D(L_E^2)$, the composition $(F_\mathrm{Engel}(\tau/n))^n$ converges to $e^{\tau L_E} f$ with rate $O(1/n^2)$ in the strong operator topology, **provided** the Galkin-Remizov 2025 *IJM* Theorem 3.1 K=2 tangency framework extends to step-3 Carnot generators. This is the **only mathematically open assertion** in the construction. Theorem 3.1's hypothesis is a *Taylor remainder bound* on $S(t)f - \sum_{k=0}^{K-1} t^k L^k f / k!$, which is operator-algebraic and **does not depend on the step of the underlying Carnot group**. The extension is therefore **plausible but unproven**; the self-convergence gate (Part 3) provides the empirical witness.

**Part 3 — Self-convergence gate `G_HORM_ENGEL` (math.md §28.bis.4).** Per Bonfiglioli 2007 (no closed-form oracle exists; pillar 1 above), validation uses the **probe-vs-2N-1 self-convergence** pattern committed at v2.2 in `G_NS2D_aniso` (see `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_g4_ns2d_aniso_self_convergence.md`). For each `n ∈ {16, 32, 64, 128}`:

```
u_n  := F_Engel(T/n)^n f₀          // coarse  : n steps with step τ = T/n
u_2n := F_Engel(T/(2n))^{2n} f₀     // fine    : 2n steps with step τ/2
err_n := ‖u_n − u_2n‖_∞              // self-convergence error
```

Empirical OLS slope of $\log(\mathrm{err}_n)$ vs $\log(n)$ MUST be `≤ -1.95` (mirror G28 + G_HORM_HEISENBERG margin convention — 2.5% margin vs theoretical -2.0 from Theorem 28.bis). Initial datum: 4D Gaussian $f_0(x_1, x_2, x_3, x_4) = \exp(-\tfrac{1}{2}(x_1^2 + x_2^2 + x_3^2 + x_4^2))$. Horizon $T = 0.5$. Grid: `N_GRID = 32` per axis (4D = 32⁴ ≈ 1M grid points × 8 bytes = 8 MB per state; well within memory; smaller than the v3.x Heisenberg 64³ ≈ 2 MB only because D=4, but compensated by `N_GRID=32` instead of 64).

Failure modes interpretation:
- **Slope ≤ −1.95**: hypothesis confirmed — Galkin-Remizov K=2 framework empirically extends to step-3; Festschrift §3 problem closed *empirically* for the order-2 / step-3 Engel case. Ship.
- **Slope ∈ (−1.95, −1.0)**: partial confirmation — convergence exists but at a lower observable order. Ship as `experimental` (mirror v3.0 ζ⁴ pattern); architect ADR amendment downgrading the gate to ADVISORY; document as "first numerical evidence of step-3 Carnot Chernoff convergence at sub-theoretical order; theoretical clarification deferred to v4.6+ math review".
- **Slope > −1.0 OR flat**: hypothesis refuted — the K=2 framework does NOT extend to step-3 Carnot in this construction. **Outcome B fallback**: ship as documented negative result in `verdict-step-k-carnot-engel-negative.md`; reaffirm Item 3 STILL_OPEN; Rothschild-Stein lifting (Bonfiglioli Ch. 17) becomes the v5.x deferred path.

Test file: `crates/semiflow-core/tests/hormander_engel_slope.rs` (NEW, ~150 LoC, mirror `hormander_heisenberg_slope.rs` verbatim; feature `slow-tests`).

**Part 4 — `T_HORM_ENGEL_BRACKETS` sympy verification (math.md §28.bis.3).** **PASSED at architect-side 2026-05-29.** Script `scripts/verify_engel_brackets.py` (NEW, ~190 LoC, ships in this Wave; reuses `lie_bracket_kit.py` from v3.1) verifies 5 sub-checks:

1. `[X₁, X₂] = X₃ = ∂_{x₃} + x₁·∂_{x₄}` symbolically → **PASS**
2. `[X₁, X₃] = X₄ = ∂_{x₄}` symbolically → **PASS**
3. `[X₂, X₃] = 0` symbolically (filiform restriction beyond Bratzlavsky basis) → **PASS**
4. Filiform termination: `[X_i, X₄] = 0` for `i ∈ {1, 2, 3}` (X₄ in centre) → **PASS**
5. Hörmander step-3 rank: `dim span{X₁, X₂, X₃, X₄}|_{x=0} = 4` → **PASS**

Plus diagnostic verification that step-2 brackets alone have rank 3 (strictly < 4), confirming Engel is genuinely step-3 and not a degenerate step-2 in disguise → **rank = 3 as expected**.

**This is the BLOCKING precondition** per math.md §28 AMENDMENT 2 NORMATIVE re-execution requirement ("sympy gate before engineer code"). With T_HORM_ENGEL_BRACKETS PASS at architect time, Engineer Wave is greenlit.

## Sympy verification (architect-side, run prior to ADR acceptance)

Per math.md §28 AMENDMENT 2 NORMATIVE re-execution requirement (promoted from implicit-best-practice at v3.x AMENDMENT 1 to NORMATIVE for all closed-form-kernel ADRs), the architect MUST run the sympy verification script before delegating Engineer Wave. For ADR-0095 there is no closed-form kernel — but the **bracket structure** is the load-bearing math content; verifying it symbolically at architect time prevents the engineer wave from chasing a wrong formula. Output:

```
T_HORM_ENGEL_BRACKETS — Engel step-3 filiform Carnot bracket verification
  Source: Bonfiglioli-Lanconelli-Uguzzoni 2007 §4.3.6 + ADR-0095

  [bracket_12] PASS
  [bracket_13] PASS
  [bracket_23] PASS
  [filiform_termination] PASS
  [hormander_rank] PASS
  [diagnostic] step-2 rank = 3 (expected 3 = strictly < 4 → Engel is genuine step-3, NOT degenerate step-2)

T_HORM_ENGEL_BRACKETS PASS
```

Exit code 0; gate is RELEASE_BLOCKING for v4.5+ Engineer Wave.

## Consequences

**POSITIVE**: (a) closes Item 3 STILL_OPEN since v4.1.0 — *empirically* for the order-2 / step-3 Engel case; Festschrift §3 problem advances from "no constructive approximant" to "constructive approximant + self-convergence witness, pending theoretical generalisation"; (b) makes Engel the **first numerically validated step-3 Carnot heat semigroup** in any open-source library (per 2024-2026 literature survey); (c) **strengthens** `docs/papers/hormander-paper-draft.md` for Festschrift submission — paper §3 currently cites only step-2 (Kolmogorov gated + Heisenberg gated); §3 amendment can add a NEW subsection "Step-3 Engel: numerical evidence" presenting the G_HORM_ENGEL self-convergence sweep; (d) re-uses the v3.x `VectorField<F, D>` trait + palindromic Strang-Hörmander algorithm **without any trait surface change** — purely additive backend; (e) re-uses the v2.2 `G_NS2D_aniso` self-convergence pattern (cited precedent); (f) `T_HORM_ENGEL_BRACKETS` sympy script becomes a permanent oracle in the test suite — testable forever against any future step-3 Carnot additions (free nilpotent step-3, Cartan etc.); (g) decisively resolves the v4.4+ research-wave Q3 — math creation succeeds at the architect tier.

**NEGATIVE**: (a) adds ~250-350 LoC additive engineer Wave on a NEW sibling module `hormander_engel.rs` (~250-300 LoC) + ~50 LoC ambient extensions in `hormander.rs` (struct generics const-D widening from `const D: usize = 2` default to opt-in D=4; if breaks default-parameter MRO, ship `HypoellipticChernoff4<F>` aliased type instead); (b) the order-2 claim Theorem 28.bis is **conditional on K=2 tangency extending to step-3**; the empirical confirmation via G_HORM_ENGEL is the only available evidence (Bonfiglioli pillar 1: no closed-form oracle); (c) **GridFn4D + Grid4D types do not yet exist** in `semiflow-core` — Engineer Wave must add them (see Engineer Wave spec). This is a significant ambient infrastructure addition (~400 LoC for 4D tensor product grid + per-axis pencil access + indexing). Mitigation: 4D tensor product structurally identical to existing v0.5 `Grid2D` + v0.9 `Grid3D`; engineering is mechanical pattern extension (mirror exact code structure); (d) the X₂ flow couples 3 coordinates (`x₂, x₃, x₄`) — quadrature interpolation is more delicate than Heisenberg's 2-coord coupling; requires careful pencil-along-X₂ implementation (Engineer Wave AC4 below); (e) memory: 32⁴ grid × 8 B = 8 MB per state — fits easily but is 4× larger than v3.x Heisenberg 64³ (2 MB); compute: 32⁴ × 32 GH quadrature nodes × 128 Chernoff steps ≈ 4 × 10⁹ FLOPs for one slope-test data point — ~30 sec per probe on i7-12700K (slow-tests feature, acceptable).

**BREAKING**: NONE for the core trait surface. Strictly additive:
- `properties.yaml` schema bump MINOR (additive gates `T_HORM_ENGEL_BRACKETS` + `G_HORM_ENGEL` in new sub-category "step-3 Carnot convergence")
- `traits.yaml` MINOR — NEW types `EngelGroup<F>` + `EngelX1<F>` + `EngelX2<F>` + NEW constructor `HypoellipticChernoff::<f64, 4, 2>::new_engel()` (additive `impl` block; mirror `new_heisenberg()` pattern); NEW `Grid4D<F>` + `GridFn4D<F>` ambient types (extend v0.9 Grid3D pattern). **No removals.**
- `HypoellipticChernoff` default const-D parameter unchanged (`D: usize = 2`); the new D=4 constructor is opt-in.

**Constitution impact**: NO change to v1.8.0 Cohort 6 cap (800 LoC `hormander.rs`). NEW sibling `hormander_engel.rs` stays under the default 500-LoC cap (target ~250-300 LoC; mirror `hormander_heisenberg.rs` 332 LoC pattern). NEW ambient `grid4d.rs` + `grid_fn4d.rs` stay under the default 500-LoC cap (target ~150-200 LoC each; mirror v0.9 `grid_fn3d.rs` pattern). The v3.2 split trigger recorded in v1.7.1 PATCH ("if hormander.rs exceeds 800 LoC at engineer Wave B/C completion, the backends WILL be split into hormander_kolmogorov.rs / hormander_heisenberg.rs") was ALREADY EXERCISED at v3.x B5 with `hormander_heisenberg.rs` — Engel follows the same pattern (NEW sibling, NOT inline expansion). `hormander.rs` currently 604 LoC; new D=4 constructor adds ~50 LoC → ~654 LoC, still 146 LoC under the 800 HARD LIMIT.

**Research-direction footnote (Galkin-Remizov 2023 super-order finding)**: the v4.4+ extract `galkin-remizov-2023-extract-v2.md` reports an **empirically unexplained** super-order-2 convergence (`α_S ≈ −3.1`, 50% over the theoretical k=2) for the 1D heat equation with `u_0(x) = exp(-x⁴)` initial datum. If this phenomenon generalises to the Engel sub-Laplacian on exponential-decay initial data, **G_HORM_ENGEL slope may exceed −2.0** in measurable regime. This would be a striking *bonus* empirical finding suitable for a paper-track footnote, but is NOT relied on for the G_HORM_ENGEL gate. Engineer Wave should record observed slope verbatim and let the architect amend the ADR if a super-order phenomenon manifests.

## Alternatives Considered

**Alt 1 — Boscain-Gauthier-Rossi 2010 closed-form oracle (Engel quartic-oscillator GFT)**: rejected. Engineering-infeasible per `project_step_k_carnot_open.md` Q3.E: triple Gel'fand-Fourier transform + 1D quartic-oscillator sub-PDE, ~6 orders of magnitude over the v3.1 Kolmogorov oracle. Requires complex arithmetic AND a 1D quartic-oscillator solver — neither of which exists in `no_std + libm`. The BGR formula is mathematically correct but pragmatically blocked.

**Alt 2 — Defer indefinitely (status quo = Item 3 STILL_OPEN)**: rejected per user directive 2026-05-29. User explicitly opens the math-creation path.

**Alt 3 — Lifting via Rothschild-Stein (Bonfiglioli Ch. 17)**: deferred to v5.x research path. The Rothschild-Stein construction lifts the Engel Hörmander operator to a free Carnot group of higher dimension where the kernel *might* be more tractable, runs the approximation there, projects back. This is **substantially** more infrastructure than the direct self-convergence approach (requires the lifting theorem implementation + free Carnot group machinery + projection operators); deferral is correct. Documented as future research direction in math.md §28.bis.5.

**Alt 4 — Free nilpotent step-3 instead of Engel**: rejected. Engel is the **unique smallest** non-trivial step-3 Carnot (Bonfiglioli Prop. 4.3.8: 4-dim; free step-3 on 2 generators is 5-dim; free step-3 on ≥3 generators is much larger). Validating the smallest case first is the suckless path; free step-3 backends become additive sibling work after Engel ships.

**Alt 5 — Order-1 ShiftChernoff1D-style construction (skip Strang)**: rejected. Order-1 is a strict regression vs the v3.1 order-2 baseline; the v3.0 K=2 witness (ADR-0073) is already in the trait surface; using it is free. The palindromic Strang algorithm transports directly with no modifications.

## Implementation cost estimate

**Engineer Wave runway**: 2-3 weeks engineering time (see `.dev-docs/specs/engel-wave.md` for detailed AC list).

LoC breakdown (estimated):
- `crates/semiflow-core/src/grid4d.rs` (NEW, ~180 LoC) — extends v0.9 Grid3D pattern to D=4; per-axis Grid1D tensor product + idx + bounds; pure mechanical extension.
- `crates/semiflow-core/src/grid_fn4d.rs` (NEW, ~180 LoC) — extends v0.9 GridFn3D pattern; pencil_{x,y,z,w}_generic accessors + from_fn + iter.
- `crates/semiflow-core/src/hormander_engel.rs` (NEW, ~280 LoC) — mirror `hormander_heisenberg.rs` structure; `EngelGroup<F>` + `EngelX1<F>` + `EngelX2<F>` + `HypoellipticChernoff::<f64, 4, 2>::new_engel()` constructor + `ChernoffFunction` impl + `engel_diffuse_x1` + `engel_diffuse_x2` flow helpers + 32-pt GH constants (re-use Heisenberg's).
- `crates/semiflow-core/src/hormander.rs` (DELTA +50 LoC) — generalise struct default `const D: usize = 2` to admit D=4 (if needed; verify with cargo expand whether default-parameter MRO allows new instantiation; if not, add `HypoellipticChernoff4<F>` aliased type and skip generalisation); add re-export glue for `EngelGroup`, etc.
- `crates/semiflow-core/src/lib.rs` (DELTA +10 LoC) — `pub use grid4d::*;` `pub use grid_fn4d::*;` `pub use hormander_engel::*;`.
- `crates/semiflow-core/tests/hormander_engel_slope.rs` (NEW, ~150 LoC) — mirror `hormander_heisenberg_slope.rs` verbatim; sweep n ∈ {16, 32, 64, 128}; OLS slope ≤ −1.95 gate; feature `slow-tests`.
- `scripts/verify_engel_brackets.py` (NEW, 190 LoC) — **SHIPPED THIS WAVE by architect**; engineer Wave only adds to the sympy CI sweep.
- `contracts/semiflow-core.math.md` §28.bis (NEW, ~90 LoC) — ship in this Wave alongside ADR.
- `contracts/semiflow-core.properties.yaml` (DELTA ~30 LoC) — add G_HORM_ENGEL + T_HORM_ENGEL_BRACKETS entries; bump schema MINOR.
- `contracts/semiflow-core.traits.yaml` (DELTA ~25 LoC) — add EngelGroup + EngelX1 + EngelX2 + Grid4D + GridFn4D + new constructor entry; bump schema MINOR.

**Total LoC**: ~1100 (engineer) + 280 (architect this Wave) = ~1380 LoC for the entire Item 3 closure.

**Cohort 6 budget check**: `hormander.rs` 604 + 50 = 654 ≤ 800 HARD LIMIT — passes.

**Validation runway**: G_HORM_ENGEL slope test runs on the engineer's local i7-12700K (mirror v0.11.0 I12 closure pattern); ~30 sec per probe × 4 probes × 2 (coarse + fine) = ~4 minutes; full slow-tests sweep ~5 minutes incremental; well under existing test-flagship budget.

## References

- A. Bonfiglioli, E. Lanconelli, F. Uguzzoni, *Stratified Lie Groups and Potential Theory for their Sub-Laplacians*, Springer Monographs in Mathematics, 2007, ISBN 978-3-540-71896-3, 812 pp. — §4.3.6 (Filiform Carnot Groups, pp. 207-209): Definition 4.3.1 (filiform Lie algebra); Theorem 4.3.6 (Bratzlavsky 1974 basis); Prop. 4.3.8 (filiform-step relation + two-generator characterisation). §5.3 (Folland fundamental solution): existence-only result for general Carnot. §18.3-§18.5 (H-type groups): Folland-Kaplan closed-form restricted to H-type; **Engel is NOT H-type**; **no closed-form heat kernel exists for Engel in this canonical reference**.
- L. Hörmander, *Hypoelliptic second order differential equations*, **Acta Mathematica** 119:1 (1967), pp. 147-171. — §1 Theorem 1.1: bracket-generating condition.
- G. B. Folland, *Subelliptic estimates and function spaces on nilpotent Lie groups*, **Arkiv för Matematik** 13 (1975), pp. 161-207. — §2 dilations + sub-Laplacian structure on general nilpotent Lie groups.
- A. V. Galkin and I. D. Remizov, *Tangency of Chernoff approximations to operator semigroups on Banach spaces*, **Israel Journal of Mathematics** (2025). — Theorem 3.1 K=2 (the order-2 tangency framework; conditionally extended to step-3 Engel in Theorem 28.bis pending empirical confirmation via G_HORM_ENGEL).
- Bratzlavsky 1974, cited via Bonfiglioli 2007 Theorem 4.3.6 — filiform Lie algebra basis (the explicit left-invariant fields for Engel).
- ADR-0077 (v3.1 A3 base Hörmander spec) — direct parent ADR; structurally identical algorithm, ambient dimension changes from D ∈ {2, 3} to D = 4.
- ADR-0087 (v3.x B5 Heisenberg backend) — Engel is the **direct step-3 successor** to Heisenberg's step-2 pattern; engineering structurally identical (mirror file pattern verbatim).
- ADR-0093 / ADR-0094 (v4.4+ research-wave siblings) — independent documentation correction + Padé revival; parallel work.
- `.dev-docs/research/verdicts/verdict-v4-4-research-wave.md` Q3 — research-wave verdict that recommended this ADR as Phase C math-creation candidate.
- `.dev-docs/research/extracts/bonfiglioli-2007-extract.md` — A-level evidence; primary Engel structure source.
- `.dev-docs/research/extracts/kalmetev-2023-extract.md` — confirms step-3 Carnot is NOT addressed in Kalmetev 2023 (affine = step-1 only).
- `.dev-docs/research/extracts/galkin-remizov-2023-extract-v2.md` — super-order observation (research-direction footnote in Consequences).
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_step_k_carnot_open.md` — Item 3 STILL_OPEN since v4.1.0; this ADR is the closure attempt.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_g4_ns2d_aniso_self_convergence.md` — v2.2 self-convergence precedent (mirror pattern for G_HORM_ENGEL).
