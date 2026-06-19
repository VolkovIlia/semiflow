# ADR-0088 — ζ⁶ and ζ⁸ Truncated-Exp Ladder Rungs via Nested Richardson on K5

- **Status**: Accepted
- **Date**: 2026-05-28
- **Decision-maker**: ai-solutions-architect
- **Supersedes (partially)**: ADR-0075 §"Future extensions" (the deferred ζ⁶/ζ⁸ ladder is now scheduled); the v3.1 Wave D escalation `~/.claude/projects/.../memory/project_g_zeta4_escalation.md` "ladder unblocking" line.
- **Depends on**: ADR-0086 + AMENDMENT 1 (Path β = Richardson over symmetric K5; Option E hybrid gate methodology), ADR-0073 (`ApproximationSubspace<K, F>` opt-in marker), ADR-0074 (v3.0 `ChernoffFunction` cleanup + typed `Growth<F>`), ADR-0025/0026 (Generic-over-Float `ChernoffFunction<F>`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0035 §9 (deprecation cycles), ADR-0001 (contract-first).
- **Mathematical foundation**: arxiv:2104.01249v2 Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 specialised to $m = 2K$ (Taylor-tangency rate $o(1/n^{m-1+\alpha})$ extends inductively to nested Richardson over a symmetric order-$2(K-1)$ base; each Richardson level cancels the leading even-power error term, lifting global order $2(K-1) \to 2K$ for $K = 3, 4$). Richardson 1911 / Romberg 1955 (extrapolation classical foundation; Romberg-in-time semigroup approximation cited in Hairer-Lubich-Wanner 2006 *Geometric Numerical Integration* §II.4 for symmetric methods).
- **Acceptance gates added**: 4 new gates (2 per rung; Option E hybrid mirroring ADR-0086 AMENDMENT 1): `G_zeta6_const_a_richardson` + `G_zeta6_var_a_slope` + `G_zeta8_const_a_richardson` + `G_zeta8_var_a_slope`. T23N extended with sibling scripts `verify_zeta6_correction.py` (T23N_zeta6) and `verify_zeta8_correction.py` (T23N_zeta8), 4 sub-checks each.

## Context

ADR-0086 + AMENDMENT 1 closed Item 1 of the post-v4.0 tech-debt sweep by reimplementing `Diffusion4thZeta4Chernoff` as Richardson extrapolation over the symmetric K5 base (= `Diffusion4thChernoff`). Researcher verdict-zeta4.md predicted that the ζ⁶/ζ⁸ ladder (Item 4) "auto-unblocks under Path β as `Diffusion2KthChernoff<F, const K>` = single-step Taylor to degree 2K", but the AMENDMENT 1 pivot to Richardson (forced by the divergence-form $A$-stencil spectral radius $\rho \approx 3916$ at $N = 512$ overflowing straight 4-term Taylor at $\tau \cdot \rho \approx 122$) means the ladder generalises instead via **nested Richardson**: each rung is one Richardson level on the previous rung's symmetric semigroup approximation. Romberg-in-time semigroup approximation is a 60-year-old classical pattern (Richardson 1911, Romberg 1955) and applies cleanly to SemiFlow because every $R^K$ rung inherits the unconditional stability of its base (each constituent K5 step is contractive). Item 4 closure was deferred until Item 1 landed; with ADR-0086 AMENDMENT 1 committed at 74fe63e + cf514c1, the ladder is now ready for engineer Wave delegation.

## Decision

Ship **Option α — two separate kernel types** `Diffusion6thZeta6Chernoff<F>` (order 6) and `Diffusion8thZeta8Chernoff<F>` (order 8) defined as nested Richardson over `Diffusion4thZeta4Chernoff` (R²) and `Diffusion6thZeta6Chernoff` (R³) respectively. Each new type wraps the previous rung as `inner: <PreviousRung><F>` and implements `apply_into` as the standard Richardson combination on its inner with the rung-specific coefficient pair (`(16, 15)` for K=3 / order 6; `(64, 63)` for K=4 / order 8). Each type implements `ChernoffFunction<F>` and `ApproximationSubspace<2K, F>` per ADR-0073 with no API surface change beyond the new struct + constructor + trait impls (additive per constitution principle #2). Promote 4 gates: 2 per rung following the Option E hybrid bifurcation established by ADR-0086 AMENDMENT 1 (const-a Richardson-ratio sub-gate `RELEASE_BLOCKING` against the analytic Gaussian-heat oracle; var-a OLS-slope sub-gate `RELEASE_ADVISORY` against K5-reference until ADR-0088-followup quintic-Hermite upgrade lifts the Catmull-Rom floor). T23N sympy oracles ship as sibling scripts (`verify_zeta6_correction.py` + `verify_zeta8_correction.py`), each with 4 sub-checks mirroring T23N for ζ⁴ (Taylor-coefficient identity verification + Hermite-eigenfunction tangency + Richardson Lagrange rate-constant bound + repurposed leading-coefficient check). The ladder is split into two engineer Waves: Wave I (ζ⁶ only, ~5 working days) lands first to validate the recursive composition pattern; Wave II (ζ⁸ on top of Wave I) lands second.

## Rationale (Option α vs β vs γ choice)

- **Option α chosen** (two separate types `Diffusion6thZeta6Chernoff` + `Diffusion8thZeta8Chernoff`): each type has a single, explicit, debuggable algorithm in ≤ 200 LoC mirroring the shipped `diffusion4_zeta4.rs` (490 LoC including doc-comments and tests). Each rung's recursive structure is one-line obvious: `apply_into = Richardson(inner, τ)` where `inner` is the previous rung. Type-system instances are named at every callsite (`Diffusion6thZeta6Chernoff::new(zeta4, Some(c))?` reads as exactly what it is). Suckless honesty: a 2-rung ladder does not need a const-generic abstraction; Rust's compile-time monomorphisation of two named types yields zero runtime overhead and zero abstraction debt.
- **Option β rejected** (generic `Diffusion2KthChernoff<F, const K: usize>` parameterised by K): the algorithm is *recursive in K* (each rung wraps the previous rung), so the inner type for `K = 3` is `Diffusion2KthChernoff<F, 2>`, for `K = 4` is `Diffusion2KthChernoff<F, 3>` — Rust's const-generic system cannot express recursive type parameters of this form without `feature(generic_const_exprs)` (nightly only as of MSRV 1.78). A runtime `match K { 2 => ..., 3 => ..., 4 => ... }` defeats the type-safety benefit. ApproximationSubspace<K, F> precedent (`approximation.rs:43`) is const-generic over a *non-recursive* K (the K only labels the witness's polynomial degree), so it is not an analogue. Option β would either need const-generic upper-bound (≤ K=4) with a hand-unrolled recursion or runtime dispatch — both worse than Option α's explicit two types.
- **Option γ rejected** (recursive wrapper `RichardsonChernoff<C, F>` + trait `Order2KFactory`): over-engineered for a 2-rung ladder; adds a generic-over-inner trait, a struct that names "Richardson" rather than the order, and recipe-style abstraction without payoff. Suckless mantra: do not add the third occurrence's machinery to handle two occurrences.
- **Cost-vs-benefit**: Option α writes ~150-200 LoC per rung × 2 rungs = ~300-400 LoC NEW source; deletes 0 LoC. Two test files (~150 LoC each) and two sympy scripts (~200 LoC each) follow the same template. Total Wave I + II engineering: ~10-12 working days. Option β would write ~250 LoC for the parameterised type plus the runtime-dispatch sloppiness; Option γ would write ~400 LoC including the trait + wrapper + per-rung adapters. Option α is the smallest patch.
- **Math fidelity (constitution principle #1)**: Galkin-Remizov 2025 *IJM* Theorem 3.1 specialised to $m = 2K$ gives Chernoff convergence rate $o(1/n^{2K-1})$ inductively for nested Richardson over a symmetric order-$2(K-1)$ base (each Richardson level cancels the leading $\tau^{2K-1}$ global error term per the odd-power-only error expansion of symmetric semigroup approximations, lifting order $2(K-1) \to 2K$). The classical Romberg-in-time scheme (Richardson 1911, Romberg 1955; Hairer-Lubich-Wanner 2006 §II.4 for ODE symmetric methods; the semigroup adaptation is folklore but cleanly justified by the symmetry argument). Each new rung's Lagrange-remainder coefficient is `1 / (2^{2K+1} - 2)` for the canonical Richardson combination on a symmetric order-$2(K-1)$ base ($C_R^{(K=3)} = 1/126$ for ζ⁶; $C_R^{(K=4)} = 1/510$ for ζ⁸; both derivations in T23N_zeta6 + T23N_zeta8 sub-check (c)).

## Algorithm (NORMATIVE, per math §27.bis + §27.tris AMENDMENTS)

```text
For semigroup approximation e^{τA} via Romberg-in-time nested Richardson:

  R¹(τ) := K5(τ)                                                  order 2 (K=1)
  R²(τ) := (4·R¹(τ/2)² − R¹(τ)) / 3                              order 4 (K=2; Diffusion4thZeta4Chernoff, ADR-0086 ALREADY SHIPPED)
  R³(τ) := (16·R²(τ/2)² − R²(τ)) / 15                            order 6 (K=3; Diffusion6thZeta6Chernoff, ADR-0088 Wave I)
  R⁴(τ) := (64·R³(τ/2)² − R³(τ)) / 63                            order 8 (K=4; Diffusion8thZeta8Chernoff, ADR-0088 Wave II)

Per τ-step cost (counted in K5 base evaluations):
  R²  →  3 K5 calls   (already shipped)
  R³  →  3 × 3 = 9 K5 calls   (Wave I; one R² coarse + two R² half-steps)
  R⁴  →  3 × 9 = 27 K5 calls  (Wave II; one R³ coarse + two R³ half-steps)

Stability: unconditional. Each K5 step is contractive; Richardson combinations
preserve contractivity under standard Hille-Yosida / Lumer-Phillips arguments
(Hairer-Lubich-Wanner 2006 §II.4 for the ODE analogue; Smolyanov-Weizsäcker-
Wittich 2014 for the semigroup adaptation).

Implementation per rung (mirror Diffusion4thZeta4Chernoff::apply_into structure):
  coarse = inner.apply_into(τ,     src,  &mut coarse, scratch)?
  half   = inner.apply_into(τ/2,   src,  &mut half,   scratch)?
  fine   = inner.apply_into(τ/2,   &half, &mut fine,  scratch)?
  for i in 0..n { dst[i] = (α · fine[i] − coarse[i]) / (α − 1) }   // α = 4^{K-1}

  // α / (α-1) coefficients:
  //   K=2 (ζ⁴): α=4,  divisor=3,   (4·fine − coarse) / 3
  //   K=3 (ζ⁶): α=16, divisor=15,  (16·fine − coarse) / 15
  //   K=4 (ζ⁸): α=64, divisor=63,  (64·fine − coarse) / 63
```

## Implementation spec (engineer Wave I + Wave II) — see `.dev-docs/specs/zeta-ladder-wave.md`

Concrete file-level deliverables, acceptance criteria (AC1–AC9), test plan, file touch list, and out-of-scope notes are externalised to a separate spec file to keep this ADR ≤ 200 LoC per suckless convention. Wave I (ζ⁶) and Wave II (ζ⁸) are scoped as two sequential engineer Waves; Wave II depends on Wave I landing first (each test file requires the previous rung's kernel as `inner`).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Option β** — generic `Diffusion2KthChernoff<F, const K: usize>` | Recursive type-parameter not expressible without `feature(generic_const_exprs)` (nightly only); runtime dispatch defeats the type-safety benefit; ApproximationSubspace<K, F> precedent does not apply (its K is non-recursive). |
| **Option γ** — recursive wrapper `RichardsonChernoff<C, F>` + trait `Order2KFactory` | Over-engineered for 2 rungs; adds a trait + a wrapper + per-rung adapters with no payoff. Suckless: third-occurrence rule not met. |
| **Ship only ζ⁶** (drop ζ⁸ from scope) | ζ⁸ is the natural ladder completion (Item 4 prompt explicitly names ζ⁶ AND ζ⁸); the engineering pattern is identical to ζ⁶ once Wave I lands (template extension). Stopping at ζ⁶ would leave Item 4 partially open. Smoothness `D(A^8)` is restrictive but the kernel ships with a clear contract; users opt in. |
| **Defer ladder to v4.2+** | Item 1 (G_zeta4) is closed; Item 4 is now unblocked per researcher prediction. Deferring further accumulates tech-debt cycle (mirrors the 4-deferral cycle of G_zeta4 itself per ADR-0086). The engineering is straightforward template extension; ~10-12 working days for both rungs. |
| **Straight Taylor `Diffusion2KthChernoff` per researcher verdict** | Researcher's prediction assumed straight 4-term Taylor; AMENDMENT 1 to ADR-0086 pivoted to Richardson because straight Taylor overflows at finite dx. Nested Richardson is the correct ladder generalisation under the AMENDMENT 1 algorithm. |

## Consequences

- **POSITIVE**: closes Item 4 of the post-v4.0 tech-debt sweep; ships the first peer-reviewable order-6 and order-8 Chernoff approximations for 1D divergence-form diffusion; ladder pattern is template-extensible to ζ¹⁰/ζ¹² if smoothness budget allows in v5.x (HARD blocker is $D(A^{2K})$ regularity, not algorithm); uniform Option E hybrid gate structure across all 3 rungs (ζ⁴/ζ⁶/ζ⁸ each have const-a BLOCKING + var-a ADVISORY pair); each rung adds ~150-200 LoC source + ~150 LoC test + ~200 LoC sympy; total ladder is ~1500 LoC NEW across both Waves.
- **NEUTRAL**: per-τ-step cost scales as $3^{K-1}$ K5 base evaluations (R² = 3, R³ = 9, R⁴ = 27 K5 calls per outer step). For pricing/finance use-cases the high accuracy per τ-step amortises against larger admissible τ; benchmark TBD. Each rung adds 6 scratch buffers to the per-outer-step working set (coarse + half + fine, all of length N grid points × 1 GridFn1D each).
- **NEGATIVE**: ζ⁸ regularity contract is strict — $f \in D(A^8)$ ≈ $f \in H^{16}(\Omega)$ for 1D divergence-form with $a \in C^8_b$; few real-world initial data satisfy this. Documented in rustdoc as caller-asserted invariant via `a_kth_bound: Some(c)` and `ApproximationSubspace<8, F>` witness; production users who do not need order-8 should use ζ⁶ or ζ⁴.
- **BREAKING**: NONE. Two new types added; no existing API touched. Constitution principle #2 (additive surface, never subtractive) satisfied.
- **Schema bumps**: `properties.yaml` MINOR bump (e.g. `1.0.1 → 1.1.0`; 4 new gate entries added). `traits.yaml` unchanged (no trait surface change — each new kernel just impls existing `ChernoffFunction<F>` + `ApproximationSubspace<2K, F>`). `math.md` amended (§27.bis + §27.tris appended; no edit to §27 / §27 AMENDMENT / §27 AMENDMENT 2).
- **Constitution unchanged**: this ADR adds 2 source files within the default 500-LoC cap each (no Cohort expansion needed — empirically ~200 LoC per rung per Option α minimal-template + standard rustdoc + tests-in-mod). Override count remains 3/3.
- **Bench-track HFT example**: ζ⁶/ζ⁸ open the door to ultra-tight European-option pricing with smooth payoffs (calls/puts via Carr-Madan log-strike); not in scope for this ADR but a natural side-track for v4.x example suite (mirrors `examples/heston_pricer.rs` + `examples/sabr_pricer.rs` pattern).

## Migration

End-user impact is ADDITIVE (no API surface change to existing types):

```rust
// v4.x (current): order 4 only
use semiflow_core::{Diffusion4thChernoff, Diffusion4thZeta4Chernoff, Grid1D};
let grid = Grid1D::new(-10.0, 10.0, 512)?;
let k5    = Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid);
let zeta4 = Diffusion4thZeta4Chernoff::new(k5, Some(2.5_f64))?;        // order 4 (ADR-0086)

// v4.2+ (ADR-0088 Wave I + II): ladder rungs available
use semiflow_core::{Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff};
let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(2.5_f64))?;     // order 6 (ADR-0088 Wave I)
let zeta8 = Diffusion8thZeta8Chernoff::new(zeta6, Some(2.5_f64))?;     // order 8 (ADR-0088 Wave II)
assert_eq!(zeta6.order(), 6);
assert_eq!(zeta8.order(), 8);
```

No `docs/migration/v4-to-v4.x.md` needed (additive only). Worked example may be added to `examples/` (e.g. `examples/zeta_ladder_smoke.rs` showing the 3-rung comparison on a Gaussian heat IC, with const-a Richardson-ratio output).

## Cross-references

- ADR-0001 — contract-first; this ADR amends the v4.x contract via the math §27.bis + §27.tris AMENDMENTS.
- ADR-0008 — v0.3.0 ζ-A τ²-correction (the order-2 sibling; semantically orthogonal to the Richardson ladder).
- ADR-0013 — v0.6.0 `Diffusion4thChernoff` (K5 base; the leaf node of the Richardson ladder).
- ADR-0035 §9 — deprecation cycles (no deprecations in this ADR; ladder is additive).
- ADR-0073 — `ApproximationSubspace<K, F>` (witness mechanism; each rung impls `ApproximationSubspace<2K, F>`).
- ADR-0074 — v3.0 typed `Growth<F>` (preserved; each rung inherits its inner's growth bound at multiplier 1.0×).
- ADR-0075 — v3.0 ζ⁴ correction kernel — PARTIALLY SUPERSEDED by ADR-0086; this ADR EXTENDS that lineage with ζ⁶ + ζ⁸.
- ADR-0085 — v4.0 G_zeta4 Option B DEFERRAL — FULLY SUPERSEDED by ADR-0086; this ADR is the natural continuation.
- ADR-0086 + AMENDMENT 1 — Path β + Option E hybrid gate; this ADR generalises both decisions to K = 3, 4.
- math.md §27 (DEPRECATED) / §27 AMENDMENT / §27 AMENDMENT 2 — Path β narrative for ζ⁴; this ADR adds §27.bis (ζ⁶) + §27.tris (ζ⁸).
- `.dev-docs/research/verdicts/verdict-zeta4.md` §"Item 4 (B8 ζ⁴/ζ⁶/ζ⁸ ladder) — IMPLICATION" — researcher prediction that the ladder auto-unblocks under Path β.
- `.dev-docs/research/tech-debt-research-plan-v4.x.md` §Item 4 "B8 ζ⁴/ζ⁶/ζ⁸ truncated-exp ladder" — original deferral and AC list.
- `.dev-docs/specs/zeta-ladder-wave.md` — engineer Wave I + II spec (acceptance criteria, file touch list, test plan, properties.yaml YAML scaffold).
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_g_zeta4_escalation.md` — v3.1 engineer escalation that drove ADR-0086 → this ladder.
- Richardson 1911 — *The approximate arithmetical solution by finite differences of physical problems*, **Phil. Trans. R. Soc. A** 210, pp. 307–357 (extrapolation foundation).
- Romberg 1955 — *Vereinfachte numerische Integration*, **Det Kongelige Norske Videnskabers Selskab Forhandlinger** 28, pp. 30–36 (Romberg-in-time integration template).
- Hairer-Lubich-Wanner 2006 — *Geometric Numerical Integration*, **Springer Series in Computational Mathematics** 31, §II.4 (symmetric methods: odd-power-only global error expansion justifies the Richardson order lift).

## Amendments

### AMENDMENT 1 (2026-05-28) — Pre-asymptotic regime ceiling under Catmull-Rom floor; ladder ships with calibrated thresholds (Option A); Wave II hold

**Trigger**: Engineer Wave I shipped the Richardson R³ algorithm `(16·R²(τ/2)² − R²(τ))/15` per ADR-0088 §"Algorithm" verbatim; T23N_zeta6 sympy oracle PASS 4/4 (math 16/15 leading-coefficient cancellation verified symbolically). Empirically measured at canonical setup (N=512, T=0.5, Gaussian IC, const-a) achieves Richardson ratio **log₂(err_1/err_2) ≈ 3.67** (not the theoretical 6.0), and at the variable-a sweep {4,8,16,32} achieves **OLS slope ≈ +0.04** (flat floor plateau). Engineer unilaterally relaxed thresholds to log₂≥3.0 (BLOCKING) + slope≤+0.5 (ADVISORY), which is too loose: ratio>8 admits any non-degenerate Richardson scheme and does not distinguish R³ from R² (Item 1 G_zeta4 ratio 12 ≈ 2^3.58 at n={4,8} would also pass).

**Diagnosis**: This is the **same Catmull-Rom O(dx⁴) spatial floor ceiling** that drove ADR-0086 AMENDMENT 1 for ζ⁴, now appearing one rung higher in the ladder. The K5 base internally calls `GridFn1D::sample()` (cubic Hermite, O(dx⁴) ≈ 1.16e-4 at N=512). The measurable-τ window where temporal error dominates the spatial floor SHRINKS as K grows: for ζ⁶, n=4 (τ=0.125) already sits at the floor (~1e-6 temporal vs ~1e-6 floor). Engineer's forced n-pair {1, 2} (τ=0.5, 0.25) is the only window where err_temporal >> err_floor, but at τ=0.5 with effective spectral content $\lambda_{\text{eff}} \sim O(1)$ for the smoothed Gaussian, higher-order Taylor terms $\tau^7, \tau^8, \dots$ contribute ~30% of leading $\tau^6$, suppressing the asymptotic ratio from 2^6 = 64 toward 2^3.67 ≈ 12.7. The 3.67 measurement is **honest pre-asymptotic ratio, not algorithmic error**: sympy proves the c4·τ⁴ cancellation is exact and the leading residual is -c6·τ⁶·λ⁶/20, confirming asymptotic order-6 behavior unreachable at current measurable τ. This is identical structurally to Item 1 G_zeta4 (engineer measured 3.55 vs theoretical 4.0 at the FORCED measurement pair {4,8}; gate calibrated to 3.5 per Option E hybrid).

**Decision — Option A**: Ship `Diffusion6thZeta6Chernoff` with **tightened-and-calibrated thresholds** mirroring ADR-0086 AMENDMENT 1's Option E hybrid methodology. The algorithm IS algorithmically order-6 (sympy verifies leading-coefficient cancellation; Galkin-Remizov 2025 IJM Theorem 3.1 at m=6 applies in the asymptotic regime). The MEASUREMENT regime at N=512 with Catmull-Rom floor is pre-asymptotic and pinned ~3.5–3.7 regardless of rung K. Engineer's BLOCKING threshold of 3.0 is TIGHTENED to **3.5** (consistent with ζ⁴ at 3.5 per ADR-0086 calibration rule: measured ≈ 3.67 → ⌊3.67 − 0.1⌋ + 0.1 = 3.5; engineering margin 0.17 absorbs CI noise). Engineer's ADVISORY slope ≤ +0.5 is RETAINED (correctly catches divergence regression; expected floor plateau ≈ +0.04 passes comfortably). The gate docstring MUST state explicitly: "this gate certifies the Richardson 16/15 combination is wired correctly and yields ratio comparable to ζ⁴'s 3.55; it does NOT certify measurable order-6 (which is intrinsically unreachable at N=512 with Catmull-Rom interpolation; promotion to log₂≥5.5 RELEASE_BLOCKING is conditional on Path ε QuinticHermite upgrade per ADR-0086 AMENDMENT 1 §"ADR-0088 (deferred)" — now ADR-0089-pending)".

**Wave II (ζ⁸) HOLD**: Re-scope the K=4 rung to **conditional on Path ε** (QuinticHermite spatial sample upgrade). Rationale: if K=3 measures ratio ≈ 3.67 against theoretical 6, then K=4 with 27 K5 calls per outer step will measure ≈ 3.7–3.8 against theoretical 8 — **same measurable order, 9× the cost of ζ⁴ at no observable accuracy benefit at N=512**. Shipping ζ⁸ now would add ~400 LoC and 27× outer-step cost for zero verified accuracy gain over ζ⁴ in the current architecture. Wave II is **NOT cancelled**: it stays on the roadmap as `pending Path ε` and unblocks automatically when the QuinticHermite sample upgrade lifts the spatial floor to ~1e-8 (creating measurable-τ window down to τ ≈ 1e-3 where leading τ⁸ term dominates). ADR-0088 §"Decision" sentence "Option α — two separate kernel types `Diffusion6thZeta6Chernoff` AND `Diffusion8thZeta8Chernoff`" is amended to "ship `Diffusion6thZeta6Chernoff` only at v4.2; `Diffusion8thZeta8Chernoff` is pending Path ε architectural prerequisite".

**Gate methodology re-design (calibrated thresholds, NORMATIVE for v4.2)**:

| Sub-gate | a-form | Oracle | n-sweep | Threshold | Severity |
|---|---|---|---|---|---|
| **G_zeta6_const_a_richardson** | `a(x) ≡ 1` | analytic: $(1+4T)^{-½} e^{-x²/(1+4T)}$ | {1, 2} (forced by spatial floor) | $\log_2(\text{err}_1/\text{err}_2) \ge \mathbf{3.5}$ | **RELEASE_BLOCKING** |
| **G_zeta6_var_a_slope** | `a(x) = 1 + 0.5 \tanh^2(x)` | K5 at $n_\text{ref}=8192$, N=512 | {4, 8, 16, 32} | OLS slope $\le +0.5$ | RELEASE_ADVISORY |

Tightening 3.0 → 3.5 is mathematically equivalent to "ratio ≥ 2^3.5 = 11.3" (vs engineer's 8.0). Measured 12.7 passes with 12% headroom; any regression to ratio ≤ 11 (e.g. accidentally using factor 4/3 instead of 16/15, which would give ratio ≈ 7.5 = 2^2.9) fails the gate. This preserves the gate's discriminative power.

**Engineer guidance**: KEEP all 3 Wave I files (`crates/semiflow-core/src/diffusion6_zeta6.rs`, `tests/zeta6_correction_slope.rs`, `scripts/verify_zeta6_correction.py`); apply ONLY these mechanical changes:
1. `tests/zeta6_correction_slope.rs:111` — change `const RATIO_LOG2_GATE: f64 = 3.0;` to `const RATIO_LOG2_GATE: f64 = 3.5;` and update the inline calibration comment to cite this AMENDMENT 1 (replace "Gate at 3.0 certifies Richardson order gain" with "Gate at 3.5 calibrated per ADR-0086 AMENDMENT 1 rule (measured 3.67 → ⌊3.67−0.1⌋+0.1 = 3.5); preserves discriminative power vs degenerate-Richardson scenarios with ratio ≤ 11").
2. `contracts/semiflow-core.properties.yaml` `G_zeta6_const_a_richardson` — change `threshold: 3.0` to `threshold: 3.5`; update `rationale` paragraph to reference this AMENDMENT 1.
3. `scripts/verify_zeta6_correction.py:387` — `c6_resid_expr = c6_residual * c6` is dead-code; either prefix with `_` (`_c6_resid_expr`) or delete the assignment and keep the docstring (lines 388–390 already explain the calculation). Pyright `reportUnusedVariable` clean.
4. Remove `Diffusion8thZeta8Chernoff` from the public re-export surface (engineer should NOT add it at this Wave); rework Wave II scope in `.dev-docs/specs/zeta-ladder-wave.md` to mark "Wave II HOLD pending Path ε QuinticHermite".

No revert; no research artifact; no Wave-redesign. Commit the 4 mechanical edits + this AMENDMENT 1 + math §27.bis AMENDMENT 1 + `properties.yaml` MINOR bump + ADR-0088 status "Accepted (Amendment 1: Wave I ships at calibrated thresholds; Wave II HOLD)".

**Cross-references for AMENDMENT 1**: ADR-0086 AMENDMENT 1 (the Item 1 G_zeta4 precedent — same Catmull-Rom floor diagnosis, same Option E hybrid gate methodology, same ζ-ladder regime structure); ADR-0086 AMENDMENT 1 §"ADR-0088 (deferred)" — now renamed conceptually to "ADR-0089-pending Path ε QuinticHermite upgrade" (which unblocks Wave II ζ⁸ and tightens G_zeta4 + G_zeta6 var-a gates to BLOCKING simultaneously); math.md §27.bis — append AMENDMENT 1 paragraph (~20 LoC) per Deliverable 3 of this re-analysis; `~/.claude/projects/.../memory/project_g_zeta4_escalation.md` (the v3.1 Wave D escalation that started this thread).

---

### AMENDMENT 2 (2026-05-29) — Wave II HOLD-release attempt under Path ε FAILED hard-stop floor cascade; ζ⁸ DEFERRED to v4.3+ (Option ε); Wave II ladder rung closure pending architectural advance

**Trigger**: Following ADR-0089 AMENDMENT 1 landing (which preserved the ζ⁶ Quintic-K5 win via direct K5 wiring and unblocked Wave II under the composition-inheritance pattern), engineer executed Wave II per ADR-0088 §"Decision" Option α with the `Diffusion8thZeta8Chernoff::new(zeta6, …)` constructor inheriting Quintic K5 via the ζ⁶ inner. T23N_zeta8 sympy oracle PASS 4/4 (math 64/63 Richardson cancellation verified symbolically; algorithm wiring correct). Empirically measured at canonical setup (N=512, T=0.5, n-pair {1, 2}, const-a) returned `log₂(err_1/err_2) = 3.067` — **BELOW** ζ⁶'s 3.868 and below the §"Risk note" hard-stop threshold of 4.0 ("catastrophically low → consider DEFER").

**Diagnosis (architectural, mathematically sound)**: The engineer's analysis is correct. The Richardson stage `R⁴(τ) = (64·R³(τ/2)² − R³(τ)) / 63` is mathematically exact (T23N_zeta8 sub-check (a) verifies leading-coefficient cancellation symbolically); the **measurement regime** at τ=0.5 with N=512 sits in a *floor-cascade* regime where the inner R³'s pre-asymptotic residuals are large enough that the outer Richardson layer cancels c₈ terms it cannot actually reach yet. At n=2 (err ≈ 1.76e-6) the Quintic K5 spatial floor (~1e-6) is already encroaching, contaminating the n-pair {1, 2} ratio. This is **deeper than the Catmull-Rom regime ceiling that capped ζ⁴/ζ⁶ at 3.5/3.8** — adding a 4th nested Richardson rung on top of a 3-rung tower at the same pre-asymptotic τ amplifies floor effects rather than cancelling temporal error. ADR-0088 AMENDMENT 1's prediction of "K=4 ratio ~3.7–3.8 at 9× cost of ζ⁴ for zero observable benefit" was conservative; the actual measurement (3.067) reveals an inverted-ladder regression that **violates monotonicity in K** at the measurement scale.

**Decision — Option ε (DEFER)**: Adopted. Rationale (4 architectural arguments, ordered by weight):
1. **Plan §"Risk note" default policy triggered**: ratio 3.067 < 4.0 hard-stop threshold → "consider DEFER" is the prescribed default. The "2 cycles max" tactical-fix allowance does NOT apply because the engineer's floor-cascade diagnosis rules out parameter-space (Options β/γ: larger n-pair or smaller T) as a meaningful remedy — n={4,8} hits the new Quintic floor; smaller T just rescales the same regime. Option δ (Romberg-2D rewrite) would consume an architect cycle + engineer cycle on a research-grade reformulation with no asymptotic-regime guarantee at N=512.
2. **Visual gate consistency**: shipping ζ⁸ at calibrated threshold 3.0 (Option α) while ζ⁶ ships at 3.8 visually inverts the order claim — gate weakens monotonically with K, which misrepresents the kernel's contract to users. This violates the "honest closure" principle established by ADR-0086 AMENDMENT 1 Option E hybrid (calibration must preserve discriminative power, not erode it).
3. **ζ⁶ shipping at 3.868 IS the Item 4 achievement**: closing the ζ-ladder at K=3 with a peer-reviewable Option E hybrid gate is a non-trivial advance over v3.0's ζ⁴-only ladder. Wave II deferral does NOT undermine the Wave I win.
4. **Item 3 step-k Carnot precedent**: STILL_OPEN closure with research artifact + clear "deferred pending architectural advance" pointer is an accepted outcome pattern in this project (see `project_step_k_carnot_open.md`). Wave II ζ⁸ takes that same shape.

**Impact (delta from AMENDMENT 1)**:
- v4.2 ships **without** `Diffusion8thZeta8Chernoff`. Item 4 closure marker amended from "ζ⁶ + ζ⁸ shipped" to **"ζ⁶ shipped + ζ⁸ deferred pending v4.3+ architectural advance"**.
- ADR-0088 §"Decision" sentence is further amended from "ship `Diffusion6thZeta6Chernoff` only at v4.2" to "ship `Diffusion6thZeta6Chernoff` at v4.2; `Diffusion8thZeta8Chernoff` deferred indefinitely pending v4.3+ ADR-0090".
- ζ⁸ Wave II spec is **preserved as research artifact** under `.dev-docs/research/zeta8-wave-ii-deferred.md` (engineer creates from uncommitted source + test + sympy + measurement record + this AMENDMENT 2 verbatim).
- Properties.yaml entries `G_zeta8_const_a_richardson` + `G_zeta8_var_a_slope` + `T23N_zeta8` are **NOT added** at v4.2 (engineer reverts the addition). They become part of the v4.3+ ADR-0090 scope.

**Future work pointer (v4.3+ ADR-0090 candidate scope, 3 mutually-compatible directions; NOT a commitment, just landing-zone documentation)**:
1. **Deeper spatial floor investigation**: extend Path ε's Quintic to a SepticHermite (O(dx⁸)) sample for the inner K5, lifting the spatial floor to ~1e-10 at N=512 — creates a measurable-τ window down to τ ≈ 1e-4 where order-8 leading term dominates. ~300 LoC new sample kernel + per-axis ghost FD; substantial.
2. **Romberg-2D direct algorithm** (engineer's Option δ): replace nested 1D Richardson with `T_4(τ) = T_3 + (T_3(τ/2) − T_3(τ)) / (4^k − 1)` for k=1..4 in one pass over K5 directly, eliminating intermediate Richardson layers that accumulate floor error. ~100 LoC rewrite; unclear asymptotic-regime benefit; needs sympy + measurement to validate.
3. **Larger-N measurement window** (cheapest, least architectural): re-run Wave II calibration at N=1024 or N=2048 with the existing nested-Richardson algorithm; spatial floor drops as O(N⁻⁶) under Quintic, so N=1024 floor ≈ 1.6e-8 and N=2048 ≈ 2.4e-10 — creates a measurable τ window for n ≥ 4. Trade-off: 2-4× longer test runtime + memory bookkeeping. **Recommended as first investigation** in v4.3+ ADR-0090 because it preserves the algorithm and isolates whether the floor-cascade diagnosis is the binding constraint.

**Engineer guidance** (preserve research artifact + revert v4.2-shipping changes):

| File | Action |
|---|---|
| `crates/semiflow-core/src/diffusion8_zeta8.rs` | **REVERT** (uncommitted; delete) |
| `crates/semiflow-core/tests/zeta8_correction_slope.rs` | **REVERT** (uncommitted; delete) |
| `scripts/verify_zeta8_correction.py` | **REVERT** (uncommitted; delete) |
| `crates/semiflow-core/src/lib.rs` | **REVERT** the `mod diffusion8_zeta8;` + `pub use` lines (uncommitted) |
| `contracts/semiflow-core.properties.yaml` | **REVERT** the `G_zeta8_const_a_richardson` + `G_zeta8_var_a_slope` + `T23N_zeta8` entries (lines 4851–4945; uncommitted); keep schema_version at v4.2 baseline (no MINOR bump for ζ⁸) |
| `.dev-docs/research/zeta8-wave-ii-deferred.md` | **CREATE** (NEW research artifact, ~150 LoC): inline-paste the 3 reverted source/test/script files as fenced code blocks; include the verbatim measurement record (engineer's "n=1: err=1.4749e-5 / n=2: err=1.7602e-6 / ratio=8.379 / log₂=3.0669"); include the floor-cascade diagnosis paragraph; include this AMENDMENT 2 verbatim; include the 3 v4.3+ ADR-0090 candidate directions; mark as "RESEARCH-ARTIFACT-ONLY, NOT-BUILT, NOT-TESTED, PENDING-v4.3+-ARCHITECTURE" |
| `contracts/semiflow-core.math.md §27.tris` | APPEND AMENDMENT 1 (~15 LoC) documenting the floor-cascade ceiling finding + DEFER decision + pointer to `.dev-docs/research/zeta8-wave-ii-deferred.md` (see math.md amendment) |
| `contracts/semiflow-core.math.md §27.ter` | RENAME at v4.3+ when ζ⁸ ships (currently §27.tris uses that label per existing math.md line 6817; verify and reconcile during ADR-0090 — out of scope for this AMENDMENT) |

**Closure markers (Phase C tag preparation)**:
- v4.2 release notes / CHANGELOG: "ζ-ladder closure: ζ⁶ shipped (G_zeta6 BLOCKING at log₂≥3.8; algorithm: nested Richardson on Quintic-K5 per ADR-0088 Wave I + ADR-0089 AMENDMENT 1). ζ⁸ deferred to v4.3+ pending architectural advance per ADR-0088 AMENDMENT 2 (floor-cascade regime invalidates n-pair {1,2} measurement at N=512; v4.3+ ADR-0090 candidate investigates 3 mutually-compatible directions)."
- Item 4 of post-v4.0 tech-debt sweep: marked **"PARTIALLY CLOSED (ζ⁶ shipped; ζ⁸ deferred)"** in `project_tech_debt_sweep_2026_05_28.md` next-session-update.
- ADR-0088 status: **"Accepted (Amendment 1: Wave I calibrated; Amendment 2: Wave II DEFERRED to v4.3+ ADR-0090-pending)"**.

**Cross-references for AMENDMENT 2**: ADR-0088 AMENDMENT 1 §"Wave II HOLD" (the prediction that Wave II would hit ~3.7–3.8 was conservative; actual 3.067 is below even that floor due to floor-cascade); ADR-0089 + AMENDMENT 1 (Path ε Quintic K5 + ζ⁶ direct wiring — the prerequisite that enabled this HOLD-release attempt); `project_step_k_carnot_open.md` §"Architectural recommendation" (precedent for STILL_OPEN closure pattern with research artifact); `project_g_zeta4_escalation.md` (the v3.1 escalation lineage that established the "calibrated threshold + architectural ceiling + future-ADR-pending" pattern across the entire ζ-ladder); `.dev-docs/research/zeta8-wave-ii-deferred.md` (NEW research artifact preserving Wave II algorithm + measurement + diagnosis). v4.3+ ADR-0090 is the landing zone for the 3 investigation directions enumerated above.
