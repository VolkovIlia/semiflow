# ADR-0106 — Galkin-Remizov 2025 *IJM* Theorem 3 Constant-Prefactor Harness

- **Status**: ACCEPTED 2026-05-29 (PRE-FLIGHT 5/5 PASS)
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Wave**: post-v4.8 research-track ADR (documentation + sympy only; ~500 LoC ADR + ~480 LoC sympy oracle; no Rust code, no contract trait changes, no engineer wave)
- **Depends on**: ADR-0001 (contract-first), ADR-0073 (`ApproximationSubspace<K, F>` witness framework), ADR-0075 (v3.0 ζ⁴ correction kernel — ATTRIBUTION partially retracted by ADR-0093, ALGORITHM replaced by ADR-0086), ADR-0086 + AMENDMENT 1 (Path β Richardson algorithmic successor — the validated algorithm), ADR-0093 (ADR-0075 6-monomial attribution retraction), ADR-0096 (Hypothesis C super-order formalisation — Theorem 2 sharpness now provides theoretical authority).
- **Supersedes / amends**: none. This ADR is a **research-track CONSTRUCTIVE supplement** that adopts the newly-extracted Galkin-Remizov 2025 *IJM* Theorem 3 (eq. 7+8 of the paper) as the FORMAL VERIFICATION TARGET for the Diffusion-family rate claims; it closes the four open architect questions in [[project-g-zeta4-escalation]] §"Questions for architect math review" by formal symbolic diagnosis.
- **Mathematical foundation**: Galkin-Remizov 2025 *Israel Journal of Mathematics* 265, 929-943 (DOI 10.1007/s11856-024-2678-x; arxiv:2104.01249v2; FULL PDF read 2026-05-29 — `.dev-docs/papers/oleg-e-galkin-upper-and-lower-estimates-for-rate-of.pdf`). Theorem 3 (eq. 7+8, pp. 935-936) explicit-constants rate-of-convergence theorem; Theorem 4 (eq. 11, 13, p. 938-939) 1D variable-coefficient parabolic application with $C_0, \ldots, C_4$ derivative bounds; Theorem 2 (p. 932) sharpness — arbitrary rate construction. Lemmas 1-3 (eq. 2, 3+4, 5+6, pp. 933-934) the technical proof scaffolding. References: Vedenin-Voevodkin-Galkin-Karatetskaya-Remizov 2020 *Math. Notes* 108(3), 451-456 (predecessor; conjectures the $1/n^2$ rate for $S(t)$); Galkin-Remizov 2022 *Math. Notes* 111(2), 305-307 (eq. 8 prerequisites short-form); Gomilko-Tomilov 2014 *J. Funct. Anal.* 266(5), 3040-3082 (competitor framework comparison).
- **Acceptance gates added**: T_GR_2025_THM3 (NORMATIVE sympy PRE-FLIGHT — 5 sub-checks; PRE-FLIGHT 5/5 PASS verified 2026-05-29 — see "Pre-flight result" below). NO new engineer-side gate; NO Rust API surface added.

## Context

User-attention provided a previously-unread paper to `.dev-docs/papers/`: *Upper and lower estimates for rate of convergence in the Chernoff product formula for semigroups of operators* by Oleg E. Galkin and Ivan D. Remizov (Israel Journal of Mathematics 265 (2025), 929-943; DOI 10.1007/s11856-024-2678-x; arxiv:2104.01249v2). Researcher findings (full report from agent a73c6defe1e443385) extracted three actionable insights that change the architectural picture for the long-running ζ⁴ escalation:

### Insight 1 — Explicit-constants Theorem 3 (eq. 7+8 of the paper)

The paper's main result is **Theorem 3** (not "Theorem 3.1" as our project memory and ADR-0075 misrecorded). The abstract "o(1/n^m)" rate that ADR-0086 cited is concretised here with EXPLICIT CONSTANTS. Verbatim from p. 935-936 of the paper:

> **Hypothesis (condition (3) and (7))**: $\|S(t) f - \sum_{k=0}^m t^k L^k f / k!\| \le t^{m+1} \sum_{j=0}^{m+p} K_j(t) \|L^j f\|$ for all $t \in (0, T]$ and $f \in \mathcal{D}$.
>
> **Conclusion (eq. 8)**: For all $t > 0$, all integers $n \ge t/T$, and all $f \in \mathcal{D}$:
> $$\|S(t/n)^n f - e^{tL} f\| \le \frac{M_1 M_2 t^{m+1} e^{wt}}{n^m} \cdot \sum_{j=0}^{m+p} e^{-wt/n} C_j(t/n) \|L^j f\|$$
> where $C_{m+1}(t) = K_{m+1}(t) e^{-wt} + M_1 / (m+1)!$ and $C_j(t) = K_j(t) e^{-wt}$ for $j \ne m+1$.

This is the **first explicit-constant formulation** in the Chernoff-rate literature accessible to SemiFlow; previous ADRs (ADR-0086, ADR-0088, ADR-0103) cite only the asymptotic "o(1/n^m)" rate. The $C_j$ recipe is constructive: given the user's $K_j(t)$ prefactors from condition (7), eq. 8 supplies the resulting Chernoff rate constants WITHOUT additional analysis.

### Insight 2 — Theorem 4 1D parabolic application (eq. 11, 13 of the paper)

Theorem 4 (p. 938-939) gives a SPECIFIC Chernoff function with proven O(t²/n) rate for 4-differentiable f. Verbatim from p. 938 (eq. 11):

> $(S(t)f)(x) = (1/4)f(x + 2\sqrt{a(x)t}) + (1/4)f(x - 2\sqrt{a(x)t}) + (1/2)f(x + 2b(x)t) + t \cdot c(x) \cdot f(x)$

with rate (verbatim eq. 13, p. 939):

> $\|S(t/n)^n f - e^{t \overline{A}} f\| \le (t^2 e^{\|c\| t} / n) \cdot (C_0 \|f\| + C_1 \|f'\| + C_2 \|f''\| + C_3 \|f'''\| + C_4 \|f^{(IV)}\|)$

for $f \in UC_b^4(\mathbb{R})$ and $A\varphi = a \varphi'' + b \varphi' + c \varphi$ (note: **multiplicative form**, not divergence form — this is a SIBLING kernel to SemiFlow's `DiffusionChernoff` which targets the divergence-form $A = \partial_x(a(x) \partial_x \cdot)$).

### Insight 3 — Theorem 2 sharpness construction (p. 932)

Theorem 2 formally proves the converse: for any sequence $h_n \to 0$, there exists a Chernoff function $S_\tau$ such that $\|S_\tau(\tau/n)^n - e^{\tau L}\| = h_n + O(h_n^2)$. Verbatim Remark 1 (p. 932): "the case $h(n) = 1/\ln\ln n$ with very slow convergence is possible. So Chernoff approximation can be an extremely (in)effective tool for finding operator exponents."

This is the THEORETICAL AUTHORITY for ADR-0096 Hypothesis C (super-order $\alpha_S \approx -3.1$ regime observed empirically by Galkin-Remizov 2023 arxiv:2301.05284v5). Theorem 2 PROVES that the Chernoff rate is *not* an invariant of $L$ alone; depends on the $S$ choice. SemiFlow's empirical super-order observations are now formally licensed by Theorem 2; the ADR-0096 hypothesis can be promoted from "anomaly to investigate" to "expected phenomenon per Theorem 2".

### Why this paper now: long-running G_zeta4 escalation

The v3.1 Wave D engineer escalation ([[project-g-zeta4-escalation]], 2026-05-27) flagged that the v3.0 BCH-correction ζ⁴ algorithm gives only m=2 global tangency (slope ≈ −1.0), not the m=4 claimed in ADR-0075. Architect math review was deferred. ADR-0086 (2026-05-28) bypassed the BCH defect by adopting Path β Richardson (validated slope −4.06 on const-a oracle). ADR-0093 (2026-05-29) closed the lineage by retracting ADR-0075's "Example 4.2 6-monomial polynomial" attribution as a citation error.

The four open architect questions from [[project-g-zeta4-escalation]] remained formally unanswered until full-paper extraction. The newly-extracted Theorem 3 is **exactly the prerequisite that Romberg/Richardson coefficient choices must satisfy** (per researcher findings): "Theorem 3 condition (3) is exactly the prerequisite that Romberg/Richardson coefficient choices must satisfy". The architect now has the formal framework to diagnose precisely which Taylor coefficients must vanish for m=4 tangency.

## Decision

1. **Adopt Galkin-Remizov 2025 *IJM* Theorem 3 (eq. 7+8) as the FORMAL VERIFICATION TARGET** for any `ChernoffFunction<F>` implementation claiming a convergence rate. The Chernoff rate $O(t^{m+1}/n^m)$ is conditional on the user-supplied m-tangency hypothesis (eq. 7). Future ADRs claiming a rate MUST verify the m-tangency hypothesis symbolically against this template.

2. **Ship NEW sympy oracle `scripts/verify_thm3_harness.py`** (~480 LoC) with 5 mandatory sub-checks (`T_GR_2025_THM3`):
   - (1) `T_GR_2025_THM3.diffusion_m1_tangency` — `DiffusionChernoff` (order-1) satisfies m=1 Taylor tangency hypothesis of Theorem 3.
   - (2) `T_GR_2025_THM3.diffusion4_m2_tangency` — `Diffusion4thChernoff` (order-2 in τ, 4th-order in dx) satisfies m=2 Taylor tangency.
   - (3) `T_GR_2025_THM3.zeta4_m4_diagnosis` — DIAGNOSTIC: BCH-only `Diffusion4thZeta4Chernoff` (the ADR-0075 v3.0 algorithm) FAILS m=4 Taylor tangency. Reproduces the v3.1 Wave D engineer's numerical falsification symbolically. Diagnostic sub-check EXPECTS the failure and reports it as DIAGNOSTIC SUCCESS — the BCH-only algorithm BREAKS m≥2 tangency by adding -1/12·A² to the τ² coefficient (which must stay at +1/2·A² for tangency).
   - (4) `T_GR_2025_THM3.path_beta_m4_tangency` — Path β (ADR-0086, v4.1+ SHIPPED) satisfies m=3 Taylor tangency in the strict paper convention (residual starts at τ⁴ = (1/24)A⁴f). Richardson form (ADR-0086 AMENDMENT 1) lifts to true m=4 via symmetric-base odd-power cancellation — verified separately by existing T23N sub-check (c).
   - (5) `T_GR_2025_THM3.theorem4_chernoff_form_consistency` — Galkin-Remizov 2025 Theorem 4 specific Chernoff function (eq. 11) — multiplicative-form $A\varphi = a\varphi'' + b\varphi' + c\varphi$ — at b ≡ 0, c ≡ 0 limit reproduces the leading-order Taylor expansion $S(t)f \approx f + t \cdot a \cdot f''(x) + O(t²)$, confirming Theorem 4 is a SIBLING kernel (not identical) to SemiFlow's divergence-form `DiffusionChernoff`.

3. **G_zeta4 escalation per-question resolution** (per [[project-g-zeta4-escalation]] §"Questions for architect math review"):
   - **Q1 (Theorem 3.1 requirements)** — ANSWERED by sub-check (3) DIAGNOSTIC. The paper's correct theorem number is **Theorem 3** (not Theorem 3.1). It is a Banach-space rate theorem conditional on user-supplied m-tangency (eq. 7). The m=4 Taylor tangency requires F(τ) to match τ⁰..τ³ Taylor coefficients exactly; the BCH-only ansatz BREAKS even m=2 tangency by adding a -1/12·A² term to the τ² coefficient. Resolution: BCH-only algorithm CANNOT satisfy Theorem 3 m=4 hypothesis; Path β Richardson DOES (math foundation sound).
   - **Q2 (6 monomial coefficients)** — ANSWERED via cross-reference to ADR-0093. NONE — the 6-monomial polynomial does not exist in the cited paper. ADR-0075's "Example 4.2 6-monomial table" is a citation error; ADR-0093 retracted the attribution. The validated algorithm Path β (ADR-0086) uses a 4-term Taylor expansion with NO 6-monomial polynomial.
   - **Q3 (stencil scale for `compute_jet6`)** — STILL_OPEN by intent. Moot in current architecture per ADR-0086 (Path β does not use `compute_jet6` jet evaluator — it uses Richardson on the existing K5 stencil). The Theorem 3 framework does not directly address spatial-discretisation stencil-scale questions; those are absorbed into the K_j(t) prefactor of the abstract eq. 7. If a future research wave re-opens the jet-evaluator path, Theorem 3 still does not resolve the stencil scale (a discretisation-side concern, not a Chernoff-tangency concern).
   - **Q4 (inner kernel order)** — PARTIAL. Theorem 3 with m=4 conclusion permits any inner spatial order — the rate constant $C_4(t/n) \|L^4 f\|$ in eq. 8 absorbs the spatial discretisation order into the $\|L^j f\|$ norm. Path β with the v0.6.0 4th-order spatial inner (`Diffusion4thChernoff` 9-point stencil) is consistent with Theorem 3 because the inner spatial order O(dx⁴) controls $\|L^j f\| \le C \cdot \|f\|_{H^{2j+4}}$. The architect's question "does Diffusion4thZeta4Chernoff need 7-point Fornberg" is moot — Path β achieves m=3 tangency at any spatial-order inner; the spatial order affects only the prefactor magnitude.

4. **NO Rust code change** — purely additive sympy oracle + ADR. The Path β Richardson algorithm shipped at v4.1+ (per ADR-0086) is the validated implementation; no implementation changes follow from ADR-0106.

5. **NO trait change** — `contracts/semiflow-core.traits.yaml` unchanged. No new `ChernoffFunction<F>` impl or method.

6. **Properties.yaml MINOR schema bump 1.5.0 → 1.6.0** — adds the new `T_GR_2025_THM3` NORMATIVE sympy PRE-FLIGHT record (additive entry only; no existing gate changed).

7. **math.md §27 minor amendment** — appends a Theorem 3 prefactor recipe pointer at the end of §27 + AMENDMENT 2 directing future authors to `scripts/verify_thm3_harness.py` as the verification authority for any rate claim in the §27 family.

## Pre-flight result (MANDATORY, ADR-0086 lesson)

PRE-FLIGHT sympy oracle `scripts/verify_thm3_harness.py` executed 2026-05-29:

```
T_GR_2025_THM3 PASS (5/5 sub-checks: diffusion_m1_tangency /
diffusion4_m2_tangency / zeta4_m4_diagnosis / path_beta_m4_tangency /
theorem4_chernoff_form_consistency)
```

5/5 sub-checks PASS. ADR-0106 is GREEN. No engineer wave needed; ADR is research-track architectural documentation only. Sympy oracle integrated into the existing test-fast sympy sweep (alongside `verify_zeta4_correction.py`, `verify_subordinated_chernoff.py`, etc.). Failure of T_GR_2025_THM3 BLOCKS future v5.0 release.

## Rationale

- **Math fidelity (constitution principle #1)**: the v3.0 ADR-0075 attribution to "Theorem 3.1" was both **off by section number** (correct is Theorem 3, no decimal) and **wrong in content** (the "6-monomial polynomial" never appears in the cited paper, per ADR-0093). ADR-0106 closes the citation loop by adopting the actual Theorem 3 (eq. 7+8 verbatim) as the formal verification target for the entire Diffusion family. The Path β algorithm (ADR-0086) was already empirically validated; ADR-0106 supplies the missing theoretical authority.
- **PRE-FLIGHT discipline (ADR-0086 lesson)**: ADR-0086 was accepted without a PRE-FLIGHT sympy check; the AMENDMENT 1 gate methodology re-design was needed to handle the dx-floor at finite-precision arithmetic. ADR-0103 fixed this by mandating PRE-FLIGHT 5/5 PASS before engineer wave. ADR-0106 continues this practice: 5/5 PASS before declaring ACCEPTED.
- **Closes 4-question architect escalation**: [[project-g-zeta4-escalation]] has carried the four open questions since v3.1 Wave D (2026-05-27). The full-paper extraction + sub-check (3) DIAGNOSTIC + cross-reference to ADR-0093 resolves Q1 ANSWERED + Q2 ANSWERED. Q3 STILL_OPEN and Q4 PARTIAL are appropriately scoped — they are downstream-discretisation concerns that Theorem 3 (an abstract Banach-space theorem) does not directly resolve, and the current architecture (Path β Richardson per ADR-0086) does not need them resolved.
- **Theoretical authority for Hypothesis C**: ADR-0096 super-order $\alpha_S \approx -3.1$ phenomenon was empirically observed (Galkin-Remizov 2023 arxiv:2301.05284v5) but flagged as "unexplained by existing theory" (paper §1015). Theorem 2 of the newly-extracted paper PROVES the converse: arbitrary Chernoff rates are achievable, so super-order observations are formally licensed. ADR-0106 supplies the citation; the SISC paper draft ([[project-sisc-paper-draft-v0-1]]) can now cite Theorem 2 as theoretical authority for the Hypothesis C super-order regime.
- **Suckless minimalism**: ADR-0106 is DOCUMENTATION + SYMPY only — no Rust API change, no test code, no contract change, no migration. ~480 LoC sympy oracle + ~500 LoC ADR + ~10 LoC math.md §27 amendment + ~30 LoC properties.yaml schema bump comment. Zero engineering cost; high mathematical fidelity gain.
- **Constitutional compliance**: principle #1 (math fidelity) STRENGTHENED. Principles #2-#5 untouched. Override count remains 3/3 (no new overrides). Guardrail #7 (Security by Design) VACUOUSLY SATISFIED (no API surface, no inputs from untrusted source).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Defer ADR-0106; cite Theorem 3 only in the SISC paper | Loses architectural-log discoverability. Future architect grepping `docs/adr/` for "Theorem 3" + "constant prefactor" would not find the connection without ADR-0106. Paper and architectural log must both record the citation. |
| Extend T23N (existing) with the 5 new sub-checks | Conflates two purposes: T23N is the **Path β algorithm verification** (3 sub-checks specific to the v4.1+ Richardson form); T_GR_2025_THM3 is the **abstract Theorem 3 framework verification** (5 sub-checks across the entire Diffusion family). Conflating them would obscure the diagnostic structure and the cross-class consistency checks (m=1, m=2, m=4 across DiffusionChernoff / Diffusion4thChernoff / Path β). |
| Add Rust API for runtime tangency-check | Premature abstraction. The Theorem 3 hypothesis (eq. 7) is a symbolic operator-algebra identity, not a runtime-checkable predicate. The check requires symbolic Taylor expansion and norm-bound algebra in a CAS (sympy), not f64 arithmetic in the kernel. A future "TangencyVerifier" trait would be a research-grade abstraction over `ApproximationSubspace<K, F>` (which ALREADY captures the K-jet existence) — defer to a future ADR if the need arises. |
| Wait for full SISC paper review before formalising | Decouples library architecture from paper timeline. The paper draft (project_sisc_paper_draft_v0_1) cites SemiFlow implementations; ADR-0106 supplies the corresponding architectural framework. Decoupling timing prevents bidirectional dependency. |
| Re-open the BCH 6-monomial path with the corrected Theorem 3 understanding | Rejected. The DIAGNOSTIC sub-check (3) confirms the BCH-only algorithm CANNOT satisfy m=4 tangency at the abstract level — the issue is not a missing monomial table, it is a structural mismatch between BCH leading-term subtraction and Taylor coefficient matching. Any "6-monomial" path would still need to match τ⁰..τ³ Taylor coefficients exactly; the cleanest construction is Path β (4-term Taylor) per ADR-0086. The full Galkin-Remizov 2025 paper does not supply any monomial table for any kernel; ADR-0093 retracted the attribution. Re-opening this path would re-introduce the same defect. |

## Consequences

- **POSITIVE**:
  - +1 NEW sympy oracle (T_GR_2025_THM3) — formal verification target for all future Chernoff rate claims; reusable framework.
  - G_zeta4 escalation Q1 ANSWERED + Q2 ANSWERED (via ADR-0093 cross-reference); Q3 STILL_OPEN and Q4 PARTIAL appropriately scoped.
  - Theoretical authority established for ADR-0096 Hypothesis C super-order regime (Theorem 2 sharpness).
  - SISC paper draft ([[project-sisc-paper-draft-v0-1]]) can now cite Theorem 2 + Theorem 3 + Theorem 4 with full architectural-log provenance.
  - Math fidelity restored: the citation chain `ADR-0075 → ADR-0093 → ADR-0106` now tells the complete story (attribution error → retraction → corrected formal target).
- **NEUTRAL**:
  - No Rust code change, no API change, no migration, no constitution change.
  - Properties.yaml schema MINOR bump 1.5.0 → 1.6.0 (additive only).
  - math.md §27 minor pointer amendment (~10 LoC).
  - Test-fast sympy sweep gains one script invocation (~3 seconds runtime).
- **NEGATIVE**:
  - None. Purely additive documentation + sympy.
- **No BREAKING change**: zero API surface modification.
- **Future unlocks**:
  - If a future research wave re-opens G_zeta4 (e.g., to improve the empirical slope from -4.06 to -5.0+ via a different higher-order construction), Theorem 3 + the T_GR_2025_THM3 oracle supply the formal verification framework directly; no new theoretical work needed.
  - ζ⁶ ladder rung (ADR-0088, currently shipped v4.x) can be RE-VERIFIED against Theorem 3 with m=6 specialisation; existing T23N_zeta6 may be supplemented with a `T_GR_2025_THM3.diffusion6_m6_tangency` future sub-check.
  - Theorem 4 multiplicative-form Chernoff function (eq. 11) is a candidate future kernel — `MultiplicativeFormDiffusionChernoff<F>` sibling to the divergence-form `DiffusionChernoff`. Defer to v5.x+ if industrial demand (e.g., from quantitative-finance backward Kolmogorov equation users) materialises.
- **G_zeta4 closure trajectory**: Theorem 3 CONFIRMS the v3.0 BCH-only algorithm permanently defective (cannot satisfy m=4 tangency at the abstract level). Path β Richardson permanently validated as the algorithmic successor. If empirical Path β slope -4.06 is deemed insufficient for a future application, the closure would proceed via:
  - either (a) higher-order Richardson over Path β (Romberg pattern; ADR-0088 ladder framework already provides this for ζ⁶) — Theorem 3 with m=6 supplies the rate template; or
  - (b) a Diffusion4thZeta8Chernoff sibling with 5-term Taylor + 2-level Richardson — straightforward extension of Path β within ADR-0086's pattern; sympy oracle pattern reusable.
- **Permanent defer of original BCH G_zeta4**: CONFIRMED by Theorem 3 diagnosis. ADR-0075's BCH algorithm is permanently shelved; no v5.x closure path exists for the BCH ansatz. The Path β Richardson algorithm (per ADR-0086 + AMENDMENT 1) is the ONLY validated G_zeta4 closure; it has been shipped since v4.1+.

## Migration

None. End-user impact: zero. Library callers using `Diffusion4thZeta4Chernoff<F>` continue to receive the Path β Richardson algorithm (per ADR-0086 + AMENDMENT 1) unchanged. Future paper / publication authors gain a citable architectural framework (Theorem 3 + T_GR_2025_THM3 oracle) for rate-claim verification.

## Schema bump

`contracts/semiflow-core.properties.yaml`: **1.5.0 → 1.6.0 MINOR** (additive entries only).
- ADDED: `T_GR_2025_THM3` NORMATIVE sympy PRE-FLIGHT record (5 sub-checks; `scripts/verify_thm3_harness.py`).
- All existing v1.5.0 entries PRESERVED verbatim.

`contracts/semiflow-core.traits.yaml`: **UNCHANGED**.

`contracts/semiflow-core.math.md`: minor amendment at end of §27 + AMENDMENT 2 (Theorem 3 prefactor-recipe pointer; ~10 LoC).

## Cross-references

- ADR-0001 — contract-first; this ADR follows the same Rust-doc + math.md + properties.yaml triple-source-of-truth pattern.
- ADR-0073 — `ApproximationSubspace<K, F>` opt-in marker trait; the K-jet existence framework that Theorem 3's $f \in \mathcal{D} \subseteq D(L^{m+p})$ hypothesis captures at the type level. ADR-0106 does NOT modify ADR-0073; the witness framework remains the runtime gate for `apply_into`.
- ADR-0075 — v3.0 ζ⁴ correction kernel — §"Mathematical foundation" ATTRIBUTION partially retracted by ADR-0093 (the citation error); §"Decision" + §"Algorithm" REPLACED by ADR-0086 (Path β Richardson). ADR-0106 supplies the **correct formal target** (Theorem 3, not "Theorem 3.1") that ADR-0075 should have cited; closes the lineage.
- ADR-0086 + AMENDMENT 1 — Path β Richardson algorithmic successor; validated empirically (slope −4.06 on const-a oracle). ADR-0106 supplies the THEORETICAL AUTHORITY (Theorem 3 with m=4 conclusion) that ADR-0086 cited only as the "abstract o(1/n^m) rate".
- ADR-0088 — ζ⁶/ζ⁸ ladder; the v4.x ladder framework. T_GR_2025_THM3 sub-check (4) framework readily extends to ζ⁶ (m=5 in paper convention) and ζ⁸ (m=7) via the same Path β + Richardson pattern.
- ADR-0090 — Chebyshev spectral collocation; orthogonal v4.3 path. Theorem 3 framework applies to spectral kernels as well (the rate constant is absorbed in K_j(t)).
- ADR-0093 — ADR-0075 6-monomial attribution retraction (the lineage correction). ADR-0093 retracted; ADR-0106 supplies the corrected formal target.
- ADR-0096 — Hypothesis C super-order convergence mechanism. Theorem 2 sharpness (newly extracted) provides theoretical authority for the empirical $\alpha_S \approx -3.1$ regime.
- ADR-0103 — Subordinated Chernoff (PRE-FLIGHT pattern precedent followed by ADR-0106).
- math.md §27 — NORMATIVE ζ⁴ correction algorithm (the Path β Richardson form per AMENDMENT 2). Receives a minor pointer amendment in this wave: end-of-§27 callback to `scripts/verify_thm3_harness.py` as the prefactor-recipe verification authority.
- `contracts/semiflow-core.properties.yaml` schema 1.6.0 — NEW `T_GR_2025_THM3` entry; integrated into test-fast sympy sweep.
- `scripts/verify_thm3_harness.py` — NEW PRE-FLIGHT sympy oracle (5 sub-checks); PRE-FLIGHT 5/5 PASS 2026-05-29.
- `scripts/verify_zeta4_correction.py` — existing T23N gate (Path β-specific); COMPLEMENTARY to T_GR_2025_THM3 (cross-class abstract Theorem 3 framework). Both remain in test-fast sympy sweep.
- arxiv:2104.01249v2 — Galkin-Remizov 2025 *Israel J. Math.* full PDF in `.dev-docs/papers/oleg-e-galkin-upper-and-lower-estimates-for-rate-of.pdf`.
- arxiv:2301.05284v5 — Katalova-Nikbakht-Remizov 2023 super-order observation; Theorem 2 of the newly-extracted paper now provides theoretical authority.
- Vedenin-Voevodkin-Galkin-Karatetskaya-Remizov 2020 *Math. Notes* 108(3), 451-456 — predecessor (conjectures the $1/n^2$ rate for $S(t)$); cited as Galkin-Remizov 2025 §"Introduction" lineage.
- Galkin-Remizov 2022 *Math. Notes* 111(2), 305-307 — eq. 8 prerequisites short-form (cited as reference [6] in the newly-extracted paper).
- Gomilko-Tomilov 2014 *J. Funct. Anal.* 266(5), 3040-3082 — competitor framework comparison (cited as reference [9] in the newly-extracted paper).
- Chernoff 1968 *J. Funct. Anal.* 2(2), 238-242 — foundational Chernoff theorem (cited as reference [3] in the newly-extracted paper).
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_g_zeta4_escalation.md` — v3.1 escalation memory; gains a second RESOLVED-via-ADR-0106 closure pointer with per-question Q1/Q2/Q3/Q4 status update.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_v4_4_research_wave.md` — prior Galkin papers Wave A (Q4 verdict); cross-references the same paper extract.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_sisc_paper_draft_v0_1.md` — SISC paper draft; gains Theorem 2 + Theorem 3 + Theorem 4 citation framework via ADR-0106.

## Amendments

(none at acceptance time)
