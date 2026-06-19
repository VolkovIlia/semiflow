# ADR-0093 — G_zeta4 Lineage Correction: ADR-0075 6-Monomial Attribution Retracted

- **Status**: Accepted (corrects ADR-0075; supplements ADR-0086 AMENDMENT 1)
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Wave**: v4.4+ research wave Phase A (documentation-only; ~120 LoC ADR; no code, no contract, no schema changes)
- **Depends on**: ADR-0075 (the corrected target), ADR-0086 + AMENDMENT 1 (the validated successor Path β Richardson), ADR-0073 (`ApproximationSubspace<K, F>` framework — preserved verbatim).
- **Supersedes / amends**: ADR-0075 §"Mathematical foundation" attribution to *Galkin-Remizov 2025 Israel J. Math. Theorem 3.1 / Example 4.2 6-monomial table* — RETRACTED; ADR-0086 already supersedes ADR-0075 §"Decision" + §"Algorithm" — this ADR closes the lineage-attribution loop.
- **Mathematical foundation**: arXiv:2104.01249v2 Galkin-Remizov 2025 *Israel J. Math.* (full PDF read 2026-05-29, extract `.dev-docs/research/extracts/galkin-remizov-2025-extract.md`); arXiv:2301.05284v5 Katalova-Nikbakht-Remizov 2023 (fresh download 2026-05-29, extract `.dev-docs/research/extracts/galkin-remizov-2023-extract-v2.md`); Vedenin-Voevodkin-Galkin-Karatetskaya-Remizov 2020 *Math. Notes* 108(3) (extract `.dev-docs/research/extracts/vedenin-speed-extract.md`); Remizov 2018 *Appl. Math. Comput.* 328 (lineage citation).
- **Acceptance gates added**: none (documentation-only).

## Context

ADR-0075 (v3.0; 2026-05-27) attributed a "6-monomial polynomial $P_2[A]$ with leading $-1/12$ entry" Chernoff ζ⁴-correction construction to *Galkin-Remizov 2025 Israel Journal of Mathematics* Theorem 3.1 / Example 4.2, declaring the polynomial "uniquely determined" by the cancellation requirement and citing the paper as the closed-form source for the v0.6.0 9-point-stencil specialisation.

The v3.1 Wave D engineer escalation (project_g_zeta4_escalation.md, 2026-05-27) flagged the attribution as numerically inconsistent: BCH-correction-only gave order-2 global, not order-4, and 5 of the 6 monomial coefficients sat at placeholder zero. Phase 5a (v4.3.0, 2026-05-28) flagged the attribution as likely false via *partial* PDF extract. The v4.4+ research wave (2026-05-29) — user provided the full arXiv 2104.01249v2 PDF in `.dev-docs/papers/` — completed the verification by full read.

The full-paper read **CONFIRMS the attribution is FALSE**. Example 4.2 (paper p. 29, eq. (33)) reads verbatim:

> $(S(t)f)(x) = (1/4) f(x + 2\sqrt{a(x) t}) + (1/4) f(x - 2\sqrt{a(x) t}) + (1/2) f(x + 2 b(x) t) + t \, c(x) \, f(x)$

This is a 4-shift order-1 tangent kernel for the variable-coefficient operator $Au = a u'' + b u' + c u$. The residual bound (eq. (34)) is a clean 2-term Lagrange remainder: $\|S(t) f - (f + tAf)\| \le t^2 \bigl( (2/3) \|a\|^2 \|f^{IV}\| + \|b\|^2 \|f''\| \bigr)$ — **NOT a 6-monomial operator polynomial**. Theorem 3.1 (paper pp. 18-19) is an *abstract rate-of-convergence* theorem conditional on a user-supplied $m$-tangency hypothesis: it does not *construct* any specific high-order Chernoff function for variable-coefficient diffusion; it only states that *if* the user supplies an order-$m$-tangent $S(t)$, *then* the convergence is $O(n^{-m})$. **No 6-monomial polynomial appears anywhere in the paper.**

Cross-checks: Vedenin 2020 *Math. Notes* 108(3) publishes only the order-1 $G(t)$ and order-2 $S(t)$ heat-equation Chernoff functions; conjectures the $1/n^2$ rate for $S(t)$ explicitly, noting "the proof will soon be published by Galkin and Remizov" — confirming that Galkin-Remizov 2025 *IJM* is the conjecture-closing paper for $k=2$, not for $k \ge 3$. Galkin-Remizov 2023 (arXiv:2301.05284v5; fresh download confirmed 2026-05-29) is an empirical follow-up that reports a surprising super-order observation $\alpha_S \approx -3.1$ for $u_0(x) = e^{-x^4}$ explicitly labelled "unexplained by existing theory" (Conclusion §1015). The v3.0 ADR-0075 6-monomial attribution sits in a literature gap that no published paper fills.

## Decision

1. **Retract** the ADR-0075 §"Mathematical foundation" attribution of the "6-monomial polynomial $P_2[A]$ with leading $-1/12$ entry" to *Galkin-Remizov 2025 IJM Theorem 3.1 / Example 4.2*. Record this as a v3.0 engineering hypothesis without literature precedent.
2. **Document the actual lineage** of the v3.0+v4.x ζ⁴ effort:
   - **Remizov 2018** *Appl. Math. Comput.* 328 — direct lineage of `DiffusionChernoff::new` (the order-1 3-point heat Chernoff).
   - **Vedenin et al. 2020** *Math. Notes* 108(3) — canonical Remizov ApproximationSubspace framework (Definition 2 + Proposition 1) + the $k=1$ $G(t)$ and $k=2$ $S(t)$ heat-equation Chernoff functions + the conjecture $A_w^\tau$ for $S(t)$ at order $1/n^2$.
   - **Galkin-Remizov 2025** *Israel J. Math.* (arXiv:2104.01249v2) — Theorem 3.1 abstract rate theorem $\|S(t/n)^n f - e^{tL} f\| \le (C \, t^{m+1} / n^m) \cdot \sum K_j \|L^j f\|$ conditional on user-supplied $m$-tangency. Specialised to $m = 4$, this is the abstract guarantor of order-4 Chernoff convergence — without supplying the construction recipe.
   - **Katalova-Nikbakht-Remizov 2023** (arXiv:2301.05284v5) — empirical study of $G$ and $S$ Chernoff functions on the heat semigroup; reports super-order observation $\alpha_S \approx -3.1$ for $u_0(x) = e^{-x^4}$ explicitly labelled unexplained.
3. **Successor reference**: the v3.0 6-monomial ansatz was deleted at v4.1 in favor of the **single-step 4-term Taylor expansion** (Path β) per ADR-0086 + AMENDMENT 1 (Richardson form on the K5 base, RELEASE_BLOCKING const-a sub-gate + RELEASE_ADVISORY var-a sub-gate). Path β is empirically validated (slope $-4.06$ on constant-$a$ analytic oracle) and correctly cites Theorem 3.1 specialised to $m=4$.
4. **ζ⁴/ζ⁶/ζ⁸ ladder genuine novelty CONFIRMED**: four independent extracts (Vedenin 2020, Galkin-Remizov 2025, Galkin-Remizov 2023, Shamarova 2007) + knowledge-fetcher 2010-2026 literature sweep (8 web searches; `.dev-docs/reports/RAW_FINDINGS_HIGHAM_SAAD_2026.md`) show **no published Chernoff order-6 or order-8 kernels exist** outside the SemiFlow v3.0+v4.x ζ-ladder. The v4.3 Chebyshev spectral collocation path (ADR-0090) and the v4.x Richardson nested ladder (ADR-0088) are genuinely novel engineering contributions, no SWW prior art (Shamarova-Smolyanov-Weizsäcker-Wittich caps at order-1 tangency).
5. **NO source code changes** — the v3.0 6-monomial implementation was already DELETED at v4.1 commit (see ADR-0086 + AMENDMENT 1 for the Path β replacement). The v3.0 work is preserved as project history (memory file project_g_zeta4_escalation.md updated with closure pointer).

## Rationale

- **Constitution principle #1 (math fidelity)**: ADR-0086 already corrected the *algorithm*; this ADR closes the loop by correcting the *citation*. Carrying a phantom attribution in ADR-0075 invites the same misattribution to leak into downstream documentation (ROADMAP, papers, migration guides) and undermines the documented lineage trust.
- **Closes tracking debt**: project_g_zeta4_escalation.md has carried the open question "what does Galkin-Remizov 2025 IJM Theorem 3.1 actually require?" since v3.1 Wave D. The full-PDF read decisively answers it (rate theorem conditional on user-supplied tangency, not a construction recipe). Closing the memory entry with explicit pointer to this ADR + ADR-0086 + AMENDMENT 1 prevents the question being re-asked at future research waves.
- **Documents genuine novelty**: the ζ⁴/ζ⁶/ζ⁸ ladder beyond $k=2$ is SemiFlow's original research contribution (4 independent extracts + knowledge-fetcher sweep confirm). Recording this in an architectural decision record establishes provenance for the v3.0+v4.x ζ-ladder ahead of paper publication (SISC primary + SINUM companion track per project_sisc_paper_draft_v0_1).
- **Documentation-only — zero risk**: no API, contract, schema, or code change. ADR-0086 already shipped the validated Path β algorithm; ADR-0093 retroactively corrects the attribution in the architectural log.
- **Suckless honesty**: per AGENT.md anti-pattern table, "carrying phantom citations" is the documentation equivalent of "claimed success without tool evidence". Retracting the attribution is the suckless choice; preserving it would propagate a false provenance trail.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Edit ADR-0075 in-place to fix the attribution | ADRs are immutable architectural log per ADR-0001; retroactive edits break the historical decision trail. Issuing ADR-0093 as a corrective record preserves the history while making the correction discoverable. |
| Amend ADR-0086 with a second AMENDMENT for the citation correction | ADR-0086 is scoped to *algorithm* correction (Path β); the *attribution* correction is a distinct architectural concern. Cleaner to separate the algorithmic record (ADR-0086) from the attribution record (ADR-0093). |
| Skip the ADR; just close the memory entry | Loses architectural-log discoverability — future research waves grepping `docs/adr/` for "Galkin-Remizov 2025" would still find ADR-0075's incorrect attribution without the corrective pointer. |
| Wait for SISC paper submission to formalise the correction | Decouples library architecture from paper timeline; ADRs must be self-contained. The paper draft (project_sisc_paper_draft_v0_1) can cite ADR-0093 as the provenance record. |

## Consequences

- **POSITIVE**: closes v3.1 Wave D tracking debt (project_g_zeta4_escalation.md); restores citation fidelity for math §27 + ADR-0075; documents v3.0+v4.x ζ-ladder genuine novelty; establishes provenance trail for upcoming SISC paper.
- **NEUTRAL**: no code, contract, schema, or test changes. No migration. No build/run impact. Constitution unchanged (principle #1 re-affirmed, not amended).
- **NEGATIVE**: none — purely additive documentation.
- **No BREAKING change**: ADR-0086 already handled the algorithm replacement; ADR-0093 closes the citation loop.

## Migration

None. End-user impact: zero. Library callers using `Diffusion4thZeta4Chernoff<F>` continue to receive the Path β Richardson algorithm per ADR-0086 + AMENDMENT 1.

## Cross-references

- ADR-0001 — contract-first; ADRs are the immutable architectural log; corrective records preserve history.
- ADR-0075 — v3.0 ζ⁴ correction kernel — §"Mathematical foundation" attribution PARTIALLY RETRACTED by this ADR (kernel preserved per ADR-0086; only the citation is corrected).
- ADR-0086 — v4.1 Path β Richardson algorithmic resolution + AMENDMENT 1 (gate methodology re-design) — the validated algorithmic successor.
- ADR-0088 — v4.x ζ⁶/ζ⁸ Richardson ladder — also descends from the abstract Galkin-Remizov 2025 Theorem 3.1 rate theorem ($m = 6, 8$ specialisations).
- ADR-0090 — v4.3 Chebyshev spectral collocation — the orthogonal v4.3 path that also achieves order-4+ via spatial spectral exactness.
- math.md §27 LINEAGE NOTE (this wave) — appended to §27 documenting the corrected attribution.
- arXiv:2104.01249v2 — Galkin-Remizov 2025 *Israel J. Math.* full PDF; extract `.dev-docs/research/extracts/galkin-remizov-2025-extract.md` (verbatim Theorem 3.1 + Example 4.2 + Theorem 4.1/4.2).
- arXiv:2301.05284v5 — Katalova-Nikbakht-Remizov 2023; extract `.dev-docs/research/extracts/galkin-remizov-2023-extract-v2.md` (super-order observation; open problem §322-346).
- Vedenin et al. 2020 *Math. Notes* 108(3); extract `.dev-docs/research/extracts/vedenin-speed-extract.md` (canonical ApproximationSubspace framework).
- Remizov 2018 *Appl. Math. Comput.* 328 — first-order heat Chernoff (`DiffusionChernoff::new` direct lineage).
- `.dev-docs/research/verdicts/verdict-v4-4-research-wave.md` — Q4 verdict synthesising the 4 extracts + the full-PDF confirmation.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_g_zeta4_escalation.md` — v3.1 escalation memory; updated with RESOLVED status pointing to this ADR.
- `.dev-docs/reports/RAW_FINDINGS_HIGHAM_SAAD_2026.md` — knowledge-fetcher 2010-2026 sweep (8 web searches) confirming no competing Chernoff order-6/8 publications.

## Amendments

(none at acceptance time)
