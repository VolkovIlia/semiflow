# ADR-0174 — Bounded ≤900-LoC carve-out for cohesive numeric-kernel families

**Status:** ACCEPTED (2026-06-20) · **Reinstates (reuses retired slot):** constitution Override #1 · **Constitution:** v6.0.0 → v7.0.0 (MAJOR)

**Decision.** The flat 500-LoC/file suckless cap currently forces *artificial*
fragmentation of single cohesive numeric-kernel families into `_helpers`/`_generic`
sibling files whose only reason to exist is the line cap, not a design boundary — e.g. the
diffusion4 family (`diffusion4.rs` 466 + `diffusion4_helpers.rs` 138 +
`diffusion4_generic.rs` 144 = 748 if merged) and the diffusion6 family
(360+190+144 = 694). This scatters the math-co-location (sympy-derived coefficient tables,
proof-cited rustdoc, the generic `RemizovFloat`/`Dual<F>` impl surface) that
`contracts/*.math.md` cites *by file:section*. We reinstate the (retired) constitution
**Override #1** as a **bounded ≤900-LoC carve-out** for self-contained numeric-kernel
families, NOT a blanket raise: the default cap stays ENFORCED 500; the carve-out applies
ONLY to a file that satisfies ALL of — *(a)* it implements ONE cohesive algorithm family
(one ζ-order kernel + its helpers/generic specialisation, or one tensor-Strang dimension),
*(b)* the merge has NO cross-file coupling introduced (the helpers exist solely to dodge the
cap), *(c)* it carries proof-cited rustdoc whose math.md cross-references break under
splitting, *(d)* function cap (≤50 lines) and dep cap (≤3) remain ENFORCED unchanged.
**Honesty note:** every production `.rs` is currently ≤500 LoC, so this is *forward-looking*
authorization for the consolidation refactor — it does NOT retroactively legalize bloat.
Qualifying families (post-merge LoC): diffusion4 (748), diffusion6 (694), diffusion8 if
re-merged, grid_chebyshev septic/octonic clusters. **Explicit non-qualifier:** the
`*_parallel.rs` siblings (`strang3d.rs` 493 + `strang3d_parallel.rs` 494 = 987 > 900) stay
split — their separation is a *deliberate* ADR-0018 bit-identical-mirror boundary, not a
cap dodge, and 987 exceeds the ceiling anyway. Override count stays **3 / 3** (reuses the
retired #1 slot; #2 MCP and #3 spec-doc unchanged). Guardrail #7 (Security-by-Design)
UNTOUCHED and IMMUTABLE.

---

## Execution Amendment — 2026-06-21 (commit 72ead1a)

**Outcome:** The consolidation refactor authorized by this ADR was executed in 2026-06
and found that the ADR's premise of "artificial fragmentation" was largely incorrect
for the two named qualifying families.

| Family | Finding | Action taken |
|---|---|---|
| Diffusion ζ-ladder (diffusion4/6/8) | FD `apply`/stencil routines are order-specific (7-pt no-SIMD vs 9-pt 4+4+1 SIMD, distinct delta margins); sharing them would break bit-equality | Merged only the genuinely-identical validators (`validate_tau` / `validate_a_x`, f64 + generic) into `crates/semiflow-core/src/diffusion_zeta_common.rs` (net −114 LoC) |
| Chebyshev septic/octonic | Stencil widths differ (8/7/6-pt septic vs 10/9/10-pt octonic) and `quad_prime` exists in octonic only; no genuine duplication | Consolidation ABORTED — audit/plan premise was incorrect; no files merged |

**Conclusion.** The ~900-LoC carve-out remains essentially unexercised: no file needed
to exceed 500 LoC after the real (small) consolidation. The per-order numeric files are
legitimately distinct for mathematical reasons, not ceiling-driven fragmentation. The
carve-out stays in the constitution as a sanctioned-but-currently-unexercised allowance
in case a genuinely cohesive family is introduced in future.
