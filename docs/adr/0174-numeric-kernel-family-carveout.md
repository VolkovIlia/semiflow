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
