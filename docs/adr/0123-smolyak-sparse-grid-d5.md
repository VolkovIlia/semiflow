# ADR-0123 — Smolyak sparse-grid quadrature for `AnisotropicShiftChernoffND` at D≥5 (additive `SmolyakGridND`, in-tree, NO 4th dep)

- **Status**: Accepted (PRE-FLIGHT GO; engineer wave authorised — additive sparse-grid quadrature backend)
- **Date**: 2026-06-05
- **Decision-maker**: ai-solutions-architect
- **Backlog item**: v7.0.0 freeze item #22 (Phase 5, 🚫 dep-watch). *Freeze pre-allocated ADR-0130; reconciled to actual next-free 0123.*
- **Depends on**: ADR-0081 / math.md §32.3-§32.6 (the d-D shift kernel; §32.6 names Smolyak as the D≥6 future extension), ADR-0112 (corrected eq 32.3), constitution v2.0.0 Override #1 (3/3 dep cap ENFORCED — **a sparse-grid crate is PROHIBITED**; the combination technique is hand-rolled in-tree).
- **Mathematical foundation**: PRE-FLIGHT `scripts/verify_smolyak_sparse.py` (executed 2026-06-05). Exactness, D=5 accuracy < tensor node count, and bakeability all PASS — see §PRE-FLIGHT.
- **Acceptance gate added**: `G_SMOLYAK_D5` (RELEASE_BLOCKING) — D=5 sparse-grid self-convergence slope ≤ −1.95, at node count < full tensor 3125, with NO 4th dependency.

## Context

The tensor-product Gauss-Hermite quadrature in eq 32.3 costs `q^D` nodes per evaluation point: `5^5 = 3125` at the D=5 gated ceiling, exploding for finer q or D≥6. The Smolyak combination technique builds a SPARSE grid from nested low-order 1-D rules,
`A(ℓ,D) = Σ_{ℓ−D+1 ≤ |i| ≤ ℓ} (−1)^{ℓ−|i|} \binom{D−1}{ℓ−|i|} ⊗_k U_{i_k}`,
retaining a total-degree exactness class at far fewer nodes.

## Decision

Add an ADDITIVE `SmolyakGridND<F, const D: usize>` quadrature backend selectable on `AnisotropicShiftChernoffND` for D≥5 (the tensor path stays the default for D≤4). The 1-D rule ladder reuses the in-tree `{1,3,5,7,9}`-pt Gauss-Hermite tables (the same ones ADR-0122 bakes); the combination coefficients `(−1)^{ℓ−|i|}\binom{D−1}{ℓ−|i|}` are compile-time constants. The merged sparse `(node, weight)` set is emitted as a GENERATED const table (mirroring `generate_chebyshev_nodes.py`) — **zero new dependencies**. Default level `ℓ = D+3` (the operative accuracy/cost knee).

## PRE-FLIGHT (executed 2026-06-05; `scripts/verify_smolyak_sparse.py`)

D=5, nested GH ladder, kernel-class deg-10 product integrand at shift s=0.3 (tensor q=5 is the exact f64 reference):

| level ℓ | #nodes | tensor 5⁵ | max total-deg exact | rel err vs tensor |
|---------|--------|-----------|---------------------|-------------------|
| 5 | 1 | 3125 | 1 | 1.05e-1 |
| 6 | 11 | 3125 | 3 | 4.63e-3 |
| 7 | 71 | 3125 | 5 | 1.03e-4 |
| 8 | **341** | 3125 | 7 | **1.15e-6** |
| 9 | 1341 | 3125 | — | 5.16e-9 |

- **ℓ=8 → 341 nodes** matches the tensor 5⁵ to rel **1.15e-6** — a **9.2× node reduction** at the operative accuracy.
- **ℓ=9 → 1341 nodes**, rel 5.16e-9 (essentially exact), still 2.3× cheaper than tensor.
- Exactness scales cleanly (ℓ=7 nails total-degree-5 at 71 nodes — 44× reduction).
- Bakeable as in-tree `const` arrays from itertools-style loop unrolling + existing GH tables + binomial constants → **NO 4th dep** (the constitution Override #1 boundary is respected).

## Consequences / engineer caveats

- Smolyak weights carry SIGNS (the `(−1)^{ℓ−|i|}` combination coefficients) — unlike the all-positive tensor GH weights. The kernel accumulator must not assume positivity; the `F(0)=I` unit witness still holds (Σ merged weights = π^{D/2} by construction) and MUST be asserted as a smoke sub-test, exactly like §32.4.
- The merged node set must be deduplicated at GENERATION time (nested nodes coincide across sub-rules); the runtime sees a flat `(node, weight)` list.
- Gate is `G_SMOLYAK_D5` self-convergence slope ≤ −1.95 (the freeze's threshold — the sparse rule must not degrade the kernel below its tensor accuracy class). The PRE-FLIGHT rel-err ≤ 1.15e-6 at ℓ=8 is comfortably below the order-1 temporal-truncation signal at the gated step counts (the ADR-0112 AMENDMENT 1 coarse-grid protocol applies verbatim).
- D≥6 follows the SAME construction; only D=5 is gated in v7.0.0 (CI budget).

## AMENDMENT 1 — D≥6 in-scope for v8.0.0 (Phase-4 carry-forward item C1)

- **Date**: 2026-06-07. **Decision-maker**: ai-solutions-architect. **Status**: ACCEPTED (contract-first; engineer spec authorised).
- **What changes**: D≥6 (the §32.6 "deferred to v4.1+" / "CI budget" bullet) is now IN-SCOPE for v8.0.0. The deferral was a CI-budget choice, NOT a mathematical limit — `scripts/verify_smolyak_sparse.py` D=6 sweep (re-run 2026-06-07) confirms the combination technique at $\ell = D+3 = 9$ gives **533 nodes** vs tensor $6^6 = 46656$ (**87.5× reduction**) with the unit witness $\sum w = \pi^3$ holding to $10^{-13}$ at every level. The GH ladder depth (max nested level 5) suffices ($\ell-D+1 = 4 \le 5$ at the gated $\ell$).
- **Genericity**: `SmolyakGridND<F, const D>` and `build_smolyak<D>` / `enumerate_multi_index<D>` / `add_tensor_product<D>` are ALREADY fully generic over `const D` (default 5). No new public type; D=6 is the SAME type instantiated.
- **Sole structural blocker**: `SquareMatrix<F, D>` (in `shift_nd.rs`) stores a fixed `[F; 25]` oversized-static ("first D·D slots used"), sized for $D \le 5$. At $D = 6$, $D \cdot D = 36 > 25$ → `set(i,j)` indexes $i + jD$ up to 35, out of bounds. Fix: widen `[F; 25] → [F; 36]` (covers $D \le 6$). Internal-storage change only (no public-API break — array length is an implementation detail). The TRIZ-resolved choice (fixed-`[F;36]` over `generic_const_exprs` `[F; D*D]`) is documented in the C1 engineer spec.
- **New gate**: `G_SMOLYAK_D6` (RELEASE_BLOCKING; slope ≤ **−0.95** — order-1, mirroring the CORRECTED G_SMOLYAK_D5 threshold, NOT the stale −1.95 prose in §"Acceptance gate added" above; nodes < 46656; Σw ≈ π³ smoke). slow-tests; runs on prod HW in Phase-6 (mirrors G5_3D / G3⁶-2D heavy-validation posture — the prod-HW deferral handles CI budget WITHOUT relaxing severity). properties.yaml 4.5.1 → 4.6.0 MINOR; traits.yaml 4.3.1 → 4.4.0 MINOR (D≥6 audit note on the existing entry; no new type).
- **Gate-prose correction (audit)**: the §"Acceptance gate added" line above lists G_SMOLYAK_D5 at "slope ≤ −1.95". The SHIPPED test (`tests/g_smolyak_d5.rs`) gates at −0.95 because `SmolyakGridND::order() = 1` (the −1.95 figure targets order-2 Strang/RK2 kernels). G_SMOLYAK_D6 inherits the honest −0.95.
