# ADR-0125 — Matrix-exponential for M≥5 coupling blocks via hand-rolled Padé[13/13] (Higham 2005); `expmv` does NOT generalise, scaling-and-squaring Taylor is insufficient

- **Status**: Accepted (PRE-FLIGHT GO; engineer wave authorised — additive `MatrixExpPade<M>` path lifting the `if M>=5 { Err }` guard)
- **Date**: 2026-06-05
- **Decision-maker**: ai-solutions-architect
- **Backlog item**: v7.0.0 freeze item #25 (predicted ADR-0132; actual sequential allocation = **0125**).
- **Depends on**: ADR-0082 (`MatrixDiffusionChernoff<F, M>`; this lifts its M≤4 cap), ADR-0094 (in-tree scaling-and-squaring scalar machinery — the squaring loop reused), `matrix_inv.rs` (in-tree LU solve — the Padé `V−U` solve reuses it), constitution v2.0.0 (3/3 dep cap INVIOLATE — Padé hand-rolled, ZERO new deps).
- **Mathematical foundation**: `scripts/verify_matrix_pade_m5_preflight.py` (executed 2026-06-05). PASS — see §PRE-FLIGHT.
- **Acceptance gates added**: `G_MATRIX_PADE_M5` (slow-tests; **relative** Frobenius error `≤ 1e-12` vs reference, in the regime `‖τ·C(x)/2‖_∞ ≤ 10`).

## Context

`MatrixDiffusionChernoff<F, M>` needs `exp(τ·C(x))` for the dense M×M per-grid-point reaction coupling block. v4.0 ships closed-form M≤4 and an explicit `if M >= 5 { Err(Unsupported) }`. Two questions had to be answered before authorising an engineer wave: **(Q1)** does the just-shipped `expmv` (ADR-0121) generalise to the M≥5 case (the cleanest no-dep path)? **(Q2)** does the existing generic `mat_exp_taylor` (already parameterised over `dim`) simply work at M≥5?

## Decision

**Q1 — NO, `expmv` does NOT generalise.** `DiffusionExpmvChernoff` computes the *action* `e^{τA}·v` of the N×N **spatial** divergence-form operator on a **scalar** field (M=1 implicitly). Feature 2 needs the M×M **coupling-block** exponential as a *matrix* (cached `exp_half_c[k]`, then matrix-vector-multiplied per node). Different object on both axes (action-on-N-vector vs M×M matrix; spatial operator vs component coupling). REUSE of `expmv` is impossible.

**Q2 — NO, `mat_exp_taylor` is insufficient.** PRE-FLIGHT measured its worst relative error at 2.13e-12 (regime `‖A‖≤10`) and ~1e-9 at `‖A‖=20`; raising degree/scaling makes it *worse* (the squaring round-off "hump", Moler-Van Loan 2003). It floors above 1e-12.

**Therefore: hand-roll Padé[13/13] (Higham 2005)** — the algorithm `scipy.linalg.expm` uses: `θ₁₃ = 5.3719`, `U/V` even/odd split, single in-tree LU solve (`matrix_inv.rs`), `s = ⌈log₂(‖A‖/θ₁₃)⌉` squarings. It reaches relative error **1.80e-13** for symmetric reaction matrices (reaction matrices ARE symmetric, math §33.1) in the physical half-step regime. Ship as additive `MatrixExpPade<M>`; dispatch keeps Cayley-Hamilton for M≤4, Padé[13/13] for M≥5. ZERO new deps.

## PRE-FLIGHT numpy (executed 2026-06-05; `scripts/verify_matrix_pade_m5_preflight.py`)

120 symmetric M×M cases, M∈{5,6,8}, `‖A‖_∞∈{0.1,1,5,10}`, relative Frobenius error vs `scipy.linalg.expm`:

| Candidate | worst rel-err | verdict |
|-----------|---------------|---------|
| A: existing `mat_exp_taylor` (deg-12 + ⌈log₂‖A‖⌉ scaling) | 2.13e-12 | INSUFFICIENT |
| B: **Padé[13/13]** (Higham 2005) | **1.80e-13** | **PASS** |

VERDICT: **GO** with Padé[13/13]; gate measures **relative** error in `‖τC/2‖≤10`.

## Consequences

- **Honest regime bound**: at `‖τC/2‖ ≈ 20` even Padé[13/13] dips to ~1.9e-12 (squaring hump). The gate and rustdoc MUST state the `≤10` half-step regime; stiffer reaction needs more Chernoff sub-steps (smaller τ), which is the correct user response, not a tighter exponential.
- **Dispatch**: `matrix_exp_dispatch` gains an `_ => pade13` arm; the `if M>=5 { Err }` guards in `apply_into` are removed. M≤4 closed-form paths UNCHANGED (byte-identity preserved).
- **No 4th dep** (freeze §2 row satisfied): Padé coefficients are a compile-time `const` table; the `V−U` solve is the existing in-tree LU.
