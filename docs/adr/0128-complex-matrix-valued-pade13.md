# ADR-0128 — Complex matrix-valued operators via Padé[13/13] over SemiflowComplex

**Status:** PROPOSED (PRE-FLIGHT GREEN) · **Date:** 2026-06-06 · **Branch:** `feat/v7.0.0-debt-closure`
**Item:** v7.0.0 backlog #17 (§33.5 deferred) · **Gate:** `G_CPLX_MATRIX`
**Supersedes numbering:** backlog freeze assigned this item ADR-0126, but 0126 was consumed by higher-order-soft-killing; reallocated to **0128** (next free on disk).

## Context

§33.5 deferred `MatrixDiffusionChernoff<C, M>` with `C: SemiflowComplex` (complex M×M coupling blocks: non-Hermitian absorbing potentials, complex cross-diffusion, rough-Heston Markov blocks), citing "Higham 2008 §10.4 different rational approximants" as the cost. The shipped real `matrix_pade.rs` (ADR-0125, Padé[13/13], M≥5) is generic over `F: SemiflowFloat` only.

## Decision

Ship `MatrixDiffusionChernoff` over `C: SemiflowComplex` as an **additive** sibling. The §33.5 premise ("different rational approximants") is FALSE: Higham 2005 Padé[13/13] uses **real** coefficients `PADE_B` applied to a matrix argument — the argument may be complex with no change to the approximant. The engineer change is mechanical: `[[F;M];M]` → `[[C;M];M]`, real arithmetic → `num-complex` arithmetic. `compute_squarings` (inf-norm `s = ⌈log₂(‖Z‖∞/θ₁₃)⌉`) lifts unchanged (complex-modulus row sums). Order-2 Strang composition (§33.7 AMENDMENT 2 block-Cayley) is inherited verbatim. ZERO new deps (`num-complex` is dep 3/3, already direct). PRE-FLIGHT `scripts/verify_complex_matrix.py` 3/3 PASS: worst rel-Frobenius err **9.3e-23** (mpmath/50-dps) on M={5,6,8} complex cases; anti-Hermitian `exp(iH)` unitary to 2e-52; complex squaring formula matches real. Gate `G_CPLX_MATRIX`: rel-Frobenius err ≤ 1e-12 (f64 Rust hits ~1e-13, matching the real M≥5 path). No slope gate (exact-to-1e-12 exponential + inherited order-2).

## Consequences

Additive only; no breakage. `num-complex` budget unchanged (3/3). Trait surface: new `SemiflowComplex`-parameterised `MatrixDiffusionChernoff` impl. Holomorphic/non-symmetric-coupling cases remain as §33.5 notes. Dispatch keeps Cayley-Hamilton M≤4 (byte-identity), routes M≥5 to Padé[13/13] — identical structure for real and complex.
