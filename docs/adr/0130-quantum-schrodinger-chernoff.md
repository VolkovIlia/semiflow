# ADR-0130 — QuantumSchrödingerChernoff on quantum graphs (complex Kirchhoff)

**Status:** PROPOSED (PRE-FLIGHT GREEN) · **Date:** 2026-06-06 · **Branch:** `feat/v7.0.0-debt-closure`
**Item:** v7.0.0 backlog #16 (§29.7 + §30.5 deferred) · **Gate:** `G_QSCHROD`
**Numbering:** backlog assigned 0125 (consumed by matrix-pade13-m5); reallocated to **0130**.

## Context

§29.7 + §30.5 deferred complex Schrödinger evolution on a metric graph "to v4.1+ once SemiflowComplex available", flagging that the per-edge probability-current law `∑ₑ Im(ψ̄ ∂ₓψ)=0` is "a NEW conservation law that needs its own vertex condition class". All prerequisites now ship: `SemiflowComplex` (ADR-0079), `SchrödingerChernoffComplex` real-space Cayley map (§30), `QuantumGraphHeatChernoff` Kirchhoff-heat (ADR-0078).

## Decision

Ship `QuantumSchrödingerChernoff<C>` as an additive composition reusing the EXISTING heat Kirchhoff projector `Q_v=(1/d)𝟙𝟙ᵀ` around a complex Cayley kinetic step. PRE-FLIGHT (`scripts/verify_quantum_schrodinger.py`, 3/3 PASS) **falsifies the §30.5 "needs its own vertex condition class" premise**: (1) `Q_v` over ℂ is Hermitian/idempotent/rank-1 — the projector (29.1) lifts to ℂ^{d_v} verbatim; (2) `Q·Cayley·Q` is norm-preserving on the continuity subspace (Cayley of anti-Hermitian `(iτ/2)L` is exactly unitary, `UᴴU=I` symbolic); (3) the Schrödinger current `∑ₑ Im(ψ̄ψ′) = Im(ψ̄·∑ₑψ′) = 0` is **IMPLIED** by continuity (`Q`) + the heat-Kirchhoff derivative balance `∑ₑψ′=0` — no separate vertex class required. Two-half-step structure (Phase-1 per-edge complex Cayley + Phase-2 vertex `Q`) mirrors the shipped heat kernel. Gate `G_QSCHROD`: eigenmode max_err ≤ 5e-4 (mirror G30) or unitarity drift ≤ 5e-4.

## Consequences

Additive; no breakage, no new deps. Spec-trap caught: the deferral's "new conservation law" concern dissolves — the probability current is a consequence of two already-realised conditions, so the heat `QuantumGraph` framework is reused unchanged. Order-1 per-edge inheritance from the heat kernel applies (order-2 per-edge deferred per §29.7); path-graph P₃ eigenmode gating mirrors G30.
