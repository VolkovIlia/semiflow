# ADR-0133 — F1 Dual-number forward-mode AD over the Chernoff product structure

**Status:** ACCEPTED · **Date:** 2026-06-06 · **Shipped:** 2026-06-08 · **Branch:** `feat/v8.0.0-planning`
**Theme:** v8.0.0 — Differentiable Chernoff (F1, HEADLINE, low-risk)
**Gates:** `G_DUAL_AD_GRADIENT`, `G_DUAL_ZERO_ALLOC`
**Parent:** ADR-0132

## Context

Parameter and state sensitivities (Greeks Δ/Γ, ML-training gradients through PDE kernels, calibration Jacobians) are economically important. Two incumbent approaches both fail the library's constraints: adjoint backward-mode tape-AD allocates heap memory proportional to trajectory length, shattering the 45 ns / L1d-resident hot-loop invariant; central finite-differences require ≥2k evaluations for k parameters and are inexact at finite step. The contradiction is declared: "exact gradient AND zero-alloc / cache-resident" is impossible with these tools. The Chernoff product `(F(τ))ⁿ` is the ВПР: by the product rule, `d/dθ (F(τ))ⁿ` equals a sum of n products each with exactly one factor replaced by `(dF/dθ)(τ)` — the derivative rides in the **same SIMD register** as the value when the scalar field `F` is replaced by the `Dual<F>` field `F + ε·F'`.

## Decision

Introduce `Dual<F: SemiflowFloat>: SemiflowFloat` (value + tangent pair) and implement `ChernoffFunction<Dual<F>>` blanket coverage so that every existing kernel gains forward-mode AD at zero new allocation. The TRIZ resolution is: TRIZ-6 (universality — one field extension covers all 25+ kernels) + TRIZ-5 (merging in time — value and derivative advance in the same loop pass) + dual-number arithmetic. `SemiflowFloat` is already generic-over-Float (v0.9.0 Block D), so `Dual<F>: SemiflowFloat` integrates at no abstraction tax; the ping-pong buffer doubles register width but remains allocated-once. Hyper-dual `Dual<Dual<F>>` extends to Γ (second derivatives) via the same blanket. SIMD specialisations stay `f64`-only per ADR-0018; the `Dual` path falls through to scalar Rust ops, which is acceptable for calibration workloads (not 45 ns hot-loop). Gate `G_DUAL_AD_GRADIENT`: forward-mode gradient matches central-differences to tolerance 1e-10 for a representative set of kernels and parameters. Gate `G_DUAL_ZERO_ALLOC`: steady-state allocation count per `evolve` call is unchanged relative to `F=f64` baseline.

## Consequences

Unlocks Δ/Γ Greeks on ALL 25+ library kernels (moat: incumbents allocate a tape, fall out of L1d). Enables ML training through any PDE layer in WASM/no_std environments (JAX/Diffrax require Python+XLA runtime). Zero new external dependencies; dual arithmetic is ~15 lines of trait impl. No existing kernel semantics change; purely additive surface extension. Risk is LOW: `SemiflowFloat` generic infrastructure already proven across all kernel families in v0.9.0+ waves.

## Amendment 1 (2026-06-07) — interpolation-order coverage of the AD seam

The "ALL 25+ kernels" claim is universal at the **kernel** level (the blanket `ChernoffFunction<Dual<F>>` covers every kernel) but is qualified along one axis: the **spatial interpolant** used by `GridFn1D::sample` at off-node positions. The AD path runs through the generic scalar sampler `Grid1D::interp_generic`, whose `InterpKind` dispatch differs from the `f64`-specialised `Grid1D::interp`. QA discovered that pre-v8 `interp_generic` supported only `CubicHermite`/`Linear`, while `Grid1D::new` **defaults** to `SepticHermite` (ADR-0109) — so AD silently returned `SemiflowError::Unsupported` on the *default* grid, and the F1 gates passed only because they pinned `.with_interp(InterpKind::CubicHermite)`.

Resolution (ruling C, hybrid): genericise the **default** `SepticHermite` sampler over `F: SemiflowFloat` so AD composes with the default grid (closed-form weights + rational-constant FD stencils are pure `SemiflowFloat` field ops — mechanical, low-risk; see math.md §46.5.bis). `OctonicHermite` (ADR-0117, non-default precision opt-in) and `ChebyshevSpectralWithBC` (node/weight tables + barycentric reduction) remain `f64`-only on the generic path as a **documented, honest-defer known-limitation to v8.x** — no headline-blocking user exists for them since the default is `SepticHermite`. `G_DUAL_AD_GRADIENT` is amended to REQUIRE at least one kernel on a default `Grid1D::new` (SepticHermite) grid, so the gate can no longer be satisfied by the CubicHermite-pinned dodge. The "all 25+ kernels" claim is true for the default grid and is qualified, not retracted.
