# ADR-0069 — Laplace-Chernoff Resolvent (A1)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v2.7 (first math pillar; additive minor)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float with `F = f64` default), ADR-0026 (`ChernoffFunction` trait generic over `F`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0068 (L-gate harness, BoundaryPolicy widening — v2.6 infrastructure that A1 builds on).
- **Supersedes / amends**: none — strictly additive on the public surface.
- **Mathematical foundation**: math.md §22 (NORMATIVE library — `LaplaceChernoffResolvent` semantics; CITATION Remizov 2025 *Vladikavkaz Math. J.* 27(4) Theorem 3 for the underlying convergence result).
- **Acceptance gates added**: G24 (RELEASE_BLOCKING residual + slope), T19N (NORMATIVE sympy Laplace-transform identity), L_RESOLVENT_N64_P99 (advisory in v2.7.0-rc.1 → blocking in v2.7.0).

## Context

For a closed densely-defined operator `A` generating a C₀-semigroup `{S(t)}_{t≥0}` on a Banach space `X` with growth bound `ω`, the **resolvent** at `λ ∈ ℂ` with `Re(λ) > ω` is given by the Laplace transform of the semigroup:

```
R(λ; A) g := (λI − A)⁻¹ g = ∫₀^∞ e^{−λt} S(t) g dt        (Hille-Yosida)
```

This identity is the foundational tool for spectral analysis, Tikhonov regularization, and semianalytic boundary terms in pricing models (e.g., Heston model log-price PDF closure — the v2.7 HFT side-track example). It is **unique to the semigroup setting**: there is no Trotter-Kato analog that gives the resolvent directly from a product formula (Trotter-Kato gives `S(t)` from a sequence of approximants; the resolvent is recovered indirectly via an additional Laplace integral).

Remizov (2025, *Vladikavkaz Math. J.* 27(4) Theorem 3) proves that the Chernoff approximation `S(t) ≈ (F(t/n))^n` (with consistency order `m ≥ 1`) lifts directly into the resolvent via the same Laplace integral:

```
R̃_n(λ) g := ∫₀^∞ e^{−λt} (F(t/n))^n g dt   →   (λI − A)⁻¹ g   as n → ∞     (Theorem 3)
```

with convergence rate `O(1/n)`. The Chernoff resolvent gives a **preconditioner-free** path to `(λI − A)⁻¹` — no sparse-solver, no iterative GMRES/BiCG, no scope-creep into linear-algebra dependencies. It builds entirely on the existing `ChernoffFunction` machinery.

v2.7 ships the contract surface for this construction: a wrapper `LaplaceChernoffResolvent<C, F>` over any `ChernoffFunction<F>`, plus a small `LaplaceQuadrature<F>` strategy enum (Gauss-Laguerre 32-pt for the well-conditioned regime `Re(λ) ≫ ω`; trapezoid-with-tail for the marginal regime `Re(λ) → ω⁺`).

## Decision

Ship three additive public-surface items in v2.7:

- **`LaplaceChernoffResolvent<C: ChernoffFunction<F>, F: SemiflowFloat = f64>`** — wrapper over any inner Chernoff function. Methods: `eval(&self, lambda: F, g: &C::S) -> Result<C::S, SemiflowError>` (resolvent applied to a full state, returns a new state); `eval_at_point(&self, lambda: F, x0: &[F], g: &dyn Fn(&[F]) -> F) -> Result<F, SemiflowError>` (point-evaluation entrypoint — the L-gate benches this method). Internally calls the inner `apply_into` `n` times per quadrature node, accumulating `Σ_k w_k · (F(t_k/n))^n g` into a working state.
- **`LaplaceQuadrature<F: SemiflowFloat = f64>`** — `enum`. Variants: `GaussLaguerre32` (32-point Gauss-Laguerre nodes/weights as `const [F; 32]` arrays — no_std-safe, zero allocation, exact for polynomial · `e^{-s}` integrands up to degree 63); `TrapezoidWithTail { t_max: F }` (trapezoid on `[0, t_max]` + analytical tail bound `‖g‖ · e^{−λ·t_max} / (λ − ω)`, used when Gauss-Laguerre stalls at `λ → ω⁺`). The enum is **public** because the user chooses the strategy per call; internal Gauss-Laguerre nodes/weights are `pub(crate) const`.
- **`L_RESOLVENT_N64_P99`** — new L-gate entry in `properties.yaml` for per-call latency of `eval_at_point` at `n=64`, `λ=1.0`, Gauss-Laguerre 32-pt. **Advisory** in v2.7.0-rc.1 (one RC cycle of calibration on `i7-12700K` + collection of `m2-pro` and `aws-c7g-large` samples); **blocking** in v2.7.0 final.

Substitution `s = λt`, `dt = ds/λ`:
```
R̃_n(λ) g = (1/λ) ∫₀^∞ e^{−s} · (F(s/(λn)))^n g · ds
         ≈ (1/λ) Σ_{k=0}^{31} w_k · (F(s_k/(λn)))^n g          (Gauss-Laguerre 32-pt)
```
This factoring (substituting `s = λt`) gives a `λ`-independent set of quadrature nodes `(s_k, w_k)`, allowing the const-array approach.

File layout: `crates/semiflow-core/src/resolvent.rs` (~350 LoC; functions ≤50 lines each; fits the default 500-LoC cap with headroom — no constitution carve-out needed). Module added to `traits.yaml` `modules:` list. Schema bumps: `properties.yaml` 0.8.0 → 0.9.0 (per-gate `advisory:` field becomes load-bearing — see ADR §"Consequences"); `traits.yaml` 0.6.0 → 0.7.0 (new types).

## Rationale

- **Why Laplace integral via Chernoff (not direct sparse-solver)?** A direct sparse linear solve for `(λI − A_disc)⁻¹ g_disc` requires either (a) factorization via SuiteSparse/MUMPS (heavy dependency, GPL-incompatible) or (b) iterative GMRES/BiCG with problem-dependent preconditioners. Both kill the suckless dep-count budget (the `semiflow-core` 3-dep cap is hit — no room for nalgebra+SuiteSparse adapters). The Laplace path requires **zero new dependencies**: every operation is already in the `ChernoffFunction` trait surface. It is also **preconditioner-free** by design: the convergence is dictated by `(λ − ω)` and `n`, both user-tunable.
- **Why Gauss-Laguerre 32-pt as the default?** The Laplace integrand `e^{-s} · h(s)` is the canonical Gauss-Laguerre weight × smooth function. 32-point quadrature is exact for `h(s)` polynomial of degree ≤ 63; for the analytic `h(s) = (F(s/(λn)))^n g` the error decays super-algebraically in the node count (Trefethen 2008). 32 nodes is a sweet spot between accuracy (sub-1e-10 on test integrands) and per-call cost (32 × `n` `apply_into` calls = 32n total Chernoff steps). The nodes/weights are stored as `const [F; 32]` arrays — no_std-safe, zero allocation, ABI-stable across `F = f64` / `F = f32` (separate const arrays per concrete `F`; the engineer picks `f64` first wave, `f32` later if needed).
- **Why `TrapezoidWithTail` as the secondary strategy?** Near the spectrum (`λ → ω⁺`) the integrand `e^{-(λ-ω)t} · ‖S(t)g‖` decays slowly; Gauss-Laguerre 32-pt may have a long tail error because the integrand support extends past the largest node `s_31 ≈ 100`. The trapezoid-with-tail fallback truncates at `t_max = 50/λ` (chosen so `e^{-λ·t_max} = e^{-50} < 10^{-21}`) and adds an analytical tail bound. This is the recommended path when the user knows `λ` is close to `ω` (stress-test regime in G24).
- **Why two distinct entrypoints `eval` and `eval_at_point`?** `eval` returns a full state `(λI − A)⁻¹ g` — useful for spectral computation, regularization, and bulk operations. `eval_at_point` returns a scalar `F` — useful for HFT semianalytic terms (one boundary value per tick), and is the form the L-gate benches (tight per-call loop). The two share a common inner kernel; `eval_at_point` avoids the final-state allocation by accumulating per-quadrature-node values into a scalar reduction.
- **Why `L_RESOLVENT_N64_P99` advisory in RC, blocking in final?** Latency calibration is host-dependent (ADR-0068 §"Rationale" track 2). The v2.7.0-rc.1 advisory phase lets the project collect one cycle of `i7-12700K` measurements (calibrate the `i7-12700K` profile budget), plus optional advisory samples from `m2-pro` and `aws-c7g-large`. The v2.7.0 final promotion bumps `advisory: false` for the `i7-12700K` profile only; other profiles stay advisory. Symmetric to the v2.6 → v2.7 promotion of `L_CEV_PTICK`.
- **Why ship the per-gate `advisory:` field as load-bearing?** v2.6 introduced the `advisory: <bool>` field per-profile under `percentile_budgets_ns:` but the v2.6 harness clamped ALL gates to advisory regardless of severity (the v2.6 transitional behaviour). v2.7 promotes the field to load-bearing: the harness reads `advisory: true|false` and dispatches to warn-only / exit-1 accordingly. This is a schema semantics change (the field already exists but its enforcement changes) — schema bump 0.8.0 → 0.9.0 documents the change.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Direct sparse linear solve via `nalgebra` + SuiteSparse | Adds 2 deps (over budget); GPL-incompatible (SuiteSparse); preconditioner-dependent (BiCG/GMRES convergence depends on problem); kills the "preconditioner-free" narrative that is the entire industrial-priority story for C6 Tikhonov (v4.0 example). |
| Iterative BiCG / GMRES on `(λI − A_disc)` | Same problem as direct solver: preconditioner-dependent, problem-specific, kills suckless dep-count. Also: would require a generic matrix-vector-product trait — premature abstraction. |
| Faber polynomials (Crouzeix-Knizhnerman) | Scope creep into operator-theoretic complex analysis. Hits the same complex-arithmetic blocker that drives B6 SemiflowComplex to v4.0; would require complex-valued operator support before v3.0 trait redesign. Defer. |
| Tanh-sinh (double-exponential) quadrature instead of Gauss-Laguerre | Higher node count needed (typically 256–512) for the same accuracy on `e^{-s} · h(s)`; per-call cost dominates. Gauss-Laguerre is the natural choice for `[0, ∞)` integrands of this weight. |
| Adaptive quadrature (e.g., QUADPACK QAG) | Allocating, recursive (no_std-hostile), and not bit-reproducible — contract-violating for the deterministic test suite. Fixed-node Gauss-Laguerre is reproducible by construction. |
| Single `eval` entrypoint (no `eval_at_point`) | Forces the L-gate harness to allocate a full state per tick — kills the per-tick latency story (HFT use case). Two entrypoints, shared kernel, no_std-safe. |
| Ship `LaplaceQuadrature` as a trait (not enum) | Two variants only in v2.7; a trait would be premature. The enum is a closed set; adding `GaussLegendreOnFiniteInterval { a, b }` or `TanhSinhDoubleExp` later is an additive `#[non_exhaustive]` enum extension (semantically equivalent surface). |
| Make `LaplaceQuadrature::GaussLaguerre32` carry runtime node count `N` | Const-array storage forbids runtime `N`. Re-computing nodes/weights per call would be O(N²) inside the hot path and require `libm::erf` + Newton iteration. `32` is a fixed const-array choice; if we ever need other counts they become separate variants (`GaussLaguerre64`, etc.). |
| Defer L_RESOLVENT_N64_P99 to v2.8 | v2.7 ships the resolvent — without a latency gate the contract is incomplete. The advisory-in-RC, blocking-in-final pattern mirrors the L_CEV_PTICK v2.6→v2.7 promotion and is the project's established calibration ladder. |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; no existing trait or struct is modified.
- **New module `crates/semiflow-core/src/resolvent.rs`** (~350 LoC budget, default 500-LoC cap with 150-LoC headroom). No constitution amendment needed.
- **Dependency count unchanged** at 2/3 budget (still `num-traits`, `libm` for `semiflow-core` core; `num-complex` reserved for v4.0 B6).
- **Schema bumps**: `properties.yaml` 0.8.0 → 0.9.0 (per-profile `advisory:` field load-bearing); `traits.yaml` 0.6.0 → 0.7.0 (new types). math.md is append-only (§22 NEW).
- **New gates**: G24 (RELEASE_BLOCKING — residual ≤ 1e-3 at n=64 + sweep slope ≥ 1.0); T19N (NORMATIVE sympy — Laplace transform of `exp(−π²t)·sin(πx)` equals `sin(πx)/(λ+π²)`); L_RESOLVENT_N64_P99 (advisory v2.7.0-rc.1 → blocking v2.7.0).
- **xtask change**: `xtask/src/latency_gate.rs` reads the per-profile `advisory:` field (currently TODO at lines 416, 421; promote to live logic). When `advisory: false` AND severity is `RELEASE_BLOCKING` AND a floor is breached, exit 1. Default `advisory:` if field is absent: `true` (backward-compatible with v2.6 entries that don't declare it).
- **Existing `L_CEV_PTICK` promotion**: `percentile_budgets_ns."i7-12700K".advisory: true → advisory: false` (explicit edit; documented in CHANGELOG as the v2.7 promotion). The other profiles (`m2-pro`, `aws-c7g-large`) remain `advisory: true` placeholders.
- **CITATIONs added to math.md §22**: Remizov 2025 *Vladikavkaz Math. J.* 27(4) Theorem 3 (Laplace-Chernoff convergence); Trefethen 2008 *SIAM Rev.* 50:1 (Gauss-Laguerre superalgebraic convergence on analytic integrands); Hille-Yosida theorem (textbook, cited for the resolvent ↔ Laplace identity).

## Migration

None for end-users. v2.6 binaries / crates link against v2.7 without recompilation. The L_CEV_PTICK promotion (`advisory: true → false`) tightens CI but does not invalidate any prior measurement: the v2.6 baseline `45 ns p999` is below the budget `50 ns p999` and continues to pass. Existing `latency_gates:` entries that don't declare `advisory:` per-profile default to `advisory: true` (warn-only) — no silent CI breakage.

The HFT side-track example `examples/heston_pricer.rs` (Wave D engineer task) is opt-in; it uses `LaplaceChernoffResolvent` for the semianalytic boundary terms but is not part of the test gate.

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0003 — no_std + alloc; const-array Gauss-Laguerre nodes are no_std-safe.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; reused for `LaplaceChernoffResolvent<C, F>` and `LaplaceQuadrature<F>`.
- ADR-0026 — `ChernoffFunction<F>` trait generic over `F`; the resolvent wraps any impl.
- ADR-0041 — `apply_into` + `ScratchPool`; the resolvent's hot loop uses `apply_into` for zero-alloc Chernoff steps.
- ADR-0068 — L-gate harness; the L_RESOLVENT_N64_P99 entry rides this infrastructure. Also: per-gate `advisory:` field semantics promoted from cosmetic-only (v2.6) to load-bearing (v2.7).
- ADR-0070 — Howland nonautonomous lift (v2.7 companion ADR; independent math, shared release window).
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v2.7 — release-level roadmap.
- math.md §22 (NEW v2.7) — Laplace-Chernoff resolvent normative spec.
- math.md §3.6.bis (v2.6) — amended with v2.7 advisory/blocking semantics (small amendment block).

## Amendments

(none at acceptance time)
