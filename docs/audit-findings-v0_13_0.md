---
title: "Audit findings — v0.13.0"
version: 0.13.0
status: FINALIZED — pending iter-4 bench + tag
date: 2026-05-20
---

# Audit findings — v0.13.0

## Scope

Architectural performance milestone per plan `/home/volk/.claude/plans/vast-riding-bachman.md`.
All waves shipped as additive perf refactors with bit-equal regression gates; no
math identity changed.

| Wave | Commit | Summary |
|------|--------|---------|
| A2 | `52bac39` | Strang3D serial in-place rewrite (5× O(N³) alloc → 1× clone + 3 in-place passes) |
| B1 | `e0a993c` | TruncatedExp4 wasted-sample bypass at exact grid nodes (`catmull_rom(s=0) ≡ p0`) |
| B2 | `28d3134` | `HalfNodeCoeffCache<F>` storage refactor eliminates closure indirection |
| B3 | `9e93d6e` | AVX2 (x86_64) + NEON (aarch64) SIMD on cached K=4 stencil, FMA-disabled |
| C1 | `6a1273d` | `REMIZOV_PARALLEL_THRESHOLD` env override on `parallel1d.rs` |
| C2-abort | `6a1273d` | Small-N fused-axis Strang aborted; `apply_fused` retained as `#[allow(dead_code)]` |
| C3 | `6a1273d` | `strang_fused_order_confirmation_{2d,3d}` alert gates asserting O(τ¹) on fused path |
| D1 | `cf430ce` | `DiffusionChernoff` const-a fast path skips ζ-A τ²-correction (mathematically identical when `a' ≡ 0`) |
| D2 | `56ce23b` | `[profile.release-wasm]` cuts bundle below 500 KB target |
| D3 | `9b2df63` | xtask `binary-size-check` + CI workflow gate + ADR-0040 |

Cross-cutting v1.0.0 hygiene (out of perf scope but landed in same milestone):

- `b6ae1d3` ADR-0038 — State trait experimental marker for v2.0.0
- `86a1f00` + `35318e6` — `remizov-graph-spike` crate (publish=false, 3-trait
  shape validation for v2.0.0 Trait State, no impact on shipped API surface)

## DEVIATIONS

| ID | Severity | Description | Mitigation | ADR | Status |
|----|----------|-------------|------------|-----|--------|
| D-001 | INFO | Wave C2 small-N fused-axis Strang fast path measured slopes **+1.137 (2D)** and **+1.261 (3D)** vs required ≤ −1.8 for O(τ²). The fused `Y(τ) ∘ X(τ)` path is O(τ¹) relative to palindromic Strang because `[L_x, L_y] = 0` ensures the *exact* semigroups commute but the *Chernoff discrete* approximations do **not** commute at O(τ²); palindromic symmetry is required for τ²-error cancellation. | `apply_fused` retained as `#[allow(dead_code)]` private method. `strang_fused_order_confirmation_{2d,3d}` alert gates assert O(τ¹) — they will trip if fused ever becomes τ²-accurate via unrelated change. User-facing `apply()` API unchanged. | ADR-0039 | ACCEPTED — extends F2 algorithmic floor disclosure (ADR-0037) to F5/F7 bench gaps. |

### D-001 details

The fused fast path was hypothesised in plan `vast-riding-bachman.md` Wave C2 as
a small-N optimisation skipping the palindromic half-step decomposition:

```
fused:       e^{τL_y} ∘ e^{τL_x}                  (one round-trip)
palindromic: e^{τL_x/2} ∘ e^{τL_y} ∘ e^{τL_x/2}    (Strang sandwich)
```

For exact semigroups with `[L_x, L_y] = 0` these are identical. The Chernoff
approximations `Tx(τ) ≈ e^{τL_x}` carry per-step τ²-residue, however, and only
palindromic placement around `Ty(τ)` cancels the cross-term at O(τ²). Wave C2's
empirical slopes (+1.137, +1.261) confirm the τ¹ regime — exactly the predicted
mathematical behaviour, not a numerical bug. The decision to **abort** rather
than ship is documented in ADR-0039; the code stays in-tree behind an alert gate
to prevent silent re-emergence.

## SIMPLIFICATIONS

Storage refactors that preserve mathematical identity (bit-equal vs the prior
implementation on the same input). Each row is gated by a regression test that
fails on any byte-level divergence.

| ID | Description | ADR | Bit-equality gate | # tests |
|----|-------------|-----|-------------------|---------|
| S-001 | Strang3D serial in-place rewrite (5× O(N³) alloc → 1× clone + 3 in-place passes) | ADR-0022 Amendment 1 | `STRANG3D_SERIAL_SCRATCH_BIT_EQUAL` | 3 |
| S-002 | TExp4 sample bypass at exact grid nodes (`catmull_rom(s=0) ≡ p0` exact) | ADR-0019 Amendment 2 | `TEXP4_BYPASS_BIT_EQUAL` | 2 |
| S-003 | `HalfNodeCoeffCache<F>` precomputes `a(x)` at half-nodes once (closure indirection eliminated) | ADR-0034 Amendment 1 | `TEXP4_CACHED_COEFF_BIT_EQUAL` | 2 |
| S-004 | AVX2/NEON SIMD on K=4 cached stencil, FMA-disabled (bit-equality preserved per ADR-0019 v0.8.0 precedent) | ADR-0019 Amendment 2 | `TEXP4_SIMD_BIT_EQUAL` | 3 |
| S-005 | `DiffusionChernoff` const-a fast path skips ζ-A correction (mathematically identical when `a' ≡ 0`) | ADR-0040 | `CONST_A_BIT_EQUAL` | 5 |

### Notes on FMA

S-004 follows the ADR-0019 v0.8.0 precedent: SIMD intrinsics are used with FMA
**disabled** so that the rounding behaviour exactly matches the scalar reference.
This is the only way to preserve bit-equality across architectures (x86_64 with
AVX2, aarch64 with NEON) and against the pre-SIMD baseline. Performance gain
comes from vector-width parallelism, not from FMA's reduced rounding.

### Notes on const-a fast path

S-005 short-circuits the ζ-A τ²-correction only when the variable-coefficient
closure is detected as constant (caller opts in via the `ConstA` variant). The
math says: when `a(x) ≡ a₀`, the τ²-correction term `(τ²/12) · (a²(x))''` is
identically zero, so the corrected and uncorrected stencils coincide pointwise.
The fast path encodes this identity directly instead of computing zero. Bit-equal
gate covers 5 test points spanning the supported envelope.

## APPROXIMATIONS

**None.** All shipped changes preserve mathematical identity. The only behaviour
change is in `DiffusionChernoff::ConstA` (S-005), and that change is provably
equivalent to the prior `apply()` output when `a' ≡ 0` — which is the precondition
for the `ConstA` variant to be selected.

## Math fidelity gates

All flagship slope gates re-verified green on the v0.13.0 close-out session:

| Gate | Status | Notes |
|------|--------|-------|
| G3⁶-2D (ADR-0020) | green | unchanged from v0.8.1 |
| G5_3D (3D tensor) | green | unchanged from v0.11.0 |
| G4_NS2D_aniso | green | unchanged from v0.9.0 |
| G3⁴-2D | green | unchanged |
| G3-2D | green | unchanged; Wave C2 abort did **not** degrade this gate |
| T9N_* sympy | 6/6 PASS | NORMATIVE |
| T10N_* sympy | 6/6 PASS | NORMATIVE |
| T7N_* sympy | green | NORMATIVE |

### New gates added in v0.13.0

| Gate | # tests | Wave | Purpose |
|------|---------|------|---------|
| `STRANG3D_SERIAL_SCRATCH_BIT_EQUAL` | 3 | A2 | byte-equality vs pre-refactor 3D serial |
| `TEXP4_BYPASS_BIT_EQUAL` | 2 | B1 | byte-equality at exact grid nodes |
| `TEXP4_CACHED_COEFF_BIT_EQUAL` | 2 | B2 | byte-equality vs closure-dispatch baseline |
| `TEXP4_SIMD_BIT_EQUAL` | 3 | B3 | byte-equality scalar ↔ AVX2/NEON |
| `CONST_A_BIT_EQUAL` | 5 | D1 | byte-equality fast path ↔ general path when `a' ≡ 0` |
| `strang_fused_order_confirmation_2d` | 1 | C3 | asserts O(τ¹) on fused (alert gate per D-001) |
| `strang_fused_order_confirmation_3d` | 1 | C3 | asserts O(τ¹) on fused (alert gate per D-001) |

**Total new bit-equal / order-confirmation tests in v0.13.0: 17.**

## Algorithmic floor disclosures

Consolidated statement per **ADR-0037** + **ADR-0039**, covering wallclock gaps
observed in iter-3 benches F2 (kiops Krylov-Magnus), F5 (scipy-mol-2d), and F7
(scipy-mol-3d):

> These wallclock gaps are **not** implementation-bound — they reflect algorithmic
> strategies that are intrinsically superior for those operator classes:
>
> - **F2** — Krylov-Arnoldi adaptive substepping (kiops) exploits low-rank action
>   structure that operator-splitting Chernoff schemes cannot match at low rank.
> - **F5 / F7** — method-of-lines (scipy-mol) avoids the operator-splitting
>   τ²-cancellation requirement entirely; the variable-coefficient correction
>   that RC pays per step is amortised by the ODE solver across many steps.
>
> RC preserves **H-MEM Pareto dominance** in all three cases (50–254× less memory
> per iter-3 §5.1), which remains the project's flagship hypothesis.

## Audit results for v0.10.0 + v0.11.0 (carry-overs)

Per `docs/audit-findings-v0_10_0.md` and `docs/audit-findings-v0_11_0.md`: no
v0.13.0 work introduced new DEVIATIONs to those prior releases. The bindings
boundary (ADR-0028) and the I13 audit DEVIATION inventory remain unchanged.

Specifically verified:

- `semiflow-ffi` (v0.10.0 Wave A) — no math touched; const-a fast path is
  invoked through the same `apply()` entrypoint and inherits S-005 bit-equality.
- `semiflow-py` (v0.10.0 Wave B) — no PyO3 surface change; GIL release path from
  v0.11.0 I6 unchanged.
- `semiflow-wasm` (v0.10.0 Wave C) — Wave D2's `[profile.release-wasm]` only
  affects bundle size (binary-size CI gate at 500 KB); JS class surface and
  numerical output unchanged.

## Open questions for v0.13.x patch releases

(populated as patch needs surface; none open at tag time)

## References

- ADR-0019 Amendment 2 (SIMD on TExp4) — `docs/adr/0019-simd-intrinsics.md`
- ADR-0022 Amendment 1 (Strang3D serial scratch) — `docs/adr/0022-parallel-tile-scratch.md`
- ADR-0034 Amendment 1 (HalfNodeCoeffCache) — `docs/adr/0034-with-closure-api.md`
- ADR-0037 (F2 algorithmic floor) — `docs/adr/0037-f2-algorithmic-floor.md`
- ADR-0038 (State trait experimental marker for v1.0.0) — `docs/adr/0038-state-trait-experimental.md`
- ADR-0039 (small-N fused-axis abort) — `docs/adr/0039-small-n-fused-axis.md`
- ADR-0040 (const-a + WASM size + CI binary-size gate) — `docs/adr/0040-const-a-fast-path-binary-size.md`
- Plan: `/home/volk/.claude/plans/vast-riding-bachman.md`
- Iter-3 H-MEM Pareto dominance: project memory `feedback_bench_memory_first`
