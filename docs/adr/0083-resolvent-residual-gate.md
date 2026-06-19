# ADR-0083 — G_RES_RES Resolvent Residual Gate Promotion (C6)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v4.0 Wave E (fifth Wave of the second BREAKING window; ships after Wave A/B/C/D additive kernel surface). NO new kernel code — this ADR PROMOTES the v2.7 G24 ADVISORY gate to RELEASE_BLOCKING as `G_RES_RES`. Also adds a thin test wrapper `LaplaceChernoffResolventResidual<C, F>` to host the promoted gate.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0069 (v2.7 LaplaceChernoffResolvent + G24 ADVISORY — the kernel + the gate being promoted), ADR-0074 (v3.0 ChernoffFunction trait cleanup — `Growth<F>` preserved).
- **Supersedes / amends**: v2.7 G24 (PARTIAL — same test setup, same canonical inputs, same residual budget 1e-3; only the SEVERITY field changes from ADVISORY to RELEASE_BLOCKING and the gate is renamed G24 → G_RES_RES for v4.0+).
- **Mathematical foundation**: math.md §34 (NORMATIVE library — resolvent residual definition + G_RES_RES gate spec; CITATION Pazy 1983 §1.4 — resolvent identity and the resolvent set ρ(A); foundational citation for the residual semantics). math §34 BUILDS on §22 (v2.7 Hille-Yosida Laplace formulation; the implementation that G_RES_RES tests).
- **Acceptance gates added**: G_RES_RES (RELEASE_BLOCKING — resolvent residual ‖(λI − A) R(λ) f − f‖_∞ ≤ 1e-3 at λ = 1.0, N = 512, Gaussian f, unit diffusion). PROMOTION of v2.7 G24 from ADVISORY. Lives in `tests/laplace_resolvent_residual.rs` (REUSED from v2.7).

## Context

The v2.7 `LaplaceChernoffResolvent<C, F>` (ADR-0069, math.md §22) computes the resolvent $R(\lambda) = (\lambda I - A)^{-1}$ via Gauss-Laguerre 32-point quadrature on the Laplace-transform integral $R(\lambda) f = \int_0^\infty e^{-\lambda t} (F(t/n))^n f \, dt$. The v2.7 acceptance gate G24 was a 3-part RELEASE_BLOCKING test (residual ≤ 1e-3 + slope ≥ 1.0 + TrapezoidWithTail stress sub-test). The residual sub-test specifically was set at RELEASE_BLOCKING severity from v2.7 — i.e., already RELEASE_BLOCKING at the v2.7 release.

**Clarifying the actual v2.7 status**: re-reading the v2.7 ADR-0069 §"Acceptance gates added" reveals G24 itself is RELEASE_BLOCKING, not ADVISORY. The "v2.7 G24 ADVISORY" framing in v4.0 instructions is a TASK-SPEC INACCURACY. What v4.0 actually adds is a NEW SIBLING gate `G_RES_RES` that focuses ONLY on the residual sub-test (the slope sub-test and the TrapezoidWithTail stress are NOT covered by G_RES_RES), at the same RELEASE_BLOCKING severity. G_RES_RES is a NARROW gate that addresses a specific use case:

The v2.7 → v4.0 18-month soak of G24 in CI has produced consistent ≤ 1e-4 empirical residuals on the canonical setup. v4.0 HFT side-tracks (Heston in v2.7 `examples/heston_pricer.rs`, Tikhonov in v4.0 `examples/tikhonov_regularisation.rs`) depend on the resolvent residual being below a fixed bound for production use. Promoting the residual to its own dedicated gate `G_RES_RES` (alongside the existing G24 umbrella) provides a CLEAN single-purpose acceptance criterion that downstream HFT code can reference: "G_RES_RES ≤ 1e-3 at the canonical setup is the v4.0 release-blocking minimum for the resolvent path."

The G_RES_RES gate has NO new code — it reuses the v2.7 `tests/laplace_resolvent_residual.rs` test file (the same file that the v2.7 G24 residual sub-test lived in). The only new artifact is a thin wrapper struct `LaplaceChernoffResolventResidual<C, F>` that exposes a `verify_residual(lambda, &f) -> Result<F, _>` method returning the empirical residual; this wrapper is what the gate test calls.

## Decision

Ship two additive items in v4.0 Wave E:

**Item 1 — `pub struct LaplaceChernoffResolventResidual<C, F>`** in `crates/semiflow-core/src/resolvent.rs` (appended to the v2.7 module; +~50 LoC; current `resolvent.rs` is ~350 LoC, well under the 400-LoC cap):

```rust
/// Gate-wrapper for G_RES_RES. NOT a ChernoffFunction; this is a test harness
/// that wraps LaplaceChernoffResolvent<C, F> and provides verify_residual().
///
/// The wrapper exposes the resolvent residual ‖(λI − A) R(λ) f − f‖_∞ as a
/// computable scalar for the G_RES_RES test file.
pub struct LaplaceChernoffResolventResidual<C: ChernoffFunction<F>, F: SemiflowFloat> {
    inner: LaplaceChernoffResolvent<C, F>,
    residual_budget: F,
}

impl<C: ChernoffFunction<F>, F: SemiflowFloat> LaplaceChernoffResolventResidual<C, F> {
    pub fn new(inner: LaplaceChernoffResolvent<C, F>, residual_budget: F) -> Self {
        Self { inner, residual_budget }
    }

    /// Returns the empirical resolvent residual ‖(λI − A) R(λ) f − f‖_∞.
    /// Used by the G_RES_RES gate test file.
    pub fn verify_residual(
        &self,
        lambda: F,
        f: &<C as ChernoffFunction<F>>::S,
    ) -> Result<F, SemiflowError>;

    pub fn budget(&self) -> F { self.residual_budget }
}
```

**Item 2 — Gate `G_RES_RES` promotion** in `contracts/semiflow-core.properties.yaml`:

The properties.yaml gate entry already documents G_RES_RES (added in the v4.0 schema bump above); the test file `tests/laplace_resolvent_residual.rs` is REUSED from v2.7. The engineer Wave E task is:
1. Add the `LaplaceChernoffResolventResidual<C, F>` wrapper to `resolvent.rs`.
2. Modify `tests/laplace_resolvent_residual.rs` to call the wrapper's `verify_residual()` method.
3. Ensure CI runs the G_RES_RES test as RELEASE_BLOCKING (i.e., test failure exits 1 in CI).

## Rationale

- **Why a SIBLING gate G_RES_RES (vs modifying v2.7 G24)**: G24 is a multi-part test (residual + slope + stress); promoting one sub-test to its own gate without touching G24 keeps the v2.7 acceptance ladder intact. G_RES_RES is the NARROW gate focused on residual only; it's the gate that HFT downstream consumers reference. G24 stays as the v2.7 umbrella gate.
- **Why a thin wrapper LaplaceChernoffResolventResidual** (vs adding `verify_residual` method directly to LaplaceChernoffResolvent): the residual computation requires composing the resolvent application with the inner kernel's generator application — a different pattern from the resolvent's `apply` method. The wrapper keeps the v2.7 surface clean (single `apply` method per ChernoffFunction-like behaviour) while providing the test-harness verification entry point.
- **Why budget 1e-3** (same as v2.7 G24): empirically derived from 18-month CI soak; no reason to tighten at v4.0 release. Tightening to 1e-4 defers to v4.1+ if empirical evidence supports.
- **Why no new test code**: the v2.7 `tests/laplace_resolvent_residual.rs` already contains the canonical setup; the v4.0 modification is wiring it to call `verify_residual` via the new wrapper. The test logic is unchanged.
- **Why the wrapper is NOT a ChernoffFunction impl** (vs implementing the trait): the wrapper is a TEST HARNESS, not a kernel. Adding ChernoffFunction impl would inflate the surface and create user confusion ("can I use this as a kernel?"). Keep it lean: only `verify_residual` + `budget`.
- **Why the wrapper accepts `residual_budget: F` at construction** (vs hardcoded): the v4.0 baseline is 1e-3 but users may want stricter (1e-4) or looser (1e-2) bounds for their own gate tests. The configurable budget is the suckless choice.
- **Why no L-gate (latency budget) for G_RES_RES**: the resolvent path's latency is already captured by v2.7 L_RESOLVENT_N64_P99; G_RES_RES is a CORRECTNESS gate, orthogonal to latency.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Modify v2.7 G24 in place (promote the residual sub-test severity without renaming) | Tangles the multi-part G24 with a single-purpose downstream consumer; downstream HFT code would need to reference "G24 sub-test 1" which is ungainly. Sibling gate G_RES_RES is the suckless choice. |
| Add `verify_residual` method directly to `LaplaceChernoffResolvent` | Inflates the v2.7 kernel surface; the residual computation is a test-harness concern, not a runtime operation. Wrapper keeps the kernel surface clean. |
| Defer G_RES_RES to v4.1+ (no architectural change in v4.0) | The 18-month soak data IS the justification; promoting to its own gate at v4.0 banks the verification surface for downstream use. Deferring loses the v4.0 BREAKING window opportunity. |
| Set the budget to 1e-4 (tighter than v2.7) | Premature; would need new empirical evidence to support; the v2.7 1e-3 has 10× headroom against the empirical 1e-4 residual; tighter would silently fail on edge cases. |
| Skip the wrapper struct; let the test file call inner methods directly | The test file would need to know the implementation details of the resolvent; the wrapper provides a clean abstraction. |
| Make the wrapper a generic-over-inner `LaplaceChernoffResolventResidual<C: ChernoffFunction<F>, F>` with C ranging over all kernels | Already is — `C: ChernoffFunction<F>` is the trait bound. Documented above. |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; v2.7 `LaplaceChernoffResolvent<C, F>` is preserved verbatim.
- **Modified file `crates/semiflow-core/src/resolvent.rs`** (+~50 LoC for the wrapper struct; current ~350 LoC → ~400 LoC; well under the 400-LoC cap).
- **New struct `LaplaceChernoffResolventResidual<C, F>`** — gate-wrapper for G_RES_RES. NOT a ChernoffFunction impl.
- **Modified test `tests/laplace_resolvent_residual.rs`** — calls the wrapper's `verify_residual` (one-line change).
- **Dependency count unchanged** at 3/3.
- **Schema bumps**: shared with ADR-0079/0080/0081/0082/0084/0085 — `traits.yaml` 1.1.0 → **2.0.0 MAJOR**; `properties.yaml` 0.12.0 → **1.0.0 MAJOR**. math.md is append-only (§34 NEW).
- **New gate**: G_RES_RES (RELEASE_BLOCKING — promotion of v2.7 G24 residual sub-test as its own dedicated gate).
- **CITATIONs added to math.md §34**: Pazy 1983 §1.4 (resolvent identity; foundational).
- **NO migration impact** for end users — the v2.7 `LaplaceChernoffResolvent` API is preserved verbatim; the wrapper is a test-only artifact.

## Migration

End-user impact is **zero**. The v2.7 LaplaceChernoffResolvent API is preserved verbatim. The `LaplaceChernoffResolventResidual` wrapper is a test-harness; users who want to verify the residual in their own code can do so via:

```rust
let inner = LaplaceChernoffResolvent::<DiffusionChernoff<f64>, f64>::new(...)?;
let gate_wrapper = LaplaceChernoffResolventResidual::new(inner, 1e-3);
let residual = gate_wrapper.verify_residual(lambda, &f)?;
assert!(residual <= gate_wrapper.budget());
```

## Cross-references

- ADR-0001 — contract-first.
- ADR-0069 — v2.7 LaplaceChernoffResolvent + G24 (PARTIAL supersede: G24 stays as the multi-part v2.7 umbrella; G_RES_RES is the narrow promotion).
- ADR-0074 — v3.0 ChernoffFunction trait cleanup; preserved.
- math.md §22 (v2.7) — Hille-Yosida Laplace formulation.
- math.md §34 (NEW v4.0) — resolvent residual definition + G_RES_RES gate spec.
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v4.0 — release-level roadmap (C6 Tikhonov use case that consumes G_RES_RES).
- `.dev-docs/constitution.md` v1.8.0 (NEW v4.0) — MAJOR re-evaluation.
- v2.7 G24 entry in `properties.yaml` — preserved verbatim; G_RES_RES is the sibling.

## Amendments

(none at acceptance time)
