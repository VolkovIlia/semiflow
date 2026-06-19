# ADR-0033 — `NonSeparable2DChernoff` (scalar-c) deprecation policy

**Status**: SUPERSEDED-BY-0058 (superseded 2026-05-21)
**Date**: 2026-05-09
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0016 (scalar-`c` introduction, v0.7.0), ADR-0023
(anisotropic-`β` introduction, v0.9.0), ADR-0029 (v0.11.0 milestone —
docs-only release; this ADR satisfies the zero-core-diff gate),
constitution v1.1.0 §"Project-Specific Principles" #2 (additive, never
subtractive), `docs/audit-findings-v0_9_0.md` §OPEN/O-3 (this ADR resolves
the finding). No source code change in v0.11.0; no follow-up implementation
work scheduled.

## Context

v0.7.0 (ADR-0016) shipped `NonSeparable2DChernoff<X, Y, F>` for the
scalar-coefficient case `L = L_x ⊗ I + I ⊗ L_y + c(x,y) · ∂_x∂_y`. v0.9.0
(ADR-0023) shipped `NonSeparable2DAnisotropicChernoff<X, Y, F>` for the
position-dependent generalisation `β(x,y) · ∂_x∂_y`. The aniso type is
mathematically a strict generalisation: scalar-`c` is the degenerate case
`β ≡ const`. The v0.9.0 audit (O-3, severity LOW, planning-only) flagged
that the v1.0.0 freeze must record an explicit policy: keep both forever
(Option A), `#[deprecated]` cycle removing scalar-`c` at v2.0.0 (Option B),
or reimplement scalar-`c` as a documented sugar wrapper around aniso
(Option C).

## Decision

**Option A — Keep both types at v1.0.0 freeze and indefinitely thereafter.**
Both `NonSeparable2DChernoff<X, Y, F>` and
`NonSeparable2DAnisotropicChernoff<X, Y, F>` are first-class public APIs
with no deprecation marker. Rationale: (1) ADR-0023 §Consequences is an
explicit "purely additive, v0.7.0/v0.8.x callers remain bit-equal" promise
the project has already shipped — Option B would retroactively break that
contract at v2.0.0. (2) Constitution v1.1.0 "additive, never subtractive"
applies directly. (3) Suckless guardrail #1 favours small surface area for
the *median caller's* dependency footprint, not minimum type count;
scalar-`c` callers pay zero cost for the existence of aniso, and aniso
callers pay zero cost for the existence of scalar-`c` — neither type
imports the other at the source level (`nonseparable2d.rs` and
`nonseparable2d_aniso.rs` are independent modules per ADR-0023
§Consequences). (4) Cargo SemVer's `#[deprecated]` mechanism (Option B)
is the canonical Rust tool when an API is *wrong*; here scalar-`c` is the
*correct, simpler* API for genuinely isotropic callers (e.g. CEV
option-pricing with a single covariance scalar) — forcing migration to a
verbose `|_x,_y| c` closure is anti-suckless ("explicit input over
implicit estimator", same axis as ADR-0023's `c_norm_bound` /
`beta_norm_bound` rule). (5) Option C (sugar-wrapper) hard-depends on a
v0.12.0+ `with_closure` constructor that ADR-0029 explicitly defers and
that does not exist at v0.11.0; Option C also breaks the bit-equal fast
path documented at ADR-0023 §Decision (the `is_zero_c == 0.0` branch goes
direct to `Strang2D::apply` for FP determinism — a wrapper would have to
re-prove that property after the closure indirection). The math-level
relationship (scalar-`c` is the `β ≡ const` degeneration of aniso) is
already documented in `nonseparable2d_aniso.rs:65-68` rustdoc and in
math.md §10.7-ter cross-refs to §10.7-bis; no further documentation
amendment is needed for v0.11.0.

## Consequences

- **Pro**: zero migration burden for v0.7.0+ callers through v1.x and
  beyond; ADR-0023 additivity promise honoured at v1.0.0 freeze.
- **Pro**: ADR-0029 zero-core-diff gate preserved (docs-only ADR; no
  `crates/semiflow-core/**` change).
- **Pro**: scalar-`c` callers keep the simpler API surface (one f64
  bound + one `fn(F,F) -> F`) instead of being forced into the aniso
  signature with a constant closure.
- **Con**: two types covering overlapping math is a soft minimalism
  violation — accepted because the violation is internal-to-the-crate
  surface area, not caller-visible complexity (independent modules,
  independent rustdoc, no shared trait that forces both types into the
  same documentation page).
- **Follow-up**: none. Audit O-3 is closed by this ADR with no v0.12.0
  or v1.0.0 implementation work. v1.0.0 audit (per ROADMAP.md "Math
  fidelity commitment" rule #5) re-affirms the policy; if a future
  caller-survey reveals scalar-`c` has no users, a separate v2.0.0 ADR
  may revisit (not committed).

## Supersession Note

This ADR was superseded by ADR-0058 (`NonSeparableAniso<S: Discrete<F>>`
unification) on 2026-05-21. ADR-0058 extends the "keep both" Option A policy
into a unified generic trait hierarchy; the scalar-`c` / aniso policy decision
recorded here remains historically correct but ADR-0058 is now the normative
reference for `NonSeparable2D*` type governance.

## Alternatives Considered

- **Option B (`#[deprecated]` at v0.12.0, remove at v2.0.0)**: rejected.
  Breaks ADR-0023 §Consequences "purely additive" promise; forces
  migration cost onto v0.7.0+ callers for whom scalar-`c` is the natural
  API; `#[deprecated]` is the canonical Rust tool for *wrong* APIs, not
  *simpler-than-necessary* APIs.
- **Option C (sugar-wrapper around aniso)**: rejected. Hard-depends on
  v0.12.0+ `with_closure` constructor that ADR-0029 defers; breaks the
  bit-equal `is_zero_c` fast path documented at ADR-0023; introduces
  closure-indirection runtime cost (negligible but non-zero) for a
  refactor with no user-visible benefit.
