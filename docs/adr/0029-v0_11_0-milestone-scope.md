# ADR-0029 — v0.11.0 milestone scope (bindings-polish, zero core surface additions)

**Status**: Accepted (planning ADR for v0.11.0)
**Date**: 2026-05-09
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0028 (v0.10.0 bindings milestone — supersedes its "Out of scope"
list for I1/I2/I12/I13 only), ADR-0027 (MCP withdrawn — re-affirmed for v0.11.0),
ADR-0008 (additive-surface principle), constitution v1.1.0 §"Project-Specific
Principles" #2 (additive, never subtractive)

## Context

v0.10.0 (commit `92bd484`, tag `v0.10.0`, 2026-05-09) shipped three additive
binding crates restricted to 1D heat with constant unit diffusion `a(x) = 1.0`.
Product Strategist's Stage-1 scope (`acceptance.md` + `clarity-scan.md`,
quality-gate PASSED) inventoried 14 deferred items (I1–I14) and applied MoSCoW.
The driving constraint identified in `clarity-scan.md` F7.1 (CRITICAL,
resolved): variable-`a(x)` for FFI/PyO3/WASM (I3) requires a closure-capable
constructor on `semiflow-core::DiffusionChernoff` (Strategy A — additive
sibling `with_closure`) which is itself a v0.12.0-scoped Architect milestone.
Landing it speculatively in v0.11.0 would (a) couple two unrelated changes
("unblock the API" vs "expose to all 3 bindings"), (b) risk a v0.11.0 → MAJOR
escalation if the closure type ends up touching a public trait, and (c) burn
the v0.11.0 → v0.12.0 → v1.0.0 budget on a single concern. v0.11.0 must
therefore be a polish release: ship npm distribution, lift the GIL, close
v0.9.0+v0.10.0 math-fidelity debt, and add nothing to `semiflow-core`.

## Decision

v0.11.0 is the **bindings-polish milestone**: a SemVer MINOR release that
ships **zero `semiflow-core` API additions** (verified by reviewer-suckless
gate against `crates/semiflow-core/src/**/*.rs` diff at tag time) and confines
all changes to the three sibling binding crates plus CI workflow files.
The MUST set is exactly four items: **I1** (npm publish for `semiflow-wasm`,
ADR-0030), **I6** (PyO3 GIL release, ADR-0031), **I12** (heavy-validation
runs on production hardware recorded in `docs/audit-findings-v0_11_0.md`,
ADR-0032), and **I13** (researcher-driven math-fidelity audit for v0.9.0 +
v0.10.0 producing `docs/audit-findings-v0_9_0.md` and `docs/audit-findings-v0_10_0.md`,
non-blocking on findings per AC-4). SHOULD set (I2 cross-engine WASM smoke,
I7 Python `.pyi` stubs, I8 pyo3/numpy/wasm-bindgen lockstep version bumps)
ships in v0.11.0 if engineer effort permits, else slips to v0.11.x. WON'T
set (I3 variable-`a` bindings, I4 2D bindings, I5 3D bindings, I14 async
primitives) is hard-deferred to v0.12.0+ — these items MUST NOT appear in
the v0.11.0 deltas, source diff, or CHANGELOG. Reviewer-suckless gates
v0.11.0 release on the bindings-only invariant: any non-empty
`git diff v0.10.0..v0.11.0 -- crates/semiflow-core/` blocks tag.

## Consequences

- **Pro**: separates the closure-API decision (v0.12.0 architecture work)
  from binding-distribution polish; unblocks persona P1 (npm-install) and
  P2 (GIL release) without re-opening core.
- **Pro**: closes the v0.9.0+v0.10.0 math-fidelity debt before v1.0.0
  freeze, making the v1.0.0 audit (ROADMAP rule #5) incremental rather
  than retrospective.
- **Pro**: constitution overrides unchanged (file-cap RELAX touches only
  math-heavy core modules; v0.11.0 changes are all <80 LoC outside core).
- **Con**: persona P2's "real PDE problems have variable diffusion" pain
  point persists through v0.11.0 — explicitly accepted as v0.12.0 scope.
- **Follow-up**: v0.12.0 Architect ADR will define `DiffusionChernoff::with_closure`
  signature + `Box<dyn Fn>` vs generic-parameter trade-off (Strategy A);
  v0.12.0 will then expose it through all three bindings.
- **Follow-up**: v1.0.0 audit re-evaluates Override #1 (file cap) and
  Override #2 (MCP) per constitution §Override-rules.

## Alternatives Considered

- **Land Strategy A (`with_closure`) in v0.11.0 to seed v0.12.0 bindings**
  — rejected (clarity-scan F7.1 CRITICAL; couples unrelated decisions; risks
  surface leak through `Box<dyn Fn>` lifetime parameter into trait bounds).
- **Land 2D/3D constant-`a` bindings in v0.11.0** — rejected (degenerate without
  variable coefficients; closed-form 2D Gaussian oracle is uninteresting; the
  binding wrapper would be rewritten in v0.12.0 once `β(x,y)` lands).
- **Defer I12 heavy validation to v1.0.0 audit** — rejected (the slope
  numbers feed I13 audit; doing them in two passes wastes the wallclock
  budget twice and breaks the "audit reads, doesn't run" boundary set in
  clarity-scan F1.3).
- **Skip I13 audit, fold into v1.0.0** — rejected (v1.0.0 must audit only
  the v0.11→v1.0 delta, not the v0.9→v1.0 superset; doing I13 now keeps
  v1.0.0 audit scoped and reviewable).
