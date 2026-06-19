# ADR-0035 — v1.0.0 API stability commitment (four-crate freeze)

**Status**: Accepted (planning ADR for v1.0.0)
**Date**: 2026-05-10
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0001 (contract-first math library — this ADR finalises the
contract for v1.x), ADR-0028 §"API stability" (seed of the v1.0.0 freeze;
deferred experimental status of v0.10.0 bindings), ADR-0028 Amendment 1 (Wave C
profile divergence — preserved at freeze), ADR-0033 (NonSeparable2D
coexistence — both types are first-class at v1.0.0), ADR-0034 §"Suckless audit"
(Copy → Clone SemVer note — last pre-1.0 breaking change), ADR-0018 (parallel
`Strang2D` `Send + Sync` invariant — preserved), ADR-0019 (SIMD intrinsics — bit-equality
gate is part of the v1.0.0 contract), ADR-0026 (`ChernoffFunction` generic-over-`F` —
trait shape frozen), ADR-0027 (MCP withdrawn — re-affirmed; no MCP in v1.0.0),
ROADMAP §"v1.0.0 (stability commitment)", `docs/api-stability.md` (companion
user-facing policy doc, S2.1).

## Context

v0.x has reached the natural transition to v1.0.0: the math surface is closed
(2D anisotropic non-separable per ADR-0023; 3D tensor per ADR-0024; 6th-order
spatial per ADR-0015; Magnus integrators per ADR-0011; adaptive PI controller
per ADR-0014), the binding crates are published (`semiflow-ffi` per ADR-0028
Wave A; `semiflow-py` per ADR-0028 Wave B; `semiflow-wasm` per ADR-0028 Wave C +
Amendment 1), npm distribution is live (ADR-0030), the v0.9.0 + v0.10.0
math-fidelity audits have closed (ADR-0029 §I13; `docs/audit-findings-v0_9_0.md`,
`docs/audit-findings-v0_10_0.md`), the GIL-release optimisation has shipped
(ADR-0031), and the variable-`a(x)` closure API has landed (ADR-0034 / v0.12.0
`DiffusionChernoff::with_closure`). The forces requiring this ADR now: (1) v1.0.0
is the first SemVer-MAJOR release and the moment to commit users to a stable
public surface — without an explicit freeze, downstream code (research scripts,
external bindings, paper-reproduction artifacts) cannot reason about upgrade
cost; (2) ADR-0028 §"API stability" already promised that the v1.0.0 freeze
covers Rust + FFI + Python + WASM **simultaneously** — this ADR honours that
commitment; (3) the library is a math crate where stability has two distinct
axes — *symbolic* (identifiers, signatures, error variants stay callable) and
*semantic* (numerical outputs of named operators remain bit-stable for the
deterministic `f64` path) — both axes need an explicit policy; (4) the
companion `docs/api-stability.md` (S2.1, written in parallel by docs-writer) is
the user-facing policy; this ADR is the architectural decision record that
backs it. Per Anchor's S2.2 audit (in flight), `crates/semiflow-core` exposes
~53 public items; the three binding crates mirror the relevant subset. The
constitution v1.1.0 §"Project-Specific Principles" #2 ("additive, never
subtractive") covers source-file evolution but is silent on caller-visible API
freeze policy — this ADR fills that gap for v1.0.0 onward.

## Decision

**Four-crate simultaneous v1.0.0 freeze.** `semiflow-core` v1.0.0,
`semiflow-ffi` v1.0.0, `semiflow-py` v1.0.0, and `semiflow-wasm` v1.0.0 ship in a
single coordinated release. From v1.0.0 onward, all four crates take SemVer
commitments per `docs/api-stability.md` (companion user-facing policy doc).
The MAJOR-bump barrier (v1 → v2) is triggered exclusively by:

1. **Removal** of any pub item (function, struct, enum, trait, type alias, module).
2. **Signature change** of any pub item that breaks source compatibility for
   downstream callers (parameter type, return type, generic bound, lifetime).
3. **Behavioural change** that violates a contract documented in rustdoc, in
   `contracts/semiflow-core.math.md`, in `docs/api-stability.md`, or in a
   NORMATIVE math.md section (sympy gates).
4. **Error-variant rename or removal** for any public error enum
   (`SemiflowError`, `SemiflowStatus`, `SemiflowError.kind` for PyO3, the
   `kind`-discriminated `JsValue` for WASM per ADR-0028 Amendment 1). Adding
   variants is MINOR-tolerable when the enum is `#[non_exhaustive]`; for v1.0.0
   `SemiflowError` is **not** marked `#[non_exhaustive]` (decision: a new
   variant is a MAJOR bump, exchanged for caller-side exhaustive `match`
   ergonomics — see Considered alternatives §c).

**Numerical-output stability commitment** (the semantic axis). The
deterministic scheme outputs of every named integrator on the `f64` `FnPtr`
path — `Strang2D`, `Strang3D`, `DiffusionChernoff`, `TruncatedExp`,
`TruncatedExp4`, `Diffusion6thChernoff`, `DriftReactionChernoff`,
`NonSeparable2DChernoff`, `NonSeparable2DAnisotropicChernoff`,
`AdaptivePI`, `Magnus4thDiffusionChernoff`, and the `ChernoffSemigroup`
iteration around any of them — are **bit-stable across PATCH and MINOR bumps**
on a fixed hardware/target tuple (canonically `x86_64-unknown-linux-gnu`,
SIMD-on, parallel-on with deterministic reduction order per ADR-0018). The
SIMD bit-equality gate (`tests/simd_bit_equal.rs`, `tests/simd_speedup.rs`) is
part of the v1.0.0 contract: `--features simd` results MUST byte-equal
scalar-fallback results to ULP for every regression-tested input. The
`Storage::Closure` path (ADR-0034 / v0.12.0) is bit-stable for a given
closure (the closure is the user's responsibility); bit-equality between the
`FnPtr` and `Closure` variants for the same coefficient function is itself a
regression gate (`tests/diffusion_with_closure.rs`).

**HW-dependent bit-stability** (the explicit non-promise). Bit-equality is
**not** promised across non-x86_64 architectures: parallel impls are `f64`-only
per ADR-0018, and ARM (NEON), AVX-512, and other SIMD widenings may differ in
the lowest bits of the result depending on how the compiler vectorises
reductions. The `tests/simd_bit_equal.rs` gate runs only on the host
architecture in CI and that is the only architecture where bit-equality is
contractually promised. Cross-architecture results agree to documented
tolerances (`5e-4` sup-error for the heat-equation smoke per ADR-0028, matching
across Wave A FFI, Wave B PyO3, and Wave C WASM at 3-digit precision per Wave
C's recorded `1.460302e-6`).

## Considered alternatives

(a) **No formal freeze; ship 1.0.0 as a snapshot with no stability
commitments.** Fastest path: tag `v1.0.0`, write nothing in
`docs/api-stability.md`, leave callers to read the SemVer spec and infer policy.
**Rejected.** A math crate has no value if downstream code (paper-reproduction
artifacts, research scripts, external bindings, course materials, the Vladikavkaz
2025 reference implementation itself) cannot rely on a named operator producing
the same numerical output across a MINOR bump. The cost of writing this ADR + the
companion policy doc is small; the cost of *not* writing it would be paid for the
entire lifetime of v1.x by every caller who has to grep CHANGELOG to discover
which methods moved. The whole point of v1.0.0 is the commitment.

(b) **Freeze only the Rust API; leave the three binding crates on 0.x
indefinitely.** Possible: keep `semiflow-core` at v1.0.0 frozen, mark
`remizov-{ffi,py,wasm}` as "stable Rust core, experimental bindings".
**Rejected.** ADR-0028 §"API stability" already committed to simultaneous
freeze, and that commitment was the basis for shipping the bindings as
experimental in v0.10.0. Reneging now would create an awkward UX: Python users
see `pip install remizov` returning a 0.x wheel while Rust users on `cargo add
semiflow-core` see `1.x` — the version skew tells nobody anything useful about
which surface to depend on. The bindings are thin wrappers around the frozen
core; freezing the core means the binding signatures are also stable enough to
freeze. Simultaneous freeze, as promised.

(c) **Freeze with `#[non_exhaustive]` on every public enum as an escape valve.**
Marking `SemiflowError`, `SemiflowStatus`, and `Axis` as `#[non_exhaustive]`
would let us add variants in MINOR bumps without breaking exhaustive `match`
arms in caller code (callers would be forced to add a `_ =>` arm, which costs
nothing but discipline). **Rejected for v1.0.0** as default policy — too eager.
The v1.0.0 freeze should commit to a stable enum surface; if a future caller
survey or a v1.x audit reveals a forced enum-extension demand, the
case-by-case escape is to add the variant in a vN.0.0 MAJOR bump (which is the
correct signal — adding a new error class is a real change of caller
expectations). `#[non_exhaustive]` may be adopted **selectively post-v1.0**
when a concrete evolution pressure appears, recorded in a separate ADR per
enum. The default at v1.0.0: closed enums, MAJOR bump for additions.

## Stability matrix

| Surface | Frozen at v1.0.0? | MAJOR-bump trigger | Notes |
|---------|-------------------|--------------------|-------|
| **Rust API** (`semiflow-core` pub items, ~53 items per S2.2 audit) | Yes | Item removal; signature break; trait-bound break; lifetime change | Includes traits (`State`, `ChernoffFunction`, `SemiflowFloat`), structs, enums, free functions; `pub(crate)` items are not part of the surface |
| **FFI ABI** (`semiflow-ffi` `extern "C"` symbols, opaque handle, `SemiflowStatus` repr, header drift) | Yes | Symbol removal/rename; struct layout change; enum repr change; ABI-breaking `cbindgen` output drift | Per ADR-0028 §C ABI; cbindgen header check in CI; `panic = "unwind"` profile preserved |
| **Python API** (`semiflow-py` pyclass/pyfunction surface, `SemiflowError.kind` discriminator, `.pyi` stubs) | Yes | Method removal/rename; signature break; `kind` value rename | abi3-py310 single wheel preserved per ADR-0028 Wave B; PEP 561 `.pyi` stubs (v0.11.0 I7) frozen |
| **WASM API** (`semiflow-wasm` `#[wasm_bindgen]` exports, JS `kind` discriminator on thrown errors) | Yes | Export removal/rename; signature break; `kind` value rename | `panic = "abort"` profile preserved per ADR-0028 Amendment 1; npm package name `@semiflow/wasm` frozen per ADR-0030 |
| **Numerical output** (`f64` deterministic path, fixed HW/target) | Yes (PATCH/MINOR) | Algorithm change that perturbs lowest-bit results; this is a v2.0.0 trigger | SIMD bit-equality gate part of contract; non-x86_64 bit-equality not promised |
| **Performance** (criterion baselines per `benches/`) | No (separate doc) | N/A — perf regressions are CI-blocking but not SemVer-relevant | Tracked separately in `docs/perf-commitment-v1_0_0.md` (S2.3) |
| **Error variants** (`SemiflowError`, `SemiflowStatus`, PyO3 `kind`, WASM `kind`) | Yes (closed enums) | Variant addition, removal, or rename | Per Considered alternatives §c — `#[non_exhaustive]` not adopted at v1.0.0 |
| **Internal storage** (`Storage<F>` enum in `DiffusionChernoff`, parallel reduction order, SIMD intrinsic choice) | No (private) | N/A — `pub(crate)` and below | Implementation may evolve freely so long as the symbolic + semantic stability axes hold |

## Pre-1.0 record (informative)

The pre-1.0 history below is recorded for caller awareness; none of these
events constitute a v1.x commitment.

| Version | Change | SemVer role pre-1.0 |
|---------|--------|---------------------|
| v0.3.0 → v0.4.0 | Magnus K=4 added (ADR-0011) | Additive |
| v0.7.0 | 6th-order spatial (ADR-0015) + non-separable 2D scalar-`c` (ADR-0016) | Additive |
| v0.10.0 | FFI/PyO3/WASM crate split (ADR-0028) | Additive at the user-facing level (new crates) |
| v0.12.0 | `DiffusionChernoff::with_closure` + `Copy` → `Clone` (ADR-0034 §"Suckless audit") | Last pre-1.0 breaking change: cascading `Clone` migration through `Strang2D`, `Strang3D`, `NonSeparable2DChernoff`, `NonSeparable2DAnisotropicChernoff`, `AxisLift` (composition types that hold `DiffusionChernoff` by value lost `Copy` transitively) |

## Migration path

Pre-1.0 callers upgrading to v1.0.0: **the only breaking change since v0.10.0
is the v0.12.0 `Copy` → `Clone` shift on `DiffusionChernoff` and the
composition types that hold it (`Strang2D`, `Strang3D`, `NonSeparable2DChernoff`,
`NonSeparable2DAnisotropicChernoff`, `AxisLift`).** Callers who held any of
these by value and assumed `Copy` semantics — `let dc2 = dc;` followed by use
of `dc` — get a clear `move` error from the compiler at upgrade and replace
with `let dc2 = dc.clone();`. No tooling, no `cargo fix`, no migration script
required; the `clippy` and `rustc` diagnostics are sufficient. All other
v0.10.0+ surface is forward-compatible to v1.0.0 unchanged. Binding callers
(C, Python, JavaScript) see no breaking changes from v0.12.0 → v1.0.0 because
the binding wrappers absorb the Rust-side `Clone` migration internally; the C
ABI, the Python pyclass surface, and the WASM JS surface are all source-stable
across the freeze.

## Out of scope

- **Performance regression bounds.** Criterion baselines and
  regression-blocking thresholds live in `docs/perf-commitment-v1_0_0.md`
  (S2.3). This ADR commits to numerical *output* stability; performance
  stability is a separate, narrower contract handled by the perf commitment
  doc.
- **ABI-level C compatibility across compilers.** The C ABI surface is what
  Rust + cbindgen emit; how a downstream caller's specific C compiler
  (gcc/clang/msvc, plus version) lays out struct fields when cbindgen says
  `#[repr(C)]` is not our promise. This is the ordinary C interop contract;
  consult `cbindgen` and `cc` documentation.
- **SO-versioning policy for `libsemiflow_ffi.so`.** We ship per-release; we
  do not embed an SO-name version field. Downstream Linux packagers
  (Debian, Fedora, Nix) handle SO-versioning per their own conventions; we
  do not promise binary compat across `libsemiflow_ffi.so.1.x` minor bumps
  at the ELF level — only at the source level (rebuilding against v1.x
  headers is the supported upgrade path).
- **Async runtime addition (I14 from ROADMAP).** Per ADR-0034 §"Out of
  scope", an async-callback API is deferred and tracked separately. If
  I14 lands in a v1.x.y MINOR release post-freeze, it follows the standard
  additive-only deprecation cycle: add `Heat1D.with_a_async_function`
  alongside the existing sync constructor, `#[deprecated]` is **not**
  used (the sync constructor remains the correct API for non-async
  callers, mirroring the ADR-0033 NonSeparable2D coexistence precedent).
- **MCP introspection.** Withdrawn per ADR-0027 (no runtime to introspect;
  rustdoc + cargo cover what MCP would have provided). Re-affirmed at
  v1.0.0 freeze: MCP is not part of the v1.x surface and cannot be added
  without a v2.0.0 ADR.
- **`#[non_exhaustive]` adoption per enum.** Reserved for case-by-case
  post-v1.0 ADRs per Considered alternatives §c.

## Suckless audit

This ADR is a freeze, not an addition. **0 new direct dependencies** are
introduced (the dep graph at v1.0.0 = the dep graph at v0.12.x).
**0 ABI-breaking changes** are triggered by the freeze itself (the freeze
ratifies the v0.12.0 surface as stable; the only pre-1.0 break — `Copy` →
`Clone` — already shipped in v0.12.0 per ADR-0034). The ADR has multiple
sections per the v0.11.0+ planning-ADR convention (ADR-0029, ADR-0033,
ADR-0034 all use the same multi-section structure), with each section terse
and bounded; the suckless ADR convention "≤1 paragraph per section" is
honoured per individual decision-bearing section (§Decision is one paragraph
per topic; §Migration path is one paragraph; §Out of scope items are one
sentence each). Total file length stays within the freeze ADR budget; no
narrative bloat.
