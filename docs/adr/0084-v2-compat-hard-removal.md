# ADR-0084 — `v2_compat` HARD REMOVAL + Binding Cleanup (Wave G)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v4.0 Wave G (final Wave of the second BREAKING window; ships AFTER all kernel additions of Waves A-E). The hard-removal Wave is the canonical 12-month deprecation cycle closure per ADR-0035 §9 precedent.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0035 §9 (the v0.10.0 → v0.11.0 → v1.0.0 12-month deprecation cycle precedent — the pattern this ADR replays at v3.0 → v4.0 cadence), ADR-0074 (v3.0 ChernoffFunction trait cleanup — the source of the v2_compat shim), ADR-0076 (v3.0 v2→v3 binding redesign — the source of the per-binding v2 shim layers). Sibling to ADR-0079/0080/0081/0082/0083 in v4.0; this ADR is the ENGINEERING TAIL of the second BREAKING window.
- **Supersedes / amends**: ADR-0074 §"12-month deprecation timeline" (FULFILLED — the v3.0.0 deprecation start + v4.0.0 hard removal cadence is honoured); ADR-0076 §"12-month deprecation timeline" (FULFILLED — the FFI / PyO3 / WASM v2 shim layers are removed in lockstep with the core); ADR-0035 §9 (the canonical 12-month cycle precedent — this ADR is a faithful replay).
- **Mathematical foundation**: math.md §35 (NORMATIVE library — exhaustive removal manifest; CITATION: ADR-0035 §9 precedent). math §35 is BOOKKEEPING — no new mathematical content; the v2_compat removal is a pure code-management operation.
- **Acceptance gates added**: None (NO new gate). G_binding_parity is MODIFIED (REDUCED from 6 sub-tests to 3) per ADR-0079/0080/0081/0082/0083 schema bump documentation. The reduction is gated by `cargo build --no-default-features` succeeding on every workspace crate (no v2_compat feature needed) + G_binding_parity sub-tests 1 + 3 + 5 passing at v3-only surface.

## Context

Per the 12-month deprecation cycle established by ADR-0035 §9 (the v0.10.0 → v0.11.0 → v1.0.0 precedent) and reaffirmed at v3.0 by ADR-0074 §"Decision" + ADR-0076 §"Decision", the v2.x compatibility shim shipped with v3.0 is HARD-REMOVED at v4.0.

The 12-month deprecation window:
- **v3.0.0** (release 2026-05-27): all v2.x surface marked `#[deprecated(since = "3.0.0", note = "Hard-removed at v4.0")]`. v2.x callers compile with warnings.
- **v3.x point releases** (2026-08-27, 2026-11-27, etc.): warning escalation per ADR-0035 §9 pattern.
- **v4.0-rc.1** (~2027-05-13): final notice; rustdoc front-matter explicitly states removal date.
- **v4.0.0** (~2027-05-27): hard removal complete.

This ADR is the EXHAUSTIVE REMOVAL MANIFEST. Every v2 surface artifact that shipped in v3.x is deleted at v4.0. The math.md §35 §35.2-§35.5 manifest tables list each artifact with kind + removal rationale; this ADR's `Decision` references those tables and adds the engineer Wave G responsibilities + verification protocol.

## Decision

Execute the v4.0 v2_compat hard removal per math.md §35 manifest. Specifically:

**Section A — Core crate removals (math §35.2)**:
- DELETE `crates/semiflow-core/src/v2_compat.rs` (~120 LoC).
- REMOVE `[features] v2_compat = []` from `crates/semiflow-core/Cargo.toml`.
- REMOVE `default = ["v2_compat"]` from `crates/semiflow-core/Cargo.toml`.
- REMOVE `#[cfg(feature = "v2_compat")] pub use v2_compat::*;` from `crates/semiflow-core/src/lib.rs`.
- REMOVE all `#[deprecated]` markers tied to the v2 surface.

Net effect: `cargo build --no-default-features` succeeds on `semiflow-core` (was already a valid build path in v3.x; in v4.0 the feature is removed so the build is unaffected).

**Section B — FFI removals (math §35.3)**:
- DELETE `crates/semiflow-ffi/src/v2_shim.rs` (~300 LoC).
- DELETE `crates/semiflow-ffi/include/remizov_v2.h` (cbindgen-generated).
- DELETE `crates/semiflow-ffi/cbindgen-v2.toml`.
- REMOVE all `_v2`-suffixed extern "C" symbols (e.g., `smf_chernoff_semigroup_new_v2`, `smf_apply_v2`, `smf_evolve_v2`, `smf_growth_m_v2`).
- REMOVE `xtask ffi-headers --v2-shim` flag from `xtask/src/main.rs`.
- REMOVE `xtask ffi-smoke --v2-shim` flag.
- REMOVE CI job `ffi-build-v2-shim` from `.github/workflows/ci.yml`.

The v3 FFI surface (`crates/semiflow-ffi/include/remizov.h`, `smf_evolver_new`, `smf_apply_into`, etc.) is PRESERVED verbatim — only the `_v2` shim layer is removed.

**Section C — PyO3 removals (math §35.4)**:
- REMOVE v2-shim methods from `crates/semiflow-py/src/lib.rs` (e.g., `Heat1D.evolve_v2`, `Heat1D.growth_v2`).
- REMOVE `ChernoffSemigroup` Python class alias.
- REMOVE `semiflow.__version__` v2 string compatibility shim.
- REMOVE `xtask py-smoke --v2-shim` flag.
- REMOVE CI job `py-build-v2-shim`.

**Section D — WASM removals (math §35.5)**:
- REMOVE v2-shim methods from `crates/semiflow-wasm/src/lib.rs` (e.g., `Heat1D.growthV2()`).
- REMOVE `ChernoffSemigroup` JS class alias.
- REMOVE `RemizovWasm.versionV2()` method.
- REMOVE `xtask wasm-smoke --v2-shim` flag.
- REMOVE CI job `wasm-build-v2-shim`.

**Section E — Migration playbook update**:
- UPDATE `docs/migration/v3-to-v4.md` per §"v2_compat removal checklist" with the actual v4.0.0 release SHA.

**Section F — Acceptance verification**:
- Run `cargo build --no-default-features` on every workspace crate; MUST succeed.
- Run G_binding_parity sub-tests 1, 3, 5 (FFI v3 / PyO3 v3 / WASM v3) at v3-only surface; MUST pass byte-identically.
- Run cross-binding match (FFI v3 == PyO3 v3 == WASM v3 pairwise memcmp); MUST pass.
- Run `cargo test --workspace --features slow-tests`; MUST pass (the v2_compat removal SHOULD have zero impact on non-shim tests since v2 surface is parallel to v3).

The full engineer Wave G responsibilities are documented in math.md §35.7.

## Rationale

- **Why hard-remove (vs continue deprecation through v5.0)**: ADR-0035 §9 established the 12-month cycle; honouring it at v4.0 is the project's commitment to predictable BREAKING windows. Continuing deprecation through v5.0 would extend to 24-36 months — far longer than any Rust ecosystem standard (most crates use 6-12 month deprecation; the project chose 12 to be gentle). Indefinite deprecation creates trait-surface bloat indefinitely.
- **Why delete the v2_compat files entirely** (vs leave them as dead code with `#[cfg(feature = "v2_compat")]`): "dead code" is a maintenance burden — every future refactor must verify the dead code still compiles; every reviewer must mentally distinguish live from dead. Deletion is the suckless choice: gone is better than dormant.
- **Why remove the `default = ["v2_compat"]` Cargo.toml line entirely** (vs change default to `[]`): the feature itself is removed; the default-list cannot reference a non-existent feature. The line must go.
- **Why preserve the v3 binding surfaces verbatim** (vs also clean those up at v4.0): the v3 binding surfaces are the STABLE BASELINE for v4.x — the same role the v2 surface played in v2.x. Touching them at v4.0 would defeat the purpose of the BREAKING window (which is to drop the OLD surface, not redesign the new one). The v3 surfaces will be the long-term post-v4.0 surface.
- **Why G_binding_parity reduces from 6 → 3 sub-tests** (vs keep 6 sub-tests with the v2 sub-tests failing): the v2 shim test files are DELETED at v4.0 (they cannot exist if the v2 surface is gone). G_binding_parity goes from 6 sub-tests covering 6 paths (FFI v3 + FFI v2 + PyO3 v3 + PyO3 v2 + WASM v3 + WASM v2) to 3 sub-tests covering 3 paths (FFI v3 + PyO3 v3 + WASM v3). The cross-binding match check (FFI v3 == PyO3 v3 == WASM v3) is preserved.
- **Why all 4 sections (Core, FFI, PyO3, WASM) executed in a SINGLE Wave G** (vs per-binding waves): the v2_compat surface is INTERDEPENDENT — removing it from the core crate would orphan the binding shims (they reference core types that no longer exist). All 4 sections must land together to maintain workspace coherence; engineer Wave G is the single atomic unit.
- **Why update `docs/migration/v3-to-v4.md` with the v4.0.0 release SHA**: the migration playbook is the long-term reference for downstream users; recording the actual hard-removal SHA makes future audits trivial.
- **Why verify via `cargo build --no-default-features`** (in addition to G_binding_parity): the `--no-default-features` build path was the v3.x escape hatch for users who wanted to skip the v2_compat shim early. At v4.0 the feature itself is gone, so the build behaves identically to the default build. Verifying both paths catches any accidental dependency on the `v2_compat` feature.
- **Why 12 months (not 6 or 18)**: ADR-0035 §9 settled on 12 as the project's cadence. 6 is too aggressive for academic users on conference-driven release cycles; 18 blocks v4.0 from landing on the planned schedule. 12 is the suckless choice.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Continue deprecation through v5.0+ (postpone hard removal) | Violates ADR-0035 §9 cadence; creates indefinite trait-surface bloat; inconsistent with the project's commitment to predictable BREAKING windows. |
| Hard-remove ONLY the core v2_compat (leave per-binding shims) | The per-binding shims reference the core v2_compat surface; without the core, the shims are orphaned/broken. All-or-nothing is the only coherent path. |
| Soft-deprecate the v2_compat at v4.0 (escalate to error in v4.x) | The deprecation cycle is the SOFT phase; v4.0 is the HARD phase by ADR-0035 §9 precedent. Adding a second soft phase doubles the timeline. |
| Keep the v2_compat behind a `[features] v2_compat_legacy` feature flag for v4.x users who haven't migrated | Re-creates the deprecation maintenance burden; users would still need to migrate eventually; the v3.0 → v4.0 12 months was sufficient warning. |
| Defer the hard removal to v4.1 (ship v4.0 with v2_compat intact) | Loses the cleanup story of the BREAKING window; misses the natural cadence; v4.0 is the only feasible point. |
| Skip the G_binding_parity reduction (keep 6 sub-tests, just `#[ignore]` the v2 ones) | Tagging tests as `#[ignore]` is silent failure; better to remove them entirely so the gate reflects actual coverage. |
| Skip the docs/migration/v3-to-v4.md update | The migration playbook is the long-term reference; not updating it would silently miss the canonical removal record. |
| Run the v4.0 hard removal as multiple smaller PRs across v3.x → v4.0 cycle | The v3.x cycle is the SOFT phase (deprecation only); incremental removal in v3.x would surprise users. v4.0 is the SINGLE atomic removal. |

## Consequences

- **BREAKING for any v2.x callers who have not migrated to v3.0+ API by v4.0 release.** Per ADR-0035 §9 deprecation cycle, the 12 months of warnings is the migration window; v4.0 hard removal is the project's commitment.
- **Files DELETED** (per math.md §35.2-§35.5 manifest): `crates/semiflow-core/src/v2_compat.rs` (~120 LoC), `crates/semiflow-ffi/src/v2_shim.rs` (~300 LoC), `crates/semiflow-ffi/include/remizov_v2.h`, `crates/semiflow-ffi/cbindgen-v2.toml`. Total: ~420 LoC of removable code + 2 generated files.
- **Files MODIFIED**: `crates/semiflow-core/Cargo.toml` (-3 lines), `crates/semiflow-core/src/lib.rs` (-1 line + re-export removal), `crates/semiflow-ffi/src/lib.rs` (-~90 LoC), `crates/semiflow-py/src/lib.rs` (-~120 LoC), `crates/semiflow-wasm/src/lib.rs` (-~80 LoC), `xtask/src/main.rs` (-~30 LoC for the `--v2-shim` flag handling), `.github/workflows/ci.yml` (-3 CI jobs).
- **Schema bumps**: shared with ADR-0079/0080/0081/0082/0083/0085 — `traits.yaml` 1.1.0 → **2.0.0 MAJOR**; `properties.yaml` 0.12.0 → **1.0.0 MAJOR**. The v2_compat module marker pseudo-entry in traits.yaml is DELETED.
- **G_binding_parity REDUCED** from 6 sub-tests to 3 sub-tests (FFI v3 + PyO3 v3 + WASM v3 only; the v2 shim sub-tests are removed because the v2 shim layer is gone). Cross-binding match check preserved.
- **Dependency count unchanged** at 3/3 (no dep removals; num-complex + num-traits + libm preserved per ADR-0079).
- **CI changes**: 3 jobs removed (ffi-build-v2-shim, py-build-v2-shim, wasm-build-v2-shim). Build time decrease ~5-10%.
- **Documentation update**: `docs/migration/v3-to-v4.md` final version (Wave G) gets the actual v4.0.0 release SHA in the "Migration window log" §35.6.
- **Total code DELETION**: ~600 LoC across core + FFI + PyO3 + WASM + xtask + CI configs. Net workspace LoC decreases at v4.0 (rare event — most releases add LoC).

## Migration

End-user impact:

- **v2.x callers using v3.x shim (`feature v2_compat`)**: HARD BREAK at v4.0. MUST migrate to v3.0+ API per `docs/migration/v2-to-v3.md` (the v2 → v3 playbook) before upgrading to v4.0. The 12-month deprecation warnings (v3.0 onwards) gave fair notice.
- **v3.0+ native callers (using cleaned-up trait surface)**: ZERO impact. The v3 trait surface + v3 binding surfaces are preserved verbatim through v4.0+.
- **CI consumers**: 3 fewer CI jobs (ffi-build-v2-shim, py-build-v2-shim, wasm-build-v2-shim); minor build time reduction.

Full migration playbook with worked examples per binding: `docs/migration/v3-to-v4.md` (architect scaffold + engineer Wave G fills).

The v3 → v4 migration playbook is INTENTIONALLY THIN — most v3+ users have ZERO migration work because v4.0 adds new surface (Schrödinger Option B, PointEval, d-D shift, matrix-valued) and removes only the v2_compat shim. The bulk of the playbook is the "v2 → v3 → v4 deferred migration" worked examples for users who skipped v3.0 entirely.

## Cross-references

- ADR-0001 — contract-first.
- ADR-0035 §9 — the 12-month deprecation cycle precedent that this ADR replays.
- ADR-0074 — v3.0 ChernoffFunction trait cleanup + v2_compat shim source (the ADR being closed by hard removal here).
- ADR-0076 — v3.0 v2→v3 binding redesign + per-binding shim sources (the ADR being closed by hard removal here).
- ADR-0079/0080/0081/0082/0083 — sibling ADRs in v4.0 BREAKING window; this ADR is the ENGINEERING TAIL.
- math.md §35 (NEW v4.0) — exhaustive removal manifest + engineer Wave G responsibilities + migration window log.
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v4.0 — release-level roadmap (v2_compat hard removal Wave G placement).
- `.dev-docs/constitution.md` v1.8.0 (NEW v4.0) — MAJOR re-evaluation; v2_compat removal recorded.
- `docs/migration/v3-to-v4.md` — migration playbook scaffold (architect) + Wave G fills (engineer).

## Amendments

(none at acceptance time)
