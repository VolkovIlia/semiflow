# ADR-0076 — v2→v3 Binding Redesign (FFI/PyO3/WASM)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v3.0 Wave D-F (one wave per binding: D=FFI, E=PyO3, F=WASM). Rides on Wave A (ADR-0073 ApproximationSubspace + ADR-0074 ChernoffFunction trait cleanup) and Wave C (ADR-0075 ζ⁴). The binding redesign is the LAST architectural change before Wave G migration scaffolding + cross-binding parity gate; no further BREAKING surface changes in v3.0.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0028 (v0.10.0 binding strategy — three sibling crates: semiflow-ffi cdylib, semiflow-py PyO3 wheel, semiflow-wasm wasm-bindgen; the [profile.release-ffi] panic=unwind requirement per Wave A v0.10.0; the [profile.release] panic=abort divergence in Wave C wasm per ADR-0028 Amendment 1), ADR-0035 (v1.0.0 API freeze — bindings included; this ADR is the v3.0 RESET), ADR-0059 (graph-binding cross-binding sup-error ≤ 3 ULP gate — precedent for G_binding_parity), ADR-0072 (v2.8 ADR-0072 §"Cross-references" — sibling design pattern reused), ADR-0073 (v3.0 ApproximationSubspace<K, F> — NOT exposed in v3 bindings; see Rationale), ADR-0074 (v3.0 ChernoffFunction trait cleanup — the `growth()` return-type change is what the v3 binding surfaces transparently expose).
- **Supersedes / amends**: ADR-0028 v0.10.0 binding strategy (PARTIAL — preserves the three-sibling-crate structure; redesigns the per-binding API surface to align with the v3.0 cleaned-up Rust trait). ADR-0035 v1.0.0 API freeze (PARTIAL — v3.0 is the v0.x → v1.x → v2.x → v3.x BREAKING cadence's third BREAKING window; the v1.0.0 binding freeze was implicitly time-boxed to "until v3.0.0 BREAKING window").
- **Mathematical foundation**: none — this is a pure binding-surface redesign. The mathematical content is in ADR-0073 / ADR-0074 / ADR-0075 (math §26 + §27).
- **Acceptance gates added**: G_binding_parity (RELEASE_BLOCKING — cross-binding FFI/PyO3/WASM byte-identical to v2.5.1 baseline on the canonical CEV smoke suite; the binding redesign MUST NOT alter the binding-observable numerical output by even 1 ULP). The canonical smoke suite reuses the v2.5.1 HFT-latency-tail baseline per ADR-0067.

## Context

v0.10.0 (ADR-0028) shipped three additive sibling crates as Wave A/B/C:
- **`semiflow-ffi`** (cdylib + opaque handle + `smf_*` extern "C" functions; v0.10.0 Wave A commit c6a4a93)
- **`semiflow-py`** (PyO3 + maturin pyclass wheels; v0.10.0 Wave B commit c3a6fe5)
- **`semiflow-wasm`** (wasm-bindgen JS classes; v0.10.0 Wave C commit 7efd3c3)

v2.0–v2.8 added kernels (graph PDE, adjoint, Magnus6, Schrödinger, ReflectedHeat, Manifold) and graph-binding surfaces (ADR-0059, v2.2 Wave C); the binding surfaces grew organically with each minor release. v3.0's BREAKING ChernoffFunction trait cleanup (ADR-0074) and the new ApproximationSubspace<K, F> marker trait (ADR-0073) raise the question: **how do the v0.10.x → v2.8 binding surfaces survive the v3.0 trait redesign?**

Three constraints frame the answer:

1. **v2.x binding consumers MUST NOT be broken cold.** Per ADR-0035 §9 precedent (the v1.0.0 → v2.0.0 cycle), every BREAKING window provides 12 months of deprecation shim. The v3.0 → v4.0 cycle MUST provide the same.
2. **The v3 binding surfaces MUST expose the v3.0 cleaned-up Rust trait** (the `Growth<F>` return; the absence of Clone bound on `Self::S`; the `Evolver<C, F>` rename; the `apply_into` zero-alloc primacy). Otherwise the binding redesign is purely cosmetic and doesn't deliver the v3.0 cleanup story.
3. **The v3.0 const-generic `ApproximationSubspace<K, F>` (ADR-0073) is NOT yet binding-exposable.** None of the three binding ABIs (C, PyO3, JS) have a clean way to express const-generic traits at the surface (C lacks templates entirely; PyO3 lacks compile-time const-generics on Python classes; WASM/wasm-bindgen lacks the same). Exposing the const-generic K to bindings would require either (a) hardcoding K=2,4,6 as separate symbol-name suffixes (`smf_in_subspace_k2`, `smf_in_subspace_k4`, `smf_in_subspace_k6`) — ugly and combinatorial; or (b) waiting for the Rust → binding-ABI tooling to add const-generic support. Defer to v3.1+ per "out of scope".

The v3.0 binding redesign therefore: **regenerates v3 binding surfaces with `_v3` symbol-name suffix; ships v2 binding surfaces as parallel shims for 12 months.** This is the same pattern used by the GLib / Cairo / OpenSSL 1.0 → 1.1 transitions and by Python's `python` / `python3` parallel CLI invocations — well-trodden in the OSS ecosystem.

## Decision

Ship the v3.0 binding redesign across the three sibling crates with the following per-binding pattern:

### Pattern: parallel v2 + v3 surfaces

For each binding, the cdylib / wheel / WASM package compiles BOTH the v2 surface (verbatim from v2.x) AND the v3 surface (clean) from the SAME Rust source crate. The caller selects which surface to use at the C-include / Python-import / JS-import level, NOT at the cdylib / wheel / WASM-package level. Symbol coexistence is achieved via:

- **FFI**: separate header files `remizov.h` (v3) and `remizov_v2.h` (v2-deprecated). Both headers declare `smf_*` functions from the same cdylib; v3 functions use the bare names (`smf_evolver_new`, `smf_evolver_evolve_into`, `smf_growth_multiplier`); v2 functions use `_v2` suffix (`smf_chernoff_semigroup_new_v2`, `smf_apply_v2`, `smf_growth_multiplier_v2`). Both compile from the same Rust source via two `#[no_mangle]` extern "C" blocks per affected function (one with bare name, one with `_v2` suffix). Caller picks `#include "remizov.h"` (v3) or `#include "remizov_v2.h"` (v2 deprecated, warns).
- **PyO3**: pyclasses keep their canonical names (`Heat1D`, `GraphHeat`, etc.) but expose `evolve_into` as the canonical method (v3) AND `apply` / `evolve` as deprecation-warned shims (v2). The `__module__` attribute version-bumps to `"3.0"`. The wheel exports a single Python package `semiflow` whose `__version__ == "3.0.0"`; callers using the v2 shim methods see `DeprecationWarning` per CPython convention.
- **WASM**: the JS class `RemizovWasm` exposes `version()` returning `"3.0"` (v3) AND keeps the `versionV2()` shim returning `"2.x"` for 12 months. The JS class methods themselves follow the same shim pattern: `evolveInto` is canonical (v3), `evolve` is deprecation-warned (v2). The npm package exports the JS class verbatim; consumers see deprecation messages via `console.warn` in dev builds.

### Specific changes per binding

**FFI (Wave D)** — `crates/semiflow-ffi/`:

| v2.x symbol | v3 symbol | v2 shim symbol | Migration kind |
|---|---|---|---|
| `smf_chernoff_semigroup_new(func, n)` | `smf_evolver_new(func, n)` | `smf_chernoff_semigroup_new_v2(func, n)` | type-rename (ChernoffSemigroup → Evolver) |
| `smf_apply(func, tau, src, dst)` (allocating in v2) | `smf_apply_into(func, tau, src, dst, scratch)` (zero-alloc, v3) | `smf_apply_v2(func, tau, src, dst)` (shim wraps apply_into) | API rewrite (allocating → zero-alloc) |
| `smf_growth_m(func)`, `smf_growth_omega(func)` | `smf_growth_multiplier(func)`, `smf_growth_omega(func)` | `smf_growth_m_v2(func)` (alias) | field-rename (M → multiplier; omega unchanged) |
| `smf_evolve(func, t, src, dst)` | `smf_evolver_evolve_into(evolver, t, src, dst, scratch)` | `smf_evolve_v2(func, t, src, dst)` (shim) | API rewrite |
| Per-kernel `smf_<kernel>_new(...)` (Heat1D, Diffusion4th, GraphHeat, etc.) | unchanged in v3; same signatures | n/a (no v2 shim needed — bare names work for both) | none |

xtask change: `xtask ffi-headers` regenerates BOTH `remizov.h` (v3 surface) AND `remizov_v2.h` (v2 shim surface). Same cdylib; two `cbindgen.toml` files (one per header). Per-binding tests: `xtask ffi-smoke` runs the canonical CEV smoke against BOTH headers (`#include "remizov.h"` and `#include "remizov_v2.h"`); both MUST produce byte-identical results to the v2.5.1 baseline (G_binding_parity gate).

**PyO3 (Wave E)** — `crates/semiflow-py/`:

| v2.x Python API | v3 Python API | v2 shim | Migration kind |
|---|---|---|---|
| `heat = Heat1D(...)` then `psi1 = heat.evolve(t, psi0)` (allocating) | `heat.evolve_into(t, psi0, out)` (zero-alloc preferred); `psi1 = heat.evolve(t, psi0)` continues to work (allocating convenience, NOT deprecated — it's the v3 inherent `apply_chernoff` per-impl convenience exposed to Python) | shim NOT needed (v3 keeps `evolve` as inherent) | none — soft compatibility |
| `m, om = heat.growth()` (tuple return) | `g = heat.growth(); g.multiplier; g.omega` (Growth namedtuple) | `m, om = heat.growth_v2()` (tuple, deprecation-warned) | tuple → namedtuple migration |
| `ChernoffSemigroup(func, n)` | `Evolver(func, n)` | `ChernoffSemigroup` alias (deprecation-warned at construction) | type-rename |
| `semiflow.__version__ == "0.10.0"` (or v2.x) | `semiflow.__version__ == "3.0.0"` | n/a | version metadata |

The PyO3 binding has the easiest migration path because Python's runtime introspection lets us alias-and-warn cleanly. The `growth()` namedtuple is `collections.namedtuple('Growth', ['multiplier', 'omega'])` — destructuring `m, om = heat.growth()` continues to work via Python's tuple-iteration on namedtuples. The deprecation is over the *field-access* style: callers using `heat.growth()[0]` keep working (positional access on namedtuple); callers using `heat.growth().multiplier` are the v3-recommended path.

Per-binding tests: `xtask py-smoke` runs the canonical CEV smoke against the v3 API; `xtask py-smoke --v2-shim` runs against the v2 shim methods. Both MUST produce byte-identical results to the v2.5.1 baseline AND byte-identical results to the FFI v3 surface (cross-binding parity).

**WASM (Wave F)** — `crates/semiflow-wasm/`:

| v2.x JS API | v3 JS API | v2 shim | Migration kind |
|---|---|---|---|
| `import {Heat1D, RemizovWasm} from '@semiflow/wasm'` | unchanged in v3; same imports | n/a | none |
| `RemizovWasm.version() === "2.x"` | `RemizovWasm.version() === "3.0"` | `RemizovWasm.versionV2() === "2.x"` | version metadata |
| `heat.evolve(t, psi0)` returning new Float64Array (allocating) | `heat.evolveInto(t, psi0, out)` zero-alloc preferred; `heat.evolve(t, psi0)` continues to work (allocating convenience) | shim NOT needed | none — soft compatibility |
| `heat.growth() === [M, omega]` (Array return) | `heat.growth() === {multiplier, omega}` (object return) | `heat.growthV2() === [M, omega]` (Array, deprecation-warned via console.warn) | array → object migration |
| `ChernoffSemigroup` JS class | `Evolver` JS class | `ChernoffSemigroup` alias (console.warn on construction) | type-rename |

Per-binding tests: `xtask wasm-smoke` runs the canonical CEV smoke (via wasm-bindgen-test in Node + headless Chrome) against the v3 API; `xtask wasm-smoke --v2-shim` runs against the v2 shim methods. Both MUST produce byte-identical results to the v2.5.1 baseline AND to the FFI/PyO3 v3 surfaces.

### Per-binding profile choices (carry-over from v0.10.0)

- **FFI**: `[profile.release-ffi]` with `panic=unwind` (v0.10.0 Wave A; ADR-0028 Wave A requirement for `catch_unwind` on every extern "C"). UNCHANGED in v3.0.
- **PyO3**: reuses `[profile.release-ffi]` (v0.10.0 Wave B). UNCHANGED.
- **WASM**: `[profile.release]` with `panic=abort` (v0.10.0 Wave C; ADR-0028 Amendment 1 — intentional divergence from FFI/PyO3 because `catch_unwind` is meaningless in WASM and `panic=abort` produces smaller binaries). UNCHANGED.

### G_binding_parity acceptance gate (RELEASE_BLOCKING)

The cross-binding parity gate verifies that the v3 binding redesign does NOT alter the binding-observable numerical output. Specifically:

- **Canonical smoke suite**: 1M CEV pricing ticks per ADR-0067 (the v2.5.1 HFT-latency-tail baseline). Seed `PCG64(0xC0FFEE_BABE_DEAD_BEEF)`; N=1536; n=200; Diffusion4thChernoff kernel.
- **Baseline reference**: the v2.5.1 commit `dabf5a1` baseline output captured as a fixed Float64Array of 1M entries with SHA-256 `beac36ebe1c541bff4641debe170cb00b35aedb0e893ac8603470d98508b2863` (per `properties.yaml` `L_CEV_PTICK.canonical_input.sha256`). Stored in `crates/semiflow-core/examples/data/cev_baseline_v2_5_1.bin`.
- **Per-binding output**: the v3 binding surfaces (FFI v3 `smf_*`, PyO3 v3 `Heat1D.evolve_into`, WASM v3 `Heat1D.evolveInto`) MUST each produce a 1M Float64Array byte-identical (`memcmp == 0`) to the v2.5.1 baseline.
- **Cross-binding match**: the three binding outputs MUST also be byte-identical to each other (FFI v3 == PyO3 v3 == WASM v3 — pairwise `memcmp == 0`).
- **The v2 shim surfaces** (FFI `_v2`, PyO3 `growth_v2`, WASM `growthV2`) MUST ALSO produce byte-identical output to the v2.5.1 baseline AND to the v3 surfaces — the shim is a pass-through, NOT a re-implementation.

Test files: `tests/binding_parity_ffi_v3.rs`, `tests/binding_parity_py_v3.rs` (via PyO3 test harness), `tests/binding_parity_wasm_v3.rs` (via wasm-bindgen-test on Node). All three MUST pass; any failure BLOCKS v3.0 release.

The G_binding_parity gate is the ADR-0059 graph-binding ≤3-ULP gate's strict tightening (ULP=0; byte-identical). Justified because the v3 binding redesign is structurally pure pass-through to the same Rust core; no numerical algorithm change means zero ULP error is achievable.

### File layout

- `crates/semiflow-ffi/src/lib.rs` — v3 surface (bare-name `smf_*` functions); ~600 LoC target (current 510 LoC + ~90 LoC for the 4 v3-named entries that wrap the cleaned-up trait).
- `crates/semiflow-ffi/src/v2_shim.rs` — NEW file, v2 surface (`_v2`-suffixed `smf_*` functions); ~300 LoC; default 500-LoC cap. `#[cfg(feature = "v2_compat")]` gated (default ON in v3.x, OFF in v4.0).
- `crates/semiflow-ffi/include/remizov.h` — generated by `xtask ffi-headers` from the v3 cbindgen config.
- `crates/semiflow-ffi/include/remizov_v2.h` — NEW, generated by `xtask ffi-headers --v2-shim` from a separate cbindgen config (different namespace prefix).
- `crates/semiflow-py/src/lib.rs` — v3 + v2 shim methods inline (pyo3-deprecated decorator on v2 methods); ~700 LoC target (current ~450 LoC + ~250 LoC for v3 + v2 shim coverage).
- `crates/semiflow-wasm/src/lib.rs` — v3 + v2 shim methods inline; ~500 LoC target (current ~350 LoC + ~150 LoC for v3 + v2 shim coverage).

Schema bumps: shared with ADR-0073 / ADR-0074 / ADR-0075 — `traits.yaml` 0.8.0 → 1.0.0, `properties.yaml` 0.10.0 → 0.11.0. math.md is append-only (§26, §27 NEW for ADR-0073 + ADR-0075; no §28 for this ADR — pure binding-surface redesign, no math content).

## Rationale

- **Why parallel v2 + v3 surfaces (not a single migrated v3 surface)?** Per ADR-0035 §9 precedent, every BREAKING window provides 12 months of shim. The parallel-surface pattern is the lowest-friction shim: v2.x consumers `#include "remizov_v2.h"` (FFI), call `heat.growth_v2()` (PyO3), or call `RemizovWasm.versionV2()` (WASM) and get the v2.x semantics verbatim with a deprecation warning. The v3 consumers `#include "remizov.h"` and use bare names. No CI breakage at v3.0 upgrade; 12 months to migrate; hard removal at v4.0.
- **Why `_v3` symbol-name suffix for FFI (not `_v2` suffix and bare names for v3)?** Two alternatives: (a) bare names for v3 + `_v2` suffix for shim (chosen); (b) `_v3` suffix for v3 + bare names preserved for shim (rejected). Choice (a) is suckless: the v3 surface is the FUTURE long-term surface; bare names should belong to it. The v2 shim is the TRANSITIONAL surface; suffix makes the deprecation visible at every call site. After v4.0 shim removal, the bare-name surface (= v3) is what remains — no rename needed at v4.0. Choice (b) would force a rename at v4.0 (bare-name v2 shim removed, v3 bare names move in) — extra churn. Choice (a) is the v0.10.0 → v0.11.0 npm precedent (Wave B precedent: `@semiflow/wasm` v3.x bare names are the long-term).
- **Why separate `remizov.h` and `remizov_v2.h` header files (not a single header with `#ifdef REMIZOV_V2_SHIM`)?** Two reasons: (a) separate headers make the deprecation MORE visible — the act of changing `#include` declares the migration intent at the C build-script level (CMake, Makefile, Meson); (b) cbindgen can generate two cleanly-separated headers from two cbindgen.toml configs, avoiding error-prone `#ifdef` regions in generated files. The two-header pattern matches the OpenSSL 1.0 → 1.1 transition (separate header `openssl/opensslv.h` + `openssl/opensslv_legacy.h`).
- **Why PyO3 keeps `Heat1D.evolve()` working WITHOUT deprecation (unlike FFI's hard-renamed `smf_apply` → `smf_apply_v2`)?** Python's runtime introspection makes the `evolve()` method intrinsically forgiving: callers using `heat.evolve(t, psi0)` get the allocating convenience (the v3 inherent `apply_chernoff` exposed under the v2-familiar name `evolve`). The PyO3 binding can ALIAS without semantic change because the v3 inherent `apply_chernoff` is BEHAVIOURALLY identical to the v2.x `apply` (both allocate; both return a fresh state). The deprecation is over `evolve_v2` (the explicit v2-compatibility name) and `growth_v2` (the tuple-return shim). Bare `evolve` continues without warning because it works without modification.
- **Why explicitly NOT expose `ApproximationSubspace<K, F>` in v3 bindings?** Three reasons: (a) C lacks templates entirely — would require hardcoded `smf_in_subspace_k2`, `smf_in_subspace_k4`, `smf_in_subspace_k6` ABI symbols (ugly combinatorial; doesn't generalise to user-defined K); (b) PyO3 lacks compile-time const-generics on Python classes — `Heat1D.in_subspace::<4>(f)` is not expressible in Python; the alternative `Heat1D.in_subspace(4, f)` loses the compile-time order verification that motivates the const-generic in the first place; (c) wasm-bindgen has the same const-generic gap as PyO3. The trait is Rust-only in v3.0; v3.1+ may add binding exposure once the Rust → binding-ABI tooling supports const-generic traits (current state: not yet). Suckless: don't ship a degraded binding surface that loses the trait's design intent.
- **Why does the v3 binding redesign get its own ADR (rather than rolling into ADR-0074)?** ADR-0074 is the Rust-trait cleanup; the binding redesign is the C/Python/JS-surface consequence of that cleanup. Each binding has distinct constraints (FFI panic=unwind requirement; PyO3 namedtuple ergonomics; WASM panic=abort + smaller-binary discipline) that warrant a focused ADR. The two ADRs are tightly coupled but separately ownable: Wave D-F engineering is a different work surface from Wave A.
- **Why a byte-identical (0 ULP) parity gate (not the v2.2 ADR-0059 ≤3 ULP gate)?** The v3 binding redesign is structurally PURE pass-through to the same Rust core. The Rust core's numerical algorithms are UNCHANGED at v3.0 (ADR-0073/0074 are surface-only; ADR-0075 is a NEW kernel, not a modification of existing kernels). Therefore the v3 binding surfaces calling the v2.x baseline kernels (Heat1D, Diffusion4th, GraphHeat, etc.) MUST produce numerically-identical output to the v2.x bindings on the same kernels. Zero ULP is achievable; setting the gate at ≤3 ULP would silently tolerate a regression. The 0-ULP gate matches the v0.10.0 Wave A cross-binding sup_error == 0 baseline (FFI vs PyO3 Float64Array equality per Wave B commit c3a6fe5).
- **Why a 12-month deprecation timeline (not 6 months or 18 months)?** ADR-0035 §9 precedent: the v1.0.0 → v2.0.0 cycle gave 12 months. The cadence stays at 12 months unless the engineering reality changes. 6 months is too aggressive for downstream consumers (especially academic users on conference-driven release cycles); 18 months blocks v4.0 from landing on the v3.0 + 12mo schedule per `roadmap-reflective-biscuit.md`. 12 months is the suckless cadence choice.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Single v3 surface (no v2 shim) at v3.0 | Breaks every v2.x binding consumer cold; violates ADR-0035 §9 deprecation cadence. The v2.x consumers include the existing PyPI / npm / vcpkg ecosystem; a hard break would force same-day migration. |
| Defer the v3 binding redesign to v3.1+ (ship v3.0 with v2.x binding surfaces unchanged) | The v3.0 BREAKING window is the only opportunity to redesign before v4.0; v3.1 is a MINOR release that can't add breaking changes. Defer-to-v3.1 means defer-to-v4.0 — another 12 months of v2.x-binding-surface lag. |
| Single header `remizov.h` with `#ifdef REMIZOV_V2_SHIM` regions | Generated header would have hand-maintained `#ifdef` regions — error-prone for cbindgen; clutters the v3-canonical header file. Two separate headers (per cbindgen-config) is the suckless choice. |
| `_v3` suffix for v3 functions (preserve bare names for v2 shim) | Forces a rename at v4.0 (bare-names move from v2 shim to v3 surface). The bare-names-for-v3 choice avoids the v4.0 rename. |
| Expose `ApproximationSubspace<K, F>` via hardcoded `smf_in_subspace_k2/k4/k6` ABI symbols | Combinatorial explosion at v3.1 (Magnus order-6 K=6, A3 Hörmander K=2, etc.); doesn't generalise to user-defined K. Pull the surface out entirely until binding-ABI tooling supports const-generic traits. |
| ABI version field in the v3 header (e.g., `SMF_ABI_VERSION == 3.0` constant) | Adds a runtime version-check burden on every caller; conflicts with cdylib soname versioning (the cdylib already declares soname v3 → v4 as a SemVer break). The header-name choice (`semiflow.h` vs `semiflow_v2.h`) is the version-check mechanism. |
| Strict ≤3 ULP G_binding_parity gate (not byte-identical) | Tolerates silent numerical regression; the v3 binding redesign is pure pass-through, so 0 ULP is achievable. Stricter is suckless. |
| 6-month deprecation timeline (faster shim sunset) | Insufficient for academic downstream consumers (conference-cycle migrations take 6+ months). ADR-0035 §9 12-month cadence stays. |
| 18-month deprecation timeline (gentler sunset) | Blocks v4.0 from landing at v3.0 + 12 months per `roadmap-reflective-biscuit.md` schedule. 12 months is the project-established cadence. |
| Ship a v3.0-rc.1 with v2 surfaces hard-removed (force-test downstream migration) | Inverts the deprecation cadence; v3.0-rc.1 is a tag, not a deprecation milestone. Per ADR-0035 §9 the rc.1 IS the early-access for v3.0 (with shim); hard removal is v4.0 only. |
| Skip the canonical CEV smoke suite for G_binding_parity (use a different oracle) | The v2.5.1 baseline `dabf5a1` is the project's existing reproducible reference (sha256 captured per `L_CEV_PTICK.canonical_input`); reusing it is the suckless choice. Any new oracle would require its own baseline capture + 5-rep replication discipline. |

## Consequences

- **BREAKING changes at v3.0** (mitigated by v2 shim with 12-month deprecation):
  - FFI callers using `smf_apply`, `smf_chernoff_semigroup_new`, `smf_growth_m` MUST switch to `#include "remizov_v2.h"` (shim header, deprecation warnings on every call) OR rewrite to the v3 surface (`smf_apply_into`, `smf_evolver_new`, `smf_growth_multiplier`).
  - PyO3 callers using `m, om = heat.growth()` (tuple destructure) keep working via namedtuple-iteration; callers using `heat.growth()[0]` keep working (positional access); callers using `heat.growth().M` MUST migrate to `.multiplier` (no shim).
  - PyO3 callers using `ChernoffSemigroup` keep compiling with deprecation warning on construction; migrate to `Evolver` within 12 months.
  - WASM callers using `RemizovWasm.version()` get `"3.0"` (was `"2.x"` in v2); callers using `heat.growth()` get `{multiplier, omega}` object (was `[M, omega]` array) — MUST migrate via `growthV2()` shim.
- **NEW files**:
  - `crates/semiflow-ffi/src/v2_shim.rs` (~300 LoC, default 500-LoC cap, `#[cfg(feature = "v2_compat")]` gated, deleted at v4.0).
  - `crates/semiflow-ffi/include/remizov_v2.h` (cbindgen-generated, not hand-maintained).
- **MODIFIED files**:
  - `crates/semiflow-ffi/src/lib.rs` (~510 → ~600 LoC; +90 LoC for v3 bare-name surface; remains within the 700-LoC carve-out per constitution v1.6.1 Override #1 Cohort 4 — `ffi.rs` is already in the list).
  - `crates/semiflow-py/src/lib.rs` (~450 → ~700 LoC; +250 LoC for v3 surface + v2 shim methods + deprecation decorators).
  - `crates/semiflow-wasm/src/lib.rs` (~350 → ~500 LoC; +150 LoC for v3 surface + v2 shim methods + console.warn).
- **Dependency count unchanged** at 2/3 budget for `semiflow-core` (still `num-traits`, `libm`). Per-binding crates have their own deps (PyO3, wasm-bindgen, etc.) per ADR-0028; v3.0 does NOT change them.
- **Schema bumps**: shared with ADR-0073 / ADR-0074 / ADR-0075 (`traits.yaml` 1.0.0; `properties.yaml` 0.11.0). math.md is append-only (§26, §27 NEW for ADR-0073 + ADR-0075; no §28 — this ADR is binding-surface only).
- **New gate**: G_binding_parity (RELEASE_BLOCKING — 1M CEV smoke, byte-identical FFI v3 == PyO3 v3 == WASM v3 == v2.5.1 baseline; also verifies the v2 shims produce byte-identical output to the v3 surfaces). Per-binding test files: `tests/binding_parity_ffi_v3.rs`, `tests/binding_parity_py_v3.rs`, `tests/binding_parity_wasm_v3.rs`.
- **xtask changes** (Wave G engineering surface):
  - `xtask ffi-headers` regenerates BOTH `remizov.h` (v3) AND `remizov_v2.h` (v2 shim). Two cbindgen.toml configs.
  - `xtask ffi-smoke` runs against BOTH v3 + v2 shim surfaces.
  - `xtask py-smoke --v2-shim` adds the v2-shim test invocation.
  - `xtask wasm-smoke --v2-shim` adds the v2-shim test invocation.
- **CI changes** (Wave G):
  - Three new jobs per binding (`ffi-build-v2-shim`, `py-build-v2-shim`, `wasm-build-v2-shim`); each exercises the v2 shim feature flag in addition to the default v3-only build.
  - One new job `cross-binding-parity` running G_binding_parity across all three bindings.
- **12-month deprecation timeline**:
  - **v3.0.0** (release): v2 shims active by default (feature `v2_compat` = on); deprecation warnings on v2 callers.
  - **v3.1.0 – v3.x.y** (12 months): shims continue; engineer escalates warnings per ADR-0035 §9 pattern at minor releases.
  - **v4.0.0** (12 months after v3.0): shim files (`v2_shim.rs`, `remizov_v2.h`, PyO3 `*_v2` methods, WASM `*V2` methods) DELETED; `#[cfg(feature = "v2_compat")]` removed; hard removal complete.

## Migration

End-user impact per binding (full coverage in `docs/migration/v2-to-v3.md` §7-§10 — Wave G):

**FFI consumers (`#include "remizov.h"` in C/C++)**:

```c
// v2.x — works in v3.0 if you switch to the v2 shim header:
#include "remizov_v2.h"  // deprecation warning at include time
RemizovChernoffSemigroup *cs = smf_chernoff_semigroup_new_v2(func, 200);
double *psi1 = smf_evolve_v2(cs, 0.5, psi0, /* allocates */);
// CALL THE V2 SHIM EXPLICITLY — symbol names have _v2 suffix.

// v3.0 RECOMMENDED — switch include + rewrite to zero-alloc:
#include "remizov.h"
RemizovEvolver *ev = smf_evolver_new(func, 200);
double *psi1_buf = malloc(N * sizeof(double));      // pre-allocated
RemizovScratchPool *pool = smf_scratch_pool_new();
smf_evolver_evolve_into(ev, 0.5, psi0, psi1_buf, pool);
```

**PyO3 consumers (`import semiflow` in Python)**:

```python
# v2.x — works in v3.0 with namedtuple-tuple iteration:
heat = Heat1D(...)
psi1 = heat.evolve(0.5, psi0)              # OK — same as v2 (allocating; not deprecated)
m, omega = heat.growth()                    # OK — namedtuple destructures like tuple
# M field-rename:
multiplier = heat.growth()[0]               # OK — positional access on namedtuple
# multiplier = heat.growth().M             # ERROR — no .M; use .multiplier instead

# v3.0 RECOMMENDED — named fields + zero-alloc:
psi1_buf = np.zeros_like(psi0)
heat.evolve_into(0.5, psi0, psi1_buf)       # zero-alloc
g = heat.growth()
multiplier, omega = g.multiplier, g.omega   # named-field access (preferred)
```

**WASM consumers (`import {Heat1D, RemizovWasm} from '@semiflow/wasm'` in JS)**:

```javascript
// v2.x — works in v3.0 with growthV2() shim:
const heat = new Heat1D(...);
const psi1 = heat.evolve(0.5, psi0);              // OK (allocating; not deprecated)
const [M, omega] = heat.growthV2();                // deprecation warning via console.warn
// V2 STYLE: heat.growth() now returns {multiplier, omega} object, NOT [M, omega] array.

// v3.0 RECOMMENDED — object + zero-alloc:
const psi1Buf = new Float64Array(psi0.length);
heat.evolveInto(0.5, psi0, psi1Buf);              // zero-alloc
const {multiplier, omega} = heat.growth();         // object destructure
```

Full migration playbook with per-binding worked examples: `docs/migration/v2-to-v3.md` §7 (FFI), §8 (PyO3), §9 (WASM), §10 (cross-binding parity verification).

## Cross-references

- ADR-0001 — contract-first; this ADR adds new binding contracts before any Rust impl ships.
- ADR-0028 — v0.10.0 binding strategy; PARTIALLY supersedes; preserves the three-sibling-crate structure.
- ADR-0035 — v1.0.0 API freeze; this ADR is the v3.0 binding-RESET (first BREAKING since v2.0). 12-month deprecation cadence per §9.
- ADR-0059 — graph-binding cross-binding sup-error ≤3 ULP gate; G_binding_parity tightens to 0 ULP for the CEV smoke (achievable due to pure pass-through).
- ADR-0072 — sibling design pattern (v2 vs v3 binding surfaces as siblings, not subtypes).
- ADR-0073 — v3.0 ApproximationSubspace<K, F>; NOT exposed in v3 bindings (deferred to v3.1+ per "Out of scope").
- ADR-0074 — v3.0 ChernoffFunction trait cleanup; the cleanup is what the v3 binding surfaces transparently expose.
- ADR-0075 — v3.0 ζ⁴ correction; the `Diffusion4thZeta4Chernoff<F>` kernel is NOT exposed in v3 bindings for v3.0 (the K=6 ApproximationSubspace witness is Rust-only; binding exposure deferred to v3.1+).
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v3.0 — release-level roadmap (binding redesign Wave D-F placement).
- `docs/migration/v2-to-v3.md` (NEW v3.0 — Wave G; engineer fills worked examples per binding).
- `.dev-docs/constitution.md` v1.7.0 (NEW v3.0).
- `properties.yaml` G_binding_parity (NEW v3.0).
- `traits.yaml` schema 1.0.0 (MAJOR per Override re-eval).

## Amendments

(none at acceptance time)
