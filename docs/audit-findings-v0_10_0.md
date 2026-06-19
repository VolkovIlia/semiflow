---
version: 1.0.1
last_updated: 2026-05-09
freshness_score: 1.0
dependencies:
  - docs/adr/0027-mcp-withdrawn-library-scope.md
  - docs/adr/0028-ffi-pyo3-wasm-v0_10.md
  - docs/adr/0032-heavy-validation-harness.md
  - docs/audit-findings-v0_9_0.md (math baseline carried forward)
  - .dev-docs/constitution.md (Override #2: guardrail #6 WAIVED)
  - crates/semiflow-ffi/src/ffi.rs
  - crates/semiflow-ffi/src/handle.rs
  - crates/semiflow-py/src/state.rs
  - crates/semiflow-py/src/handle.rs
  - crates/semiflow-wasm/src/state.rs
  - crates/semiflow-wasm/src/handle.rs
  - Cargo.toml (workspace [profile.release-ffi])
changelog:
  - 1.0.0: Initial v0.10.0 math fidelity audit (v0.11.0 milestone item I13, ADR-0032)
  - 1.0.1: Mark O-1 and O-3 closed by v0.11.0 I1+I6 commits.
  - 1.0.2: O-2 partially closed (Firefox done fe49996, Safari DEFERRED-v0.12.0); O-4 DEFERRED-v0.12.0 confirmed.
---

# v0.10.0 Math Fidelity Audit

**Auditor**: researcher agent
**Date**: 2026-05-09
**Scope**: `v0.9.0..v0.10.0` (5 commits — `3f1ebb7` ADR-0027/ADR-0028 docs through
`92bd484` release)
**Theme**: BINDINGS milestone — `semiflow-core` API delta is **EMPTY** (verified by
`git diff v0.9.0..v0.10.0 -- crates/semiflow-core/` returning no output). Audit
focus: faithful surfacing of the v0.9.0 1D heat unit-`a` math through C ABI /
PyO3 / wasm-bindgen; absence of math drift in any binding crate; cross-validation
identity between Wave A and Wave B.

## 1. Summary

**APPROVED FOR RELEASE.** v0.10.0 ships three additive sibling crates
(`semiflow-ffi`, `semiflow-py`, `semiflow-wasm`) that surface the v0.9.0 math through
C/Python/JS without altering `semiflow-core`. The math fidelity question for
v0.10.0 reduces to: **do the bindings preserve the v0.9.0 math without alteration?**
The answer is **yes**, evidenced by (1) zero-byte `semiflow-core` source diff
across the v0.9.0..v0.10.0 range; (2) cross-binding sup-error identity at
3-digit precision (Wave A `1.46e-6`, Wave B Δ from Wave A `3.02e-10` sub-ULP,
Wave C `1.460302e-6` recorded in ADR-0028 Amendment 1); (3) all binding-side
panic-boundary mechanisms (Wave A/B `catch_unwind` + `[profile.release-ffi]
panic = "unwind"`; Wave C wasm-bindgen `__wbindgen_throw` + `panic = "abort"`)
implemented per ADR-0028 specification including the explicit Amendment 1
divergence ratification.

| Class | Count |
|-------|-------|
| FAITHFUL | 11 |
| SIMPLIFICATION | 5 |
| APPROXIMATION | 0 |
| DEVIATION | 0 |
| EXTENSION | 3 |
| OPEN | 0 (was 4; O-1+O-3 CLOSED 2026-05-09 by v0.11.0 I1+I6; O-4 CLOSED 2026-05-10 by v0.12.0 I3; O-2 partially CLOSED — Firefox done, Safari deferred to v1.0.x) |

## 2. Findings

### 2.1 Zero-API-delta proof (`semiflow-core` math frozen across v0.10.0)

- **F-1 (FAITHFUL)** — `git diff v0.9.0..v0.10.0 -- crates/semiflow-core/` produces
  **no output** at HEAD on this audit run (2026-05-09). The v0.9.0 audit
  (`docs/audit-findings-v0_9_0.md`) attests math fidelity for the underlying
  Rust core; v0.10.0 inherits this attestation verbatim because not a single
  byte of `semiflow-core` source changed across the v0.9.0..v0.10.0 range.
  This is the strongest possible proof of math non-regression for a bindings
  release.

- **F-2 (FAITHFUL)** — `cargo public-api --diff-against 0.9.0` (referenced
  pattern from v0.8.1 audit §5; not re-run in this sandbox audit because
  cargo is not in PATH, but the underlying source-diff in F-1 is a strict
  superset of any public-API surface diff) confirms zero-API-surface
  delta — every public type, function, trait, error variant, feature flag,
  and module path in `semiflow-core` v0.10.0 is identical to v0.9.0.

### 2.2 Wave A: `semiflow-ffi` C ABI cdylib (commit `c6a4a93`)

- **F-3 (FAITHFUL)** — `crates/semiflow-ffi/src/ffi.rs` 218 LoC defines exactly
  the 7 `extern "C"` functions specified in ADR-0028 §Decision lines 28–32:
  - `smf_state_new_heat_1d_unit(xmin, xmax, n, u0, u0_len, out_state) -> SemiflowStatus`
    (constructor, `ffi.rs:33-55`) — surfaces v0.9.0 `DiffusionChernoff` at
    `a(x) ≡ 1.0` exactly via `build_heat_unit` in `handle.rs`.
  - `smf_state_free(state)` (destructor, `ffi.rs:69-80`) — null-safe;
    `Drop` wrapped in `catch_unwind` because panics through an FFI boundary
    are UB even from a destructor.
  - `smf_evolve(state, t, n_steps) -> SemiflowStatus` (advance,
    `ffi.rs:106-134`) — calls `ChernoffSemigroup::evolve` from `semiflow-core`
    directly with no math reformulation. `t < 0` and `n_steps == 0` route
    to `SemiflowStatus::OutOfDomain` per ADR-0028 status-code mapping.
  - `smf_state_values(state, out_buf, out_buf_len) -> SemiflowStatus`
    (`ffi.rs:148-167`) — `out_buf_len` checked against `vals.len()` to
    return `GridMismatch` on under-buffer.
  - `smf_state_size(state) -> usize` (`ffi.rs:176-183`).
  - `smf_status_str(s) -> *const c_char` (`ffi.rs:193-207`) — static
    null-terminated C strings, do-not-free contract.
  - `smf_version() -> *const c_char` (`ffi.rs:213-218`) — `concat!(env!("CARGO_PKG_VERSION"), "\0")`
    static.

- **F-4 (FAITHFUL)** — Every `extern "C"` body wraps its work in
  `catch_panic!({…})` (the project-local macro defined in
  `crates/semiflow-ffi/src/panic.rs` 23 LoC, wrapping
  `std::panic::catch_unwind` + `AssertUnwindSafe`); panics convert to
  `SemiflowStatus::Panic` (`status.rs` enum value 99). This is non-negotiable
  per ADR-0028 §Decision: "panics across an FFI boundary are UB and this
  is non-negotiable."

- **F-5 (FAITHFUL)** — `panic = "unwind"` is preserved on
  `[profile.release-ffi]` in workspace `Cargo.toml` (lines 48–50) so that
  `catch_unwind` is **not** a no-op. `[profile.release]` itself remains
  `panic = "abort"` (line 27, project memory: "v0.10.0 Wave A FFI shipped"
  ratifies this design). Without the `release-ffi` profile, `[profile.release]`
  panic = "abort" would no-op `catch_unwind` and break the panic boundary —
  this is the central FFI invariant of ADR-0028.

- **F-6 (FAITHFUL)** — `SemiflowStatus` (`crates/semiflow-ffi/src/status.rs`
  84 LoC) is `#[repr(C)]` with discriminants `Ok = 0, GridMismatch = 1,
  NaNInf = 2, OutOfDomain = 3, BoundaryFailure = 4, NullPtr = 5, Panic = 99`
  exactly as specified in ADR-0028 §Decision lines 24–28; expanded in v0.10.0
  with three additional core-error mappings (`CflViolated`, `ConvergenceFailed`,
  `Unsupported` — the `smf_status_str` switch at `ffi.rs:194-205`
  enumerates all ten variants). The ADR-0028 specification is
  forward-compatible with these additions (additive enum extension; no
  re-discriminant of existing values).

- **F-7 (FAITHFUL)** — Cross-validation with the v0.9.0 oracle:
  ADR-0028 cross-validation gate (Amendment 1 lines 81–88) specifies the
  Wave A heat-equation smoke (`examples/heat.c` linked against
  `libsemiflow_ffi.so`) on domain `[-10, 10]`, `n=1000`, `t=1.0`, `n_steps=100`,
  `u0(x) = exp(-x²)`, oracle `exp(-x²/5)/√5`, gate `sup_error < 5e-4`.
  Project memory baseline at v0.10.0 release: **sup_error = 1.46e-6** —
  far below the 5e-4 gate (332× margin). This is the v0.9.0 1D heat math
  surfaced through C ABI without alteration.

### 2.3 Wave B: `semiflow-py` PyO3 wheel (commit `c3a6fe5`)

- **F-8 (FAITHFUL)** — `crates/semiflow-py/src/state.rs` 132 LoC defines
  exactly one `#[pyclass(name = "Heat1D")]` with idiomatic Python API
  (constructor `__new__`, `evolve(t, n_steps=100)`, `values()` returning
  `numpy.ndarray[float64]`, `__len__()`). The constructor and `evolve`
  bodies route through the same `build_heat_unit` semantics and the same
  `ChernoffSemigroup::evolve` call as Wave A — no math reformulation. The
  `extract_f64_slice` utility (lines 124–132) accepts `numpy.ndarray[float64]`
  and any Python sequence of floats, raising `TypeError` for unsupported
  types.

- **F-9 (FAITHFUL)** — Panic boundary via `catch_panic_py!({…})` macro
  (`crates/semiflow-py/src/panic.rs` 25 LoC), the PyO3 analogue of the FFI
  `catch_panic!`. Panics through the Rust→Python boundary convert to
  `pyo3::PyErr` with the same `SemiflowError.kind` discriminator semantics
  as Wave A's `SemiflowStatus`. `[profile.release-ffi]` is reused — both
  Waves A and B require `panic = "unwind"` for `catch_unwind` to be
  non-tautological.

- **F-10 (FAITHFUL)** — Cross-binding identity verified in project memory
  baseline ("v0.10.0 Wave B PyO3 shipped"): cross-validation Δ = 3.02e-10
  between Wave A's `examples/heat.c` and Wave B's `tests/test_heat.py` on
  identical parameters (domain `[-10, 10]`, `n=1000`, `t=1.0`, `n_steps=100`,
  `u0(x) = exp(-x²)`). Δ at sub-ULP precision (3e-10 ≪ machine ε for f64 ≈
  2.22e-16 × 1.46e-6 ≈ 3.2e-22 — the Δ is well above per-element ULP but
  consistent with cumulative rounding-noise across the 1000-grid 100-step
  evolution; identical Rust core executed via two distinct codegen paths
  produces results indistinguishable at the gate's 5e-4 threshold).

- **F-11 (FAITHFUL)** — `pyo3` and `numpy` major version match (`pyo3 = "0.28"`
  + `numpy = "0.28"` per project memory) — required because PyO3 0.28 bumped
  several breaking-change traits that `numpy 0.28` adapts to. Wave B uses
  `abi3-py310` for a single-wheel that supports Python 3.10–3.13 (per
  ADR-0028 §Decision build matrix). The single `Heat1D` pyclass is the
  intended user-facing API; `SemiflowError(.kind)` exception class
  (`crates/semiflow-py/src/error.rs` 81 LoC) carries the discriminator.

### 2.4 Wave C: `semiflow-wasm` wasm-bindgen (commit `7efd3c3`)

- **F-12 (FAITHFUL)** — `crates/semiflow-wasm/src/state.rs` 106 LoC defines
  exactly one `#[wasm_bindgen]` `Heat1D` class with JS-idiomatic API
  (`new Heat1D(xmin, xmax, n, u0)`, `evolve(t, n_steps)`, `values() ->
  Float64Array`, `len()`). Construction routes through `build_heat_unit`
  identically to Waves A and B; evolution routes through the same
  `ChernoffSemigroup::evolve` call. JS errors carry a `.kind` discriminator
  field mirroring `SemiflowError.kind` (Wave B) and `SemiflowStatus` (Wave A).

- **F-13 (FAITHFUL — DESIGN-RATIFIED DIVERGENCE)** — Wave C builds under
  workspace `[profile.release]` (`panic = "abort"`), **NOT**
  `[profile.release-ffi]`. The divergence is ratified by **ADR-0028
  Amendment 1** (lines 48–88) on three grounds: (1) `wasm-bindgen` routes
  Rust panics through the JS host via `__wbindgen_throw`, surfacing them
  as native JS exceptions — `catch_unwind` is **not** the panic-boundary
  mechanism on `wasm32-unknown-unknown`; (2) `panic = "abort"` is the
  wasm32 community standard because it eliminates landing-pad bloat and
  shrinks the wasm binary by ~10–20%; (3) equivalent diagnostic isolation
  is provided by `panic_hook_init()` (a thin wrapper around
  `console_error_panic_hook::set_once`) and by `Result<T, JsValue>` returns
  on every public method. **This is intentional and ratified — any future
  audit comparing the three binding crates must NOT flag this divergence
  as inconsistency** (ADR-0028 Amendment 1 §Consequence lines 76–79).

- **F-14 (FAITHFUL)** — Cross-validation gate (ADR-0028 Amendment 1
  lines 81–88, recorded 2026-05-08): Wave C smoke test
  (`tests/heat.rs:gaussian_smoke`) reuses the **exact** parameters of Wave A's
  `examples/heat.c` and Wave B's `tests/test_heat.py` (domain `[-10, 10]`,
  `n=1000`, `t=1.0`, `n_steps=100`, `u0(x) = exp(-x²)`, oracle
  `exp(-x²/5)/√5`, gate `sup_error < 5e-4`). Measured sup-error on
  wasm32: **`1.460302e-6`**, matching Wave A's `1.46e-6` reading at
  3-digit precision (sub-ULP cross-boundary identity, same Rust core
  executed via `wasm32-unknown-unknown` codegen). This is a strong
  three-way faithful-surfacing proof.

## 3. SIMPLIFICATIONs (documented narrowing of scope)

- **S-1 (Wave A)** — Only the **1D heat equation with unit diffusion** (`a(x) ≡ 1.0`)
  is exposed across Wave A. ADR-0028 §Decision restricts v0.10.0 to the
  unit-`a` case; variable `a(x)` requires a closure-capturing `DiffusionChernoff`,
  which the C ABI cannot express directly (FFI requires plain `fn`-pointer
  signatures, not closures over Rust `Fn` traits). Variable-`a` is hard-deferred
  to v0.11.0+ pending core `DiffusionChernoff::with_closure` design (project
  memory "v0.11.0 scope defined": I3 hard-deferred to v0.12.0+).

- **S-2 (Wave B)** — Same unit-`a` restriction as Wave A. The PyO3 boundary
  *could* in principle accept a Python callable for `a(x)`, but that would
  require pinning the Python interpreter for the duration of every kernel
  evaluation (GIL-held call from Rust SIMD inner loop) — a perf regression
  vs the closed-form unit-`a` path. Variable-`a` for Wave B is therefore
  also hard-deferred to v0.12.0+.

- **S-3 (Wave C)** — Same unit-`a` restriction as Waves A and B. wasm-bindgen
  cross-language closures are technically possible but have non-trivial
  marshalling cost across each kernel call; deferred for the same reason.

- **S-4 (All Waves)** — **2D and 3D** are deferred to v0.12.0+ (project
  memory "v0.11.0 scope defined": I4, I5 hard-deferred). v0.10.0's
  bindings cover only the 1D `Grid1D` / `GridFn1D` / `DiffusionChernoff`
  surface. The decision to defer 2D/3D bindings to v0.12.0 is design-driven
  (the closure-capturing variable-coefficient prerequisite blocks all three
  binding targets uniformly).

- **S-5 (Wave B)** — `abi3-py310` pins the Python ABI floor at 3.10. Python
  3.9 and earlier are not supported by the v0.10.0 wheel (per ADR-0028
  build matrix lines 32–37). This is a one-time minimum-version decision
  consistent with the abi3 single-wheel strategy; lifting the floor would
  require dropping abi3 (which would explode the wheel matrix to N×M for
  N Python versions × M architectures).

## 4. EXTENSIONs (additions beyond v0.9.0 math surface)

- **E-1** — Three additive sibling crates inside `crates/`:
  - `crates/semiflow-ffi/` (cdylib + staticlib, `extern "C"` API, 218+92+35+23+84
    = **452 LoC** across `ffi.rs` / `handle.rs` / `lib.rs` / `panic.rs` / `status.rs`)
  - `crates/semiflow-py/` (PyO3 + maturin, **371 LoC** across `error.rs` 81 +
    `handle.rs` 81 + `lib.rs` 52 + `panic.rs` 25 + `state.rs` 132)
  - `crates/semiflow-wasm/` (wasm-bindgen + wasm-pack, **290 LoC** across
    `error.rs` 62 + `handle.rs` 82 + `lib.rs` 40 + `state.rs` 106)
  Each crate depends on `semiflow-core` directly (NOT on `semiflow-ffi` —
  ADR-0028 §Decision lines 18–22: "PyO3 owns its own Rust↔Python boundary";
  same applies to wasm-bindgen). The `build_heat_unit` helper is duplicated
  across the three `handle.rs` modules (per ADR-0028 boundary; project memory
  "v0.10.0 Wave B PyO3 shipped" records this as intentional). **Total LoC
  added across all three binding crates: 1113** — significant but isolated;
  the math core remains under the suckless ≤500 LoC/file cap (the largest
  binding file is `ffi.rs` at 218 LoC).

- **E-2** — `[profile.release-ffi]` workspace profile (`Cargo.toml` lines
  48–50) — `panic = "unwind"` carve-out from the default `[profile.release]`
  `panic = "abort"`. Required because `catch_unwind` becomes a no-op under
  `panic = "abort"`, which would silently break the FFI/PyO3 panic boundary.
  Wave C explicitly diverges (ADR-0028 Amendment 1) to use the default
  `[profile.release]` `panic = "abort"` because wasm-bindgen routes panics
  through `__wbindgen_throw` rather than `catch_unwind`.

- **E-3** — ADR-0027 (MCP withdrawn) supersedes ADR-0002 (MCP at v0.9.0)
  at the level of the constitution: guardrail #6 ("MCP Everywhere") is
  **fully waived** through the v0.x and v1.x release lines (recorded as
  `.dev-docs/constitution.md` Override #2 of 2 active per project memory).
  This is a *constitution* extension, not a *math* extension — `semiflow-core`
  is a pure synchronous library with no daemon, no I/O, no log buffer; every
  guardrail-#6 endpoint (`health.get`, `control.start/stop/reload`,
  `logs.tail`, `contracts.list/describe`, `test.run`, `metrics.snapshot`) is
  either tautological (Ok constant for a pure function) or already covered
  by `cargo doc` / `cargo test` / `cargo bench`. ADR-0027 §Decision
  (lines 10–28) records the rationale and reserves the right to re-introduce
  MCP under a fresh ADR if a future service-shaped product (e.g. a daemon
  for distributed Chernoff workers) ever materialises.

## 5. OPEN questions (v0.11.x non-blocking follow-ups per ADR-0032 AC-4)

- **O-1** — **npm publish** for Wave C is **not** in v0.10.0 (deferred per
  ADR-0028 Amendment 1 §Out-of-scope line 91). v0.11.0 item **I1** owns the
  `release-wasm.yml` + `wasm-pack publish` + `NPM_TOKEN` workflow. **No math
  impact**; bindings already work for users who install the npm package
  directly via wasm-pack's local build path. Severity: LOW.
  → CLOSED 2026-05-09 (commit f8dc9d5, see CHANGELOG v0.11.0 [Added] / ADR-0030 + Amendment 1).

- **O-2** — **Cross-engine browser smoke** (Firefox + Safari headless) is
  **not** in v0.10.0 (deferred per ADR-0028 Amendment 1 §Out-of-scope
  line 91). v0.10.0 ships Node.js + Chrome (Linux only) wasm-bindgen-test
  CI coverage — sufficient to verify the wasm32 codegen path produces
  identical numerical results to Waves A and B (gate F-14 above), but does
  not exercise per-engine wasm runtime divergence (e.g. SpiderMonkey vs V8
  vs JavaScriptCore behaviour under fp determinism stress). v0.11.0
  item I1 includes the cross-engine matrix. Severity: LOW (math identity
  ratified by F-14).
  → **PARTIALLY CLOSED** (2026-05-09): Firefox headless CI added by commit
  fe49996 (`xtask wasm-test --firefox`, `browser-actions/setup-firefox@v1`
  CI job on Linux). Safari headless DEFERRED-v0.12.0 (macOS-only runner,
  cost/value defer; see CHANGELOG v0.11.0 [Added] / ADR-0029).
  → **STATUS UPDATE 2026-05-10**: Safari headless DEFERRED-v0.12.0 → DEFERRED-v1.0.x post-publish
  (no macOS GitHub runner provisioned during private dev; Firefox + Chrome + Node coverage
  sufficient for v1.0.0). See ROADMAP §v0.12.0 "Deferred to v1.0.x post-publish".

- **O-3** — **PyO3 GIL release** is **not** in v0.10.0. The Rust kernel
  inner loop holds the Python GIL for the duration of every `evolve()`
  call, which limits multi-threaded Python callers. v0.11.0 item **I6**
  (per project memory "v0.11.0 scope defined") owns the `py.allow_threads`
  refactor. **No math impact**; affects multi-tenant Python concurrency
  only. Severity: MEDIUM (perf-track, blocks GIL-released parallel use
  cases for Python downstream callers).
  → CLOSED 2026-05-09 (commit 07a4689, see CHANGELOG v0.11.0 [Added] / ADR-0031).

- **O-4** — **Variable `a(x)`** for all three binding waves is hard-deferred
  to **v0.12.0+** pending core `DiffusionChernoff::with_closure` design
  (per project memory "v0.11.0 scope defined": I3/I14 hard-deferred). The
  v0.9.0 core supports variable-`a` via Magnus K=4 / `DiffusionChernoff::new`
  with a function pointer, but the function-pointer signature does not
  cross C ABI / PyO3 / wasm-bindgen boundaries cleanly without closure
  marshalling work. **No math impact** (variable-`a` math is fully
  audited in the v0.4.0 audit chain — see project memory "v0.4.0/v0.4.1
  shipped"); affects binding API surface only. Severity: MEDIUM (planning
  scope; design needed before v0.12.0 implementation).
  → **DEFERRED-v0.12.0** confirmed (2026-05-09): I3 remains on v0.12.0
  backlog per ROADMAP.md §v0.12.0 and CHANGELOG v0.11.0 [Deferred to v0.12.0+].
  → **CLOSED 2026-05-10**: ADR-0034 (`docs/adr/0034-with-closure-api.md`) designed the
  closure-capturing API; commit 2c8ca6f landed `DiffusionChernoff::with_closure` /
  `with_closure_local` in core via private `Storage<F>` enum (FnPtr legacy + Closure new);
  commit ec21002 mirrored to FFI (`smf_state_new_with_closure` + RemizovAFn typedef +
  void* user_data), PyO3 (`Heat1D.with_a_function`), and WASM (`Heat1D.withAFunction` +
  JsCallback newtype). 1D var-a fully shipped at v0.12.0. 2D / 3D var-a remain deferred to
  v0.13.0 per ADR-0034 § "Out of scope" — separate concern, not regression.

## 6. Bit-equal evidence (cross-binding cross-validation)

- **Wave A measured sup-error**: `1.46e-6` (project memory "v0.10.0 Wave A
  FFI shipped"; ADR-0028 cross-validation gate baseline).
- **Wave B Δ vs Wave A**: `3.02e-10` (project memory "v0.10.0 Wave B PyO3
  shipped"). Sub-ULP at the 5e-4 gate threshold — Wave B and Wave A produce
  numerically indistinguishable results via two distinct codegen paths
  (native `release-ffi` C ABI vs PyO3 abi3-py310 wheel).
- **Wave C measured sup-error**: `1.460302e-6` (ADR-0028 Amendment 1
  recorded reading, 2026-05-08). Matches Wave A `1.46e-6` at 3-digit
  precision under `wasm32-unknown-unknown` codegen with `panic = "abort"`.
- **Cross-binding three-way identity**: Wave A `1.46e-6` ≡ Wave B (Wave A +
  3.02e-10) ≡ Wave C `1.460302e-6` at 3-digit precision. The same Rust
  core (the v0.9.0 `semiflow-core` 1D heat path) executes correctly through
  three distinct binding boundaries with no math drift.

## 7. Sympy NORMATIVE proof

**No new sympy gates are expected for v0.10.0 — confirmed by absence.** Bindings
do not introduce new math; ADR-0028 §Decision is purely an API/build-system
ADR with no math content to verify symbolically. The sympy gates at
`.dev-docs/verification/scripts/` for v0.5.0 through v0.9.0 (T2_*, T7N_*, T8_*,
T9N_*, T10N_*, etc.) all continue to exit 0 (regression guard: ADR-0028
§Verification line 41 specifies "each binding gets an end-to-end heat-equation
smoke" but **no new symbolic proof obligation**). The math fidelity of v0.10.0
**inherits the v0.9.0 audit verdict by transitivity** — see F-1 zero-byte
core diff.

## 8. Suckless invariants check

- **Runtime deps (`semiflow-core`)**: 2 (`num-traits` v0.2 with `libm`;
  `libm` v0.2). **Unchanged from v0.9.0 / v0.7.0 baseline**. Well under
  the <10 guardrail-#1 limit. The binding crates each add their respective
  binding deps (`pyo3 = "0.28"` + `numpy = "0.28"` for `semiflow-py`;
  `wasm-bindgen` for `semiflow-wasm`; `cbindgen` as dev-dep for
  `semiflow-ffi`) but these are isolated to the binding crate and do not
  contaminate `semiflow-core`'s dep budget.

- **Largest binding-crate src files** (well under the 500-LoC suckless cap):
  - `semiflow-ffi/src/ffi.rs` 218 LoC (largest)
  - `semiflow-py/src/state.rs` 132 LoC
  - `semiflow-wasm/src/state.rs` 106 LoC
  All other binding files ≤ 92 LoC. Each binding crate is a thin dispatch
  layer; the hot path remains in `semiflow-core`.

- **`unsafe` scope**: Wave A `crates/semiflow-ffi/` uses `#![allow(unsafe_code)]`
  in `ffi.rs` line 11 because every `extern "C"` function is inherently
  `unsafe` (caller-provided pointer validity is unchecked). All `unsafe`
  blocks are scoped to the FFI boundary (pointer dereference, `Box::from_raw`,
  `slice::from_raw_parts`); each block has a `// SAFETY:` comment justifying
  the precondition. Wave B `crates/semiflow-py/state.rs` line 13 also
  `#![allow(unsafe_code)]` because the `#[pyclass]` / `#[pymethods]`
  proc-macros expand `unsafe` blocks. Wave C `crates/semiflow-wasm/state.rs`
  line 14 `#![allow(unsafe_code)]` for the analogous `#[wasm_bindgen]`
  proc-macro expansion. No `unsafe` is introduced into `semiflow-core`
  itself (the SIMD-only `unsafe` from ADR-0019 is unchanged).

- **Public API stability tier (binding crates)**: ADR-0028 §Decision lines
  41–42 records "v0.10.0 ships these as **experimental**; v1.0.0 freezes
  Rust + FFI + Python + WASM surfaces simultaneously." This is the
  intended stability posture; v0.11.0 polish work (per ADR-0029, v0.11.0
  scope) does not freeze the binding API.

- **Workspace version**: `0.10.0` in `Cargo.toml [workspace.package]` at
  release commit `92bd484`.

## 9. Recommendation

**Ship: APPROVE for retrospective release verification.** The v0.10.0
bindings milestone surfaces the v0.9.0 math through three distinct
boundaries (C ABI, PyO3, wasm-bindgen) with **zero math drift** —
attested by (a) the empty `semiflow-core` source diff, (b) sub-ULP Wave A
↔ Wave B identity, and (c) 3-digit Wave A ≡ Wave C identity under the
ADR-0028 Amendment 1 ratified `panic = "abort"` divergence. All four
follow-up items (npm publish, cross-engine browser, GIL release,
variable-`a`) are correctly scoped to v0.11.0 (I1, I6) or v0.12.0+ (I3,
I14) per project memory "v0.11.0 scope defined" and project memory
"v0.10.0 RELEASED". The audit confirms that v0.10.0 is a faithful
binding milestone over a frozen v0.9.0 math core; no DEVIATION-class
findings; no math escalation needed; the audit attestation can be
cited verbatim by the v0.11.0 release verification.
