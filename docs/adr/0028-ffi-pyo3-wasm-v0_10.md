# ADR-0028 — FFI / PyO3 / WASM crate split for v0.10.0

**Status**: Accepted (planning ADR for v0.10.0)
**Date**: 2026-05-08
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0001 (contract-first for math libraries), ADR-0003 (`no_std + alloc`),
ADR-0011 (`SemiflowError` enum), ADR-0027 (MCP withdrawn — FFI bindings inherit
no-runtime semantics), ADR-0017 (perf baseline lint, applies to FFI smokes)

## Decision

v0.10.0 is the **bindings milestone**: real-user reach for a math `rlib` flows
through FFI, not through more Rust crates. We add three additive sibling crates
inside `crates/`, each pinned to a single binding target, leaving `semiflow-core`'s
`no_std + alloc` core and ≤3-direct-dep budget untouched:

- **`semiflow-ffi`** — `cdylib + staticlib`, `extern "C"` API, depends on `semiflow-core`
  only (0 runtime deps; `cbindgen` as dev-dep for header generation).
- **`semiflow-py`** — PyO3 + `maturin` build, depends on `semiflow-core` directly
  (NOT through `semiflow-ffi` — PyO3 owns its own Rust↔Python boundary); `pyo3` is the only direct dep.
- **`semiflow-wasm`** — `wasm-bindgen` + `wasm-pack`, depends on `semiflow-core`
  with `std` feature on (`wasm-bindgen` is `no_std`-incompatible); `wasm-bindgen` is the only direct dep.

The C ABI ships an opaque-handle pattern: `pub struct SemiflowState { _private: [u8; 0] }`
returned as `*mut SemiflowState`, freed by `smf_state_free`; the status code enum
is `#[repr(C)] pub enum SemiflowStatus { Ok = 0, GridMismatch = 1, NaNInf = 2,
OutOfDomain = 3, BoundaryFailure = 4, NullPtr = 5, Panic = 99 }` mapping
`SemiflowError`. Core entry points are sketched as `smf_state_new(grid_*) -> *mut
SemiflowState`, `smf_state_free(*mut SemiflowState)`, `smf_evolve(state*, t,
n, out_state**) -> SemiflowStatus` — exact signatures finalised by the engineer per
workspace member. **Panic boundary**: every `extern "C"` function wraps its body
in `std::panic::catch_unwind` and returns `SemiflowStatus::Panic` on unwind —
panics across an FFI boundary are UB and this is non-negotiable. **CI build
matrix**: `semiflow-ffi` builds for `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`,
`x86_64-pc-windows-msvc` (cargo + cbindgen); `semiflow-py` ships wheels via
`cibuildwheel` for `manylinux_2_28`, `macos-13`/`macos-14`, `windows-msvc`, Python
3.10–3.13; `semiflow-wasm` builds for `wasm32-unknown-unknown` via `wasm-pack
build --target web` and `--target nodejs`. **Capability boundary**: FFI is
"trusted caller" — no JWT, no scopes; the `rlib` already enforces invariants via
`SemiflowError`, and adding token machinery would duplicate `Result` semantics
across a sync function call (security-model.md is not extended). **API stability**:
v0.10.0 ships these as **experimental**; v1.0.0 freezes Rust + FFI + Python +
WASM surfaces simultaneously. **Heavy validation**: each binding gets an
end-to-end heat-equation smoke (`tests/test_heat.py` for Python, an
`examples/heat.c` linked against `libsemiflow_ffi.so`, and a `wasm-bindgen-test`
for WASM) gated in CI before tag.

## Amendment 1 — Wave C profile divergence (2026-05-08)

**Status**: Accepted (records WASM-specific deviation from Waves A/B)
**Context**: Wave C (`semiflow-wasm`) ships under v0.10.0 alongside Waves A and B.

Waves A and B pin `[profile.release-ffi]` (`panic = "unwind"`) because both
wrap entry-points (`extern "C"` for FFI, `#[pyfunction]` for PyO3) in
`std::panic::catch_unwind`. With workspace `[profile.release]` (`panic = "abort"`)
`catch_unwind` becomes a no-op, breaking the panic boundary.

**Wave C is different.** `wasm-bindgen` routes Rust panics through the JS
host via the generated `__wbindgen_throw` shim, surfacing them as native
JavaScript exceptions in the calling code. `catch_unwind` is **not** the
panic-boundary mechanism on `wasm32-unknown-unknown` — `Result<T, JsValue>`
returns are. Furthermore, `panic = "abort"` is the wasm32 community standard
because it eliminates landing-pad bloat and shrinks the wasm binary by
~10-20%.

**Decision**: `semiflow-wasm` builds under the workspace `[profile.release]`
(panic=abort), NOT `[profile.release-ffi]`. The crate does NOT export a
`wasm_catch_panic` macro. Equivalent diagnostic isolation is provided by:

1. `panic_hook_init()` (a thin wrapper around `console_error_panic_hook::set_once`)
   exposed to JS callers — improves panic diagnostics in development without
   adding runtime overhead in production.
2. All public methods return `Result<T, JsValue>` so domain errors surface
   as JS errors with a `kind` discriminator field (mirrors `SemiflowError.kind`
   in Wave B).

**Consequence**: any future audit comparing the three binding crates must
not flag this divergence as inconsistency — it is intentional and reflects
the platform-specific panic-boundary semantics of WebAssembly.

**Cross-validation gate** (recorded 2026-05-08): Wave C smoke test
(`tests/heat.rs:gaussian_smoke`) reuses the exact parameters of Wave A's
`examples/heat.c` and Wave B's `tests/test_heat.py` (domain `[-10, 10]`,
`n=1000`, `t=1.0`, `n_steps=100`, `u0(x) = exp(-x²)`, oracle
`exp(-x²/5)/√5`, gate `sup_error < 5e-4`). Measured sup-error on wasm32:
**`1.460302e-6`**, matching Wave A's `1.46e-6` reading at 3-digit
precision (sub-ULP cross-boundary identity, same Rust core executed via
`wasm32-unknown-unknown` codegen).

**Out of scope**:
- Cross-engine browser smoke (Firefox + Safari headless) — deferred to v0.11.0.
- npm publish (`release-wasm.yml` with `wasm-pack publish` + `NPM_TOKEN`) — deferred to v0.11.0.
- Variable `a(x)` (closure-capturing `DiffusionChernoff`) — deferred to v0.11.0
  (matches Wave A and Wave B unit-a-only restriction).

## Amendment 2 — v8.0.0 Differentiable-Chernoff binding surface (2026-06-07)

**Status**: Accepted (records the v8.0.0 binding-scope decision)
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0133 (F1 Dual-AD), ADR-0135 Amendment 2 (F3 KilledDirichlet),
ADR-0076 (v3 binding redesign — extended additively here), math.md §46 (Dual<F>).

### Context — the scope judgment (the key decision)

v8.0.0's theme is "Differentiable Chernoff." The binding-relevant headline is
**F1 Dual-AD Greeks** (ADR-0133 applied directions A1 = HFT Greeks on the 45 ns
hot loop via the C ABI, and A3 = differentiable PDE in WASM/edge ML). The three
binding crates (`remizov-{ffi,py,wasm}`) are EXPERIMENTAL per the v0.10.0 decision
above (not ABI-stable until v1.0.0) and lag the Rust core by many kernels.

The naive completeness target — expose ALL six v8 kernels (F1 Dual-AD, F2
resolvent-jump, F3 killed-Dirichlet, F4 order-4 complex 5D, C1 Smolyak D6, C2
adjoint measure-state) through ALL three bindings — is **~18 implementations +
parity tests**. For experimental, pre-`v1.0.0`, no-stable-user surfaces this is
disproportionate effort and would dilute the one honest, headline-completing
deliverable. This is a value/effort allocation, **not** a physical contradiction
(complete-vs-bounded here has a clean honest answer: tier and defer with named
reasons), so no TRIZ resolution is forced. The genuine *design* subtlety lives
inside F1 itself (flat C ABI must surface value + Δ + Γ vectors, and Γ rides a
4-wide `Dual<Dual<f64>>` field) and is resolved in deliverables 2–5.

### Decision — TIER 1 / TIER 2 / TIER 3 split

**TIER 1 — SHIP in v8.0.0 (honest, complete, all three bindings):**
F1 Dual-AD Greeks (Δ = ∂u/∂θ and Γ = ∂²u/∂θ²) for the **1D unit-diffusion heat
kernel**, w.r.t. the **diffusion-coefficient scale θ** (the `scale` argument of
`DiffusionChernoff::new`). This is THE differentiable-Chernoff headline and MUST
be honest and complete across core ↔ FFI ↔ PyO3 ↔ WASM. Rationale for choosing
the diffusion-scale parameter: it is the cleanest single scalar that (a) demos
forward-mode AD end-to-end, (b) maps directly onto ADR-0133 A1's HFT Greek story
(Δ/Γ of price w.r.t. a volatility-like scale), and (c) reuses the existing
`*EvolverHeat1DUnitV3` constructor surface verbatim — the Greeks variant just
seeds the scale as `Dual::variable(θ)` / `Dual::variable(Dual::variable(θ))` on
the generic `apply_f` path and reads the `.tangent`(s). Output is the full grid
(value[N], delta[N], gamma[N]); the HFT point-Greek is a thin reduction (read one
node), NOT a separate binding entry.

**TIER 2 — OPPORTUNISTIC (only if genuinely a thin wrapper):**
F3 `KilledDirichletChernoff` 1D (hard absorbing wall) through **PyO3 only**, as a
non-blocking stretch. Justification: F3's binding is a clean
constructor-+-`apply` wrapper that reuses the `*EvolverHeat1DUnit*` builder shape
(an inner heat kernel + a region), and PyO3 is the lowest-friction binding (no C
header drift, no wasm-pack). It is explicitly **NOT release-blocking**: if it does
not land cleanly inside the F1 effort it slips to v8.x with zero headline impact.
FFI and WASM exposure of F3 are deferred. No parity gate is added for F3 in v8.0.0
(experimental, single-binding).

**TIER 3 — DEFER to v8.x with named reasons (honest non-goals):**
- **F2 ResolventJumpChernoff** — numerical-Laplace-inversion surface; the binding
  would have to expose quadrature configuration and a jump kernel. High-effort,
  low present demand. Deferred.
- **F4 order-4 complex 5D (ComplexTripleJump / high-D)** — `Complex<f64>` field +
  5D state; the C ABI for a 5D complex buffer is a large, low-reuse surface.
  Deferred.
- **C1 Smolyak D6** — sparse-grid D6 has no thin 1D-style buffer mapping; the
  binding would need a sparse-index ABI. Deferred.
- **C2 adjoint measure-state** — backward-mode/adjoint surface conflicts with the
  forward-only zero-alloc binding story and needs a tape ABI. Deferred (and note:
  C2 is the *backward* counterpart; v8.0.0 deliberately ships the FORWARD Greek
  story only).

All TIER 3 items remain fully usable from Rust; only their **binding** exposure is
deferred. This is an honest, named-reason deferral, not silent omission.

### The v8 Greeks binding surface (additive, ADR-0076 §Approach A continued)

Three new additive entry points, one per binding, all suffixed `_greeks` /
`Greeks`, all reusing the established `EvolverHeat1DUnitV3` construction shape and
the zero-alloc caller-owned-buffer convention:

- **FFI**: `smf_heat1d_greeks_v3(handle, t, out_value, out_delta, out_gamma, len)
  -> SemiflowStatus` — three caller-owned `*mut c_double` buffers of length `N`.
  Opaque handle is a NEW `SmfGreeksEvolverV3` (carries the `Dual<Dual<f64>>`
  kernel). `catch_panic!` on every extern; build under `[profile.release-ffi]`
  (`panic = "unwind"`) exactly as the existing `_v3` surface.
- **PyO3**: `EvolverHeat1DGreeksV3.greeks(t) -> (value, delta, gamma)` returning a
  3-tuple of `numpy.ndarray[float64]` (length N each). GIL released via `py.detach`
  during the pure-Rust dual sweep (three-phase ADR-0031 pattern).
- **WASM**: `EvolverHeat1DGreedsV3` — JS class with `.greeks(t) -> { value, delta,
  gamma }` (object of three `Float64Array`). `[profile.release]` (`panic = "abort"`)
  per Amendment 1; errors as `Result<_, JsValue>`.

### Cross-binding parity gate

NEW gate **G_BINDING_GREEKS_PARITY** (declared in `properties.yaml`,
RELEASE_BLOCKING, slow-tests): the `(value, delta, gamma)` triple MUST be
byte-identical (0 ULP) across core ↔ FFI ↔ PyO3 ↔ WASM AND the `delta`/`gamma`
components MUST match a central-difference reference to ≤ 1e-10 (reuses the
`G_DUAL_AD_GRADIENT` central-difference oracle, extended to second order for Γ).
This is the binding-surface companion to the core-only `G_DUAL_AD_GRADIENT`; it
proves the Greek survives the binding boundary unchanged. It does NOT replace
`G_binding_parity` (which guards the v2.x value-only kernel output).

### Reuse-vs-duplicate (per the §"per-crate duplication boundary")

REUSE from core: `Dual<F>`, `Dual<Dual<F>>`, `DiffusionChernoff::<Dual<…>>::new`
on the generic `apply_f` path, `Grid1D::new_generic`, `GridFn1D::from_fn_generic`,
`Evolver::evolve`. DUPLICATE per-crate (rule-of-three, established by the existing
`build_evolver_*` trio): the tiny `build_greeks_evolver` constructor helper, the
seed-θ-as-`Dual::variable` logic, and the validation stubs — each crate owns its
boundary translation exactly as it owns `unit_a` / `zero_d` today. NO shared
binding utility crate is introduced (ADR-0028 §"PyO3 owns its own boundary"
preserved).

## Amendment 3 — WASM bundle-size target superseded by `full` feature (2026-06-20)

**Status:** Accepted (records intentional deviation from the original < 500 KB bundle target)

The v0.10.0 Wave C decision implicitly targeted a small < 500 KB raw Wasm binary (the `Heat1D`-only scope made this easy). The binding-parity wave adds a `full` cargo feature to `semiflow-wasm` that enables all heavy-grid, multi-dimensional, and hypoelliptic engines; the resulting binary exceeds the informal size target. The default/"lite" build (no `--features full`) retains the lightweight baseline — all engines present at v0.10.0 — and measures **≈ 768 KB raw**. `--features full` measures **≈ 1.4 MB raw**; this is acceptable for applications that need the complete engine surface and have the bandwidth budget. The dual-target design is explicit: `Cargo.toml` `[features] full = []` plus the `lib.rs` module-level documentation listing lite vs full contents. No existing CI smoke or cross-validation gate changes.

### Out of scope (v8.0.0 bindings)

- F3 FFI + WASM bindings (TIER 2 is PyO3-only).
- F2 / F4 / C1 / C2 bindings of any kind (TIER 3, deferred to v8.x).
- Greeks w.r.t. parameters OTHER than the diffusion scale (e.g. initial-amplitude,
  boundary, drift) — the AD machinery covers them in core, but the v8 binding
  exposes exactly ONE seeded parameter; multi-parameter Jacobian bindings are
  deferred to v8.x.
- Variable `a(x)` Greeks (still unit-a only at the binding layer, inheriting the
  v0.10.0 / v0.11.0 unit-a binding restriction).
