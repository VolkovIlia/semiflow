# ADR-0138 — v8.1.0 TIER-3 binding surface (F2, F4, C1-D6, C2)

**Status:** ACCEPTED (2026-06-08) · **Branch:** `feat/v8.1.0-debt-closure`
**Supersedes:** the TIER-3 *deferral* rows of ADR-0028 Amendment 2 (the four named
kernels are now scheduled, not indefinitely deferred). **Cross-refs:** ADR-0028
Amendment 2 (per-crate-dup mandate, binding-scope tiering), ADR-0076 (additive
`_v3` surface), ADR-0031 (PyO3 three-phase GIL release), ADR-0134/0136/0123/0107
(the four kernels' NARROW scopes), `.dev-docs/reports/V8_1_TIER3_BINDING_DESIGN.md`.

**Decision.** v8.1.0 closes the binding debt for the four v8.0.0 core kernels that
ship with no binding surface — **F2 `ResolventJumpChernoff`**, **F4
`ComplexTripleJump`**, **C1 `SmolyakGridND` at D=6**, **C2
`AdjointFokkerPlanckChernoff`** — by adding thin additive wrappers that copy the F1
Greeks template verbatim, **tripled per crate with NO shared util** (the per-crate
duplication of ADR-0028 Amendment 2 is REQUIRED, not optional: each of
`remizov-{ffi,py,wasm}` owns its own Rust↔boundary code). Surfaces are tiered by the
kernel's *shape*, not by importance: **F2 and C2 → full FFI+PyO3+WASM** (each has a
natural real-valued scalar/measure `evolve`/`jump` entry that maps onto the F1
buffer-copy pattern); **C1-D6 and F4 → PyO3-first TIER-2** (intrinsically multi-dim
`GridFnND<f64,{5,6}>` state with no thin 1D-style buffer mapping — FFI/WASM are
opportunistic and may slip within v8.1 with zero headline impact, exactly as F3
KilledDirichlet1D was PyO3-only in v8.0.0). The hard ABI-safety constraint: **NO
`SemiflowComplex`/`num_complex::Complex` type leaks across ANY boundary** — F2 and F4
carry complex internals (TWS contour arithmetic; complex-time substeps), but only
their *real-valued* entry points (`jump` taking/returning `GridFn1D<f64>`;
`apply_real` taking/returning `GridFnND<f64,5>`) are exposed; the complex math stays
sealed inside core, and `CplxGridFn5`/`Complex<f64>` never appear in a signature.
Each binding echoes its kernel's NARROW limitation in rustdoc (F2 self-adjoint /
sectorial only; F4 filiform-N5 step-4 Carnot only) so callers cannot misuse it; every
FFI entry is `catch_panic!`-wrapped (build under `[profile.release-ffi]`,
`panic=unwind`), every PyO3 compute releases the GIL via `py.detach`, no crate gains a
new dependency, and each new binding file stays ≤500 LoC (split into per-surface
helper modules if a kernel would exceed). Each bound kernel gets a 0-ULP cross-binding
parity gate `G_BINDING_<KERNEL>_PARITY` comparing whatever surfaces it is bound to
against a core golden; the contract additions are recorded in
`contracts/semiflow-core.traits.yaml` (schema **4.2.0 → 4.3.0** MINOR, additive) and the
gates in `contracts/semiflow-core.properties.yaml`. No genuine contradiction is present
(completeness vs boundedness are a budget choice, resolved by value-tiering — same
honest resolution as ADR-0028 Amendment 2 §0); no TRIZ resolution is forced.
