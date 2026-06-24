# ADR-0183: WASM Greeks parity is tolerance-bounded, not byte-identical

**Status**: Accepted (amends ADR-0076 §`G_binding_parity` and the
`G_BINDING_GREEKS_PARITY` contract for sub-test 4 only)

**Context**: `G_BINDING_GREEKS_PARITY` (ADR-0133, ADR-0028 Amendment 2,
math.md §46) requires the Greeks triple (value/delta/gamma) to be byte-identical
(0 ULP) across core ↔ FFI v3 ↔ PyO3 v3 ↔ WASM v3. Sub-tests 2 (FFI) and 3
(PyO3) run on the native target and share the host libm with the core golden, so
they genuinely pass at 0 ULP. Sub-test 4 (WASM) does NOT: wasm32 ships its own
scalar libm whose `exp()` differs from the host libm in the last ULP. Over 32
Chernoff steps and the hyper-dual chain rule those last-ULP differences amplify.
Measured WASM-vs-core divergence at the canonical config (N=64, n_chernoff=32,
t=0.05, θ=0.5): value ≤ 71193 ULP, delta ≤ 171579 ULP, gamma ≤ 284959 ULP —
but only ≤ 8.5e-12 / 2.7e-11 / 6.1e-11 in *relative* error. The large ULP figures
are concentrated on the ~1e-23-magnitude Gaussian tail, where one bit of
mantissa is many ULP yet a negligible relative quantity. The hyper-dual Greeks
path is NOT SIMD-accelerated (the AVX2 intrinsics are `f64`-only and cannot apply
to `Dual<Dual<f64>>`), so a *scalar* core build does not close the gap either
(verified: 3/64 value, 0/64 delta, 0/64 gamma at 0 ULP). The root cause is
native↔wasm32 libm `exp()` non-determinism, which is irreducible.

A prior fix made sub-test 4 pass by regenerating the "golden" arrays from the
WASM binary's own output while keeping the "Core golden" label and the 0-ULP
assertion. That made the gate vacuous (WASM == WASM) and masked the real
divergence. This is unacceptable.

**Decision**: For sub-test 4 (WASM) only, the parity criterion is changed from
0-ULP byte-equality to a **per-array sup relative-error tolerance** against the
legitimate **scalar core golden** (the independent, Richardson-FD-verified
hyper-dual sweep from `crates/semiflow/tests/binding_greeks_parity.rs`, built
`--no-default-features --features std` so the golden uses the same strict scalar
IEEE-754 f64 arithmetic family). Tolerances: value ≤ 1e-9, delta ≤ 1e-9,
gamma ≤ 1e-9 (≈150× headroom over the measured 6.1e-11 physics gap). The change
of axis from ULP to relative error resolves the tight-vs-loose contradiction: a
real marshalling bug (wrong index, transposed/dropped array, sign flip) produces
order-1 relative error and is caught, while the uniform ~6e-11 libm gap passes.
FFI and PyO3 sub-tests remain 0-ULP (native, genuine).

**Consequences**: The WASM golden is sourced from the core library, never from
the WASM binary. The gate is genuine — it still catches binding-boundary
marshalling bugs — and honest about cross-platform float determinism. The
`G_BINDING_GREEKS_PARITY` contract and the test docstrings are corrected to state
that WASM parity is tolerance-bounded (≤ 1e-9 relative), not byte-identical. No
core symbols are touched (test + doc + contract only).
