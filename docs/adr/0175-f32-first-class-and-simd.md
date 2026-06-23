# ADR-0175 — f32 as a first-class carrier with a deterministic SIMD path

**Status**: Accepted
**Date**: 2026-06-23
**Authors**: ai-solutions-architect
**Issue**: #5 (`issue-5-f32-simd`)
**Cross-refs**: ADR-0019 (SIMD intrinsics determinism contract — **scope extended to f32 here**),
ADR-0025 (Generic-over-Float — **§SIMD carve-out deferral superseded here**),
ADR-0018 (parallel Strang carve-out, untouched),
`crates/semiflow-core/src/float.rs` (`SemiflowFloat`; "f32 scalar-only" note updated),
`crates/semiflow-core/src/simd/{mod.rs,x86_64.rs,aarch64.rs}` (`SimdF64x4` → mirrored by `SimdF32x8`/`SimdF32x4`),
`crates/semiflow-core/src/grid_cubic.rs` + `diffusion6_helpers.rs` (f64 SIMD dispatch → f32 sibling),
`crates/semiflow-core/tests/simd_bit_equal.rs` (`SIMD_BIT_EQUAL` mirrored by `SIMD_F32_BIT_EQUAL` in `tests/simd_bit_equal_f32.rs`),
`contracts/semiflow-core.math.md` §46.5 (NORMATIVE SIMD note),
`contracts/semiflow-core.properties.yaml` (new gate `SIMD_F32_BIT_EQUAL`),
`docs/adr/0175-engineer-handoff.md` (file-by-file implementation checklist).

## Context

`ChernoffFunction<f32>` is not implemented on the leaf kernels. f32 reaches a result
only through the generic scalar `apply_f` path, surfaced in tests via the `WrapDiff<F>`
shim (`crates/semiflow-core/tests/generic_float_strang.rs`). Because the production f64
impls route through the AVX2/NEON SIMD path (`grid_cubic.rs` Catmull-Rom, `diffusion6_helpers.rs`
9-pt FD stencil) while f32 falls through to pure scalar, single-precision buys ~no
throughput or memory advantage (measured f32/f64 ≈ 0.93× wall). ADR-0025 type-parameterised
the surface but explicitly deferred concrete f32 SIMD to "a future ADR"; this **is** that ADR.
The `ChernoffFunction` trait has **56 direct dependents** (gitnexus: CRITICAL) — the design
is therefore strictly additive: new `impl ChernoffFunction<f32>` blocks only, **no change to
the trait signature**.

## Decision

Make f32 a first-class carrier in two independently-mergeable phases. **Phase 5a** adds an
additive `impl ChernoffFunction<f32>` to each of the seven 1D leaf kernels — DiffusionChernoff,
Diffusion4thChernoff, Diffusion6thChernoff, DriftReactionChernoff, TruncatedExpDiffusionChernoff,
TruncatedExp4thDiffusionChernoff, ShiftChernoff1D — each delegating to the kernel's existing
generic `apply_f` scalar path (no new math), retiring the test-only `WrapDiff<F>` shim. **Phase 5b**
adds an f32 SIMD kernel — a crate-private `SimdF32x8` (AVX2 `__m256`, 8 lanes) / `SimdF32x4`
(NEON `float32x4_t`, 4 lanes) trait mirroring `SimdF64x4`, wired into `grid_cubic.rs` Catmull-Rom
dispatch and `diffusion6_helpers.rs` — so the f32 leaf kernels use the wide path instead of scalar.
**5a may merge independently of 5b**: 5a delivers correctness/API (f32 ≡ scalar, already correct),
5b delivers throughput. The f32 precision floor (~1e-5 accumulated, intrinsic to single precision)
is **unchanged** — the SIMD path is a performance transform, byte-equal to the f32 scalar reference,
NOT an accuracy change.

## Determinism contract (extends ADR-0019 to f32)

`SimdF32x8`/`SimdF32x4` carry the **same** contract as `SimdF64x4`: only `splat`, `load_unaligned`,
`store_unaligned`, lane-wise `add`/`sub`/`mul`, and a single `horizontal_sum` with a **hard-fixed
reduction tree**. **FMA is FORBIDDEN** — `mul` and `add` are separate rounding steps (no `vfmadd*`).
Every method is byte-identical to the corresponding f32 scalar op. `unsafe` is confined to
`src/simd/x86_64.rs` + `src/simd/aarch64.rs` (intrinsic shims only), enforced by the existing
`xtask check-unsafe-scope` CI grep (no `unsafe` / `#[allow(unsafe_code)]` token outside `src/simd/`).
The lane count (8 for f32 AVX2 vs 4 for f64) is **not observable** in the result because the per-node
Catmull-Rom/FD computation is a fixed unrolled convolution (lane-local `add`/`sub`/`mul`) and the only
cross-lane step is the fixed-order `horizontal_sum` — see the contradiction resolution below.

## Contradiction resolution (TRIZ АП→ТП→ФП→ИКР)

**НЭ / ТП:** wide f32 lanes want to reorder accumulation (tree reduction, FMA contraction) for
throughput (useful) — but reordering breaks byte-equality with the sequential scalar reduction (harm).
ТП-1 (free reorder) = fast but non-deterministic; ТП-2 (replay scalar order serially) = deterministic
but no vectorisation. We keep ТП-1 (throughput is the main function) and must remove the harm.

**ФП:** the order of arithmetic in the kernel must be *parallel/reordered* (speed) AND *identical to
scalar* (bit-equality) at once. **Resolved in structure + time:** the conflict zone is **only the
lane-merge point** (`horizontal_sum`), not the element-wise `add`/`sub`/`mul`, which are lane-local
and bit-identical to scalar at **any** width. The Catmull-Rom / FD kernels do **not** sum over N nodes;
they pack a fixed 4 (Catmull) or 8/9 (FD) control points into lanes and take one fixed convolution per
node. Therefore: (a) element-wise ops are bit-equal to scalar by construction (width-invariant), and
(b) the single cross-lane reduction is pinned to the same addition tree the scalar path uses
(`((l0+l1)+l2)+l3` for f64x4; the engineer fixes the analogous tree for f32x8/x4 and writes the f32
scalar reference in the **same** order). **ИКР:** lane width is *unobservable* in the output — speed
comes free from the register width applied where the order is not observable, determinism comes from
keeping the only observable reduction fixed. Bit-equality is not a compromise paid for in speed; it is
structurally impossible to violate as long as the wide ops stay lane-local and the merge tree stays
fixed. The release-blocking `SIMD_F32_BIT_EQUAL` gate (below) makes this self-enforcing in CI.

## Gate spec — SIMD_F32_BIT_EQUAL (RELEASE-BLOCKING)

A new release-blocking gate `SIMD_F32_BIT_EQUAL`, living in
`crates/semiflow-core/tests/simd_bit_equal_f32.rs`, mirrors `SIMD_BIT_EQUAL`
(`tests/simd_bit_equal.rs`) **exactly**, at f32: it asserts the f32 SIMD path is byte-for-byte
identical to the f32 scalar path — forced via the existing `with_force_scalar` / `FORCE_SCALAR`
thread-local hook — for every combination of `N ∈ {64, 256, 1024, 4096}` × `boundary ∈ {Reflect,
ZeroExtend, Periodic, LinearExtrapolate}`, across the same two hot paths the f64 gate covers
(`diffusion6_apply` f32 + the Catmull/Hermite f32 sampler). Comparison is byte-exact via
`f32::to_bits()` (covers signed-zero, subnormals, NaN bit patterns) — **no tolerance**. On divergence
the reporter prints the first divergent index, scalar bits (hex), simd bits (hex), and ULP gap, exactly
as the f64 gate does. The gate is `#![cfg(feature = "simd")]`, registered RELEASE-BLOCKING in
`contracts/semiflow-core.properties.yaml` (mirror the `SIMD_BIT_EQUAL` entry), and runs under
`test-full` (`--features parallel,simd,slow-tests --release`). This gate is the acceptance condition
for **5b** (5a, being scalar, is covered by the existing generic-float tests).

## Consequences

f32 becomes a genuine throughput/memory carrier (AVX2: 8 f32 lanes vs 4 f64 = ~2× bandwidth; half the
memory footprint per grid). The `WrapDiff<F>` shim is retired (5a). No trait-signature change → all 56
`ChernoffFunction` dependents compile unchanged (additive impls only). The unsafe blast radius stays
bounded to `src/simd/` (ADR-0019 grep gate, now also covering the f32 intrinsics). Risk: a compiler
folding `mul`+`add` into FMA would silently break bit-equality — mitigated by using only non-FMA
intrinsics (as the f64 path already does) and the `SIMD_F32_BIT_EQUAL` CI gate catching any regression.
Deletion backlog: remove `WrapDiff<F>` from `generic_float_strang.rs` once 5a lands.
