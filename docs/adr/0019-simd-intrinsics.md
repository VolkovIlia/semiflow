# ADR-0019 — v0.8.0 Block C SIMD intrinsics module (AVX2 + NEON, scalar fallback)

**Status**: Accepted
**Date**: 2026-05-05
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0012 (tensor-product 2D), ADR-0015 (6th-order spatial),
ADR-0017 (v0.8.0 perf baseline + lint), ADR-0018 (parallel Strang2D),
ROADMAP.md v0.8.0 PERFORMANCE THEME items 2 + 3,
`crates/semiflow-core/src/grid.rs` (cubic-Hermite hot path),
`crates/semiflow-core/src/grid_quintic.rs` (quintic-Hermite hot path),
`crates/semiflow-core/src/diffusion6.rs` (7-pt K-kernel + 9-pt Fornberg FD),
`contracts/semiflow-core.tensor.yaml` schema 0.7.2 (`simd` block),
`contracts/semiflow-core.properties.yaml` schema 0.7.2 gates
`SIMD_BIT_EQUAL` (release-blocking) + `SIMD_HERMITE_SPEEDUP` /
`SIMD_COMBINED_SPEEDUP` (informational), `docs/perf-baseline-v0_7_0.md`.

`semiflow-core` adds a single scoped module `src/simd/` housing 4-lane f64
intrinsics — `std::arch::x86_64::*` (AVX2, `__m256d`) and
`std::arch::aarch64::*` (NEON, two `float64x2_t` registers per lane-4) —
behind a new opt-in `simd` Cargo feature defaulted ON for `cfg(any(target_arch
= "x86_64", target_arch = "aarch64"))` and silently scalar-fallback on every
other target. The crate root carries `#![deny(unsafe_code)]` (NOT `forbid`)
and the workspace `Cargo.toml` lint table mirrors it as `unsafe_code = "deny"`;
`unsafe` is permitted ONLY via `#[allow(unsafe_code)]` at the head of
`src/simd/mod.rs`, and the only inhabitants of that module are the per-arch
intrinsic shims. **`deny`-not-`forbid` rationale**: Rust's lint hierarchy makes
`forbid` non-overridable by inner `#[allow]`, so a `forbid(unsafe_code)` crate
root would refuse to compile the scoped `#[allow(unsafe_code)]` on
`src/simd/mod.rs` that the SIMD intrinsics require; `deny` is the strongest
lint level that still admits the scoped allow, and the safety-by-design
intent of the original `forbid` is preserved by an **out-of-band CI grep
enforcement gate** that fails the build whenever any `unsafe` token or
`#[allow(unsafe_code)]` attribute appears outside `src/simd/` (the lint-policy
declaration line in `src/lib.rs` itself — `#![deny(unsafe_code)]` — is
explicitly excluded as a non-`unsafe` site). The contract is:
```
! grep -rEn '\b(unsafe|allow\(unsafe_code\))\b' crates/semiflow-core/src --include='*.rs' \
    | grep -v '/simd/' \
    | grep -vE '^[^:]+:[0-9]+:#!\[(deny|allow|forbid)\(unsafe_code\)\]'
```
to be wired into `xtask` before v0.8.0 ships (engineer-domain). With `deny` + the grep gate, no `unsafe` leaks across the
module boundary, every public-from-crate path
(`Grid::sample`, `GridQuintic::sample`, `Diffusion6thChernoff::apply`)
remains a safe-Rust call site that dispatches to the trait
`simd::SimdF64x4`. Compile-time arch dispatch (`#[cfg(target_arch = ...)]`)
selects the impl; runtime feature detection (`is_x86_feature_detected!`) is
explicitly rejected because the build matrix already pins the target and a
runtime branch would (a) break inlining of the hot stencil loops and (b)
require `cfg(target_feature = "avx2")` or per-call dispatch tables that
inflate the module size beyond the suckless ≤500-line budget.
**Determinism contract — TIGHT BIT-EQUALITY (no FMA)**: the SIMD path MUST
produce a `Vec<f64>` byte-identical to the scalar path for every input,
across {AVX2, NEON, scalar-fallback} builds and across the v0.8.0 Block B
parallel kernel (gate `SIMD_BIT_EQUAL`, release-blocking). This is
achievable because the hot kernels are dense `Σ wᵢ · vᵢ` reductions where
weights `wᵢ` and lane order are compile-time constants; scalar code is
`a·b + c·d + …` with explicit IEEE-754 rounding at each multiply, and the
SIMD impl mirrors that order EXACTLY using only `_mm256_add_pd`,
`_mm256_sub_pd`, `_mm256_mul_pd` (and their `vaddq_f64` / `vsubq_f64` /
`vmulq_f64` NEON twins) — fused-multiply-add (`_mm256_fmadd_pd`,
`vfmaq_f64`) is **explicitly forbidden in vectorized hot paths** because
FMA collapses two roundings into one and would break the bit-equal
contract; this costs ≈3% peak FLOPs but buys deterministic regression
testing and lets every existing v0.5.0–v0.7.0 release-blocking gate
(G1/G2/G3-2D/G3⁴-2D/G3⁶-2D/Z⁶_const-a) carry over byte-for-byte under
`--features simd`. **Alternatives considered**: `std::simd` /
`portable_simd` (rejected: unstable, MSRV is 1.78 stable); `wide` /
`packed_simd` / `safe_arch` crates (rejected: violates the suckless
dep-count invariant of 2 — `num-traits`, `libm` — and adds opaque
abstraction layers between the FD weight constants and the assembler);
runtime CPU detection via `is_x86_feature_detected!` (rejected as above);
splitting AVX2 and NEON into two ADRs (rejected: doubles audit churn for
negligible isolation, single-module unsafe boundary is the entire point);
allowing FMA with ≤2 ULP relaxation (rejected: every v0.5.0+ release-
blocking gate is bit-equal not ULP-bounded, so an FMA-relaxed SIMD path
would fork the test surface — the 3% FLOPs cost is far cheaper than
maintaining two reference outputs). **Consequences**: `cargo build
--no-default-features` continues to compile no_std + alloc serial scalar
clean (the `simd` feature is disabled, no `std::arch` import); `cargo
build --features simd` (the new default for x86_64/aarch64) inherits
`std` and pulls in the AVX2 or NEON impl via cfg; `cargo build
--features parallel,simd` composes Block B + Block C giving the
ROADMAP-mandated ≥5× combined speedup at N=1600² heat 2D
(`SIMD_COMBINED_SPEEDUP`, informational). The `unsafe` blast radius is
bounded to `src/simd/{x86_64.rs, aarch64.rs}` — reviewer-suckless audits
exactly two files; every other touched site (`grid.rs`,
`grid_quintic.rs`, `diffusion6.rs`) calls into the trait via safe Rust
and is unmodified at the API level. The blast-radius bound is **the
real safety guarantee** (verified by grep at the time of this ADR: zero
`unsafe` and zero `#[allow(unsafe_code)]` outside `src/simd/`), not the
choice between `deny` and `forbid` at the lint level — engineer MUST
wire the CI grep gate above into `xtask` (e.g. `xtask ci`) before v0.8.0
ships so the bound is mechanically enforced on every PR, not just at
review time. Dep count stays at 2.

---

## Amendment 2026-05-06: Tighten cfg gates from `target_arch` to `all(target_arch, target_feature)`

**Status**: Accepted (engineer-domain follow-up — `src/simd/x86_64.rs` and
`src/simd/aarch64.rs` cfg lines change).
**Authors**: ai-solutions-architect.
**Cross-ref**: ADR-0020 Amendment 2026-05-06 (Block D recalibration —
co-amendment).

### Trigger

Block D first-run measurement: SIMD speedup at `N=1024` was `1.01×`
(essentially scalar) under the default `cargo test --release
--features parallel,simd,slow-tests` invocation, despite the AVX2
intrinsics in `src/simd/x86_64.rs` being compiled in. Root cause: the
file-head gate `#![cfg(target_arch = "x86_64")]` admits the AVX2
intrinsics on every x86_64 build, but the default x86_64 Rust target
ships with SSE2 only — without `target_feature = "avx2"`, LLVM cannot
prove the host CPU supports AVX2 and the `_mm256_*_pd` calls fail to
inline cleanly into the FD9/Hermite hot loops, producing scalar-equivalent
or slightly-worse codegen. Block C's bench file
`benches/diffusion6_simd.rs:8` already documents the workaround
(`RUSTFLAGS="-C target-feature=+avx2"`), but consumers (and Block D's
gate invocation) inherited the default cargo flags silently — the
"silent-scalar trap".

### Decision

Tighten the file-head cfg gate on both intrinsic files:

```rust
// src/simd/x86_64.rs
- #![cfg(target_arch = "x86_64")]
+ #![cfg(all(target_arch = "x86_64", target_feature = "avx2"))]

// src/simd/aarch64.rs
- #![cfg(target_arch = "aarch64")]
+ #![cfg(all(target_arch = "aarch64", target_feature = "neon"))]
```

And update the dispatch in `src/simd/mod.rs`:

```rust
- #[cfg(target_arch = "x86_64")]
+ #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
  mod x86_64;

- #[cfg(target_arch = "aarch64")]
+ #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  mod aarch64;

  // Scalar fallback now activates whenever the target_feature
  // requirement is unmet, NOT only on non-x86/non-arm arches.
- #[cfg(any(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
+ #[cfg(any(test, not(any(
+     all(target_arch = "x86_64", target_feature = "avx2"),
+     all(target_arch = "aarch64", target_feature = "neon")
+ ))))]
  mod scalar;
```

And update the `pub(crate) use` re-exports symmetrically (so `F64x4`
resolves to `F64x4Avx2` only when both `target_arch = "x86_64"` AND
`target_feature = "avx2"` hold; falls through to `F64x4Scalar`
otherwise).

### Compile-time arch × feature matrix

| Target arch | `target_feature = "avx2"` | `target_feature = "neon"` | `F64x4` resolves to |
|-------------|---------------------------|---------------------------|----------------------|
| x86_64 | yes | (n/a) | `F64x4Avx2` (AVX2 hot path) |
| x86_64 | no | (n/a) | `F64x4Scalar` (scalar fallback — explicit) |
| aarch64 | (n/a) | yes | `F64x4Neon` (NEON hot path) |
| aarch64 | (n/a) | no | `F64x4Scalar` (scalar fallback — explicit) |
| any other | (n/a) | (n/a) | `F64x4Scalar` |

Note: AArch64 targets enable NEON by default (it's part of the AArch64
ISA baseline), so Linux/macOS aarch64 builds keep the `F64x4Neon` path
unchanged. The change is the explicit cfg cross-product — what was
implicit becomes explicit, with no expected runtime behaviour change
on default aarch64.

### Consequences

- Default x86_64 `cargo build --features simd` now compiles to
  `F64x4Scalar` (NOT `F64x4Avx2`) — perf parity with
  `--no-default-features` for SIMD-touching paths. **Visible** scalar
  fallback closes the silent-scalar trap.
- Consumers wanting AVX2 acceleration MUST opt in via
  `RUSTFLAGS="-C target-cpu=native"` or
  `RUSTFLAGS="-C target-feature=+avx2"`. The contract is now explicit.
- Block D's `G3_6_2D` gate invocation requires the `RUSTFLAGS` (per
  ADR-0020 Amendment); without it, the test compiles to scalar and
  fails the runtime budget — but with the explicit message that the
  budget assumes AVX2 is active.
- `SIMD_BIT_EQUAL` (release-blocking) still passes because it
  composes scalar↔scalar bit-equality on default x86_64 builds and
  AVX2↔scalar bit-equality only on `target-feature=+avx2` builds —
  both contracts are intact.
- The `with_force_scalar` test hook continues to work: it
  forces the scalar dispatch in CI for bit-equality verification on
  every x86_64 build matrix entry where AVX2 is enabled.
- The CI grep enforcement (no `unsafe` outside `src/simd/`) is
  unaffected — `target_feature` cfg-gating is orthogonal to lint
  scope.
- No public API changes. No `Cargo.toml` changes. No new dependencies.
- Bench file `benches/diffusion6_simd.rs` doc-comment line 8 remains
  accurate; the pre-existing workaround instruction now matches the
  cfg requirement explicitly.

### Engineer-domain implementation notes

- The change is mechanical: 6 cfg-attribute edits across 3 files
  (`src/simd/mod.rs`, `src/simd/x86_64.rs`, `src/simd/aarch64.rs`).
- After the change, run `cargo build --features simd` (default x86_64
  target) and verify `F64x4` resolves to `F64x4Scalar` (e.g. via a
  `compile_fail` doctest or `cargo expand`).
- Run `RUSTFLAGS="-C target-cpu=native" cargo build --features simd`
  on an AVX2 host and verify `F64x4` resolves to `F64x4Avx2`.
- Run `cargo test --features simd` (default) → `SIMD_BIT_EQUAL`
  scalar↔scalar still passes.
- Run `RUSTFLAGS="-C target-cpu=native" cargo test --features simd`
  → `SIMD_BIT_EQUAL` AVX2↔scalar still passes.
- This change is a prerequisite for re-running `G3_6_2D` after
  recalibration. It does not bump the crate version
  (Block E does versioning).

---

## Amendment 2 — v0.13.0 Wave B SIMD extension to TruncatedExp4

**Status**: Proposed 2026-05-19 for v0.13.0.

**Context**: Iter-3 bench F2 (1D heat time-dep, Magnus GL4 class): RC `TruncatedExp4thDiffusionChernoff` 65× slower than kiops-magnus-gl4. Hot path `truncated_exp4.rs:411-414` has 4 closure calls (`mc.a)(...)` per node per stencil application; closure indirection defeats SIMD lane utilization regardless of width. ADR-0019 v0.8.0 covered cubic-Hermite/quintic-Hermite/FD stencils but did NOT include `TruncatedExp4` K=4 power-series accumulation loop.

**Decision**: Extend ADR-0019 SIMD coverage to `TruncatedExp4thDiffusionChernoff::apply` hot path with three pre-conditions: (a) `HalfNodeCoeffCache<F>` storage refactor (ADR-0034 Amendment 1) precomputes half-node coefficients as flat `Vec<F>` eliminating closure indirection; (b) wasted-sample bypass in `truncated_exp4.rs:432` replaces `g_grids[k].sample(x_mid)` with direct `values()[i]` indexing when `x_mid == x_i` (free 1.3-1.5× bit-equal); (c) AVX2 (`x86_64`) + NEON (`aarch64`) `#[target_feature]` paths on the 5-point stencil application kernel and K=4 Horner-style polynomial accumulation, FMA-DISABLED to preserve scalar arithmetic order (cite SLEEF arXiv 2001.09258, SimSIMD ashvardanian/SimSIMD, Compensated Horner arXiv cs/0610122). Bit-equality gate: new `TEXP4_SIMD_BIT_EQUAL` test asserts byte-identical f64 output vs scalar reference path at N=64 canonical bench.

**Consequences**: Expected wall-time gain on TExp4 path: 1.3-1.5× (sample bypass) × 2.5-3× (SIMD on stencil+Horner) ≈ **3.4-4.5× cumulative**. F2 65× gap closes to **~15-20× residual** (algorithmic Krylov floor, ADR-0037). Risk: bit-equality regression if FMA accidentally enabled or AVX2 horizontal-sum reorders adds; mitigated by `TEXP4_SIMD_BIT_EQUAL` gate and `-C target-feature=-fma` build flag verification. Memory cost: 2N·sizeof(F) ≈ 8 KB at N=512 (negligible vs H-MEM 2.8 MB baseline).
