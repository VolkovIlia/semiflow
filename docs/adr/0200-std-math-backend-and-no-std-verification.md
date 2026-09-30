# ADR-0200 — Ship the math backend the tests verify, and prove `no_std` by execution

- **Status**: Proposed
- **Date**: 2026-09-28
- **Supersedes / amends**: ADR-0003 (`no_std + alloc` core), ADR-0025 (libm for
  `Float` in `no_std`) — the `std` feature wiring and the verification contract.
- **Contract**: no `math.md` change. New CI jobs `no-std`, `no-std-test`,
  `no-std-libm`, `no-std-libm-neon`, `no-std-qemu`; no new `properties.yaml` gate.

## Context

Two facts measured on 0.13.1-beta:

1. **Shipped ≠ tested.** `semiflow`'s `std` feature was `std = []`. It never
   enabled `num-traits/std`, so a downstream crate using default features
   computed every generic `f64`/`f32` transcendental (`exp`, `powf`, `sin`, …
   through `num_traits::Float`) with the pure-Rust `libm`. The repository's
   own tests never ran that configuration: the dev-dependencies (`statrs` →
   `nalgebra`, `criterion`) enable `num-traits/std`, and with it num-traits uses
   the platform math library. `tests/adaptive_classical_bit_equal.rs` states
   that `ClassicalPI` uses "platform pow", which was true only in test builds.
2. **`no_std` was type-checked at best.** Until issue #40 the
   `--no-default-features` build did not compile at all. After #40, CI only
   ran `cargo check`. `cargo test --no-default-features` did not compile, and
   even when it does, the same dev-dependencies load `std` into the build, so
   the `libm` path that real `no_std` users get was executed by nothing.

## Decision

1. **One math backend in every build: `libm`, and the same lane arithmetic.**
   semiflow evaluates every transcendental through `SemiflowFloat::libm_*`
   (`libm` for `f32`/`f64`; `Dual<F>` applies them to its components) and
   every complex one through `complex_libm` (num-complex 0.4.6's formulas on
   top of `libm_*`). The portable SIMD lanes (`simd::{F64x4, F32x8, F32x4}`,
   scalar fallback) compile without the `simd` feature, so the lane kernels
   sum in the same order whether or not the AVX2/NEON intrinsics are compiled
   in, and the G⁴ scalar fallback mirrors the intrinsics operation for
   operation. `std` and `no_std` builds therefore produce identical bits.
   `clippy::disallowed_methods` (root `clippy.toml`, denied in `semiflow`
   outside unit tests and in `semiflow-nostd-check`) rejects any
   `num_traits::Float`, `f64`/`f32` or `num_complex::Complex` transcendental.
   `std = ["num-traits/std", "num-complex/std"]` stays, but it no longer
   affects semiflow's numerics — only downstream code that calls `Float`
   directly. Rejected: `std` → platform math library (glibc). Its results
   depend on the libm vendor and, through IFUNC, on the CPU (ADR-0191
   Amendment 5), so `std` and `no_std` could never agree, nor two `std`
   machines.
2. **`no_std` is verified in four independent ways**, all on every pull request:
   - `no-std`: `cargo build --no-default-features` on the host,
     `thumbv7em-none-eabihf`, `thumbv7m-none-eabi`, and at MSRV.
   - `no-std-test`: the whole `semiflow` test suite (unit + integration) with the
     crate compiled `#![no_std]`. It exercises the `no_std` code paths. It does
     not exercise `libm`, because the test harness links `std`.
   - `no-std-libm`: `crates/semiflow-nostd-check`, a `#![no_std]` crate whose
     only dependency is `semiflow` without default features, runs closed-form
     oracle scenarios and hashes each scenario's output bits (FNV-1a 64). The
     digests must equal the committed table (`src/expected.rs`). The job asserts
     from `cargo tree` that nothing enables `num-traits/std` in the default run,
     then reruns with `--features std-ref` (semiflow's default `std` + `simd`,
     `num-traits/std` on) and with `-C target-cpu=x86-64-v3` (AVX2 lanes);
     both must reproduce the same table. `no-std-libm-neon` repeats `std-ref`
     on an aarch64 runner (NEON lanes).
   - `no-std-qemu`: `nostd-qemu/` links the same scenarios into a
     `#![no_main]` bare-metal binary for Cortex-M3 (soft float) and Cortex-M4F
     (hard float) and runs them under QEMU, digests included. The exit code is
     the verdict.
   Rejected: a `cargo check` only (proves nothing about behaviour), and adding
   `libm` tests inside `semiflow` itself (impossible, because the dev-dependencies
   switch num-traits to `std` by feature unification).
3. **Targets without pointer-sized atomics fail with an explicit
   `compile_error!`.** The engines share coefficient closures through
   `alloc::sync::Arc`. Rejected: `portable-atomic`, which would add a fourth
   runtime dependency (the cap is three).

## Consequences

- `std` results can move by a few ULP relative to 0.13.1-beta (platform math
  → `libm`); `std` and `no_std` results are now bit-identical, on every CPU.
  No gate threshold changes. Every exact-bits golden (including
  `adaptive_classical_trace_v1.json`) and the binding parity tests passed
  unchanged; one unit test that compared against platform `cos`/`sin` now uses
  `libm_*` as its reference.
- CI gains four jobs. The QEMU job installs `qemu-system-arm` from apt.
- `nostd-qemu/` is outside the workspace (a `#![no_main]` binary cannot build
  for the host), with its own `Cargo.lock`.

## Honest limits

- The QEMU run executes a curated set of scenarios, not the full test suite.
  The full suite runs `no_std`-compiled but with `std` math.
- QEMU checks instruction-set and ABI correctness, not the timing or memory
  limits of any particular board.
- `thumbv6m` and other targets without atomic CAS stay unsupported.
- The integration-test `simd::with_force_scalar` hook is inert: it acts
  through `cfg!(test)` inside the library, which is false when the library is
  built for integration tests, so `SIMD_BIT_EQUAL` / `TEXP4_SIMD_BIT_EQUAL`
  compare SIMD with SIMD. AVX2/NEON-versus-scalar identity is proven by the
  digest jobs instead (`no-std-libm` AVX2 step, `no-std-libm-neon`,
  `no-std-qemu`).

## Gate

The five CI jobs above. No `properties.yaml` entry: they verify the build and
math-backend configuration, not a numerical property.
