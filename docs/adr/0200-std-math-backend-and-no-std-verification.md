# ADR-0200 — Ship the math backend the tests verify, and prove `no_std` by execution

- **Status**: Proposed
- **Date**: 2026-09-28
- **Supersedes / amends**: ADR-0003 (`no_std + alloc` core), ADR-0025 (libm for
  `Float` in `no_std`) — the `std` feature wiring and the verification contract.
- **Contract**: no `math.md` change. New CI jobs `no-std`, `no-std-test`,
  `no-std-libm`, `no-std-qemu`; no new `properties.yaml` gate.

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

1. **`std = ["num-traits/std", "num-complex/std"]`.** A `std` build now uses the
   platform math library, which is exactly what the test suite has always
   measured. Rejected: keeping `std = []` and testing the `libm` path for `std`
   users too. That leaves the shipped configuration different from the one
   every golden and parity test runs, which is the root problem.
2. **`no_std` is verified in four independent ways**, all on every pull request:
   - `no-std`: `cargo build --no-default-features` on the host,
     `thumbv7em-none-eabihf`, `thumbv7m-none-eabi`, and at MSRV.
   - `no-std-test`: the whole `semiflow` test suite (unit + integration) with the
     crate compiled `#![no_std]`. It exercises the `no_std` code paths. It does
     not exercise `libm`, because the test harness links `std`.
   - `no-std-libm`: `crates/semiflow-nostd-check`, a `#![no_std]` crate whose
     only dependency is `semiflow` without default features, runs closed-form
     oracle scenarios. The job asserts from `cargo tree` that nothing enables
     `num-traits/std`, so the math provably goes through `libm`.
   - `no-std-qemu`: `nostd-qemu/` links the same scenarios into a
     `#![no_main]` bare-metal binary for Cortex-M3 (soft float) and Cortex-M4F
     (hard float) and runs them under QEMU. The exit code is the verdict.
   Rejected: a `cargo check` only (proves nothing about behaviour), and adding
   `libm` tests inside `semiflow` itself (impossible, because the dev-dependencies
   switch num-traits to `std` by feature unification).
3. **Targets without pointer-sized atomics fail with an explicit
   `compile_error!`.** The engines share coefficient closures through
   `alloc::sync::Arc`. Rejected: `portable-atomic`, which would add a fourth
   runtime dependency (the cap is three).

## Consequences

- For `std` users, generic transcendentals move from `libm` to the platform
  library. Results can change in the last bits compared with 0.13.1-beta; they
  now match what the tests and the binding parity goldens were computed with.
  No gate threshold changes.
- CI gains three jobs. The QEMU job installs `qemu-system-arm` from apt.
- `nostd-qemu/` is outside the workspace (a `#![no_main]` binary cannot build
  for the host), with its own `Cargo.lock`.

## Honest limits

- The QEMU run executes a curated set of scenarios, not the full test suite.
  The full suite runs `no_std`-compiled but with `std` math.
- QEMU checks instruction-set and ABI correctness, not the timing or memory
  limits of any particular board.
- `thumbv6m` and other targets without atomic CAS stay unsupported.

## Gate

The four CI jobs above. No `properties.yaml` entry: they verify the build and
math-backend configuration, not a numerical property.
