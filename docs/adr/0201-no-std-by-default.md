# ADR-0201 — `no_std` is the default build

- **Status**: Proposed
- **Date**: 2026-09-29
- **Supersedes / amends**: ADR-0003 (`no_std + alloc` core), ADR-0019 (`simd`
  implied `std`), ADR-0200 (verification jobs now test the default build).
- **Contract**: no `math.md` change. Feature table: `default = ["simd"]`,
  `simd = []`, `std` opt-in; MSRV 1.78 → 1.81.

## Context

`no_std + alloc` is the library's main property (ADR-0003), yet the default
feature set was `["simd"]` with `simd = ["std"]`: `semiflow = "…"` produced a
`std` build and `no_std` needed `default-features = false`. Nothing in `simd`
needs `std`. The AVX2/NEON intrinsics come from `core::arch`; the only `std`
item was the `thread_local!` behind the `FORCE_SCALAR` test hook. After
ADR-0200 the math is `libm` in every build, so `std` no longer changes
results either. What still needs `std`:

- `impl std::error::Error for SemiflowError`;
- `parallel` (`std::thread::scope`).

## Decision

1. **`simd = []`; `default = ["simd"]`.** The default build is `#![no_std]` +
   `alloc` with SIMD kernels. `std` is opt-in and `parallel = ["std"]`.
2. **`impl core::error::Error for SemiflowError`, unconditionally.**
   `core::error::Error` is stable since Rust 1.81 and is the same trait as
   `std::error::Error`, so `?` into `Box<dyn Error>` keeps working in `std`
   programs without enabling `std`. MSRV goes from 1.78 to 1.81. Rejected:
   keeping MSRV 1.78 and the `std`-only impl. That breaks every downstream
   crate that relies on `SemiflowError: Error` without naming the `std`
   feature.
3. **`FORCE_SCALAR` is a unit-test-only hook.** It exists under `cfg(test)`,
   where the test harness links `std`. `simd::with_force_scalar` stays public,
   but outside the library's own unit tests it only runs its closure. That
   matches what it already did for integration tests (ADR-0200, Honest
   limits).
4. **Removing the `std` feature entirely** was rejected. It is still needed for
   `parallel`, and removing it would break manifests that name it.

## Consequences

- `semiflow = "…"` is a `no_std` build. Users who want `parallel` add
  `features = ["parallel"]`, as before. `std` alone changes nothing observable
  except linking `std`.
- The binding crates already list their features explicitly and are
  unaffected.
- CI: `no-std` builds the default configuration for the host and bare-metal
  targets and checks it at MSRV. `no-std-test` runs the full suite with
  `cargo test -p semiflow` (default features). `-p` keeps the bindings from
  unifying `std`/`parallel` into the library, as they do in the
  workspace-wide `test` job. `semiflow-nostd-check` depends on `semiflow` with
  default features, so the digests (ADR-0200) are taken from the default
  build. `std-ref` switches on `std` + `simd`.
- Breaking change for downstream code that used `std`-only items through the
  default features without naming `std`. After decision 2, the only such item
  is the implicit `num-traits/std` feature (ADR-0200). It affects only callers
  of `num_traits::Float` who relied on semiflow to enable it.

## Gate

The ADR-0200 jobs, now in the default configuration. No `properties.yaml`
entry.
