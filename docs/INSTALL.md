# Installation

## Prerequisites

Install Rust via [rustup](https://rustup.rs/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

MSRV: **1.78**. The crate includes a `rust-toolchain.toml` that pins the
toolchain automatically when building from the repository.

## Adding to a project

```toml
[dependencies]
semiflow-core = "9"
```

Default features include `simd` (AVX2/NEON auto-selected; scalar fallback on
other architectures). The core is `no_std + alloc`; `simd` and `parallel`
require `std`.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `simd` | on | Manual `#[target_feature]` AVX2 (x86_64) / NEON (aarch64) paths for Hermite interpolants and FD stencils; scalar fallback on other targets (ADR-0019) |
| `std` | off | Enables `std::error::Error` impl on `SemiflowError`; required by `simd` and `parallel` |
| `parallel` | off | Parallel `Strang2D::apply` via `std::thread::scope` — no external dep (ADR-0018) |
| `linear-interp` | off | Enables `InterpKind::Linear` (2-point fallback interpolation) |
| `diff-scipy` | off | Gates Python/scipy-dependent tests (CI only) |
| `slow-tests` | off | Gates flagship slope-rate convergence tests (prod hardware only) |
| `tracking-alloc` | off | Enables allocation tracking instrumentation for zero-alloc gate verification |

## API documentation

```sh
cargo doc --open -p semiflow-core
```

Published docs: https://docs.rs/semiflow-core

See [`crates/semiflow-core/README.md`](../crates/semiflow-core/README.md) for a
full type catalogue, usage examples, and the math reference (Theorem 6,
Remizov 2025).
