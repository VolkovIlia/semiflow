# Installation

The package names, install commands and the current version for every language
(Rust, Python, JavaScript/WASM, C) are listed in the
[project README](../README.md#install). This page covers the Rust toolchain and
where to find the build options.

## Prerequisites

Install Rust via [rustup](https://rustup.rs/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

MSRV: **1.81**. The repository includes a `rust-toolchain.toml` that pins the
toolchain automatically when building from a checkout.

## Adding to a project

```sh
cargo add semiflow
```

The default build is `#![no_std]` + `alloc` with `simd` (AVX2/NEON
auto-selected; scalar lanes on other architectures). `std` is needed only for
`parallel`.

- Feature flags (`simd`, `std`, `parallel`, `linear-interp`, …): see
  [README § Feature flags](../README.md#feature-flags).
- Embedded / bare-metal use (the default: `#![no_std]` + `alloc`):
  see [README § no_std](../README.md#no_std).

## Bindings

Python (`semiflow-pde`), JavaScript/WASM (`@semiflow/wasm`) and C
(`semiflow-ffi`, built from source) install as described in
[README § Install](../README.md#install) and
[README § C / C++](../README.md#c--c); the [Bindings guide](BINDINGS.md) has
usage examples for each.

## API documentation

```sh
cargo doc --open -p semiflow
```

Published docs: https://docs.rs/semiflow
