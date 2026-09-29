<!--
SINGLE SOURCE of every published README. Edit this file (and the fragments next
to it), then run `cargo xtask readme`; CI runs `cargo xtask readme --check`.

Generated files and where they are published:
  github -> README.md                      (GitHub front page)
  rust   -> crates/semiflow/README.md       (crates.io and the docs.rs front page)
  pypi   -> crates/semiflow-py/README.md    (PyPI, package `semiflow-pde`)
  npm    -> crates/semiflow-wasm/README.md  (npm, package `@semiflow/wasm`)

Directives are HTML comments on a line of their own:
  "only: github rust" ... "/only"   keep the enclosed lines for those channels only
  "include: file.md"                splice a fragment from docs/readme/
  "readme-test: skip ..."           the next code block is not executed by CI
Placeholders in double braces (version, pep440_version, msrv, runtime_dep_count,
runtime_deps, features_table, py_class_count, py_functions, wasm_lite_table,
wasm_full_table) are computed from the manifests and sources.
Relative links become absolute GitHub URLs in every channel except github.
Rust code blocks are doctests of the `semiflow` crate; Python and Node.js code
blocks are executed in CI (py-test-fast, wasm-test-node).
-->
<!-- only: github -->
# SemiFlow

[![CI](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/semiflow)](https://crates.io/crates/semiflow)
[![Docs.rs](https://img.shields.io/docsrs/semiflow)](https://docs.rs/semiflow)
[![PyPI](https://img.shields.io/pypi/v/semiflow-pde)](https://pypi.org/project/semiflow-pde/)
[![npm](https://img.shields.io/npm/v/@semiflow/wasm)](https://www.npmjs.com/package/@semiflow/wasm)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20837851.svg)](https://doi.org/10.5281/zenodo.20837851)
<!-- /only -->
<!-- only: rust -->
# semiflow

[![CI](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/semiflow)](https://crates.io/crates/semiflow)
[![Docs.rs](https://img.shields.io/docsrs/semiflow)](https://docs.rs/semiflow)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20837851.svg)](https://doi.org/10.5281/zenodo.20837851)
<!-- /only -->
<!-- only: pypi -->
# semiflow-pde — Python bindings for SemiFlow

[![CI](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/semiflow-pde)](https://pypi.org/project/semiflow-pde/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20837851.svg)](https://doi.org/10.5281/zenodo.20837851)
<!-- /only -->
<!-- only: npm -->
# @semiflow/wasm — WebAssembly bindings for SemiFlow

[![CI](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@semiflow/wasm)](https://www.npmjs.com/package/@semiflow/wasm)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20837851.svg)](https://doi.org/10.5281/zenodo.20837851)
<!-- /only -->

> **Status: beta ({{version}}).** The API is stabilising toward 1.0; minor
> versions may still make breaking changes. Bug reports and feedback are welcome.

**SemiFlow solves evolution equations `∂ₜu = Lu` (heat, diffusion,
Schrödinger, graph and manifold PDEs, …) by Chernoff approximation of operator
semigroups: no matrix exponentials, no linear solves, a flat memory footprint.**
The core is a `no_std + alloc` Rust library with {{runtime_dep_count}} runtime
dependencies ({{runtime_deps}}); the same engines are available from Python,
JavaScript/WebAssembly and C.

<!-- only: pypi -->
This package is the Python binding (PyO3, abi3 wheels for CPython ≥ 3.10) of
the Rust crate [`semiflow`](https://crates.io/crates/semiflow). Every class
below runs the Rust engine; the GIL is released during `evolve`.
<!-- /only -->
<!-- only: npm -->
This package is the WebAssembly binding (wasm-bindgen) of the Rust crate
[`semiflow`](https://crates.io/crates/semiflow), for Node.js (CommonJS) and
browsers/bundlers (ES modules). TypeScript declarations are included.
<!-- /only -->

The method evaluates `e^{tL}f` by iterating an explicit step operator `S(τ)`:
`(S(t/n))ⁿ f → e^{tL} f` as `n → ∞`. Each step is an explicit stencil over a
`Vec`-backed grid with reused scratch buffers, so steady-state evolution does
not allocate.

> Mathematical foundation: **Theorem 6 of I. D. Remizov (2025)**, *Vladikavkaz
> Math. J.* 27(4), 124–135 ([doi:10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)).

> **Honest performance note.** The primary measured advantage is **memory
> frugality**: a flat ~3 MB working set across the 1D/2D/3D families versus
> 50–418 MB for heavy frameworks. Wall-clock speed is **not** a general
> strength — for matched-accuracy PDE solving SemiFlow is slower than adaptive
> ODE solvers and spectral methods, and parallelism pays off only on large 3D
> grids. Two measured niches: tail-latency-sensitive pricing
> (`Diffusion4thChernoff` ≈ 41 ns p99.9 per tick) and L1-resident low-rank
> carriers (`TtChernoff`, `ReverseChernoff`). Treat memory and latency as
> measured properties of the concrete grid types, not guarantees of the traits.

## Install

<!-- only: github -->
| Language | Package | Install |
|----------|---------|---------|
| Rust | [`semiflow`](https://crates.io/crates/semiflow) | `cargo add semiflow` |
| Python ≥ 3.10 | [`semiflow-pde`](https://pypi.org/project/semiflow-pde/) | `pip install semiflow-pde`, then `import semiflow` |
| JavaScript / WASM | [`@semiflow/wasm`](https://www.npmjs.com/package/@semiflow/wasm) | `npm install @semiflow/wasm` |
| C / C++ | `semiflow-ffi` (built from source) | see [C / C++](#c--c) |

<!-- /only -->
<!-- only: github rust -->
```toml
[dependencies]
semiflow = "{{version}}"
```

MSRV: **Rust {{msrv}}**. The default build is `#![no_std]` + `alloc` with
AVX2/NEON kernels (feature `simd`, scalar lanes elsewhere); `std` is needed
only for `parallel`. See [below](#no_std).

<!-- /only -->
<!-- only: pypi -->
```sh
pip install semiflow-pde
```

The import name is `semiflow`. Wheels are abi3 (one wheel per platform covers
CPython 3.10+). To build from a source checkout instead (needs a Rust toolchain):

```sh
pip install maturin
maturin develop --release -m crates/semiflow-py/Cargo.toml
```

<!-- /only -->
<!-- only: npm -->
```sh
npm install @semiflow/wasm
```

Node.js ≥ 18 loads the CommonJS build (`require`); bundlers and browsers load
the ES-module build (`import`). To build from a source checkout (needs a Rust
toolchain and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/)):

```sh
wasm-pack build crates/semiflow-wasm --target nodejs --out-dir pkg-node
wasm-pack build crates/semiflow-wasm --target web --out-dir pkg-web
# heavy-grid engines (see "Full build" below):
wasm-pack build crates/semiflow-wasm --target web --out-dir pkg-web -- --features full
```

<!-- /only -->
## Quickstart

<!-- only: github rust -->
Solve `∂ₜu = ½·∂ₓₓu` from `u₀(x) = e^{-x²}` to `t = 1` and compare with the
closed-form solution `u(1,x) = 3^{-1/2}·e^{-x²/3}`:

```rust
use semiflow::{ChernoffSemigroup, DiffusionChernoff, Grid1D, GridFn1D, State};

let grid = Grid1D::new(-10.0, 10.0, 1000).expect("valid grid");
let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

// L = a(x)·∂²ₓ with a ≡ ½ (a' = a'' = 0), bound ‖a‖ = ½.
let diffusion = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, grid);
let semigroup = ChernoffSemigroup::new(diffusion, 100).expect("n >= 1");
let u1 = semigroup.evolve(1.0, &u0).expect("evolve");

let exact = GridFn1D::from_fn(grid, |x| (-x * x / 3.0).exp() / 3.0_f64.sqrt());
let mut err = u1.clone();
err.axpy(-1.0, &exact);
assert!(err.norm_sup() < 1e-5, "sup-norm error {}", err.norm_sup());
```

More: [Quickstart](docs/QUICKSTART.md), [User Guide](docs/USER_GUIDE.md) and
the runnable [examples](crates/semiflow/examples/README.md)
(`cargo run -p semiflow --example heat_2d_demo`).

<!-- /only -->
<!-- only: pypi -->
<!-- include: quickstart-python.md -->
<!-- /only -->
<!-- only: npm -->
<!-- include: quickstart-js.md -->
<!-- /only -->
<!-- only: github rust -->
<a id="no_std"></a>
## `no_std`

```toml
[dependencies]
semiflow = "{{version}}"   # no_std + alloc, SIMD kernels
```

The crate is `#![no_std]` by default and needs only `alloc`; the `std`
feature is needed only for `parallel`. What that means in practice:

- Your binary provides a `#[global_allocator]` and a `#[panic_handler]`, as
  for any `no_std + alloc` program (a `std` program already has both).
- The target needs pointer-sized atomics (the engines share coefficient
  closures through `alloc::sync::Arc`): Cortex-M3/M4/M7/M33, RISC-V with the
  `a` extension, `x86_64-unknown-none` and similar work; `thumbv6m` does not.
- `SemiflowError` implements `Debug`, `Display` and `core::error::Error`
  (the same trait as `std::error::Error`) in every build.
- `f32`/`f64` transcendentals come from the pure-Rust [`libm`](https://crates.io/crates/libm)
  in every build, and the SIMD kernels use the same lane arithmetic with or
  without intrinsics, so `std`, `no_std`, `simd` and non-`simd` builds give
  bit-identical results on every CPU.
- `simd` works without `std` (the intrinsics come from `core::arch`);
  `parallel` requires `std`.

This is verified on every pull request, not just type-checked:

| CI job | What it proves |
|--------|----------------|
| `no-std` | the default and the featureless build for the host, `thumbv7em-none-eabihf`, `thumbv7m-none-eabi`, `riscv32imac-unknown-none-elf`, and at MSRV {{msrv}} |
| `no-std-test` | the full `semiflow` test suite passes in the default (`no_std`) configuration |
| `no-std-libm` | [`semiflow-nostd-check`](crates/semiflow-nostd-check) — closed-form oracle checks plus per-scenario output digests: the default `no_std` build (the job fails if anything enables `num-traits/std`), the `std` build and its AVX2 lanes must all produce the committed bits |
| `no-std-libm-neon` | the same digests from the `std` build on aarch64 (NEON lanes) |
| `no-std-qemu` | the same checks and digests as a bare-metal `#![no_main]` binary on Cortex-M3 (soft float) and Cortex-M4F (hard float) under QEMU — see [`nostd-qemu`](nostd-qemu/README.md) |

## Feature flags

{{features_table}}

<!-- include: engines.md -->
<!-- /only -->
<!-- only: pypi -->
<!-- include: python-api.md -->
<!-- /only -->
<!-- only: npm -->
<!-- include: wasm-api.md -->
<!-- /only -->
## Bindings

| Language | Package | Notes |
|----------|---------|-------|
| Rust | [`semiflow`](https://crates.io/crates/semiflow) ([docs.rs](https://docs.rs/semiflow)) | The full engine catalogue; `no_std + alloc` |
| Python | [`semiflow-pde`](https://pypi.org/project/semiflow-pde/) | {{py_class_count}} classes and the functions {{py_functions}}; NumPy in/out; complete `.pyi` stubs; see the [PyPI page](https://pypi.org/project/semiflow-pde/) |
| JavaScript / WASM | [`@semiflow/wasm`](https://www.npmjs.com/package/@semiflow/wasm) | Lite build on npm; the heavy-grid engines need a `--features full` build; see the [npm page](https://www.npmjs.com/package/@semiflow/wasm) |
| C / C++ | `semiflow-ffi` | `extern "C"` ABI with `catch_unwind` on every entry point; header `semiflow.h` |

The bindings mirror the user-facing engines, not the internal composition types
(`AxisLift`, `StrangSplit`, … are Rust-side building blocks). Variable
coefficients cross the language boundary as pre-sampled arrays (evaluated in
Rust, no call back into the host language during `evolve`); some classes also
accept a host-language callback (e.g. `Heat1D.with_a_function` in Python,
`Heat1D.withAFunction` in JS, `smf_state_new_with_closure` in C).
The [Bindings guide](docs/BINDINGS.md) has side-by-side examples.

<!-- only: github -->
### C / C++

The C ABI ships as source; build it with:

```sh
# `release-ffi` keeps panic = "unwind", which the catch_unwind boundary needs;
# the plain `release` profile aborts on panic.
cargo build -p semiflow-ffi --profile release-ffi
# library: target/release-ffi/libsemiflow_ffi.{so,dylib,a} / semiflow_ffi.dll
# header:  crates/semiflow-ffi/include/semiflow.h
```

See [`crates/semiflow-ffi/README.md`](crates/semiflow-ffi/README.md).

<!-- /only -->
## Documentation

| If you want to… | Read |
|-----------------|------|
| Get started | [Quickstart](docs/QUICKSTART.md) · [User Guide](docs/USER_GUIDE.md) · [Install](docs/INSTALL.md) |
| Use a binding | [Bindings guide](docs/BINDINGS.md) (C / Python / WASM) |
| Browse the Rust API | [docs.rs/semiflow](https://docs.rs/semiflow) |
| See worked examples | [`crates/semiflow/examples`](crates/semiflow/examples/README.md) |
| Understand accuracy and stability policy | [Precision policy](docs/precision-policy.md) · [API stability](docs/api-stability.md) |
| Read design decisions | [Architecture Decision Records](docs/adr/) |
| Follow changes | [CHANGELOG](CHANGELOG.md) · [GitHub Releases](https://github.com/VolkovIlia/semiflow/releases) |

## Accuracy and design

Every numerical claim is gated in CI against a closed-form or high-order
reference oracle (convergence-order and sup-norm tests). Design principles:
`no_std + alloc` core; {{runtime_dep_count}} runtime dependencies; SIMD hot
paths isolated to `src/simd/` (AVX2 on x86_64, NEON on aarch64, scalar
fallback elsewhere); bit-reproducible parallelism across thread counts; no
`unsafe` in the math kernels. Builds with and without `simd` agree to rounding
but not always bit for bit, so golden files are tied to one configuration.

## How to cite

If you use SemiFlow in academic work, cite both the software and the theorem.

```bibtex
@software{volkov2026semiflow,
  author  = {Volkov, Ilia},
  title   = {{SemiFlow}: {Chernoff} Approximation of Operator Semigroups},
  year    = {2026},
  version = {{{version}}},
  doi     = {10.5281/zenodo.20837851},
  url     = {https://doi.org/10.5281/zenodo.20837851}
}

@article{Remizov2025,
  author  = {I. D. Remizov},
  title   = {Chernoff Approximations of the Solution of Linear ODE with Variable Coefficients},
  journal = {Vladikavkaz Math. J.},
  volume  = {27},
  number  = {4},
  pages   = {124--135},
  year    = {2025},
  doi     = {10.46698/a3908-1212-5385-q}
}
```

The concept DOI [10.5281/zenodo.20837851](https://doi.org/10.5281/zenodo.20837851)
always resolves to the latest version; [`CITATION.cff`](CITATION.cff) has the
full entry (GitHub's "Cite this repository" reads it).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option.

## Author, contributing, security

SemiFlow is created and maintained by **Ilia Volkov**. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md) —
not in public issues.
