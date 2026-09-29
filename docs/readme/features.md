<!--
Feature-flag descriptions for the generated READMEs (`cargo xtask readme`).
One row per feature of crates/semiflow/Cargo.toml `[features]` (except `default`).
The generator fails if this list and Cargo.toml disagree, and fills the
"Default" column from Cargo.toml itself.
-->
| `simd` | AVX2 (x86_64) / NEON (aarch64) kernels via `core::arch`, scalar lanes elsewhere. Works in `no_std`. Results are bit-identical to a non-`simd` build. |
| `std` | Links `std`. Needed only by `parallel`; math is `libm` either way, so results do not change. Without it the crate is `no_std + alloc` (the default). |
| `parallel` | Multi-threaded `apply` for 1D kernels and 2D/3D compositions via `std::thread::scope`; bit-identical across thread counts. Implies `std`; not for `wasm32`. |
| `linear-interp` | Enables `InterpKind::Linear` grid sampling (otherwise it returns `SemiflowError::Unsupported`). |
| `s3-poc` | Experimental S³ evolvers with a proven-boundary API (ADR-0169). |
| `slow-tests` | Internal: enables the long convergence gates. Not needed by users. |
| `diff-scipy` | Internal: SciPy differential test (needs Python). Not needed by users. |
| `tracking-alloc` | Internal: allocation counting in the `latency_tail` example. Not needed by users. |
