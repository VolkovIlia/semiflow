# Security Policy

`semiflow` ships four packages: `semiflow-core` (rlib, pure math, no I/O,
no networking), `semiflow-ffi` (C ABI cdylib via cbindgen), `semiflow-py`
(PyO3 wheel via maturin, abi3-py310), and `semiflow-wasm` (`wasm-bindgen`).
The library performs no network I/O, authentication, or PII handling. The
relevant attack surface is **boundary safety** (panic propagation, NULL
pointers, memory ownership at FFI / Python / WASM edges) plus **supply-chain**
integrity of dependencies.

## Supported Versions

| Version | Status                                             |
|---------|----------------------------------------------------|
| 9.x     | Supported                                          |
| 8.x     | Security fixes only (90 days post v9.0.0 release)  |
| < 8.0   | Unsupported                                        |

## Reporting a Vulnerability

- Email **ilia.volkov@outlook.com**. Do **not** open public GitHub issues for
  security reports.
<!-- TODO: publish maintainer GPG fingerprint once uploaded to a keyserver. -->
- Acknowledgement within **72 hours**.
- Initial assessment within **7 days**.
- Coordinated disclosure window: **90 days** (negotiable for critical vulns).
- Credit: reporter named in `CHANGELOG.md` and release notes unless they
  prefer otherwise.

## Scope

**In-scope (we will fix):**

- Memory unsafety in `unsafe` blocks (`crates/remizov-{ffi,py,wasm}/src/**`,
  `crates/semiflow-core/src/simd/**`; ADR-0019, `xtask check-unsafe-scope`).
- Panics that escape an `extern "C"` boundary (FFI). Note: `wasm-bindgen`
  deliberately routes panics to JS via `__wbindgen_throw` — this is documented
  behaviour, not a vulnerability.
- Undefined behaviour from misuse of public APIs that we should reject (NULL
  deref, double-free of opaque handles, malformed numpy buffers — provided
  the input matches the documented type contract; wrong Python type is a
  Python issue, not ours).
- Supply-chain: vulnerable transitive dependency that affects shipped crates.
- Build-system issues (e.g. malicious crates.io fork; we will triage).

**Out-of-scope (these are not vulnerabilities):**

- Numerical instability or low convergence at extreme parameter regimes —
  use the issue tracker.
- Performance regressions — use the issue tracker.
- Crashes from values explicitly forbidden by `SemiflowError::DomainViolation`
  (e.g. negative `t`). The library reports these as errors (Rust `Result`,
  FFI status code, Python exception, JS error) per the documented contract.
- Vulnerabilities in your environment or unrelated dependencies (e.g. NumPy
  CVEs).
- Attacks requiring control of the build pipeline or local filesystem.

## Hardening Defaults

- All FFI entry points wrap in `catch_unwind` and translate panics to
  `SemiflowStatus::Panic` (status code 99). `[profile.release-ffi]` sets
  `panic = "unwind"` so `catch_unwind` is effective (ADR-0028).
- PyO3 boundary: `Heat1D.evolve` releases the GIL via `py.detach`
  (ADR-0031); `Send + Sync` is verified at compile time with
  `static_assertions`.
- WASM: `console_error_panic_hook` available via `panic_hook_init()` for
  dev; production builds use workspace `panic = "abort"` (ADR-0028 Am. 1).
- Workspace dependency licensing enforced by `deny.toml` (allowlist;
  `unlicensed = deny`).
- MSRV pinned at Rust 1.78.

## Update Channels

- **crates.io**: `cargo update -p semiflow-core`. Security advisories on
  RustSec (<https://rustsec.org>).
- **PyPI**: `pip install --upgrade semiflow-py`. Security notices via
  maintainer email if a PyPI advisory is filed.
- **npm**: `npm update @semiflow/wasm`.
- **GitHub releases**: <https://github.com/VolkovIlia/semiflow/releases>
  (use *Watch -> Custom -> Releases*).
