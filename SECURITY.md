# Security Policy

SemiFlow ships four packages: the Rust crate `semiflow` (crates.io; rlib,
pure math, no I/O, no networking), the Python wheel `semiflow-pde` (PyPI;
PyO3 via maturin, abi3-py310, crate `semiflow-py`), the npm package
`@semiflow/wasm` (`wasm-bindgen`, crate `semiflow-wasm`), and the C ABI
`semiflow-ffi` (cdylib/staticlib + cbindgen header, built from source — no
prebuilt binaries are distributed).
The library performs no network I/O, authentication, or PII handling. The
relevant attack surface is **boundary safety** (panic propagation, NULL
pointers, memory ownership at FFI / Python / WASM edges) plus **supply-chain**
integrity of dependencies.

## Supported Versions

SemiFlow is in beta (`0.x`). Only the latest minor release line receives
security fixes; fixes ship as a new patch or minor release of all four
packages (they share one version number).

| Version  | Status      |
|----------|-------------|
| 0.13.x   | Supported   |
| < 0.13   | Unsupported — upgrade to the latest 0.13.x |

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

- Memory unsafety in `unsafe` blocks (`crates/semiflow-{ffi,py,wasm}/src/**`,
  `crates/semiflow/src/simd/**`; ADR-0019, `xtask check-unsafe-scope`).
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

- **crates.io**: `cargo update -p semiflow`. Security advisories on
  RustSec (<https://rustsec.org>).
- **PyPI**: `pip install --upgrade semiflow-pde`. Security notices via
  maintainer email if a PyPI advisory is filed.
- **npm**: `npm update @semiflow/wasm`.
- **C ABI** (`semiflow-ffi`): rebuild from the latest tagged source.
- **GitHub releases**: <https://github.com/VolkovIlia/semiflow/releases>
  (use *Watch -> Custom -> Releases*).
