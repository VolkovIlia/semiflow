# Contributing to semiflow-core

Thank you for contributing. This guide covers the full workflow for the semiflow-core workspace.

## Repository Layout

Four-crate workspace:

| Crate | Kind | Purpose |
|-------|------|---------|
| `crates/semiflow-core` | `rlib` | Core math: Chernoff operator semigroup approximations |
| `crates/semiflow-ffi` | `cdylib` (C ABI) | C-facing bindings; design in ADR-0028 |
| `crates/semiflow-py` | PyO3 wheel | Python bindings via maturin |
| `crates/semiflow-wasm` | wasm-bindgen | WebAssembly bindings |

Key directories:

- `docs/adr/` — 155 Architecture Decision Records. To add one: copy
  `docs/adr/0000-template.md`, increment the number, keep rationale ≤1 paragraph.
- `contracts/semiflow-core.math.md` — normative mathematical specification
  (4554 lines). Updates require a companion ADR.
- `xtask/` — workspace task runner (replaces Make/shell scripts).

## MSRV & Toolchain

Minimum Supported Rust Version: **1.78**.

An active issue with `clap_lex`/edition2024 requires a stable-toolchain
workaround when invoking xtask commands:

```bash
RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- <command>
```

The `rust-toolchain.toml` at repo root pins the default channel; override only
when the toolchain resolver misbehaves with nightly features.

## Build & Test Matrix

```bash
cargo run -p xtask -- test-fast      # opt-level=2, ~5-10× faster; default for PR iteration
cargo run -p xtask -- test-full      # native SIMD + parallel + slow-tests, release mode
cargo run -p xtask -- test-flagship  # G3⁶-2D + G4_NS2D_aniso + G5_3D slope gates (minutes)
```

Why the extra flags in `test-full` and `test-flagship`:

- `RUSTFLAGS="-C target-cpu=native"` — activates AVX2/NEON SIMD hot paths.
- `--features parallel` — `Strang2D` uses `std::thread::scope` across all cores.
- `--features slow-tests` — includes CEV sweeps and the G3⁶-2D flagship gate.

Flagship tests run on production hardware before each release.

### CI gate map

Two workflow files cover the full gate set:

| Workflow | Trigger | Gates |
|----------|---------|-------|
| `ci.yml` | every push + PR | fmt, clippy, test (fast), doc, suckless, **unsafe-scope** (NEW), deny, coverage, ffi/py/wasm builds and smokes |
| `flagship-gates.yml` | nightly + `workflow_dispatch` | **all RELEASE_BLOCKING slow gates** (see below) |

`unsafe-scope` (C-C3 fix) — `cargo run -p xtask -- check-unsafe-scope` — is now
a **blocking CI job** on every push/PR.  It exits 0 since the ADR-0019 allowlist
was last audited.

#### Flagship-gates job breakdown

| Job | Gate IDs | Mechanism |
|-----|----------|-----------|
| `flagship-slope-gates` | G3⁶-2D, G4\_NS2D\_aniso, G5\_3D | `xtask test-flagship` (Pattern A: `#![cfg(feature="slow-tests")]`) |
| `anisotropic-ddim-gates` | G\_DDIM D=2–5 | Pattern A: named `--test` binaries |
| `zeta-truthful-order-gates` | G\_zeta4\_TRUTHFUL\_ORDER, G\_zeta4/6/8 correction slopes | Pattern B: `--features slow-tests -- --ignored` |
| `magnus-schrodinger-gates` | G17, G18 | Pattern B |
| `hormander-quantum-gates` | Hörmander Engel/Heisenberg/Kolmogorov, quantum graph | Pattern B (Pattern A + `#[ignore]`) |
| `misc-slow-gates` | Robin boundary, subordinated order-1, resolvent | Pattern B |
| `latency-gate` | L\_CEV\_PTICK | `xtask latency-gate --all` (advisory on hosted runner; blocking on self-hosted i7-12700K) |

**Pattern A** = `#![cfg(feature = "slow-tests")]` at file level, no `#[ignore]`.
Needs `--features slow-tests`; `-- --ignored` is NOT passed.

**Pattern B** = `#[ignore = "..."]` per test (may or may not also have Pattern A).
Needs both `--features slow-tests` AND `-- --ignored`.

#### Running flagship gates locally

```bash
# The three main slope gates (test-flagship):
cargo run -p xtask -- test-flagship

# All ignored slow gates:
RUSTFLAGS="-C target-cpu=native" cargo test --workspace \
  --features parallel,simd,slow-tests --release -- --ignored --nocapture

# Latency gate (prod hardware recommended):
cargo run -p xtask -- latency-gate --all
```

Note: GitHub Actions billing may be paused; local validation on `i7-12700K` (or
equivalent) is the primary gate before each release.  The `flagship-gates.yml`
workflow exists to make the gates _runnable_ in CI even if hosted-runner timing
differs from the calibration hardware.

Lints:

```bash
cargo run -p xtask -- check-lints        # suckless budget checks
cargo run -p xtask -- check-unsafe-scope # ADR-0019 unsafe allowlist (fast, always green)
cargo clippy --all-targets --all-features -- -D warnings
```

Bindings smoke tests:

```bash
cargo run -p xtask -- ffi-smoke    # compiles heat.c against generated header
cargo run -p xtask -- py-smoke     # maturin develop + pytest
cargo run -p xtask -- wasm-test    # wasm-bindgen-test on Node (--chrome for browser)
```

## Suckless Budgets (Hard)

| Constraint | Limit |
|------------|-------|
| Lines per function | ≤ 50 |
| Lines per file | ≤ 500 |
| Dependencies per crate | < 10 |

**Grandfathered exceptions** (pre-v0.10.0): `grid.rs`, `diffusion6.rs`,
`truncated_exp4.rs`. Do not push these files past their current size; do not
add new grandfather exceptions without an ADR.

`unsafe` code is only permitted in:

- `crates/remizov-{ffi,py,wasm}/src/**`
- `crates/semiflow-core/src/simd/**`

Scope is enforced by `cargo run -p xtask -- check-unsafe-scope` (ADR-0019).
New `unsafe` outside these paths requires an ADR.

## Adding a New ChernoffFunction Type

1. Add a module file: `crates/semiflow-core/src/<name>.rs`.
2. Implement `ChernoffFunction<F: SemiflowFloat = f64>` — methods `apply`,
   `order`, `growth`.
3. Re-export from `crates/semiflow-core/src/lib.rs`.
4. Add a unit test in the same file (≤50 LoC).
5. If order > 2 or the type introduces 2D/3D coupling: write an ADR and add a
   derivation in `contracts/semiflow-core.math.md` §X.Y.
6. Add an entry to the `Unreleased` section of `CHANGELOG.md`.

## Contract-First & Math Fidelity

New mathematical content lives in `contracts/semiflow-core.math.md`. Pull
requests must link the relevant section in the PR description.

Since v0.5.0 every PDE family has a sympy gate under
`crates/semiflow-core/sympy/T*N_*.py`. The gate must pass before merge:

```bash
python crates/semiflow-core/sympy/<gate>.py
```

For audit reference, see `docs/audit-findings-v0_10_0.md` and
`docs/audit-findings-v0_9_0.md` for the v0.11.0 audit pattern.

## ADR Process

Files live at `docs/adr/NNNN-short-title.md`. Rationale must be ≤1 paragraph.

**MAJOR architectural decisions** (foundational policies, multi-version commitments,
or decisions with amendment chains) MAY exceed the ≤1-paragraph rule. Examples in
the current corpus: ADR-0015 (6th-order spatial), ADR-0020 (G3⁶-2D FLAGSHIP gate),
ADR-0025 (generic-over-Float), ADR-0028 (FFI/PyO3/WASM bindings split), ADR-0035
(v1.0.0 API stability). Such ADRs SHOULD carry a `severity: major` line in their
status frontmatter so reviewers can identify them at a glance. Routine ADRs remain
bound by the ≤1-paragraph rule.

Steps for a new ADR:

1. Check the highest existing number: `ls docs/adr/ | tail -1`.
2. Copy `docs/adr/0000-template.md`; increment the number.
3. Set Status: `Proposed`.
4. After merge set Status: `Accepted`. If later superseded, mark
   `Superseded by ADR-XXXX`.
5. For milestone-affecting decisions, link the ADR from `ROADMAP.md`.

## Pull Request Workflow

- Branch from `master` (single-trunk development).
- Use conventional commits: `type(scope): description`
  (`feat`, `fix`, `docs`, `chore`, `perf`, `refactor`, `test`).
- Required commit trailers at end of commit body:
  ```
  Agent: <agent-name-or-human>
  Task-ID: <kebab-case-id>
  ```
  Bug-fix commits additionally require:
  ```
  Fixes-Agent: <agent-that-introduced-bug>
  Fixes-Commit: <sha>
  ```
- CI must be green: lints, fast tests, ffi-headers drift check, ffi-smoke,
  py-smoke, wasm-test.
- For changes touching SIMD or any `unsafe` code, run `test-full` locally
  before opening the PR.

## Bindings Workflow

**FFI (`semiflow-ffi`)**

Regenerate the C header after any ABI change:

```bash
cargo run -p xtask -- ffi-headers
```

CI runs `ffi-headers --check`; drift from the committed header fails the build.

**PyO3 (`semiflow-py`)**

```bash
cargo run -p xtask -- py-build
```

Type stubs are at `crates/semiflow-py/python/semiflow/__init__.pyi` and must
be kept in sync with the public API.

**WASM (`semiflow-wasm`)**

```bash
cargo run -p xtask -- wasm-build   # produces web + nodejs targets
cargo run -p xtask -- wasm-pack-npm  # packages dist/npm/ for npm publish
```

**Cross-language parity rule**: when a symbol is added to FFI, mirror it in
PyO3 and WASM within the same milestone, or file an ADR explaining the
intentional asymmetry.

## Documentation Updates

- Public API changes require rustdoc updates with at least one runnable
  `doctest` or a reference to `examples/`.
- Math spec changes require updating `contracts/semiflow-core.math.md` and, if
  the change affects convergence order or envelope, also updating the relevant
  sympy gate.

## Reporting Bugs / Requesting Features

Open a GitHub issue. Templates are available under `.github/ISSUE_TEMPLATE/`.

## License

By contributing you agree that your work is submitted under the project's
dual license: **MIT OR Apache-2.0**.

## Code of Conduct

This project follows the Contributor Covenant 2.1. See `CODE_OF_CONDUCT.md`.
Report violations to ilia.volkov@outlook.com.
