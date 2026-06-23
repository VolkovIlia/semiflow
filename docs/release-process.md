# Release Process

## Overview

Four crates ship under a single lockstep workspace version (`vN.M.K`):

| Crate | Kind | Registry |
|-------|------|----------|
| `semiflow` | `rlib` | crates.io (`publish = true`) |
| `semiflow-ffi` | `cdylib` | GitHub Releases only (`publish = false`) |
| `semiflow-py` | PyO3 wheel | GitHub Releases (`.whl` artefacts) |
| `semiflow-wasm` | wasm-bindgen | npmjs.org (`@semiflow/wasm`) |

All four crates share the `version` field in `Cargo.toml`. See `docs/api-stability.md`.

---

## Pre-Release Checklist

Complete all steps in order. Do not push the tag until every item is done.

### 1. ROADMAP MUSTs closed

Every MUST item for the target version in `ROADMAP.md` must be `[x]`.

### 2. Math fidelity audit approved

`docs/audit-findings-vN_M_K.md` must be **APPROVED**, 0 OPEN, 0 DEVIATION.

Verify sympy gates locally (all must print `PASS`):

```bash
python crates/semiflow/sympy/<gate>.py   # repeat for all T*N_*.py
```

**For math-creation ADRs** (any ADR that introduces a new mathematical construction
or oracle): the ADR must record a PRE-FLIGHT pass result (all sub-checks PASS)
before the engineer wave proceeds. Example: ADR-0107 records `T_ADJOINT_FP_TIGHTNESS`
6/6 PRE-FLIGHT PASS. This gate is checked as part of the release audit step above.

### 3. Heavy validation on production hardware

Run on an i7-12700K-class host (see `audit-findings-v1_0_0.md` §2 for spec):

```bash
RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=target-flagship \
    RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- test-flagship
```

Acceptance gates:

| Gate | Threshold |
|------|-----------|
| G3⁶-2D | slope ∈ [-6.15, -5.85] |
| G4_NS2D_aniso | slope ≤ -1.95 |
| G5_3D | slope ≤ -1.95 |
| NS2D_ANISO_PARALLEL_BIT_EQUAL | `abs_diff == 0.0` |

Fill in hardware block and slope numbers in `docs/audit-findings-vN_M_K.md`;
flip `[ ]` → `[x]`; promote DRAFT → APPROVED.

### 3a. Heavy `#[ignore]` gate sweep

Run all RELEASE_BLOCKING gates marked `#[ignore]` (distinct from the three
named flagship binaries above):

```bash
RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=target-flagship \
    RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- test-ignored-gates
```

This executes every `#[ignore]`-annotated test in the workspace under the same
flag profile as `test-full` (`parallel,simd,slow-tests --release`). Gates
covered include (non-exhaustive):

| Test binary | Gate |
|-------------|------|
| `g17_magnus6_slope` | G17 Magnus-6 slope |
| `g18_schrodinger_unitarity` | G18 Schrödinger unitarity |
| `hormander_kolmogorov_slope` | Kolmogorov hypoelliptic slope |
| `hormander_heisenberg_slope` | Heisenberg hypoelliptic slope |
| `hormander_engel_slope` | Engel step-3 Carnot slope |
| `robin_heat_slope` | Robin BC convergence slope |
| `subordinated_order1_slope` | Subordinated semigroup order-1 |
| `zeta4_truthful_order` | ζ⁴ truthful order gate |
| `diff_scipy` | SciPy cross-validation stub |
| `capture_trace_v1` | Trace capture regression |

All must exit 0 before tagging.

### 4. Test suite and lints clean

```bash
RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- test-fast
RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- test-full
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- check-lints
RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- ffi-smoke
RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- py-smoke
RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- wasm-test
```

All must exit 0.

### 5. Version bump consistent

`Cargo.toml` `[workspace.package] version` drives all four crates.
`pyproject.toml` is dynamic (maturin reads Cargo.toml). The npm package.json
is rendered by xtask — verify:

```bash
grep '^version' Cargo.toml
RUSTUP_TOOLCHAIN=stable cargo run -p xtask -- wasm-pack-npm
grep '"version"' dist/npm/package.json
```

### 6. CHANGELOG updated

`CHANGELOG.md` must have a `## [N.M.K] — YYYY-MM-DD` entry, no `(DRAFT)`.

---

## Tagging

```bash
git tag -a vN.M.K -m "chore(release): vN.M.K"
```

The version-match guard in `release-wasm.yml` strips the `v` prefix and
compares the tag to `[workspace.package] version` — a mismatch fails the job.

For MAJOR releases (BREAKING windows), the sign-off commit is a `docs(vN.0.0):`
commit (no code changes) that updates CHANGELOG + ROADMAP only; the BREAKING code
ships in the preceding `feat(vN.0.0)!:` commit. Pattern established at v3.0.0
(Window #1, 2026-05-27) and v5.0.0 (Window #2, 2026-05-29).

**BREAKING window cadence**: v3.0.0 = Window #1; v5.0.0 = Window #2; v7.0.0 = Window #3;
v9.0.0 = Window #4 (last; current workspace version is v9.2.0 — two additive MINORs since).
Future BREAKING windows follow the same pattern; see ROADMAP.md for the next candidate and
ADR-0035 §9 for deprecation-clock rules.

The bump commit preceding the tag must carry:
```
Agent: <human|agent-name>
Task-ID: release-vN-M-K
```

**Tag locality note**: Tags are created locally and pushed separately
(`git push origin master vN.M.K`). This pattern was established at v4.8.0 and
v5.0.0 when GitHub Actions billing was paused; local tags + manual push is the
primary flow (CI validation is secondary).

---

## Required GitHub Secrets

Set under **Settings → Secrets and variables → Actions**
([docs](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions)):

| Secret | Used by | Purpose |
|--------|---------|---------|
| `CARGO_REGISTRY_TOKEN` | manual | `cargo publish` to crates.io |
| `NPM_TOKEN` | `release-wasm.yml` | `npm publish --provenance` (+ OIDC `id-token: write`) |
| `PYPI_API_TOKEN` | manual | `twine upload` (not yet automated) |

---

## Publication Order

1. Push the tag:

   ```bash
   git push origin master vN.M.K
   ```

2. **Automatic** — `release-wasm.yml`: builds WASM, guards idempotency,
   publishes `@semiflow/wasm@N.M.K` to npmjs.org.

3. **Automatic** — `release-wheels.yml`: builds `semiflow-py` wheels (CPython
   3.10–3.13, Linux/macOS/Windows) and attaches them to the GitHub Release.

4. **Manual** — publish the Rust crate:

   ```bash
   cargo publish -p semiflow
   ```

   `semiflow-ffi`, `semiflow-py`, `semiflow-wasm` have `publish = false`; do not
   run `cargo publish` on them.

5. **Manual** — upload Python wheels to PyPI (download from GitHub Release):

   ```bash
   pip install twine
   twine upload dist/*.whl   # PYPI_API_TOKEN in env or ~/.pypirc
   ```

---

## Post-Release Verification

Wait 5–15 minutes for registries to propagate:

```bash
cargo search semiflow | grep "^semiflow "   # crates.io
npm view @semiflow/wasm version                       # npmjs.org
pip index versions semiflow-py                        # PyPI
# docs.rs: https://docs.rs/semiflow/N.M.K (allow ~15 min)
```

Smoke-test each surface:

```bash
cargo add semiflow@N.M.K && cargo build
npm install @semiflow/wasm@N.M.K && \
    node -e "const r=require('@semiflow/wasm'); console.log(typeof r.Heat1D)"
pip install semiflow-py==N.M.K && \
    python -c "import semiflow; print(semiflow.__version__)"
```

---

## Hot-Fix Process

1. Branch from the tag:
   ```bash
   git checkout -b hotfix/vN.M.K+1 vN.M.K
   ```
2. Apply the minimal fix. Add a sympy gate if math changes. Update `CHANGELOG.md`.
3. Bump `version` in `Cargo.toml` to `N.M.K+1`.
4. Run the full checklist above. Heavy validation is mandatory if any numerical
   code changed.
5. Tag and push:
   ```bash
   git tag -a vN.M.K+1 -m "chore(release): vN.M.K+1"
   git push origin hotfix/vN.M.K+1 vN.M.K+1
   ```
6. Open a PR from `hotfix/vN.M.K+1` → `master` to carry the fix forward.

---

## PyPI Trusted Publishing setup (one-time)

`release-wheels.yml` uses [OIDC Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
to publish `semiflow-pde` without storing a long-lived API token anywhere.  
Complete the steps below **once**; subsequent tag pushes publish fully automatically.

### 1. Register the pending publisher on PyPI

Go to **pypi.org → Your account → Publishing** (direct link:
`https://pypi.org/manage/account/publishing/`).

Click **"Add a new pending publisher"** (this works before the project exists on PyPI).

Fill in the form:

| Field | Value |
|---|---|
| **PyPI Project Name** | `semiflow-pde` |
| **Owner** | `VolkovIlia` |
| **Repository name** | `semiflow` |
| **Workflow name** | `release-wheels.yml` |
| **Environment name** | `pypi` |

The **Environment name** field must exactly match the `environment: name: pypi` declared
in the `publish-pypi` job; this scopes the OIDC token and prevents other workflows from
publishing under the same project.

Click **"Add"**. PyPI will show a pending publisher entry; it becomes active on first use.

### 2. Create the matching GitHub Environment (if not present)

In the repository go to **Settings → Environments → New environment**, name it `pypi`.
No additional protection rules are required; Trusted Publishing OIDC is the only auth
mechanism.

### 3. Remove the now-unnecessary PYPI_API_TOKEN secret

The `PYPI_API_TOKEN` secret listed in the "Required GitHub Secrets" table above is no
longer needed for automated publishing.  You may delete it from
**Settings → Secrets and variables → Actions** to reduce the attack surface.

### 4. Publish by pushing a version tag

```bash
git tag -a v0.9.0-beta2 -m "chore(release): v0.9.0-beta2"
git push origin master v0.9.0-beta2
```

The workflow will:
1. Build CPython 3.10–3.13 wheels on Linux, macOS (Intel + ARM), and Windows via `cibuildwheel`.
2. Build an sdist via `maturin sdist`.
3. Upload both wheels and sdist to the `pypi` environment with OIDC — no token exchanged.

**Version normalisation note**: maturin derives the Python package version from the
Cargo.toml workspace version.  A pre-release suffix such as `0.9.0-beta` becomes
`0.9.0b0` under PEP 440 normalisation; `0.9.0-rc.1` → `0.9.0rc1`.

**First-publish coverage note**: The first successful publish must include at least the
`manylinux` wheel (produced by the `ubuntu-latest` matrix leg) and the sdist so that
`pip install semiflow-pde` works broadly.  All four matrix legs run in parallel and
their artifacts are merged before upload, so a single tag push satisfies this
requirement automatically.
