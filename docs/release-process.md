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
    cargo run -p xtask -- test-flagship
```

Acceptance gates:

| Gate | Threshold |
|------|-----------|
| G3⁶-2D | slope ∈ [-6.30, -5.85] AND wallclock ≤ 600 s |
| G4_NS2D_aniso | slope ≤ -1.95 |
| G5_3D | slope ≤ -1.95 |
| NS2D_ANISO_PARALLEL_BIT_EQUAL | `abs_diff == 0.0` |

The G3⁶-2D window is two-sided on purpose: `-5.85` catches order degradation,
`-6.30` catches the interpolation floor returning as fake super-convergence.
Both bounds and the 600 s budget come from `properties.yaml::G3_6_2D`, which the
test file mirrors verbatim — recalibrated for the `SepticHermite` floor by
ADR-0163 (the pre-0163 window `[-6.15, -5.85]` and 3300 s budget are dead).

Fill in hardware block and slope numbers in `docs/audit-findings-vN_M_K.md`;
flip `[ ]` → `[x]`; promote DRAFT → APPROVED.

### 3a. Heavy `#[ignore]` gate sweep

Run all RELEASE_BLOCKING gates marked `#[ignore]` (distinct from the three
named flagship binaries above):

```bash
RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=target-flagship \
    cargo run -p xtask -- test-ignored-gates
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

### 3b. Which workflow actually runs which gate

`properties.yaml` declares 117 `RELEASE_BLOCKING` gates. Audited 2026-08-18,
their execution was:

| Executed by | Gates |
|---|---|
| `ci.yml` — plain `cargo test --workspace --release` | 30 |
| `flagship-gates.yml` / `nightly.yml` — named `--test` binaries | 16 |
| `py-smoke` (Python) | 3 |
| **nothing — no workflow, on any trigger** | **62** |
| marker entries with no `test_file` | 4 |
| pointers this checker cannot resolve | 2 |

> The first published version of this table said 59, not 62. The audit script
> behind it classified gates with a single regex and had three blind spots:
> it recognised only the bare `#![cfg(feature = "slow-tests")]` form and missed
> the compound `#![cfg(all(feature = "parallel", feature = "slow-tests"))]`; it
> never looked at per-test `#[cfg(feature = "slow-tests")]`; and a trailing
> comment after an attribute broke its attribute-chain match, which is how
> `sym_op_dense` read as reachable. The numbers above come from
> `scripts/check_gate_coverage.py`, which parses attribute blocks properly.

Gates are unreachable in two ways, and the second reads as health: files gated
by `#![cfg(...)]` on `slow-tests` are never compiled by CI, while gates carrying
`#[ignore]` are compiled, skipped, and reported inside the `N ignored` tally of
a green run.

**The enumeration is no longer trusted.** `scripts/check_gate_coverage.py` runs
as a `ci.yml` job on every PR and fails if any `RELEASE_BLOCKING` gate is
executed by no workflow. It exists because the hand-maintained list of ~78
`--test` flags drifted within a day of being written: review of the very PR that
added it found four binaries already missing (`sym_op_dense`,
`obstacle_vi_slope`, `strang2d_parallel_bit_equal`,
`chernoff1d_parallel_bit_equal`, together 7 gates). Add a gate, and this job
tells you if you forgot to wire it up.

Steps 3 and 3a above were the only thing standing between that set and a
release, and they are manual. Before v0.13.0-beta they were not run, which is
how the gates written FOR that release's issue campaign — `G_ASND_MOMENT` (the
second-moment oracle written to catch issue #17), the five `G_CONS_*` (#26),
`G_SHIFT1D_COEFF_FD` (#25), `G_PENCIL_ORDER2` (#21) — shipped without ever
having executed in CI.

Those 62 are now bound to `flagship-gates.yml`, which triggers nightly **and on
every `v*` tag**, in six concern-grouped jobs: `campaign-gates`,
`operator-exponential-gates`, `geometry-hypoelliptic-gates`,
`resolvent-sampling-gates`, `wentzell-multilayer-ad-gates` and
`nonseparable-2d-gates`. They run with `-- --include-ignored`, so a file is
covered regardless of which of the two gating mechanisms it uses.

Measuring that set before wiring it up — its first execution — immediately
turned up one stale cost claim. `G_SMOLYAK_D5` documents "~10-30 s on release"
and exceeded a 40-minute cap on a 12-core host; `G_SMOLYAK_D6` documents "≤ 2 min"
and hit the same cap. Both pass; what went stale is the estimate, because
ADR-0191's `K^D` sampler made a `D = 5` sample read 1024 nodes instead of 32 and
nobody re-measured a gate that ran nowhere. They now live in
`nightly.yml::smolyak-d5-d6` beside `ddim-d5`, off the tag lane.

That expectation held: the completed sweep turned up a third hours-long gate,
`strang_nonseparable_slope` (`G3_NS2D`, `G3_NS2D_var`), also over the 40-minute
cap and also moved to nightly. Nothing was stale about it — it is expensive by
construction — but it is at least twice the local cost of its tag-lane sibling
`strang_nonseparable_aniso_slope`, which already takes 118 min hosted.

Final sweep result, 12-core host, `-- --include-ignored`: **52 binaries, 49 PASS
in 1035 s, 3 over the cap** — the three now on the nightly schedule. Nothing else
in the newly-covered set is slow; the next-slowest is `g_killing_order2` at 90 s
and the median is ~13 s. Turning these gates on does not make CI slow or red.

This does **not** retire steps 3/3a: the tag lane is a record bound to the
released SHA, not a blocker on publication (see the `tags:` comment in
`flagship-gates.yml`), and hosted runners are not the calibrated bench hardware.
It does mean a skipped manual run is now visible within a day instead of never.

### 4. Test suite and lints clean

```bash
cargo run -p xtask -- test-fast
cargo run -p xtask -- test-full
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo run -p xtask -- check-lints
cargo run -p xtask -- check-unsafe-scope
cargo run -p xtask -- ffi-headers --check
cargo run -p xtask -- ffi-smoke
cargo run -p xtask -- py-smoke
cargo run -p xtask -- wasm-test
```

All must exit 0.

Also run the two gates the release workflows themselves enforce, so a failure
surfaces here rather than after the tag is already pushed — `release-crate.yml`,
`release-wheels.yml` and `release-wasm.yml` each block publication on both:

```bash
cargo run -p xtask -- doc-check
cargo run -p xtask -- changelog-check
```

**Resolved 2026-08-18 — clippy now covers `slow-tests`.** The line above matches
`ci.yml`, which gained `--all-features`. Until that day neither job saw the
`#![cfg(feature = "slow-tests")]` files: `clippy --all-targets` without
`--all-features` skips them, and `cargo test --workspace --release` does not
build them either. 27 of the files carrying RELEASE_BLOCKING gates were
therefore type-checked by no workflow at all.

The backlog that had accumulated behind that blind spot measured **346 unique
diagnostics across 31 files** (`cargo clippy --workspace --all-targets
--all-features --keep-going`, deduplicated by lint × file × line; earlier
counts of 412/444 double-counted sites reported under more than one target).
It was cleared as follows, with no gate threshold, tolerance or assertion
touched:

| Disposition | Count | What |
|---|---|---|
| Fixed by `cargo clippy --fix` | 183 | `doc_markdown` (127), `uninlined_format_args`, `cloned_instead_of_copied`, `map_unwrap_or`, `enum_glob_use`, `let_and_return`, `manual_range_contains`, … |
| Fixed by hand | 10 | `ignore_without_reason` ×8 — every ignored gate now states WHY in the attribute — plus `vec_init_then_push`, `empty_line_after_doc_comments` |
| File-level `#![allow]` + justification | 153 | numeric-cast family, `similar_names`, `needless_range_loop`, `too_many_lines`, `too_many_arguments`, `identity_op`/`erasing_op`, `manual_clamp`, `match_same_arms` |

The `#![allow]` route is the repo's existing convention for test files, not a
new escape hatch: 130 test files already carried one before this change, and
each new line states its reason inline. Three of those reasons are load-bearing
rather than cosmetic and should not be "cleaned up" later:

- `identity_op` / `erasing_op` — row-major index formulas are written uniformly
  (`gen[0 * nd + 0]`, `d_tri[1 * d + 0]`). The redundant `0 *` and `+ 0` terms
  document the layout; `cargo clippy --fix` rewrote them to `gen[0 * nd]` and
  the rewrite was reverted on purpose.
- `manual_clamp` — `x.max(lo).min(hi)` is **not** `f64::clamp(x, lo, hi)`:
  `max`/`min` map a NaN input onto a bound, `clamp` propagates it. Rewriting
  would change what the gate does with non-finite input.
- `match_same_arms` — the `d = 4 => 4` arm mirrors the §3.1 table explicitly;
  folding it into the default would hide the spec it transcribes.

Note `--keep-going` if you re-measure: `.cargo/config.toml` sets
`rustflags = ["-D", "warnings"]` and `[workspace.lints.clippy] all = "deny"`, so
lints are hard errors and the build aborts at the first failing target. Without
it you see one file per run and walk the backlog one step at a time, which badly
understates it. Add `RUSTFLAGS="--cap-lints warn"` to see every lint at once.

### 5. Version bump consistent

`Cargo.toml` `[workspace.package] version` drives all four crates.
`pyproject.toml` is dynamic (maturin reads Cargo.toml). The npm package.json
is rendered by xtask — verify:

```bash
grep '^version' Cargo.toml
cargo run -p xtask -- wasm-pack-npm
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

**BREAKING window cadence**: the windows above (v3.0.0 = #1, v5.0.0 = #2, v7.0.0 = #3,
v9.0.0 = #4) are all **pre-rebrand**, on the `remizov-*` version line that ended at
v9.2.0. The rebrand to SemiFlow reset the public version to `0.9.0-beta`, so the
workspace is now on a `0.x` beta line and no BREAKING window has opened since. While
the major is `0`, SemVer permits breaking changes in a MINOR — the deprecation-clock
discipline in ADR-0035 §9 still applies by policy, not by SemVer obligation. See
ROADMAP.md for the road to `1.0.0`, which is when windows resume being load-bearing.

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
| `CRATES_IO_TOKEN` | `release-crate.yml` | `cargo publish -p semiflow` |
| `NPM_TOKEN` | `release-wasm.yml` | `npm publish --provenance` (+ OIDC `id-token: write`) |

PyPI needs **no secret**: `release-wheels.yml` publishes through OIDC Trusted
Publishing (`pypa/gh-action-pypi-publish`, job `publish-pypi`, environment `pypi`).
That environment must exist in repo settings and the PyPI project `semiflow-pde`
must list this repository + workflow as a trusted publisher, or the job fails at
the token-exchange step. There is no `PYPI_API_TOKEN` and no `twine upload`.

---

## Publication Order

Publication is **fully automatic**. All three release workflows trigger on
`push: tags: ["v*"]`, so pushing the tag is the only manual act — there is
nothing left to run by hand afterwards.

1. Push the tag:

   ```bash
   git push origin master vN.M.K
   ```

2. **Automatic** — `release-crate.yml`: publishes `semiflow` to crates.io
   (`CRATES_IO_TOKEN`). Guards: tag/`Cargo.toml` version match, idempotency
   probe against the crates.io API, and `cargo test -p semiflow --release`.

3. **Automatic** — `release-wasm.yml`: builds WASM and publishes
   `@semiflow/wasm@N.M.K` to npmjs.org with `--provenance` (`NPM_TOKEN` + OIDC).

4. **Automatic** — `release-wheels.yml`: `cibuildwheel` builds CPython 3.10–3.13
   wheels (Linux x86-64 manylinux_2_28, macOS arm64, Windows) plus a maturin
   sdist, attaches them to the GitHub Release, then job `publish-pypi` uploads
   everything to PyPI as **`semiflow-pde`** via Trusted Publishing.

All three gate on `doc-check` + `changelog-check` before publishing, so a stale
CHANGELOG or a doc-drift failure blocks the release rather than shipping it.

Only `semiflow` is a published crate; `semiflow-ffi`, `semiflow-py` and
`semiflow-wasm` carry `publish = false` and never go to crates.io.

**Idempotency**: crates.io and npm are guarded explicitly; PyPI uses
`skip-existing: true`. Re-running a workflow on an already-published tag is
therefore safe and is the normal way to recover from a single failed job.

---

## Post-Release Verification

Wait 5–15 minutes for registries to propagate:

```bash
cargo search semiflow | grep "^semiflow "   # crates.io
npm view @semiflow/wasm version                       # npmjs.org
pip index versions semiflow-pde                       # PyPI
# docs.rs: https://docs.rs/semiflow/N.M.K (allow ~15 min)
```

**PyPI distribution name is `semiflow-pde`**, not `semiflow-py`. `semiflow-py`
is the *crate* name in this workspace and is `publish = false`; it has never
been a PyPI project. Installing it fetches nothing.

PyPI's JSON API reports the new release before the `/simple/` index that `pip`
resolves against does, so `pip install` can still fail with "No matching
distribution found" for a minute or two after the workflow goes green. That is
propagation lag, not a failed publish — confirm with
`curl -s https://pypi.org/simple/semiflow-pde/ | grep <version>` before
investigating anything else.

Smoke-test each surface (`N.M.K` is the Cargo version; PyPI normalises it per
PEP 440, so `0.13.1-beta` is installed as `0.13.1b0`):

```bash
cargo add semiflow@N.M.K && cargo build
npm install @semiflow/wasm@N.M.K && \
    node -e "const r=require('@semiflow/wasm'); console.log(typeof r.Heat1D)"
pip install semiflow-pde==<PEP440 version> && \
    python -c "import semiflow; print(semiflow.version())"
```

`semiflow.version()` is a function, and is the only version accessor the module
exposes — there is no `semiflow.__version__`.

Prefer a smoke test that computes something over one that only imports. The
import succeeds even if the compiled extension is broken in ways that matter:

```bash
python - <<'PY'
import semiflow, numpy as np
n = 64; xs = np.linspace(-4.0, 4.0, n)
s = semiflow.Heat1D(-4.0, 4.0, n, np.exp(-xs**2))
s.evolve(0.1, 200)
got = np.asarray(s.values())
exact = np.exp(-xs**2 / 1.4) / np.sqrt(1.4)          # closed form at t=0.1
print("version:", semiflow.version())
print("sup error vs closed form: %.3e" % np.abs(got - exact).max())   # ~9e-06
PY
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
