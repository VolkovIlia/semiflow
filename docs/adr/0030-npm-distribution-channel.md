# ADR-0030 — npm distribution channel for `semiflow-wasm`

**Status**: Accepted (v0.11.0 contract, item I1)
**Date**: 2026-05-09
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0028 §"Out of scope" (deferred npm publish to v0.11.0),
ADR-0029 (v0.11.0 milestone — I1 MUST), Wave C profile (ADR-0028 Amendment 1)

## Context

ADR-0028 shipped `semiflow-wasm` (Wave C) as a buildable but unpublished crate.
Persona P1 (JS/TS web/Node developer; `acceptance.md` §1) cannot consume the
package without cloning the repo and running `wasm-pack build` locally —
adoption blocker. v0.11.0 must close this with a release workflow.
Decision points raised in `clarity-scan.md`: F1.1 (registry — npm public,
not GitHub Packages — resolved), F3.1 (package name `@semiflow/wasm` vs
unscoped — resolved with org-scope preference), F4.3 (CI runtime budget
≤10 min — resolved), F5.1 (lockstep with crate version), F6.1 (idempotency
on re-run / no `--force`), F10.2 (D-level: `@remizov` org availability —
mitigated by Risk R7 fallback).

## Decision

`semiflow-wasm` ships to the **public npm registry** under the org-scope
package name **`@semiflow/wasm`** via a tag-triggered GitHub Actions workflow
`.github/workflows/release-wasm.yml` that runs on `v*` tag push. The workflow
(a) checks out the tag, (b) runs `cargo run -p xtask -- wasm-build` (existing
xtask, produces both `--target web` and `--target nodejs` outputs), (c)
generates TypeScript declarations via `wasm-bindgen --typescript`
(removing the `--no-typescript` flag currently used for Wave C dev builds),
(d) merges the two builds into a single npm package with `package.json`
`exports` field routing `import` to web build and `require` to nodejs
build, (e) runs `npm pack` + dry-install in a scratch directory to verify
`import { Heat1D } from '@semiflow/wasm'` resolves before publishing,
(f) publishes via `wasm-pack publish` (or equivalent `npm publish --access public`)
using `NPM_TOKEN` stored as a GitHub Actions secret. The workflow MUST
fail-fast if `package.json` `version` field disagrees with the git tag
stripped of the `v` prefix (guard against accidental re-publish), and MUST
fail-fast if `npm view @semiflow/wasm@<version>` returns a result before the
publish step (idempotency guard against tag-rerun overlap; no `--force`).
**Token custody**: `NPM_TOKEN` is an automation-token with publish-only
scope on `@remizov/*`, rotated yearly, 2FA required on the npm account
(`auth-only` 2FA mode — automation tokens bypass 2FA at publish time but
the account itself enforces 2FA for token issuance/rotation). If the
`@remizov` org is unavailable on npm at workflow first-run (Risk R7
fallback path), Engineer falls back to `@semiflow-core/wasm` and amends
this ADR with the chosen scope (no separate ADR needed; this is a
pre-flight finding, not a design change).

## Consequences

- **Pro**: persona P1 unblocked with `npm install @semiflow/wasm`; matches the
  Rust crate name's module structure (`semiflow-wasm` crate → `@semiflow/wasm`
  package) for one-step mental mapping.
- **Pro**: tag-triggered publish ties npm version to git tag — single source
  of truth for distribution version, no parallel version state to drift.
- **Pro**: idempotency guards prevent the most common Wave-C-style accidents
  (workflow re-run, manual re-tag) from producing duplicate or conflicting
  registry entries.
- **Con**: `NPM_TOKEN` is a single bus-factor-1 secret (Risk R1 in
  `acceptance.md`); recovery requires npm support if maintainer loses
  access. Mitigated by yearly rotation and 2FA on issuance — accepted.
- **Follow-up**: if v0.12.0 adds variable-`a` bindings (I3), the same
  workflow re-runs unchanged — package contents grow, version follows
  git tag, no workflow edit needed.
- **Follow-up**: v1.0.0 freezes the package name; renaming post-v1.0.0
  requires an `npm deprecate` + new package with redirect note, similar
  cost to a Cargo crate rename.

## Alternatives Considered

- **Unscoped name `semiflow-wasm`** — rejected (F3.1 resolution): polluting
  the global npm namespace is anti-suckless; org-scope clarifies provenance
  and reserves namespace for future siblings (`@remizov/cli`, `@remizov/types`).
- **GitHub Packages instead of public npm** — rejected (F1.1): adds friction
  for users who must configure `.npmrc` with a GitHub auth token to install;
  defeats the persona-P1 "npm install and go" goal.
- **Manual `wasm-pack publish` from maintainer's laptop** — rejected: violates
  reproducibility; CI-driven publish keeps the build environment pinned to
  the repo's `rust-toolchain.toml` and `Cargo.lock`.
- **Hand-written `.d.ts`** — rejected: `wasm-bindgen --typescript` already
  generates them from the `#[wasm_bindgen]` annotations; hand-written would
  drift. (`clarity-scan.md` F2.1 analogue, decided by symmetry.)
- **Publish on every commit to main** — rejected: spams the registry with
  pre-release versions and drains the maintainer's daily registry-publish
  rate-limit budget.

## Amendment 1 — SLSA provenance opt-in (2026-05-09)

Per session-locked decision, `npm publish` adds `--provenance` in the
`release-wasm.yml` workflow.  The workflow job grants `id-token: write`
permission (required for GitHub's OIDC token issuance that backs npm
provenance attestation).  Cost: one extra YAML line and the `id-token: write`
permission block; benefit: free supply-chain attestation visible on npmjs.org
as a provenance badge, enabling consumers to verify the package was built from
this exact commit by the registered GitHub Actions workflow — no external
tooling or separate signing key required.
