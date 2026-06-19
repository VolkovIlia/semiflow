# ADR-0032 — Heavy-validation harness for v0.11.0 (G3⁶-2D, G4_NS2D_aniso, G5_3D)

**Status**: Accepted (v0.11.0 contract, items I12 + I13)
**Date**: 2026-05-09
**Authors**: ai-solutions-architect
**Cross-refs**: ROADMAP.md "Heavy validation" subsection (defers gates from
v0.9.0 to production hardware), ADR-0020 (G3⁶-2D flagship gate), ADR-0023
(anisotropic 2D non-separable G4 gate), ADR-0024 (3D tensor G5 gate),
ADR-0029 (v0.11.0 milestone — I12, I13 MUST)

## Context

Three slope-convergence gates from v0.9.0 carry "deferred to production
hardware" status in `ROADMAP.md`: G3⁶-2D (slope ≈ -6.08, ~50 min wallclock
on prod HW), G4_NS2D_aniso (slope ≈ -2.20), G5_3D (slope ≈ -2.0). GitHub
Actions runners cannot execute these in the CI budget — wallclock per gate
exceeds the per-job ceiling for the public ubuntu-latest pool. v0.11.0
must (a) execute these gates one final time before v1.0.0 freeze (item I12),
(b) record results in a reproducibility-audited document that the v0.9.0+v0.10.0
math-fidelity audit (I13, researcher-driven) consumes. Findings from
`clarity-scan.md`: F1.3 CRITICAL (audit reviews; does not re-run — resolved),
F4.2 (HW spec recording — resolved), F6.3 (slope-drift threshold — gate
values stay in `tests/strang_*_slope.rs`; >5% drift from historical recorded
but non-blocking — resolved), F9.1 (done-signal: 4 artifacts — 3 slope numbers
+ 1 audit doc section; resolved), F10.1 (D-level: HW available — mitigated
by Risk R2 community-validation fallback). The maintainer's local hardware
(no GitHub-Actions self-hosted runner provisioned, no budget for one in
v0.11.0) is the execution environment.

## Decision

The three heavy-validation gates run on the **maintainer's local production
hardware** (single human-driven invocation per gate, no CI orchestration in
v0.11.0) using the exact commands recorded in `ROADMAP.md` "Heavy validation"
subsection — `RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=target-flagship
cargo run -p xtask -- test-flagship -- {gate-name}`. Results land in a single
new document **`docs/audit-findings-v0_11_0.md`** containing for each gate:
(a) raw slope value (≥4 significant figures), (b) wallclock per-step in
seconds, (c) total wallclock, (d) commit SHA at time of run, (e) hardware
record block (CPU model, physical core count, logical thread count, RAM
size, OS+version, `rustc --version --verbose`, `cargo --version`). The same
file gains a "Math fidelity audit" §2 produced by the researcher agent
(item I13) covering v0.9.0 + v0.10.0 commits, with one **separate** sibling
file per audit target — `docs/audit-findings-v0_9_0.md` and
`docs/audit-findings-v0_10_0.md` — matching the v0.6.0 / v0.8.0 audit
template (`Findings`, `SIMPLIFICATIONs`, `EXTENSIONs`, `OPEN`). Audit doc
generation is a **researcher pipeline** delegation; the audit reads the
v0.11.0 slope numbers from `audit-findings-v0_11_0.md` and the existing
math.md sections (§10.7-ter, §10.8, Lemma 10.1) and ADRs 0023/0024/0025/0026/0028
without re-running gates (per F1.3). **Gate-pass criterion**: each `cargo
run -p xtask -- test-flagship -- {gate}` invocation exits 0 (test source
embeds the slope threshold; no Architect re-encoding); v0.11.0 release
blocks if any gate exits non-zero. **Drift detection** (non-blocking
diagnostic): if the recorded slope drifts >5% from historical
(G3⁶-2D = -6.0837 reference per project memory `v0.8.1 G3⁶-2D FLAGSHIP closed`),
the engineer adds a "Drift note" subsection to `audit-findings-v0_11_0.md`
explaining the delta; release proceeds. **Audit findings DO NOT block
v0.11.0** (AC-4, Risk R4): a CRITICAL finding is filed as a v0.11.0-follow-up
issue, recorded in the audit doc, and v0.11.0 ships with the finding
documented.

## Consequences

- **Pro**: clears v0.9.0 math-correctness debt before v1.0.0 freeze
  (ROADMAP rule #5 satisfied incrementally rather than in a single
  v1.0.0-blocker audit cycle).
- **Pro**: reproducibility metadata (HW + toolchain block) lets any
  community contributor with comparable HW reproduce the slope independently
  — Risk R2 fallback path is viable rather than aspirational.
- **Pro**: separating gate-execution (I12) from audit-review (I13) per F1.3
  matches the labour split — engineer/maintainer runs gates,
  researcher-agent reads results.
- **Con**: no CI gating on heavy validation — depends on maintainer
  discipline + this ADR's release checklist. Acceptable for v0.11.0;
  v1.0.0 may revisit (self-hosted runner) but is out of scope here.
- **Con**: audit-doc proliferation (3 files: v0.9.0, v0.10.0, v0.11.0 sharing
  the same template) is mild documentation duplication. Accepted: each
  major version's audit is independently grep-able and citable from
  CHANGELOG; consolidating into a single file would lose per-tag locality.
- **Follow-up**: v1.0.0 audit re-uses this template + reads from these three
  files; net new audit work in v1.0.0 covers only the v0.11→v1.0 delta.
- **Follow-up**: if Risk R2 fires (no maintainer HW), the v0.11.0 release
  process delegates to GitHub Discussions for community-validation runs
  before tag — the audit doc records reporter identity and HW spec from
  whoever volunteers. Document mechanics deferred until/unless triggered.

## Alternatives Considered

- **Provision a self-hosted GitHub Actions runner for heavy validation**
  — rejected: capital + ongoing cost not justified by single-release usage;
  v1.0.0 may revisit but is outside v0.11.0 scope.
- **Lift slope thresholds into config so the gate is parameterized** —
  rejected: thresholds in source (`tests/strang_*_slope.rs`) are the
  contract; making them config-driven invites silent relaxation. Drift
  detection is the proper escape hatch for non-blocking-but-noteworthy
  changes.
- **Single combined audit doc covering v0.9.0+v0.10.0+v0.11.0 in one
  file** — rejected (F8.1 implication): per-tag locality matters for
  citation from CHANGELOG and for the v1.0.0 audit's reading scope.
- **Re-run v0.6.0/v0.8.0 historical gates as part of I12** — rejected:
  out of scope; those tags' gates were green at their time of release
  and are not part of the v0.9.0→v0.10.0 deferral. v1.0.0 audit may
  spot-check historical gates if needed.
- **Block v0.11.0 release on a clean audit (zero CRITICAL findings)** —
  rejected (AC-4, Risk R4): defeats the purpose of separating audit from
  feature work; the audit's job is to surface findings, not to gate.
