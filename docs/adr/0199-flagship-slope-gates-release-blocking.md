# ADR-0199 — Make G4_NS2D_aniso and G5_3D RELEASE_BLOCKING in the contract

- **Status**: Proposed (issue #34)
- **Date**: 2026-09-28
- **Supersedes / amends**: ADR-0023 (G4_NS2D_aniso), ADR-0024 (G5_3D) — severity
  field only.
- **Contract**: `contracts/semiflow-core.properties.yaml` entries `G4_NS2D_aniso`
  and `G5_3D`. No `math.md` section changes.

## Context

The three flagship slope gates did not have the same severity in the contract:
`G3_6_2D` was `RELEASE_BLOCKING`, but `G4_NS2D_aniso` and `G5_3D` were
`NORMATIVE`. Everything else in the repo treats all three as blocking:

- `docs/release-process.md` step 3 lists all three as acceptance gates that must
  pass before a tag.
- `.github/workflows/flagship-gates.yml` is titled *"Flagship Gates
  (RELEASE_BLOCKING)"* and its header calls all three RELEASE_BLOCKING.
- ADR-0024's contract note says "Failure of `g5_3d_slope` BLOCKS v0.9.0 release".
- The `G_SMOLYAK_D6` rationale in the contract says "G5_3D / G3⁶-2D are
  RELEASE_BLOCKING but run on prod HW".

The mismatch had a practical cost. `scripts/check_gate_coverage.py` only checks
RELEASE_BLOCKING gates, so removing `strang_3d_slope` or
`strang_nonseparable_aniso_slope` from every workflow went unnoticed. Those were
exactly the two gates the release process treats as blocking.

## Decision

1. Set `severity: RELEASE_BLOCKING` on `G4_NS2D_aniso` and `G5_3D`. The contract
   was the one wrong document; the release process, the workflow and ADR-0024
   already agree with each other. Rejected alternative: keep them `NORMATIVE` and
   rewrite `release-process.md` step 3 to say they are advisory. That would
   quietly demote two convergence gates that have been blocking in practice since
   v0.9.0, and leave the workflow title wrong.
2. `check_gate_coverage.py` now drops a YAML trailing comment from unquoted
   `severity:` / `test_file:` values. Before this, a line such as
   `severity: RELEASE_BLOCKING  # note` was read as a different severity, and the
   gate silently fell outside the coverage check.

## Consequences

- No threshold, basket or test body changes: both gates keep `slope <= -1.95` and
  the same test files. This is not a threshold change, so it needs no
  `Gate-Change-Approved-By:` trailer.
- `check_gate_coverage.py` now covers both gates. Both binaries are already named
  by `xtask test-flagship` and `flagship-gates.yml`, so the check still passes
  without workflow changes.
- A failing `G4_NS2D_aniso` or `G5_3D` now formally blocks a tag, which matches
  what `release-process.md` already required.

## Honest limits

- This does not make either gate run on every PR. Both are still `slow-tests`
  gates run nightly and before a tag on production-class hardware (G5_3D takes
  about 50 min).
- `NS2D_ANISO_PARALLEL_BIT_EQUAL`, the fourth row of the step-3 table, has no
  contract entry at all. It is out of scope here and is still not covered by the
  coverage check.

## Gate

- `G4_NS2D_aniso` — `crates/semiflow/tests/strang_nonseparable_aniso_slope.rs::g4_ns2d_aniso_slope`
- `G5_3D` — `crates/semiflow/tests/strang_3d_slope.rs::g5_3d_slope`
- The coverage check itself: `python3 scripts/check_gate_coverage.py`, run by the
  `gate-coverage` job in `ci.yml`.
