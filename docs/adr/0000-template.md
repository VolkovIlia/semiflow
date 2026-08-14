# ADR-NNNN — <one-line decision, in the imperative>

- **Status**: Proposed | Accepted | Superseded by ADR-MMMM
- **Date**: YYYY-MM-DD
- **Supersedes / amends**: none, or `ADR-MMMM §Section`
- **Contract**: the `contracts/semiflow-core.math.md` section this decision
  writes or changes, and the gate names it registers in
  `contracts/semiflow-core.properties.yaml`.

## Context

What forced the decision. State the observed behaviour, not the intent — a
measurement, a failing gate, a reported defect, a capability that cannot be
expressed with the current types. If a number motivated this, put the number
here.

## Decision

The decision itself, ≤1 paragraph per numbered point (suckless guardrail #1).
Alternatives that were considered and rejected belong here too, with the reason
each was rejected; "we picked X" without "and not Y because Z" is not an ADR.

## Consequences

What this costs. Step counts, wallclock, memory, API breakage, gates that have
to move. If a gate threshold changes, say so explicitly — a threshold change
needs a `Gate-Change-Approved-By:` commit trailer from the architect, and the
producing agent must not self-approve.

## Honest limits

What this does NOT do, stated plainly enough that a reader can tell whether it
covers their case. Every ADR needs this section; if it is empty the decision has
not been thought through. Include the failure modes the new gates cannot see.

## Gate

The gate(s) that make the decision falsifiable, with their registry names, the
file and test function that implement them, and the measured value. A gate whose
non-vacuity is not obvious should carry a companion assertion that fails if the
datum stops exercising the property (see `G_THETA_M_TABLE` for the pattern).
