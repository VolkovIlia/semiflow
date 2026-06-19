# ADR-0007 — Implement `Periodic` and `LinearExtrapolate` boundary policies in v0.2.1

**Status**: Accepted
**Date**: 2026-04-29
**Authors**: ai-solutions-architect (Stage 3)
**Resolves risk**: declared-but-unsupported gap from v0.1.0 (R5 carry-over)

## Decision

Promote `BoundaryPolicy::Periodic` and `BoundaryPolicy::LinearExtrapolate` from
`SemiflowError::Unsupported` stubs to fully native implementations in v0.2.1
(consolidation release before the v0.3.0 2D tensor-product extension). The
public `BoundaryPolicy` enum stays at four variants — no API breakage. The
internal `bc_index` helper is retyped from `Result<Option<usize>, SemiflowError>`
to a total `BoundaryHit` enum (`Inside(usize) | Zero | OutsideLeft(u32) | OutsideRight(u32)`)
and `bc_value` is taught the affine extension for `LinearExtrapolate` (3-point
boundary slope `(-3 f_0 + 4 f_1 - f_2) / (2 dx)`, exact for affine inputs).
Rationale: closes the declared-but-unsupported gap before v0.3.0 lifts these
adapters into 2D where stub-Err propagation would force every 2D call site to
carry boundary precondition checks. Trade-off: `LinearExtrapolate` can amplify
boundary noise by `1 / dx` for high-frequency boundary data, mitigated by
property `boundary_linear_extrap_affine_exact` (machine-eps for affine input)
and a future test G7 bounding `‖f̃‖_∞` growth at fixed shift magnitude
`|s| ≤ ¼` (the Chernoff-shift envelope at `tau ≤ 0.1`, `a ≤ 5`). All four
policies share strict-interior behaviour (invariant I5, traits.yaml), so
existing oracles (G1-legacy, G2-legacy, G1, G2, G3-strang) remain unchanged.
