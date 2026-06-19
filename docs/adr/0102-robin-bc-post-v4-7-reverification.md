# ADR-0102: Robin BC post-v4.7 reverification (light hygiene)

**Status**: ACCEPTED 2026-05-29
**Outcome**: NEGATIVE (all gates PASS — no engineer follow-up required)

## Context

Robin BC (`BoundaryPolicy::Robin { alpha, beta }` + `RobinHeatChernoff<C, R, F>`)
shipped at **v4.6.0** (engineer wave `385580e` + cleanup `82b6398`). ADR-0098
AMENDMENT 1 corrected the Carslaw-Jaeger 1959 §14.2 eq 5 erfc factor from
`2·(α/β)` to `(α/β)` mid-wave; math §3.5.tris.2 was AMENDED accordingly.

**v4.7.0** SHIPPED A.6 LadderRung trait (`40edaa7`) — additive sealed catalogue
with **zero API surface change** to Robin BC code (`robin.rs`, `robin_regions.rs`,
sympy oracle, slope test all untouched).

Per the max-2-retries causal-protection protocol, every post-release boundary
that touches numerical invariants requires an explicit "did anything drift?"
checkpoint. This ADR is that checkpoint for Robin BC at the v4.6 → v4.7 transition.

## Decision

Robin BC v4.6 invariants HOLD under v4.7 LadderRung. **No re-measurement required.**
This ADR is CLOSED at v4.8 docs sign-off — no engineer Wave spec produced.

Conditional triggers (NOT activated):
- (a) If `tests/robin_heat_slope.rs` slope drifted → ADVISORY engineer follow-up.
- (b) If `scripts/verify_robin_kernel.py` sub-checks FAIL → BLOCKING engineer fix.

## Acceptance gates (all PASS)

1. **Sympy oracle** — `python3 scripts/verify_robin_kernel.py`:
   ```
   T_ROBIN.coefficient: PASS (Neumann limit r=+1; r(α,β,0)=1 verified)
   T_ROBIN.boundary:    PASS (α·K^Robin − β·∂_x K^Robin = 0 at x=0; outward-normal form)
   T_ROBIN.heat_pde:    PASS (∂_t K^Robin - ∂_xx K^Robin = 0)
   T_ROBIN.oracle_match:PASS (sympy=0.479990800421184, python=0.479990800421184, rel_err=1.16e-16)
   T_ROBIN PASS  (4/4 sub-checks)
   ```

2. **Fast-bin presence** — `crates/semiflow-core/tests/robin_heat_slope.rs`
   present (5118 B); contains `g_robin_halfline_slope` (#[ignore], slow-tests
   gated, Carslaw-Jaeger oracle with corrected `(α/β)` factor) and the
   `g_robin_self_2d_slope` engineer-todo stub (Strang-2D pattern, ADR-0098 §6).

3. **test-fast** — `cargo run -p xtask -- test-fast`:
   ```
   passed=824  failed=0  ignored=28  bins=176
   ```
   Robin code compiled and linked into all relevant bins; zero failures.

## Consequences

- v4.6 Robin BC ships unchanged into v4.7 and v4.8.
- `g_robin_halfline_slope` remains `#[ignore]` + `slow-tests` feature-gated
  per v4.6 acceptance (slope ≤ -0.95 order-1 cap, math §3.5.tris.5); not
  promoted to test-fast since order-1 sweep over `[16, 32, 64, 128]` is
  intentionally a release-gate not a hot-loop check.
- `g_robin_self_2d_slope` 2D engineer-todo stub is **explicitly out of scope**
  for this reverification — tracked separately under ADR-0098 §6 future work.
- POSITIVE-outcome path (`.dev-docs/specs/robin-rehygiene-wave.md`) NOT
  produced — both gates PASS.

## References

- ADR-0098 (Robin BC partial-additive) + AMENDMENT 1 (erfc factor fix)
- math.md §3.5.tris.2 (AMENDED Carslaw-Jaeger 3-term kernel)
- v4.6 SHIP commits: `385580e` (engineer wave) + `82b6398` (cleanup)
- v4.7 SHIP commit: `40edaa7` (A.6 LadderRung trait — additive, no Robin touch)
- ADR-0100 (LadderRung trait sealed catalogue)
- `scripts/verify_robin_kernel.py` (sympy oracle, 4 sub-checks)
- `crates/semiflow-core/tests/robin_heat_slope.rs` (G_ROBIN_HALFLINE + 2D stub)

**Superseded re: convergence by ADR-0098 Amendment 2 (v6.2.3)** — the v4.6 operator-level method verified here was non-convergent on half-line grids; see Amendment 2.
