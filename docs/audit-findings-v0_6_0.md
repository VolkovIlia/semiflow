---
version: 1.0.0
last_updated: 2026-05-01
freshness_score: 1.0
dependencies:
  - contracts/semiflow-core.math.md
  - docs/adr/ADR-0004.md
  - docs/adr/ADR-0007.md
  - docs/adr/ADR-0013.md
  - docs/adr/ADR-0014.md
  - src/diffusion4.rs
  - src/magnus4.rs
  - src/adaptive.rs
changelog:
  - 1.0.0: Initial audit findings record for v0.6.0
---

# v0.6.0 Math Fidelity Audit — Findings

**Audit date**: 2026-05-02
**Auditor**: researcher agent (autonomous)
**Audit reference**: `.dev-docs/reports/v0_6_0_math_fidelity_audit_2026-05-02.md`
(full report, gitignored, dev-only)
**Scope**: v0.6.0 implementation vs Remizov 2025 (*Vladikavkaz Math. J.* 27(4),
DOI 10.46698/a3908-1212-5385-q) + Chernoff 1968 + HLW 2006 §III.5 +
Söderlind 2002 + BCO-R 2009.

---

## Summary (17 findings)

| Class | Count | Items |
|-------|-------|-------|
| FAITHFUL | 6 | Theorem 6, Chernoff iteration, gamma-A weights, Strang, RK2 drift, Theorem 7 |
| SIMPLIFICATION | 3 | f64 (ADR-0004), bounded domain (ADR-0007), `a >= 0` |
| APPROXIMATION | 3 | Cubic-Hermite joint (math.md §9.2.4), delta=3·dx floor, boundary layer |
| DEVIATION | 2 | order() mismatch (D1), Magnus naming (D2) |
| EXTENSION | 3 | Strang (HLW), zeta-A correction, AdaptivePI (Söderlind) |

---

## Faithful (no action required)

- **Theorem 6 formula (6)** — `ShiftChernoff1D` implements the exact Chernoff
  approximation `S(t/n)^n → exp(tL)` per Remizov 2025 formula (6).
- **Chernoff iteration** — `ChernoffSemigroup::evolve` iterates `n` applications
  of `S(τ)` with `τ = t/n`; structure is faithful to Theorem 6.
- **gamma-A K-kernel weights** — `(7/12, 3/16, 1/48)` weights in
  `DiffusionChernoff` and `Diffusion4thChernoff` match the paper's Appendix A
  γ-A scheme exactly.
- **Strang sandwich** — `StrangSplit` implements palindromic
  `D(τ/2) ∘ R(τ) ∘ D(τ/2)` per HLW 2006 §III.5 (credited as EXTENSION below;
  faithful to chosen reference).
- **RK2 drift-reaction** — `DriftReactionChernoff` uses Heun's method (explicit
  RK2) for the `b(x)∂ + c(x)` operator; yields order-2 convergence.
- **Theorem 7 2D** — `Strang2D` applies per-axis Strang composition with
  `AxisLift`; correctly instantiates the 2D semigroup approximation from
  Theorem 7 (math.md §10).

---

## Simplifications (acknowledged, ADR-documented)

- **f64 monomorphism** — all arithmetic fixed to `f64`; no generic `Float`
  parameter. Documented in ADR-0004. Planned lift in v0.9.0.
- **Bounded domain via BoundaryPolicy** — implementation requires a finite
  grid `[x_lo, x_hi]` with explicit boundary conditions
  (`Periodic | LinearExtrapolate`). The original theorem is on ℝ.
  Documented in ADR-0007.
- **`a(x) >= 0` allowed at isolated points** — strict ellipticity (`a > 0`
  uniformly) is not enforced; degenerate points are permitted.
  Consistency is the caller's responsibility (no panic, undefined behavior
  at degenerate points is documented).

---

## Approximations (now documented post-audit)

- **Cubic-Hermite joint mechanism** (math.md §9.2.4, post-audit clarification
  committed b8327d1) — spatial 4th-order accuracy in `Diffusion4thChernoff`
  arises from cubic-Hermite interpolation in `f.sample(off-grid)` (ADR-0005),
  not from a standalone 4th-order stencil. The 7-point FD computes derivatives
  inside the τ²-correction; the interpolant provides the actual off-grid
  evaluation. Both work jointly to yield the O(dx⁴) spatial slope confirmed
  in gate G3⁴.
- **Delta floor `3·dx` dominating in production** (math.md §9.2.4 line 1938,
  committed b8327d1) — the shift `Δ = sqrt(2a·τ) + |ζ_correction|` is floored
  at `3·dx` to keep the cubic-Hermite kernel within the stencil support.
  In production regimes (moderate τ) the floor is often active; the effective
  Δ is `3·dx`, not `sqrt(2aτ)`. This is documented and expected.
- **Boundary-layer order degradation** (ADR-0007) — near grid boundaries,
  `LinearExtrapolate` reduces local accuracy to O(dx). Global convergence
  is still O(dx⁴) in the interior but the boundary strip degrades.
  Reaffirmed by audit as known and documented.

---

## Deviations (require follow-up)

### D1 — `order()` mismatch (RED FLAG, real defect)

**Status**: Defect confirmed. Fix targeted v0.6.1.

**What is wrong**: `Diffusion4thChernoff::order()` returns `4` and
`Magnus4thDiffusionChernoff::order()` returns `4`. However, math.md §11.1
(normative) states `p = 2` for these types in v0.6.0, and the sympy
consistency gates (`Z⁴_τ⁰..τ²`) only verify the τ²-consistency slice.

**Mathematical reality**: The local truncation in `τ` is O(τ²) for variable
`a`, O(τ³) for constant `a`. The "4th order" refers to spatial accuracy (in
`dx`), not to the temporal-Chernoff consistency order, which governs
Lady-Windermere accumulation in `AdaptivePI`.

**Impact on AdaptivePI**: `AdaptivePI<Diffusion4thChernoff>` reads
`func.order() = 4` and:
- Uses Richardson divisor `2⁴ - 1 = 15` instead of `2² - 1 = 3`
  (Richardson error estimate is ~5× too small)
- Uses PI gains `alpha = 0.175, beta = 0.1` instead of `alpha = 0.35,
  beta = 0.2` (slower adaptation, larger accepted error)
- Result: `AdaptivePI` violates the user's `tol` contract for `Diffusion4th`
  and `Magnus4th` inner types; the adaptive loop accepts steps with true
  error larger than `tol` by a constant factor

**G_PI gate note**: The gate currently uses `DiffusionChernoff` (order 2) as
the inner type, so the gate passes. A gate exercising `Diffusion4thChernoff`
as the inner type would fail the tol contract.

**Recommended fix (Option A — correct)**:
- Change `Diffusion4thChernoff::order()` to return `2`
- Change `Magnus4thDiffusionChernoff::order()` to return `2`
- Update math.md §11.1 with a normative clarification distinguishing
  spatial order (dx axis) from Chernoff consistency order (τ axis)
- Re-run G_PI gate with `Diffusion4thChernoff` inner to confirm tol contract
  holds after fix

**Option B (rejected)**: Keeping `order() = 4` would require re-deriving
PI gains and Richardson divisor for genuinely O(τ⁴) truncation, which
the current implementation does not achieve. Option A is correct.

**Target**: v0.6.1 (no API surface break; `AdaptiveOutcome` step counts
will change).

---

### D2 — "Magnus" naming misnomer (cosmetic, academically misleading)

**Status**: Naming inaccuracy confirmed. Partial fix (rustdoc) targeted
v0.6.1; full rename deferred to v0.7.0 (semver MAJOR break).

**What is wrong**: `MagnusDiffusionChernoff` and
`Magnus4thDiffusionChernoff` implement a truncated power series
`sum_{k=0..4} (τ^k / k!) G^k f` — a Taylor approximation of `exp(τG)`
at constant `G`. This is NOT the Magnus expansion (Magnus 1954), which
constructs the exponential of integrated nested commutators (BCO-R 2009,
Blanes et al., *Phys. Rep.* 470, 151-238).

**Why the difference matters**: For variable-coefficient operators, the
genuine Magnus expansion requires time-ordering and commutator nesting.
The current implementation uses the truncated Taylor series without
commutator terms; the variable-coefficient correction is instead
"subsumed by the outer Strang sandwich" (math.md §9.2.3.C:1518). This
is a valid and documented design choice, but the type name implies
genuine Magnus.

**Correct name**: `TruncatedExpDiffusionChernoff` or
`TaylorExpDiffusionChernoff`.

**Practical impact at present**: nil for constant-coefficient + Strang
regime. Variable-coefficient users expecting genuine Magnus variable-
coefficient accuracy may misunderstand the error bounds.

**Partial fix (v0.6.1)**: Expand rustdoc on both types:
- Explicitly state this is a truncated Taylor expansion of `exp(τG)`
- Cite BCO-R 2009 (Blanes, Casas, Oteo, Ros, *Phys. Rep.* 470, 2009)
- Note genuine Magnus is deferred to a future implementation

**Full fix (v0.7.0)**: Rename types with clean break (no deprecation alias):
`MagnusDiffusionChernoff` → `TruncatedExpDiffusionChernoff`,
`Magnus4thDiffusionChernoff` → `TruncatedExp4thDiffusionChernoff`.
Callers must do a mechanical `s/MagnusDiffusionChernoff/TruncatedExpDiffusionChernoff/g`
substitution. SHIPPED in v0.7.0.

---

## Extensions (intentional, beyond Remizov 2025)

- **Strang composition** — `StrangSplit`, `Strang2D` implement palindromic
  operator splitting. Reference: HLW 2006 §III.5 (Hairer, Lubich, Wanner,
  *Geometric Numerical Integration*). Not present in Remizov 2025.
- **zeta-A tau²-correction** — `DiffusionChernoff` and `Diffusion4thChernoff`
  apply a BCH-derived τ²-correction to the shift argument for variable `a(x)`.
  First shipped v0.3.0; extended to 4th-order in v0.6.0. Not in Remizov 2025.
- **AdaptivePI** — `AdaptivePI<C>` wraps any `ChernoffFunction` with a Söderlind
  2002 PI step-size controller. Reference: Söderlind, *BIT Numer. Math.* 42,
  2002. Not in Remizov 2025.

---

## Process notes

- Researcher agent dispatched 2026-05-02 after v0.6.0 tag.
- Methodology: paper-by-implementation cross-reference using local sources
  (math.md, ADRs, src/, literature-validation.md).
- 30 tool calls, 17 findings classified.
- Full report (600+ words): `.dev-docs/reports/v0_6_0_math_fidelity_audit_2026-05-02.md`
  (gitignored, dev-only).

## Next steps

See [ROADMAP.md](../ROADMAP.md) for prioritized fix plan.
