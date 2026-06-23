# ADR-0178 — TT-carrier variable-coefficient phase-1: per-axis diagonal `a_j(x_j)` with bounded rank, fail-loud wall for everything else

**Status:** PROPOSED (design-only; no implementation this round). **Date:** 2026-06-23.
**Branch:** `issue-2-tt-varcoef`. **Issue:** #2 (extend `TtChernoff` toward variable / non-diagonal A).
**Builds on:** ADR-0159 (TT carrier), ADR-0162 (band-split coupling + fail-loud walls),
ADR-0166 (additive-separable variable-coef, FLAT `n^d` POC). **Math:** §52.10 (new), §52.4 (rank cap).
**Engineer hand-off:** `docs/adr/0178-engineer-handoff.md`.

## Context

`TtChernoff` (§52, `tt_chernoff.rs`) escapes the exponential curse `n^d → O(d·n·r²)` **only** for
constant diagonal-A: each per-axis shift `h_j = 2√(a_jτ)` is a **single permutation** band (QTT-rank
≤ 2), so an evolved rank-1 IC stays rank-1 and the Rohrbach (2022) cap `r ≤ ⌊d/2⌋` holds (§52.4).

The variable-coefficient case **already exists on this branch — but on FLAT `n^d` storage**:
`S3VarCoefEvolver` (`tt_varcoef_spectral.rs`, ADR-0166) proves the per-axis step
`P₂(τ/2)·k(τ)·P₂(τ/2)` is order-2 for additive-separable `a_j(x_j)`, validated by
`g_s3_varcoef_spectral`. That gate proves convergence and an **operator**-TT-rank-1 fact, but it
allocates `n.pow(d)` and never measures the **state carrier's** `peak_rank()`. **The curse-escape
of the variable-coef step is therefore unproven** — it has never been run on `TtState`.

**The rank-cap contradiction (the inventive core).** The cap holds because constant-diagonal-A keeps
each axis operator a permutation. A variable-coefficient step needs `R_j = L_j − a₀·Lap`, whose
core is `diag(a_j(x_j))·(tridiagonal stencil)` — a **position-dependent multiplier**. Applying
`diag(a_j(x))` to a TT core multiplies bond rank by the TT-rank of `a_j` viewed as a grid function.
For arbitrary `a_j(x)` that rank is `O(n)` and the curse returns. We need **heterogeneous diffusion
AND a bounded rank cap** — properties that ordinarily exclude each other.

## Decision

Phase-1 scope: **per-axis diagonal variable diffusion `a_j(x_j)`, drift `b_j(x_j)`, reaction
`v_j(x_j)`, additive-separable (`L = Σ_j L_j`), carried natively on `TtState`** — the TRIZ-(a)
class. New **additive** sibling `src/tt_varcoef.rs`; `tt_chernoff.rs` is **untouched**.

**Discretised variable step (per axis `j`, acting on core `G_j` only).** The inter-axis split is
EXACT (`[L_j,L_k]=0`, rank-1 operators on disjoint axes — ADR-0166 Layer-1), so the whole
`d`-dependence is solved by sweeping one axis at a time. Per axis the carrier-native factor is the
ADR-0166 sandwich rewritten as **TT-core operators**:

```
G_j ← P₂(τ/2) · k(τ) · P₂(τ/2) · G_j          (acts on the single core G_j; identity elsewhere)
k(τ)  = exp(τ·a₀_j·Lap_j)              a₀_j = mean_x a_j(x_j)   (CONSTANT factor → permutation-band, QTT-rank ≤ 2)
R_j   = L_j − a₀_j·Lap_j               (the variable residual, periodic tridiagonal)
P₂(s) = I + s·R_j + (s²/2)·R_j²        (2 tridiagonal mat-vecs on the core's mode axis)
```

`R_j` and `k(τ)` act **only on the `n`-mode index of core `G_j`** (a `r_{j-1} × n × r_j` slab),
leaving the bond indices `r_{j-1}, r_j` as passive spectators. A mode-axis tridiagonal mat-vec is a
**bond-rank-preserving** operation: it remaps the `n` slices of the slab without touching the bond
dimensions, so it cannot increase `r_{j-1}` or `r_j`. After the full forward+backward axis sweep,
`tt_round(eps_round)` recompresses — exactly as in `tt_chernoff::step`.

**Why the rank stays bounded for THIS class (bounded-rank story).** Two facts compose:
1. **Inter-axis is rank-exact.** Each `exp(τL_j)` is `E_j ⊗ I^{⊗(d−1)}` — a rank-1 TT operator
   (ADR-0166 Layer-1, operator-TT-rank=1). Applying a rank-1 operator multiplies state bond rank by
   1 — it cannot grow it. This is the entire `d`-scaling, and it is exact.
2. **Intra-axis acts on ONE core's mode axis.** `R_j`, `R_j²`, `k(τ)` are `n×n` operators on the
   mode index of `G_j` alone. A mode-axis linear map is `r_{j-1}×n×r_j → r_{j-1}×n×r_j` with the
   bond dims fixed — **structurally rank-preserving on every bond**. The cubic-band `tt_round`
   tolerance only ever *shrinks* rank. Therefore an IC of bond rank `r₀` evolves at peak rank
   `≤ r₀` for the lifetime of the evolution; for a rank-1 (separable-Gaussian-like) IC the state
   stays **exactly rank-1**, storage `O(d·n)` — the §52.3 Strang⊗ floor, now with heterogeneous
   per-axis `a_j(x_j)`. The Rohrbach cap `r ≤ ⌊d/2⌋` continues to bound any low-rank correlated IC
   because the per-axis factors never inject cross-axis correlation (they are `E_j ⊗ I`).

**The explicit fail-loud wall (everything NOT covered).** The phase-1 type `VarCoefTt` accepts only
**per-axis arrays** `a_axis[j]: [F; n_j]` (mirroring `AxisCoef`). By construction it CANNOT represent:
- **non-separable / cross-axis `a(x_i,x_j)`** — no per-axis array can encode a 2-D field; the
  constructor has no slot for it. (Low-CP-rank non-separable already lives in `S3NonSepVarCoefEvolver`
  on flat `n^d` — out of *this* TT-carrier phase-1; a future ADR-0179 may lift it.)
- **non-diagonal constant A (cross-axis coupling)** — handled by `CoupledTtChernoff`/ADR-0162 for
  the adjacent-pair band; dense/non-adjacent remain ADR-0162's `CouplingError` walls.
- **dense full-rank `a(x)`, nonlinear `f(u)`, time-dependent `a(x,t)`** — return
  `SemiflowError::VarCoefOutOfClass { detail }` at construction (fail-loud, never a silent
  wrong-operator floor). A parabolicity check `a_j(x) > 0 ∀x` is fail-loud at construction.

## TRIZ resolution (ИКР)

- **НЭ:** a variable-coefficient diffusion step injects a position-dependent multiplier, which grows
  TT bond rank without an algebraic cap → the curse-escape (the only reason the carrier is worth it)
  is destroyed.
- **ТП:** *инструмент* = the per-axis variable factor `P₂·k·P₂`; *изделие* = the TT state's bonds.
  - ТП-1: a factor expressive enough to be a full position multiplier `diag(a(x))` over the whole
    `n^d` grid gives genuine heterogeneity (польза) but its multiplier has rank `O(n)` and explodes
    the bonds (вред).
  - ТП-2: a constant-coefficient permutation factor (the §52 status quo) keeps bonds rank-1 (no вред)
    but cannot vary `a` in space (no польза). **Chosen half: ТП-2, strengthened** — keep the
    bond-preserving permutation backbone, but make the variation *appear on the mode axis of a single
    core*, not as a global multiplier.
- **ФП:** the residual `R_j` must be **expressive in space** (vary `a_j` node-by-node) AND
  **invisible to the bonds** (rank-preserving). Resolution **in structure (super-system) + in space**:
  separate the two conflicting properties onto *different index sets of the same tensor* — the **mode
  index** carries the spatial variation, the **bond indices** carry the cross-axis correlation, and
  the variable operator is confined to the mode index of one core, where the bond index simply rides
  along untouched.
- **Ресурсы (ВПР — already in the topology, ~free):** (1) the per-axis Strang split → inter-axis
  factors already commute and are rank-1 — no new mechanism; (2) the `P₂·k·P₂` sandwich already
  exists in `tt_varcoef_spectral.rs` — reuse the math, re-target it from a flat `n`-line to a TT
  core slab; (3) `tt_core::tt_round` already recompresses — reuse verbatim.
- **ИКР:** *the variable coefficient varies `a_j` freely in space yet, because its operator lives
  only on one core's mode axis while the bond indices ride through untouched, the bond rank is
  unchanged — the rank cap survives the variable coefficient at zero extra rank cost.* The "machine"
  that would normally grow the rank is structurally absent: there is no global `diag(a(x))` multiplier
  anywhere, only `n×n` mode maps confined to a single core.

This is a genuine resolution, not a compromise: the phase-1 class has **full per-axis spatial
heterogeneity AND the unchanged rank cap simultaneously** — neither property is traded away. The
honest price paid is *scope* (per-axis-separable only), not *rank* — and that scope boundary is a
hard fail-loud wall, not a soft degradation.

## Consequences / limits (the honest INTRINSIC_LIMIT)

- **Covered (real win):** per-axis diagonal `a_j(x_j), b_j(x_j), v_j(x_j)`, additive-separable, on
  the TT carrier, order-2 in τ, peak rank bounded (rank-1 IC ⇒ exactly rank-1; Rohrbach cap ⇒
  `≤ ⌊d/2⌋`). Proven by the NEW gate `G_TT_VARCOEF` (convergence AND measured `peak_rank`) — the
  first gate to run a variable-coef step on `TtState` and check the carrier rank.
- **INTRINSIC_LIMIT (uncovered, honest):** **truly non-separable `a(x_i,x_j)` cannot be carried at
  bounded rank by ANY representation when its CP/TT-rank grows with the grid** — this is intrinsic to
  the solution's rank structure (§52.4 / §52.6, Rohrbach–Dolgov–Grasedyck–Scheichl), not a defect of
  this design. The low-CP-rank non-separable sub-case is *representable but unproven on the TT
  carrier* (it exists only on flat `n^d` as `S3NonSepVarCoefEvolver`); lifting it to `TtState` with a
  measured rank-growth gate is a candidate **phase-2 (ADR-0179)**, explicitly deferred. Non-diagonal
  constant A is owned by ADR-0162 (adjacent-pair) with its dense/non-adjacent walls.
- **Boundary discipline:** every uncovered case is a typed `VarCoefOutOfClass` / existing
  `CouplingError` / `S3OutOfClass` fail-loud error or a structurally-unrepresentable type — **never a
  silent wrong-operator floor**, mirroring ADR-0162's `b≠0` / non-adjacent-pair walls. A documented
  INTRINSIC_LIMIT here is the honest outcome, not a gap.
- **No trait change:** `tt_varcoef.rs` is a standalone evolver (like `TtChernoff`); it does NOT touch
  the `ChernoffFunction` trait signature. Additive sibling; ≤3 deps unchanged (in-tree `tt_core` only).
