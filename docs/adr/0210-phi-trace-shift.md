# ADR-0210 — Diagonal shift of the augmented φ operator

- **Status**: Accepted
- **Date**: 2026-10-13
- **Supersedes / amends**: amends ADR-0202 §62.4 (the φ sweep schedule); output bits
  of φ-actions on CSR / div-form / graph generators change (one digest re-recorded).
- **Contract**: math §67 (`contracts/semiflow-core.math-phi-shift.md`); gates
  `G_PHI_SHIFT_COST`, `G_PHI_SHIFT_DENSE`, `G_PHI_SHIFT_HOMOGENEITY`.

## Context

`phi_action`, `phi_action_batched`, `phi_combination` and everything built on them
(ETDRK4, ETD/affine steps, the steady-state `φ₁` limit) evaluate `e^{B̃_η}` of the
augmented operator `B̃_η = [[τG, ηW], [0, J_p]]` by `s` Taylor sweeps of degree `m`
chosen from the bound `τ‖G‖ + 2` (ADR-0198/0202). For a diffusion generator the
Gershgorin hull of `G` is `[−‖G‖, 0]`: half of the bound is spent on the hull's
offset from the origin, which a scalar shift removes for free.

## Decision

1. **Shift.** `μI` commutes with every block, so `e^{B̃_η} = e^{μ}·e^{B̃_η − μI}` for
   any scalar `μ`. With `μ = τ·ḡ` the shifted operator is bounded by
   `max(τ‖G − ḡI‖_∞, |μ|) + 2` (top rows: shifted generator plus the η-scaled coupling
   `≤ 1`; bottom rows: `J_p − μI`). Each of the `s` sweeps applies
   `T_m((B̃_η − μI)/s)` and rescales by `e^{μ/s}`:
   `w_u ← (τ/js)·Gw_u − (μ/js)·w_u + Σ c_j col_j`,
   `w_c[i] ← (1/js)·w_c[i+1] − (μ/js)·w_c[i]`, bottom `w_c[p−1] ← −(μ/js)·w_c[p−1]`.
2. **Which `ḡ`.** The centre of `G`'s Gershgorin hull,
   `ḡ = (maxᵢ(dᵢ + rᵢ) + minᵢ(dᵢ − rᵢ))/2` (`dᵢ` diagonal, `rᵢ` off-diagonal absolute
   row sum), which minimises `‖G − ḡI‖_∞` over all scalars. The trace mean (used by
   the Taylor `expmv`, ADR-0205, and kept there) is a poor centre when the diagonal
   varies: with a lumped mass of contrast 50 it saved 8 % of the generator
   applications; the midpoint saves 46 %.
3. **When.** A provided method `GeneratorAction::diagonal_shift() -> Option<(ḡ,
   ‖G − ḡI‖_∞)>` (default `None`: external generators keep today's arithmetic bit
   for bit) is implemented for `CsrGenerator` (from the CSR or the row-normalised
   mass values), `DivFormGenerator` (from the face coefficients) and
   `NegLaplacianGenerator` (through a new provided `SymmetricLinearOp::csr_view`,
   so it computes the same `ḡ` from the same triples as
   `CsrGenerator::from_symmetric(op, None)` and `G_PHI_MASS_DENSE` (3) stays bitwise).
   The shifted schedule is used only when its `s·m` is strictly smaller than the
   unshifted one; otherwise the unshifted code runs verbatim (not with `μ = 0`).
   `phi_cost_probe` stays the unshifted, norm-only probe; the new `phi_sweep_probe(op,
   τ)` returns the schedule actually used.
4. `PHI_NORM_TIGHTEN` (ADR-0198) applies to the shifted bound unchanged.

## Consequences

- Generator applications per φ sweep, 64-node diffusion operators (deterministic,
  `G_PHI_SHIFT_COST`): `τ‖G‖ = 40`: 720 → 390; `τ‖G‖ = 400`: 6 840 → 3 510 (DivForm
  3 450). ETDRK4 (four `phi_combination` sweeps per step) gains the same factor on
  these generators. Wall clock, same session back to back (shift disabled vs
  enabled): `phi_action` φ₁, N = 1000, `‖G‖t ≈ 4e3`: 300 → 145 ms; ETDRK4
  Allen–Cahn, N = 256, 100 steps: 195 → 96 ms.
- Bitwise homogeneity `φ(2ʲv) = 2ʲφ(v)` holds on the shifted path (scalar products
  commute with power-of-two scaling); gated.
- Bits change for φ outputs on the built-in CSR / div-form / graph generators: the
  no_std digest `phi_combination` is re-recorded once.

## Honest limits

- Gershgorin bounds only; a generator whose hull is already centred (skew drift)
  gains nothing and keeps the unshifted path.
- `e^{μ/s}` is applied per sweep; with `|μ|/s ≤ θ_m` it cannot overflow.

## Gates

- `G_PHI_SHIFT_COST` (`tests/g_phi_shift.rs`): exact counter around four generators
  (graph, CSR, CSR + mass, div-form; n = 64) at `τ‖G‖ ∈ {40, 400}`: calls `= s·m` of
  `phi_sweep_probe` and `≤ 0.6×` the unshifted `s·m`.
- `G_PHI_SHIFT_DENSE`: shifted path asserted active, `φ_k` (`k ≤ 3`) and a full
  combination vs the dense augmented exponential, n = 10, mass contrast 50, symmetric,
  general and graph generators, `τ‖G‖ ∈ {2, 20, 200}`: `≤ 1e−12·max(1, ‖want‖∞)`
  (measured ≤ 7.1e−15). The combination reference sums the dense `φ_k`: the dense
  augmented exponential has no η-scaling, so `τᵏ ≈ 5e5` inside it cost it 4 digits.
- `G_PHI_SHIFT_HOMOGENEITY`: `φ_k(2⁴⁰v) = 2⁴⁰φ_k(v)` and the same for a combination,
  bitwise, on the shifted path.
- All existing φ gates pass unchanged (`G_PHI_MASS_DENSE` incl. its bitwise (3),
  `G_PHI_AUG_DENSE`, `G_PHI_COMBINATION_DENSE`, `G_PHI_GENERAL_DENSE`,
  `G_PHI_COST_V_INVARIANT`, `G_ETDRK4_SWEEPS`, `G_ETDRK4_ORDER`, `G_ETD_*`).
