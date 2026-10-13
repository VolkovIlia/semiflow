# §67 — Diagonal shift of the augmented φ operator (ADR-0210, NORMATIVE library)

## §67.1 — Identity

For `B̃_η = [[τG, ηW], [0, J_p]]` (§62.4) and any scalar `μ`,
`e^{B̃_η} = e^{μ}·e^{B̃_η − μI}` (`μI` commutes with `B̃_η`). With `μ = τḡ`:
`‖B̃_η − μI‖_∞ ≤ max(τ‖G − ḡI‖_∞ + ‖ηW‖_∞, |μ| + 1) ≤ max(τ‖G − ḡI‖_∞, |μ|) + 2`.

## §67.2 — Choice of `ḡ` (NORMATIVE)

`ḡ = (maxᵢ(dᵢ + rᵢ) + minᵢ(dᵢ − rᵢ))/2`, `dᵢ = Gᵢᵢ`, `rᵢ = Σ_{j≠i}|Gᵢⱼ|`: the
minimiser of `maxᵢ(|dᵢ − ḡ| + rᵢ) = ‖G − ḡI‖_∞`. Used iff the resulting Taylor
schedule `(s, m)` (`select_s_m` of the shifted bound × `PHI_NORM_TIGHTEN`) has
`s·m` strictly below the unshifted one.

## §67.3 — Sweep

`y ← e^{μ/s}·T_m((B̃_η − μI)/s)·y`, `s` times, with the step formulas of ADR-0210
Decision 1. Power-of-two homogeneity in the coupling is preserved.

## §67.4 — Gates

`G_PHI_SHIFT_COST`, `G_PHI_SHIFT_DENSE`, `G_PHI_SHIFT_HOMOGENEITY` (ADR-0210).
