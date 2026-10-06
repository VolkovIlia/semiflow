# ADR-0203 — Fréchet gradient at large `λ_max·t`: two-sided graded Duhamel quadrature

- **Status**: Proposed
- **Date**: 2026-10-06
- **Amends**: ADR-0185 (A2 `graph_expmv_frechet`, §54.5), ADR-0186 (§55.5 entry
  sensitivities, same code path). Independent of ADR-0202 (PR #43); numbering
  skips 0202 only because that number is taken on the unmerged branch.
- **Contract**: new `contracts/semiflow-core.math-frechet-large-t.md` (§63; pointer
  stub and a §54.5 note in `semiflow-core.math.md`). 5 RELEASE_BLOCKING + 1
  ADVISORY gates in `semiflow-core.properties.yaml`. Additive API only.

## Context

The SOFC network study (`remizov-publications/applications/sofc-channel/network/
PREREG-B1A.md` §9 F1) measured the edge gradients of `symmetric_op_expmv_frechet`
against `scipy.linalg.expm_frechet`: relative error 6e-14 at `λ_max t = 2.6`,
6.5e-8 at 13, 2.8e-4 at 53. Thermal networks need `λ_max t ≈ 1e5–2e6`
(`λ_max ≈ 300 s⁻¹`, windows of 30–120 min). All-edge gradients in `O(nnz)` memory
are the load-bearing element of the proposed application.

**Root cause (measured, §63.2).** `graph_expmv_frechet` evaluates the exact Duhamel
integral with ONE 8-point Gauss–Legendre panel on `[0, t]`. Mode pairs contribute
`e^{μσ}` with `|μ|` up to `λ_max`, so the integrand has boundary layers of width
`1/λ_max` at both ends and the GL remainder grows like `C_8 (|μ|t)^{17}`. It is not
Krylov dimension, restarts, Lanczos orthogonality or the forward action: a numpy
replica of the same 8-node rule with *exact* eigen propagators reproduces the
semiflow errors digit for digit (Evidence). A second defect on the same path: the
contraction loops `apply_param_deriv` over all parameters, zeroing an `n`-vector
each time, so an all-edges gradient cost `O(n·|E|)` per node, not `O(nnz)`.

## Contradiction (TRIZ gate)

Accuracy needs many quadrature nodes clustered in the two boundary layers; without
trajectory storage each node needs both `a(σ) = e^{−(t−σ)L}d` and `b(σ) = e^{−σL}v`,
which are propagated in opposite directions; storing either costs `O(N·nodes)`.
Physical contradiction: the paired vector must be *propagated incrementally*
(cheap) and *available out of order* (no storage). Resolved in space: split at
`t/2`. On each half one vector is **near** its source (distance `≤ t/2`, and on a
geometric mesh the sum of those distances is only `3·t/2`) and is recomputed
directly per panel into 8 buffers; the other is **far** and is advanced
monotonically. The resource already in the system is the decay of `e^{−rL}`:
it makes the mesh geometric (log nodes) and it bounds what a decay skip discards.

## Decision

1. **Mesh (§63.3).** Two halves, each graded towards its outer endpoint:
   `d_0 = min(t/2, 2/ρ̄)`, `d_k = min(t/2, 1.5·d_{k−1})`, GL8 per panel; nodes are
   carried as distances to the endpoint, never as `σ = t − r`. Built by a
   multiply loop (no `ln`): bit-identical across builds.
2. **A-priori bound (§63.4).** Relative quadrature error per mode pair
   `≤ ε_Q = 1.1e−14`, independent of `λ_max·t` (proof from the GL remainder and the
   geometric sum; probe measures `4.9e−16`).
3. **Evaluation order (§63.5).** Near/far sweep per half, outer panel to inner;
   memory `≤ 15·N` + operator, independent of `λ_max·t`, node count and
   `n_params`. Exact-accounted decay skip when `‖far‖ ≤ tol·‖far_src‖`.
4. **One-pass contraction (§63.6).** `GeneratorSensitivity::accumulate_bilinear`
   (provided default = old loop) with `O(nnz)` overrides for `EdgeWeightSensitivity`,
   `EntrySensitivity`, `NodeTimescaleSensitivity`.
5. **Plan predictor (§63.8).** `graph_expmv_frechet_plan(ρ̄, t, tol, path)` returns
   node count, chain length, max degree and an SpMV upper bound; the accuracy gate
   computes its threshold from it. Accessors `lambda_max_bound`, `tol`, `path` on
   `GraphKrylovChernoff`. Python: `tol` keyword on `symmetric_op_expmv_frechet`.
6. **Gate threshold (§63.7).** `τ_k = (ε_Q + 2n²u)·G_k + η·N_k` with `η` from
   propagator tolerance, Chebyshev rounding and the inherent `(r+n)·u·ρ̄t`
   conditioning term. Derived; the legacy rule (run with exact propagators) must
   fail its propagator-free part at every `λ_max t ≥ 10`.

## Complexity

Per channel, `n_nodes = 16(K+1)`, `K = ⌈log_{1.5}(ρ̄t/4)⌉₊` (512 at `ρ̄t = 1e6`).
SpMVs `≈ 5·C(t)` on the substepped Chebyshev path (old rule: `≈ 8·C(t)`), where
`C(t) ≈ 0.24·ρ̄t` today. Contractions `O(n_nodes·nnz)`, independent of
`n_params`. Memory `O(nnz + 15N)`. The number of actions does not depend on the
number of edges, so the all-edges gradient costs the same actions as one edge.

## Rejected

- **More fixed nodes / adaptive GL on `[0,t]`**: needs `O(λ_max t)` uniform nodes or
  an error estimator; the geometric mesh is optimal for exponential layers.
- **Dyadic semigroup recurrence `L(2τ) = e^{τA}L(τ) + L(τ)e^{τA}`**: exact, but the
  bilinear form needs `2^k` leaves or `O(k)` stored operator applications per level;
  either `O(λt)` work or trajectory storage.
- **Augmented `2N` block action (§54.5 text)**: one action per *direction*; the VJP
  over all edges still needs the Duhamel contraction, and the block action inherits
  the forward cost.
- **Store the far chain (`O(N·n_nodes)` memory, `≈ 2·C(t)` SpMVs)**: 2.5× cheaper,
  but memory grows with `log(λt)·N` (512·N at `1e6`). Kept as a possible opt-in.
- **Truncating stiff modes (quasi-static)**: changes the quantity; not exact.

## Consequences

- `graph_expmv_frechet` / `symmetric_op_expmv_frechet` / Python `graph_expmv_frechet`
  results change numerically (more accurate), not bit-identically. Existing gates
  keep their thresholds: `G_GRAPH_FRECHET_FD`, `g_graph_frechet_fd_triangle`,
  `G_SYMOP_ENTRY_FRECHET` (all `≤ 1e−7`), `T_ADJOINT_STATE_SENSITIVITY`.
- For small `ρ̄t` the call count rises from 16 to 32 actions; per-call minimum degree
  dominates there. Documented, ADVISORY gate only.
- The inherent floor `(r+n)·u·ρ̄t` (`≈ 2e−9` at `1e6`) is now stated: slow-mode
  gradients at `λ_max t = 1e6` cannot be computed to `1e−15` by any f64 method.
- `ImplicitEuler` path: accepted, outside the bound (`O(Δt)` bias), documented.
- MCP / capability / message-flow deliverables: not applicable (rlib, constitution
  Override #2, ADR-0027).

## Follow-up (not in this ADR)

The forward action is `C(t) ≈ 0.24·ρ̄t` SpMVs because the Chebyshev path substeps
at `z ≤ Z_SAFE = 200` (unscaled `I_k(z)` overflow). Computing `e^{−z}I_k(z)` scaled
removes substepping and gives degree `≈ √(2z·ln(1/tol))`: about `1e4` instead of
`1e6` SpMVs at `ρ̄t = 4e6`. That benefits every graph action and makes the
gradient `O(√(ρ̄t))`; it needs its own ADR and gates. Python `graph_expmv_frechet`
`params="all_edges"` (currently rejected by test) is a separate binding change.

## Evidence

Probe `frechet_exp.py` (scratchpad, reproduced in the engineer brief), `n = 12`,
max error over parameters relative to `G_k`:

| `λ_max t` | semiflow GL8 | GL8, exact props | graded, exact | graded, chained |
|---|---|---|---|---|
| random graph 1 / 10 / 53 / 1e2 | 1e-13 / 1e-10 / 7e-4 / 9e-3 | same | 1e-16 / 2e-16 / 5e-16 / 6e-16 | ≤ 9e-15 |
| random graph 1e4 / 1e6 | 6e-3 / 6e-5 | same | 2e-15 / 3e-15 | ≤ 8e-15 |
| stiff (`λ_max/λ_min = 2e8`) 10 / 53 / 1e3 / 1e6 | 2e-10 / 3e-4 / 1e-2 / 2e-5 | same | 4e-16 / 7e-16 / 5e-16 / 7e-16 | ≤ 1.4e-14 |

The eigen oracle agrees with `scipy.linalg.expm_frechet` to `≤ 2e−15·G_k` for
`λ_max t ≤ 1e2`.

## Amendment 1 — gate decisions after implementation (2026-10-06)

Gate-Change-Approved-By: ai-solutions-architect

1. **Cost model corrected (`G_FRECHET_COST_LOG_NODES` (ii), ADVISORY).** The
   "≈ 5·C(t)" estimate assumed every call costs SpMVs in proportion to its time
   step. Below `Z_SAFE` a call costs its degree `m(z) ~ √z`, and the near-vector
   sum missed the clipped last panel (`≤ 4H` per half, not `3H`). Derived
   replacement (§63.6.c): `spmv_upper ≤ m_Z·(3ρ̄t/Z_SAFE + 2 + 32(K+1))`, which is
   about `6·C(t)` asymptotically. That gives `B/C = 180 / 33 / 9.3 / 6.4` at
   `ρ̄t = 1e3…1e6`; measured `25.1 / 9.8 / 5.6 / 5.4`. Part (ii) now asserts the
   bound at all four points instead of a constant 6 from `1e3`. I did not restrict
   the claim to `ρ̄t ≥ 1e5`: the bound holds everywhere, and at small `ρ̄t` it is
   loose, not wrong.
2. **Non-vacuity is per carrier, not pooled (`G_FRECHET_LARGE_T_ORACLE`,
   `G_PY_FRECHET_LARGE_T`).** In the implementation, the 68 F2 entry parameters
   certified every point. Meanwhile no edge parameter was informative from
   `λt = 1e2` (F1) or at `1e6` (F2), so the edge path was untested exactly where
   the SOFC application needs it. This is expected for connected graphs with
   generic signals: edge gradients decay like `e^{−λ₂t}`. New fixture **F3**:
   four 3-node clusters with stiff internal weights `W`, three unit-scale bridges,
   and deterministic weights and signals (spec in properties.yaml). `W` is solved
   per point so that `λ₂·t = 1` whenever `λ_max t ≥ 47.5`, i.e. slow inter-cluster
   modes are resolved while intra-cluster modes are `λ_max t`-stiff. That is the
   thermal-network regime. The oracle probe gives the 2nd-largest bridge
   `|g|/N_k ≥ 2e−3` at every point, against `1e3·η ≈ 1e−4`. The carriers are
   F3-edge (≥ 2 of 3 bridges informative per point) and F2-entry (≥ 1 per point).
   The Python gate adds the F3 bridge entry pairs. F1-edge and F2-edge remain
   accuracy-only. The thresholds `τ_k` and `τ_k^Q` are unchanged.

## Amendment 2 — sharp algorithmic gate and the PSD precondition (2026-10-06)

Gate-Change-Approved-By: ai-solutions-architect

1. **New RELEASE_BLOCKING gate `G_FRECHET_SWEEP_EXACT_PROP` (§63.7.b).** Review
   showed that the end-to-end bound τ_k barely discriminates at `λ_max t = 1e6`.
   The measured error is about `2.9e−7·τ_k`, and the legacy rule exceeds τ_k by
   only about 4×. The cause is the `(r+n)·u·ρ̄t·N_k` floor. That floor is genuine
   for any method that applies `L` in floating point (SpMV cancellation on slow
   vectors), so I keep it in the end-to-end gate and do not tighten it there.
   Rejected alternatives:
   - A tighter spectral estimate than Gershgorin gains at most 2×.
   - A `G_k`-relative bound on informative parameters has no a-priori constant:
     propagator errors are vector-norm errors and can project onto any `M_k`.
   - Testing τ^Q against a test-side replica checks the test, not the library.

   Adopted: make the library's own sweep generic over a crate-private propagator
   and drive it, in an in-crate test, with exact-eigenvalue propagators that share
   the oracle's Jacobi decomposition. The eigensolver's backward error then
   cancels, and the bound becomes
   `τ_k^E = (ε_Q + 2n²u)·G_k + (2(1+8(K+1))·κ_E + n_nodes + 4)·u·N_k`,
   with `κ_E = 2n^{3/2} + n + 2`. At `ρ̄t = 1e6` this is about `5.6e−12·N_k`:
   about 600× tighter than the floor term alone, and independent of `ρ̄t` apart
   from `log(ρ̄t)`. It catches mesh, weight, step-length, role-swap and skip
   regressions. The production path must stay bit-identical after the refactor.
   No existing threshold changes.
2. **PSD precondition documented (§63.1.d).** `SymmetricOperator::from_csr` checks
   symmetry and `diag ≥ 0`, not positive semidefiniteness. The Chebyshev mapping,
   the decay skip and Proposition 63.4 all assume `σ(L) ⊂ [0, ρ̄]`; for indefinite
   input the results are unspecified and no error is raised. Weak diagonal
   dominance is a sufficient `O(nnz)` check, and conductance networks with
   non-negative leaks satisfy it. Turning it into a runtime check is a follow-up
   API decision outside this ADR.

## Amendment 3 — the decay skip gets a real test (2026-10-06)

Gate-Change-Approved-By: ai-solutions-architect

Amendment 2 claimed that `G_FRECHET_SWEEP_EXACT_PROP` covers the decay skip through
a second run with `ε_skip = 1e−14`. That claim was false. Propagator call counts
were identical with the skip on and off at every point. On the Laplacian fixtures
the zero mode keeps the far vector alive, and F2 has `λ_min·t ≲ 2`. SOFC networks
leak to ambient, so the skip will fire in production; it needs a test that
discriminates.

- **Why a uniform leak cannot test it (§63.7.c).** With `L + cI`, every Duhamel
  term carries the same factor `e^{−ct}`. The skip fires only when the whole
  gradient is already below `ε_skip·N_k`, so the dropped-part bound covers it,
  but no broken skip can be detected that way. Discrimination needs non-uniform
  decay between the two sources.
- **Fixture S1.** F3 with `dj` projected onto the fast intra-cluster eigenspace;
  `u0` keeps its slow content. The left half's far vector then decays and the
  skip fires there. The right half carries the whole gradient, so a wrongly fired
  skip loses `|g_k|`. A-priori onset: `λ_f(t − δ/ρ̄) ≥ ln(2/ε_skip)`, which with
  `ε_skip = 1e−12` holds at `λ_max t ∈ {1e2, 1e4, 1e6}`.
- **Bound.** The skip fires at most once per half (§63.5 returns on the first
  success), so the dropped part is at most `(ε_skip + L_E κ_E u)·N_k/2` per half.
  The assertion is `|g_k − g_k^ref| ≤ τ_k^E + ε_skip·N_k`; no new constant.
- **Non-vacuity, two ways.**
  - Call counts: strictly fewer calls with the skip on at the three onset points.
  - Two negative tests must violate the bound at every `λ_max t ≥ 10`: a skip
    that always fires, and an unscaled-norm skip on signals scaled by `1e−16`.
- **Probe (numpy replica of the sweep).** Correct skip: err/bound `≤ 6.6e−4`.
  Calls `322/178`, `674/354`, `1058/546` at `1e2/1e4/1e6`. Negative tests:
  `3.1e3` to `2.9e9` × the bound.
- The §63.7.b wording is corrected; part A of the gate now runs with the skip off
  only.
- Informational: the end-to-end legacy margin on F3-edge at `λ_max t = 1e6` is
  only 1.026 against `τ^Q`. The sharp check at that point is part A of
  `G_FRECHET_SWEEP_EXACT_PROP` (legacy err/`τ^E` = 4.1e2).
