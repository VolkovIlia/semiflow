# ADR-0185 — Depth-independent graph-semigroup action + edge-weight Fréchet gradient via Krylov

- **Status**: Proposed (design only — Phase 1 of the `revssm`/ML roadmap; branch TBD)
- **Date**: 2026-06-26
- **Supersedes**: none — purely ADDITIVE over ADR-0121 (`expmv` 1-D action),
  ADR-0047 (graph-heat), ADR-0051 (Magnus graph), ADR-0115 (graph adjoint
  sensitivity), ADR-0180 (presampled adjoint), ADR-0184 (batched multi-channel
  evolve). No existing kernel, gate, or contract entry is changed.
- **Contract**: `contracts/graph-batched-evolve.contract.md` §5 (delta),
  `contracts/semiflow-core.math.md` §54 (new NORMATIVE section).

## Context

The graph-heat kernels (`GraphHeatChernoff`, `MagnusGraphHeatChernoff`) integrate
`e^{−tL_G}·v` by `n` small Chernoff steps `(S(t/n))^n`. ADR-0121's `expmv`
(Al-Mohy–Higham 2011, `DiffusionExpmvChernoff`) already replaces per-step
stepping with a single tolerance-driven action `e^{τA}·v` — but **only for the
1-D divergence-form carrier** (`apply_div_form`); it is not wired to the sparse
graph Laplacian SpMV. So two of the five #10 speed ceilings remain open
(CHANGELOG v0.9.1-beta):

- **(c)** graph kernels still take `O(n_steps)` SpMVs for a depth-`t` action;
- **(d)** the backward edge-weight gradient `adjoint_state_gradient_batched`
  (§43.4) is Rust-compute-bound `O(edges · C · n_steps)` — batching (ADR-0184)
  does not lower it.

**Genuine contradiction (already resolved upstream, recorded here as NORMATIVE
rationale).** The number of operator applications (matvecs) must be **large**
(to reach accuracy / large "depth" `t`) and **small** (for speed) at once. The
roadmap's АРИЗ chain (АП → ТП → ФП → ИКР) resolves it by *separation in
structure*: black-box stepping discards the **spectral structure of `L_G`**
(symmetric PSD, bounded spectrum) — a resource already present in the topology.
Krylov (Lanczos) and Chebyshev recover it for free: the matvec count for
accuracy `ε` is `≈ √(t‖L‖)·polylog(1/ε)`, set by `t‖L‖` and `ε` **and is flat
in the number of "steps"**. Depth `t` becomes a continuous parameter decoupled
from a step count. This is not a compromise: in the linear/symmetric regime the
system holds **both** O(1)-in-depth memory **and** few-matvec speed.

**Second, hidden contradiction (closes ceiling (d)).** Even a one-call forward
leaves the gradient `∂/∂w (e^{−tL(w)}·v)` at `O(edges·n_steps)`. The Fréchet
derivative of the matrix exponential has the exact augmented-matrix
representation (Al-Mohy & Higham 2009)
`exp([[A,E],[0,A]]) = [[e^A, L(A,E)],[0,e^A]]`, so the directional derivative is
**one augmented action at ≈ forward cost**, depth-independent. For "all edge
weights at once" the VJP/adjoint form (one augmented solve seeded by the
upstream cotangent) replaces per-parameter JVP.

This ADR is **mechanism only** (a calculator: operator + vectors in → numbers /
gradients out). No autograd, tape, or training loop — those stay in `revssm`
(ADR-0115 boundary, reaffirmed).

## Decision

Add two ADDITIVE core primitives, reusing the existing sparse Laplacian SpMV,
`ScratchPool`, the `[C, N]` batched contract (ADR-0184), and the
`GeneratorSensitivity` trait (§43.2). New files `graph_krylov.rs` and
`graph_frechet.rs`; one new NORMATIVE math section (§54); a contract delta; a
gate triple. Engineer ships **A1 first, then A2**.

### D1 — A1 `graph_expmv`: Chebyshev default (no basis), Lanczos adaptive path; both depth-independent

`graph_expmv(L, v, t, ε)` computes `e^{−tL_G}·v`. Two paths, one public surface:

- **Chebyshev (default, O(1) extra vectors).** `L_G` symmetric PSD ⇒ spectrum in
  `[0, λ_max]`. Compute `λ_max` ONCE per graph via Gershgorin (free from CSR row
  sums; `Laplacian::spectral_radius_bound()` already supplies a bound) or a
  cheap power iteration; expand `e^{−tλ}` in Chebyshev polynomials on
  `[0, λ_max]`, applied by the 3-term recurrence — **two work vectors, no Krylov
  basis stored**, degree `m(ε, tλ_max)` set by Bessel-coefficient decay.
- **Lanczos (adaptive / tight-tolerance).** Mirrors `expmv.rs`'s
  tolerance-driven `(s, m)` selection (Al-Mohy–Higham 2011) but on the graph
  SpMV: build a depth-independent Krylov basis `m ≈ 20–40` by the symmetric
  3-term recurrence, exponentiate the tridiagonal `T_m` densely (reuse
  `matrix_system_exp::mat_exp_pade13`), project back. Stores `m` basis vectors
  (`O(m·N)`, depth-INDEPENDENT). Used when the Chebyshev spectral bound is loose
  or a strict per-call tolerance is requested.

Both are exposed as a sibling evolver next to `graph_heat.rs`. `order()` returns
`u32::MAX` (tolerance-driven, NOT fixed-order — same contract as
`DiffusionExpmvChernoff`, ADR-0121); callers MUST NOT compare it to slope gates.
The mandatory marshalling and one-`py.detach` boundary (ADR-0184 D1/D2) are
unchanged: the new path is a dispatch target inside `evolve_batched`.

### D2 — A2 `graph_expmv_frechet`: one augmented VJP for the full edge-weight gradient

`graph_expmv_frechet` returns `∂J/∂w` (length `n_params`) for
`J` a scalar of `e^{−tL(w)}·v`, via the augmented operator
`Â = [[−tL, −tE],[0, −tL]]` acting on `2N` vectors (Al-Mohy–Higham 2009). The
**adjoint/VJP** form seeds the augmented action once with the upstream cotangent
and reads the rank-1 edge stencil `∂L/∂w_{ij} = (e_i−e_j)(e_i−e_j)ᵀ` (§43.2) out
of `GeneratorSensitivity::apply_param_deriv` — **one augmented Krylov/Chebyshev
solve for all edges**, not `O(edges)` JVPs and not `O(edges·n_steps)` per-step
sweeps. The augmented `Â` is block-upper-triangular with the SAME symmetric
`−tL` on the diagonal, so the diagonal action reuses the A1 path; only the
off-diagonal coupling `−tE` differs per seed.

### D3 — Same summed `∂J/∂w`, but "same" is NUMERICAL, not 0-ULP (D5 does not transfer across algorithms)

A2 returns the SAME mathematical quantity and the SAME shape/length and the SAME
ascending-channel accumulation as `adjoint_state_gradient_batched` (ADR-0184 D4
0-ULP invariant is preserved **within** A2, across the channel sum). But A2 is a
*different algorithm* (augmented Krylov of the continuous action) from the
existing *per-step Magnus discrete adjoint*, so it is **not** bit-identical to
it: the two agree only as `n_steps → ∞` / at matched accuracy. Therefore:

- ADR-0184 **D5 structural bit-equality transfers to the channel-batching axis
  only** (batched-Krylov == per-channel-Krylov loop, same kernel `C` times →
  0-ULP, no new oracle for that axis), exactly as in #10.
- Equivalence to the *old* per-step gradient is a **numerical** claim, gated by
  the existing §43.6 finite-difference oracle (`T_ADJOINT_STATE_SENSITIVITY`,
  `scripts/verify_adjoint_state_sensitivity.py`) — which finite-differences
  `J(θ±εδθ)` and is **method-agnostic**, so it covers A2 unchanged. No new sympy
  oracle is needed for A2.

### D4 — Gate triple (definitions in §54; entries in `properties.yaml`)

1. `G_GRAPH_EXPMV_DENSE` (RELEASE_BLOCKING): `graph_expmv` vs dense `e^{−tL}·v`
   (dense `L` from CSR + `mat_exp_pade13`) on a small graph, `sup_error ≤ 1e-10`.
   Type `BACKWARD_ERROR`; no slope (tolerance-driven). REUSES the dense oracle —
   no sympy (mirrors `G_MATRIX_PADE_M5`).
2. `G_GRAPH_FRECHET_FD` (RELEASE_BLOCKING): A2's `⟨∂J/∂w, δw⟩` vs central FD
   `(J(w+εδw)−J(w−εδw))/(2ε)`, rel-err `≤ 1e-7`. REUSES `T_ADJOINT_STATE_SENSITIVITY`
   (§43.6) harness — no new oracle.
3. `G_GRAPH_EXPMV_DEPTH_FLAT` (RELEASE_BLOCKING): matvec-count(`ε`, `t`) is flat
   in `t` at fixed `ε` (instrumented counter; assert count for `t ∈ {1,4,16,64}`
   stays within a small constant band, vs the linear-in-`t` per-step baseline).
   Structural/perf gate; no oracle.

### D5 — Honest boundary

Symmetric `L_G` only (Lanczos/Chebyshev). **Non-symmetric / drift / directed**
graphs need Arnoldi (loses the 3-term recurrence, stores full Hessenberg) —
DEFERRED, does not block the symmetric graph case. **Time-varying `L(t)`** ⇒
evolution family, not a semigroup; Magnus/Howland (already in core) apply but
matvec count grows — OUT OF SCOPE here. **Genuinely nonlinear / state-dependent
generators** have no semigroup — `revssm` uses checkpointing there; the core does
not pretend to (no moat). These are fail-loud `SemiflowError::Unsupported`, not
silent fallbacks.

## Consequences

- **Positive**: closes #10 ceilings (c) and (d) — forward and edge-weight
  backward both become depth-independent (`O(m·edges)`, `m` set by `ε, t‖L‖`,
  flat in `t`) at O(1)-in-depth memory; reuses every existing primitive (SpMV,
  `ScratchPool`, `[C,N]` batching, `GeneratorSensitivity`, dense `mat_exp_pade13`,
  the §43.6 FD oracle); zero new dependency; ADDITIVE; the `revssm` Semigroup
  Layer (Track B) gets a forward + full-gradient native call with no per-step
  trajectory storage.
- **Negative / risk**: Lanczos stores `m` basis vectors (`O(m·N)`, depth-
  independent, but not O(1)-in-`N`) — Chebyshev is the O(1)-vector default for
  exactly this reason; Lanczos loss of orthogonality at large `m` mitigated by
  the `m ≈ 20–40` cap (depth-independence makes large `m` unnecessary). The
  augmented `2N` Fréchet action doubles the working set vs forward (still
  depth-independent). "Same gradient as the per-step path" is numerical, not
  0-ULP (D3) — the contract and gate names state this explicitly to avoid a false
  bit-equality expectation.
- **Boundary (ADR-0115 reaffirmed)**: core ships the depth-independent MATH
  primitives only; tape / autograd / training stay in `revssm`.

## References

- A. H. Al-Mohy, N. J. Higham (2011), *Computing the action of the matrix
  exponential*, SIAM J. Sci. Comput. 33(2):488–511, DOI 10.1137/100788860 — A1
  tolerance-driven action (already cited §45 / ADR-0121).
- A. H. Al-Mohy, N. J. Higham (2009), *Computing the Fréchet derivative of the
  matrix exponential, with an application to condition number estimation*, SIAM
  J. Matrix Anal. Appl. 30(4):1639–1657, DOI 10.1137/080716426 — A2 augmented-
  matrix Fréchet identity.
- Y. Saad (1992), *Analysis of some Krylov subspace approximations to the matrix
  exponential operator*, SIAM J. Numer. Anal. 29(1):209–228 — Lanczos/Krylov
  action error bound.
- math.md §42 (transpose-exact state adjoint), §43 (adjoint-state sensitivity +
  §43.6 FD oracle), §45 (1-D `expmv`); ADR-0121, ADR-0184.
