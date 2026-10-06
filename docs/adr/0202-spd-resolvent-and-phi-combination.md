# ADR-0202 — SPD resolvent / steady solve, operator composition, and the φ-combination

- **Status**: Proposed
- **Date**: 2026-10-06
- **Supersedes / amends**: ADDITIVE. Amends ROADMAP "Out of scope: Fully-implicit
  schemes" (wording narrowed, see D6). Amends the `(s, m)` selection of
  `phi_action` / `phi_action_batched` (§58.2): cost no longer depends on `‖v‖`
  (a defect fix; results move by at most the truncation-tolerance level, see D4).
  Implements the IC(0) preconditioner that §59.2 already declares NORMATIVE but
  `pcg.rs` never shipped.
- **Contract**: math §62 (`contracts/semiflow-core.math-spd-resolvent.md`, split out of math.md for size); `contracts/semiflow-core.spd-resolvent-api.md`;
  gates `G_SPDR_*`, `G_SYMOP_COMPOSE_*`, `G_PHI_COST_V_INVARIANT`,
  `G_PHI_COMBINATION_DENSE`, `G_PHI_MASS_DENSE`, `G_PHI_GENERAL_DENSE`,
  `G_ETD_AFFINE_EXACT`; errors.yaml entries; nostd-check digests (ADR-0200 jobs).

## Context

An applied case study (1-D conservative heat equation with distributed sink and
source: `−∂ₓ(k∂ₓT) + G(x)T = s(x)`, plus a stiff semilinear transient) measured
the library against scipy. Steady: `τφ₁(−τÂ)s` at `τ = 40/λ_min` was exact
(1e-10..1e-8 vs `spsolve`) but cost 3.4 s / 187 s at N = 161 / 641 against
0.28 s / 1.1 s. Transient (hand-written ETD-RK2 on `phi_action`): 70 s against
25 s for BDF, error 0.058 K against a 1e-3 K target. Nothing in this ADR is
specific to that study; it only exposed three universal defects.

**Diagnosis (what already exists and why it did not help).**

1. *The N³ is inherent to every polynomial method used for a steady solve.*
   Taylor `expmv` needs `s·m ∝ τ‖A‖` matvecs (`expmv.rs:128`, `phi_action.rs:51`).
   For the steady limit `τ ≥ 40/λ_min`, so the work is `∝ κ(A) = λ_max/λ_min ∝ N²`
   matvecs of `O(N)` each. Measured: `τλ_max ≈ 2.9·10⁴` at N = 161. Chebyshev
   (`KrylovPath::Chebyshev`) would cut this to `O(√κ)` matvecs (`O(N²)` total) but
   computes `e^{−τA}` only, not `φ₁` or a solve. Lanczos (`m_max ≤ 18`,
   `graph_krylov.rs:65`) substeps `∝ τλ_max` too. `KrylovPath::ImplicitEuler`
   (`(I+ΔtÂ)^{−n}` by Jacobi-PCG, `pcg.rs:272`) is a first-order propagator, not a
   solve with a source; abusing `n=1, Δt→∞` gives relative error `1/(Δtλ_min)` and
   `κ(S) → ∞`. `LaplaceChernoffResolvent` requires `λ > 0` (`resolvent.rs:154`) and
   its Gauss–Laguerre quadrature degrades as `λ → 0`. `ResolventJumpChernoff`
   (`resolvent_jump.rs:74`) uses an O(N) complex Thomas solve internally, but only
   for the unit 3-point Laplacian and only to return `e^{tA}g`.
   `steady_state_dirichlet_1d` (`conservative.rs:372`) is an O(N) Thomas solve but
   has no source, no reaction, Dirichlet only. The O(N) primitives (`thomas_solve`
   `conservative.rs:247`, `pcg_shifted` `pcg.rs:145`, `Jacobi`) are `pub(crate)`.
   **No existing path gives an O(N) steady or resolvent solve with a source.**
2. *`phi_action` has a cost defect.* `aug_norm_bound = τ‖A‖ + ‖v‖_∞ + 1`
   (`phi_action.rs:51`) feeds `select_s_m`, so the substep count grows with the
   magnitude of the input vector. φ_k is linear in `v`; Al-Mohy–Higham scale the
   coupling block by a power of two `η` so that it never drives `s`. Measured
   (N = 161, `phi_action(op, 1, τ, c·v)`): `τ = 10⁻⁶`: 0.02 ms at `c = 10⁻³`, 20 ms
   at `c = 10³`; `τ = 0.5`: 1.36 ms vs 17 ms. This is the "~50 ms fixed cost" the
   case study reported, and about 40% of its transient wall time.
3. *The transient error is not stiffness.* The explicit part has Lipschitz
   constant ≈ 0.34 s⁻¹ (`L·Δt ≈ 0.17` at Δt = 0.5). The error is the plain
   second-order constant of ETD-RK2, made large because the explicit term nearly
   cancels the linear part on smooth modes. BDF wins mainly by order and by
   growing its step after the response settles (~120 s of a 600 s window).

## TRIZ contradiction scan (gate, ADR template step)

- **C1 (steady).** The steady solve must come from the semigroup (the library is
  matvec-only and explicit). It must also be a direct O(N) solve. *Resolution in
  the super-system*: the resolvent `R(λ) = ∫₀^∞ e^{−λt}S(t)dt` IS the semigroup's
  Laplace transform (Hille–Yosida). `λ = 0` is the steady state of a dissipative
  generator, and `R(0)s = lim τφ₁(−τA)s` is exactly what the study computed. The
  operator supplies its own inverse through a structural resource it already has:
  tridiagonality (LDLᵀ, O(N), exact) or SPD sparsity (PCG). IFR: the operator
  inverts itself at the cost of one sweep. Not a compromise. The polynomial path
  stays for evolution, the rational path serves solves, and dispatch is by
  structure.
- **C2 (φ cost).** `v` must sit inside the augmented matrix (one sweep for all φ_k),
  and `v` must not affect that matrix's norm. *Resolution by parameter*: scale the
  coupling block by `η = 2^{−⌈log₂ M⌉}`, `M = Σ_k τ^k‖w_k‖_∞` (τ-weighted, summed
  over columns; `M = ‖v‖_∞` for `phi_action`), and the companion slot of the
  initial vector by `1/η`, so `‖ηW_τ‖_∞ ≤ 1` for every `τ` and column count
  (§62.4). Powers of two are exact in IEEE arithmetic, so `φ(2^j v) = 2^j φ(v)`
  bit for bit.
- **C3 (transient).** The coupling term must stay explicit (it is problem-specific
  and nonlinear, and Python cannot call back per step: ADR-0189 D3). It must also
  be implicit (it cancels the linear part). *Resolution in structure*: split it into
  a linear part that the user moves into the operator (`with_diagonal` for local
  terms, `GeneralOperator` for nonlocal ones, both φ-capable after this ADR) and a
  weak explicit remainder. Supply `φ_combination`, so any ETD/EPIRK/exponential
  Rosenbrock stage is one library call from a Python-driven loop, without callbacks.
  Adaptive stepping costs nothing extra: the ETD-RK2 correction `Δt φ₂(N(a)−N(u))`
  is already an embedded ETD1–ETD2 error estimate.

## Decision

**D1 — `SpdResolvent<F>`: factor once, solve many** (new `spd_resolvent.rs`).
`SymmetricOperator::resolvent(λ, mass, solver, tol)` builds a solver for
`(λM + A)x = b` with `λ ≥ 0` and `M = diag(mass) > 0` (default `I`). `solve_into`
then solves for each right-hand side. Dispatch by structure:
`SpdSolver::Auto` → **symmetric tridiagonal LDLᵀ** when every row couples only
`i−1, i, i+1` (exact, factor `O(N)`, solve `O(N)`, positive pivots certify SPD);
otherwise **PCG** with IC(0) (fallback to Jacobi on breakdown, reported in
`method()`, as §59.2 prescribes) or Jacobi. Deterministic sequential arithmetic
(`+ − × ÷`, `sqrt` via the ADR-0200 `libm` route), `no_std + alloc`, bit-identical
std/no_std. `λ = 0` with a singular `A` returns `DomainViolation`: certified by
the LDLᵀ pivot test (relative threshold `n·ε·max|aᵢᵢ|`), or flagged a priori when
all row sums vanish (the pure-Neumann null space). Non-convergence returns
`ConvergenceFailed` (first production use of that variant). Rejected: a
general sparse direct Cholesky (fill-in, a dependency, or ≥ 500 LoC of symbolic
factorisation; the ROADMAP non-goal); extending `LaplaceChernoffResolvent` to `λ = 0`
(the quadrature diverges); a `path="steady"` on `evolve_batched` (a solve is not an
evolution and has no `t`).

**D2 — operator composition** (on `SymmetricOperator`, additive).
`with_diagonal(c)` returns `A + diag(c)` (`c ≥ 0`, finite; inserts missing diagonal
entries, keeps sorted columns), which covers reaction, sink, Newton cooling,
Robin boundary rows, the cable equation and linearised kinetics. `csr()` returns
the borrowed `(row_ptr, col_idx, vals)` with zero copies. Python gains
`to_csr()`, `with_diagonal()`, and `lumped_congruence()` (already in Rust).
Rejected: new parameters on `assemble_conservative_csr_1d` (that would break the
signature, and composition is more general).

**D3 — φ-combination and generators.** `phi_combination(op, τ, [w₀..w_p], out)`
computes `Σ_k τ^k φ_k(τG) w_k` (`p ≤ PHI_MAX = 3`) with ONE augmented Horner sweep
(Al-Mohy–Higham 2011, Thm 2.1). This covers the affine evolve
`u' = Gu + s` (G1: `[u₀, s]`), every ETD/Krogstad/Cox–Matthews stage, and
exponential Rosenbrock. New adapter `CsrGenerator<F>`
(`from_symmetric(op, mass)`, `from_general(op, mass)`) provides `G = −M⁻¹A`
with an exact row-wise Gershgorin bound `maxᵢ Σⱼ|aᵢⱼ|/mᵢ` and an exact transpose
`−AᵀM⁻¹`. This gives φ-functions with a diagonal mass (G4) and φ-functions on the
non-symmetric `GeneralOperator` (G9), and therefore user-assembled linear coupling
in the operator (C3).

**D4 — φ cost fix (§58.2 amended).** `phi_action`, `phi_action_batched` and
`phi_combination` select `(s, m)` from `τ‖G‖ + 2` (`PHI_NORM_TIGHTEN` kept) with
the power-of-two scaling `η` applied to the coupling block. The augmented matrix
is `B̃_η = [[τG, ηW_τ],[0, J]]` (column `j` of `W_τ` is `τ^{p−j}w_{p−j}`, `J`
unscaled) and `η` comes from the τ-weighted column sum, which is what makes
`‖B̃_η‖_∞ ≤ τ‖G‖ + 2` hold (§62.4). New probe
`phi_cost_probe(norm_g, τ) -> (s, m)`. Existing gate thresholds (`G_PHI_AUG_DENSE`,
`G_ETDRK4_ORDER`, `G_ETD_ADJOINT_FD`) are unchanged and must stay green. Values change only through the
`(s, m)` choice, at the truncation-error level (≤ 1e-14 on the existing oracles).

**D5 — Python only** (`semiflow-py`). Exposed: `SymmetricOperator.resolvent(...)` →
`SpdResolvent` (`solve`, `solve_batched`, `solve_info`, `method`, `n`);
`SymmetricOperator.to_csr/with_diagonal/lumped_congruence`; `phi_combination`;
a `mass=` keyword on `phi_action`, `phi_action_batched` and `phi_combination`;
and acceptance of `GeneralOperator` in all three. GIL: ADR-0031 three-phase. FFI and
WASM are DEFERRED under the ADR-0186/0195 asymmetry: neither binding has a
`SymmetricOperator` or `GeneralOperator` surface to extend. Adding them is a
separate binding-parity ADR, triggered by a C or JS user request.

**D6 — ROADMAP wording.** "Fully-implicit schemes — Chernoff approach is explicit
by design" is narrowed to "fully-implicit *time-stepping* schemes (BDF, implicit RK,
Newton–Krylov on nonlinear systems)". The linear resolvent of a dissipative
generator is semigroup theory, and the library already ships three resolvent
kernels (§22, §47, §59). A steady solve is `R(0)`, not a time-stepping scheme.

## Consequences

- New file `spd_resolvent.rs` (≤ 500 LoC) with IC(0) in `pcg.rs` (`pcg.rs` stays ≤ 500).
  `phi_action.rs` adds `phi_combination` (move to `phi_combination.rs` if it would
  exceed 500). `generator_action.rs` adds `CsrGenerator`. No new dependency
  (3/3 cap untouched).
- The case-study steady loop drops from `O(N³)` to `O(N)` per linear solve, with the
  factor reused across outer iterations. The transient loses the `‖v‖`-driven φ cost
  and gains a high-order, adaptive, callback-free ETD path. Perf acceptance is
  measured in the case-study repo (not a library gate): steady at N = 641 within
  2× of the `splu` loop; transient within 1× of BDF at equal T_max accuracy.
- The constitution header still says MSRV 1.78. ADR-0201 moved it to 1.81, so the
  docs-writer should fix that drift. Nothing in this ADR needs a newer compiler.

## Honest limits

- PCG is `O(√κ·nnz)` per solve: fine for 2-D and 3-D Laplacians (IC(0) improves
  the constant, not the exponent). No multigrid, no direct sparse Cholesky. 1-D
  tridiagonal is the only exact-direct path. Block-tridiagonal and banded direct
  paths are deferred.
- φ-actions stay polynomial: cost `∝ τ‖G‖`. A rational (TWS contour) φ-action for
  tridiagonal SPD operators, reusing the `resolvent_jump.rs` contour with `s^{−k}`
  weights, would make φ cost independent of `τ`. It is DEFERRED. Trigger:
  a transient at N ≥ 641 where `τ‖G‖` dominates wall time.
- `λ < 0` (indefinite shifts, Helmholtz-type) is out of scope: SPD only.
- The 1×-BDF transient target may need case-study-side changes (adaptive Δt,
  linear coupling moved into the operator). The library supplies the means. It
  does not promise the number.

## Amendment 1 — gate calibration (2026-10-06, Gate-Change-Approved-By: ai-solutions-architect)

The `G_SPDR_*` thresholds were set before implementation. This amendment
calibrates a constant that had never been measured. It does not weaken a
measured gate. `G_SPDR_STEADY_MMS` had `err(513) ≤ 1e-5`, estimated from the
interior truncation alone (C = 1.565). The half-cell boundary row has an O(h)
truncation at x = 1, where `k'(1) ≠ 0`. That row adds an O(h²) Neumann flux
defect `k w'(1) = −π²/2` to the global error. The a-priori error equation
(`scripts/verify_spdr_mms.py`, Part 2) gives `C* = 3.506917`. The measurement is
`3.50692`, and an independent Thomas oracle reproduces the Rust errors to 6 digits.
The new gate is `err(513) ≤ 1.02·C*·h² = 1.365e-5`, plus a two-sided check
`|C(n)/C* − 1| ≤ 2 %`. The `#[ignore]` is gone. Two deviations are approved:
`G_SPDR_PCG_DENSE`/`G_SPDR_REJECT` use 4×4 because `GridND` needs n ≥ 4 per
axis, and `G_SPDR_IMPLICIT_CROSS` raises the reference's CG cap (`Some(20n)`)
with its threshold of 1e-10 kept. Derivations are in properties.yaml.

## Amendment 2 — PCG result semantics (2026-10-06)

On the PCG path, `Ok` means the recursive CG residual met `tol`. The true residual
is recomputed on both paths and reported in `SolveReport`. Near the `ε·κ(S)`
attainable-accuracy floor it may exceed `tol`, and that is not an error. The shared
CG loop returns `Ok` when it stagnates, so stagnation could pass as success: on a
singular `lumped_congruence` at `λ = 0`, `Pcg(IC0)` returned `Ok` with a residual of
1.43. A gross-residual guard turns any true residual that is NaN or
`> max(10³·tol, 10⁻³)` into `ConvergenceFailed`. That threshold is approved over
two alternatives. `10·tol` falsely rejects legitimately converged solves at tight
`tol`, where the floor is about `ε·κ`. A bound relative to an `ε·κ` estimate needs
a κ estimator, which adds code for a detector that only has to separate an `O(1)`
failure from a floor of at most `10⁻⁶`. That separation holds only for
`κ(S) ≲ 10¹⁰`. In the band between the guard and the `8ε` null-space check
(`κ ≈ 10¹³–10¹⁴` measured), PCG returns `ConvergenceFailed` while LDLᵀ returns
`Ok` at a comparable residual. This is accepted and documented, not fixed: the
only cheap κ-aware floor, `ε‖S‖‖x‖/‖b‖`, is inflated by null-mode growth during
singular stagnation, which is the very case the guard exists to catch.
`ConvergenceFailed` reports the configured cap and the relative true residual. `solve_into` prescales `b` by an exact
power of two, which makes the solve bitwise scale-equivariant. The null-space
rejection is now relative to rounding: every `|row sum| ≤ 8ε·Σ|aᵢⱼ|`. `n = 0` is
rejected. The normative text is in §62.1 and §62.2.b/c of
math-spd-resolvent.md, in spd-resolvent-api.md and in errors.yaml.

## Amendment 3 — PR #43 review fixes (2026-10-06)

Contract changes after review:
- `with_diagonal` returns `DomainViolation` when an assembled entry of `A + diag(c)` overflows.
- With a mass, `CsrGenerator` stores the row-normalised `Q = M⁻¹A` once and applies `−Qv`.
- Its norm bound `maxᵢ Σⱼ(|aᵢⱼ|/mᵢ)` is computed divide-then-sum, and a non-finite quotient or bound is a `DomainViolation`, in both constructors. `mass = None` is unchanged.
- In the `solve_into` power-of-two prescale, an exponent whose factor is not representable is clamped toward 0 until both factors are finite and nonzero, replacing the `(1,1)` fallback. A subnormal RHS is now solved correctly.
- In Python, `resolvent(solver="auto")` tries the tridiagonal path and on `Unsupported` rebuilds as `Pcg{precond, max_iter}`, so those keywords are honoured. The core `SpdSolver::Auto` is unchanged.
- The test-side gate helpers (`sup`, `sup_diff`, `rel_sup_err`) reject non-finite input, so a NaN can no longer pass a `≤` threshold vacuously.

No gate threshold changes.
