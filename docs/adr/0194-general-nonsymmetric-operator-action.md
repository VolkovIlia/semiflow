# ADR-0194 — Non-symmetric operator action via the existing Taylor `expmv`

- **Status**: Proposed (Issue #24; branch `fix/issue-campaign-17-26`)
- **Date**: 2026-08-13
- **Supersedes / amends**: none — purely ADDITIVE. `SymmetricOperator`,
  `GraphKrylovChernoff` and `expmv.rs` are untouched. Answers the question
  ADR-0185 left open ("non-symmetric / drift / directed graphs need Arnoldi…
  deferred") and math.md §55.6 ("requires Arnoldi/GMRES-type methods, explicitly
  OUT OF SCOPE") — with a different answer than either anticipated.
- **Contract**: gates `G_GENOP_DENSE`, `G_GENOP_NONNORMAL`,
  `G_GENOP_ASYM_ACCEPTED`, `G_GENOP_COST_LINEAR` (advisory).

## Context

`SymmetricOperator::from_csr` validates symmetry, so the whole Krylov surface —
`evolve_batched` with `chebyshev`/`lanczos`, `phi_action`, `Etdrk4`,
`mass_lumped_evolve` — is closed to non-self-adjoint generators. Two concrete
operators are shut out: **drifted Fokker–Planck**
`∂_t p = ∂_x(D(x)∂_x p) − ∂_x(μ(x)p)`, where the drift breaks symmetry, and
**inventory-ladder** generators (Cartea–Jaimungal), whose Hopf–Cole system has
upper-bidiagonal `A` — structurally non-symmetric, tiny (`N ~ 100`), and an
obvious `expmv` workload.

## Decision

**1. `GeneralOperator::from_csr` routed to scaled truncated Taylor — not Arnoldi.**
The issue offered either. Taylor wins on three counts. The engine is *already
present and already symmetry-agnostic*: `expmv.rs` implements Al-Mohy & Higham's
matvec-only method, `phi_action` reuses it, and math.md §58.2 already records
that it needs "only a matvec and a real spectral interval — NOT symmetry". It is
also the **safer** engine for the two named targets: the inventory ladder is
nilpotent-plus-diagonal, i.e. maximally non-normal, exactly where Arnoldi's
residual-based stopping is unreliable and unrestarted Arnoldi can stagnate,
whereas the backward-error bound holds for any `A`. And it is ~250 lines against
Arnoldi's full MGS + Hessenberg + happy-breakdown + restart policy, with no new
numerical claim to gate. Arnoldi stays deferred, now with a reason rather than an
absence.

**2. `norm_inf_bound()`, not `lambda_max_bound()`.** For a non-symmetric `A` the
Gershgorin row sum is an induced *norm* bound, not a spectral interval. The name
is the API's way of not lying; over-estimation only raises the scaling `s`.

**3. No `path=` argument.** Chebyshev's Bessel coefficients require a real
spectrum in `[0, λ_max]` and are not fixable; Lanczos is structurally
symmetric-only (3-term recurrence without reorthogonalisation, projecting onto a
*force-symmetrised* tridiagonal). `GeneralOperator` exposes no `krylov()`, so
reaching those paths is a compile error in Rust and an `AttributeError` in
Python — a class boundary, not a tolerance failure.

**4. A real transpose.** `apply_transpose_into_slice` uses a transposed CSR built
once at construction. The `GeneratorAction` trait's `apply_generator_transpose`
defaults to `apply_generator` — correct for self-adjoint operators and silently
wrong here — so relying on that default was not an option.

**5. This kernel selects `(s, m)` itself instead of calling `expmv::select_s_m`.**
See the finding below. The criterion used is derived rather than tabulated: the
degree-`m` truncation term at per-substep argument `z` is `z^{m+1}/(m+1)!`, so
requiring it at unit roundoff gives `z ≤ (u·(m+1)!)^{1/(m+1)}`, which at `m = 18`
is `1.1894`. Fixing `m = 18` is optimal within the available degrees: cost per
unit argument is `m / z_max(m)` = 2075 (m=4), 106 (m=8), 28 (m=13), **15
(m=18)**.

## Finding — `expmv::select_s_m`'s θ table is optimistic when fed a tight norm

Not fixed here, and worth stating plainly because it is not confined to this ADR.

Feeding the *tight* `‖A‖_∞` for the drifted-Fokker–Planck datum
(`n = 10`, `t = 0.3`, `‖A‖_∞ = 23.4`, so `arg = 7.02`), `select_s_m` returns
`(s, m) = (1, 18)`. The measured truncation error of that choice is **1.635e−4**
— not double precision. Verified three ways: the kernel's output at `(1, 18)`,
an independent numpy replication of the same `(s, m)`, and an independent
matrix-level scaling-and-squaring reference that agrees with the naive series to
machine precision. Re-running at `(2, 18)` gives 2.6e−11 and at `(6, 18)`
4.4e−16, so the algorithm is fine and the *selection* is what is loose.

The existing callers do not see this because they pass deliberately loose norm
bounds — `DiffusionExpmvChernoff` uses `4·a_norm_bound/dx²`, an over-estimate
that inflates `s` enough to compensate — so `G_PHI_AUG_DENSE` and the ADR-0121
gates pass on merit. `GeneralOperator` passes a tight bound, which removes the
compensating slack and exposes the table.

Retuning `THETA_M` would change the step counts, results and runtimes of two
shipped, gated kernels (`DiffusionExpmvChernoff`, `phi_action`) for a reason
unrelated to issue #24. That is an architect decision with its own gate re-run,
not a side effect of opening a new operator class. Recorded here, in the rustdoc
at `select_s_m_tight`, and left for a separate ADR.

## Consequences

- New `crates/semiflow/src/general_operator.rs` (`GeneralOperator`,
  `CsrExpmvChernoff`); `pub mod` appended to an existing `lib.rs` line.
- `CsrExpmvChernoff` implements `ChernoffFunction<F, S = GraphSignal<F>>`, so it
  composes with the existing batched/parallel layer unchanged.
- `order()` returns `u32::MAX` — the honest answer for a tolerance-driven kernel
  with no fixed consistency order. Callers that schedule PI gains from `order()`
  must not use it.
- PyO3: `GeneralOperator.from_csr / n / norm_inf_bound / evolve_batched /
  apply_transpose`. PyO3-only, inheriting the ADR-0186 asymmetry — no
  `SymmetricOperator` exists in FFI or WASM either.

## Honest limits

- **Not depth-flat.** Cost is `Θ(τ‖A‖_∞)` matvecs — *linear*, not flat.
  `G_GRAPH_EXPMV_DEPTH_FLAT` must **not** be extended to this path. The advisory
  gate `G_GENOP_COST_LINEAR` measures the ~2× per doubling so the anti-claim is
  falsifiable rather than merely asserted.
- **Backward error only.** For `A = VΛV⁻¹` with ill-conditioned `V` the forward
  error can exceed the backward radius by `κ(V)`; `κ(V)` is not estimated.
  `G_GENOP_NONNORMAL` measures one instance at a deliberately looser tolerance
  (1e−9 vs 1e−10) and does not generalise.
- **Large `τ‖A‖` is hump-dominated.** The per-substep criterion controls
  truncation but not cancellation growth across substeps; for a strongly
  indefinite or non-normal operator at large argument, forward accuracy degrades
  regardless of `s`. Seen while validating: at `arg = 70` even a converged
  reference loses all precision in f64.
- **Chebyshev and Lanczos remain unavailable.** A class boundary, not a
  tolerance.
- **No conservation claim.** Discrete mass conservation for a Fokker–Planck
  assembly holds only if the caller's `A` has exactly zero column sums;
  `GeneralOperator` checks nothing of the sort and the Taylor truncation
  perturbs it at the backward-error level anyway.
- **Real `F` only** — `SemiflowFloat` is real; complex `A` is out of scope.
