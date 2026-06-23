# ADR-0177 — Reverse-AD multi-parameter (K>1) via piecewise-region coefficients

**Status:** ACCEPTED (2026-06-23) · **Amends:** ADR-0172 (lifts the K=1-only
boundary) · **Extends:** ADR-0156 Amendment 2 (§51.9 genuine cotangent sweep)
· **Issue:** #1 (reverse-AD K>1)

## Context

`ReverseChernoff::value_and_grad` rejects `theta.len()!=1` fail-loud
(`reverse_ad.rs:299-306`, `SemiflowError::UnsupportedOperation`). ADR-0172
established **why** the rejection is correct, not a bug: the constant-a
`DiffusionChernoff` kernel carries a single scalar coefficient `a(x) ≡ θ`, so a
K-vector θ has no per-component basis — the backward sweep recomputes the
*identical* `b_k` for every slot `p` and broadcasts one gradient into all K
slots ("the degenerate broadcast is the correct value of an ill-posed
question"). We now want the genuine K-vector reverse gradient — K **distinct**
∂J/∂θ_r in ONE backward pass, with the O(1)-in-K advantage over forward dual-AD
(`G_REVERSE_AD_ADVANTAGE`) — to enable gradient-based calibration of
**spatially-varying** diffusion coefficients. ADR-0172 named two correct paths:
(a) K piecewise-region coefficients `a(x)=θ_r on Ω_r` with per-region dual
seeding; (b) K independent loss targets. This ADR chooses (a).

## TRIZ resolution (АП→ТП→ФП→ИКР)

**ТП.** ТП-1 (a is one scalar θ): Jᵀ=F self-adjoint and exact, K=1
byte-identical — but all K seeds coincide → degenerate. ТП-2 (a is K
independent scalars): K gradients distinct — but K different `a_r` break the
single self-adjoint stencil and the `ChernoffFunction` contract. **ФP.** `a(x)`
must be **one function of one kernel** (Jᵀ=F, contract, K=1 preserved) AND
**K distinct numbers** (non-degenerate seeds). **Resource (already in the
topology):** the spatial grid Ω is *already partitioned into nodes*; the region
indicator `𝟙_{Ω_r}` is pure geometry — no new field. **Resolution in space:**
keep one function `a(x) = Σ_r θ_r · 𝟙_{Ω_r}(x)` that takes **K distinct values
on K disjoint regions**. "One" lives at the kernel/contract level; "K distinct"
lives at the values on disjoint Ω_r. The contradiction is dissolved — both
properties hold fully, not halved. **ИКР:** *K>1 becomes well-posed because each
θ_r owns a disjoint region Ω_r, so the per-region dual seed is non-degenerate
and the reverse sweep accumulates K genuinely-distinct gradients in ONE backward
pass* — adding no new kernel, preserving Jᵀ=F and the byte-identical K=1 path.
Path (b) is rejected: K cotangent targets on one θ do not calibrate a(x) (the
stated motivation) and add an external target loop (a new subsystem → lower
ideality); path (a) spends zero new subsystems.

## Decision

1. **Region map, NOT a new kernel.** Introduce a region partition
   `Ω = ⊔_{r=0}^{K-1} Ω_r` aligned to grid **DoF** (a `region_of(node_i) → r`
   map). The diffusion coefficient is `a(x_i) = θ_{region_of(i)}`, built as ONE
   `DiffusionChernoff` via the existing `with_closure` constructor (closure is
   `Send+Sync`, ADR-0034) — the `ChernoffFunction` trait signature is
   **untouched** (CRITICAL: 56 dependents). Const-per-region ⇒ inside each Ω_r
   the stencil is the same self-adjoint const-a stencil ⇒ `Jᵀ = F` stays exact.

2. **Per-region dual seeding.** For parameter r, the dual coefficient closure
   returns `Dual::variable(θ_r)` on nodes `i ∈ Ω_r` and `Dual::constant(θ_{region_of(i)})`
   elsewhere. Hence `b_k^{(r)} = ∂F/∂θ_r` has support ⊆ Ω_r. The backward sweep
   accumulates `grad[r] += ⟨λ_k, b_k^{(r)}⟩` for `r ∈ 0..K`, reusing the SAME
   checkpointed `u_{k-1}` — no extra forward trajectory (the O(1)-in-K resource).

3. **Lift the K>1 guard.** `value_and_grad` accepts `theta.len() == K` and
   returns `grad ∈ ℝ^K`. The `theta.len()!=1` rejection in `reverse_ad.rs` is
   replaced by `theta.len() == region_count` validation.

## The genuine-reverse argument (NOT the degenerate broadcast)

The per-region seeds are **structurally distinct**: `supp(b_k^{(r)}) ⊆ Ω_r` and
the Ω_r are disjoint, so `⟨λ_k, b_k^{(r)}⟩` reads distinct DoF for distinct r —
the v9.0.0 broadcast (identical b_k for all p) cannot occur. K=1 is the special
case (one region = all of Ω): seed `Dual::variable(θ)` everywhere, one
`b_k^{(0)}`, identical to today's path → **byte-identical regression**.
`Jᵀ = F` is unchanged (it is the *state* Jacobian, independent of how θ is
partitioned); transport stays in `b_k` per §51.9, never in `Jᵀλ`.

## Consequences / limits (narrow scope)

- **In scope:** self-adjoint **const-per-region** `a` (one constant value per
  Ω_r). Region boundaries are **DoF-aligned** (a node belongs to exactly one
  Ω_r) — the septic-Hermite stencil width is captured *inside* `b_k` by dual
  arithmetic and routed to the owning `grad[r]`, because the seed is bound to
  the **node value** θ_r, not a continuous sample position.
- **Out of scope (fail-loud):** variable-a *within* a region, non-self-adjoint
  kernels, non-DoF-aligned region boundaries — `SemiflowError::UnsupportedOperation`
  (the §51.5 narrow-linear/self-adjoint boundary is unchanged otherwise).
- **Additive:** new module `src/reverse_region.rs`; K=1 path in `reverse_ad.rs`/
  `reverse_sweep.rs` stays byte-identical. Zero new runtime deps.
