# ADR-0156 — Shift B: Reverse-mode-differentiable Chernoff layer (checkpointed adjoint)

**Status:** ACCEPTED (v9.0.0 SHIPPED 2026-06-10) · **Date:** 2026-06-08 · **Branch:** `feat/v9.0.0-planning`
**Theme:** v9.0.0 — Reverse-mode AD (Shift B, SHIP-TRACK, low-risk, R2 = no-solver ⇒ exact transpose)
**Gates:** `G_REVERSE_AD_GRADIENT` (RELEASE_BLOCKING) + `G_REVERSE_AD_CHECKPOINT` (memory sub-check) + `G_REVERSE_AD_STRUCTURE` (Amdt 2, RELEASE_BLOCKING) + `G_REVERSE_AD_ADVANTAGE` (Amdt 1, RELEASE_BLOCKING)
**Amendments:** 1 (genuine reverse wiring + K>1) · 2 (corrected self-defeating 0-ULP gate → tight `<1e-12` + structural conformance)
**Math:** `contracts/semiflow-core.math.md` §51 · **Oracle:** `T_REVERSE_TRANSPOSE`
**Parent:** ADR-0154 · **Source:** research §3

## Context

The v8.0.0 differentiability axis shipped **forward-mode** dual-AD (§46, `Dual<F>`, zero-alloc, `G_DUAL_AD_GRADIENT` PASS) and the supporting transpose machinery: §42 transpose-exactness (`T_MAGNUS_TRANSPOSE`) and §43 adjoint-state parameter sensitivity. **Reverse-mode** is the missing complement — and it is *cheaper* here than anywhere else because of resource **R2**: `F(τ)` has **no implicit solver**, so the adjoint of the trajectory `(F(τ))ⁿ` is simply the **transposed product of the same shift-and-scale steps**, algebraically exact (no implicit-function-theorem backward solve, unlike Neural-ODE / Diffrax continuous adjoints, which are either approximate or store the whole trajectory). Forward-mode is `O(K)` for `K` parameters; reverse-mode gives the full gradient in `O(1)` passes — the right tool for many-parameter calibration / ML-training-through-a-PDE-layer.

## Decision

Implement **reverse-mode AD over the whole Chernoff trajectory via binomial checkpointing**, NOT a tape. **TRIZ resolution (research §3.3).** **ТП**: an *exact* adjoint normally requires *storing the whole forward trajectory* (`O(n·state)` memory) OR *re-solving an implicit backward equation* (optimize-then-discretize, which is *approximate*). We want *exact AND cheap-memory*. **ФП**: the trajectory must be *stored* (for exactness) AND *not stored* (for memory). **ИКР**: the trajectory *recomputes itself* — because the forward step is a cheap deterministic solver-free shift-and-scale, recomputation (checkpointing) is nearly free, and the transpose is algebraically exact (§42). **ВПР**: the *absence of a solver* (R2) is precisely what makes the backward pass both exact and cheaply recomputable. A classic Wengert-list tape would need dynamic allocation incompatible with the suckless posture; **binomial checkpointing stores every `k`-th state and recomputes segments backward** — `alloc`-only (a `Vec` of checkpoints), `no_std`-clean, **no external autodiff crate** (the project already hand-rolls `Dual<F>`; this respects Override #1's ≤3-direct-dep budget). Integration surface is **PyO3 / WASM `value_and_grad`** (reusing the 0-ULP binding-parity culture of `G_BINDING_GREEKS_PARITY`); a PyTorch/JAX custom-op is downstream glue, kept OUT of the `no_std` core.

## Consequences

Additive surface extending `dual.rs` / `adjoint.rs` / `magnus_graph_adjoint.rs`; no kernel semantics change; zero new dependency. **Strongest failure mode (adversarial, research §3.5):** the **ecosystem is the moat-breaker, not the math** — PyTorch/JAX/Diffrax own differentiable-solver mindshare, so a Rust layer that cannot be *trivially* dropped into a Python training loop sees no mainstream-SciML adoption regardless of being exact and memory-cheap. The honest niche is **embedded / edge / HFT differentiable control** (Rust, `no_std`, deterministic, bit-reproducible) where Python autodiff cannot go. **Secondary (R-B3):** §42 proves transpose-exactness only for the degree-4 truncated-Magnus / linear family — reverse-mode over *variable-coefficient / nonlinear* kernels may lose the clean transpose, so the exactness guarantee MUST be re-established per-kernel or honestly scoped to the linear/Magnus family (carried into §51 NARROW scope). Gates: `G_REVERSE_AD_GRADIENT` (reverse-mode gradient matches central-difference reference to <1e-9 relative, AND matches the forward-mode `Dual<F>` gradient (§46) to **0 ULP** for `K=1` — cross-mode parity); `G_REVERSE_AD_CHECKPOINT` (peak memory `O(√n)` not `O(n)`, by allocation counting, reusing the `G_DUAL_ZERO_ALLOC` harness); `T_REVERSE_TRANSPOSE` (extends `T_MAGNUS_TRANSPOSE` to the full trajectory transpose). Risk is LOW: ~80% of the machinery (§42/43/46) already ships. All gates PLANNED.

## v9.0.0 Implementation Outcome (2026-06-10)

All gates PASS. `reverse_ad.rs` (483 LoC, commits 3a4abaa + 778b854). Measured results:

| Gate | Threshold | Measured | Result |
|------|-----------|----------|--------|
| `G_REVERSE_AD_GRADIENT` (FD agreement) | rel error < 1e-9 | 8.09e-12 | PASS |
| `G_REVERSE_AD_GRADIENT` (cross-mode 0-ULP, K=1) | 0 ULP vs §46 `Dual<F>` | 0 ULP | PASS |
| `G_REVERSE_AD_CHECKPOINT` (peak memory scaling) | slope ≤ 0.6 | slope 0.39 | PASS |
| `G_BINDING_REVERSE_AD_PARITY` (PyO3 + WASM) | 0 ULP vs Rust | 0 ULP | PASS |

NARROW scope enforcement is in place: `ReverseChernoff<C, F>` is constructable only for `C: LinearChernoffFamily` (§51.4 marker trait); transpose-exactness is not claimed for variable-coefficient or nonlinear kernels.

---

## Amendment 1 — v9.1.0: make it GENUINELY reverse-mode (2026-06-10)

**Status:** ACCEPTED (v9.1.0 PLAN) · **Branch:** `feat/v9.1.0-genuine-scurve` · **Math:** §51 STATUS + §51.9 (amended)
**Trigger:** two independent adversarial audits (`.dev-docs/reports/v9-third-scurve-audit-reviewer.md`, `v9-math-fidelity-audit.md`) found the v9.0.0 "reverse-mode AD" headline ships as **forward-mode `Dual<F>` AD relabelled**.

### Honest correction
The v9.0.0 public gradient API `value_and_grad_k1` computed the gradient via the forward-mode `Dual<F>` pass (§46), not a reverse/adjoint backward sweep (`reverse_ad.rs:290–313`, confirmed by its own docstring). The genuine reverse machinery (`recompute_segment`, `step_jacobian_col`, the cotangent recursion `λ_{k-1}=Fᵀλ_k`) shipped `#[allow(dead_code)]` and was never wired. Consequences: (a) "reverse-mode" was a relabel of v8 forward dual-AD; (b) the K=1 "0-ULP cross-mode parity" gate was tautological (`reverse_grad == forward_dual` — same computation); (c) **no K>1 gradient shipped** — the actual capability tier over §46 was absent. The gradient *value* was correct (FD-validated 8.09e-12); the *method* claim was not honored. This is the audit's finding D1/D2/D3 (severity HIGH on novelty, LOW on correctness).

### Decision (v9.1.0)
Wire the genuine cotangent backward sweep as the PUBLIC gradient path and add a **K>1** multi-parameter gradient computed in a SINGLE backward pass — the real tier over forward dual-AD's `O(K)` passes (§51.9 NORMATIVE). The contradiction "exact adjoint AND O(√n) memory AND genuinely reverse" was already resolved by binomial checkpointing in the original ADR (R2 = no solver ⇒ cheap replay + exact transpose); v9.0.0 built the scaffold but never ran the backward sweep on it. v9.1.0 finishes the wiring — no new mathematics, an unfinished resolution completed. Forward-dual is retained ONLY as the §51.4 parity reference (now a real cross-mode check between two distinct computations, with a NORMATIVE anti-tautology clause).

### Gate changes
- `T_REVERSE_TRANSPOSE` corrected to **3/3**: NEW sub-check (3) proves the non-vacuous adjoint/VJP identity `⟨v,Ju⟩ = ⟨Jᵀv,u⟩` on explicit symbolic `J, Jᵀ` plus `Jᵀ == transpose(J)` entrywise (structurally-independent oracle, §2.3 of the design spec).
- `G_REVERSE_AD_GRADIENT`: leg (ii) now compares the GENUINE reverse sweep against forward-dual to 0 ULP (real cross-mode parity, not `x==x`).
- **`G_REVERSE_AD_ADVANTAGE`** (NEW, RELEASE_BLOCKING for the capability claim): asserts reverse is O(1)-in-K vs forward-dual O(K) (ratio(64)/ratio(1) ≥ 8). This gate FAILS on the v9.0.0 forward-dual relabel — it is what makes the new tier non-vacuous.

### Consequences
Additive within `reverse_ad.rs` (split to `reverse_ad_backward.rs` if it would exceed the 500-LoC default cap — additive sibling, no Cohort carve-out). Scope stays NARROW (linear/Magnus/constant-a self-adjoint; K params of a LINEAR kernel). Zero new deps. PyO3/WASM `value_and_grad(theta)` updated; `G_BINDING_REVERSE_AD_PARITY` stays 0-ULP. Phased plan: design spec §5 Phases 1–3.

---

## Amendment 2 — v9.1.0: the §51.4 gate was self-defeating; correct it (2026-06-10)

**Status:** ACCEPTED (v9.1.0) · **Branch:** `feat/v9.1.0-genuine-scurve` · **Math:** §51 STATUS + §51.4 + §51.6 + §51.9 (amended)
**Trigger:** Phase-2 engineering (`reverse_sweep.rs::backward_sweep_k1`, uncommitted WIP) reproduced the v9.0.0 defect — a forward-mode JVP relabelled "reverse" — and Anchor's verification traced the cause to the Amendment-1 gate itself.

### Honest correction (the gate forced the tautology it forbade)
Amendment 1's §51.4 required, for `K=1`, that the reverse-mode gradient match the forward-mode `Dual<F>` gradient to **0 ULP** "by the adjoint identity, not shared code." This is **mathematically impossible** for two genuinely independent floating-point computations: a reverse VJP (`g = Σ_k ⟨λ_k, b_k⟩`, backward `λ_{k-1}=Jᵀλ_k`) and a forward JVP (`g = ⟨w, t_n⟩`, forward `t_k=J·t_{k-1}+b_k`) are equal only in **exact** arithmetic — in IEEE-754 they use different reduction orders and agree to `O(n)·κ·ε ≈ 1e-13`, NOT 0 ULP. The ONLY way to make a "reverse" path match forward-dual to 0 ULP is to make it run the *same forward ops* — exactly the tautology the anti-tautology clause was written to prevent. The Phase-2 engineer, cornered, implemented a forward `k=1..n` tangent accumulator (`Dual.tangent` extraction, no `Jᵀ`, `step_jacobian_col` left dead) and documented it as "structurally independent" reverse-mode. **Root cause: the gate, not the engineer.**

### A second obstruction, resolved (transport terms)
The Phase-2 docstring argued the θ-dependent sample positions (`h₀=2√(θτ)`) produce transport terms that a plain `apply_f(τ,&t)` cannot capture, justifying the forward-dual dodge. This is TRUE but MISPLACED. `F_θ` is **linear in its state argument**, so the state Jacobian `∂F/∂u = J(θ)` is exactly the banded stencil (no transport); transport appears **only** in the parameter sensitivity `b_k = ∂F/∂θ` (differentiating the sample positions). `Jᵀλ` carries no transport; `b_k` carries all of it and is computed correctly by one §46 dual step with **zero state tangent** and the **θ-seed in the coefficient closure** (the repaired `step_jacobian_col`). The conflation of `∂F/∂u`-transport (none) with `∂F/∂θ`-transport (all) is what drove the dodge.

### Decision (gate corrections)
1. **§51.4 leg (ii)**: drop 0-ULP; reverse-vs-forward-`Dual<F>` now `< 1e-12` relative (justified `C·n·ε`, `n≤1024`), still 3 orders tighter than the FD leg (i) `< 1e-9`.
2. **0-ULP confined to same-algorithm cross-platform**: `G_BINDING_REVERSE_AD_PARITY` (Rust ≡ PyO3 ≡ WASM) STAYS 0-ULP — that is reproducibility of *one* computation, the legitimate home of bit-exactness.
3. **NEW `G_REVERSE_AD_STRUCTURE`** (RELEASE_BLOCKING, `test-fast`): mutation oracle — replacing `apply_transpose_step` with identity MUST change the gradient `> 1e-6` rel (proves `Jᵀ` is load-bearing); plus a backward-direction (`k=n→1`) witness. A forward-tangent implementation FAILS it.
4. **`G_REVERSE_AD_ADVANTAGE`** (Amendment 1) remains the K-scaling gate that makes the tier non-vacuous; the genuine-reverse requirement binds at K>1, and K=1 routes through the SAME backward machinery (no forward shortcut).
5. **`step_jacobian_col` repaired**: seed `∂/∂θ_p` in the coefficient closure (NOT in `tau_dual`), zero state tangent, extract `.tangent` ⇒ transport-complete `b_k^{(p)}`.

### Consequences
No new mathematics, no new deps, scope unchanged (NARROW constant-a / linear-Magnus). Phase-2 WIP (`reverse_sweep.rs` new; `reverse_ad.rs`/`g_reverse_ad.rs`/`lib.rs` modified) will be **reworked**: the forward-tangent `backward_sweep_k1` is REPLACED by the genuine `Jᵀ` cotangent sweep; `value_and_grad_k1` is subsumed by the K-vector `value_and_grad` (K=1 = length-1 θ). **VERDICT: genuine reverse-mode is ACHIEVABLE as re-specified** for the narrow scope — no irreducible obstruction; the transport concern is handled in `b_k`, and `Jᵀ=J` for the symmetric constant-a stencil.
