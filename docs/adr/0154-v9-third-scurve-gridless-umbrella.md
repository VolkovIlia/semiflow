# ADR-0154 — v9.0.0 umbrella: Gridless / Solver-Free Chernoff (third S-curve)

**Status:** ACCEPTED (v9.0.0 SHIPPED 2026-06-10 — Shift B + Shift C-via-TT shipped; Shift A GPU DEFERRED) · **Date:** 2026-06-08 · **Branch:** `feat/v9.0.0-planning`
**Theme:** v9.0.0 strategic umbrella · **Child ADRs:** 0155, 0156, 0157, 0158
**Source:** `.dev-docs/research/v9-paradigm-shifts-research.md` (graded C > B > A)
**Math:** `contracts/semiflow-core.math.md` §50 (Shift C), §51 (Shift B); Shift A = perf-only, no math section.

## Context

Two S-curves are saturated. The **kernel-accuracy** axis (ζ²→ζ⁸, manifolds, hypoelliptic, resolvent, complex, matrix, soft-killing; ADRs 0117–0131) closed at v7.0.0. The **product-structure** axis (`(F(τ))ⁿ` differentiated / jump-amortized / boundary-regularized / recursively deepened; ADRs 0132–0137) closed at v8.0.0 — F1 forward dual-AD (§46) and F2 resolvent-jump (§47) shipped, F5 eigenrotated WITHDRAWN. The two *unused structural resources* of Theorem-6 formula (6) `S(τ)f = ¼f(x+2√(aτ)) + ¼f(x−2√(aτ)) + ½f(x+2bτ) + τc·f` are now the next axis. **(R1)** the formula *is a Markov transition kernel* — already proven mass-conservative and tight on `M(ℝ^d)` (math §38, `T_ADJOINT_FP_TIGHTNESS` 6/6) — so iterating it on a *measure / particle ensemble* instead of a `Vec<F>` grid escapes the `O(N^d)` curse of dimensionality. **(R2)** the formula has *no linear solver* (pure local shifts + scalar multiplies; no FFT/Krylov/LU) — so its trajectory transpose is algebraically exact, giving a clean reverse-mode-differentiable layer, and its locality maps to data-parallel hardware. The research report ranks the three resulting directions C (gridless) > B (reverse-AD) > A (GPU spike).

## Decision

Open v9.0.0 on the **transition-law / solver-free axis** (the third S-curve), and ratify three child ADRs, each resolving a named TRIZ contradiction (АП→ТП→ФП→ИКР→решение), not a compromise. **ADR-0155 (Shift C, SHIP-NARROW)**: gridless high-dimensional Chernoff — iterate the §38 adjoint kernel on a `MeasureState<F,D>` particle ensemble, `O(d·P)` per step independent of grid `N`; SHIP-TRACK is the *bounded-d* (d≤~10) linear-coefficient deterministic-branching evolver, gated `G_GRIDLESS_DIM_SCALING` with a **go/no-go variance sub-check `G_GRIDLESS_VARIANCE`**. The go/no-go FIRED (measured 2026-06-09): d=2 validated, variance 1.417× < 2× gate, high-d intrinsic-limit (spatial-merge curse); Shift C ships NARROW. **ADR-0156 (Shift B, HEADLINE)**: reverse-mode AD over `(F(τ))ⁿ` via binomial checkpointing (no tape, no new dep, stays `no_std+alloc`), completing the v8 differentiability axis; gated `G_REVERSE_AD_GRADIENT` + `G_REVERSE_AD_CHECKPOINT`. Shift B is the **v9.0.0 headline** per the pre-registered §50.6 go/no-go consequence. **ADR-0157 (Shift A, SPIKE)**: feature-gated `remizov-gpu` `wgpu` translator crate, advisory gate `G_GPU_PARITY` (parity WAIVED, withdraw-on-dep-budget-breach). **ADR-0158 (research-track)**: path-space RQMC functional estimation — the TRIZ-reframed research direction to escape the spatial-merge curse. Adopt the research-track / ship-track / spike-only split (research §5.2): ship-track = C-narrow (d=2 validated) + B (headline); research-track = high-d path-space RQMC (ADR-0158), high-d (d>10) nonlinear/variable-coef, general 4ⁿ-reduction theory; spike-only = A.

## Consequences

All v9.0.0 additions are **additive** to the v8.0.0 public surface (no kernel semantics change). Shift A may introduce a **new crate `remizov-gpu`**, isolated behind `--features gpu` so the `no_std` core dependency budget (Override #1: ≤3 direct deps in `semiflow-core`) is **inviolate** — GPU deps live only in the optional crate. The **variance go/no-go fired as designed** (measured 2026-06-09): Shift C demoted to NARROW (d=2 validated, intrinsic spatial-merge limit at d≥3); **Shift B is the v9.0.0 headline** (research §5.3 Phase 2 activated). Override #2 (MCP WAIVED) is RE-AFFIRMED for the `semiflow-core` rlib and bindings; the `remizov-gpu` spike is a compile-time backend, not a runtime, so it does not change the MCP posture. Child gates: `G_GRIDLESS_DIM_SCALING` (MEASURED — d=2 PASS), `G_GRIDLESS_VARIANCE` (MEASURED — NO-GO), `G_REVERSE_AD_GRADIENT` (PLANNED), `G_REVERSE_AD_CHECKPOINT` (PLANNED), `G_GPU_PARITY` (PLANNED).

## v9.0.0 Shipped Status (2026-06-10)

**Shift B (HEADLINE, SHIPPED)** — `ReverseChernoff` (ADR-0156, §51): reverse-mode AD via binomial checkpointing. Commits 3a4abaa, 778b854. All gates PASS (G_REVERSE_AD_GRADIENT FD 8.09e-12, 0-ULP cross-mode, G_REVERSE_AD_CHECKPOINT slope 0.39, G_BINDING_REVERSE_AD_PARITY 0-ULP).

**Shift C (co-HEADLINE, RESOLVED, SHIPPED)** — `TtChernoff` (ADR-0159, §52): tensor-train carrier escapes the curse for the linear diagonal-A Gaussian class. Commit f7e0c16. `GridlessChernoff` (ADR-0155, §50) retained as the d=2 particle primitive and the documented negative result (G_GRIDLESS_VARIANCE NO-GO, spatial-merge INTRINSIC LIMIT). Both are normative.

**Shift A (DEFERRED)** — `remizov-gpu` (ADR-0157) not built this release. Advisory gate G_GPU_PARITY advisory-only; withdraw-on-dep-budget-breach posture unchanged.

**Child gate summary:**
- `G_GRIDLESS_DIM_SCALING` — d=2 PASS; d≥4 INTRINSIC LIMIT (pre-registered outcome)
- `G_GRIDLESS_VARIANCE` — NO-GO (1.417× MSE ratio at d=2; pre-registered outcome)
- `G_REVERSE_AD_GRADIENT` — PASS (8.09e-12 FD rel error; 0 ULP cross-mode K=1)
- `G_REVERSE_AD_CHECKPOINT` — PASS (slope 0.39, sub-O(n))
- `G_BINDING_REVERSE_AD_PARITY` — PASS (0 ULP PyO3 + WASM)
- `G_TT_CHERNOFF_DIMSCALING` — PRE-REGISTERED (`slow-tests` `--ignored`; prod-HW pending)
- `G_GPU_PARITY` — DEFERRED with Shift A

## Anti-directions (carried from research §5.4 — permanent rejection criteria)

- **(X3 — moat) NOT a general-purpose PDE framework.** Everything stays an iterate of `(F(τ))ⁿ`. The defensible asset is the math-fidelity moat (normative `math.md` + sympy gates + per-release audit), not a kernel/solver catalogue.
- **NOT a trained / data-driven solver.** SemiFlow *computes* the operator exactly and bit-reproducibly; it does not *learn* it (the deep-BSDE / FNO lane is a deliberate differentiator, not a target).
- **NO GPU stack in the `no_std` core.** Any GPU work is a separate, feature-gated, optional crate; the dependency-count and binary-size guardrails are inviolate (Override #1).
- **NOT an analog / optical Fourier-domain port.** That primitive is structurally incompatible with the solver-free shift kernel (it re-introduces the FFT that R2 avoids).
- **NO "golden-middle" compromise gates.** Each `G_*` is a sharp threshold (log-log slope / variance ratio / ULP), per the project's TRIZ-resolution-not-compromise culture.
