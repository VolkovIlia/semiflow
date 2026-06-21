# ADR-0172 — Reverse-AD K>1 semantics: K=1-only, fail-loud above

**Status:** ACCEPTED (2026-06-20) · **Supersedes the K>1 claim in:** ADR-0156 Amendment 2 (§51.9 "genuine K-vector cotangent backward sweep")

**Decision.** `ReverseChernoff` parameterises a **single scalar constant** diffusion
coefficient `a(x) ≡ θ` (`reverse_ad.rs:200-213` lifts `a(x)=Dual::variable(θ)`; gate
`with_closure(move |_| theta, …)` at `g_reverse_ad.rs:251`). A K-vector of θ has **no
well-defined meaning** for a one-scalar-coefficient kernel: there is no spatial region
structure, no per-component basis, and the dual kernel carries exactly one tangent seed,
so `backward_step` (`reverse_sweep.rs:54-61`) recomputes the *identical* `b_k` and dot
product for every `p` and broadcasts one gradient into all K slots — the observed
degenerate-broadcast bug is not a coding slip but the **correct** value of an ill-posed
question. A genuine K>1 reverse-mode gradient would require either (a) θ as K
piecewise-region coefficients `a(x)=θ_r on Ω_r` with per-region dual seeding, or (b) K
independent loss targets — neither of which is in the v9.x narrow linear/self-adjoint scope
(§51.5). Adding (a)/(b) is a research-track design, not an audit fix. Therefore we choose
**correctness over the broadcast illusion**: restrict the public surface to K=1 and reject
K>1 fail-loud at the boundary, returning `SemiflowError`. K=1 keeps the genuine cotangent
backward sweep (the §51.6 `G_REVERSE_AD_STRUCTURE` load-bearing transpose path is
unaffected). True multi-parameter reverse-AD is deferred to a future region-coefficient or
multi-target ADR with its own seeding model and gate. This honestly scopes the v9.1.0
"genuine reverse-mode" claim to the single-parameter case it actually computes.
