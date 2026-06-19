# ADR-0009 — CEV European call as v0.3.0 real-world benchmark

**Status**: Accepted
**Date**: 2026-04-30
**Authors**: ai-solutions-architect (post-research / post-knowledge-fetcher Stage 4)
**Supersedes**: none. Complements ADR-0008 (ζ-A) and ADR-0007 (boundary policies).

## Decision

Adopt the **Schroder (1989) closed-form CEV European call** as the single
real-world validation benchmark for v0.3.0, codified in
`contracts/tests/cev_european_call.yaml` and implemented in
`crates/semiflow-core/tests/cev_european_call.rs`. Use the **spot-normalized
σ₀ convention** matched against QuantLib's `analyticcevengine.cpp` and R's
`FER::CevPrice`: `δ² = σ₀² · S₀^(2-2β)`, giving reference price **14.2421**
(NOT 13.70 as cited by knowledge-fetcher — independent recomputation against
Larguinho-Dias-Braumann eq 11-12d showed the cited value was a different-
parameterization artefact). Compute the noncentral χ² CDF in-tree via the
**Poisson-weighted central χ² series** (Larguinho eq 3 / Schroder series form):
`F(w; v, λ) = Σ_j Poisson(j; λ/2) · F_central(w; v + 2j, 0)` with `j_max=100`
and an early-exit at term magnitude `< 1e-12`. Use `statrs::ChiSquared` (central
only) as a `[dev-dependencies]` entry — production dep count stays at 2
(`num-traits`, `libm`); G7 budget unaffected. Three acceptance gates
(G_real_world_1: sup-norm < 5e-2 over `S∈[50, 150]`; G_real_world_2: pointwise
ATM < 1e-2; G_real_world_3: log-log slope ≤ −0.95 across `n ∈ {64,128,256,512}`)
respect the global O(τ¹) variable-`a` ceiling proven in ADR-0008 Amendment 2.
Rejected alternatives: **Option II** (`strafe-distribution` v0.1.1) — too new
(April-2026 release), unverified license/MSRV, accuracy degrades at λ > 1e4
(safe for our λ ≈ 45 but reckless to depend on); **Option III** (Python +
QuantLib subprocess) — adds Python toolchain to CI, slow, brittle. Trade-off
of Option I: ~40 LoC of in-tree numerical helper plus a dev-dep on a vetted
crate, vs zero deps (impractical: writing central χ² CDF by hand would be
much longer and would duplicate `statrs`). The series convergence was
verified by the architect: at λ ≈ 45 the helper terminates in 49 iterations
to 1e-13 precision, well under the j_max=100 budget. Acceptance: all three
gates pass on `cargo test --release`; failure blocks v0.3.0 release.
