# ADR-0010 — v0.3.1 CEV hardening release (parameter sweep + boundary stress)

**Status**: Accepted
**Date**: 2026-04-30
**Authors**: ai-solutions-architect
**Supersedes**: none. Complements ADR-0008 (ζ-A) and ADR-0009 (CEV bench at single point).

## Context

v0.3.0 shipped the Schroder (1989) CEV European-call closed form as the first
real-world benchmark, validated at ONE canonical point (S₀=100, K=100, σ₀=0.20,
β=0.5, T=1, r=0.05) — see ADR-0009 and `contracts/tests/cev_european_call.yaml`.
That gate confirms the v0.3.0 ζ-A self-adjoint diffusion (ADR-0008) + RK2 drift +
Strang composition pipeline at one point in parameter space. The
τ²-correction polynomial in `crates/semiflow-core/src/diffusion.rs` is `a''`-aware
(ADR-0008 §Decision), but β=0.5 ⇒ a''(S) ≡ 0, so the variable-`a''` term has had
no empirical closed-form validation. v0.3.1 closes that gap: a parameter sweep
across moneyness × maturity × vol × elasticity (β ∈ {0.3, 0.5, 0.7}) plus a
boundary-truncation stress test. No API change, no production deps added.

## Decision

Ship two new tests + two new test-contract YAMLs + one worked example, all
under SemVer-patch envelope. The new artefacts are:

- `crates/semiflow-core/tests/cev_european_call_sweep.rs` (parameter grid;
  default ~25 combos always-on, full 135 combos behind Cargo feature
  `slow-tests`) governed by `contracts/tests/cev_european_call_sweep.yaml`
  with gates G_cev_sweep_A/B/C.
- `crates/semiflow-core/tests/cev_boundary_stress.rs` (S_max sweep across
  six values 150 → 400 with dx held ≈ 0.39) governed by
  `contracts/tests/cev_boundary_stress.yaml` with gates G_cev_boundary_A/B/C.
- `crates/semiflow-core/examples/cev_european_call.rs` — first option-pricing
  worked example for the crate (~70 LOC).
- This ADR-0010 (~80 LOC).

The single-point bench `cev_european_call.yaml` is preserved unchanged as
historical record. `properties.yaml` is NOT touched: these are TEST contracts
(per-instance benchmarks), not invariant properties (proptest gates).

## Rationale

- **Why feature-flag `slow-tests` instead of always-on full grid**: the 135-combo
  full set runs ~3 min; the default ~25-combo subset runs <30s. Daily `cargo test`
  must stay fast (suckless: build/run hostile to friction). Nightly CI and
  pre-release verification run with `--features slow-tests`. The full set is
  not a release-blocker; the default subset is.
- **Why we DO NOT add an adaptive-grid heuristic in v0.3.1**: variable spacing
  would change `Grid1D::dx` from a scalar to a function, breaking every existing
  call-site that assumes uniform `dx`. That violates the SemVer-patch envelope.
  Adaptive grid is deferred to v0.3.2 (minor) or v0.4.0 (Magnus integrator
  release).
- **Why we DO NOT chase the Magnus integrator now**: ADR-0008 Amendment 2 puts
  Magnus explicitly in the v0.4.0 milestone — it raises the global O(τ¹)
  variable-`a` ceiling to O(τ²) but requires a non-trivial spectral
  decomposition redesign. v0.3.1 hardens the existing pipeline; v0.4.0
  changes it.
- **Why σ₀=0.30 (not 0.20) in the boundary-stress test**: thicker right-tail
  at higher vol stresses LinearExtrapolate harder; a tightly-passing 0.20
  would mask the truncation sensitivity that the stress test is designed
  to expose. Documented in `cev_boundary_stress.yaml` §2 rationale_sigma0.

## Consequences

- **Positive**: first empirical `a''≠0` validation of ζ-A τ²-correction polynomial.
  Documented saturation profile for `S_max` choice. Two reusable benchmarks for
  future schemes (Magnus, adaptive grid). Worked example improves doc value
  (first option-pricing example in the crate).
- **Conditional**: if `G_cev_sweep_C` fails specifically at β∈{0.3, 0.7}: the
  triage is (a) real bug in `crates/semiflow-core/src/diffusion.rs`
  τ²-correction polynomial → ship as v0.3.1 patch; OR (b) edge-of-tolerance →
  document as known-limitation in CHANGELOG, defer correction to v0.4.0
  Magnus. Both paths are acceptable; the deferral path DOES NOT alter the
  v0.3.1 release scope or its other gates.
- **Negative**: ~3 min slow-test runtime under `--features slow-tests` adds CI
  cost on nightly only — acceptable (nightly already ~10 min). j_max bumped
  100 → 200 in the local `noncentral_chi2_cdf` helper; `panic!` on overshoot.

## Cross-references

- ADR-0008 — ζ-A self-adjoint variable-`a` (the τ²-correction polynomial under test).
- ADR-0009 — v0.3.0 CEV bench at single point (the foundation this hardens).
- `contracts/tests/cev_european_call_sweep.yaml` — parameter grid contract.
- `contracts/tests/cev_boundary_stress.yaml` — S_max stress contract.
- `contracts/tests/cev_european_call.yaml` — v0.3.0 single-point bench (unchanged).

## Amendment 1 — Empirical findings (post-implementation)

**Status**: Adopted, 2026-04-30 (rolled into the v0.3.1 release).

### Validated regime (empirical)

The sweep test confirmed the validity envelope of ζ-A on CEV oracle:

```
σ < 0.50  AND  σ·T·β < 0.40  AND  lam_peak < 1400
```

Inside this regime: all sweep gates (sup-norm, ATM, relative) pass with margin
≥10× the threshold. Outside: τ²-correction polynomial loses numerical
stability (sup_err can grow to 1e85+ at the worst corner). Out-of-regime
combos are tagged `[OOR]` in `cev_sweep_full` (slow-tests) and excluded from
the strict-pass default set.

- **σ < 0.50** — PDE stability boundary. ζ-A τ²-correction loses numerical stability
  at σ=0.50 in several (T, K/S, β) combinations. Not limited to β≥0.70 — also fails
  at β=0.30 and β=0.50 depending on T and K/S.
- **σ·T·β < 0.40** — combined-magnitude corner; sup-norm error exceeds the 5e-2 budget
  when this product is too large (e.g. K/S=1.20, T=2.00, σ=0.30, β=0.70).
- **lam_peak < 1400** — Schroder oracle (test-side) numerical limit; `exp(-lam/2)`
  underflows to 0 in f64 when `half_lam ≥ 745`. Fixed in test helper by peak-pass
  guard + j_max=2000. Worst case: (K/S=1.0, T=0.25, σ=0.15, β=0.70) → λ≈1983.

### Test-side helper fixes

- `ncx2_cdf`: peak-pass guard (`(j as f64) > half_lam.max(5.0)`); j_max bumped
  100 → 2000. Prior early-exit at `j > 5` returned sum ≈ 0 when `e^(-half_lam) << 1e-12`,
  masquerading as a PDE blow-up at β=0.7.
- `cev_boundary_stress`: sup-norm window guard `[50, min(150, S_max − 25)]` excludes
  the LinearExtrapolate boundary residual from the measurement at the most aggressive
  truncation. Threshold G_cev_boundary_C tightened from 5e-1 → 2.5e-1 with the guard
  band (empirical err(150) = 1.16e-1; threshold = 1.16e-1 × ~2 with margin).

### Deferral confirmed

The high-(σ × β) and high-(σ·T·β) corners require either Magnus
integrator (avoids manual `f''', f''` FD) or a higher-resolution grid +
adaptive step controller. Both land in **v0.4.0** per ADR-0008
Amendment 2 + roadmap.

## v0.4.1 follow-up — oracle λ-underflow lifted (test-side)

The `lam_peak < 1400` clause of the v0.3.1 envelope has been removed in v0.4.1.
The underflow described above (test helper `noncentral_chi2_cdf` returning 0 silently
when `exp(-λ/2)` underflows at `λ/2 > 745`) was fixed by rewriting the helper as a
log-space Poisson recurrence (`log P_j` updated additively; `exp` only applied when
`log_p > -700`). With `j_max=2000`, the helper now supports `λ_peak ≲ 3500`, comfortably
covering all CEV combos generated by the default and slow-tests sweeps.

Pure test-side patch — no ζ-A or Magnus PDE code changed; the Magnus v0.4.0 envelope
(`σ<0.70 ∧ σ·T·β<0.80`) is unchanged. The combo
`(K/S=1, T=0.25, σ=0.15, β=0.70)` (λ≈1983), previously excluded from `DEFAULT_COMBOS`
as an oracle-OOR case, is now restored.
