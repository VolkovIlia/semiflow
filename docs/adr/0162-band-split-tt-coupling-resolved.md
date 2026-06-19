# ADR-0162 — Band-split interpolation shift + spectral pair factor resolve the coupled-TT convergence boundary (TRIZ verdict A)

**Status:** ACCEPTED · **Date:** 2026-06-11 · **Branch:** `feat/v9.1.0-genuine-scurve`
**Theme:** v9.1.0 — Shift C: genuine no-solver coupled high-d PDE evolver for the constant-coef correlated-Gaussian class
**Parent:** ADR-0159 Amendment 1 (superseded for the boundary verdict) · **Math:** §52.2 Amendment 1 + §52.9 Round-4 NORMATIVE
**Supersedes:** ADR-0159 Amendment 2 (BOUNDARY verdict, hereby WITHDRAWN as a false compromise) · **Withdraws:** ADR-0161 (contingency NO-GO, outcome (i) achieved)
**Source:** `.dev-docs/specs/v9.1.0-s3-triz-resolution.md` §8.3 + §10.13.3 + §11.7 (all rounds)

## Context

ADR-0159 Amendment 2 issued a BOUNDARY (fundamental) verdict on the coupled TT evolver: the integer-index periodic shift is rank-O(1) but cannot converge (quantization floor O(1)), so rank-preservation and PDE accuracy were deemed mutually exclusive. The TRIZ analysis (`v9.1.0-s3-triz-resolution.md`) found that premise empirically false. ADR-0161 (contingency NO-GO) is therefore WITHDRAWN.

## Decision

**Round 1 — diagonal leg (band-split shift):** Replace the integer-index periodic shift `P_{round(h/dx)}` with a fixed-width (`w=4` cubic-Lagrange) band-split interpolation shift `Sₕ = Σ_m c_m(h/dx)·P_{s₀+m}`. Each band is a permutation (QTT-op-rank ≤ 2, Kazeev–Khoromskij); the 4-band sum rounds to rank 3 constant in grid resolution and `d` (measured; `T_TT_BAND_SHIFT_RANK`). Band-weights carry the continuum-exact displacement → `O(τ²)` convergence (`T_TT_BAND_SHIFT_RANK PASS`).

**Round 2/3 honesty correction:** The round-2 "stable rotated correlated-shift pair factor / `R·diag(S_h)·R^T`" design was SOUND in the dense-expm sense but mis-stated the solver-free property. A shift along a rotated eigendirection is inherently two-axis (op-rank 6 ≠ 1, proven `probe_adjudicate_rotated_shift.py`; R1 literal construction rel-err 1.16 — FAILS) and cannot be a pure per-axis Markov band-shift. The round-2 "no solver" prose therefore overclaimed.

**Round 4 — coupling leg (spectral pair factor):** The solver-free, exact realisation is the spectral (FFT-diagonal) apply `exp(τ·L_pair)·u = ifft2(exp(τ·symbol)⊙fft2(u))`. This is: machine-exact (1.2e-15 vs dense `expm`, `probe_adjudicate_rotated_shift.py` R3); no LU (FFT is a fixed unitary + elementwise `exp`); same op-rank 6 as the dense factor (curse-escape identical). The `exp(τ·symbol)` diagonal is built once per `(pair,τ)` and reused every step (no per-step solve). Paired with the band-split diagonal shift, this yields a per-step operator with no linear solver anywhere (Theorem-6 R2 honoured throughout, coupling factor included).

## Consequences

The §52.9 genuine high-d coupled-PDE-solver claim is ACHIEVED for the **constant-coefficient correlated-Gaussian / linear cross-diffusion class** (constant diagonal diffusion `aⱼ`, **drift `b = 0`**, scalar reaction `c`, constant `ρ`; **tridiagonal or block-disjoint ADJACENT pairs** only; `|ρ|<1`): storage `O(d·n·r²)`, pair-factor op-rank `r=O(1)` (measured 6), no linear solver, EXACT for this class (commuting circulant generators). Variable-coef/nonlinear/slowly-decaying-precision remain research-track (ADR-0158) — genuine O(τ²) Strang at best, rank uncapped.

**`CoupledTtChernoff::new` enforces fail-loud construction guards (NORMATIVE):**
- Any axis with `b_j ≠ 0` (drift advection) — panics: "CoupledTtChernoff drift b≠0 is not supported in v9.1.0 (drift advection unimplemented — ADR-0162 / §52.9 v9.2.0 deferral); pass b = 0".
- Any non-adjacent pair `(j,k)` with `k > j+1` — panics: "CoupledTtChernoff non-adjacent pair (j,k) with k>j+1 is not supported in v9.1.0 (only tridiagonal / block-disjoint adjacent pairs; true dense coupling deferred to v9.2.0)".

**DEFERRED to v9.2.0:** drift advection (`b ≠ 0`) and true dense / non-adjacent-pair coupling. Both were surfaced by v9.1.0 QA coverage testing and are rejected fail-loud rather than silently wrong.

The spectral R3 factor is **SHIPPED** (`crates/semiflow-core/src/tt_spectral.rs` + `tt_coupled_pair.rs`); the intermediate dense-LU R0 path has been replaced and survives only as a `#[cfg(test)]` reference. The `G_TT_COUPLED_EXACT` exactness gate (≤1e-12 vs independent dense `expm(τL_h^{dx})`, `d∈{3,4}`, `ρ≠0`) is RELEASE_BLOCKING and **PASSES** (`#[ignore]`+`slow-tests` for ~140s runtime only); the `no_lu_in_coupling` source-scan test verifies no linear solver is on the production coupling path. Form (i) (exact, no-solver) is the certified claim. No new dependency added (Constitution Override #1 ≤3 deps intact): the spectral apply is implemented as an in-tree direct O(n²) DFT, requiring no external crate. A faster in-tree FFT (radix-2 or Bluestein) is an optional future performance optimisation if the O(n²) cost is profiled as a bottleneck on large `n`; that path remains in-tree and dep-free. ADR-0161 withdrawn at v9.1.0 tag.

**ROUND-3 honesty addendum (2026-06-11 — HISTORICAL; superseded by spectral R3 implementation this session):** At the time of the initial v9.1.0-s3 documentation pass, `tt_coupled.rs` still implemented the round-1 explicit additive indefinite cross (not the spectral factor); that scheme reproduced the P4 failure (joint slope 1.394, finest 3.9e-2 vs `expm`; `probe_crux2_shipped_gate.py`) and was NOT exact. That state has been **superseded**: the spectral R3 factor is now shipped (`crates/semiflow-core/src/tt_spectral.rs` + `tt_coupled_pair.rs`), the dense-LU R0 path is `#[cfg(test)]`-only, and the exactness gate PASSES. Form (i) (exact, no-solver, third S-curve) is the current certified claim. The P4 failure measurements and the form (ii) fallback are archived here as the historical record of the pre-spectral state; they are no longer normative.
