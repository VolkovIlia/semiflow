# ADR-0178 — Engineer hand-off: implement `tt_varcoef::VarCoefTt` (per-axis variable-coef TT step)

**For:** agentic-engineer. **Design:** ADR-0178 + math §52.10. **Branch:** `issue-2-tt-varcoef`.
**Discipline:** ADDITIVE — do NOT modify `tt_chernoff.rs`. New sibling. ≤50-line functions, ≤500-line
files, ≤3 deps (in-tree `tt_core` only — no new crates). Do NOT touch the `ChernoffFunction` trait.

## What already exists (read first — do NOT duplicate)

- `crates/semiflow-core/src/tt_chernoff.rs` — `TtChernoff`/`TtState` (constant diagonal-A). **UNTOUCHED.**
- `crates/semiflow-core/src/tt_varcoef_spectral.rs` — `S3VarCoefEvolver` + the math you REUSE:
  `residual_tridiag`, `p2_apply_tridiag`, `varcoef_axis_step` (the `P₂(τ/2)·k(τ)·P₂(τ/2)` sandwich on a
  FLAT 1-D line). Your job is to apply that **per-axis line algorithm to each TT core's mode axis**.
- `crates/semiflow-core/src/tt_core.rs` — `TtCore` (`r_left × n × r_right`, `.get/.set/.data`),
  `tt_round`. REUSE `tt_round` verbatim.
- `crates/semiflow-core/src/tt_drift_spectral.rs` — `apply_drift_spectral_axis` (the const-coef
  `k_j(τ)` factor). REUSE for `k_j`.

## File checklist

1. **NEW** `crates/semiflow-core/src/tt_varcoef.rs` (≤500 lines). Implements:
   - `pub struct VarCoefTt<F> { a_axis: Vec<Vec<F>>, b_axis: Vec<Vec<F>>, v_axis: Vec<Vec<F>>, domain: Vec<(F,F)>, eps_round: F }`
     (per-axis arrays only — this IS the fail-loud wall: non-separable `a(x_i,x_j)` is unrepresentable).
   - `pub fn new(...) -> Result<Self, SemiflowError>` — validate `a_axis[j].len()==n`, `b/v` shapes,
     `domain.len()==d`, **parabolicity `a_axis[j][i] > 0 ∀i,j`** (fail-loud). Out-of-class ⇒
     `Err(SemiflowError::VarCoefOutOfClass { detail })`.
   - `pub fn step(&self, tau: F, state: &mut TtState<F>)` — symmetric per-axis Strang sweep
     `(j=0:τ/2)…(j=d-1:τ)…(j=0:τ/2)`, each axis applying `apply_varcoef_core(&mut state.cores[j], …)`,
     then `tt_round(&mut state.cores, self.eps_round)`. (Mirror `TtChernoff::step` structure.)
   - `pub fn evolve(&self, t_final: F, n_steps: usize, state: &mut TtState<F>)` — n_steps × `step`.
   - `fn apply_varcoef_core(core: &mut TtCore<F>, a, b, v, dx, tau)` — the ONE new kernel: for each
     `(r_left, r_right)` slab pair, extract the length-`n` mode line, call the SAME
     `P₂(τ/2)·k(τ)·P₂(τ/2)` sandwich as `varcoef_axis_step` (reuse `residual_tridiag`/`p2_apply_tridiag`/
     `apply_drift_spectral_axis`), write back. **The bond indices `(r_left, r_right)` are spectators —
     never index-mixed.** That spectator property IS the §52.10d rank-preservation; preserve it
     exactly (no cross-bond arithmetic). Extract helpers to stay ≤50 lines/fn.
2. **NEW** `crates/semiflow-core/src/tt_varcoef_api.rs` (optional, only if `VarCoefTt` + the doc stanza
   would exceed 500 lines — otherwise keep the type in `tt_varcoef.rs`). Mirror `tt_nonsep_varcoef_api.rs`.
3. **`crates/semiflow-core/src/lib.rs`** — `pub mod tt_varcoef;` + `pub use tt_varcoef::VarCoefTt;`
   (mirror the `tt_chernoff::{TtChernoff, TtState}` and `tt_varcoef_spectral` export blocks).
4. **`SemiflowError`** (wherever the enum lives — grep `S3OutOfClass`) — add variant
   `VarCoefOutOfClass { detail: &'static str }` next to `S3OutOfClass`. Do not reuse `S3OutOfClass`
   (distinct boundary).
5. **GATE** `crates/semiflow-core/tests/g_tt_varcoef.rs` — ALREADY WRITTEN (spec, this ADR). Wire the
   two `unimplemented!()` points ONLY: `run_varcoef_tt` (construct + evolve `VarCoefTt`) and
   `oracle_inner_d` (closed-form linear-`a` reference matching `scripts/verify_tt_varcoef.py` — NOT a
   self-comparison). Do NOT weaken any threshold or assert. The const-mean comparison in the variation
   check: call `run_varcoef_tt` with a flat `a_j = mean(a_j)` profile (or route through `TtChernoff` for
   the constant case) so `‖var − const‖` is meaningful.
6. **ORACLE** `scripts/verify_tt_varcoef.py` — ALREADY WRITTEN and PASSING (3/3). Keep the Rust
   `oracle_inner_d` consistent with its derivation.
7. **math §52.10** — ALREADY WRITTEN in `contracts/semiflow-core.math.md`. Cite it in the module rustdoc.

## Explicit fail-loud WALL locations (mirror ADR-0162 panics; never a silent floor)

| Uncovered case | Where it errors | Mechanism |
|---|---|---|
| non-separable `a(x_i,x_j)` | constructor type | structurally unrepresentable (no per-axis slot) — no code path can build it |
| `a_axis[j][i] ≤ 0` (non-parabolic) | `VarCoefTt::new` | `Err(VarCoefOutOfClass{detail:"a_axis[j][i] must be > 0"})` |
| shape mismatch (`a/b/v/domain`) | `VarCoefTt::new` | `Err(VarCoefOutOfClass{detail:"per-axis arrays must have length d/n"})` |
| `n < 2` | `VarCoefTt::new` | `Err(VarCoefOutOfClass{detail:"n must be >= 2"})` |
| non-diagonal constant A | NOT this module | owned by `CoupledTtChernoff` / ADR-0162 walls |
| dense `a(x)` / nonlinear / time-dep | constructor type | unrepresentable by per-axis arrays (INTRINSIC_LIMIT, §52.10.5) |

Add a `#[cfg(test)] mod boundary_tests` (mirror `tests_s3` in `tt_varcoef_spectral.rs`) asserting each
`new(...)` rejection above `.is_err()`, and an accept-valid case.

## Reduction invariant (load-bearing — add as a fast unit test)

`a_axis[j] = const ⇒ R_j = 0 ⇒ P₂ = I ⇒ step ≡ const-coef spectral`. Assert: with every
`a_axis[j]` flat, one `VarCoefTt` core step is bit-close (≤1e-12) to the corresponding
`TtChernoff` / `apply_drift_spectral_axis` const path on the same core. (Mirrors
`varcoef_step_const_a_zero_drift_equals_spectral` in `tt_varcoef_spectral.rs`.)

## No-solver invariant (keep R2)

`tt_varcoef.rs` MUST contain NO `lu_solve_inplace(` and NO `dense_expm(` (grep-clean). Only
FFT-diagonal `k_j` and tridiagonal mat-vecs, as in `tt_varcoef_spectral.rs`.

## Verification commands

```bash
# 1. Oracle (must already pass — re-confirm after any §52.10 edit)
python3 scripts/verify_tt_varcoef.py            # expect: T_TT_VARCOEF PASS (3/3)

# 2. Fast tests (reduction invariant + boundary rejections + build)
cargo run -p xtask -- test-fast                 # all green, no regressions

# 3. The slow carrier curse-escape gate (the two assertions)
cargo test -p semiflow-core --features slow-tests \
    --test g_tt_varcoef -- --ignored --nocapture
#   PASS iff: convergence slope ≤ −1.95 AND rank-1 IC ⇒ peak_rank()==1 at every d
#             AND log-rank-vs-d slope < 0.70 AND byte-reproducible.

# 4. No-solver grep (R2)
! grep -nE 'lu_solve_inplace\(|dense_expm\(' crates/semiflow-core/src/tt_varcoef.rs

# 5. Keep ALL existing TT gates green (no regression on the constant-A path)
cargo test -p semiflow-core --features slow-tests \
    --test g_tt_chernoff --test g_tt_coupled --test g_tt_strang_identity \
    --test g_tt_band_converge -- --ignored
```

## Done criteria

- `tt_chernoff.rs` diff = ∅ (additive sibling only).
- `VarCoefTt` exported; `VarCoefOutOfClass` added; per-axis-only type (the wall).
- `g_tt_varcoef` PASSES both assertions on the carrier; existing TT gates still green.
- Oracle `T_TT_VARCOEF PASS`; no-solver grep clean; reduction invariant test green.
- ≤50-line fns, ≤500-line file, ≤3 deps, no trait-signature change.
