# ADR-0181 — Engineer hand-off: production rough-Heston pricer (issue #9)

Design: `docs/adr/0181-production-rough-heston.md`.
Oracle (already passes): `scripts/verify_rough_heston_mc.py`.
**Additive only** — `examples/rough_heston_pricer.rs` capability/L-gate mode MUST keep
running (`--rate 0.0` recovers pre-#9 forward-ish behaviour). Contract-first: implement
against ADR-0181 + ADR-0082 §33.7 AMENDMENT 2; do NOT invent the kernel — `c_00 = −r` rides
the EXISTING `MatrixDiffusionChernoff<4>` matrix-exp.

## The two things that will bite you

1. **Gate I is NOT vs true rough-Heston.** It is the Chernoff kernel vs an MC of the kernel's
   OWN linearised/frozen-V₀ model. Simulate the SAME approximation the kernel encodes
   (frozen `a_00 = ½V₀`, `c_{0,k} = ρξw_k` reaction coupling, `c_kk = −γ_k` decay), NOT the
   full nonlinear SDE. Mixing them silently injects model bias (II) into the numerical gate
   (I) and the gate becomes meaningless. The oracle's `mc_price_kernel_model` is the
   reference layout.

2. **Discount enters through the reaction matrix, not a post-multiply.** Set
   `mat[0][0] = -r` in `fill_c_ij`. The block-CN Strang half-steps `exp(τC/2)` compound to
   `e^{−rT}` over the backward sweep (ADR-0181 §D1; oracle `discount_factor_check` proves
   1e-12). Do NOT add a separate `× e^{−rT}` multiply — that is a second, desyncable code
   path.

## File checklist

### 1. Example — `crates/semiflow-core/examples/rough_heston_pricer.rs` (extend, additive)
- [ ] Add `struct RoughHestonParams { r: f64, /* + the existing consts, parameterised */ }`
      with a `Default` matching the current constants (`r = 0.05`). Keep `HURST`, `V_0`,
      etc. as defaults; expose `r` (and optionally `frozen_v0: bool`) for the advisory
      bias measurement.
- [ ] `fill_c_ij`: change `mat[0][0] = 0.0;` → `mat[0][0] = -params.r;`. This is THE
      discounting change. ≤50-line fn preserved (thread `params` or `r` in via a closure
      capture / small struct).
- [ ] Add `--rate <f64>` CLI flag (default 0.05). `--rate 0.0` MUST reproduce the old
      forward-ish output bit-for-bit (regression guard).
- [ ] Add a `--price` mode (vs the existing latency mode): build the IC, evolve
      `n = T/τ` backward steps, read component-0 at `x=0` (or interpolate at strike) →
      print the discounted call price per strike. Keep the latency mode untouched.

### 2. Oracle gate — `crates/semiflow-core/tests/rough_heston_mc_oracle.rs` (NEW)
Mirror `tests/cev_high_lam_oracle.rs` structure (self-contained, deterministic, slow-tests).
- [ ] `#[cfg_attr(not(feature="slow-tests"), ignore)]` — 1M paths is slow.
- [ ] `G_ROUGH_HESTON_MC_PARITY` (RELEASE_BLOCKING). Implement the MC of the kernel's own
      model in Rust mirroring `verify_rough_heston_mc.py::mc_price_kernel_model`:
      Euler log-spot + **QE (Andersen 2008) CIR** factors, antithetic, seed
      `PCG64(0xC0FFEE_BABE_DEAD_BEEF)`, `n_paths=1_000_000` (500k antithetic pairs),
      `n_steps=200`. Estimator: `e^{−rT}·mean[(S_T−K)_+]`, `MC_stderr = e^{−rT}·std/√N`.
- [ ] Build the Chernoff price: `MatrixDiffusionChernoff<f64,4>` on the example's grid,
      evolve `n=T/τ` steps, read component-0 → call price per strike
      `K ∈ {90, 100, 110}`.
- [ ] Assert `|C_chernoff − C_mc| ≤ K_SIGMA·MC_stderr + DELTA_KERNEL` with `K_SIGMA=3.0`.
      **`DELTA_KERNEL` is MEASURED**, not guessed: add a sub-test that runs the kernel at
      `N_GRID=48` vs `N_GRID=192` (self-convergence) and fit `δ_kernel`; write the fitted
      value as the test literal AND back-annotate ADR-0181 §D3 + math §33.9 (replace the
      `≈0.55` placeholder with the measured number). Initial target: ≤ 0.55 price units
      (~0.6% ATM). If the coarse demonstrator grid cannot hit it without breaking the
      latency story, run gate I on a dedicated accuracy grid (`N_GRID=192`, `TAU=0.01`) and
      document the accuracy-mode/latency-mode split in the test header.
- [ ] Discount sub-test: coupling zeroed, flat IC `u₀≡1` on component 0, after `n` steps
      component-0 == `e^{−rT}` to ≤1e-12 (isolates the discount; mirrors oracle
      `discount_factor_check`).

### 3. Advisory record — `crates/semiflow-core/tests/rough_heston_model_bias.rs` (NEW, advisory)
- [ ] `A_ROUGH_HESTON_MODEL_BIAS` — NOT release-blocking; prints measured magnitudes,
      never asserts-fail (warn-only; `#[cfg_attr(not(feature="slow-tests"), ignore)]`).
      Three measured sub-biases (ADR-0181 §D4), each reported as % of the ATM call:
      (a) frozen-V₀ vs stochastic-`√V_t` spot MC; (b) reaction-coupling vs exact correlated
      cross-term MC; (c) 3-factor vs high-factor (≥20) El Euch–Rosenbaum MC at H=0.1.
      Emit a JSONL/stdout line per sub-bias. Honest expectation: aggregate O(H) ≈ 1–5%.

### 4. Contract — `contracts/semiflow-core.properties.yaml`
- [ ] Add `G_ROUGH_HESTON_MC_PARITY` under `properties:` (RELEASE_BLOCKING; slow-tests;
      test file `tests/rough_heston_mc_oracle.rs`; params + strikes + tolerance literals
      from §2). Follow the existing entry shape (name / cases / purpose / invariant).
- [ ] Add a NEW top-level `advisory_records:` section (sibling to `latency_gates:`) with
      `A_ROUGH_HESTON_MODEL_BIAS` (severity ADVISORY; sub-bias list; expected O(H) 1–5%).
- [ ] Add a `notes:` entry documenting the two-tier (I/II) design + that gate I is vs the
      SAME model, not true rough-Heston. Bump `schema_version` per the project convention.

### 5. Math — `contracts/semiflow-core.math.md` §33.9 (NEW, append-only)
- [ ] §33.9 — discounting: `c_00 = −r` ⟹ per-step `exp(−rτ/2)·exp(−rτ/2)=exp(−rτ)` ⟹
      `e^{−rT}` over `n` steps (Feynman-Kac `∂_τ u = Lu − ru`). Cite §33.7 AMENDMENT 2 for
      the reaction-matrix entry point.
- [ ] §33.9 — the TWO-ERROR-SOURCE decomposition: (I) numerical (kernel vs MC of same
      model, gate I, RELEASE, target ~0.6%); (II) model approximation (frozen-V₀ +
      leading-order coupling + O(H) 3-factor CF error, gate II, ADVISORY, expected 1–5% at
      H=0.1). State the defensible bounded price-claim verbatim from ADR-0181 §D5. Cite
      Andersen 2008 (QE), El Euch–Rosenbaum 2019 (multifactor reference), Carr-Cisek-Pintar
      2021 (the 3-factor model).

### 6. Oracle — `scripts/verify_rough_heston_mc.py` (ALREADY WRITTEN, passes)
- [ ] No change needed; keep as the language-independent pre-flight reference. When the
      Rust kernel price is wired, the synthetic `kernel_prices` offset in the oracle MAY be
      replaced by a fixture exported from the Rust run (optional cross-check).

### 7. CHANGELOG.md
- [ ] `### Added` — production rough-Heston pricer (issue #9): risk-neutral discounting
      (`c_00 = −r`), `G_ROUGH_HESTON_MC_PARITY` gate I (RELEASE), `A_ROUGH_HESTON_MODEL_BIAS`
      advisory (gate II). State the honest claim: oracle-validated solver of a documented
      4-factor Markov model (~0.6%), itself O(H)-biased ~1–5% vs true rough-Heston at H=0.1.
      Note `--rate 0.0` preserves the demonstrator.

## Verification commands

```bash
# 0. design oracle (already green)
python3 scripts/verify_rough_heston_mc.py

# 1. fast suite (no regressions; capability mode + --rate 0.0 regression)
cargo run -p xtask -- test-fast

# 2. gate I — kernel vs MC of the same model (RELEASE_BLOCKING, slow)
cargo test -p semiflow-core --features slow-tests rough_heston_mc_oracle

# 3. advisory — model-bias magnitudes (warn-only)
cargo test -p semiflow-core --features slow-tests rough_heston_model_bias -- --nocapture

# 4. example: price mode + latency mode both run
cargo run --release -p semiflow-core --example rough_heston_pricer -- --price --rate 0.05
cargo run --release -p semiflow-core --example rough_heston_pricer -- --n-ticks 100 --rep 0

# 5. lints / suckless gates
cargo clippy --all-targets -- -D warnings
cargo run -p xtask -- check-lints
```

## Do NOT

- Do NOT compare gate I against true rough-Heston (no closed form / no engine; that is
  gate II, advisory).
- Do NOT add a post-evolution `× e^{−rT}` multiply — discount rides `c_00 = −r` through the
  existing matrix-exp.
- Do NOT promote `A_ROUGH_HESTON_MODEL_BIAS` to RELEASE_BLOCKING — it gates a modelling
  choice, not implementation correctness.
- Do NOT break the latency demonstrator / L-gate path (`--rate 0.0` must reproduce it).
- Do NOT guess `DELTA_KERNEL` — measure it via the N=48-vs-192 self-convergence sub-test and
  back-annotate ADR-0181 §D3 + math §33.9.
- Do NOT use Euler full-truncation for the CIR factors — use QE (Andersen 2008); the
  positivity bias would contaminate `δ_kernel`.
```
