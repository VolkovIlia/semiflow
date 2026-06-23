# ADR-0177 — Engineer hand-off: multi-parameter (K>1) reverse-AD via regions

**For:** agentic-engineer · **Design:** ADR-0177 + math.md §51.10 · **Issue #1**
**Principle:** ADDITIVE wherever possible; the **K=1 path MUST stay byte-identical**.

## CRITICAL invariants (do not violate)

- **NEVER alter the `ChernoffFunction` trait signature** (56 dependents). The
  region kernel is ONE `DiffusionChernoff` built via the existing `with_closure`
  — no new trait, no new generic on `ChernoffFunction`.
- **K=1 byte-identical regression.** `value_and_grad_k1` and `value_and_grad`
  with a length-1 θ MUST produce bit-identical results to today
  (`G_REVERSE_AD_GRADIENT` cross-mode + the K=1 unit tests are the witnesses).
- Functions ≤50 lines, files ≤500 lines, ≤3 new deps (target: **zero** new deps).
- Errors as values (`Result<_, SemiflowError>`); fail-loud on out-of-scope.

## File checklist

### 1. NEW `crates/semiflow-core/src/reverse_region.rs` (additive, declare in `lib.rs`)

The region map + per-region dual seeding. Suggested surface (engineer may refine
names; keep ≤500 lines):

```rust
/// DoF-aligned region partition ρ: node index → region id (math.md §51.10).
/// Contiguous by default; `region_count` regions over `n_grid` nodes.
pub struct RegionMap {
    /// `region_of[i]` = region id r of node i.  len == n_grid.
    region_of: Vec<usize>,
    region_count: usize,
}

impl RegionMap {
    /// Contiguous DoF-aligned partition of `n_grid` nodes into `k` regions.
    pub fn contiguous(n_grid: usize, k: usize) -> Result<Self, SemiflowError>;
    /// Region id of node `i`.
    #[inline] pub fn region_of(&self, i: usize) -> usize;
    #[inline] pub fn region_count(&self) -> usize;
}
```

- Build the primal kernel ONCE via `DiffusionChernoff::with_closure` with
  `a(x_i) = θ[region_of(i)]` (a closure capturing `theta: Vec<F>` + `RegionMap`).
  This is the single self-adjoint kernel; `apply_transpose_step` (F^⊤=F) unchanged.
- Validation: `theta.len() == region_count` else `UnsupportedOperation`.

### 2. `crates/semiflow-core/src/reverse_sweep.rs` — per-region dual seeding + grad[r]

- Generalise `backward_step` accumulation from `grad[0] += dot` to a loop over
  `r ∈ 0..K`: `grad[r] += ⟨λ_k, b_k^{(r)}⟩`.
- **Per-region dual seed** in the column eval: build `kernel_dual` for region r so
  its coefficient closure returns `Dual::variable(θ_r)` on nodes `i ∈ Ω_r` and
  `Dual::constant(θ_{ρ(i)})` elsewhere (state zero-tangent, `Dual::constant(τ)`).
  Two clean options — pick the lower-churn one:
  - (A) one `DiffusionChernoff<Dual<F>>` whose closure reads a `current_region:
    Cell<usize>` / arg, re-seeding per r (one heap kernel, K closure evals); or
  - (B) `step_jacobian_col` gains a `region: usize` + `&RegionMap` parameter and
    masks the seed to Ω_r. Prefer (B) — it keeps the dual kernel stateless and is
    the closest additive change to the existing `step_jacobian_col`.
- The **K=1 special case** (`region_count == 1`) MUST take the existing code path
  unchanged (seed everywhere, single `b_k`, single dot) — gate this with an early
  branch so the byte-identical regression is structural, not incidental.
- `backward_sweep` returns `grad: Vec<F>` of length K (already a `Vec`; just size
  it to `region_count` instead of hard-coded `1` at `reverse_sweep.rs:115`).

### 3. `crates/semiflow-core/src/reverse_ad.rs` — lift the K>1 guard

- Replace the `theta.len() != 1` rejection (lines ~299–306) with
  `theta.len() == self.region_map.region_count()` validation (carry a `RegionMap`
  on `ReverseChernoff`, or pass it to `value_and_grad`). Out-of-scope cases
  (variable-a within a region, non-DoF-aligned, non-self-adjoint) stay fail-loud.
- `value_and_grad_k1` stays a thin wrapper: it constructs a 1-region map ⇒ routes
  through the SAME sweep ⇒ byte-identical. **Do not** add a forward shortcut.
- Update docstrings: cite ADR-0177 + §51.10, drop the "K>1 not supported" prose.

### 4. Gate edits

**`tests/g_reverse_ad.rs` — extend `G_REVERSE_AD_GRADIENT` to K-vector FD parity:**
- Add a K-region case (e.g. `N_GRID=128`, `K=4`, contiguous regions, distinct
  `θ_r`). For each `r ∈ 0..K`: central-difference `∂J/∂θ_r` by perturbing ONLY
  `θ_r` (`θ_r ± h`), assert `|grad[r] − fd_r| / |fd_r| < 1e-9` (reuse `GRAD_REL_GATE`).
- Keep the existing K=1 cross-mode (`< 1e-12`) + anti-tautology asserts untouched
  (they are the K=1 regression witnesses).
- New assertion message prefix `G_REVERSE_AD_GRADIENT (K-vector) …`.

**`tests/g_reverse_ad_advantage.rs` — confirm K>1 binding with the region kernel:**
- Swap `make_dual_kernel` / `reverse_step_count` to drive the **region** kernel so
  the K accumulation is genuinely per-region (not a relabel). The harness already
  parameterises `K ∈ {1,4,16,64}`; the assertion `ratio(64)/ratio(1) ≥ 8`
  (`ADVANTAGE_GATE`) is unchanged — it now binds at K>1 because the reverse sweep
  stays O(1)-in-K trajectory passes while forward dual-AD is O(K).

### 5. NEW oracle `scripts/verify_reverse_ad_kvector.py` (ALREADY WRITTEN — design spec)

- The file exists (executable sympy spec, PASSES today). Wire it into the
  `test-fast` sympy oracle harness alongside `reverse_transpose_kit.py`. Prints
  `T_REVERSE_AD_KVECTOR PASS`. RELEASE_BLOCKING (§51.10). No edits required unless
  you tighten `step_matrix` to mirror the real stencil (support + adjoint identity
  are invariant — keep PASS).

### 6. Math section — `contracts/semiflow-core.math.md` §51.10 (ALREADY WRITTEN)

- NORMATIVE. No engineer edit needed; implement to it.

## Verification commands

```sh
# 1. Sympy oracle (must already PASS — math fidelity, pre-flight):
python3 scripts/verify_reverse_ad_kvector.py            # expect T_REVERSE_AD_KVECTOR PASS

# 2. Fast suite incl. K=1 regression + structure gate (test-fast):
cargo run -p xtask -- test-fast
#   wraps: cargo test --workspace --features parallel,simd
#   includes G_REVERSE_AD_STRUCTURE (must stay PASS — Jᵀ still load-bearing).

# 3. Slow K-vector gates (ignored; run explicitly):
cargo test -p semiflow-core --features slow-tests --test g_reverse_ad \
    -- --ignored --nocapture          # G_REVERSE_AD_GRADIENT (K-vector FD parity) + CHECKPOINT
cargo test -p semiflow-core --features slow-tests --test g_reverse_ad_advantage \
    -- --ignored --nocapture          # G_REVERSE_AD_ADVANTAGE binds at K>1

# 4. K=1 byte-identity spot-check (no behavioural drift):
cargo test -p semiflow-core reverse_ad   # unit tests in reverse_ad_tests.rs
```

## Done criteria

- [ ] `reverse_region.rs` added + declared in `lib.rs`; `RegionMap` DoF-aligned.
- [ ] `backward_sweep` accumulates `grad[r]` for `r∈0..K` via per-region seeds.
- [ ] K>1 guard lifted; `theta.len()==region_count` validated; out-of-scope fail-loud.
- [ ] K=1 path byte-identical (cross-mode `<1e-12` + unit tests unchanged).
- [ ] `G_REVERSE_AD_GRADIENT` K-vector FD parity `<1e-9` for each `grad[r]`.
- [ ] `G_REVERSE_AD_ADVANTAGE` `ratio(64)/ratio(1) ≥ 8` with the region kernel.
- [ ] `G_REVERSE_AD_STRUCTURE` still PASS (Jᵀ load-bearing, k=n→1).
- [ ] `T_REVERSE_AD_KVECTOR PASS`. `ChernoffFunction` signature untouched. ≤3 deps.
