# ADR-0152 — Obstacle evolver inactive-set Γ primitive + multi-asset D≥2 kernel

**Status:** ACCEPTED · **Date:** 2026-06-08 · **Branch:** `feat/v8.2.0-math`
**Theme:** v8.2.0 — Wave-2 B-7 (second-order Greeks + multi-asset obstacles)
**Designed-By:** ai-solutions-architect (engineer-spec format, mirrors ADR-0134)
**Parent / pre-flight:** ADR-0150 (research GO, `scripts/obstacle_gamma_kit.py` 5/5 PASS)
**Extends:** ADR-0116 (projective-splitting `ObstacleChernoff`), math.md §44.
**Gates:** `G_OBSTACLE_GAMMA` (RELEASE_BLOCKING), `G_OBSTACLE_SLOPE_2D` (RELEASE_BLOCKING),
`T_OBSTACLE_GAMMA` (NORMATIVE oracle). **Math:** math.md §44.5 additions (NORMATIVE).

## Context

ADR-0116 shipped `ObstacleChernoff<C, O, F>` (`V^{n+1} = Π_g(S(Δτ)Vⁿ)`, `Π_g = max(·,g)`)
with two declared OPEN items in math.md §44.5/§44.7: **second-order Greeks (Γ) at the free
boundary** and **`D = 1` only**. ADR-0150's research pre-flight closed both as GO. (7a)
The value function is C¹ across the contact line `x*` (smooth-fit — Peskir 2005) but Γ = V″
is *discontinuous* there (Γ = 0 on the active/stopping set where `V ≡ g` is linear, Γ > 0
just inside the continuation set; `obstacle_gamma_kit.py` verified the perpetual-American-put
jump `Γ(S*⁺)=4.90` vs `Γ(S*⁻)=0` exactly). A single *global* Γ is therefore intrinsically
ill-posed at `x*`. (7b) The projection `Π_g(W)=max(W,g)` and the active-set mask `𝟙[W>g]`
are elementwise and carry no dimension assumption; `grid_nd.rs` already exposes the flat
storage + `x_at(d,k)` surface the D=1 kernel uses.

## Decision

Ship, in v8.2.0, (7a) an **inactive-set Γ primitive** that reports a classical, O(Δx²)-
convergent Γ strictly on the OPEN continuation set `{W > g}` and *refuses* it on the active
set and across `x*` (resolving the contradiction *in space*, reusing the existing active-set
mask), with a rustdoc HONESTY contract that no global C² Greek exists; and (7b) a
**`GridFnND`-generalized `ObstacleChernoff`** for multi-asset (D≥2) obstacles. The TRIZ
resolution for 7a (ADR-0150) is not a golden-middle averaged Γ across the kink — it gives the
*full* classical Γ where Γ lives and an honest *absent* flag where it does not. No
mollification is shipped in core (the mollified route is a documented SHOULD only — it cannot
manufacture a classical Γ in the O(ε) boundary layer).

## 7a — `apply_inactive_gamma_into` primitive (engineer spec)

**File: `crates/semiflow-core/src/obstacle_gamma.rs` (NEW, additive).**
RATIONALE for the new module: `obstacle.rs` is **441 LoC** at HEAD (verify with
`wc -l crates/semiflow-core/src/obstacle.rs`); adding the Γ primitive (~60 LoC) AND the D≥2
generalization (7b) would push it over the constitution's default 500-LoC cap. The
constitution (v4.0.0) records that *every* shipped v8 module stays under 500 by splitting into
sibling modules rather than invoking a Cohort carve-out (precedent: `reflection.rs` /
`reflection_regions.rs`). Therefore the Γ primitive ships in a **sibling module
`obstacle_gamma.rs`**; the D≥2 generalization (7b) stays in `obstacle.rs` (it is a type/loop
edit, not new code volume). Declare `mod obstacle_gamma;` in `lib.rs` next to `mod obstacle;`.

### Γ-refusal API — RECOMMENDED: companion boolean mask array (NOT a sentinel/NaN)

```rust
/// Inactive-set Γ = V″ of the projected value field, on the OPEN continuation
/// set only (math §44.5). Writes the central-difference second derivative
/// `gamma[i] = (v[i+1] − 2·v[i] + v[i−1]) / dx²` at every node `i` that is
/// strictly inside the continuation set AND whose 3-point stencil is itself
/// entirely inside it; sets `defined[i] = true` there. On the active set, at
/// the contact line `x*`, and within a one-node guard band of any active node
/// (the stencil would straddle the kink), writes `gamma[i] = F::zero()` and
/// `defined[i] = false` — Γ is REFUSED, not fabricated.
///
/// Returns the number of nodes where Γ is defined (the inactive-set size).
///
/// # Honesty (NORMATIVE, math §44.5)
///
/// Γ is C¹-not-C² at the free boundary: it JUMPS across `x*` (perpetual-put
/// witness Γ(S*⁺)=4.90, Γ(S*⁻)=0). There is NO classical global Γ; this
/// primitive deliberately exposes Γ ONLY where it mathematically exists. Callers
/// MUST consult `defined[i]` before reading `gamma[i]`; a `false` entry means
/// "Γ undefined here", never "Γ = 0".
///
/// # Errors
/// `DomainViolation` if `v.grid.n != gamma.grid.n` or `!= defined.len()`, or if
/// `v.grid.n < 3` (no interior 3-point stencil).
pub fn apply_inactive_gamma_into(
    &self,
    v: &GridFn1D<F>,           // the (already-projected) value field Vⁿ
    gamma: &mut GridFn1D<F>,   // OUT: Γ on the inactive set, 0 elsewhere
    defined: &mut [bool],      // OUT: companion mask — true iff Γ[i] is classical
) -> Result<usize, SemiflowError>
```

**Why companion mask, not sentinel/NaN (DECISION):**
1. **no_std + alloc, generic over `F: SemiflowFloat`** — a NaN sentinel assumes IEEE NaN
   semantics and is fragile under `--ffast-math`-style codegen and for any non-IEEE `F`; a
   `bool` mask is unconditionally correct.
2. **The kernel already produces exactly this mask** — `apply_active_set_adjoint_into`
   computes `active = w.values[i] > g(x_i)` nodewise (`obstacle.rs:279`). The Γ primitive
   REUSES that boundary (ВПР in ADR-0150's TRIZ): `defined[i]` is the *inactive* indicator
   intersected with the "stencil clear of the kink" guard. Zero extra state, one extra pass.
3. **Honest by construction** — `gamma[i] = 0` at refused nodes is a *valid Γ value* on the
   active set (Γ = 0 there is mathematically true), so a NaN sentinel would be both
   over-strong and ambiguous. The `defined` flag carries the orthogonal "is this a *classical
   interior* Γ I am willing to certify" bit. A caller that wants the masked field can do
   `gamma.values[i] * (defined[i] as u8 as f64-via-F)`.
4. **Mirrors the existing `active_set_into(w, active: &mut [bool])` signature** — same
   out-param `&mut [bool]` convention, same `DomainViolation` on length mismatch.

**Guard band (NORMATIVE):** node `i` is `defined` iff `i ∈ {1,…,n−2}` (interior, stencil
exists) AND all three of `{i−1, i, i+1}` are strictly inactive (`v > g`, the §44.5 strict
convention). One inactive node adjacent to the contact set is therefore correctly REFUSED
because its `i±1` stencil reaches the kink — this is the `≥1 node off x*` guard ADR-0150
requires. Implementation: compute the inactive indicator once into a scratch `&[bool]`
(reuse `Obstacle::active_set_into` against the *projected* `v` — note `{v > g}` is the
continuation set), then a single second pass ANDs the 3-point window.

**Boundary policy:** Γ is undefined at `i = 0` and `i = n−1` (no centred stencil) →
`defined = false`, `gamma = 0`. No one-sided stencil is fabricated (suckless: refuse, don't
approximate).

**Honesty boundary (NORMATIVE, mirrors ADR-0116 §44.5):** `apply_inactive_gamma_into` is a
**separate inherent method on `ObstacleChernoff`**, NOT a `ChernoffFunction` Greek and NOT
part of any `AdjointApply`/Greek supertrait. `ObstacleChernoff` MUST NOT expose a global C²
Greek. The rustdoc above MUST state the C¹-not-C² limit verbatim (the perpetual-put witness).

**No mollification in core.** Route B (softplus `g_ε`) from ADR-0150 is a documented SHOULD
only; it is provably non-classical in the O(ε) layer and is NOT shipped. If a future
consumer needs a bounded global Γ field, it lives downstream, not in core.

**Suckless:** `apply_inactive_gamma_into` ≤ 50 lines — extract the inactive-indicator pass
and the guard-band AND into a private `fn inactive_defined_mask(...)` helper in the same
module. No new deps. no_std + alloc (the one scratch `Vec<bool>` via `alloc`).

## 7b — `ObstacleChernoff` over `GridFnND<F, D>` (engineer spec)

**File: `crates/semiflow-core/src/obstacle.rs` (edit in place; mechanical, GO-by-construction).**

The generalization is a state-type swap `GridFn1D<F> → GridFnND<F, D>` on the projection path.
The projection and mask are elementwise over the flat `values: Vec<F>` (ADR-0150 sub-check 5),
so the inner loops do not change shape — only coordinate access generalizes from `x_at(i)` to
`x_at(d, k)` via a multi-index decode.

**Exact type/trait changes:**

1. **`Obstacle<F>` trait** — generalize `value_at(&self, point: &[F]) -> F` is ALREADY
   D-agnostic (it takes a `&[F]` coordinate slice — see `obstacle.rs:67`). KEEP it. Add a
   D-generic projection/active-set surface:
   ```rust
   // ADD (generic), KEEP the D=1 GridFn1D methods for back-compat:
   fn project_in_place_nd<const D: usize>(&self, dst: &mut GridFnND<F, D>)
       -> Result<(), SemiflowError> {
       // decode each flat index → [x₀,…,x_{D-1}] via grid.x_at(d, k), then
       // dst.values[flat] = max(dst.values[flat], self.value_at(&coords));
   }
   fn active_set_nd_into<const D: usize>(&self, w: &GridFnND<F, D>, active: &mut [bool])
       -> Result<(), SemiflowError> { /* same decode; active[flat] = w>g (strict) */ }
   ```
   Reuse the **existing row-major decode** pattern from `grid_nd.rs::enumerate_nd`
   (`remaining = flat; for d: k = remaining % n_d; remaining /= n_d`). DO NOT re-derive it.

2. **`ObstacleChernoff<C, O, F>`** — the struct is unchanged (`inner`, `obstacle`, `_f`).
   The bound on `C::S` is what generalizes. Provide **two impl blocks** (additive, no removal):
   - KEEP the existing `impl … where C: ChernoffFunction<F, S = GridFn1D<F>>` (D=1 path,
     ships unchanged — back-compat for all current users).
   - ADD `impl<C, O, F, const D: usize> ChernoffFunction<F> for ObstacleChernoff<C, O, F>
     where C: ChernoffFunction<F, S = GridFnND<F, D>>, O: Obstacle<F>` with
     `type S = GridFnND<F, D>`, `apply_into` = `inner.apply_into` then
     `self.obstacle.project_in_place_nd(dst)`, `order() = 1`, `growth() = inner.growth()`.

   These two impls do NOT overlap (distinct `C::S` associated types), so they coexist without
   a coherence conflict.

**What stays D=1-specialized (NORMATIVE):**
- `apply_active_set_adjoint_into` (the §44.5 adjoint primitive) — **stays D=1-only** in
  v8.2.0. The mask is D-agnostic, but the *inner adjoint* `S*(Δτ)` for ND generators is not
  yet a shipped surface (no ND `apply_adjoint_into` consumer). Document as deferred.
- `apply_inactive_gamma_into` (7a) — **stays D=1-only** in v8.2.0. Multi-asset Γ requires a
  per-axis second-difference stencil and an ND guard band (the kink becomes a free *surface*);
  this is real new work, not mechanical, and is explicitly DEFERRED (see Consequences).

So v8.2.0 ships: D≥2 *forward evolution* (projection) + D=1 Γ + D=1 adjoint. This is the
honest minimal cut — 7b's "mechanical" claim holds for the projection path only.

## math.md §44.5 additions (NORMATIVE spec — engineer writes the prose)

Append to §44.5 (after the existing "Honesty (greeks)" paragraph), two NEW normative blocks.
The prose MUST state, NORMATIVELY:

**§44.5.bis — Inactive-set Γ (C¹-not-C²; PROVEN ill-posed at x*, GATED on the open set).**
- Theorem statement: for the obstacle VI, `Γ = V″` is *continuous and positive on the open
  continuation set* `{V > g}` and *zero on the interior of the contact set*, with a
  **strictly positive jump across the free boundary** `x*` — hence V is C¹ but **not C²**
  there (Peskir 2005 smooth-fit gives C¹; the jump is the obstruction to C²). No classical
  *global* Γ exists.
- Closed-form witness (perpetual American put, `q=0`): `V = A·S^{−γ}`, `γ = 2r/σ²`,
  `S* = γ/(γ+1)·K`, `Γ(S) = A·γ(γ+1)·S^{−γ−2}` on `{S>S*}`, `Γ ≡ 0` on `{S≤S*}`, jump
  `= γ(γ+1)(K−S*)/(S*)² > 0`. At the canonical `(K,r,σ)=(1,0.05,0.20)` the numeric witness
  is **Γ(S*⁺)=4.90 vs Γ(S*⁻)=0** (cite `obstacle_gamma_kit.py` sub-checks 1–2).
- Primitive contract: `apply_inactive_gamma_into` reports the central-difference Γ on the
  open inactive set, converging at **O(Δx²)** (sub-check 3, no mollification), and REFUSES Γ
  (companion mask `defined = false`) on the active set / at `x*` / within the one-node guard
  band. The mollified route B is documented as a SHOULD that is **bounded but non-classical
  in the O(ε) layer** and is NOT shipped (cite ADR-0150).
- Citations: Peskir 2005; Casabán–Company–Egorova–Jódar 2024 (arXiv:2401.13361, inactive-set
  region-of-interest without smoothing); Howison–Reisinger–Witte 2013 (penalty boundary
  layer); Forsyth–Vetzal (spurious Γ oscillations near the boundary).

**§44.5.ter — Multi-asset D≥2 generalization (mechanical; projection path only).**
- State the elementwise identity `Π_g(W)=W·𝟙[W>g] + g·𝟙[W≤g]` and the mask `𝟙[W>g]` are
  dimension-agnostic over flat row-major storage (sub-check 5), so `ObstacleChernoff` over
  `GridFnND<F, D>` is well-defined for any `D≥1`, with the SAME order-1 declaration and the
  SAME `Π_g`-nonexpansiveness stability certificate (Theorem 44.1 is stated for a general
  convex cone — no D=1 assumption). Cite Casabán et al. 2024 (multi-asset American context).
- NORMATIVE scope note: v8.2.0 ships D≥2 *forward evolution only*; the active-set adjoint and
  the inactive-set Γ remain D=1 (the multi-asset free *surface* Γ stencil/guard is deferred).
- Update §44.7 "Limitations": change "`D = 1` only" to "D≥2 forward evolution shipped (v8.2.0);
  D≥2 adjoint and D≥2 Γ deferred"; change "second-order greeks (Γ) OPEN" to "inactive-set Γ
  shipped D=1 (§44.5.bis); global/contact-line Γ provably non-classical (honest limit)".

## Gates (engineer + QA spec)

### `G_OBSTACLE_GAMMA` (RELEASE_BLOCKING, EMPIRICAL_SLOPE + correctness)
- **File:** `tests/obstacle_vi_slope.rs::g_obstacle_gamma` (or a new `obstacle_gamma_slope.rs`).
- **Convergence sub-gate:** on the perpetual-American-put closed form (`K=1, r=0.05, σ=0.20`,
  `S* = γ/(γ+1)`, `γ=2r/σ²=2.5`), sample `V` analytically on a grid of the *continuation*
  region `[S*, S_max]`, run `apply_inactive_gamma_into`, and at a probe node *strictly inside*
  the continuation set (≥1 node off `x*`, e.g. `S = S* + 0.40·(S_max − S*)`) take the error
  `|Γ_h − Γ_analytic|` over a Δx-halving sweep. **OLS log-log slope in Δx ≤ −1.95** (O(Δx²)),
  mirroring `obstacle_gamma_kit.py` sub-check 3 (observed ≈2.00).
- **Refusal sub-gate (correctness):** assert that for every node `i` with `S_i ≤ S*` (active
  set) and for the one-node guard band around `x*`, `defined[i] == false`; and that at least
  one interior continuation node has `defined[i] == true`. I.e. Γ is *refused* (mask absent)
  at/inside the active set, never fabricated.
- **Threshold:** `slope ≤ −1.95` AND `refusal mask correct`. `feature_gate: slow-tests`
  (sweep cost), `blocks_release: true`, `authority: ADR-0152`, `introduced_in: v8.2.0`.

### `G_OBSTACLE_SLOPE_2D` (RELEASE_BLOCKING, EMPIRICAL_SLOPE)
- **File:** `tests/obstacle_vi_slope.rs::g_obstacle_slope_2d`.
- **Setup (mirror `G_OBSTACLE_SLOPE_SMOOTH`/`G_OBSTACLE_STATIONARY` in 2D):** D=2 inner =
  a tensor-product / `GridFnND<F,2>` diffusion (axis-separable heat), composed with a 2D
  obstacle. Two checks:
  1. **Self-convergence slope:** evolve `Vⁿ⁺¹ = Π_g(S(Δτ)Vⁿ)` on a 2D grid, sweep `n_steps`
     (Δτ halving) vs a finer reference; **OLS slope(log sup_err vs log n_steps) ≤ −0.95**
     (order-1 ceiling, projection-capped — same posture as the D=1 smooth gate).
  2. **2D stationary correctness:** a 2D stationary-membrane / smooth obstacle oracle (e.g.
     radially-symmetric `g(x,y) = A − B((x−½)² + (y−½)²)` with Dirichlet box BCs, contact set
     a disc) — sup-error vs the closed-form fixed point below a tight tolerance, mirroring
     `G_OBSTACLE_STATIONARY`. If a clean closed form is not available, use a manufactured
     stationary solution `u*` with `Π_g(S(Δτ)u*) = u*` verified at construction.
- **Threshold:** `slope ≤ −0.95` AND 2D stationary `sup_err ≤ tol`. `feature_gate: slow-tests`,
  `blocks_release: true`, `authority: ADR-0152`, `introduced_in: v8.2.0`.

### `T_OBSTACLE_GAMMA` (NORMATIVE oracle)
- **Script:** `scripts/obstacle_gamma_kit.py` (EXISTS, 5/5 PASS) — wire into the `test-fast`
  sympy sweep exactly as `T_OBSTACLE_PROJECTION` wires `verify_obstacle_projection.py`.
- **5 sub-checks (already implemented):** `closed_form` (ODE + C¹ smooth-fit + Γ jump),
  `gamma_jump` (jump > 0), `inactive_restrict` (O(Δx²) on the open set), `mollified_eps`
  (bounded + faithful on inactive set, non-classical in layer — SHOULD evidence),
  `d2_mechanical` (D-agnostic mask → 7b).
- **Engineer action:** add a `T_OBSTACLE_GAMMA PASS` print line at the end of `main()` (mirror
  `T_OBSTACLE_PROJECTION PASS`) and register the script in the `xtask test-fast` sympy list.
  `invocation: "python3 scripts/obstacle_gamma_kit.py"`, `blocks_release: true`,
  `authority: ADR-0152`, `introduced_in: v8.2.0`.

## Contract stanzas (spec)

> **SCHEMA BUMP — grep HEAD first.** The C-9 kernel is bumping `traits.yaml` /
> `properties.yaml` concurrently. DO NOT hardcode a target version. At impl time:
> `grep -n '^schema_version:' contracts/semiflow-core.traits.yaml` and
> `grep -n '^schema_version:' contracts/semiflow-core.properties.yaml`, then bump each by ONE
> MINOR over the **actual HEAD value** (both changes are ADDITIVE: new methods + new gates,
> no removal). At the time of writing HEAD is traits `4.9.0`, properties `4.10.0` — treat
> these as STALE references, not targets.

**`contracts/semiflow-core.traits.yaml`** (ADDITIVE; documented via the `#`-comment change_log
block under `schema_version`, the established mechanism — `ObstacleChernoff` has no dedicated
`- name:` type stanza, it lives in the change_log):
- New change_log entry: "vX.Y.Z (v8.2.0 B-7 obstacle Γ + D≥2 — MINOR, ADDITIVE; ADR-0152).
  `ObstacleChernoff<C,O,F>` gains a SECOND `ChernoffFunction` impl over
  `C::S = GridFnND<F, D>` (D≥2 forward evolution; the existing `GridFn1D` impl is UNCHANGED).
  `Obstacle<F>` gains D-generic `project_in_place_nd<const D>` / `active_set_nd_into<const D>`
  (default methods; `value_at(&[F])` already D-agnostic). NEW inherent primitive
  `ObstacleChernoff::apply_inactive_gamma_into(v, gamma, defined) -> Result<usize>` —
  inactive-set Γ on the OPEN continuation set, companion `&mut [bool]` refusal mask, NOT a
  `ChernoffFunction`/Greek surface, NO global C² Greek (math §44.5.bis honesty). NEW module
  `crates/semiflow-core/src/obstacle_gamma.rs` (~60 LoC, default 500 cap — sibling split of
  `obstacle.rs`, no Cohort carve-out). `apply_active_set_adjoint_into` + Γ remain D=1
  (math §44.5.ter scope note). NO existing type/trait/method changed or removed."

**`contracts/semiflow-core.properties.yaml`** (ADDITIVE; new gate stanzas mirroring the existing
obstacle gates at lines 8257–8384, plus a change_log entry):
- THREE new gate entries `G_OBSTACLE_GAMMA`, `G_OBSTACLE_SLOPE_2D`, `T_OBSTACLE_GAMMA` in the
  v8.2.0 gate block, EACH mirroring the shape of `G_OBSTACLE_SLOPE_AMERICAN` /
  `T_OBSTACLE_PROJECTION` respectively (`type`, `severity: RELEASE_BLOCKING`,
  `blocks_release: true`, `threshold`/`sub_checks`, `pass_status` filled by QA after the run,
  `rationale`, `gate`, `purpose`, `test_file`, `feature_gate`, `introduced_in: v8.2.0`,
  `authority: ADR-0152`). Field values per the Gates section above.
- New change_log entry recording the three gates + the `obstacle_gamma_kit.py` wiring into
  `test-fast`.

## Consequences

Second-order Greeks become available where they mathematically exist (the open continuation
set), with an honest, mask-carried refusal at the free boundary — closing the §44.5 OPEN item
without claiming a Greek that does not exist. Multi-asset obstacle *forward evolution* ships at
the cost of one additive impl block, validating the §44.7 D≥2 deferral as mechanical. The
single genuinely-new numerical primitive is the central-difference Γ pass with the guard-band
mask (~60 LoC). Risk is LOW: 7a is gate-backed by an exact closed form (O(Δx²) margin to
−1.95), 7b's projection path is sub-check-5 mechanical. DEFERRED to v8.x (real new work, NOT
in scope): D≥2 active-set adjoint (needs an ND `apply_adjoint_into` consumer) and D≥2 / free-
*surface* Γ (per-axis stencil + ND guard band). The mollified global-Γ route stays
out-of-core permanently per ADR-0150 (provably non-classical in the layer).

## References

ADR-0150 (research GO); ADR-0116 (projective-splitting obstacle); math.md §44.5/§44.6/§44.7;
Peskir 2005 (smooth-fit C¹); Jaillet–Lamberton–Lapeyre 1990; Casabán–Company–Egorova–Jódar
2024 (arXiv:2401.13361, inactive-set Γ region-of-interest without smoothing); Howison–
Reisinger–Witte 2013 (doi:10.1137/090776089, penalty boundary layer); Forsyth–Vetzal 2002
(spurious Γ oscillations); constitution v4.0.0 (default 500-LoC cap, sibling-split precedent
`reflection.rs`/`reflection_regions.rs`).
