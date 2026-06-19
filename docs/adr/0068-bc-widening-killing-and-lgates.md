# ADR-0068 — BoundaryPolicy widening, B3 Dirichlet via killing, L-gate infrastructure

- **Status**: Accepted
- **Date**: 2026-05-26
- **Wave**: v2.6 (additive infrastructure release; no math pillars)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0007 (existing 4-variant `BoundaryPolicy`), ADR-0025 (Generic-over-Float with `F = f64` default), ADR-0026 (`ChernoffFunction` trait generic over `F`), ADR-0028 (FFI/PyO3/WASM bindings — relevant only for the L-gate output JSONL schema parity), ADR-0066 (`tracking-alloc` feature), ADR-0067 (`latency_tail.rs` example — the canonical v2.5.1 latency baseline).
- **Supersedes / amends**: none — purely additive on the contract surface.
- **Mathematical foundation**: §3.5.bis (boundary-policy widening, NORMATIVE library), §3.6.bis (L-gate semantics, NORMATIVE acceptance gate), §21 (operator-level Dirichlet via Feynman–Kac killing; CITATION Butko 2018 *Fract. Calc. Appl. Anal.* 21).

## Context

The v2.6 release bundles three loosely-coupled but jointly-motivated tracks. Each on its own does not justify a dedicated ADR; together they form the **infrastructure layer** for the v2.7–v2.8 math pillars (resolvent, Howland, manifold, image-method Neumann), which all need at least one of {boundary-condition vocabulary, operator-killing primitive, latency floor harness}.

1. **BoundaryPolicy is too narrow.** The 4 v0.2.1 variants (`Reflect`, `ZeroExtend`, `Periodic`, `LinearExtrapolate`) cover stencil out-of-range extension. They do **not** express two physical conditions that v2.7+ semigroups (in particular B4 reflection / B3 killing / A4 manifold) need to consume from user code: **`Dirichlet { value }`** (fixed-value boundary, e.g., absorbing-wall heat equation, vanishing PDF) and **`Neumann`** (zero-flux boundary, e.g., insulated wall, reflecting random walk). v2.6 widens the enum **additively** so the v2.7+ pillars do not have to ship an enum widening at the same time as the math.
2. **Operator-level Dirichlet (killing) is a primitive, not a stencil hack.** Stencil-level Dirichlet (via the new enum variant in track 1) is a *boundary-value-substitution* policy on the interpolant: it answers *"what value should I read at index `j ∉ [0, n)`?"* with a fixed scalar. **Operator-level Dirichlet** is a different mathematical object: it answers *"what is the semigroup that absorbs trajectories on `∂R`?"* via the Feynman–Kac killing functional `𝟙_R(x) · (S_n f)(x)`. Both are needed; they ship together in v2.6 with a dual-policy section in `math.md §3.5.bis` to prevent confusion. The killing wrapper is the **Tier-B item** in the roadmap and the math content in this release.
3. **Latency floors need a contract before they can be CI-enforced.** ADR-0067 measured RC N=1536 CEV per-tick latency at p99.9 = 45 ns on `i7-12700K` and reported it in the release notes. There is currently no machine-readable contract that asserts this number, no `--hardware-profile` mechanism that distinguishes the 45 ns observation from an `aws-c7g-large` measurement, and no `xtask` command that re-runs the latency bench and gates on the floors. v2.6 ships the L-gate **schema** (in `properties.yaml`) and the `xtask latency-gate` subcommand in **advisory mode** (exit 0 with warnings). v2.7 promotes the first L-gate (`L_CEV_PTICK_P999`) to release-blocking once the multi-host profile table has two entries.

## Decision

Three additive contracts ship together in v2.6:

- **Track 1 — `BoundaryPolicy` widening**: extend the existing public enum with two new variants — `Dirichlet { value: F }` (parameterized) and `Neumann` (unit). The enum becomes `BoundaryPolicy<F = f64>` (generic with default), mirroring the ADR-0025 generic-over-float defaulting pattern; all existing call-sites compile unchanged because they elide the type parameter. Existing variants are byte-identical to v2.5.1. Math.md §3.5.bis (NEW) distinguishes stencil-BC (this enum) from operator-killing (track 3) and gives the continuous + discrete dispatch for both new variants. ADR amendment to Override #1 (file-cap carve-out for `grid.rs`): **the enum and its `bc_*` helpers MOVE to a new `crates/semiflow-core/src/boundary.rs` module** (≤400 LoC target) rather than expanding the 715-LoC `grid.rs` carve-out further. This is a structurally cleaner outcome than the alternative — extracting BoundaryPolicy was already a latent reviewer suggestion at v2.0 — and it RELIEVES override-budget pressure ahead of v2.8 (`manifold.rs` 700-LoC carve-out planned).
- **Track 2 — L-gate harness**: new top-level `latency_gates:` section in `properties.yaml` (schema mirrors existing `properties:` list shape: id, severity, gate, purpose, test_file, feature_gate, plus L-specific fields `hardware_profile`, `percentile_budgets_ns`, `replication`). Initial entry `L_CEV_PTICK` populated for `i7-12700K` profile, all other profiles advisory placeholders. New `xtask latency-gate` subcommand reads the section, runs the designated bench, parses HDR JSONL output, asserts floors with warning (v2.6 advisory) → blocking (v2.7). Math.md §3.6.bis (NEW) defines L-gate semantics normatively.
- **Track 3 — B3 Dirichlet via killing**: new file `crates/semiflow-core/src/killing.rs` (~350 LoC, NEW Override #1 carve-out *not* required — fits the unmodified 500 LoC cap with headroom). Generic `KillingChernoff<C, R>` wraps any `ChernoffFunction<F>` with a `KillingRegion<F>` indicator (post-multiply each Chernoff step by `𝟙_R(x)`). Trait `KillingRegion<F>` with two shipped impls: `BoxRegion<F, const D>` and `BallRegion<F, const D>`. New gates G23 (eigenmode convergence slope ≤ −0.95, order-1 globally per Butko 2018 §3) and T18N (sympy symbolic identity for first 4 sin-eigenmodes on `[0,1]`). Sympy script `scripts/verify_killing_dirichlet.py` ships with the engineering wave.

Schema-version bump: `properties.yaml` advances `0.7.6 → 0.8.0` (MINOR-style: new top-level `latency_gates:` section is a schema addition, not a breaking change to existing `properties:` entries). `traits.yaml` advances `0.8.0 → 0.9.0` (new public surface: `KillingChernoff`, `KillingRegion`, `BoxRegion`, `BallRegion`, `HdrSnapshot`, `BoundaryPolicy::Dirichlet`, `BoundaryPolicy::Neumann`). `math.md` is append-only; no version field. The bump is justified because both files gain new top-level / new-public-trait surface — a downstream contract-parser MUST distinguish v2.5.1 from v2.6 to know which sections to expect.

## Rationale

### Track 1 (BoundaryPolicy widening)

- **Why additive enum, not new types?** A new top-level type (e.g., `BoundaryCondition<F>` separate from `BoundaryPolicy`) would force every internal call-site in `diffusion.rs`, `truncated_exp.rs`, `nonseparable_mixed.rs` etc. to branch on TWO enums or be re-keyed on a new trait. The 4 existing variants and the 2 new ones share the same dispatch shape (`bc_index` → `BoundaryHit` → `bc_value`), so a single widened enum is the minimal-diff design.
- **Why `Dirichlet { value: F }` parameterized?** `Dirichlet { value: 0.0 }` is a degenerate case — most real users want `Dirichlet { value: f(x_min) }` (clamped to initial-condition boundary value) or `Dirichlet { value: g(t) }` (time-dependent absorbing wall, handled by re-constructing the policy each step). The non-parameterized alternative (`Dirichlet`, no value, always zero) would force a second variant `DirichletConst { value: F }` later; ship the parameterized form once.
- **Why `Neumann` unit (no parameter)?** Zero-flux is the dominant Neumann case (insulated wall, reflecting random walk on `[a, b]`); non-zero-flux Neumann would be `Robin` (mixed). Robin is C5 in the roadmap and is deferred. Shipping `Neumann` as unit now matches user expectations and leaves `Robin { alpha: F, beta: F, value: F }` available as a future ADR.
- **Why move to `boundary.rs`?** `grid.rs` is at 715/715 LoC against its per-file carve-out cap. Adding the enum widening + dispatch + rustdoc would push it past 720 LoC, triggering another constitution PATCH amendment. Extracting `BoundaryPolicy`, `BoundaryHit`, `bc_index`, `bc_value`, `bc_value_generic`, `reflect_index` to a new `boundary.rs` (which is what they were always logically grouped as) shrinks `grid.rs` to ~470 LoC (well under the 500 cap) and gives the new module ~245 LoC headroom for the two new variants + their tests. This is the **cheaper** path against constitution overrides than expanding the carve-out for the third time.

### Track 2 (L-gate harness)

- **Why advisory in v2.6, blocking in v2.7?** L-gate floors are host-dependent: 45 ns p99.9 on `i7-12700K` may be 38 ns on a future Zen5 part and 90 ns on a power-constrained `aws-c7g-large`. The v2.6 advisory mode lets us collect floors on additional hardware profiles before any CI green-light depends on them. The v2.7 promotion is itself a contract change (severity bump in `properties.yaml`) and gets its own ADR.
- **Why a separate `latency_gates:` section, not entries inside `properties:`?** L-gates differ from `properties:` entries in two ways: (1) they require a `hardware_profile` field (existing G-gates assume any reasonable CPU + RAM); (2) they assert **percentile budgets** in nanoseconds, not slope or sup-norm. Mixing the two would force the existing `properties:` schema to grow `hardware_profile` and `percentile_budgets_ns` as optional fields on every entry — strictly worse than a parallel section.
- **Why HDR JSONL output for the bench?** The v2.5.1 `latency_tail.rs` already emits `{rep, p50_ns, p99_ns, p999_ns, p9999_ns, ...}` JSONL (one record per replication). The L-gate harness just parses this output. No format invention.
- **Why NIST nearest-rank (ASTM E29-13 §6) instead of HDR log-bucket histograms?** `latency_tail.rs` already uses `nearest-rank` on sorted data (lines 306–310). Log-buckets (e.g., the `hdrhistogram` crate, ~2.5k LoC + 3 deps) would force a fourth direct dep on `semiflow-core` (currently ≤3 budget). For per-tick counts ≤ 1M and known bounded latency range (sub-microsecond to milliseconds), nearest-rank is sufficient and exact. The `HdrSnapshot` public API is named "HDR" by convention; the implementation is array-backed nearest-rank.

### Track 3 (B3 Dirichlet via killing)

- **Why post-multiply (`𝟙_R · C.apply`) instead of pre-multiply?** Butko 2018 §3 derives the Feynman–Kac killing semigroup as `(P^R_t f)(x) = E^x[f(X_t) · 𝟙_{τ_R > t}]` where `τ_R` is the first exit time. The Chernoff approximation of this is `(F_n(τ) f)(x) := 𝟙_R(x) · (C(τ) f)(x)` — pre-applying the indicator would zero out interior points whose Chernoff stencil reaches into `R^c` (over-killing). Post-applying preserves Chernoff's order-1 consistency on the unrestricted operator and adds a single O(τ) commutator term for the killing.
- **Why `KillingRegion` as a trait, not a `Fn(&[F; D]) -> bool`?** Concrete impls (`BoxRegion`, `BallRegion`) ship `is_inside`, `is_inside_into` (batch SIMD-friendly), and `volume_estimate` (for future use in oracle calibration). A bare `Fn` would forbid the batch path; a trait gives the engineer a clean specialization surface.
- **Why two shipped regions only?** `BoxRegion` covers the eigenmode oracle on `[0, 1]ᴰ`. `BallRegion` covers the manifold-pillar prep (v2.8 A4 needs spherical caps). Half-spaces, polyhedra, and user-defined `Fn` adapters are deferred to v2.8.
- **Why order-1 globally?** Butko 2018 §3.2 proves that the killing semigroup convergence rate is dominated by the O(τ) commutator `[L, 𝟙_R]`. The order-2 splittings (Strang, ζ-A) do not lift this without an order-2 killing correction (an open research direction). v2.6 ships the order-1 contract; v2.9+ may revisit.
- **Why eigenmode oracle on `[0, 1]`?** Heat equation on `[0,1]` with absorbing boundary has a closed-form eigenfunction expansion `u(t,x) = Σ_k a_k sin(kπx) exp(-(kπ)² · t / 2)` — exact for arbitrary smooth initial data with sufficient mode-decay. The first 8 modes give a sup-norm reference that the killing chernoff can be slope-tested against (G23). Sympy verifies the eigenfunction expansion symbolically (T18N: validates the first 4 modes against `∂_t = ½∂_xx`, `u(t, 0) = u(t, 1) = 0`).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Don't widen `BoundaryPolicy`; users supply `Dirichlet`/`Neumann` via a separate `BoundaryCondition<F>` enum on a per-call basis | Forces every Chernoff impl to take TWO enums; doubles the dispatch tables; user-error-prone (mismatched policy + BC pairs). |
| Add `Dirichlet { value: f64 }` (non-generic) to keep enum non-generic | Lossy for `F = f32` and future `f128`; would require an awkward `From<F> for f64` constraint in the variant or a second `Dirichlet_f64` variant. |
| Ship `Robin { alpha, beta, value }` instead of `Neumann` (more general) | Robin is C5 (research-priority — needs a paper-track ADR). Zero-flux Neumann covers the v2.7+ B4 image-method use case; ship narrow now, widen later. |
| Don't extract to `boundary.rs`; expand `grid.rs` carve-out to 750 LoC | Burns constitution-override budget. `grid.rs` already has the `InterpKind`/`Grid1D` cluster; `BoundaryPolicy` is logically a separate adapter. Extraction is the suckless answer. |
| Use the `hdrhistogram` crate for `HdrSnapshot` | Adds a 4th direct dep on `semiflow-core` (over the ≤3 cap). Not worth it for arrays of ≤1M i64. Nearest-rank is exact and ~30 LoC. |
| Defer L-gate to v2.7 (skip v2.6 advisory phase) | Would require shipping schema + tooling + multi-host floor table + CI integration in one wave — too much. The advisory phase exists precisely to de-risk the v2.7 promotion. |
| Killing-via-pre-multiply (`𝟙_R · f` before Chernoff step) | Over-kills the interior — Chernoff stencil width may reach into `R^c` and zero-multiply destroys consistency. Post-multiply is the Feynman–Kac form. |
| Skip the `KillingRegion` trait; take a closure `Fn(&[F; D]) -> bool` | Forbids batch SIMD paths; closure dispatch overhead per cell at hot-loop scale. Trait + 2 concrete impls is the suckless answer. |
| Use NIST nearest-rank without naming it "HDR" | Convention loss — every benchmarking literature reader expects "HDR" for tail-latency snapshots; the inner implementation is private. |

## Consequences

- **Pre-existing call-sites compile unchanged.** All existing uses of `BoundaryPolicy` elide the new type parameter (`BoundaryPolicy` resolves to `BoundaryPolicy<f64>`). No version-2 migration burden.
- **`grid.rs` shrinks ~715 → ~470 LoC** after extraction; `boundary.rs` is new at ~470–500 LoC including widening; `killing.rs` is new at ~350 LoC; `hdr.rs` is new at ~120 LoC (impl ships in the engineer wave). Net: 3 new files, 1 carve-out file shrinks back below cap, no override expansion needed.
- **Constitution Override #1 file-list amendment**: `grid.rs` per-file cap REVERTS from 715 → 700 (joins the default for the remaining 4 carve-out files); `boundary.rs` does NOT need a carve-out (estimated 470–500 LoC, within the 500 cap). Recorded as a `v1.6.2 PATCH` amendment in `.dev-docs/constitution.md` at the same commit as the engineer wave.
- **`xtask` gains a new subcommand** (`latency-gate`), adding ~200 LoC to `xtask/src/main.rs` (already at 4 dispatch arms — small). Advisory in v2.6, becomes blocking in v2.7 via a properties.yaml `severity` bump.
- **New sympy script** (`scripts/verify_killing_dirichlet.py`) ships in the engineer wave (~150 LoC); becomes part of the `test-fast` sympy-gate sweep alongside `verify_v2_2_*.py`.
- **No change to existing gates.** G1..G22, T1N..T17N, all `STRANG*_BIT_EQUAL` and SIMD gates are unaffected. The L-gate is opt-in advisory; the G23 + T18N gates are new (release-blocking on v2.6 itself, but only assert the new killing module).
- **CITATIONs added to math.md §21**: Butko 2018 *Fract. Calc. Appl. Anal.* 21:5 — Chernoff approximation of killed Feynman–Kac semigroups; Chernoff 1968 (already cited) for the underlying theorem.

## Migration

None. v2.6 is a strict additive minor release:

- v2.5.1 binaries / crates link against v2.6 without recompilation (FFI/PyO3/WASM ABI unchanged — new enum variants are not exposed across the FFI boundary in v2.6; the C bindings keep their existing 4-variant `smf_boundary_policy_e` enum).
- v2.5.1 source code compiles against v2.6 without modification (generic-over-F default keeps existing `BoundaryPolicy` mentions as `BoundaryPolicy<f64>`).
- The new `KillingChernoff` / `KillingRegion` / `BoxRegion` / `BallRegion` types are opt-in. No existing Chernoff impl is touched.

## Cross-references

- ADR-0007 — established the original 4-variant `BoundaryPolicy` enum that this ADR widens.
- ADR-0025 — Generic-over-Float defaulting pattern (`F = f64`) that this ADR reuses for `BoundaryPolicy<F = f64>`.
- ADR-0026 — `ChernoffFunction<F>` trait generic over `F`; `KillingChernoff<C, R>` is a generic wrapper preserving this contract.
- ADR-0066 — `tracking-alloc` feature; L-gate harness uses `--measure` flag to assert zero per-tick allocations.
- ADR-0067 — `latency_tail.rs` example; L-gate `L_CEV_PTICK` re-uses its JSONL output schema.
- `.dev-docs/constitution.md` v1.6.1 → v1.6.2 (PATCH): `grid.rs` carve-out 715 → 700 (reverts after extraction); no new override added.
- math.md §3.5 (existing) — boundary-policy continuous extensions; §3.5.bis extends with Dirichlet + Neumann.
- math.md §3.6 (currently absent — `§3.6` slot is reserved for `bc_value` dispatch helper docs; §3.6.bis ships now as the L-gate normative definition; the original `§3.6` slot remains free for a future formal `bc_value` spec).
- math.md §21 (NEW) — operator-level Dirichlet via Feynman–Kac killing.

## Amendments

(none at acceptance time)
