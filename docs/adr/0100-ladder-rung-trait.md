# ADR-0100 — `LadderRung<const K: usize, F>` Formal Sealed Super-Trait (A.6)

- **Status**: Accepted
- **Date**: 2026-05-29
- **Wave**: v5.0 BREAKING WINDOW #3 (A.6 of 2-item release; B.1 Padé DECISION ships separately as ADR-0101; B.2 Chebyshev default REMOVED per ADR-0097 AMENDMENT 1 RED verdict — deferred to v5.1+ conditional revival).
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0073 (`ApproximationSubspace<K, F>` opt-in super-trait — A.6 super-bounds it; freeze invariant preserved), ADR-0086 + AMENDMENT 1 (Path β = Richardson over symmetric K5; K=2 base identity = `Diffusion4thChernoff`), ADR-0088 + AMENDMENT 1/2 (ladder rung population at K=4/6/8 via nested Richardson), ADR-0089 (Path ε Quintic-K5 spatial floor lift), ADR-0090 (v4.3 `Diffusion8thZeta8Chernoff` Chebyshev-base K=8 rung shipping), ADR-0025/0026 (Generic-over-Float; trait generic over `F: SemiflowFloat = f64`).
- **Supersedes / amends**: NONE. **Strictly additive** per ADR-0073 freeze invariant. The `ApproximationSubspace<K, F>` trait surface is UNCHANGED. `LadderRung<K, F>` is a NEW opt-in super-trait that super-bounds `ApproximationSubspace<K, F>` and adds the rung-to-rung structural invariant (`PREDECESSOR_K`).
- **Mathematical foundation**: math.md §36 (NEW, NORMATIVE library — sealed-sibling formal trait surface + 4-rung catalogue); CITATION Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 (the order-K tangency theorem; `LadderRung<K, F>` instances are the type-level catalogue of impls that joint-witness the K-2 → K Richardson order lift); Richardson 1911 / Romberg 1955 (Romberg-in-time extrapolation foundation); Hairer-Lubich-Wanner 2006 *Geometric Numerical Integration* §II.4 (odd-power-only error expansion of symmetric methods that justifies each rung's order lift). Reuses §26 (ApproximationSubspace semantics) verbatim — no new mathematical claim beyond formalising the existing ladder structure as a sealed typing surface.
- **Acceptance gates added**: T_LADDER_RUNG (RELEASE_BLOCKING — sympy oracle verifying the K → K-2 invariant symbolically across the 4-rung catalogue; ~60 LoC sibling to T23N_zeta6/T23N_zeta8 per ADR-0088 pattern). NO new runtime gate (typing-surface formalisation only; runtime behavior of existing rungs UNCHANGED — they ship with their own G_zeta4 / G_zeta6 / G_zeta8 gates per ADRs 0086/0088/0090).

## Context

The v3.0–v4.3 ζ-ladder shipped 4 rungs of nested Richardson semigroup approximation, each impl'ing `ApproximationSubspace<2K, F>` per ADR-0073:

| Rung K | Type                          | Order | ADR  | `ApproximationSubspace<K>` |
|--------|-------------------------------|-------|------|----------------------------|
| 2      | `Diffusion4thChernoff<F>`     | 2     | 0013 | impl `<2, F>` (palindromic Strang K5 base) |
| 4      | `Diffusion4thZeta4Chernoff<F>`| 4     | 0086 | impl `<4, F>` (Richardson on K=2 base)     |
| 6      | `Diffusion6thZeta6Chernoff<F>`| 6     | 0088 W-I | impl `<6, F>` (Richardson on K=4)      |
| 8      | `Diffusion8thZeta8Chernoff<F>`| 8     | 0090 | impl `<8, F>` (Chebyshev-base K=8 rung)    |

The ladder structure (each rung K is Richardson over the K-2 rung) is currently encoded only in **prose comments** + per-rung constructor parameter types (`zeta4: Diffusion4thZeta4Chernoff<F>` accepts the K=4 inner; `zeta6: Diffusion6thZeta6Chernoff<F>` accepts the K=6 inner). There is no type-level surface that declares "this type is a formal rung of the ζ-ladder at index K, with predecessor at K-2", and no way for a downstream caller (e.g., a generic Richardson driver, a documentation generator, or a future ADR-0101 Padé Wave bench harness) to enumerate the ladder catalogue at compile time.

Per `~/.claude/plans/roadmap-reflective-biscuit.md` §v5.0, v5.0 A.6 ships the formal sealed typing surface that unifies these 4 rungs without reshaping `ApproximationSubspace<K, F>` (ADR-0073 freeze). This closes the long-deferred A.6 item ("Higher-order ζ⁸ formal `ApproximationSubspace<K>` ladder") originally scoped for v3.0 and bumped through v3.1/v4.0/v4.5 (4-deferral cycle — see ADR-0086 precedent). The `LadderRung<K, F>` trait is the v5.0 typing-surface completion of the ζ-ladder; B.1 Padé DECISION (ADR-0101) optionally adds Path α/β/γ as a *separate* approximation family (NOT a ζ-rung — Padé operates by scaling-and-squaring on the matrix exponential, not Richardson on a symmetric base), so the `LadderRung<K, F>` trait MUST NOT subsume Padé.

## Decision

Ship **one additive public-surface item** in v5.0 (super-bound only; NO reshape of `ApproximationSubspace<K, F>`):

```rust
/// Sealed sibling super-trait of `ApproximationSubspace<K, F>` codifying
/// formal membership in the ζ-ladder of nested-Richardson Chernoff rungs.
///
/// Each `LadderRung<K, F>` instance asserts:
///   - The kernel is order-K Chernoff-tangent on D(A^K) (via `ApproximationSubspace<K, F>` super-bound)
///   - The rung's algorithmic predecessor (`PREDECESSOR_K`) is either Some(K-2) for ladder rungs K ≥ 4,
///     or None for the K=2 base (Diffusion4thChernoff palindromic Strang K5 — ADR-0086 §"Decision")
///   - The (K=2K-1)-th Richardson combination on the predecessor closes the K-2 → K order lift
///     per Hairer-Lubich-Wanner 2006 §II.4 odd-power error expansion of symmetric methods
///
/// Sealed via the `sealed::Sealed` private marker (mirror rustls / thiserror pattern):
/// downstream crates CANNOT impl `LadderRung<K, F>` without going through ADR-0100 +
/// PRE-FLIGHT sympy verification of the K → K-2 invariant. This preserves the formal
/// catalogue (4 rungs in v5.0; future rungs only via ADR-0090-style ADR + impl block
/// inside `semiflow-core`).
///
/// ## v5.0 catalogue
///
/// - `LadderRung<2, F>` for `Diffusion4thChernoff<F>` (K=2 base; PREDECESSOR_K = None)
/// - `LadderRung<4, F>` for `Diffusion4thZeta4Chernoff<F>` (PREDECESSOR_K = Some(2))
/// - `LadderRung<6, F>` for `Diffusion6thZeta6Chernoff<F>` (PREDECESSOR_K = Some(4))
/// - `LadderRung<8, F>` for `Diffusion8thZeta8Chernoff<F>` (PREDECESSOR_K = Some(6))
///
/// ## References
///
/// - ADR-0073 — `ApproximationSubspace<K, F>` super-bounded by this trait.
/// - ADR-0086/0088/0090 — populate the catalogue at K=2/4/6/8 respectively.
/// - math.md §36 — NORMATIVE typing surface spec.
pub trait LadderRung<const K: usize, F: SemiflowFloat = f64>:
    ApproximationSubspace<K, F> + sealed::Sealed
{
    /// Predecessor rung index, or `None` for the K=2 ladder base.
    ///
    /// `Some(K - 2)` for every rung K ≥ 4; the K → K-2 invariant is verified
    /// symbolically by the T_LADDER_RUNG sympy oracle at compile-time-equivalent
    /// (build-time) per AC6.
    const PREDECESSOR_K: Option<usize>;
}

mod sealed {
    pub trait Sealed {}
    impl<F: crate::float::SemiflowFloat> Sealed for crate::diffusion4::Diffusion4thChernoff<F> {}
    impl<F: crate::float::SemiflowFloat> Sealed for crate::diffusion4_zeta4::Diffusion4thZeta4Chernoff<F> {}
    impl<F: crate::float::SemiflowFloat> Sealed for crate::diffusion6_zeta6::Diffusion6thZeta6Chernoff<F> {}
    impl<F: crate::float::SemiflowFloat> Sealed for crate::diffusion8_zeta8::Diffusion8thZeta8Chernoff<F> {}
}
```

File layout: addition to `crates/semiflow-core/src/approximation.rs` (~150 LoC additive — current file 303 LoC; HARD LIMIT 500 LoC default cap with 47 LoC headroom; NO Override #1 expansion required). The 4 sealed impls live in `approximation.rs` (not co-located with the kernel types) for the same reason ADR-0073 chose this location: the witness logic is independent of the kernel's `apply_into`, and future rungs accumulate in one place.

Schema bumps: `traits.yaml` 2.0.0 → **2.1.0** (MINOR — adds `LadderRung<K, F>` opt-in super-trait, NO breaking change to existing trait surface); `properties.yaml` 1.0.0 → 1.1.0 (NEW gate T_LADDER_RUNG, NORMATIVE rationale = K → K-2 invariant verification). math.md is append-only (§36 NEW).

## Rationale (Decisions A/B/C)

- **Decision A: Option (i) ADDITIVE ONLY (chosen)** — `LadderRung<K, F>: ApproximationSubspace<K, F>` is a super-bound; NO reshape of the v3.0 ADR-0073 trait surface. The four existing `impl ApproximationSubspace<K, f64>` blocks at K=2/4/6/8 are UNCHANGED; they each gain a sibling `impl LadderRung<K, f64>` block declaring the K-2 predecessor. Option (ii) reshape (forcing `ApproximationSubspace<K, F>` to require ladder semantics) is REJECTED because (a) it would break the non-rung opt-in impls already shipped — `DiffusionChernoff<F>` impls `<2, F>` but is NOT a ζ-ladder rung (it's the v0.3.0 ζ-A τ²-correction kernel, semantically orthogonal per ADR-0088 cross-reference list), and similarly `TruncatedExp4thDiffusionChernoff` impls `<6, F>` as the ζ⁴ correction witness consumer, NOT a Richardson rung; (b) it would violate the ADR-0073 freeze invariant ("strictly additive on the public surface; existing `ChernoffFunction<F>` impls compile UNCHANGED") which the v3.0 BREAKING window deliberately froze; (c) per R5.0-3 (`roadmap-reflective-biscuit.md` §v5.0 contingency), reshape forces v6.0 deferral — there is no v5.0 reshape allowance. Option (i) is the *only* design that ships A.6 at v5.0 within the freeze.

- **Decision B: Option (c) NO type-level rung inheritance + `const PREDECESSOR_K: Option<usize>` (chosen)** — the K → K-2 relationship is expressed as a per-impl `const PREDECESSOR_K: Option<usize>` associated constant, verifiable at build-time by the T_LADDER_RUNG sympy oracle (AC6). Option (a) (where-clause `LadderRung<K, F>: LadderRung<K - 2, F>`) is REJECTED because `K - 2` const-arithmetic in trait bounds requires `feature(generic_const_exprs)` — nightly only as of MSRV 1.78, same blocker that drove ADR-0088 Option β rejection (recursive type-parameter rejected for identical reasons). Option (b) (associated type `type Predecessor: LadderRung<?, F>`) is REJECTED because GAT in trait bounds breaks `dyn ApproximationSubspace<K, F>` trait-object dispatch in stable Rust until ≥1.85 (same blocker ADR-0073 §"Alternatives considered" cited for rejecting associated-type design on `ApproximationSubspace`). The simple associated-const Option (c) preserves stable-MSRV compilation, preserves trait-object dispatch on `ApproximationSubspace<K, F>` (since `LadderRung<K, F>` super-bounds it without GAT machinery), and gives the same compile-time-equivalent K-2 invariant verification through the sympy oracle. Suckless minimal-surface choice (associated-const is one Rust language feature; recursive const-generic where-clauses are three).

- **Decision C: Option (x) YES — K=2 is the ladder base (chosen)** — `LadderRung<2, F> for Diffusion4thChernoff<F>` ships with `PREDECESSOR_K = None`. Rationale: the ζ-ladder per ADR-0088 §"Algorithm" starts at `R¹(τ) := K5(τ)` (order 2, the palindromic Strang K5 base from ADR-0086 AMENDMENT 1); all higher Richardson levels (`R²`, `R³`, `R⁴`) inductively build on it. Excluding K=2 (Option (y)) would orphan the K5 base from its own ladder and force the catalogue to begin at K=4, which would (a) misrepresent the algorithm (Richardson on K=2 → K=4 cancellation is the FIRST rung, not the base); (b) leave `Diffusion4thChernoff` without a formal LadderRung witness despite its central role as the K=2 base in every downstream rung's constructor (`Diffusion4thZeta4Chernoff::new(k5: Diffusion4thChernoff<F>, ...)` per ADR-0086). The `PREDECESSOR_K = None` sentinel cleanly distinguishes the base case from inductive rungs; T_LADDER_RUNG AC6 verifies this is the unique base (exactly one rung in the catalogue has `PREDECESSOR_K == None`).

- **Why sealed (`sealed::Sealed` private marker)?** The ζ-ladder catalogue is a closed structural invariant verified by T_LADDER_RUNG sympy oracle. Allowing downstream crates to add `impl LadderRung<K, F> for MyKernel<F>` would (a) bypass the sympy verification of the K → K-2 invariant (catalogue integrity hole), (b) allow false catalogue claims (a downstream impl could declare `PREDECESSOR_K = Some(4)` without actually being Richardson on a K=4 rung). Sealing via the `mod sealed` private marker is the established Rust pattern (rustls, thiserror, hyper use this); it preserves "additivity" (any kernel can still impl `ApproximationSubspace<K, F>` per ADR-0073 — only `LadderRung<K, F>` is sealed) and matches the "this is a curated catalogue, not an open extension point" semantics. New rungs (e.g., a hypothetical K=10 ζ¹⁰ at v5.x) require ADR + impl block inside `semiflow-core` — same gating that already governs every other ladder rung.

- **Why NOT a runtime `assert_ladder_invariant<K>(rung)` helper?** Build-time T_LADDER_RUNG sympy oracle verification is strictly stronger than runtime: it catches catalogue regression at CI time, before any binary ships. A runtime helper would add code-size + runtime cost for zero compile-time-equivalent safety win. The sympy oracle is the suckless choice.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Option (ii) reshape `ApproximationSubspace<K, F>` to require ladder semantics | Breaks 4+ non-rung opt-in impls (DiffusionChernoff `<2>`, TruncatedExp4thDiffusion `<6>`, KolmogorovHypoelliptic `<2>`) shipped under ADR-0073 freeze; violates the v3.0 BREAKING-window freeze invariant; per R5.0-3 forces v6.0 deferral. |
| Option (a) const-generic where-clause `LadderRung<K, F>: LadderRung<K - 2, F>` | Requires `feature(generic_const_exprs)` — nightly only as of MSRV 1.78. Same blocker that drove ADR-0088 Option β rejection (recursive type-param). |
| Option (b) associated-type `type Predecessor: LadderRung<?, F>` (GAT) | Breaks `dyn ApproximationSubspace<K, F>` trait-object dispatch in stable Rust until ≥1.85; same blocker ADR-0073 cited. |
| Option (y) ladder starts at K=4 (skip K=2 base) | Orphans the K5 base from its own ladder; misrepresents the algorithm (Richardson on K=2 base is the FIRST rung lift, not the base); leaves `Diffusion4thChernoff` without a formal LadderRung witness. |
| NON-sealed `LadderRung<K, F>` (downstream-extensible) | Allows downstream catalogue claims that bypass T_LADDER_RUNG verification of the K → K-2 invariant; catalogue integrity hole. |
| Documentation-only catalogue (no trait at all) | Fails to give downstream callers (Padé bench harness, doc generators, future generic Richardson driver) a typed surface to enumerate the ladder. The v5.0 A.6 closure rationale explicitly names "typing surface unification". |
| Free function `enumerate_ladder_rungs() -> &'static [&'static dyn Any]` | Type-erased; loses the per-K static dispatch and the const-evaluated K-2 invariant. |
| Subsume Padé under `LadderRung<K, F>` | Padé operates by scaling-and-squaring on the matrix exponential, NOT by Richardson on a symmetric base; the K → K-2 invariant does not apply. Padé is a separate approximation family per ADR-0101 (B.1 DECISION). |

## Consequences

- **POSITIVE**: closes A.6 (formal `LadderRung<K, F>` trait), unblocks v5.x downstream consumers (Padé bench harness at ADR-0101 conditional, future ADR-0101+ generic Richardson driver, doc generators); preserves ADR-0073 freeze (strictly additive); preserves trait-object dispatch on `ApproximationSubspace<K, F>`; closes the long-deferred A.6 item without scope creep (no new mathematical claim, no new runtime behavior, no algorithm change to existing rungs); the sealed catalogue gives a curated invariant-verified 4-rung surface that future rungs (K=10, K=12, etc.) can be added to via ADR + impl block within `semiflow-core` (~30 LoC per new rung).
- **NEUTRAL**: per-rung addition cost ~10 LoC (one `impl LadderRung<K, F>` block + one `impl Sealed` line); the 4 existing rungs add ~40 LoC total to `approximation.rs`. T_LADDER_RUNG sympy oracle (~60 LoC) is a sibling of T23N_zeta6/T23N_zeta8 per ADR-0088 pattern. No runtime cost.
- **NEGATIVE**: introduces `sealed::Sealed` private marker (one new mod in `approximation.rs`); downstream crates cannot impl `LadderRung<K, F>` without sending a PR to `semiflow-core` (this is INTENTIONAL — catalogue integrity).
- **BREAKING**: NONE. `LadderRung<K, F>` is strictly additive; `ApproximationSubspace<K, F>` surface is UNCHANGED; existing 4 rungs gain a sibling impl block without touching their `ChernoffFunction<F>` or `ApproximationSubspace<K, F>` impls.
- **Schema bumps**: `traits.yaml` 2.0.0 → 2.1.0 (MINOR — new opt-in trait); `properties.yaml` 1.0.0 → 1.1.0 (NEW gate T_LADDER_RUNG). `math.md` append-only (§36 NEW).
- **Constitution unchanged**: ~150 LoC addition to `approximation.rs` (303 LoC → ~450 LoC), well under the default 500-LoC cap with 50 LoC headroom; NO Override #1 file-list expansion. Override count remains 3/3.

## Implementation cost

| Item | LoC | Days |
|---|---|---|
| `LadderRung<K, F>` trait + `sealed::Sealed` private mod | ~30 | 0.5 |
| 4 `impl LadderRung<K, F>` blocks at K=2/4/6/8 + 4 `impl Sealed` lines | ~40 | 0.5 |
| T_LADDER_RUNG sympy oracle (`scripts/verify_ladder_rung.py`) | ~60 | 0.5 |
| `properties.yaml` T_LADDER_RUNG entry + `traits.yaml` 2.1.0 MINOR bump | ~30 | 0.25 |
| Doc updates (rustdoc, math.md §36 reference) | ~20 | 0.25 |
| **Total** | **~180** | **2** |

Conservative engineer estimate: **2-3 days** (single Wave; no Wave I/II split since the catalogue is fixed at 4 rungs). Validated against the ~150 LoC budget from `roadmap-reflective-biscuit.md` §v5.0 A.6 — within budget.

## References

- ADR-0073 — `ApproximationSubspace<K, F>` opt-in super-trait (super-bounded by this ADR; freeze invariant preserved).
- ADR-0086 + AMENDMENT 1 — Path β Richardson on K5; establishes `Diffusion4thChernoff` as the K=2 ladder base.
- ADR-0088 + AMENDMENT 1/2 — ζ⁶ + ζ⁸ ladder rungs; this ADR formalises the algorithm-level ladder as a typing surface.
- ADR-0089 + AMENDMENT 1 — Path ε Quintic-K5 spatial floor lift; preserved by additivity.
- ADR-0090 — v4.3 ζ⁸ Chebyshev-base shipping; populates the K=8 rung.
- ADR-0101 — B.1 Padé DECISION (v5.0 sibling release); Padé is NOT a `LadderRung<K, F>` (different approximation family).
- math.md §26 — `ApproximationSubspace<K, F>` semantics (reused verbatim).
- math.md §36 (NEW v5.0) — `LadderRung<K, F>` formal typing surface + 4-rung catalogue + K → K-2 invariant.
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v5.0 A.6 — release-level scope for this ADR.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_v4_5_research_wave.md` §"Architectural patterns" Pattern #2 — Direct K5 wiring; informs the K=2 base catalogue entry.

## Amendments

(none at acceptance time)
