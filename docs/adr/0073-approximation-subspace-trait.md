# ADR-0073 — `ApproximationSubspace<const K: usize, F>` Marker Trait (B1)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v3.0 (B1 — first BREAKING window of the academic-priority trajectory; opt-in marker trait shipped alongside the ChernoffFunction trait cleanup of ADR-0074 and the ζ⁴ correction of ADR-0075)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float with `F = f64` default), ADR-0026 (`ChernoffFunction<F>` trait generic over `F`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0070 (v2.7 `TimedChernoffFunction<F>` super-trait — same opt-in additive pattern; this ADR generalises that mechanism to K-jet structure).
- **Supersedes / amends**: none — strictly additive on the public surface. Establishes a NEW marker trait `ApproximationSubspace<const K: usize, F>` as an OPT-IN super-trait of `ChernoffFunction<F>` (Design Option C per `roadmap-reflective-biscuit.md` §v3.0). Existing `ChernoffFunction<F>` impls compile UNCHANGED; impls that wish to expose order-K convergence witnesses opt in by writing one or more `impl<F: SemiflowFloat> ApproximationSubspace<K, F> for X<F> { ... }` blocks.
- **Mathematical foundation**: math.md §26 (NORMATIVE library — `ApproximationSubspace<K, F>` semantics; CITATION Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — the order-K Chernoff tangency theorem; Example 4.2 — the multi-K co-witness construction; Vedenin-Smolyanov-Voskresenskaya 2020 *Math. Notes* — the original characterisation of D(A^K) Chernoff approximation cores).
- **Acceptance gates added**: G_AS_K (RELEASE_BLOCKING — per-K=2,4,6 symbolic in-subspace witness; the K=2 instance underwrites the v2.x compatibility shim's order-2 floor, K=4 underwrites the ζ⁴ correction of ADR-0075, K=6 underwrites the v3.1 Hörmander-track and v4.0 B8 order-6/8 expansion).

## Context

For a Chernoff function $F(\tau)$ approximating $\exp(\tau A)$, the *consistency order* $m$ in the bound $\|F(\tau) f - \exp(\tau A) f\| \le K \tau^{m+1}$ holds only on a **dense core** — typically $f \in D(A^m)$ or a refinement thereof. The order claim is therefore *jointly* a statement about the operator and the initial datum: the same Chernoff function can be order-2 on one core and order-4 on a tighter core (Galkin-Remizov 2025 *IJM* Example 4.2).

The v2.x library encodes order via the single `fn order(&self) -> u32` trait method on `ChernoffFunction<F>` — a *single* number per impl. This forces a per-impl committal to one global order regardless of the initial-datum smoothness, and offers no compile-time path for the caller to *witness* that their input lies in the order-K core. Two consequences:

- The **ζ⁴ correction** (ADR-0075, A5) requires `f ∈ D(A^6)` and `a ∈ C^6_b` (Galkin-Remizov §3.2); there is no v2.x mechanism to express or verify this requirement at the type level — the gate currently lives in test fixtures only.
- The **Hörmander order-2 sub-Riemannian** track (v3.1+ A3 research) and the **v4.0 B8 order-6/8 truncated-exp** track both need the same "f lives in a tighter subspace" expression — and would each invent ad-hoc per-trait machinery if a shared abstraction is not shipped now.

Galkin-Remizov 2025 *Israel Journal of Mathematics* — *Tangency of Chernoff approximations to operator semigroups on Banach spaces* — formalises this via the notion of **order-K tangency**: $F$ is order-$K$ tangent to $\{S(t)\}$ on a core $\mathcal{D}_K \subset D(A^K)$ if for every $f \in \mathcal{D}_K$,
$$
F(\tau) f \;=\; f + \tau A f + \tfrac{\tau^2}{2} A^2 f + \cdots + \tfrac{\tau^K}{K!} A^K f + O(\tau^{K+1}),
$$
with the $O(\tau^{K+1})$ remainder uniform on bounded subsets of $\mathcal{D}_K$. Theorem 3.1 of Galkin-Remizov 2025 lifts this pointwise tangency to a global $O(1/n^K)$ Chernoff convergence rate on the same core. **The trait surface that codifies "this datum and this Chernoff function jointly support order $K$" is the v3.0 `ApproximationSubspace<K, F>` marker trait.**

This is a **BREAKING-window** addition (v3.0 = first BREAKING release per `roadmap-reflective-biscuit.md`) — but the trait itself is *purely additive* on the public surface (opt-in). It rides the v3.0 release window because (a) it requires the v3.0 ADR-0074 ChernoffFunction trait cleanup (removing `Clone` bound on `Self::S` is needed for `jet` to write into pre-allocated slices), and (b) shipping it BEFORE v4.0 banks the trait surface for B1/A3/B8 use cases that all need it.

## Decision

Ship four additive public-surface items in v3.0:

- **`pub trait ApproximationSubspace<const K: usize, F: SemiflowFloat = f64>: ChernoffFunction<F>`** — new opt-in marker trait. Required methods:
  ```rust
  /// Returns true if `f` lies in the order-K approximation subspace D(A^K).
  /// Witness for order-K convergence claims via Galkin-Remizov 2025 IJM
  /// Theorem 3.1 tangency.
  fn in_subspace(&self, f: &Self::S) -> bool;

  /// K-jet operator: writes [A^0 f, A^1 f, ..., A^K f] into `out`.
  /// out MUST have length K + 1; out[0] = f (identity); out[K] = A^K f.
  /// Returns DomainViolation if out.len() != K + 1, or if any iterated A^k f
  /// triggers DomainViolation from the implementor's discrete operator.
  fn jet(&self, f: &Self::S, out: &mut [Self::S]) -> Result<(), SemiflowError>;
  ```
  The trait is generic over **both** a const-generic `K: usize` AND the float type `F: SemiflowFloat = f64` (default per ADR-0025). The same impl block can declare BOTH `ApproximationSubspace<4, f64>` AND `ApproximationSubspace<6, f64>` with different IC-class requirements per Galkin-Remizov 2025 *IJM* Example 4.2 — the const-generic on the trait (not on the type) gives compile-time order verification AND per-K specialisation.

- **Opt-in pattern, NOT default-bridged.** Unlike v2.7's `TimedChernoffFunction<F>` super-trait (ADR-0070) — which ships a default `apply_at` that bridges to `apply_into` ignoring `t`, so every existing autonomous impl gets the super-trait for free — `ApproximationSubspace<K, F>` has NO default impl. An impl must explicitly write the `in_subspace` / `jet` bodies. This is a deliberate choice (see Rationale): "witnessing K-jet membership" is a non-trivial mathematical claim that we refuse to auto-confess on every `ChernoffFunction<F>` impl; the caller MUST request K-jet eligibility by writing the explicit impl block.

- **`pub fn assert_in_subspace<C, F, const K: usize>(chernoff: &C, f: &C::S) -> Result<(), SemiflowError>`** — module-level convenience helper in `crates/semiflow-core/src/approximation.rs`. Returns `Ok(())` if `chernoff.in_subspace(f)` AND `Ok(())` propagated from `jet`-via-discard-buffer; returns `DomainViolation { reason: "f ∉ D(A^K)" }` otherwise. Used by gate G_AS_K tests + the ζ⁴ correction (ADR-0075) at construction time.

- **Three v3.0 reference impls** — opt-in on existing v2.x kernels (no new structs):
  - `impl<F: SemiflowFloat> ApproximationSubspace<2, F> for DiffusionChernoff<F>` — the v0.3.0 ζ-A diffusion. `in_subspace`: returns `true` iff the underlying `GridFn1D` has at least 5 grid points (the central-difference stencil width). `jet`: writes `[f, A f, A² f]` via two iterations of the discrete divergence-form operator $\partial_x(a(x) \partial_x \cdot)$ on the grid. Replaces the v2.x test-fixture witness with a public-surface witness.
  - `impl<F: SemiflowFloat> ApproximationSubspace<4, F> for Diffusion4thChernoff<F>` — the v0.6.0 4th-order diffusion. `in_subspace`: ≥9 grid points AND `a ∈ C^4_b` (caller-provided via `a_kth_bound: F` field on the type — additive non-breaking field, default `F::INFINITY`). `jet`: 4 iterations of the 4th-order spatial operator.
  - `impl<F: SemiflowFloat> ApproximationSubspace<6, F> for TruncatedExp4thDiffusionChernoff<F>` — the v0.6.0 K=4 truncated-exp variant (the type name retains "4" for backwards-compat per ADR-0011; the *subspace* it witnesses is K=6 per the math §27 derivation). `in_subspace`: ≥13 grid points AND `a ∈ C^6_b`. `jet`: 6 iterations. **This is the witness the ζ⁴ correction (ADR-0075) consumes.**

File layout: `crates/semiflow-core/src/approximation.rs` (~250 LoC target — trait + helper + 3 opt-in impls move to here; HARD LIMIT 400 LoC well under the default 500-LoC cap; NO Override #1 expansion). The 3 opt-in `impl ApproximationSubspace` blocks live in `approximation.rs` (not co-located with the kernel types) because (a) the witness logic is independent of the kernel's `apply_into`, (b) Wave B can add or remove witness impls without touching kernel files, (c) future K-jet impls (Magnus K=4, NonSeparable2D, etc.) accumulate in one place.

Schema bumps: `traits.yaml` 0.8.0 → **1.0.0** (MAJOR per Override re-eval; first stable schema baseline — see ADR-0074 §"Decision"); `properties.yaml` 0.10.0 → 0.11.0 (NEW gate category "approximation-subspace witness" — see Acceptance gates). math.md is append-only (§26 NEW).

## Rationale

- **Why const-generic `K` on the trait (not on `ChernoffFunction<F>` itself)?** Pushing `K` up to `ChernoffFunction<const K, F>` would force EVERY existing impl (26+ types as of v2.8) to declare a single global K — mass-rename churn, and worse: it would force a single committal per type when Galkin-Remizov §3.1 explicitly contemplates the *same* Chernoff function having different K-tangency on different cores (Example 4.2). The opt-in marker trait pattern (Option C from the roadmap) lets `TruncatedExp4thDiffusionChernoff` declare `ApproximationSubspace<2>` (the weak core), `ApproximationSubspace<4>` (the medium core), AND `ApproximationSubspace<6>` (the strict core) as three independent impl blocks. The caller picks the order it wants via the const-generic at call site: `chernoff.in_subspace::<4>(f)` is a different question from `chernoff.in_subspace::<6>(f)`. This is the *only* design that scales.
- **Why an associated-type `type Subspace` was rejected.** A natural alternative is to add `type Subspace<const K: usize>` to `ChernoffFunction<F>` and force every impl to declare `type Subspace<const K> = SomeWitnessType<K>`. The fatal flaw: associated types breakdown trait-object usage (`dyn ChernoffFunction<F>` can't see GATs in stable Rust until 1.85+), and several v2.x callers (the resolvent eval loop, the Howland lift inner-step dispatch) rely on trait-object dispatch for heterogeneous-kernel composition. The marker-trait Option C preserves trait-object dispatch on `ChernoffFunction<F>` while gating K-jet access through static-dispatch sites only (where `K` is known at compile time). Suckless minimal-surface choice.
- **Why a free-function `verify_in_subspace<K>(chernoff, f)` outside any trait was rejected.** A free function would force every K-jet computation to re-implement the per-kernel iteration logic (the `A^k f` evaluation) inside the verify-helper. Encapsulating `jet` as a trait method keeps the iteration logic co-located with the kernel — the kernel knows how to evaluate its own discrete generator efficiently (e.g., `Diffusion4thChernoff` reuses its 9-point stencil for both `apply_into` and `jet`). Per-impl specialisation matters for performance (K-jet evaluation is the hot path inside the ζ⁴ correction).
- **Why NO default-bridged impl (unlike v2.7's `TimedChernoffFunction<F>`)?** `TimedChernoffFunction::apply_at(t, τ, ...)` defaults to bridging to `apply_into(τ, ...)` ignoring `t` — that's a *mathematically honest* default for autonomous Chernoff functions (the time coordinate genuinely doesn't enter the computation). The analog for `ApproximationSubspace<K, F>` would be `default in_subspace(_, _) { false }` and `default jet(_, _) { Err(NotImplemented) }` — but a false witness is *worse* than no witness (the caller silently gets `false` and assumes the impl doesn't support K-jet, when actually the impl just hasn't opted in). Forcing the explicit `impl ApproximationSubspace<K, F> for X<F> { ... }` block prevents the silent-false-witness footgun. Suckless explicit-over-implicit.
- **Why opt-in three impls in v3.0 (not all 26+ ChernoffFunction impls)?** The three opt-in impls (`DiffusionChernoff`, `Diffusion4thChernoff`, `TruncatedExp4thDiffusionChernoff`) underwrite the three downstream use cases: K=2 = the compatibility shim's order-2 floor (the v2.x callers that relied on `order() == 2` now get a K=2 witness instead), K=4 = the v3.1+ A3 Hörmander track and v4.0 B8 order-6 expansion, K=6 = the ζ⁴ correction of ADR-0075 (immediate consumer in v3.0). Other v2.x impls (Strang2D, MagnusGraph, Schrödinger, etc.) MAY opt in at v3.1+ when concrete use cases demand it. Shipping all 26 in v3.0 is scope creep — the trait surface is what banks the future work, not the impl coverage.
- **Why does `in_subspace` return `bool` (not `Result<(), SemiflowError>`)?** A pure-predicate boolean fits the standard Rust trait idiom (`fn contains(&self) -> bool`); the convenience helper `assert_in_subspace::<K>` wraps the bool into a `Result` for caller ergonomics. Keeping the trait method itself a bool keeps the surface minimal and matches the v2.6 `KillingRegion::is_inside` precedent (also a bool, no Result).
- **Why does `jet` write to `&mut [Self::S]` (an externally-allocated slice) rather than returning `Vec<Self::S>`?** Zero-allocation hot path. The ζ⁴ correction (ADR-0075) calls `jet(f, &mut self.scratch_jet)` once per Chernoff step with `K = 6` — that's 7 state-buffer allocations per step if `jet` returned a `Vec`. Pre-allocating the K+1 buffers once at construction time (ScratchPool reuse) and passing the slice in is the suckless `apply_into` pattern from ADR-0041. The trait CAN be used in allocating contexts via the standalone helper `let mut jet = vec![f.zeroed_like(); K+1]; chernoff.jet(f, &mut jet)?;` — explicit at the call site, no hidden allocation in the trait method.
- **Why `ApproximationSubspace<K, F>` and not `ApproxSubspace<K, F>` (per the roadmap-reflective-biscuit.md draft name)?** Full word in public-surface trait names; mirrors `ChernoffFunction` / `TimedChernoffFunction` / `BoundedGeometryManifold` / `ReflectingRegion` naming convention. The roadmap draft `ApproxSubspace<k>` is the conversational shorthand.
- **Why ship the trait surface before the ζ⁴ correction (ADR-0075) consumes it?** Strict ordering inside Wave A: the trait must compile, the G_AS_K gates must pass, BEFORE Wave C ζ⁴ engineering work begins (the ζ⁴ ADR depends on this one). v3.0 ships the trait + 3 opt-in impls + G_AS_K gates in Wave A; the ζ⁴ correction lands in Wave C. Each wave can independently fail without blocking the next-wave dependency analysis.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Add `const K: usize` to `ChernoffFunction<F>` as a top-level const-generic | Mass-rename churn (26+ existing impls); forces single committal per impl (Galkin-Remizov §3.1 Example 4.2 explicitly contemplates multi-K co-witness for the same Chernoff function); breaks trait-object dispatch on `dyn ChernoffFunction<F>` patterns used in the v2.7 resolvent loop. |
| Associated-type `type Subspace<const K>` on `ChernoffFunction<F>` (GAT) | Breaks `dyn ChernoffFunction<F>` trait-object usage in stable Rust until ≥1.85; the v2.7 resolvent uses trait-object dispatch heavily. |
| Free-function `verify_in_subspace<K>(chernoff, f)` outside any trait | Forces re-implementing the per-kernel K-jet iteration logic inside the helper; loses per-impl specialisation (e.g., `Diffusion4thChernoff` reusing its 9-point stencil for both `apply_into` and `jet`). |
| Default-bridged impl: `default fn in_subspace(_, _) -> bool { false }` | Silent-false-witness footgun — the caller queries `chernoff.in_subspace(f)` on an impl that hasn't opted in and gets `false`, assuming the impl doesn't support K-jet when actually it just hasn't declared the impl. Explicit opt-in matches the suckless explicit-over-implicit posture. |
| Return `Vec<Self::S>` from `jet` (allocating) | Per-step heap allocation in the ζ⁴ correction hot path — kills the per-tick latency story. The `&mut [Self::S]` slice form matches the v2.0 ADR-0041 `apply_into` zero-alloc pattern. |
| Ship the trait but NO opt-in impls in v3.0 (impls deferred to v3.1+) | Leaves the ζ⁴ correction (ADR-0075) without a witness type. The two ADRs are tightly coupled; shipping one without the other is non-functional. The three v3.0 opt-in impls are the minimum that closes the ζ⁴ dependency loop. |
| Ship opt-in impls for ALL 26+ v2.x ChernoffFunction types | Scope creep — the trait surface is what banks future work, not the impl coverage. Strang2D / Magnus / Schrödinger / etc. can opt in at v3.1+ when concrete use cases (e.g., Magnus order-6 needing K=6 witness) demand it. |
| Generic-over-K associated type on the trait method (`fn in_subspace<const K: usize>(&self, f: &Self::S) -> bool`) | Per-call generic-on-method dispatch is awkward at usage sites (`chernoff.in_subspace::<4>(f)` rather than `chernoff_as_4.in_subspace(f)`); doesn't compose well with `dyn` dispatch. Trait-level const-generic K is the suckless choice. |
| Bool-only trait (drop `jet` entirely; let callers manually iterate `apply_into`) | Forces every K-jet consumer to re-implement the `[f, A f, A² f, ..., A^K f]` loop with their own scratch-pool management. Encapsulating in `jet` is the suckless minimal-surface choice (one place to get the K-jet right; one place to bug-fix). |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; no existing trait or struct is modified. The opt-in `impl ApproximationSubspace<K, F> for {Diffusion,Diffusion4th,TruncatedExp4thDiffusion}Chernoff` blocks are NEW; the three kernel structs themselves are UNCHANGED.
- **New module `crates/semiflow-core/src/approximation.rs`** (~250 LoC target; HARD LIMIT 400 LoC; default 500-LoC cap with 250-LoC headroom; NO Override #1 expansion).
- **New trait `ApproximationSubspace<const K: usize, F>`** — opt-in super-trait of `ChernoffFunction<F>`. Independent of `TimedChernoffFunction<F>` (ADR-0070); a type CAN implement both (`HowlandLift<C, F>` will get an `ApproximationSubspace<K, F>` opt-in at v3.1 once the time-axis K-jet is worked out — out of scope for v3.0). Independent of `BoundedGeometryManifold<F>` (ADR-0071); `ManifoldChernoff<M, F>` is NOT opted in to `ApproximationSubspace<K, F>` in v3.0 (the manifold K-jet is open math — `[\Delta_M, \cdot]` involves curvature derivatives at high K; defer to v3.1+ research).
- **Dependency count unchanged** at 2/3 budget (still `num-traits`, `libm`). The trait + helper + opt-in impls use only basic arithmetic + the existing kernel `apply_into`.
- **Schema bumps**: `traits.yaml` 0.8.0 → **1.0.0** (MAJOR per Override re-evaluation schedule — see ADR-0074); `properties.yaml` 0.10.0 → 0.11.0 (NEW gate category "approximation-subspace witness"). math.md is append-only (§26 NEW).
- **New gates**: G_AS_K (RELEASE_BLOCKING — per-K=2,4,6 symbolic in_subspace witness; 3 sub-tests, one per opt-in impl; lives in `tests/approximation_subspace_witness.rs` new file, feature `slow-tests`).
- **Backwards-compatibility note**: the v2.x compatibility shim (ADR-0074 §"12-month deprecation shim") preserves the v2.x `fn order(&self) -> u32` semantics; callers that previously relied on `chernoff.order() == 2` continue to work via the shim. New v3.0 callers that need order witnessing SHOULD use `chernoff.in_subspace::<2>(f)` (the K=2 marker-trait witness) — strictly stronger than the v2.x global-order check (it also verifies datum membership).
- **CITATIONs added to math.md §26**: Galkin-Remizov 2025 *Israel Journal of Mathematics* — *Tangency of Chernoff approximations to operator semigroups on Banach spaces*, Theorem 3.1 (the order-K tangency theorem); Example 4.2 (the multi-K co-witness construction); Vedenin-Smolyanov-Voskresenskaya 2020 *Mathematical Notes* (the original characterisation of D(A^K) Chernoff approximation cores — cited for historical lineage of the K-jet idea).

## Migration

End-user impact is **opt-in zero**. No v2.x caller is forced to use the new trait; the v2.x `ChernoffFunction<F>` trait surface (minus the cleanup of ADR-0074) is preserved verbatim via the 12-month shim.

New v3.0 callers who want order witnessing have a strictly stronger replacement for the v2.x `order()` check:

```rust
// v2.x (still works via shim, deprecation warning):
assert_eq!(chernoff.order(), 2);  // global-order check

// v3.0 (the strict witness, replaces the above):
assert!(chernoff.in_subspace::<2>(f));  // f ∈ D(A^2) ∧ chernoff is order-2 on D(A^2)
```

The K=4/K=6 witnesses are NEW capability (no v2.x equivalent); see `docs/migration/v2-to-v3.md` §6 for worked examples (Wave G).

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0003 — no_std + alloc; the trait + helper use only stdlib + the existing kernel API.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; reused for `ApproximationSubspace<K, F>` with `F = f64` default.
- ADR-0026 — `ChernoffFunction<F>` super-trait; `ApproximationSubspace<K, F>` is a super-trait of it.
- ADR-0041 — `apply_into` + `ScratchPool` zero-alloc pattern; reused for `jet` (writes to externally-allocated slice).
- ADR-0070 — v2.7 `TimedChernoffFunction<F>` super-trait pattern; `ApproximationSubspace<K, F>` follows the same additive-super-trait pattern but with explicit opt-in (no default-bridged impl — see Rationale).
- ADR-0074 — v3.0 ChernoffFunction trait cleanup; this ADR DEPENDS on the cleanup (Clone bound removal on `Self::S` is needed for `jet` slice-writing).
- ADR-0075 — v3.0 ζ⁴ correction; the immediate consumer of `ApproximationSubspace<6, F>`.
- ADR-0076 — v3.0 binding redesign; the FFI/PyO3/WASM surfaces do NOT expose `ApproximationSubspace<K, F>` in v3.0 (Rust-only API; binding exposure deferred to v3.1+ once the const-generic-K binding ABI is worked out — see ADR-0076 §"Out of scope").
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v3.0 — release-level roadmap (Design Option C for the trait surface).
- math.md §26 (NEW v3.0) — `ApproximationSubspace<K, F>` normative spec.
- math.md §27 (NEW v3.0) — ζ⁴ correction algorithm (depends on §26).
- `.dev-docs/constitution.md` v1.7.0 (NEW v3.0) — MAJOR re-evaluation; all 3 overrides RE-AFFIRMED.

## Amendments

(none at acceptance time)
