# ADR-0074 — `ChernoffFunction<F>` Trait Cleanup (v3.0 BREAKING)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v3.0 (the BREAKING centrepiece; first BREAKING release since v2.0.0; first MAJOR window in the academic-priority trajectory per `roadmap-reflective-biscuit.md` §v3.0). The v3.0 ChernoffFunction trait is the **stable baseline** for the entire post-v3 trajectory through v4.0.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (v0.9.0 ChernoffFunction generic over `F`), ADR-0035 (v1.0.0 API freeze — first SemVer-stable surface; this ADR is the v3.0 RESET of that freeze), ADR-0041 (`apply_into` + `ScratchPool` additive method, v2.0 Wave 1), ADR-0043 (v2.0 State<F> 3-layer trait hierarchy), ADR-0070 (v2.7 `TimedChernoffFunction<F>` — the trait redesign explicitly considers folding this in; **v3.0 retains it as a separate super-trait** per Rationale).
- **Supersedes / amends**: ADR-0035 v1.0.0 API freeze (PARTIAL — v3.0 is the v2.0.0 → v3.0.0 → v4.0.0 cadence's first BREAKING change; the v1.0.0 freeze was implicitly time-boxed to "until v3.0.0 BREAKING window"). Supersedes the v2.x `ChernoffFunction<F>` trait surface (the `apply` method, the implicit `Clone` bound on `Self::S`, the `(f64, f64)` return of `growth()`). PRESERVES the v2.x trait NAME (`ChernoffFunction<F>`) and the v0.9.0 generic-over-F design.
- **Mathematical foundation**: math.md §26 (NORMATIVE — ApproximationSubspace witness semantics; see ADR-0073). The trait cleanup does NOT add new mathematical content; it CONSOLIDATES the v2.x surface around the v2.0 ADR-0041 `apply_into` zero-alloc pattern and removes legacy v0.x ergonomic baggage. CITATIONs preserved: Chernoff 1968 *J. Funct. Anal.* 2:2; Remizov 2025 *Vladikavkaz Math. J.* 27:4 Theorem 6 (foundational).
- **Acceptance gates added**: G_binding_parity (RELEASE_BLOCKING — cross-binding FFI/PyO3/WASM byte-identical to v2.5.1 baseline on the canonical CEV smoke suite — see ADR-0076; the cleanup MUST NOT alter the binding-observable numerical output by even 1 ULP).

## Context

The v0.1.0–v2.8 `ChernoffFunction<F>` trait surface has accumulated four legacy items that the v3.0 BREAKING window is the only feasible time to remove. After v3.0 the SemVer-stable surface is frozen again through the v4.0 BREAKING window — so cleanup deferred past v3.0 means waiting another ~12 months.

The four items:

1. **`fn apply(&self, tau: F, f: &Self::S) -> Result<Self::S, SemiflowError>`** — the v0.1.0 ergonomic entrypoint. It allocates a fresh `Self::S` per call (the return value), which kills the per-tick latency story for HFT use cases. ADR-0041 (v2.0 Wave 1) added `apply_into(τ, src, dst, scratch)` as an *additive* zero-alloc method; the v2.x trait carried both methods. v3.0 removes `apply` from the trait (moves it to an inherent-impl convenience `apply_chernoff(τ, &src)` per type that wants it, behind a `Self::S: Clone` bound).
2. **Implicit `Self::S: Clone` bound** — the v0.1.0 `apply` method's `Result<Self::S, _>` return signature requires `Self::S: Clone` (the caller calls `func.apply(τ, &f)?` and gets a fresh state). The `Clone` bound has leaked into many consumer sites (`ChernoffSemigroup::evolve`, the L-gate harness, the FFI/PyO3 binding glue). Removing `apply` from the trait lets us drop the `Clone` bound on `Self::S` — the bound becomes opt-in at the call site (`where Self::S: Clone`) only where the *caller* wants the allocating convenience.
3. **`fn growth(&self) -> (f64, f64)`** — the v0.1.0 returns an `(M, omega)` tuple. Two problems: (a) tuple-field access `let (m, om) = func.growth();` or `func.growth().0` is brittle and unmotivated; (b) the tuple type is `(f64, f64)` regardless of the kernel's float parameter `F` — a *latent* v0.9.0 (ADR-0026) Generic-over-Float bug. v3.0 ships `Growth<F>` as a typed struct with named `multiplier: F` and `omega: F` fields, generic over `F: SemiflowFloat = f64`.
4. **`fn order(&self) -> u32` is a required method with no default** (good) BUT the v0.1.0 spec did not enforce explicit declaration. Some v2.x impls (mostly composition types) silently inherit `1` via accident-of-history default. v3.0 reaffirms: `order(&self) -> u32` is required, no default, all impls MUST declare. This is BREAKING for any impl that relied on a now-removed default — but a `grep -rn 'impl ChernoffFunction' crates/` (Wave A audit) identifies all 26+ existing impls and confirms each already has an explicit `order()` body, so the impact is zero on shipped impls.

Additionally, the v2.0 ChernoffSemigroup struct is renamed to `Evolver<C, F>`:
- `ChernoffSemigroup<C>` is the v0.1.0 name; "Semigroup" is mathematically inaccurate (the type evolves a Chernoff *approximation*, which is not itself a semigroup — it's an n-step iterate of a Chernoff function), and the name is wordy.
- `Evolver<C, F>` is the v3.0 name. Aligns naming with v2.7 `HowlandLift<C, F>` / v2.6 `KillingChernoff<C, R, F>` / v2.8 `ReflectedHeatChernoff<C, R, F>` / v2.8 `ManifoldChernoff<M, F>` / v2.7 `LaplaceChernoffResolvent<C, F>` — all single-noun wrapper types.

The v3.0 BREAKING window provides the unique opportunity to land all four cleanups + the rename atomically with a 12-month deprecation shim per ADR-0035 §9 precedent — callers compile with deprecation warnings (not hard errors), then migrate over 12 months, then the shim is removed at v4.0.

## Decision

The v3.0 `ChernoffFunction<F>` trait surface is:

```rust
pub trait ChernoffFunction<F: SemiflowFloat = f64> {
    /// Per-impl state type. NO Clone bound (v2.x had implicit Clone via apply's
    /// return; v3.0 removes both apply and the Clone bound).
    type S: State<F>;

    /// Zero-allocation apply: dst := F(τ) src. Caller provides pre-allocated
    /// dst and a scratch pool for hot-path temporaries. The v2.0 ADR-0041
    /// signature, now the SOLE apply method on the trait.
    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;

    /// Consistency order m ≥ 1 such that ‖F(τ) f − exp(τ A) f‖ = O(τ^{m+1})
    /// on the core (per Theorem 6, Remizov 2025). REQUIRED method; NO default.
    /// All v2.x impls already declare explicitly (Wave A audit verified).
    fn order(&self) -> u32;

    /// Growth bound for ‖F(τ)‖ ≤ multiplier · exp(omega · τ). Generic over F.
    /// v3.0 replaces the v2.x (f64, f64) tuple return.
    fn growth(&self) -> Growth<F>;
}
```

and:

```rust
/// v3.0 typed return for ChernoffFunction::growth.
/// Generic over F: SemiflowFloat = f64 per ADR-0025.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Growth<F: SemiflowFloat = f64> {
    pub multiplier: F,
    pub omega: F,
}

impl<F: SemiflowFloat> Growth<F> {
    /// Idiomatic constructor.
    pub fn new(multiplier: F, omega: F) -> Self { Self { multiplier, omega } }
}
```

and:

```rust
/// v3.0 wrapper that iterates a ChernoffFunction n times with step τ = t / n,
/// producing the Chernoff approximant (S(t/n))^n g of exp(tA) g.
/// RENAMED from v2.x ChernoffSemigroup<C>. v2.x type alias retained for 12-month
/// shim (see Migration).
pub struct Evolver<C: ChernoffFunction<F>, F: SemiflowFloat = f64> {
    func: C,
    n: usize,
    _phantom_f: core::marker::PhantomData<F>,
}
```

**Per-impl inherent convenience for callers that want the allocating apply** (NOT a trait method):

```rust
impl<C, F> C
where
    C: ChernoffFunction<F>,
    C::S: Clone,                         // opt-in Clone bound at the impl site, NOT the trait
    F: SemiflowFloat,
{
    /// Allocating apply: returns a fresh state. Convenience for non-hot-path
    /// callers (one-shot evaluations, REPL exploration). Hot-path callers
    /// MUST use apply_into instead. v2.x's `apply` is the closest equivalent.
    pub fn apply_chernoff(&self, tau: F, src: &C::S) -> Result<C::S, SemiflowError> {
        let mut dst = src.clone();
        let mut scratch = ScratchPool::<F>::new();
        self.apply_into(tau, src, &mut dst, &mut scratch)?;
        Ok(dst)
    }
}
```

**12-month compatibility shim** (lives in `crates/semiflow-core/src/v2_compat.rs`, ~120 LoC, default 500-LoC cap):

```rust
//! v2.x compatibility shim — hard-removed at v4.0.
//!
//! Re-exports v2.x type aliases and provides deprecated method shims so v2.x
//! callers compile with WARNINGS (not hard errors) for 12 months from v3.0.0.
//! Per ADR-0035 §9 deprecation precedent; ADR-0074 §"Migration".

#[deprecated(
    since = "3.0.0",
    note = "Renamed to Evolver<C, F> in v3.0. The ChernoffSemigroup alias \
            is preserved until v4.0; migrate to Evolver before then."
)]
pub type ChernoffSemigroup<C> = crate::Evolver<C, f64>;

/// Deprecated v2.x apply shim. Equivalent to the v3.0 inherent
/// `apply_chernoff(τ, &src)` convenience.
#[deprecated(
    since = "3.0.0",
    note = "Renamed to apply_chernoff(τ, &src) in v3.0; or use the zero-alloc \
            apply_into(τ, &src, &mut dst, &mut scratch) directly. Hard-removed at v4.0."
)]
pub trait ChernoffFunctionApplyShim<F: SemiflowFloat>: crate::ChernoffFunction<F>
where
    Self::S: Clone,
{
    #[deprecated(since = "3.0.0", note = "Use apply_chernoff or apply_into.")]
    fn apply(&self, tau: F, f: &Self::S) -> Result<Self::S, crate::SemiflowError> {
        self.apply_chernoff(tau, f)
    }
}

impl<C, F> ChernoffFunctionApplyShim<F> for C
where
    C: crate::ChernoffFunction<F>,
    C::S: Clone,
    F: SemiflowFloat,
{}
```

The shim file is GATED by `#[cfg(feature = "v2_compat")]` (default = on through v3.x; default = off at v4.0 → shim file deleted entirely). The feature is documented in the crate root rustdoc as "v3.0 → v4.0 transitional; do not depend on for new code."

For `growth()` callers using the v2.x tuple-access pattern, the shim is a per-call helper, NOT a method-shim (tuple-destructuring `let (m, om) = func.growth()` is a syntax that no shim can transparently support across the type change). v2.x callers MUST migrate to `let g = func.growth(); g.multiplier; g.omega;` — documented in `docs/migration/v2-to-v3.md` §3 (Wave G).

**Migration table** (full coverage):

| v2.x surface | v3.0 surface | Migration kind | Shim coverage |
|---|---|---|---|
| `func.apply(τ, &f)?` returning `Result<C::S, _>` | `func.apply_chernoff(τ, &f)?` (allocating, where `C::S: Clone`) OR `func.apply_into(τ, &src, &mut dst, &mut pool)?` (zero-alloc, recommended) | API rename or zero-alloc port | Deprecation warning via `ChernoffFunctionApplyShim` (v3.x default-on); HARD REMOVED v4.0 |
| `func.growth().0` (tuple `.0` access for M) | `func.growth().multiplier` | Field-rename (no shim possible across type change) | NONE — caller migration required (documented in migration guide) |
| `func.growth().1` (tuple `.1` access for ω) | `func.growth().omega` | Field-rename | NONE — caller migration required |
| `let (m, om) = func.growth();` | `let g = func.growth(); let (m, om) = (g.multiplier, g.omega);` (or destructure: `let Growth { multiplier: m, omega: om } = func.growth();`) | Destructure migration | NONE — caller migration required |
| `ChernoffSemigroup::new(c, n)?` | `Evolver::new(c, n)?` (the rename) | Type-rename | `ChernoffSemigroup<C>` type alias for `Evolver<C, f64>` (deprecation warning); HARD REMOVED v4.0 |
| `where C::S: Clone` as a trait bound (consumer site) | Keep the bound — moves from trait-implicit to consumer-explicit | Code rewrites in trait-bound positions on consumer types | Compiles unchanged (the consumer site's bound is still valid); NO shim needed |
| `impl ChernoffFunction<F> for X { fn apply(...) {...} }` (custom v2.x impl) | `impl ChernoffFunction<F> for X { fn apply_into(...) {...} }` (rewrite to zero-alloc) | API REWRITE — the old `apply` is no longer a trait method | NO transparent shim possible; documented in migration guide §4 with worked example |

File layout: `crates/semiflow-core/src/chernoff.rs` — UNCHANGED location (still hosts the `ChernoffFunction<F>` trait + `Evolver<C, F>` struct). NEW file `crates/semiflow-core/src/v2_compat.rs` (~120 LoC, default 500-LoC cap; NO Override expansion). The 4 cleanup changes happen inline in `chernoff.rs`; the `Growth<F>` struct moves to a new module if `chernoff.rs` exceeds the cap (deferred; current `chernoff.rs` is well under 200 LoC).

Schema bumps: `traits.yaml` 0.8.0 → **1.0.0** (MAJOR per Override re-evaluation schedule — see Rationale below; first stable schema baseline since the v0.x trajectory began); `properties.yaml` 0.10.0 → 0.11.0 (NEW gate G_binding_parity — see ADR-0076). math.md is append-only (§26 NEW for ADR-0073).

## Rationale

- **Why a v3.0 MAJOR cleanup (not a v2.9 MINOR with `apply_into` becoming "preferred")?** A v2.9 MINOR with both methods on the trait does not buy the cleanup — it just adds another method. The Clone bound continues to leak through `apply`'s return signature; the growth tuple continues to be `(f64, f64)`; the `apply` method continues to be the v0.x ergonomic entrypoint that allocates per call. Only a MAJOR BREAKING release lets us *remove* `apply` from the trait and the `Clone` bound from `Self::S`. The v3.0 → v4.0 cadence (12-month BREAKING windows per `roadmap-reflective-biscuit.md`) means cleanup deferred past v3.0 waits another ~12 months. Suckless-discipline: do the cleanup at the first BREAKING window that's available, not later.
- **Why retain the trait NAME `ChernoffFunction<F>` (not rename to `ChernoffApprox<F>` or similar)?** Brand stability; the v0.x rustdoc citations, the v1.0.0 API freeze documentation, the paper draft `draft-ru/paper-main-ru.ipynb`, the iter-3 bench results, and the README all reference "ChernoffFunction" as the trait identity. Renaming the trait is gratuitous churn for zero mathematical content. The cleanup is INTERNAL to the trait surface; the name stays.
- **Why `Growth<F>` as a struct (not a `(F, F)` tuple)?** Field-access ergonomics: `growth().multiplier` and `growth().omega` are self-documenting; `growth().0` and `growth().1` require the caller to remember the tuple order (or look at rustdoc on every call). The struct also enables future additive fields (`pub struct Growth<F> { ..., pub deferred_field: Option<F> }`) without breaking callers — the tuple form forbids any future expansion. The struct is `#[derive(Clone, Copy, PartialEq)]` so f64 callers get the same lightweight pass-by-value behaviour as the tuple.
- **Why `apply_chernoff` as an inherent-impl convenience (not a trait method)?** Two-pronged: (a) **trait-method** + Clone bound → forces every consumer using trait-object dispatch (`dyn ChernoffFunction<F>`) to know the dynamic-Clone-availability story, which is awkward; (b) **inherent-impl** + Clone bound at the impl site → the bound is opt-in at the *call site* (the caller knows whether their kernel's `C::S` is Clone). Per-impl inherent works because Rust's coherence rules let us add `impl<C: ChernoffFunction<F>> C { fn apply_chernoff(...) where C::S: Clone {} }` — the bound at the impl is enforced at the call site, not the trait definition. Suckless minimal surface: trait stays zero-bound; convenience is opt-in.
- **Why remove the `Self::S: Clone` bound from the trait (rather than keep it for v2.x compatibility)?** Three downstream wins: (a) **HFT/embedded callers** can build kernels with non-Clone state types (e.g., a memory-mapped `MmapState<F>` that can't be Clone because the underlying mmap is single-owner); (b) **trait-object dispatch sites** no longer need to confess Clone in the bound; (c) **bindings** (FFI/PyO3/WASM) don't need to provide a Clone shim for opaque types. The bound IS LEFT to opt in at the consumer site for callers that need the allocating convenience — explicit-over-implicit.
- **Why `order() -> u32` is REQUIRED with NO default (not `Default = 1`)?** Default-1 was the v0.1.0 footgun: composition types (Strang2D, AxisLift, etc.) accidentally inherited `order = 1` from an absent override, silently advertising worse asymptotic behaviour than they delivered. The v3.0 trait makes `order(&self)` required (no default) — Wave A audit confirms all 26+ v2.x impls already declare it explicitly, so zero impact on shipped impls; new impls MUST declare. Suckless explicit-over-implicit.
- **Why rename `ChernoffSemigroup<C>` → `Evolver<C, F>`?** "Semigroup" is mathematically inaccurate: the type evolves an n-step Chernoff iterate (S(t/n))^n, which CONVERGES to the semigroup exp(tA) but is itself NOT a semigroup (the n-step iterate is an *approximant*; composition like (S(t/n))^n ∘ (S(s/m))^m doesn't have the semigroup law). "Evolver" reads as "drives the evolution forward by n Chernoff steps" — accurate, concise, single-noun. Aligns naming with v2.6 `KillingChernoff<C, R, F>` / v2.7 `HowlandLift<C, F>` / v2.7 `LaplaceChernoffResolvent<C, F>` / v2.8 `ReflectedHeatChernoff<C, R, F>` / v2.8 `ManifoldChernoff<M, F>` — all single-noun wrappers parameterised over inner kernel + float.
- **Why a 12-month deprecation shim (not hard removal at v3.0)?** Per ADR-0035 §9 precedent: the v1.0.0 → v2.0.0 cycle gave 12 months of `#[deprecated]` warnings before hard removal at v2.0. The v3.0 → v4.0 cycle SHOULD give the same 12 months. The shim file is small (~120 LoC) and `#[cfg(feature = "v2_compat")]` gated; consumers compile with WARNINGS (not hard errors) for 12 months, then the warnings escalate at v3.x.y minor releases (the engineer adds `#[deprecated(since="3.x")] note: HARD REMOVE AT 4.0` per ADR-0035 §9 pattern), then the shim file is deleted at v4.0. Standard Rust ecosystem cadence.
- **Why explicitly NOT fold `TimedChernoffFunction<F>` (ADR-0070) into the redesigned `ChernoffFunction<F>` (despite the roadmap noting this as a consideration)?** Three reasons: (a) the time-dependence is a *real* additive surface (autonomous impls genuinely don't carry a time coordinate; the trait split makes this typed); (b) folding it in would force every existing autonomous impl to declare `apply_at` (or carry a default-bridge), inflating every impl block by 5+ lines for zero gain; (c) the v2.7 ADR-0070 default-bridge mechanism already gives autonomous impls `TimedChernoffFunction<F>` "for free" via a one-line `impl TimedChernoffFunction<F> for X<F> {}` blanket — that's the suckless choice. The roadmap's "explicitly considers folding" became "explicitly preserved as separate super-trait" at v3.0 architecture review (this ADR).
- **Why traits.yaml schema 0.8.0 → 1.0.0 (MAJOR jump)?** The v3.0 cleanup is the FIRST stable schema baseline. The v0.x → v2.x schema versions (0.1.x through 0.8.0) were tracking minor incremental additions; v3.0 establishes the v1.x schema family as the post-v3-stable surface. SemVer MAJOR per the established convention (`grid_fn.rs` schema bumped 0.5.0 → 0.6.0 at v0.6 release; 0.6.0 → 0.7.0 at v2.7; etc. — pattern is MINOR per release window). The 0.8.0 → 1.0.0 BREAKING bump at v3.0 reflects: (a) the trait surface itself is BREAKING; (b) the v3.0 ADR-0073/0074/0075/0076 jointly establish the stable baseline; (c) future MINOR additions (v3.x ApproximationSubspace opt-in expansion, etc.) increment 1.0.x within the stable schema family.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Keep `apply` on the trait; add `#[deprecated]` only | Forces every consumer to disable the warning or migrate; the Clone bound on `Self::S` continues to leak through the `Result<Self::S, _>` return; the v3.0 BREAKING window is wasted. |
| Keep the `(f64, f64)` tuple return of `growth()` | Caller ergonomics regression (`growth().0` vs `growth().multiplier`); blocks future additive fields; latent Generic-over-Float bug (the tuple is always `(f64, f64)` regardless of `F`). |
| Keep `ChernoffSemigroup<C>` name | Mathematically misleading; out of step with the v2.6/v2.7/v2.8 single-noun wrapper naming convention. The rename has clear pedagogical and aesthetic value. |
| Hard-remove the v2.x surface at v3.0 (no shim) | Breaks every v2.x caller cold; forces an immediate 100% migration on the entire downstream user base. The 12-month shim per ADR-0035 §9 is the project's established BREAKING-window protocol. |
| Defer the cleanup to v4.0 (keep v3.0 ChernoffFunction surface = v2.x) | The v3.0 BREAKING window is the only feasible time before v4.0; deferring means another ~12 months of accumulated baggage. Suckless: do the cleanup at the first opportunity. |
| Add a NEW trait `ChernoffFunctionV3<F>` (parallel to the v2.x `ChernoffFunction<F>`) | Two trait names doing the same job; every impl needs to declare both; doubles the rustdoc surface. The in-place rename is the clean choice. |
| Rename trait to `ChernoffApprox<F>` (drop "Function") | Loses brand stability (v0.x–v2.x rustdoc, papers, READMEs reference "ChernoffFunction"); zero mathematical content in the rename. Keep the name. |
| Make `Growth<F>` a `#[non_exhaustive]` struct | Forbids the caller from destructuring with `let Growth { multiplier, omega } = func.growth();` — which is one of the recommended migration paths from the v2.x tuple-destructure. The struct is explicitly EXHAUSTIVE in v3.0; future additive fields will go through a v4.0 BREAKING release (or v3.x with `#[non_exhaustive]` retroactively added). |
| Fold `TimedChernoffFunction<F>` into `ChernoffFunction<F>` (per roadmap consideration) | Inflates every autonomous-impl block by 5+ lines for the `apply_at` default-bridge; the v2.7 ADR-0070 blanket pattern is already the suckless choice. Architecture review (this ADR) decides AGAINST folding. |
| Keep `order(&self) -> u32` default = 1 (don't make it required) | Reaffirms the v0.1.0 footgun (composition types silently advertising worse asymptotic behaviour); Wave A audit confirms zero current impl impact, so required-explicit is free. |

## Consequences

- **BREAKING changes for v2.x callers** (mitigated by 12-month shim):
  - `func.apply(τ, &f)?` → `func.apply_chernoff(τ, &f)?` or `func.apply_into(...)` — shim warns, doesn't break.
  - `func.growth().0` / `.1` → `func.growth().multiplier` / `.omega` — NO shim (field-rename across types); caller migration required (documented in migration guide).
  - `ChernoffSemigroup::new(c, n)?` → `Evolver::new(c, n)?` — type-alias shim warns, doesn't break.
  - `where C::S: Clone` bound in consumer trait-bound positions: keeps compiling unchanged (the bound is still valid at the consumer); no caller change needed.
- **NEW file `crates/semiflow-core/src/v2_compat.rs`** (~120 LoC, default 500-LoC cap, `#[cfg(feature = "v2_compat")]` gated; deleted at v4.0).
- **`Growth<F>` struct + `Evolver<C, F>` type** added to `crates/semiflow-core/src/chernoff.rs` (~50 LoC additive; current `chernoff.rs` ~200 LoC, well under cap).
- **Trait surface**: 4 fewer methods total compared to v2.x (`apply` removed; `Clone` bound dropped; `growth` return type changed; `order` reaffirmed required). The trait is now **3 methods**: `apply_into`, `order`, `growth`.
- **Inherent convenience**: `apply_chernoff` is a per-impl inherent method available wherever `Self::S: Clone` — covers the vast majority of v2.x callers via the shim.
- **Schema bumps**: `traits.yaml` 0.8.0 → **1.0.0** (MAJOR — first stable schema baseline); `properties.yaml` 0.10.0 → 0.11.0 (NEW gate G_binding_parity — see ADR-0076). math.md is append-only (§26, §27 NEW for ADR-0073 + ADR-0075).
- **New gate**: G_binding_parity (RELEASE_BLOCKING — cross-binding FFI/PyO3/WASM byte-identical to v2.5.1 baseline on the canonical CEV smoke suite; the trait cleanup MUST NOT alter binding-observable numerical output by even 1 ULP). Engineer Wave G ships the bench harness.
- **Dependency count unchanged** at 2/3 budget. The cleanup adds zero deps.
- **12-month deprecation timeline**:
  - **v3.0.0** (release): shim active by default (feature `v2_compat` = on); deprecation warnings on v2.x callers.
  - **v3.1.0 – v3.x.y** (next 12 months): shim continues; engineer escalates warnings per ADR-0035 §9 pattern at minor releases.
  - **v4.0.0** (12 months after v3.0): shim file `v2_compat.rs` DELETED; `#[cfg(feature = "v2_compat")]` removed; hard removal complete.
- **CITATIONs preserved** (no new math citations needed — this is a trait cleanup): Chernoff 1968 *J. Funct. Anal.* 2:2 (foundational Chernoff theorem); Remizov 2025 *Vladikavkaz Math. J.* 27:4 Theorem 6 (foundational Chernoff product formula). The v3.0 cleanup does NOT add new mathematical content.

## Migration

End-user impact:

- **v2.x callers using only `apply`**: compile with deprecation warning; migrate to `apply_chernoff` (one-line rename) or `apply_into` (zero-alloc port) within 12 months. Worked example in `docs/migration/v2-to-v3.md` §2.
- **v2.x callers using `growth().0` / `.1`**: HARD BREAK (no shim). Migrate to `growth().multiplier` / `.omega` immediately at v3.0 upgrade. Worked example in §3.
- **v2.x callers using `ChernoffSemigroup`**: compile with deprecation warning; migrate to `Evolver` (one-line rename) within 12 months. Worked example in §4.
- **v2.x impls of `ChernoffFunction`**: REWRITE `fn apply(...)` to `fn apply_into(...)` (signature change is the only mechanical action); REWRITE `fn growth() -> (f64, f64)` to `fn growth() -> Growth<F>` (one-line return-type and constructor); REWRITE `fn order(&self) -> u32` (already required; reaffirm explicit body). Worked example in §5.
- **Binding consumers (FFI/PyO3/WASM)**: see ADR-0076. The v2 binding surface is preserved via parallel headers/pyclasses/JS classes for 12 months; the v3 binding surface uses the cleaned-up trait.

Full migration playbook with worked examples per binding: `docs/migration/v2-to-v3.md` (engineer Wave G fills the per-binding worked examples; architect ships the SCAFFOLD).

## Cross-references

- ADR-0001 — contract-first; this ADR is a contract-layer redesign before any Rust impl ships.
- ADR-0003 — no_std + alloc; the cleanup preserves the no_std posture.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; the `Growth<F>` struct adheres.
- ADR-0026 — v0.9.0 ChernoffFunction generic-over-F; this ADR preserves the design and removes the latent `(f64, f64)` Generic-over-Float bug.
- ADR-0035 — v1.0.0 API freeze; this ADR is the v3.0 RESET (first BREAKING since v2.0). The 12-month deprecation shim cadence per §9 is reused here.
- ADR-0041 — `apply_into` + `ScratchPool`; the v3.0 trait centres on this method.
- ADR-0043 — v2.0 State<F> trait hierarchy; this ADR removes the implicit `Self::S: Clone` bound (Clone moves to opt-in via consumer trait-bounds).
- ADR-0070 — v2.7 `TimedChernoffFunction<F>`; v3.0 PRESERVES as separate super-trait per Rationale.
- ADR-0073 — v3.0 `ApproximationSubspace<K, F>` opt-in marker trait; depends on the Clone-bound removal in this ADR.
- ADR-0075 — v3.0 ζ⁴ correction; depends on ADR-0073 which depends on this ADR.
- ADR-0076 — v3.0 v2→v3 binding redesign; the binding surfaces use the cleaned-up trait via the v3 header / pyclass / JS-class surfaces.
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v3.0 — release-level roadmap.
- math.md §26 (NEW v3.0 — ApproximationSubspace witness semantics).
- `.dev-docs/constitution.md` v1.7.0 (NEW v3.0 — MAJOR re-evaluation; all 3 overrides RE-AFFIRMED).
- `docs/migration/v2-to-v3.md` (NEW v3.0; engineer Wave G).

## Amendments

(none at acceptance time)
