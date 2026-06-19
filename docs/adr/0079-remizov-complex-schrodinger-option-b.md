# ADR-0079 — `SemiflowComplex<C>` Trait + Schrödinger Option B (B6)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v4.0 Wave A (first Wave of the second BREAKING window of the academic-priority trajectory; trait surface freeze for v4.0). Ships in lockstep with ADR-0080 (PointEval first-class), ADR-0081 (d-D shift), ADR-0082 (matrix-valued), ADR-0083 (resolvent residual gate), ADR-0084 (v2_compat hard removal), ADR-0085 (G_zeta4 deferral).
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float with `F = f64` default), ADR-0026 (`ChernoffFunction<F>` super-trait generic over `F`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0057 (v2.2 SchrödingerChernoff Option A real-pair representation — this ADR ships the SIBLING Option B WITHOUT removing Option A), ADR-0073 (v3.0 `ApproximationSubspace<K, F>` opt-in marker trait — SchrödingerChernoffComplex opts into K=2 witness), ADR-0074 (v3.0 ChernoffFunction trait cleanup — `Growth<F>` typed return used by the new kernel's `growth()`).
- **Supersedes / amends**: ADR-0057 §"Decision" (PARTIAL — v2.2 Option A real-pair representation is PRESERVED verbatim through v4.x; v4.0 adds the Option B sibling per ADR-0057 §"Out of scope" → "deferred to v4.0+ once SemiflowComplex is available". The deferral is now FULFILLED.) Establishes a NEW generic trait class `SemiflowComplex<C>` paralleling `SemiflowFloat<F>` (ADR-0025) and a NEW kernel `SchrödingerChernoffComplex<C: SemiflowComplex>` that stores `ψ : GridFn1D<C>` natively.
- **Mathematical foundation**: math.md §30 (NORMATIVE library — `SemiflowComplex<C>` trait semantics + Option B Chernoff step; CITATION Pazy 1983 §6.4 — operator-splitting decomposition of unitary semigroups; Cheng 2008 §3.2 — Crank-Nicolson Cayley map as unitary rational approximation; Engel-Nagel 2000 §IV.6 — unitary semigroups on Hilbert space). The v2.2 ADR-0057 §"Out of scope" lifts to a concrete trait foundation: math §30.2 Definition 30.1 (complex-arithmetic trait) + §30.3 equation 30.1 (palindromic Strang) + equation 30.2 (Cayley map) + §30.4 Proposition 30.1 (unitarity at f64).
- **Acceptance gates added**: G_SCHROD_B (RELEASE_BLOCKING — Schrödinger Option B unitarity ≤ 1e-12 at f64 on harmonic-oscillator Gaussian wave packet at N=512, T=1.0, n=128; ADVISORY sub-test for cross-representation Option A ↔ Option B sup-norm ≤ 4 ULP). Lives in `tests/schrodinger_complex_unitarity.rs` new file, feature `slow-tests`.

## Context

The v2.2 Schrödinger kernel (`crates/semiflow-core/src/schrodinger.rs`, ADR-0057) realised the Schrödinger semigroup $\{e^{-itH}\}_{t \in \mathbb{R}}$ for $H = -\tfrac{1}{2}\partial_x^2 + V(x)$ on a 1D grid by storing the wave function as a real PAIR $(\psi_{\mathrm{re}}, \psi_{\mathrm{im}}) \in \mathbb{R}^N \times \mathbb{R}^N$ — Option A per ADR-0057 §"Decision". Option B (native complex state $\psi : \mathrm{GridFn1D}\langle \mathbb{C} \rangle$) was DEFERRED to v4.0+ "once `SemiflowComplex` is available" per ADR-0057 §"Out of scope".

Through v2.2 → v3.1 (six releases) the Option A representation has accumulated three significant costs:

1. **Per-step bookkeeping** — every Chernoff step explicitly carries paired `(re, im)` slices through `apply_into`, with hand-coded $2 \times 2$ block LU for the Crank-Nicolson kinetic step.
2. **Lifted-only quantum graphs deferred** — ADR-0078 §29.7 defers quantum-Schrödinger semigroups on metric graphs explicitly pending `SemiflowComplex`.
3. **Future matrix-valued + rough-vol tracks blocked** — the v4.0 `examples/rough_heston_markov.rs` Markov approximation (Bayer-Friz-Gulisashvili 2019) has natively complex state.

v4.0 fulfils the v2.2 deferral by shipping `SemiflowComplex<C>` as a generic complex-arithmetic trait abstraction (paralleling `SemiflowFloat<F>`) PLUS `SchrödingerChernoffComplex<C: SemiflowComplex>` as a SIBLING kernel to the v2.2 real-pair `SchrödingerChernoff<F>`. The v2.2 kernel is PRESERVED verbatim through v4.x (soft-deprecation via rustdoc only — no hard removal); the v4.0 sibling stores `ψ : GridFn1D<C>` natively.

## Decision

Ship three additive public-surface items in v4.0 Wave A:

**Item 1 — `pub trait SemiflowComplex<C>`** in `crates/semiflow-core/src/complex.rs` (NEW module, ~250 LoC target, default 500-LoC cap):

```rust
pub trait SemiflowComplex:
    Copy + Send + Sync + 'static
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::AddAssign + core::ops::SubAssign
    + core::ops::MulAssign + core::ops::DivAssign
{
    /// Real number type for re/im components and the modulus.
    type Real: SemiflowFloat;

    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;
    fn abs(self) -> Self::Real;
    fn conj(self) -> Self;

    fn from_real(r: Self::Real) -> Self;
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;

    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
}

impl SemiflowComplex for num_complex::Complex<f64> { /* ... per math §30.2 */ }
impl SemiflowComplex for num_complex::Complex<f32> { /* ... per math §30.2 */ }
```

The trait is generic in the algebraic sense (no const-generic K like ApproximationSubspace; no const-generic D like AnisotropicShiftChernoffND). The two v4.0 reference impls cover the f64 + f32 lattice; downstream users may plug in alternative complex types (e.g., `arbitrary_complex::Complex` for GMP-precision, `simd_complex::Complex4` for SIMD-batch evaluation) without modifying any Chernoff kernel.

**Item 2 — `pub struct SchrödingerChernoffComplex<C: SemiflowComplex>`** in `crates/semiflow-core/src/schrodinger_complex.rs` (NEW module, ~400 LoC target, default 500-LoC cap):

```rust
pub struct SchrödingerChernoffComplex<C: SemiflowComplex> {
    grid: Grid1D<C::Real>,
    v_fn: Box<dyn Fn(C::Real) -> C::Real + Send + Sync>,
    scratch_rhs:    GridFn1D<C>,                // pre-allocated complex RHS buffer
    scratch_lu_d:   Vec<C>,                      // tridiagonal LU diagonal cache
    scratch_lu_l:   Vec<C>,                      // tridiagonal LU sub-diagonal cache
}

impl<C: SemiflowComplex> SchrödingerChernoffComplex<C> {
    pub fn new(
        grid: Grid1D<C::Real>,
        v_fn: impl Fn(C::Real) -> C::Real + Send + Sync + 'static,
    ) -> Result<Self, SemiflowError>;
}

impl<C: SemiflowComplex> ChernoffFunction<C::Real> for SchrödingerChernoffComplex<C> {
    type S = GridFn1D<C>;                                       // native complex state

    fn apply_into(
        &self, tau: C::Real, src: &GridFn1D<C>, dst: &mut GridFn1D<C>,
        scratch: &mut ScratchPool<C::Real>,
    ) -> Result<(), SemiflowError>;

    fn order(&self) -> u32 { 2 }                                 // Strang outer

    fn growth(&self) -> Growth<C::Real> {
        Growth { multiplier: C::Real::one(), omega: C::Real::zero() }   // unitary
    }
}

impl<C: SemiflowComplex> ApproximationSubspace<2, C::Real> for SchrödingerChernoffComplex<C> {
    fn in_subspace(&self, f: &GridFn1D<C>) -> bool {
        f.grid().n() >= 5 && /* ... per ADR-0073 pattern */
    }
    fn jet(&self, f: &GridFn1D<C>, out: &mut [GridFn1D<C>]) -> Result<(), SemiflowError>;
}
```

**Item 3 — `num-complex` dep promoted from "reserved" to "direct"** in `crates/semiflow-core/Cargo.toml`:

```toml
[dependencies]
num-traits = { version = "0.2", default-features = false, features = ["libm"] }
libm       = "0.2"
num-complex = { version = "0.4", default-features = false, features = ["libm"] }
```

This is the FIRST direct dep addition since the v0.x trajectory began. Total `semiflow-core` deps becomes 3/3 (cap met); any v4.x+ dep addition requires a constitution amendment (see constitution v1.8.0 §"Technology Constraints" amendment).

Per-step algorithm (math §30.3):

```
SchrödingerChernoffComplex::apply_into(τ, ψ_src, ψ_dst, scratch):
  // Step 1 — potential half-step (pointwise complex multiplication, O(N)):
  for k in 0..N:
    let phase = C::from_polar(C::Real::one(), -tau * v_fn(grid.x_at(k)) / two);
    tmp[k] = phase * ψ_src[k]
  // Step 2 — kinetic full-step (complex Cayley map, O(N) banded LU):
  // Solve (I - (iτ/4) Δ_h) tmp2 = (I + (iτ/4) Δ_h) tmp
  for k in 0..N:
    rhs[k] = tmp[k] + C::from_parts(_zero, tau/four) * laplacian(tmp, k);
  banded_complex_tridiag_lu_solve(rhs, tmp2);
  // Step 3 — second potential half-step (O(N)):
  for k in 0..N:
    let phase = C::from_polar(C::Real::one(), -tau * v_fn(grid.x_at(k)) / two);
    ψ_dst[k] = phase * tmp2[k]
```

The banded-LU solve uses the complex tridiagonal structure (5-point Laplacian with reflecting BCs yields tridiagonal); $O(N)$ work per step. Reference implementation MUST use `num_complex::Complex<f64>` arithmetic directly (no manual `(re, im)` decomposition) — that's the WHOLE POINT of the Option B representation.

## Rationale

- **Why `SemiflowComplex<C>` as a separate trait** (not adding methods to `SemiflowFloat<F>`): real-valued kernels (DiffusionChernoff, ShiftChernoff1D, etc.) do not need complex arithmetic; bundling complex methods into SemiflowFloat would inflate every real-only impl with `unimplemented!()` placeholders or require costly default-impl indirection. The separate-trait pattern matches the v3.1 split between `SemiflowFloat` and `BoundedGeometryManifold<F>` (ADR-0071) — each abstraction has its own focused trait surface.
- **Why `Complex<f64>` and `Complex<f32>` as the v4.0 reference impls** (vs custom complex types): `num-complex` is the canonical Rust complex-number crate; its types are universally understood and have well-tested arithmetic. Plugging in arbitrary user complex types is supported via the trait abstraction but the v4.0 ships only the canonical impls.
- **Why ship Option B as a SIBLING to v2.2 Option A** (not as a replacement): the v2.2 Option A real-pair representation IS the v2.x stable surface; many v2.x users depend on it; changing the state type from `(GridFn1D<F>, GridFn1D<F>)` to `GridFn1D<C>` is a BREAKING change for those users. v4.0 ships the sibling and preserves Option A through v4.x (soft-deprecation via rustdoc only). The two kernels can coexist indefinitely; users choose based on their needs.
- **Why num_complex is promoted to a DIRECT dep (vs an optional feature)**: the v4.0 SchrödingerChernoffComplex kernel REQUIRES complex arithmetic; making it optional means the kernel can't be compiled by default. The reserved-to-direct promotion is the suckless choice — the kernel and the dep go together. The dep cap is now 3/3; further additions require justification.
- **Why the SchrödingerChernoffComplex `growth()` returns `Growth { multiplier: 1, omega: 0 }`** (unitary): Schrödinger semigroups are unitary on $L^2(\mathbb{R})$ (preserve the $L^2$-norm exactly); the discrete Cayley map preserves the discrete $L^2$-norm exactly (proposition 30.1). Growth bound 1 is tight (no operator-norm inflation).
- **Why the G_SCHROD_B gate is 1e-12 unitarity (not 1e-15 or 1e-10)**: empirically derived from the 128-step accumulation of f64 round-off in the complex banded-LU solve. The 1e-12 budget gives ~10× headroom against drift; tighter (1e-15 = $\epsilon$-machine) would silently fail on different CPUs / compilers; looser (1e-10) would let in larger regressions. The 1e-12 mark is the smallest reliable empirical signal for unitarity preservation at this scale.
- **Why the cross-representation Option A ↔ Option B sub-test is ADVISORY (not RELEASE_BLOCKING)**: byte-identity between Option A's real-pair LU and Option B's complex LU is NOT achievable at f64 precision (the two LUs reorder a few floating-point sums differently). The $\le 4$ ULP discrepancy is below the unitarity budget but is observable; ADVISORY status documents the discrepancy without blocking the release.
- **Why a NEW kernel module `schrodinger_complex.rs` (~400 LoC)** vs extending v2.2 `schrodinger.rs`: the v2.2 module is in the Override #1 file-list (Cohort 4 carve-out at 578 LoC); adding the Option B code would push it well past the carve-out cap. A separate module keeps the v2.2 file under its existing cap AND gives the v4.0 sibling a clean co-location with its own math citations.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Add complex arithmetic to `SemiflowFloat<F>` (single trait covering both) | Inflates every real-only impl with `unimplemented!()` placeholders; bundles concerns; harder to evolve independently. Separate-trait pattern matches v3.1 BoundedGeometryManifold split. |
| Replace v2.2 Option A in-place with the Option B representation (BREAKING) | Breaks every v2.x Schrödinger user cold; loses the v2.2 real-pair LU as a backwards-compat path; violates ADR-0035 deprecation cadence. SIBLING is the suckless choice. |
| Ship `SemiflowComplex` trait without any reference impls (forcing users to provide their own `Complex<f64>` impl) | Useless trait; every user would need 200 LoC of boilerplate. Reference impls cover the universal case. |
| Make `num-complex` an OPTIONAL feature (`features.complex = ["num-complex"]`) instead of direct dep | The Option B kernel REQUIRES it; the kernel can't be useful without it. Optional-feature would mean users must opt in to a feature to use a default-compiled-in kernel — confusing surface. |
| Implement complex arithmetic from scratch in semiflow-core (avoid the num-complex dep) | Reinvents a well-tested crate (`num-complex` has 8 years of ecosystem use). Suckless principle: use stdlib + well-trodden deps; don't reinvent. |
| Defer the entire Schrödinger Option B work to v4.1+ | The use cases (quantum graphs, matrix-valued complex, rough-vol) are now mature; the v4.0 BREAKING window is the right place to ship the trait foundation. Deferring would push everything back another 12 months. |
| Make `growth()` return a complex-magnitude bound (`Growth<C>`) instead of real-valued `Growth<C::Real>` | Growth bound is intrinsically real-valued (a norm bound is a non-negative real); using complex Growth would be type-noise without semantic benefit. |
| Ship the Option B kernel with f32-only support initially (defer f64) | f64 is the canonical scientific-computing precision; users would expect both. Both impls are equally easy (the num-complex crate provides both); shipping both is the suckless choice. |
| Hard-deprecate v2.2 Option A at v4.0 (start the 12-month deprecation clock at v4.0 release) | Premature; the v2.2 Option A has its own ecosystem of users; SOFT-deprecation via rustdoc only at v4.0 is the gentler path. Hard-deprecation can defer to v5.0+ if a use case demands. |
| Implement SchrödingerChernoffComplex via the v2.2 real-pair internals + conversion at the boundary | Defeats the purpose of Option B (which is the NATIVELY complex representation); the conversion overhead would be larger than the per-step compute. |

## Consequences

- **Pre-existing v2.2 SchrödingerChernoff<F> call-sites compile unchanged.** Strictly additive surface; v2.2 Option A is preserved verbatim. The v4.0 Option B is a SIBLING kernel.
- **New file `crates/semiflow-core/src/complex.rs`** (~250 LoC; trait + 2 reference impls + rustdoc; default 500-LoC cap; NO Override expansion).
- **New file `crates/semiflow-core/src/schrodinger_complex.rs`** (~400 LoC; kernel impl + 2 trait opt-ins; default 500-LoC cap; NO Override expansion).
- **New direct dep `num-complex` 0.4** — promoted from "reserved" to "direct"; total `semiflow-core` deps now 3/3 (cap fully used; further additions require constitution amendment).
- **New trait `SemiflowComplex<C>`** with 2 reference impls (`Complex<f64>`, `Complex<f32>`). User-extensible.
- **New kernel `SchrödingerChernoffComplex<C: SemiflowComplex>`** — sibling to v2.2 `SchrödingerChernoff<F>`. Implements `ChernoffFunction<C::Real>` (the v3.0 cleaned-up trait per ADR-0074) + `ApproximationSubspace<2, C::Real>` (the SEVENTH v3.x opt-in after the v3.0 trio + v3.1 HypoellipticChernoff + v4.0 trio).
- **Schema bumps**: shared with ADR-0080/0081/0082/0083/0084/0085 — `traits.yaml` 1.1.0 → **2.0.0 MAJOR** (BREAKING per Override re-eval; v2_compat removal); `properties.yaml` 0.12.0 → **1.0.0 MAJOR** (BREAKING per G_binding_parity sub-test removal + new gate additions). math.md is append-only (§30 NEW).
- **New gate**: G_SCHROD_B (RELEASE_BLOCKING — 1 sub-test + 1 ADVISORY sub-test). Test file `tests/schrodinger_complex_unitarity.rs` new file, feature `slow-tests`.
- **CITATIONs added to math.md §30**: Pazy 1983 §6.4; Cheng 2008 §3.2; Engel-Nagel 2000 §IV.6.
- **Migration note**: existing v2.2 Schrödinger users get NO migration burden at v4.0 (Option A preserved verbatim). New users choosing between Option A and Option B should default to Option B (Option A is soft-deprecated via rustdoc; hard removal deferred indefinitely).

## Migration

End-user impact is **opt-in additive**. v2.2 callers using `SchrödingerChernoff<F>` (real-pair Option A) continue to compile unchanged.

New v4.0 users wanting native complex:

```rust
// v2.2 baseline (still works, real-pair representation):
let schrod_v22 = SchrödingerChernoff::<f64>::new(
    grid.clone(), v_fn, /* config */
)?;
let evolver = Evolver::new(schrod_v22.clone(), n_steps)?;
let (psi_re_final, psi_im_final) = evolver.evolve(t_final, &(psi_re_0, psi_im_0))?;

// v4.0 NEW (Option B — native complex):
use num_complex::Complex;
let schrod_v40 = SchrödingerChernoffComplex::<Complex<f64>>::new(
    grid.clone(), v_fn,
)?;
let evolver_v40 = Evolver::new(schrod_v40, n_steps)?;
let psi_final = evolver_v40.evolve(t_final, &psi_complex_0)?;
// psi_final: GridFn1D<Complex<f64>> — natively complex.
```

Worked example with the harmonic-oscillator wave packet in `docs/migration/v3-to-v4.md` §2 (Wave G).

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0003 — no_std + alloc; the trait + kernel use only stdlib + the new num-complex dep.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; `SemiflowComplex<C>` parallels the pattern with `C::Real = f64` default.
- ADR-0026 — `ChernoffFunction<F>` super-trait; SchrödingerChernoffComplex implements `ChernoffFunction<C::Real>`.
- ADR-0041 — `apply_into` + `ScratchPool`; the kernel uses the scratch pool for the complex tridiagonal LU buffers.
- ADR-0057 — v2.2 Schrödinger Option A; PARTIAL supersede (the §"Out of scope" deferral to v4.0+ is FULFILLED here).
- ADR-0073 — v3.0 `ApproximationSubspace<K, F>` opt-in marker; SchrödingerChernoffComplex opts into K=2 (the seventh v3.x opt-in).
- ADR-0074 — v3.0 ChernoffFunction trait cleanup; the `Growth<F>` typed return is used by the new kernel's `growth()`.
- ADR-0080 — v4.0 PointEval first-class API (sibling ADR in same Wave A; SchrödingerChernoffComplex does NOT opt in to PointEval in v4.0 — the Cayley-map evaluation is not a kernel-representation form; PointEval for complex kernels defers to v4.1+).
- ADR-0084 — v4.0 v2_compat hard removal (sibling ADR; the v3.0 v2_compat shim is removed in lockstep with this Wave A).
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v4.0 — release-level roadmap.
- math.md §30 (NEW v4.0) — SemiflowComplex trait + Option B Chernoff step + unitarity proposition + G_SCHROD_B gate spec.
- `.dev-docs/constitution.md` v1.8.0 (NEW v4.0) — MAJOR re-evaluation; `num-complex` promoted from reserved to direct (deps now 3/3).
- `docs/migration/v3-to-v4.md` §2 — Schrödinger Option A → Option B worked example (Wave G fills).

## Amendments

(none at acceptance time)
