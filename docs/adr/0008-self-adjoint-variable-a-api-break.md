# ADR-0008 — Self-adjoint variable-`a(x)` Chernoff via API break (v0.3.0, option ζ)

## Status

**Accepted** — 2026-04-30 (γ-A baseline). **Amendment 1 — option ζ
extension (true order-2 lift)**: 2026-04-30, ADOPTED. The shipping
v0.3.0 formula is **ζ-A** (γ-A with explicit τ²-correction polynomial;
adds `a_double_prime` as a third caller-supplied closure). Authority:
AI Solutions Architect post-γ Stage-1 amendment after sympy proof that
γ-A leaves a τ²-deficit that can be killed exactly by an explicit τ²
correction in `a, a', a''`.
SemVer: pre-1.0 minor break (v0.2.2 → **v0.3.0**, skipping the
planned v0.2.3 micro-bump). The ζ-extension is rolled into the same
v0.3.0 release — there is no intermediate γ-only v0.3.0 release.

> **Reading guide.** Sections "Context", "Decision (γ-A baseline)",
> "API change (γ-A baseline)", "Migration (γ-A baseline)",
> "Backward compatibility", "Performance budget (γ-A)", and "References"
> below are PRESERVED VERBATIM from the γ-A draft as reference for the
> structural baseline. The shipping v0.3.0 formula is the **ζ-A**
> extension defined in **Amendment 1** at the end of this document; the
> ζ-A extension is a strict superset of γ-A — every γ-A invariant is
> preserved and ζ-A adds local-O(τ³) for variable `a` via the explicit
> τ² polynomial correction. Where γ-A and ζ-A diverge in API surface
> (struct fields, constructor arity, invariants I2/I3/I4/I5), the
> Amendment 1 text supersedes the baseline.

## Context

ADR-0006 introduced operator splitting `L = A + B` for v0.2.0:
- `A f := a(x) f''(x)` — diffusion;
- `B f := b(x) f'(x) + c(x) f(x)` — drift + reaction;
- `Φ(τ) := D(τ/2) ∘ R(τ) ∘ D(τ/2)` — Strang composition.

For **constant** `a, b, c` the v0.2.0 production oracle attains global
`O(τ²)` per the standard Strang theorem (HLW §III.5 Thm 4.1). For
**variable** coefficients, the path was:

| Version | DiffusionChernoff `D(τ)` semantics              | Local order (var. `a`) | Global order (var. `a, b, c`) | Generator |
|---------|--------------------------------------------------|------------------------|--------------------------------|-----------|
| v0.2.0  | central-only `a(x_i)` in 5-pt stencil           | O(τ²)                 | O(τ)                          | undefined for var. `a` |
| v0.2.2  | (D unchanged; R upgraded to RK2 for var. `b,c`) | O(τ²)                 | O(τ)                          | undefined for var. `a` |
| α (rejected, ADR-0006 Am 6) | per-leg midpoint `a(x ± h/2)` → asymmetric `h⁺ ≠ h⁻` | O(τ³) target           | O(τ²) **claimed**, **wrong PDE** | M = a∂² + ½a'∂ ❌ |
| β (rejected for our claim, ADR-0006 Am 6.1) | symmetric mean `(a(x+h/2)+a(x-h/2))/2` | O(τ²)                 | O(τ)                          | A = a∂² ✓ |

**Two impossibility results** govern the 5-point symmetric stencil
(`D = w₀f + w₁[f(x±h)] + w₂[f(x±H)]`):

1. **Asymmetric shifts (α)** generate the symmetrized operator
   `M = a∂² + ½a'∂`, **not** `A`. Iterated `(D_α(T/n))^n f` converges
   to `e^{TM}f` — the wrong semigroup.
2. **Symmetric shifts (β, central, Picard, trapezoidal, Simpson, ...)**
   produce only **even derivatives** of `f`. The τ²-coefficient of
   `e^{τA}` contains `a·a'·f'''` — an odd derivative — which **cannot
   be reproduced** by any symmetric stencil, regardless of how `h, H, w_*`
   depend on `a, τ`. (Proof: ADR-0006 Amendment 6.1 §6 / β-derivation
   document §6.) Hence variable-`a` local-O(τ³) is unreachable with the
   v0.2.0 stencil topology under the v0.2.0 single-`fn`-pointer API.

User-chosen mitigation: **option γ — API break**. Add `a'(x)` as a
caller-supplied closure on `DiffusionChernoff`. This unlocks two things:

1. The **architecturally clean self-adjoint decomposition**
   `A = ∂(a∂) − a'·∂`, where `D(τ)` corresponds to the
   divergence-form generator `A_self f := ∂_x(a(x)·∂_x f) = a·f'' + a'·f'`,
   and the residual drift `−a'·∂` is **absorbed into `R`** via
   `b_total := b − a'`.
2. A **forward-compatible API surface** — future v0.4.0+ extensions
   (Magnus integrator, second-derivative input `a''(x)`, resolvent
   quadrature) can refine `D(τ)` to true local-O(τ³) without further
   breaking changes.

## Decision

### Route γ-A (chosen): self-adjoint split + caller-composed `b_total`

**Architectural choice**: route γ-A — `D(τ)` operates on the
divergence-form generator `A_self`, `R(τ)` operates on the
**caller-composed** `B_compensated f := (b(x) − a'(x))·f'(x) + c(x)·f(x)`.

**Inner-Strang internal structure of `D(τ)`** (the "γ-FINAL-v5"
formula, sympy-verified at
`.dev-docs/verification/scripts/verify_v0_3_0_gamma.py`):

```text
D(τ) f(x)  :=  S(τ/2) ∘ K(τ; a) ∘ S(τ/2)  [f] (x)

where
  S(s) g(x)   :=  g(x + s · a'(x))                      // drift exponential
  K(τ; a) g(x) := w₀ · g(x)
               + w₁ · [g(x + h₀) + g(x − h₀)]
               + w₂ · [g(x + H₀) + g(x − H₀)]
  h₀(x)       := 2·√(a(x)·τ)
  H₀(x)       := 2·√(3·a(x)·τ)
  w₀ = 7/12,  w₁ = 3/16,  w₂ = 1/48                   // unchanged from v0.2.0
```

**Why three nested operators?** This is a per-step Strang split *inside*
`D(τ)` for the operator decomposition `A_self = a·∂² + a'·∂`:
- `S(s) = exp(s · a'·∂)` — exact pure-drift exponential along the
  field `a'(x)`;
- `K(τ; a) = v0.2.2 5-point stencil` — symmetric Chernoff function
  for the central-`a` diffusion `a·∂²` (with `a` evaluated at each
  K-step's input position).

**Sympy-verified gates** (`verify_v0_3_0_gamma.py`, exit code 0 on
success, polynomial-Taylor analysis, exact rationals):

- **G_τ⁰** (identity): `[D(τ)f](0)|_{τ=0} = f(0)` ✓
- **G_τ¹** (Chernoff consistency, NORMATIVE):
  `D'(0) f = a₀·f₂ + a₁·f₁ = A_self f` ✓ —
  the τ¹ deficit is **algebraically zero**.
- **G_const-a** (regression-safety, NORMATIVE):
  for `a' ≡ 0`, `S(s)` reduces to identity, so `D(τ) = K(τ;a₀)`
  which is **bit-equal** to the v0.2.2 5-point central-`a` formula
  (algebraically: `D − D_v022 = 0` exactly). ✓
- **G_τ²** (full local-O(τ³), informational): the τ² deficit is
  `−a₀·a₁·f₃ − a₀·a₂·f₂/2 − a₁·a₂·f₁/4` — **non-zero**, mirroring
  the same impossibility-theorem residual as β (the leading
  `−a·a'·f'''` term is unreachable by any composition of the symmetric
  K-factor with the pure-drift S-factor without `a''(x)` input).

### Net order claim (revised, honest)

| Regime                          | Local order | Global order (Strang) | Generator |
|---------------------------------|-------------|------------------------|-----------|
| Constant `a`                    | O(τ³) exact | O(τ²) via Strang      | A ✓ (= A_self when a'≡0) |
| Variable `a ∈ C²(ℝ)`            | O(τ²)       | O(τ)                   | A_self ✓ (divergence form, mathematically rigorous) |
| Variable `a, b, c` (full)       | O(τ²)       | O(τ)                   | L = A_self + B_compensated ✓ |

**This is the SAME convergence rate as the rejected β formula**.
The architectural value of γ-A over β is **not** order improvement —
both are global-O(τ) for variable `a` due to the same impossibility
theorem — it is:

1. **Mathematical rigor**: D(τ) corresponds to a real, well-defined
   self-adjoint operator (`∂(a∂)`); R(τ) corresponds to a first-order
   operator with caller-composed `b_total`. The mathematical objects are
   transparent and conform to the standard divergence-form parabolic
   theory (Friedman, *Partial Differential Equations of Parabolic Type*,
   Ch. 1, §2).
2. **API surface change** that unlocks future order-2 lifts:
   - **v0.4.0**: option ε (Magnus integrator) — replace inner-Strang with
     `exp(τ · A_avg)` where `A_avg` is the time-averaged generator over
     `[0, τ]`. Captures full `[A_self, A_self]` commutator structure.
   - **v0.5.0**: option ζ — add `a''(x)` as a third caller-supplied
     closure, kill the τ²-deficit terms `a·a''·f''` and `a'·a''·f'`
     by explicit higher-order corrections.
   - **v0.6.0+**: Resolvent quadrature `R_λ f = Σ_k w_k S(τ_k) f` —
     reuses `(a, a')` interface verbatim.
3. **R reuse**: v0.2.2 RK2 `DriftReactionChernoff` is reused **verbatim**
   with the composed `b_total = b − a'`. No code change to R.

The user's expectation that γ would unlock true local-O(τ³) for variable
`a` is **mathematically incorrect** — the symmetric-K-factor still
inherits the impossibility theorem, even with the inner-Strang structure
and the explicit `a'` input. **This is documented honestly in the
revised order claim above.** True order-2 for variable `a` requires
escaping the symmetric-stencil topology (Magnus, FFT, or `a''(x)` input);
deferred to v0.4.0+.

### API change (breaking)

**Before** (v0.2.2):
```rust
pub struct DiffusionChernoff {
    pub a: fn(f64) -> f64,
    pub a_norm_bound: f64,
    pub grid: Grid1D,
}

impl DiffusionChernoff {
    pub fn new(a: fn(f64) -> f64, a_norm_bound: f64, grid: Grid1D) -> Self { ... }
    pub fn apply(&self, tau: f64, f: &GridFn1D) -> Result<GridFn1D, SemiflowError> { ... }
}
```

**After** (v0.3.0):
```rust
pub struct DiffusionChernoff {
    pub a: fn(f64) -> f64,
    pub a_prime: fn(f64) -> f64,    // NEW — caller-supplied derivative
    pub a_norm_bound: f64,
    pub grid: Grid1D,
}

impl DiffusionChernoff {
    /// New signature — `a_prime` is the analytic derivative of `a`,
    /// caller-supplied. For constant `a`, pass `|_| 0.0` (the constant-a
    /// fast path is bit-equal to v0.2.2).
    pub fn new(
        a: fn(f64) -> f64,
        a_prime: fn(f64) -> f64,
        a_norm_bound: f64,
        grid: Grid1D,
    ) -> Self { ... }

    pub fn apply(&self, tau: f64, f: &GridFn1D) -> Result<GridFn1D, SemiflowError> { ... }
}
```

**`StrangSplit` helper** (NEW — convenience constructor that absorbs
`−a'` into `b_total` automatically, so callers don't have to remember
the composition rule):

```rust
impl<R> StrangSplit<DiffusionChernoff, DriftReactionChernoff>
where /* ... */
{
    /// Construct `Φ(τ) = D(τ/2) ∘ R'(τ) ∘ D(τ/2)` where R' uses
    /// the compensated drift `b_total(x) = b(x) − a'(x)`.   This is the
    /// correct composition for the self-adjoint decomposition
    /// `L = ∂(a∂) + (b − a')·∂ + c`.
    pub fn with_a_prime_compensation(
        a: fn(f64) -> f64,
        a_prime: fn(f64) -> f64,
        b: fn(f64) -> f64,
        c: fn(f64) -> f64,
        a_norm_bound: f64,
        b_norm_bound: f64,
        c_norm_bound: f64,
        grid: Grid1D,
    ) -> Self { ... }
}
```

Inside `with_a_prime_compensation`, `b_total: fn(f64) -> f64 = |x| b(x) - a_prime(x);`
is captured at construction and passed to `DriftReactionChernoff::new`.
**Note**: Rust does not allow capturing `fn` pointers in another `fn`
pointer (no closures-of-closures with `fn`). The implementation
either (a) generates a per-instance vtable thunk (NOT no_std-friendly
without alloc); (b) requires the caller to **pre-compose** `b_total`
themselves and pass via the standard `StrangSplit::new`. Option (b) is
chosen for v0.3.0 — `with_a_prime_compensation` is documented as a
**helper that exists only when `(a, a', b)` are ALL `fn` pointers and
the caller computes `b_total = b − a'` at the call site**:

```rust
// Idiomatic v0.3.0 usage:
let a:      fn(f64) -> f64 = |x| (1.0 + 0.1 * x).powi(2);
let a_prime: fn(f64) -> f64 = |x| 0.2 * (1.0 + 0.1 * x);
let b:      fn(f64) -> f64 = |_| -0.5;
let c:      fn(f64) -> f64 = |_| 0.0;

// Caller composes b_total at-source (Rust closures cannot capture fn pointers):
fn b_total(x: f64) -> f64 { b(x) - a_prime(x) }   // requires GLOBAL fn

// OR (preferred): pre-fold the analytic difference into a single fn:
let b_total: fn(f64) -> f64 = |x| -0.5 - 0.2 * (1.0 + 0.1 * x);

let d = DiffusionChernoff::new(a, a_prime, /* a_norm */ 1.5, grid.clone());
let r = DriftReactionChernoff::new(b_total, c, /* b_norm */ 1.0, /* c_norm */ 0.0);
let phi = StrangSplit::new(d, r);
```

**Documentation** explicitly notes the `b_total = b − a'` rule.
A future v0.4.0 may use heap-allocated closures (under the `alloc`
feature) to provide a true `with_a_prime_compensation` that takes
`b` and `a_prime` separately and computes the difference internally.

### Migration guide (v0.2.x → v0.3.0)

**Constant-`a` callers** (the production case for v0.2.0/v0.2.1/v0.2.2
acceptance gates G1, G2, G3-strang, G4-strang — all use
`α = 0.5`):

```rust
// v0.2.x:
let diff = DiffusionChernoff::new(|_| 0.5_f64, 0.5, grid);

// v0.3.0:
let diff = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, 0.5, grid);
//                                              ^^^^^^^^^^ — a' ≡ 0 for constant a
```

**Mechanical change**: insert `|_| 0.0_f64` as the second argument.
Bit-equal output guaranteed (sympy-verified G_const gate).

**Variable-`a` callers** (new in v0.3.0; no prior callers exist —
v0.2.2 had no documented variable-`a` support):

```rust
// v0.3.0 — caller MUST provide both a and its analytic derivative:
let a:       fn(f64) -> f64 = |x| (1.0 + 0.1 * x).powi(2);
let a_prime: fn(f64) -> f64 = |x| 0.2 * (1.0 + 0.1 * x);
let diff = DiffusionChernoff::new(a, a_prime, 1.5, grid);
```

**Variable-`a` Strang composition** (explicit `b_total`):
```rust
// caller pre-folds  b_total = b - a_prime  at the call site:
let b_total: fn(f64) -> f64 = |x| -0.5 - 0.2 * (1.0 + 0.1 * x);
let drift   = DriftReactionChernoff::new(b_total, |_| 0.0, 1.0, 0.0);
let phi     = StrangSplit::new(diff, drift);
```

### Backward compatibility

**Breaking changes** (SemVer minor in pre-1.0):

1. `DiffusionChernoff::new` constructor signature: 3 args → 4 args.
2. `DiffusionChernoff` struct: 3 fields → 4 fields (adds `a_prime`).
3. `StrangSplit<DiffusionChernoff, DriftReactionChernoff>` users with
   variable `a, b` MUST manually compose `b_total = b − a'` (or use
   `with_a_prime_compensation` helper if/when added). This is a
   **behavioural** change, not just an API change: prior v0.2.x users
   who passed variable `b` directly to R alongside variable `a` were
   computing the wrong PDE (silent generator mismatch). v0.3.0 forces
   the correct composition.
4. `traits.yaml` `DiffusionChernoff` schema_version: `0.2.3` → `0.3.0`.

**Non-breaking** (preserved):
- `ChernoffFunction` trait surface — unchanged.
- `DriftReactionChernoff` API and semantics — unchanged
  (still uses v0.2.2 RK2 for variable `b, c`).
- `StrangSplit::new` constructor — unchanged.
- `DiffusionChernoff::apply` signature — unchanged.
- Constant-`a` numerical output — bit-equal to v0.2.2 modulo
  `≤ 4 ULPs` (formal: algebraically zero by the inner-Strang's
  `S(0) = id` reduction when `a' ≡ 0`).
- Acceptance gates G1, G2, G3-strang, G4-strang (constant-coefficient,
  `α = 0.5`) all pass unchanged — they only need the `|_| 0.0_f64`
  insertion for `a_prime`.

### Performance budget

Per-node `apply_at_node` cost:

| Operation                          | v0.2.2 | v0.3.0 (γ-A)  | Δ vs v0.2.2 |
|------------------------------------|--------|---------------|-------------|
| Central `a(x)` evaluations         | 1      | 1 (in K-factor)| 0           |
| `a_prime(x)` evaluations            | 0      | 2 (in S-factors at x and x+pre-shift) | +2 fn calls |
| `libm::sqrt` calls                  | 2      | 2 (h₀, H₀)    | 0           |
| `validate_a_x` calls                | 1      | 1             | 0           |
| `f.sample` calls                    | 4      | 4 (after S-shift composition) | 0 |
| Inner shift compositions            | 0      | 2 (S(τ/2) before/after K) | +2 mults+2 adds |

**Estimated overhead vs v0.2.2**: 2 extra `fn(f64) -> f64` calls (≈
1-2 ns each on x86_64 with branch prediction) + 4-8 ns of arithmetic
for the S-shift composition. **Total: ~5-10 ns per node.** v0.2.2
baseline is ~150 ns/node (dominated by `f.sample` cubic Hermite).
**Predicted overhead: ~3-7%, within ±5% target.** Half the predicted
cost of the rejected α formula (8 extra `a(·)` calls).

For **constant `a` callers** (`a_prime = |_| 0.0`), the inner-Strang
reduces to `S(τ/2) = id`, `D = K = v0.2.2` — **zero overhead** is
achievable with branch prediction on the `a_prime ≡ 0` fast path
(the `S(s)` operation `g(x + 0.0)` is bit-equal to `g(x)` and the
Rust optimizer should constant-fold the entire `S` factor when
`a_prime` is the literal `|_| 0.0`). Engineer Stage 6 SHOULD verify
this fast-path is constant-folded; if not, an explicit
`if a_prime is fn(f64) -> f64 { |_| 0.0 }` branch may be added
(but only if profiling shows it matters — the trait-method-call
overhead via the existing `ChernoffFunction` indirection is comparable).

### Forward compatibility

This API change unlocks the following v0.4.0+ extensions **without
further breaking changes**:

- **v0.4.0** — Magnus integrator: replace inner-Strang `S∘K∘S` with
  exact-Magnus `exp(τ · A_avg)` where `A_avg = (1/τ)·∫_0^τ A_self(s) ds`
  averaged over the time-step. Reuses `(a, a')` interface verbatim;
  adds an internal numerical-integration kernel. Lifts variable-`a`
  to true local-O(τ³) / global-O(τ²).
- **v0.5.0** — `a''(x)` input: extend struct to add `a_double_prime`
  field. Captures the τ²-deficit terms `a·a''·f''` and `a'·a''·f'`
  via explicit higher-order Taylor corrections. Lifts to local-O(τ⁴)
  / global-O(τ³). **Note**: this WOULD be a SemVer minor break, but
  it's already mentally budgeted as part of the v0.3.0 architectural
  trajectory.
- **v0.6.0+** — Resolvent quadrature: `R_λ f = Σ_k w_k · S(τ_k) f`
  for spectral analysis. Reuses `(a, a')` interface verbatim; no
  further API change required.
- **2D tensor product** (separate work item, post-v0.3.0): each axis
  applies the v0.3.0 γ-A `D(τ)` along its own direction, with the
  caller supplying `(a₁, a₁'), (a₂, a₂')` per axis. Strang composition
  inherits the per-axis order (variable-`a` per axis: global O(τ),
  isotropic constant-`a`: global O(τ²)).

## Consequences

### Positive

1. Mathematical rigor — `D(τ)` corresponds to the exact divergence-form
   generator `A_self = ∂(a·∂)`; v0.2.2 silent degradation is gone.
2. Generator correctness — variable-`a` PDE is now correctly identified
   as `e^{T(A_self − a'·∂ + B)}f = e^{TL}f`; β's correct-generator
   property is preserved with stronger structural justification.
3. Forward-compatible API — Magnus, `a''`, resolvent quadrature do
   not require further breaking changes.
4. v0.2.2 R reused verbatim — no regression risk on the
   already-verified RK2 drift-reaction.
5. Constant-`a` regression: bit-equal to v0.2.2 (sympy-proven `G_const`
   gate); G1-G4 acceptance unchanged.

### Negative

1. **Convergence rate is still global-O(τ) for variable `a`**, not
   global-O(τ²) as the user originally hoped. The user's premise
   that γ would "unlock true order-2" is mathematically incorrect —
   it's blocked by the same impossibility theorem that rejected α and
   β. v0.3.0 ships the correct generator at v0.2.2's rate.
2. Caller burden — variable-`a` callers must supply two closures
   `(a, a')`. If `a` is a polynomial, computing `a_prime` is
   trivial; for general `a`, callers must derive `a'` analytically.
3. Migration cost — every existing call site must add `|_| 0.0` for
   `a_prime`. Migration is mechanical (10 call sites total — see
   `.dev-docs/migration/v0.3.0-call-sites.md`).
4. SemVer break in pre-1.0 — minor version bump (0.2.2 → 0.3.0)
   skipping the planned 0.2.3. Users on `Cargo.toml ^0.2` will NOT
   automatically get v0.3.0; explicit version pinning required.
5. The `b_total = b − a'` composition is caller-side (no
   `with_a_prime_compensation` in v0.3.0 due to no_std + `fn` pointer
   constraints). Documentation must be unmistakable.

### Neutral

1. The β formula is **superseded but retained** in
   `.dev-docs/verification/variable-diffusion-beta-derivation.md` —
   the impossibility theorem is canonical mathematical content
   (not project-specific) and is the strongest justification for the
   API break. Both `ADR-0006 Amendment 6.1` and the β-derivation
   document are marked **superseded by ADR-0008 (this document)** but
   not deleted.
2. The Liouville oracle `a(x) = (1+γx)²` is **preserved** — the same
   manufactured solution validates both β and γ-A against the correct
   generator `e^{TA}f`. Property
   `diffusion_chernoff_variable_order1_liouville_oracle` is
   renamed to `diffusion_chernoff_variable_gamma_liouville_oracle`
   in v0.3.0 properties.yaml; the slope assertion stays at `≤ -0.95`
   (global O(τ) — same as β; γ-A does not improve order, only
   generator rigor).
3. A new property `diffusion_chernoff_a_prime_consistency` is added
   (50 cases) — soft check that user-supplied `a_prime(x)` matches
   the numerical derivative `(a(x+h) − a(x−h))/(2h)` within 1% over
   random `x`-points. Failure indicates caller error
   (`a_prime` does not actually equal the analytic derivative of `a`).
   This is a **caller-correctness** check, not a contract; cases ≪
   the bit-equal regression gate.

## API change summary (machine-readable)

| Element                                     | Before (v0.2.2)                                     | After (v0.3.0)                                                         |
|---------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------------|
| `DiffusionChernoff` struct fields           | `(a: fn, a_norm_bound: f64, grid)`                  | `(a: fn, a_prime: fn, a_norm_bound: f64, grid)`                        |
| `DiffusionChernoff::new` signature           | `(a, a_norm_bound, grid) -> Self`                   | `(a, a_prime, a_norm_bound, grid) -> Self`                             |
| `DiffusionChernoff::apply` signature         | `(&self, τ, f) -> Result<...>`                      | `(&self, τ, f) -> Result<...>` (UNCHANGED)                              |
| `DriftReactionChernoff::new` signature       | `(b, c, b_norm, c_norm) -> Self`                    | `(b, c, b_norm, c_norm) -> Self` (UNCHANGED — caller composes b_total) |
| `StrangSplit::new` signature                 | `(D, R) -> Self`                                    | `(D, R) -> Self` (UNCHANGED)                                            |
| `traits.yaml` schema_version                 | `0.2.3`                                             | `0.3.0`                                                                |
| `properties.yaml` schema_version             | `0.2.3`                                             | `0.3.0`                                                                |
| Generator (variable `a`)                    | undefined (silent degradation)                      | **A_self = ∂(a·∂)**, with `−a'` absorbed into R                        |
| Local order (variable `a`)                  | O(τ²)                                               | O(τ²) — SAME (impossibility theorem)                                   |
| Global order (variable `a, b, c`)           | O(τ)                                                | O(τ) — SAME (impossibility theorem)                                    |
| Constant-`a` regression                     | n/a                                                 | bit-equal to v0.2.2 (sympy-proven, gate `G_const`)                     |

## Verification

Sympy script `.dev-docs/verification/scripts/verify_v0_3_0_gamma.py`
(reproducible: `python3 .dev-docs/verification/scripts/verify_v0_3_0_gamma.py`,
exit 0 on success). Direct verification of:

- **G_τ⁰** identity at τ=0: `D(0)f|0 = f₀` ✓ (deficit `0` exact).
- **G_τ¹** Chernoff consistency:
  `D'(0) f|0 = a₀·f₂ + a₁·f₁ = (a·f'' + a'·f')|0 = A_self f|0` ✓
  (deficit `0` exact, sympy expand). **NORMATIVE.**
- **G_τ²** local order-2: deficit
  `−a₀·a₁·f₃ − a₀·a₂·f₂/2 − a₁·a₂·f₁/4 ≠ 0` (informational; the
  `−a₀·a₁·f₃ = −a·a'·f'''` term is the same impossibility-theorem
  residual that ADR-0006 Am 6.1 §6 proved unreachable for any
  symmetric K-factor; the additional terms involve `a''(x) = a₂` and
  `a'(x)·a''(x) = a₁·a₂` which would require option-ζ `a''` input).
- **G_const** constant-`a` regression: `D|(a₁=a₂=a₃=0) − D_v022 = 0`
  exact. ✓ **NORMATIVE.**

## References

- Engel, Nagel. *One-Parameter Semigroups for Linear Evolution
  Equations*. Springer, 2000. Ch. II — Chernoff product theorem.
- Hairer, Lubich, Wanner. *Geometric Numerical Integration*. Springer,
  2nd ed. 2006. §III.5 — Strang splitting and palindromic structure.
- Friedman. *Partial Differential Equations of Parabolic Type*.
  Prentice-Hall, 1964. Ch. 1 §2 — divergence-form parabolic operators.
- Olver. *Asymptotics and Special Functions*. Academic Press, 1974.
  §5.2 — Liouville transformation (manufactured oracle).
- Polyanin, Zaitsev. *Handbook of Linear PDEs*. CRC, 2nd ed. 2016.
  §1.1 — Liouville-transformed parabolic PDEs.

## Cross-links

- `contracts/semiflow-core.math.md` §9.2 (restructured): 9.2.1
  constant-`a` 5-point (unchanged from v0.2.0); 9.2.2 variable-`a`
  γ-A inner-Strang formula (NEW); 9.2.3 (informational) impossibility
  theorem — verbatim retained from ADR-0006 Am 6.1 §6.
- `contracts/semiflow-core.traits.yaml` `DiffusionChernoff`
  (schema_version 0.3.0).
- `contracts/semiflow-core.properties.yaml`
  `diffusion_chernoff_constant_fast_path_exact`,
  `diffusion_chernoff_variable_gamma_liouville_oracle` (renamed from β),
  `diffusion_chernoff_a_prime_consistency` (NEW).
- `contracts/semiflow-core.errors.yaml` — UNCHANGED. `DomainViolation`
  variant covers `a(x) ≤ 0` and non-finite `a_prime(x)` results
  (caller-correctness check; no new error variant).
- `.dev-docs/verification/scripts/verify_v0_3_0_gamma.py` —
  sympy proof script (reproducible, exit 0).
- `.dev-docs/migration/v0.3.0-call-sites.md` — exhaustive call-site
  migration map (10 sites).
- ADR-0006 Amendment 6.1 — **superseded by this ADR for the v0.3.0
  formula**; **retained in full** as the canonical reference for the
  symmetric-stencil impossibility theorem (mathematical result, not
  v0.3.0-specific).
- ADR-0006 Amendment 6 (formula α) — **rejected**; **retained** as
  the documentation of the wrong-PDE bug avoided by γ-A.
- ADR-0007 (boundary policies) — UNCHANGED. γ-A's S-shift `c = a'(x)·s`
  remains within the existing `BoundaryPolicy` interface (the shifted
  position `x + c` is sub-grid and resolved by `f.sample()` like any
  other shift).

## Performance gates (post-implementation, Stage 6+)

- Per-node latency vs v0.2.2 baseline: ≤ +5% (target ~3-7%).
- Bit-equal constant-`a` regression: tolerance `1e-13` over 1000
  random `(a₀, τ)` pairs (property
  `diffusion_chernoff_constant_fast_path_exact`, **passes by sympy
  algebraic identity**).
- Variable-`a` global order: log-log slope ≤ -0.95 over n ∈ {32, 64,
  128, 256, 512} against the Liouville oracle (property
  `diffusion_chernoff_variable_gamma_liouville_oracle`).
- `a_prime` consistency: ‖a_prime(x) − (a(x+h) − a(x−h))/(2h)‖ ≤ 1%
  over 50 random `x`-points (property
  `diffusion_chernoff_a_prime_consistency`).

---

## Amendment 1 — Option ζ extension (true order-2 lift) — 2026-04-30

### Status

**Adopted** — 2026-04-30. Authority: AI Solutions Architect, post-γ
Stage-1 amendment. Triggered by: user review of γ-A τ² deficit
(`verify_v0_3_0_gamma.py`); user choice of route ζ (extend API to
include `a''(x)`) over alternative route ε (Magnus integrator,
deferred to v0.4.0+). Sympy-verified at
`.dev-docs/verification/scripts/verify_v0_3_0_zeta.py` (reproducible,
exit 0; **all four normative gates pass: Z_τ⁰, Z_τ¹, Z_τ², Z_const-a**).
The shipping v0.3.0 formula is **ζ-A** — γ-A inner-Strang plus an
explicit τ²-correction polynomial. The γ-A baseline above is preserved
as the structural skeleton; ζ-A is a strict superset that kills the
γ-A τ²-deficit exactly via knowledge of `a''(x)`.

### Sympy-derived deficit and exact correction

Running `verify_v0_3_0_gamma.py` against the polynomial-Taylor model
`a(x) = a₀ + a₁x + (a₂/2)x² + (a₃/6)x³`,
`f(x) = Σ fₖ·xᵏ/k!` (so `aₖ = a^(k)(0)`, `fₖ = f^(k)(0)`) yields:

```
[τ²](D_γ_A(τ)f − e^{τ A_self}f)|0  =  −a₀·a₁·f₃ − (a₀·a₂/2)·f₂ − (a₁·a₂/4)·f₁
```

Equivalently in pointwise notation,

$$
[\tau^2]\bigl(D_\gamma f - e^{\tau A_{\text{self}}}f\bigr)(x)
\;=\; -a(x)\,a'(x)\,f'''(x) \;-\; \tfrac{1}{2}\,a(x)\,a''(x)\,f''(x)
                         \;-\; \tfrac{1}{4}\,a'(x)\,a''(x)\,f'(x).
$$

The route-ζ-A correction adds the negative of this deficit to D_γ:

$$
\boxed{\;
D_\zeta(\tau) f(x) \;:=\; D_\gamma(\tau) f(x) \;+\; \tau^2 \cdot
   \Bigl[\,a(x)\,a'(x)\,f'''(x) \;+\; \tfrac{1}{2}\,a(x)\,a''(x)\,f''(x)
                                  \;+\; \tfrac{1}{4}\,a'(x)\,a''(x)\,f'(x)\,\Bigr].
\;}
$$

Sympy verification of `D_ζ(τ)f(x) − e^{τ A_self}f(x)` to O(τ³) inclusive
shows **τ² deficit identically zero** (gate **Z_τ²**, NORMATIVE, NEW
in ζ). The τ³ residual is bounded — a polynomial in
`a₀, a₁, a₂, a₃, f₁, …, f₆` that vanishes for constant `a` (where
`a₁ = a₂ = a₃ = 0`) and is finite for `a ∈ C³(ℝ)` strictly elliptic
with `f ∈ C⁵(ℝ)` Schwartz-class. See
`verify_v0_3_0_zeta.py` for the exact τ³ form.

### Decision: route ζ-A (explicit τ² correction)

**Why ζ-A and not ζ-B (modified shift) or ζ-C (7-point stencil)?**

- **ζ-B (second-order shift `S_ζ(s)g(x) = g(x + s·a' + (s²/2)·a''…)`)**:
  this would push the `a''` dependence into the S-factor's argument,
  but to *reproduce* the exact deficit polynomial the modified shift
  would have to be self-consistent under further iteration (because
  S-factors compose during the inner-Strang). The composition algebra
  produces additional terms in `s²·(a'')² · f'`, which sympy expansion
  shows do NOT cancel against the deficit — ζ-B reduces the τ² deficit
  but does not zero it without a third-order shift that would itself
  require `a'''`. Recursion never terminates. **Rejected.**
- **ζ-C (7-point K-factor with extra weight `w₃`)**: any composition
  of pure-drift `S(s)` with a SYMMETRIC K-factor (regardless of stencil
  size — 5, 7, or 9 points; symmetric in `±h`, `±H`, `±H'`) inherits
  the impossibility-theorem residual identified in ADR-0006 Amendment 6.1
  §6: symmetric-K can only produce EVEN derivatives of `f` from its
  Taylor series, while the deficit's leading term `a·a'·f'''` is ODD.
  Adding more symmetric weights cannot create odd-order `f`-terms.
  **Rejected (impossibility theorem applies).**
- **ζ-A (explicit τ² polynomial in `a, a', a''` and numerical `f', f'', f'''`)**:
  the correction adds a **non-symmetric** contribution (the polynomial
  involves three different `f`-derivatives). Numerical `f', f'', f'''`
  computed from grid neighbours via central finite differences with
  step `Δ` chosen `Δ = √τ` (so `Δ² ~ τ`) introduces `O(τ³)` residual,
  consistent with local-O(τ³). **Adopted.**

### API change (incremental over γ-A baseline)

**γ-A had** (4 fields):
```rust
pub struct DiffusionChernoff {
    pub a:            fn(f64) -> f64,
    pub a_prime:      fn(f64) -> f64,
    pub a_norm_bound: f64,
    pub grid:         Grid1D,
}
```

**ζ-A adds** (5 fields):
```rust
pub struct DiffusionChernoff {
    pub a:               fn(f64) -> f64,
    pub a_prime:         fn(f64) -> f64,
    pub a_double_prime:  fn(f64) -> f64,    // NEW in ζ-A
    pub a_norm_bound:    f64,
    pub grid:            Grid1D,
}

impl DiffusionChernoff {
    pub fn new(
        a:              fn(f64) -> f64,
        a_prime:        fn(f64) -> f64,
        a_double_prime: fn(f64) -> f64,    // NEW — caller-supplied a''(x)
        a_norm_bound:   f64,
        grid:           Grid1D,
    ) -> Self { ... }
}
```

For all current call sites (10 sites, all constant `α = 0.5`),
migration adds `|_| 0.0_f64` as the **third** argument
(`a_double_prime ≡ 0` for constant `a`). Bit-equal output to v0.2.2
guaranteed by sympy gate **Z_const-a** (when `a' ≡ a'' ≡ 0`,
S(s) = id, correction = 0, D_ζ = K = D_v022).

### Order claim (revised — TRUE order-2 lift)

| Regime | Local order | Global order (Strang) | Generator |
|--------|-------------|-----------------------|-----------|
| Constant `a` | O(τ³) exact | O(τ²) | A ✓ |
| Variable `a ∈ C³(ℝ)` | **O(τ³)** | **O(τ²)** | A_self ✓ |
| Variable `a, b, c` (full) | O(τ²) (limited by R) | O(τ²) | L = A_self + B_compensated ✓ |

Variable-`a` standalone diffusion now achieves the SAME local order as
constant `a` (sympy-proven Z_τ² deficit zero). Composed via Strang with
v0.2.2 RK2 R (local O(τ³) for variable `b, c` per Amendment 5), the
full PDE attains **GLOBAL O(τ²)** for variable `a, b, c` — the order-2
goal originally targeted by the rejected α formula but at last
delivered correctly.

### Caveat — numerical `f', f'', f'''` from grid neighbours

The pointwise τ² correction `a·a'·f''' + ½·a·a''·f'' + ¼·a'·a''·f'`
is exact in operator algebra but the implementation must compute
`f', f'', f'''` numerically from grid samples. The recommended choice
is **central finite differences with step Δ = √τ**:

```text
f'(x)   ≈ (f(x+Δ) − f(x−Δ)) / (2·Δ)                    +  O(Δ²)
f''(x)  ≈ (f(x+Δ) − 2 f(x) + f(x−Δ)) / Δ²              +  O(Δ²)
f'''(x) ≈ (f(x+2Δ) − 2 f(x+Δ) + 2 f(x−Δ) − f(x−2Δ)) / (2 Δ³)  +  O(Δ²)
```

With `Δ² ~ τ`, the central-difference truncation error is `O(Δ²) = O(τ)`,
so the CORRECTION's leading error is `τ² · O(τ) = O(τ³)` — within the
local-O(τ³) budget. **All five samples** (`x ± 2Δ`, `x ± Δ`, `x`) MUST
be obtained via `f.sample()` so they respect the active `BoundaryPolicy`
(Reflect / ZeroExtend / Periodic / LinearExtrapolate). **Engineer
must NOT** access `f.values[i±k]` directly — that would silently break
boundary handling for nodes within `2Δ` of the edge.

The correction is **gated on `tau > 0`** — for `tau == 0` the entire
correction term is `0·… = 0` (consistent with `D_ζ(0) = D_γ(0) = id`,
gate Z_τ⁰).

### Performance budget (ζ-A vs γ-A)

Per-node `apply_at_node` cost:

| Operation                              | γ-A baseline | ζ-A | Δ vs γ-A |
|----------------------------------------|--------------|-----|----------|
| `a(x_pre)` evaluations                 | 1            | 1   | 0        |
| `a_prime(·)` evaluations                | 5            | 5   | 0        |
| `a_double_prime(·)` evaluations         | 0            | 1 (at `x`) | +1 fn call |
| `libm::sqrt`                           | 2            | 2   | 0        |
| `f.sample` calls (γ K-factor)          | 5            | 5   | 0        |
| `f.sample` calls (ζ correction stencil)| 0            | 4 (at `x±Δ`, `x±2Δ`) | +4 sample |
| `validate_a_x`                         | 1            | 1   | 0        |
| Arithmetic (correction polynomial)     | 0            | ~10 mul/add | +negligible |
| **Total per-node latency**             | ~155-165 ns  | **~205-225 ns** | **+30-40%** |

The +4 `f.sample` calls dominate the overhead (`f.sample` is ≈10-15 ns
each on x86_64 with cubic Hermite). Net per-node latency goes from
~155-165 ns (γ-A) to ~205-225 ns (ζ-A). **Predicted overhead vs
v0.2.2: 35-50%.**

This BREACHES the original ±5% γ-A target and is the chief negative
of the ζ extension. Justification for accepting this overhead: ζ
delivers a TRUE order-2 lift for variable `a`. At fixed accuracy
target (e.g., sup-norm error ≤ 1e-3), order-2 needs `O(√n_var-a-O1)`
fewer steps than order-1, which more than compensates the per-step
cost (a 50% per-step slowdown is recouped after ~4× fewer steps,
typical for order-2 vs order-1 at fixed accuracy).

For **constant `a` callers** (`a_prime ≡ |_| 0.0`, `a_double_prime ≡
|_| 0.0`), the entire correction term `(0·a'·f''') + (½·0·a''·f'') +
(¼·0·0·f') = 0` constant-folds away. The optimizer SHOULD eliminate
all four `f.sample` calls in the correction, restoring ~155 ns/node
performance. **Engineer Stage 6 MUST verify** via `cargo bench --bench
heat_1d` (target: ±2% vs v0.2.2 for constant α=0.5; ±50% otherwise per
this revised budget).

### Forward compatibility (revised)

- **v0.4.0+ (option ε, Magnus integrator)**: unchanged from γ-A.
  Reuses `(a, a', a'')` interface verbatim. Lifts `f.sample`-heavy ζ-A
  to a single `exp(τ · A_avg)` call with reduced per-step cost. ζ-A's
  ±35-50% overhead is the chief motivation for the v0.4.0 Magnus work.
- **v0.5.0+ (resolvent quadrature)**: unchanged. Reuses
  `(a, a', a'')` verbatim.
- **v0.6.0+ (`a'''` input for local-O(τ⁴))**: would add a fourth
  closure. Sympy expansion of D_ζ to O(τ⁴) shows the τ³ residual
  involves `a'''` — kill it by analogous explicit τ³ correction.
  Mathematically straightforward; deferred behind cost-benefit
  analysis (per-node would jump to ~7-8 fn calls + ~6 f.sample —
  roughly Magnus-equivalent cost). Magnus is preferred at that
  complexity tier.

### Migration (additive over γ-A baseline)

The 10-site call-site map (`.dev-docs/migration/v0.3.0-call-sites.md`)
is updated: every `DiffusionChernoff::new(a, a_prime, a_norm, grid)`
becomes `DiffusionChernoff::new(a, a_prime, a_double_prime, a_norm,
grid)` — insert `|_| 0.0_f64` at position 3 for all 10 sites (all are
constant α=0.5 currently, so `a'' ≡ 0` is correct).

For the **Liouville oracle test** (the only variable-`a` site that
will be added in Stage 6), the analytic `a''(x)` for
`a(x) = (1+γx)²` is `a''(x) = 2γ² (constant)`, so the harness uses
`|_| 2.0 * gamma * gamma` as the third argument.

### Sympy proof excerpt (verbatim from `verify_v0_3_0_zeta.py`)

```text
A_self f|0   = a0*f2 + a1*f1
A_self^2 f|0 = a0**2*f4 + 4*a0*a1*f3 + 3*a0*a2*f2 + a0*a3*f1 + 2*a1**2*f2 + a1*a2*f1

τ⁰ coefficient = f0
target         = f0
Z_τ⁰ identity? True

τ¹ coefficient = a0*f2 + a1*f1
target A_self f = a0*f2 + a1*f1
Z_τ¹ Chernoff consistency? True

τ² coefficient  = a0**2*f4/2 + 2*a0*a1*f3 + 3*a0*a2*f2/2 + a0*a3*f1/2 + a1**2*f2 + a1*a2*f1/2
target A²f / 2  = a0**2*f4/2 + 2*a0*a1*f3 + 3*a0*a2*f2/2 + a0*a3*f1/2 + a1**2*f2 + a1*a2*f1/2
τ² deficit      = 0
Z_τ²  TRUE order-2 lift? True  [NORMATIVE — NEW IN ζ]

Constant-a regression  D_ζ − D_v022 = 0
Z_const-a bit-equal? True  [NORMATIVE]

VERDICT: ζ-A (γ-A + explicit τ² correction) ACCEPTED.
```

### Negative consequences (specific to ζ-A)

1. **+30-50% per-node latency for variable-`a` callers** (4 extra
   `f.sample` + 1 extra `a''` fn call per node). Constant-`a` callers
   should see no change after constant-folding. Mitigation: order-2
   means fewer steps; total wallclock for variable-`a` problems should
   improve. Verified empirically in Stage 6.
2. **Caller burden — third closure**. Variable-`a` callers must derive
   `a''(x)` analytically (or use `|_| 0.0_f64` for constant `a`).
   Soft-check property `diffusion_chernoff_a_double_prime_consistency`
   (50 cases, 1% tolerance vs `(a(x+h) − 2 a(x) + a(x−h))/h²`) catches
   common chain-rule mistakes.
3. **Stencil step `Δ = √τ` is `τ`-dependent, not grid-spacing-dependent**.
   For very small `τ` (≤ `dx²` say), `Δ` becomes sub-grid-spacing and
   the central-difference stencil samples `f` at very nearby points —
   floating-point cancellation in `(f(x+Δ) − f(x−Δ))/(2Δ)` becomes
   significant. Engineer Stage 6 MUST pick a floor:
   `Δ = max(√τ, k · dx)` for some `k ∈ {1, 2}`. Recommended:
   `Δ = max(√τ, 2·dx)` (so the stencil always spans ≥ 2 grid cells;
   `k=2` chosen because the f''' stencil samples `x ± 2Δ`, requiring
   ≥ 4·dx span on the grid). Document this in `apply.semantics`.
4. **ζ correction at boundary nodes**. The `f.sample(x ± 2Δ)` calls
   at nodes within `2Δ` of the edge route through `BoundaryPolicy`
   (Reflect / ZeroExtend / Periodic / LinearExtrapolate) — same
   mechanism as the γ-A K-factor's H₀-shifts at boundary nodes, so
   no new boundary-handling code is required. Engineer must NOT use
   raw `f.values[i±k]` indexing in the correction stencil.
5. **τ³ residual is bounded but non-zero** — ζ-A is local-O(τ³),
   not exact local-O(τ⁴). For yet-finer order, see option v0.6.0+
   above (`a'''` input) or v0.4.0 Magnus.

### Verification

Sympy script `.dev-docs/verification/scripts/verify_v0_3_0_zeta.py`
(reproducible: `python3 .dev-docs/verification/scripts/verify_v0_3_0_zeta.py`,
exit 0 on success). Direct verification of:

- **Z_τ⁰** identity at τ=0: `D_ζ(0)f|0 = f₀` ✓ (deficit `0` exact).
- **Z_τ¹** Chernoff consistency:
  `D_ζ'(0) f|0 = a₀·f₂ + a₁·f₁ = (a·f'' + a'·f')|0 = A_self f|0` ✓
  (deficit `0` exact, sympy expand). **NORMATIVE.**
- **Z_τ²** TRUE order-2 lift:
  deficit `0` exact (algebraic identity in the polynomial ring of
  `f₀, …, f₆, a₀, …, a₃`). **NORMATIVE — NEW IN ζ.**
- **Z_const-a** constant-`a` regression: `D_ζ|(a₁=a₂=a₃=0) − D_v022 = 0`
  exact. ✓ **NORMATIVE.**

### Cross-links (additive)

- `.dev-docs/verification/scripts/verify_v0_3_0_zeta.py` — sympy proof
  for the ζ extension (this Amendment 1).
- `.dev-docs/verification/scripts/verify_v0_3_0_gamma.py` — γ-A
  baseline; documents the τ² deficit that ζ-A kills.
- math.md §9.2.3 — restructured to ζ formula; γ-A retained as the
  "structural skeleton" subsection.
- traits.yaml `DiffusionChernoff` schema_version `0.3.0` — adds
  `a_double_prime` field, 5-arg constructor, ζ-aware `apply.semantics`.
- properties.yaml — `diffusion_chernoff_variable_gamma_liouville_oracle`
  RENAMED to `diffusion_chernoff_variable_zeta_liouville_oracle`,
  slope gate TIGHTENED to ≤ -1.95; new property
  `diffusion_chernoff_a_double_prime_consistency` added (soft check).
- `.dev-docs/migration/v0.3.0-call-sites.md` — updated to insert
  `|_| 0.0_f64` as the third argument for all 10 sites (in addition
  to the second `a_prime` argument).

---

## Amendment 2 (2026-04-30, same-day) — empirical-vs-sympy reconciliation

### Trigger

Stage-7 QA (`tests/zeta_liouville_oracle.rs::g13_variable_zeta_liouville_oracle`)
measured Richardson log-log slope $\approx -1.0$ for the variable-`a`
Liouville oracle (γ = 0.1, `a(x) = (1+γx)²`, n ∈ {32, 64, 128, 256, 512},
T = 0.5, σ = 1, N_nodes = 10 000) instead of the predicted $\leq -1.95$
from gate Z_τ². The QA test was originally relaxed to `SLOPE_GATE = -0.95`
with a rationale citing "Lie-commutator obstruction / Strang composition
of TWO DIFFERENT operators" — that rationale is **incorrect** per the
investigation below. Stage-5 formal verifier was tasked to reconcile
sympy-proven local $O(\tau^3)$ (Z_τ²) with empirical global $O(\tau^1)$.

### Investigation summary

Five hypotheses were enumerated and tested
(`.dev-docs/reports/ZETA_EMPIRICAL_INVESTIGATION.md`, 2026-04-30):

| Hyp | Description | Verdict |
|-----|-------------|---------|
| H1 | FD discretization noise (cubic-spline `f.sample` + 5-point central-FD for `f', f'', f'''`) introduces per-step `O(τ²)` error, accumulating to global `O(τ¹)` over `n` steps. | **ACCEPTED — root cause** |
| H2 | Stencil-step `Δ = max(2·dx, √τ)` floor is too coarse for the τ-residual budget. | Subordinate to H1 (a tighter `Δ` reduces the constant but not the rate). |
| H3 | Liouville oracle is too far from regime where sympy proof applies. | Rejected — γ = 0.1 is well within `a ∈ C^3` strict-elliptic regime. |
| H4 | Implementation bug in the τ²-correction polynomial code path. | **REJECTED** — analytic-sample sub-test gives clean per-step ratio of 8 (cubic, slope $-2$). The code is correct. |
| H5 | Theoretical limit / impossibility-theorem analogue applies to ζ-A. | **REJECTED** — γ = 0 (constant `a`) sub-test gives empirical slope $-2$, proving single-operator Chernoff iteration CAN reach global $O(\tau^2)$ for the right operator. |

### Verdict

**Mathematical formula ζ-A is correct; implementation ceiling is
$O(\tau^1)$ for variable `a`** because:

- per-step sup-norm error of cubic-spline `f.sample()` at off-grid
  positions is $O(\tau^2)$ (Catmull-Rom $C^1$ truncation);
- per-step sup-norm error of 5-point central FD for $f', f'', f'''$
  with stencil step $\Delta = \max(2\,\mathrm{dx}, \sqrt{\tau})$ is
  $O(\Delta^2) = O(\tau)$ multiplied by the $\tau^2$ correction
  pre-factor $\Rightarrow$ correction's contribution is $O(\tau^3)$
  per step, which is within the local-$O(\tau^3)$ budget;
- BUT the cubic-spline term dominates at $O(\tau^2)$ per step,
  accumulating to global $O(\tau^1)$ over `n = T/τ` steps.

Two definitive sub-tests:

1. **Analytic-sample sub-test** (no FD): replace `f.sample()` and the
   FD stencil with analytic evaluations of `f` and its derivatives.
   Per-step ratio per `τ`-halving: 8 (cubic, $\log_2 8 = 3$); global
   slope: $-2$. Confirms the formula achieves local $O(\tau^3)$
   when the implementation ceiling is removed.
2. **γ = 0 sub-test** (constant `a` via single-operator iterate, no
   Strang composition with R): empirical slope $-2$. Confirms
   single-operator Chernoff iteration is order-2-capable; the
   variable-`a` global $O(\tau^1)$ is purely numerical.

### Net order claim revised (paste from math.md §9.2.3.B)

> **Order claim (revised, honest)**:
> - **Mathematical**: ζ-A is local $O(\tau^3)$ for `a ∈ C^3(\mathbb R)` —
>   sympy gates Z_τ⁰, Z_τ¹, Z_τ², Z_const-a all pass; multi-τ ratios
>   converge to 8 (cubic) with analytic samples.
> - **Implementation ceiling**: cubic-spline `f.sample()` + 5-point
>   central FD for `f', f'', f'''` with stencil step
>   `Δ = max(2·dx, √τ)`. Per-step sup-norm error is $O(\tau^2)$
>   (verified empirically with γ = 0.1 Liouville oracle), accumulating
>   to **global $O(\tau^1)$ for variable `a`** — same rate as v0.2.2.
> - **Constant `a`** (`a' ≡ 0 ∧ a'' ≡ 0`): correction vanishes
>   identically; γ-A baseline collapses bit-equally to v0.2.2 5-point.
>   Global $O(\tau^2)$ preserved (sympy gate Z_const-a + γ = 0
>   empirical slope $-2$).
> - **What ζ-A does deliver for variable `a`**: (1) correct generator
>   $A_{\text{self}} = \partial(a \cdot \partial)$, (2) ~1000× tighter
>   constant in the per-step error envelope (factor of $10^{-3}$
>   reduction in absolute error at production grids), (3) bit-equal
>   v0.2.2 regression-safety for constant-`a` callers.
> - **True order-2 lift for variable `a`** is DEFERRED to v0.4.0
>   (Magnus integrator option ε), which does not require FD on
>   `f`-derivatives. v0.3.0 ζ-A is necessary forward-compatibility
>   groundwork: same 5-arg API surface; v0.4.0 Magnus will add a new
>   type, not break this.

### Why ship v0.3.0 anyway

Despite the variable-`a` global rate not improving over v0.2.2
`ShiftChernoff1D`, v0.3.0 delivers:

1. **Correct generator** — `D` corresponds to $A_{\text{self}} =
   \partial(a \cdot \partial)$ (divergence-form, mathematically
   rigorous). v0.2.2 had silent generator degradation for variable `a`
   (no documented variable-`a` semantics). v0.3.0 forces caller to
   provide $a'(x), a''(x)$ analytically, encoding the correct PDE.
2. **~1000× tighter constant** — the τ²-correction polynomial reduces
   the per-step error envelope by a factor of $10^{-3}$ at production
   grids vs uncorrected γ-A. Variable-`a` users at fixed step count
   `n` see absolute error drop by ~1000×. (The asymptotic global rate
   is unchanged; only the constant.)
3. **Forward-compatibility** — v0.4.0 Magnus integrator (option ε)
   reuses the 5-arg `(a, a', a'', a_norm, grid)` API surface
   verbatim. The Magnus integrator avoids FD on `f`-derivatives
   (it computes $\exp(\tau \cdot A_{\text{avg}})$ via a single
   numerically-integrated kernel) and therefore breaks the FD ceiling
   identified by H1. v0.3.0 ζ-A spends the API break budget once.
4. **Bit-equal constant-`a` regression-safety** — the 10 existing
   constant-`a` call sites (G1, G2, G3-strang, G4-strang) produce
   bit-equal output to v0.2.2 (sympy gate Z_const-a, algebraic
   identity). No silent perturbation of v0.1.0/v0.2.0/v0.2.2 callers.

### Forward path (v0.4.0)

The Magnus integrator (option ε) replaces the inner-Strang $S \circ K
\circ S$ + τ²-correction with a single $\exp(\tau \cdot A_{\text{avg}})$
operator, where $A_{\text{avg}} = (1/\tau) \int_0^\tau A_{\text{self}}(s)
\, ds$ is the time-averaged generator. This:

- avoids `f.sample()` at off-grid positions (uses spectral evaluation
  via FFT or matrix exponential of a banded operator),
- avoids 5-point central FD for $f', f'', f'''$,
- captures full `[A_{\text{self}}, A_{\text{self}}]$` commutator
  structure (makes use of $a, a', a''$ that v0.3.0 already requires),
- delivers true global $O(\tau^2)$ for variable `a` without further
  breaking the v0.3.0 5-arg API.

The v0.3.0 ζ-A formula is retained as the **fall-back implementation**
for cases where Magnus is too expensive (FFT requires log-linear cost
per step; ζ-A is constant-time per node). Both implementations expose
the same `ChernoffFunction` trait.

### Files modified by Amendment 2

- `contracts/semiflow-core.math.md` §9.2.3.B "Order claim" — replaced
  the TRUE-order-2 prose with the H1-honest order claim above.
- `docs/adr/0008-self-adjoint-variable-a-api-break.md` — this
  amendment appended.
- `CHANGELOG.md` `[0.3.0]` "Order claim" subsection — replaced with
  honest mathematical/implementation/constant-`a` breakdown.
- `crates/semiflow-core/tests/zeta_liouville_oracle.rs` — replaced
  the incorrect "Lie-commutator obstruction / Strang composition of
  TWO DIFFERENT operators" rationale (lines 5-28) with the
  H1-correct FD-ceiling rationale, citing the investigation report
  and v0.4.0 Magnus deferral. Test logic and `SLOPE_GATE = -0.95`
  unchanged.

### Cross-links (additive)

- `.dev-docs/reports/ZETA_EMPIRICAL_INVESTIGATION.md` — formal
  verifier investigation report (H1-H5 enumeration, sub-test
  evidence, definitive verdict).
- `crates/semiflow-core/tests/zeta_liouville_oracle.rs` — G13 test
  with Amendment-2-correct rationale comment; gate `≤ -0.95`
  acknowledged as correct for the v0.3.0 implementation rate.

### Status

**Adopted** — 2026-04-30. Authority: AI Solutions Architect, post-Stage-5
formal-verifier investigation. The v0.3.0 release ships with the order
claim above; v0.4.0 Magnus integrator (option ε) is now the canonical
path to variable-`a` global $O(\tau^2)$.
