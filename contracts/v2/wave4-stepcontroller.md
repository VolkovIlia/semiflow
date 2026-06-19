# Wave 4 Contract: `StepController<F>` + Generic `AdaptivePI<C, F, K>`

**Status**: NORMATIVE
**ADR**: docs/adr/0044-stepcontroller-trait-h211b-advisory.md
**Scope**: semiflow-core v2.0 Wave 4
**Depends on**: contracts/v2/wave1-scratch.md, contracts/v2/wave2-inplace-strang.md, contracts/v2/wave3-state-trait.md
**Math fidelity**: `ClassicalPI` is NORMATIVE per math.md §11.1.bis. `H211bFilter` is ADVISORY (ADR-only).

---

## §1 — `AdaptivePI<C, F, K>` generic signature change

### 1.1 Before (v1.x / post-Wave-3)

```rust
// crates/semiflow-core/src/adaptive.rs — current (post-Wave-3, line 69)
pub struct AdaptivePI<C: ChernoffFunction> {
    pub func: C,
    pub tol_abs: f64,
    pub tol_rel: f64,
    pub safety: f64,
    pub alpha: f64,
    pub beta: f64,
    pub min_ratio: f64,
    pub max_ratio: f64,
    pub max_substeps: usize,
}
```

The inner `C: ChernoffFunction` defaults to `ChernoffFunction<f64>` (ADR-0025/0026)
but `AdaptivePI` itself bakes in `f64` for `tol_abs`, `safety`, gains, ratios, and
the `richardson_err` divisor. The PI step-size law is **inlined** as
`pi_step_factor` / `reject_step_factor` private helpers (`adaptive.rs:159–177`).

### 1.2 After (Wave 4)

```rust
// crates/semiflow-core/src/adaptive.rs — NORMATIVE Wave 4
pub struct AdaptivePI<
    C: ChernoffFunction<F>,
    F: SemiflowFloat = f64,
    K: StepController<F> = ClassicalPI<F>,
> {
    pub func: C,
    /// Absolute tolerance (in the same scalar type as the state).
    pub tol_abs: F,
    /// Relative tolerance.
    pub tol_rel: F,
    /// Safety factor on the controller multiplier.
    pub safety: F,
    /// Minimum allowed step-size ratio (next_tau / prev_tau).
    pub min_ratio: F,
    /// Maximum allowed step-size ratio.
    pub max_ratio: F,
    /// Hard cap on total substeps (accepted + rejected).
    pub max_substeps: usize,
    /// Step-size law. Defaults to `ClassicalPI<F>`.
    controller: K,
}
```

**Removed public fields**: `alpha`, `beta`. They move into `ClassicalPI<F>`. Source
callers reading `pi.alpha` / `pi.beta` migrate to `pi.controller().alpha()` /
`pi.controller().beta()` accessor methods on `ClassicalPI<F>`. (See §3.3 for
accessor surface — controllers expose their own state.)

**Defaulting rules** (MUST):
- `F = f64` default: source callers writing `AdaptivePI::new(func)` get back
  `AdaptivePI<C, f64, ClassicalPI<f64>>`.
- `K = ClassicalPI<F>` default: NORMATIVE per §11.1.bis. Switching requires explicit
  `.with_controller(...)`.

**Construction defaults** (MUST match v1.0.0 in the f64+ClassicalPI case):
- `tol_abs = F::from(1e-8).unwrap()` — equals `1e-8_f64` for `F=f64`.
- `tol_rel = F::from(1e-6).unwrap()`.
- `safety = F::from(0.9).unwrap()`.
- `min_ratio = F::from(0.2).unwrap()`.
- `max_ratio = F::from(5.0).unwrap()`.
- `max_substeps = 100_000`.
- `controller = ClassicalPI::<F>::with_order(func.order())` → seeds gains.

The `F::from(...).unwrap()` constants are call-once at `new`; they collapse to
literals under monomorphisation for `F = f64` / `F = f32`.

### 1.3 Outcome type (unchanged surface, F-generic body)

```rust
pub struct AdaptiveOutcome<S, F: SemiflowFloat = f64> {
    pub final_state: S,
    pub steps_accepted: usize,
    pub steps_rejected: usize,
    pub last_tau: F,
}
```

`last_tau` becomes `F`. For `F = f64` this is source-compatible — callers reading
`outcome.last_tau` as `f64` are unchanged.

### 1.4 Method surface (NORMATIVE)

```rust
impl<C, F, K> AdaptivePI<C, F, K>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
    K: StepController<F>,
    C::S: HilbertState<F> + Clone,
{
    pub fn evolve_adaptive(
        &mut self,
        t: F,
        u0: &C::S,
    ) -> Result<AdaptiveOutcome<C::S, F>, SemiflowError>;
}
```

`&mut self` (was `&self`): controllers carry mutable state (previous filtered error).
The hot loop borrows `&mut self.controller`. Documented breaking change since v2.0 is
MAJOR.

`C::S: HilbertState<F>` (was `State`): Wave 4 requires `dot` / `norm_sq` for the
zero-alloc Richardson error norm. All concrete `GridFn{1,2,3}D<F>` already implement
`HilbertState<F>` per Wave 3.

`+ Clone` retained: the accepted-state path still does `u_curr = u_half` semantically,
but Wave 4 replaces the clone with `u_curr.copy_from(&u_half)` (Wave 3 zero-alloc
primitive). The `Clone` bound is kept on `evolve_adaptive` for backward compat with
the `apply` bridge default in `ChernoffFunction::apply`; concrete callers' `GridFnXD`
all derive `Clone`.

### 1.5 Builder

```rust
impl<C, F> AdaptivePI<C, F, ClassicalPI<F>>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    pub fn new(func: C) -> Self;
    pub fn with_tolerance(mut self, abs: F, rel: F) -> Self;
}

impl<C, F, K> AdaptivePI<C, F, K>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
    K: StepController<F>,
{
    /// Replace the step-size law. Consumes `self`, returns the re-typed instance.
    /// Calling with `H211bFilter::default()` returns
    /// `AdaptivePI<C, F, H211bFilter<F>>`.
    pub fn with_controller<K2: StepController<F>>(self, ctrl: K2)
        -> AdaptivePI<C, F, K2>;

    pub fn controller(&self) -> &K;
    pub fn controller_mut(&mut self) -> &mut K;
}
```

Note `with_controller` is a *type-changing* builder — uses self-consuming pattern.
This is zero-runtime-cost via monomorphisation; the only practical drawback is that
trait objects cannot be used (acceptable for a numerics core).

---

## §2 — `StepController<F>` trait surface

### 2.1 Trait

```rust
/// File: crates/semiflow-core/src/controller.rs (NEW, Wave 4)
///
/// Step-size law for `AdaptivePI`. The controller owns its own state
/// (previous filtered error, history buffer for digital filters) and
/// returns the next-τ multiplier per call.
///
/// `propose_accept` and `propose_reject` are separate to mirror the
/// existing two-branch flow in `adaptive.rs::substep` and to let
/// digital-filter controllers (e.g. `H211bFilter`) update history only
/// on accepted steps.
pub trait StepController<F: SemiflowFloat> {
    /// Called after a substep is ACCEPTED. Returns the multiplier `r`
    /// such that `next_tau = clamp(prev_tau * r)`.
    ///
    /// Inputs:
    /// - `err_norm`: Richardson error norm of the just-accepted step.
    /// - `tol`:      mixed abs/rel tolerance at the current state.
    /// - `safety`:   safety factor (passed by `AdaptivePI`, e.g. 0.9).
    /// - `p_order`:  inner function consistency order (`func.order()`).
    ///
    /// The controller MUST update its internal state if it carries one.
    fn propose_accept(&mut self, err_norm: F, tol: F, safety: F, p_order: u32) -> F;

    /// Called after a substep is REJECTED. Returns the shrink multiplier.
    /// Classical implementations use only the I-term to avoid amplification.
    /// Controllers MAY or MAY NOT touch their internal state on reject —
    /// `ClassicalPI` does not, `H211bFilter` does not.
    fn propose_reject(&mut self, err_norm: F, tol: F, safety: F, p_order: u32) -> F;

    /// Reset internal state to initial conditions. Called at the top of
    /// `evolve_adaptive` so consecutive evolutions are independent.
    fn reset(&mut self);
}
```

### 2.2 Semantics — return value

The returned `F` is a **pure multiplier** on the previous τ. Clamping to
`[min_ratio, max_ratio]` is performed in `AdaptivePI`, NOT in the controller.
This separation lets controllers be unit-tested without an `AdaptivePI` context.

### 2.3 Where state lives

- `ClassicalPI<F>`: holds `alpha`, `beta`, `err_prev` (the I-term seed).
- `H211bFilter<F>`: holds two previous error values + last accepted multiplier.

Neither holds the *previous τ* — that flows through `AdaptivePI` as the loop-local
`tau_step`.

### 2.4 Algebraic laws (MUST hold)

- `propose_accept(err_norm = tol, ...) ≈ safety` for `ClassicalPI` after the first
  step (when `err_prev == tol`). Documented in tests.
- `propose_reject` returns ≤ `safety` (never amplifies a rejected step).
- `reset()` is idempotent: `c.reset(); c.reset();` is equivalent to `c.reset();`.

---

## §3 — `ClassicalPI<F>` implementation (NORMATIVE default)

### 3.1 Struct

```rust
pub struct ClassicalPI<F: SemiflowFloat> {
    /// Söderlind PI.4.7 P-gain exponent = 0.7 / p.
    pub alpha: F,
    /// Söderlind PI.4.7 I-gain exponent = 0.4 / p.
    pub beta: F,
    /// Previous-step error norm (I-term memory). Seeded to `F::one()`
    /// so the first step's I-term is neutral.
    err_prev: F,
}
```

### 3.2 Constructors

```rust
impl<F: SemiflowFloat> ClassicalPI<F> {
    /// Construct with §11.1.bis gains for the given inner order `p`.
    pub fn with_order(p: u32) -> Self {
        let pf = F::from(f64::from(p)).unwrap();
        Self {
            alpha: F::from(0.7).unwrap() / pf,
            beta:  F::from(0.4).unwrap() / pf,
            err_prev: F::one(),
        }
    }

    /// Direct constructor — for tests and advanced users.
    pub fn new(alpha: F, beta: F) -> Self {
        Self { alpha, beta, err_prev: F::one() }
    }

    pub fn alpha(&self) -> F { self.alpha }
    pub fn beta(&self) -> F  { self.beta  }
}
```

### 3.3 `Default for ClassicalPI<F>` (REQUIRED for type-defaulting)

```rust
impl<F: SemiflowFloat> Default for ClassicalPI<F> {
    /// Defaults assume `p = 2` (matches `DiffusionChernoff::order()` post-D1).
    fn default() -> Self { Self::with_order(2) }
}
```

### 3.4 `StepController<F> for ClassicalPI<F>` — bit-identical to v1.0.0

The Wave 4 implementation MUST produce the **byte-identical** sequence of
multiplied steps as the v1.0.0 inlined helpers. The reference v1.0.0 code is:

```rust
// v1.0.0 — adaptive.rs:159..171
fn pi_step_factor(err_norm: f64, err_prev: f64, tol: f64,
                  alpha: f64, beta: f64, safety: f64) -> f64 {
    let safe_err = err_norm.max(1e-300);
    let e        = pow(tol / safe_err, alpha);
    let e_prev   = pow(err_prev / safe_err, beta);
    safety * e * e_prev   // ← exact FP association: ((safety * e) * e_prev)
}

// v1.0.0 — adaptive.rs:174..177
fn reject_step_factor(err_norm: f64, tol: f64, alpha: f64, safety: f64) -> f64 {
    let safe_err = err_norm.max(1e-300);
    safety * pow(tol / safe_err, alpha)
}
```

**Wave 4 NORMATIVE encoding (MUST preserve FP association):**

```rust
impl<F: SemiflowFloat> StepController<F> for ClassicalPI<F> {
    fn propose_accept(&mut self, err_norm: F, tol: F, safety: F, _p: u32) -> F {
        // The "1e-300" floor stays a literal-via-F::from to keep f64 byte-identical.
        let safe_err = err_norm.max(F::from(1e-300).unwrap());
        let e        = (tol / safe_err).powf(self.alpha);
        let e_prev   = (self.err_prev / safe_err).powf(self.beta);
        let factor   = safety * e * e_prev;   // associate left-to-right; NORMATIVE
        self.err_prev = err_norm;             // I-term update on accept
        factor
    }

    fn propose_reject(&mut self, err_norm: F, tol: F, safety: F, _p: u32) -> F {
        let safe_err = err_norm.max(F::from(1e-300).unwrap());
        safety * (tol / safe_err).powf(self.alpha)
        // err_prev NOT updated on reject (matches v1.0.0)
    }

    fn reset(&mut self) { self.err_prev = F::one(); }
}
```

**Bit-equality rule** (NORMATIVE):
- The expression `safety * e * e_prev` MUST associate left-to-right exactly as the
  v1.0.0 inlined helper does. A compiler-defeating right-association
  (`safety * (e * e_prev)`) would diverge at ULP level on some FP edges and break
  the F9 oracle. The test `adaptive_classical_bit_equal` enforces this.
- `(tol / safe_err).powf(alpha)` MUST use `libm::pow` for `F = f64` to match v1.0.0
  exactly. For `F = f32` use `libm::powf`. The `SemiflowFloat` trait already exposes
  `powf` per ADR-0025 — confirm in implementation.

### 3.5 Source-compat shims on `AdaptivePI`

For zero-source-diff migration of the existing public field accessors, add inherent
methods on `AdaptivePI<C, F, ClassicalPI<F>>`:

```rust
impl<C, F> AdaptivePI<C, F, ClassicalPI<F>>
where C: ChernoffFunction<F>, F: SemiflowFloat {
    pub fn alpha(&self) -> F { self.controller.alpha() }
    pub fn beta(&self)  -> F { self.controller.beta()  }
}
```

The v1.0.0 tests (`tests/adaptive_unit.rs::default_construction`) read `pi.alpha`
as a *field*. Wave 4 breaks that one usage — the test must migrate to
`pi.alpha()` (call). Same for `beta`. This is the **only** known
public-surface break inside the workspace and is documented in the migration note.

---

## §4 — `H211bFilter<F>` implementation (ADVISORY, opt-in only)

### 4.1 Mathematical formula

H211b is a digital low-pass filter on the step-size sequence. Per Söderlind 2003,
the multiplier `r` for the next τ is:

```
ρ_n = (tol / err_n)^(1 / (b · p))
    × (tol / err_{n-1})^(1 / (b · p))
    × r_{n-1}^(-c / b)
r_n = safety · ρ_n
```

with the H211b-specific parameters

```
b = 4         (filter order parameter; H211b convention)
c = 1         (filter feedback coefficient)
```

so the per-step exponent on each error term is `1/(b·p) = 1/(4·p)`, and the
multiplier-feedback exponent is `-c/b = -1/4`. The accumulator `r_{n-1}` is the
*previous accepted multiplier* — initial value `1` (neutral).

### 4.2 Struct

```rust
pub struct H211bFilter<F: SemiflowFloat> {
    /// Previous error norm `err_{n-1}`. Seeded to `F::one()`.
    err_prev: F,
    /// Previous accepted multiplier `r_{n-1}`. Seeded to `F::one()`.
    r_prev: F,
}
```

No stored gains — `b=4` and `c=1` are baked into the impl as `F::from(0.25)` and
`F::from(-0.25)` constants. The order `p` arrives via `propose_accept`'s `p_order`
argument so the controller adapts to whichever inner Chernoff function is wrapped.

### 4.3 Constructors

```rust
impl<F: SemiflowFloat> H211bFilter<F> {
    pub fn new() -> Self {
        Self { err_prev: F::one(), r_prev: F::one() }
    }
}

impl<F: SemiflowFloat> Default for H211bFilter<F> {
    fn default() -> Self { Self::new() }
}
```

### 4.4 `StepController<F> for H211bFilter<F>`

```rust
impl<F: SemiflowFloat> StepController<F> for H211bFilter<F> {
    fn propose_accept(&mut self, err_norm: F, tol: F, safety: F, p_order: u32) -> F {
        let p = F::from(f64::from(p_order)).unwrap();
        let exp_e   = F::one() / (F::from(4.0).unwrap() * p);   // 1 / (b·p) with b=4
        let exp_r   = F::from(-0.25).unwrap();                  // -c/b with c=1, b=4
        let safe_e  = err_norm.max(F::from(1e-300).unwrap());
        let safe_ep = self.err_prev.max(F::from(1e-300).unwrap());
        let term_e  = (tol / safe_e).powf(exp_e);
        let term_ep = (tol / safe_ep).powf(exp_e);
        let term_r  = self.r_prev.powf(exp_r);
        let factor  = safety * term_e * term_ep * term_r;       // left-to-right
        // Update state on accept only:
        self.err_prev = err_norm;
        self.r_prev   = factor;
        factor
    }

    fn propose_reject(&mut self, err_norm: F, tol: F, safety: F, p_order: u32) -> F {
        // On reject, fall back to I-term shrink (classical-style). H211b literature
        // is silent on the reject branch; using the I-term keeps the shrink predictable.
        let p = F::from(f64::from(p_order)).unwrap();
        let alpha = F::from(0.7).unwrap() / p;
        let safe_e = err_norm.max(F::from(1e-300).unwrap());
        safety * (tol / safe_e).powf(alpha)
        // err_prev / r_prev NOT updated on reject (state advances only on accept).
    }

    fn reset(&mut self) {
        self.err_prev = F::one();
        self.r_prev   = F::one();
    }
}
```

### 4.5 Status

**NOT NORMATIVE**. Documented at ADR-0044 + this contract scope only. No math.md
mention. Acceptance metric: F9 IQR(step) ≥ 2× smaller than `ClassicalPI` at L²
error ≤ 1.05× baseline.

---

## §5 — Richardson error path: zero-alloc via Wave 1 + Wave 3

### 5.1 Current (post-Wave-3) implementation

`adaptive.rs:189–207` (`richardson_err`) allocates one fresh `S` via
`u_half.clone()` per substep attempt — the classical "diff = clone(u_half); diff
+= -1 · u_full; err = ‖diff‖" pattern. Combined with `apply_full_and_half`
returning two new owned states (`u_full`, `u_half`), one substep attempt costs
**3 owned-state allocations** on the heap.

### 5.2 Wave 4 NORMATIVE refactor

Inject a `ScratchPool<F>` member into `AdaptivePI`:

```rust
pub struct AdaptivePI<C, F, K> {
    // ... existing fields ...
    scratch: ScratchPool<F>,
}
```

Add `apply_into`-driven helpers that write into pool-managed states. Because the
`apply_into` signature returns into `&mut C::S`, not `&mut [F]`, we need *state*
scratch, not raw-vec scratch. Two options:

**Option A (RECOMMENDED)**: persist three `Option<C::S>` slots inside `AdaptivePI`
that are lazily allocated on the first call to `evolve_adaptive` (sized from
`u0.len()`) and reused thereafter:

```rust
pub struct AdaptivePI<C: ChernoffFunction<F>, F: SemiflowFloat, K> {
    // ...
    /// Lazily-allocated state scratch (sized on first evolve_adaptive call).
    /// Three slots: u_full, u_half_a, u_half (or diff).
    state_scratch: Option<[C::S; 3]>,
    /// Vec-level scratch pool (passed through to inner apply_into calls).
    scratch: ScratchPool<F>,
}
```

`C::S: Clone` is needed at first allocation (`u0.clone()` × 3 then `zero_into()`).
This is paid **once per evolution call**, not per substep. Subsequent substeps
reuse the slots via `copy_from` / `apply_into`.

**Option B**: hand the user a builder method `with_state_buffers([s1, s2, s3])` so
they pre-allocate. Cleaner but breaks ergonomic continuity with v1.0.0. **REJECTED**
for Wave 4 — re-evaluate in v2.1.

### 5.3 Hot loop (NORMATIVE pseudo-code)

```rust
fn substep(&mut self, tau_step: F, u_curr: &C::S, p: u32) -> Result<...> {
    let [s_full, s_half_a, s_half] = self.state_scratch.as_mut().unwrap();

    // Full step:
    self.func.apply_into(tau_step,     u_curr,   s_full,   &mut self.scratch)?;
    // Two half steps:
    self.func.apply_into(tau_step/2,   u_curr,   s_half_a, &mut self.scratch)?;
    self.func.apply_into(tau_step/2,   s_half_a, s_half,   &mut self.scratch)?;

    // Richardson error via HilbertState — zero-alloc:
    //   ‖u_half − u_full‖₂ / (2^p − 1)
    //   ≡ sqrt( ⟨u_half, u_half⟩ − 2·⟨u_half, u_full⟩ + ⟨u_full, u_full⟩ ) / div
    let dot_hh = s_half.dot(s_half);     // == norm_sq(s_half)
    let dot_hf = s_half.dot(s_full);
    let dot_ff = s_full.dot(s_full);
    let two = F::from(2.0).unwrap();
    let diff_sq = (dot_hh - two*dot_hf + dot_ff).max(F::zero()); // guard ≥0
    let divisor = F::from(((1u64 << p) - 1) as f64).unwrap();
    let err_norm = diff_sq.sqrt() / divisor;

    // Mixed tolerance:
    let u_curr_norm = u_curr.norm_sup();
    let u_full_norm = s_full.norm_sup();
    let tol = self.tol_abs + self.tol_rel * u_curr_norm.max(u_full_norm);

    // Controller:
    if err_norm <= tol {
        let factor = self.controller.propose_accept(err_norm, tol, self.safety, p);
        // ... accept path: copy s_half into u_curr storage and return ...
    } else {
        let factor = self.controller.propose_reject(err_norm, tol, self.safety, p);
        // ... reject path ...
    }
}
```

**Three allocations** removed per substep: no more `apply -> Vec<F>` returns, no more
`u_half.clone()` for `diff`. Wave 4 substep cost is **0 heap allocations** in the
steady state (modulo first-call buffer allocation).

### 5.4 Norm choice — L² vs sup

The v1.0.0 path used `‖·‖_∞`. Wave 4 switches to `‖·‖_{ℓ²}` (counting-measure) because:
- `HilbertState::dot` is the Wave 3 zero-alloc primitive.
- L² norm is the standard for digital-filter controllers (Söderlind 2003 §2.1).
- Recovering sup-norm would require a third scratch state for the difference.

**Math fidelity question**: changing the norm from sup to L² shifts the *numerical
value* of `err_norm` while keeping the *qualitative behaviour* (both go to zero
as τ → 0 at rate `O(τ^{p+1})`). The bit-equality test in §8 specifically requires
the **accepted-step trajectory** to be invariant, NOT the err_norm itself. As long
as the L²/sup ratio is bounded across the F9 sweep and the controller multipliers
land in the same `[min_ratio, max_ratio]` clamp regime, the accepted-step path is
preserved.

**SAFEGUARD (NORMATIVE)**: the bit-equality test MUST capture v1.0.0 *accepted-τ
trajectory* (not err_norm). If implementation reveals that L²-vs-sup substantively
changes the accepted-τ path on F9, the implementer SHALL switch to a sup-norm
zero-alloc path (allocate one extra scratch state for the diff) and document the
deviation in the test rationale. This is an implementation-level fallback, not a
contract violation.

---

## §6 — Builder API

### 6.1 Default path

```rust
let pi = AdaptivePI::new(func);          // AdaptivePI<C, f64, ClassicalPI<f64>>
let pi = pi.with_tolerance(1e-10, 1e-8); // still ClassicalPI default
```

Behaviour: byte-identical to v1.0.0 accepted-step trajectory on F9.

### 6.2 Opt-in H211b

```rust
let pi = AdaptivePI::new(func)
    .with_controller(H211bFilter::default());      // re-typed
// type: AdaptivePI<C, f64, H211bFilter<f64>>
```

### 6.3 f32 path (NEW)

```rust
let func: DiffusionChernoff<f32> = ...;     // F = f32 via ADR-0025/0026
let pi = AdaptivePI::<_, f32>::new(func)
    .with_tolerance(1e-5_f32, 1e-3_f32);    // f32 ULP-aware tolerances
```

### 6.4 Mutability change

`evolve_adaptive` is now `&mut self`. Callers MUST hold `AdaptivePI` mutably.
Doc-tests and integration tests update accordingly. This is a documented v2.0
breaking change.

---

## §7 — Migration matrix

| Caller pattern (v1.0.0)                          | Wave 4 equivalent                              | Source change? |
|--------------------------------------------------|------------------------------------------------|----------------|
| `let pi = AdaptivePI::new(func);`                | unchanged                                      | none           |
| `let pi = pi.with_tolerance(1e-8, 1e-6);`        | unchanged                                      | none           |
| `let outcome = pi.evolve_adaptive(t, &u0)?;`     | `let mut pi = ...; pi.evolve_adaptive(t,&u0)?` | `let mut`      |
| `pi.alpha` (field read)                          | `pi.alpha()` (method)                          | `()` suffix    |
| `pi.beta` (field read)                           | `pi.beta()`                                    | `()` suffix    |
| (no equivalent — no H211b in v1.0.0)             | `.with_controller(H211bFilter::default())`     | NEW            |
| (no equivalent — f64-only)                       | `AdaptivePI::<_, f32>::new(...)`               | NEW            |

The `let mut pi` requirement is the only **forced** v1.0.0→Wave-4 source change
(plus `.alpha()`/`.beta()` accessor change in one test file). Both are documented
in the v2.0 migration guide.

---

## §8 — Test plan

### 8.1 `tests/adaptive_classical_bit_equal.rs` (NEW — NORMATIVE gate)

**Goal**: prove `ClassicalPI` produces the v1.0.0 accepted-τ trajectory byte-for-byte.

**Strategy**:
1. Capture the v1.0.0 accepted-step trajectory **once** before Wave 4 lands. Generate
   a fixture `tests/fixtures/adaptive_classical_trace_v1.json` (or `.bin`) by running
   the v1.0.0 binary against a fixed `(grid, func, t, u0, tol)` tuple and serialising
   the sequence of `(step_index, tau_accepted)` pairs.
2. Wave 4 test re-runs the same `(grid, func, t, u0, tol)` and asserts equality of
   the trajectory under `AdaptivePI::new(func)` (default `ClassicalPI`).
3. Tuple count: **3 fixtures** — F9 CEV oracle params (the production-critical one),
   smooth 1D heat (sanity), and a stiff diffusion (controller-stress).
4. Proptest layer: random `(tol_rel ∈ [1e-7, 1e-3])` with a fixed seed; assert that
   the accepted-τ trajectory under Wave 4 matches a freshly-recomputed v1.0.0
   reference (build the v1.0.0 helpers as a `#[cfg(test)]` `mod legacy` in the same
   file so the comparison is in-process).

**Fail mode**: if any step's τ differs by > 0 ULP, fail with the diverging step index
and both values.

### 8.2 `tests/adaptive_f9_cev_variance.rs` (NEW — H211b acceptance gate)

**Goal**: prove H211b reduces F9 step-size variance by ≥ 2× without loss of L²
accuracy.

**Setup**:
- Reuse the F9 CEV oracle params from `tests/cev_european_call.rs` (same `S0, K,
  R, SIGMA0, BETA_PDE, T, X_MIN, X_MAX, N_GRID`).
- Compute `reference` = high-resolution Schroder closed form (already in the F9
  oracle as the noncentral-χ² CDF series).
- Run two adaptive evolutions:
  - `classical = AdaptivePI::new(func).evolve_adaptive(T, &u0)?`
  - `h211b = AdaptivePI::new(func).with_controller(H211bFilter::default())
            .evolve_adaptive(T, &u0)?`
- Collect accepted-τ sequence from each via a `#[cfg(test)]` instrumentation hook
  on `AdaptivePI` (a `tau_log: Option<&mut Vec<F>>` field gated by `#[cfg(test)]`,
  or — cleaner — a public `AdaptiveOutcomeTrace { taus: Vec<F> }` returned from a
  test-only `evolve_adaptive_with_trace` method).

**Assertions**:
1. `iqr(h211b.taus) <= iqr(classical.taus) / 2.0` — IQR reduction ≥ 2×.
2. `l2_err(h211b.final_state, reference) <= 1.05 * l2_err(classical.final_state, reference)` —
   L² error ratio ≤ 1.05.

**Tolerance**: `tol_rel = 1e-5`, `tol_abs = 1e-8` (tuned to land in the stiff CEV
regime where H211b shows the strongest benefit).

### 8.3 `tests/adaptive_generic_f32.rs` (NEW — Wave 4 generic-over-F gate)

**Goal**: smoke-test the f32 path.

```rust
let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
let diff = DiffusionChernoff::<f32>::new(/* f32 closures */, 0.5_f32, grid);
let mut pi = AdaptivePI::<_, f32>::new(diff).with_tolerance(1e-4_f32, 1e-3_f32);
let u0 = GridFn1D::<f32>::from_fn(grid, |x| (-x * x).exp());
let outcome = pi.evolve_adaptive(0.1_f32, &u0).unwrap();
assert!(outcome.steps_accepted > 0);
assert!(outcome.last_tau.is_finite());
// Optional: convergence vs analytic Gaussian solution at relaxed f32 tolerance.
```

### 8.4 `tests/adaptive_unit.rs` migration

The six existing unit tests stay in place. Required edits:
- `default_construction`: replace `pi.alpha` / `pi.beta` with `pi.alpha()` /
  `pi.beta()`. Confirm same numeric values.
- All tests: convert `let pi = ...` to `let mut pi = ...` where `evolve_adaptive`
  is called.

### 8.5 Regression — Wave 1 / 2 / 3

All existing tests under `crates/semiflow-core/tests/`:
- `adaptive_2d_heat.rs`, `adaptive_cev_efficiency.rs`,
  `adaptive_chernoff_consistency.rs`, `adaptive_unit.rs` — re-pass.
- `cev_european_call.rs`, `cev_european_call_sweep.rs`,
  `cev_boundary_stress.rs`, `cev_high_lam_oracle.rs` — re-pass.
- All `grid_*`, `strang*`, `diffusion*` tests — unaffected (no AdaptivePI usage).
- Sympy gates (T9N_*, T10N_*, 18 NORMATIVE) — unaffected (symbolic only).
- Slope gates (G1, G2, G3 family, G4_NS2D_aniso, G5_3D) — unaffected (use
  `ChernoffSemigroup`, not `AdaptivePI`).

### 8.6 SIMD bit-equality

Preserved at the leaf-Chernoff layer per existing AVX2/NEON tests. Wave 4 changes
only `adaptive.rs` and adds `controller.rs`; no SIMD kernel is touched.

---

## §9 — Risk table (top 5 + mitigation)

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | Classical step trajectory drifts at ULP (e.g. FP re-association of `safety * e * e_prev`) → F9 oracle fails | **CRITICAL** (math fidelity) | `tests/adaptive_classical_bit_equal.rs` proptest + 3 fixtures; explicit "left-to-right" comment in `ClassicalPI::propose_accept` (§3.4); `#[allow(clippy::suspicious_arithmetic_impl)]` if needed; `cargo test --release` covers SIMD path |
| 2 | H211b state on first step (no `err_prev`, no `r_prev`) over-shoots | HIGH | Seed `err_prev = F::one()` and `r_prev = F::one()` (neutral exponents). Asserted in `propose_accept` on synthetic first-step inputs (`err_norm = tol`). |
| 3 | `ScratchPool` aliasing — Wave 1 enforces single-borrow via `&mut self`, but Wave 4 needs 3 simultaneous state scratch slots | HIGH | Use **state scratch** (3 owned `C::S` slots in `AdaptivePI`) for live cross-call slots; use `ScratchPool<F>` only for vec-level scratch *inside* `apply_into` (which receives `&mut ScratchPool` and is single-borrow-safe). |
| 4 | f32 Richardson norm underflow when `err_norm ~ 1e-20` (below f32 normal range 1e-38 still OK but `safe_err = 1e-300` is below f32 representable) | MEDIUM | Make the "tiny" floor `F::from(1e-30).unwrap()` for f32, `F::from(1e-300).unwrap()` for f64 — or unconditional `F::from(1e-30)` (still safe for f64). Re-verify v1.0.0 bit-equality with the unified floor; if it diverges, use `cfg_if` / generic specialisation hack. **Resolution**: keep `1e-300` literal-via-`F::from` and rely on f32's `F::from(1e-300).unwrap_or(F::min_positive_value())` semantics; document f32 underflow caveat in `SemiflowError::DomainViolation`. |
| 5 | `adaptive.rs` exceeds 500-line file cap after Wave 4 expansion (current 330 + ~150 controller scaffolding + ~30 state scratch) | MEDIUM | **Split**: move `ClassicalPI` and `H211bFilter` into new `crates/semiflow-core/src/controller.rs`. Target sizes: `adaptive.rs` ≤ 420, `controller.rs` ≤ 220. Keeps both well under the 500 cap, no carve-out needed. |

---

## §10 — Build / run / verify

Same workspace commands; no new tooling:

```bash
cargo run -p xtask -- test-fast       # 5–10× faster, default
cargo run -p xtask -- test-full       # SIMD + slow-tests + release
cargo run -p xtask -- test-flagship   # G3⁶-2D + G4_NS2D_aniso + G5_3D
cargo clippy --all-targets --workspace -- -D warnings
cargo doc   --no-deps   --workspace
```

No new direct deps (cap stays at 2). No new feature flags. No new MSRV bump.

---

## §LoC budget

| File                                                | Action  | LoC delta | Final LoC | Cap          |
|-----------------------------------------------------|---------|-----------|-----------|--------------|
| `crates/semiflow-core/src/adaptive.rs`               | EDIT    | +90       | ≤ 420     | 500          |
| `crates/semiflow-core/src/controller.rs`             | NEW     | +220      | 220       | 500          |
| `crates/semiflow-core/src/lib.rs`                    | EDIT    | +4        | (current+4)| 500         |
| `crates/semiflow-core/tests/adaptive_classical_bit_equal.rs` | NEW | +150 | 150 | 500 |
| `crates/semiflow-core/tests/adaptive_f9_cev_variance.rs`     | NEW | +180 | 180 | 500 |
| `crates/semiflow-core/tests/adaptive_generic_f32.rs`         | NEW | +80  | 80  | 500 |
| `crates/semiflow-core/tests/adaptive_unit.rs`        | EDIT    | +6        | (current+6)| 500         |
| `tests/fixtures/adaptive_classical_trace_v1.json`   | NEW     | data      | N/A       | N/A          |
| `docs/adr/0044-stepcontroller-trait-h211b-advisory.md` | NEW (this PR) | +180 | 180 | N/A   |
| `contracts/v2/wave4-stepcontroller.md`              | NEW (this PR) | +500 | 500 | N/A      |
| `contracts/semiflow-core.math.md`                    | **UNCHANGED** | 0 | n/a | n/a         |

**Function-cap audit (50 lines)**:
- `AdaptivePI::evolve_adaptive` — currently 51 lines (adaptive.rs:231–282); Wave 4
  refactor brings it to ~45 lines by extracting the `state_scratch` init into a
  helper.
- `AdaptivePI::substep` — currently 24 lines; Wave 4 brings it to ~38 lines (still
  ≤ 50) by inlining the controller dispatch.
- `ClassicalPI::propose_accept`, `propose_reject` — each ≤ 10 lines.
- `H211bFilter::propose_accept`, `propose_reject` — each ≤ 15 lines.

All within the 50-line cap.

---

## §11 — Per-test verification commands (for QA)

```bash
# Bit-equality gate (NORMATIVE)
cargo test -p semiflow-core --test adaptive_classical_bit_equal

# H211b advisory gate (NEW)
cargo test -p semiflow-core --test adaptive_f9_cev_variance --release

# f32 generic gate (NEW)
cargo test -p semiflow-core --test adaptive_generic_f32

# F9 CEV oracle re-pass
cargo test -p semiflow-core --test cev_european_call           --release
cargo test -p semiflow-core --test cev_european_call_sweep     --release
cargo test -p semiflow-core --test cev_boundary_stress         --release
cargo test -p semiflow-core --test cev_high_lam_oracle         --release

# Full Wave 1/2/3 regression
cargo run -p xtask -- test-full
```

Acceptance: all green; bit-equality test reports zero ULP divergence on the
captured fixtures.
