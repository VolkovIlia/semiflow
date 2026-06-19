# Wave 3 Contract: `State<F>` 3-Layer Trait Split

**Status**: NORMATIVE
**ADR**: docs/adr/0043-state-trait-three-layer-split.md
**Scope**: semiflow-core v2.0 Wave 3
**Depends on**: contracts/v2/wave1-scratch.md, contracts/v2/wave2-inplace-strang.md

---

## §1 — `State<F>` trait surface (Layer 1)

**File**: `crates/semiflow-core/src/state.rs`. Replaces the v1.x `State<F>` trait.

### 1.1 Method signatures (NORMATIVE)

```rust
pub trait State<F: SemiflowFloat = f64> {
    fn len(&self) -> usize;
    fn axpy_into(&mut self, alpha: F, src: &Self);
    fn copy_from(&mut self, src: &Self);
    fn zero_into(&mut self);
    fn norm_sup(&self) -> F;
    // scale_into: default is unimplemented!(); all concrete types override.
    fn scale_into(&mut self, k: F) { unimplemented!("scale_into: override required") }
}
```

`Clone` is **not** a supertrait. Algebraic laws:
- `axpy_into(F::zero(), &x)` is a no-op.
- After `copy_from(&src)`: node-wise equal to `src`, `len()` unchanged.
- After `zero_into()`: `norm_sup() == F::zero()`, `len()` unchanged.
- `len()` invariant under `axpy_into`, `copy_from`, `zero_into`.
- `norm_sup()` returns `NaN` iff any node is `NaN`.

### 1.2 Backward-compat inherent shims on `GridFnXD<F>`

```rust
impl<F: SemiflowFloat> GridFn1D<F> {
    pub fn axpy(&mut self, a: F, x: &Self) { self.axpy_into(a, x); }
    pub fn scale(&mut self, k: F) { self.scale_into(k); }
    pub fn zeroed_like(&self) -> Self { /* clone + zero */ }
}
// identical on GridFn2D<F>, GridFn3D<F>
```

Concrete-type callers: **no source change required**.

---

## §2 — `HilbertState<F>` trait (Layer 2)

```rust
pub trait HilbertState<F: SemiflowFloat = f64>: State<F> {
    fn dot(&self, other: &Self) -> F;
    fn norm_sq(&self) -> F { self.dot(self) }
    fn norm_l2(&self) -> F where F: num_traits::Float { self.norm_sq().sqrt() }
}
```

Counting-measure ℓ²: `dot(a,b) = Σᵢ aᵢ·bᵢ`. No grid-spacing weights.
Symmetric: `dot(a,b) == dot(b,a)`. Non-negative: `norm_sq(a) ≥ 0`.

---

## §3 — `Discrete<F>` trait (Layer 3)

```rust
pub trait Discrete<F: SemiflowFloat = f64>: State<F> {
    type Idx: Copy + Eq + core::hash::Hash;
    type Neighbours<'a>: Iterator<Item = (Self::Idx, F)> where Self: 'a;
    fn get(&self, idx: Self::Idx) -> F;
    fn set(&mut self, idx: Self::Idx, val: F);
    fn indices(&self) -> impl Iterator<Item = Self::Idx> + '_;
    fn neighbours(&self, idx: Self::Idx) -> Self::Neighbours<'_>;
}
```

GAT eliminates `Box<dyn Iterator>` per v0.14.0 spike finding.
Implicit BC: empty `neighbours` → Dirichlet zero.
MSRV 1.78 covers both GATs (stable 1.65) and RPITIT (stable 1.75).

---

## §4 — Concrete trait impls

| Type | `State<F>` | `HilbertState<F>` | `Discrete<F>` |
|------|:----------:|:-----------------:|:-------------:|
| `GridFn1D<F>` | ✅ | ✅ | ❌ |
| `GridFn2D<F>` | ✅ | ✅ | ❌ |
| `GridFn3D<F>` | ✅ | ✅ | ❌ |
| `PathGraphFn<F>` (example) | ✅ | ✅ | ✅ |

Tensor types do NOT implement `Discrete<F>` — no canonical neighbour set;
tensor pipelines use slice pencils (Wave 2), not neighbour iteration.

Shared macro `impl_state_for_gridfn!` in `state.rs` covers all three
tensor types identically, keeping `grid_fn{,2d,3d}.rs` under 500-LoC cap.

---

## §5 — `ChernoffFunction::apply_into` pivot

### Before (v1.x)
```rust
fn apply_into(&self, tau: F, src: &Self::S, dst: &mut Self::S, scratch: &mut ScratchPool<F>)
    -> Result<(), SemiflowError>
{
    let _ = scratch;
    *dst = self.apply(tau, src)?;  // Clone via assignment
    Ok(())
}
```

### After (v2.0)
```rust
fn apply(&self, tau: F, f: &Self::S) -> Result<Self::S, SemiflowError>
where Self::S: Clone;

fn apply_into(&self, tau: F, src: &Self::S, dst: &mut Self::S, scratch: &mut ScratchPool<F>)
    -> Result<(), SemiflowError>
where Self::S: Clone,  // default bridge only
{
    let _ = scratch;
    let tmp = self.apply(tau, src)?;
    dst.copy_from(&tmp);  // explicit zero-alloc copy
    Ok(())
}
```

Override impls (DiffusionChernoff etc.) do NOT need `Clone` — they use
scratch-arena buffers and never allocate a fresh `Self::S`.

---

## §6 — `ChernoffSemigroup::evolve` rewrite

```rust
impl<C, S> ChernoffSemigroup<C, S>
where
    C: ChernoffFunction<f64, S = S>,
    S: State<f64> + Clone,
{
    pub fn evolve(&self, t: f64, f: &S) -> Result<S, SemiflowError> {
        // ...
        let mut buf_a: S = f.clone();
        let mut buf_b: S = f.clone();
        buf_b.zero_into();  // replaces f.zeroed_like() — explicit allocation
        // ping-pong loop unchanged
    }
}
```

`S: Clone` retained for `buf_a/b` initialisation from `f`.

---

## §7 — `cargo semver-checks` integration

Expected-breakage whitelist: `.cargo/semver-checks-allowlist.toml`.

Breaking changes in whitelist:
- `State::axpy` removed (renamed to `axpy_into`; inherent shim retained)
- `State::scale` removed (inherent shim retained)
- `State::zeroed_like` removed (inherent shim retained)
- `State::len` added (new required method)
- `State::axpy_into` added (new required method)
- `State::copy_from` added (new required method)
- `State::zero_into` added (new required method)
- `State` supertrait `Clone` removed
- `HilbertState` trait added (additive)
- `Discrete` trait added (additive)

---

## §8 — Test plan

### 8.1 Carry-forward (must re-pass unchanged)

| Test | Origin | Gate |
|------|--------|------|
| `apply_into_byte_equal.rs` | Wave 1 | 6/6 |
| `zero_alloc_steady.rs` | Wave 1 | 2/2 |
| `parallel_scratch_drain.rs` | Wave 1 | 7/7 |
| `strang_inplace_byte_equal.rs` | Wave 2 | 7/7 |
| `strang_inplace_alloc_count.rs` | Wave 2 | 2/2 |
| T9N_* sympy ×6 | v0.9.0+ | 6/6 |
| T10N_* sympy ×6 | v0.9.0+ | 6/6 |
| T11N_* sympy ×6 | v0.11.0+ | 6/6 |

### 8.2 New tests (Wave 3)

**(a)** `tests/state_trait_contract.rs` — proptest, 10 invariants:
1. `axpy_into(0.0, &x)` no-op.
2. `axpy_into(α, &x)` then `axpy_into(-α, &x)` ≈ no-op (f64 rounding).
3. After `copy_from`: `norm_sup` equal.
4. After `zero_into`: `norm_sup == 0.0`, `len` unchanged.
5. `norm_sup(axpy_into(α, &zero))` == `norm_sup(self)`.
6. `dot(a,b) == dot(b,a)`.
7. `dot(a,a) == norm_sq(a)`.
8. `norm_sq(a) ≥ 0`.
9. After `zero_into`: `dot(s, anything) == 0`.
10. `len()` invariant under `axpy_into`, `copy_from`, `zero_into`.

Tested on `GridFn1D<f64>`, `GridFn2D<f64>`, `GridFn3D<f64>`, `PathGraphFn<f64>`.
Config: 256 cases, 200 max_shrink_iters.

**(b)** `tests/default_bridge_compat.rs` — mock `ChernoffFunction` without
`apply_into` override; verifies default bridge writes `dst` correctly (AC-11).

**(c)** `tests/graph_heat_oracle.rs` — `PathGraphFn` + `GraphHeatChernoff`
against eigenvalue oracle `λ_k = 2(1 − cos(kπ/N))`.
Tolerance: `‖u_chernoff − u_exact‖_∞ < 5e-3` for `N=64, t=0.1, n_steps=400`.

---

## §9 — Migration (summary)

Full guide: `docs/migration/v1-to-v2.md`.

| v1.x trait method | v2.0 | Concrete shim? |
|-------------------|------|:--------------:|
| `s.axpy(a, &x)` | `s.axpy_into(a, &x)` | ✅ |
| `s.scale(k)` | `s.scale_into(k)` | ✅ |
| `s.zeroed_like()` | `clone + zero_into` | ✅ |
| `s.norm_sup()` | unchanged | — |
| `T: State<F>` + `.clone()` | add `+ Clone` bound | — |

---

## §10 — Risk table

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|:----------:|:------:|------------|
| R1 | Macro coherence collision | LOW | HIGH | No blanket impls; documented in rustdoc |
| R2 | Monomorphisation bloat | LOW | LOW | `cargo bloat` check; wasm size gate re-runs |
| R3 | Wave 4 shape mismatch | MEDIUM | HIGH | Wave 4 prototype compiles against Wave 3 surface |
| R4 | SIMD bit-equality regression | LOW | CRITICAL | AC-3 gate; `axpy_into` body identical to v1.x `axpy` |
| R5 | Custom downstream breakage | MEDIUM | MEDIUM | Migration guide; v2.0.0-rc.1 pre-release window |
