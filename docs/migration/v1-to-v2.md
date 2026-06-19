# Migration Guide: semiflow-core v1 → v2.0

**Scope**: v1.0.0 → v2.0.0 (Wave 3, ADR-0043).
**Most users need ZERO code changes.** This guide is for the minority who hit a
breaking pattern.

---

## TL;DR — Decision tree

```
Custom `impl State<F> for MyType`?
  NO  → re-build; if it compiles, you're done.
  YES → see §3.

Calling State methods through a generic bound `T: State<F>`?
  NO  → §1 inherent shims cover you.
  YES → see §2.

Calling `.clone()` on a `T: State<F>` generic?
  → add `+ Clone` to your bound.
```

---

## §1 — What did NOT change (concrete-type inherent methods)

All three concrete grid-function types retain v1.x methods as **inherent** methods:

```rust
impl<F: SemiflowFloat> GridFn1D<F> {
    pub fn axpy(&mut self, a: F, x: &Self) { /* v1.x semantics */ }
    pub fn scale(&mut self, k: F) { /* v1.x semantics */ }
    pub fn zeroed_like(&self) -> Self { /* same-shape zero state */ }
}
// Same on GridFn2D<F>, GridFn3D<F>.
```

Code like `u.axpy(0.5, &v)` or `let z = v.zeroed_like()` on a concrete
`GridFnXD` compiles **unchanged** against v2.0.

---

## §2 — Method-mapping table

| v1.x trait method | v2.0 trait method | Notes |
|-------------------|-------------------|-------|
| `s.axpy(a, &x)` | `s.axpy_into(a, &x)` | Same semantics, zero allocation |
| `s.scale(k)` | `s.scale_into(k)` | Default is `unimplemented!`; override or use concrete shim |
| `s.zeroed_like()` | `clone + zero_into` | See §2.1 |
| `s.norm_sup()` | `s.norm_sup()` | Unchanged |
| `T: State<F>` + `.clone()` | Add `+ Clone` to bound | `Clone` removed from supertrait |

### 2.1 Migrating `zeroed_like()` through a generic bound

v1.x:
```rust
fn helper<S: State<f64>>(s: &S) -> S {
    let mut z = s.zeroed_like();
    z.axpy(0.5, s);
    z
}
```

v2.0 Option A (Clone-based, allocates once):
```rust
fn helper<S: State<f64> + Clone>(s: &S) -> S {
    let mut z = s.clone();
    z.zero_into();
    z.axpy_into(0.5, s);
    z
}
```

v2.0 Option B (scratch-arena, zero allocation in steady state):
```rust
// See contracts/v2/wave1-scratch.md for the full scratch-arena pattern.
fn helper<S: State<f64>>(s: &S, scratch: &mut ScratchPool<f64>, /* fresh buf */ dst: &mut S) {
    dst.copy_from(s);
    dst.axpy_into(0.5, s);
}
```

Use Option B for hot loops; Option A for one-shot setup.

---

## §3 — Custom `impl State<F>` migration

New required surface:

```rust
use semiflow_core::{State, SemiflowFloat};

impl<F: SemiflowFloat> State<F> for MyType<F> {
    fn len(&self) -> usize { /* NEW */ }
    fn axpy_into(&mut self, alpha: F, src: &Self) { /* replaces axpy */ }
    fn copy_from(&mut self, src: &Self) { /* NEW */ }
    fn zero_into(&mut self) { /* NEW */ }
    fn norm_sup(&self) -> F { /* unchanged */ }
    fn scale_into(&mut self, k: F) {
        // Must override — default is unimplemented!()
        for v in &mut self.data { *v = *v * k; }
    }
}
```

Optional Hilbert extension:
```rust
impl<F: SemiflowFloat> semiflow_core::HilbertState<F> for MyType<F> {
    fn dot(&self, other: &Self) -> F { /* node-wise sum of products */ }
    // norm_sq, norm_l2 inherit defaults
}
```

Optional graph extension (non-tensor types only):
```rust
impl<F: SemiflowFloat> semiflow_core::Discrete<F> for MyType<F> {
    type Idx = u32;
    type Neighbours<'a> = MyNeighbourIter<'a> where Self: 'a;
    fn get(&self, idx: u32) -> F { /* ... */ }
    fn set(&mut self, idx: u32, val: F) { /* ... */ }
    fn indices(&self) -> impl Iterator<Item = u32> + '_ { /* ... */ }
    fn neighbours(&self, idx: u32) -> Self::Neighbours<'_> { /* ... */ }
}
```

Removed from v1.x required surface: `Clone` supertrait, `axpy`, `scale`, `zeroed_like`.

---

## §4 — Why `scale_into` is `unimplemented!` by default

A pure-trait default for `scale` without aliasing tricks is impossible in Rust.
Rather than ship a default that silently allocates, v2.0's `scale_into` is
`unimplemented!()`. Every concrete type in `semiflow-core` overrides it with a
3-line loop. Concrete `GridFnXD<F>` users use inherent `scale(k)` and never notice.

---

## §4.1 — `AdaptivePI` generic parameter

`AdaptivePI` now takes a third type parameter `K: StepController<F>`:

```rust
// v1.x
use semiflow_core::AdaptivePI;
let api: AdaptivePI<MyC> = AdaptivePI::new(inner, tol, dt_min, dt_max);

// v2.0 — ClassicalPI is byte-equal with the v1.x controller
use semiflow_core::{AdaptivePI, ClassicalPI};
let api: AdaptivePI<MyC, f64, ClassicalPI> = AdaptivePI::new(inner, tol, dt_min, dt_max);
// or, using type inference:
let api = AdaptivePI::<_, f64, ClassicalPI>::new(inner, tol, dt_min, dt_max);
```

`ClassicalPI` produces bit-identical step-size decisions to the v1.x hardcoded
controller. `H211bFilter` is a new opt-in type for higher-order step rejection.

---

## §4.2 — Zero-copy `evolve_into` (bindings only)

If you call `evolve` through FFI, PyO3, or WASM and allocate a result buffer on
every step, switch to the new zero-copy path:

| Language | v1.x | v2.0 |
|----------|------|-------|
| Python | `u1 = heat.evolve(t, u0)` | `heat.evolve_into(t, u0, u1_buf)` |
| WASM/JS | `const u1 = heat.evolve(t, u0)` | `heat.evolveInto(t, u0, u1Buf)` |
| C | `smf_evolve(state, t, src, &dst)` | `remizov_state_apply_into(state, tau, src, dst)` |

The `evolve` methods remain available; the `evolve_into` / `evolveInto` /
`apply_into` variants avoid the per-step `Vec` clone at the language boundary.

---

## §4.3 — Graph PDE (new in v2.1)

Graph PDE types (`Graph<F>`, `Laplacian<F>`, `GraphSignal<F>`,
`GraphHeatChernoff`, `GraphHeat4thChernoff`, `StrangSplitGraph`,
`MagnusGraphHeatChernoff`) are **new API in v2.1**. There is no v1.x
counterpart; no migration is required. See the README "Graph PDE (v2.1+)"
quickstart for usage.

---

## §5 — `cargo semver-checks` setup

```bash
cargo install cargo-semver-checks --locked
cargo semver-checks --baseline-rev v1.0.0 --release-type major --workspace
```

Expected-breakage whitelist: `.cargo/semver-checks-allowlist.toml` (see
`contracts/v2/wave3-state-trait.md §7`).

---

## §6 — FAQ

**Q: Will my v1.x code keep compiling?**
A: If you use concrete `GridFnXD` types, yes. If you used generic `State<F>` bounds,
see §2. If you wrote `impl State for MyType`, see §3.

**Q: Why was `Clone` removed from the `State` supertrait?**
A: `Clone` made allocation invisible. Forcing explicit `+ Clone` bounds makes cost
auditable. Concrete types still derive `Clone` — nothing changes for them.

**Q: Do I need the Wave 1 scratch arena?**
A: No. The default-bridge `apply_into` still works. Scratch is a performance option.

**Q: Does v2.0 change any numerical results?**
A: No. All per-node arithmetic is byte-identical on f64. All 6 slope gates and
18 NORMATIVE sympy gates re-pass.

**Q: When will `WeightedGraphFn` ship?**
A: v2.x post-release. The `Discrete<F>` trait ships in v2.0; concrete production
graph types follow based on user demand.
