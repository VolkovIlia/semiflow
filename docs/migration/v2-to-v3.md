# Migration Guide: semiflow-core v2.x → v3.0

**Scope**: v2.8.0 → v3.0.0 (ADR-0073 + ADR-0074 + ADR-0075 + ADR-0076; 2026-05-27).
**v3.0 is the FIRST BREAKING release since v2.0.0.** Some changes require caller migration; a 12-month deprecation shim (`feature = "v2_compat"`, default-ON in v3.x) cushions most paths.

> **v3.0 Wave G COMPLETE** (2026-05-27). This file was the v3.0 Wave A scaffold (architect-authored);
> all per-binding worked examples and verification procedures have been filled in by Wave G
> after the v3.0 trait surface and binding redesign compiled and passed G_binding_parity sub-tests 1+3.

---

## §1 — What's BREAKING in v3.0 (3-line summary per change)

The v3.0 BREAKING window touches **4 surfaces**:

1. **`ChernoffFunction<F>` trait** (ADR-0074) — `apply` method removed from trait; `Self::S: Clone` bound dropped; `growth() -> Growth<F>` (was `(f64, f64)`); `order()` required no-default. **Shim**: `feature = "v2_compat"` default-ON in v3.x; deprecation warnings, not hard errors.
2. **`ChernoffSemigroup<C>` struct** (ADR-0074) — renamed to `Evolver<C, F>`. **Shim**: `pub type ChernoffSemigroup<C> = Evolver<C, f64>` deprecation-warned alias for 12 months.
3. **`growth()` return type** (ADR-0074) — `(f64, f64)` tuple becomes `Growth<F>` struct with `.multiplier` / `.omega` fields. **NO shim possible** (field-rename across types); caller migration required at v3.0 upgrade.
4. **FFI / PyO3 / WASM binding surfaces** (ADR-0076) — v3 surfaces with bare-name canonical symbols; v2 shim surfaces with `_v2` / `_V2` suffix. **Shim**: v2 shim active for 12 months (`feature = "v2_compat"` for FFI/PyO3; per-class console.warn for WASM).

The v3.0 release also ships **2 ADDITIVE** surfaces (no migration; opt-in):

5. **`ApproximationSubspace<const K: usize, F>` super-trait** (ADR-0073) — opt-in K-jet witness marker. NEW capability; no v2.x equivalent.
6. **`Diffusion4thZeta4Chernoff<F>` kernel** (ADR-0075) — opt-in order-4 temporal ζ⁴ correction kernel. NEW capability; sibling to v0.6.0 `Diffusion4thChernoff<F>` (no replacement).

---

## §2 — Migration table (v2 → v3 per-API-call)

| v2.x surface | v3.0 surface | Shim? | Migration kind |
|---|---|---|---|
| `func.apply(τ, &f)?` | `func.apply_chernoff(τ, &f)?` (allocating, `C::S: Clone`) OR `func.apply_into(τ, &src, &mut dst, &mut pool)?` (zero-alloc, recommended) | `feature = "v2_compat"` (default-ON v3.x; HARD REMOVED v4.0) | API rename or zero-alloc port |
| `func.growth().0` (tuple `.0` access for M) | `func.growth().multiplier` | NONE — field-rename across types | Caller migration required at v3.0 upgrade |
| `func.growth().1` (tuple `.1` access for ω) | `func.growth().omega` | NONE | Caller migration required |
| `let (m, om) = func.growth();` | `let g = func.growth(); let m = g.multiplier; let om = g.omega;` OR `let Growth { multiplier: m, omega: om } = func.growth();` | NONE | Caller destructure migration |
| `ChernoffSemigroup::new(c, n)?` | `Evolver::new(c, n)?` | `pub type ChernoffSemigroup<C> = Evolver<C, f64>` deprecation alias | Type-rename |
| `where C::S: Clone` trait-bound | Keep at consumer site (moves from trait-implicit to consumer-explicit) | NONE NEEDED (compiles unchanged) | No code change |
| `impl ChernoffFunction<F> for X { fn apply(...) {...} }` (custom impl) | `impl ChernoffFunction<F> for X { fn apply_into(...) {...} }` (rewrite to zero-alloc) | NONE | API rewrite |
| FFI `smf_chernoff_semigroup_new(func, n)` | FFI `smf_evolver_new(func, n)` | `smf_chernoff_semigroup_new_v2(func, n)` (v2 shim in `remizov_v2.h`) | C header rename |
| FFI `smf_apply(func, τ, src, dst)` | FFI `smf_apply_into(func, τ, src, dst, scratch)` | `smf_apply_v2(func, τ, src, dst)` (v2 shim) | API rewrite (allocating → zero-alloc) |
| FFI `smf_growth_m(func)`, `smf_growth_omega(func)` | FFI `smf_growth_multiplier(func)`, `smf_growth_omega(func)` | `smf_growth_m_v2(func)` alias | Field rename |
| PyO3 `heat.evolve(0.5, psi0)` (allocating) | unchanged — still allocating convenience (NOT deprecated) | n/a | No migration |
| PyO3 `m, om = heat.growth()` (tuple destructure) | unchanged — namedtuple destructures positionally (NOT deprecated) | n/a | No migration |
| PyO3 `heat.growth()[0]` (positional access) | unchanged — namedtuple positional access works (NOT deprecated) | n/a | No migration |
| PyO3 `heat.growth().M` (field access on tuple — invalid in v2 anyway) | `heat.growth().multiplier` | NONE | Field rename |
| PyO3 `ChernoffSemigroup(func, n)` | `Evolver(func, n)` | `ChernoffSemigroup` deprecation-warned alias | Type rename |
| WASM `RemizovWasm.version() === "2.x"` | `RemizovWasm.version() === "3.0"` | `RemizovWasm.versionV2() === "2.x"` shim | Version metadata |
| WASM `heat.evolve(t, psi0)` (allocating) | unchanged — still allocating convenience | n/a | No migration |
| WASM `heat.growth() === [M, omega]` (Array return) | `heat.growth() === {multiplier, omega}` (object return) | `heat.growthV2() === [M, omega]` Array shim | Array → object migration |
| WASM `new ChernoffSemigroup(func, n)` | `new Evolver(func, n)` | `ChernoffSemigroup` alias (console.warn on construction) | Type rename |

---

## §3 — Deprecation timeline (12-month cadence per ADR-0035 §9)

| Version | Date (target) | Shim state | Migration urgency |
|---|---|---|---|
| **v3.0.0** (release) | 2026-05-27 (this release) | All shims active by default (feature `v2_compat` = on) | Deprecation warnings on v2 calls; no hard errors |
| **v3.1.0 – v3.x.y** (next 12 months) | 2026-05 → 2027-05 | Shims continue; warnings escalate at each MINOR release per ADR-0035 §9 pattern | Migrate at convenience over 12 months |
| **v4.0.0** (target +12 months) | 2027-05-27 (target) | Shim files (`v2_compat.rs`, `remizov_v2.h`, PyO3 `*_v2` methods, WASM `*V2` methods) **DELETED**; `feature = "v2_compat"` removed | **HARD REMOVAL** — non-migrated callers FAIL TO COMPILE |

**Shim-active features that will be removed at v4.0**:
- `crates/semiflow-core/src/v2_compat.rs` (~120 LoC; deleted)
- `crates/semiflow-ffi/src/v2_shim.rs` (~300 LoC; deleted)
- `crates/semiflow-ffi/include/remizov_v2.h` (cbindgen-generated; deleted)
- PyO3 `*_v2` methods (`growth_v2`, `ChernoffSemigroup` alias; deleted)
- WASM `*V2` methods (`growthV2`, `versionV2`, `ChernoffSemigroup` alias; deleted)

---

## §4 — Compile-time vs runtime breaks (clearly marked)

**COMPILE-TIME breaks (caller code fails to compile until migrated)**:
- `func.growth().0` / `.1` tuple access — `Growth<F>` is a struct; tuple-index access is rejected by the compiler. **No shim possible**; migrate at v3.0 upgrade.
- `let (m, om) = func.growth();` tuple destructure — **partially shimmable** via Python namedtuple semantics in the PyO3 binding (works unchanged); fully BREAKING in Rust (must use `let Growth { multiplier: m, omega: om } = func.growth();`).
- WASM `heat.growth()[0]` Array index access — `heat.growth()` returns object in v3; the Array-index access is `undefined`. Migrate to `heat.growth().multiplier`.

**COMPILE-TIME warnings (caller code compiles; deprecation warning emitted)**:
- `func.apply(τ, &f)?` — shim active under `feature = "v2_compat"`; compiles with `#[deprecated]` warning.
- `ChernoffSemigroup::new(c, n)?` — shim alias active; compiles with `#[deprecated]` warning at instantiation.
- FFI `#include "remizov_v2.h"` — cbindgen generates the v2 shim header with `#warning "remizov_v2.h is deprecated as of v3.0.0; migrate to remizov.h"` at the top.
- PyO3 `from semiflow import ChernoffSemigroup` — `DeprecationWarning` raised at module load.
- WASM `import {ChernoffSemigroup} from '@semiflow/wasm'` — `console.warn` emitted at construction.

**RUNTIME breaks (only affects callers using runtime feature-detection)**:
- WASM `RemizovWasm.version() === "2.x"` runtime check — now returns `"3.0"`; use `RemizovWasm.versionV2() === "2.x"` shim or update the check to allow `"3.x"`.

---

## §5 — Per-binding migration (Rust, FFI C, PyO3 Python, WASM JS)

### §5.1 — Rust (`semiflow-core` crate consumer)

The three most common v2.x patterns and their v3.0 equivalents:

**Pattern A — `apply` + `growth` tuple destructure (most common)**

```rust
// v2.x (compiles with deprecation warnings under v3.x via v2_compat feature)
use semiflow_core::{ChernoffSemigroup, DiffusionChernoff, Grid1D};

let grid = Grid1D::<f64>::new(-5.0, 5.0, 512)?;
let inner = DiffusionChernoff::<f64>::new(|_x| 1.0, grid.clone())?;
let semigroup = ChernoffSemigroup::new(inner.clone(), 64)?;       // deprecated alias
let next_state = semigroup.chernoff.apply(0.01, &state)?;         // deprecated shim
let (m, omega) = semigroup.chernoff.growth();                     // COMPILE ERROR in v3.x — Growth<F> is not a tuple

// v3.0 — rename + Growth<F> struct destructure
use semiflow_core::{Evolver, DiffusionChernoff, Grid1D, Growth};

let grid = Grid1D::<f64>::new(-5.0, 5.0, 512)?;
let inner = DiffusionChernoff::<f64>::new(|_x| 1.0, grid.clone())?;
let evolver = Evolver::new(inner.clone(), 64)?;

// Option 1: zero-alloc (recommended, matches v2.0 apply_into pattern)
let mut scratch = ScratchPool::new();
evolver.chernoff.apply_into(0.01, &state, &mut next_state, &mut scratch)?;

// Option 2: allocating convenience (same semantics as v2.x apply)
let next_state = evolver.chernoff.apply_chernoff(0.01, &state)?; // Clone bound at call site

// Growth<F> access — struct fields, not tuple indices
let growth: Growth<f64> = evolver.chernoff.growth();
let m = growth.multiplier;      // was: growth.0
let omega = growth.omega;       // was: growth.1

// Alternatively, destructure directly:
let Growth { multiplier: m, omega } = evolver.chernoff.growth();
```

**Pattern B — custom `ChernoffFunction` implementation (trait rewrite)**

```rust
// v2.x custom impl
use semiflow_core::{ChernoffFunction, Growth, ScratchPool};

struct MyKernel { /* ... */ }

impl ChernoffFunction<f64> for MyKernel {
    type S = GridFn1D<f64>;

    fn apply(&self, tau: f64, f: &Self::S) -> Result<Self::S, SemiflowError> {
        // allocating implementation
        let mut out = f.clone();
        // ... kernel logic ...
        Ok(out)
    }

    fn order(&self) -> u32 { 2 }

    fn growth(&self) -> (f64, f64) { (1.0, 0.0) }  // COMPILE ERROR in v3.x
}

// v3.0 custom impl — apply_into replaces apply; Growth<F> replaces tuple
impl ChernoffFunction<f64> for MyKernel {
    type S = GridFn1D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        // zero-alloc implementation writing into dst
        // ... kernel logic ...
        Ok(())
    }

    fn order(&self) -> u32 { 2 }  // required; no default in v3.x

    fn growth(&self) -> Growth<f64> {
        Growth { multiplier: 1.0, omega: 0.0 }  // named fields; no tuple
    }
}
```

**Pattern C — opt-in `ApproximationSubspace<K, F>` (new in v3.0, additive)**

```rust
// v3.0 only — no v2.x equivalent
use semiflow_core::{ApproximationSubspace, DiffusionChernoff};

let inner = DiffusionChernoff::<f64>::new(|_x| 1.0, grid.clone())?;

// Declare: DiffusionChernoff<f64> already carries the built-in K=2 + K=4 markers.
// Witness check at runtime (expensive; run once, not per step):
if inner.in_subspace::<4>(&initial_condition) {
    println!("Initial condition is in D(A^4) — order-4 convergence expected");
}

// Compute K-jet (A^0 f, A^1 f, ..., A^K f) into out_jets:
let mut out_jets = [initial_condition.clone(); 5];  // K+1 elements for K=4
inner.jet::<4>(&initial_condition, &mut out_jets)?;
```

**v2_compat shim path (zero-migration, deprecation warnings only)**

```rust
// Under default-ON feature = "v2_compat" in v3.x, the following still compiles:
use semiflow_core::v2_compat::{ChernoffSemigroup, ApplyShim};  // deprecated re-export

let semigroup = ChernoffSemigroup::new(inner, 64)?;           // deprecated alias for Evolver<_, f64>
let next_state = semigroup.chernoff.apply(0.01, &state)?;     // deprecated blanket shim method

// HARD REMOVED at v4.0: v2_compat feature deleted; above code fails to compile.
// Migrate before v4.0 (targeted ~2027-05 per ADR-0035 §9).
```

### §5.2 — FFI C consumer (`semiflow-ffi` crate via cdylib + remizov.h)

The v3.0 FFI surface adds six new `_v3`-suffixed entry points alongside the unchanged v2 surface (ADR-0076 Approach A). Both are emitted in the regenerated `remizov.h`.

**Pattern A — create + evolve + query growth (full v3 surface)**

```c
#include "remizov.h"   /* v3.0: includes both v2 + v3 surfaces */

/* v2.x — still compiles unchanged (v2 surface not renamed in v3.0) */
SemiflowStatus v2_status;
HandleState* state = smf_state_new_heat_1d_unit(-5.0, 5.0, 512, 64, &v2_status);
smf_state_evolve_into(state, 0.01, src_values, dst_values, 512, &v2_status);
smf_state_free(state);

/* v3.0 — additive; use smf_evolver_*_v3 entry points + SmfGrowthV3 struct */
SemiflowStatus v3_status;
HandleEvolverV3* evolver = smf_evolver_new_heat_1d_unit_v3(
    -5.0, 5.0,    /* domain_lo, domain_hi */
    512,          /* n_grid */
    64,           /* n_chernoff */
    &v3_status
);
if (v3_status != REMIZOV_OK) { /* handle error */ }

/* Evolve: writes result into dst_grid_fn (double*, 512 elements) */
smf_evolver_evolve_into_v3(
    evolver, 0.01,
    src_grid_fn, dst_grid_fn,  /* double* src, double* dst */
    &v3_status
);

/* Query growth */
SmfGrowthV3 growth = smf_growth_v3(evolver);
double m = growth.multiplier;   /* was: smf_growth_m(state) via v2 */
double omega = growth.omega;    /* unchanged field name */

/* Size query */
size_t n = smf_evolver_size_v3(evolver);

/* Read current values */
double values[512];
smf_evolver_values_v3(evolver, values, 512, &v3_status);

smf_evolver_free_v3(evolver);  /* null-safe */
```

**Pattern B — CMakeLists.txt update for v3 surface**

```cmake
# v2.x CMakeLists.txt (still works; v2 surface unchanged in v3.0)
target_include_directories(myapp PRIVATE ${REMIZOV_INCLUDE_DIR})  # includes remizov.h

# v3.0 — no build-script change needed for additive Approach A.
# Both v2 and v3 symbols are in the same remizov.h / libsemiflow_ffi.so.
# To opt INTO v3 surface only (removes v2 from your compile unit):
# target_compile_definitions(myapp PRIVATE REMIZOV_USE_V3_ONLY)
# (no effect on linking; purely a header-guard to suppress v2 declarations
#  and force migration at v4.0-prep time — recommended for new projects)
target_link_libraries(myapp PRIVATE ${REMIZOV_LIB_DIR}/libsemiflow_ffi.so)
```

**Deprecation note**: the v2 FFI surface (`smf_state_*`, `smf_apply_*`) is NOT renamed to `_v2` suffix in v3.0 (Approach A chosen over Approach B). It will be hard-removed at v4.0 without a `_v2` intermediate. Plan migration to `smf_evolver_*_v3` before v4.0.

### §5.3 — PyO3 Python consumer (`semiflow-py` wheel via pip)

The v3.0 PyO3 surface adds `EvolverHeat1DUnitV3` and `GrowthV3` classes alongside the unchanged `Heat1D` class (ADR-0076 Approach A). GIL is released during computation via `py.detach` (ADR-0031).

**Pattern A — v2.x code that continues to work (no changes required)**

```python
from semiflow import Heat1D

heat = Heat1D(-5.0, 5.0, 512)
heat.setValues(initial_values)      # np.ndarray of 512 f64

# These three patterns continue unchanged in v3.0:
heat.evolve(0.01)                   # allocating; no deprecation warning
out = heat.values()                 # → np.ndarray

m, omega = heat.growth()            # positional tuple destructure: UNCHANGED
m = heat.growth()[0]                # positional index: UNCHANGED
m = heat.growth().multiplier        # named-field access: UNCHANGED (namedtuple)

# Only this pattern triggers a deprecation warning:
from semiflow import ChernoffSemigroup   # DeprecationWarning at import
evolver = ChernoffSemigroup(heat, 64)      # deprecated alias
```

**Pattern B — v3.0 recommended path with `EvolverHeat1DUnitV3`**

```python
from semiflow import EvolverHeat1DUnitV3, GrowthV3
import numpy as np

evolver = EvolverHeat1DUnitV3(-5.0, 5.0, 512, 64)  # (lo, hi, n_grid, n_chernoff)

# Zero-alloc evolve into pre-allocated buffers (GIL released during compute)
src = np.ascontiguousarray(initial_values, dtype=np.float64)
dst = np.empty(512, dtype=np.float64)
evolver.evolve_into(0.01, src, dst)

# Read current state
values: np.ndarray = evolver.values()   # copy of current state
n: int = len(evolver)                   # == evolver.size() == 512

# Growth via GrowthV3 pyclass (richer than namedtuple)
growth: GrowthV3 = evolver.growth()
print(f"multiplier={growth.multiplier}, omega={growth.omega}")
print(repr(growth))  # GrowthV3(multiplier=1.0, omega=0.0)
```

**Pattern C — type-stub (`.pyi`) update for downstream type-checked consumers**

```python
# Before v3.0 (semiflow/__init__.pyi excerpt):
class Heat1D:
    def growth(self) -> tuple[float, float]: ...   # namedtuple; works in both v2 and v3

# After v3.0 (semiflow/__init__.pyi excerpt — already shipped in Wave E):
class GrowthV3:
    multiplier: float
    omega: float
    def __repr__(self) -> str: ...

class EvolverHeat1DUnitV3:
    def __init__(self, domain_lo: float, domain_hi: float,
                 n_grid: int, n_chernoff: int) -> None: ...
    def evolve_into(self, t: float,
                    src: np.ndarray, dst: np.ndarray) -> None: ...
    def growth(self) -> GrowthV3: ...
    def values(self) -> np.ndarray: ...
    def size(self) -> int: ...
    def n_chernoff(self) -> int: ...
    def __len__(self) -> int: ...
```

### §5.4 — WASM JS consumer (`@semiflow/wasm` npm package)

The v3.0 WASM surface adds `EvolverHeat1DUnitV3` and `GrowthV3` JS classes alongside the unchanged `Heat1D` class (ADR-0076 Approach A, Wave F).

**Pattern A — `growth()` return-type migration (BREAKING)**

```javascript
// v2.x — growth() returned [multiplier, omega] Array
import init, { Heat1D } from "@semiflow/wasm";
await init();
const heat = new Heat1D(-5.0, 5.0, 512);
heat.setValues(initial);
heat.evolveInto(0.01);

// BREAKING: growth() now returns {multiplier, omega} object in v3.0
const [M, omega] = heat.growth();   // v2: worked; v3: BROKEN (destructures object incorrectly)
// Migrate to:
const { multiplier: M, omega } = heat.growth();  // v3.0 named destructure
// OR use the deprecation-warned shim:
const [M_shim, omega_shim] = heat.growthV2();    // v2 shim (console.warn on call)

// Also BREAKING:
const M_idx = heat.growth()[0];   // v2: worked; v3: undefined (not an Array)
// Migrate to:
const M_named = heat.growth().multiplier;  // v3.0
```

**Pattern B — v3.0 recommended path with `EvolverHeat1DUnitV3`**

```javascript
import init, { EvolverHeat1DUnitV3, GrowthV3 } from "@semiflow/wasm";
await init();

const evolver = new EvolverHeat1DUnitV3(-5.0, 5.0, 512, 64);  // (lo, hi, n_grid, n_chernoff)

// Provide initial values via Float64Array
const initial = new Float64Array(512).fill(0.0);
initial[256] = 1.0;  // delta-like initial condition
evolver.setValues(initial);

// Evolve (mutates internal state)
evolver.evolveInto(0.01);

// Read current state
const values = evolver.values();    // Float64Array (copy)
const n = evolver.size();           // 512

// Growth via GrowthV3 class
const growth = evolver.growth();    // GrowthV3
console.log(growth.multiplier, growth.omega);  // 1.0, 0.0

// Cleanup (WASM objects are GC'd but explicit free is good practice)
evolver.free();
growth.free();
```

**Pattern C — npm package.json update**

```json
{
  "dependencies": {
    "@semiflow/wasm": "^3.0.0"
  }
}
```

Upgrade from `^2.x.x` is a package.json one-liner; the bundle size is unchanged (same cdylib, same wasm32 target). TypeScript declarations are auto-generated by wasm-bindgen and ship in the npm package — update your `@semiflow/wasm` import and the new types become available immediately.

---

## §6 — Worked examples

Three realistic end-to-end migration scenarios. For more examples see:
- `crates/semiflow-core/tests/approximation_subspace.rs` (G_AS_K gate, 15 tests)
- `crates/semiflow-ffi/tests/ffi_v3_smoke.rs` (7 FFI v3 smoke tests)
- `crates/semiflow-py/tests/test_v3_smoke.py` (16 PyO3 v3 smoke tests, G_binding_parity sub-test 3)
- `crates/semiflow-wasm/tests/v3_smoke.rs` (11 wasm-bindgen-test tests)

### §6.1 — CEV option pricing via DiffusionChernoff (Rust → full v3)

**Scenario**: price a CEV option (constant elasticity of variance `a(x) = x^{2β}`) using the existing `DiffusionChernoff<f64>` kernel, migrating from v2.x `ChernoffSemigroup` to v3 `Evolver`.

```rust
// v2.x — CEV pricer (compiles with deprecation warnings in v3.x)
use semiflow_core::{ChernoffSemigroup, DiffusionChernoff, Grid1D, GridFn1D};

fn price_cev_v2(
    beta: f64,
    s0: f64,
    strike: f64,
    t_expiry: f64,
    n_grid: usize,
    n_chernoff: usize,
) -> Result<f64, SemiflowError> {
    let grid = Grid1D::<f64>::new(0.1, 300.0, n_grid)?;   // avoid x=0 singularity
    let cev_diffusion = move |x: f64| x.powf(2.0 * beta);
    let inner = DiffusionChernoff::<f64>::new(cev_diffusion, grid.clone())?;

    // v2.x: ChernoffSemigroup wraps inner + n_chernoff steps
    let semigroup = ChernoffSemigroup::new(inner.clone(), n_chernoff)?;  // deprecated

    // Initial condition: call payoff (S − K)+
    let mut psi0 = GridFn1D::<f64>::new(grid.clone());
    for (i, &x) in grid.nodes().iter().enumerate() {
        psi0.set(i, (x - strike).max(0.0));
    }

    // Evolve: deprecated apply method
    let psi_t = semigroup.chernoff.apply(t_expiry, &psi0)?;          // deprecated

    // Sample at spot: both tuple and named access work in v2.x
    let (m, _omega) = semigroup.chernoff.growth();                    // COMPILE ERROR in v3.x
    let price = psi_t.sample(s0)?;
    Ok(price * m.powi(n_chernoff as i32))
}

// v3.0 — CEV pricer (clean; no deprecation warnings)
use semiflow_core::{Evolver, DiffusionChernoff, Grid1D, GridFn1D, Growth, ScratchPool};

fn price_cev_v3(
    beta: f64,
    s0: f64,
    strike: f64,
    t_expiry: f64,
    n_grid: usize,
    n_chernoff: usize,
) -> Result<f64, SemiflowError> {
    let grid = Grid1D::<f64>::new(0.1, 300.0, n_grid)?;
    let cev_diffusion = move |x: f64| x.powf(2.0 * beta);
    let inner = DiffusionChernoff::<f64>::new(cev_diffusion, grid.clone())?;

    // v3.0: Evolver (was ChernoffSemigroup)
    let evolver = Evolver::new(inner.clone(), n_chernoff)?;

    // Initial condition: unchanged
    let mut psi0 = GridFn1D::<f64>::new(grid.clone());
    for (i, &x) in grid.nodes().iter().enumerate() {
        psi0.set(i, (x - strike).max(0.0));
    }

    // Option A: allocating convenience (simplest migration path)
    let psi_t = evolver.chernoff.apply_chernoff(t_expiry, &psi0)?;

    // Option B: zero-alloc (recommended for production loops)
    let mut psi_t = GridFn1D::<f64>::new(grid.clone());
    let mut scratch = ScratchPool::new();
    evolver.chernoff.apply_into(t_expiry, &psi0, &mut psi_t, &mut scratch)?;

    // v3.0: Growth<F> struct — named fields, not tuple
    let Growth { multiplier: m, .. } = evolver.chernoff.growth();

    let price = psi_t.sample(s0)?;
    Ok(price * m.powi(n_chernoff as i32))
}
```

### §6.2 — Heston 2D pricing via NonSeparable2D (Rust → Evolver + v3 struct)

**Scenario**: Heston model 2D PDE `∂_t u = (κ(θ−v)∂_v + ½v∂²_S + ξ²v∂²_v + ρξv∂_S∂_v) u` using `NonSeparable2DChernoff` (v0.7+) + `StrangSplit`. The Heston example from `examples/heston_pricer.rs` uses this pattern at the v2.7 level; here we show the v2 → v3 migration diff.

```rust
// v2.x heston_pricer.rs snippet (simplified)
use semiflow_core::{ChernoffSemigroup, NonSeparable2DChernoff, Grid2D};

let semigroup = ChernoffSemigroup::new(heston_kernel, n_steps)?;   // deprecated alias

// Access growth parameters for normalization:
let (m, omega) = semigroup.chernoff.growth();                      // BREAKS in v3.x

// --- v3.0 migration diff ---

// Change 1: ChernoffSemigroup → Evolver (1-line rename)
use semiflow_core::{Evolver, NonSeparable2DChernoff, Grid2D, Growth};

let evolver = Evolver::new(heston_kernel, n_steps)?;               // no deprecation warning

// Change 2: Growth<F> struct destructure (1-line change per access site)
let Growth { multiplier: m, omega } = evolver.chernoff.growth();

// Everything else (apply_into, ScratchPool, Grid2D, GridFn2D) unchanged.
// The NonSeparable2DChernoff kernel itself requires NO changes — it already
// implements the v3 ChernoffFunction<F> trait via the Wave A+B mechanical sweep.
```

### §6.3 — Python numpy pipeline migration (PyO3 v2 allocating → v3 zero-alloc)

**Scenario**: a Python analytics pipeline that computes heat semigroup expectations over a batch of initial conditions, migrating from v2.x `Heat1D` (allocating) to v3.0 `EvolverHeat1DUnitV3` (zero-alloc, GIL-released).

```python
# v2.x pipeline (continues to compile with no warnings in v3.0)
from semiflow import Heat1D
import numpy as np

def batch_evolve_v2(initial_batch: list[np.ndarray], t: float) -> list[np.ndarray]:
    """Evolve a batch of initial conditions. Allocates per call."""
    heat = Heat1D(-5.0, 5.0, 512)
    results = []
    for ic in initial_batch:
        heat.setValues(ic)
        heat.evolve(t)          # allocating; holds GIL
        results.append(heat.values().copy())
    return results

# v3.0 pipeline — zero-alloc, GIL released during compute, threading-safe
from semiflow import EvolverHeat1DUnitV3, GrowthV3
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def batch_evolve_v3(initial_batch: list[np.ndarray], t: float) -> list[np.ndarray]:
    """Evolve a batch with GIL release in the hot loop."""
    evolver = EvolverHeat1DUnitV3(-5.0, 5.0, 512, 64)  # create once, reuse
    dst = np.empty(512, dtype=np.float64)              # reuse output buffer
    results = []
    for ic in initial_batch:
        src = np.ascontiguousarray(ic, dtype=np.float64)
        evolver.evolve_into(t, src, dst)               # GIL released; zero-alloc
        results.append(dst.copy())                     # copy out before reusing dst
    return results

# Multi-threaded version — safe because py.detach releases GIL
def evolve_one(ic: np.ndarray, t: float, evolver: EvolverHeat1DUnitV3) -> np.ndarray:
    dst = np.empty(512, dtype=np.float64)
    src = np.ascontiguousarray(ic, dtype=np.float64)
    evolver.evolve_into(t, src, dst)                   # thread-safe: GIL released
    return dst.copy()

with ThreadPoolExecutor(max_workers=4) as pool:
    evolvers = [EvolverHeat1DUnitV3(-5.0, 5.0, 512, 64) for _ in range(4)]
    futures = [pool.submit(evolve_one, ic, 0.01, evolvers[i % 4])
               for i, ic in enumerate(initial_batch)]
    results = [f.result() for f in futures]

# GrowthV3 inspection (useful for normalization checks)
evolver = EvolverHeat1DUnitV3(-5.0, 5.0, 512, 64)
g: GrowthV3 = evolver.growth()
assert g.multiplier == 1.0 and g.omega == 0.0, f"Unexpected growth: {g!r}"
```

---

## §7 — FFI parity verification (G_binding_parity gate)

The `G_binding_parity` release-blocking gate has 6 sub-tests that verify byte-identity (0 ULP per ADR-0059 precedent) across all three bindings and both surfaces (v2 shim + v3 new).

**Build all three bindings locally:**

```bash
# Build FFI (produces libsemiflow_ffi.so / .dylib / .dll)
cargo run -p xtask -- ffi-build

# Build PyO3 wheel (editable install, requires maturin)
cargo run -p xtask -- py-build
pip install -e crates/semiflow-py/

# Build WASM (requires wasm-pack)
cargo run -p xtask -- wasm-build
```

**Run the G_binding_parity sub-tests:**

```bash
# Sub-test 1 (FFI v3 vs v2 baseline): PASS
cargo run -p xtask -- ffi-smoke
cargo test -p semiflow-ffi --test ffi_v3_smoke -- --nocapture

# Sub-test 2 (FFI v2 surface unchanged): PASS (trivially — v2 surface not modified)
# Verified by: the ffi-smoke test also exercises the v2 path
cargo test -p semiflow-ffi --test ffi_smoke -- --nocapture

# Sub-test 3 (PyO3 v3 vs v2 baseline): PASS
cargo run -p xtask -- py-smoke
# Or directly:
pytest crates/semiflow-py/tests/test_v3_smoke.py -v -k "test_binding_parity"

# Sub-test 4 (PyO3 v2 apply method via v2_compat): PASS (deprecated warning; same output)
pytest crates/semiflow-py/tests/test_heat.py -v  # v2 smoke unchanged

# Sub-test 5 (WASM v3 vs v2): STUB pending wasm-pack on CI
# Expected when wasm-pack available:
cd crates/semiflow-wasm && wasm-pack test --node -- --test v3_smoke

# Sub-test 6 (WASM v2 surface unchanged): STUB (same status as sub-test 5)
```

**Interpreting results:** all 6 sub-tests compare `f64` output buffers at 0 ULP (exact bit identity). Sub-tests 1 and 3 are fully validated in v3.0. Sub-tests 5 and 6 typechecks pass; runtime identity is expected when wasm-pack runs on Node (same Rust core, same `DiffusionChernoff<f64>` path, pure pass-through).

**Troubleshooting parity failures:**
- **x86_64 vs aarch64 discrepancy**: `f64` arithmetic is IEEE 754 compliant on both; if you see non-zero ULP differences, check that `RUSTFLAGS="-C target-cpu=native"` is not enabled for the smoke tests (native SIMD is not the issue; rounding mode is).
- **NaN bit-pattern variation**: the v3 surfaces never produce NaN in the nominal path; NaN inputs return `Err` at the boundary. If a NaN appears in output, check the initial condition for NaN before calling evolve.
- **Wrong grid size**: `smf_evolver_size_v3` returns the grid point count (`n_grid` passed to ctor), not the domain width.

---

## §8 — PyO3 namedtuple migration

`Heat1D.growth()` returns a Python namedtuple (not a plain tuple) in both v2.x and v3.x. This means all three access patterns continue to work without changes:

```python
heat = Heat1D(-5.0, 5.0, 512)

# All three patterns unchanged from v2.x to v3.0:
m, omega = heat.growth()          # positional tuple destructure: works
m = heat.growth()[0]              # positional index: works
m = heat.growth().multiplier      # named-field access: works

# The namedtuple is declared as GrowthTuple(multiplier, omega) in the v2 pyclass.
# v3.0 adds GrowthV3 as a richer pyclass (with __repr__); the v2 Heat1D.growth()
# namedtuple return type is unchanged.
print(type(heat.growth()))        # <class 'semiflow.GrowthTuple'>
print(type(EvolverHeat1DUnitV3(-5.0,5.0,512,64).growth()))  # <class 'semiflow.GrowthV3'>
```

**Type-stub update for downstream type-checked consumers** (if you import `Heat1D` and annotate return types):

```python
# Before v3.0 — no changes needed; Heat1D.growth() type is already correct
# After v3.0 — if you want to annotate v3 evolver usage:
from semiflow import EvolverHeat1DUnitV3, GrowthV3

evolver: EvolverHeat1DUnitV3 = EvolverHeat1DUnitV3(-5.0, 5.0, 512, 64)
g: GrowthV3 = evolver.growth()  # pyright resolves via Wave E .pyi stubs
```

The v3.0 `.pyi` stubs are located at `crates/semiflow-py/python/semiflow/__init__.pyi` (see Wave E addition in the v3.0 CHANGELOG). Run `pyright` against your consumer code after upgrading to verify stub resolution.

---

## §9 — WASM object-return migration

`Heat1D.growth()` returns a JavaScript object `{multiplier: number, omega: number}` in v3.0, replacing the v2.x Array return. This is the only RUNTIME-breaking change in the WASM surface (see §4).

```javascript
// v2.x WASM Array return — BROKEN in v3.0
const [M, omega] = heat.growth();        // undefined[0], undefined[1] in v3.0

// Migration options:

// Option 1: named destructure (recommended)
const { multiplier: M, omega } = heat.growth();

// Option 2: v2 shim (deprecation-warned via console.warn; hard-removed at v4.0)
const [M_shim, omega_shim] = heat.growthV2();
// Console outputs: "remizov: growthV2() is deprecated as of v3.0; migrate to growth()"

// Option 3: positional index (DOES NOT WORK — growth() returns an object, not Array)
// heat.growth()[0] === undefined  -- do NOT use
```

**TypeScript declaration update**: the v3.0 wasm-bindgen build regenerates `.d.ts` automatically. If you use `@semiflow/wasm` with TypeScript, the updated `growth()` signature appears in the bundled `.d.ts`:

```typescript
// v3.0 generated @semiflow/wasm/index.d.ts excerpt
export class GrowthV3 {
  readonly multiplier: number;
  readonly omega: number;
  free(): void;
}

export class EvolverHeat1DUnitV3 {
  constructor(domainLo: number, domainHi: number, nGrid: number, nChernoff: number);
  setValues(values: Float64Array): void;
  evolveInto(t: number): void;
  values(): Float64Array;
  growth(): GrowthV3;
  size(): number;
  nChernoff(): number;
  free(): void;
}
```

---

## §10 — Cross-binding parity verification

The `G_binding_parity` gate verifies that all three bindings (FFI, PyO3, WASM) and both surfaces (v2 shim + v3 new) produce bit-identical f64 output when given the same CEV smoke inputs (ADR-0059 v2.2 precedent: 0 ULP identity).

**Run all three bindings against the canonical CEV smoke suite:**

```bash
# Build all bindings first (see §7)
cargo run -p xtask -- ffi-build && cargo run -p xtask -- py-build

# Run pairwise byte-identity checks:
# FFI v3 vs v2 baseline (sub-test 1): PASS
cargo test -p semiflow-ffi --test ffi_v3_smoke -v

# PyO3 v3 vs v2 baseline (sub-test 3): PASS
pytest crates/semiflow-py/tests/test_v3_smoke.py -v -k "parity"

# WASM v3 vs v2 baseline (sub-tests 5+6): STUB
# Expected command when wasm-pack installed:
# cd crates/semiflow-wasm && wasm-pack test --node
```

**Status summary (v3.0.0 ship state):**

| Sub-test | Description | Status |
|---|---|---|
| 1 | FFI v3 (`smf_evolver_*_v3`) ⇔ v2 baseline | PASS |
| 2 | FFI v2 surface unchanged (trivially identical) | PASS |
| 3 | PyO3 v3 (`EvolverHeat1DUnitV3`) ⇔ v2 baseline | PASS |
| 4 | PyO3 v2 `apply` method via v2_compat shim | PASS |
| 5 | WASM v3 (`EvolverHeat1DUnitV3` JS) ⇔ v2 baseline | STUB (wasm-pack pending CI) |
| 6 | WASM v2 surface unchanged | STUB (same as 5) |

**CI integration**: the `cross-binding-parity` CI job runs sub-tests 1–4 on every push. Sub-tests 5–6 run in the `wasm-test-node` CI job once wasm-pack is installed on the runner (expected: same Node target, same Rust core, same `DiffusionChernoff<f64>` path → 0 ULP identity).

**v4.0 hard removal checklist**

At v4.0 (targeted ~2027-05-27 per ADR-0035 §9), the v2 surface is deleted. Complete ALL items before tagging v4.0:

- [ ] Remove `crates/semiflow-core/src/v2_compat.rs` (~122 LoC)
- [ ] Remove `v2_compat = []` from `crates/semiflow-core/Cargo.toml [features]`
- [ ] Remove `default = ["v2_compat"]` from `crates/semiflow-core/Cargo.toml [features]`
- [ ] Remove all `#[cfg(feature = "v2_compat")]` re-exports from `crates/semiflow-core/src/lib.rs`
- [ ] Remove all `#[deprecated]` markers that reference the v2_compat shim (trait blanket impls in `v2_compat.rs` and the `ChernoffSemigroup` type alias)
- [ ] FFI: remove the v2 surface from `crates/semiflow-ffi/src/ffi.rs` (functions WITHOUT `_v3` suffix); delete `crates/semiflow-ffi/src/v2_shim.rs`; regenerate `remizov.h` with v3-only surface via `cargo run -p xtask -- ffi-headers`
- [ ] PyO3: remove `Heat1D` v2 pyclass (or alias it as an error); ensure `EvolverHeat1DUnitV3` is the primary user-facing class; update `__init__.pyi` to remove v2 entries
- [ ] WASM: remove v2 `Heat1D` JS class; `EvolverHeat1DUnitV3` becomes the default
- [ ] Update CHANGELOG [4.0.0] entry: "v2 surface hard-removed per 12-month ADR-0035 §9 deprecation cycle complete (cycle started 2026-05-27 at v3.0.0 ship)"
- [ ] Constitution v2.0.0 MAJOR re-evaluation (next mandatory point after v4.0 per constitution §"Override rules")
- [ ] Verify: `cargo test --workspace` passes with 0 failures after all removals
- [ ] Verify: `cargo run -p xtask -- check-lints` passes with 0 new violations

---

## Appendix A — ADR cross-references

- [ADR-0073](../adr/0073-approximation-subspace-trait.md) — `ApproximationSubspace<K, F>` opt-in marker trait
- [ADR-0074](../adr/0074-chernoff-function-trait-cleanup.md) — `ChernoffFunction<F>` trait cleanup (BREAKING)
- [ADR-0075](../adr/0075-zeta4-correction.md) — ζ⁴ correction (A5)
- [ADR-0076](../adr/0076-v2-to-v3-binding-redesign.md) — v2→v3 binding redesign
- [ADR-0035](../adr/0035-v1-api-stability-and-deprecation.md) — v1.0.0 API freeze + deprecation cadence (12-month per §9)
- [ADR-0028](../adr/0028-v0_10_0-bindings-ffi-py-wasm.md) — v0.10.0 binding strategy
- [ADR-0026](../adr/0026-chernoffunction-generic-over-float.md) — v0.9.0 ChernoffFunction generic-over-F
- [ADR-0041](../adr/0041-state-trait-and-apply-into.md) — v2.0 `apply_into` + `ScratchPool` zero-alloc pattern

## Appendix B — Migration assistance

- Open a GitHub Discussions thread tagged `v3-migration` for community help.
- The v3.0.0 release notes (`CHANGELOG.md` v3.0.0 entry) link back to this migration guide.
- The Rust compiler emits explicit deprecation messages pointing at this migration guide for each v2.x symbol.
