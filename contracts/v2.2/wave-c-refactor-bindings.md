# Wave 2.2C Contract — NonSeparable Refactor + Graph Bindings

**Status**: NORMATIVE — engineer implements verbatim against this contract.
**ADRs**: 0058 (`NonSeparableMixedChernoff` unification — SUPERSEDES ADR-0033),
0059 (graph bindings FFI/PyO3/WASM), 0060 (parallel bench suite, OPTIONAL).
**Depends on**: v2.2 Waves A + B — all v2.2 types must be shipped
before Wave C begins (graph bindings expose Wave A types; refactor is
independent but tested last).
**Math**: contracts/semiflow-core.math.md §18 (refactor pointer — no math change).
**Sympy gates**: none new (refactor preserves numerics).
**Slope gates**: G20 (alias-identity, new); G_cross_binding_graph_identity
(extension of v0.10.0 G).
**Author**: ai-solutions-architect · **Date**: 2026-05-21.

This wave is a TWO-PART hybrid:

- **Part 1 — Refactor**: `NonSeparable2DChernoff` and `NonSeparable2DAnisotropicChernoff`
  become type aliases of a new generic `NonSeparableMixedChernoff<X, Y, F, S>`
  (ADR-0058).
- **Part 2 — Bindings**: extend `semiflow-ffi`, `semiflow-py`, `semiflow-wasm` with
  Graph PDE modules (ADR-0059). Mirrors v0.10.0 three-way split.
- **Part 3 (OPTIONAL) — Bench**: parallel bench suite (ADR-0060). Engineer
  may defer to v2.2.1 if LoC budget tight.

---

## §1 — Part 1: NonSeparable refactor (NORMATIVE — ADR-0058)

### 1.1 File operations

```text
crates/semiflow-core/src/nonseparable_mixed.rs    (NEW FILE, ~550 LoC; Override #1 carve-out)
crates/semiflow-core/src/nonseparable2d.rs        (THINNED to ~30 LoC re-export shim)
crates/semiflow-core/src/nonseparable2d_aniso.rs  (THINNED to ~30 LoC re-export shim)
```

Net code reduction: 995 LoC (514+481) of duplicated logic → 550 LoC unified.

### 1.2 Public API (verbatim)

```rust
//! crates/semiflow-core/src/nonseparable_mixed.rs

use crate::{Grid2D, GridFn1D, GridFn2D, SemiflowError, ScratchPool};
use crate::axis::{Axis, AxisLift};
use crate::chernoff::ChernoffFunction;
use crate::float::SemiflowFloat;

/// Coupling for mixed second derivative `∂x∂y`. Two impls in v2.2:
/// `ScalarCoupling<F>` (v0.7.0 scalar c) and `BetaCoupling<F>` (v0.9.0 β(x,y)).
/// PRIVATE to module — exposed only via type aliases below.
trait MixedDerivOperator<F: SemiflowFloat, S>: Send + Sync {
    fn norm_bound(&self) -> F;
    fn apply_mixed_into(&self, src: &S, dst: &mut S, grid: &Grid2D<F>);
    fn is_zero(&self) -> bool;
}

/// Unified non-separable mixed Chernoff. See math.md §10.7-ter + §18.
///
/// Type aliases:
/// - `NonSeparable2DChernoff<X, Y, F>` (scalar coupling `c`, v0.7.0 surface)
/// - `NonSeparable2DAnisotropicChernoff<X, Y, F>` (position-dep coupling `β(x,y)`, v0.9.0 surface)
#[derive(Clone, Debug)]
pub struct NonSeparableMixedChernoff<X, Y, F: SemiflowFloat = f64, S = GridFn2D<F>>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    pub x: AxisLift<X, F>,
    pub y: AxisLift<Y, F>,
    coupling: alloc::boxed::Box<dyn MixedDerivOperator<F, S>>,
    pub grid: Grid2D<F>,
}

// v2.2: collapses to S = GridFn2D<F> only (graph case = v2.3+).
impl<X, Y, F: SemiflowFloat> NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// Constant scalar coupling. Mirrors v0.7.0 `NonSeparable2DChernoff::new`.
    pub fn with_scalar_c(
        x_inner: X,
        y_inner: Y,
        c: fn(F, F) -> F,
        c_norm_bound: f64,
        grid: Grid2D<F>,
    ) -> Result<Self, SemiflowError>;

    /// Position-dependent coupling. Mirrors v0.9.0 anisotropic constructor.
    pub fn with_beta(
        x_inner: X,
        y_inner: Y,
        beta: fn(F, F) -> F,
        beta_norm_bound: f64,
        grid: Grid2D<F>,
    ) -> Result<Self, SemiflowError>;
}

// Backward-compat type aliases — zero source migration burden.
pub type NonSeparable2DChernoff<X, Y, F = f64> =
    NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>;

pub type NonSeparable2DAnisotropicChernoff<X, Y, F = f64> =
    NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>;

// ChernoffFunction impl (sequential + parallel paths — copy from v0.7.0/v0.9.0).
impl<X, Y, F: SemiflowFloat> ChernoffFunction<F> for NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    type S = GridFn2D<F>;
    fn apply(/* … */) -> Result<Self::S, SemiflowError>;
    fn apply_into(/* … */) -> Result<(), SemiflowError>;
    fn order(&self) -> u32 { 2 }
    fn growth(&self) -> (f64, f64);
}
```

### 1.3 Constructor migration (NORMATIVE)

The v2.1 constructors `NonSeparable2DChernoff::new(x, y, c, c_bound, grid)`
and `NonSeparable2DAnisotropicChernoff::new(x, y, β, β_bound, grid)` MUST
continue to compile. Achieve this by making `new` a thin re-export of
`with_scalar_c` / `with_beta` respectively:

```rust
// In src/nonseparable2d.rs (thinned shim):
pub use crate::nonseparable_mixed::NonSeparableMixedChernoff;

pub type NonSeparable2DChernoff<X, Y, F = f64> =
    NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>;

impl<X, Y, F: SemiflowFloat> NonSeparable2DChernoff<X, Y, F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// v2.1-compatible constructor; equivalent to `with_scalar_c`.
    pub fn new(
        x_inner: X,
        y_inner: Y,
        c: fn(F, F) -> F,
        c_norm_bound: f64,
        grid: Grid2D<F>,
    ) -> Result<Self, SemiflowError> {
        Self::with_scalar_c(x_inner, y_inner, c, c_norm_bound, grid)
    }
}
```

(Symmetric in `src/nonseparable2d_aniso.rs` with `with_beta` delegation.)

### 1.4 R4 zero-alloc invariant

Steady-state `apply_into` MUST allocate zero bytes in the hot loop. The
v-table hop on `Box<dyn MixedDerivOperator>` is one indirect call per step
(~1-2 ns overhead, negligible). All scratch buffers from `ScratchPool`.

### 1.5 Generic-over-F coverage

`F: SemiflowFloat`. Same as v0.7.0 / v0.9.0 precedents.

---

## §2 — Part 2: Graph bindings (NORMATIVE — ADR-0059)

### 2.1 File locations

```text
crates/semiflow-ffi/src/graph_ffi.rs               (NEW, ~400 LoC; under cap)
crates/semiflow-ffi/include/remizov_graph.h        (NEW, ~150 LoC; C header)
crates/semiflow-py/src/graph_py.rs                 (NEW, ~350 LoC)
crates/semiflow-py/tests/smoke_graph.py            (NEW, ~80 LoC)
crates/semiflow-wasm/src/graph_wasm.rs             (NEW, ~350 LoC)
crates/semiflow-wasm/tests/smoke_graph.ts          (NEW, ~100 LoC)
```

### 2.2 FFI surface (mirrors v0.10.0 §"Opaque handles")

```c
// crates/semiflow-ffi/include/remizov_graph.h (NEW)

typedef struct smf_graph_t      smf_graph_t;       // opaque handle
typedef struct smf_laplacian_t  smf_laplacian_t;   // opaque handle
typedef struct smf_graphsig_t   smf_graphsig_t;    // opaque handle
typedef struct smf_ghc_t        smf_ghc_t;         // GraphHeatChernoff
typedef struct smf_mghc_t       smf_mghc_t;        // MagnusGraphHeatChernoff
typedef struct smf_traj_t       smf_traj_t;        // GraphTraj

// Graph constructors (path/cycle/from_edges)
int smf_graph_path(uint32_t n_nodes, smf_graph_t** out);
int smf_graph_n_nodes(const smf_graph_t* g, uint32_t* out);
void smf_graph_drop(smf_graph_t* g);

// LaplacianAtTime callback
typedef int (*smf_lap_at_t_fn)(double t, void* user_data, smf_laplacian_t** out);

// MagnusGraphHeatChernoff
int smf_mghc_new(
    const smf_graph_t* topology,
    smf_lap_at_t_fn lap_fn,
    void* lap_user_data,
    double rho_bar_max,
    int convergence_radius_check,
    smf_mghc_t** out
);
int smf_mghc_apply_into(
    smf_mghc_t* mghc, double tau, const smf_graphsig_t* src, smf_graphsig_t* dst
);
void smf_mghc_drop(smf_mghc_t* mghc);
```

Every `extern "C"` function MUST:

1. Use `catch_unwind` (panics return error code, never unwind across ABI).
2. Validate non-null inputs; return `SMF_ERR_NULL` on null pointer.
3. Use the `[profile.release-ffi]` panic=unwind (per ADR-0028).
4. Carry 30–50 lines of NORMATIVE rustdoc (panic-boundary + user_data
   lifetime + null-handling), matching `crates/semiflow-ffi/src/ffi.rs`
   precedent.

### 2.3 PyO3 surface (mirrors v0.10.0 §"Python wheel")

```python
# crates/semiflow-py/src/graph_py.rs (Rust source — Python visible class)

#[pyclass]
struct GraphPath { /* ... */ }

#[pymethods]
impl GraphPath {
    #[new]
    fn new(n_nodes: u32) -> PyResult<Self>;

    fn n_nodes(&self) -> u32;
}

#[pyclass]
struct GraphHeat { /* ... */ }

#[pymethods]
impl GraphHeat {
    #[new]
    fn new(graph: &GraphPath, rho_bar: f64) -> PyResult<Self>;

    fn evolve(&self, py: Python, t_final: f64, n_steps: u32, f0: &PyArray1<f64>) -> PyResult<Py<PyArray1<f64>>>;
}

#[pyclass]
struct MagnusGraphHeat { /* ... */ }

#[pymethods]
impl MagnusGraphHeat {
    #[new]
    fn new(graph: &GraphPath, lap_at_t: &PyAny, rho_bar: f64) -> PyResult<Self>;
    // lap_at_t is a Python callable taking f64 -> Laplacian
    fn evolve(&mut self, py: Python, t_final: f64, n_steps: u32, f0: &PyArray1<f64>) -> PyResult<Py<PyArray1<f64>>>;
}
```

**GIL release**: per ADR-0031 precedent, the `evolve` method releases the
GIL during the Rust hot loop and re-acquires only for the per-step
`lap_at_t` Python callback. Document expected callback overhead in
rustdoc.

### 2.4 WASM surface (mirrors v0.10.0 §"WASM API")

```rust
// crates/semiflow-wasm/src/graph_wasm.rs

#[wasm_bindgen]
pub struct GraphPath { /* ... */ }

#[wasm_bindgen]
impl GraphPath {
    #[wasm_bindgen(constructor)]
    pub fn new(n_nodes: u32) -> Result<GraphPath, JsValue>;
    pub fn n_nodes(&self) -> u32;
}

#[wasm_bindgen]
pub struct GraphHeat { /* ... */ }

#[wasm_bindgen]
impl GraphHeat {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &GraphPath, rho_bar: f64) -> Result<GraphHeat, JsValue>;
    pub fn evolve(&self, t_final: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue>;
}
```

WASM uses `[profile.release]` with `panic=abort` (ADR-0029). Errors are
`JsValue::from_str(format!(...))` — NOT `catch_unwind` (panics abort).

### 2.5 Cross-binding identity gate (NORMATIVE)

Extend v0.10.0's `G_cross_binding_identity` gate to graph kernels:

```text
Test setup:
  - P_64 path graph, combinatorial Laplacian
  - GraphSignal: u₀(i) = exp(−i² / 64)
  - t_final = 0.5, n_steps = 50

Compute u(t_final) via:
  - Rust reference (direct call to MagnusGraphHeatChernoff)
  - FFI binding (C smoke test)
  - PyO3 binding (Python smoke)
  - WASM binding (Node smoke)

Cross-binding invariant:
  |sup_err_Rust − sup_err_FFI| < 3 ULP
  |sup_err_Rust − sup_err_PyO3| < 3 ULP
  |sup_err_Rust − sup_err_WASM| < 3 ULP
```

**3 ULP threshold** (relaxed from v0.10.0's 1 ULP per ADR-0059 §"Cross-binding
sup-error gate" rationale). Sparse mat-vec summation order reorder is the
source.

### 2.6 R4 zero-alloc invariant (FFI hot path)

FFI `smf_mghc_apply_into` MUST allocate zero bytes in the hot loop after
the first call (subsequent calls reuse internal `ScratchPool`).

### 2.7 Generic-over-F coverage

**f64 ONLY for graph bindings** (per ADR-0059 §"Out of scope"). f32 graph
bindings deferred to v2.3+.

---

## §3 — Part 3 (OPTIONAL): Parallel bench suite (ADR-0060)

### 3.1 File locations

```text
benchmarks/v2_2_0/parallel_bench.rs              (NEW, ~280 LoC)
benchmarks/v2_2_0/baseline-v2_2_0.json           (NEW, output snapshot)
benchmarks/v2_2_0/parallel_bench_report.md       (NEW, ~250 LoC markdown)
xtask new subcommand: `bench-parallel`           (~50 LoC)
```

### 3.2 Schema for `baseline-v2_2_0.json`

```json
{
  "version": "2.2.0",
  "host": "bestfriend",
  "cpu": "i7-12700K, 12C20T",
  "rust_version": "1.78.0",
  "date": "2026-05-XX",
  "kernels": [
    {
      "name": "Strang2D::apply",
      "feature_flags": ["parallel"],
      "n": 1024,
      "thread_counts": [1, 2, 4, 8, 12, 16],
      "time_per_step_ns": [...],
      "allocs_per_step": [0, 0, 0, 0, 0, 0]
    }
  ]
}
```

### 3.3 Acceptance gates (NORMATIVE — only if Part 3 ships)

- `cargo bench -p remizov-bench parallel_bench` exits 0.
- `baseline-v2_2_0.json` schema-validates.

---

## §4 — Sympy gates (no new — refactor preserves numerics)

The unification (Part 1) does NOT change any math; the v0.7.0 `T_NS2D_*`
and v0.9.0 `T_NS2D_aniso_*` gates re-pass byte-identical via the type-aliased
call paths. **G20 alias-identity gate** (NORMATIVE — new) verifies this:

```text
For arbitrary f, τ ∈ {0.001, 0.01, 0.1}:
  <NonSeparable2DChernoff::new(x, y, c=|x,y| 0.3, c_bound=0.3, grid) as ChernoffFunction>::apply_into(τ, f)
== <NonSeparableMixedChernoff::with_scalar_c(x, y, |x,y| 0.3, 0.3, grid) as ChernoffFunction>::apply_into(τ, f)

Threshold: 0 ULP (byte-equal).
```

---

## §5 — Slope gates (NORMATIVE)

### 5.1 G20 alias-identity (`tests/g20_alias_identity.rs`)

See §4 above. ULP threshold 0 (byte equality).

### 5.2 Existing G_NS2D + G_NS2D_aniso (re-run after refactor)

Both v0.7.0 and v0.9.0 slope gates MUST re-pass byte-identical (slope
within ±0.001 of previous releases). If they regress, refactor is
backed out.

### 5.3 G_cross_binding_graph (new — ADR-0059)

See §2.5. C smoke + Python smoke + Node smoke. CI matrix:
`ubuntu-latest`, `macos-latest`, `windows-latest`.

---

## §6 — Capability / security (NORMATIVE)

**STRIDE on graph bindings** (ADR-0059):

- **S** (spoofing): caller passes opaque handle (`*const smf_graph_t`).
  `catch_unwind` + non-null + n_nodes > 0 check.
- **T** (tampering): handles immutable post-construction.
- **R** (repudiation): N/A (sync function-call API; no transaction log).
- **I** (information disclosure): `row_ptr`/`col_idx` exposed via FFI
  return COPIES into caller-owned buffers (NOT raw pointers).
- **D** (DoS): `GraphTraj` capped at 65_535 segments (ADR-0052).
- **E** (elevation of privilege): N/A in `rlib`/`cdylib` scope
  (constitution §5 VACUOUSLY SATISFIED).

**No new capability tokens; STRIDE applies the same as v0.10.0 bindings.**

---

## §7 — Build/run path (NORMATIVE)

### Core crate

```bash
cargo run -p xtask -- test-fast       # default (Wave A + B + C Part 1)
cargo run -p xtask -- test-full       # full
cargo run -p xtask -- test-flagship   # slope gates
```

### Bindings

```bash
# FFI
cargo run -p xtask -- ffi-build         # build cdylib
cargo run -p xtask -- ffi-headers       # regenerate headers
cargo run -p xtask -- ffi-smoke         # C smoke test

# PyO3
cargo run -p xtask -- py-build          # maturin build
cargo run -p xtask -- py-smoke          # pytest invocation

# WASM
cargo run -p xtask -- wasm-build        # wasm-pack
cargo run -p xtask -- wasm-smoke        # wasm-bindgen-test
```

### Parallel bench (optional, Part 3)

```bash
cargo run -p xtask -- bench-parallel    # produces baseline-v2_2_0.json
```

---

## §8 — Engineer pickup ordering (NORMATIVE)

Step 1 (Part 1 — Refactor): Read ADR-0058 + math.md §18.

Step 2: Implement `nonseparable_mixed.rs` (consolidate v0.7.0 + v0.9.0
logic). Thin `nonseparable2d.rs` and `nonseparable2d_aniso.rs` to
shims. Verify ALL existing v0.7/v0.9 tests re-pass byte-identical.

Step 3: Add `tests/g20_alias_identity.rs`. Confirm 0 ULP.

Step 4 (Part 2 — Bindings): Read ADR-0059. Read v0.10.0 contracts
(`contracts/v0.10/wave-{a,b,c}-*.md`) for binding patterns.

Step 5: Implement `crates/semiflow-ffi/src/graph_ffi.rs` + header generation.
Add `tests/ffi_smoke_graph.c`. Confirm cross-binding identity.

Step 6: Implement `crates/semiflow-py/src/graph_py.rs` + Python smoke.
GIL release pattern (ADR-0031) applies.

Step 7: Implement `crates/semiflow-wasm/src/graph_wasm.rs` + Node smoke.

Step 8: Update CI (`release-ffi.yml`, `release-wheels.yml`, `release-wasm.yml`)
to include graph kernels in cross-binding gate.

Step 9 (OPTIONAL — Part 3): Read ADR-0060. Implement parallel bench
if time permits; otherwise defer to v2.2.1.

Step 10: Update CHANGELOG.md with Wave 2.2C entry.

Step 11: Handoff to git-workflow for Wave 2.2C commit. Trailers:
`Agent: agentic-engineer`, `Task-ID: v2.2-wave-c-refactor-bindings`.

Step 12: Update `docs/migration/v2.1-to-v2.2.md` (NEW) with the type-alias
note (v2.1 callers unaffected; new code MAY use generic form).
