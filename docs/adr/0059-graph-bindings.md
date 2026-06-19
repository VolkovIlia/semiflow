# ADR-0059 — Graph bindings: FFI / PyO3 / WASM

- **Status**: ACCEPTED (v2.2 Wave C)
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave C
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0028 (`semiflow-ffi/-py/-wasm` v0.10.0 bindings),
  ADR-0029 (Amendment 1, WASM panic=abort), ADR-0030 (Amendment 2, npm
  publish), ADR-0031 (PyO3 GIL release), ADR-0047 (`GraphHeatChernoff`),
  ADR-0051 (`MagnusGraphHeatChernoff`), ADR-0052 (`GraphTraj<F>`).
- **Supersedes / amends**: nothing; answers v2.0 "Out of scope" item
  "Graph bindings (FFI/PyO3/WASM)" (ROADMAP.md line ~440).
- **Mathematical foundation**: math.md §14 unchanged (bindings are
  surface-level; no math content).

## Context

v0.10.0 shipped C ABI (`semiflow-ffi`), PyO3 (`semiflow-py`), and WASM
(`semiflow-wasm`) bindings for the **1D heat unit-a kernel**
(`DiffusionChernoff` with `a ≡ 1`). Graph PDE kernels (v2.1) — and the
v2.2 Wave A / B kernels (`GraphTraj<F>`, `VarCoefGraphHeatChernoff`,
`MagnusGraphHeat6thChernoff`, `SchrodingerChernoff`) — currently have
no bindings.

Customer demand has accumulated since v0.10.0:
- Python data scientists need `pip install semiflow-py` for graph PDE
  on heterogeneous networks (drug-diffusion modelling, power-grid
  thermal transient).
- Web-based interactive demos need WASM for in-browser graph PDE
  visualisations.
- C/C++ scientific computing teams need FFI for embedding in larger HPC
  codes.

## Decision

Mirror v0.10.0's three-way binding split by adding **new modules**
(not new crates) to the existing binding crates:

```text
crates/semiflow-ffi/src/graph_ffi.rs       (NEW, ~400 LoC)
crates/semiflow-py/src/graph_py.rs         (NEW, ~350 LoC)
crates/semiflow-wasm/src/graph_wasm.rs     (NEW, ~350 LoC)
```

Each new module exposes the v2.1+ graph types:

- `Graph<f64>` → opaque handle (FFI) / `pyclass` (PyO3) / `#[wasm_bindgen]` class.
- `Laplacian<f64>` → opaque handle / `pyclass` / `#[wasm_bindgen]`.
- `GraphSignal<f64>` → owned buffer wrapper.
- `GraphHeatChernoff<f64>` → opaque handle / `pyclass` / `#[wasm_bindgen]`.
- `MagnusGraphHeatChernoff<f64>` → same (with `LaplacianAtTime` closure
  expressed as a caller-supplied callback function pointer in FFI / a
  Python callable in PyO3 / a JS function in WASM).
- `GraphTraj<f64>` (Wave A) → opaque handle.

**No `f32` support in bindings.** Graph PDE is the highest-precision
demand pattern; `f32` adds 2× source surface for marginal benefit.
Documented in rustdoc; v2.3+ may revisit.

**Panic policies** (unchanged from ADR-0028 + ADR-0029):

| Crate | Profile | Panic policy | Error path |
|---|---|---|---|
| `semiflow-ffi` | `[profile.release-ffi]` panic=unwind | `catch_unwind` at every `extern "C"` entry | `int_t` return code with `last_error()` lookup |
| `semiflow-py` | `[profile.release-ffi]` panic=unwind | `panic::catch_unwind` | `pyo3::exceptions::PyRuntimeError` |
| `semiflow-wasm` | `[profile.release]` panic=abort | abort (no catch) | JS `throw` via `wasm_bindgen` |

### Opaque handle ABI (FFI; mirrors v0.10.0 §"Opaque handles")

```c
// New types in crates/semiflow-ffi/include/remizov_graph.h
typedef struct smf_graph_t      smf_graph_t;
typedef struct smf_laplacian_t  smf_laplacian_t;
typedef struct smf_graphsig_t   smf_graphsig_t;
typedef struct smf_ghc_t        smf_ghc_t;   // GraphHeatChernoff
typedef struct smf_mghc_t       smf_mghc_t;  // MagnusGraphHeatChernoff
typedef struct smf_traj_t       smf_traj_t;  // GraphTraj

// Constructor + accessor + destructor pattern (one per type)
int smf_graph_path(uint32_t n_nodes, smf_graph_t** out);
int smf_graph_n_nodes(const smf_graph_t* g, uint32_t* out);
void smf_graph_drop(smf_graph_t* g);

// LaplacianAtTime closure: caller passes a function pointer + user_data
typedef int (*smf_lap_at_t_fn)(double t, void* user_data, smf_laplacian_t** out);

int smf_mghc_new(
    const smf_graph_t* topology,
    smf_lap_at_t_fn lap_fn,
    void* lap_user_data,
    double rho_bar_max,
    int convergence_radius_check,
    smf_mghc_t** out
);

int smf_mghc_apply_into(
    smf_mghc_t* mghc,
    double tau,
    const smf_graphsig_t* src,
    smf_graphsig_t* dst
);
```

The `smf_lap_at_t_fn` callback approach mirrors GLib/libuv idiom and
keeps the FFI ABI C-compatible. `user_data` is opaque (typically a
struct pointer with the time-varying weight closure).

### PyO3 surface (PyO3; mirrors v0.10.0 §"Python wheel")

```python
# Wheel: semiflow-py-2.2.0-cp310-abi3-{linux,macos,windows}_amd64.whl
import semiflow

g = semiflow.GraphPath(64)   # P_64
gh = semiflow.GraphHeat(g, rho_bar=4.0)
signal = semiflow.GraphSignalFromArray(np.array([1.0, 0.0, 0.0, ...]))

# Magnus K=4 with Python callable as LaplacianAtTime
def lap_at_t(t):
    g = semiflow.GraphPath(64)
    g.set_weights(np.full(63, 1 + 0.3 * np.sin(np.pi * t)))
    return g.laplacian_combinatorial()

mghc = semiflow.MagnusGraphHeat(g, lap_at_t, rho_bar=4.0)
out = mghc.evolve(t_horizon=0.5, n_steps=50, f0=signal)
```

### WASM surface (WASM; mirrors v0.10.0 §"WASM API")

```typescript
// npm: @semiflow/wasm@2.2.0
import init, { GraphPath, GraphHeat, MagnusGraphHeat } from '@semiflow/wasm';

await init();
const g = new GraphPath(64);
const gh = new GraphHeat(g, 4.0 /* rho_bar */);
const signal = new Float64Array(64);
signal[0] = 1.0;
const out = gh.evolve(0.5 /* t_final */, 50 /* n_steps */, signal);
```

## Cross-binding sup-error gate

Extend v0.10.0's `G_cross_binding_identity` gate (originally for 1D
heat) to include graph kernels:

```text
For inputs:
  - P_64 path graph, combinatorial Laplacian
  - GraphSignal: u₀(i) = exp(−i²/64)
  - t_final = 0.5, n_steps = 50

Compute u(t_final) via three bindings:
  - FFI (Rust via C interface), sup-err vs reference Rust
  - PyO3 (Rust via Python), sup-err
  - WASM (Rust via wasm-bindgen), sup-err

Cross-binding identity: |sup_err_FFI − sup_err_PyO3| < 3 ULP
                         |sup_err_FFI − sup_err_WASM| < 3 ULP
```

**3 ULP threshold (relaxed from v0.10.0's 1 ULP)**: sparse-mat-vec
operations on the graph kernel introduce reorder-of-summation
sensitivity (different summation orders in the inner loop) which is
~10×-worse on f64 than the dense 1D kernel. 3 ULP is empirically
within reach on modern (post-2020) x86_64 and aarch64 CPUs.

## Rationale

- **Reuses v0.10.0 infrastructure.** No new crate boundary; just new
  modules in existing crates. CI workflows (`release-ffi.yml`,
  `release-wheels.yml`, `release-wasm.yml`) extend with new test
  targets but no new pipelines.
- **Cross-binding identity gate** ensures the three bindings remain
  numerically interchangeable — a customer can switch from FFI to PyO3
  to WASM and get the same answer (within precision band).
- **Static_assertions for thread safety** (ADR-0031 precedent):
  `smf_mghc_t` is `Send + Sync` — verified at compile time via
  `static_assertions::assert_impl_all!`.

## Consequences

- 3 new files (~400 + 350 + 350 = 1100 LoC); all under the 500-LoC cap
  with one exception (FFI module at 400; safe). The constitution
  Override #1 file-list does NOT expand here (none exceeds the cap).
- 3 new wheel/package distribution targets:
  - `crates.io`: `semiflow-ffi` v2.2.0
  - `pypi.org`: `semiflow-py` v2.2.0 (abi3-py310)
  - `npmjs.com`: `@semiflow/wasm` v2.2.0
- Existing v0.10.0 1D-heat surfaces continue to work; bindings are
  additive.

## Acceptance gates

- **G_cross_binding_graph_identity gate** (NORMATIVE — see above).
  Re-run on every push; must remain green for tag.
- **G_FFI_smoke_graph gate** (NORMATIVE). C smoke test: `smf_graph_path`,
  `smf_ghc_new`, `smf_ghc_apply_into`, verify result against reference
  Rust. `sup_error < 5e-4` (matches v0.10.0 G_FFI_smoke).
- **G_PyO3_smoke_graph gate** (NORMATIVE). Python smoke test in
  `crates/semiflow-py/tests/smoke_graph.py`: same logic, `pytest`
  invocation, `assert np.allclose(out_py, out_rust, atol=5e-4)`.
- **G_WASM_smoke_graph gate** (NORMATIVE). JS test in
  `crates/semiflow-wasm/tests/smoke_graph.ts`: same logic, Node smoke,
  `assert close to 1.46e-6 sup_err` (consistent with v0.10.0 Wave A).

## Out of scope (v2.2)

- **f32 graph bindings.** Bindings are f64-only. Use ADR-0046
  precision-policy bands for in-Rust f32 usage.
- **Wave B Schrödinger bindings** — Schrödinger does have a working
  v2.2 Rust API but binding for it (especially WASM where Float32Array
  reordering becomes complex) is deferred.
- **`AdjointChernoff` wrapper through bindings.** Internal trait
  composition; not exposed through bindings (FFI lacks Higher-Order
  Function semantics).
- **Browser cross-engine bench** (v0.11.0 Firefox CI deferral).

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Cross-binding sup-error exceeds 3 ULP on aarch64 (M1/M2 macOS, ARM linux) | CI runs on `ubuntu-latest`, `macos-latest`, `windows-latest`; if M1 sees >3 ULP, document in `docs/api-stability.md` §"Cross-binding precision". |
| R2 | Python `LaplacianAtTime` callback adds GIL hot-spot — single Python callable per K=4 sample = 4 GIL acquires per step | ADR-0031 PyO3 GIL release pattern: acquire once per step (around all 4 callback calls). Document in `crates/semiflow-py/src/graph_py.rs` rustdoc. |
| R3 | WASM closure-as-callback for `LaplacianAtTime` adds Function.prototype.call() per sample — costly | WASM impl uses `js_sys::Function` wrapper; benchmark in `tests/smoke_graph.ts`; document expected slowdown vs Rust. |
| R4 | `smf_lap_at_t_fn` callback signature MUST take `void* user_data` — easy to mis-use | Generate `bindgen` headers automatically; documented in C examples. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `crates/semiflow-ffi/src/graph_ffi.rs` | ~400 |
| `crates/semiflow-py/src/graph_py.rs` | ~350 |
| `crates/semiflow-wasm/src/graph_wasm.rs` | ~350 |
| `crates/semiflow-ffi/include/remizov_graph.h` | ~150 |
| `crates/semiflow-py/tests/smoke_graph.py` | ~80 |
| `crates/semiflow-wasm/tests/smoke_graph.ts` | ~100 |
| CI YAML extensions | ~100 |
| ADR-0059 (this) | ~250 |
| **Total** | **~1780** |

## References

- ADR-0028 (FFI/PyO3/WASM v0.10.0) — direct precedent.
- ADR-0029 (WASM panic=abort).
- ADR-0030 (npm publish workflow).
- ADR-0031 (PyO3 GIL release).
- math.md §12 (graph PDE NORMATIVE).
- math.md §14 (variable-topology NORMATIVE — Wave A binding additive).
