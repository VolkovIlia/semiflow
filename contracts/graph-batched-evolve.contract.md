# Contract — Batched multi-channel graph evolve + adjoints (Issue #10)

- **ADR**: `docs/adr/0184-batched-multichannel-graph-evolve.md`
- **Status**: Proposed (design only — no implementation in this branch yet)
- **Scope**: ADDITIVE. New `*_batched` methods only; existing 1-D `evolve`,
  `evolve_state_adjoint`, `edge_weight_grad` are byte-unchanged.
- **Suckless**: new core code in sibling module
  `crates/semiflow/src/graph_batched.rs` (+ `graph_batched_tests.rs`), split via
  `include!` if it approaches 500 lines; functions ≤ 50 lines; NO new deps in
  `semiflow` core.

## 0. Layout invariants (NORMATIVE — see ADR-0184 D1)

| Surface | Layout | Meaning |
|---------|--------|---------|
| **Python** (PyO3) | `[N, C]` row-major | torch GNN feature matrix, passed as-is (no `.t()`). `N` = nodes, `C` = channels. Channel `c` = column `c` (stride `C`). |
| **Core** (Rust) | `[C, N]` flat, channel-major | channel `c` occupies `cols[c*N .. c*N + N]`, contiguous. Zero-dep (`&[F]` / `&mut [F]`). |

The PyO3 wrapper bridges the two via a strided gather/scatter fused into the
mandatory GIL-boundary copy (no separate transpose pass).

Common validation (all batched entry points):
- `src_cols.len() == n_cols * n_nodes` (else `DomainViolation`).
- `dst_cols.len() == n_cols * n_nodes`.
- `n_cols >= 1`, `n_nodes == graph.n_nodes()`.
- per-channel `tau`/`t`/`n_steps` validation identical to the 1-D path.

---

## 1. Core Rust — forward kernels (`graph_batched.rs`)

### 1a. Plain ChernoffFunction kernels — ONE generic helper

Covers `GraphHeatChernoff` (order-1 & order-2), `GraphHeat4thChernoff`,
`GraphHeat6Chernoff` — all `ChernoffFunction<F, S = GraphSignal<F>>` driven by
`Evolver`/`ChernoffSemigroup`. The concrete `S = GraphSignal<F>` bound lets the
helper build/read state from a slice (no new generic-State machinery).

```rust
/// Evolve `n_cols` channels of `(S(t/n))^n` in one call.
///
/// `src_cols`/`dst_cols` are flat `[C, N]` (channel-major). Allocates ONE
/// ping-pong buffer pair + ONE `ScratchPool`, reused across all channels.
/// Bit-identical (0 ULP) to looping `Evolver::evolve_into` per channel.
///
/// # Errors
/// `DomainViolation` (length/`tau`/`n` checks); kernel errors propagate.
pub fn evolve_batched<C, F>(
    func: &C,
    graph: alloc::sync::Arc<crate::graph::Graph<F>>,
    t_final: F,
    n_steps: usize,
    n_nodes: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), crate::error::SemiflowError>
where
    C: crate::chernoff::ChernoffFunction<F, S = crate::graph_signal::GraphSignal<F>>,
    F: crate::float::SemiflowFloat;
```

### 1b. Magnus trajectory kernels — typed helpers (hoist GL₄ sampling)

`MagnusGraphHeatChernoff` (K=4) and `MagnusGraphHeat6Chernoff` (K=6) are
time-dependent (driven by `evolve_with_traj`, not `Evolver`). The
channel-independent GL₄ Laplacian sampling is hoisted out of the channel loop
and shared by all channels (big win over the 1-D binding which re-samples per
call).

```rust
/// Batched forward for the Magnus K=4 graph-heat kernel. GL₄ Laplacian samples
/// are computed ONCE and replayed for every channel. `[C, N]` in/out.
pub fn evolve_batched_magnus<F: SemiflowFloat>(
    mc: &crate::magnus_graph::MagnusGraphHeatChernoff<F>,
    t_final: F,
    n_steps: usize,
    n_nodes: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError>;

/// Batched forward for the Magnus K=6 graph-heat kernel (same shape as K=4).
pub fn evolve_batched_magnus6<F: SemiflowFloat>(
    mc: &crate::magnus6_graph::MagnusGraphHeat6Chernoff<F>,
    t_final: F,
    n_steps: usize,
    n_nodes: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError>;
```

### 1c. VarCoef kernels — typed helpers (hoist Laplacian + a-sequence)

`VarCoefGraphHeatChernoff` (ζ-A, order-2) and `VarCoefMagnusGraphHeatChernoff`
(GL₄). Channel-independent `a(·)` + Laplacian sampling hoisted once.

```rust
pub fn evolve_batched_varcoef_heat<F: SemiflowFloat>(
    vc: &crate::graph_heat::GraphHeatChernoff<F>,   // ζ-A order-2 ctor
    /* or the dedicated VarCoef ζ-A type if distinct */
    t_final: F, n_steps: usize, n_nodes: usize,
    src_cols: &[F], dst_cols: &mut [F],
) -> Result<(), SemiflowError>;

pub fn evolve_batched_varcoef_magnus<F: SemiflowFloat>(
    vc: &crate::varcoef_magnus_graph::VarCoefMagnusGraphHeatChernoff<F>,
    t_final: F, n_steps: usize, n_nodes: usize,
    src_cols: &[F], dst_cols: &mut [F],
) -> Result<(), SemiflowError>;
```

> Engineer note: confirm the exact public type names for the ζ-A VarCoef forward
> kernel during implementation (`graph_heat.rs::with_zeta_a` vs a dedicated
> type) and fold 1b/1c into a single generic if they share a driving trait —
> prefer one generic over four typed helpers if the trait bound is clean.

### 1d. (Optional, follow-up) channel-parallel variant

Behind `#[cfg(feature = "parallel")]` + threshold `n_cols >= 2`: split channels
across `std::thread::scope` workers (ADR-0018), one `ScratchPool` per worker.
Bit-identical to serial (no cross-channel reduction in forward). Same public
signatures as 1a–1c (internal dispatch on feature + threshold).

---

## 2. Core Rust — adjoint stack

### 2a. Stacked state-VJP (presampled, ADR-0180 types)

```rust
impl<F: SemiflowFloat> PreSampledMagnusAdj<F> {
    /// Batched backward costate sweep. `src_cols`/`dst_cols` are `[C, N]`.
    /// Each channel's λ is evolved INDEPENDENTLY → stacked `[C, N]` output.
    /// Pre-sampled Laplacian sequence is shared across channels (hoisted).
    /// Bit-identical to per-channel `evolve_state_adjoint_into`.
    pub fn evolve_state_adjoint_batched_into(
        &self,
        tau: F,
        n_steps: usize,
        n_nodes: usize,
        n_cols: usize,
        src_cols: &[F],
        dst_cols: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
}

impl<F: SemiflowFloat> PreSampledVarCoefAdj<F> {
    pub fn evolve_state_adjoint_batched_into(
        &self, tau: F, n_steps: usize, n_nodes: usize, n_cols: usize,
        src_cols: &[F], dst_cols: &mut [F], scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
}
```

### 2b. Summed edge-weight gradient (`graph_sensitivity.rs`)

```rust
/// Multi-channel assembled adjoint-state gradient. `u0_cols`/`dj_cols` are
/// `[C, N]`. Returns ONE gradient `∂J/∂θ = Σ_c ∂J_c/∂θ` of length
/// `param_deriv.n_params()` (math.md §43.4 + linearity, ADR-0184 D4).
///
/// `grad_theta` is zeroed ONCE, then each channel's contribution is ADDED in
/// ascending channel index (NOT re-zeroed) → bit-identical to summing `n_cols`
/// separate `adjoint_state_gradient` calls in index order.
#[allow(clippy::too_many_arguments)]
pub fn adjoint_state_gradient_batched<F, P>(
    mc: &MagnusGraphHeatChernoff<F>,
    u0_cols: &[F],
    dj_cols: &[F],
    n_cols: usize,
    n_nodes: usize,
    n_steps: usize,
    tau: F,
    param_deriv: &P,
    grad_theta: &mut [F],   // SUMMED over channels, len = n_params
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where F: SemiflowFloat, P: GeneratorSensitivity<F>;
```

---

## 3. PyO3 wrappers (thin `[N, C]` adapters)

Pattern for every wrapper (ADR-0031 three-phase, ONE `py.detach`):
1. **GIL held**: validate; read `PyReadonlyArray2<f64>` view `[N, C]`; gather
   columns into owned `[C, N]` `Vec<f64>` (strided `.column(c)` → contiguous
   subslice — this IS the mandatory marshalling copy).
2. **`py.detach`**: call the core `*_batched` fn on the `[C, N]` buffers.
3. **GIL held**: scatter `[C, N]` result into one `PyArray2<f64>` `[N, C]`.

### 3a. Forward kernels — add to each pyclass

```rust
// GraphHeat (graph_py.rs), GraphHeat4th/GraphHeat6 (graph_v2_4.rs),
// MagnusGraphHeat (graph_py.rs), MagnusGraphHeat6 / VarCoef* (graph_extra.rs,
// magnus_graph_py.rs) — one method per class:
#[pyo3(signature = (t_final, n_steps, f0))]
fn evolve_batched<'py>(
    &self,
    py: Python<'py>,
    t_final: f64,
    n_steps: u32,
    f0: PyReadonlyArray2<'py, f64>,   // [N, C]
) -> PyResult<Bound<'py, PyArray2<f64>>>;  // [N, C]
```

Optional `dtype="f32"` kwarg parity with the 1-D `evolve` (Issue #3 / ADR-0115)
— same f64→f32 post-cast.

### 3b. Adjoint — `GraphAdjoint` + `GraphAdjointPresampled` (graph_adjoint.rs)

```rust
#[pyo3(signature = (lambda_cols, n_steps=None))]
fn evolve_state_adjoint_batched<'py>(
    &self,
    py: Python<'py>,
    lambda_cols: PyReadonlyArray2<'py, f64>,  // [N, C] stacked RHS
    n_steps: Option<u32>,
) -> PyResult<Bound<'py, PyArray2<f64>>>;     // [N, C] stacked state-VJP
```

### 3c. `edge_weight_grad` — batched RHS, summed output (graph_sensitivity_py.rs)

```rust
/// `u0`/`dj_du_n` accept 2-D `[N, C]`. Returns ONE summed ∂J/∂w of length
/// n_params (1-D). 1-D inputs remain valid via the existing function (additive
/// new `edge_weight_grad_batched`, OR overload by ndim — prefer a new name).
#[pyfunction]
#[pyo3(signature = (graph=None, a=None, *, u0, dj_du_n, t, n_steps, rho_bar, params))]
fn edge_weight_grad_batched<'py>(
    py: Python<'py>,
    graph: Option<&Bound<'_, PyAny>>,
    a: Option<&Bound<'_, PyAny>>,
    u0: PyReadonlyArray2<'py, f64>,        // [N, C]
    dj_du_n: PyReadonlyArray2<'py, f64>,   // [N, C]
    t: f64, n_steps: u32, rho_bar: f64,
    params: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>>;  // [n_params], summed over channels
```

---

## 4. Test + benchmark plan

### 4a. Correctness gate — 0-ULP bit-equality (NO new sympy oracle)

`batched == per-channel-loop` is a structural identity → bit-equality suffices.

- **Rust** `crates/semiflow/src/graph_batched_tests.rs`: for each of the 6
  forward kernels + both adjoint paths, build `C ∈ {1, 4}` random channels,
  compute (i) `*_batched` `[C, N]` output, (ii) per-channel loop of the existing
  1-D method; `assert_eq!` on `f64`/`Vec<f64>` (exact 0 ULP; in the NaN-free
  domain `assert_eq!` is identical to checking `a.to_bits() == b.to_bits()`, so
  the tests use `assert_eq!` directly). Edge-weight grad: assert summed batched grad equals
  `Σ_c grad_c` accumulated in ascending channel index, exactly.
- **Rust parallel parity** (if D3 1d lands): `--features parallel` run must equal
  the serial `--features ""` run, 0 ULP (forward and order-pinned grad).
- **Python** `crates/semiflow-py/tests/`: numpy parity —
  `np.array_equal(gh.evolve_batched(t, n, X), np.stack([gh.evolve(t, n, X[:,c]) for c in range(C)], axis=1))`
  for every kernel + both adjoint methods; assert exact equality.
- Existing §43.6 finite-difference oracle (`T_ADJOINT_STATE_SENSITIVITY`)
  unchanged — still covers underlying gradient correctness.

### 4b. Benchmark spec — prove the `O(C)` Python overhead is removed

`crates/semiflow-py/benches/` (criterion) + a Python timing harness:
- **Fixture**: 90-node graph, `C ∈ {1, 4, 8, 16}`, `n_steps = 8`, seeded GBM/
  random features (match the Issue #10 profile so numbers are comparable).
- **Metrics** per `C`: (i) batched forward wall-time; (ii) per-channel Python
  loop wall-time (baseline = current behaviour); (iii) raw 1-channel Rust
  evolve; (iv) torch dense-unroll fwd+bwd (≈ 620 µs reference).
- **Targets / acceptance**:
  - batched-`C` forward ≈ raw-1ch × small constant (hoist + shared buffers),
    **not** × C — i.e. the per-channel Python/PyO3/numpy overhead (≈ 70 % at
    `C=4`) is gone; Rust fraction → ~100 %.
  - batched fwd+bwd **beats** the per-channel loop and **approaches** the
    ≈ 620 µs dense baseline (current ≈ 7700 µs at `C=4`).
  - GIL-release count = 1 per batched call (vs `C` today) — assert via the
    ADR-0031 detach pattern, not per-channel.
- **Memory**: one `[C, N]` owned buffer + reused ping-pong (no per-channel
  re-alloc) — record peak vs the per-channel loop (memory-first per project
  convention).
- Report to `benchmarks/results/aggregate/` with matched-`C` columns (the
  aggregate `.md` is the citable source of truth).

### 4c. Wheel-features verification (sub-issue, separate change)

After the D6 quick win (`features = ["std", "simd"]`) lands: rebuild the wheel,
re-run 4b, and confirm the per-channel SpMV SIMD speedup is present (the batched
hoist + SIMD compound). Verify manylinux_2_28 portability (runtime SIMD feature
detection, no illegal-instruction on baseline x86-64).
