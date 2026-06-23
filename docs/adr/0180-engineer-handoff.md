# ADR-0180 — Engineer hand-off: GraphAdjoint batched time-grid sampler

Design: `docs/adr/0180-graphadjoint-batched-time-grid-sampler.md`.
Oracle (already passes): `scripts/verify_graphadjoint_sampled.py`.
**Additive only** — the existing closure constructor (`MagnusGraphHeatChernoff::new`
+ PyO3 `lap_at_t`) MUST keep working. Contract-first: implement against the ABI
and types specified in ADR-0180; do NOT invent signatures.

## The one thing that will bite you

The grid is **`2·n_steps`**, NOT `n_steps`. Magnus K=4 samples `L_G` at two GL₄
abscissae `t_start + c₁τ`, `t_start + c₂τ` per step (`magnus_graph_adjoint.rs:169-172`).
`vals_seq` is laid out **per (step, abscissa)** in schedule order
`[(0,c1),(0,c2),(1,c1),(1,c2),…]`, length `2·n_steps·nnz`. The adjoint sweep uses
`t_start=(n_steps−1−k)·τ`; replay MUST index the SAME schedule (block `2k` = c1,
`2k+1` = c2). Get this wrong and parity fails at O(τ²). The oracle's `WRONG`
variant demonstrates the failure mode.

## File checklist

### 1. Core — `crates/semiflow-core/src/`
- [ ] **`graph.rs`** — add `Laplacian::from_csr_parts(n_nodes, row_ptr: Vec<usize>,
      col_idx: Vec<u32>, vals: Vec<F>, kind: LaplacianKind) -> Result<Self, SemiflowError>`.
      Recompute `spectral_radius_bound` via the existing Gershgorin path
      (`compute_gershgorin_bound` over reconstructed rows) so the cached-bound
      invariant holds. Validate `row_ptr.len()==n_nodes+1`, monotone non-decreasing,
      `col_idx.len()==vals.len()==row_ptr[n_nodes]`, indices `< n_nodes`
      → else `DomainViolation`. ≤50 lines (extract a validator helper if needed).
- [ ] **`magnus_graph_adjoint.rs`** — add `PreSampledLaplacianSeq<F>` struct (fields
      per ADR-0180) + `MagnusGraphHeatChernoff::from_presampled(graph, seq, rho_bar,
      conv_check)` + `VarCoefMagnusGraphHeatChernoff::from_presampled(graph, seq,
      a_seq, rho_bar, a_sup_max)`. Add `evolve_state_adjoint_into_presampled` (or
      route through a private replay that indexes `vals_seq` blocks instead of
      calling `self.lap_at_t`). Reconstruct `lap1`/`lap2` per step via
      `Laplacian::from_csr_parts`; feed the SAME `apply_exp_omega4_adj_kernel` /
      `apply_exp_omega4_la_adj_kernel` so float ops are byte-identical. Validate
      `vals_seq.len()==2·n_steps·nnz` and `n_steps`-at-evolve == `n_steps`-at-ctor
      → `DomainViolation`/`OutOfDomain`. Reuse R4 zero-alloc scratch discipline.
- [ ] **`lib.rs`** — re-export `PreSampledLaplacianSeq` (mirror `MagnusGraphHeatChernoff`).

### 2. FFI — `crates/semiflow-ffi/src/graph_adjoint_ffi.rs` (NEW)
- [ ] `SmfGraphAdjoint` opaque handle (`#[repr(C)] { _private: [u8;0] }`) + inner.
- [ ] `smf_graph_adjoint_new_presampled(...)` — exact signature in ADR-0180.
      Null-check BEFORE `catch_panic!`; copy `row_ptr`/`col_idx`/`vals_seq` into
      owned `Vec`s (caller may free immediately); validate pattern against `topo`
      (`smf_graph_*`); `kind` 0/1 → `LaplacianKind`. No new error variants
      (ADR-0171): mismatch → `OutOfDomain`, null → `NullPtr`.
- [ ] `smf_graph_adjoint_new_presampled_varcoef(...)` — adds `a_seq` (len
      `2·n_steps·n_nodes`) + `a_sup_max`.
- [ ] `smf_graph_adjoint_evolve_state_adjoint(...)` — `lambda_n` in, `out` written;
      `GridMismatch` if `lambda_len`/`out_len < n_nodes`; `OutOfDomain` if evolve
      `n_steps` != ctor `n_steps`.
- [ ] `smf_graph_adjoint_n_nodes` + `smf_graph_adjoint_free` (null-safe, idempotent,
      `catch_unwind` drop).
- [ ] Register module in `crates/semiflow-ffi/src/lib.rs`; regen C header
      (`cargo run -p xtask -- ffi-headers`).

### 3. PyO3 — `crates/semiflow-py/src/graph_adjoint.rs` (extend, do NOT replace)
- [ ] Add a sample-once path: classmethod `GraphAdjoint.from_presampled(...)` OR a
      `presample=True` kwarg. Under the GIL: compute grid `T` from
      `(t_horizon, n_steps, C1, C2)`, call the Python `lap_at_t` once per grid point
      (preferred: accept a vectorized `lap_at_ts(ts)->list[Laplacian]` to collapse
      crossings), `extract` each `vals` (validate CSR pattern vs topology → `PyErr`
      raised BEFORE compute), build `PreSampledLaplacianSeq`; VarCoef also samples
      `a(t)` on the same grid.
- [ ] `evolve_state_adjoint`: when presampled, run the replay inside `py.detach`
      with **no** `Python::attach` in the loop (delete the per-step attach at
      `graph_adjoint.rs:206-209` for this path). ADR-0031 GIL release preserved.
- [ ] Keep the existing closure constructor + per-step path intact (additive).

### 4. WASM — `crates/semiflow-wasm/src/` (mirror)
- [ ] `GraphAdjoint.fromPresampled(topo, rowPtr: Uint32Array, colIdx: Uint32Array,
      valsSeq: Float64Array, nSteps, tHorizon, rhoBar, kind)` + `evolveStateAdjoint`.
      Single JS↔WASM crossing at construction; `Result<_, JsValue>`; `panic=abort`
      profile (ADR-0028 Amendment 1). Single-threaded → no `Send+Sync` concern.

### 5. Gate test — `crates/semiflow-core/tests/graph_adjoint_sampled_parity.rs` (NEW)
- [ ] `G_GRAPH_ADJOINT_SAMPLED_PARITY` (`#[cfg_attr(not(feature="slow-tests"), ignore)]`,
      RELEASE_BLOCKING). Path-8, `w_k(t)=1+0.5·sin(t+0.1k)`, `t_horizon=0.5`,
      `n_steps=64`, both Magnus and VarCoef kernels. Build `vals_seq` by sampling the
      SAME closure on the GL₄ grid; assert
      `presampled.evolve_state_adjoint(...).values() == closure.evolve_state_adjoint(...).values()`
      via `assert_eq!` on `&[f64]` (**0 ULP, bit-exact** — NOT ε; both paths run
      identical float ops). Mirror the oracle's numbers.

### 6. Oracle — `scripts/verify_graphadjoint_sampled.py` (ALREADY WRITTEN, passes)
- [ ] No change needed; keep as the language-independent pre-flight reference.

### 7. CHANGELOG.md
- [ ] `### Added` — pre-sampled time-grid Laplacian path for `GraphAdjoint`
      (FFI/PyO3/WASM); closes the PyO3-only deferral (ADR-0180, extends ADR-0179).
      Note the fixed-topology / time-varying-weight scope wall.

## Verification commands

```bash
# 0. design oracle (already green)
python3 scripts/verify_graphadjoint_sampled.py

# 1. core build + parity gate (RELEASE_BLOCKING)
cargo test -p semiflow-core --features slow-tests graph_adjoint_sampled_parity

# 2. fast suite (no regressions on the closure path)
cargo run -p xtask -- test-fast

# 3. FFI build + header drift + smoke
cargo build -p semiflow-ffi --profile release-ffi
cargo run -p xtask -- ffi-headers     # must be drift-free
cargo run -p xtask -- ffi-smoke

# 4. PyO3 + WASM
cargo run -p xtask -- py-build && cargo run -p xtask -- py-smoke
cargo run -p xtask -- wasm-build      # if present

# 5. lints / suckless gates
cargo clippy --all-targets -- -D warnings
cargo run -p xtask -- check-lints
cargo run -p xtask -- check-unsafe-scope
```

## Do NOT

- Do NOT call any host callback inside the evolve loop on the presampled path.
- Do NOT add a new `SemiflowError`/`SemiflowStatus` variant (ADR-0171 — reuse
  `OutOfDomain`/`GridMismatch`/`NullPtr`).
- Do NOT sample on the step grid (`n_steps` points) — it MUST be the `2·n_steps`
  GL₄ grid.
- Do NOT break or change the existing closure constructor / per-step path.
- Do NOT widen scope to topology-varying or state-dependent Laplacians — they are
  walled `OutOfDomain` / out of scope per ADR-0180.
