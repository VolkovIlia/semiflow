# ADR-0214 — Separable whole-array sampling for the constant-coefficient N-D shift engines

- **Status**: Accepted
- **Date**: 2026-10-11
- **Supersedes / amends**: none (bit-identical speed-up; no API change).
- **Contract**: math §65 (`contracts/semiflow-core.math-nd-sweep.md`); gates
  `G_ND_SWEEP_BIT_EQUAL`, `G_ND_SWEEP_ZERO_ALLOC` (`contracts/semiflow-core.properties.yaml`).

## Context

`SmolyakGridND` and `AnisotropicShiftChernoffND` compute, per grid node `k`,
`exp(τc(x_k))·π^{−D/2}·Σ_q w_q f(y_q(k))` with `y_q(k) = x_k + τb(x_k) + 2√τ·L_k η_q`.
Each `f(y)` is a `GridFnND::sample`: a tensor-product recursion over `K^D` node
values (`K = 4` for Catmull–Rom). On the `4⁶` grid of `SmolyakD6V8` (533 Smolyak
nodes) one step costs `4096 · 533 · 4096 ≈ 9·10⁹` reads — ≈ 50 s per `apply`, and
the Python test file took ≈ 8 minutes for ten steps.

## Decision

1. **Separable path.** When `b(x_k)` and `L_k` are bitwise the same at every node
   (checked once per call; `c(x)` may vary — it multiplies after the sum), axis
   `d` of `y_q(k)` is `(x_{k_d} + τb_d) + 2√τ(Lη_q)_d`: it depends only on
   `(q, k_d)`, so the stencil (weights and resolved boundary hits) of axis `d`
   depends only on `k_d`. The recursion then equals `D` whole-array passes, axis 0
   first: pass `d` writes, for every node, `Σ_j w_j·bc(hit_j, i ↦ pass_{d−1} at
   axis index i)` with `acc = 0; acc += w_j·v_j` in `j` order — exactly
   `GridFnND::collapse`, where a boundary hit that reads several nodes (affine
   extrapolation, Robin, odd reflection) reads the stored previous-pass values the
   recursion recomputes. The coordinate is evaluated with the engine's own
   association `(x + τb) + 2√τ·lη` (a pre-summed offset would round differently
   when `b ≠ 0`). The outer sum `acc[k] += w_q·g_q[k]` runs over `q` in stored
   order. Same operations in the same order: **bit-identical** for every non-NaN
   output, `D·K` instead of `K^D` reads per node and `q`. (Where two NaNs with
   different payloads meet in one addition the result's payload is unspecified by
   IEEE 754 and Rust; NaN outputs stay NaN.)
2. **Prefix sharing.** Consecutive `q` whose axis offsets agree bitwise on axes
   `0..=d` share passes `0..=d`. The Smolyak table is sorted axis 0 first, so at
   `D = 6, ℓ = 9` 1 238 passes replace 533 · 6. The sharing key is the computed
   offset itself, not `η`, so no property of `L` or of the ordering is assumed;
   the tensor Gauss–Hermite order (axis 0 fastest) simply shares nothing.
3. **Scratch** from the pool: `(D + 1)·N` elements with sharing, `3·N` without;
   above `2²⁴` elements the per-`q` passes, above that the per-node loop. The
   quadrature list `(η_q, w_q)` is built at construction and the per-axis stencils
   are computed once per axis node inside each pass (no table), so `apply_into`
   keeps the zero-allocation contract of ADR-0041 (`G_ND_SWEEP_ZERO_ALLOC`).
4. **Fallback** (variable coefficients, non-primitive `F` such as `Dual` — no
   bitwise test —, shape mismatch, an interpolant without an N-D stencil): the
   per-node loop, moved verbatim into `grid_nd_sweep::node_value` and shared by
   both engines; for `F = f64` its nodes are evaluated through
   `parallel_eval_into` (independent nodes, same code per node: bit-identical).

## Consequences

- `SmolyakD6V8.apply` (4⁶ nodes, CPython 3.13, release wheel): ≈ 50 s → 0.09 s per
  step (≈ 550×); `test_smolyak_v8.py`: ≈ 8 min → 1.1 s, goldens unchanged.
  `binding_smolyak_parity` drops from tens of seconds to milliseconds.
- The end-to-end bit pins are unchanged: the no_std digest `smolyak_heat_d2` and
  the Python goldens `_GOLDEN_FIRST8/_GOLDEN_LAST4` (both `b = 0`, `A = I`); the
  D = 2–6 Smolyak and anisotropic tolerance/slope/moment tests still pass.
- The nightly variable-`A` gates (`g_smolyak_d5/d6`, tanh-coupled `A`) stay on the
  per-node path; they gain only the parallel node loop.

## Gate

`G_ND_SWEEP_BIT_EQUAL` (`src/grid_nd_sweep_tests.rs`, `src/grid_nd_sweep_engine_tests.rs`):
the separable path (both scratch modes), the shared per-node path and the
pre-ADR-0214 loop copied verbatim agree to the bit, every case also asserting
non-vacuity `max|Sf − f| > 1e−3`:
- D = 2 (9×7) and D = 3 (6×5×4): CubicHermite and Linear × all eight boundary
  policies (D = 3 rotated per axis) × τ ∈ {0.05, 0.6} × {Smolyak ℓ = 5, tensor
  Gauss–Hermite}, dense lower-triangular `L`, `b ≠ 0`, variable `c`, in f64; the
  D = 2 Smolyak matrix also in f32.
- D = 5 (4⁵), ℓ = 8 (341 nodes), three policy rotations: Linear on the full table,
  CubicHermite on its first 40 nodes. D = 6 (4⁶), ℓ = 9 (533 nodes): Linear on the
  first 160, CubicHermite on the first 8 and on every 41st node (all axis-0 runs) —
  the reference loop reads 4⁶ values per sample, so the full cubic table is too slow
  for a gate.
- Engine level: `SmolyakGridND::apply_into` and `AnisotropicShiftChernoffND::apply_into`
  with a constant dense `A`, `b ≠ 0`, variable `c` (separable path) and with a
  variable `b` (per-node path), D = 2 and 3, against the verbatim loop fed the same
  Cholesky cache and quadrature.
- Refusal: variable `b`, an `L` one ULP off at one node, `±0` drift, a scratch cap
  below `3N`; `Dual` has no bitwise test (path disabled).
- NaN inputs (NaN, payload NaN, ±inf in the field): non-NaN outputs bit-identical,
  NaN outputs NaN on both paths (payload unspecified).

`G_ND_SWEEP_ZERO_ALLOC` (`tests/nd_shift_zero_alloc.rs`): a steady-state
`apply_into` of either engine allocates nothing (separable path; per-node path in
the serial build).
