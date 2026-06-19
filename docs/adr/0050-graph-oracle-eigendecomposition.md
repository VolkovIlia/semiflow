# ADR-0050 — Test-only Jacobi eigendecomposition oracle for graph PDE

- **Status**: PROPOSED
- **Date**: 2026-05-20
- **Wave**: v2.1 Wave A (Graph PDE Foundations)
- **Companion**: ADR-0047, ADR-0048, ADR-0049.

## Decision

Add a **test-only**, hand-rolled symmetric Jacobi eigendecomposition routine in
`crates/semiflow-core/src/graph_oracle.rs` gated by `#[cfg(test)]`, capable of
diagonalising a symmetric `Laplacian<F>` of size `N ≤ 1024`. The oracle computes

```
   u_oracle(t) = Σ_k φ_k · ⟨φ_k, f₀⟩ · exp(−t · λ_k)
```

where `(λ_k, φ_k)` are the eigenpairs of `L_G`. It exists exclusively to serve as
the reference solution for the **G7 slope-rate gate** comparing
`ChernoffSemigroup<GraphHeatChernoff<F>>` output against `e^{−t L_G} f₀`.

## Why test-only, not production

- **Production path**: callers want `e^{−t L_G} f₀` for graphs of `N ∈ [10³, 10⁵]`
  where dense `O(N³)` eigendecomposition is unacceptable.
- **Wave 2.1A scope**: ship Chernoff approximation + slope gate vs an
  independent reference. The Chernoff path is the production path; the oracle
  exists to validate convergence.
- **`#[cfg(test)]` gating**: the oracle never ships in release artifacts —
  `cargo build --release` excludes it; `cargo doc --no-deps` excludes it
  (`#[doc(hidden)]` belt-and-braces); binary-size CI gate (ADR-0040) untouched.

## Why hand-rolled Jacobi

- **No new dep**: `semiflow-core` direct-dep cap is 3 (`num-traits`, `libm` +
  reserved `num-complex`); using `nalgebra` (≈ 50 transitive deps) or even
  `simba` is prohibited by the constitution.
- **Algorithm fits the file budget**: a clean symmetric Jacobi rotation loop is
  ≈ 100–150 LoC including convergence detection. Well under the 500 LoC
  file cap. No carve-out requested.
- **Numerically robust for `N ≤ 1024`**: Jacobi has worst-case `O(N³)` complexity
  and is known to be backward-stable; for `N = 64` (G7 gate) and `N ≤ 1024`
  (oracle eigenmode parity test) it converges in `≤ 10 N²` rotations and gives
  `‖L_G − Σ λ_k φ_k φ_k^T‖_F ≤ 1e-12 · ‖L_G‖_F` (f64) in practice.
- **f32 + f64**: generic over `F: SemiflowFloat`, with relaxed convergence
  threshold `1e-5` for f32 per ADR-0046 precision policy.

`nalgebra` / `ndarray-linalg` rejected: each pulls in `> 30` transitive deps and
optional BLAS / LAPACK; both violate the suckless dep cap and would force a
project-constitution override.

## API surface (test-only)

```rust
// crates/semiflow-core/src/graph_oracle.rs
#![cfg(test)]

use crate::{float::SemiflowFloat, graph::Laplacian, graph_signal::GraphSignal};
use alloc::vec::Vec;

/// Eigendecomposition `L_G = Q Λ Q^T` of a symmetric Laplacian.
///
/// `eigenvalues[k]` ↔ column `k` of `eigenvectors_col_major` (length `n_nodes`).
/// Returned eigenvalues are sorted ascending.
#[doc(hidden)]
pub(crate) struct EigDecomp<F> {
    pub eigenvalues: Vec<F>,
    pub eigenvectors_col_major: Vec<F>,
    pub n: usize,
}

/// Compute `L_G = Q Λ Q^T` via symmetric Jacobi (off-diagonal Frobenius
/// minimisation). Converges when `Σ_{i≠j} L[i,j]² < tol² · ‖L‖_F²`.
///
/// # Panics (debug)
///
/// `assert!(n <= 1024)` — oracle is dense O(N³); not for production graphs.
#[doc(hidden)]
pub(crate) fn jacobi_eig<F: SemiflowFloat>(lap: &Laplacian<F>) -> EigDecomp<F>;

/// Closed-form heat-semigroup oracle: `u(t) = Σ_k φ_k ⟨φ_k, f₀⟩ e^{−t λ_k}`.
#[doc(hidden)]
pub(crate) fn heat_oracle<F: SemiflowFloat>(
    decomp: &EigDecomp<F>,
    f0: &GraphSignal<F>,
    t: F,
) -> GraphSignal<F>;
```

The decomposition is cached **once per test** to avoid re-running the `O(N³)`
loop for each `N_VALUES` Chernoff iteration count.

## Math fidelity

The oracle implements `u(t) = e^{−t L_G} f₀` exactly (modulo f64 rounding):

1. `f₀ → α = Q^T f₀` (basis change).
2. `α_k → α_k · exp(−t λ_k)` (diagonal exponential).
3. `α → Q α` (back-transform).

`Q^T Q = I` to `1e-14` (f64) post-Jacobi, so back-and-forth incurs only round-off.
Acceptable for the slope-rate gate threshold ≤ −0.95.

## How it's used

```rust
// tests/convergence_graph.rs (G7 slope gate)
let graph = Graph::<f64>::erdos_renyi(64, 0.15, 0xDEADBEEF);
let lap = Arc::new(Laplacian::assemble_combinatorial(&graph));
let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));
let decomp = jacobi_eig(&lap);              // ← oracle, computed once

let f0 = GraphSignal::from_fn(64, |i| /* … */);
let oracle = heat_oracle(&decomp, &f0, 0.5);

let mut errs = Vec::with_capacity(5);
for &n in &[25, 50, 100, 200, 400] {
    let semi = ChernoffSemigroup::new(chernoff_for(n), n).unwrap();
    let u_t = semi.evolve(0.5, &f0).unwrap();
    let mut diff = u_t.clone();
    <GraphSignal<f64> as State<f64>>::axpy_into(&mut diff, -1.0, &oracle);
    errs.push(diff.norm_sup());
}
let slope = log_log_slope(&[25, 50, 100, 200, 400], &errs);
assert!(slope <= -0.95, "G7 slope = {slope:.4}");
```

The pattern mirrors `tests/convergence_rate.rs` (Gaussian heat-kernel oracle for
G3) — same `log_log_slope` helper, same `N_VALUES` template, same gate threshold.

## Eigenmode parity test (separate file)

`tests/graph_heat_oracle.rs` (migrated from spike) is REPURPOSED post-Wave-2.1A:

- DROP: the spike's `PathGraphFn` smoke tests (already covered by the new
  `Graph::path` builder + Wave 2.1A invariant tests).
- ADD: an **eigenmode parity** test — pick a single Laplacian eigenvector
  `φ_k`, apply `GraphHeatChernoff::apply_into(τ, φ_k)` ONCE, check the result is
  `(1 − τ λ_k) φ_k` to f64 round-off (`< 1e-12`) and f32 round-off (`< 1e-5`).
  This is an independent algebraic check separate from the G7 slope gate.

## Acceptance criteria

1. `jacobi_eig` converges to `‖Q^T Q − I‖_F ≤ 1e-13` (f64) on 100 random
   symmetric matrices up to `N = 128` (property test).
2. `heat_oracle` reproduces a hand-derived analytical solution on the
   path-graph (closed-form eigenvalues `λ_k = 2(1 − cos(kπ/N))`, eigenvectors
   `φ_k(j) = √(2/N) sin(jkπ/N)`) to `< 1e-12` for `N = 16`, `t = 0.5`.
3. G7 slope gate ≤ −0.95 passes (f64) and ≤ −0.90 passes (f32, per ADR-0046).
4. Oracle code lives BEHIND `#[cfg(test)]` — verified by `cargo build --release`
   binary-size CI gate (binary size unchanged from v2.0).
5. `#[doc(hidden)]` on the oracle module — no public docs leak.
6. File `crates/semiflow-core/src/graph_oracle.rs` ≤ 300 LoC (well under 500
   cap).

## Open questions (none — all resolved)

| Question | Resolution |
|----------|------------|
| Why not use the matrix exponential directly via Padé? | Pad'e on a non-diagonal `N × N` matrix needs an external linear-algebra dep for the matrix multiplications + LU solve. Jacobi only needs `Vec<F>` + scalar ops. |
| Why `N ≤ 1024` cap? | `1024³ ≈ 10⁹` Jacobi flops; on a single-threaded test runner that's `> 1 s` — fine for `#[ignore]`-able flagship test, but the slope gate uses `N = 64` and the eigenmode parity uses `N = 32`, both well within. |
| Why not `nalgebra`? | Cap on direct deps (3); transitive deps explosion. |
| Determinism? | Jacobi pivot selection: scan upper-triangle for max `|L[i,j]|`. Deterministic for fixed input → bit-identical eigenvectors across runs (up to sign convention: enforce `φ_k[0] ≥ 0` by flipping sign). |
