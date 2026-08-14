//! G_SYMOP_IMPLICIT_STIFF — TEETH: stiff SINGULAR Neumann operator, N=400,
//! scale=1e7 (§59.7 gate B, strengthened for non-vacuous accuracy).
//!
//! ## Why Neumann (singular), not Dirichlet
//!
//! The original Dirichlet Laplacian ×1e7 sends ALL modes to underflow at t=1
//! (λ_min·t ≈ 617 → e^{-617} ≈ 0).  Both oracle and solver return ≈ 0, so
//! `sup_error ≤ 1e-6` proves nothing about accuracy — any method that does not
//! blow up trivially passes.
//!
//! This gate uses a SINGULAR 1-D Neumann Laplacian (scaled ×1e7) instead:
//!
//!   - Boundary rows:  A[0,0]=S,     A[0,1]=-S
//!                     A[N-1,N-2]=-S, A[N-1,N-1]=S   (where S=SCALE)
//!   - Interior rows:  A[i,i-1]=-S,  A[i,i]=2S,  A[i,i+1]=-S
//!
//! All row sums = 0 → the constant vector **1** is an exact null eigenvector (λ_0=0).
//! Non-zero eigenvalues: λ_k = S·2·(1−cos(kπ/N)) for k=1..N-1; smallest ≈ 617.
//!
//! ## Non-trivial surviving-mode oracle
//!
//! At t=1: `e^{−tA}·v → (⟨1,v⟩/N)·1` (constant mode preserved; all others underflow).
//! With `v = linspace(1/N, 1, N)`, mean(v) ≈ 0.501 — oracle is an O(1) vector, NOT ≈ 0.
//!
//! ## Assertions
//! (a) Wallclock ≤ 5 s (explicit Chebyshev path with λ_max ≈ 4e7 would time out).
//! (b) `sup_error ≤ 1e-6` vs the non-trivial surviving-mode oracle.
//! (c) Real total `apply_into_slice` count (measured via `CountedOp` wrapper)
//!     is ≪ ⌈τ·λ_max⌉ ≈ 4×10⁷ — proves actual cost saving, not a fabricated budget.
//!
//! ## Stability-only note
//! The pure Dirichlet (all-underflow) case demonstrates L-stability of backward-Euler
//! (stiff modes correctly damped to ≈ 0) but cannot serve as an accuracy assertion
//! because any output ≈ 0 trivially passes.  It is excluded here; if a separate
//! L-stability regression is needed, add a dedicated gate.

#![cfg(test)]
// Test doc comments use mathematical notation and gate identifiers without backticks.
#![allow(clippy::doc_markdown, clippy::doc_overindented_list_items)]
// N=400; all usize→f64 casts involve values far below 2^53.
#![allow(clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use semiflow::{
    graph_expmv_krylov, scratch::ScratchPool, KrylovPath, SymmetricLinearOp, SymmetricOperator,
};

const N: usize = 400;
const SCALE: f64 = 1e7_f64;
const TAU: f64 = 1.0_f64;
const N_STEPS: usize = 100;

// ── CSR builder — Neumann (singular) scaled Laplacian ────────────────────────

/// Singular 1-D Neumann Laplacian scaled by `SCALE` (row sums = 0 everywhere).
///
/// Boundary rows:  diag = SCALE,   one off-diagonal = −SCALE.
/// Interior rows:  diag = 2·SCALE, two off-diagonals = −SCALE each.
///
/// Constant vector **1** is an exact null eigenvector (λ_0 = 0).
/// All other eigenvalues: λ_k = SCALE·2·(1−cos(kπ/N)) ≥ 617 for k=1..N-1.
// N=400; column indices < N fit in u32 — no truncation possible.
#[allow(clippy::cast_possible_truncation)]
fn build_neumann_csr() -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let mut row_ptr = vec![0usize; N + 1];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for i in 0..N {
        let is_boundary = i == 0 || i == N - 1;
        let diag = if is_boundary { SCALE } else { 2.0 * SCALE };
        if i > 0 {
            col_idx.push((i - 1) as u32);
            vals.push(-SCALE);
        }
        col_idx.push(i as u32);
        vals.push(diag);
        if i + 1 < N {
            col_idx.push((i + 1) as u32);
            vals.push(-SCALE);
        }
        row_ptr[i + 1] = col_idx.len();
    }
    (row_ptr, col_idx, vals)
}

// ── Matvec-counting wrapper ───────────────────────────────────────────────────

/// Transparent `SymmetricLinearOp` wrapper that counts all `apply_into_slice` calls.
///
/// Captures ALL matvec calls inside PCG: Jacobi build (N calls), residual (1/step),
/// and CG iterates (1/CG-iter/step).  Used by gate (c) to measure actual cost.
struct CountedOp<'a> {
    inner: &'a SymmetricOperator<f64>,
    count: AtomicUsize,
}

impl<'a> CountedOp<'a> {
    fn new(inner: &'a SymmetricOperator<f64>) -> Self {
        Self {
            inner,
            count: AtomicUsize::new(0),
        }
    }
    /// Total `apply_into_slice` calls since creation.
    fn total(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

impl SymmetricLinearOp<f64> for CountedOp<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn lambda_max_bound(&self) -> f64 {
        self.inner.lambda_max_bound()
    }
    fn apply_into_slice(&self, src: &[f64], dst: &mut [f64]) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_into_slice(src, dst);
    }
}

// ── Surviving-mode oracle ─────────────────────────────────────────────────────

/// Analytic `e^{−τ·A}·v` for Neumann Laplacian at t = TAU = 1.0.
///
/// Only the null mode (λ_0 = 0) survives; all λ_k ≥ 617 underflow.
/// Result: `(⟨1, v⟩ / N) · 1` — the constant projection, O(1) when mean(v) ≠ 0.
fn neumann_surviving_oracle(v: &[f64]) -> Vec<f64> {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    vec![mean; v.len()]
}

// ── Non-vacuity pre-checks ────────────────────────────────────────────────────

/// Assert structural non-vacuity of the Neumann operator and the initial condition.
///
/// Returns `(lam1_neumann, oracle_inf)` for the eprintln report.
fn assert_non_vacuous(
    op: &SymmetricOperator<f64>,
    row_ptr: &[usize],
    vals: &[f64],
    oracle: &[f64],
) -> (f64, f64) {
    // (i) Still stiff: explicit Chebyshev would need ≥ SCALE matvecs.
    assert!(
        op.lambda_max_bound() >= SCALE,
        "non-vacuity: λ_max_bound={} < SCALE={SCALE:.0e}",
        op.lambda_max_bound()
    );
    // (ii) Smallest non-zero mode causes underflow → surviving mode is the only signal.
    let lam1 = SCALE * 2.0 * (1.0 - (std::f64::consts::PI / N as f64).cos());
    assert!(
        lam1 > 100.0,
        "non-vacuity: λ_1≈{lam1:.1} < 100 — not stiff enough"
    );
    // (iii) Null space: verify row sums are exactly 0 (constant vector ∈ ker A).
    for i in 0..N {
        let s: f64 = (row_ptr[i]..row_ptr[i + 1]).map(|k| vals[k]).sum();
        assert!(s.abs() < 1e-9, "null-space: row {i} sum={s:.3e} ≠ 0");
    }
    // (iv) Oracle is non-trivial (mean(v) ≈ 0.501, well above 0).
    let oracle_inf = oracle.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        oracle_inf > 0.1,
        "non-vacuity: oracle ≈ 0 (value={oracle_inf:.3e})"
    );
    (lam1, oracle_inf)
}

// ── Gate ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::too_many_lines)]
fn g_symop_implicit_stiff() {
    let t_start = Instant::now();

    let (row_ptr, col_idx, vals) = build_neumann_csr();
    let op =
        SymmetricOperator::from_csr(N, &row_ptr, &col_idx, &vals, 1e-6).expect("Neumann CSR build");

    // v = linspace 1/N .. 1: nonzero mean ≈ 0.501 → non-trivial oracle.
    let v: Vec<f64> = (0..N).map(|i| (i as f64 + 1.0) / N as f64).collect();
    let exact = neumann_surviving_oracle(&v);

    let (lam1, oracle_inf) = assert_non_vacuous(&op, &row_ptr, &vals, &exact);

    // Explicit-path cost bound: ⌈τ·λ_max⌉ ≈ 4×10⁷ matvecs.
    // f64::ceil() ≥ 0 (λ_max > 0); value < usize::MAX in any realistic system.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let explicit_mv_budget = (TAU * op.lambda_max_bound()).ceil() as usize;

    // Solve via backward-Euler, counting ALL apply_into_slice calls.
    let counted = CountedOp::new(&op);
    let path = KrylovPath::ImplicitEuler { n_steps: N_STEPS, cg_max_iter: None };
    let mut out = vec![0.0_f64; N];
    let mut scratch = ScratchPool::new();
    graph_expmv_krylov(&counted, TAU, &v, &mut out, path, 1e-10, &mut scratch)
        .expect("implicit Neumann stiff expmv");

    let total_mv = counted.total();
    let elapsed = t_start.elapsed().as_secs_f64();
    let sup_error: f64 = exact
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    eprintln!(
        "G_SYMOP_IMPLICIT_STIFF  N={N}  scale={SCALE:.0e}  n_steps={N_STEPS}  \
         lam1_neumann={lam1:.1}  oracle_inf={oracle_inf:.6}  \
         sup_error={sup_error:.3e}  total_mv={total_mv}  \
         explicit_mv_budget={explicit_mv_budget}  \
         mv_ratio={:.1}×  wallclock={elapsed:.2}s",
        explicit_mv_budget as f64 / total_mv as f64
    );

    // (a) Wallclock ≤ 5 s.
    assert!(
        elapsed <= 5.0,
        "G_SYMOP_IMPLICIT_STIFF (a): wallclock={elapsed:.2}s > 5s"
    );

    // (b) Accuracy vs NON-TRIVIAL surviving-mode oracle (oracle_inf ≈ 0.501·1, not ≈ 0).
    //     Backward-Euler preserves the null space exactly ((I+Δt·0)^{-n}·c = c),
    //     so the constant component is computed exactly; error comes only from
    //     CG residual tolerance (tol_cg=1e-10, n_steps=100 → ≤ 1e-7 accumulated).
    assert!(
        sup_error <= 1e-6,
        "G_SYMOP_IMPLICIT_STIFF (b): sup_error={sup_error:.3e} > 1e-6 \
         vs surviving-mode oracle (oracle_inf={oracle_inf:.6})"
    );

    // (c) Real total matvec count ≪ explicit cost — proves actual (not fabricated) savings.
    //     Includes: N Jacobi-setup calls (once) + n_steps × (1 + CG-iters) calls.
    //     Expect total_mv ~ O(1 000) vs explicit ~ 4×10⁷.
    assert!(
        total_mv < explicit_mv_budget,
        "G_SYMOP_IMPLICIT_STIFF (c): total_mv={total_mv} ≥ explicit_mv_budget={explicit_mv_budget}"
    );
    // Tighter: implicit must be ≥ 100× cheaper — CG's sub-linear cost vs Chebyshev.
    assert!(
        total_mv * 100 < explicit_mv_budget,
        "G_SYMOP_IMPLICIT_STIFF (c′): implicit not dramatically cheaper — \
         total_mv={total_mv}  explicit_mv_budget={explicit_mv_budget}  \
         ratio={:.1}× (need ≥ 100×)",
        explicit_mv_budget as f64 / total_mv as f64
    );

    eprintln!("G_SYMOP_IMPLICIT_STIFF  PASS");
}
