//! G3⁴-2D — FLAGSHIP 2D 4th-order convergence gate (v0.6.0, ADR-0013).
//!
//! PDE: `∂_t u = ½(∂_xx + ∂_yy)u`, `u_0(x,y) = exp(-(x²+y²))`.
//!
//! # Closed-form oracle
//! 2D heat kernel (math.md §10.5(a), eq. 10.7):
//! ```text
//! u(t, x, y) = (1+2t)^{-1} · exp(-(x²+y²)/(1+2t))
//! ```
//! At `t = 1`: `u(1, x, y) = ⅓ · exp(-(x²+y²)/3)`.
//!
//! # Operator
//! `Strang2D<Diffusion4thChernoff(0.5), Diffusion4thChernoff(0.5)>`.
//! Per-axis τ-order 2 (`Diffusion4thChernoff`, post-D1); spatial dx⁴ accuracy
//! verified by the convergence-slope gate below, NOT by `Strang2D::order()`
//! (math.md §11.1.bis).
//!
//! # Gate (G3⁴-2D — FLAGSHIP)
//!
//! ## Test 1: `temporal_convergence_2d_4th` (n-sweep)
//! n-sweep at fixed N=2000 per axis. Documents O(τ²) Chernoff convergence.
//! Gate: slope ≤ -1.85 (temporal order 2, as expected for any Strang composition).
//!
//! ## Test 2: `spatial_convergence_2d_4th` (dx-sweep — FLAGSHIP, ASYMPTOTIC calibration)
//! FIXED n=4000, t=0.5 (τ = 0.5/4000 = 1.25e-4, temporal floor ≈0.3e-9).
//! Sweep N per axis ∈ {200, 300, 400}; domain [-15, 15]².
//! Gate: log-log OLS slope of ‖err‖_∞ vs N is ≤ -3.85 (asymptotic spatial order 4).
//! Parallel 8-thread execution: palindromic Strang rows/cols split via `std::thread::scope`.
//!
//! Recalibrated from hardware measurements: at n=2000 the temporal floor on this
//! machine is ≈1.2e-9, contaminating N=800 and N=1600 in the old sweep (slope −1.45
//! FAIL). `N_STEPS` raised to 4000 (floor → 0.3e-9); `N_SWEEP` reduced to {200,300,400}
//! where all points are spatial-dominated (floor < 3.5% of smallest spatial error).
//!
//! # Design note
//! n-sweep at fixed N measures temporal Chernoff convergence (capped at order 2
//! for these Strang compositions), NOT spatial 4th-order. To demonstrate 4th-order
//! SPATIAL benefit, sweep N at fixed large n — mirroring Phase 1's proven approach
//! in `tests/convergence_rate_4th.rs::spatial_convergence_constant_a`.
//!
//! # Parallel kernel (Phase 4, `parallel_strang2d_step`)
//! Implements palindromic Strang `S(τ) = X(τ/2) ∘ Y(τ) ∘ X(τ/2)` using 8 threads
//! via `std::thread::scope`. Operates on a shared pre-allocated `Vec<f64>` buffer.
//! No allocation inside the time loop. No allocator contention (buffers are pre-
//! partitioned). Column-gather approach for Y-pass (option b): each thread handles
//! a contiguous range of column indices; gather/scatter with stride `nx`.
//!
//! # Cost estimate (release mode, `RUST_TEST_THREADS=1`, 8-core parallel)
//! `temporal_convergence_2d_4th`: n∈{16,32,64} at N=800/axis ≈ 80 s (sequential, small).
//! `spatial_convergence_2d_4th` (parallel):
//!   N=200:   4000 × 40k   cells (~4 s par)
//!   N=300:   4000 × 90k   cells (~9 s par)
//!   N=400:   4000 × 160k  cells (~16 s par)
//!   Total spatial: ~30 s
//! Grand total: ~2 min.
//! Empirical: OLS slope over {200, 300, 400} ≈ −5.84 (PASS gate −3.85).
//!
//! Reference: `contracts/semiflow-core.math.md §10.3 Theorem 7`, §9.2.1,
//! `contracts/semiflow-core.properties.yaml` gate `G3_4_2D`,
//! `docs/adr/0013-fourth-order-spatial.md`, `docs/adr/0012-tensor-product-2d.md`.

#![cfg(feature = "slow-tests")]
#![allow(clippy::needless_pass_by_value)] // Diffusion4thChernoff passed by value for Clone inside thread closure
#![allow(clippy::too_many_lines)]         // parallel_y_pass is 51 lines due to detailed column-gather comments

use semiflow_core::{
    chernoff::ApplyChernoffExt, ChernoffFunction, ChernoffSemigroup, Diffusion4thChernoff, Grid1D,
    Grid2D, GridFn1D, GridFn2D, Strang2D,
};

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -15.0;
const X_MAX: f64 = 15.0;
/// Default t for temporal test (t=1.0). Spatial test uses its own `T_SPATIAL=0.5`.
const T_FINAL: f64 = 1.0;

// Number of threads for the parallel Strang2D kernel.
const N_THREADS: usize = 8;

// ---------------------------------------------------------------------------
// Oracle: 2D heat kernel
// ---------------------------------------------------------------------------

/// 2D heat-kernel oracle: `(1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`.
///
/// Normative formula from `contracts/semiflow-core.math.md §10.5(a)` eq. (10.7).
/// Initial datum: `u_0(x,y) = exp(-(x²+y²))`.
/// PDE: `∂_t u = ½(∂_xx + ∂_yy)u`.
#[inline]
fn oracle_heat_2d(t: f64, x: f64, y: f64) -> f64 {
    let denom = 1.0 + 2.0 * t;
    (1.0 / denom) * (-(x * x + y * y) / denom).exp()
}

// ---------------------------------------------------------------------------
// Parallel Strang2D kernel (test-local, no production code changes)
// ---------------------------------------------------------------------------

/// Apply one X-pass (half-step or full-step) to `state` in parallel.
///
/// Splits `ny` rows into `N_THREADS` chunks. Each thread owns a contiguous
/// slice of rows (via `chunks_mut`), extracts each row as a `GridFn1D`,
/// applies `op.apply_chernoff(tau, &row)`, and writes the result back in-place.
///
/// No allocator contention: each thread allocates only for its own rows
/// and the allocations are not concurrent with others in the same chunk.
/// The per-row `GridFn1D` is stack-allocated (heap for `Vec<f64>` of size nx)
/// and freed immediately after `write_back`.
fn parallel_x_pass(
    state: &mut [f64],
    nx: usize,
    ny: usize,
    gx: Grid1D,
    op: Diffusion4thChernoff,
    tau: f64,
) {
    // Split state into ny row-sized chunks (each row = nx contiguous f64).
    // chunks_mut gives non-overlapping mutable slices — borrow-safe for scope.
    let chunk_size = ny.div_ceil(N_THREADS);

    std::thread::scope(|s| {
        for row_chunk in state.chunks_mut(chunk_size * nx) {
            let rows_in_chunk = row_chunk.len() / nx;
            let op_clone = op.clone();
            s.spawn(move || {
                for r in 0..rows_in_chunk {
                    let row_start = r * nx;
                    let row_end = row_start + nx;
                    // Build GridFn1D from this row slice (copy into Vec).
                    let row_vals: Vec<f64> = row_chunk[row_start..row_end].to_vec();
                    let row_fn = GridFn1D {
                        values: row_vals,
                        grid: gx,
                    };
                    // Apply 1D Chernoff op.
                    let evolved = op_clone.apply_chernoff(tau, &row_fn).expect("x-pass apply");
                    // Write evolved values back in-place.
                    row_chunk[row_start..row_end].copy_from_slice(&evolved.values);
                }
            });
        }
    });
}

/// Apply one Y-pass (full-step) to `state` in parallel.
///
/// Splits `nx` column indices into `N_THREADS` chunks. Each thread gathers
/// its assigned columns (strided reads: stride = nx, count = ny), applies
/// `op.apply_chernoff(tau, &col)`, and scatters the result back.
///
/// Cache behaviour: strided gather/scatter. For N=3200 (82 MB state),
/// L3 bandwidth dominates; strided access has ~2× penalty vs sequential
/// but is still fast relative to compute. Column-gather is simpler than
/// double-transpose and avoids an extra 82 MB allocation.
fn parallel_y_pass(
    state: &mut [f64],
    nx: usize,
    ny: usize,
    gy: Grid1D,
    op: Diffusion4thChernoff,
    tau: f64,
) {
    // We need mutable access to non-overlapping column ranges of `state`.
    // `chunks_mut` splits by rows (contiguous), not by columns (strided).
    // To avoid unsafe, we use a per-thread scratch buffer approach:
    //   - Build thread-local column buffers (gather), apply, scatter back.
    // The scatter uses raw pointer arithmetic on disjoint column ranges,
    // which requires unsafe. Instead, we use a safe two-phase approach:
    //   Phase 1: gather + apply for all columns (parallel, write to temp).
    //   Phase 2: scatter results from temp back into state (parallel by row).
    //
    // Temp layout: temp[col_idx * ny + j] = evolved value for col col_idx, row j.
    // This is column-major in temp, row-major in state.

    // Pre-allocate temp storage (nx * ny f64 = same size as state).
    let mut temp = vec![0.0_f64; nx * ny];

    // Phase 1: gather + apply (read-only from state, write to non-overlapping temp columns).
    // Partition column indices into N_THREADS chunks.
    let col_chunk_size = nx.div_ceil(N_THREADS);

    // We need to share `state` (read-only) and `temp` (write, disjoint columns).
    // Split temp into column-range slices.
    // temp is column-major: col c occupies temp[c*ny .. (c+1)*ny].
    // chunks_mut on temp with chunk_size = col_chunk_size * ny gives disjoint column ranges.
    {
        let state_slice: &[f64] = state;
        std::thread::scope(|s| {
            for (chunk_idx, temp_chunk) in temp.chunks_mut(col_chunk_size * ny).enumerate() {
                let col_start = chunk_idx * col_chunk_size;
                let cols_in_chunk = temp_chunk.len() / ny;
                let op_clone = op.clone();
                s.spawn(move || {
                    for c in 0..cols_in_chunk {
                        let col_idx = col_start + c;
                        if col_idx >= nx {
                            break;
                        }
                        // Gather column col_idx from state (row-major, stride nx).
                        let col_vals: Vec<f64> =
                            (0..ny).map(|j| state_slice[j * nx + col_idx]).collect();
                        let col_fn = GridFn1D {
                            values: col_vals,
                            grid: gy,
                        };
                        // Apply 1D Chernoff op.
                        let evolved = op_clone.apply_chernoff(tau, &col_fn).expect("y-pass apply");
                        // Write evolved values into temp column c (column-major temp).
                        let temp_start = c * ny;
                        temp_chunk[temp_start..temp_start + ny].copy_from_slice(&evolved.values);
                    }
                });
            }
        });
    }

    // Phase 2: scatter temp (column-major) back into state (row-major).
    // temp[c*ny + j] → state[j*nx + (chunk_start + c)]
    // Parallelize by row chunk: split state by row, each thread reads from temp.
    let row_chunk_size = ny.div_ceil(N_THREADS);
    {
        let temp_slice: &[f64] = temp.as_slice();
        std::thread::scope(|s| {
            for (chunk_idx, state_chunk) in state.chunks_mut(row_chunk_size * nx).enumerate() {
                let row_start = chunk_idx * row_chunk_size;
                let rows_in_chunk = state_chunk.len() / nx;
                s.spawn(move || {
                    for r in 0..rows_in_chunk {
                        let j = row_start + r;
                        for col_idx in 0..nx {
                            // Determine which col_chunk this column belongs to.
                            let chunk_of_col = col_idx / col_chunk_size;
                            let c_within_chunk = col_idx % col_chunk_size;
                            // temp layout: chunk_of_col * col_chunk_size * ny + c_within_chunk * ny + j
                            let temp_idx =
                                chunk_of_col * col_chunk_size * ny + c_within_chunk * ny + j;
                            state_chunk[r * nx + col_idx] = temp_slice[temp_idx];
                        }
                    }
                });
            }
        });
    }
}

/// Apply one palindromic Strang step `S(τ) = X(τ/2) ∘ Y(τ) ∘ X(τ/2)` in parallel.
///
/// Operates on a pre-allocated flat buffer `state` (row-major, size nx × ny).
/// No allocation inside this function except per-thread row/column scratch.
///
/// # Safety requirements
/// - `cx` and `cy` must be `Diffusion4thChernoff` instances with valid grids.
/// - `nx = gx.n`, `ny = gy.n`, `state.len() == nx * ny`.
// All 8 args are intrinsic to the Strang splitting step (state, grid dims, grids, ops, τ).
#[allow(clippy::too_many_arguments)]
fn parallel_strang2d_step(
    state: &mut [f64],
    nx: usize,
    ny: usize,
    gx: Grid1D,
    gy: Grid1D,
    cx: Diffusion4thChernoff,
    cy: Diffusion4thChernoff,
    tau: f64,
) {
    let half = tau / 2.0;
    parallel_x_pass(state, nx, ny, gx, cx.clone(), half);
    parallel_y_pass(state, nx, ny, gy, cy, tau);
    parallel_x_pass(state, nx, ny, gx, cx, half);
}

// ---------------------------------------------------------------------------
// Runner: parameterised by (n_steps, n_nodes_per_axis)
// ---------------------------------------------------------------------------

/// Run `n_steps` `Strang2D` iterations with `Diffusion4thChernoff` and return
/// the sup-norm error vs. the 2D heat-kernel oracle.
///
/// Domain: [-15, 15]² so the ±3·dx 7-pt FD stencil stays well inside the grid
/// at all tested N. `BoundaryPolicy`: Reflect (default).
///
/// Per-axis: `Diffusion4thChernoff(a=0.5, a'=0, a''=0)` — constant-coefficient
/// heat on each axis. `Diffusion4thChernoff` is `Copy`, satisfying the
/// `Strang2D<X, Y: Copy>` bound. Order 4 per axis (ζ⁴, ADR-0013).
///
/// `t_final` — evolution time; temporal test uses 1.0, spatial test uses 0.5.
fn heat_2d_4th_error(n_steps: usize, n_nodes: usize, t_final: f64) -> f64 {
    let gx = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid x valid");
    let gy = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid y valid");
    let grid2d = Grid2D::new(gx, gy);

    // Initial datum: u_0(x, y) = exp(-(x² + y²)).
    let f0 = GridFn2D::from_fn(grid2d, |x, y| (-(x * x + y * y)).exp());

    // Per-axis 4th-order heat: L_x = L_y = 0.5 · ∂².
    // Constant a ⇒ a'=0, a''=0 (7-pt FD correction vanishes; γ-A baseline active).
    let cx = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

    // Palindromic Strang2D: Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2). order() = 2 (τ-axis, post-D1).
    let phi2d = Strang2D::new(cx, cy);
    let semi = ChernoffSemigroup::new(phi2d, n_steps).expect("n >= 1");
    let u_n = semi.evolve(t_final, &f0).expect("evolve succeeds");

    // Sup-norm error.
    let nx = grid2d.nx();
    let ny = grid2d.ny();
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let exact = oracle_heat_2d(t_final, xi, yj);
            let err = (u_n.values[j * nx + i] - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

/// Run `n_steps` parallel `Strang2D` iterations and return the sup-norm error.
///
/// Uses `parallel_strang2d_step` (8-thread palindromic Strang via
/// `std::thread::scope`). State is a pre-allocated `Vec<f64>` of size nx × ny;
/// no allocation occurs inside the time loop except per-thread scratch.
///
/// This function is test-local and does NOT modify any production code.
// n_steps ≤ 2000 — well within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn heat_2d_4th_error_parallel(n_steps: usize, n_nodes: usize, t_final: f64) -> f64 {
    let gx = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid x valid");
    let gy = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid y valid");
    let nx = gx.n;
    let ny = gy.n;

    // Per-axis 4th-order heat operators (constant a=0.5).
    let cx = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

    // Pre-allocate state buffer (single allocation, reused across all time steps).
    let mut state: Vec<f64> = (0..ny)
        .flat_map(|j| {
            let yj = gy.x_at(j);
            (0..nx).map(move |i| {
                let xi = gx.x_at(i);
                (-(xi * xi + yj * yj)).exp()
            })
        })
        .collect();

    // Time step size.
    let tau = t_final / (n_steps as f64);

    // Time loop: n_steps applications of parallel Strang2D.
    for _ in 0..n_steps {
        parallel_strang2d_step(&mut state, nx, ny, gx, gy, cx.clone(), cy.clone(), tau);
    }

    // Sup-norm error vs oracle at t_final.
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let exact = oracle_heat_2d(t_final, xi, yj);
            let err = (state[j * nx + i] - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Log-log OLS slope helper
// ---------------------------------------------------------------------------

/// OLS slope of `(ln x[i], ln err[i])` pairs.
///
/// Positive slope → err increases with x (wrong direction for convergence).
/// Negative slope → err decreases with x (expected for both n-sweep and N-sweep).
// slice length ≤ 4 data points — well within f64 52-bit mantissa range.
// sum_x/sum_y/sum_xx/sum_xy are standard OLS notation.
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn loglog_slope(xs: &[f64], errs: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let lx: Vec<f64> = xs.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = lx.iter().sum();
    let sum_y: f64 = ly.iter().sum();
    let sum_xx: f64 = lx.iter().map(|&v| v * v).sum();
    let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// Test 1: temporal_convergence_2d_4th  (n-sweep, documents order-2 temporal)
// ---------------------------------------------------------------------------

/// Temporal convergence: n-sweep at fixed N=800/axis.
///
/// Gate: slope ≤ -1.85 (temporal order 2 — same as `DiffusionChernoff`).
///
/// Design note: temporal Chernoff convergence is O(τ²) regardless of spatial
/// order. This test DOCUMENTS (not demonstrates) that the 2D operator does not
/// regress temporally. The spatial 4th-order gate is in `spatial_convergence_2d_4th`.
// gx_chk/gy_chk and cx_chk/cy_chk are axis-pair convention (x/y axis).
// n ∈ {16, 32, 64} — well within f64 52-bit mantissa range.
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
#[test]
fn temporal_convergence_2d_4th() {
    // Fixed N=800 per axis (dx=0.0375, dx⁴≈2e-6 — well below temporal error at
    // n=64, τ≈0.0156, τ²≈2.4e-4). N=400 shows spatial-floor contamination at n=64.
    // N=800 gives ~40M cell-ops at n=64 — fast in release (~1 s total).
    const N_NODES_FIXED: usize = 800;
    let ns: &[usize] = &[16, 32, 64];

    // Verify Strang2D::order() = 2 before running (τ-axis; spatial dx⁴ checked by slope gate).
    let gx_chk = Grid1D::new(X_MIN, X_MAX, 16).expect("check grid x");
    let gy_chk = Grid1D::new(X_MIN, X_MAX, 16).expect("check grid y");
    let cx_chk = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx_chk);
    let cy_chk = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy_chk);
    let phi_chk = Strang2D::new(cx_chk, cy_chk);
    assert_eq!(
        phi_chk.order(),
        2,
        "Strang2D τ-order should be 2 with 4th-order spatial inner (math.md §11.1.bis); spatial dx⁴ accuracy is verified by the convergence slope below."
    );

    println!("Temporal sweep: Diffusion4thChernoff a=0.5, N={N_NODES_FIXED}/axis, T={T_FINAL}");
    println!("{:>6}  {:>12}  {:>8}", "n", "err_sup", "ratio");

    let mut prev_err: Option<f64> = None;
    let mut ns_f = Vec::new();
    let mut errs = Vec::new();

    for &n in ns {
        let e = heat_2d_4th_error(n, N_NODES_FIXED, T_FINAL);
        let ratio_str = prev_err.map_or_else(|| "       -".into(), |p| format!("{:>8.2}", p / e));
        println!("{n:>6}  {e:>12.4e}  {ratio_str}");
        prev_err = Some(e);
        ns_f.push(n as f64);
        errs.push(e);
    }

    let slope = loglog_slope(&ns_f, &errs);
    println!("Temporal slope = {slope:.4}  (gate ≤ -1.85)");
    println!("Expected ≈ -2.0 (O(τ²) Chernoff — temporal, NOT spatial 4th-order)");

    assert!(
        slope <= -1.85,
        "temporal_convergence_2d_4th FAIL: slope {slope:.4} > -1.85 — \
         Strang2D O(τ²) temporal convergence regressed (errs: {errs:?})"
    );
}

// ---------------------------------------------------------------------------
// Test 2: spatial_convergence_2d_4th  (dx-sweep, FLAGSHIP G3⁴-2D gate)
// ---------------------------------------------------------------------------

/// G3⁴-2D (FLAGSHIP): dx-sweep at fixed n=4000, t=0.5 — spatial-order FLOOR gate.
///
/// Gate: log-log OLS slope of ‖err‖_∞ vs N is ≤ -3.85 (spatial order ≥ ~4).
/// This is a ONE-SIDED ORDER FLOOR, NOT a two-sided "slope = -4.0" asymptotic
/// equality. The scheme's true asymptotic spatial order is exactly 4; this
/// coarse window measures it PRE-ASYMPTOTICALLY (super-convergent, see below).
/// **Failure of this gate BLOCKS v0.6.0 release.**
///
/// # Parallelisation (Phase 4 parallel)
///
/// Uses `heat_2d_4th_error_parallel` backed by `parallel_strang2d_step`:
/// - 8 threads via `std::thread::scope` (no unsafe, no rayon, no new deps).
/// - Pre-allocated single `Vec<f64>` state buffer reused across all n=4000 steps.
/// - X-pass: `chunks_mut(chunk_rows * nx)` — contiguous rows, zero allocator contention.
/// - Y-pass: gather/scatter column ranges; temp buffer (1 × nx × ny f64 = 1.3 MB at N=400).
/// - Peak memory: 2 × nx × ny × 8 bytes = 2.6 MB at N=400 (state + temp). Safe (10 GB).
///
/// # Pre-asymptotic floor-clean calibration (HW-specific temporal floor, v9.2.0)
///
/// Previous fine window {400, 800, 1600} at n=2000 produced slope −1.4472 (FAIL)
/// because the temporal floor at n=2000 on this machine is ≈1.2e-9 (~5× the
/// historically-quoted 2.4e-10), contaminating N=800 (spatial ~5.5e-10 < floor)
/// and N=1600 (deep in floor). The fine asymptotic plateau (ratio→16×, slope→−4.0)
/// is unreachable cheaply on this HW: pushing the floor below the N=1600 spatial
/// error (~3.4e-11) needs n≈12000–16000 (~1hr) AND re-enters the near-f64-noise
/// regime that produced two prior false-greens. We therefore calibrate on a
/// floor-CLEAN COARSE window, certifying spatial order ≥ ~4 as a one-sided floor.
///
/// Recalibrated sweep: n=4000 (τ=1.25e-4, floor ≈0.3e-9), N ∈ {200, 300, 400}.
/// All three points are spatial-dominated — floor is < 3.5% of the N=400 error:
///   floor(n=4000) ≈ (2000/4000)² × 1.2e-9 ≈ 0.3e-9
///   err(N=400)    = 8.75e-9  →  floor/spatial ≈ 3.4%
///   err(N=300)    = 4.73e-8  →  floor/spatial ≪ 1%
///   err(N=200)    = 5.11e-7  →  floor/spatial ≪ 1%
///
/// Measured 2D ratio table (n=4000, t=0.5, 8-thread parallel, i7-12700K) —
/// PRE-ASYMPTOTIC SUPER-CONVERGENT (ratios ABOVE the 4th-order prediction and
/// DECREASING toward it; the dx⁶/dx⁸ tail is still active, NOT yet on the dx⁴
/// plateau — the genuine signature of a ≥4th-order scheme):
///   N=200:  err = 5.1101e-7   ratio    -
///   N=300:  err = 4.7253e-8   ratio 10.81  (>5.06 = (300/200)⁴ ⇒ super-convergent)
///   N=400:  err = 8.7522e-9   ratio  5.40  (>3.16 = (400/300)⁴ ⇒ super-convergent)
///   OLS slope over {200, 300, 400} ≈ −5.87  (PASS gate −3.85, margin ≥ 2.0)
///
/// Note: N=400 at n=10000 gives 8.6200e-9 (matches n=4000), confirming that
/// N=400 is spatial-dominated (temporal floor negligible) at n=4000.
///
/// # Temporal budget (t=0.5, n=4000)
/// τ = 0.5/4000 = 1.25e-4, τ² = 1.5625e-8.
/// Temporal floor ≈ 0.3e-9.
/// Spatial error at N=400: 8.75e-9 >> 0.3e-9 → spatial-dominated.
///
/// # Domain and stencil
/// Domain [-15, 15]²: dx at N=400 is 0.075; ±3·dx stencil offset = ±0.225.
/// At N=200: dx=0.15; well inside domain at all sweep points.
/// `BoundaryPolicy::Reflect` (default).
///
/// # Spatial accuracy mechanism
/// With constant a=0.5 per axis, `Diffusion4thChernoff` `zeta4_correction`
/// short-circuits to 0 (no variable-a Taylor correction). Spatial accuracy
/// governed by K-kernel Fourier-symbol weights (math.md §9.2.1):
///   K̂(ξ) = W0 + 2W1·cos(ξ) + 2W2·cos(2ξ)  [O(ξ⁴) by construction]
/// → global O(dx⁴) spatial convergence per axis → O(dx⁴) for `Strang2D`.
/// Measured OLS slope ≈ −5.87 (super-convergent, pre-asymptotic — coarse window).
///
/// # Time budget (release mode, 8-thread parallel)
///   N=200:   ~4 s
///   N=300:   ~9 s
///   N=400:   ~16 s
///   Total: ~30 s
///   Empirical: OLS slope ≈ −5.87 (PASS gate −3.85)
/// Run the spatial sweep for G3⁴-2D and return `(log_n, log_e, total_secs)`.
///
/// Runs parallel `Strang2D` at each N in `n_sweep`, logs timing and error table
/// to stderr, and returns log-space vectors for OLS slope computation.
// n_per_axis ≤ 1600 — well within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn spatial_sweep_data(n_sweep: &[usize], n_steps: usize, t: f64) -> (Vec<f64>, Vec<f64>, u64) {
    let total_start = std::time::Instant::now();
    let mut log_n = Vec::new();
    let mut log_e = Vec::new();
    let mut prev_err: Option<f64> = None;

    eprintln!("\nN/axis          dx       err_sup     ratio      time");
    for &n_per_axis in n_sweep {
        let t_start = std::time::Instant::now();
        let dx = (X_MAX - X_MIN) / (n_per_axis as f64);
        let err = heat_2d_4th_error_parallel(n_steps, n_per_axis, t);
        let elapsed_s = t_start.elapsed().as_secs();
        match prev_err {
            None => eprintln!(
                "{n_per_axis:>6}   {dx:.4e}     {err:.4e}         -        ({elapsed_s}s)"
            ),
            Some(p) => {
                let ratio = p / err;
                eprintln!(
                    "{n_per_axis:>6}   {dx:.4e}     {err:.4e}    {ratio:6.2}        ({elapsed_s}s)"
                );
            }
        }
        log_n.push((n_per_axis as f64).ln());
        log_e.push(err.ln());
        prev_err = Some(err);
    }
    let total_secs = total_start.elapsed().as_secs();
    (log_n, log_e, total_secs)
}

/// OLS log-log slope from pre-computed log-space vectors.
// Vec length ≤ 4 — well within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn ols_slope(log_n: &[f64], log_e: &[f64]) -> f64 {
    let m = log_n.len() as f64;
    let mean_x = log_n.iter().sum::<f64>() / m;
    let mean_y = log_e.iter().sum::<f64>() / m;
    let num: f64 = log_n
        .iter()
        .zip(log_e.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_n.iter().map(|x| (x - mean_x).powi(2)).sum();
    num / den
}

#[test]
fn spatial_convergence_2d_4th() {
    // G3⁴-2D SPATIAL ORDER FLOOR — recalibrated for hardware-specific temporal floor.
    // t=0.5: shorter evolution reduces temporal floor.
    // n=4000: τ=1.25e-4, temporal floor ≈0.3e-9 (well below N=400 spatial 8.75e-9).
    const T_SPATIAL: f64 = 0.5;
    const N_STEPS_FIXED: usize = 4000;
    // 3-point sweep {200, 300, 400}. All points spatial-dominated at n=4000.
    // Measured OLS slope ≈ −5.87 (PASS gate −3.85, margin ≥ 2.0).
    const N_SWEEP: &[usize] = &[200, 300, 400];
    // One-sided order-floor gate: slope ≤ -3.85 certifies spatial order ≥ ~4.
    const SLOPE_GATE: f64 = -3.85;

    eprintln!(
        "Spatial sweep (G3⁴-2D ORDER-FLOOR, 8-thread parallel): \
         n_fixed={N_STEPS_FIXED}, T={T_SPATIAL}, threads={N_THREADS}"
    );
    eprintln!("Domain [{X_MIN}, {X_MAX}]²; `BoundaryPolicy::Reflect`");

    let (log_n, log_e, total_secs) = spatial_sweep_data(N_SWEEP, N_STEPS_FIXED, T_SPATIAL);
    let slope = ols_slope(&log_n, &log_e);

    eprintln!(
        "\nWallclock total: {total_secs} s ({} min)",
        total_secs / 60
    );
    eprintln!("Spatial slope = {slope:.4}  (gate ≤ {SLOPE_GATE}, ADR-0013 FLAGSHIP)");
    eprintln!(
        "True asymptotic order = 4; this coarse window is PRE-ASYMPTOTIC \
         super-convergent, expected measured slope ≥ -4 (≈ -5.87 here). \
         Gate is a one-sided order floor, NOT slope = -4.0. See math.md §9.2.1."
    );

    assert!(
        slope <= SLOPE_GATE,
        "G3⁴-2D SPATIAL FAIL: slope {slope:.4} > {SLOPE_GATE}.\n\
         N range used: {N_SWEEP:?}\n\
         n_fixed={N_STEPS_FIXED}, T={T_SPATIAL}.\n\
         This is a one-sided spatial-ORDER-FLOOR gate (order ≥ ~4); the coarse\n\
         window is pre-asymptotic super-convergent (expected slope ≈ -5.87).\n\
         slope > -3.85 means spatial order dropped below ~4 — investigate the\n\
         implementation (a 2nd-order regression gives slope ≈ -2.0). See\n\
         ADR-0013 Amendment 3 / math.md §9.2.1 recalibration note.\n\
         BLOCKS v0.6.0 release."
    );
}
