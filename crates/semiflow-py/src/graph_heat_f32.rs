//! f32 dispatch helpers for graph/diffusion kernels (Issue #3, ADR-0115).
//!
//! Contains the pure-Rust compute functions called from within `py.detach`
//! for the f32 path.  No Python types cross this boundary.
//!
//! ## Pattern
//!
//! 1. Convert `Arc<Graph<f64>>` to `Arc<Graph<f32>>` by casting edge weights.
//! 2. Assemble the f32 core kernel.
//! 3. Run the evolution.
//! 4. Cast the `Vec<f32>` result back to `Vec<f64>` for numpy.
//!
//! The round-trip cast (f64 → f32 → f64) is the intended boundary loss; the
//! caller compares against `rtol=1e-5` in tests (see `test_dtype_f32.py`).

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use semiflow_core::{
    DiffusionChernoff, Evolver, Graph, GraphHeatChernoff, GraphSignal, Grid1D, GridFn1D, Laplacian,
    LaplacianAtTime, MagnusGraphHeatChernoff, SemiflowError, ScratchPool, VarCoefGraphHeatChernoff,
};

use crate::dtype_dispatch::{cast_f32_to_f64, cast_f64_to_f32};

// ---------------------------------------------------------------------------
// Graph type-conversion: Graph<f64> → Graph<f32>
// ---------------------------------------------------------------------------

/// Convert a `Laplacian<f64>` into an `Arc<Laplacian<f32>>` for use in
/// `LaplacianAtTime<f32>` callbacks (Magnus f32 path).
///
/// In a combinatorial Laplacian `L = D − W`, off-diagonal entries are `−w_{ij}`
/// (negative for edges).  We extract upper-triangle edges as `(u, v, -val)`.
pub(crate) fn build_lap_f32_from_lap_f64(
    lap64: &Laplacian<f64>,
) -> Arc<semiflow_core::Laplacian<f32>> {
    let n = lap64.n_nodes();
    let row_ptr = lap64.row_ptr();
    let ci = lap64.col_idx();
    let vf = lap64.vals();
    let edges_f64: Vec<(u32, u32, f64)> = row_ptr
        .windows(2)
        .enumerate()
        .flat_map(|(u, w)| {
            #[allow(clippy::cast_possible_truncation)]
            let u32 = u as u32;
            ci[w[0]..w[1]]
                .iter()
                .zip(vf[w[0]..w[1]].iter())
                .filter_map(move |(&c, &v)| {
                    if u32 < c && v < 0.0 {
                        Some((u32, c, -v))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let g64 = if edges_f64.is_empty() {
        Graph::<f64>::path(n.max(1))
    } else {
        Graph::<f64>::from_edges(n, edges_f64).unwrap_or_else(|_| Graph::<f64>::path(n.max(1)))
    };
    let g32 = graph_f64_to_f32(&g64).unwrap_or_else(|_| Graph::<f32>::path(n.max(1)));
    Arc::new(Laplacian::<f32>::assemble_combinatorial(&g32))
}

/// Re-assemble a `Graph<f32>` by casting edge weights from a `Graph<f64>`.
///
/// The topology (`row_ptr`, `col_idx`) is identical; only the weight type changes.
/// Returns a fresh allocation; called once per kernel construction.
pub(crate) fn graph_f64_to_f32(g: &Graph<f64>) -> Result<Graph<f32>, SemiflowError> {
    let n = g.n_nodes();
    let row_ptr = g.row_ptr();
    let col_idx = g.col_idx();
    let vals_f64 = g.vals();
    let edges = row_ptr.windows(2).enumerate().flat_map(|(u, w)| {
        let start = w[0];
        let end = w[1];
        #[allow(clippy::cast_possible_truncation)]
        let u32 = u as u32;
        col_idx[start..end]
            .iter()
            .zip(vals_f64[start..end].iter())
            .map(move |(&c, &v)| (u32, c, v as f32))
            .collect::<Vec<_>>()
    });
    // Only store upper-triangle to satisfy from_edges symmetry requirement.
    let directed: Vec<(u32, u32, f32)> = edges.filter(|&(u, v, _)| u < v).collect();
    Graph::<f32>::from_edges(n, directed)
}

/// Assemble a `Laplacian<f32>` combinatorial from a `Graph<f32>`.
fn lap_f32(g: &Arc<Graph<f32>>) -> Arc<Laplacian<f32>> {
    Arc::new(Laplacian::assemble_combinatorial(g))
}

// ---------------------------------------------------------------------------
// GraphHeat f32 — order-1 graph heat
// ---------------------------------------------------------------------------

/// Order-1 graph heat evolution on f32.  No GIL held.
pub(crate) fn compute_graph_heat_f32(
    graph_f64: Arc<Graph<f64>>,
    input_f64: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    let g32 = Arc::new(graph_f64_to_f32(&graph_f64)?);
    let lap32 = lap_f32(&g32);
    let chernoff = GraphHeatChernoff::<f32>::new(lap32);
    let ev = Evolver::<GraphHeatChernoff<f32>, f32>::new(chernoff, n_steps)?;
    let input_f32 = cast_f64_to_f32(input_f64);
    let f0 = GraphSignal::from_fn(g32, |i| input_f32[i as usize]);
    let result = ev.evolve(t_final as f32, &f0)?;
    Ok(cast_f32_to_f64(result.values()))
}

// ---------------------------------------------------------------------------
// MagnusGraphHeat f32 — Magnus K=4 heat with time-varying weights
// ---------------------------------------------------------------------------

/// Magnus K=4 graph heat evolution on f32.  No GIL held outside callbacks.
///
/// The `lap_at_t` closure re-acquires the GIL internally (ADR-0059 R2 pattern).
pub(crate) fn compute_magnus_graph_f32(
    graph_f64: Arc<Graph<f64>>,
    lap_at_t_f32: LaplacianAtTime<f32>,
    rho_bar_max: f64,
    convergence_check: bool,
    input_f64: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    let g32 = Arc::new(graph_f64_to_f32(&graph_f64)?);
    let mghc = MagnusGraphHeatChernoff::<f32>::new(
        Arc::clone(&g32),
        lap_at_t_f32,
        rho_bar_max as f32,
        convergence_check,
    )?;

    #[allow(clippy::cast_possible_truncation)]
    let tau = (t_final / n_steps as f64) as f32;
    let input_f32 = cast_f64_to_f32(input_f64);
    let mut state = GraphSignal::from_fn(Arc::clone(&g32), |i| input_f32[i as usize]);
    let mut scratch = ScratchPool::<f32>::new();

    for step in 0..n_steps {
        #[allow(clippy::cast_possible_truncation)]
        let t_start = (step as f64 * (t_final / n_steps as f64)) as f32;
        let mut next = state.clone();
        mghc.apply_into_at(t_start, tau, &state, &mut next, &mut scratch)?;
        state = next;
    }
    Ok(cast_f32_to_f64(state.values()))
}

// ---------------------------------------------------------------------------
// VarCoefGraphHeat f32 — variable-coefficient order-2
// ---------------------------------------------------------------------------

/// Variable-coefficient graph heat evolution on f32.  No GIL held.
pub(crate) fn compute_var_coef_f32(
    graph_f64: Arc<Graph<f64>>,
    a_f64: &[f64],
    rho_bar: f64,
    input_f64: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    let g32 = Arc::new(graph_f64_to_f32(&graph_f64)?);
    let a_f32: Vec<f32> = cast_f64_to_f32(a_f64);
    let g32_2 = Arc::clone(&g32);
    let chernoff = VarCoefGraphHeatChernoff::<f32>::new(Arc::clone(&g32), a_f32, rho_bar as f32)?;
    let ev = Evolver::<VarCoefGraphHeatChernoff<f32>, f32>::new(chernoff, n_steps)?;
    let input_f32 = cast_f64_to_f32(input_f64);
    let f0 = GraphSignal::from_fn(g32_2, |i| input_f32[i as usize]);
    let result = ev.evolve(t_final as f32, &f0)?;
    Ok(cast_f32_to_f64(result.values()))
}

// ---------------------------------------------------------------------------
// Heat1D f32 — unit diffusion
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_f32(_: f32) -> f32 {
    1.0_f32
}

extern "Rust" fn zero_deriv_f32(_: f32) -> f32 {
    0.0_f32
}

/// 1-D heat evolution on f32, unit diffusion `a = 1`.  No GIL held.
///
/// Uses `DiffusionChernoff::<f32>::apply_f` (direct inherent method, bypasses
/// the `ChernoffFunction` dispatch machinery).  Although
/// `ChernoffFunction<f32>` IS now first-class for leaf kernels (ADR-0175,
/// issue #5), `apply_f` is retained here to avoid an unnecessary intermediate
/// `GridFn1D` allocation on the hot inner loop of the f32 graph-heat path.
pub(crate) fn compute_heat1d_f32(
    xmin: f64,
    xmax: f64,
    n: usize,
    input_f64: &[f64],
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    if n_steps == 0 {
        return Ok(input_f64.to_vec());
    }
    let grid = Grid1D::<f32>::new_generic(xmin as f32, xmax as f32, n)?;
    // a_norm_bound takes f64 as per the constructor signature
    let chernoff =
        DiffusionChernoff::<f32>::new(unit_a_f32, zero_deriv_f32, zero_deriv_f32, 1.0_f64, grid);
    let tau = (t / n_steps as f64) as f32;
    let input_f32 = cast_f64_to_f32(input_f64);
    let mut state = GridFn1D::<f32>::new_generic(grid, input_f32)?;
    for _ in 0..n_steps {
        state = chernoff.apply_f(tau, &state)?;
    }
    Ok(cast_f32_to_f64(&state.values))
}
