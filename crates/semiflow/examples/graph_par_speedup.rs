//! Multicore speedup probe for graph-batched channel-parallel (ADR-0184 D3).
//!
//! Measures wall-clock time for `evolve_batched_magnus` with C=16 channels
//! on an Erdős–Rényi random graph (N=256), comparing parallel vs serial build.
//!
//! Run with parallel:
//! ```sh
//! cargo run -p semiflow --example graph_par_speedup --features parallel,simd --release
//! ```
//!
//! Run serial reference:
//! ```sh
//! cargo run -p semiflow --example graph_par_speedup --release
//! ```

#![allow(missing_docs)]

use std::{sync::Arc, time::Instant};

use semiflow::{
    graph_batched::evolve_batched_magnus, Graph, Laplacian, LaplacianAtTime,
    MagnusGraphHeatChernoff,
};

const N: usize = 256;
const N_COLS: usize = 16;
const N_STEPS: usize = 50;
const T_FINAL: f64 = 0.5;
const RHO: f64 = 64.0;
const REPS: usize = 5;

fn main() {
    let g = Arc::new(Graph::<f64>::path(N));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let lap_fn: LaplacianAtTime<f64> = Box::new(move |_t| Arc::clone(&lap));
    let mc = MagnusGraphHeatChernoff::new(Arc::clone(&g), lap_fn, RHO, false).unwrap();

    let src: Vec<f64> = (0..N * N_COLS)
        .map(|i| if i % N == 0 { 1.0 } else { 0.0 })
        .collect();
    let mut dst = vec![0.0_f64; N * N_COLS];

    // Warm-up
    evolve_batched_magnus(&mc, T_FINAL, N_STEPS, &src, &mut dst).unwrap();

    // Timed reps
    let mut total_ns = 0u128;
    for _ in 0..REPS {
        let t0 = Instant::now();
        evolve_batched_magnus(&mc, T_FINAL, N_STEPS, &src, &mut dst).unwrap();
        total_ns += t0.elapsed().as_nanos();
    }
    let avg_ms = total_ns as f64 / REPS as f64 / 1_000_000.0;

    let feature = if cfg!(feature = "parallel") { "parallel" } else { "serial" };
    println!(
        "graph_par_speedup: N={N} C={N_COLS} n_steps={N_STEPS} build={feature} avg={avg_ms:.2}ms over {REPS} reps"
    );
    println!("  dst[0]={:.6e} (non-zero → real work done)", dst[0]);
}
