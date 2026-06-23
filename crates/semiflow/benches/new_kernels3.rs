//! Criterion benchmark: v3.1–v4.8 kernel families (subordinated, Kolmogorov,
//! quantum-graph, matrix-diffusion, anisotropic-shift, Schrödinger).
//!
//! Run: `cargo bench -p semiflow-core --bench new_kernels3 --features simd`

#![allow(missing_docs)]
// Integration test/bench: allows for numerical patterns.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex;
use semiflow_core::{
    chernoff::ChernoffFunction,
    drift_reaction::DriftReactionChernoff,
    grid_nd::{GridFnND, GridND},
    hormander::{HypoellipticChernoff, KolmogorovPhaseSpace},
    matrix_system::{MatrixDiffusionChernoff, MatrixGridFn1D},
    schrodinger_complex::{GridFnComplex1D, SchrödingerChernoffComplex},
    subordinated::{
        GammaSubordinator, InverseGaussianSubordinator, StableSubordinator, SubordinatedChernoff,
    },
    AnisotropicShiftChernoffND, Grid1D, Grid2D, GridFn1D, GridFn2D, QuantumGraph,
    QuantumGraphHeatChernoff, QuantumGraphSignal, ScratchPool, SquareMatrix,
};

// Bench configuration

/// Time step for subordinated and Kolmogorov benchmarks.
const TAU: f64 = 0.005;
/// Time step for matrix-diffusion benchmarks (unconditionally stable).
const TAU_MATRIX: f64 = 0.01;
/// Time step for quantum-graph benchmarks.
const TAU_GRAPH: f64 = 0.001;
/// Time step for Schrödinger benchmarks.
const TAU_SCHROD: f64 = 0.01;
/// Time step for anisotropic shift (small for GH node displacement stability).
const TAU_ANISO: f64 = 1e-4;

const WARM_UP_SECS: u64 = 3;
const MEASUREMENT_SECS: u64 = 10;

// Group 1: SubordinatedChernoff (three backends, N=256 and N=1024)

fn add_stable_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ic: &GridFn1D<f64>,
    make_base: &impl Fn() -> DriftReactionChernoff<f64>,
) {
    let sub = StableSubordinator::new(0.5_f64).expect("stable sub");
    let kernel = SubordinatedChernoff::new(make_base(), sub);
    let mut dst = ic.clone();
    let mut scratch = ScratchPool::new();
    group.bench_with_input(BenchmarkId::new("stable_alpha0.5", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(black_box(TAU), black_box(ic), &mut dst, &mut scratch)
                .expect("ok");
        });
    });
}

fn bench_subordinated(c: &mut Criterion, n: usize) {
    let grid = Grid1D::new(0.0_f64, 1.0, n).expect("grid");
    let make_base = || DriftReactionChernoff::new(|_| 0.0_f64, |_| -1.0_f64, 1.0, grid);
    let ic = GridFn1D::from_fn(grid, |_| 1.0_f64);
    let mut group = c.benchmark_group(format!("new_kernels3/subordinated/N={n}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n as u64));
    add_stable_bench(&mut group, &ic, &make_base);
    add_gamma_bench(&mut group, &ic, &make_base);
    add_ig_bench(&mut group, &ic, &make_base);
    group.finish();
}

fn add_gamma_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ic: &GridFn1D<f64>,
    make_base: &impl Fn() -> DriftReactionChernoff<f64>,
) {
    let sub = GammaSubordinator::new(1.0_f64).expect("gamma sub");
    let kernel = SubordinatedChernoff::new(make_base(), sub);
    let mut dst = ic.clone();
    let mut scratch = ScratchPool::new();
    group.bench_with_input(BenchmarkId::new("gamma_c1.0", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(black_box(TAU), black_box(ic), &mut dst, &mut scratch)
                .expect("ok");
        });
    });
}

fn add_ig_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ic: &GridFn1D<f64>,
    make_base: &impl Fn() -> DriftReactionChernoff<f64>,
) {
    let sub = InverseGaussianSubordinator::new(1.0_f64).expect("ig sub");
    let kernel = SubordinatedChernoff::new(make_base(), sub);
    let mut dst = ic.clone();
    let mut scratch = ScratchPool::new();
    group.bench_with_input(BenchmarkId::new("inv_gaussian_c1.0", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(black_box(TAU), black_box(ic), &mut dst, &mut scratch)
                .expect("ok");
        });
    });
}

fn bench_subordinated_256(c: &mut Criterion) {
    bench_subordinated(c, 256);
}
fn bench_subordinated_1024(c: &mut Criterion) {
    bench_subordinated(c, 1024);
}

// Group 2: HypoellipticChernoff (Kolmogorov) on small 2D phase-space grid

fn bench_hypoelliptic_kolmogorov(c: &mut Criterion) {
    // Small grid for fast iteration: 32×32 phase-space nodes.
    // The Strang-Hörmander decomposition cost is dominated by the DiffusionChernoff
    // inner step on the velocity axis plus one ShiftChernoff1D on the x-axis.
    let n_axis: usize = 32;
    let n_nodes = n_axis * n_axis;

    let gx = Grid1D::new(-6.0_f64, 6.0, n_axis).expect("x grid");
    let gv = Grid1D::new(-6.0_f64, 6.0, n_axis).expect("v grid");
    let grid2d = Grid2D::new(gx, gv);

    let x0 = Box::new(KolmogorovPhaseSpace::x0_drift());
    let x1 = Box::new(KolmogorovPhaseSpace::x1_diffusion());
    let kernel = HypoellipticChernoff::<f64, 2, 1>::new(x0, [x1]).expect("Kolmogorov Hormander ok");

    let ic = GridFn2D::from_fn(grid2d, |x, v| (-(x * x + v * v)).exp());
    let mut dst = GridFn2D {
        values: vec![0.0_f64; n_nodes],
        grid: grid2d,
    };
    let mut scratch = ScratchPool::new();

    let mut group = c.benchmark_group(format!("new_kernels3/kolmogorov/N={n_axis}x{n_axis}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n_nodes as u64));

    group.bench_with_input(
        BenchmarkId::new("hypoelliptic_kolmogorov", ""),
        &(),
        |b, ()| {
            b.iter(|| {
                kernel
                    .apply_into(black_box(TAU), black_box(&ic), &mut dst, &mut scratch)
                    .expect("ok");
            });
        },
    );

    group.finish();
}

// Group 3: QuantumGraphHeatChernoff (path graph, two sizes)

fn make_quantum_graph_path(
    n_edges: usize,
    n_grid: usize,
) -> (QuantumGraphHeatChernoff<f64>, QuantumGraphSignal<f64>) {
    let graph = QuantumGraph::path(n_edges, 1.0_f64, n_grid).expect("path graph");
    let kernel = QuantumGraphHeatChernoff::new(graph.clone()).expect("quantum graph kernel");
    let ic = QuantumGraphSignal::from_fn(&graph, |e, x| {
        let phi = (e as f64) * 0.5;
        (-(x - 0.5).powi(2)).exp() * phi.cos()
    });
    (kernel, ic)
}

fn bench_quantum_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_kernels3/quantum_graph_per_step");
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    add_qgraph_bench(&mut group, 4, 500, "path4_n500");
    add_qgraph_bench(&mut group, 16, 500, "path16_n500");
    group.finish();
}

fn add_qgraph_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n_edges: usize,
    n_grid: usize,
    label: &str,
) {
    let (kernel, ic) = make_quantum_graph_path(n_edges, n_grid);
    let mut dst = ic.clone();
    let mut scratch = ScratchPool::new();
    group.throughput(Throughput::Elements((n_edges * n_grid) as u64));
    group.bench_with_input(BenchmarkId::new(label, ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(black_box(TAU_GRAPH), black_box(&ic), &mut dst, &mut scratch)
                .expect("ok");
        });
    });
}

// Group 4: MatrixDiffusionChernoff M ∈ {2, 3, 4}

fn make_matrix_kernel<const M: usize>(n: usize) -> MatrixDiffusionChernoff<f64, M> {
    let grid = Grid1D::new(-5.0_f64, 5.0, n).expect("grid");
    MatrixDiffusionChernoff::<f64, M>::new(
        // Identity diffusion tensor.
        |_, a| {
            for i in 0..M {
                for j in 0..M {
                    a[i][j] = if i == j { 1.0 } else { 0.0 };
                }
            }
        },
        // Zero drift.
        |_, b| {
            for row in b.iter_mut() {
                for v in row.iter_mut() {
                    *v = 0.0;
                }
            }
        },
        // Small reaction: c_ii = -0.1.
        |_, c| {
            for i in 0..M {
                for j in 0..M {
                    c[i][j] = if i == j { -0.1 } else { 0.0 };
                }
            }
        },
        grid,
    )
    .expect("matrix kernel")
}

fn add_matrix_m4_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
) {
    let kernel = make_matrix_kernel::<4>(n);
    let ic = MatrixGridFn1D::<f64, 4>::from_fn(Grid1D::new(-5.0_f64, 5.0, n).unwrap(), |x| {
        [
            (-x * x).exp(),
            (-x * x * 0.5).exp(),
            (-x * x * 0.3).exp(),
            (-x * x * 0.2).exp(),
        ]
    });
    let mut dst = MatrixGridFn1D::<f64, 4>::new(Grid1D::new(-5.0_f64, 5.0, n).unwrap());
    let mut scratch = ScratchPool::new();
    group.throughput(Throughput::Elements((n * 4) as u64));
    group.bench_with_input(BenchmarkId::new("M=4", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(
                    black_box(TAU_MATRIX),
                    black_box(&ic),
                    &mut dst,
                    &mut scratch,
                )
                .expect("ok");
        });
    });
}

fn bench_matrix_diffusion(c: &mut Criterion, n: usize) {
    let mut group = c.benchmark_group(format!("new_kernels3/matrix_diffusion/N={n}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    add_matrix_m2_bench(&mut group, n);
    add_matrix_m3_bench(&mut group, n);
    add_matrix_m4_bench(&mut group, n);
    group.finish();
}

fn add_matrix_m2_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
) {
    let kernel = make_matrix_kernel::<2>(n);
    let ic = MatrixGridFn1D::<f64, 2>::from_fn(Grid1D::new(-5.0_f64, 5.0, n).unwrap(), |x| {
        [(-x * x).exp(), (-x * x * 0.5).exp()]
    });
    let mut dst = MatrixGridFn1D::<f64, 2>::new(Grid1D::new(-5.0_f64, 5.0, n).unwrap());
    let mut scratch = ScratchPool::new();
    group.throughput(Throughput::Elements((n * 2) as u64));
    group.bench_with_input(BenchmarkId::new("M=2", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(
                    black_box(TAU_MATRIX),
                    black_box(&ic),
                    &mut dst,
                    &mut scratch,
                )
                .expect("ok");
        });
    });
}

fn add_matrix_m3_bench(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
) {
    let kernel = make_matrix_kernel::<3>(n);
    let ic = MatrixGridFn1D::<f64, 3>::from_fn(Grid1D::new(-5.0_f64, 5.0, n).unwrap(), |x| {
        [(-x * x).exp(), (-x * x * 0.5).exp(), (-x * x * 0.3).exp()]
    });
    let mut dst = MatrixGridFn1D::<f64, 3>::new(Grid1D::new(-5.0_f64, 5.0, n).unwrap());
    let mut scratch = ScratchPool::new();
    group.throughput(Throughput::Elements((n * 3) as u64));
    group.bench_with_input(BenchmarkId::new("M=3", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(
                    black_box(TAU_MATRIX),
                    black_box(&ic),
                    &mut dst,
                    &mut scratch,
                )
                .expect("ok");
        });
    });
}

fn bench_matrix_diffusion_n64(c: &mut Criterion) {
    bench_matrix_diffusion(c, 64);
}
fn bench_matrix_diffusion_n256(c: &mut Criterion) {
    bench_matrix_diffusion(c, 256);
}

// Group 5: AnisotropicShiftChernoffND D ∈ {2, 3}

fn add_aniso_d2_bench(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    const N_AXIS: usize = 8;
    let ax = Grid1D::new(-5.0_f64, 5.0, N_AXIS).expect("ax");
    let grid = GridND::<f64, 2>::new([ax, ax]).expect("grid2");
    let kernel = AnisotropicShiftChernoffND::<f64, 2>::new(
        |x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            let off = 0.25 * (x[0] + x[1]).tanh();
            a.set(0, 1, off);
            a.set(1, 0, off);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x: &[f64; 2]| 0.0_f64,
        grid.clone(),
    )
    .expect("D=2 kernel");
    let ic = GridFnND::from_fn(grid.clone(), |x: &[f64; 2]| {
        (-x[0] * x[0] - x[1] * x[1]).exp()
    });
    let mut dst = GridFnND::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();
    group.throughput(Throughput::Elements((N_AXIS * N_AXIS) as u64));
    group.bench_with_input(BenchmarkId::new("D=2_N=8x8", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(black_box(TAU_ANISO), black_box(&ic), &mut dst, &mut scratch)
                .expect("ok");
        });
    });
}

fn bench_anisotropic_shift(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_kernels3/anisotropic_shift_per_step");
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    add_aniso_d2_bench(&mut group);
    add_aniso_d3_bench(&mut group);
    group.finish();
}

// D = 3, N_AXIS = 5 (minimum: grid.len() >= 5^D = 125; keep grid small).
fn add_aniso_d3_bench(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    const N_AXIS3: usize = 5;
    let ax = Grid1D::new(-3.0_f64, 3.0, N_AXIS3).expect("ax");
    let grid = GridND::<f64, 3>::new([ax, ax, ax]).expect("grid3");
    let kernel = AnisotropicShiftChernoffND::<f64, 3>::new(
        |_x: &[f64; 3], a: &mut SquareMatrix<f64, 3>| {
            for i in 0..3 {
                for j in 0..3 {
                    a.set(i, j, if i == j { 1.0 } else { 0.0 });
                }
            }
        },
        |_x: &[f64; 3], b: &mut [f64; 3]| {
            b[0] = 0.0;
            b[1] = 0.0;
            b[2] = 0.0;
        },
        |_x: &[f64; 3]| 0.0_f64,
        grid.clone(),
    )
    .expect("D=3 kernel");
    let ic = GridFnND::from_fn(grid.clone(), |x: &[f64; 3]| {
        (-x[0] * x[0] - x[1] * x[1] - x[2] * x[2]).exp()
    });
    let mut dst = GridFnND::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();
    group.throughput(Throughput::Elements(N_AXIS3.pow(3) as u64));
    group.bench_with_input(BenchmarkId::new("D=3_N=5x5x5", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(black_box(TAU_ANISO), black_box(&ic), &mut dst, &mut scratch)
                .expect("ok");
        });
    });
}

// Group 6: SchrödingerChernoffComplex N ∈ {256, 1024}

fn make_schrodinger(
    n: usize,
) -> (
    SchrödingerChernoffComplex<Complex<f64>>,
    GridFnComplex1D<Complex<f64>>,
) {
    use std::f64::consts::PI;
    let grid = Grid1D::<f64>::new(-10.0, 10.0, n).expect("grid");
    // Harmonic oscillator V(x) = 0.5 x².
    let kernel = SchrödingerChernoffComplex::<Complex<f64>>::new(grid, |x: f64| 0.5 * x * x)
        .expect("schrodinger kernel");
    // Gaussian wave packet with momentum k₀ = 1.
    let k0 = 1.0_f64;
    let norm = PI.powf(-0.25_f64);
    let ic = GridFnComplex1D::from_fn(grid, |x: f64| {
        let env = norm * (-x * x / 2.0).exp();
        Complex::from_polar(env, k0 * x)
    });
    (kernel, ic)
}

fn bench_schrodinger(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_kernels3/schrodinger_complex_per_step");
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    for &n in &[256_usize, 1024] {
        let (kernel, ic) = make_schrodinger(n);
        let mut dst = ic.clone();
        let mut scratch = ScratchPool::new();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(format!("N={n}"), ""), &(), |b, ()| {
            b.iter(|| {
                kernel
                    .apply_into(
                        black_box(TAU_SCHROD),
                        black_box(&ic),
                        &mut dst,
                        &mut scratch,
                    )
                    .expect("ok");
            });
        });
    }

    group.finish();
}

// Criterion entry points

criterion_group!(
    benches,
    bench_subordinated_256,
    bench_subordinated_1024,
    bench_hypoelliptic_kolmogorov,
    bench_quantum_graph,
    bench_matrix_diffusion_n64,
    bench_matrix_diffusion_n256,
    bench_anisotropic_shift,
    bench_schrodinger,
);
criterion_main!(benches);
