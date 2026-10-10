//! Wall-clock and accuracy table for representative SemiFlow workloads.
//!
//! One row per workload: best-of-`reps` wall time and, where a closed-form
//! oracle exists, the sup-norm error. Used to record before/after numbers in
//! `docs/perf/overhaul-v0_15.md`; it is not a gate (timings depend on the host).
//!
//! ```sh
//! cargo run -p semiflow --release --example perf_suite            # all rows
//! cargo run -p semiflow --release --example perf_suite -- krylov  # rows whose name contains "krylov"
//! ```

#![allow(
    missing_docs,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

use std::time::Instant;

use semiflow::general_operator::GeneralOperator;

use semiflow::{
    graph_expmv_krylov, phi_action, AllenCahn, ChernoffSemigroup, CsrGenerator,
    Diffusion4thChernoff, Diffusion6thChernoff, DiffusionChernoff, Etdrk4, Grid1D, Grid2D, Grid3D,
    GridFn1D, GridFn2D, GridFn3D, KrylovPath, MatrixDiffusionChernoff, MatrixGridFn1D,
    NegLaplacianGenerator, ScratchPool, ShiftChernoff1D, SpdResolvent, SpdSolver, Strang2D,
    Strang3D, SymmetricOperator,
};

/// Best-of-`reps` wall time in milliseconds, plus the last result.
fn time_best<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let r = f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        out = Some(r);
    }
    (best, out.expect("reps >= 1"))
}

fn row(name: &str, ms: f64, err: Option<f64>, note: &str) {
    let e = err.map_or_else(|| "      —".to_string(), |e| format!("{e:9.2e}"));
    println!("| {name:<34} | {ms:>11.3} | {e} | {note} |");
}

fn gauss_err(u: &[f64], grid: Grid1D<f64>, a: f64, t: f64) -> f64 {
    // u0 = exp(-x^2), u_t = a u_xx  =>  u = exp(-x^2/(1+4at)) / sqrt(1+4at)
    let s = 1.0 + 4.0 * a * t;
    u.iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = grid.x_at(i);
            (v - (-x * x / s).exp() / s.sqrt()).abs()
        })
        .fold(0.0, f64::max)
}

fn heat_1d() {
    let grid = Grid1D::new(-10.0, 10.0, 1000).unwrap();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    let semi = ChernoffSemigroup::new(
        ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.0, grid),
        100,
    )
    .unwrap();
    let (ms, u) = time_best(5, || semi.evolve(1.0, &u0).unwrap());
    row(
        "heat1d/shift1d N=1000 n=100",
        ms,
        Some(gauss_err(&u.values, grid, 0.5, 1.0)),
        "order 1",
    );

    let semi = ChernoffSemigroup::new(DiffusionChernoff::new_const_a(0.5, 0.5, grid), 100).unwrap();
    let (ms, u) = time_best(5, || semi.evolve(1.0, &u0).unwrap());
    row(
        "heat1d/diffusion const-a",
        ms,
        Some(gauss_err(&u.values, grid, 0.5, 1.0)),
        "",
    );

    let semi = ChernoffSemigroup::new(
        DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid),
        100,
    )
    .unwrap();
    let (ms, u) = time_best(5, || semi.evolve(1.0, &u0).unwrap());
    row(
        "heat1d/diffusion fn-ptr a≡½",
        ms,
        Some(gauss_err(&u.values, grid, 0.5, 1.0)),
        "README path",
    );

    let semi = ChernoffSemigroup::new(
        DiffusionChernoff::with_closure(
            |x: f64| 0.5 + 0.2 * (0.5 * x).sin(),
            |x: f64| 0.1 * (0.5 * x).cos(),
            |x: f64| -0.05 * (0.5 * x).sin(),
            0.7,
            grid,
        ),
        100,
    )
    .unwrap();
    let (ms, _) = time_best(5, || semi.evolve(1.0, &u0).unwrap());
    row("heat1d/diffusion var-a closure", ms, None, "");

    let g6 = Grid1D::new(-10.0, 10.0, 1024).unwrap();
    let u6 = GridFn1D::from_fn(g6, |x| (-x * x).exp());
    let semi = ChernoffSemigroup::new(
        Diffusion6thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, g6),
        50,
    )
    .unwrap();
    let (ms, u) = time_best(3, || semi.evolve(1.0, &u6).unwrap());
    row(
        "heat1d/diffusion6 N=1024 n=50",
        ms,
        Some(gauss_err(&u.values, g6, 0.5, 1.0)),
        "",
    );

    let g4 = Grid1D::new(-10.0, 10.0, 256).unwrap();
    let u4 = GridFn1D::from_fn(g4, |x| (-x * x).exp());
    let semi = ChernoffSemigroup::new(
        Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, g4).with_chebyshev_sampling(),
        20,
    )
    .unwrap();
    let (ms, u) = time_best(1, || semi.evolve(1.0, &u4).unwrap());
    row(
        "heat1d/diffusion4 chebyshev N=256 n=20",
        ms,
        Some(gauss_err(&u.values, g4, 0.5, 1.0)),
        "",
    );
}

fn heat_nd() {
    let gx = Grid1D::new(-4.0, 4.0, 400).unwrap();
    let g2 = Grid2D::new(gx, gx);
    let f0 = GridFn2D::from_fn(g2, |x, y| (-x * x - y * y).exp());
    let d = || DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx);
    let semi = ChernoffSemigroup::new(Strang2D::new(d(), d()), 10).unwrap();
    let (ms, u) = time_best(1, || semi.evolve(1.0, &f0).unwrap());
    let s = 3.0_f64; // 1 + 4·½·1
    let err = (0..400 * 400)
        .map(|k| {
            let (i, j) = (k % 400, k / 400);
            let (x, y) = (gx.x_at(i), gx.x_at(j));
            (u.values[k] - (-(x * x + y * y) / s).exp() / s).abs()
        })
        .fold(0.0, f64::max);
    row("heat2d/strang 400² n=10", ms, Some(err), "");

    let g = Grid1D::new(-4.0, 4.0, 64).unwrap();
    let g3 = Grid3D::new(g, g, g).unwrap();
    let f0 = GridFn3D::from_fn(g3, |x, y, z| (-(x * x + y * y + z * z)).exp());
    let d = || DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, g);
    let semi = ChernoffSemigroup::new(Strang3D::new(d(), d(), d()), 10).unwrap();
    let (ms, _) = time_best(1, || semi.evolve(1.0, &f0).unwrap());
    row("heat3d/strang 64³ n=10", ms, None, "");
}

fn matrix_system() {
    let grid = Grid1D::new(-8.0, 8.0, 512).unwrap();
    let eng = MatrixDiffusionChernoff::<f64, 3>::new(
        |_, a| {
            *a = [[0.5, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.3]];
        },
        |_, b| *b = [[0.0; 3]; 3],
        |_, c| *c = [[-0.1, 0.05, 0.0], [0.05, -0.1, 0.05], [0.0, 0.05, -0.1]],
        grid,
    )
    .unwrap();
    let u0 = MatrixGridFn1D::from_fn(grid, |x| {
        let g = (-x * x).exp();
        [g, 0.5 * g, 0.25 * g]
    });
    let semi = ChernoffSemigroup::new(eng, 100).unwrap();
    let (ms, _) = time_best(3, || semi.evolve(1.0, &u0).unwrap());
    row("matrix/diffusion M=3 N=512 n=100", ms, None, "");
}

fn laplacian_1d(n: usize, scale: f64, neumann: bool) -> SymmetricOperator<f64> {
    let mut rp = vec![0usize];
    let mut ci = Vec::new();
    let mut va = Vec::new();
    for i in 0..n {
        let edge = neumann && (i == 0 || i == n - 1);
        if i > 0 {
            ci.push((i - 1) as u32);
            va.push(-scale);
        }
        ci.push(i as u32);
        va.push(if edge { scale } else { 2.0 * scale });
        if i + 1 < n {
            ci.push((i + 1) as u32);
            va.push(-scale);
        }
        rp.push(ci.len());
    }
    SymmetricOperator::from_csr(n, &rp, &ci, &va, 1e-12).unwrap()
}

fn krylov() {
    let n = 400;
    let v: Vec<f64> = (0..n).map(|i| (i + 1) as f64 / n as f64).collect();
    let mean = v.iter().sum::<f64>() / n as f64;
    let mut scratch = ScratchPool::new();
    for &(scale, label, lanczos) in &[
        (1.0e3, "krylov/cheb N=400 λt≈4e3", true),
        (1.0e5, "krylov/cheb N=400 λt≈4e5", true),
        (1.0e7, "krylov/cheb N=400 λt≈4e7 (issue16)", false),
    ] {
        let op = laplacian_1d(n, scale, true);
        let mut out = vec![0.0; n];
        let (ms, ()) = time_best(1, || {
            graph_expmv_krylov(
                &op,
                1.0,
                &v,
                &mut out,
                KrylovPath::Chebyshev,
                1e-10,
                &mut scratch,
            )
            .unwrap();
        });
        let err = (scale >= 1e7).then(|| out.iter().map(|o| (o - mean).abs()).fold(0.0, f64::max));
        row(label, ms, err, "");
        if lanczos {
            let (ms, ()) = time_best(1, || {
                graph_expmv_krylov(
                    &op,
                    1.0,
                    &v,
                    &mut out,
                    KrylovPath::Lanczos { m_max: 18 },
                    1e-10,
                    &mut scratch,
                )
                .unwrap();
            });
            let err =
                (scale >= 1e7).then(|| out.iter().map(|o| (o - mean).abs()).fold(0.0, f64::max));
            row(&label.replace("cheb", "lanczos"), ms, err, "");
        }
    }
}

fn expmv_phi() {
    // Non-symmetric 1D advection–diffusion, ‖A‖∞ ≈ 4e3.
    let n = 1000;
    let (d, c) = (1.0e3, 1.0e2);
    let mut rp = vec![0usize];
    let mut ci = Vec::new();
    let mut va = Vec::new();
    for i in 0..n {
        if i > 0 {
            ci.push((i - 1) as u32);
            va.push(-d - c);
        }
        ci.push(i as u32);
        va.push(2.0 * d + c);
        if i + 1 < n {
            ci.push((i + 1) as u32);
            va.push(-d);
        }
        rp.push(ci.len());
    }
    let gop = GeneralOperator::from_csr(n, &rp, &ci, &va).unwrap();
    let v: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.01).sin()).collect();
    let mut out = vec![0.0; n];
    let kern = gop.expmv();
    let (ms, ()) = time_best(3, || kern.action_into_slice(1.0, &v, &mut out).unwrap());
    row("expmv/general N=1000 ‖A‖t≈4e3", ms, None, "");

    let op = laplacian_1d(n, 1.0e3, false);
    let gen = CsrGenerator::from_symmetric(&op, None).unwrap();
    let mut scratch = ScratchPool::new();
    let (ms, ()) = time_best(3, || {
        phi_action(&gen, 1, 1.0, &v, &mut out, &mut scratch).unwrap();
    });
    row("phi/phi1 N=1000 ‖G‖t≈4e3", ms, None, "");

    let op = laplacian_1d(256, 256.0 * 256.0 * 1e-2, false);
    let et = Etdrk4::new(NegLaplacianGenerator::new(op), AllenCahn::new(), 0.01).unwrap();
    let u0: Vec<f64> = (0..256).map(|i| (0.1 * i as f64).sin()).collect();
    let mut u1 = vec![0.0; 256];
    let (ms, ()) = time_best(3, || et.integrate(&u0, 100, &mut u1, &mut scratch).unwrap());
    row("etdrk4/allen-cahn N=256 100 steps", ms, None, "");

    let op = laplacian_1d(100_000, 1.0, false);
    let res = SpdResolvent::new(&op, 1.0, None, SpdSolver::Auto, 1e-12).unwrap();
    let b: Vec<f64> = (0..100_000).map(|i| (i as f64 * 1e-3).cos()).collect();
    let mut x = vec![0.0; 100_000];
    let (ms, _) = time_best(5, || res.solve_into(&b, &mut x, &mut scratch).unwrap());
    row("spdr/tridiag N=1e5 solve", ms, None, "");
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    println!("| workload | ms (best) | sup err | note |");
    println!("|---|---:|---:|---|");
    let groups: [(&str, fn()); 5] = [
        ("heat1d", heat_1d),
        ("heatnd", heat_nd),
        ("matrix", matrix_system),
        ("krylov", krylov),
        ("expmv", expmv_phi),
    ];
    for (name, f) in groups {
        if filter.is_empty() || name.contains(&filter) {
            f();
        }
    }
}
