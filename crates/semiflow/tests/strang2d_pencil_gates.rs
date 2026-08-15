//! `G_PENCIL_*` — per-pencil 2-D Strang composition (ADR-0196, Issue #21).
//!
//! `Strang2D` applies one `X` kernel to every row, so a coefficient varying
//! along the *transverse* axis is inexpressible. `Strang2DPencil` carries one
//! kernel per pencil. Two gates:
//!
//! * `G_PENCIL_REDUCTION` — when the coefficients happen to be separable, the
//!   new type must agree with the old one to machine precision. This is the
//!   cheap, sharp anchor: it proves the pencil dispatch and the palindromic leg
//!   ordering are right, independently of any convergence question.
//! * `G_PENCIL_ORDER2` — τ-self-convergence with `a_x` varying **only**
//!   transversally, which `Strang2D` cannot represent at all.

// Grid-index -> coordinate arithmetic; `ax`/`ay` name the two axis
// coefficients the kernel under test names them.
#![allow(clippy::cast_precision_loss, clippy::similar_names)]

use semiflow::{
    strang2d_pencil::Strang2DPencil, ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    ScratchPool, Strang2D,
};

const N: usize = 33;

fn grids() -> (Grid1D<f64>, Grid1D<f64>, Grid2D<f64>) {
    let gx = Grid1D::new(0.0, 1.0, N).unwrap();
    let gy = Grid1D::new(0.0, 1.0, N).unwrap();
    (gx, gy, Grid2D::new(gx, gy))
}

/// Constant-coefficient diffusion kernel on one axis.
fn const_kernel(a: f64, g: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::with_closure(move |_| a, |_| 0.0, |_| 0.0, a, g)
}

fn gaussian_ic(grid: Grid2D<f64>) -> GridFn2D<f64> {
    let mut v = vec![0.0_f64; N * N];
    for j in 0..N {
        for i in 0..N {
            let x = (i as f64) / ((N - 1) as f64);
            let y = (j as f64) / ((N - 1) as f64);
            v[i + j * N] = (-40.0 * ((x - 0.5).powi(2) + (y - 0.5).powi(2))).exp();
        }
    }
    GridFn2D::new(grid, v).unwrap()
}

fn evolve<C: ChernoffFunction<f64, S = GridFn2D<f64>>>(
    k: &C,
    grid: Grid2D<f64>,
    n_steps: usize,
    t: f64,
) -> Vec<f64> {
    let mut src = gaussian_ic(grid);
    let mut dst = GridFn2D::new(grid, vec![0.0; N * N]).unwrap();
    let mut pool = ScratchPool::<f64>::new();
    let tau = t / n_steps as f64;
    for _ in 0..n_steps {
        k.apply_into(tau, &src, &mut dst, &mut pool).unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src.values
}

/// `G_PENCIL_REDUCTION` — separable coefficients reproduce `Strang2D`.
///
/// Non-vacuity: the datum uses `a_x = 0.7 ≠ a_y = 0.3`, so the two legs are
/// distinguishable and a swapped/misordered leg would show up immediately.
#[test]
fn g_pencil_reduction_matches_strang2d() {
    let (gx, gy, grid) = grids();
    let (ax, ay) = (0.7_f64, 0.3_f64);

    let plain = Strang2D::new(const_kernel(ax, gx), const_kernel(ay, gy));
    let pencil = Strang2DPencil::new(
        (0..N).map(|_| const_kernel(ax, gx)).collect(),
        (0..N).map(|_| const_kernel(ay, gy)).collect(),
        grid,
    )
    .unwrap();

    let want = evolve(&plain, grid, 20, 0.01);
    let got = evolve(&pencil, grid, 20, 0.01);
    let err = got
        .iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max);
    assert!(err <= 1e-13, "separable reduction sup_error={err:.3e}");

    // Non-vacuity: with the axes swapped the answer really is different, so the
    // agreement above is not an artefact of an isotropic datum.
    let swapped = Strang2D::new(const_kernel(ay, gx), const_kernel(ax, gy));
    let other = evolve(&swapped, grid, 20, 0.01);
    let diff = other
        .iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        diff > 1e-6,
        "datum is axis-symmetric; the gate proves nothing"
    );
}

/// Transverse-variation amplitude of the gate datum.
///
/// The order-2 *slope* is independent of this; the error **constant** is not —
/// it carries the double commutators `[B,[B,A]]`, `[A,[A,B]]` that the
/// symmetric-splitting BCH residue leaves behind, and those grow with
/// `‖∂_y a_x‖`. Measured slope on the `n_steps ∈ {20, 40, 80}`, `t = 0.02`
/// ladder (see `diag_transverse_sweep`):
///
/// | amplitude | t = 0.005 | t = 0.02 |
/// |---|---|---|
/// | 0.1 | 2.871 | **2.053** |
/// | 0.3 | 2.723 | **1.913** |
/// | 0.6 | 1.774 | 1.175 |
///
/// At 0.6 the ladder has not reached the asymptotic regime — the commutator
/// term still dominates at `τ = 2.5e−4` — which is the concrete form of the
/// warning in the `strang2d_pencil` module doc, not a defect. The gate uses 0.3,
/// which reaches order 2 at feasible step counts while still being a genuinely
/// transverse field (±30%).
const GATE_AMPLITUDE: f64 = 0.3;

/// Kernel list for `a_x` varying **only transversally**: `a_x = a_x(y)`.
fn transverse_x(gx: Grid1D<f64>) -> Vec<DiffusionChernoff<f64>> {
    (0..N)
        .map(|j| {
            let y = (j as f64) / ((N - 1) as f64);
            const_kernel(
                1.0 + GATE_AMPLITUDE * (2.0 * core::f64::consts::PI * y).sin(),
                gx,
            )
        })
        .collect()
}

/// Kernel list for `a_y = a_y(x)`.
fn transverse_y(gy: Grid1D<f64>) -> Vec<DiffusionChernoff<f64>> {
    (0..N)
        .map(|i| {
            let x = (i as f64) / ((N - 1) as f64);
            const_kernel(
                1.0 + GATE_AMPLITUDE * (2.0 * core::f64::consts::PI * x).cos(),
                gy,
            )
        })
        .collect()
}

/// `G_PENCIL_ORDER2` — τ-self-convergence with transverse-varying coefficients.
///
/// This is the capability `Strang2D` does not have: `a_x` depends on `y` alone,
/// which no single shared X-kernel can express. Order 2 here rests on the
/// classical symmetric-splitting BCH argument, NOT on `[L_x, L_y] = 0`, which is
/// false for this datum — see the module doc of `strang2d_pencil`.
#[test]
#[ignore = "G_PENCIL_ORDER2 — self-convergence sweep; Pattern B slow gate"]
fn g_pencil_order2_transverse() {
    let (gx, gy, grid) = grids();
    let build = || Strang2DPencil::new(transverse_x(gx), transverse_y(gy), grid).unwrap();
    let t = 0.02;
    // Reference-free Richardson: |u(n) - u(2n)| / |u(2n) - u(4n)| -> 2^p.
    // A fixed reference at 640 steps carries its own O(tau^2) error, which
    // depresses the measured slope; the successive-difference form cancels it.
    let sup_diff = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    let u1 = evolve(&build(), grid, 20, t);
    let u2 = evolve(&build(), grid, 40, t);
    let u4 = evolve(&build(), grid, 80, t);
    let d1 = sup_diff(&u1, &u2);
    let d2 = sup_diff(&u2, &u4);
    assert!(d1 > 0.0 && d2 > 0.0, "difference ladder is degenerate");
    let slope = (d1 / d2).log2();
    assert!(
        (1.7..=2.3).contains(&slope),
        "transverse-varying slope {slope:.3} outside [1.7, 2.3] (d1={d1:.3e}, d2={d2:.3e})"
    );
}

/// Construction validates the pencil counts.
#[test]
fn pencil_counts_are_validated() {
    let (gx, gy, grid) = grids();
    assert!(Strang2DPencil::new(
        (0..N - 1).map(|_| const_kernel(1.0, gx)).collect(),
        (0..N).map(|_| const_kernel(1.0, gy)).collect(),
        grid,
    )
    .is_err());
    assert!(Strang2DPencil::new(
        (0..N).map(|_| const_kernel(1.0, gx)).collect(),
        (0..=N).map(|_| const_kernel(1.0, gy)).collect(),
        grid,
    )
    .is_err());
}

/// A transverse-varying field genuinely differs from any separable one.
///
/// Teeth for the whole feature: if this failed, `Strang2D` could have expressed
/// the datum and the new type would be unnecessary.
#[test]
fn transverse_field_is_not_expressible_as_separable() {
    let (gx, gy, grid) = grids();
    let pencil = Strang2DPencil::new(transverse_x(gx), transverse_y(gy), grid).unwrap();
    let got = evolve(&pencil, grid, 40, 0.02);

    // Best separable stand-in: the mean of each transverse profile.
    let ax_mean = 1.0;
    let ay_mean = 1.0;
    let plain = Strang2D::new(const_kernel(ax_mean, gx), const_kernel(ay_mean, gy));
    let sep = evolve(&plain, grid, 40, 0.02);
    let diff = got
        .iter()
        .zip(sep.iter())
        .map(|(g, s)| (g - s).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        diff > 1e-4,
        "transverse field is indistinguishable from separable (diff={diff:.3e}) — \
         the new type would not be needed"
    );
}

/// Diagnostic control: the SAME estimator on a constant-coefficient datum,
/// Sup-norm of the difference, shared by the diagnostics below.
///
/// A free function rather than a local closure: the diagnostic that used it is a
/// three-way comparison whose body is already at the 50-line cap.
fn sup_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// through both `Strang2D` and `Strang2DPencil`.
#[test]
#[ignore = "diagnostic"]
fn diag_const_coeff_slope_control() {
    let (gx, gy, grid) = grids();
    let t = 0.02;
    let plain = Strang2D::new(const_kernel(1.0, gx), const_kernel(1.0, gy));
    let (p1, p2, p4) = (
        evolve(&plain, grid, 20, t),
        evolve(&plain, grid, 40, t),
        evolve(&plain, grid, 80, t),
    );
    println!(
        "Strang2D  const-a slope = {:.3}",
        (sup_diff(&p1, &p2) / sup_diff(&p2, &p4)).log2()
    );

    let pen = Strang2DPencil::new(
        (0..N).map(|_| const_kernel(1.0, gx)).collect(),
        (0..N).map(|_| const_kernel(1.0, gy)).collect(),
        grid,
    )
    .unwrap();
    let (q1, q2, q4) = (
        evolve(&pen, grid, 20, t),
        evolve(&pen, grid, 40, t),
        evolve(&pen, grid, 80, t),
    );
    println!(
        "Pencil    const-a slope = {:.3}",
        (sup_diff(&q1, &q2) / sup_diff(&q2, &q4)).log2()
    );

    // Separable-but-varying: a_x(x), a_y(y) — expressible by Strang2D too.
    let sep_x: Vec<_> = (0..N)
        .map(|_| {
            DiffusionChernoff::with_closure(
                |x: f64| 1.0 + 0.6 * (2.0 * core::f64::consts::PI * x).sin(),
                |_| 0.0,
                |_| 0.0,
                1.6,
                gx,
            )
        })
        .collect();
    let sep_y: Vec<_> = (0..N).map(|_| const_kernel(1.0, gy)).collect();
    let sp = Strang2DPencil::new(sep_x, sep_y, grid).unwrap();
    let (r1, r2, r4) = (
        evolve(&sp, grid, 20, t),
        evolve(&sp, grid, 40, t),
        evolve(&sp, grid, 80, t),
    );
    println!(
        "Pencil  a_x(x) slope    = {:.3}  (a'=0 passed, so this is the a'=0 regime)",
        (sup_diff(&r1, &r2) / sup_diff(&r2, &r4)).log2()
    );
}

/// Diagnostic sweep: how the transverse slope depends on amplitude, grid and t.
#[test]
#[ignore = "diagnostic"]
fn diag_transverse_sweep() {
    let sup_diff = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    for amp in [0.1_f64, 0.3, 0.6] {
        for t in [0.005_f64, 0.02] {
            let (gx, gy, grid) = grids();
            let xs: Vec<_> = (0..N)
                .map(|j| {
                    let y = (j as f64) / ((N - 1) as f64);
                    const_kernel(1.0 + amp * (2.0 * core::f64::consts::PI * y).sin(), gx)
                })
                .collect();
            let ys: Vec<_> = (0..N)
                .map(|i| {
                    let x = (i as f64) / ((N - 1) as f64);
                    const_kernel(1.0 + amp * (2.0 * core::f64::consts::PI * x).cos(), gy)
                })
                .collect();
            let k = Strang2DPencil::new(xs, ys, grid).unwrap();
            let (a, b, c) = (
                evolve(&k, grid, 20, t),
                evolve(&k, grid, 40, t),
                evolve(&k, grid, 80, t),
            );
            let (d1, d2) = (sup_diff(&a, &b), sup_diff(&b, &c));
            println!(
                "amp={amp:.1} t={t:.3}  slope={:.3}  d1={d1:.2e} d2={d2:.2e}",
                (d1 / d2).log2()
            );
        }
    }
}
