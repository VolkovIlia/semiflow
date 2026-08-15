//! `G_ASND_MOMENT` — second-moment growth of `AnisotropicShiftChernoffND` (ADR-0191).
//!
//! The kernel's generator is `∂_t u = A·∇²u`, so a Gaussian initial condition
//! must gain **exactly** `2·a_dd·t` of variance along axis `d`, independently of
//! how the interval `t` is chopped into `n_steps`.
//!
//! This is the oracle that was missing when issue #17 was filed. Every previous
//! oracle for this kernel — `F(0) = I`, constant-preservation, temporal
//! self-convergence — is blind to the defect, because a sampler that injects a
//! fixed amount of spurious variance *per step* preserves constants exactly,
//! is the identity at `τ = 0`, and self-converges to its own wrong answer.
//!
//! Pre-ADR-0191 measurement (multilinear sampler, 96² grid, `A = I`, `t = 0.5`):
//! `dVar = 1.2113 / 2.2449 / 4.4901` at `n_steps = 100 / 400 / 1600` against an
//! exact `1.0` — i.e. the error grew with the very parameter users increase to
//! improve accuracy.

// Grid-index -> coordinate arithmetic; every index here is a small test
// constant, far below the f64 mantissa.
#![allow(clippy::cast_precision_loss)]

use semiflow::{
    grid_nd::{GridFnND, GridND},
    shift_nd::AnisotropicShiftChernoffND,
    ChernoffFunction, Grid1D, ScratchPool,
};

const NX: usize = 96;
const LO: f64 = -8.0;
const HI: f64 = 8.0;
/// Variance of the Gaussian initial condition, per axis.
const VAR0: f64 = 0.5;
const T_FINAL: f64 = 0.5;

/// Second central moments `(var_x, var_y, cov_xy)` of `u` treated as a density.
fn moments(u: &[f64], nx: usize) -> (f64, f64, f64) {
    let x_at = |k: usize| LO + (HI - LO) * (k as f64) / ((nx - 1) as f64);
    let (mut mass, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for iy in 0..nx {
        for ix in 0..nx {
            let w = u[ix + iy * nx];
            mass += w;
            sx += x_at(ix) * w;
            sy += x_at(iy) * w;
        }
    }
    let (ex, ey) = (sx / mass, sy / mass);
    let (mut vx, mut vy, mut cxy) = (0.0, 0.0, 0.0);
    for iy in 0..nx {
        for ix in 0..nx {
            let w = u[ix + iy * nx];
            let (dx, dy) = (x_at(ix) - ex, x_at(iy) - ey);
            vx += dx * dx * w;
            vy += dy * dy * w;
            cxy += dx * dy * w;
        }
    }
    (vx / mass, vy / mass, cxy / mass)
}

/// Evolve a unit Gaussian under constant tensor `a` and return the moment gain.
fn run(a: [f64; 4], n_steps: usize) -> (f64, f64, f64) {
    run_on(NX, a, n_steps)
}

/// `run` on an explicit per-axis node count.
fn run_on(nx: usize, a: [f64; 4], n_steps: usize) -> (f64, f64, f64) {
    let gx = Grid1D::new(LO, HI, nx).unwrap();
    let grid = GridND::<f64, 2>::new([gx, gx]).unwrap();
    let kernel = AnisotropicShiftChernoffND::<f64, 2>::new(
        move |_x, mat| {
            mat.set(0, 0, a[0]);
            mat.set(0, 1, a[1]);
            mat.set(1, 0, a[2]);
            mat.set(1, 1, a[3]);
        },
        |_x, bv| {
            bv[0] = 0.0;
            bv[1] = 0.0;
        },
        |_x| 0.0,
        grid.clone(),
    )
    .unwrap();

    let u0 = GridFnND::<f64, 2>::from_fn(grid.clone(), |x: &[f64; 2]| {
        (-(x[0] * x[0] + x[1] * x[1]) / (2.0 * VAR0)).exp()
    });
    let mut src = u0;
    let mut dst = GridFnND::<f64, 2>::new(grid, vec![0.0; nx * nx]).unwrap();
    let mut scratch = ScratchPool::<f64>::new();
    let tau = T_FINAL / n_steps as f64;
    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &src, &mut dst, &mut scratch)
            .unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    let (vx, vy, cxy) = moments(&src.values, nx);
    (vx - VAR0, vy - VAR0, cxy)
}

/// Fast regression guard: the excess at `n_steps = 200` on a 48² grid.
///
/// The full `G_ASND_MOMENT` ladder below is `slow-tests`-gated because 1600
/// steps on a 96² grid costs ~200 s, which does not belong in `xtask test-fast`.
/// This cheap variant still catches the defect decisively: the pre-ADR-0191
/// sampler carried `n·dx²/6` of spurious variance, which at `n = 200` on this
/// coarser grid is a ~90% error against a 2% band. Refining the grid makes the
/// old failure *smaller*, so a coarse grid is the conservative choice here.
#[test]
fn asnd_moment_isotropic_smoke() {
    let n_axis = 48;
    let dx = (HI - LO) / (n_axis - 1) as f64;
    let exact = 2.0 * 1.0 * T_FINAL;
    let (dvx, dvy, _) = run_on(n_axis, [1.0, 0.0, 0.0, 1.0], 200);
    // Sanity on the non-vacuity claim: the old floor would have been this big.
    let old_floor = 200.0 * dx * dx / 6.0;
    assert!(
        old_floor / exact > 0.2,
        "datum no longer discriminates: old floor would be only {:.1}% of exact",
        100.0 * old_floor / exact
    );
    for (axis, got) in [("x", dvx), ("y", dvy)] {
        let rel = (got - exact).abs() / exact;
        assert!(rel <= 2e-2, "axis={axis}: dVar={got:.6}, exact={exact:.6}");
    }
}

/// `G_ASND_MOMENT` — isotropic gain is `2·a·t` and is FLAT in the step count.
///
/// Non-vacuity: the pre-ADR-0191 sampler produced 1.2113 / 2.2449 / 4.4901 here,
/// so the 2% band cannot be met by an implementation that carries a per-step
/// interpolation floor. The three step counts differ by 16×, so a step-count
/// dependence of the kind issue #17 reported cannot hide inside the band.
///
/// Pattern B (CONTRIBUTING.md): plain `#[ignore]`, so the `flagship-gates.yml`
/// slow job picks it up via `--features slow-tests -- --ignored` while
/// `xtask test-fast` skips the ~200 s cost. The cheap
/// `asnd_moment_isotropic_smoke` above keeps fast-iteration coverage.
#[test]
#[ignore = "G_ASND_MOMENT full ladder (~200 s) — Pattern B slow gate"]
fn g_asnd_moment_isotropic_is_step_count_flat() {
    let exact = 2.0 * 1.0 * T_FINAL; // 2·a·t with a = 1
    for n_steps in [100_usize, 400, 1600] {
        let (dvx, dvy, cxy) = run([1.0, 0.0, 0.0, 1.0], n_steps);
        for (axis, got) in [("x", dvx), ("y", dvy)] {
            let rel = (got - exact).abs() / exact;
            assert!(
                rel <= 2e-2,
                "n_steps={n_steps} axis={axis}: dVar={got:.6}, exact={exact:.6}, rel={rel:.3e}"
            );
        }
        assert!(
            cxy.abs() <= 2e-2,
            "n_steps={n_steps}: dCov={cxy:.6} should vanish"
        );
    }
}

/// `G_ASND_MOMENT` — a diagonal tensor diffuses each axis by its OWN entry.
///
/// This is the assertion that pins the state layout (`flat = ix + iy·nx`): if
/// the axes were transposed anywhere between the coefficient closure, the
/// Cholesky cache and the sampler, `a11` and `a22` would swap and the 2% band
/// would fail by a factor of two.
#[test]
fn g_asnd_moment_diagonal_tensor_does_not_mix_axes() {
    let (a11, a22) = (1.0, 0.5);
    let (dvx, dvy, _) = run([a11, 0.0, 0.0, a22], 400);
    let (ex, ey) = (2.0 * a11 * T_FINAL, 2.0 * a22 * T_FINAL);
    assert!(
        (dvx - ex).abs() / ex <= 2e-2,
        "x axis (a11={a11}): dVar={dvx:.6}, exact={ex:.6}"
    );
    assert!(
        (dvy - ey).abs() / ey <= 2e-2,
        "y axis (a22={a22}): dVar={dvy:.6}, exact={ey:.6}"
    );
}

/// Diagnostic: print the measured table that `pass_status` in
/// `contracts/semiflow-core.properties.yaml` records. Not a gate.
///
/// ```text
/// cargo test -p semiflow --release --test aniso_nd_moment -- --ignored --nocapture
/// ```
#[test]
#[ignore = "diagnostic — prints the G_ASND_MOMENT measurement table"]
fn asnd_moment_table() {
    for n_steps in [100_usize, 400, 1600] {
        let (dvx, dvy, cxy) = run([1.0, 0.0, 0.0, 1.0], n_steps);
        println!(
            "A=I      n_steps={n_steps:5}  dVar=({dvx:.6}, {dvy:.6})  dCov={cxy:.3e}  exact=1.0"
        );
    }
    let (dvx, dvy, _) = run([1.0, 0.0, 0.0, 0.5], 400);
    println!("A=diag   n_steps=  400  dVar=({dvx:.6}, {dvy:.6})  exact=(1.0, 0.5)");
    let (_, _, cxy) = run([1.0, 0.4, 0.4, 1.0], 400);
    println!("A=offdiag n_steps=  400  dCov={cxy:.6}  exact=0.4");
}

/// `G_ASND_MOMENT` — the off-diagonal entry drives the cross moment.
#[test]
fn g_asnd_moment_off_diagonal_drives_covariance() {
    let a12 = 0.4;
    let (_, _, cxy) = run([1.0, a12, a12, 1.0], 400);
    let exact = 2.0 * a12 * T_FINAL;
    assert!(
        (cxy - exact).abs() / exact <= 2e-2,
        "dCov={cxy:.6}, exact={exact:.6}"
    );
}
