//! `G_POINTEVAL` — `PointEval` byte-identity gate (`RELEASE_BLOCKING`).
//!
//! Properties.yaml v1.0.0 §`G_POINTEVAL`: `f64::to_bits()` equality between
//! `kernel.eval_at(τ, src, x_query, n)` and the full-grid path
//! `(apply_into^n src).sample_at(x_query)` on FIVE active backends.
//!
//! - **Backend A**: `DiffusionChernoff<f64>` (variable-a heat, 1-D).
//! - **Backend B**: `ShiftChernoff1D<f64>` (Theorem 6 shift kernel, 1-D).
//! - **Backend C**: `ManifoldChernoff<Sphere2<f64>, f64>` (S² heat, 2-D chart).
//! - **Backend D**: `HypoellipticChernoff<f64, 2, 1>` (Kolmogorov, 2-D phase).
//! - **Backend E**: `AnisotropicShiftChernoffND<f64, 2>` (d-D anisotropic shift, 2-D).
//!
//! Math §31.3 Proposition 31.1 is the byte-identity claim.
//! Math §31.4 is the test protocol.
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]

use semiflow::{
    hormander::{HypoellipticChernoff, KolmogorovPhaseSpace},
    point_eval::sample_gridfn2d,
    ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, GridFn1D, GridFn2D, ManifoldChernoff,
    PointEval, ScratchPool, ShiftChernoff1D, Sphere2,
};

// ---------------------------------------------------------------------------
// Gate constants (math §31.4)
// ---------------------------------------------------------------------------

const T: f64 = 0.5;
const N: u32 = 64;

// ---------------------------------------------------------------------------
// Full-grid path helpers
// ---------------------------------------------------------------------------

/// Full-grid path for 1-D kernels: run `apply_into` n times, then sample at x.
fn full_grid_1d<C>(kernel: &C, tau: f64, src: &GridFn1D<f64>, n: u32, x: f64) -> f64
where
    C: ChernoffFunction<f64, S = GridFn1D<f64>>,
{
    let mut pool = ScratchPool::<f64>::new();
    let mut cur = src.clone();
    let mut nxt = src.clone();
    for _ in 0..n {
        kernel.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur.sample(x).unwrap()
}

/// Full-grid path for 2-D kernels: run `apply_into` n times, then bilinear at (cx, cy).
fn full_grid_2d<C>(kernel: &C, tau: f64, src: &GridFn2D<f64>, n: u32, cx: f64, cy: f64) -> f64
where
    C: ChernoffFunction<f64, S = GridFn2D<f64>>,
{
    let mut pool = ScratchPool::<f64>::new();
    let grid = src.grid;
    let nn = src.values.len();
    let mut cur = src.clone();
    let mut nxt = GridFn2D {
        values: vec![0.0_f64; nn],
        grid,
    };
    for _ in 0..n {
        kernel.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    sample_gridfn2d(&cur, cx, cy)
}

// ---------------------------------------------------------------------------
// Backend A — DiffusionChernoff<f64>
// ---------------------------------------------------------------------------

/// `G_POINTEVAL` sub-test 1: `DiffusionChernoff` byte-identity.
///
/// Setup: variable-a kernel `a(x) = 1 + 0.5·tanh²(x)`, IC `exp(-x²)`.
/// Grid: [-10, 10] × 512 nodes (math §31.4 Backend A spec).
/// Query: grid centre (x=0.0).
#[test]
fn g_pointeval_byte_identity_diffusion() {
    let grid = Grid1D::new(-10.0_f64, 10.0, 512).unwrap();
    let kernel = DiffusionChernoff::new(
        |x: f64| 1.0 + 0.5 * x.tanh().powi(2),
        |x: f64| x.tanh() * (1.0 - x.tanh().powi(2)),
        |x: f64| {
            let th = x.tanh();
            let sech2 = 1.0 - th * th;
            2.0 * sech2 * sech2 - 2.0 * th * th * sech2
        },
        1.5_f64,
        grid,
    );
    let src = GridFn1D {
        values: (0..grid.n)
            .map(|i| {
                let x = grid.x_at(i);
                (-x * x).exp()
            })
            .collect(),
        grid,
    };

    let tau = T / f64::from(N);
    // Query at grid centre (x = 0.0 — midpoint of [-10, 10]).
    let x_query = 0.0_f64;
    let x_slice = [x_query];

    let pe_val = kernel.eval_at(tau, &src, &x_slice, N).unwrap();
    let fg_val = full_grid_1d(&kernel, tau, &src, N, x_query);

    assert_eq!(
        pe_val.to_bits(),
        fg_val.to_bits(),
        "G_POINTEVAL Backend A: eval_at {pe_val:.16e} != full_grid {fg_val:.16e}",
    );
}

// ---------------------------------------------------------------------------
// Backend B — ShiftChernoff1D<f64>
// ---------------------------------------------------------------------------

/// `G_POINTEVAL` sub-test 2: `ShiftChernoff1D` byte-identity.
///
/// Setup: constant-a unit diffusion `a=1, b=0, c=0`, IC `exp(-x²)`.
/// Grid: [-10, 10] × 512 nodes.
/// Query: x = 0.0 (grid centre).
#[test]
fn g_pointeval_byte_identity_shift1d() {
    let grid = Grid1D::new(-10.0_f64, 10.0, 512).unwrap();
    let kernel = ShiftChernoff1D::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        0.0_f64,
        grid,
    );
    let src = GridFn1D {
        values: (0..grid.n)
            .map(|i| {
                let x = grid.x_at(i);
                (-x * x).exp()
            })
            .collect(),
        grid,
    };

    let tau = T / f64::from(N);
    let x_query = 0.0_f64;
    let x_slice = [x_query];

    let pe_val = kernel.eval_at(tau, &src, &x_slice, N).unwrap();
    let fg_val = full_grid_1d(&kernel, tau, &src, N, x_query);

    assert_eq!(
        pe_val.to_bits(),
        fg_val.to_bits(),
        "G_POINTEVAL Backend B: eval_at {pe_val:.16e} != full_grid {fg_val:.16e}",
    );
}

// ---------------------------------------------------------------------------
// Backend C — ManifoldChernoff<Sphere2<f64>, f64>
// ---------------------------------------------------------------------------

/// `G_POINTEVAL` sub-test 3: `ManifoldChernoff` (`Sphere2`) byte-identity.
///
/// Setup: unit sphere, IC = Y(0,0) (constant function), order-2 correction.
/// Grid: 32×64 (θ,φ) chart (smaller than G26 — faster; byte-identity only).
/// Query: chart centre (theta\_mid, phi\_mid).
#[test]
fn g_pointeval_byte_identity_manifold_sphere2() {
    use core::f64::consts::PI;

    let eps = 0.02_f64;
    let g_theta = Grid1D::new(eps, PI - eps, 32).unwrap();
    let g_phi = Grid1D::new(0.0_f64, 2.0 * PI, 64).unwrap();
    let grid = Grid2D::new(g_theta, g_phi);

    let sphere = Sphere2::unit();
    let kernel = ManifoldChernoff::new(sphere, false); // order-1, no curvature

    // IC: constant function (Y_{0,0} up to normalisation).
    let src = GridFn2D::from_fn(grid, |_theta, _phi| 1.0_f64);

    let tau = T / f64::from(N);
    // Query at chart centre: mid θ, mid φ.
    let theta_mid = g_theta.x_at(g_theta.n / 2);
    let phi_mid = g_phi.x_at(g_phi.n / 2);
    let x_query = [theta_mid, phi_mid];

    let pe_val = kernel.eval_at(tau, &src, &x_query, N).unwrap();
    let fg_val = full_grid_2d(&kernel, tau, &src, N, theta_mid, phi_mid);

    assert_eq!(
        pe_val.to_bits(),
        fg_val.to_bits(),
        "G_POINTEVAL Backend C: eval_at {pe_val:.16e} != full_grid {fg_val:.16e}",
    );
}

// ---------------------------------------------------------------------------
// Backend D — HypoellipticChernoff<f64, 2, 1> (Kolmogorov)
// ---------------------------------------------------------------------------

/// `G_POINTEVAL` sub-test 4: `HypoellipticChernoff` (Kolmogorov) byte-identity.
///
/// Setup: Kolmogorov phase-space L = v·∂x + ½∂²v, IC = Gaussian.
/// Grid: 64×64 on [-3,3]² (smaller than G28/G29 — faster; byte-identity only).
/// Query: phase-space centre (x=0, v=0).
#[test]
fn g_pointeval_byte_identity_hypoelliptic_kolmogorov() {
    let gx = Grid1D::new(-3.0_f64, 3.0, 64).unwrap();
    let gv = Grid1D::new(-3.0_f64, 3.0, 64).unwrap();
    let grid = Grid2D::new(gx, gv);

    let kernel = HypoellipticChernoff::<f64, 2, 1>::new(
        Box::new(KolmogorovPhaseSpace::<f64>::x0_drift()),
        [Box::new(KolmogorovPhaseSpace::<f64>::x1_diffusion())],
    )
    .expect("Kolmogorov Hörmander step-2 condition satisfied");

    let src = GridFn2D::from_fn(grid, |x, v| (-(x * x + v * v) * 0.5).exp());

    let tau = T / f64::from(N);
    let x_mid = gx.x_at(gx.n / 2);
    let v_mid = gv.x_at(gv.n / 2);
    let x_query = [x_mid, v_mid];

    let pe_val = kernel.eval_at(tau, &src, &x_query, N).unwrap();
    let fg_val = full_grid_2d(&kernel, tau, &src, N, x_mid, v_mid);

    assert_eq!(
        pe_val.to_bits(),
        fg_val.to_bits(),
        "G_POINTEVAL Backend D: eval_at {pe_val:.16e} != full_grid {fg_val:.16e}",
    );
}

// ---------------------------------------------------------------------------
// Backend E — AnisotropicShiftChernoffND<f64, 2>
// ---------------------------------------------------------------------------

/// `G_POINTEVAL` sub-test 5: `AnisotropicShiftChernoffND` D=2 byte-identity.
///
/// Setup: anisotropic 2-D kernel `A = I + 0.25·tanh(x₀+x₁)·off-diag`, b=0, c=0.
/// Grid: 32×32 on [-5,5]² (small — byte-identity only, not convergence).
/// Query: grid centre (0.0, 0.0).
///
/// Byte-identity holds by Proposition 31.1: `eval_at` is defined as `apply_into`^n
/// followed by `GridFnND::sample`, identical to the full-grid path.
#[test]
fn g_pointeval_byte_identity_anisotropic_shift_nd() {
    use semiflow::{
        grid_nd::{GridFnND, GridND},
        AnisotropicShiftChernoffND, SquareMatrix,
    };

    let ax = Grid1D::new(-5.0_f64, 5.0, 32).unwrap();
    let grid = GridND::<f64, 2>::new([ax; 2]).unwrap();

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
    .unwrap();

    let src = GridFnND::from_fn(grid.clone(), |x: &[f64; 2]| {
        (-x[0] * x[0] - x[1] * x[1]).exp()
    });

    let tau = T / f64::from(N);
    // Query at grid centre.
    let x_query = [0.0_f64, 0.0_f64];

    let pe_val = kernel.eval_at(tau, &src, &x_query, N).unwrap();

    // Full-grid path: apply_into^N then sample.
    let mut pool = ScratchPool::<f64>::new();
    let mut cur = src.clone();
    let mut nxt = src.clone();
    for _ in 0..N {
        kernel.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let fg_val = cur.sample(&x_query).unwrap();

    assert_eq!(
        pe_val.to_bits(),
        fg_val.to_bits(),
        "G_POINTEVAL Backend E: eval_at {pe_val:.16e} != full_grid {fg_val:.16e}",
    );
}
