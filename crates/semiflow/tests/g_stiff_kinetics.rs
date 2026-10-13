//! Stiff kinetics gates (`RELEASE_BLOCKING`, ADR-0212, math §69).
//!
//! * `G_STIFF_KINETICS_LINEAR`: `u′ = Cu`, `C` symmetric `2×2` with eigenvalues
//!   `−10⁶` (along `(1, 1)`) and `−1` (along `(1, −1)`), on a field whose diffusion is
//!   the identity, over `T = 1`. RK4 with one substep per reaction step (`n = 10`:
//!   `hλ = −5·10⁴` on the two end half-steps, `−10⁵` on the merged steps) must blow
//!   up (error `> 10⁶×` the solution) — the gate is not vacuous; ROS2 single steps
//!   (tolerances so loose that every step is accepted whole) converge to the exact
//!   `e^{TC}u₀` with order 2 (slope `∈ [1.8, 2.2]` over `n = 20, 40, 80`); the pure
//!   stiff mode `(1, 1)` after two whole steps at `hλ = −5·10⁵` is `≤ 1e−9`
//!   (L-stability, `R(∞) = 0`: a non-L-stable `γ` leaves `≈ R(∞)²`); adaptive ROS2
//!   meets `rtol ∈ {1e−4, 1e−6, 1e−8}` to `≤ 10·rtol` (relative, sup norm). The same
//!   system NON-AUTONOMOUS (forcing `±cos t` on the slow mode, `∂f/∂t` omitted from
//!   the method): fixed steps converge with order 2 (`n = 80, 160, 320`) and the
//!   adaptive runs meet `≤ 10·rtol` — a wrong stage time drops to order 1.
//! * `G_STIFF_KINETICS_FD_JAC`: every catalogue analytic Jacobian (Nagumo,
//!   Gray–Scott, FitzHugh–Nagumo, Brusselator) equals central differences of its
//!   `eval_batch` to `≤ 1e−7` relative at 64 random states; a NON-SYMMETRIC stiff
//!   linear system run with forward differences agrees with the analytic-Jacobian
//!   run to `≤ 1e−12` (a transposed or mis-scaled difference Jacobian fails it); the
//!   forward-difference layout per node is checked directly in the unit tests.
//! * `G_STIFF_ROBERTSON`: Robertson's chemistry (`k₁ = 0.04`, `k₂ = 3·10⁷`,
//!   `k₃ = 10⁴`) on a uniform 4-node field, `T = 40`, against SciPy Radau
//!   (`rtol = 1e−13`, `atol = 1e−20`, SciPy 1.18.1; agrees with the Hairer–Wanner
//!   reference to `1e−13`): per-species relative error `≤ 10·rtol` for
//!   `rtol ∈ {1e−3, 1e−5, 1e−7, 1e−8}`, analytic and forward-difference Jacobians,
//!   `|Σyₖ − 1| ≤ 1e−12`, `min yₖ ≥ −1e−12`.
//!
//! The per-node gate (node-dependent stiffness and pivoting) is
//! `tests/g_stiff_kinetics_nodes.rs`.

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

mod stiff_common;

use semiflow::{
    Brusselator, FitzHughNagumo, GrayScott, Kinetics, Nagumo, ReactionSolver, SemiflowError,
};
use stiff_common::{evolve, rel_sup, ros};

/// `u′ = Cu + (cos t)(1, −1)·[forced]` with `C = [[a, b], [b, a]]`,
/// `a ± b = −1, −10⁶` (the forcing drives only the slow mode); analytic `J = C`
/// unless `fd` (`∂f/∂t` is never supplied: the W-method claim).
struct StiffLinear {
    fd: bool,
    forced: bool,
}

const A: f64 = (-1e6 - 1.0) / 2.0;
const B: f64 = (-1e6 + 1.0) / 2.0;

impl Kinetics<f64> for StiffLinear {
    fn species(&self) -> usize {
        2
    }
    fn eval_batch(
        &self,
        t: f64,
        n: usize,
        _x: &[f64],
        u: &[f64],
        du: &mut [f64],
    ) -> Result<(), SemiflowError> {
        let s = if self.forced { t.cos() } else { 0.0 };
        for i in 0..n {
            du[i] = A * u[i] + B * u[n + i] + s;
            du[n + i] = B * u[i] + A * u[n + i] - s;
        }
        Ok(())
    }
    fn jacobian_batch(
        &self,
        _t: f64,
        n: usize,
        _x: &[f64],
        _u: &[f64],
        jac: &mut [f64],
    ) -> Option<Result<(), SemiflowError>> {
        if self.fd {
            return None;
        }
        for (b, v) in [A, B, B, A].into_iter().enumerate() {
            jac[b * n..(b + 1) * n].fill(v);
        }
        Some(Ok(()))
    }
}

/// Robertson's chemistry; analytic Jacobian unless `fd`.
struct Robertson {
    fd: bool,
}

impl Kinetics<f64> for Robertson {
    fn species(&self) -> usize {
        3
    }
    fn eval_batch(
        &self,
        _t: f64,
        n: usize,
        _x: &[f64],
        u: &[f64],
        du: &mut [f64],
    ) -> Result<(), SemiflowError> {
        for i in 0..n {
            let (a, b, c) = (u[i], u[n + i], u[2 * n + i]);
            du[i] = -0.04 * a + 1e4 * b * c;
            du[n + i] = 0.04 * a - 1e4 * b * c - 3e7 * b * b;
            du[2 * n + i] = 3e7 * b * b;
        }
        Ok(())
    }
    fn jacobian_batch(
        &self,
        _t: f64,
        n: usize,
        _x: &[f64],
        u: &[f64],
        jac: &mut [f64],
    ) -> Option<Result<(), SemiflowError>> {
        if self.fd {
            return None;
        }
        for i in 0..n {
            let (b, c) = (u[n + i], u[2 * n + i]);
            let rows = [
                [-0.04, 1e4 * c, 1e4 * b],
                [0.04, -1e4 * c - 6e7 * b, -1e4 * b],
                [0.0, 6e7 * b, 0.0],
            ];
            for (r, row) in rows.iter().enumerate() {
                for (l, &v) in row.iter().enumerate() {
                    jac[(r * 3 + l) * n + i] = v;
                }
            }
        }
        Some(Ok(()))
    }
}

const ROBERTSON_T40: [f64; 3] = [
    0.715_827_068_719_408_4,
    9.185_534_764_557_822e-6,
    0.284_163_745_745_829_87,
];

/// `e^{TC}u₀` in the eigenbasis `(1, 1)/√2`, `(1, −1)/√2`.
fn linear_exact(u0: [f64; 2], t: f64) -> [f64; 2] {
    let (p, m) = ((u0[0] + u0[1]) / 2.0, (u0[0] - u0[1]) / 2.0);
    let (ep, em) = ((-1e6 * t).exp() * p, (-t).exp() * m);
    [ep + em, ep - em]
}

/// The forced system's exact solution: the slow mode solves `m′ = −m + cos t`.
fn forced_exact(u0: [f64; 2], t: f64) -> [f64; 2] {
    let p = (-1e6 * t).exp() * (u0[0] + u0[1]) / 2.0;
    let m = (-t).exp() * (u0[0] - u0[1]) / 2.0 + (t.cos() + t.sin() - (-t).exp()) / 2.0;
    [p + m, p - m]
}

/// `u′ = Cu` with the NON-symmetric `C = [[−10⁶, 0], [999 999, −1]]` (eigenvalues
/// `−10⁶`, `−1`); analytic Jacobian unless `fd`.
struct SkewLinear {
    fd: bool,
}

const SKEW: [f64; 4] = [-1e6, 0.0, 999_999.0, -1.0];

impl Kinetics<f64> for SkewLinear {
    fn species(&self) -> usize {
        2
    }
    fn eval_batch(
        &self,
        _t: f64,
        n: usize,
        _x: &[f64],
        u: &[f64],
        du: &mut [f64],
    ) -> Result<(), SemiflowError> {
        for i in 0..n {
            du[i] = SKEW[0] * u[i] + SKEW[1] * u[n + i];
            du[n + i] = SKEW[2] * u[i] + SKEW[3] * u[n + i];
        }
        Ok(())
    }
    fn jacobian_batch(
        &self,
        _t: f64,
        n: usize,
        _x: &[f64],
        _u: &[f64],
        jac: &mut [f64],
    ) -> Option<Result<(), SemiflowError>> {
        if self.fd {
            return None;
        }
        for (b, v) in SKEW.into_iter().enumerate() {
            jac[b * n..(b + 1) * n].fill(v);
        }
        Some(Ok(()))
    }
}

/// The stiff linear system (`forced` or not) over `T = 1` in `n` Strang steps.
fn linear_run(forced: bool, solver: ReactionSolver, u0: &[f64], n: usize) -> Vec<f64> {
    let kin = Box::new(StiffLinear { fd: false, forced });
    evolve(kin, solver, 1, u0, (1.0, n)).expect("ros2")
}

/// Fixed-step slopes over `ns` and adaptive errors over `rtol ∈ {1e−4, 1e−6,
/// 1e−8}` (as multiples of `rtol`) against `exact`.
fn order_and_tolerance(forced: bool, ns: [usize; 3], exact: [f64; 2]) -> ([f64; 2], Vec<f64>) {
    let u0 = [1.0, 0.3];
    let errs: Vec<f64> = ns
        .iter()
        .map(|&n| rel_sup(&linear_run(forced, ros(0.5, 1e3), &u0, n), &exact))
        .collect();
    let slopes = [(errs[0] / errs[1]).log2(), (errs[1] / errs[2]).log2()];
    let tol = [1e-4, 1e-6, 1e-8]
        .iter()
        .map(|&rtol| rel_sup(&linear_run(forced, ros(rtol, 1e-14), &u0, 1), &exact) / rtol)
        .collect();
    eprintln!("G_STIFF_KINETICS_LINEAR forced={forced}: errors {errs:?} slopes {slopes:?} err/rtol {tol:?}");
    (slopes, tol)
}

#[test]
fn g_stiff_kinetics_linear() {
    let u0 = [1.0, 0.3];
    let exact = linear_exact(u0, 1.0);
    let rk4 = evolve(
        Box::new(StiffLinear {
            fd: false,
            forced: false,
        }),
        ReactionSolver::Rk4,
        1,
        &u0,
        (1.0, 10),
    );
    eprintln!("G_STIFF_KINETICS_LINEAR rk4: {rk4:?}");
    let blown = rk4.as_ref().map_or(true, |v| rel_sup(v, &exact) > 1e6);
    assert!(blown, "RK4 at hλ = −5e4/−1e5 must blow up: {rk4:?}");
    // Single ROS2 steps: every step accepted whole (rtol 0.5, atol 1e3).
    for (forced, ns, want) in [
        (false, [20, 40, 80], exact),
        (true, [80, 160, 320], forced_exact(u0, 1.0)),
    ] {
        let (slopes, tol) = order_and_tolerance(forced, ns, want);
        assert!(slopes.iter().all(|s| (1.8..=2.2).contains(s)), "{slopes:?}");
        assert!(tol.iter().all(|&r| r <= 10.0), "forced={forced}: {tol:?}");
    }
    // L-stability: the pure stiff mode after two whole steps at z = −5·10⁵.
    let stiff = linear_run(false, ros(0.5, 1e3), &[1.0, 1.0], 1);
    let damped = stiff.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    eprintln!("G_STIFF_KINETICS_LINEAR stiff mode after 2 steps: {damped:.3e}");
    assert!(
        damped <= 1e-9,
        "stiff mode not annihilated (L-stability): {damped:e}"
    );
}

/// Central-difference check of `kin.jacobian_batch` at `n` random states.
fn jacobian_matches_differences(name: &str, kin: &dyn Kinetics<f64>, n: usize) -> f64 {
    let k = kin.species();
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut rand = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let u: Vec<f64> = (0..k * n).map(|_| 2.0 * rand() - 0.5).collect();
    let mut jac = vec![0.0; k * k * n];
    kin.jacobian_batch(0.0, n, &[], &u, &mut jac)
        .expect("catalogue Jacobian")
        .expect("ok");
    let (mut fp, mut fm) = (vec![0.0; k * n], vec![0.0; k * n]);
    let mut worst = 0.0_f64;
    for l in 0..k {
        let (mut up, mut um) = (u.clone(), u.clone());
        let h = 1e-6;
        for i in 0..n {
            up[l * n + i] += h;
            um[l * n + i] -= h;
        }
        kin.eval_batch(0.0, n, &[], &up, &mut fp).expect("eval");
        kin.eval_batch(0.0, n, &[], &um, &mut fm).expect("eval");
        for r in 0..k {
            for i in 0..n {
                let fd = (fp[r * n + i] - fm[r * n + i]) / (2.0 * h);
                let an = jac[(r * k + l) * n + i];
                worst = worst.max((fd - an).abs() / an.abs().max(1.0));
            }
        }
    }
    eprintln!("G_STIFF_KINETICS_FD_JAC {name}: max rel diff {worst:.2e}");
    worst
}

#[test]
fn g_stiff_kinetics_fd_jac() {
    let catalogue: [(&str, Box<dyn Kinetics<f64>>); 4] = [
        ("nagumo", Box::new(Nagumo { a: 0.3 })),
        (
            "gray_scott",
            Box::new(GrayScott {
                feed: 0.04,
                kill: 0.06,
            }),
        ),
        (
            "fitzhugh_nagumo",
            Box::new(FitzHughNagumo {
                a: 0.7,
                b: 0.8,
                eps: 0.08,
                current: 0.5,
            }),
        ),
        ("brusselator", Box::new(Brusselator { a: 1.0, b: 3.0 })),
    ];
    for (name, kin) in &catalogue {
        assert!(
            jacobian_matches_differences(name, kin.as_ref(), 64) <= 1e-7,
            "{name}"
        );
    }
    let run = |fd: bool| {
        let kin = Box::new(SkewLinear { fd });
        evolve(kin, ros(1e-6, 1e-14), 1, &[1.0, 0.3], (1.0, 1)).expect("ros2")
    };
    let lin = rel_sup(&run(true), &run(false));
    eprintln!("G_STIFF_KINETICS_FD_JAC non-symmetric linear, fd vs analytic: {lin:.3e}");
    assert!(lin <= 1e-12, "{lin:e}");
}

#[test]
fn g_stiff_robertson() {
    for fd in [false, true] {
        for rtol in [1e-3, 1e-5, 1e-7, 1e-8] {
            let got = evolve(
                Box::new(Robertson { fd }),
                ros(rtol, 1e-12),
                1,
                &[1.0, 0.0, 0.0],
                (40.0, 1),
            )
            .expect("ros2");
            let rel = (0..3).fold(0.0_f64, |m, k| {
                m.max((got[k] - ROBERTSON_T40[k]).abs() / ROBERTSON_T40[k])
            });
            let mass = (got.iter().sum::<f64>() - 1.0).abs();
            let min = got.iter().fold(f64::INFINITY, |a, &v| a.min(v));
            eprintln!("G_STIFF_ROBERTSON fd={fd} rtol={rtol:e}: rel {rel:.3e} mass {mass:.1e} min {min:.3e}");
            assert!(rel <= 10.0 * rtol, "fd={fd} rtol={rtol:e}: rel {rel:e}");
            assert!(mass <= 1e-12 && min >= -1e-12, "mass {mass:e} min {min:e}");
        }
    }
}
