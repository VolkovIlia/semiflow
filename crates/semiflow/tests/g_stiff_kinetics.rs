//! Stiff kinetics gates (`RELEASE_BLOCKING`, ADR-0212, math §69).
//!
//! * `G_STIFF_KINETICS_LINEAR`: `u′ = Cu`, `C` symmetric `2×2` with eigenvalues
//!   `−10⁶` (along `(1, 1)`) and `−1` (along `(1, −1)`), on a field whose diffusion is
//!   the identity, over `T = 1`. RK4 with one substep per half step (`hλ = −10⁴`)
//!   must blow up (error `> 10⁶×` the solution) — the gate is not vacuous; ROS2 single steps
//!   (tolerances so loose that every step is accepted whole) converge to the exact
//!   `e^{TC}u₀` with order 2 (slope `∈ [1.8, 2.2]` over `n = 20, 40, 80`), the stiff
//!   mode damped (L-stable), and adaptive ROS2 meets `rtol ∈ {1e−4, 1e−6, 1e−8}`
//!   to `≤ 10·rtol` (relative, sup norm).
//! * `G_STIFF_KINETICS_FD_JAC`: every catalogue analytic Jacobian (Nagumo,
//!   Gray–Scott, FitzHugh–Nagumo, Brusselator) equals central differences of its
//!   `eval_batch` to `≤ 1e−7` relative at 64 random states; the linear system without
//!   an analytic Jacobian (forward differences) agrees with the analytic run to
//!   `≤ 10·rtol` relative at `rtol = 1e−8`; Robertson with forward differences meets
//!   the `G_STIFF_ROBERTSON` accuracy (checked there).
//! * `G_STIFF_ROBERTSON`: Robertson's chemistry (`k₁ = 0.04`, `k₂ = 3·10⁷`,
//!   `k₃ = 10⁴`) on a uniform 4-node field, `T = 40`, against SciPy Radau
//!   (`rtol = 1e−13`, `atol = 1e−20`, SciPy 1.18.1; agrees with the Hairer–Wanner
//!   reference to `1e−13`): per-species relative error `≤ 10·rtol` for
//!   `rtol ∈ {1e−3, 1e−5, 1e−7}`, `|Σyₖ − 1| ≤ 1e−12`, `min yₖ ≥ −1e−12`.

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use semiflow::{
    Brusselator, ChernoffFunction, FitzHughNagumo, GrayScott, Grid1D, GridFn1D, Growth, Kinetics,
    Nagumo, ReactionDiffusion, ReactionSolver, RosenbrockOptions, ScratchPool, SemiflowError,
    Species, SpeciesEngine,
};

/// The identity "diffusion" (the gates isolate the reaction flow).
struct Identity;

impl ChernoffFunction<f64> for Identity {
    type S = GridFn1D<f64>;
    fn apply_into(
        &self,
        _tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        dst.values.copy_from_slice(&src.values);
        Ok(())
    }
    fn order(&self) -> u32 {
        u32::MAX
    }
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }
}

/// `u′ = Cu` with `C = [[a, b], [b, a]]`, `a ± b = −1, −10⁶`; analytic Jacobian
/// unless `fd`.
struct StiffLinear {
    fd: bool,
}

const A: f64 = (-1e6 - 1.0) / 2.0;
const B: f64 = (-1e6 + 1.0) / 2.0;

impl Kinetics<f64> for StiffLinear {
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
            du[i] = A * u[i] + B * u[n + i];
            du[n + i] = B * u[i] + A * u[n + i];
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

/// Evolve `u0` (one value per species, uniform on 4 nodes) over `[0, t]` in `n`
/// Strang steps; returns the per-species node-0 values.
fn evolve(
    kin: Box<dyn Kinetics<f64>>,
    solver: ReactionSolver,
    substeps: u32,
    u0: &[f64],
    (t, n): (f64, usize),
) -> Result<Vec<f64>, SemiflowError> {
    let grid = Grid1D::new(0.0, 1.0, 4).expect("grid");
    let engines: Vec<SpeciesEngine<'static, f64, GridFn1D<f64>>> = u0
        .iter()
        .map(|_| Box::new(Identity) as SpeciesEngine<'static, f64, GridFn1D<f64>>)
        .collect();
    let rd = ReactionDiffusion::new(engines, kin)?
        .with_reaction_solver(solver)
        .with_reaction_substeps(substeps);
    let fields = u0
        .iter()
        .map(|&v| GridFn1D::new(grid, vec![v; 4]).expect("field"))
        .collect();
    let src = Species::new(fields);
    let mut dst = src.clone();
    rd.evolve_into(0.0, t, n, &src, &mut dst, &mut ScratchPool::new())?;
    Ok(dst.fields.iter().map(|f| f.values[0]).collect())
}

/// `e^{TC}u₀` in the eigenbasis `(1, 1)/√2`, `(1, −1)/√2`.
fn linear_exact(u0: [f64; 2], t: f64) -> [f64; 2] {
    let (p, m) = ((u0[0] + u0[1]) / 2.0, (u0[0] - u0[1]) / 2.0);
    let (ep, em) = ((-1e6 * t).exp() * p, (-t).exp() * m);
    [ep + em, ep - em]
}

fn rel_sup(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |a, w| a.max(w.abs()));
    got.iter()
        .zip(want)
        .fold(0.0_f64, |a, (g, w)| a.max((g - w).abs()))
        / scale
}

fn ros(rtol: f64, atol: f64) -> ReactionSolver {
    ReactionSolver::Rosenbrock(RosenbrockOptions::default().with_rtol(rtol).with_atol(atol))
}

#[test]
fn g_stiff_kinetics_linear() {
    let u0 = [1.0, 0.3];
    let exact = linear_exact(u0, 1.0);
    let rk4 = evolve(
        Box::new(StiffLinear { fd: false }),
        ReactionSolver::Rk4,
        1,
        &u0,
        (1.0, 10),
    );
    eprintln!("G_STIFF_KINETICS_LINEAR rk4: {rk4:?}");
    let blown = rk4.as_ref().map_or(true, |v| rel_sup(v, &exact) > 1e6);
    assert!(blown, "RK4 at hλ = −1e4 must blow up: {rk4:?}");
    // Single ROS2 steps: every step accepted whole (rtol 0.5, atol 1e3).
    let errs: Vec<f64> = [20, 40, 80]
        .iter()
        .map(|&n| {
            let got = evolve(
                Box::new(StiffLinear { fd: false }),
                ros(0.5, 1e3),
                1,
                &u0,
                (1.0, n),
            )
            .expect("ros2");
            rel_sup(&got, &exact)
        })
        .collect();
    let slopes = [(errs[0] / errs[1]).log2(), (errs[1] / errs[2]).log2()];
    eprintln!("G_STIFF_KINETICS_LINEAR fixed-step errors {errs:?} slopes {slopes:?}");
    assert!(slopes.iter().all(|s| (1.8..=2.2).contains(s)), "{slopes:?}");
    for rtol in [1e-4, 1e-6, 1e-8] {
        let got = evolve(
            Box::new(StiffLinear { fd: false }),
            ros(rtol, 1e-14),
            1,
            &u0,
            (1.0, 1),
        )
        .expect("ros2");
        let err = rel_sup(&got, &exact);
        eprintln!("G_STIFF_KINETICS_LINEAR adaptive rtol={rtol:e}: err {err:.3e}");
        assert!(err <= 10.0 * rtol, "rtol={rtol:e}: {err:e}");
    }
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
    let u0 = [1.0, 0.3];
    let run = |fd: bool| {
        evolve(
            Box::new(StiffLinear { fd }),
            ros(1e-8, 1e-14),
            1,
            &u0,
            (1.0, 1),
        )
        .expect("ros2")
    };
    let lin = rel_sup(&run(true), &run(false));
    eprintln!("G_STIFF_KINETICS_FD_JAC linear: {lin:.3e}");
    assert!(lin <= 1e-7, "{lin:e}");
}

#[test]
fn g_stiff_robertson() {
    for (fd, rtols) in [(false, [1e-3, 1e-5, 1e-7]), (true, [1e-3, 1e-5, 1e-7])] {
        for rtol in rtols {
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
