//! Subordination, gridless particles, Smolyak quadrature and reverse-mode AD.

use semiflow::{
    subordinated::{GammaSubordinator, LevySubordinator},
    SubordinatedChernoff,
};
use semiflow::{
    CheckpointSchedule, ChernoffFunction, DiffusionChernoff, DriftReactionChernoff, Dual, Grid1D,
    GridFn1D, GridFnND, GridND, GridlessChernoff, MeasureState, ParticleReduction, ReverseChernoff,
    ScratchPool, SmolyakGridND,
};

use crate::{check, digest_of, m, sup_diff, Digest, Failure, ScenarioResult};

/// Reaction coefficient `c(x)` as a plain function pointer.
type Reaction = fn(f64) -> f64;

/// Value at node 0 after 128 steps of `SubordinatedChernoff<DriftReaction,
/// Gamma(c = 1)>` with reaction `c ≡ −λ` on `f ≡ 1` (`T = 1`).
fn subordinated_run(c_fn: Reaction, lam: f64) -> Result<f64, Failure> {
    let grid = Grid1D::new(0.0, 1.0, 4)?;
    let base = DriftReactionChernoff::new(|_| 0.0, c_fn, lam, grid);
    let wrapper = SubordinatedChernoff::new(base, GammaSubordinator::new(1.0)?);
    let mut src = GridFn1D::from_fn(grid, |_| 1.0);
    let mut dst = GridFn1D::from_fn(grid, |_| 0.0);
    let mut scratch = ScratchPool::new();
    for _ in 0..128 {
        wrapper.apply_into(1.0 / 128.0, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values[0])
}

/// Gamma-subordinated semigroup `exp(−T φ(λ))`, `φ(λ) = ln(1 + λ)` (Laguerre
/// quadrature in `gen_quadrature.rs`).
///
/// `tests/subordinated_order1_slope.rs` (`G_SUBORD_ORDER1`), Gamma backend,
/// `λ ∈ {4, 16}`, `n = 128`: (A) `|f − exact| ≤ 5e-3·max(1, φ(λ))` — held to
/// `5e-3` flat here — and (B) closer to `exp(−Tφ(λ))` than to the linearised
/// `exp(−Tφ(1)λ)` by a factor 5.
pub(crate) fn subordinated_gamma() -> ScenarioResult {
    let sub = GammaSubordinator::new(1.0)?;
    let cases: [(Reaction, f64); 2] = [(|_| -4.0, 4.0), (|_| -16.0, 16.0)];
    let mut err: f64 = 0.0;
    let mut digest = Digest::new();
    for (c_fn, lam) in cases {
        let f = subordinated_run(c_fn, lam)?;
        digest.push(f);
        let exact = m::exp(-sub.laplace_exponent(lam));
        let wrong = m::exp(-sub.laplace_exponent(1.0) * lam);
        let e = m::abs(f - exact);
        if e.is_nan() || e >= 0.2 * m::abs(f - wrong) {
            return Err(Failure::Invariant(
                "subordinated: converged to linearised limit",
            ));
        }
        err = err.max(e);
    }
    check(err, 5e-3, digest.finish())
}

/// `GridlessChernoff` particle ensemble in D = 2 (`WeightedVoronoi`
/// reduction in `gridless_reduce.rs`): mass and mean are invariants.
///
/// The kernel's branches `x ± 2√(aτ)`, `x + 2bτ` (weights ¼, ¼, ½) move the
/// mean by exactly `bτ` per step, and the reduction preserves mass and first
/// moment exactly (`gridless_reduce.rs` docs; `tests/t_gridless.rs` sub-checks
/// `mass_conservation`, `voronoi_moment_match`). Oracle after 20 steps of
/// `τ = 0.05` from a unit Dirac at 0: mass 1, mean `b·t`.
pub(crate) fn gridless_moments() -> ScenarioResult {
    const B: [f64; 2] = [0.3, -0.2];
    let cap = 64;
    let ev = GridlessChernoff::<f64, 2>::new(
        [0.5, 0.5],
        B,
        0.0,
        ParticleReduction::WeightedVoronoi { cap },
    );
    let mut src = MeasureState::<f64, 2>::dirac([0.0, 0.0], 1.0);
    let mut dst = src.clone();
    let mut pool = ScratchPool::new();
    for _ in 0..20 {
        ev.apply_into(0.05, &src, &mut dst, &mut pool)?;
        core::mem::swap(&mut src, &mut dst);
    }
    if src.n_diracs() > cap || src.n_diracs() < 2 {
        return Err(Failure::Invariant("gridless: Dirac count outside [2, cap]"));
    }
    let mean = src.first_moment();
    let var = src.variance_per_axis();
    let err = m::abs(src.total_variation() - 1.0)
        .max(m::abs(mean[0] - B[0]))
        .max(m::abs(mean[1] - B[1]));
    #[allow(clippy::cast_precision_loss)] // n_diracs ≤ cap = 64
    let count = src.n_diracs() as f64;
    let digest = digest_of(&[
        src.total_variation(),
        mean[0],
        mean[1],
        src.second_moment(),
        var[0],
        var[1],
        count,
    ]);
    check(err, 1e-12, digest)
}

/// Step size and count for the Smolyak scenario (`t = 0.2`).
const SM_TAU: f64 = 0.05;
const SM_STEPS: u32 = 4;

/// `SmolyakGridND` (D = 2, default level `ℓ = D + 3`) heat flow of
/// `exp(−|x|²)` with `A = I`: oracle `(1+4t)^{-1} exp(−|x|²/(1+4t))`.
///
/// Kernel constructor as in `tests/g_smolyak_d5.rs` (there D = 5 with a
/// `tanh`-coupled `A`, self-convergence only); with constant `A = I` each step
/// is the exact heat convolution up to quadrature and interpolation error, so
/// a closed form applies. 41 nodes per axis on `[−5, 5]`, 4 steps of
/// `τ = 0.05`. Measured `3.70e-4`, dominated by the N-D interpolation at
/// `dx = 0.25` (`7.9e-5` at `dx = 0.125`); tolerance `1e-3`. Memory: the
/// kernel caches one `SquareMatrix` (fixed 36-slot storage, 288 bytes) per
/// node, about 0.5 MB here.
pub(crate) fn smolyak_heat_d2() -> ScenarioResult {
    let ax = Grid1D::new(-5.0, 5.0, 41)?;
    let grid = GridND::new([ax; 2])?;
    let kernel = SmolyakGridND::<f64, 2>::new(
        |_x: &[f64; 2], a| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            a.set(0, 1, 0.0);
            a.set(1, 0, 0.0);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| *b = [0.0; 2],
        |_x: &[f64; 2]| 0.0,
        grid.clone(),
    )?;
    let mut src = GridFnND::from_fn(grid.clone(), |x| m::exp(-(x[0] * x[0] + x[1] * x[1])));
    let mut dst = GridFnND::from_fn(grid.clone(), |_| 0.0);
    let mut pool = ScratchPool::new();
    for _ in 0..SM_STEPS {
        kernel.apply_into(SM_TAU, &src, &mut dst, &mut pool)?;
        core::mem::swap(&mut src, &mut dst);
    }
    let s = 1.0 + 4.0 * SM_TAU * f64::from(SM_STEPS);
    let exact = GridFnND::from_fn(grid, |x| m::exp(-(x[0] * x[0] + x[1] * x[1]) / s) / s);
    check(
        sup_diff(&src.values, &exact.values),
        1e-3,
        digest_of(&src.values),
    )
}

/// Parameters of `tests/g_reverse_ad.rs`, except 64 grid nodes instead of 128
/// (halves the emulated run time; the gradient identities do not depend on it).
const THETA0: f64 = 0.5;
const RAD_N: usize = 32;
const RAD_GRID: usize = 64;

fn a_seeded_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::variable(THETA0)
}
fn zero_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::constant(0.0)
}

fn dual_kernel() -> Result<DiffusionChernoff<Dual<f64>>, Failure> {
    let grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(-10.0), Dual::constant(10.0), RAD_GRID)?;
    Ok(DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual,
        zero_dual,
        zero_dual,
        THETA0,
        grid,
    ))
}

/// `J(θ) = ‖u_n‖²` for diffusivity `θ`, plain `f64` path.
fn loss_at(theta: f64, tau: f64) -> Result<f64, Failure> {
    let grid = Grid1D::new(-10.0, 10.0, RAD_GRID)?;
    let k = DiffusionChernoff::with_closure(move |_| theta, |_| 0.0, |_| 0.0, theta, grid);
    let mut u = GridFn1D::from_fn(grid, |x| m::exp(-x * x));
    for _ in 0..RAD_N {
        u = k.apply_f(tau, &u)?;
    }
    Ok(u.values.iter().map(|v| v * v).sum())
}

/// `dJ/dθ` by forward-mode `Dual<f64>`.
fn forward_grad(tau: f64) -> Result<f64, Failure> {
    let k = dual_kernel()?;
    let mut u = GridFn1D::from_fn_generic(k.grid, |x| Dual::constant(m::exp(-x.value * x.value)));
    for _ in 0..RAD_N {
        u = k.apply_f(Dual::constant(tau), &u)?;
    }
    Ok(u.values
        .iter()
        .fold(0.0, |acc, d| acc + 2.0 * d.value * d.tangent))
}

/// `ReverseChernoff::value_and_grad_k1` with `CheckpointSchedule::sqrt_n`.
///
/// `tests/g_reverse_ad.rs` (`G_REVERSE_AD_GRADIENT`): `θ = ½`, `T = 1`,
/// `n = 32`, 64 nodes on `[−10, 10]`; (i) reverse vs 4-point Richardson FD
/// (`h = 1e-3`) relative `< 1e-9`, (ii) reverse vs forward `Dual` relative
/// `< 1e-12` and not bit-identical. Reported error is (ii).
pub(crate) fn reverse_ad_sqrt_n() -> ScenarioResult {
    #[allow(clippy::cast_precision_loss)] // RAD_N = 32
    let tau = 1.0 / RAD_N as f64;
    let sched = CheckpointSchedule::sqrt_n(RAD_N);
    if sched.stride != 6 || sched.checkpoint_count() != 6 {
        return Err(Failure::Invariant(
            "sqrt_n(32): expected stride 6, 6 checkpoints",
        ));
    }
    let grid = Grid1D::new(-10.0, 10.0, RAD_GRID)?;
    let k64 = DiffusionChernoff::with_closure(|_| THETA0, |_| 0.0, |_| 0.0, THETA0, grid);
    let rc = ReverseChernoff::new(k64, dual_kernel()?, sched);
    let u0 = GridFn1D::from_fn(grid, |x| m::exp(-x * x));
    let target = GridFn1D::from_fn(grid, |_| 0.0);
    let (_, rev) = rc.value_and_grad_k1(tau, RAD_N, &u0, &target)?;
    let fwd = forward_grad(tau)?;
    let h = 1e-3;
    let fd = (-loss_at(THETA0 + 2.0 * h, tau)? + 8.0 * loss_at(THETA0 + h, tau)?
        - 8.0 * loss_at(THETA0 - h, tau)?
        + loss_at(THETA0 - 2.0 * h, tau)?)
        / (12.0 * h);
    let digest = digest_of(&[rev, fwd, fd]);
    check(m::abs(rev - fd) / m::abs(fd), 1e-9, digest)?;
    if rev.to_bits() == fwd.to_bits() {
        return Err(Failure::Invariant(
            "reverse == forward bit-exactly (tautology)",
        ));
    }
    check(m::abs(rev - fwd) / m::abs(fwd), 1e-12, digest)
}
