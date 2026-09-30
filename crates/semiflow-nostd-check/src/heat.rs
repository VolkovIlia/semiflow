//! 1-D heat-equation scenarios against the Gaussian / cosine closed forms.

use semiflow::{
    chernoff::ApplyChernoffExt, ChernoffSemigroup, Diffusion6thChernoff, DiffusionChernoff,
    DriftReactionChernoff, Evolver, Grid1D, GridFn1D, InterpKind, SemiflowFloat, ShiftChernoff1D,
    StrangSplit, TruncatedExp4WithCache, TruncatedExpDiffusionChernoff,
};

use crate::{check, digest_of, m, sup_diff, Digest, ScenarioResult};

/// Diffusion coefficient: `∂_t u = A ∂_xx u` with `A = ½`.
const A: f64 = 0.5;
/// Final time for the Gaussian scenarios.
const T: f64 = 1.0;
/// Domain half-width; the Gaussian tails are below 1e-40 at the edges.
const L: f64 = 10.0;
/// Nodes on `[-L, L]` (`dx = 0.1`). The quick-start doctest uses 1000; the
/// errors below are time-step dominated, so the coarser grid measures the same
/// value (`heat_shift1d`: `3.211e-4` at both sizes) at a fifth of the cost.
const NODES: usize = 201;

/// `u(t,x) = (1+2t)^{-1/2} exp(-x²/(1+2t))`: heat flow of `exp(-x²)` with `A = ½`.
fn gauss_heat(t: f64, x: f64) -> f64 {
    let s = 1.0 + 4.0 * A * t;
    m::exp(-x * x / s) / m::sqrt(s)
}

/// `exp(-x²)` sampled on `grid`.
fn gauss0(grid: Grid1D) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| m::exp(-x * x))
}

/// Sup-norm distance between `u` and `oracle(x_i)`.
fn err_vs(u: &GridFn1D, oracle: impl Fn(f64) -> f64) -> f64 {
    let exact: alloc::vec::Vec<f64> = (0..u.grid.n).map(|i| oracle(u.grid.x_at(i))).collect();
    sup_diff(&u.values, &exact)
}

/// `ShiftChernoff1D` + `ChernoffSemigroup` heat flow — the crate-level
/// "Quick start" doctest of `semiflow` (`src/lib.rs`): `[-10, 10]`, `n = 100`,
/// oracle `3^{-1/2} exp(-x²/3)`, tolerance `5e-4` (first-order scheme,
/// measured `3.211e-4`); 201 instead of 1000 nodes, see [`NODES`].
pub(crate) fn shift1d_heat() -> ScenarioResult {
    let grid = Grid1D::new(-L, L, NODES)?;
    let func = ShiftChernoff1D::new(|_| A, |_| 0.0, |_| 0.0, 0.0, grid);
    let semi = ChernoffSemigroup::new(func, 100)?;
    let u = semi.evolve(T, &gauss0(grid))?;
    check(err_vs(&u, |x| gauss_heat(T, x)), 5e-4, digest_of(&u.values))
}

/// `DiffusionChernoff` heat flow vs the same Gaussian oracle.
///
/// Setup of `tests/adaptive_chernoff_consistency.rs` (`DiffusionChernoff`,
/// `a = ½`, `[-10, 10]`, oracle `(1+2t)^{-1/2} exp(-x²/(1+2t))`) with fixed
/// steps; that test allows `5e-3` on 500 nodes. Here 201 nodes and `n = 20`:
/// measured `2.83e-5` (`1.08e-6` at 1000 nodes, `n = 100`; the scheme is
/// second order in time); tolerance `1e-4`.
pub(crate) fn diffusion_heat() -> ScenarioResult {
    let grid = Grid1D::new(-L, L, NODES)?;
    let func = DiffusionChernoff::new(|_| A, |_| 0.0, |_| 0.0, A, grid);
    let semi = ChernoffSemigroup::new(func, 20)?;
    let u = semi.evolve(T, &gauss0(grid))?;
    check(err_vs(&u, |x| gauss_heat(T, x)), 1e-4, digest_of(&u.values))
}

/// Constant decay rate for the Strang scenario (`c(x) = -½`).
const DECAY: f64 = -0.5;

/// `StrangSplit<DiffusionChernoff, DriftReactionChernoff>` with `b = 0`,
/// `c = -½`: `∂_t u = ½ u_xx - ½ u`, oracle = Gaussian heat flow × `exp(-t/2)`.
///
/// Adapter and grid family of `tests/strang_advdiff.rs` (G1: `1e-4`, which needs
/// its 100 000-node grid). With a constant reaction in place of the drift the
/// two parts commute, so the split adds no error: measured `4.17e-6` on 201
/// nodes, `n = 20` (two half-step diffusions per step); tolerance `2e-5`.
pub(crate) fn strang_heat_decay() -> ScenarioResult {
    let grid = Grid1D::new(-L, L, NODES)?;
    let diff = DiffusionChernoff::new(|_| A, |_| 0.0, |_| 0.0, A, grid);
    let react = DriftReactionChernoff::new(|_| 0.0, |_| DECAY, -DECAY, grid);
    let semi = ChernoffSemigroup::new(StrangSplit::new(diff, react), 20)?;
    let u = semi.evolve(T, &gauss0(grid))?;
    check(
        err_vs(&u, |x| gauss_heat(T, x) * m::exp(DECAY * T)),
        2e-5,
        digest_of(&u.values),
    )
}

/// `TruncatedExpDiffusionChernoff` heat flow of `cos(πx)` on `[-1, 1]`.
///
/// `tests/truncated_exp_heat_kernel.rs`, `n = N = 64` point of its diagonal
/// sweep: `T = 0.02`, `A = ½`, oracle `exp(-π²AT) cos(πx)`, CFL factor
/// `2τA/dx² ≈ 0.31 < 1`. That test gates the slope, not an absolute error;
/// measured `7.41e-5` here, tolerance `5e-4` (below `dx² ≈ 1e-3`).
pub(crate) fn truncated_exp_heat() -> ScenarioResult {
    use core::f64::consts::PI;
    const N: usize = 64;
    const T_TE: f64 = 0.02;
    let grid = Grid1D::new(-1.0, 1.0, N)?;
    #[allow(clippy::cast_precision_loss)] // N = 64
    let tau = T_TE / N as f64;
    let dx = grid.dx();
    if 2.0 * tau * A >= dx * dx {
        return Err(crate::Failure::Invariant(
            "truncated-exp CFL 2τA < dx² violated",
        ));
    }
    let func = TruncatedExpDiffusionChernoff::new(|_| A, |_| 0.0, |_| 0.0, A, grid);
    let mut u = GridFn1D::from_fn(grid, |x| m::cos(PI * x));
    for _ in 0..N {
        u = func.apply_chernoff(tau, &u)?;
    }
    let decay = m::exp(-PI * PI * A * T_TE);
    check(
        err_vs(&u, |x| decay * m::cos(PI * x)),
        5e-4,
        digest_of(&u.values),
    )
}

// ---------------------------------------------------------------------------
// Lane kernels (`semiflow::simd` portable lanes; AVX2/NEON under `simd`).
// Each runs in every build, so its digest pins the lane arithmetic too.
// ---------------------------------------------------------------------------

/// Gaussian heat flow of `func` with `steps` steps to `T`, checked against
/// [`gauss_heat`].
fn gauss_flow<C>(func: C, grid: Grid1D, steps: usize, tol: f64) -> ScenarioResult
where
    C: semiflow::ChernoffFunction<f64, S = GridFn1D>,
{
    let semi = ChernoffSemigroup::new(func, steps)?;
    let u = semi.evolve(T, &gauss0(grid))?;
    check(err_vs(&u, |x| gauss_heat(T, x)), tol, digest_of(&u.values))
}

/// `TruncatedExp4WithCache` (cached G⁴ stencil: 4-node lane kernel
/// `apply_g4_4nodes_simd`; default `SepticHermite` sampling at the ends).
///
/// Constructor of `tests/truncated_exp4_simd_bit_equal.rs` with constant
/// `a = ½`; `[-10, 10]`, 201 nodes. The 4th-order CFL `τ < 3dx²/(8A) = 0.0075`
/// needs `n ≥ 134` steps for `T = 1`; `n = 200` (`τ = 0.005`). Measured
/// `1.42e-6`; tolerance `3e-6`.
pub(crate) fn texp4_cached_heat() -> ScenarioResult {
    let grid = Grid1D::new(-L, L, NODES)?;
    let func = TruncatedExp4WithCache::with_cached_coefficients(|_| A, |_| 0.0, |_| 0.0, A, grid);
    gauss_flow(func, grid, 200, 3e-6)
}

/// Amplitude of the `a(x)` perturbation that makes `a′, a″ ≠ 0`, so
/// `Diffusion6thChernoff` evaluates its ζ⁶ correction (9-point FD lanes).
const EPS: f64 = 1e-3;

/// `Diffusion6thChernoff<f64>` (`fd9_simd`: `F64x4` 4+4+1 stencil).
///
/// Constructor of `tests/simd_bit_equal.rs` (`a = ½ + ε sin x`, `ε = 10⁻³`
/// keeps the correction term live); oracle is the constant-`A` Gaussian flow,
/// so the error includes the `O(ε)` model difference. `[-10, 10]`, 201 nodes,
/// `n = 20`. Measured `1.71e-4`; tolerance `4e-4`.
pub(crate) fn diffusion6_heat_f64() -> ScenarioResult {
    let grid = Grid1D::new(-L, L, NODES)?;
    let func = Diffusion6thChernoff::new(
        |x| A + EPS * m::sin(x),
        |x| EPS * m::cos(x),
        |x| -EPS * m::sin(x),
        A + EPS,
        grid,
    );
    gauss_flow(func, grid, 20, 4e-4)
}

/// `Diffusion6thChernoff<f32>` on a `CubicHermite` grid (`catmull_rom_f32`:
/// `F32x4` lanes; `fd9_simd_f32`: `F32x8` 8+1 stencil).
///
/// Constructor of `tests/simd_bit_equal_f32.rs` (`new_generic`,
/// `InterpKind::CubicHermite`) with the `a` of [`diffusion6_heat_f64`];
/// error measured in `f64` against [`gauss_heat`]. `[-10, 10]`, 201 nodes,
/// `n = 20`. Measured `1.84e-4`; tolerance `4e-4`.
pub(crate) fn diffusion6_catmull_f32() -> ScenarioResult {
    const A32: f32 = 0.5;
    const EPS32: f32 = 1e-3;
    const L32: f32 = 10.0;
    let grid = Grid1D::<f32>::new_generic(-L32, L32, NODES)?.with_interp(InterpKind::CubicHermite);
    let func = Diffusion6thChernoff::<f32>::new_generic(
        |x: f32| A32 + EPS32 * x.libm_sin(),
        |x: f32| EPS32 * x.libm_cos(),
        |x: f32| -EPS32 * x.libm_sin(),
        A + EPS,
        grid,
    );
    let evolver = Evolver::new(func, 20)?;
    #[allow(clippy::cast_possible_truncation)]
    let u0 =
        GridFn1D::<f32>::from_fn_generic(grid, |x| m::exp(-f64::from(x) * f64::from(x)) as f32);
    #[allow(clippy::cast_possible_truncation)] // T = 1
    let state = evolver.evolve(T as f32, &u0)?;
    let mut digest = Digest::new();
    let mut err: f64 = 0.0;
    for (i, &val) in state.values.iter().enumerate() {
        let val = f64::from(val);
        digest.push(val);
        let diff = m::abs(val - gauss_heat(T, f64::from(grid.x_at(i))));
        err = if diff.is_nan() || err.is_nan() {
            f64::NAN
        } else {
            err.max(diff)
        };
    }
    check(err, 4e-4, digest.finish())
}

/// `ShiftChernoff1D` on a `CubicHermite` grid (`catmull_rom`: `F64x4` lanes).
///
/// [`shift1d_heat`] with `InterpKind::CubicHermite` and `n = 20`. Measured
/// `1.58e-3`; tolerance `4e-3`.
pub(crate) fn cubic_hermite_heat() -> ScenarioResult {
    let grid = Grid1D::new(-L, L, NODES)?.with_interp(InterpKind::CubicHermite);
    let func = ShiftChernoff1D::new(|_| A, |_| 0.0, |_| 0.0, 0.0, grid);
    gauss_flow(func, grid, 20, 4e-3)
}

/// `ShiftChernoff1D` on an `OctonicHermite` grid (`fd_scaled_prime` of
/// `grid_chebyshev_octonic.rs`: `F64x4` 4+4+1 stencil).
///
/// [`shift1d_heat`] with `InterpKind::OctonicHermite` and `n = 20`. Measured
/// `1.61e-3`; tolerance `4e-3`.
pub(crate) fn octonic_heat() -> ScenarioResult {
    let grid = Grid1D::new(-L, L, NODES)?.with_interp(InterpKind::OctonicHermite);
    let func = ShiftChernoff1D::new(|_| A, |_| 0.0, |_| 0.0, 0.0, grid);
    gauss_flow(func, grid, 20, 4e-3)
}
