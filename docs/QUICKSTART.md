# Quickstart — Solving the heat equation

> Current release: **v0.9.0-beta** (first public beta). For a full type catalogue and feature flag reference see
> [`crates/semiflow/README.md`](../crates/semiflow/README.md).

We numerically integrate `∂_t u = (1/2)·∂_xx u` from `u_0(x) = exp(-x²)` to
`t = 1`, and compare against the closed-form Gaussian heat kernel
`u(t, x) = (1 + 2t)^{-1/2} exp(-x² / (1 + 2t))`.

At `t = 1` the oracle is `3^{-1/2} exp(-x²/3)`.

## Full example

```rust
use semiflow_core::{Grid1D, GridFn1D, ShiftChernoff1D, ChernoffSemigroup};

fn main() {
    // Uniform grid: [-10, 10] with N=1000 nodes.
    // Defaults: BoundaryPolicy::Reflect, InterpKind::CubicHermite.
    let grid = Grid1D::new(-10.0, 10.0, 1000)
        .expect("grid bounds and node count are valid");

    // Initial condition: u_0(x) = exp(-x^2).
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    // Operator L = (1/2) d^2/dx^2.
    // ShiftChernoff1D::new(a, b, c, c_norm_bound, grid)
    let chernoff = ShiftChernoff1D::new(
        |_| 0.5_f64,  // a(x) = 1/2  (diffusion)
        |_| 0.0_f64,  // b(x) = 0    (no drift)
        |_| 0.0_f64,  // c(x) = 0    (no reaction)
        0.0,          // sup |c|  (used for growth estimate)
        grid,
    );

    // Iterate (S(t/n))^n with n=100 steps to t=1.
    let semigroup = ChernoffSemigroup::new(chernoff, 100)
        .expect("n=100 satisfies the n >= 1 precondition");
    let u_t = semigroup.evolve(1.0, &u0)
        .expect("evolve should not fail for valid inputs");

    // Compare to oracle: u(1,x) = (3)^{-1/2} exp(-x^2/3).
    let inv_sqrt3 = (3.0_f64).sqrt().recip();
    let mut max_err: f64 = 0.0;
    for i in 0..u_t.values.len() {
        let x = grid.x_at(i);
        let oracle = inv_sqrt3 * (-(x * x) / 3.0).exp();
        let err = (u_t.values[i] - oracle).abs();
        if err > max_err {
            max_err = err;
        }
    }
    println!("max sup-norm error: {:.3e}", max_err);
    // Output: max sup-norm error: 3.207e-4
}
```

## What is happening

1. `Grid1D::new(-10.0, 10.0, 1000)` creates a uniform grid with 1000 nodes and
   spacing `dx ≈ 0.02`. The default boundary policy is `Reflect`; the default
   sub-grid interpolation is `CubicHermite` (Catmull-Rom).

2. `ShiftChernoff1D` encodes formula (6) of Theorem 6 (Remizov 2025):

   ```text
   (S(τ) f)(x) = ¼ f(x + 2√(a(x)·τ))
               + ¼ f(x − 2√(a(x)·τ))
               + ½ f(x + 2·b(x)·τ)
               + τ·c(x)·f(x)
   ```

   For `a = 0.5`, `b = c = 0` the shift is `±√(2τ)` ≈ ±0.14 grid units at
   `τ = 0.01` (n=100, t=1), well within the Catmull-Rom stencil's accuracy.

3. `ChernoffSemigroup::evolve(t, &f)` applies `S(t/n)` exactly `n` times,
   threading the state forward. The error bound from Theorem 6 (inequality 9)
   scales as `O(t²/n)`. With `n=100` the empirical error is ≈ 3.2e-4.

## Convergence sweep

```rust
for &n in &[25_usize, 50, 100, 200, 400, 1000] {
    let sc   = ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.0, grid);
    let semi = ChernoffSemigroup::new(sc, n)
        .expect("n >= 1 always holds here");
    let u_t  = semi.evolve(1.0, &u0)
        .expect("evolve should not fail for valid inputs");
    let err  = /* compute sup-norm vs oracle as above */ 0.0_f64;
    println!("n={n:5}  err={err:.3e}");
}
// Expected (first-order O(1/n) convergence, slope ≈ −1.00):
// n=   25  err=1.3e-3
// n=   50  err=6.4e-4
// n=  100  err=3.2e-4
// n=  200  err=1.6e-4
// n=  400  err=8.0e-5
// n= 1000  err=3.1e-5
```

## Non-trivial coefficients

Vary the coefficients to model `L = a(x)∂² + b(x)∂`:

```rust
let sc = ShiftChernoff1D::new(
    |x| 0.5 + 0.1 * x.tanh(),  // space-varying diffusion
    |x| -0.2 * x,               // linear drift (Ornstein-Uhlenbeck-like)
    |_| 0.0,
    0.0,
    grid,
);
```

Theorem 6 requires `a(x) > 0` everywhere and `a, b, c` uniformly bounded
with bounded derivatives up to order 3. The library validates `a(x_i) >= 0`
at each grid node during `apply`; callers are responsible for ellipticity.

## Next steps

- See [contracts/semiflow-core.math.md](../contracts/semiflow-core.math.md) for the
  full mathematical specification of formula (6) and the convergence bound.
- See [crates/semiflow/tests/heat_kernel.rs](../crates/semiflow/tests/heat_kernel.rs)
  for the gate tests (G1 at n=100, G2 at n=1000) run in CI.
- Read the API docs: `cargo doc --open -p semiflow`.

---

## v0.2.0 — order-2 with `StrangSplit`

The v0.1.0 example above uses `ShiftChernoff1D`, which has first-order global
convergence (O(1/n) error). The v0.2.0 release ships `StrangSplit`, an
operator-splitting composer that achieves **order-2** (O(1/n²) error) by
symmetrizing a pure-diffusion step with an exact drift+reaction step.

The v0.2.0 acceptance gates are tighter than v0.1.0: G1 `< 1e-4` at `n=100`
(was 5e-4) and G2 `< 1e-6` at `n=1000` (was 5e-5). These are the original
PRD targets, now achievable with Strang order-2.

### Example — advection-diffusion `∂_t u = (1/2) ∂_xx u + (1/2) ∂_x u`

```rust
use semiflow_core::{
    Grid1D, GridFn1D, ChernoffSemigroup,
    DiffusionChernoff, DriftReactionChernoff, StrangSplit,
};

fn main() {
    // Uniform grid: [-10, 10] with N=100_000 nodes.
    // The fine grid pushes the cubic Hermite spatial-discretization floor
    // below the tighter v0.2.0 tolerance gates.
    let grid = Grid1D::new(-10.0, 10.0, 100_000)
        .expect("grid bounds and node count are valid");

    // Initial condition: u_0(x) = exp(-x^2).
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    // Diffusion operator A = (1/2) d^2/dx^2.
    // v0.3.0 (ADR-0008 Amendment 1, ζ-A):
    //   DiffusionChernoff::new(a, a_prime, a_double_prime, a_norm_bound, grid)
    // For constant `a`, pass `|_| 0.0_f64` for BOTH a_prime AND a_double_prime
    // (a' ≡ a'' ≡ 0; bit-equal to v0.2.2 by sympy gate Z_const-a).
    let diff = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, grid);

    // Drift+reaction operator B = (1/2) d/dx + 0.
    // DriftReactionChernoff::new(b, c, c_norm_bound, grid)
    let drift = DriftReactionChernoff::new(|_| 0.5_f64, |_| 0.0, 0.0, grid);

    // Strang sandwich: Phi(tau) = D(tau/2) o R(tau) o D(tau/2).
    let strang = StrangSplit::new(diff, drift);

    // Iterate (Phi(t/n))^n with n=1000 steps to t=1.
    let semigroup = ChernoffSemigroup::new(strang, 1000)
        .expect("n=1000 satisfies the n >= 1 precondition");
    let u_t = semigroup.evolve(1.0, &u0)
        .expect("evolve should not fail for valid inputs");

    // Oracle: u(t,x) = (1+2*alpha*t)^{-1/2} * exp(-(x+beta*t)^2 / (1+2*alpha*t))
    // with alpha=beta=0.5, t=1 => u(1,x) = (2)^{-1/2} * exp(-(x+0.5)^2 / 2).
    let inv_sqrt2 = 2.0_f64.sqrt().recip();
    let mut max_err: f64 = 0.0;
    for i in 0..u_t.values.len() {
        let x = grid.x_at(i);
        let oracle = inv_sqrt2 * (-((x + 0.5) * (x + 0.5)) / 2.0).exp();
        let err = (u_t.values[i] - oracle).abs();
        if err > max_err {
            max_err = err;
        }
    }
    println!("max sup-norm error: {:.3e}", max_err);
    // Output: max sup-norm error: 2.676e-9   (G2 gate: < 1e-6)
}
```

### Convergence table (advection-diffusion, N=100_000)

| n | sup-norm error | gate |
|---|---------------|------|
| 100 | ≈ 2.7e-7 | G1: < 1e-4 |
| 1000 | ≈ 2.7e-9 | G2: < 1e-6 |

The empirical log-log slope over `n ∈ {32, 64, 128, 256, 512, 1024}` is
`-2.004`, confirming order-2 convergence (G3-strang gate: slope ≤ -1.95).

---

## 2D heat (v0.5.0)

> See `crates/semiflow/examples/heat_2d_demo.rs` for the full demo
> (run via `cargo run --release --example heat_2d_demo -p semiflow`).

We numerically integrate `∂_t u = ½(∂_xx + ∂_yy)u` from
`u_0(x,y) = exp(-(x²+y²))` to `t = 1` on `[-10, 10]²`,
and compare against the closed-form 2D Gaussian heat oracle
`u(t,x,y) = (1+2t)^{-1} exp(-(x²+y²)/(1+2t))`.

At `t = 1` the oracle is `(1/3) exp(-(x²+y²)/3)`.

```rust
use semiflow_core::{
    ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D, GridFn2D, Strang2D,
};

fn main() {
    // Uniform 2D grid: [-10, 10]² with N=1000 nodes per axis (1M cells).
    let gx = Grid1D::new(-10.0, 10.0, 1000).expect("grid x OK");
    let gy = Grid1D::new(-10.0, 10.0, 1000).expect("grid y OK");
    let g  = Grid2D::new(gx, gy);  // infallible — Grid1D preconditions already validated

    // Initial condition: u_0(x, y) = exp(-(x² + y²)).
    let u0 = GridFn2D::from_fn(g, |x, y| (-(x * x + y * y)).exp());

    // Per-axis diffusion: L_x = L_y = ½∂²_z.
    // DiffusionChernoff::new(a, a_prime, a_double_prime, a_norm_bound, grid)
    // Constant a=0.5 ⇒ a'=a''=0 (ζ-A fast path, bit-equal to v0.2.2 by Z_const-a).
    let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

    // Palindromic Strang2D: Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2), global order 2.
    // Strang2D::new wraps each inner 1D function in an AxisLift automatically.
    let strang = Strang2D::new(cx, cy);
    let semi = ChernoffSemigroup::new(strang, 1000).expect("n >= 1");
    let u1 = semi.evolve(1.0, &u0).expect("evolve OK");

    // Compare to oracle: u(1,x,y) = (1/3) exp(-(x²+y²)/3).
    let nx = g.nx();
    let ny = g.ny();
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let oracle = (1.0 / 3.0) * (-(xi * xi + yj * yj) / 3.0).exp();
            let err = (u1.values[j * nx + i] - oracle).abs();
            if err > max_err { max_err = err; }
        }
    }
    println!("max sup-norm error: {:.3e}", max_err);
    // Smoke gate (n=50, N=1000): err < 5e-4
}
```

### What is happening

1. `Grid2D::new(gx, gy)` is infallible — each `Grid1D` has already been
   validated by `Grid1D::new` (requires `n >= 4`, finite endpoints,
   `xmin < xmax`). The 2D geometry invariant I-T2 is therefore implied.

2. `GridFn2D` stores values in a single `Vec<f64>` with row-major
   layout `idx(i, j) = j * nx + i` (x is the fast axis, I-T1).

3. `Strang2D::new(cx, cy)` internally wraps `cx` in `AxisLift<Axis::X>`
   and `cy` in `AxisLift<Axis::Y>`. The palindromic composition
   `Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2)` achieves global order 2 because
   `[Lx ⊗ I, I ⊗ Ly] = 0` (Theorem 7, `contracts/semiflow-core.math.md §10`).

4. `AxisLift::apply` for `Axis::X` sweeps each row (fixed j) as an
   independent 1D problem; for `Axis::Y` it sweeps each column (fixed i).
   Both re-use the same `DiffusionChernoff::apply` kernel.

### Acceptance gates (v0.5.0)

| Gate | n | N | sup-norm err | threshold |
|------|---|---|-------------|-----------|
| G1-2D | 100 | 200×200 | 3.687e-5 | < 5e-4 |
| G2-2D (slow-tests) | 1000 | 500×500 | 1.666e-5 | < 5e-5 |
| G3-2D slope (slow-tests) | 8…64 | 1000×1000 | −2.056 | ≤ −1.95 |

### Next steps

- See `contracts/semiflow-core.math.md` §10 for Theorem 7 and Lemma 10.2
  (the Y-independent reduction lemma that validates per-axis sweep).
- See `contracts/semiflow-core.tensor.yaml` for the NORMATIVE schema
  (schema_version 0.5.0) covering `Grid2D`, `GridFn2D`, `AxisLift`, `Strang2D`.
- See `docs/adr/0012-tensor-product-2d.md` for architectural rationale.
- Read the API docs: `cargo doc --open -p semiflow`.
