//! End-to-end v0.2.0 verification demo.
//!
//! Solves `∂_t u = ½·∂_xx u + ½·∂_x u` from `u(0,x) = exp(-x²)` on [-10, 10]
//! over `t ∈ [0, 1]` using `StrangSplit<DiffusionChernoff, DriftReactionChernoff>`.
//!
//! Compares the Strang result against the closed-form oracle
//!     `u(t,x) = (1+2αt)^(-1/2) · exp(-(x+βt)²/(1+2αt))`,    α=β=½
//! at three n values to demonstrate empirical second-order convergence.
//!
//! Also runs v0.1.0 `ShiftChernoff1D` on the same problem for direct comparison.
//!
//! Run with:  `cargo run --release --example strang_advdiff_demo -p semiflow-core`

use semiflow::{
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, ShiftChernoff1D,
    StrangSplit,
};

const ALPHA: f64 = 0.5;
const BETA: f64 = 0.5;
const T_FINAL: f64 = 1.0;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;

/// Closed-form solution of `∂_t u = α·∂_xx u + β·∂_x u`, `u(0,x) = exp(-x²)`.
/// Galilean-reduced heat equation: `∂_t v = α·∂_yy v`, `v(0,y)=exp(-y²)`.
/// Gaussian-initial-datum solution: `v(t,y) = (1+4αt)^(-1/2)·exp(-y²/(1+4αt))`.
/// Reverse Galilean substitution `y = x + βt`. For α=β=0.5, t=1: denom = 3.
fn oracle(x: f64) -> f64 {
    let denom = 1.0 + 4.0 * ALPHA * T_FINAL;
    let arg = (x + BETA * T_FINAL).powi(2) / denom;
    denom.sqrt().recip() * (-arg).exp()
}

fn sup_norm_err(u_n: &GridFn1D, grid: Grid1D) -> f64 {
    let mut max_err: f64 = 0.0;
    for i in 0..u_n.values.len() {
        let x = grid.x_at(i);
        max_err = max_err.max((u_n.values[i] - oracle(x)).abs());
    }
    max_err
}

fn run_strang(n: usize, n_nodes: usize) -> f64 {
    let grid = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid OK");
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    // v0.3.0 (ADR-0008 Amendment 1, ζ-A): a_prime = a_double_prime = |_| 0.0 for constant α
    // (a' ≡ a'' ≡ 0 ⇒ S(s) = id AND τ²-correction = 0 ⇒ D_ζ = D_γ = K = v0.2.2 bit-equal).
    let diff = DiffusionChernoff::new(|_| ALPHA, |_| 0.0_f64, |_| 0.0_f64, ALPHA, grid);
    let drift = DriftReactionChernoff::new(|_| BETA, |_| 0.0, 0.0, grid);
    let strang = StrangSplit::new(diff, drift);

    let semi = ChernoffSemigroup::new(strang, n).expect("n>=1");
    let u_n = semi.evolve(T_FINAL, &u0).expect("evolve OK");
    sup_norm_err(&u_n, grid)
}

fn run_v0_1_baseline(n: usize, n_nodes: usize) -> f64 {
    let grid = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid OK");
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    let func = ShiftChernoff1D::new(|_| ALPHA, |_| BETA, |_| 0.0, 0.0, grid);
    let semi = ChernoffSemigroup::new(func, n).expect("n>=1");
    let u_n = semi.evolve(T_FINAL, &u0).expect("evolve OK");
    sup_norm_err(&u_n, grid)
}

fn print_header() {
    println!("=== semiflow-core v0.2.0 — real-world verification ===");
    println!();
    println!("PDE: ∂_t u = (1/2)·∂_xx u + (1/2)·∂_x u");
    println!("IC:  u(0,x) = exp(-x²)");
    println!("BC:  reflective on Ω = [-10, 10]");
    println!("T:   1.0");
    println!("Oracle: u(t,x) = (1+4αt)^(-1/2) · exp(-(x+βt)²/(1+4αt))");
    println!("        with α=β=0.5, t=1 → u(1,x) = (1/√3)·exp(-(x+0.5)²/3)");
}

fn print_convergence_table() {
    println!();
    println!("--- v0.2.0 StrangSplit (order-2) — N=100000 ---");
    let mut prev: Option<f64> = None;
    for n in [100_usize, 1000, 10000] {
        let err = run_strang(n, 100_000);
        let ratio = prev.map_or(0.0, |p| p / err);
        println!(
            "  n={n:5}  sup-norm err = {err:.4e}  (ratio vs prev: {ratio:.2}× — expect 100× per 10× n)"
        );
        prev = Some(err);
    }
    println!();
    println!("--- v0.1.0 ShiftChernoff1D (order-1) — N=1000 ---");
    let mut prev: Option<f64> = None;
    for n in [100_usize, 1000, 10000] {
        let err = run_v0_1_baseline(n, 1000);
        let ratio = prev.map_or(0.0, |p| p / err);
        println!(
            "  n={n:5}  sup-norm err = {err:.4e}  (ratio vs prev: {ratio:.2}× — expect 10× per 10× n)"
        );
        prev = Some(err);
    }
}

fn print_gate_comparison() {
    println!();
    println!("--- Direct gate comparison (G1 < 1e-4, G2 < 1e-6) ---");
    let g1_strang = run_strang(100, 100_000);
    let g2_strang = run_strang(1000, 100_000);
    let g1_legacy = run_v0_1_baseline(100, 1000);
    let g2_legacy = run_v0_1_baseline(1000, 1000);
    println!(
        "  G1: Strang  = {:.4e}  (gate <1e-4, margin {:.0}×){}",
        g1_strang,
        1e-4 / g1_strang,
        if g1_strang < 1e-4 { "  ✓" } else { "  ✗" }
    );
    println!(
        "  G2: Strang  = {:.4e}  (gate <1e-6, margin {:.0}×){}",
        g2_strang,
        1e-6 / g2_strang,
        if g2_strang < 1e-6 { "  ✓" } else { "  ✗" }
    );
    println!("  G1-legacy:    {g1_legacy:.4e}  (v0.1.0 ShiftChernoff vs same advdiff oracle)");
    println!("  G2-legacy:    {g2_legacy:.4e}  (v0.1.0 ShiftChernoff vs same advdiff oracle)");
    println!();
    println!("Speed-up Strang vs legacy at the same problem:");
    println!(
        "  n=100:   {:.0}× more accurate  (Strang uses N=100000, legacy N=1000)",
        g1_legacy / g1_strang
    );
    println!("  n=1000:  {:.0}× more accurate", g2_legacy / g2_strang);
}

fn main() {
    print_header();
    print_convergence_table();
    print_gate_comparison();
}
