//! `boundary_demo` — tabulate `f(x) = exp(-x²)` sampled with all four boundary
//! policies on a small `Grid1D` (n=8, [-2, 2]) at x in {-3, -2.5, …, 2.5, 3}.
//!
//! Run with: `cargo run --example boundary_demo --package semiflow-core`

use semiflow::{BoundaryPolicy, Grid1D, GridFn1D};

fn main() {
    let grid_base = Grid1D::new(-2.0, 2.0, 8).expect("valid grid");
    let f_fn = |x: f64| (-x * x).exp();

    let policies = [
        ("Reflect        ", BoundaryPolicy::Reflect),
        ("ZeroExtend     ", BoundaryPolicy::ZeroExtend),
        ("Periodic       ", BoundaryPolicy::Periodic),
        ("LinearExtrap   ", BoundaryPolicy::LinearExtrapolate),
    ];

    // Print header.
    print!("{:>8}", "x");
    for (name, _) in &policies {
        print!("  {name:>16}");
    }
    println!();

    // Sample at x in {-3, -2.5, -2, -1.5, …, 2, 2.5, 3} (step 0.5).
    let mut x = -3.0_f64;
    while x <= 3.0 + 1e-9 {
        print!("{x:>8.2}");
        for (_, policy) in &policies {
            let grid = grid_base.with_boundary(*policy);
            let f = GridFn1D::from_fn(grid, f_fn);
            let v = f.sample(x).unwrap_or(f64::NAN);
            print!("  {v:>16.8}");
        }
        println!();
        x += 0.5;
    }
}
