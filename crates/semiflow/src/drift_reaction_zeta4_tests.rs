// Included into `drift_reaction_zeta4.rs` via `include!` inside `mod tests`.
// This gives access to `super::*` (PalindromicStrang, etc.) without duplication.
// DO NOT declare a `mod tests` here — this file is included inside one.

fn make_kernel(n: usize) -> DriftReactionZeta4Chernoff {
    let grid = Grid1D::new(-5.0, 5.0, n).expect("grid");
    let diff = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    DriftReactionZeta4Chernoff::new(diff, |x| -0.3 * x, |_| -0.3, |_| -0.3_f64, 0.3, grid)
}

#[test]
fn order_is_4() {
    assert_eq!(make_kernel(32).order(), 4);
}

#[test]
fn apply_into_produces_finite_output() {
    let k = make_kernel(64);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    let mut dst = f.zeroed_like();
    let mut scratch = ScratchPool::new();
    k.apply_into(0.01, &f, &mut dst, &mut scratch)
        .expect("apply_into must succeed");
    assert!(dst.values.iter().all(|v| v.is_finite()));
}

#[test]
fn apply_into_rejects_negative_tau() {
    let k = make_kernel(32);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    let mut dst = f.zeroed_like();
    let mut scratch = ScratchPool::new();
    assert!(k.apply_into(-0.01, &f, &mut dst, &mut scratch).is_err());
}

#[test]
fn apply_into_tau_zero_returns_src() {
    let k = make_kernel(64);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    let mut dst = f.zeroed_like();
    let mut scratch = ScratchPool::new();
    k.apply_into(0.0, &f, &mut dst, &mut scratch)
        .expect("tau=0 must succeed");
    let max_diff = f
        .values
        .iter()
        .zip(dst.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-12, "tau=0 must return src unchanged; max_diff={max_diff:.2e}");
}

/// Palindrome round-trip test for R_sym: verify R_sym(τ)·R_sym(-τ) = I + O(τ^p).
#[test]
fn r_sym_palindromic_to_order_4() {
    const GAMMA: f64 = 0.3;
    const C: f64 = -0.3;
    let grid = crate::Grid1D::new(-5.0, 5.0, 512).expect("grid");
    let f0 = crate::GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let diff = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let strang = PalindromicStrang {
        diffusion: diff,
        b: |x| -GAMMA * x,
        b_prime: |_| -GAMMA,
        c: |_| C,
        c_norm_bound: C.abs(),
        grid,
    };
    let mut scratch = ScratchPool::new();
    let n = f0.values.len();
    for &tau in &[0.5f64, 0.25, 0.125, 0.0625] {
        let mut f1 = crate::GridFn1D { grid, values: scratch.take_vec(n) };
        f1.values.resize(n, 0.0);
        for i in 0..n { f1.values[i] = strang.apply_r_sym_at(tau, &f0, i).unwrap(); }
        f1.grid = grid;
        let mut f2 = crate::GridFn1D { grid, values: scratch.take_vec(n) };
        f2.values.resize(n, 0.0);
        for i in 0..n { f2.values[i] = strang.apply_r_sym_at(-tau, &f1, i).unwrap(); }
        f2.grid = grid;
        let max_err = f0.values.iter().zip(f2.values.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        eprintln!("τ={tau:.4e}: R_sym round-trip err = {max_err:.4e}");
        assert!(max_err < 1e-5, "R_sym round-trip too large at τ={tau}: err={max_err:.4e}");
        scratch.return_vec(f1.values);
        scratch.return_vec(f2.values);
    }
}

/// With b=0, c=0 the kernel reduces to double-Richardson-over-K5 (pure diffusion).
#[test]
// Step count n (usize) cast to f64 to compute τ = T/n; n ≪ 2^52.
#[allow(clippy::cast_precision_loss)]
fn pure_diffusion_case_order4() {
    const T: f64 = 0.5;
    let grid = crate::Grid1D::new(-10.0, 10.0, 1024).expect("grid");
    let f0 = crate::GridFn1D::from_fn(grid, |x| libm::exp(-x * x));
    let diff = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let k = DriftReactionZeta4Chernoff::new(diff, |_| 0.0, |_| 0.0, |_| 0.0, 0.0, grid);
    let denom = 1.0 + 4.0 * T;
    let u_exact = crate::GridFn1D::from_fn(grid, |x| libm::exp(-x*x/denom)/denom.sqrt());
    let mut scratch = crate::ScratchPool::new();
    let ns = [4usize, 8, 16];
    let mut errs = [0.0f64; 3];
    for (idx, &n) in ns.iter().enumerate() {
        let tau = T / n as f64;
        let mut cur = f0.clone();
        let mut nxt = f0.zeroed_like();
        for _ in 0..n {
            k.apply_into(tau, &cur, &mut nxt, &mut scratch).unwrap();
            core::mem::swap(&mut cur, &mut nxt);
        }
        errs[idx] = cur.values.iter().zip(u_exact.values.iter())
            .map(|(a, b)| (a-b).abs()).fold(0.0_f64, f64::max);
    }
    let mid_slope = (errs[2]/errs[1]).ln() / 2.0_f64.ln();
    eprintln!("pure diffusion: errs={errs:?}, mid-pair slope={mid_slope:.3}");
    assert!(mid_slope < -3.5, "pure diffusion slope {mid_slope:.3} should be ≤ -3.5; errs={errs:?}");
}

/// Test order with small drift γ=0.01 (near pure diffusion, should show ~order 4).
#[test]
// Step count n (usize) cast to f64 to compute τ = T/n; n ≪ 2^52.
#[allow(clippy::cast_precision_loss)]
fn small_drift_order4() {
    const T: f64 = 0.5;
    const GAMMA: f64 = 0.01;
    let grid = crate::Grid1D::new(-8.0, 8.0, 512).expect("grid");
    let oracle = |t: f64, x: f64| -> f64 {
        let sigma2 = (1.0 - libm::exp(-2.0*GAMMA*t))/(2.0*GAMMA);
        let denom = 1.0 + 4.0*sigma2;
        libm::exp(-(x*libm::exp(-GAMMA*t)).powi(2)/denom)/denom.sqrt()
    };
    let f0 = crate::GridFn1D::from_fn(grid, |x| oracle(0.0, x));
    let u_exact = crate::GridFn1D::from_fn(grid, |x| oracle(T, x));
    let diff = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let k = DriftReactionZeta4Chernoff::new(diff, |x| -GAMMA*x, |_| -GAMMA, |_| 0.0, 0.0, grid);
    let mut scratch = crate::ScratchPool::new();
    let ns = [2usize, 4, 8, 16];
    let mut errs = [0.0f64; 4];
    for (idx, &n) in ns.iter().enumerate() {
        let tau = T / n as f64;
        let mut cur = f0.clone();
        let mut nxt = f0.zeroed_like();
        for _ in 0..n {
            k.apply_into(tau, &cur, &mut nxt, &mut scratch).unwrap();
            core::mem::swap(&mut cur, &mut nxt);
        }
        errs[idx] = cur.values.iter().zip(u_exact.values.iter())
            .map(|(a, b)| (a-b).abs()).fold(0.0_f64, f64::max);
    }
    let mid_slope = (errs[2]/errs[1]).ln() / 2.0_f64.ln();
    eprintln!("γ=0.01: mid-pair slope={mid_slope:.3}  errs={errs:?}");
    assert!(mid_slope < -3.5, "small-drift slope {mid_slope:.3} should be ≤ -3.5; errs={errs:?}");
}

/// Convergence order check with analytic oracle (OU + reaction).
#[test]
// Step count and array length (usize) cast to f64 for slope computation; values ≪ 2^52.
#[allow(clippy::cast_precision_loss)]
fn approx_order_4_ou_oracle() {
    const GAMMA: f64 = 0.3;
    const C: f64 = -0.3;
    const T: f64 = 0.5;
    let oracle = |t: f64, x: f64| -> f64 {
        let sigma2 = (1.0 - libm::exp(-2.0 * GAMMA * t)) / (2.0 * GAMMA);
        let denom = 1.0 + 4.0 * sigma2;
        let mu = x * libm::exp(-GAMMA * t);
        libm::exp(C * t) / denom.sqrt() * libm::exp(-mu * mu / denom)
    };
    let grid = crate::Grid1D::new(-8.0, 8.0, 1024).expect("grid");
    let f0 = crate::GridFn1D::from_fn(grid, |x| oracle(0.0, x));
    let u_exact = crate::GridFn1D::from_fn(grid, |x| oracle(T, x));
    let diff = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let k = DriftReactionZeta4Chernoff::new(
        diff, |x| -GAMMA * x, |_| -GAMMA, |_| C, C.abs(), grid,
    );
    let mut scratch = crate::ScratchPool::new();
    let ns = [2usize, 4, 8, 16];
    let mut errs = [0.0f64; 4];
    for (idx, &n) in ns.iter().enumerate() {
        let tau = T / n as f64;
        let mut cur = f0.clone();
        let mut nxt = f0.zeroed_like();
        for _ in 0..n {
            k.apply_into(tau, &cur, &mut nxt, &mut scratch).unwrap();
            core::mem::swap(&mut cur, &mut nxt);
        }
        errs[idx] = cur.values.iter().zip(u_exact.values.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    }
    let slope_mid = (errs[2] / errs[1]).ln() / 2.0_f64.ln();
    let m = ns.len() as f64;
    let lx: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sx: f64 = lx.iter().sum::<f64>();
    let sy: f64 = ly.iter().sum::<f64>();
    let sxx: f64 = lx.iter().map(|&x| x * x).sum();
    let sxy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    let ols = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    eprintln!("OU oracle: errs={errs:?}, mid-pair slope={slope_mid:.3}, OLS={ols:.3}");
    assert!(
        slope_mid < -3.2,
        "mid-pair slope {slope_mid:.3} < -3.2 expected for symmetric base; errs={errs:?}"
    );
}
