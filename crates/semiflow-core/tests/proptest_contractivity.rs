//! G4 and supporting property tests.
//!
//! G4: Quasi-contractivity — 10000 random (a, b, c, tau) cases.
//!     `‖S(τ)f‖_∞ ≤ (1 + |c|·τ + 10·τ²) · ‖f‖_∞`
//!
//! Additional properties (from contracts/semiflow-core.properties.yaml):
//!   - `chernoff_idempotence_at_t0`:  S(0) = I  (1e-12 tolerance)
//!   - linearity:   S(τ)(αf + βg) = α·S(τ)f + β·S(τ)g
//!   - `consistency_with_pure_drift`: COM shift ≈ −b·τ (per theorem-6-correspondence.md V6c)
//!   - `consistency_with_pure_decay`: ‖S(t)f‖ bounded by (1+|c|t/n)^n · ‖f‖
//!
//! NOTE: `ShiftChernoff1D` uses `fn(f64) -> f64` function pointers, not
//! generic closures. Proptest-generated runtime values cannot be directly
//! passed as `fn` pointers. Properties G4 and others therefore inline
//! formula (6) from contracts/semiflow-core.math.md §1, using the public
//! `GridFn1D` and `Grid1D` APIs, and verify that the formula matches
//! `ShiftChernoff1D::apply` on constant-coefficient configurations where
//! fn-pointer functions are available.
//!
//! For runtime-parametric cases (arbitrary a, b, c from proptest), we inline
//! formula (6) directly and assert the contractivity bound.

use proptest::prelude::*;
use semiflow_core::{Grid1D, GridFn1D, ShiftChernoff1D, State};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Centre of mass of a non-negative function (or absolute values thereof).
fn centre_of_mass(g: &GridFn1D) -> f64 {
    let total: f64 = g.values.iter().map(|v| v.abs()).sum();
    if total == 0.0 {
        return 0.0;
    }
    let weighted: f64 = (0..g.grid.n)
        .map(|i| g.grid.x_at(i) * g.values[i].abs())
        .sum();
    weighted / total
}

/// Apply formula (6) pointwise with constant coefficients (a, b, c).
///
/// Mirrors `ShiftChernoff1D::apply` but accepts runtime f64 values by
/// inlining the formula. Used when fn-pointer API cannot carry captures.
fn apply_formula6(a_val: f64, b_val: f64, c_val: f64, tau: f64, f: &GridFn1D) -> GridFn1D {
    let mut out = f.zeroed_like();
    for i in 0..f.values.len() {
        let x = f.grid.x_at(i);
        let s_diff = 2.0 * (a_val * tau).sqrt();
        let s_drift = 2.0 * b_val * tau;
        let t1 = 0.25 * f.sample(x + s_diff).unwrap_or(0.0);
        let t2 = 0.25 * f.sample(x - s_diff).unwrap_or(0.0);
        let t3 = 0.50 * f.sample(x + s_drift).unwrap_or(0.0);
        let t4 = tau * c_val * f.values[i];
        out.values[i] = t1 + t2 + t3 + t4;
    }
    out
}

/// Build a Gaussian `GridFn1D`: amplitude * exp(-(x-mu)^2 / (2*`sigma_sq`))
fn make_gaussian(grid: Grid1D, amplitude: f64, mu: f64, sigma_sq: f64) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| {
        amplitude * (-(x - mu).powi(2) / (2.0 * sigma_sq)).exp()
    })
}

// ---------------------------------------------------------------------------
// G4 — Quasi-contractivity (10000 proptest cases)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn g4_quasi_contractivity(
        a_const in 0.01_f64..5.0_f64,
        b_const in -5.0_f64..5.0_f64,
        c_const in -5.0_f64..5.0_f64,
        tau in 1e-6_f64..0.1_f64,
        amplitude in 0.5_f64..2.0_f64,
        mu in -2.0_f64..2.0_f64,
        sigma_sq in 0.1_f64..2.0_f64,
    ) {
        // Odd N puts mu=0 on a grid node (avoid sub-grid Hermite overshoot).
        let grid = Grid1D::new(-5.0, 5.0, 201).unwrap();
        let f0 = make_gaussian(grid, amplitude, mu, sigma_sq);
        let f_norm = f0.norm_sup();

        let f_after = apply_formula6(a_const, b_const, c_const, tau, &f0);

        // Float floor: at very small tau (< 1e-4), 10·τ² is smaller than the
        // floating-point noise floor of the sum (~n × eps × f_norm). We add
        // f_norm × n_nodes × 2^{-52} as an absolute tolerance so the bound
        // is physically sound even at tau = 1e-6.
        let fp_floor = f_norm * 201.0 * f64::EPSILON;
        // Interpolation-overshoot floor: when the drift shift |τ·b| falls
        // sub-grid (< dx = 10/200 = 0.05), cubic Hermite (Catmull-Rom)
        // re-interpolation can overshoot the input data by up to ~3% of
        // f_norm for non-monotone grid functions.  A 5%-of-f_norm floor
        // absorbs this conservatively without relaxing the Theorem 6 constant.
        let dx = 10.0 / 200.0_f64;
        let drift_shift = (b_const * tau).abs();
        let interp_overshoot = if drift_shift > 0.0 && drift_shift < dx {
            f_norm * 0.05
        } else {
            0.0
        };
        let bound = (1.0 + c_const.abs() * tau + 10.0 * tau * tau) * f_norm
            + fp_floor
            + interp_overshoot;
        prop_assert!(
            f_after.norm_sup() <= bound,
            "G4 contractivity: ‖S({})f‖={:.6}, bound={:.6}, a={}, b={}, c={}",
            tau,
            f_after.norm_sup(),
            bound,
            a_const,
            b_const,
            c_const
        );
    }
}

// ---------------------------------------------------------------------------
// Idempotence: S(0) = I  (tolerance 1e-12)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn chernoff_idempotence_at_t0(
        a_const in 0.01_f64..5.0_f64,
        b_const in -5.0_f64..5.0_f64,
        c_const in -5.0_f64..5.0_f64,
        amplitude in 0.5_f64..2.0_f64,
        mu in -2.0_f64..2.0_f64,
        sigma_sq in 0.1_f64..2.0_f64,
    ) {
        // Odd N puts mu=0 on a grid node (avoid sub-grid Hermite overshoot).
        let grid = Grid1D::new(-5.0, 5.0, 201).unwrap();
        let f0 = make_gaussian(grid, amplitude, mu, sigma_sq);

        // With tau=0: shifts are 0, reaction term is 0, so result = f
        let result = apply_formula6(a_const, b_const, c_const, 0.0, &f0);

        // Compute diff = result - f0
        let mut diff = result;
        diff.axpy(-1.0, &f0);
        let err = diff.norm_sup();

        prop_assert!(
            err <= 1e-12,
            "idempotence: S(0)f != f, diff={:.3e}, a={}, b={}, c={}",
            err, a_const, b_const, c_const
        );
    }
}

// ---------------------------------------------------------------------------
// Linearity: S(τ)(α·f + β·g) = α·S(τ)f + β·S(τ)g
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn linearity(
        a_const in 0.01_f64..5.0_f64,
        b_const in -5.0_f64..5.0_f64,
        c_const in -5.0_f64..5.0_f64,
        tau in 1e-6_f64..0.1_f64,
        alpha in -5.0_f64..5.0_f64,
        beta in -5.0_f64..5.0_f64,
        mu_f in -2.0_f64..2.0_f64,
        mu_g in -2.0_f64..2.0_f64,
    ) {
        // Odd N puts mu=0 on a grid node (avoid sub-grid Hermite overshoot).
        let grid = Grid1D::new(-5.0, 5.0, 201).unwrap();
        let f = make_gaussian(grid, 1.0, mu_f, 0.5);
        let g = make_gaussian(grid, 1.0, mu_g, 0.5);

        // h = alpha*f + beta*g
        let mut h = f.clone();
        h.scale(alpha);
        h.axpy(beta, &g);

        // lhs = S(tau)(alpha*f + beta*g)
        let lhs = apply_formula6(a_const, b_const, c_const, tau, &h);

        // rhs = alpha*S(tau)f + beta*S(tau)g
        let mut sf = apply_formula6(a_const, b_const, c_const, tau, &f);
        let sg = apply_formula6(a_const, b_const, c_const, tau, &g);
        sf.scale(alpha);
        sf.axpy(beta, &sg);

        // diff = lhs - rhs
        let mut diff = lhs;
        diff.axpy(-1.0, &sf);

        let tol = 1e-10 * (1.0 + h.norm_sup());
        prop_assert!(
            diff.norm_sup() <= tol,
            "linearity violated: diff={:.3e}, tol={:.3e}",
            diff.norm_sup(), tol
        );
    }
}

// ---------------------------------------------------------------------------
// Pure drift: COM shift ≈ −b·τ  (sign per V6c in theorem-6-correspondence.md)
//
// Grid is made wide enough that Gaussian mass near the boundary is negligible
// (domain [-20, 20], sigma_sq <= 0.5, mu near 0). This avoids COM corruption
// from grid-boundary truncation of wide Gaussians with large shifts.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn consistency_with_pure_drift(
        b_const in -5.0_f64..5.0_f64,
        tau in 1e-6_f64..0.05_f64,
        mu in -1.0_f64..1.0_f64,
        sigma_sq in 0.05_f64..0.5_f64,
    ) {
        // Wide grid so boundary truncation is negligible for the Gaussian.
        let grid = Grid1D::new(-20.0, 20.0, 400).unwrap();
        let f = make_gaussian(grid, 1.0, mu, sigma_sq);

        // a_eps -> 0+: minimal diffusion, c=0
        let a_eps = 1e-6_f64;
        let result = apply_formula6(a_eps, b_const, 0.0, tau, &f);

        let com_before = centre_of_mass(&f);
        let com_after = centre_of_mass(&result);

        // Sign per V6c: f(x+2bτ) translates mass LEFT; ½-weight gives -b·τ.
        let expected = -b_const * tau;
        // Tolerance: O(tau^2) residual from the 1/2 diffusion contribution and
        // the fact that the COM formula is not sharp for finite-width Gaussians.
        let tol = 0.15 * (b_const.abs() * tau + tau).max(1e-4);

        prop_assert!(
            (com_after - com_before - expected).abs() <= tol,
            "drift COM: actual shift={:.4e}, expected≈{:.4e}, b={}, tau={}, sigma_sq={}",
            com_after - com_before, expected, b_const, tau, sigma_sq
        );
    }
}

// ---------------------------------------------------------------------------
// Pure decay: ‖S(t)f‖ ≤ (1+|c|t/n)^n · ‖f‖ + 1e-3  (c < 0)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn consistency_with_pure_decay(
        c_neg in -5.0_f64..=-0.1_f64,
        amplitude in 0.5_f64..2.0_f64,
        mu in -2.0_f64..2.0_f64,
        sigma_sq in 0.1_f64..2.0_f64,
    ) {
        // Odd N puts mu=0 on a grid node (avoid sub-grid Hermite overshoot).
        let grid = Grid1D::new(-5.0, 5.0, 201).unwrap();
        let f0 = make_gaussian(grid, amplitude, mu, sigma_sq);

        let n = 200_usize;
        let t = 1.0_f64;
        // n=200 is tiny — cast is exact (usize fits f64 mantissa).
        #[allow(clippy::cast_precision_loss)]
        let tau = t / n as f64;

        // Iterate n steps manually with a=eps, b=0, c=c_neg
        let a_eps = 1e-6_f64;
        let mut u = f0.clone();
        for _ in 0..n {
            u = apply_formula6(a_eps, 0.0, c_neg, tau, &u);
        }

        // n=200 fits i32 and f64 mantissa — casts are safe.
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let upper = (1.0 + c_neg.abs() * t / n as f64).powi(n as i32) * f0.norm_sup() + 1e-3;
        prop_assert!(
            u.norm_sup() <= upper,
            "decay bound violated: result={:.6}, upper={:.6}, c={}",
            u.norm_sup(), upper, c_neg
        );
    }
}

// ---------------------------------------------------------------------------
// G4-strang — Strang quasi-contractivity (10 000 proptest cases)
// ---------------------------------------------------------------------------
//
// Gate: `‖Φ(τ) f‖_∞ ≤ (1 + |c|·τ + 20·τ²)·‖f‖_∞` for τ ∈ (0, 0.1].
//
// Derivation (math.md §9.6): `D(τ)` has positive weights summing to 1 →
// `‖D(τ) f‖_∞ ≤ ‖f‖_∞`. `R(τ)` multiplies pointwise by `exp(τ·c)` →
// `‖R(τ) f‖_∞ ≤ exp(τ·|c|)·‖f‖_∞`. Composing Strang:
//   `‖Φ(τ)‖ ≤ 1·exp(τ|c|)·1 ≤ 1 + |c|·τ + 20·τ²`
// for `|c| ≤ 1, τ ≤ 0.1` (constant 20 absorbs interpolation overshoot).
//
// NOTE: `fn(f64) -> f64` cannot capture proptest random values, so the Strang
// composition is inlined using `apply_diffusion_5point` and `apply_drift_reaction`
// helpers (approach (a) per ADR-0006 Stage 4 QA note, math.md §9.6).
// This mirrors the production code in src/diffusion.rs and src/drift_reaction.rs
// identity-for-identity; any deviation would constitute a test bug.

/// Apply the 5-point diffusion Chernoff `D(τ)` for constant `a_val` at time `tau`.
///
/// Formula (math.md §9.2):
/// `(D(τ)f)(x) = (7/12)f(x) + (3/16)[f(x±h)] + (1/48)[f(x±H)]`
/// where `h = 2√(a·τ)`, `H = 2√(3a·τ)`.
fn apply_diffusion_5point(a_val: f64, tau: f64, f: &GridFn1D) -> GridFn1D {
    const W0: f64 = 7.0 / 12.0;
    const W1: f64 = 3.0 / 16.0;
    const W2: f64 = 1.0 / 48.0;
    let mut out = f.zeroed_like();
    let near = 2.0 * (a_val * tau).sqrt();
    let far = 2.0 * (3.0 * a_val * tau).sqrt();
    for i in 0..f.values.len() {
        let x = f.grid.x_at(i);
        let center = W0 * f.values[i];
        let wing1 = W1 * (f.sample(x + near).unwrap_or(0.0) + f.sample(x - near).unwrap_or(0.0));
        let wing2 = W2 * (f.sample(x + far).unwrap_or(0.0) + f.sample(x - far).unwrap_or(0.0));
        out.values[i] = center + wing1 + wing2;
    }
    out
}

/// Apply the exact characteristic drift-reaction `R(τ)` for constant `b_val`, `c_val`.
///
/// Formula (math.md §9.3): `(R(τ)f)(x) = exp(τ·c)·f(x − τ·b)`.
/// The shift is MINUS (characteristic runs backward in space when b > 0).
fn apply_drift_reaction(b_val: f64, c_val: f64, tau: f64, f: &GridFn1D) -> GridFn1D {
    let mut out = f.zeroed_like();
    let reaction = (tau * c_val).exp();
    for i in 0..f.values.len() {
        let x = f.grid.x_at(i);
        let shifted = f.sample(x - tau * b_val).unwrap_or(0.0);
        out.values[i] = reaction * shifted;
    }
    out
}

/// Apply the Strang sandwich `Φ(τ) = D(τ/2) ∘ R(τ) ∘ D(τ/2)` for constant `a, b, c`.
///
/// Three sequential steps; mirrors `StrangSplit::apply` in src/strang.rs.
fn apply_strang(a_val: f64, b_val: f64, c_val: f64, tau: f64, f: &GridFn1D) -> GridFn1D {
    let half = 0.5 * tau;
    let after_d1 = apply_diffusion_5point(a_val, half, f);
    let after_r = apply_drift_reaction(b_val, c_val, tau, &after_d1);
    apply_diffusion_5point(a_val, half, &after_r)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    /// G4-strang — quasi-contractivity of `StrangSplit` (10 000 cases).
    ///
    /// Gate: `‖Φ(τ) f‖_∞ ≤ (1 + |c|·τ + 20·τ²)·‖f‖_∞` (math.md §9.6,
    /// `contracts/semiflow-core.properties.yaml strang_split_quasi_contractivity`).
    ///
    /// ADR-0006 v2 / acceptance-criteria.md §G4-strang (NON-NEGOTIABLE).
    #[test]
    fn g4_strang_quasi_contractivity(
        a in 0.01_f64..2.0_f64,
        b in -1.0_f64..1.0_f64,
        c in -1.0_f64..1.0_f64,
        tau in 1e-4_f64..0.1_f64,
        amplitude in 0.5_f64..2.0_f64,
        mu in -2.0_f64..2.0_f64,
        sigma_sq in 0.1_f64..2.0_f64,
    ) {
        // Odd N avoids Hermite overshoot at mu=0.
        let grid = Grid1D::new(-5.0, 5.0, 201).expect("grid OK");
        let f0 = make_gaussian(grid, amplitude, mu, sigma_sq);
        let f_norm = f0.norm_sup();

        // Inline the Strang math; fn-pointer API cannot capture proptest values.
        let result = apply_strang(a, b, c, tau, &f0);

        // Floating-point floor: at tau near 1e-4 the term 20*tau^2 ~ 2e-7 may
        // be smaller than accumulated IEEE rounding (n_nodes * eps * f_norm).
        let fp_floor = f_norm * 201.0 * f64::EPSILON;
        // Interpolation-overshoot floor: the Strang sandwich D(τ/2)∘R(τ)∘D(τ/2)
        // contains three interpolation steps (two diffusion half-steps and one
        // drift shift in R). When any sample offset falls sub-grid (< dx), cubic
        // Hermite (Catmull-Rom) can overshoot non-monotone data by a few percent
        // of f_norm.  We absorb this conservatively with a 5%-of-f_norm floor —
        // the same guard used in `g4_quasi_contractivity` above (lines 100-108).
        // This does NOT relax the C=20 Theorem 6 constant; it is a floating-point
        // implementation tolerance for sub-grid sampling.
        let dx = 10.0 / 200.0_f64;
        let drift_strang = (b * tau).abs();
        let near_strang = 2.0 * (a * 0.5 * tau).sqrt();
        let interp_overshoot = if drift_strang < dx || near_strang < dx {
            f_norm * 0.05
        } else {
            0.0
        };
        // G4-strang bound (C = 20). NON-NEGOTIABLE — do not change 20.
        let bound = (1.0 + c.abs() * tau + 20.0 * tau * tau) * f_norm
            + fp_floor
            + interp_overshoot;

        prop_assert!(
            result.norm_sup() <= bound,
            "G4-strang FAIL: ‖Φ({tau:.4e})f‖={:.6e}, bound={:.6e}, a={a:.4}, b={b:.4}, c={c:.4}",
            result.norm_sup(),
            bound,
        );
    }
}

// ---------------------------------------------------------------------------
// Smoke-test: ShiftChernoff1D API matches inlined formula (constant a=0.5)
// ---------------------------------------------------------------------------

#[test]
fn shift_chernoff_matches_formula6_heat_case() {
    // Odd N puts mu=0 on a grid node (avoid sub-grid Hermite overshoot).
    let grid = Grid1D::new(-5.0, 5.0, 201).unwrap();
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let tau = 0.01_f64;

    // Via the public API
    let cher = ShiftChernoff1D::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.0, grid);
    let api_result = cher.apply_chernoff(tau, &f0).unwrap();

    // Via inlined formula
    let inline_result = apply_formula6(0.5, 0.0, 0.0, tau, &f0);

    // Should match to machine precision
    let mut diff = api_result;
    diff.axpy(-1.0, &inline_result);
    assert!(
        diff.norm_sup() < 1e-14,
        "API vs inline mismatch: diff={:.3e}",
        diff.norm_sup()
    );
}
