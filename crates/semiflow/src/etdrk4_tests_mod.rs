// Included via `include!` from etdrk4.rs `#[cfg(test)]` block.
// Items from etdrk4.rs are in scope: Etdrk4.

// Items already in scope from the parent etdrk4.rs use block:
// SemiflowError, GeneratorAction, Nonlinearity, ScratchPool.
use crate::nonlinearity::AllenCahn;

// ---------------------------------------------------------------------------
// Minimal helpers: scalar generator and zero nonlinearity
// ---------------------------------------------------------------------------

/// Scalar generator: A = scalar multiplication by `a`, dim = 1.
struct ScalarGen {
    a: f64,
}

impl GeneratorAction<f64> for ScalarGen {
    fn dim(&self) -> usize { 1 }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) { dst[0] = self.a * src[0]; }
    fn norm_bound(&self) -> f64 { self.a.abs() }
}

/// Nonlinearity that always outputs zero: `N(u) = 0`.
struct ZeroNl;

impl Nonlinearity<f64> for ZeroNl {
    fn eval(&self, _u: &[f64], n_out: &mut [f64]) -> Result<(), SemiflowError> {
        for x in n_out.iter_mut() { *x = 0.0; }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn etdrk4_step_linear_exact() {
    // With N = 0 the ETDRK4 formula reduces to e^{hL}·u (exact for linear ODE).
    // For scalar a = -1, h = 0.1: exact = e^{-0.1}.
    let a = -1.0_f64;
    let h = 0.1_f64;
    let driver = Etdrk4::new(ScalarGen { a }, ZeroNl, h).unwrap();

    let u = [1.0_f64];
    let mut u_next = [0.0_f64];
    let mut scratch = ScratchPool::new();
    driver.step(&u, &mut u_next, &mut scratch).unwrap();

    let exact = (a * h).exp(); // e^{-0.1} ≈ 0.904837418
    // |τa| = 0.1 → T_13 truncation error ~1e-14; expect 1e-12.
    assert!(
        (u_next[0] - exact).abs() < 1e-12,
        "linear exact: got {}, expected {exact}", u_next[0]
    );
}

#[test]
fn etdrk4_integrate_matches_step() {
    // integrate(u0, 1, …) must equal step(u0, …).
    let op = ScalarGen { a: -0.5 };
    let nl = AllenCahn::<f64>::new();
    let h = 0.05;
    let driver = Etdrk4::new(op, nl, h).unwrap();

    let u = [0.3_f64];
    let mut u_step = [0.0_f64];
    let mut u_int = [0.0_f64];
    let mut scratch = ScratchPool::new();

    driver.step(&u, &mut u_step, &mut scratch).unwrap();
    driver.integrate(&u, 1, &mut u_int, &mut scratch).unwrap();

    assert!(
        (u_step[0] - u_int[0]).abs() < 1e-15,
        "step vs integrate(1): {} != {}", u_step[0], u_int[0]
    );
}

#[test]
fn etdrk4_integrate_two_steps_vs_manual() {
    // integrate(u0, 2, …) must equal two manual step calls.
    let op = ScalarGen { a: -0.5 };
    let nl = AllenCahn::<f64>::new();
    let h = 0.02;
    let driver = Etdrk4::new(op, nl, h).unwrap();

    let u0 = [0.5_f64];
    let mut scratch = ScratchPool::new();

    // Two manual steps
    let mut u1 = [0.0_f64];
    let mut u2 = [0.0_f64];
    driver.step(&u0, &mut u1, &mut scratch).unwrap();
    driver.step(&u1, &mut u2, &mut scratch).unwrap();

    // integrate
    let mut u_int = [0.0_f64];
    driver.integrate(&u0, 2, &mut u_int, &mut scratch).unwrap();

    assert!(
        (u2[0] - u_int[0]).abs() < 1e-15,
        "two steps vs integrate(2): {} != {}", u2[0], u_int[0]
    );
}

#[test]
fn etdrk4_rejects_zero_dim() {
    struct ZeroDimGen;
    impl GeneratorAction<f64> for ZeroDimGen {
        fn dim(&self) -> usize { 0 }
        fn apply_generator(&self, _s: &[f64], _d: &mut [f64]) {}
        fn norm_bound(&self) -> f64 { 0.0 }
    }
    let err = Etdrk4::new(ZeroDimGen, ZeroNl, 0.1_f64);
    assert!(err.is_err(), "expected DomainViolation for dim=0");
}
