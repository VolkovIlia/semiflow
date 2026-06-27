// Included via `include!` from phi_action.rs #[cfg(test)] block.
// Fast (non-ignored) unit tests for phi_action / phi_action_batched.
// All items from phi_action.rs are in scope (same module): phi_action,
// phi_action_batched, PHI_MAX, GeneratorAction, ScratchPool.

// ---------------------------------------------------------------------------
// Minimal scalar generator: A = scalar multiplication by `a`
// dim=1, apply_generator: dst[0] = a * src[0], norm_bound = |a|.
// Exact answers: phi_k(τa) = Σ_{n≥0} (τa)^n / (n+k)!
// For k=0: e^{τa}.  For k=1: (e^{τa}-1)/(τa) when τa≠0.
// ---------------------------------------------------------------------------

struct ScalarGen {
    a: f64,
}

impl GeneratorAction<f64> for ScalarGen {
    fn dim(&self) -> usize { 1 }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        dst[0] = self.a * src[0];
    }
    fn norm_bound(&self) -> f64 { self.a.abs() }
}

/// Exact `phi_k(z)` for scalar z (Taylor series, 50 terms).
#[allow(clippy::cast_precision_loss)]
fn phi_k_exact(k: usize, z: f64) -> f64 {
    // Compute 1/k! first, then accumulate.
    let mut fact = 1.0_f64;
    for n in 0..k {
        fact *= (n + 1) as f64;
    }
    let mut term = 1.0 / fact;
    let mut sum = term;
    for n in 1..50_usize {
        term *= z / (n + k) as f64;
        sum += term;
        if term.abs() < 1e-17 * sum.abs() { break; }
    }
    sum
}

#[test]
fn phi0_scalar_unit_time() {
    // phi_0(tau * a) * 1.0 should equal e^{tau*a}
    // Tolerance: T_13 truncation at |τa|=1.0 is |z|^14/14! ≈ 1.15e-11.
    let op = ScalarGen { a: -2.0 };
    let tau = 0.5_f64;
    let v = [1.0_f64];
    let mut out = [0.0_f64];
    let mut scratch = ScratchPool::new();
    phi_action(&op, 0, tau, &v, &mut out, &mut scratch).unwrap();
    let expected = (-1.0_f64).exp(); // e^{0.5 * (-2.0)}
    assert!(
        (out[0] - expected).abs() < 1e-9,
        "phi0: got {}, expected {}", out[0], expected
    );
}

#[test]
fn phi1_scalar() {
    // phi_1(tau * a) * 1.0 = (e^{tau*a} - 1) / (tau*a)   [for tau*a != 0]
    // Tolerance: T_13 truncation at |τa|=1.2 is ~1.47e-10.
    let op = ScalarGen { a: -3.0 };
    let tau = 0.4_f64;
    let v = [1.0_f64];
    let mut out = [0.0_f64];
    let mut scratch = ScratchPool::new();
    phi_action(&op, 1, tau, &v, &mut out, &mut scratch).unwrap();
    let z = tau * op.a;
    let expected = phi_k_exact(1, z);
    assert!(
        (out[0] - expected).abs() < 1e-8,
        "phi1: got {}, expected {}", out[0], expected
    );
}

#[test]
fn phi2_scalar() {
    let op = ScalarGen { a: -1.5 };
    let tau = 0.6_f64;
    let v = [1.0_f64];
    let mut out = [0.0_f64];
    let mut scratch = ScratchPool::new();
    phi_action(&op, 2, tau, &v, &mut out, &mut scratch).unwrap();
    let z = tau * op.a;
    let expected = phi_k_exact(2, z);
    assert!(
        (out[0] - expected).abs() < 1e-11,
        "phi2: got {}, expected {}", out[0], expected
    );
}

#[test]
fn phi3_scalar() {
    // Tolerance: T_13 truncation at |τa|=1.4 is ~4.3e-10 (observed).
    let op = ScalarGen { a: -2.0 };
    let tau = 0.7_f64;
    let v = [1.0_f64];
    let mut out = [0.0_f64];
    let mut scratch = ScratchPool::new();
    phi_action(&op, 3, tau, &v, &mut out, &mut scratch).unwrap();
    let z = tau * op.a;
    let expected = phi_k_exact(3, z);
    assert!(
        (out[0] - expected).abs() < 1e-7,
        "phi3: got {}, expected {}", out[0], expected
    );
}

#[test]
fn batched_matches_single() {
    // phi_action_batched should match phi_action for each k.
    let op = ScalarGen { a: -2.5 };
    let tau = 0.5_f64;
    let v = [1.0_f64];
    let mut scratch = ScratchPool::new();

    let mut batched_out = [0.0_f64; 4]; // 4 * 1
    phi_action_batched(&op, 3, tau, &v, &mut batched_out, &mut scratch).unwrap();

    for (k, &batched_val) in batched_out.iter().enumerate() {
        let mut single_out = [0.0_f64];
        phi_action(&op, k, tau, &v, &mut single_out, &mut scratch).unwrap();
        assert!(
            (batched_val - single_out[0]).abs() < 1e-12,
            "k={k}: batched={}, single={}", batched_val, single_out[0]
        );
    }
}

#[test]
fn phi0_zero_tau_is_identity() {
    // phi_0(0) = 1, so phi_0(0*A)*v = v.
    let op = ScalarGen { a: -10.0 };
    let v = [3.0_f64];
    let mut out = [0.0_f64];
    let mut scratch = ScratchPool::new();
    phi_action(&op, 0, 0.0_f64, &v, &mut out, &mut scratch).unwrap();
    // e^0 = 1, so out = 1.0 * 3.0 = 3.0
    assert!((out[0] - 3.0).abs() < 1e-12, "phi0(0): got {}", out[0]);
}

#[test]
fn phi1_zero_tau_is_1() {
    // phi_1(0) = 1/1! = 1, so phi_1(0*A)*v = 1*v.
    let op = ScalarGen { a: -5.0 };
    let v = [2.0_f64];
    let mut out = [0.0_f64];
    let mut scratch = ScratchPool::new();
    phi_action(&op, 1, 0.0_f64, &v, &mut out, &mut scratch).unwrap();
    // phi_1(0) = 1, out = 2.0
    assert!((out[0] - 2.0).abs() < 1e-12, "phi1(0): got {}", out[0]);
}

#[test]
fn phi_action_domain_error_k_exceeds_max() {
    let op = ScalarGen { a: -1.0 };
    let v = [1.0_f64];
    let mut out = [0.0_f64];
    let mut scratch = ScratchPool::new();
    let err = phi_action(&op, PHI_MAX + 1, 0.5, &v, &mut out, &mut scratch);
    assert!(err.is_err(), "expected error for k > PHI_MAX");
}
