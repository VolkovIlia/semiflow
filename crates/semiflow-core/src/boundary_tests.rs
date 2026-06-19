// Tests for boundary policies — moved from boundary.rs (batch H5).
use super::{bc_index, bc_value, BoundaryHit, BoundaryPolicy};

// ------------------------------------------------------------------
// proptest: bc_index_dirichlet_neumann_robin_totality (1 000 cases)
// properties.yaml: I1 (no panic) + I5 (outside ⇒ Dirichlet/clamped)
// ------------------------------------------------------------------

use proptest::prelude::*;

/// Check I5 properties for the out-of-range case (extracted from proptest body).
///
/// Verifies:
/// - Dirichlet outside → `Dirichlet(v)` with the correct value
/// - Neumann outside → `Inside(0)` or `Inside(n-1)`
/// - Robin outside → `RobinSkew { reflected ∈ [0, n) }`
/// - [`bc_value`] round-trip for Dirichlet
fn assert_outside_hits(
    n: usize,
    idx: i64,
    v: f64,
    hit_d: BoundaryHit<f64>,
    hit_n: BoundaryHit<f64>,
    hit_r: BoundaryHit<f64>,
) -> Result<(), TestCaseError> {
    let dirichlet = BoundaryPolicy::Dirichlet { value: v };
    prop_assert!(
        matches!(hit_d, BoundaryHit::Dirichlet(_)),
        "expected Dirichlet hit outside grid, got {hit_d:?}"
    );
    if let BoundaryHit::Dirichlet(got) = hit_d {
        prop_assert!(
            (got - v).abs() < 1e-12,
            "Dirichlet value mismatch: got {got}, want {v}"
        );
    }
    prop_assert!(
        matches!(hit_n, BoundaryHit::Inside(_)),
        "Neumann outside should return Inside, got {hit_n:?}"
    );
    if let BoundaryHit::Inside(i) = hit_n {
        prop_assert!(
            i == 0 || i == n - 1,
            "Neumann clamp must be 0 or n-1={}, got {i}",
            n - 1
        );
    }
    prop_assert!(
        matches!(hit_r, BoundaryHit::RobinSkew { .. }),
        "Robin outside should return RobinSkew, got {hit_r:?}"
    );
    if let BoundaryHit::RobinSkew { reflected, .. } = hit_r {
        prop_assert!(
            reflected < n,
            "Robin reflected must be in [0, n={n}), got {reflected}"
        );
    }
    // bc_value round-trip: Dirichlet outside returns the value
    let vals = vec![0.0f64; n];
    // dx=0.0 is safe: Dirichlet does not use dx
    let got = bc_value(dirichlet, &vals, n, idx, 0.0);
    prop_assert!(
        (got - v).abs() < 1e-12,
        "bc_value round-trip: got {got}, want {v}"
    );
    Ok(())
}

/// Check I5 fast-path for the in-range case (extracted from proptest body).
///
/// All three policies must return `Inside(idx)` when `0 ≤ idx < n`.
fn assert_inside_hits(
    expected_i: usize,
    hit_d: BoundaryHit<f64>,
    hit_n: BoundaryHit<f64>,
    hit_r: BoundaryHit<f64>,
) -> Result<(), TestCaseError> {
    prop_assert!(
        matches!(hit_d, BoundaryHit::Inside(i) if i == expected_i),
        "Dirichlet in-range should be Inside({expected_i}), got {hit_d:?}"
    );
    prop_assert!(
        matches!(hit_n, BoundaryHit::Inside(i) if i == expected_i),
        "Neumann in-range should be Inside({expected_i}), got {hit_n:?}"
    );
    prop_assert!(
        matches!(hit_r, BoundaryHit::Inside(i) if i == expected_i),
        "Robin in-range should be Inside({expected_i}), got {hit_r:?}"
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]

    /// I1 — never panics on any (n, idx, value) in the spec range.
    /// I5 — out-of-range Dirichlet returns `Dirichlet(value)`;
    ///      out-of-range Neumann/Robin returns `Inside(0)` or `Inside(n-1)`.
    #[test]
    fn bc_index_dirichlet_neumann_robin_totality(
        n in 1usize..=1000,
        idx in -2000i64..=2000,
        v in -1e6f64..=1e6f64,
    ) {
        // n >= 1 but bc_index requires n >= 2 for Neumann's "n-1" (n=1 ⇒ n-1=0).
        // For n == 1 the grid is degenerate; skip gracefully rather than panic.
        if n < 2 {
            return Ok(());
        }

        let dirichlet = BoundaryPolicy::Dirichlet { value: v };
        let neumann    = BoundaryPolicy::<f64>::Neumann;
        let robin      = BoundaryPolicy::<f64>::Robin { alpha: 1.0, beta: 1.0 };

        // I1: no call panics.
        let hit_d = bc_index(dirichlet, n, idx);
        let hit_n = bc_index(neumann, n, idx);
        let hit_r = bc_index(robin, n, idx);

        #[allow(clippy::cast_possible_wrap)]
        let n_i = n as i64;

        if idx < 0 || idx >= n_i {
            assert_outside_hits(n, idx, v, hit_d, hit_n, hit_r)?;
        } else {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let expected_i = idx as usize;
            assert_inside_hits(expected_i, hit_d, hit_n, hit_r)?;
        }
    }
}

// --- Dirichlet variant: in-range returns Inside, out-of-range returns value ---
#[test]
fn dirichlet_in_range_returns_inside() {
    let n = 8usize;
    let hit = bc_index(BoundaryPolicy::Dirichlet { value: 42.0_f64 }, n, 3);
    assert_eq!(hit, BoundaryHit::Inside(3));
}

#[test]
fn dirichlet_out_left_returns_value() {
    let n = 8usize;
    let hit = bc_index(BoundaryPolicy::Dirichlet { value: 7.5_f64 }, n, -1);
    // Matches structurally because f64 has no Eq
    if let BoundaryHit::Dirichlet(v) = hit {
        assert!((v - 7.5).abs() < 1e-15);
    } else {
        panic!("expected Dirichlet hit, got {hit:?}");
    }
}

#[test]
fn dirichlet_out_right_returns_value() {
    let n = 8usize;
    // 3.14 is an arbitrary test value, not π — the test checks identity, not transcendence
    #[allow(clippy::approx_constant)]
    let val = 3.14_f64;
    let hit = bc_index(BoundaryPolicy::Dirichlet { value: val }, n, 10);
    if let BoundaryHit::Dirichlet(v) = hit {
        assert!((v - val).abs() < 1e-15);
    } else {
        panic!("expected Dirichlet hit, got {hit:?}");
    }
}

#[test]
fn dirichlet_bc_value_returns_constant() {
    let n = 8usize;
    let values = [1.0f64; 8];
    // dx=0.0 safe: Dirichlet does not use dx
    let result = bc_value(
        BoundaryPolicy::Dirichlet { value: 99.0 },
        &values,
        n,
        -2,
        0.0,
    );
    assert!((result - 99.0).abs() < 1e-15);
}

// --- Neumann variant: clamps to 0 or n-1 ---
#[test]
fn neumann_out_left_clamps_to_zero() {
    let n = 8usize;
    let hit = bc_index(BoundaryPolicy::<f64>::Neumann, n, -5);
    assert_eq!(hit, BoundaryHit::Inside(0));
}

#[test]
fn neumann_out_right_clamps_to_last() {
    let n = 8usize;
    let hit = bc_index(BoundaryPolicy::<f64>::Neumann, n, 15);
    assert_eq!(hit, BoundaryHit::Inside(n - 1));
}

#[test]
fn neumann_in_range_returns_inside() {
    let n = 8usize;
    let hit = bc_index(BoundaryPolicy::<f64>::Neumann, n, 4);
    assert_eq!(hit, BoundaryHit::Inside(4));
}

// --- Robin variant: in-range Inside, out-of-range RobinSkew (v6.2.3, ADR-0098 Am.2) ---
#[test]
fn robin_bc_index_clamp() {
    let n = 8usize;
    let bc = BoundaryPolicy::Robin {
        alpha: 1.0_f64,
        beta: 2.0_f64,
    };
    // In-range: Inside(idx)
    assert_eq!(bc_index(bc, n, 3), BoundaryHit::Inside(3));
    // idx=-5: depth=5, reflected=reflect_index(8,-5)=5
    assert!(matches!(
        bc_index(bc, n, -5),
        BoundaryHit::RobinSkew {
            reflected: 5,
            depth: 5
        }
    ));
    // idx=15: depth=15-(8-1)=8, reflected=reflect_index(8,15)=1
    assert!(matches!(
        bc_index(bc, n, 15),
        BoundaryHit::RobinSkew {
            reflected: 1,
            depth: 8
        }
    ));
}

// --- Existing variants: quick smoke tests ---
#[test]
fn reflect_in_range() {
    let hit = bc_index(BoundaryPolicy::<f64>::Reflect, 10, 5);
    assert_eq!(hit, BoundaryHit::Inside(5));
}

#[test]
fn zero_extend_out_of_range() {
    let hit = bc_index(BoundaryPolicy::<f64>::ZeroExtend, 10, 15);
    assert_eq!(hit, BoundaryHit::Zero);
}
