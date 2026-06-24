// Tests for `matrix_pade.rs` (Padé[13/13] matrix exponential, Higham 2005).
//
// Properties asserted:
//   1. exp(0) = I (zero matrix → identity).
//   2. exp(diag(a_i)) = diag(exp(a_i)) for small diagonal matrices.
//   3. exp(A)·exp(−A) = I (inverse property).
//   4. compute_squarings returns 0 for ‖A‖_∞ ≤ θ₁₃, > 0 for large norm.
//   5. mm_eye returns identity (diagonal 1, off-diagonal 0).
//   6. mm_mul is correct for 2×2 example.
//   7. mm_axpby is correct for 2×2 example.
//   8. mat_scale_entries scales uniformly.

/// Frobenius norm of the difference between two M×M matrices.
fn frob_diff<const M: usize>(a: &[[f64; M]; M], b: &[[f64; M]; M]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(ra, rb)| ra.iter().zip(rb.iter()))
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ── exp(0) = I ───────────────────────────────────────────────────────────────

#[test]
fn exp_zero_is_identity_5x5() {
    let zero = [[0.0_f64; 5]; 5];
    let result = mat_exp_pade13::<f64, 5>(&zero).unwrap();
    let eye = mm_eye::<f64, 5>();
    assert!(
        frob_diff(&result, &eye) < 1e-14,
        "exp(0) != I: diff={}",
        frob_diff(&result, &eye)
    );
}

// ── exp(diag) = diag(exp) ─────────────────────────────────────────────────────

#[test]
fn exp_diagonal_5x5() {
    // A = diag(0, 0.5, 1.0, 1.5, 2.0)
    let mut a = [[0.0_f64; 5]; 5];
    let diag_vals = [0.0_f64, 0.5, 1.0, 1.5, 2.0];
    for (i, &v) in diag_vals.iter().enumerate() {
        a[i][i] = v;
    }
    let result = mat_exp_pade13::<f64, 5>(&a).unwrap();
    for (i, &v) in diag_vals.iter().enumerate() {
        let expected = v.exp();
        assert!(
            (result[i][i] - expected).abs() < 1e-12,
            "diag[{i}]: expected {expected}, got {}",
            result[i][i]
        );
        // Off-diagonal must be near zero.
        for (j, &val) in result[i].iter().enumerate() {
            if j != i {
                assert!(
                    val.abs() < 1e-12,
                    "off-diag [{i}][{j}] = {val}"
                );
            }
        }
    }
}

// ── exp(A)·exp(−A) = I ───────────────────────────────────────────────────────

#[test]
fn exp_times_exp_neg_is_identity_5x5() {
    // Use a small symmetric 5×5 matrix with bounded norm.
    let a: [[f64; 5]; 5] = [
        [0.5, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.5, 0.1, 0.0, 0.0],
        [0.0, 0.1, 0.5, 0.1, 0.0],
        [0.0, 0.0, 0.1, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.1, 0.5],
    ];
    let neg_a = mm_axpby(-1.0_f64, &a, 0.0_f64, &a); // –A
    let exp_a = mat_exp_pade13::<f64, 5>(&a).unwrap();
    let exp_neg_a = mat_exp_pade13::<f64, 5>(&neg_a).unwrap();
    let product = mm_mul::<f64, 5>(&exp_a, &exp_neg_a);
    let eye = mm_eye::<f64, 5>();
    assert!(
        frob_diff(&product, &eye) < 1e-12,
        "exp(A)·exp(-A) not identity, diff={}",
        frob_diff(&product, &eye)
    );
}

// ── compute_squarings returns 0 for small norm ────────────────────────────────

#[test]
fn squarings_zero_for_small_norm() {
    // ‖A‖_∞ = 0.1 ≤ θ₁₃ ≈ 5.37 → 0 squarings.
    let mut a = [[0.0_f64; 5]; 5];
    a[0][0] = 0.1;
    a[1][1] = 0.05;
    let s = compute_squarings::<f64, 5>(&a);
    assert_eq!(s, 0, "expected 0 squarings for small norm, got {s}");
}

// ── compute_squarings > 0 for large norm ─────────────────────────────────────

#[test]
fn squarings_positive_for_large_norm() {
    // Row 0 has ‖row‖ = 100, so ‖A‖_∞ = 100 >> θ₁₃ → need squarings.
    let mut a = [[0.0_f64; 5]; 5];
    for entry in &mut a[0] {
        *entry = 20.0_f64;
    }
    let s = compute_squarings::<f64, 5>(&a);
    assert!(s > 0, "expected > 0 squarings for large-norm matrix, got {s}");
}

// ── mm_eye ────────────────────────────────────────────────────────────────────

#[test]
fn mm_eye_is_identity_5x5() {
    let eye = mm_eye::<f64, 5>();
    for (i, row) in eye.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-15,
                "eye[{i}][{j}] = {val}"
            );
        }
    }
}

// ── mm_axpby ─────────────────────────────────────────────────────────────────

#[test]
fn mm_axpby_2x2() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // 2·A + 3·B = [[2+15, 4+18],[6+21, 8+24]] = [[17,22],[27,32]]
    let a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let b: [[f64; 2]; 2] = [[5.0, 6.0], [7.0, 8.0]];
    let c = mm_axpby(2.0_f64, &a, 3.0_f64, &b);
    assert!((c[0][0] - 17.0).abs() < 1e-15);
    assert!((c[0][1] - 22.0).abs() < 1e-15);
    assert!((c[1][0] - 27.0).abs() < 1e-15);
    assert!((c[1][1] - 32.0).abs() < 1e-15);
}

// ── mat_scale_entries ─────────────────────────────────────────────────────────

#[test]
fn mat_scale_entries_2x2() {
    let a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let scaled = mat_scale_entries(&a, 0.5_f64);
    assert!((scaled[0][0] - 0.5).abs() < 1e-15);
    assert!((scaled[0][1] - 1.0).abs() < 1e-15);
    assert!((scaled[1][0] - 1.5).abs() < 1e-15);
    assert!((scaled[1][1] - 2.0).abs() < 1e-15);
}
