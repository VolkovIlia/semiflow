//! Proptest — Wave 3 (ADR-0043) `State<F>` algebraic axioms for `GridFn1D<f64>`.
//!
//! Ten invariants mandated by the Wave 3 contract (§5):
//!
//! - **I1** `axpy_into(0, x)` is a no-op on `self`.
//! - **I2** After `copy_from(&src)`, `self` is node-wise equal to `src`.
//! - **I3** After `zero_into()`, `norm_sup() == 0.0`.
//! - **I4** `len()` is invariant under `axpy_into`, `copy_from`, `zero_into`.
//! - **I5** `norm_sup() >= 0.0` for all finite inputs.
//! - **I6** `HilbertState::dot(a, a) == norm_sq(a)` (default impl identity).
//! - **I7** `dot(a, b) == dot(b, a)` (symmetry).
//! - **I8** After `zero_into()`, `dot(self, anything) == 0.0`.
//! - **I9** `scale_into(1.0)` is a no-op.
//! - **I10** `axpy_into(alpha, x)` then `axpy_into(-alpha, x)` round-trips to
//!   within IEEE-754 tolerance.
//!
//! Reference: `contracts/v2/wave3-state-trait.md` §5, `docs/adr/0043-state-trait-three-layer-split.md`.

use proptest::prelude::*;
use semiflow::{
    state::{HilbertState, State},
    Grid1D, GridFn1D,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Arbitrary small grid (4..=32 nodes) and random values.
fn arb_gridfn() -> impl Strategy<Value = GridFn1D<f64>> {
    (4_usize..=32_usize, -10.0_f64..=10.0_f64, 1.0_f64..=5.0_f64).prop_flat_map(
        |(n, center, spread)| {
            let vals = proptest::collection::vec((center - spread)..(center + spread), n);
            vals.prop_map(move |vs| {
                let grid = Grid1D::new(-1.0, 1.0, n).unwrap();
                let mut gf = GridFn1D::from_fn(grid, |_| 0.0);
                for (v, &src) in gf.values.iter_mut().zip(vs.iter()) {
                    *v = src;
                }
                gf
            })
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// I1: `axpy_into(0, x)` is a no-op on `self`.
    #[test]
    fn i1_axpy_zero_is_noop(mut u in arb_gridfn(), v in arb_gridfn()) {
        let n = u.len().min(v.len());
        // Resize both to same length for the shape-match contract.
        u.values.resize(n, 0.0);
        let mut v2 = v;
        v2.values.resize(n, 0.0);
        let grid = Grid1D::new(-1.0, 1.0, n).unwrap();
        u.grid = grid;
        v2.grid = grid;
        let original: Vec<f64> = u.values.clone();
        <GridFn1D<f64> as State<f64>>::axpy_into(&mut u, 0.0, &v2);
        prop_assert_eq!(&u.values, &original, "axpy_into(0, x) must be a no-op");
    }

    /// I2: After `copy_from(&src)`, `self` is node-wise equal to `src`.
    #[test]
    fn i2_copy_from_equals_src(src in arb_gridfn()) {
        let n = src.len();
        let grid = Grid1D::new(-1.0, 1.0, n).unwrap();
        let mut dst = GridFn1D::from_fn(grid, |_| 999.0);
        <GridFn1D<f64> as State<f64>>::copy_from(&mut dst, &src);
        prop_assert_eq!(&dst.values, &src.values, "copy_from: node-wise equality");
    }

    /// I3: After `zero_into()`, `norm_sup() == 0.0`.
    // After zeroing, all values are exactly 0.0 — bit-exact comparison is correct.
    #[allow(clippy::float_cmp)]
    #[test]
    fn i3_zero_into_zeroes_norm(mut u in arb_gridfn()) {
        <GridFn1D<f64> as State<f64>>::zero_into(&mut u);
        let ns = <GridFn1D<f64> as State<f64>>::norm_sup(&u);
        prop_assert!(ns == 0.0, "zero_into: norm_sup must be 0.0, got {ns}");
    }

    /// I4: `len()` is invariant under `axpy_into`, `copy_from`, `zero_into`.
    #[test]
    fn i4_len_invariant(mut u in arb_gridfn(), v in arb_gridfn()) {
        let n = u.len().min(v.len());
        u.values.resize(n, 0.0);
        let mut v2 = v;
        v2.values.resize(n, 0.0);
        let grid = Grid1D::new(-1.0, 1.0, n).unwrap();
        u.grid = grid;
        v2.grid = grid;

        let len_before = u.len();
        <GridFn1D<f64> as State<f64>>::axpy_into(&mut u, 1.0, &v2);
        prop_assert_eq!(u.len(), len_before, "len must be invariant after axpy_into");
        <GridFn1D<f64> as State<f64>>::copy_from(&mut u, &v2);
        prop_assert_eq!(u.len(), len_before, "len must be invariant after copy_from");
        <GridFn1D<f64> as State<f64>>::zero_into(&mut u);
        prop_assert_eq!(u.len(), len_before, "len must be invariant after zero_into");
    }

    /// I5: `norm_sup() >= 0.0` for all finite inputs.
    #[test]
    fn i5_norm_sup_nonneg(u in arb_gridfn()) {
        let ns = <GridFn1D<f64> as State<f64>>::norm_sup(&u);
        prop_assert!(ns >= 0.0, "norm_sup must be non-negative; got {}", ns);
    }

    /// I6: `dot(a, a) == norm_sq(a)` (default impl consistency).
    #[test]
    fn i6_norm_sq_via_dot(u in arb_gridfn()) {
        let dot_aa = <GridFn1D<f64> as HilbertState<f64>>::dot(&u, &u);
        let norm_sq = <GridFn1D<f64> as HilbertState<f64>>::norm_sq(&u);
        let diff = (dot_aa - norm_sq).abs();
        prop_assert!(
            diff == 0.0,
            "dot(a,a) must equal norm_sq(a); diff = {}",
            diff
        );
    }

    /// I7: `dot(a, b) == dot(b, a)` (symmetry).
    #[test]
    fn i7_dot_symmetric(a in arb_gridfn(), b in arb_gridfn()) {
        let n = a.len().min(b.len());
        let grid = Grid1D::new(-1.0, 1.0, n).unwrap();
        let mut a2 = a;
        a2.values.resize(n, 0.0);
        a2.grid = grid;
        let mut b2 = b;
        b2.values.resize(n, 0.0);
        b2.grid = grid;
        let dab = <GridFn1D<f64> as HilbertState<f64>>::dot(&a2, &b2);
        let dba = <GridFn1D<f64> as HilbertState<f64>>::dot(&b2, &a2);
        let rel = (dab - dba).abs() / (dab.abs().max(1e-300));
        prop_assert!(rel < 1e-14, "dot must be symmetric; diff_rel = {}", rel);
    }

    /// I8: After `zero_into()`, `dot(self, anything) == 0.0`.
    // dot(0, v) = Σᵢ 0·vᵢ = 0 exactly in IEEE 754.
    #[allow(clippy::float_cmp)]
    #[test]
    fn i8_dot_with_zero_is_zero(mut u in arb_gridfn(), v in arb_gridfn()) {
        let n = u.len().min(v.len());
        let grid = Grid1D::new(-1.0, 1.0, n).unwrap();
        u.values.resize(n, 0.0);
        u.grid = grid;
        let mut v2 = v;
        v2.values.resize(n, 0.0);
        v2.grid = grid;
        <GridFn1D<f64> as State<f64>>::zero_into(&mut u);
        let d = <GridFn1D<f64> as HilbertState<f64>>::dot(&u, &v2);
        prop_assert!(d == 0.0, "dot with zero state must be 0.0, got {d}");
    }

    /// I9: `scale_into(1.0)` is a no-op.
    #[test]
    fn i9_scale_one_is_noop(mut u in arb_gridfn()) {
        let original: Vec<f64> = u.values.clone();
        <GridFn1D<f64> as State<f64>>::scale_into(&mut u, 1.0);
        prop_assert_eq!(&u.values, &original, "scale_into(1.0) must be a no-op");
    }

    /// I10: `axpy_into(alpha, x); axpy_into(-alpha, x)` round-trips within ULP tolerance.
    #[test]
    fn i10_axpy_roundtrip(
        mut u in arb_gridfn(),
        alpha in -5.0_f64..=5.0_f64
    ) {
        let n = u.len();
        let grid = Grid1D::new(-1.0, 1.0, n).unwrap();
        u.grid = grid;
        let x = GridFn1D::from_fn(grid, |i| i * 0.1);
        let original: Vec<f64> = u.values.clone();
        <GridFn1D<f64> as State<f64>>::axpy_into(&mut u, alpha, &x);
        <GridFn1D<f64> as State<f64>>::axpy_into(&mut u, -alpha, &x);
        for (i, (&recovered, &orig)) in u.values.iter().zip(original.iter()).enumerate() {
            let diff = (recovered - orig).abs();
            // axpy round-trip: u += alpha*x; u -= alpha*x
            // Absolute error ≤ 4·ε·|alpha|·|x_i| + 4·ε·|u_i|.
            // Use mixed absolute+relative tolerance: tol = 1e-12 * (|orig| + |alpha| + 1).
            let tol = 1e-12 * (orig.abs() + alpha.abs() + 1.0);
            prop_assert!(
                diff < tol,
                "axpy round-trip failed at [{}]: orig={} recovered={} diff={} tol={}",
                i, orig, recovered, diff, tol
            );
        }
    }
}
