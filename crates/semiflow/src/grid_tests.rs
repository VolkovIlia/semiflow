// Property tests for `bc_index` (extracted per suckless ≤500-line cap).
use alloc::format; // required by prop_assert! in no_std context

use proptest::prelude::*;

use crate::boundary::{bc_index, BoundaryHit, BoundaryPolicy};

fn any_policy() -> impl Strategy<Value = BoundaryPolicy<f64>> {
    prop_oneof![
        Just(BoundaryPolicy::Reflect),
        Just(BoundaryPolicy::ZeroExtend),
        Just(BoundaryPolicy::Periodic),
        Just(BoundaryPolicy::LinearExtrapolate),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5_000))]

    /// `bc_index_totality` (I1): total over all (policy, n, idx).
    #[test]
    fn bc_index_totality(
        policy in any_policy(),
        n in 2usize..=10_000usize,
        idx in -1_000_000i64..=1_000_000i64,
    ) {
        let hit = bc_index(policy, n, idx);
        let n_i64 = i64::try_from(n).expect("n bounded to 10_000 fits i64");
        match (policy, hit) {
            (BoundaryPolicy::Reflect, BoundaryHit::Inside(i)) => {
                prop_assert!(i < n);
            }
            (BoundaryPolicy::Periodic, BoundaryHit::Inside(i)) => {
                prop_assert!(i < n);
            }
            (BoundaryPolicy::ZeroExtend, BoundaryHit::Inside(i)) => {
                prop_assert!(i < n);
                prop_assert!(idx >= 0 && idx < n_i64);
            }
            (BoundaryPolicy::ZeroExtend, BoundaryHit::Zero) => {
                prop_assert!(idx < 0 || idx >= n_i64);
            }
            (BoundaryPolicy::LinearExtrapolate, BoundaryHit::Inside(i)) => {
                prop_assert!(i < n);
                prop_assert!(idx >= 0 && idx < n_i64);
            }
            (BoundaryPolicy::LinearExtrapolate, BoundaryHit::OutsideLeft(d)) => {
                prop_assert!(idx < 0);
                prop_assert!(i64::from(d) == -idx);
            }
            (BoundaryPolicy::LinearExtrapolate, BoundaryHit::OutsideRight(d)) => {
                prop_assert!(idx >= n_i64);
                prop_assert!(i64::from(d) == idx - (n_i64 - 1));
            }
            _ => prop_assert!(false),
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1_000))]

    /// `bc_index_strict_interior_agreement` (I5): for all 4 policies and
    /// `i ∈ [0, n)`, `bc_index(policy, n, i) == Inside(i)`.
    #[test]
    fn bc_index_strict_interior_agreement(
        policy in any_policy(),
        n in 4usize..=1000usize,
        i_raw in 0usize..=999usize,
    ) {
        let i = i_raw.min(n - 1);
        let i_i64 = i64::try_from(i).expect("i bounded to 1000 fits i64");
        let hit = bc_index(policy, n, i_i64);
        prop_assert!(hit == BoundaryHit::Inside(i));
    }
}
