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

/// Binding gate (ADR-0190): the nodal weights that `interp_stencil` hands to the
/// N-D tensor-product sampler must reproduce the arithmetic the generic 1-D
/// sampler actually performs.
///
/// This calls the real `catmull_rom_scalar_generic`, not a re-typed copy, so the
/// 1-D and N-D interpolation paths cannot drift apart silently. `grid.rs` keeps
/// its own evaluation order — re-expressing it via the weights would perturb the
/// f32 and `Dual` scalar paths bit-for-bit — and this test is what makes the two
/// forms one contract rather than two implementations.
#[test]
fn catmull_rom_matches_interp_stencil() {
    use crate::interp_stencil::interp_stencil;
    let p = [0.3_f64, -1.2, 2.5, 0.7];
    for k in 0..=16 {
        let s = f64::from(k) / 16.0;
        let (n_k, offsets, w) = interp_stencil::<f64>(crate::grid::InterpKind::CubicHermite, s)
            .expect("CubicHermite is supported");
        assert_eq!(n_k, 4);
        assert_eq!(&offsets[..4], &[-1, 0, 1, 2]);
        let via_weights: f64 = w.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
        let direct = super::catmull_rom_scalar_generic(p[0], p[1], p[2], p[3], s);
        assert!(
            (via_weights - direct).abs() < 1e-14,
            "s={s}: weights={via_weights}, sampler={direct}"
        );
    }
}
