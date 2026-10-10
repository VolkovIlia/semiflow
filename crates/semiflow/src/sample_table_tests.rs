// Unit tests for `sample_table` — included as `mod tests` (keeps sample_table.rs ≤ 500 lines).
//
// `G_PLAN_BIT_EQUAL` (RELEASE_BLOCKING, ADR-0204): a prepared sample is
// bit-identical to `Grid1D::interp` for every interpolant × boundary policy,
// inside the grid, inside the ghost margin and beyond it (direct fallback),
// on the lane path and under the force-scalar hook.

use alloc::vec::Vec;

use super::{PreparedGridFn, Sample1D, MARGIN};
use crate::{
    grid::{BoundaryPolicy, Grid1D, InterpKind, OobPolicy},
    grid_chebyshev::virtual_node_x,
    grid_chebyshev_nodes::chebyshev_nodes,
};

fn lcg(n: usize, mut s: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let u = (s >> 11) as f64 / (1_u64 << 53) as f64;
            2.0 * u - 1.0
        })
        .collect()
}

fn policies() -> [BoundaryPolicy; 8] {
    [
        BoundaryPolicy::Reflect,
        BoundaryPolicy::ZeroExtend,
        BoundaryPolicy::Periodic,
        BoundaryPolicy::LinearExtrapolate,
        BoundaryPolicy::Dirichlet { value: 0.37 },
        BoundaryPolicy::Neumann,
        BoundaryPolicy::Robin {
            alpha: 0.7,
            beta: 1.3,
        },
        BoundaryPolicy::OddReflect,
    ]
}

fn interps() -> [InterpKind; 7] {
    [
        InterpKind::SepticHermite,
        InterpKind::OctonicHermite,
        InterpKind::CubicHermite,
        InterpKind::Linear,
        InterpKind::ChebyshevSpectralWithBC {
            m: 16,
            oob_policy: OobPolicy::Inherit,
        },
        InterpKind::ChebyshevSpectralWithBC {
            m: 32,
            oob_policy: OobPolicy::ForceReflect,
        },
        InterpKind::ChebyshevSpectralWithBC {
            m: 8,
            oob_policy: OobPolicy::ForcePeriodic,
        },
    ]
}

/// Probe points: a sweep from far left to far right (past the ghost margin,
/// exercising the direct fallback), every node, and every virtual node.
#[allow(clippy::cast_precision_loss)]
fn probes(grid: &Grid1D) -> Vec<f64> {
    let dx = grid.dx();
    let reach = (MARGIN as f64 + 9.0) * dx;
    let (lo, hi) = (grid.xmin - reach, grid.xmax + reach);
    let k = 997;
    let mut xs: Vec<f64> = (0..=k)
        .map(|i| lo + (hi - lo) * (f64::from(i) / f64::from(k)))
        .collect();
    xs.extend((0..grid.n).map(|i| grid.x_at(i)));
    if let InterpKind::ChebyshevSpectralWithBC { m, .. } = grid.interp {
        let nodes = chebyshev_nodes(m).expect("supported m");
        xs.extend(nodes.iter().map(|&t| virtual_node_x(grid, t)));
    }
    xs
}

fn check_all(force_scalar: bool) {
    crate::simd::FORCE_SCALAR.with(|c| c.set(force_scalar));
    let mut checked = 0_usize;
    for n in [8_usize, 37, 128] {
        for bnd in policies() {
            for interp in interps() {
                let grid = Grid1D::new(-3.0, 2.5, n)
                    .expect("grid")
                    .with_boundary(bnd)
                    .with_interp(interp);
                let values = lcg(n, 17 + n as u64);
                let mut work = alloc::vec![0.0; PreparedGridFn::work_len(&grid)];
                PreparedGridFn::fill(&values, &grid, &mut work);
                let prepared = PreparedGridFn::view(&values, grid, &work);
                for x in probes(&grid) {
                    let want = grid.interp(&values, x);
                    let got = prepared.sample(x);
                    match (want, got) {
                        (Ok(a), Ok(b)) => assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "G_PLAN_BIT_EQUAL: n={n} {bnd:?} {interp:?} x={x}: \
                             direct {a:e} vs prepared {b:e} (force_scalar={force_scalar})"
                        ),
                        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                        (a, b) => panic!("G_PLAN_BIT_EQUAL: Ok/Err mismatch {a:?} vs {b:?}"),
                    }
                    checked += 1;
                }
            }
        }
    }
    crate::simd::FORCE_SCALAR.with(|c| c.set(false));
    assert!(
        checked > 100_000,
        "G_PLAN_BIT_EQUAL: vacuous ({checked} probes)"
    );
}

#[test]
fn g_plan_bit_equal_samplers() {
    check_all(false);
}

#[test]
fn g_plan_bit_equal_samplers_force_scalar() {
    check_all(true);
}
