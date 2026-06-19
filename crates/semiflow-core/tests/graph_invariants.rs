//! Property tests for `Graph<F>` builder invariants I1–I7 and
//! `Laplacian<F>` invariants L1–L4.
//!
//! Uses `proptest` 1.4.0 (already in dev-deps).
//! See `contracts/v2.1/wave-a-graph-foundations.md` §1.3 / §3.2 and ADR-0048.

use proptest::prelude::*;
use semiflow_core::{Graph, Laplacian};

// ---------------------------------------------------------------------------
// Invariant checkers
// ---------------------------------------------------------------------------

/// Assert I1–I7 on a `Graph<f64>`.
fn assert_graph_invariants(g: &Graph<f64>) {
    let n = g.n_nodes();
    let rp = g.row_ptr();
    let ci = g.col_idx();
    let vs = g.vals();

    // I1: row_ptr length, starts at 0, ends at col_idx.len().
    assert_eq!(rp.len(), n + 1, "I1: row_ptr.len() != n+1");
    assert_eq!(rp[0], 0, "I1: row_ptr[0] != 0");
    assert_eq!(rp[n], ci.len(), "I1: row_ptr[n] != col_idx.len()");
    assert_eq!(ci.len(), vs.len(), "I1: col_idx.len() != vals.len()");

    for i in 0..n {
        let lo = rp[i];
        let hi = rp[i + 1];

        // I2: non-strictly monotonic row_ptr.
        assert!(hi >= lo, "I2: row_ptr not monotonic at {i}");

        let row_ci = &ci[lo..hi];
        let row_vs = &vs[lo..hi];

        // I3: col_idx strictly sorted ascending within row, no duplicates.
        for k in 1..row_ci.len() {
            assert!(
                row_ci[k] > row_ci[k - 1],
                "I3: not strictly sorted at row {i}, pos {k}"
            );
        }

        for (j_idx, (&j, &w)) in row_ci.iter().zip(row_vs.iter()).enumerate() {
            // I4: no self-loops.
            assert_ne!(j as usize, i, "I4: self-loop at node {i}");

            // I5: positive finite weights.
            assert!(
                w.is_finite() && w > 0.0,
                "I5: invalid weight at row {i}, pos {j_idx}"
            );

            // I6: all col_idx < n_nodes.
            assert!((j as usize) < n, "I6: col_idx {j} >= n_nodes {n}");
        }
    }

    // I7: symmetric storage — check every (i -> j) has a matching (j -> i).
    for i in 0..n {
        for k in rp[i]..rp[i + 1] {
            let j = ci[k] as usize;
            let w = vs[k];
            let j_row = &ci[rp[j]..rp[j + 1]];
            let j_vals = &vs[rp[j]..rp[j + 1]];
            let found = j_row
                .iter()
                .zip(j_vals.iter())
                .any(|(&c, &v)| c as usize == i && (v - w).abs() < 1e-15);
            assert!(found, "I7: edge ({i},{j}) has no symmetric counterpart");
        }
    }
}

/// Assert L1–L4 on a `Laplacian<f64>`.
fn assert_laplacian_invariants(lap: &Laplacian<f64>, g: &Graph<f64>) {
    let n = lap.n_nodes();
    let rp = lap.row_ptr();
    let ci = lap.col_idx();
    let vs = lap.vals();

    assert_eq!(rp.len(), n + 1, "Laplacian I1: row_ptr length");

    for i in 0..n {
        let lo = rp[i];
        let hi = rp[i + 1];

        // L1: diagonal is LAST entry of each row.
        if hi > lo {
            let last_col = ci[hi - 1] as usize;
            assert_eq!(last_col, i, "L1: diagonal not last in row {i}");
        }

        // L2: off-diagonal < 0, diagonal >= 0 (= 0 only for isolated nodes). L3: all finite.
        for k in lo..hi {
            let j = ci[k] as usize;
            let v = vs[k];
            assert!(v.is_finite(), "L3: non-finite value at row {i}");
            if j == i {
                assert!(v >= 0.0, "L2: diagonal negative at row {i}");
            } else {
                assert!(v < 0.0, "L2: off-diagonal not negative at row {i}, col {j}");
            }
        }
    }

    // L4: spectral_radius_bound >= max |diag| (Gershgorin).
    let rho = lap.spectral_radius_bound();
    assert!(rho > 0.0, "L4: Gershgorin bound not positive");

    // Check bound is non-trivial: all row-sums |L[i,j]| <= rho.
    for i in 0..n {
        let row_sum: f64 = vs[rp[i]..rp[i + 1]].iter().map(|v| v.abs()).sum();
        assert!(
            row_sum <= rho + 1e-10,
            "L4: row {i} sum {row_sum} > spectral_radius_bound {rho}"
        );
    }

    // Verify degree sums match graph (I6 analog for Laplacian).
    for i in 0..n {
        let g_deg: f64 = g.vals()[g.row_ptr()[i]..g.row_ptr()[i + 1]].iter().sum();
        let l_diag = vs[rp[i + 1] - 1];
        assert!(
            (l_diag - g_deg).abs() < 1e-12,
            "Laplacian diagonal at {i}: expected {g_deg}, got {l_diag}"
        );
    }
}

// ---------------------------------------------------------------------------
// Path graph smoke tests
// ---------------------------------------------------------------------------

#[test]
fn path_graph_invariants_n8() {
    let g = Graph::<f64>::path(8);
    assert_graph_invariants(&g);
    let lap = Laplacian::assemble_combinatorial(&g);
    assert_laplacian_invariants(&lap, &g);
}

#[test]
fn cycle_graph_invariants_n6() {
    let g = Graph::<f64>::cycle(6);
    assert_graph_invariants(&g);
    let lap = Laplacian::assemble_combinatorial(&g);
    assert_laplacian_invariants(&lap, &g);
}

#[test]
fn erdos_renyi_invariants() {
    let g = Graph::<f64>::erdos_renyi(32, 0.3, 0xDEAD_BEEF);
    assert_graph_invariants(&g);
    let lap = Laplacian::assemble_combinatorial(&g);
    assert_laplacian_invariants(&lap, &g);
}

#[test]
fn gershgorin_nontrivial_i7() {
    let g = Graph::<f64>::path(10);
    let lap = Laplacian::assemble_combinatorial(&g);
    // Boundary nodes have degree 1, interior nodes degree 2.
    // Gershgorin bound = 2 * max_deg = 4 for interior nodes.
    assert!(
        lap.spectral_radius_bound() >= 3.9,
        "Gershgorin bound too small: {}",
        lap.spectral_radius_bound()
    );
}

#[test]
fn erdos_renyi_determinism() {
    let g1 = Graph::<f64>::erdos_renyi(20, 0.3, 42);
    let g2 = Graph::<f64>::erdos_renyi(20, 0.3, 42);
    assert_eq!(
        g1.n_directed_edges(),
        g2.n_directed_edges(),
        "determinism: edge count"
    );
    // Check same col_idx.
    for (a, b) in g1.col_idx().iter().zip(g2.col_idx().iter()) {
        assert_eq!(a, b, "determinism: col_idx mismatch");
    }
}

// ---------------------------------------------------------------------------
// Proptest: from_edges preserves invariants
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn from_edges_preserves_invariants(
        n in 2usize..32,
        raw_edges in prop::collection::vec(
            (0u32..32, 0u32..32, 1e-6f64..1e6f64),
            0..100
        )
    ) {
        let valid: Vec<(u32, u32, f64)> = raw_edges.into_iter()
            .filter(|(u, v, w)| {
                *u != *v
                    && (*u as usize) < n
                    && (*v as usize) < n
                    && w.is_finite()
                    && *w > 0.0
            })
            // Deduplicate (keep first of each (min, max) pair).
            .fold(Vec::new(), |mut acc, (u, v, w)| {
                let lo = u.min(v);
                let hi = u.max(v);
                if !acc.iter().any(|(a, b, _)| *a == lo && *b == hi) {
                    acc.push((lo, hi, w));
                }
                acc
            });

        if let Ok(g) = Graph::<f64>::from_edges(n, valid) {
            assert_graph_invariants(&g);
            // Laplacian invariants L2/L4 require at least one edge (isolated-only
            // graphs have diagonal = 0 and Gershgorin bound = 0 by definition).
            if g.n_directed_edges() > 0 {
                let lap = Laplacian::assemble_combinatorial(&g);
                assert_laplacian_invariants(&lap, &g);
            }
        }
    }
}
