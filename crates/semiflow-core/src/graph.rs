//! Frozen sparse weighted graph (`Graph<F>`) and graph Laplacian (`Laplacian<F>`)
//! in hand-rolled CSR layout.
//!
//! See `contracts/v2.1/wave-a-graph-foundations.md` §1 / §3 and ADR-0048 for
//! the full invariant set (I1–I7, L1–L4).
//!
//! ## Invariants (enforced at build time, preserved post-assembly)
//!
//! **Graph<F>:**
//! - I1: `row_ptr.len() == n_nodes + 1`; `row_ptr[0] == 0`; `row_ptr[n_nodes] == col_idx.len() == vals.len()`.
//! - I2: `row_ptr` is non-strictly monotonic.
//! - I3: Within each row, `col_idx` is strictly sorted ascending (no duplicates).
//! - I4: No self-loops.
//! - I5: All `vals` are finite and positive.
//! - I6: All `col_idx[k] < n_nodes`.
//! - I7: Symmetric: edge `(u,v)` appears in both row `u` and row `v`.
//!
//! **Laplacian<F>** (extends I1–I7):
//! - L1: Diagonal entry is the LAST in each row's CSR slice.
//! - L2: Off-diagonal `vals < 0`; diagonal `vals > 0`.
//! - L3: All `vals` are finite.
//! - L4: `spectral_radius_bound` is cached at assembly.

use alloc::vec;
use alloc::vec::Vec;

use crate::{error::SemiflowError, float::SemiflowFloat};

// ---------------------------------------------------------------------------
// LaplacianKind
// ---------------------------------------------------------------------------

/// Normalization convention for the assembled [`Laplacian`].
///
/// Stored for documentation only — `apply_into_slice` is identical for both
/// (the CSR carries the computed values). See ADR-0048, Chung 1997 §1.2.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LaplacianKind {
    /// Combinatorial Laplacian `L = D − W`. Default.
    Combinatorial,
    /// Symmetric normalized Laplacian `L_sym = I − D^{−½} W D^{−½}` (Chung 1997 §1.2).
    SymNormalized,
}

// ---------------------------------------------------------------------------
// Graph<F>
// ---------------------------------------------------------------------------

/// Frozen sparse weighted graph in symmetric CSR layout. See ADR-0048.
///
/// **Immutable post-construction** — no mutator methods. Invariants I1–I7 hold.
pub struct Graph<F: SemiflowFloat = f64> {
    n_nodes: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    vals: Vec<F>,
}

impl<F: SemiflowFloat> Graph<F> {
    /// Number of nodes. O(1).
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of directed edge-entries stored (= `2 · undirected_edges`).
    #[must_use]
    pub fn n_directed_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Row-pointer slice. Length `n_nodes + 1`.
    #[must_use]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Column-index slice (neighbour node ids).
    #[must_use]
    pub fn col_idx(&self) -> &[u32] {
        &self.col_idx
    }

    /// Edge-weight slice (same length as `col_idx`).
    #[must_use]
    pub fn vals(&self) -> &[F] {
        &self.vals
    }

    // -----------------------------------------------------------------------
    // Internal builder helper
    // -----------------------------------------------------------------------

    /// Assemble a CSR graph from an adjacency list (one `Vec<(u32, F)>` per row).
    /// Rows are sorted by `col_idx` ascending (invariant I3).
    fn from_rows(n_nodes: usize, mut rows: Vec<Vec<(u32, F)>>) -> Self {
        for row in &mut rows {
            row.sort_unstable_by_key(|(c, _)| *c);
        }
        let total: usize = rows.iter().map(Vec::len).sum();
        let mut row_ptr = Vec::with_capacity(n_nodes + 1);
        let mut col_idx = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        let mut ptr = 0_usize;
        row_ptr.push(0);
        for row in &rows {
            for &(c, v) in row {
                col_idx.push(c);
                vals.push(v);
            }
            ptr += row.len();
            row_ptr.push(ptr);
        }
        Self {
            n_nodes,
            row_ptr,
            col_idx,
            vals,
        }
    }

    // -----------------------------------------------------------------------
    // Builders
    // -----------------------------------------------------------------------

    /// Path graph `0 — 1 — 2 — … — (n − 1)` with unit edge weights.
    ///
    /// # Panics
    /// Panics if `n == 0`.
    #[must_use]
    pub fn path(n: usize) -> Self {
        assert!(n > 0, "Graph::path requires n >= 1");
        let mut rows: Vec<Vec<(u32, F)>> = (0..n).map(|_| Vec::new()).collect();
        for i in 0..(n - 1) {
            #[allow(clippy::cast_possible_truncation)]
            let iu32 = i as u32;
            rows[i].push((iu32 + 1, F::one()));
            rows[i + 1].push((iu32, F::one()));
        }
        Self::from_rows(n, rows)
    }

    /// Cycle graph: path graph plus edge `(n − 1) — 0`, all unit weights.
    ///
    /// # Panics
    /// Panics if `n < 3` (cycle on 1 or 2 nodes is degenerate).
    #[must_use]
    pub fn cycle(n: usize) -> Self {
        assert!(n >= 3, "Graph::cycle requires n >= 3");
        let mut rows: Vec<Vec<(u32, F)>> = (0..n).map(|_| Vec::new()).collect();
        for i in 0..(n - 1) {
            #[allow(clippy::cast_possible_truncation)]
            let iu32 = i as u32;
            rows[i].push((iu32 + 1, F::one()));
            rows[i + 1].push((iu32, F::one()));
        }
        // wrap-around edge
        #[allow(clippy::cast_possible_truncation)]
        let n_minus_1 = (n - 1) as u32;
        rows[n - 1].push((0_u32, F::one()));
        rows[0].push((n_minus_1, F::one()));
        Self::from_rows(n, rows)
    }

    /// Build from an iterator of undirected edges `(u, v, w)`.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] on:
    /// - Self-loop (`u == v`).
    /// - Node index out of range.
    /// - Non-positive or non-finite edge weight.
    /// - Duplicate edge.
    pub fn from_edges(
        n_nodes: usize,
        edges: impl IntoIterator<Item = (u32, u32, F)>,
    ) -> Result<Self, SemiflowError> {
        let mut rows: Vec<Vec<(u32, F)>> = (0..n_nodes).map(|_| Vec::new()).collect();
        for (u, v, w) in edges {
            if u == v {
                return Err(SemiflowError::DomainViolation {
                    what: "Graph::from_edges: self-loop not allowed",
                    value: f64::from(u),
                });
            }
            if u as usize >= n_nodes || v as usize >= n_nodes {
                return Err(SemiflowError::DomainViolation {
                    what: "Graph::from_edges: node index >= n_nodes",
                    value: f64::from(u.max(v)),
                });
            }
            if !w.is_finite() || w <= F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what: "Graph::from_edges: weight must be finite and positive",
                    value: w.to_f64().unwrap_or(f64::NAN),
                });
            }
            // Check duplicate
            if rows[u as usize].iter().any(|(c, _)| *c == v) {
                return Err(SemiflowError::DomainViolation {
                    what: "Graph::from_edges: duplicate edge",
                    value: f64::from(u),
                });
            }
            rows[u as usize].push((v, w));
            rows[v as usize].push((u, w));
        }
        Ok(Self::from_rows(n_nodes, rows))
    }

    /// Erdős–Rényi `G(n_nodes, p)` with seeded splitmix64 RNG.
    ///
    /// All edge weights = `F::one()`. Deterministic for a given `seed`.
    #[must_use]
    pub fn erdos_renyi(n_nodes: usize, p: f64, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut rows: Vec<Vec<(u32, F)>> = (0..n_nodes).map(|_| Vec::new()).collect();
        for u in 0..n_nodes {
            for v in (u + 1)..n_nodes {
                let r = rng.next_f64();
                if r < p {
                    #[allow(clippy::cast_possible_truncation)]
                    let (v_u32, u_u32) = (v as u32, u as u32);
                    rows[u].push((v_u32, F::one()));
                    rows[v].push((u_u32, F::one()));
                }
            }
        }
        Self::from_rows(n_nodes, rows)
    }
}

// ---------------------------------------------------------------------------
// SplitMix64 — deterministic seeded RNG (≤ 20 LoC, no new dep)
// ---------------------------------------------------------------------------

/// Minimal splitmix64 RNG for deterministic Erdős–Rényi graphs.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        // Map to [0, 1) using upper 53 bits.
        // The right-shifted value and divisor (2^53) are both exact in f64.
        #[allow(clippy::cast_precision_loss)]
        let hi53 = (self.next_u64() >> 11) as f64;
        // 2^53 is representable exactly in f64 (53-bit mantissa).
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f64 / (1u64 << 53) as f64;
        hi53 * scale
    }
}

// ---------------------------------------------------------------------------
// Laplacian<F>
// ---------------------------------------------------------------------------

/// Frozen sparse Laplacian in CSR layout. See ADR-0048 invariants L1–L4.
///
/// Diagonal entry is the **LAST** entry of each row (invariant L1).
pub struct Laplacian<F: SemiflowFloat = f64> {
    n_nodes: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    vals: Vec<F>,
    spectral_radius_bound: F,
    kind: LaplacianKind,
}

impl<F: SemiflowFloat> Laplacian<F> {
    /// Number of nodes.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Gershgorin spectral-radius upper bound `ρ̄ ≥ ρ(L_G)`. Cached at assembly.
    #[must_use]
    pub fn spectral_radius_bound(&self) -> F {
        self.spectral_radius_bound
    }

    /// Normalization kind (documentation only).
    #[must_use]
    pub fn kind(&self) -> LaplacianKind {
        self.kind
    }

    /// Row-pointer slice. Length `n_nodes + 1`.
    #[must_use]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Column-index slice.
    #[must_use]
    pub fn col_idx(&self) -> &[u32] {
        &self.col_idx
    }

    /// Value slice (L[i,j] for each CSR entry).
    #[must_use]
    pub fn vals(&self) -> &[F] {
        &self.vals
    }

    /// Combinatorial Laplacian `L = D − W` from a graph.
    #[must_use]
    pub fn assemble_combinatorial(g: &Graph<F>) -> Self {
        assemble_laplacian(g, LaplacianKind::Combinatorial)
    }

    /// Symmetric normalized Laplacian `L_sym = I − D^{−½} W D^{−½}` (Chung 1997 §1.2).
    ///
    /// Isolated nodes (`deg = 0`) get `L_sym[i, i] = 0` per Chung 1997 §1.2 convention.
    #[must_use]
    pub fn assemble_normalized(g: &Graph<F>) -> Self {
        assemble_laplacian(g, LaplacianKind::SymNormalized)
    }

    /// `dst ← L_G · src`. Zero-alloc; `src.len() == dst.len() == self.n_nodes`.
    pub fn apply_into_slice(&self, src: &[F], dst: &mut [F]) {
        debug_assert_eq!(
            src.len(),
            self.n_nodes,
            "apply_into_slice: src len mismatch"
        );
        debug_assert_eq!(
            dst.len(),
            self.n_nodes,
            "apply_into_slice: dst len mismatch"
        );
        for (i, d) in dst.iter_mut().enumerate() {
            let mut acc = F::zero();
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                acc += self.vals[k] * src[self.col_idx[k] as usize];
            }
            *d = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Laplacian assembly (shared logic)
// ---------------------------------------------------------------------------

/// Build rows for the Laplacian: off-diagonal entries first (sorted ascending),
/// then the diagonal as the last entry (invariant L1).
fn assemble_laplacian<F: SemiflowFloat>(g: &Graph<F>, kind: LaplacianKind) -> Laplacian<F> {
    let n = g.n_nodes();

    // Compute weighted degrees.
    let mut deg: Vec<F> = vec![F::zero(); n];
    for (i, d) in deg.iter_mut().enumerate() {
        for k in g.row_ptr()[i]..g.row_ptr()[i + 1] {
            *d += g.vals()[k];
        }
    }

    let rows = fill_laplacian_rows(g, kind, &deg, n);
    let rho_bar = compute_gershgorin_bound(&rows);
    let (row_ptr, col_idx, vals) = flatten_to_csr(&rows);

    Laplacian {
        n_nodes: n,
        row_ptr,
        col_idx,
        vals,
        spectral_radius_bound: rho_bar,
        kind,
    }
}

/// Fill per-row (off-diagonal + diagonal) entries for the Laplacian.
///
/// Off-diagonal entries are sorted ascending (invariant I3/L1); diagonal
/// is appended last (invariant L1).
fn fill_laplacian_rows<F: SemiflowFloat>(
    g: &Graph<F>,
    kind: LaplacianKind,
    deg: &[F],
    n: usize,
) -> Vec<Vec<(u32, F)>> {
    let mut rows: Vec<Vec<(u32, F)>> = (0..n).map(|_| Vec::new()).collect();
    match kind {
        LaplacianKind::Combinatorial => {
            for (i, row) in rows.iter_mut().enumerate() {
                for k in g.row_ptr()[i]..g.row_ptr()[i + 1] {
                    let j = g.col_idx()[k]; // u32 (I6 guarantees < n)
                    row.push((j, -g.vals()[k]));
                }
            }
        }
        LaplacianKind::SymNormalized => {
            for (i, row) in rows.iter_mut().enumerate() {
                let di = deg[i];
                for k in g.row_ptr()[i]..g.row_ptr()[i + 1] {
                    let j = g.col_idx()[k]; // u32
                    let dj = deg[j as usize];
                    let w = g.vals()[k];
                    let val = if di > F::zero() && dj > F::zero() {
                        -w / (di * dj).sqrt()
                    } else {
                        F::zero()
                    };
                    row.push((j, val));
                }
            }
        }
    }
    // Sort off-diagonal ascending (invariant I3/L1).
    for row in &mut rows {
        row.sort_unstable_by_key(|(c, _)| *c);
    }
    // Append diagonal LAST (invariant L1).
    append_diagonal_entries(&mut rows, kind, deg);
    rows
}

/// Append the diagonal entry `(i, diag_i)` to each row (invariant L1).
///
/// Extracted from `fill_laplacian_rows` — batch H9b.
fn append_diagonal_entries<F: SemiflowFloat>(
    rows: &mut [Vec<(u32, F)>],
    kind: LaplacianKind,
    deg: &[F],
) {
    for (i, row) in rows.iter_mut().enumerate() {
        let diag = match kind {
            LaplacianKind::Combinatorial => deg[i],
            LaplacianKind::SymNormalized => {
                if deg[i] > F::zero() {
                    F::one()
                } else {
                    F::zero()
                }
            }
        };
        #[allow(clippy::cast_possible_truncation)]
        row.push((i as u32, diag));
    }
}

/// Flatten adjacency rows into CSR `(row_ptr, col_idx, vals)`.
fn flatten_to_csr<F: Copy>(rows: &[Vec<(u32, F)>]) -> (Vec<usize>, Vec<u32>, Vec<F>) {
    let total: usize = rows.iter().map(Vec::len).sum();
    let mut row_ptr = Vec::with_capacity(rows.len() + 1);
    let mut col_idx = Vec::with_capacity(total);
    let mut vals = Vec::with_capacity(total);
    let mut ptr = 0_usize;
    row_ptr.push(0);
    for row in rows {
        for &(c, v) in row {
            col_idx.push(c);
            vals.push(v);
        }
        ptr += row.len();
        row_ptr.push(ptr);
    }
    (row_ptr, col_idx, vals)
}

/// Gershgorin row-sum bound: `ρ̄ = max_i Σ_j |L[i, j]|`.
fn compute_gershgorin_bound<F: SemiflowFloat>(rows: &[Vec<(u32, F)>]) -> F {
    let mut max_sum = F::zero();
    for row in rows {
        let row_sum: F = row.iter().fold(F::zero(), |acc, &(_, v)| {
            let av = if v < F::zero() { F::zero() - v } else { v };
            acc + av
        });
        if row_sum > max_sum {
            max_sum = row_sum;
        }
    }
    max_sum
}
