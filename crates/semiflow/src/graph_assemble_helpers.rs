// Private Laplacian assembly and validation helpers for `graph.rs` — included via
// `include!` at module scope (keeps graph.rs within the 500-line budget).

/// Build Laplacian CSR rows: off-diagonals first (sorted asc), diagonal last (L1).
fn assemble_laplacian<F: SemiflowFloat>(g: &Graph<F>, kind: LaplacianKind) -> Laplacian<F> {
    let n = g.n_nodes();
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

/// Fill per-row entries: off-diagonals sorted ascending (I3/L1), diagonal appended last.
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
                        -w / normalized_denominator(di, dj)
                    } else {
                        F::zero()
                    };
                    row.push((j, val));
                }
            }
        }
        LaplacianKind::GeneralSymmetric => unreachable!("GeneralSymmetric bypasses assembly"),
    }
    for row in &mut rows {
        row.sort_unstable_by_key(|(c, _)| *c);
    }
    append_diagonal_entries(&mut rows, kind, deg);
    rows
}

/// Append the diagonal entry `(i, diag_i)` to each row (invariant L1).
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
            LaplacianKind::GeneralSymmetric => unreachable!("GeneralSymmetric bypasses assembly"),
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

/// Validate CSR parts (dimensions + `col_idx` range) for `Laplacian::from_csr_parts`.
fn validate_csr_parts<F: SemiflowFloat>(
    n_nodes: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[F],
) -> Result<(), SemiflowError> {
    if row_ptr.len() != n_nodes + 1 {
        return Err(SemiflowError::DomainViolation {
            what: "from_csr_parts: row_ptr.len() must equal n_nodes+1",
            #[allow(clippy::cast_precision_loss)]
            value: row_ptr.len() as f64,
        });
    }
    for i in 0..n_nodes {
        if row_ptr[i] > row_ptr[i + 1] {
            return Err(SemiflowError::DomainViolation {
                what: "from_csr_parts: row_ptr not monotone non-decreasing",
                #[allow(clippy::cast_precision_loss)]
                value: i as f64,
            });
        }
    }
    let nnz = row_ptr[n_nodes];
    if col_idx.len() != nnz || vals.len() != nnz {
        return Err(SemiflowError::DomainViolation {
            what: "from_csr_parts: col_idx.len()/vals.len() must equal row_ptr[n_nodes]",
            #[allow(clippy::cast_precision_loss)]
            value: nnz as f64,
        });
    }
    for &c in col_idx {
        if c as usize >= n_nodes {
            return Err(SemiflowError::DomainViolation {
                what: "from_csr_parts: col_idx entry >= n_nodes",
                value: f64::from(c),
            });
        }
    }
    if let Some(bad) = vals.iter().find(|v| !v.is_finite()) {
        return Err(SemiflowError::DomainViolation {
            what: "from_csr_parts: non-finite matrix entry",
            value: bad.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// Reject weighted degrees that overflow: `dᵢ` is the combinatorial Laplacian's
/// diagonal entry, so an `∞` there is an invalid operator, not just a loose bound.
fn check_degrees_finite<F: SemiflowFloat>(rows: &[Vec<(u32, F)>]) -> Result<(), SemiflowError> {
    for row in rows {
        let deg = row.iter().fold(F::zero(), |acc, &(_, w)| acc + w);
        if !deg.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "Graph::from_edges: weighted degree overflows (non-finite d_i)",
                value: deg.to_f64().unwrap_or(f64::INFINITY),
            });
        }
    }
    Ok(())
}

/// `√(dᵢ·dⱼ)`, without overflow when the product exceeds the float range.
///
/// The product form is kept whenever it is finite so in-range results are
/// bit-identical to the original `(dᵢ·dⱼ).sqrt()`.
fn normalized_denominator<F: SemiflowFloat>(di: F, dj: F) -> F {
    let prod = di * dj;
    if prod.is_finite() {
        prod.sqrt()
    } else {
        di.sqrt() * dj.sqrt()
    }
}

/// Gershgorin row-sum bound from flat CSR: `ρ̄ = max_i Σ_j |L[i,j]|`.
fn gershgorin_bound_csr<F: SemiflowFloat>(row_ptr: &[usize], vals: &[F]) -> F {
    let n = row_ptr.len().saturating_sub(1);
    let mut max_sum = F::zero();
    for i in 0..n {
        let mut row_sum = F::zero();
        for v in &vals[row_ptr[i]..row_ptr[i + 1]] {
            let av = if *v < F::zero() { F::zero() - *v } else { *v };
            row_sum += av;
        }
        if row_sum > max_sum {
            max_sum = row_sum;
        }
    }
    max_sum
}

// Same as `gershgorin_bound_csr` but from rows vec (pre-assembly).
fn compute_gershgorin_bound<F: SemiflowFloat>(rows: &[Vec<(u32, F)>]) -> F {
    let mut max_sum = F::zero();
    for row in rows {
        let row_sum: F = row.iter().fold(F::zero(), |acc, &(_, v)| {
            let av = if v < F::zero() { F::zero() - v } else { v };
            acc + av
        });
        max_sum = if row_sum > max_sum { row_sum } else { max_sum };
    }
    max_sum
}
