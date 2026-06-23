//! ADR-0123 — `SmolyakGridND<F, const D: usize>` sparse-grid quadrature backend.
//!
//! Smolyak combination over the in-tree GH ladder {1,3,5,7,9}-pt.
//! Default level `ℓ = D+3`: D=5 → ℓ=8 → 341 nodes (tensor 5⁵=3125, 9.2× reduction).
//! NO new dependency. Weights carry SIGNS (combination coefficients).
//! Gate: `G_SMOLYAK_D5` — slope ≤ −0.95, nodes < 3125, F(0)=I verified at construction.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid_nd::{GridFnND, GridND},
    scratch::ScratchPool,
    shift_nd::{
        build_cholesky_cache, flat_to_x, SquareMatrix, GH1_NODES_F64, GH1_WEIGHTS_F64,
        GH3_NODES_F64, GH3_WEIGHTS_F64, GH5_NODES_F64, GH5_WEIGHTS_F64, GH7_NODES_F64,
        GH7_WEIGHTS_F64, GH9_NODES_F64, GH9_WEIGHTS_F64,
    },
};

// 1-D GH ladder: level ∈ {1..=5} → (q, nodes, weights).
struct GhLevel {
    q: usize,
    nodes: &'static [f64],
    weights: &'static [f64],
}

const GH_LADDER: [GhLevel; 5] = [
    GhLevel {
        q: 1,
        nodes: &GH1_NODES_F64,
        weights: &GH1_WEIGHTS_F64,
    },
    GhLevel {
        q: 3,
        nodes: &GH3_NODES_F64,
        weights: &GH3_WEIGHTS_F64,
    },
    GhLevel {
        q: 5,
        nodes: &GH5_NODES_F64,
        weights: &GH5_WEIGHTS_F64,
    },
    GhLevel {
        q: 7,
        nodes: &GH7_NODES_F64,
        weights: &GH7_WEIGHTS_F64,
    },
    GhLevel {
        q: 9,
        nodes: &GH9_NODES_F64,
        weights: &GH9_WEIGHTS_F64,
    },
];

/// `GH_LADDER` entry for 1-based level `lev` (1..=5); `None` otherwise.
fn gh_level(lev: usize) -> Option<&'static GhLevel> {
    if lev == 0 || lev > 5 {
        return None;
    }
    Some(&GH_LADDER[lev - 1])
}

/// C(n, k) for n,k ≤ 12; 0 for k > n.  Fits in i64 (max C(12,6)=924).
fn binom(n: usize, k: usize) -> i64 {
    if k > n {
        return 0;
    }
    // Use the smaller k for efficiency.
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 0..k {
        #[allow(clippy::cast_possible_wrap)]
        let num = (n - i) as i64;
        #[allow(clippy::cast_possible_wrap)]
        let den = (i + 1) as i64;
        result = result * num / den;
    }
    result
}

// ---------------------------------------------------------------------------
// Smolyak node/weight builder
// ---------------------------------------------------------------------------

/// Build the Smolyak sparse grid A(ell, D) as a list of (node, weight) pairs,
/// with duplicate nodes merged (nested rules share nodes across sub-rules).
///
/// Nodes rounded to 14 decimal places for deduplication; weights carry SIGNS.
fn build_smolyak<const D: usize>(ell: usize) -> Result<Vec<([f64; D], f64)>, SemiflowError> {
    if ell < D {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "SmolyakGridND: ell must be >= D",
            value: ell as f64,
        });
    }
    let mut raw: Vec<([f64; D], f64)> = Vec::new();
    let mut stack = Vec::new();
    enumerate_multi_index::<D>(ell, &mut stack, &mut raw)?;

    // Sort by node lexicographically, then merge coincident nodes.
    raw.sort_by(|a, b| {
        for d in 0..D {
            let c = a.0[d].total_cmp(&b.0[d]);
            if c != core::cmp::Ordering::Equal {
                return c;
            }
        }
        core::cmp::Ordering::Equal
    });
    let nodes_eq = |a: &[f64; D], b: &[f64; D]| -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| x.total_cmp(y).is_eq())
    };
    let mut merged: Vec<([f64; D], f64)> = Vec::new();
    for (node, w) in raw {
        if let Some(last) = merged.last_mut() {
            if nodes_eq(&last.0, &node) {
                last.1 += w;
                continue;
            }
        }
        merged.push((node, w));
    }
    // Drop near-zero combined weights (cancellation artifact).
    merged.retain(|(_, w)| w.abs() > 1e-15);

    Ok(merged)
}

/// Fill `raw` with all (tensor-product node, weight) pairs for admissible multi-indices.
fn enumerate_multi_index<const D: usize>(
    ell: usize,
    partial: &mut Vec<usize>,
    raw: &mut Vec<([f64; D], f64)>,
) -> Result<(), SemiflowError> {
    let depth = partial.len();
    if depth == D {
        let s: usize = partial.iter().sum();
        if s < ell.saturating_sub(D - 1) || s > ell {
            return Ok(());
        }
        let diff = ell - s;
        let coeff_abs = binom(D - 1, diff);
        if coeff_abs == 0 {
            return Ok(());
        }
        let sign: f64 = if diff % 2 == 0 { 1.0 } else { -1.0 };
        #[allow(clippy::cast_precision_loss)]
        let coeff = sign * coeff_abs as f64;
        let rules: Vec<&GhLevel> = partial
            .iter()
            .map(|&lev| {
                #[allow(clippy::cast_precision_loss)]
                gh_level(lev).ok_or(SemiflowError::DomainViolation {
                    what: "SmolyakGridND: level exceeds GH ladder depth (max 5)",
                    value: lev as f64,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        add_tensor_product::<D>(&rules, coeff, raw);
        return Ok(());
    }
    let remaining = D - depth;
    for lev in 1..=ell.min(5) {
        let partial_sum: usize = partial.iter().sum();
        if partial_sum + lev + (remaining - 1) > ell {
            break;
        }
        partial.push(lev);
        enumerate_multi_index::<D>(ell, partial, raw)?;
        partial.pop();
    }
    Ok(())
}

/// Add all tensor-product (node, weight) pairs from `rules` scaled by `coeff` into `raw`.
fn add_tensor_product<const D: usize>(
    rules: &[&GhLevel],
    coeff: f64,
    raw: &mut Vec<([f64; D], f64)>,
) {
    // Total number of tensor combinations.
    let total: usize = rules.iter().map(|r| r.q).product();
    for qi in 0..total {
        let mut node = [0.0f64; D];
        let mut w_prod = coeff;
        let mut rem = qi;
        for d in 0..D {
            let q = rules[d].q;
            let digit = rem % q;
            rem /= q;
            // Round to 14 decimal places for stable deduplication.
            node[d] = (rules[d].nodes[digit] * 1e14).round() / 1e14;
            w_prod *= rules[d].weights[digit];
        }
        raw.push((node, w_prod));
    }
}

// ---------------------------------------------------------------------------
// SmolyakGridND<F, D>
// ---------------------------------------------------------------------------

/// Boxed drift closure type alias (suppress `type_complexity` lint).
type DriftFn<F, const D: usize> = alloc::boxed::Box<dyn Fn(&[F; D], &mut [F; D]) + Send + Sync>;
/// Boxed reaction closure type alias (suppress `type_complexity` lint).
type ReactionFn<F, const D: usize> = alloc::boxed::Box<dyn Fn(&[F; D]) -> F + Send + Sync>;

/// Sparse-grid Smolyak quadrature backend for d-D anisotropic shift (ADR-0123).
///
/// Implements the same `ChernoffFunction<F>` interface as
/// `AnisotropicShiftChernoffND` but uses a Smolyak sparse grid instead of the
/// full tensor product. For D=5 at the default level ℓ=D+3=8 this gives
/// **341 quadrature nodes** vs 3125 for the tensor rule — a 9.2× reduction.
///
/// Weights carry SIGNS. The `F(0)=I` unit witness is verified at construction.
///
/// # Construction
///
/// ```rust,ignore
/// # use semiflow::{Grid1D, smolyak::SmolyakGridND, grid_nd::GridND};
/// # let ax = Grid1D::new(-5.0_f64, 5.0, 8).unwrap();
/// # let grid = GridND::<f64, 5>::new([ax; 5]).unwrap();
/// let kernel = SmolyakGridND::<f64, 5>::new(
///     |_x, a| { for i in 0..5 { a.set(i, i, 1.0); } },
///     |_x, b| { for v in b.iter_mut() { *v = 0.0; } },
///     |_x| 0.0_f64,
///     grid,
/// ).unwrap();
/// ```
pub struct SmolyakGridND<F: SemiflowFloat = f64, const D: usize = 5> {
    /// Drift vector `b_i(x)`.
    b_i: DriftFn<F, D>,
    /// Reaction coefficient `c(x)`.
    c: ReactionFn<F, D>,
    /// Cached Cholesky factors at each grid point.
    cholesky_cache: Vec<SquareMatrix<F, D>>,
    /// Smolyak sparse-grid nodes (physical space, pre-scaled; raw η values).
    sm_nodes: Vec<[f64; D]>,
    /// Smolyak sparse-grid weights (signed combination coefficients × GH weights).
    sm_weights: Vec<f64>,
    /// Smolyak level ℓ (default: D+3).
    level: usize,
    /// Reference grid geometry.
    grid: GridND<F, D>,
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat, const D: usize> SmolyakGridND<F, D> {
    /// Construct with default Smolyak level `ℓ = D + 3`.
    ///
    /// # Errors
    /// - `DomainViolation` if `D == 0`.
    /// - `DomainViolation` if grid has fewer than 2 nodes per axis.
    /// - `DomainViolation` if `a_ij` is not SPD at any grid point.
    pub fn new(
        a_ij: impl Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync + 'static,
        b_i: impl Fn(&[F; D], &mut [F; D]) + Send + Sync + 'static,
        c: impl Fn(&[F; D]) -> F + Send + Sync + 'static,
        grid: GridND<F, D>,
    ) -> Result<Self, SemiflowError> {
        Self::with_level(a_ij, b_i, c, grid, D + 3)
    }

    /// Construct with explicit Smolyak level `ell`.
    ///
    /// # Errors
    /// Same as `new` plus `DomainViolation` if `ell < D` or `ell - D + 1 > 5`
    /// (level exceeds the in-tree GH ladder depth of 5).
    pub fn with_level(
        a_ij: impl Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync + 'static,
        b_i: impl Fn(&[F; D], &mut [F; D]) + Send + Sync + 'static,
        c: impl Fn(&[F; D]) -> F + Send + Sync + 'static,
        grid: GridND<F, D>,
        ell: usize,
    ) -> Result<Self, SemiflowError> {
        if D == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "SmolyakGridND: D must be >= 1",
                value: 0.0,
            });
        }
        // Build Smolyak nodes/weights (validates ell ≥ D, level ≤ 5).
        let nw = build_smolyak::<D>(ell)?;
        let n_nodes = nw.len();
        if n_nodes == 0 {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "SmolyakGridND: empty node set (ell too small?)",
                value: ell as f64,
            });
        }
        // Verify F(0)=I unit witness: Σ weights = π^{D/2}.
        let wsum: f64 = nw.iter().map(|(_, w)| *w).sum();
        #[allow(clippy::cast_precision_loss)]
        let pi_dhalf = core::f64::consts::PI.powf(D as f64 / 2.0);
        let rel_err = (wsum - pi_dhalf).abs() / pi_dhalf;
        if rel_err > 1e-10 {
            return Err(SemiflowError::DomainViolation {
                what: "SmolyakGridND: weight sum ≠ π^{D/2} (F(0)=I violated)",
                value: rel_err,
            });
        }
        let sm_nodes: Vec<[f64; D]> = nw.iter().map(|(n, _)| *n).collect();
        let sm_weights: Vec<f64> = nw.iter().map(|(_, w)| *w).collect();

        let cholesky_cache = build_cholesky_cache(&a_ij, &grid)?;
        Ok(Self {
            b_i: alloc::boxed::Box::new(b_i),
            c: alloc::boxed::Box::new(c),
            cholesky_cache,
            sm_nodes,
            sm_weights,
            level: ell,
            grid,
            _f: PhantomData,
        })
    }

    /// Number of Smolyak quadrature nodes (sparse grid size).
    pub fn n_nodes(&self) -> usize {
        self.sm_nodes.len()
    }

    /// Smolyak level `ℓ` used by this kernel.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Return a shared reference to the kernel's grid geometry.
    pub fn grid(&self) -> &GridND<F, D> {
        &self.grid
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl — math §32.4 algorithm with Smolyak quadrature
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat, const D: usize> ChernoffFunction<F> for SmolyakGridND<F, D> {
    type S = GridFnND<F, D>;

    /// Apply one Chernoff step using the Smolyak sparse-grid quadrature.
    ///
    /// Same formula as `AnisotropicShiftChernoffND::apply_into` (math §32.4
    /// eq 32.3) but the sum over `q_idx` uses the Smolyak sparse grid instead
    /// of the full tensor product. Weights may be negative (combination signs).
    fn apply_into(
        &self,
        tau: F,
        src: &GridFnND<F, D>,
        dst: &mut GridFnND<F, D>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "SmolyakGridND::apply_into: tau must be finite >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let ns: [usize; D] = core::array::from_fn(|d| self.grid.axes[d].n);
        let total = src.values.len();
        // Node scale: 2√τ (matches AnisotropicShiftChernoffND, ADR-0112).
        let two_sqrt_tau = from_f64::<F>(2.0_f64) * tau.sqrt();
        // Normalization: π^{-D/2} (required for F(0)=I, ADR-0112 §Decision 1).
        #[allow(clippy::cast_precision_loss)]
        let inv_pi_dhalf =
            from_f64::<F>(core::f64::consts::PI).powf(from_f64::<F>(-(D as f64) / 2.0_f64));
        let n_q = self.sm_nodes.len();

        for flat in 0..total {
            let xk = flat_to_x::<F, D>(flat, &ns, &self.grid);
            let mut b_val = [F::zero(); D];
            (self.b_i)(&xk, &mut b_val);
            let c_val = (self.c)(&xk);
            let exp_factor = (tau * c_val).exp();
            let l_k = &self.cholesky_cache[flat];
            let mut acc = F::zero();
            for qi in 0..n_q {
                // η_q from the Smolyak table (pre-stored as f64, cast to F).
                let eta_q: [F; D] = core::array::from_fn(|d| from_f64::<F>(self.sm_nodes[qi][d]));
                let w_q = from_f64::<F>(self.sm_weights[qi]);
                // y_q = x_k + τ·b(x_k) + 2√τ · L_k · η_q
                let mut l_eta = [F::zero(); D];
                l_k.mat_vec(&eta_q, &mut l_eta);
                let y_q: [F; D] =
                    core::array::from_fn(|d| xk[d] + tau * b_val[d] + two_sqrt_tau * l_eta[d]);
                let f_y = src.sample(&y_q).unwrap_or(F::zero());
                // CRITICAL: weight may be negative — do NOT clamp or abs.
                acc += w_q * f_y;
            }
            dst.values[flat] = exp_factor * inv_pi_dhalf * acc;
        }
        Ok(())
    }

    /// Order 1: same order class as `AnisotropicShiftChernoffND` (math §32.5).
    fn order(&self) -> u32 {
        1
    }

    /// Growth bound: multiplier 1.5, omega 0 (consistent with ADR-0081).
    fn growth(&self) -> Growth<F> {
        Growth::new(from_f64::<F>(1.5_f64), F::zero())
    }
}

// ---------------------------------------------------------------------------
// Inline unit tests — extracted to sibling file per ≤500-line cap.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "smolyak_tests.rs"]
mod tests;
