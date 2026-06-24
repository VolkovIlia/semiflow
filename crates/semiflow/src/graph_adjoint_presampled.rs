//! Pre-sampled time-grid path for graph state-adjoint (ADR-0180).
//!
//! Provides [`PreSampledLaplacianSeq`] — a fixed-topology, GL₄-aware weight
//! sequence — and additive constructors for [`MagnusGraphHeatChernoff`] and
//! [`VarCoefMagnusGraphHeatChernoff`] that replay the sequence without any
//! live host callback during evolve.
//!
//! ## Layout invariant (CRITICAL)
//!
//! `vals_seq` is indexed per `(step, abscissa)` in adjoint-sweep order:
//! block `2k` = c₁ sample for step k, block `2k+1` = c₂ sample for step k.
//! The adjoint sweep uses `t_start = (n_steps−1−k)·τ`, so block `2k` is the
//! pre-sampled Laplacian at `t_start + c₁·τ` for that step — identical to
//! what the closure path samples. Getting this wrong causes O(τ²) divergence
//! (the oracle's WRONG variant).

use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    graph::{Laplacian, LaplacianKind},
    graph_signal::GraphSignal,
    magnus_graph::{MagnusGraphHeatChernoff, GL4_C1_F64, GL4_C2_F64},
    magnus_graph_adjoint::{apply_exp_omega4_adj_kernel, apply_exp_omega4_la_adj_kernel},
    scratch::ScratchPool,
    state::State,
    varcoef_magnus_graph::VarCoefMagnusGraphHeatChernoff,
};

// ---------------------------------------------------------------------------
// PreSampledLaplacianSeq
// ---------------------------------------------------------------------------

/// Pre-sampled, fixed-topology, GL₄-aware Laplacian weight sequence (ADR-0180).
///
/// `vals_seq.len() == 2 * n_steps * nnz`, laid out per (step, abscissa) in
/// schedule order `[(0,c₁),(0,c₂),(1,c₁),(1,c₂),…]`; each nnz-block reuses
/// the shared CSR pattern `row_ptr`/`col_idx`.
pub struct PreSampledLaplacianSeq<F> {
    /// Shared CSR row-pointer (len `n_nodes + 1`).
    pub(crate) row_ptr: Vec<usize>,
    /// Shared CSR column indices (len `nnz`).
    pub(crate) col_idx: Vec<u32>,
    /// Flat weight sequence: length `2 * n_steps * nnz`.
    pub(crate) vals_seq: Vec<F>,
    /// Number of adjoint steps.
    pub(crate) n_steps: usize,
    /// Normalization kind (stored for documentation; passed to `from_csr_parts`).
    pub(crate) kind: LaplacianKind,
}

impl<F: SemiflowFloat> PreSampledLaplacianSeq<F> {
    /// Construct and validate a pre-sampled sequence.
    ///
    /// # Errors
    /// `DomainViolation` if `vals_seq.len() != 2 * n_steps * nnz`.
    pub fn new(
        row_ptr: Vec<usize>,
        col_idx: Vec<u32>,
        vals_seq: Vec<F>,
        n_steps: usize,
        kind: LaplacianKind,
    ) -> Result<Self, SemiflowError> {
        let nnz = col_idx.len();
        let expected = 2 * n_steps * nnz;
        if vals_seq.len() != expected {
            return Err(SemiflowError::DomainViolation {
                what: "PreSampledLaplacianSeq: vals_seq.len() must equal 2*n_steps*nnz",
                // cast_precision_loss: diagnostic value only; exact for grid sizes < 2^52.
                #[allow(clippy::cast_precision_loss)]
                value: vals_seq.len() as f64,
            });
        }
        Ok(Self {
            row_ptr,
            col_idx,
            vals_seq,
            n_steps,
            kind,
        })
    }

    /// Reconstruct a `Laplacian` for adjoint step `k`, abscissa index `ci` (0 or 1).
    ///
    /// Block index = `2k + ci`; copies `nnz` values from `vals_seq`.
    fn lap_for_step(&self, k: usize, ci: usize) -> Result<Laplacian<F>, SemiflowError> {
        let nnz = self.col_idx.len();
        let block = 2 * k + ci;
        let start = block * nnz;
        let vals: Vec<F> = self.vals_seq[start..start + nnz].to_vec();
        let n_nodes = self.row_ptr.len().saturating_sub(1);
        Laplacian::from_csr_parts(
            n_nodes,
            self.row_ptr.clone(),
            self.col_idx.clone(),
            vals,
            self.kind,
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: abscissa times for host pre-sampling
// ---------------------------------------------------------------------------

/// Fill `out` (length `2 * n_steps`) with the GL₄ abscissa times in schedule
/// order `[(0,c₁),(0,c₂),(1,c₁),(1,c₂),…]`.
///
/// `t_start` for adjoint step k is `(n_steps − 1 − k) · τ`; the two sample
/// times are `t_start + c₁·τ` and `t_start + c₂·τ`.
///
/// This is the public helper for C/FFI callers; identical ordering is used by
/// `evolve_state_adjoint_presampled`.
// cast_precision_loss: n_steps < 2^52 for any realistic time-stepping scenario.
#[allow(clippy::cast_precision_loss)]
pub fn fill_abscissa_times(t_horizon: f64, n_steps: usize, out: &mut [f64]) {
    let tau = t_horizon / n_steps as f64;
    for k in 0..n_steps {
        let t_start = (n_steps - 1 - k) as f64 * tau;
        out[2 * k] = t_start + GL4_C1_F64 * tau;
        out[2 * k + 1] = t_start + GL4_C2_F64 * tau;
    }
}

// ---------------------------------------------------------------------------
// MagnusGraphHeatChernoff — presampled constructor + evolve
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> MagnusGraphHeatChernoff<F> {
    /// Construct from a pre-sampled GL₄ weight sequence (ADR-0180).
    ///
    /// `seq.n_steps` is stored; `evolve_state_adjoint_presampled` checks it
    /// matches the runtime `n_steps` argument.
    ///
    /// # Errors
    /// Same as [`MagnusGraphHeatChernoff::new`]: `DomainViolation` if
    /// `rho_bar_max <= 0` or `graph.n_nodes() == 0`.
    pub fn from_presampled(
        seq: PreSampledLaplacianSeq<F>,
        rho_bar_max: F,
        convergence_check: bool,
    ) -> Result<PreSampledMagnusAdj<F>, SemiflowError> {
        crate::magnus_graph_helpers::validate_rho(rho_bar_max)?;
        let n_nodes = seq.row_ptr.len().saturating_sub(1);
        if n_nodes == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "from_presampled: n_nodes must be >= 1",
                value: 0.0,
            });
        }
        Ok(PreSampledMagnusAdj {
            seq,
            n_nodes,
            rho_bar_max,
            convergence_check,
        })
    }
}

/// Presampled Magnus graph adjoint (ADR-0180, Magnus K=4 variant).
pub struct PreSampledMagnusAdj<F: SemiflowFloat = f64> {
    pub(crate) seq: PreSampledLaplacianSeq<F>,
    pub(crate) n_nodes: usize,
    pub(crate) rho_bar_max: F,
    pub(crate) convergence_check: bool,
}

impl<F: SemiflowFloat> PreSampledMagnusAdj<F> {
    /// Number of graph nodes.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of construction-time steps (immutable; replayed in evolve).
    #[must_use]
    pub fn n_steps(&self) -> usize {
        self.seq.n_steps
    }

    /// Backward costate sweep using pre-sampled sequence.
    ///
    /// # Errors
    /// `DomainViolation` if `n_steps != self.n_steps()`.
    /// `OutOfMagnusRadius` if convergence check enabled and `τ` too large.
    pub fn evolve_state_adjoint_into(
        &self,
        tau: F,
        n_steps: usize,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        validate_presampled_evolve(
            self.seq.n_steps,
            n_steps,
            self.rho_bar_max,
            self.convergence_check,
            tau,
        )?;
        run_presampled_adj(tau, n_steps, &self.seq, src, dst, scratch)
    }
}

// ---------------------------------------------------------------------------
// VarCoefMagnusGraphHeatChernoff — presampled constructor + evolve
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> VarCoefMagnusGraphHeatChernoff<F> {
    /// Construct from a pre-sampled GL₄ weight + a-sequence (ADR-0180).
    ///
    /// `a_seq.len()` must equal `2 * n_steps * n_nodes`.
    ///
    /// # Errors
    /// `DomainViolation` if `a_seq` length is wrong or `n_nodes == 0`.
    pub fn from_presampled(
        seq: PreSampledLaplacianSeq<F>,
        a_seq: Vec<F>,
        rho_bar_max: F,
        a_sup_max: F,
    ) -> Result<PreSampledVarCoefAdj<F>, SemiflowError> {
        crate::magnus_graph_helpers::validate_rho(rho_bar_max)?;
        let n_nodes = seq.row_ptr.len().saturating_sub(1);
        if n_nodes == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "from_presampled(varcoef): n_nodes must be >= 1",
                value: 0.0,
            });
        }
        let expected_a = 2 * seq.n_steps * n_nodes;
        if a_seq.len() != expected_a {
            return Err(SemiflowError::DomainViolation {
                what: "from_presampled(varcoef): a_seq.len() must equal 2*n_steps*n_nodes",
                #[allow(clippy::cast_precision_loss)]
                value: a_seq.len() as f64,
            });
        }
        Ok(PreSampledVarCoefAdj {
            seq,
            a_seq,
            n_nodes,
            rho_bar_max,
            a_sup_max,
        })
    }
}

/// Presampled `VarCoef` Magnus graph adjoint (ADR-0180, `VarCoef` variant).
pub struct PreSampledVarCoefAdj<F: SemiflowFloat = f64> {
    pub(crate) seq: PreSampledLaplacianSeq<F>,
    pub(crate) a_seq: Vec<F>,
    pub(crate) n_nodes: usize,
    pub(crate) rho_bar_max: F,
    /// Stored for convergence-radius check and FFI surface; not used by the
    /// pure-replay path (all values come from `a_seq`).
    #[allow(dead_code)]
    pub(crate) a_sup_max: F,
}

impl<F: SemiflowFloat> PreSampledVarCoefAdj<F> {
    /// Number of graph nodes.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of construction-time steps.
    #[must_use]
    pub fn n_steps(&self) -> usize {
        self.seq.n_steps
    }

    /// Backward costate sweep using pre-sampled Laplacian + a-sequences.
    ///
    /// # Errors
    /// `DomainViolation` if `n_steps != self.n_steps()`.
    pub fn evolve_state_adjoint_into(
        &self,
        tau: F,
        n_steps: usize,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        validate_presampled_evolve(self.seq.n_steps, n_steps, self.rho_bar_max, false, tau)?;
        run_presampled_varcoef_adj(
            tau,
            n_steps,
            &self.seq,
            &self.a_seq,
            self.n_nodes,
            src,
            dst,
            scratch,
        )
    }
}

// ---------------------------------------------------------------------------
// Shared private helpers
// ---------------------------------------------------------------------------

/// Validate `n_steps` match and optional Magnus radius check.
fn validate_presampled_evolve<F: SemiflowFloat>(
    ctor_steps: usize,
    evolve_steps: usize,
    rho_bar: F,
    conv_check: bool,
    tau: F,
) -> Result<(), SemiflowError> {
    if evolve_steps != ctor_steps {
        return Err(SemiflowError::DomainViolation {
            what: "presampled: n_steps at evolve must equal n_steps at construction",
            #[allow(clippy::cast_precision_loss)]
            value: evolve_steps as f64,
        });
    }
    crate::magnus_graph_helpers::validate_tau(tau)?;
    if conv_check {
        crate::magnus_graph_helpers::validate_magnus_radius(rho_bar, tau)?;
    }
    Ok(())
}

/// Core adjoint sweep for Magnus variant using pre-sampled sequence.
fn run_presampled_adj<F: SemiflowFloat>(
    tau: F,
    n_steps: usize,
    seq: &PreSampledLaplacianSeq<F>,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if n_steps == 0 {
        dst.copy_from(src);
        return Ok(());
    }
    let mut lam = src.clone();
    let mut lam_next = dst.clone();
    for k in 0..n_steps {
        let lap1 = seq.lap_for_step(k, 0)?;
        let lap2 = seq.lap_for_step(k, 1)?;
        apply_exp_omega4_adj_kernel(&lap1, &lap2, tau, &lam, &mut lam_next, scratch);
        core::mem::swap(&mut lam, &mut lam_next);
    }
    dst.copy_from(&lam);
    Ok(())
}

/// Core adjoint sweep for `VarCoef` variant using pre-sampled sequences.
// All 8 arguments are required by the GL4 Magnus adjoint protocol (ADR-0180).
#[allow(clippy::too_many_arguments)]
fn run_presampled_varcoef_adj<F: SemiflowFloat>(
    tau: F,
    n_steps: usize,
    seq: &PreSampledLaplacianSeq<F>,
    a_seq: &[F],
    n_nodes: usize,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if n_steps == 0 {
        dst.copy_from(src);
        return Ok(());
    }
    let mut lam = src.clone();
    let mut lam_next = dst.clone();
    for k in 0..n_steps {
        let lap1 = seq.lap_for_step(k, 0)?;
        let lap2 = seq.lap_for_step(k, 1)?;
        let a1 = &a_seq[2 * k * n_nodes..(2 * k + 1) * n_nodes];
        let a2 = &a_seq[(2 * k + 1) * n_nodes..(2 * k + 2) * n_nodes];
        let sqrt_a1: Vec<F> = a1.iter().map(|&x| x.sqrt()).collect();
        let sqrt_a2: Vec<F> = a2.iter().map(|&x| x.sqrt()).collect();
        crate::magnus_graph_adjoint::validate_varcoef_sqrt_a(&sqrt_a1, &sqrt_a2)?;
        let mut sa1_buf = scratch.take_vec(n_nodes);
        let mut sa2_buf = scratch.take_vec(n_nodes);
        sa1_buf.copy_from_slice(&sqrt_a1);
        sa2_buf.copy_from_slice(&sqrt_a2);
        apply_exp_omega4_la_adj_kernel(
            &lap1,
            &sa1_buf,
            &lap2,
            &sa2_buf,
            tau,
            &lam,
            &mut lam_next,
            scratch,
        );
        scratch.return_vec(sa2_buf);
        scratch.return_vec(sa1_buf);
        core::mem::swap(&mut lam, &mut lam_next);
    }
    dst.copy_from(&lam);
    Ok(())
}

// GL4 constants re-exported at crate level as pub(crate) for internal use;
// `fill_abscissa_times` uses the raw f64 constants from magnus_graph directly.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::Graph,
        magnus_graph::{GL4_C1_F64, GL4_C2_F64},
    };

    #[test]
    fn from_csr_parts_roundtrip() {
        let n = 4usize;
        let g = Graph::<f64>::path(n);
        let lap = Laplacian::assemble_combinatorial(&g);
        let rp = lap.row_ptr().to_vec();
        let ci = lap.col_idx().to_vec();
        let vs = lap.vals().to_vec();
        let lap2 = Laplacian::from_csr_parts(n, rp, ci, vs, LaplacianKind::Combinatorial).unwrap();
        assert_eq!(lap.vals(), lap2.vals());
    }

    #[test]
    fn presampled_seq_wrong_len() {
        let g = Graph::<f64>::path(4);
        let lap = Laplacian::assemble_combinatorial(&g);
        let rp = lap.row_ptr().to_vec();
        let ci = lap.col_idx().to_vec();
        // Only 1 block instead of 2*n_steps*nnz
        let err =
            PreSampledLaplacianSeq::new(rp, ci, vec![0.0_f64; 3], 2, LaplacianKind::Combinatorial);
        assert!(err.is_err());
    }

    #[test]
    fn fill_abscissa_times_order() {
        let n_steps = 4usize;
        let t_horizon = 0.5_f64;
        #[allow(clippy::cast_precision_loss)]
        let tau = t_horizon / n_steps as f64;
        let mut out = vec![0.0_f64; 2 * n_steps];
        fill_abscissa_times(t_horizon, n_steps, &mut out);
        // Block 0: step 0 (last adjoint step), t_start = (n_steps-1)*tau
        #[allow(clippy::cast_precision_loss)]
        let t_start0 = (n_steps - 1) as f64 * tau;
        assert!((out[0] - (t_start0 + GL4_C1_F64 * tau)).abs() < 1e-15);
        assert!((out[1] - (t_start0 + GL4_C2_F64 * tau)).abs() < 1e-15);
    }
}
