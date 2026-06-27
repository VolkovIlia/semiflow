//! Fréchet sensitivity w.r.t. individual CSR entries of a symmetric operator (ADR-0186, §55.5).
//!
//! [`EntrySensitivity`] implements [`GeneratorSensitivity`] for a stencil `∂A/∂L_{ij}`.
//! Plugs into the unchanged `graph_expmv_frechet` infrastructure (A2).

use alloc::vec::Vec;

use crate::{error::SemiflowError, float::SemiflowFloat, graph_sensitivity::GeneratorSensitivity};

/// Parameter list for single-entry symmetric-stencil sensitivity (§55.5).
///
/// Each entry `(i, j)` in `entries` (with `i ≤ j`) represents the pair `{L_{ij}, L_{ji}}`
/// which are tied by symmetry.  `n_nodes` is the operator dimension.
pub struct EntrySensitivity {
    /// List of `(row, col)` pairs with `row ≤ col`.  One parameter per entry.
    pub entries: Vec<(usize, usize)>,
    /// Operator dimension (number of nodes).
    pub n_nodes: usize,
}

impl<F: SemiflowFloat> GeneratorSensitivity<F> for EntrySensitivity {
    fn n_params(&self) -> usize {
        self.entries.len()
    }

    /// `out ← (∂A/∂L_{ij}) · v` for the k-th parameter.
    ///
    /// Since `A = −L` (ADR-0186 sign convention), `∂A/∂L_{ij} = −∂L/∂L_{ij}`.
    ///
    /// Stencil (§55.5):
    /// - Diagonal (`i == j`): `(∂L/∂L_{ii}) v = v[i] e_i`, so `out[i] = −v[i]`.
    /// - Off-diagonal (`i ≠ j`): `(∂L/∂L_{ij}) v = v[j] e_i + v[i] e_j`,
    ///   so `out[i] = −v[j]`, `out[j] = −v[i]`.
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if `k ≥ self.n_params()`.
    fn apply_param_deriv(
        &self,
        k: usize,
        _t: F,
        v: &[F],
        out: &mut [F],
    ) -> Result<(), SemiflowError> {
        if k >= self.entries.len() {
            return Err(SemiflowError::DomainViolation {
                what: "EntrySensitivity::apply_param_deriv: k out of range",
                #[allow(clippy::cast_precision_loss)]
                value: k as f64,
            });
        }
        for x in out.iter_mut() {
            *x = F::zero();
        }
        let (i, j) = self.entries[k];
        if i == j {
            out[i] = -v[i]; // diagonal entry: ∂A/∂L_{ii} = −e_i eᵢᵀ
        } else {
            out[i] = -v[j]; // off-diagonal: ∂A/∂L_{ij} = −(e_i eⱼᵀ + e_j eᵢᵀ)
            out[j] = -v[i];
        }
        Ok(())
    }
}
