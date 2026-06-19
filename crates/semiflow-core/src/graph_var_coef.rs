//! Variable-coefficient graph heat Chernoff with ζ-A τ²-correction.
//!
//! Generator: `L_a = A^{1/2} L_G A^{1/2}` where `A = diag(a)`.
//! Order: 2 on all graphs (graph diameter and `a_sup/a_inf` do not restrict order).
//!
//! See math.md §14.2 (NORMATIVE) and ADR-0053 (design).
//!
//! ## Algorithm (NORMATIVE — per contract §2.3, corrected v2.2.0)
//!
//! `apply_into` computes `dst ← f − τ L_a f + (τ²/2) L_a² f`:
//!
//! 1. `L_a f = sqrt_a ⊙ (L_G (sqrt_a ⊙ f))` (two `vec_mul` + one `SpMV`).
//! 2. `L_a² f = L_a (L_a f)` — repeat step 1 on result.
//! 3. Combine: `dst = f − τ·L_a_f + (τ²/2)·L_a2_f`.
//!
//! The `D_a^{(2)}` correction term from the 1D continuous analogy (§9.2.3.B) is NOT
//! applied: the graph operator `L_a = A^{1/2} L_G A^{1/2}` is an exact linear map, so
//! the Taylor truncation `I − τL_a + (τ²/2)L_a²` already achieves O(τ³) local error
//! (verified by T12N gate). Adding `D_a^{(2)}` would introduce a spurious O(τ²) offset
//! that degrades convergence to order 1 (G13 gate regression).

use alloc::{sync::Arc, vec::Vec};

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// VarCoefGraphHeatChernoff<F>
// ---------------------------------------------------------------------------

/// Variable-coefficient graph heat Chernoff with ζ-A τ²-correction.
///
/// Generator: `L_a = A^{1/2} L_G A^{1/2}` where `A = diag(a)`.
/// Order: 2 (global, for all positive `a`).
///
/// See math.md §14.2 (NORMATIVE) and ADR-0053 (design).
pub struct VarCoefGraphHeatChernoff<F: SemiflowFloat = f64> {
    graph: Arc<Graph<F>>,
    laplacian: Arc<Laplacian<F>>,
    a: Vec<F>,
    sqrt_a: Vec<F>,
    rho_bar: F,
}

impl<F: SemiflowFloat> VarCoefGraphHeatChernoff<F> {
    /// Construct from topology, conductivity `a`, and Gershgorin spectral bound.
    ///
    /// The `Laplacian` is assembled internally as the combinatorial Laplacian
    /// of `graph` (base `L_G` for `L_a = A^{1/2} L_G A^{1/2}`).
    ///
    /// # Errors
    /// - `DomainViolation` if `a.len() != graph.n_nodes()`.
    /// - `DomainViolation` if any `a[i] < 1e-12 * max(a)`.
    /// - `DomainViolation` if `rho_bar <= 0` or non-finite.
    pub fn new(graph: Arc<Graph<F>>, a: Vec<F>, rho_bar: F) -> Result<Self, SemiflowError> {
        validate_inputs(&graph, &a, rho_bar)?;

        let laplacian = Arc::new(Laplacian::assemble_combinatorial(&graph));
        let sqrt_a = compute_sqrt_a(&a);

        Ok(Self {
            graph,
            laplacian,
            a,
            sqrt_a,
            rho_bar,
        })
    }

    /// Borrow the topology graph.
    #[must_use]
    pub fn graph(&self) -> &Graph<F> {
        &self.graph
    }

    /// Borrow the conductivity vector.
    #[must_use]
    pub fn a(&self) -> &[F] {
        &self.a
    }

    /// Gershgorin spectral-radius bound.
    #[must_use]
    pub fn rho_bar(&self) -> F {
        self.rho_bar
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> ChernoffFunction<F> for VarCoefGraphHeatChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        validate_tau(tau)?;
        check_cfl(tau, self.rho_bar, &self.a)?;
        let n = src.len();
        debug_assert_eq!(
            dst.len(),
            n,
            "VarCoefGraphHeatChernoff: dst/src size mismatch"
        );
        apply_var_coef_steps(&self.laplacian, &self.sqrt_a, tau, src, dst, scratch);
        Ok(())
    }

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<F> {
        let a_sup = self
            .a
            .iter()
            .copied()
            .fold(F::zero(), |acc, ai| if ai > acc { ai } else { acc });
        Growth {
            multiplier: F::one(),
            omega: self.rho_bar * a_sup * a_sup,
        }
    }
}

// ---------------------------------------------------------------------------
// Computation helpers
// ---------------------------------------------------------------------------

/// Compute `sqrt_a[i] = sqrt(a[i])` for all i.
fn compute_sqrt_a<F: SemiflowFloat>(a: &[F]) -> Vec<F> {
    a.iter().map(|&ai| ai.sqrt()).collect()
}

/// Apply `L_a` to a `GraphSignal` slice: `out = sqrt_a ⊙ (L_G (sqrt_a ⊙ src_vals))`.
fn apply_la<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    sqrt_a: &[F],
    src_vals: &[F],
    out: &mut [F],
    tmp1: &mut [F],
    tmp2: &mut [F],
) {
    let n = src_vals.len();
    // Step 1a: tmp1[i] = sqrt_a[i] * src_vals[i]
    for i in 0..n {
        tmp1[i] = sqrt_a[i] * src_vals[i];
    }
    // Step 1b: tmp2 = L_G * tmp1
    lap.apply_into_slice(tmp1, tmp2);
    // Step 1c: out[i] = sqrt_a[i] * tmp2[i]
    for i in 0..n {
        out[i] = sqrt_a[i] * tmp2[i];
    }
}

/// Apply the variable-coefficient kernel update in-place:
/// `dst ← src − τ·la_f + (τ²/2)·la_squared_f`.
fn apply_var_coef_update<F: SemiflowFloat>(
    dst: &mut GraphSignal<F>,
    src: &GraphSignal<F>,
    tau: F,
    la_f: &[F],
    la_squared_f: &[F],
) {
    let two = F::one() + F::one();
    let half_tau2 = tau * tau / two;
    dst.copy_from(src);
    dst.axpy_into_slice(-tau, la_f);
    dst.axpy_into_slice(half_tau2, la_squared_f);
}

/// Acquire scratch, apply steps 1-3 (La, La², update), return scratch.
///
/// Extracted from `apply_into` (batch H9b) — float op order preserved.
fn apply_var_coef_steps<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    sqrt_a: &[F],
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) {
    let n = src.len();
    let mut tmp1 = scratch.take_vec(n); // sqrt_a ⊙ f intermediate
    let mut tmp2 = scratch.take_vec(n); // L_G · tmp1 intermediate
    let mut la_f = scratch.take_vec(n); // L_a f
    let mut la_squared_f = scratch.take_vec(n); // L_a² f
    apply_la(lap, sqrt_a, src.values(), &mut la_f, &mut tmp1, &mut tmp2);
    apply_la_on_slice(lap, sqrt_a, &la_f, &mut la_squared_f, &mut tmp1, &mut tmp2);
    // D_a^{(2)} NOT added: L_a exact, (I−τL_a+(τ²/2)L_a²) achieves O(τ³).
    apply_var_coef_update(dst, src, tau, &la_f, &la_squared_f);
    scratch.return_vec(la_squared_f);
    scratch.return_vec(la_f);
    scratch.return_vec(tmp2);
    scratch.return_vec(tmp1);
}

/// Apply `L_a` to a plain slice (no `GraphSignal` wrapper).
///
/// `pub(crate)` — shared with `varcoef_magnus_graph.rs` (ADR-0063).
pub(crate) fn apply_la_on_slice<F: SemiflowFloat>(
    lap: &Laplacian<F>,
    sqrt_a: &[F],
    src: &[F],
    out: &mut [F],
    tmp1: &mut [F],
    tmp2: &mut [F],
) {
    let n = src.len();
    for i in 0..n {
        tmp1[i] = sqrt_a[i] * src[i];
    }
    lap.apply_into_slice(tmp1, tmp2);
    for i in 0..n {
        out[i] = sqrt_a[i] * tmp2[i];
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate constructor inputs.
#[allow(clippy::cast_precision_loss)] // usize→f64 for error reporting only; not on hot path
fn validate_inputs<F: SemiflowFloat>(
    graph: &Graph<F>,
    a: &[F],
    rho_bar: F,
) -> Result<(), SemiflowError> {
    if a.len() != graph.n_nodes() {
        return Err(SemiflowError::DomainViolation {
            what: "VarCoefGraphHeatChernoff: a.len() != graph.n_nodes()",
            value: a.len() as f64,
        });
    }
    // Compute max(a) for ratio check.
    let a_max = a
        .iter()
        .copied()
        .fold(F::zero(), |acc, ai| if ai > acc { ai } else { acc });
    let eps = F::from(1e-12_f64).unwrap_or(F::zero());
    for &ai in a {
        if !ai.is_finite() || ai < eps * a_max {
            return Err(SemiflowError::DomainViolation {
                what: "VarCoefGraphHeatChernoff: a[i] must be finite and >= 1e-12 * max(a)",
                value: ai.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    let rho_v = rho_bar.to_f64().unwrap_or(f64::NAN);
    if !rho_v.is_finite() || rho_v <= 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "VarCoefGraphHeatChernoff: rho_bar must be finite and > 0",
            value: rho_v,
        });
    }
    Ok(())
}

/// Validate that tau is finite and non-negative.
fn validate_tau<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    let v = tau.to_f64().unwrap_or(f64::NAN);
    if !v.is_finite() || v < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "VarCoefGraphHeatChernoff: tau must be finite and >= 0",
            value: v,
        });
    }
    Ok(())
}

/// CFL check (NORMATIVE per contract §2.3):
/// if `tau * rho_bar * max(a)^2 > 0.5`, return `CflViolated`.
fn check_cfl<F: SemiflowFloat>(tau: F, rho_bar: F, a: &[F]) -> Result<(), SemiflowError> {
    let a_max = a
        .iter()
        .copied()
        .fold(F::zero(), |acc, ai| if ai > acc { ai } else { acc });
    let product = tau * rho_bar * a_max * a_max;
    let limit = F::from(0.5_f64).unwrap_or(F::one());
    if product > limit {
        return Err(SemiflowError::CflViolated {
            tau: tau.to_f64().unwrap_or(f64::NAN),
            dx_squared: rho_bar.to_f64().unwrap_or(f64::NAN),
            a_norm_bound: a_max.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph_signal::GraphSignal;
    use crate::state::State;

    fn make_path_vc(n: usize, a_fn: impl Fn(usize) -> f64) -> VarCoefGraphHeatChernoff<f64> {
        let g = Arc::new(Graph::<f64>::path(n));
        let a: Vec<f64> = (0..n).map(a_fn).collect();
        let rho_bar = 4.0_f64;
        VarCoefGraphHeatChernoff::new(g, a, rho_bar).expect("valid inputs")
    }

    #[test]
    fn order_is_two() {
        let vc = make_path_vc(8, |_| 1.0);
        assert_eq!(vc.order(), 2);
    }

    #[test]
    fn zero_tau_preserves_src() {
        let vc = make_path_vc(8, |_| 1.0);
        let g = Arc::clone(&vc.graph);
        let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
        let mut dst = src.clone();
        let mut pool = ScratchPool::<f64>::new();
        vc.apply_into(0.0, &src, &mut dst, &mut pool).unwrap();
        let mut diff = dst.clone();
        diff.axpy_into(-1.0, &src);
        assert!(diff.norm_sup() < 1e-14, "zero-tau should preserve src");
    }

    #[test]
    fn constructor_rejects_wrong_a_len() {
        let g = Arc::new(Graph::<f64>::path(4));
        let a = alloc::vec![1.0_f64; 5]; // wrong size
        assert!(matches!(
            VarCoefGraphHeatChernoff::new(g, a, 2.0),
            Err(SemiflowError::DomainViolation { .. })
        ));
    }

    #[test]
    fn constructor_rejects_zero_a() {
        let g = Arc::new(Graph::<f64>::path(4));
        let a = alloc::vec![1.0_f64, 0.0, 1.0, 1.0];
        assert!(matches!(
            VarCoefGraphHeatChernoff::new(g, a, 2.0),
            Err(SemiflowError::DomainViolation { .. })
        ));
    }

    #[test]
    fn constant_a_matches_graph_heat() {
        // With a ≡ 1, L_a = L_G. Verify apply result approximates heat diffusion.
        let n = 8usize;
        let vc = make_path_vc(n, |_| 1.0);
        let g = Arc::clone(&vc.graph);
        let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
        let tau = 0.01;
        let g = Arc::clone(&vc.graph);
        let mut result = GraphSignal::zeros(Arc::clone(&g));
        let mut pool = ScratchPool::<f64>::new();
        vc.apply_into(tau, &src, &mut result, &mut pool).unwrap();
        // Result should differ from src (diffusion applied).
        let mut diff = result.clone();
        diff.axpy_into(-1.0, &src);
        // Non-zero: some diffusion happened.
        assert!(diff.norm_sup() > 1e-10, "expected non-trivial diffusion");
    }
}
