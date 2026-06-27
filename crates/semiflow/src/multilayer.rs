//! Multilayer thermal stack for stiff multi-material conduction (§57, issue #14).
//!
//! Physics: ρc ∂ₜT = ∂ₓ(k ∂ₓT) → M dT/dt = `L_k` T (M = diag(ρc)).
//! Propagation via [`multilayer_evolve`] (Krylov expmv, §55.3) or per-step via
//! [`MassWeightedConservativeChernoff`] (CN Thomas, order 2). Authority: ADR-0188 §57.

use alloc::vec::Vec;

use crate::{
    boundary::BoundaryPolicy,
    chernoff::{ChernoffFunction, Growth},
    conservative::thomas_solve,
    conservative_assemble::{assemble_conservative_csr_1d, build_faces},
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    graph_krylov::KrylovPath,
    mass_operator::mass_lumped_evolve,
    scratch::ScratchPool,
    symmetric_operator::SymmetricOperator,
};

/// One material layer (thickness [m], k [W/(m·K)], `rho_c` [J/(m³·K)]).
#[derive(Debug, Clone, Copy)]
pub struct Layer<F: SemiflowFloat = f64> {
    /// Layer thickness [m].
    pub thickness: F,
    /// Thermal conductivity [W/(m·K)].
    pub k: F,
    /// Volumetric heat capacity ρc [J/(m³·K)].
    pub rho_c: F,
}

/// Uniform-dx discretisation of a 1-D multilayer stack.
///
/// Built via [`MultilayerStack::from_layers`]; propagated via [`multilayer_evolve`].
#[derive(Debug, Clone)]
pub struct MultilayerStack<F: SemiflowFloat = f64> {
    /// Grid covering [0, L]; `grid.n` = `total_cells` + 1.
    pub grid: Grid1D<F>,
    /// Node-sampled thermal conductivity k.
    pub k_nodes: Vec<F>,
    /// Node-sampled volumetric heat capacity ρc.
    pub rho_c_nodes: Vec<F>,
}

impl<F: SemiflowFloat> MultilayerStack<F> {
    /// Build from layer definitions with uniform target cell size.
    ///
    /// Cells per layer: `max(1, round(thickness / target_dx))`.
    /// Interface nodes assigned to the left layer (ADR-0188 §57.1).
    ///
    /// # Errors
    /// `DomainViolation` if `layers` is empty or the resulting `n < 4`.
    pub fn from_layers(layers: &[Layer<F>], target_dx: F) -> Result<Self, SemiflowError> {
        if layers.is_empty() {
            return Err(SemiflowError::DomainViolation {
                what: "MultilayerStack::from_layers: empty layers slice",
                value: 0.0,
            });
        }
        let mut n_cells: Vec<usize> = Vec::with_capacity(layers.len());
        for lay in layers {
            let c = (lay.thickness / target_dx).round().to_f64().unwrap_or(1.0_f64).max(1.0_f64);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            n_cells.push(c as usize);
        }
        let mut cumul = vec![0_usize];
        for &c in &n_cells {
            cumul.push(cumul.last().copied().unwrap_or(0) + c);
        }
        let total_cells = cumul.last().copied().unwrap_or(1);
        let n = total_cells + 1;
        let total_len = layers.iter().fold(F::zero(), |a, l| a + l.thickness);
        let grid = Grid1D::new_generic(F::zero(), total_len, n)?;
        let mut k_nodes = Vec::with_capacity(n);
        let mut rho_c_nodes = Vec::with_capacity(n);
        for i in 0..n {
            let idx = cumul.partition_point(|&c| c < i).saturating_sub(1).min(layers.len() - 1);
            k_nodes.push(layers[idx].k);
            rho_c_nodes.push(layers[idx].rho_c);
        }
        Ok(Self { grid, k_nodes, rho_c_nodes })
    }

    /// Assemble `A = −L_k` and return `(A, rho_c_nodes)`.
    ///
    /// # Errors
    /// Propagates from [`assemble_conservative_csr_1d`].
    pub fn to_stiffness_and_mass(
        &self,
        boundary: BoundaryPolicy<F>,
    ) -> Result<(SymmetricOperator<F>, Vec<F>), SemiflowError> {
        let a = assemble_conservative_csr_1d(self.grid, &self.k_nodes, None, boundary)?;
        Ok((a, self.rho_c_nodes.clone()))
    }
}

/// Evolve `u0` by `tau` under `M⁻¹ L_k` via Krylov expmv (§55.3 bridge).
///
/// # Errors
/// Propagates from [`MultilayerStack::to_stiffness_and_mass`] or [`mass_lumped_evolve`].
#[allow(clippy::too_many_arguments)]
pub fn multilayer_evolve<F: SemiflowFloat>(
    stack: &MultilayerStack<F>,
    boundary: BoundaryPolicy<F>,
    tau: F,
    u0: &[F],
    out: &mut [F],
    path: KrylovPath,
    tol: F,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let (a, masses) = stack.to_stiffness_and_mass(boundary)?;
    mass_lumped_evolve(&a, &masses, tau, u0, out, path, tol, scratch)
}

/// CN-step Chernoff function for the mass-weighted system `M dT/dt = L_k T`.
///
/// Each call solves `(M − ½τL_k) T^{n+1} = (M + ½τL_k) T^n` via Thomas.
/// Order 2, contractive (§57.2). For long stiff runs, prefer [`multilayer_evolve`].
#[derive(Debug, Clone)]
pub struct MassWeightedConservativeChernoff<F: SemiflowFloat = f64> {
    faces: Vec<F>, // k_harm/dx per interface, n−1 entries
    rho_c: Vec<F>, // nodal ρc, n entries
    dx: F,
    grid: Grid1D<F>,
}

impl<F: SemiflowFloat> MassWeightedConservativeChernoff<F> {
    /// Build from a `MultilayerStack` (Neumann BCs at both ends).
    ///
    /// # Errors
    /// Propagates from [`build_faces`] (e.g. k ≤ 0).
    pub fn from_stack(stack: &MultilayerStack<F>) -> Result<Self, SemiflowError> {
        let dx = stack.grid.dx();
        let faces = build_faces(&stack.k_nodes, dx, None)?;
        Ok(Self { faces, rho_c: stack.rho_c_nodes.clone(), dx, grid: stack.grid })
    }
}

impl<F: SemiflowFloat> ChernoffFunction<F> for MassWeightedConservativeChernoff<F> {
    type S = GridFn1D<F>;

    #[inline]
    fn order(&self) -> u32 { 2 }

    #[inline]
    fn growth(&self) -> Growth<F> { Growth::contraction() }

    /// CN step `(M − ½τL_k) T^{n+1} = (M + ½τL_k) T^n` solved by Thomas.
    #[allow(clippy::similar_names)]
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn1D<F>,
        dst: &mut GridFn1D<F>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "MassWeightedConservativeChernoff: tau must be finite and >= 0",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let ht = tau / (F::one() + F::one());
        let n = self.rho_c.len();
        let dx = self.dx;
        let mut sub_d = vec![F::zero(); n];
        let mut diag = vec![F::zero(); n];
        let mut sup_d = vec![F::zero(); n];
        let mut rhs = vec![F::zero(); n];
        for i in 0..n {
            let t_l = if i > 0 { self.faces[i - 1] / dx } else { F::zero() };
            let t_r = if i + 1 < n { self.faces[i] / dx } else { F::zero() };
            let s_l = if i > 0 { src.values[i - 1] } else { F::zero() };
            let s_r = if i + 1 < n { src.values[i + 1] } else { F::zero() };
            sub_d[i] = -ht * t_l;
            diag[i] = self.rho_c[i] + ht * (t_l + t_r);
            sup_d[i] = -ht * t_r;
            rhs[i] = (self.rho_c[i] - ht * (t_l + t_r)) * src.values[i]
                + ht * t_l * s_l + ht * t_r * s_r;
        }
        dst.values.resize(n, F::zero());
        thomas_solve(n, &sub_d, &diag, &sup_d, &rhs, &mut dst.values)?;
        dst.grid = self.grid;
        Ok(())
    }
}
