//! Cross-diffusion with a constant matrix: `∂ₜu = DΔu + f(t, x, u)` (ADR-0213,
//! math §70).
//!
//! For a constant `K×K` matrix `D = PΛP⁻¹` with real eigenvalues `λ_k ≥ 0`
//! (diagonalisable), pointwise species mixing commutes with any scalar linear
//! spatial operator `L` applied identically to every species, so
//! `e^{tD⊗L} = P·diag(e^{tλ_k L})·P⁻¹` exactly. A reaction–diffusion step mixes the
//! species into modes `m = P⁻¹u`, diffuses mode `k` with an engine for `λ_k L`, and
//! mixes back `u = Pm`; the reaction acts on the physical species. Every engine of
//! the library (1-D/2-D/3-D, all interpolants, Strang compositions, graphs) is
//! reused unchanged.
//!
//! The decomposition is computed by a small dense eigen-solver and VERIFIED before
//! use: `cond₁(P) ≤ 10⁸` and `‖PΛP⁻¹ − D‖_max ≤ 10⁻¹⁰·‖D‖_max`. Complex or negative
//! eigenvalues and defective (or nearly defective) `D` are rejected with the reason.
//! `D` is `f64` data for every scalar type `F`: the decomposition is a constant of
//! the step (a forward-mode `Dual` state still carries its own tangents through the
//! mixing, which is linear; there is no tangent in `D` to drop).

use alloc::{vec, vec::Vec};

use crate::{
    dense_la::{lu_in_place, lu_solve},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
};

/// Largest accepted `cond₁(P)` (beyond: nearly defective, rejected).
const MAX_COND: f64 = 1e8;
/// Relative tolerance of the reconstruction check `‖PΛP⁻¹ − D‖ ≤ tol·‖D‖`.
const RECON_TOL: f64 = 1e-10;
/// Eigenvalues closer than this (relative to `‖D‖`) form one cluster; also the
/// imaginary-part and negative-eigenvalue tolerance.
const CLUSTER_TOL: f64 = 1e-6;

fn domain(what: &'static str, value: f64) -> SemiflowError {
    SemiflowError::DomainViolation { what, value }
}

/// A verified modal decomposition `D = PΛP⁻¹` of a constant diffusion matrix.
#[derive(Clone, Debug)]
pub struct CrossDiffusion<F: SemiflowFloat = f64> {
    k: usize,
    p: Vec<F>,
    p_inv: Vec<F>,
    lambda: Vec<F>,
}

impl<F: SemiflowFloat> CrossDiffusion<F> {
    /// Decompose the row-major `K×K` matrix `d` (`1 ≤ K ≤ 16`).
    ///
    /// # Errors
    /// `DomainViolation`: not a square of `1 ≤ K ≤ 16`; a non-finite entry; complex
    /// eigenvalues; a negative eigenvalue; a defective or nearly defective `D`
    /// (`cond₁(P) > 1e8`, or the reconstruction fails `1e−10·‖D‖`).
    pub fn new(d: &[f64]) -> Result<Self, SemiflowError> {
        let k = (1..=16)
            .find(|k| k * k == d.len())
            .ok_or_else(|| domain("CrossDiffusion: D must be K×K with 1 <= K <= 16", len(d)))?;
        if let Some(&bad) = d.iter().find(|x| !x.is_finite()) {
            return Err(domain("CrossDiffusion: non-finite matrix entry", bad));
        }
        let scale = d.iter().fold(0.0_f64, |m, x| m.max(libm::fabs(*x)));
        if scale == 0.0 {
            return Self::from_modes(&identity(k), &vec![0.0; k]);
        }
        let lambda = real_eigenvalues(d, k, scale)?;
        let p = eigenvectors(d, k, &lambda, scale)?;
        let out = Self::from_parts(k, &p, &lambda)?;
        out.verify(d, scale)?;
        Ok(out)
    }

    /// From a user-supplied decomposition: `p` row-major `K×K` (columns are the
    /// modes), `lambda` the `K` eigenvalues `≥ 0`.
    ///
    /// # Errors
    /// `DomainViolation`: shapes, a non-finite entry, a negative `λ`, a singular or
    /// ill-conditioned `P` (`cond₁(P) > 1e8`).
    pub fn from_modes(p: &[f64], lambda: &[f64]) -> Result<Self, SemiflowError> {
        let k = lambda.len();
        if k == 0 || k > 16 || p.len() != k * k {
            return Err(domain(
                "CrossDiffusion::from_modes: need P K×K and K <= 16 eigenvalues",
                len(p),
            ));
        }
        if let Some(&bad) = p.iter().chain(lambda).find(|x| !x.is_finite()) {
            return Err(domain("CrossDiffusion::from_modes: non-finite entry", bad));
        }
        if let Some(&neg) = lambda.iter().find(|&&x| x < 0.0) {
            return Err(domain(
                "CrossDiffusion::from_modes: negative eigenvalue",
                neg,
            ));
        }
        Self::from_parts(k, p, lambda)
    }

    /// `P⁻¹` by `LU`, the conditioning check, and conversion to `F`.
    fn from_parts(k: usize, p: &[f64], lambda: &[f64]) -> Result<Self, SemiflowError> {
        let p_inv = invert(k, p)?;
        let cond = norm1(k, p) * norm1(k, &p_inv);
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must fail
        if !(cond <= MAX_COND) {
            return Err(domain(
                "CrossDiffusion: eigenvector matrix ill-conditioned (D defective or nearly so)",
                cond,
            ));
        }
        let cast = |v: &[f64]| v.iter().map(|&x| from_f64::<F>(x)).collect();
        Ok(Self {
            k,
            p: cast(p),
            p_inv: cast(&p_inv),
            lambda: cast(lambda),
        })
    }

    /// `‖PΛP⁻¹ − D‖_max ≤ RECON_TOL·‖D‖_max`.
    fn verify(&self, d: &[f64], scale: f64) -> Result<(), SemiflowError> {
        let k = self.k;
        let g = |v: F| v.to_f64().unwrap_or(f64::NAN);
        let mut worst = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                let r: f64 = (0..k)
                    .map(|l| g(self.p[i * k + l]) * g(self.lambda[l]) * g(self.p_inv[l * k + j]))
                    .sum();
                worst = worst.max(libm::fabs(r - d[i * k + j]));
            }
        }
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must fail
        if !(worst <= RECON_TOL * scale) {
            return Err(domain(
                "CrossDiffusion: decomposition does not reproduce D (nearly defective)",
                worst / scale,
            ));
        }
        Ok(())
    }

    /// Number of species `K`.
    #[must_use]
    pub fn species(&self) -> usize {
        self.k
    }

    /// The eigenvalues `λ_k ≥ 0` (mode `k` diffuses with `λ_k`).
    #[must_use]
    pub fn eigenvalues(&self) -> &[F] {
        &self.lambda
    }

    /// The modes `P` (row-major, columns are the modes).
    #[must_use]
    pub fn modes(&self) -> &[F] {
        &self.p
    }

    /// `P⁻¹` (row-major): row `k` projects the species onto mode `k`.
    #[must_use]
    pub fn inverse_modes(&self) -> &[F] {
        &self.p_inv
    }

    /// `out[k·n + i] = Σ_l M[k, l]·src[l·n + i]` (species-major blocks of `n`).
    fn mix(&self, m: &[F], src: &[F], out: &mut [F], n: usize) {
        let k = self.k;
        for (row, block) in out.chunks_exact_mut(n).enumerate().take(k) {
            block.fill(F::zero());
            for (l, sblock) in src.chunks_exact(n).enumerate().take(k) {
                let c = m[row * k + l];
                for (o, &s) in block.iter_mut().zip(sblock) {
                    *o += c * s;
                }
            }
        }
    }

    /// Physical species to modes: `out = P⁻¹·src`, node by node.
    pub(crate) fn to_modes(&self, src: &[F], out: &mut [F], n: usize) {
        self.mix(&self.p_inv, src, out, n);
    }

    /// Modes to physical species: `out = P·src`, node by node.
    pub(crate) fn to_species(&self, src: &[F], out: &mut [F], n: usize) {
        self.mix(&self.p, src, out, n);
    }
}

#[allow(clippy::cast_precision_loss)]
fn len<T>(v: &[T]) -> f64 {
    v.len() as f64
}

fn identity(k: usize) -> Vec<f64> {
    (0..k * k)
        .map(|e| if e / k == e % k { 1.0 } else { 0.0 })
        .collect()
}

/// Real, non-negative eigenvalues of `d` (sorted ascending).
fn real_eigenvalues(d: &[f64], k: usize, scale: f64) -> Result<Vec<f64>, SemiflowError> {
    let (re, im) = eig::eigenvalues(d, k)
        .ok_or_else(|| domain("CrossDiffusion: eigenvalue iteration did not converge", 0.0))?;
    if let Some(&bad) = im.iter().find(|x| libm::fabs(**x) > CLUSTER_TOL * scale) {
        return Err(domain(
            "CrossDiffusion: complex eigenvalues (oscillatory D)",
            bad,
        ));
    }
    if let Some(&neg) = re.iter().find(|&&x| x < -CLUSTER_TOL * scale) {
        return Err(domain(
            "CrossDiffusion: negative eigenvalue (ill-posed backward diffusion)",
            neg,
        ));
    }
    let mut lambda: Vec<f64> = re.iter().map(|&x| x.max(0.0)).collect();
    lambda.sort_by(f64::total_cmp);
    Ok(lambda)
}

/// Eigenvectors (columns of `P`, row-major) for the sorted eigenvalues: one null
/// space of `D − λ̄I` per cluster, whose dimension must equal the multiplicity.
fn eigenvectors(
    d: &[f64],
    k: usize,
    lambda: &[f64],
    scale: f64,
) -> Result<Vec<f64>, SemiflowError> {
    let tol = CLUSTER_TOL * scale;
    let mut p = vec![0.0; k * k];
    let mut col = 0;
    while col < k {
        let end = (col..k)
            .take_while(|&j| lambda[j] - lambda[col] <= tol)
            .count()
            + col;
        let mean = lambda[col..end].iter().sum::<f64>() / len(&lambda[col..end]);
        let shifted: Vec<f64> = (0..k * k)
            .map(|e| d[e] - if e / k == e % k { mean } else { 0.0 })
            .collect();
        let basis = eig::null_space(&shifted, k, tol);
        if basis.len() != end - col {
            return Err(domain(
                "CrossDiffusion: defective D (eigenvector count below the multiplicity)",
                mean,
            ));
        }
        for (off, v) in basis.iter().enumerate() {
            for (i, &x) in v.iter().enumerate() {
                p[i * k + col + off] = x;
            }
        }
        col = end;
    }
    Ok(p)
}

/// `A⁻¹` of the row-major `k×k` matrix by `LU` (singular → `DomainViolation`).
fn invert(k: usize, a: &[f64]) -> Result<Vec<f64>, SemiflowError> {
    let (mut lu, mut piv) = (a.to_vec(), vec![0; k]);
    lu_in_place(k, &mut lu, &mut piv).map_err(|_| {
        domain(
            "CrossDiffusion: singular eigenvector matrix (defective D)",
            0.0,
        )
    })?;
    let (mut inv, mut col, mut work) = (vec![0.0; k * k], vec![0.0; k], vec![0.0; k]);
    for j in 0..k {
        col.fill(0.0);
        col[j] = 1.0;
        lu_solve(k, &lu, &piv, &mut col, &mut work);
        for i in 0..k {
            inv[i * k + j] = col[i];
        }
    }
    Ok(inv)
}

/// `‖A‖₁` (maximum column sum).
fn norm1(k: usize, a: &[f64]) -> f64 {
    (0..k)
        .map(|j| (0..k).map(|i| libm::fabs(a[i * k + j])).sum::<f64>())
        .fold(0.0, f64::max)
}

#[path = "cross_diffusion_eig.rs"]
mod eig;

#[cfg(test)]
#[path = "cross_diffusion_tests.rs"]
mod tests;
