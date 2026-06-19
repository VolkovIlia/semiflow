#!/usr/bin/env python3
"""tt_coupling_scaling.py — d-scaling of TT-rank for COUPLED generators (§52 v9.1.0).

Companion to tt_coupling_probe.py. The probe showed coupled-A semigroup rank is
BOUNDED (not 4^n). This script answers the (i)-vs-(ii) question: how does the
semigroup TT-rank SCALE with d for two coupling topologies, working in the
ANALYTIC Gaussian-density precision-matrix picture (Rohrbach et al. 2022) so we
can reach larger d without materialising n^d.

For a correlated-Gaussian density N(0, Sigma_T) the TT-rank at bond k is the
numerical rank of the off-diagonal block Sigma_T^{-1}[1..k, k+1..d] (Rohrbach).
We build Sigma_T^{-1} (the PRECISION) directly for two physically-motivated
coupling topologies and measure the per-bond rank + peak vs d:

  TRIDIAGONAL (nearest-neighbour coupling, e.g. chain of correlated assets):
      precision is banded -> off-diagonal blocks rank 1 -> peak rank O(1)
  DENSE (all-to-all constant correlation, e.g. equicorrelated basket):
      precision off-diagonal blocks -> measure actual numerical rank vs d

This separates the achievable narrow class (banded/local coupling: rank O(1),
genuine curse escape) from the hard class (dense correlation: rank may grow).
"""
import numpy as np

np.random.seed(0xBEEF)


def precision_tridiag(d, rho):
    """Precision (inverse cov) of a nearest-neighbour AR(1)-like field.

    Banded SPD precision: diag 1+rho^2 scaled, off-diag -rho on the first
    super/sub diagonal only. This is exactly the precision of a Gauss-Markov
    chain — local (nearest-neighbour) coupling.
    """
    P = np.zeros((d, d))
    for i in range(d):
        P[i, i] = 1.0 + rho**2
        if i + 1 < d:
            P[i, i + 1] = -rho
            P[i + 1, i] = -rho
    return P


def precision_dense_equicorr(d, rho):
    """Precision of an equicorrelated Gaussian: Sigma = (1-rho)I + rho 11^T.

    All-to-all constant correlation -> dense precision via Sherman-Morrison.
    """
    Sigma = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    return np.linalg.inv(Sigma)


def bond_ranks_from_precision(P, eps=1e-8):
    """TT bond ranks of a Gaussian density via off-diagonal precision blocks.

    Rohrbach et al.: r_k = numerical rank of P[0:k, k:d] (the subdiagonal block
    of the precision for the split {1..k}|{k+1..d}).
    """
    d = P.shape[0]
    ranks = []
    for k in range(1, d):
        block = P[0:k, k:d]
        if block.size == 0:
            ranks.append(0)
            continue
        s = np.linalg.svd(block, compute_uv=False)
        r = int(np.sum(s > eps * max(s.max(), 1e-300)))
        ranks.append(max(r, 1) if block.any() else 0)
    return ranks


def main():
    eps = 1e-8
    print("tt_coupling_scaling — semigroup TT-rank d-scaling under coupling (§52 v9.1.0)")
    print("Gaussian-density precision picture (Rohrbach et al. 2022)\n")
    rho = 0.5

    print(f"{'d':>3} | {'TRIDIAG (local) peak_r':>24} | {'DENSE equicorr peak_r':>22} | floor(d/2)")
    print("-" * 72)
    tri_peaks, dense_peaks, half = [], [], []
    for d in (4, 6, 8, 10, 16, 24, 32):
        Ptri = precision_tridiag(d, rho)
        Pden = precision_dense_equicorr(d, rho)
        rtri = max(bond_ranks_from_precision(Ptri, eps))
        rden = max(bond_ranks_from_precision(Pden, eps))
        tri_peaks.append(rtri)
        dense_peaks.append(rden)
        half.append(d // 2)
        print(f"{d:>3} | {rtri:>24} | {rden:>22} | {d // 2}")

    # log-rank slope (rank growth character)
    ds = np.array([4, 6, 8, 10, 16, 24, 32], dtype=float)
    def slope(peaks):
        y = np.log(np.maximum(np.array(peaks, dtype=float), 1.0))
        x = np.log(ds)
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    print()
    print(f"log-rank slope  TRIDIAG (local):    {slope(tri_peaks):+.4f}  "
          f"(0 => O(1) bounded, curse escaped)")
    print(f"log-rank slope  DENSE equicorr:     {slope(dense_peaks):+.4f}  "
          f"(0 => low-rank, ~1 => linear)")
    print()
    print("INTERPRETATION:")
    print("  TRIDIAG local coupling: peak rank is O(1) independent of d ->")
    print("    GENUINE curse escape on a NON-separable (coupled) operator.")
    print("  DENSE equicorr: rank-1 perturbation of identity (Sherman-Morrison) ->")
    print("    precision is rank-1-off-diagonal -> also low rank (special structure).")


if __name__ == "__main__":
    main()
