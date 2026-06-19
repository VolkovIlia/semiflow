# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy's MutableDenseNDimArray / Indexed expression types are dynamically
# typed through __getitem__; Pyright cannot trace them. All operations
# in this module are valid sympy at runtime (verified by the calling
# T_MANIFOLD_CURVATURE + T21N sympy oracle scripts).
"""Reusable sympy helpers for Riemannian curvature computation (v2.8, ADR-0071).

Used by:
- scripts/verify_manifold_curvature.py  (Wave A — T_MANIFOLD_CURVATURE sanity)
- scripts/verify_manifold_torus.py      (Wave A — T21N sympy gate)

Mathematical background:
  Christoffel symbols of the second kind:
    Γ^k_{ij} = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})

  Riemann curvature tensor (components):
    R^l_{ijk} = ∂_i Γ^l_{jk} − ∂_j Γ^l_{ik}
                + Σ_m (Γ^l_{im} Γ^m_{jk} − Γ^l_{jm} Γ^m_{ik})

  Ricci tensor (contraction):
    R_{ij} = R^k_{ikj}  (sum over first and third indices)

  Scalar curvature (full contraction):
    R = g^{ij} R_{ij}
"""

import sympy as sp


def christoffel_symbols(g: sp.Matrix, coords: tuple) -> sp.MutableDenseNDimArray:
    """Compute Γ^k_{ij} from metric matrix g and coordinate tuple.

    Args:
        g:      (d×d) symmetric metric matrix with sympy expressions.
        coords: tuple of d sympy symbols (coordinates).

    Returns:
        Rank-3 MutableDenseNDimArray Gamma[k, i, j] of shape (d, d, d).
    """
    d = len(coords)
    g_inv = g.inv()
    gamma = sp.MutableDenseNDimArray.zeros(d, d, d)
    for k in range(d):
        for i in range(d):
            for j in range(d):
                s = sp.S.Zero
                for l in range(d):
                    s += g_inv[k, l] * (
                        sp.diff(g[j, l], coords[i])
                        + sp.diff(g[i, l], coords[j])
                        - sp.diff(g[i, j], coords[l])
                    )
                gamma[k, i, j] = sp.simplify(sp.Rational(1, 2) * s)
    return gamma


def riemann_curvature(
    gamma: sp.MutableDenseNDimArray, coords: tuple
) -> sp.MutableDenseNDimArray:
    """Compute R^l_{ijk} from Christoffel symbols and coordinates.

    Convention (MTW / Misner-Thorne-Wheeler sign):
        R^l_{kij} = ∂_i Γ^l_{kj} − ∂_j Γ^l_{ki}
                    + Σ_m (Γ^l_{im} Γ^m_{kj} − Γ^l_{jm} Γ^m_{ki})

    Stored as riem[l, k, i, j] ≡ R^l_{kij}.
    Ricci contraction: R_{kj} = R^l_{klj} = sum_l riem[l, k, l, j].

    Args:
        gamma:  Rank-3 array Gamma[k, i, j] from christoffel_symbols().
        coords: tuple of d sympy symbols.

    Returns:
        Rank-4 MutableDenseNDimArray riem[l, k, i, j] of shape (d, d, d, d).
    """
    d = len(coords)
    riem = sp.MutableDenseNDimArray.zeros(d, d, d, d)
    for l in range(d):
        for k in range(d):
            for i in range(d):
                for j in range(d):
                    term = (
                        sp.diff(gamma[l, k, j], coords[i])
                        - sp.diff(gamma[l, k, i], coords[j])
                    )
                    for m in range(d):
                        term += (
                            gamma[l, i, m] * gamma[m, k, j]
                            - gamma[l, j, m] * gamma[m, k, i]
                        )
                    riem[l, k, i, j] = sp.simplify(term)
    return riem


def ricci_tensor(
    riem: sp.MutableDenseNDimArray, coords: tuple
) -> sp.Matrix:
    """Compute R_{kj} = R^l_{klj} = sum_l riem[l, k, l, j].

    Uses the contraction of the first and third indices of riem[l, k, i, j].

    Args:
        riem:   Rank-4 array riem[l, k, i, j] from riemann_curvature().
        coords: tuple of d sympy symbols (used only for dimension).

    Returns:
        (d×d) sympy Matrix Ric[k, j].
    """
    d = len(coords)
    ric = sp.zeros(d, d)
    for k in range(d):
        for j in range(d):
            ric[k, j] = sp.simplify(sum(riem[l, k, l, j] for l in range(d)))
    return ric


def scalar_curvature(ric: sp.Matrix, g_inv: sp.Matrix) -> sp.Expr:
    """Compute R = g^{ij} R_{ij} (full trace of Ricci with metric inverse).

    Args:
        ric:    (d×d) Ricci tensor from ricci_tensor().
        g_inv:  (d×d) inverse metric matrix.

    Returns:
        sympy scalar expression for the Riemann scalar curvature.
    """
    d = g_inv.shape[0]
    s = sp.S.Zero
    for i in range(d):
        for j in range(d):
            s += g_inv[i, j] * ric[i, j]
    return sp.simplify(s)
