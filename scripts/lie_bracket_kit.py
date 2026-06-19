# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy's MutableDenseNDimArray / Array indexing types are dynamically typed
# through __getitem__; Pyright cannot trace them. All operations in this module
# are valid sympy at runtime (verified by T_HORM oracle + future verify_* scripts).
"""Reusable sympy helpers for vector-field Lie-bracket computation (v3.1, ADR-0077).

Used by:
- scripts/verify_hormander_kolmogorov.py  (Wave A — T_HORM oracle, 4 sub-checks)
- Future v3.x research on Hörmander commutator expansions and step-checker
  validation for higher-step Carnot groups.

Mathematical background:
  Lie bracket (directional-derivative formula):
    [X, Y]^i = X^j · ∂_j Y^i − Y^j · ∂_j X^i   (Einstein summation over j)

  This is the coordinate expression of the Lie derivative:
    [X, Y] = (DY) · X − (DX) · Y

  Step-r bracket-generating (Hörmander 1967 *Acta Math.* §1 Definition 28.1):
    span{ X_i(x), [X_i, X_j](x), [X_i,[X_j,X_k]](x), ... } = T_x M
    at every x ∈ M, where iterated brackets up to depth r suffice.

Functions:
  lie_bracket(X, Y, coords)            → sp.Array, length D
  nested_bracket(X, Y, depth, coords)  → sp.Array, length D
  generates_T(brackets, coords)        → bool (Hörmander condition check)

References:
  - Hörmander 1967 *Acta Math.* 119:1, pp. 147-171 (bracket-generating cond.)
  - Kolmogorov 1934 *Math. Annalen* 108, pp. 149-160 (step-2 example)
  - math.md §28.2 (Definition 28.1, Definition 28.2)
"""

from typing import Optional

import sympy as sp


def lie_bracket(X: sp.Array, Y: sp.Array, coords: tuple) -> sp.Array:
    """Compute the Lie bracket [X, Y] of two vector fields.

    Formula (Einstein summation):
        [X, Y]^i = X^j · ∂_j Y^i − Y^j · ∂_j X^i

    Args:
        X:      D-element sp.Array representing the first vector field.
        Y:      D-element sp.Array representing the second vector field.
        coords: D-tuple of sp.Symbol (coordinate variables).

    Returns:
        sp.Array of length D representing [X, Y].

    Example (Kolmogorov phase space):
        x, v = sp.symbols('x v')
        X0 = sp.Array([v, 0])         # X₀ = v·∂_x
        X1 = sp.Array([0, 1])         # X₁ = ∂_v
        bracket = lie_bracket(X1, X0, (x, v))
        # → sp.Array([1, 0])  i.e. +∂_x (generates missing x-direction)
        # Derivation: [X₁,X₀]f = ∂_v(v·∂_x f) - v·∂_x(∂_v f) = ∂_x f
    """
    D = len(coords)
    out = sp.MutableDenseNDimArray([sp.S.Zero] * D)
    for i in range(D):
        val = sp.S.Zero
        for j in range(D):
            dYi_dxj = sp.diff(Y[i], coords[j])
            dXi_dxj = sp.diff(X[i], coords[j])
            val = val + X[j] * dYi_dxj - Y[j] * dXi_dxj
        out[i] = sp.simplify(val)
    return sp.Array(out)


def nested_bracket(
    X: sp.Array, Y: sp.Array, depth: int, coords: tuple
) -> sp.Array:
    """Compute the iterated bracket [...[[X, Y], Y], ..., Y] of depth `depth`.

    At depth 1: [X, Y].
    At depth 2: [[X, Y], Y].
    At depth k: [...[[X, Y], Y], Y, ..., Y] with Y repeated k times.

    This models the Hörmander bracket-filling process: starting from X, bracket
    repeatedly with Y to fill the missing tangent directions (math.md §28.2).

    Args:
        X:      D-element sp.Array (starting vector field).
        Y:      D-element sp.Array (vector field bracketed repeatedly).
        depth:  Number of bracket operations (≥ 1).
        coords: D-tuple of sp.Symbol.

    Returns:
        sp.Array of length D.

    Raises:
        ValueError: if depth < 1.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    result = X
    for _ in range(depth):
        result = lie_bracket(result, Y, coords)
    return result


def generates_T(
    brackets: list,
    coords: tuple,
    test_point: Optional[dict] = None,
) -> bool:
    """Check if a list of vector fields spans the tangent space (Hörmander cond.).

    Forms the D×len(brackets) matrix by evaluating each bracket at `test_point`,
    and checks if the matrix has full rank D (i.e., spans T_x M).

    This is the symbolic counterpart of the step-2 bracket-generating checker
    in `HypoellipticChernoff::new` (ADR-0077 §"Decision", Wave B step-checker).

    Args:
        brackets:   list of sp.Array, each of length D (vector fields / brackets).
        coords:     D-tuple of sp.Symbol (coordinate variables).
        test_point: dict mapping each coord symbol to a numeric value.
                    Defaults to the origin {c: 0 for c in coords}.

    Returns:
        True if the evaluated matrix has rank D (Hörmander condition satisfied).
        False otherwise (fields do not span; higher-depth brackets needed).

    Example (Kolmogorov — step-2 Carnot):
        x, v = sp.symbols('x v')
        X0 = sp.Array([v, 0])
        X1 = sp.Array([0, 1])
        b01 = lie_bracket(X1, X0, (x, v))   # → [1, 0] = +∂_x
        # X1=[0,1] and b01=[1,0] span ℝ² → step-2 Carnot verified
        ok = generates_T([X1, b01], (x, v), {x: 0, v: 0})
        assert ok  # step-2 Carnot verified
    """
    D = len(coords)
    if test_point is None:
        test_point = {c: sp.Integer(0) for c in coords}
    rows = []
    for b in brackets:
        row = [b[i].subs(test_point) for i in range(D)]
        rows.append(row)
    # Build matrix with each bracket as a row; transpose so columns are fields.
    M = sp.Matrix(rows).T  # shape D × len(brackets)
    return M.rank() == D
