# pyright: reportCallIssue=false, reportArgumentType=false
# pyright: reportOperatorIssue=false, reportAttributeAccessIssue=false
#
# Sympy's Matrix arithmetic and slicing (`A[:k, :k]`, `M * N`, `.det()`) are
# dynamically typed through __mul__ / __getitem__; Pyright resolves slices to
# an over-broad `MatrixElement | list` union and cannot trace the element
# types. Every operation in this module is valid sympy at runtime (verified by
# the T_EIGENBASIS oracle assertions at the bottom of the file).
"""T_EIGENBASIS — PRE-FLIGHT oracle for F5 eigenbasis-rotation separability.

ADR-0137 (v8.0.0 MINOR), math.md §48. Symbolically certifies that the F5
construction is EXACT for a constant symmetric-positive-definite (SPD)
off-diagonal diffusion tensor `A`:

  1. diagonalization  — for an explicit off-diagonal SPD `A`, there exists an
     orthogonal `Q` (QᵀQ = I) with `QᵀAQ = Λ` diagonal, `Λ_ii = λ_i > 0`;
  2. separability     — under the constant linear change of variables `y = Qᵀx`,
     the second-order operator `L = ∇·(A∇) = Σ_ij A_ij ∂²_{x_i x_j}` becomes the
     axis-aligned sum `L̃ = Σ_i λ_i ∂²_{y_i y_i}` (no cross terms);
  3. order-preservation / rotation-exactness — because `Q` is a CONSTANT
     orthogonal map it commutes with the time-stepping, so the per-axis
     diagonal kernel in the eigenbasis reproduces `e^{τL}` to the order of the
     inner kernel with ZERO additional rotation error. We certify this on the
     heat semigroup symbol: `Q · exp(τ Λ_∂) · Qᵀ = exp(τ A_∂)` where `A_∂`,
     `Λ_∂` are the (formal, frozen-Fourier) symbols `−ξᵀAξ`, `−ηᵀΛη` with
     `η = Qᵀξ`. The two semigroup symbols are algebraically EQUAL.

Representative dimensions: D = 3 (closed-form Jacobi-style rotation) and D = 5
(numeric eigensolve, symbolic re-verification of the three identities). D = 5
is the ADR-0137 headline regime (q^D = 5^5 = 3125 tensor nodes collapse to
D·q = 25 axis nodes).

Run:  python scripts/eigenbasis_rotation_kit.py
Prints exactly  `T_EIGENBASIS PASS`  on success (exit 0) or
                `T_EIGENBASIS FAIL: <reason>`  on failure (exit 1).
"""

from __future__ import annotations

import sys

import sympy as sp


# ---------------------------------------------------------------------------
# Construction of representative off-diagonal SPD tensors
# ---------------------------------------------------------------------------

def spd_offdiagonal_d3() -> sp.Matrix:
    """A 3x3 symmetric positive-definite tensor with nonzero off-diagonals.

    Built as `A = RᵀΛ₀R` for an explicit rational rotation so the eigenvalues
    are known exactly and the off-diagonals are guaranteed nonzero.
    """
    # Diagonal eigenvalue seed (distinct, positive) — strong anisotropy ratio 9.
    lam = sp.diag(1, 3, 9)
    # A rational orthogonal-ish mixing via a Householder-style reflector on a
    # rational unit vector; we orthonormalize symbolically to stay exact.
    v = sp.Matrix([2, 1, 2])  # |v|² = 9
    house = sp.eye(3) - sp.Rational(2, 9) * (v * v.T)  # exact orthogonal reflector
    a = house.T * lam * house
    return sp.simplify(a)


def spd_offdiagonal_d5() -> sp.Matrix:
    """A 5x5 SPD tensor with nonzero off-diagonals via a rational reflector."""
    lam = sp.diag(1, 2, 4, 8, 16)  # anisotropy ratio 16
    v = sp.Matrix([2, 2, 1, 2, 4])  # |v|² = 4+4+1+4+16 = 29
    house = sp.eye(5) - sp.Rational(2, 29) * (v * v.T)
    a = house.T * lam * house
    return sp.simplify(a)


# ---------------------------------------------------------------------------
# Core identities
# ---------------------------------------------------------------------------

def check_diagonalization(a: sp.Matrix, dim: int) -> tuple[sp.Matrix, sp.Matrix]:
    """Return (Q, Lambda) with QᵀAQ = Lambda diagonal, QᵀQ = I, λ_i > 0.

    Raises AssertionError if any property fails.
    """
    # sympy diagonalize gives A = P D P⁻¹; for symmetric A we orthonormalize P.
    p_mat, d_mat = a.diagonalize(normalize=True)
    q = sp.simplify(p_mat)
    lam = sp.simplify(d_mat)

    # (a) orthogonality: QᵀQ = I.
    ortho = sp.simplify(q.T * q - sp.eye(dim))
    assert ortho == sp.zeros(dim, dim), f"D={dim}: QᵀQ != I"

    # (b) diagonalization: QᵀAQ = Λ (diagonal).
    rotated = sp.simplify(q.T * a * q - lam)
    assert rotated == sp.zeros(dim, dim), f"D={dim}: QᵀAQ != Λ"

    # (c) Λ diagonal with strictly positive entries (SPD preserved).
    for i in range(dim):
        for j in range(dim):
            if i != j:
                assert sp.simplify(lam[i, j]) == 0, f"D={dim}: Λ not diagonal"
        assert sp.simplify(lam[i, i]) > 0, f"D={dim}: λ_{i} not positive"
    return q, lam


def check_separability(a: sp.Matrix, q: sp.Matrix, lam: sp.Matrix, dim: int) -> None:
    """Certify L = ∇·(A∇) → Σ λ_i ∂²_{y_i} under y = Qᵀx (constant A).

    Uses the (frozen-coefficient) Fourier symbol: a Fourier mode e^{i ξ·x}
    sends ∂_{x_i x_j} → −ξ_i ξ_j, so the symbol of L is −ξᵀAξ. Under the
    rotation η = Qᵀξ the symbol of L̃ is −ηᵀΛη. We verify −ξᵀAξ = −ηᵀΛη
    AND that −ηᵀΛη contains NO cross term η_i η_j (i≠j) — i.e. it is the
    separable axis-aligned sum −Σ λ_i η_i².
    """
    xi = sp.Matrix(sp.symbols(f"xi0:{dim}", real=True))
    eta = sp.Matrix(sp.symbols(f"eta0:{dim}", real=True))

    sym_l = sp.expand(-(xi.T * a * xi)[0, 0])          # symbol of L in ξ
    sym_l_rot = sp.expand(-(eta.T * lam * eta)[0, 0])    # symbol of L̃ in η

    # Separability: the rotated symbol has no η_i η_j cross terms.
    for i in range(dim):
        for j in range(dim):
            if i != j:
                coeff = sym_l_rot.coeff(eta[i] * eta[j])
                assert coeff == 0, f"D={dim}: rotated symbol has η_{i}η_{j} cross term"

    # Frame consistency: substituting η = Qᵀξ into the rotated symbol recovers L.
    subs = {eta[i]: (q.T * xi)[i, 0] for i in range(dim)}
    recovered = sp.expand(sym_l_rot.subs(subs))
    assert sp.simplify(recovered - sym_l) == 0, f"D={dim}: −ηᵀΛη|_{{η=Qᵀξ}} != −ξᵀAξ"


def check_order_preservation(a: sp.Matrix, q: sp.Matrix, lam: sp.Matrix, dim: int) -> None:
    """Certify rotation-exactness on the heat semigroup symbol (constant A).

    The frozen-coefficient heat propagator symbol is exp(τ · symbol(L)). We
    show Q · exp(τ Λ_sym) · Qᵀ = exp(τ A_sym) where A_sym = −ξᵀAξ acting per
    eigen-component. Concretely, since QᵀAQ = Λ, for ANY analytic f:
        f(A) = Q f(Λ) Qᵀ            (spectral mapping, EXACT).
    With f = exp(−τ ξ²·(·)) componentwise this is the statement that the
    per-axis diagonal kernel in y reproduces the full kernel in x with zero
    rotation error. We verify the matrix identity exp(τA) = Q exp(τΛ) Qᵀ
    symbolically (τ symbolic), which is the order-preservation certificate.
    """
    tau = sp.symbols("tau", real=True)
    # exp(τΛ) is trivially diagonal exp(τλ_i); rotate it back.
    exp_lam = sp.diag(*[sp.exp(tau * lam[i, i]) for i in range(dim)])
    lhs = sp.simplify(q * exp_lam * q.T)        # Q exp(τΛ) Qᵀ
    rhs = sp.simplify((tau * a).exp())          # exp(τA) directly
    diff = sp.simplify(lhs - rhs)
    assert diff == sp.zeros(dim, dim), f"D={dim}: exp(τA) != Q exp(τΛ) Qᵀ"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_dimension(dim: int, a: sp.Matrix) -> None:
    """Run all three identities for one representative dimension."""
    # SPD sanity: symmetric + positive-definite (all leading minors > 0).
    assert sp.simplify(a - a.T) == sp.zeros(dim, dim), f"D={dim}: A not symmetric"
    for k in range(1, dim + 1):
        minor = a[:k, :k].det()
        assert sp.simplify(minor) > 0, f"D={dim}: leading minor {k} not positive"
    # Off-diagonal non-triviality (the whole point of F5).
    off = any(sp.simplify(a[i, j]) != 0 for i in range(dim) for j in range(dim) if i != j)
    assert off, f"D={dim}: tensor has no off-diagonal coupling (F5 is vacuous)"

    q, lam = check_diagonalization(a, dim)
    check_separability(a, q, lam, dim)
    check_order_preservation(a, q, lam, dim)


def main() -> int:
    try:
        run_dimension(3, spd_offdiagonal_d3())
        run_dimension(5, spd_offdiagonal_d5())
    except AssertionError as exc:
        print(f"T_EIGENBASIS FAIL: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 — oracle must fail loud, not crash silently
        print(f"T_EIGENBASIS FAIL: unexpected {type(exc).__name__}: {exc}")
        return 1
    print("T_EIGENBASIS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
