#!/usr/bin/env python3
"""PRE-FLIGHT oracle for complex matrix-valued operators (v7.0.0 item #17, ADR-0128).

Combines ADR-0079 (SemiflowComplex) + ADR-0082/0125 (matrix-valued Padé[13/13]).
The shipped `matrix_pade.rs` is generic over `F: SemiflowFloat` (REAL only); the
§33.5 deferred extension replaces `F` with `C: SemiflowComplex` so the per-grid-point
M×M coupling-block exponential `exp(τ C(x)/2)` may be COMPLEX (non-Hermitian absorbing
potentials, complex cross-diffusion coupling, rough-Heston Markov blocks).

This PRE-FLIGHT establishes the two facts the Rust impl rests on:

  (1) Higham 2005 Padé[13/13] with REAL coefficients PADE_B applied to a COMPLEX matrix
      argument reproduces exp(Z) to relative Frobenius error <= 1e-12 in the regime
      ||Z||_inf <= theta_13 = 5.3719 (no squarings) and with scaling-and-squaring for
      larger norms. The coefficients stay real — only the matrix arithmetic becomes
      complex. The engineer change is `f64 -> Complex<f64>` throughout, NO new approximant.

  (2) The inf-norm squaring count s = ceil(log2(||Z||_inf / theta_13)) is the SAME
      formula for complex matrices (||.||_inf via complex modulus row-sums), so
      `compute_squarings` lifts unchanged.

  (3) exp(iH) for Hermitian H is unitary (the physical invariant that complex coupling
      must preserve when the imaginary/absorbing part vanishes).

Gate G_CPLX_MATRIX: relative Frobenius error <= 1e-12 (mirror of G_MATRIX_PADE_M5 in
the complex regime). NO slope gate needed — the exponential is exact-to-1e-12; the
order-2 Strang composition is inherited verbatim from §33.7 AMENDMENT 2 (real).

Implementation note: the high-precision REFERENCE exp(Z) uses mpmath at 50 decimal
digits (symbolic sympy .exp() of a dense 6x6 complex matrix is intractable). The Padé
under test is evaluated in the SAME mpmath precision so the comparison is meaningful.

Run: python3 scripts/verify_complex_matrix.py     Exit 0 = PASS (GO), 1 = FAIL.
"""
import sys

try:
    import mpmath as mp
except ImportError:
    print("verify_complex_matrix SKIP (mpmath not available)")
    sys.exit(0)

mp.mp.dps = 50  # 50 decimal digits working precision

# Higham 2005: Padé[13/13] coefficients b_0..b_13 (REAL — applied to complex matrices).
PADE_B = [
    mp.mpf("64764752532480000"), mp.mpf("32382376266240000"),
    mp.mpf("7771770303897600"), mp.mpf("1187353796428800"),
    mp.mpf("129060195264000"), mp.mpf("10559470521600"),
    mp.mpf("670442572800"), mp.mpf("33522128640"),
    mp.mpf("1323241920"), mp.mpf("40840800"),
    mp.mpf("960960"), mp.mpf("16380"), mp.mpf("182"), mp.mpf("1"),
]
THETA13 = mp.mpf("5.371920351148152")


def inf_norm(A):
    M = A.rows
    return max(sum(abs(A[i, j]) for j in range(M)) for i in range(M))


def squaring_count(A):
    norm = inf_norm(A)
    if norm <= THETA13:
        return 0
    return max(0, int(mp.ceil(mp.log(norm / THETA13) / mp.log(2))))


def pade13_exp(Z):
    """Higham 2005 Padé[13/13] + scaling-and-squaring on an mpmath complex matrix."""
    M = Z.rows
    s = squaring_count(Z)
    A = Z / (mp.mpf(2) ** s)
    I = mp.eye(M)
    A2 = A * A
    A4 = A2 * A2
    A6 = A2 * A4
    # even/odd Higham split: U = A*(b13 A6+b11 A4+b9 A2)A6 + ...; V similarly.
    U_inner = A6 * (PADE_B[13] * A6 + PADE_B[11] * A4 + PADE_B[9] * A2) \
        + (PADE_B[7] * A6 + PADE_B[5] * A4 + PADE_B[3] * A2 + PADE_B[1] * I)
    U = A * U_inner
    V = A6 * (PADE_B[12] * A6 + PADE_B[10] * A4 + PADE_B[8] * A2) \
        + (PADE_B[6] * A6 + PADE_B[4] * A4 + PADE_B[2] * A2 + PADE_B[0] * I)
    R = mp.lu_solve_mat(V - U, V + U) if hasattr(mp, "lu_solve_mat") else (V - U) ** -1 * (V + U)
    for _ in range(s):
        R = R * R
    return R


def fro_rel_err(approx, exact):
    M = exact.rows
    num = mp.sqrt(sum(abs(approx[i, j] - exact[i, j]) ** 2
                      for i in range(M) for j in range(M)))
    den = mp.sqrt(sum(abs(exact[i, j]) ** 2 for i in range(M) for j in range(M)))
    return num / den


def cmat(M, fn):
    A = mp.zeros(M)
    for i in range(M):
        for j in range(M):
            A[i, j] = fn(i, j)
    return A


def check_complex_pade(_):
    """Sub-check 1: Padé[13/13] (real coeffs) on COMPLEX matrices vs mpmath expm."""
    j = mp.mpc(0, 1)
    cases = []
    # M=5 complex non-Hermitian, small norm (no squarings)
    cases.append(cmat(5, lambda i, k:
        mp.mpf(1) / 3 * (1 if i == k else 0)
        + mp.mpf(1) / 7 * j * ((i + 2 * k) % 3 - 1)
        + mp.mpf(1) / 5 * ((3 * i + k) % 3 - 1)))
    # M=6 complex, larger norm (forces scaling-and-squaring)
    cases.append(cmat(6, lambda i, k:
        mp.mpf(3) / 2 * (2 if i == k else 0)
        + j * mp.mpf(2) / 3 * ((i * k) % 4 - 2)
        + mp.mpf(1) / 2 * ((i + k) % 3 - 1)))
    # M=8 symmetric-real-part + complex off-diagonal
    cases.append(cmat(8, lambda i, k:
        mp.mpf(1) / 4 * ((i + k) % 5 - 2)
        + j * mp.mpf(1) / 6 * (1 if abs(i - k) == 1 else 0)))
    worst = mp.mpf(0)
    for idx, Z in enumerate(cases):
        exact = mp.expm(Z)
        approx = pade13_exp(Z)
        err = fro_rel_err(approx, exact)
        worst = max(worst, err)
        if err > mp.mpf("1e-12"):
            return False, f"case {idx} (M={Z.rows}) rel-Frobenius err {mp.nstr(err,4)} > 1e-12"
    return True, f"3 complex cases (M=5,6,8), worst rel-Frobenius err {mp.nstr(worst,4)} <= 1e-12"


def check_unitarity_antihermitian(_):
    """Sub-check 2: exp(iH) for Hermitian H is unitary (U^H U = I to 1e-12)."""
    j = mp.mpc(0, 1)
    H = mp.matrix([[3, 1 - 2 * j, 0],
                   [1 + 2 * j, -1, mp.mpf(1) / 2],
                   [0, mp.mpf(1) / 2, 2]])  # Hermitian
    U = pade13_exp(j * H)
    Uh = U.transpose_conj()
    UhU = Uh * U
    err = max(abs(UhU[i, k] - (1 if i == k else 0)) for i in range(3) for k in range(3))
    if err > mp.mpf("1e-12"):
        return False, f"||U^H U - I||_max = {mp.nstr(err,4)} > 1e-12"
    return True, f"anti-Hermitian exp unitary: ||U^H U - I|| = {mp.nstr(err,4)}"


def check_squaring_formula(_):
    """Sub-check 3: complex inf-norm squaring count == real formula."""
    j = mp.mpc(0, 1)
    Z = mp.matrix([[10, 3 * j], [3, -8]])
    s = squaring_count(Z)
    norm = inf_norm(Z)  # = max(10+3, 3+8) = 13; log2(13/5.3719) ~ 1.28 => s=2
    if s != 2:
        return False, f"squaring count s={s}, expected 2 (norm={mp.nstr(norm,4)})"
    return True, f"complex inf-norm squaring: norm={mp.nstr(norm,4)} => s={s} (matches real formula)"


def main():
    checks = [
        ("complex_pade13_exp", check_complex_pade),
        ("antihermitian_unitarity", check_unitarity_antihermitian),
        ("complex_squaring_formula", check_squaring_formula),
    ]
    names = []
    for name, fn in checks:
        ok, msg = fn(None)
        if not ok:
            print(f"G_CPLX_MATRIX FAIL [{name}]: {msg}")
            return 1
        names.append(name)
        print(f"  [{name}] {msg}")
    print(f"G_CPLX_MATRIX PASS ({len(names)}/{len(names)} sub-checks: {' / '.join(names)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
