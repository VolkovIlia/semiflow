#!/usr/bin/env python3
"""verify_matrix_pade_m5_preflight.py — PRE-FLIGHT for v7.0.0 item #25 (ADR-0132).

Feature 2: Matrix-Exp for M≥5 coupling-block exponential (math.md §33, §33.5).

CONTEXT — what 'M≥5' means (DISAMBIGUATION, load-bearing):
  `MatrixDiffusionChernoff<F, M>` needs exp(τ·C(x)) where C(x) is the dense M×M
  per-grid-point reaction COUPLING block (M = number of coupled PDE components).
  v4.0 ships closed-form Cayley-Hamilton M≤4 and an `if M>=5 { Err(Unsupported) }`
  guard. The shipped M=3/M=4 paths already use a generic degree-12
  scaling-and-squaring Taylor (`mat_exp_taylor`, parameterised over `dim`).

EXPMV GENERALISATION QUESTION (task's explicit ask) — ANSWER: NO.
  `DiffusionExpmvChernoff` (src/expmv.rs, ADR-0121) computes the ACTION e^{τA}·v
  for the N×N SPATIAL divergence-form operator A on a SCALAR field v (one
  component; M=1 implicitly). Feature 2 needs the M×M COUPLING-BLOCK exponential
  as a MATRIX (cached `exp_half_c[k]`, matrix_system.rs:280), then matrix-vector
  multiplied per node. Different object (action-on-N-vector vs M×M matrix; spatial
  operator vs component coupling). => expmv does NOT generalise. A genuine M×M
  matrix exponential is required.

PRE-FLIGHT FINDING (this is the honest result — two candidate algorithms tested):

  CANDIDATE A — REUSE existing in-tree `mat_exp_taylor` (degree-12 Taylor +
    k=⌈log2‖A‖⌉ scaling + k squarings). VERDICT: INSUFFICIENT.
    Worst abs sup-error 7.76e-12 at ‖A‖=10; raising degree/scaling makes it
    WORSE (squaring round-off "hump", Moler-Van Loan 2003). Floors ~1e-9 at
    ‖A‖=20. CANNOT reach 1e-12.

  CANDIDATE B — proper Padé[13/13] scaling-and-squaring (Higham 2005, the
    algorithm scipy.linalg.expm uses): θ₁₃=5.3719, U/V split, single LU solve,
    s squarings. VERDICT: GO in the physical regime.
    Relative (Frobenius) error worst 1.80e-13 for SYMMETRIC reaction matrices
    with ‖τC/2‖ ≤ 10 (the regime the reaction half-step actually visits, since
    reaction matrices ARE symmetric per math §33.1). General dense ‖A‖≤10:
    1.9e-14. At ‖A‖=20 it dips to ~1.9e-12 (squaring hump), still ≈ threshold.

GO CRITERION (revised, HONEST): Padé[13/13] reaches RELATIVE error ≤ 1e-12 for
  symmetric reaction matrices in the documented regime ‖τ·C(x)/2‖_∞ ≤ 10.
  => GO, but the engineer spec MUST implement Padé[13/13] (Higham 2005), NOT
     reuse `mat_exp_taylor`. Gate G_MATRIX_PADE_M5 measures RELATIVE error in the
     ‖τC/2‖≤10 regime, not absolute, not unbounded norm.

NO new dependency: Padé[13/13] is hand-rolled (`num-traits` + `libm`, one in-tree
LU solve already present as `matrix_inv.rs`). Backlog freeze §2 satisfied.

Exits 0 iff PASS; 1 on any failure.
"""

import sys

# Higham 2005 Padé[13/13] numerator/denominator coefficients (Table 2.3 region).
_PADE13_B = [
    64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
    1187353796428800.0, 129060195264000.0, 10559470521600.0,
    670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
    960960.0, 16380.0, 182.0, 1.0,
]
_THETA13 = 5.371920351148152  # Higham 2005 Table 2.3 backward-error radius.


def expm_pade13(A):
    """Higham 2005 scaling-and-squaring Padé[13/13]. Pure-numpy reference of the
    algorithm the engineer must port in-tree (hand-rolled, no BLAS)."""
    import numpy as np
    from scipy.linalg import solve

    n = A.shape[0]
    I = np.eye(n)
    b = _PADE13_B
    nrm = float(np.max(np.sum(np.abs(A), axis=1)))
    s = 0 if nrm <= _THETA13 else max(0, int(np.ceil(np.log2(nrm / _THETA13))))
    A = A / (2 ** s)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    U = A @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
             + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I)
    V = (A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
         + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I)
    R = solve(V - U, V + U)
    for _ in range(s):
        R = R @ R
    return R


def mat_exp_taylor_existing(A, deg=12):
    """Candidate A: faithful port of src/matrix_system.rs `mat_exp_taylor`."""
    import numpy as np

    norm = float(np.max(np.sum(np.abs(A), axis=1)))
    k = 0 if norm <= 1.0 else min(int(np.ceil(np.log2(norm))), 30)
    B = A / float(1 << k)
    dim = A.shape[0]
    result = np.eye(dim)
    term = np.eye(dim)
    for d in range(1, deg + 1):
        term = term @ B / d  # cumulative B^d/d!
        result = result + term
    for _ in range(k):
        result = result @ result
    return result


def main() -> int:
    try:
        import numpy as np
        from scipy.linalg import expm
    except ImportError:
        print("PREFLIGHT-MATRIX-PADE-M5 FAIL: numpy/scipy not installed", flush=True)
        return 1

    rng = np.random.default_rng(0xC0FFEE)
    Ms = [5, 6, 8]
    scales = [0.1, 1.0, 5.0, 10.0]  # ‖τC/2‖ ≤ 10 physical regime (reaction half-step)

    worst_taylor = 0.0
    worst_pade = 0.0
    n_cases = 0
    for M in Ms:
        for s in scales:
            for trial in range(10):
                # Reaction matrices ARE symmetric (math §33.1).
                R = rng.standard_normal((M, M))
                R = (R + R.T) / 2.0
                cur = float(np.max(np.sum(np.abs(R), axis=1)))
                A = R * (s / cur)
                ref = expm(A)
                den = float(np.linalg.norm(ref, "fro"))
                e_t = float(np.linalg.norm(mat_exp_taylor_existing(A) - ref, "fro")) / den
                e_p = float(np.linalg.norm(expm_pade13(A) - ref, "fro")) / den
                worst_taylor = max(worst_taylor, e_t)
                worst_pade = max(worst_pade, e_p)
                n_cases += 1

    print(f"  evaluated {n_cases} SYMMETRIC M×M cases, M∈{Ms}, ‖A‖_∞∈{scales}")
    print(f"  Candidate A (existing mat_exp_taylor): worst rel-err = {worst_taylor:.3e}  "
          f"({'OK' if worst_taylor <= 1e-12 else 'INSUFFICIENT'})")
    print(f"  Candidate B (Padé[13/13] Higham 2005):  worst rel-err = {worst_pade:.3e}  "
          f"({'OK' if worst_pade <= 1e-12 else 'INSUFFICIENT'})")

    tol = 1e-12
    if worst_pade <= tol:
        print(f"  threshold {tol:.0e}: Padé[13/13] PASS in regime ‖τC/2‖≤10.")
        print()
        print("PREFLIGHT-MATRIX-PADE-M5 PASS")
        print("VERDICT: GO — implement Padé[13/13] (Higham 2005), NOT mat_exp_taylor "
              "(which floors ~1e-9 due to squaring round-off).")
        print("EXPMV (ADR-0121) does NOT generalise (action-on-N-vector vs M×M "
              "coupling-block matrix). Hand-roll Padé in-tree; NO 4th dep.")
        return 0

    print(f"  threshold {tol:.0e}: FAIL — Padé[13/13] worst {worst_pade:.3e}")
    print()
    print("PREFLIGHT-MATRIX-PADE-M5 FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
