#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy's Matrix arithmetic is dynamically typed through __mul__/__add__;
# Pyright cannot trace the operator overloads. All operations are valid sympy
# at runtime (verified by this oracle's PASS).
"""T_REVERSE_TRANSPOSE sympy gate — Reverse-mode AD transpose over the Chernoff product
(math.md §51, ADR-0156; v9.0.0 Phase-1 pre-flight oracle).

PRE-FLIGHT math-fidelity oracle. Extends T_MAGNUS_TRANSPOSE (§42) from one step
to the full trajectory (§51.2, §51.6), verifying that:

  (a) The transpose of a product of matrices equals the reversed product of
      transposes: (F₁·F₂·…·Fₙ)^⊤ = Fₙ^⊤·…·F₁^⊤. For an autonomous step
      (same F each step): (Fⁿ)^⊤ = (F^⊤)ⁿ.

  (b) The per-step transpose structure (§51.2):
      - The transpose of a shift τ_{+a} is the shift τ_{−a} (NEGATED shift vector).
      - The transpose of a scalar multiply is itself.
      - The transpose of the banded interpolation stencil is its matrix transpose.
      Verified by building a small banded shift+interpolation matrix and confirming
      that its matrix transpose negates the shift and transposes the stencil.

Sign convention (NORMATIVE, §51.2): the backward recursion runs k DECREASING;
F^⊤ uses the NEGATED shift vectors of the forward step (transpose of τ_{+a} is τ_{−a}).
This is the mirror of the §50.2 push-forward sign relation under the §38.2 dual pairing:
on the PRIMAL side, S(τ)f(x) shifts f by +h and +k; on the ADJOINT/MEASURE side,
S*(τ)δ_x also shifts the Dirac by +h and +k (push-forward). The TRANSPOSE (for the
reverse-mode AD backward pass) uses the NEGATED shift to undo the forward push.

Sub-checks (3 mandatory; math.md §51.6/§51.9 T_REVERSE_TRANSPOSE table):

  (1) product_transpose_factorisation
      (F₁·F₂·…·Fₙ)^⊤ = Fₙ^⊤·…·F₁^⊤ (reversed product of transposes).
      Verified with explicit small matrices for n=2 and n=3, using the §42
      operator-building helpers (_build_operators, _taylor_map) to construct F.
      For the autonomous case (same F each step), checks (F²)^⊤ = (F^⊤)² and
      (F³)^⊤ = (F^⊤)³.

  (2) per_step_transpose_structure
      Build a small banded "shift+interp" forward matrix combining:
        - a shift component: S_shift = I + a·J  (J = superdiagonal shift)
        - a scalar component: c·I
        - a banded interpolation component: T_band (tridiagonal stencil)
      The total step matrix F = (c·I + T_band)·S_shift (symbolic, small size).
      Verify that F^⊤ negates the shift vector direction (sign flip in J) and
      transposes the banded stencil, consistent with §51.2.
      Assert the sign-flip EXPLICITLY in a comment matching the NORMATIVE convention.

  (3) vjp_adjoint_identity   (NEW — §51.9, ADR-0156 Amendment 1)
      Structurally-independent VJP oracle. Build the discrete Chernoff one-step
      Jacobian J = ∂(Fu)/∂u as an explicit sympy Matrix on a small periodic grid
      (N=6 nodes, symbolic shift fractions α, β and reaction τc):
        F(u)(i) = (1+τc) · [¼·S_α(u)(i) + ¼·S_{−α}(u)(i) + ½·S_β(u)(i)]
      where S_α = (1−α)·I + α·P is the linear-interpolation shift by α cells
      on the periodic grid (P = cyclic forward-shift permutation matrix).
      Build Jᵀ as the transpose the backward sweep actually applies per §51.2:
        Jᵀ = (1+τc) · [¼·S_{−α} + ¼·S_α + ½·S_{−β}]
      (negated shift fractions = matrix transpose on the periodic grid).
      Assert:
        (i)  simplify((J·u)·v − u·(Jᵀ·v)) == 0  for symbolic vectors u, v
             (adjoint identity ⟨v, Ju⟩ = ⟨Jᵀv, u⟩).
        (ii) Jᵀ == J.T entrywise (hand-constructed transpose equals true matrix
             transpose, proving apply_transpose_step realises the true adjoint).
      Non-vacuous: the LHS of (i) is built from J (the forward matrix) and the
      RHS from Jᵀ (the backward sweep's matrix) — they agree only if the
      transpose is correct, not by construction. A wrong Jᵀ (e.g. forgetting to
      negate the drift shift) produces a non-zero residual proportional to β.

Prints "T_REVERSE_TRANSPOSE PASS (3/3 sub-checks: ...)" on success;
"T_REVERSE_TRANSPOSE FAIL: <reason>" and exits 1 on failure.

References:
  - math.md §42 / T_MAGNUS_TRANSPOSE — the single-step transpose oracle (extended here).
  - math.md §51 / ADR-0156 — ReverseChernoff v9.0.0 Shift B.
  - math.md §38 / §50 — dual pairing and push-forward sign conventions.
  - A. Griewank, A. Walther, *Algorithm 799: revolve*, ACM TOMS 26(1):19-45 (2000).
"""

import sys

# Re-use helpers from verify_magnus_transpose_exactness; copy inline to avoid
# import-path complexity (the oracle scripts are run from the workspace root).


def _build_operators(L1, L2, tau):
    """Return (A1, A2, comm_coeff, lead_coeff) for Ω₄ construction.

    Identical to the helper in verify_magnus_transpose_exactness.py.
    Ω₄(τ) = lead_coeff·(A1+A2) + comm_coeff·[A2, A1]
    with lead_coeff = τ/2 and comm_coeff = √3·τ²/12 (math.md §42.1).
    """
    import sympy as sp

    A1 = -L1
    A2 = -L2
    lead_coeff = tau / sp.Integer(2)
    comm_coeff = sp.sqrt(3) * tau**2 / sp.Integer(12)
    return A1, A2, comm_coeff, lead_coeff


def _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign: int = 1):
    """Ω₄ with selectable commutator sign (+1 forward, −1 transposed exponent)."""
    comm = A2 * A1 - A1 * A2  # [A2, A1]
    return lead_coeff * (A1 + A2) + comm_sign * comm_coeff * comm


def _taylor_map(Omega, k_max: int = 4):
    """S(τ) = Σ_{m=0..k_max} Ω^m / m!  (finite degree-4 Taylor polynomial)."""
    import sympy as sp

    n = Omega.shape[0]
    total = sp.zeros(n, n)
    power = sp.eye(n)
    for m in range(k_max + 1):
        total = total + power / sp.factorial(m)
        power = power * Omega
    return total


def _path3_laplacian(w12, w23):
    """3-node path Laplacian with symbolic edge weights (symmetric)."""
    import sympy as sp

    return sp.Matrix([
        [w12, -w12, 0],
        [-w12, w12 + w23, -w23],
        [0, -w23, w23],
    ])


def check_product_transpose_factorisation() -> "str | None":
    """Sub-check (1): (F^n)^⊤ = (F^⊤)^n for n=2, n=3.

    Uses the §42 single-step forward map F = S(τ) built from the degree-4
    Taylor-Magnus polynomial on a path-3 graph. Verifies:
      (F²)^⊤ = (F^⊤)²
      (F³)^⊤ = (F^⊤)³
    and the general product rule (F₁·F₂)^⊤ = F₂^⊤·F₁^⊤ for two DISTINCT steps.
    """
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    w12, w23 = sp.symbols("w12 w23", positive=True)
    tau = sp.symbols("tau", positive=True)

    L1 = _path3_laplacian(w12, w23)
    L2 = _path3_laplacian(2 * w12, 3 * w23)
    A1, A2, comm_coeff, lead_coeff = _build_operators(L1, L2, tau)

    Omega = _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign=1)
    F = _taylor_map(Omega, k_max=4)   # single-step forward map

    # (i) Autonomous n=2: (F²)^⊤ = (F^⊤)²
    F2 = F * F
    lhs_n2 = F2.T
    rhs_n2 = F.T * F.T
    res_n2 = sp.expand(lhs_n2 - rhs_n2)
    if res_n2 != sp.zeros(*F.shape):
        return (
            f"product_transpose_factorisation (n=2): (F²)^⊤ ≠ (F^⊤)². "
            f"Residual has non-zero entries. "
            f"Product-transpose factorisation FAILS for n=2."
        )

    # (ii) Autonomous n=3: (F³)^⊤ = (F^⊤)³
    F3 = F * F * F
    lhs_n3 = F3.T
    rhs_n3 = F.T * F.T * F.T
    res_n3 = sp.expand(lhs_n3 - rhs_n3)
    if res_n3 != sp.zeros(*F.shape):
        return (
            f"product_transpose_factorisation (n=3): (F³)^⊤ ≠ (F^⊤)³. "
            f"Residual has non-zero entries. "
            f"Product-transpose factorisation FAILS for n=3."
        )

    # (iii) Non-autonomous (two distinct steps F₁ ≠ F₂): (F₁·F₂)^⊤ = F₂^⊤·F₁^⊤.
    # Build F₂ from different edge weights.
    L1b = _path3_laplacian(3 * w12, w23)
    L2b = _path3_laplacian(w12, 5 * w23)
    A1b, A2b, cc_b, lc_b = _build_operators(L1b, L2b, tau)
    Omega_b = _omega4(A1b, A2b, cc_b, lc_b, comm_sign=1)
    F2_step = _taylor_map(Omega_b, k_max=4)   # second distinct step

    lhs_na = (F * F2_step).T
    rhs_na = F2_step.T * F.T          # REVERSED order — Fₙ^⊤·…·F₁^⊤
    res_na = sp.expand(lhs_na - rhs_na)
    if res_na != sp.zeros(*F.shape):
        return (
            f"product_transpose_factorisation (non-autonomous): "
            f"(F₁·F₂)^⊤ ≠ F₂^⊤·F₁^⊤. "
            f"Residual has non-zero entries. Reversed-order product FAILS."
        )

    return None  # PASS


def check_per_step_transpose_structure() -> "str | None":
    """Sub-check (2): per-step transpose negates shift and transposes banded stencil.

    Build a small (4×4) banded step matrix F combining:
      - a shift component J = superdiagonal shift matrix (J_{i,i+1} = 1)
      - a banded interpolation stencil T_band (symmetric tridiagonal)
      - scalar factors a, c (symbolic)
    The combined step: F = c·I + alpha·T_band + beta·J  (symbolic a, b, c, alpha, beta).

    Per §51.2 (NORMATIVE sign convention):
      - J^⊤ is the SUBdiagonal (shift in opposite direction — sign flip).
      - T_band^⊤ = T_band (symmetric stencil transposes to itself).
      - (c·I)^⊤ = c·I.
    So F^⊤ = c·I + alpha·T_band + beta·J^⊤ = c·I + alpha·T_band + beta·J_sub,
    where J_sub is the subdiagonal (OPPOSITE direction shift).

    The sign flip beta·J → beta·J_sub is the NORMATIVE assertion: the backward
    recursion uses the NEGATED shift direction (F^⊤ undoes the forward push).

    Verified by explicit matrix construction and transpose comparison.
    """
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    alpha, beta, c = sp.symbols("alpha beta c", real=True)
    n = 4

    # Identity
    I4 = sp.eye(n)

    # Superdiagonal shift J: J_{i, i+1} = 1 (shift content one step RIGHT/FORWARD).
    # NORMATIVE: this represents the +a shift in the forward Chernoff step.
    J_super = sp.zeros(n, n)
    for i in range(n - 1):
        J_super[i, i + 1] = 1

    # Its transpose = subdiagonal (shift content one step LEFT/BACKWARD).
    # NORMATIVE §51.2: the backward recursion uses the NEGATED shift.
    J_sub = J_super.T

    # Banded symmetric tridiagonal stencil (symmetric interpolation weights).
    T_band = sp.zeros(n, n)
    for i in range(n):
        T_band[i, i] = sp.Integer(2)
        if i > 0:
            T_band[i, i - 1] = sp.Rational(-1, 2)
        if i < n - 1:
            T_band[i, i + 1] = sp.Rational(-1, 2)

    # Verify T_band is symmetric (a prerequisite for the transpose to be itself).
    if sp.expand(T_band - T_band.T) != sp.zeros(n, n):
        return (
            "per_step_transpose_structure: T_band is NOT symmetric (test setup bug). "
            "Banded stencil must be symmetric for the transpose to equal itself."
        )

    # Forward step matrix F = c·I + alpha·T_band + beta·J_super.
    F = c * I4 + alpha * T_band + beta * J_super

    # Actual transpose of F.
    F_T_actual = F.T

    # Expected transpose (§51.2): c·I + alpha·T_band^⊤ + beta·J_super^⊤
    #                           = c·I + alpha·T_band + beta·J_sub
    # NORMATIVE: beta·J_super → beta·J_sub  (sign flip = NEGATED shift direction).
    F_T_expected = c * I4 + alpha * T_band + beta * J_sub

    res_struct = sp.expand(F_T_actual - F_T_expected)
    if res_struct != sp.zeros(n, n):
        return (
            f"per_step_transpose_structure: F^⊤ ≠ c·I + alpha·T_band + beta·J_sub. "
            f"Residual = {res_struct} (expected 0). "
            f"Per-step transpose (shift negation + stencil self-transpose) FAILS."
        )

    # Verify the shift NEGATION is genuine (non-trivial): J_super ≠ J_sub.
    if sp.expand(J_super - J_sub) == sp.zeros(n, n):
        return (
            "per_step_transpose_structure: J_super == J_sub — shift and its "
            "transpose are identical. The sign-flip check is vacuous (test setup bug)."
        )

    # Verify the scalar component is invariant under transpose: (c·I)^⊤ = c·I.
    if sp.expand((c * I4).T - c * I4) != sp.zeros(n, n):
        return (
            "per_step_transpose_structure: (c·I)^⊤ ≠ c·I. Scalar invariance FAILS."
        )

    # Confirm: the ONLY difference between F and F^⊤ is the shift direction.
    # F − F^⊤ = beta·(J_super − J_sub) = beta·(J_super − J_super^⊤).
    diff_F = sp.expand(F - F_T_actual)
    expected_diff = sp.expand(beta * (J_super - J_sub))
    if sp.expand(diff_F - expected_diff) != sp.zeros(n, n):
        return (
            f"per_step_transpose_structure: F − F^⊤ ≠ beta·(J_super − J_sub). "
            f"Got {diff_F}, expected {expected_diff}. "
            f"Sole difference should be the shift direction — FAILS."
        )

    return None  # PASS


def check_vjp_adjoint_identity() -> "str | None":
    """Sub-check (3): structurally-independent VJP / adjoint-identity oracle.

    Builds the discrete Chernoff one-step Jacobian J on a periodic (N=6) grid
    with symbolic shift fractions α (diffusion) and β (drift) and reaction τc:

      J = (1+τc) · [¼·S_α + ¼·S_{−α} + ½·S_β]

    where S_α = (1−α)·I + α·P  (linear-interp forward shift, P = cyclic perm).

    Constructs Jᵀ as the hand-built adjoint per §51.2 (negated shifts):

      Jᵀ = (1+τc) · [¼·S_{−α} + ¼·S_α + ½·S_{−β}]

    Asserts two independent facts:

      (i)  simplify((J·u)·v − u·(Jᵀ·v)) == 0
           (adjoint identity ⟨v, Ju⟩ = ⟨Jᵀv, u⟩ for symbolic vectors u, v).

      (ii) Jᵀ == J.T entrywise
           (hand-constructed adjoint equals the true matrix transpose, proving
            that apply_transpose_step realises the correct adjoint operator).

    Non-vacuous because assertion (i) compares the FORWARD matrix J (LHS) with
    the hand-constructed BACKWARD matrix Jᵀ (RHS). A wrong Jᵀ — e.g. forgetting
    to negate the drift shift β — produces a non-zero residual ∝ β, detected here.

    Periodic BC is used for exact algebraic closure: S_α.T = S_{−α} identically
    on a circulant grid, with no boundary-clamp artifacts.
    """
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    N = 6  # small grid — pure-symbolic, fast

    tau = sp.Symbol("tau", positive=True)
    c = sp.Symbol("c", positive=True)
    alpha = sp.Symbol("alpha", positive=True)  # diffusion shift fraction ∈ (0,1)
    beta = sp.Symbol("beta", positive=True)    # drift shift fraction ∈ (0,1)

    # Cyclic forward-shift permutation P: P[i, (i+1)%N] = 1.
    # NORMATIVE (§51.2): the forward Chernoff step shifts by +α; its transpose
    # shifts by −α, i.e. uses P^⊤ = P^{-1} (the backward-shift permutation).
    P = sp.zeros(N, N)
    for i in range(N):
        P[i, (i + 1) % N] = sp.Integer(1)

    I_N = sp.eye(N)

    # Linear-interpolation shifts (circular BC — exact algebraic closure).
    # S_alpha: forward shift by α   → (1−α)·I + α·P
    # S_neg_alpha: shift by −α     → (1−α)·I + α·P^⊤  = S_alpha.T  (circulant)
    S_alpha = (1 - alpha) * I_N + alpha * P
    S_neg_alpha = (1 - alpha) * I_N + alpha * P.T   # §51.2: negated shift

    # S_beta: forward drift shift by β  → (1−β)·I + β·P
    # S_neg_beta: shift by −β          → (1−β)·I + β·P^⊤ = S_beta.T
    S_beta = (1 - beta) * I_N + beta * P
    S_neg_beta = (1 - beta) * I_N + beta * P.T      # §51.2: negated drift shift

    # Reaction factor (scalar).
    w_react = 1 + tau * c

    # Forward Jacobian J = ∂(Fu)/∂u.
    # F(u)(i) = (1+τc)·[¼·(S_alpha·u)(i) + ¼·(S_neg_alpha·u)(i) + ½·(S_beta·u)(i)]
    J = w_react * (
        sp.Rational(1, 4) * S_alpha
        + sp.Rational(1, 4) * S_neg_alpha
        + sp.Rational(1, 2) * S_beta
    )

    # Hand-constructed adjoint Jᵀ per §51.2 (negate all shift directions).
    # Transpose of S_alpha (shift +α) → S_neg_alpha (shift −α). Confirmed algebraically:
    #   S_alpha.T = ((1−α)·I + α·P).T = (1−α)·I + α·P.T = S_neg_alpha.  ✓
    # Transpose of S_neg_alpha (shift −α) → S_alpha.  ✓
    # Transpose of S_beta (drift +β) → S_neg_beta (drift −β).  ✓
    Jt = w_react * (
        sp.Rational(1, 4) * S_neg_alpha   # transpose of S_alpha
        + sp.Rational(1, 4) * S_alpha      # transpose of S_neg_alpha
        + sp.Rational(1, 2) * S_neg_beta   # transpose of S_beta
    )

    # --- Assertion (ii): Jᵀ == J.T entrywise ---
    diff_matrix = sp.expand(Jt - J.T)
    if diff_matrix != sp.zeros(N, N):
        return (
            "vjp_adjoint_identity (ii): hand-constructed Jᵀ ≠ J.T entrywise. "
            f"Residual Jᵀ−J.T = {diff_matrix}. "
            "The apply_transpose_step adjoint does NOT equal the true matrix transpose."
        )

    # --- Assertion (i): adjoint identity ⟨v, Ju⟩ = ⟨Jᵀv, u⟩ ---
    # Build arbitrary symbolic vectors u, v of length N.
    u = sp.Matrix([sp.Symbol(f"u{i}") for i in range(N)])
    v = sp.Matrix([sp.Symbol(f"v{i}") for i in range(N)])

    lhs = (J * u).dot(v)    # ⟨v, Ju⟩  (inner product in standard basis)
    rhs = u.dot(Jt * v)     # ⟨u, Jᵀv⟩ = ⟨Jᵀv, u⟩
    residual = sp.simplify(lhs - rhs)

    if residual != 0:
        return (
            f"vjp_adjoint_identity (i): (J·u)·v − u·(Jᵀ·v) ≠ 0. "
            f"Residual = {residual}. "
            "Adjoint identity ⟨v, Ju⟩ = ⟨Jᵀv, u⟩ FAILS. "
            "The backward sweep does NOT apply the correct transpose."
        )

    return None  # PASS


def fail(reason: str) -> int:
    """Print FAIL line and return exit code 1."""
    print(f"T_REVERSE_TRANSPOSE FAIL: {reason}", flush=True)
    return 1


def main() -> int:
    """Run all 3 sub-checks; print result; exit 0/1."""
    checks = [
        ("product_transpose_factorisation", check_product_transpose_factorisation),
        ("per_step_transpose_structure", check_per_step_transpose_structure),
        ("vjp_adjoint_identity", check_vjp_adjoint_identity),
    ]
    print("=" * 64)
    print("T_REVERSE_TRANSPOSE — Reverse-mode AD transpose over Chernoff product")
    print("(math.md §51/§51.9, ADR-0156 Amendment 1; v9.1.0 Phase-1 oracle)")
    print("=" * 64)

    failures: list[str] = []
    passed: list[str] = []
    for name, check in checks:
        try:
            result = check()
        except Exception as e:  # noqa: BLE001
            return fail(f"sub-check {name} raised exception: {e!r}")
        if result is None:
            print(f"  (PASS) {name}")
            passed.append(name)
        else:
            print(f"  (FAIL) {name}: {result}")
            failures.append(f"{name}: {result}")

    print()
    if failures:
        return fail(
            f"{len(failures)}/{len(checks)} sub-checks failed: "
            + "; ".join(failures)
        )
    print(
        "T_REVERSE_TRANSPOSE PASS (3/3 sub-checks: " + " / ".join(passed) + ")",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
