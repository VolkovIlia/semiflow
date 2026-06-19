#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy's Matrix arithmetic is dynamically typed through __mul__/__add__;
# Pyright cannot trace the operator overloads. All operations are valid sympy
# at runtime (verified by this oracle's PASS).
"""T_MAGNUS_TRANSPOSE sympy gate — Truncated-Magnus state-adjoint transpose-exactness
(math.md §42, Theorem 42.1; ADR-0115 Issue #2).

PRE-FLIGHT math-fidelity oracle. INDEPENDENTLY verifies, before any Rust kernel
is written, that the transpose of the *implemented finite map*

    S(τ) := Σ_{m=0}^{4} Ω₄(τ)^m / m!        (NORMATIVE: m runs 0..=4, finite sum)

with the Magnus exponent (Iserles+ 2000 eq. 5.10; math.md §42.1 / §12.9)

    Ω₄(τ) = (τ/2)(A₁ + A₂) + (√3·τ²/12)·[A₂, A₁],   A_i = −L_G(t₀ + c_i·τ),

factors term-by-term under transpose, and that for SYMMETRIC node operators
(`A_iᵀ = A_i`, true for the combinatorial Laplacian and for the VarCoef
`L_a = √a · L_G · √a`) the transposed exponent equals `Ω₄` with ONLY the
commutator coefficient sign-flipped:

    Ω₄(τ)ᵀ = (τ/2)(A₁ + A₂) − (√3·τ²/12)·[A₂, A₁]   =:  Ω₄⋆,
    S(τ)ᵀ  = Σ_{m=0}^{4} (Ω₄ᵀ)^m / m!              =:  S⋆(τ).

This is the transpose of the IMPLEMENTED finite Taylor map, NOT of exp(Ω₄)
(see §42.5 boundary note: exp(Ω₄)ᵀ ≠ exp(Ω₄) at O(τ²); the two objects differ
at O(τ⁵), the Taylor remainder, which is below the declared order).

Sub-checks (5 mandatory; math.md §42.6 table):

  (a) commutator_antisymmetry
      [A₂,A₁]ᵀ = −[A₂,A₁] for symbolic SYMMETRIC A_i.
      [A₂,A₁]ᵀ = (A₂A₁ − A₁A₂)ᵀ = A₁ᵀA₂ᵀ − A₂ᵀA₁ᵀ = A₁A₂ − A₂A₁ = −[A₂,A₁].

  (b) omega4_transpose_sign_flip
      Ω₄ᵀ equals Ω₄ with the commutator coefficient negated and ALL other terms
      fixed. Verified by `Ω₄ᵀ − Ω₄⋆ == 0` AND `Ω₄ᵀ ≠ Ω₄` (the flip is genuine,
      not a no-op) on a graph whose two node operators do not commute.

  (c) taylor_transpose_factorisation
      (Σ_{m=0..4} Ω₄^m/m!)ᵀ − Σ_{m=0..4} (Ω₄ᵀ)^m/m! == 0 (ZERO residual).
      The substantive content is that this equals S⋆ built from the sign-flipped
      Ω₄⋆ (combining (b)+(c)): S(τ)ᵀ == Σ_{m=0..4} (Ω₄⋆)^m/m!.

  (d) dual_pairing_identity
      ⟨S(τ)u, g⟩ = ⟨u, S⋆(τ)g⟩ to symbolic exactness on seeded vectors u,g
      (guard against a transcription error in the reconstructed Ω₄ / S⋆).

  (e) varcoef_la_symmetry
      (√a · L_G · √a)ᵀ = √a · L_G · √a for a positive diagonal √a, so (a)–(d)
      carry verbatim to the VarCoefMagnusGraphHeatChernoff kernel. We re-run the
      core antisymmetry + sign-flip + factorisation on the L_a-conjugated
      operators to confirm.

Order-4 consistency (§42.3): S⋆ is the EXACT transpose of the same degree-4
polynomial on the same GL nodes and symmetric L_G, so its local truncation order
equals the forward map's by construction (no order loss). This is structural,
not a separable symbolic identity beyond the degree-matching of the Taylor sum,
which sub-check (c) establishes term-by-term (m = 0..4, no remainder). The forward
map's own order-4 accuracy is the subject of the SEPARATE T17N / verify_magnus_*
oracles and is NOT re-derived here (in-scope statement per §42.6).

Prints "T_MAGNUS_TRANSPOSE PASS (5/5 sub-checks: ...)" on success;
"T_MAGNUS_TRANSPOSE FAIL: <reason>" and exits 1 on failure.

References:
  - Iserles, Munthe-Kaas, Nørsett, Zanna, *Lie-group methods*, Acta Numerica 9
    (2000) 215–365, eq. (5.10) — Ω₄ structure.
  - Hochbruck, Ostermann, *Exponential integrators*, Acta Numerica 19 (2010)
    209–286, §3 — bounded-operator exp(Ω)·v Taylor truncation.
  - math.md §42 (Theorem 42.1, §42.6 oracle reference), §12.9, §15 / ADR-0114
    (genuine-exp honesty boundary, §42.5).
  - ADR-0115 — contract authority for §42.
"""

import sys

# GL₄ commutator coefficient prefactor (√3 / 12); the full coefficient is this
# times τ². Its sign is the ONLY thing that flips under transpose (Theorem 42.1).
# Kept as a module-level symbolic constant so every sub-check reuses one source.


def _build_operators(L1, L2, tau):
    """Return (A1, A2, comm_coeff, lead_coeff) for node operators A_i = −L_i.

    Ω₄(τ) = lead_coeff·(A1+A2) + comm_coeff·[A2, A1]
    with lead_coeff = τ/2 and comm_coeff = √3·τ²/12 (math.md §42.1).
    """
    import sympy as sp

    A1 = -L1
    A2 = -L2
    lead_coeff = tau / sp.Integer(2)
    comm_coeff = sp.sqrt(3) * tau**2 / sp.Integer(12)
    return A1, A2, comm_coeff, lead_coeff


def _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign=1):
    """Ω₄ with selectable commutator sign (+1 forward, −1 transposed exponent)."""
    comm = A2 * A1 - A1 * A2  # [A2, A1]
    return lead_coeff * (A1 + A2) + comm_sign * comm_coeff * comm


def _taylor_map(Omega, k_max=4):
    """S(τ) = Σ_{m=0..k_max} Ω^m / m!  (finite degree-4 Taylor polynomial)."""
    import sympy as sp

    n = Omega.shape[0]
    total = sp.zeros(n, n)
    power = sp.eye(n)  # Ω^0 = I
    for m in range(k_max + 1):
        total = total + power / sp.factorial(m)
        power = power * Omega
    return total


def _path3_laplacian(w12, w23):
    """3-node path Laplacian with symbolic edge weights w12, w23 (symmetric)."""
    import sympy as sp

    return sp.Matrix([
        [w12, -w12, 0],
        [-w12, w12 + w23, -w23],
        [0, -w23, w23],
    ])


def _cycle4_laplacian(w01, w12, w23, w30):
    """4-node cycle Laplacian with symbolic edge weights (symmetric)."""
    import sympy as sp

    return sp.Matrix([
        [w01 + w30, -w01, 0, -w30],
        [-w01, w01 + w12, -w12, 0],
        [0, -w12, w12 + w23, -w23],
        [-w30, 0, -w23, w23 + w30],
    ])


def check_commutator_antisymmetry():
    """Sub-check (a): [A₂,A₁]ᵀ = −[A₂,A₁] for symbolic symmetric A_i."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    w12, w23, w01, w23c, w30 = sp.symbols("w12 w23 w01 w23c w30", positive=True)
    tau = sp.symbols("tau", positive=True)

    cases = [
        ("3-path", _path3_laplacian(w12, w23),
                   _path3_laplacian(2 * w12, w23)),
        ("4-cycle", _cycle4_laplacian(w01, w12, w23c, w30),
                    _cycle4_laplacian(w01, 3 * w12, w23c, w30)),
    ]
    for name, L1, L2 in cases:
        # Symmetry premise must actually hold for these test operators.
        if sp.simplify(L1 - L1.T) != sp.zeros(*L1.shape):
            return f"commutator_antisymmetry ({name}): L1 is not symmetric (test setup bug)."
        if sp.simplify(L2 - L2.T) != sp.zeros(*L2.shape):
            return f"commutator_antisymmetry ({name}): L2 is not symmetric (test setup bug)."
        A1, A2, _, _ = _build_operators(L1, L2, tau)
        comm = A2 * A1 - A1 * A2          # [A2, A1]
        residual = sp.expand(comm.T + comm)  # [A2,A1]ᵀ − (−[A2,A1]) = [A2,A1]ᵀ + [A2,A1]
        if residual != sp.zeros(*comm.shape):
            return (
                f"commutator_antisymmetry ({name}): [A₂,A₁]ᵀ ≠ −[A₂,A₁]. "
                f"Residual [A₂,A₁]ᵀ + [A₂,A₁] = {residual} (expected 0). "
                f"Antisymmetry FAILS — Theorem 42.1 commutator step INVALID."
            )
    return None


def check_omega4_transpose_sign_flip():
    """Sub-check (b): Ω₄ᵀ = Ω₄ with ONLY the commutator coefficient negated."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    w12, w23 = sp.symbols("w12 w23", positive=True)
    tau = sp.symbols("tau", positive=True)
    # Two DISTINCT node operators that do NOT commute (so the flip is non-trivial).
    L1 = _path3_laplacian(w12, w23)
    L2 = _path3_laplacian(2 * w12, 3 * w23)
    A1, A2, comm_coeff, lead_coeff = _build_operators(L1, L2, tau)

    Omega = _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign=1)
    Omega_star = _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign=-1)  # §42.2 claimed Ω₄⋆

    # (i) the actual transpose equals the claimed sign-flipped exponent
    residual = sp.expand(Omega.T - Omega_star)
    if residual != sp.zeros(*Omega.shape):
        return (
            f"omega4_transpose_sign_flip: Ω₄ᵀ ≠ (τ/2)(A₁+A₂) − (√3τ²/12)[A₂,A₁]. "
            f"Residual Ω₄ᵀ − Ω₄⋆ = {residual} (expected 0). §42.2 sign-flip claim FAILS."
        )

    # (ii) the flip is GENUINE: Ω₄ᵀ ≠ Ω₄ (commutator is nonzero for these operators)
    if sp.expand(Omega.T - Omega) == sp.zeros(*Omega.shape):
        return (
            "omega4_transpose_sign_flip: Ω₄ᵀ == Ω₄ (commutator vanished). "
            "Test operators commute — the sign-flip would be vacuous. Choose "
            "non-commuting A_i so the claim has content."
        )

    # (iii) the leading symmetric term is unchanged (only the commutator term moves)
    lead = lead_coeff * (A1 + A2)
    lead_residual = sp.expand(lead.T - lead)
    if lead_residual != sp.zeros(*lead.shape):
        return (
            f"omega4_transpose_sign_flip: leading term (τ/2)(A₁+A₂) is NOT symmetric. "
            f"Residual = {lead_residual}. A_i symmetry assumption violated."
        )
    return None


def check_taylor_transpose_factorisation():
    """Sub-check (c): (Σ Ω₄^m/m!)ᵀ = Σ (Ω₄ᵀ)^m/m! = Σ (Ω₄⋆)^m/m!, zero residual."""
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
    Omega_star = _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign=-1)

    S = _taylor_map(Omega, k_max=4)
    S_from_OmegaT = _taylor_map(Omega.T, k_max=4)      # Σ (Ω₄ᵀ)^m/m!
    S_from_OmegaStar = _taylor_map(Omega_star, k_max=4)  # Σ (Ω₄⋆)^m/m!

    # (i) term-by-term transpose factorisation: Sᵀ == Σ (Ω₄ᵀ)^m/m!
    res1 = sp.expand(S.T - S_from_OmegaT)
    if res1 != sp.zeros(*S.shape):
        return (
            f"taylor_transpose_factorisation: (Σ Ω₄^m/m!)ᵀ ≠ Σ (Ω₄ᵀ)^m/m!. "
            f"Residual = {res1} (expected 0). Term-by-term transpose FAILS."
        )

    # (ii) substantive content: Sᵀ == S⋆ built from the sign-flipped Ω₄⋆
    res2 = sp.expand(S.T - S_from_OmegaStar)
    if res2 != sp.zeros(*S.shape):
        return (
            f"taylor_transpose_factorisation: S(τ)ᵀ ≠ Σ (Ω₄⋆)^m/m! (sign-flip map). "
            f"Residual = {res2} (expected 0). State-adjoint kernel identity FAILS."
        )
    return None


def check_dual_pairing_identity():
    """Sub-check (d): ⟨S(τ)u, g⟩ = ⟨u, S⋆(τ)g⟩ on seeded vectors."""
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
    Omega_star = _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign=-1)
    S = _taylor_map(Omega, k_max=4)
    Sstar = _taylor_map(Omega_star, k_max=4)

    n = S.shape[0]
    # Seeded symbolic vectors (distinct integer seeds → no accidental cancellation).
    u = sp.Matrix([sp.Integer(v) for v in (2, -3, 5)][:n])
    g = sp.Matrix([sp.Integer(v) for v in (7, 1, -4)][:n])

    lhs = sp.expand((S * u).dot(g))     # ⟨S u, g⟩
    rhs = sp.expand(u.dot(Sstar * g))   # ⟨u, S⋆ g⟩
    residual = sp.expand(lhs - rhs)
    if residual != 0:
        return (
            f"dual_pairing_identity: ⟨S u, g⟩ ≠ ⟨u, S⋆ g⟩. "
            f"Residual = {residual} (expected 0). S⋆ is NOT the dual of S — "
            f"transcription error in reconstructed Ω₄ / S⋆."
        )
    return None


def check_varcoef_la_symmetry():
    """Sub-check (e): (√a·L·√a)ᵀ = √a·L·√a; (a)–(c) carry to the VarCoef kernel."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    w12, w23 = sp.symbols("w12 w23", positive=True)
    tau = sp.symbols("tau", positive=True)
    a0, a1, a2 = sp.symbols("a0 a1 a2", positive=True)

    sqrt_a = sp.diag(sp.sqrt(a0), sp.sqrt(a1), sp.sqrt(a2))

    def L_a(L):
        return sqrt_a * L * sqrt_a

    L1 = L_a(_path3_laplacian(w12, w23))
    L2 = L_a(_path3_laplacian(2 * w12, 3 * w23))

    # (i) the conjugated operator is symmetric
    for name, La in (("L_a(1)", L1), ("L_a(2)", L2)):
        sym_res = sp.simplify(La - La.T)
        if sym_res != sp.zeros(*La.shape):
            return (
                f"varcoef_la_symmetry: ({name})ᵀ ≠ {name}. Residual = {sym_res} "
                f"(expected 0). √a-conjugation breaks symmetry — (a)–(d) would NOT "
                f"carry to VarCoefMagnusGraphHeatChernoff."
            )

    # (ii) re-run antisymmetry + sign-flip + factorisation on the L_a operators
    A1, A2, comm_coeff, lead_coeff = _build_operators(L1, L2, tau)
    comm = A2 * A1 - A1 * A2
    if sp.expand(comm.T + comm) != sp.zeros(*comm.shape):
        return "varcoef_la_symmetry: [A₂,A₁]ᵀ ≠ −[A₂,A₁] on the L_a kernel."

    Omega = _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign=1)
    Omega_star = _omega4(A1, A2, comm_coeff, lead_coeff, comm_sign=-1)
    if sp.expand(Omega.T - Omega_star) != sp.zeros(*Omega.shape):
        return "varcoef_la_symmetry: Ω₄ᵀ ≠ Ω₄⋆ (sign-flip) on the L_a kernel."

    S = _taylor_map(Omega, k_max=4)
    Sstar = _taylor_map(Omega_star, k_max=4)
    if sp.expand(S.T - Sstar) != sp.zeros(*S.shape):
        return "varcoef_la_symmetry: S(τ)ᵀ ≠ S⋆(τ) on the L_a kernel (factorisation)."
    return None


def fail(reason):
    print(f"T_MAGNUS_TRANSPOSE FAIL: {reason}", flush=True)
    return 1


def main():
    """Run all 5 sub-checks; print result; exit 0/1."""
    checks = [
        ("commutator_antisymmetry", check_commutator_antisymmetry),
        ("omega4_transpose_sign_flip", check_omega4_transpose_sign_flip),
        ("taylor_transpose_factorisation", check_taylor_transpose_factorisation),
        ("dual_pairing_identity", check_dual_pairing_identity),
        ("varcoef_la_symmetry", check_varcoef_la_symmetry),
    ]
    print("=" * 64)
    print("T_MAGNUS_TRANSPOSE — Magnus K=4 state-adjoint transpose-exactness")
    print("(math.md §42, Theorem 42.1; ADR-0115 Issue #2)")
    print("=" * 64)

    failures = []
    passed = []
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
            f"{len(failures)}/{len(checks)} sub-checks failed: " + "; ".join(failures)
        )
    print(
        "T_MAGNUS_TRANSPOSE PASS (5/5 sub-checks: " + " / ".join(passed) + ")",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
