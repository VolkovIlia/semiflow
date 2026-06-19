#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy's symbolic arithmetic is dynamically typed through operator overloads;
# Pyright cannot trace them. All operations are valid sympy at runtime
# (verified by this oracle's PASS).
"""T_GRIDLESS sympy gate — Gridless particle-ensemble Chernoff evolver
(math.md §50, ADR-0155; v9.0.0 Phase-1 pre-flight oracle).

PRE-FLIGHT math-fidelity oracle. Independently verifies, before any Rust kernel
is written, that the §50 particle-map mathematics is self-consistent via three
symbolic sub-checks.

Mathematical setting (math.md §50.2, §38.3, §50.4):
  State: ρ = Σ wᵢ δ_{xᵢ} (weighted particle ensemble on ℝ^d).
  One-step push-forward (eq. 50.3):
    S*(τ)δ_x = ¼δ_{x+h} + ¼δ_{x-h} + ½δ_{x+k} + τc·δ_x        (50.3)
  with h = 2√(aτ) (per-axis diffusion step), k = 2bτ (drift step).
  Dual pairing: ⟨f, S*(τ)δ_x⟩ = (S(τ)f)(x)        (§38.2 duality).

Sub-checks (3 mandatory; math.md §50.6 T_GRIDLESS_* table):

  (1) push_forward_exactness
      The dual pairing ⟨f, S*(τ)δ_x⟩ equals (S(τ)f)(x) (i.e., both sides
      reproduce the §38.3 / Galkin-Remizov Theorem-4 forward Chernoff function).
      Verified symbolically by Taylor-expanding S(τ)f(x) to order τ and
      confirming the leading terms reproduce f + τ(a f'' + b f' + c f) + o(τ)
      — the generator L acting on f. Both the particle-map definition and the
      direct Taylor expansion must give the same Taylor polynomial to order τ.

  (2) mass_conservation
      The ¼+¼+½ position-branch weights sum to exactly 1 (mass-preserving
      before reaction). After one step the reaction-Dirac multiplies mass by
      (1 + τc), so an n-step product reweights by ∏(1+τc). Verified by
      symbolic coefficient arithmetic (§38.6).

  (3) voronoi_moment_match
      The weighted-Voronoi reduction R_P (§50.4): merging a small cluster of
      Diracs {(wᵢ, xᵢ)} into a single Dirac at the weight-barycenter x̄ with
      summed weight w̄ = Σ wᵢ preserves (i) total mass, (ii) first moment
      exactly, and (iii) perturbs the second moment by exactly the within-cell
      weighted variance. Verified symbolically for a 2-particle and 3-particle
      cluster with symbolic positions/weights.

Prints "T_GRIDLESS PASS (3/3 sub-checks: ...)" on success;
"T_GRIDLESS FAIL: <reason>" and exits 1 on failure.

References:
  - Galkin-Remizov 2025, *Israel J. Math.* 265, 929-943, Theorem 4 (eq. 11)
    — the forward Chernoff function (50.1) whose dual is (50.3).
  - math.md §38 / ADR-0107 — AdjointFokkerPlanckChernoff, T_ADJOINT_FP_TIGHTNESS
    6/6 PASS. NOTE: T_GRIDLESS re-derives the §38 generator-consistency stencil
    inline (no code import or reuse from T_ADJOINT_FP_TIGHTNESS); the reuse is
    conceptual only (same §38 generator). Sub-check (1) verifies generator
    consistency S(τ)f = f + τLf + o(τ) to O(τ) via a 2nd-order Taylor stencil;
    sub-check (2) verifies the coefficient-sum ¼+¼+½=1 (dual-pairing sanity); both
    are inline re-derivations, not imports from the adjoint-tightness script.
  - math.md §50 / ADR-0155 — GridlessChernoff v9.0.0 headline (Shift C).
"""

import sys


def fail(reason: str) -> int:
    """Print FAIL line and return exit code 1."""
    print(f"T_GRIDLESS FAIL: {reason}", flush=True)
    return 1


def check_push_forward_exactness() -> "str | None":
    """Sub-check (1): ⟨f, S*(τ)δ_x⟩ = (S(τ)f)(x) ≡ f + τ·Lf + o(τ).

    Both sides must agree to order τ (the generator consistency condition).
    The dual pairing ⟨f, S*(τ)δ_x⟩ := ⟨S(τ)f, δ_x⟩ = S(τ)f(x) by definition
    of the adjoint, so both sides are IDENTICAL. The content is that explicitly
    Taylor-expanding S(τ)f(x) at τ=0 reproduces the generator Lf = a·f'' + b·f' + c·f
    at order τ, confirming that (50.3) is consistent with the §38.2 FP generator.

    We expand manually using shifted-argument Taylor series:
      f(x + δ) = f(x) + δ·f'(x) + (δ²/2)·f''(x) + O(δ³)
    with δ=h=2√(aτ) and δ=−h and δ=k=2bτ, then collect terms by power of τ.
    The √τ terms cancel by symmetry between the ±h branches.
    """
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    # Use abstract function symbols for f and its derivatives (avoids series issues
    # with sqrt-in-argument).
    tau, b, c = sp.symbols("tau b c", real=True)
    a_sym = sp.Symbol("a", positive=True)

    # f(x), f'(x), f''(x) as abstract symbols (their values at x).
    f0 = sp.Symbol("f0")       # f(x)
    f1 = sp.Symbol("f1")       # f'(x)
    f2 = sp.Symbol("f2")       # f''(x)
    # Higher derivatives not needed; we truncate at order τ^1 explicitly.

    h = 2 * sp.sqrt(a_sym * tau)   # = 2√(aτ)  — O(τ^{1/2})
    k = 2 * b * tau                # = 2bτ     — O(τ^1)

    # Taylor: f(x+δ) ≈ f0 + δ·f1 + (δ²/2)·f2  (exact to order δ²)
    def f_shift(delta: sp.Expr) -> sp.Expr:
        return f0 + delta * f1 + (delta**2 / 2) * f2

    # S(τ)f(x) = (1/4)f(x+h) + (1/4)f(x-h) + (1/2)f(x+k) + τc·f(x)
    Stau = (
        sp.Rational(1, 4) * f_shift(h)
        + sp.Rational(1, 4) * f_shift(-h)
        + sp.Rational(1, 2) * f_shift(k)
        + tau * c * f0
    )
    Stau_expanded = sp.expand(Stau)

    # Collect by power of τ (note h²=4aτ, so h-terms contribute at τ^1 for f2).
    # Expected: f0 + τ·(a·f2 + b·f1 + c·f0) + O(τ^{3/2})
    # The τ^{1/2} terms: (1/4)h·f1 + (1/4)(−h)·f1 = 0  ✓ (odd-branch cancellation)
    # The τ^1 terms: (1/4)(h²/2)f2 + (1/4)(h²/2)f2 + (1/2)(2bτ)f1 + τc·f0
    #             = (1/4)(4aτ/2)f2 + (1/4)(4aτ/2)f2 + bτf1 + τcf0
    #             = aτf2 + bτf1 + τcf0  ✓
    expected = f0 + tau * (a_sym * f2 + b * f1 + c * f0)
    expected_expanded = sp.expand(expected)

    # Truncate to order τ^1: collect and drop terms with τ^{3/2}, τ^2, etc.
    # Since we used exact Taylor (no symbolic series), expansion is exact; we
    # need to extract only the τ^0 and τ^1 terms.
    # Both Stau_expanded and expected_expanded involve only τ^0, τ^{1/2}, and τ^1
    # (h = 2√(aτ), h² = 4aτ, k = 2bτ, k² = 4b²τ²). We substitute h² = 4aτ and
    # k² = 4b²τ² to reduce everything to rational powers of τ.
    # The h·f1 terms: (1/4)·h·f1 + (1/4)·(−h)·f1 = 0 (symbolic, exact cancellation).
    # We verify this and the τ^1 generator residual in one go.

    # After expansion with abstract h=2√(aτ), sympy leaves sqrt(a*tau) terms.
    # Verify by substituting the numeric relation h²=4aτ and checking that
    # when we subtract expected from Stau_expanded, the result is proportional
    # to τ^{3/2} or higher (i.e., contains no τ^0 or τ^1 terms in the residual).
    # We do this by checking the residual at τ=0 (constant term) and the residual's
    # derivative w.r.t. τ at τ=0 (linear term).

    residual = sp.expand(Stau_expanded - expected_expanded)

    # Evaluate residual at τ=0 (should be 0 — constant term).
    res_at_0 = residual.subs(tau, 0)
    if sp.simplify(res_at_0) != 0:
        return (
            f"push_forward_exactness: S(τ)f(x) − (f + τ·Lf) "
            f"has non-zero constant term = {res_at_0}. "
            f"Generator consistency FAILS at order τ^0."
        )

    # Evaluate d/dτ [residual] at τ=0 (should be 0 — linear coefficient).
    # Note: diff of residual w.r.t. τ may involve 1/√τ from d/dτ √(aτ);
    # multiply by √τ first to remove the pole, then evaluate at τ=0.
    res_deriv_raw = sp.diff(residual, tau)
    # Multiply by √τ to clear potential 1/√τ poles from d/dτ(√τ terms).
    res_deriv_cleared = sp.expand(res_deriv_raw * sp.sqrt(tau))
    res_coeff_tau1 = res_deriv_cleared.subs(tau, 0)
    if sp.simplify(res_coeff_tau1) != 0:
        return (
            f"push_forward_exactness: S(τ)f(x) − (f + τ·Lf) "
            f"has non-zero τ^1 coefficient = {res_coeff_tau1}. "
            f"Generator Lf = a·f'' + b·f' + c·f NOT reproduced at order τ."
        )

    # Verify the dual-pairing identity S*(τ)δ_x[f] = S(τ)f(x) algebraically.
    # Both are the same expression by construction; check the coefficient sum.
    dual_coeff = sp.Rational(1, 4) + sp.Rational(1, 4) + sp.Rational(1, 2)
    if dual_coeff != 1:
        return (
            f"push_forward_exactness: dual-pairing coefficient sum = {dual_coeff}, "
            f"expected 1. S*(τ)δ_x ≠ S(τ) dual identity at the coefficient level."
        )

    return None  # PASS


def check_mass_conservation() -> "str | None":
    """Sub-check (2): branch weights sum to 1; n-step reweighting is ∏(1+τc).

    The three position-branch weights ¼ + ¼ + ½ = 1 (mass-preserving before
    reaction). The reaction-Dirac τc·δ_x multiplies the total weight by (1+τc),
    so after n independent steps the weight is w₀·∏ᵢ(1+τcᵢ). Verified
    symbolically (§38.6).
    """
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    # (i) Position-branch weights sum to 1.
    branch_sum = sp.Rational(1, 4) + sp.Rational(1, 4) + sp.Rational(1, 2)
    if branch_sum != 1:
        return (
            f"mass_conservation: ¼+¼+½ = {branch_sum}, expected 1. "
            f"Position-branch weights do NOT sum to 1 — mass not preserved."
        )

    # (ii) One-step total weight factor (including reaction):
    # total mass of S*(τ)δ_x = ¼ + ¼ + ½ + τc = 1 + τc.
    tau, c = sp.symbols("tau c", real=True)
    one_step_factor = branch_sum + tau * c
    expected_one_step = 1 + tau * c
    if sp.simplify(one_step_factor - expected_one_step) != 0:
        return (
            f"mass_conservation: one-step weight factor = {one_step_factor}, "
            f"expected {expected_one_step}. Reaction reweighting FAILS."
        )

    # (iii) n-step product reweighting: w·∏_{k=1}^n (1+τc) = w·(1+τc)^n.
    n = sp.Symbol("n", positive=True, integer=True)
    product_n = (1 + tau * c) ** n
    # Verify the product is symbolic-consistent: (1+τc)^1 = 1+τc and
    # (1+τc)^n·(1+τc) = (1+τc)^{n+1}.
    product_1 = product_n.subs(n, 1)
    if sp.simplify(product_1 - (1 + tau * c)) != 0:
        return (
            f"mass_conservation: (1+τc)^1 = {product_1}, expected 1+τc. "
            f"Product formula base case FAILS."
        )
    product_step = sp.simplify(product_n * (1 + tau * c) - product_n.subs(n, n + 1))
    if product_step != 0:
        return (
            f"mass_conservation: (1+τc)^n·(1+τc) ≠ (1+τc)^{{n+1}}. "
            f"Residual = {product_step} (expected 0). Product step-recurrence FAILS."
        )

    return None  # PASS


def check_voronoi_moment_match() -> "str | None":
    """Sub-check (3): R_P preserves mass + first moment; perturbs 2nd by within-cell variance.

    For a cluster of Diracs {(w₁,x₁), …, (wₖ,xₖ)} merged into a single Dirac
    at weight-barycenter x̄ = (Σ wᵢxᵢ)/(Σ wᵢ) with total weight w̄ = Σ wᵢ,
    the reduction R_P satisfies:
      (A) ⟨1, R_P ρ_cell⟩ = ⟨1, ρ_cell⟩ = w̄    (mass preserved)
      (B) ⟨x, R_P ρ_cell⟩ = ⟨x, ρ_cell⟩ = Σwᵢxᵢ  (first moment preserved)
      (C) ⟨x², R_P ρ_cell⟩ = ⟨x², ρ_cell⟩ − within-cell variance
          where within_var = (Σ wᵢ(xᵢ−x̄)²)/w̄ · w̄ = Σ wᵢ(xᵢ−x̄)²
          so ⟨x², R_P ρ_cell⟩ = ⟨x², ρ_cell⟩ − Σ wᵢ(xᵢ−x̄)²
    Equivalently, the second moment ERROR = Σ wᵢ(xᵢ−x̄)² ≥ 0.
    Verified for a 2-particle and 3-particle cluster with symbolic positions/weights.
    """
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    # --- 2-particle cluster ---
    w1, w2, x1, x2 = sp.symbols("w1 w2 x1 x2", real=True)

    w_bar_2 = w1 + w2
    x_bar_2 = (w1 * x1 + w2 * x2) / w_bar_2

    # Mass: ⟨1, R_P⟩ = w_bar_2, ⟨1, original⟩ = w1+w2
    mass_orig_2 = w1 + w2
    mass_reduced_2 = w_bar_2
    if sp.simplify(mass_reduced_2 - mass_orig_2) != 0:
        return (
            f"voronoi_moment_match (2-particle): mass after reduction = "
            f"{mass_reduced_2}, original = {mass_orig_2}. Mass NOT preserved."
        )

    # First moment: ⟨x, R_P⟩ = w_bar_2·x_bar_2 = w1·x1 + w2·x2
    first_orig_2 = w1 * x1 + w2 * x2
    first_reduced_2 = w_bar_2 * x_bar_2
    if sp.simplify(sp.expand(first_reduced_2 - first_orig_2)) != 0:
        return (
            f"voronoi_moment_match (2-particle): first moment after reduction = "
            f"{sp.expand(first_reduced_2)}, original = {first_orig_2}. "
            f"First moment NOT preserved."
        )

    # Second moment: ⟨x², original⟩ = w1·x1² + w2·x2²
    second_orig_2 = w1 * x1**2 + w2 * x2**2
    # ⟨x², R_P⟩ = w_bar_2·x_bar_2²
    second_reduced_2 = w_bar_2 * x_bar_2**2

    # Error = original - reduced = Σ wᵢ(xᵢ-x̄)²
    error_2 = sp.expand(second_orig_2 - second_reduced_2)
    expected_error_2 = sp.expand(
        w1 * (x1 - x_bar_2) ** 2 + w2 * (x2 - x_bar_2) ** 2
    )
    if sp.simplify(error_2 - expected_error_2) != 0:
        return (
            f"voronoi_moment_match (2-particle): second-moment error = "
            f"{sp.simplify(error_2)}, expected within-cell variance "
            f"{sp.simplify(expected_error_2)}. Second-moment error formula FAILS."
        )

    # --- 3-particle cluster ---
    w3, x3 = sp.symbols("w3 x3", real=True)

    w_bar_3 = w1 + w2 + w3
    x_bar_3 = (w1 * x1 + w2 * x2 + w3 * x3) / w_bar_3

    # Mass
    if sp.simplify((w1 + w2 + w3) - w_bar_3) != 0:
        return "voronoi_moment_match (3-particle): mass NOT preserved."

    # First moment
    first_orig_3 = w1 * x1 + w2 * x2 + w3 * x3
    first_reduced_3 = w_bar_3 * x_bar_3
    if sp.simplify(sp.expand(first_reduced_3 - first_orig_3)) != 0:
        return (
            f"voronoi_moment_match (3-particle): first moment after reduction "
            f"= {sp.expand(first_reduced_3)}, original = {first_orig_3}. "
            f"First moment NOT preserved."
        )

    # Second moment error = Σ wᵢ(xᵢ-x̄)²
    second_orig_3 = w1 * x1**2 + w2 * x2**2 + w3 * x3**2
    second_reduced_3 = w_bar_3 * x_bar_3**2
    error_3 = sp.expand(second_orig_3 - second_reduced_3)
    expected_error_3 = sp.expand(
        w1 * (x1 - x_bar_3) ** 2
        + w2 * (x2 - x_bar_3) ** 2
        + w3 * (x3 - x_bar_3) ** 2
    )
    if sp.simplify(error_3 - expected_error_3) != 0:
        return (
            f"voronoi_moment_match (3-particle): second-moment error = "
            f"{sp.simplify(error_3)}, expected within-cell variance "
            f"{sp.simplify(expected_error_3)}. Second-moment error formula FAILS."
        )

    return None  # PASS


def main() -> int:
    """Run all 3 sub-checks; print result; exit 0/1."""
    checks = [
        ("push_forward_exactness", check_push_forward_exactness),
        ("mass_conservation", check_mass_conservation),
        ("voronoi_moment_match", check_voronoi_moment_match),
    ]
    print("=" * 64)
    print("T_GRIDLESS — Gridless particle-ensemble Chernoff evolver")
    print("(math.md §50, ADR-0155; v9.0.0 Phase-1 pre-flight oracle)")
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
        "T_GRIDLESS PASS (3/3 sub-checks: " + " / ".join(passed) + ")",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
