#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
"""T_LADDER_RUNG sympy oracle — ζ-ladder K → K-2 invariant verification (ADR-0100).

Verifies four properties of the sealed LadderRung<K, F> catalogue:

  (a) T_LADDER_RUNG.catalogue_completeness
        Scan approximation.rs source for `impl LadderRung<` patterns.
        Assert exactly 4 matches at K ∈ {2, 4, 6, 8}.

  (b) T_LADDER_RUNG.unique_base
        Scan source for `const PREDECESSOR_K: Option<usize> = None;` lines.
        Assert exactly 1 match, and that match is inside an `impl LadderRung<2,` block.

  (c) T_LADDER_RUNG.k_minus_2_invariant
        Scan source for `const PREDECESSOR_K: Option<usize> = Some(N);` patterns
        for K=4,6,8. Assert each Some(N) equals K - 2, and N ∈ {2, 4, 6}.

  (d) T_LADDER_RUNG.romberg_identity
        For each K ∈ {4, 6, 8}, sympy-verify the Romberg combination formula:
          R^{K/2}(τ) = (α · R^{(K-2)/2}(τ/2)^2 f − R^{(K-2)/2}(τ) f) / (α - 1)
        with α = 4^((K/2)-1). Uses sympy.Symbol("tau") and truncated Taylor
        expansion for the predecessor rung at K-2.

Prints 'T_LADDER_RUNG PASS 4/4 sub-checks: catalogue_completeness / unique_base /
k_minus_2_invariant / romberg_identity' on success; 'T_LADDER_RUNG FAIL: <reason>'
and exits 1 on any failure.

References:
  - ADR-0100 §"Decision" + §"Acceptance gates added".
  - math.md §36.5 — T_LADDER_RUNG oracle specification.
  - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 (order-K Chernoff tangency).
  - Hairer-Lubich-Wanner 2006 *Geometric Numerical Integration* §II.4.
  - scripts/verify_zeta4_correction.py — sibling T23N pattern.
"""

import re
import sys
from pathlib import Path


APPROX_SRC = (
    Path(__file__).parent.parent
    / "crates"
    / "semiflow-core"
    / "src"
    / "approximation.rs"
)

EXPECTED_KS = {2, 4, 6, 8}


def fail(reason: str) -> int:
    print(f"T_LADDER_RUNG FAIL: {reason}", flush=True)
    return 1


def check_catalogue_completeness(src: str) -> str | None:
    """Sub-check (a): exactly 4 sealed LadderRung impls at K ∈ {2,4,6,8}."""
    # Match patterns like `impl LadderRung<2,` or `impl LadderRung<4,`
    matches = re.findall(r"impl\s+LadderRung<(\d+),", src)
    found_ks = {int(k) for k in matches}
    if len(matches) != 4:
        return (
            f"catalogue_completeness: expected 4 impl LadderRung<K,> blocks, "
            f"found {len(matches)} (K values: {sorted(int(k) for k in matches)})"
        )
    if found_ks != EXPECTED_KS:
        return (
            f"catalogue_completeness: expected K ∈ {{2,4,6,8}}, "
            f"found K ∈ {{{', '.join(str(k) for k in sorted(found_ks))}}} "
        )
    return None  # Pass


def check_unique_base(src: str) -> str | None:
    """Sub-check (b): exactly one PREDECESSOR_K = None, inside impl LadderRung<2, ...>."""
    # Find lines with PREDECESSOR_K = None
    none_lines = [
        i
        for i, line in enumerate(src.splitlines(), 1)
        if "PREDECESSOR_K" in line and "None" in line and "Option" in line
    ]
    if len(none_lines) != 1:
        return (
            f"unique_base: expected exactly 1 PREDECESSOR_K = None line, "
            f"found {len(none_lines)} (lines: {none_lines})"
        )

    # Verify that None sentinel appears in the context of impl LadderRung<2,
    # Search for the closest preceding impl LadderRung<K, before the None line.
    lines = src.splitlines()
    none_line_idx = none_lines[0] - 1  # 0-indexed
    for i in range(none_line_idx, max(none_line_idx - 20, -1), -1):
        if re.search(r"impl\s+LadderRung<(\d+),", lines[i]):
            m = re.search(r"impl\s+LadderRung<(\d+),", lines[i])
            assert m is not None
            k_val = int(m.group(1))
            if k_val != 2:
                return (
                    f"unique_base: PREDECESSOR_K = None is in impl LadderRung<{k_val},> "
                    f"(expected K=2 base)"
                )
            return None  # Pass

    return (
        "unique_base: could not find parent impl LadderRung<K,> block "
        "for the PREDECESSOR_K = None sentinel"
    )


def check_k_minus_2_invariant(src: str) -> str | None:
    """Sub-check (c): PREDECESSOR_K = Some(K-2) for K=4,6,8."""
    lines = src.splitlines()
    # For each K in {4,6,8}, find the impl LadderRung<K,> block and verify PREDECESSOR_K
    for k in (4, 6, 8):
        expected_pred = k - 2
        # Find all impl LadderRung<K,> lines
        impl_line_indices = [
            i
            for i, line in enumerate(lines)
            if re.search(rf"impl\s+LadderRung<{k},", line)
        ]
        if not impl_line_indices:
            return f"k_minus_2_invariant: no impl LadderRung<{k},> found in source"
        impl_idx = impl_line_indices[0]
        # Scan forward up to 15 lines for PREDECESSOR_K
        found = False
        for j in range(impl_idx, min(impl_idx + 15, len(lines))):
            m = re.search(
                r"PREDECESSOR_K\s*:\s*Option<usize>\s*=\s*Some\((\d+)\)", lines[j]
            )
            if m:
                pred_val = int(m.group(1))
                if pred_val != expected_pred:
                    return (
                        f"k_minus_2_invariant: K={k} PREDECESSOR_K = Some({pred_val}), "
                        f"expected Some({expected_pred})"
                    )
                if pred_val not in {2, 4, 6}:
                    return (
                        f"k_minus_2_invariant: K={k} PREDECESSOR_K = Some({pred_val}) "
                        f"not in catalogue {{2,4,6}}"
                    )
                found = True
                break
        if not found:
            return (
                f"k_minus_2_invariant: no PREDECESSOR_K = Some(N) found "
                f"near impl LadderRung<{k},> (searched 15 lines)"
            )
    return None  # Pass


def check_romberg_identity() -> str | None:
    """Sub-check (d): Romberg coefficient identity for K=4,6,8.

    For each K ∈ {4, 6, 8}:
      α = 4^((K/2)-1)
      R^{K/2}(τ) f = (α · R^{(K-2)/2}(τ/2)^2 f − R^{(K-2)/2}(τ) f) / (α - 1)

    The predecessor at K-2 is the truncated Taylor expansion truncated to order K-2.
    Verify that the Richardson combination cancels the τ^{K-1} leading error term,
    yielding residual at order τ^K.

    Uses sympy formal power series in τ with abstract operator symbols Af, A2f, etc.
    """
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    tau = sp.Symbol("tau", positive=True)
    # Formal operator algebra: A^k f treated as independent symbols (constructed
    # lazily per-rung inside apply_twice; outer reference kept as documentation).

    def apply_twice(order: int, t: object) -> tuple[object, object, object]:
        """Two half-steps: R(t/2, f) then R(t/2, R(t/2, f)).

        For formal verification we use the commutative approximation:
        each step is a polynomial in t, applied symbolically.
        Since A^k f are independent formal symbols, we use the fact that
        for eigenfunction f with Af = λf: R^2(τ/2)f = P(τλ/2)^2 · f.
        Use eigenvalue λ=1 scalar model for the identity check.
        """
        # Scalar model: A^k f = λ^k f with λ=1 → all A^k f = f_sym (simplest case)
        # But we need to track orders, so use λ as symbol.
        lam = sp.Symbol("lam", positive=True)

        def scalar_step(tau_val: object, ord_: int) -> object:
            return sum(
                sp.Rational(1, sp.factorial(k)) * (tau_val * lam) ** k
                for k in range(ord_ + 1)
            )

        half_step = scalar_step(t / 2, order)
        two_half_steps = sp.expand(half_step**2)
        full_step = scalar_step(t, order)
        return two_half_steps, full_step, lam

    for k in (4, 6, 8):
        pred_order = k - 2
        alpha = sp.Integer(4) ** (k // 2 - 1)
        alpha_minus_1 = alpha - 1

        two_half, full, lam = apply_twice(pred_order, tau)
        # Richardson combination: (α · two_half - full) / (α - 1)
        combined = sp.expand((alpha * two_half - full) / alpha_minus_1)

        # Reference: exact exponential up to order k (eigenvalue model)
        exp_ref = sum(
            sp.Rational(1, sp.factorial(j)) * (tau * lam) ** j for j in range(k + 2)
        )

        residual = sp.expand(exp_ref - combined)

        # Verify τ^0 through τ^{k-1} coefficients in residual vanish
        residual_expr: sp.Expr = sp.sympify(residual)
        for p in range(k):
            coeff_p = residual_expr.coeff(tau, p)
            if sp.simplify(coeff_p) != 0:
                return (
                    f"romberg_identity: K={k}, α={alpha}, τ^{p} residual = {coeff_p} "
                    f"(expected 0); Richardson combination has wrong order"
                )

        # Verify the τ^k coefficient is nonzero (i.e., we achieved exactly order k)
        coeff_k = residual_expr.coeff(tau, k)
        coeff_k_simp = sp.simplify(coeff_k)
        if coeff_k_simp == 0:
            return (
                f"romberg_identity: K={k}, τ^{k} residual coefficient vanishes — "
                f"Richardson achieved higher order than expected (degenerate case)"
            )

        # Verify α and (α-1) match the expected Romberg pair values
        expected_alpha = {4: 4, 6: 16, 8: 64}[k]
        expected_div = {4: 3, 6: 15, 8: 63}[k]
        if int(alpha) != expected_alpha:
            return (
                f"romberg_identity: K={k} α={alpha}, expected {expected_alpha} "
                f"(= 4^((K/2)-1) = 4^{k//2 - 1})"
            )
        if int(alpha_minus_1) != expected_div:
            return (
                f"romberg_identity: K={k} α-1={alpha_minus_1}, "
                f"expected {expected_div}"
            )

    return None  # Pass


def main() -> int:
    if not APPROX_SRC.exists():
        return fail(f"approximation.rs not found at {APPROX_SRC}")

    src = APPROX_SRC.read_text(encoding="utf-8")

    # Sub-check (a): catalogue completeness
    err = check_catalogue_completeness(src)
    if err is not None:
        return fail(f"T_LADDER_RUNG.catalogue_completeness: {err}")

    # Sub-check (b): unique base
    err = check_unique_base(src)
    if err is not None:
        return fail(f"T_LADDER_RUNG.unique_base: {err}")

    # Sub-check (c): K → K-2 invariant
    err = check_k_minus_2_invariant(src)
    if err is not None:
        return fail(f"T_LADDER_RUNG.k_minus_2_invariant: {err}")

    # Sub-check (d): Romberg coefficient identity (symbolic)
    err = check_romberg_identity()
    if err is not None:
        return fail(f"T_LADDER_RUNG.romberg_identity: {err}")

    print(
        "T_LADDER_RUNG PASS 4/4 sub-checks: catalogue_completeness / "
        "unique_base / k_minus_2_invariant / romberg_identity",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
