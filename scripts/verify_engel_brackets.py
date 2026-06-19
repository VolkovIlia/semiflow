#!/usr/bin/env python3
# pyright: reportUnusedVariable=false, reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy dynamic-typing through Symbol / Array / Matrix; Pyright cannot trace
# the return types. All operations are valid sympy at runtime — verified by
# the T_HORM_ENGEL_BRACKETS PASS gate.
"""T_HORM_ENGEL_BRACKETS: Engel step-3 filiform Carnot bracket sympy verification (ADR-0095).

Engel group E = filiform step-3 Carnot group with N=4 (Bonfiglioli-Lanconelli-
Uguzzoni 2007 §4.3.6, Prop. 4.3.8 + Remark 4.3.9). Coordinates (x1, x2, x3, x4)
∈ ℝ⁴. Stratification g = g₁ ⊕ g₂ ⊕ g₃ with dim(g_1)=2, dim(g_2)=dim(g_3)=1.

Left-invariant fields on E (Bonfiglioli 2007 §4.3.6 Definition 4.3.4 +
Bratzlavsky 1974 basis — Theorem 4.3.6):
  X1 = ∂_{x1}
  X2 = ∂_{x2} + x1·∂_{x3} + (x1²/2)·∂_{x4}
  X3 = ∂_{x3} + x1·∂_{x4}        (computed as [X1, X2])
  X4 = ∂_{x4}                      (computed as [X1, X3])

Stratification (Bratzlavsky 1974, cited Theorem 4.3.6):
  g_1 = span{X1, X2}        (horizontal, 2 generators per Prop. 4.3.8 two-generator rule)
  g_2 = span{X3 = [X1, X2]}
  g_3 = span{X4 = [X1, X3] = [X1, [X1, X2]]}
  g_4 = 0                   (filiform termination at depth N-1 = 3)

Sub-Laplacian (math.md §28.bis): L_E = X1² + X2²  — bracket-generating at step 3.

5 mandatory sub-checks (math.md §28.bis.4):
  (1) bracket_12          : [X1, X2] = X3 symbolically
  (2) bracket_13          : [X1, X3] = X4 symbolically
  (3) bracket_23          : [X2, X3] = 0 symbolically (filiform terminates depth-2 in g_2)
  (4) filiform_termination: [X1, X4] = 0 AND [X2, X4] = 0 AND [X3, X4] = 0
                            (g_3 is the centre; ad_{X_i} X_4 = 0 for all i)
  (5) hormander_rank      : dim(span{X1, X2, [X1,X2], [X1,[X1,X2]]}) = 4
                            (Hörmander step-3 condition: horizontal + 1+2 nested brackets span TM)

Prints exactly:
  T_HORM_ENGEL_BRACKETS PASS         — all 5 sub-checks pass → architect blesses Outcome A
  T_HORM_ENGEL_BRACKETS FAIL: <msg>  — first failing sub-check → architect → Outcome B

Exit code: 0 on PASS, 1 on FAIL.

References:
  - Bonfiglioli, Lanconelli, Uguzzoni 2007 §4.3.6 (pp. 207-209), Theorem 4.3.6
    (Bratzlavsky basis), Prop. 4.3.8 (filiform two-generator characterisation)
  - Hörmander 1967 *Acta Math.* 119:1, §1 (bracket-generating condition)
  - Folland 1975 *Ark. Mat.* 13, pp. 161-207 (nilpotent Lie group sub-Laplacians)
  - math.md §28.bis (NEW — Engel construction; Architect ADR-0095)
  - scripts/lie_bracket_kit.py — reusable Lie-bracket sympy helpers (v3.1)
  - ADR-0095 §"Sympy verification" (this script is the PASS precondition)

Usage:
    python3 scripts/verify_engel_brackets.py
"""

import os
import sys

# Make lie_bracket_kit.py importable without packaging.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import sympy as sp  # noqa: E402

from lie_bracket_kit import generates_T, lie_bracket  # noqa: E402

# ─── Engel coordinate system + left-invariant fields ─────────────────────────

x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", real=True)
coords = (x1, x2, x3, x4)

# Bonfiglioli 2007 §4.3.6 explicit filiform N=4 fields (Bratzlavsky basis).
# X1 = ∂_{x1}  → vector (1, 0, 0, 0)
# X2 = ∂_{x2} + x1·∂_{x3} + (x1²/2)·∂_{x4} → vector (0, 1, x1, x1²/2)
X1 = sp.Array([sp.S.One, sp.S.Zero, sp.S.Zero, sp.S.Zero])
X2 = sp.Array([sp.S.Zero, sp.S.One, x1, x1**2 / 2])

# X3 := [X1, X2]: computed sympy, expected = (0, 0, 1, x1)  — i.e. ∂_{x3} + x1·∂_{x4}
# X4 := [X1, X3]: computed sympy, expected = (0, 0, 0, 1)   — i.e. ∂_{x4}
# We DERIVE these (not assert) so the test verifies the canonical formulas.
X3_expected = sp.Array([sp.S.Zero, sp.S.Zero, sp.S.One, x1])
X4_expected = sp.Array([sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.One])


def check(name: str, ok: bool, detail: str = "") -> None:
    """Print PASS/FAIL and exit on first failure."""
    if ok:
        print(f"  [{name}] PASS")
    else:
        print(f"  [{name}] FAIL — {detail}")
        print(f"T_HORM_ENGEL_BRACKETS FAIL: {name}: {detail}")
        sys.exit(1)


def arrays_equal(a: sp.Array, b: sp.Array) -> bool:
    """Element-wise symbolic equality after simplify."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        diff = sp.simplify(a[i] - b[i])
        if diff != sp.S.Zero:
            return False
    return True


def is_zero_field(a: sp.Array) -> bool:
    """All components simplify to 0."""
    for i in range(len(a)):
        if sp.simplify(a[i]) != sp.S.Zero:
            return False
    return True


def main() -> None:
    """Run all 5 sub-checks; print T_HORM_ENGEL_BRACKETS PASS on success."""
    print("T_HORM_ENGEL_BRACKETS — Engel step-3 filiform Carnot bracket verification")
    print("  Source: Bonfiglioli-Lanconelli-Uguzzoni 2007 §4.3.6 + ADR-0095")
    print()

    # ─── Sub-check 1: [X1, X2] = X3 = ∂_{x3} + x1·∂_{x4} ──────────────────────
    X3_computed = lie_bracket(X1, X2, coords)
    ok1 = arrays_equal(X3_computed, X3_expected)
    detail1 = (
        f"expected (0,0,1,x1) got {tuple(sp.simplify(X3_computed[i]) for i in range(4))}"
        if not ok1
        else ""
    )
    check("bracket_12", ok1, detail1)

    # ─── Sub-check 2: [X1, X3] = X4 = ∂_{x4} ──────────────────────────────────
    X4_computed = lie_bracket(X1, X3_expected, coords)
    ok2 = arrays_equal(X4_computed, X4_expected)
    detail2 = (
        f"expected (0,0,0,1) got {tuple(sp.simplify(X4_computed[i]) for i in range(4))}"
        if not ok2
        else ""
    )
    check("bracket_13", ok2, detail2)

    # ─── Sub-check 3: [X2, X3] = 0  (filiform: g_2 commutes with g_2) ─────────
    # In a filiform step-3 algebra, g_2 = span{X3} is 1-dim Abelian so [X3, X3] = 0,
    # AND [X2, X3] lives in g_3 = span{X4}; we verify it is in fact zero (filiform
    # is stronger than generic step-3: the only non-trivial step-3 bracket is
    # [X1, X3] = X4; all others vanish).
    b23 = lie_bracket(X2, X3_expected, coords)
    ok3 = is_zero_field(b23)
    detail3 = (
        f"expected 0 got {tuple(sp.simplify(b23[i]) for i in range(4))}" if not ok3 else ""
    )
    check("bracket_23", ok3, detail3)

    # ─── Sub-check 4: filiform termination — [X_i, X4] = 0 for i=1,2,3 ────────
    # X4 = ∂_{x4} is in the centre of the algebra (filiform N=4 → step 3, depth 4 = 0).
    b14 = lie_bracket(X1, X4_expected, coords)
    b24 = lie_bracket(X2, X4_expected, coords)
    b34 = lie_bracket(X3_expected, X4_expected, coords)
    centre_ok = is_zero_field(b14) and is_zero_field(b24) and is_zero_field(b34)
    detail4 = ""
    if not centre_ok:
        detail4 = (
            f"[X1,X4]={tuple(b14)}, [X2,X4]={tuple(b24)}, [X3,X4]={tuple(b34)}"
        )
    check("filiform_termination", centre_ok, detail4)

    # ─── Sub-check 5: Hörmander rank = 4 at origin (step-3) ───────────────────
    # Span{X1, X2, [X1,X2]=X3, [X1,[X1,X2]]=X4} at x=0 must have rank 4.
    # This is the canonical step-3 bracket-generating verification.
    origin = {x1: 0, x2: 0, x3: 0, x4: 0}
    rank_ok = generates_T([X1, X2, X3_expected, X4_expected], coords, origin)
    detail5 = "" if rank_ok else "rank-deficient at origin — Engel is NOT bracket-generating?!"
    check("hormander_rank", rank_ok, detail5)

    # ─── Bonus diagnostic: confirm step-2 brackets DO NOT span (justifies step-3 need) ──
    # If span{X1, X2, [X1,X2]} had rank 4, Engel would be step-2. Verify it has rank 3.
    M_step2 = sp.Matrix(
        [
            [X1[i].subs(origin) for i in range(4)],
            [X2[i].subs(origin) for i in range(4)],
            [X3_expected[i].subs(origin) for i in range(4)],
        ]
    )
    step2_rank = M_step2.rank()
    print(
        f"  [diagnostic] step-2 rank = {step2_rank} "
        f"(expected 3 = strictly < 4 → Engel is genuine step-3, NOT degenerate step-2)"
    )
    if step2_rank != 3:
        print(
            f"T_HORM_ENGEL_BRACKETS FAIL: diagnostic: step-2 rank is {step2_rank}, "
            "expected 3 (would imply Engel is degenerate or formula error)"
        )
        sys.exit(1)

    # ─── All sub-checks pass ──────────────────────────────────────────────────
    print()
    print("T_HORM_ENGEL_BRACKETS PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
