#!/usr/bin/env python3
"""PRE-FLIGHT (v7.0.0 Phase-5 item #22) — Smolyak sparse grid for D>=5
(math.md §32.6 / §32.3; ADR-0130; 🚫 dep-watch — NO 4th dependency).

Goal
----
The d-D anisotropic shift kernel uses tensor-product Gauss-Hermite: q^D nodes
per evaluation point. At D=5, q=5 that is 5^5 = 3125 (the gated ceiling); for
finer q or D>=6 it explodes. The Smolyak combination technique builds a SPARSE
grid from low-order 1-D rules that retains a target polynomial-exactness class at
DRAMATICALLY fewer nodes, with bounded error.

This PRE-FLIGHT must establish, with NUMBERS:
  1. The Smolyak combination
        A(ℓ,D) = Σ_{ℓ-D+1 ≤ |i| ≤ ℓ} (-1)^{ℓ-|i|} C(D-1, ℓ-|i|) ⊗_k U_{i_k}
     of nested 1-D Gauss-Hermite rules U_i integrates the sparse polynomial
     exactness class (total-degree ≤ 2ℓ-1 style) EXACTLY (to f64), at node count
     FAR below the tensor product q^D.
  2. For D=5 a feasible level ℓ reaches the accuracy the kernel needs (matching
     the tensor-product 5^5 result on the kernel's Gaussian-class integrand to a
     bounded error) at NODE COUNT << 3125.
  3. The nodes/weights are GENERATED constants bakeable in-tree (no sparse-grid
     crate) — exactly like scripts/generate_chebyshev_nodes.py emits Rust tables.

GO: D=5 Smolyak hits the required exactness/accuracy at node count < full tensor
3125, AND the construction needs no 4th dependency (pure numpy here → pure
in-tree Rust there). NO-GO (honest-defer): if a CORRECT Smolyak rule for the
Gaussian-weighted d-D average is infeasible at reasonable in-tree code size or
cannot beat the tensor node count at the required accuracy.
"""

import itertools
from math import comb

import numpy as np


def gh_rule(level):
    """Nested 1-D Gauss-Hermite rule U_i. We use a growth q(level): level 1→1pt,
    2→3pt, 3→5pt, 4→7pt, 5→9pt (the {1,3,5,7,9} ladder reused from item #21).
    Returns (nodes, weights) for physicist weight e^{-x²}.
    """
    q = {1: 1, 2: 3, 3: 5, 4: 7, 5: 9}[level]
    n, w = np.polynomial.hermite.hermgauss(q)
    return n, w


def smolyak_nodes_weights(D, ell):
    """Build Smolyak sparse grid A(ell, D) over D dims using the combination
    technique. Returns dict {node_tuple: weight} (duplicate nodes merged).

    Index set: multi-indices i=(i_1..i_D), i_k>=1, with  ell-D+1 <= |i| <= ell
    (the standard Smolyak admissible set). Combination coefficient:
        c(i) = (-1)^{ell-|i|} * C(D-1, ell-|i|).
    """
    acc = {}
    for i in itertools.product(range(1, ell + 1), repeat=D):
        s = sum(i)
        if s < ell - D + 1 or s > ell:
            continue
        coeff = ((-1) ** (ell - s)) * comb(D - 1, ell - s)
        if coeff == 0:
            continue
        rules = [gh_rule(ik) for ik in i]
        node_axes = [r[0] for r in rules]
        weight_axes = [r[1] for r in rules]
        for combo in itertools.product(*[range(len(na)) for na in node_axes]):
            node = tuple(round(float(node_axes[d][combo[d]]), 14) for d in range(D))
            w = coeff
            for d in range(D):
                w *= float(weight_axes[d][combo[d]])
            acc[node] = acc.get(node, 0.0) + w
    # drop near-zero merged weights
    return {k: v for k, v in acc.items() if abs(v) > 1e-15}


def tensor_integral(D, q, g):
    """Full tensor-product GH integral of g over R^D with weight e^{-Σx²}."""
    n, w = np.polynomial.hermite.hermgauss(q)
    total = 0.0
    for combo in itertools.product(range(q), repeat=D):
        node = np.array([n[combo[d]] for d in range(D)])
        wt = 1.0
        for d in range(D):
            wt *= w[combo[d]]
        total += wt * g(node)
    return total, q ** D


def smolyak_integral(D, ell, g):
    nw = smolyak_nodes_weights(D, ell)
    total = sum(w * g(np.array(node)) for node, w in nw.items())
    return total, len(nw)


def exact_gaussian_weighted(D, g_kind):
    """Analytic ∫_{R^D} g(x) e^{-Σ x²} dx for test integrands with closed form."""
    # weight per axis integrates to √π; we build g so the answer is closed-form.
    sqrt_pi = np.sqrt(np.pi)
    if g_kind == "const":
        return sqrt_pi ** D
    raise ValueError


def run():
    print("PRE-FLIGHT: Smolyak sparse grid for D>=5 (item #22, NO 4th dep)")
    D = 5

    # ---- (1) Polynomial exactness: total-degree class ----
    # The kernel's tensor 5^5 is exact to per-axis degree 9 (=> total degree up
    # to 9 in EACH variable). Smolyak with the {1,3,5,7,9} growth at level ell
    # is exact for the TOTAL-DEGREE class d_tot(ell). We verify Smolyak
    # integrates a battery of monomials x^α e^{-Σx²} exactly and report the
    # largest total degree it nails, vs node count.
    print("\n  --- (1) exactness vs node count (D=5) ---")
    print(f"  {'ell':>3s} {'#nodes':>8s} {'tensor 5^5':>11s} {'max_total_deg_exact':>20s}")
    from math import gamma

    def mono_exact(alpha):
        val = 1.0
        for a in alpha:
            val *= 0.0 if a % 2 == 1 else gamma((a + 1) / 2.0)
        return val

    best = None
    for ell in [D, D + 1, D + 2, D + 3]:
        nw = smolyak_nodes_weights(D, ell)
        nnodes = len(nw)
        # find max total degree exactly integrated (check all monomials up to deg T)
        max_deg = 0
        for T in range(0, 13):
            ok_T = True
            # check every multi-index alpha with |alpha| == T
            for alpha in _multi_indices(D, T):
                num = sum(
                    w * np.prod([node[d] ** alpha[d] for d in range(D)])
                    for node, w in nw.items()
                )
                ana = mono_exact(alpha)
                if abs(num - ana) > 1e-7 * max(1.0, abs(ana)):
                    ok_T = False
                    break
            if ok_T:
                max_deg = T
            else:
                break
        flag = ""
        if nnodes < 5 ** D:
            flag = "< tensor"
        print(f"  {ell:>3d} {nnodes:>8d} {5**D:>11d} {max_deg:>20d}  {flag}")
        if best is None and nnodes < 5 ** D and max_deg >= 5:
            best = (ell, nnodes, max_deg)

    # ---- (2) Accuracy on the kernel's Gaussian-class integrand ----
    # Representative kernel integrand at small effective shift s: a smooth
    # product profile g(x)=Π (1 + s x_d + ½ (s x_d)²). Compare Smolyak vs full
    # tensor 5^5 (the gated reference) and the analytic value.
    print("\n  --- (2) accuracy vs full tensor 5^5 (kernel-class integrand) ---")
    s = 0.3

    def g_kernel(x):
        out = 1.0
        for d in range(len(x)):
            t = s * x[d]
            out *= (1.0 + t + 0.5 * t * t)
        return out

    # Reference: the kernel integrand is EXACTLY a degree-(2D)=10 total-degree
    # polynomial (product of per-axis quadratics), so the analytic integral is
    # closed-form. We use the full tensor q=5 as the reference (it integrates per
    # axis degree 9 ≥ 2 exactly → it IS the exact value to f64). Smolyak must
    # approach it; the operative threshold is the accuracy at which the Chernoff
    # product's order-1 TEMPORAL truncation dominates the quadrature floor. For a
    # COARSE-grid order-1 gate (slope ≤ −1.95 is the item's gate but order-1
    # kernels target ≤ −0.95; the freeze lists −1.95 as the Smolyak gate, i.e. the
    # sparse rule must not degrade the kernel below its tensor accuracy class), a
    # relative quadrature error ≤ 1e-5 is comfortably below the temporal signal at
    # the gated step counts. We report the level crossing 1e-5 (operative) AND
    # 1e-6 (stretch).
    I_tensor, n_tensor = tensor_integral(D, 5, g_kernel)
    print(f"  full tensor q=5 (=exact for this deg-10 poly):  "
          f"I={I_tensor:.12e}  nodes={n_tensor}")
    feasible = None      # operative: rel < 1e-5 at nodes < tensor
    feasible_tight = None  # stretch: rel < 1e-6
    for ell in [D, D + 1, D + 2, D + 3, D + 4]:
        I_sm, n_sm = smolyak_integral(D, ell, g_kernel)
        err = abs(I_sm - I_tensor)
        rel = err / abs(I_tensor)
        cheaper = n_sm < n_tensor
        print(f"  smolyak ell={ell}: I={I_sm:.12e}  nodes={n_sm:>5d}  "
              f"abs_err={err:.3e}  rel={rel:.3e}  cheaper={cheaper}")
        if feasible is None and cheaper and rel < 1e-5:
            feasible = (ell, n_sm, rel)
        if feasible_tight is None and cheaper and rel < 1e-6:
            feasible_tight = (ell, n_sm, rel)

    # ---- (3) bakeable as generated constants ----
    print("\n  --- (3) bakeable in-tree (no 4th dep) ---")
    nw = smolyak_nodes_weights(D, D + 1)
    print(f"  D=5 ell={D+1}: {len(nw)} unique (node,weight) pairs to bake as "
          f"const arrays (mirror generate_chebyshev_nodes.py).")
    print(f"  uses only: itertools (loop unroll in Rust), Gauss-Hermite tables "
          f"(already in-tree), binomial coeffs (const) → NO new dependency.")
    bakeable = len(nw) > 0 and len(nw) < 5 ** D

    # ---- VERDICT ----
    print("\n================ VERDICT ================")
    exact_ok = best is not None
    print(f"  (1) Smolyak exact for total-deg>=5 at nodes<tensor: {exact_ok}"
          + (f"  (ell={best[0]}, nodes={best[1]}, max_deg={best[2]})" if best else ""))
    acc_ok = feasible is not None
    print(f"  (2) D=5 matches tensor 5^5 within 1e-5 (operative) at nodes<3125: "
          f"{acc_ok}"
          + (f"  (ell={feasible[0]}, nodes={feasible[1]}, rel={feasible[2]:.1e})"
             if feasible else ""))
    if feasible_tight:
        print(f"      stretch 1e-6 reached at ell={feasible_tight[0]}, "
              f"nodes={feasible_tight[1]} (rel={feasible_tight[2]:.1e})")
    print(f"  (3) bakeable as in-tree generated constants (no 4th dep): {bakeable}")
    verdict = exact_ok and acc_ok and bakeable
    print(f"  PRE-FLIGHT: {'PASS — GO' if verdict else 'FAIL — NO-GO/honest-defer'}")
    return 0 if verdict else 1


def _multi_indices(D, total):
    """All non-negative integer D-tuples summing to `total`."""
    if D == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _multi_indices(D - 1, total - first):
            yield (first,) + rest


if __name__ == "__main__":
    raise SystemExit(run())
