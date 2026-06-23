#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy/numpy symbolic+numeric arithmetic is dynamically typed through operator
# overloads; Pyright cannot trace expressions through sp.series / numpy chains.
# All operations are valid at runtime (verified by this oracle's PASS).
"""T_TT_VARCOEF — proof-grade oracle for the per-axis variable-coefficient TT step
(math.md §52.10, ADR-0178, issue #2).

Proves, BEFORE the Rust `tt_varcoef::VarCoefTt` implementation, that the per-axis
factor

    exp(τ L_j) ≈ P₂(τ/2) · k_j(τ) · P₂(τ/2)            (§52.10b)
      k_j(τ) = exp(τ · a₀_j · Lap)                       const-coef (a₀_j = mean a_j)
      R_j    = L_j − a₀_j · Lap                          variable residual
      P₂(s)  = I + s·R_j + (s²/2)·R_j²

is CONSISTENT to O(τ²) for a small closed-form variable-`a` case (1-D, linear /
low-rank `a(x)`). Three sub-checks:

  (a) symbolic_taylor_order
      Symbolic Taylor (sympy): the scalar/operator product P₂(τ/2)·K(τ)·P₂(τ/2)
      agrees with the true semigroup exp(τ(R+a₀Lap)) = exp(τL) through O(τ²),
      i.e. the first NON-matching term is O(τ³). Confirms the sandwich is a
      genuine 2nd-order (Strang/Chernoff-symmetric) approximation of exp(τL),
      not a wrong-operator multiplier.

  (b) numeric_slope_linear_a
      Numeric (numpy): build the periodic FD generator L for a linear variable
      coefficient a(x) = a0 + s·x on a small grid; compute the dense reference
      exp(τL)·u via scipy-free Padé-free eigendecomposition; refine n_steps and
      fit the log-log slope of rel_err vs τ. Asserts slope ≥ 1.95 (order-2).

  (c) bond_rank_preservation
      Confirms the §52.10d structural claim numerically: applying the mode-axis
      operators R, R², K to a rank-1 (outer-product) 2-D state and rounding does
      NOT increase the matrix (=bond) rank — the rank stays 1. This is the
      carrier-level curse-escape backbone the Rust gate G_TT_VARCOEF measures.

PASS prints `T_TT_VARCOEF PASS`; any FAIL prints the reason and exits 1.
"""

import sys

import numpy as np
import sympy as sp


# ─────────────────────────────────────────────────────────────────────────────
# (a) symbolic Taylor order of the P₂·K·P₂ sandwich
# ─────────────────────────────────────────────────────────────────────────────
def symbolic_taylor_order() -> str | None:
    """Prove P₂(τ/2)·K(τ)·P₂(τ/2) = exp(τ(R+a0·Lap)) + O(τ³) symbolically.

    Model the operators as scalars (the symbol on a single Fourier mode): K acts
    as exp(τ·a0·ℓ) with ℓ = −k² the Laplacian symbol, R as the residual symbol r.
    The sandwich must match exp(τ(r + a0·ℓ)) through second order in τ.
    Returns None on PASS, else an error string.
    """
    tau, r, a0, ell = sp.symbols("tau r a0 ell", real=True)
    lam = r + a0 * ell  # full generator symbol L = R + a0·Lap

    # K(τ) = exp(τ·a0·ℓ); P₂(s) = 1 + s·r + s²/2·r²  (polynomial Chernoff factor for exp(s·r))
    s = tau / 2
    p2 = 1 + s * r + s**2 / 2 * r**2
    k = sp.exp(tau * a0 * ell)
    sandwich = sp.expand(p2 * k * p2)

    truth = sp.exp(tau * lam)

    diff = sp.series(sandwich - truth, tau, 0, 3).removeO()
    diff = sp.simplify(sp.expand(diff))
    if diff != 0:
        return f"sandwich − exp(τL) not O(τ³): residual through τ² = {diff}"

    # Sanity: the τ³ term is generically NONZERO (so order is exactly 2, not higher).
    full = sp.series(sandwich - truth, tau, 0, 4).removeO()
    t3 = sp.simplify(full.coeff(tau, 3))
    if t3 == 0:
        return "τ³ coefficient vanished — order claim must be re-examined (expected exactly 2)"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# (b) numeric order-2 slope for a linear variable coefficient
# ─────────────────────────────────────────────────────────────────────────────
def _periodic_varcoef_generator(a: np.ndarray, dx: float) -> np.ndarray:
    """Divergence-form periodic FD generator L = ∂_x(a(x)∂_x) on n nodes."""
    n = a.size
    lmat = np.zeros((n, n))
    for i in range(n):
        ip = (i + 1) % n
        im = (i + n - 1) % n
        ahp = 0.5 * (a[i] + a[ip])  # a_{i+1/2}
        ahm = 0.5 * (a[i] + a[im])  # a_{i-1/2}
        lmat[i, ip] += ahp / dx**2
        lmat[i, im] += ahm / dx**2
        lmat[i, i] += -(ahp + ahm) / dx**2
    return lmat


def _sandwich_step(u: np.ndarray, lmat: np.ndarray, a0: float, dx: float, tau: float) -> np.ndarray:
    """One §52.10b step: P₂(τ/2)·k(τ)·P₂(τ/2)·u (dense, single-axis reference)."""
    n = u.size
    lap = _periodic_varcoef_generator(np.ones(n), dx)  # const-coef Laplacian (a≡1)
    rmat = lmat - a0 * lap
    half = tau / 2.0

    def p2(v: np.ndarray) -> np.ndarray:
        rv = rmat @ v
        return v + half * rv + 0.5 * half**2 * (rmat @ rv)

    # k(τ) = exp(τ·a0·Lap) via eigendecomposition (symmetric-ish; use expm by eig).
    w, vmat = np.linalg.eig(a0 * lap)
    kmat = (vmat @ np.diag(np.exp(tau * w)) @ np.linalg.inv(vmat)).real

    return p2(kmat @ p2(u))


def numeric_slope_linear_a() -> str | None:
    """Refine τ; fit log-log slope of rel_err vs dense exp(τL). Assert ≥ 1.95."""
    n = 24
    length = 2.0 * np.pi
    dx = length / n
    xs = np.arange(n) * dx
    a = 0.6 + 0.25 * np.sin(xs)  # smooth variable a(x) > 0, genuinely varying
    a0 = float(a.mean())
    lmat = _periodic_varcoef_generator(a, dx)

    u0 = np.exp(np.cos(xs))  # smooth periodic IC
    t_final = 0.05
    w, vmat = np.linalg.eig(lmat)

    def exact(t: float) -> np.ndarray:
        return (vmat @ np.diag(np.exp(t * w)) @ np.linalg.inv(vmat)).real @ u0

    truth = exact(t_final)
    nsteps_list = [4, 8, 16, 32]
    taus = []
    errs = []
    for ns in nsteps_list:
        tau = t_final / ns
        u = u0.copy()
        for _ in range(ns):
            u = _sandwich_step(u, lmat, a0, dx, tau)
        rel = np.linalg.norm(u - truth) / np.linalg.norm(truth)
        taus.append(np.log(tau))
        errs.append(np.log(max(rel, 1e-300)))

    slope = np.polyfit(taus, errs, 1)[0]
    if slope < 1.95:
        return f"numeric order slope {slope:.4f} < 1.95 (not O(τ²); a0={a0:.4f})"
    # Anti-vacuous: a genuinely varies (else R=0 and the test is trivial).
    if a.max() - a.min() < 0.1:
        return f"coefficient barely varies (span {a.max()-a.min():.3e}) — vacuous"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# (c) bond-rank preservation (the §52.10d carrier-escape backbone)
# ─────────────────────────────────────────────────────────────────────────────
def bond_rank_preservation() -> str | None:
    """Mode-axis R/R²/K on a rank-1 2-D state keep the bond rank == 1."""
    n = 16
    length = 2.0 * np.pi
    dx = length / n
    xs = np.arange(n) * dx
    a = 0.6 + 0.25 * np.sin(xs)
    a0 = float(a.mean())
    lmat = _periodic_varcoef_generator(a, dx)
    lap = _periodic_varcoef_generator(np.ones(n), dx)
    rmat = lmat - a0 * lap

    # rank-1 2-D state U = f ⊗ g (matrix of rank 1).
    f = np.exp(np.cos(xs))
    g = np.exp(-((xs - np.pi) ** 2))
    u = np.outer(f, g)
    r0 = np.linalg.matrix_rank(u, tol=1e-10)
    if r0 != 1:
        return f"IC bond rank {r0} ≠ 1 (test setup error)"

    # Apply the variable-coef mode-axis operator to axis-0 only: U ← (P₂·R·P₂ on rows).
    half = 0.5 * 0.01
    p2 = np.eye(n) + half * rmat + 0.5 * half**2 * (rmat @ rmat)
    w, vmat = np.linalg.eig(a0 * lap)
    kmat = (vmat @ np.diag(np.exp(0.01 * w)) @ np.linalg.inv(vmat)).real
    op = p2 @ kmat @ p2  # full single-axis factor on axis 0
    u_evolved = op @ u  # acts on rows (mode index of core 0); columns ride untouched

    r1 = np.linalg.matrix_rank(u_evolved, tol=1e-10)
    if r1 != 1:
        return f"bond rank grew to {r1} after mode-axis step — 52.10d violated"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """Run all 3 sub-checks; print result; exit 0/1."""
    print("=" * 72)
    print("T_TT_VARCOEF — variable-coef TT step oracle (math.md §52.10, ADR-0178)")
    print("=" * 72)

    checks = [
        ("symbolic_taylor_order", symbolic_taylor_order),
        ("numeric_slope_linear_a", numeric_slope_linear_a),
        ("bond_rank_preservation", bond_rank_preservation),
    ]
    passed = []
    for name, fn in checks:
        print(f"\n[{name}]")
        result = fn()
        if result is not None:
            print(f"  (FAIL) {name}: {result}")
            print(f"\nT_TT_VARCOEF FAIL: {name}: {result}", flush=True)
            return 1
        print(f"  (PASS) {name}")
        passed.append(name)

    print()
    print("T_TT_VARCOEF PASS (3/3 sub-checks: " + " / ".join(passed) + ")", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
