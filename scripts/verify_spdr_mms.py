#!/usr/bin/env python3
"""G_SPDR_STEADY_MMS oracle (ADR-0202 + Amendment 1, math §62.3).

Manufactured steady problem on [0,1], homogeneous Neumann:
    -(k u')' + c u = s,   u = 2 + cos(pi x),  k = 1 + x^2,  c = 1 + x.
FV system solved by the library (vertex-centred, harmonic faces, lumped mass):
    (A + diag(c*m)) u_h = m*s,   m = [1/2, 1, ..., 1, 1/2],   A_ij = k_h/dx^2 stencil.

Part 1  sympy derives s(x) and checks u'(0) = u'(1) = 0.

Part 2  A-PRIORI error constant (no discrete solve involved).  With h = dx,
        q = k u' (true flux), the discrete face flux of the exact u expands as
            F_h = q + h^2 g + O(h^4),
            g = k u'''/24 + (k''/8 - k'^2/(4k)) u'        (harmonic face mean),
        both verified below by sympy series.  Cell residuals of the exact u:
            interior cell : -h^3 (g' + q'''/24) + O(h^5)
            boundary cell : +-h^2 (g + q''/8)   + O(h^3)   (half cell, one-sided)
        The boundary term is O(h) per unit cell length but lives on ONE cell, so it
        enters the global error at O(h^2) as a flux (Neumann) defect.  Hence
            u_h - u = h^2 w + o(h^2),
            -(k w')' + c w = g' + q'''/24  on (0,1),
            k w'(0) = -(g + q''/8)(0),   k w'(1) = -(g + q''/8)(1),
        and the a-priori max-norm constant is C* = max|w|.  Here
        (g + q''/8)(0) = 0 (k, u even about 0) but (g + q''/8)(1) = pi^2/2
        (k'(1) = 2 breaks the reflection symmetry), so the boundary defect
        dominates: C* = 3.506917 (interior-only C = 1.565, the value the
        pre-implementation 1e-5 bound implicitly assumed).

Part 3  INDEPENDENT pure-Python Thomas solve of the same FV system gives the
        measured err(n), n in {65,129,257,513}, the OLS slope and C(n) = err*(n-1)^2,
        and checks |C(n)/C* - 1| <= 2% for every n.

Prints 'G_SPDR_STEADY_MMS PASS' (exit 0) or 'G_SPDR_STEADY_MMS FAIL: ...' (exit 1).
The Rust gate pins the Part-2 constant and the Part-3 errors.
"""
import math
import sys

import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp

x, h = sp.symbols("x h", real=True)
u = 2 + sp.cos(sp.pi * x)
k = 1 + x**2
c = 1 + x
q = k * sp.diff(u, x)
s = sp.simplify(-sp.diff(q, x) + c * u)


def fail(msg):
    print(f"G_SPDR_STEADY_MMS FAIL: {msg}")
    sys.exit(1)


def part1():
    for b in (0, 1):
        if sp.simplify(sp.diff(u, x).subs(x, b)) != 0:
            fail(f"u'({b}) != 0")
    print("s(x) =", s)


def part2_face_expansion(g):
    """Check F_h = k_h (u(x+h/2)-u(x-h/2))/h = q + h^2 g + O(h^4) at face x."""
    kp, km = k.subs(x, x + h / 2), k.subs(x, x - h / 2)
    flux = 2 * kp * km / (kp + km) * (u.subs(x, x + h / 2) - u.subs(x, x - h / 2)) / h
    ser = sp.series(flux, h, 0, 4).removeO()
    if sp.simplify(ser - q - h**2 * g) != 0:
        fail("face-flux expansion F_h = q + h^2 g does not hold")


def part2_constant():
    g = (
        k * sp.diff(u, x, 3) / 24
        + (sp.diff(k, x, 2) / 8 - sp.diff(k, x) ** 2 / (4 * k)) * sp.diff(u, x)
    )
    part2_face_expansion(g)
    src = sp.diff(g, x) + sp.diff(q, x, 3) / 24
    beta = -(g + sp.diff(q, x, 2) / 8)
    b0, b1 = float(beta.subs(x, 0)), float(beta.subs(x, 1))
    print(f"boundary flux defect: k w'(0) = {b0:.6f}, k w'(1) = {b1:.6f} (= -pi^2/2)")
    src_f, k_f, c_f = (sp.lambdify(x, e, "numpy") for e in (src, k, c))

    def rhs(xx, yy):  # yy = [w, k w']
        return np.vstack([yy[1] / k_f(xx), c_f(xx) * yy[0] - src_f(xx)])

    grid = np.linspace(0.0, 1.0, 2001)
    sol = solve_bvp(rhs, lambda a, b: np.array([a[1] - b0, b[1] - b1]),
                    grid, np.zeros((2, grid.size)), tol=1e-9, max_nodes=10**5)
    if sol.status != 0:
        fail(f"solve_bvp status {sol.status}")
    w = sol.sol(grid)[0]
    c_star = float(np.abs(w).max())
    print(f"a-priori C* = max|w| = {c_star:.6f} at x = {grid[np.abs(w).argmax()]:.4f}")
    return c_star


def thomas(lo, di, up, rhs):
    n = len(di)
    d, r = di[:], rhs[:]
    for i in range(1, n):
        wgt = lo[i - 1] / d[i - 1]
        d[i] -= wgt * up[i - 1]
        r[i] -= wgt * r[i - 1]
    out = [0.0] * n
    out[-1] = r[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        out[i] = (r[i] - up[i] * out[i + 1]) / d[i]
    return out


def fv_err(n, u_f, s_f, k_f, c_f):
    dx = 1.0 / (n - 1)
    xs = [i * dx for i in range(n)]
    kk = [k_f(v) for v in xs]
    t = [2 * kk[i] * kk[i + 1] / (kk[i] + kk[i + 1]) / dx for i in range(n - 1)]
    m = [1.0] * n
    m[0] = m[-1] = 0.5
    di = []
    for i in range(n):
        tl = t[i - 1] if i > 0 else 0.0
        tr = t[i] if i < n - 1 else 0.0
        di.append((tl + tr) / dx + c_f(xs[i]) * m[i])
    off = [-ti / dx for ti in t]
    sol = thomas(off, di, off, [m[i] * s_f(xs[i]) for i in range(n)])
    return max(abs(sol[i] - u_f(xs[i])) for i in range(n))


def part3(c_star):
    fns = [sp.lambdify(x, e, "math") for e in (u, s, k, c)]
    ns = [65, 129, 257, 513]
    errs = [fv_err(n, *fns) for n in ns]
    lx = [math.log(1.0 / (n - 1)) for n in ns]
    ly = [math.log(e) for e in errs]
    mx, my = sum(lx) / 4, sum(ly) / 4
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
    for n, e in zip(ns, errs):
        cn = e * (n - 1) ** 2
        print(f"n={n:4d} err={e:.6e} C(n)={cn:.6f} C(n)/C*-1={cn / c_star - 1:+.2e}")
        if abs(cn / c_star - 1) > 0.02:
            fail(f"n={n}: measured constant {cn:.4f} deviates >2% from a-priori {c_star:.4f}")
    print(f"OLS slope vs log dx = {slope:.4f}")
    if not 1.8 <= slope <= 2.2:
        fail(f"slope {slope:.4f} outside [1.8, 2.2]")


if __name__ == "__main__":
    part1()
    part3(part2_constant())
    print("G_SPDR_STEADY_MMS PASS")
