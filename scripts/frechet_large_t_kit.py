#!/usr/bin/env python3
"""ADR-0203 pre-flight kit: Fréchet-gradient accuracy versus lambda_max * t.

Reproduces the ADR-0203 evidence table and the §63.4 quadrature constant.

Compared against a Daleckii-Krein eigen oracle (§63.1.b), error / G_k (§63.1.c):
  1. semiflow.symmetric_op_expmv_frechet (current binding), if importable;
  2. the pre-ADR-0203 one-panel GL8 rule with EXACT eigen propagators
     (isolates the quadrature error from the propagators);
  3. the §63.3 two-sided graded rule with exact propagators;
  4. the §63.5 near/far sweep with chained (incremental) propagation.

Usage:  python3 scripts/frechet_large_t_kit.py
Needs numpy + scipy; semiflow optional.
"""
import math
import sys

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import leggauss

try:
    import semiflow
except ImportError:  # the kit is meaningful without the binding
    semiflow = None

X8, W8 = leggauss(8)
NODES = (X8 + 1) / 2
WTS = W8 / 2
Q_RATIO = 1.5
DELTA = 2.0


def stencil(n, i, j):
    m = np.zeros((n, n))
    m[i, j] = 1.0
    m[j, i] = 1.0
    return m


def f1(t, la, lb):
    """Cancellation-free divided difference of e^{-t x} (§63.1.b)."""
    lo, hi = np.minimum(la, lb), np.maximum(la, lb)
    th = t * (hi - lo)
    safe = np.where(th > 0, th, 1.0)
    phi = np.where(th > 0, -np.expm1(-th) / safe, 1.0)
    return -t * np.exp(-t * lo) * phi


def oracle(lam, vec, u0, dj, t, entries):
    al, be = vec.T @ dj, vec.T @ u0
    big_f = f1(t, lam[:, None], lam[None, :])
    g, gabs = [], []
    for (i, j) in entries:
        mt = vec.T @ stencil(len(lam), i, j) @ vec
        terms = al[:, None] * mt * be[None, :] * big_f
        g.append(terms.sum())
        gabs.append(np.abs(terms).sum())
    return np.array(g), np.array(gabs)


def make_prop(lam, vec):
    return lambda tau, x: vec @ (np.exp(-tau * lam) * (vec.T @ x))


def bilinear(a, b, i, j):
    """<a, (dL/dL_ij) b>; the gradient sign (dA = -dL) is applied by callers."""
    return a[i] * b[j] + a[j] * b[i] if i != j else a[i] * b[i]


def gl8_legacy(prop, u0, dj, t, entries):
    g = np.zeros(len(entries))
    for s, w in zip(NODES, WTS):
        a, b = prop((1 - s) * t, dj), prop(s * t, u0)
        for k, (i, j) in enumerate(entries):
            g[k] -= t * w * bilinear(a, b, i, j)
    return g


def mesh_half(half, rho):
    """§63.3: 0 < d_0 < ... < d_K = half, multiply loop, no ln."""
    d0 = min(half, DELTA / rho) if rho > 0 else half
    pts = [0.0, d0]
    while pts[-1] < half:
        pts.append(min(pts[-1] * Q_RATIO, half))
    return np.array(pts)


def panels(half, rho):
    b = mesh_half(half, rho)
    return [(lo, hi - lo) for lo, hi in zip(b[:-1], b[1:])]


def graded_exact(prop, u0, dj, t, rho, entries):
    g = np.zeros(len(entries))
    for lo, h in panels(t / 2, rho):
        for x, w in zip(NODES, WTS):
            r = lo + h * x
            for a, b in ((prop(t - r, dj), prop(r, u0)), (prop(r, dj), prop(t - r, u0))):
                for k, (i, j) in enumerate(entries):
                    g[k] -= h * w * bilinear(a, b, i, j)
    return g


def half_sweep(prop, near_src, far_src, near_is_b, t, rho, entries, g):
    half = t / 2
    far, r_far = prop(half, far_src), half
    for lo, h in reversed(panels(half, rho)):
        r = lo + h * NODES
        near = [prop(r[0], near_src)]
        for q in range(1, 8):
            near.append(prop(r[q] - r[q - 1], near[-1]))
        for q in reversed(range(8)):
            far = prop(r_far - r[q], far)
            r_far = r[q]
            a, b = (far, near[q]) if near_is_b else (near[q], far)
            for k, (i, j) in enumerate(entries):
                g[k] -= h * WTS[q] * bilinear(a, b, i, j)


def graded_chained(prop, u0, dj, t, rho, entries):
    g = np.zeros(len(entries))
    half_sweep(prop, u0, dj, True, t, rho, entries, g)   # left half
    half_sweep(prop, dj, u0, False, t, rho, entries, g)  # right half
    return g


def add_edge(mat, i, j, w):
    mat[i, i] += w
    mat[j, j] += w
    mat[i, j] -= w
    mat[j, i] -= w


def stiff_contrast(n=12, fast=1e6, seed=0):
    """Fixture F2: two fast rings, one slow bridge, a Robin leak (SPD)."""
    rng = np.random.default_rng(seed)
    mat, edges, h = np.zeros((n, n)), [], n // 2
    for ring in (list(range(0, h)), list(range(h, n))):
        pairs = list(zip(ring[:-1], ring[1:])) + [(ring[0], ring[-1])]
        for (i, j) in pairs:
            add_edge(mat, i, j, fast * rng.uniform(0.5, 1.5))
            edges.append((min(i, j), max(i, j)))
    add_edge(mat, h - 1, h, 1.0)
    edges.append((h - 1, h))
    mat[0, 0] += 0.3
    return mat, edges + [(0, 0), (h, h)]


def random_graph(n=12, seed=1):
    """Fixture F1: random weighted graph containing a path."""
    rng = np.random.default_rng(seed)
    mat, edges = np.zeros((n, n)), []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform() < 0.35 or j == i + 1:
                add_edge(mat, i, j, rng.uniform(0.2, 2.0))
                edges.append((i, j))
    return mat, edges


def semiflow_grad(mat, u0, dj, t, entries):
    if semiflow is None:
        return None
    from scipy.sparse import csr_matrix
    csr = csr_matrix(mat)
    op = semiflow.SymmetricOperator.from_csr(
        csr.indptr.astype(np.int64), csr.indices.astype(np.int32), csr.data, mat.shape[0])
    return semiflow.symmetric_op_expmv_frechet(
        op, u0[:, None], dj[:, None], t=t, entries=entries)


def fmt(x):
    return "      n/a" if x is None else f"{x:9.2e}"


def report(name, mat, entries, lts, rng):
    n = mat.shape[0]
    u0, dj = rng.standard_normal(n), rng.standard_normal(n)
    lam, vec = np.linalg.eigh(mat)
    prop = make_prop(lam, vec)
    rho = 2 * np.max(np.diag(mat))
    print(f"\n== {name}: n={n} lam_min={lam[0]:.3g} lam_max={lam[-1]:.3g}")
    print("  lam*t |  semiflow | GL8 exact | graded ex | graded ch | nodes")
    for lt in lts:
        t = lt / lam[-1]
        g, gabs = oracle(lam, vec, u0, dj, t, entries)
        def err(x, g=g, gabs=gabs):
            return None if x is None else float(np.max(np.abs(x - g) / gabs))

        sf = err(semiflow_grad(mat, u0, dj, t, entries))
        old = err(gl8_legacy(prop, u0, dj, t, entries))
        ex = err(graded_exact(prop, u0, dj, t, rho, entries))
        ch = err(graded_chained(prop, u0, dj, t, rho, entries))
        nodes = 16 * len(panels(t / 2, rho))
        print(f" {lt:6.0e} | {fmt(sf)} | {fmt(old)} | {fmt(ex)} | {fmt(ch)} | {nodes:5d}")
        if lt <= 100:
            (i, j) = entries[0]
            _, fr = sla.expm_frechet(-t * mat, -t * stencil(n, i, j))
            print(f"         oracle vs scipy.expm_frechet: {abs(dj @ fr @ u0 - g[0]) / gabs[0]:.2e}")


def quad_constant():
    """sup rel. error of the §63.3 rule on single exponentials (G_FRECHET_QUAD_CONSTANT)."""
    worst = 0.0
    for mut in np.logspace(-3, 8, 600):
        for over in (1.0, 2.0):
            near, far = [], []
            for lo, h in panels(0.5, mut * over):
                r = lo + h * NODES
                near.extend(h * WTS * np.exp(-mut * r))
                far.extend(h * WTS * np.exp(-mut * (1.0 - r)))
            exact = -np.expm1(-mut) / mut
            worst = max(worst, abs(math.fsum(near) + math.fsum(far) - exact) / exact)
    return worst


def apriori_constant():
    """Proposition 63.4 bound."""
    c8 = math.factorial(8) ** 4 / (17 * math.factorial(16) ** 3)
    best = 0.0
    for x0 in np.linspace(1e-3, Q_RATIO, 3000)[:-1]:
        xs = x0 * Q_RATIO ** np.arange(80)
        best = max(best, float(np.sum(((Q_RATIO - 1) * xs) ** 17 * np.exp(-xs))))
    inner = c8 * DELTA ** 17
    far = c8 * (17 / 3) ** 17 * math.exp(-17) * 1.001
    return (c8 * best + inner + far) / (1 - math.exp(-1))


if __name__ == "__main__":
    print(f"eps_Q a-priori (Prop. 63.4): {apriori_constant():.3e}")
    print(f"eps_Q measured sup:          {quad_constant():.3e}")
    grid = [1, 2.6, 10, 13, 53, 1e2, 1e3, 1e4, 1e5, 1e6]
    gen = np.random.default_rng(42)
    m1, e1 = random_graph()
    report("F1 random graph", m1, e1, grid, gen)
    m2, e2 = stiff_contrast()
    report("F2 stiff contrast", m2, e2, grid, gen)
    sys.exit(0)
