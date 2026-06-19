#!/usr/bin/env python3
"""tt_coupled_crossterm_verdict.py — DECISIVE PROCEED/BOUNDARY probe for §52.9.

Question (maintainer, pre-v9.1.0): is CoupledTtChernoff's non-convergence to the
TRUE coupled semigroup e^{TL}u0 a FIXABLE cross-term scaling bug, or a FUNDAMENTAL
obstruction of the integer-shift Chernoff structure?

This script faithfully replicates the Rust CoupledTtChernoff::step algorithm
(crates/semiflow-core/src/tt_coupled.rs) on the DENSE n^d tensor (small n,d) — NOT
an explicit-Euler stand-in — and measures the error vs the analytic correlated-
Gaussian semigroup under tau-refinement at FIXED fine dx, for THREE coupling
scalings:

  (A) CURRENT  : scale = tau * rho * sqrt(a_j a_k),                D1u (un-normalised, [-1/2,0,1/2])
  (B) FIX-1/dx2: scale = tau * rho * sqrt(a_j a_k) / (dx_j dx_k),  D1u  (divide by dx_j dx_k)
  (C) FIX-1/dx2 + factor 2 : scale = 2 * tau * rho * sqrt(...) /(dx_j dx_k)

The continuous generator (correlated-Gaussian / cross-diffusion Fokker-Planck) is
    L u = a1 d1^2 u + a2 d2^2 u + 2*rho*sqrt(a1 a2) d1 d2 u   (b=c=0 for the test)
with diffusion matrix  Sigma = [[a1, rho sqrt(a1a2)],[rho sqrt(a1a2), a2]].
A Gaussian IC u0 = N(0, S0) heat-evolves to  S_T = S0 + 2 T Sigma  (closed form).

The Rust step structure replicated EXACTLY:
  1. diagonal per-axis shift sweep: each axis gets the 3-branch (1/4,1/4,1/2)
     integer-index periodic shift by  s = round(h/dx),  h = 2 sqrt(a tau) (+drift).
  2. coupling sweep: state <- state + scale * (D1_j (x) D1_k) state
  3. reaction (1+tau c)   [c=0 here]
  (TT-rounding omitted: on the dense tensor it only affects rank/storage, NOT the
   evolved values up to eps — accuracy is what we test here.)

Run:  python3 scripts/tt_coupled_crossterm_verdict.py
Deps: numpy, scipy (scipy optional; matrix-exp reference closed-form below).
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


# --- single-axis operators on a periodic grid -------------------------------

def d1_unnorm(n):
    """Un-normalised central difference [-1/2, 0, 1/2] (matches Rust apply_d1)."""
    M = np.zeros((n, n))
    for i in range(n):
        M[i, (i + 1) % n] += 0.5
        M[i, (i - 1) % n] -= 0.5
    return M


def shift_3branch(n, s):
    """3-branch (1/4,1/4,1/2) integer-index periodic shift by s grid steps.

    Matches Rust apply_per_axis_shift: G_new[i] = 1/4 G[i+s] + 1/4 G[i-s] + 1/2 G[i].
    s=0 -> identity.
    """
    M = np.zeros((n, n))
    s = s % n
    for i in range(n):
        M[i, (i + s) % n] += 0.25
        M[i, (i - s) % n] += 0.25
        M[i, i] += 0.5
    if s == 0:
        # all three land on i -> identity (0.25+0.25+0.5 = 1)
        return np.eye(n)
    return M


def compute_shift_index(h, dx):
    if dx <= 0:
        return 0
    return int(np.floor(abs(h / dx) + 0.5))


# --- faithful Rust CoupledTtChernoff::step on the dense tensor ---------------

def kron_axis(op, axis, d, n):
    """Lift a single-axis n x n operator to the full n^d tensor (Kronecker)."""
    mats = [np.eye(n)] * d
    mats[axis] = op
    out = np.array([[1.0]])
    for m in mats:
        out = np.kron(out, m)
    return out


def kron_pair(op_j, j, op_k, k, d, n):
    mats = [np.eye(n)] * d
    mats[j] = op_j
    mats[k] = op_k
    out = np.array([[1.0]])
    for m in mats:
        out = np.kron(out, m)
    return out


def coupled_step_matrix(d, n, dx, a, b, c, pairs, tau, scaling):
    """Build the FULL n^d x n^d one-step operator of the Rust scheme.

    scaling in {"current", "fix_dx2", "fix_dx2_x2"}.
    """
    N = n ** d
    # Step 1: diagonal per-axis shift sweep (product over axes, applied in order).
    S = np.eye(N)
    for axis in range(d):
        h = 2.0 * np.sqrt(a[axis] * tau)
        drift = b[axis] * tau
        s_idx = compute_shift_index(h + drift, dx)
        op = shift_3branch(n, s_idx)
        S = kron_axis(op, axis, d, n) @ S
    # Step 2: coupling sweep — additive (I + sum scale * D1_j x D1_k).
    D1u = d1_unnorm(n)
    C = np.eye(N)
    for (j, k, rho) in pairs:
        base = tau * rho * np.sqrt(a[j] * a[k])
        if scaling == "current":
            scale = base
        elif scaling == "fix_dx2":
            scale = base / (dx * dx)
        elif scaling == "fix_dx2_x2":
            scale = 2.0 * base / (dx * dx)
        else:
            raise ValueError(scaling)
        C = C + scale * kron_pair(D1u, j, D1u, k, d, n)
    # Step 3: reaction.
    R = (1.0 + tau * c)
    return R * (C @ S)


# --- analytic correlated-Gaussian semigroup reference -----------------------

def gaussian_density_grid(xs, Sinv, normconst):
    """Evaluate exp(-1/2 x^T Sinv x)*normconst on the tensor product grid xs (per axis)."""
    d = len(xs)
    grids = np.meshgrid(*xs, indexing="ij")
    quad = np.zeros_like(grids[0])
    for i in range(d):
        for j in range(d):
            quad += Sinv[i, j] * grids[i] * grids[j]
    return normconst * np.exp(-0.5 * quad)


def analytic_semigroup(d, xs, S0, Sigma, T):
    """u_T = e^{TL} u0 closed form: S_T = S0 + 2 T Sigma, density renormalised
    so the L1 mass (sum*prod(dx)) is preserved (heat semigroup conserves mass)."""
    S_T = S0 + 2.0 * T * Sigma
    Sinv = np.linalg.inv(S_T)
    return Sinv, S_T


# --- main convergence experiment --------------------------------------------

def run(d=2, n=81, half_width=6.0, T=0.2, rho=0.6,
        a=(0.5, 0.4), b=(0.0, 0.0), c=0.0):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    L = 2.0 * half_width
    dx = L / n                      # periodic grid spacing
    xs_axis = -half_width + dx * np.arange(n)
    xs = [xs_axis.copy() for _ in range(d)]

    # diffusion matrix Sigma (constant): diagonal a, cross 2*rho*sqrt(ai aj) handled
    # in L; for the COVARIANCE growth the matrix is Sigma_ij with Sigma_jk = rho sqrt.
    Sigma = np.diag(a).astype(float)
    pairs = [(0, 1, rho)] if d == 2 else \
        [(j, j + 1, rho) for j in range(d - 1)]
    for (j, k, r) in pairs:
        Sigma[j, k] = r * np.sqrt(a[j] * a[k])
        Sigma[k, j] = Sigma[j, k]

    # IC: narrow isotropic Gaussian (rank-1 separable), well inside the domain.
    s0 = 0.6
    S0 = (s0 ** 2) * np.eye(d)
    S0inv = np.linalg.inv(S0)
    u0 = gaussian_density_grid(xs, S0inv, 1.0)
    mass0 = u0.sum() * (dx ** d)
    u0 = u0 / mass0                 # L1-normalise

    # analytic truth at T
    STinv, S_T = analytic_semigroup(d, xs, S0, Sigma, T)
    uT = gaussian_density_grid(xs, STinv, 1.0)
    massT = uT.sum() * (dx ** d)
    uT = uT / massT                 # heat conserves mass

    N = n ** d
    u0v = u0.reshape(-1)
    uTv = uT.reshape(-1)

    print(f"=== d={d}, n={n}, dx={dx:.4f}, T={T}, rho={rho}, a={a.tolist()} ===")
    print(f"Sigma (diffusion matrix, incl cross):\n{Sigma}")
    print(f"S_T = S0 + 2T*Sigma:\n{S_T}\n")
    header = f"{'scaling':12s} | {'n_steps':>7} | {'tau':>9} | {'rel_L2_err':>12} | {'order':>7}"
    for scaling in ("current", "fix_dx2", "fix_dx2_x2"):
        print(header)
        print("-" * len(header))
        prev_err = None
        prev_tau = None
        for n_steps in (10, 20, 40, 80, 160):
            tau = T / n_steps
            Astep = coupled_step_matrix(d, n, dx, a, b, c, pairs, tau, scaling)
            u = u0v.copy()
            for _ in range(n_steps):
                u = Astep @ u
            err = np.linalg.norm(u - uTv) / np.linalg.norm(uTv)
            order = ""
            if prev_err is not None and err > 0 and prev_err > 0:
                order = f"{np.log(prev_err / err) / np.log(prev_tau / tau):+.3f}"
            print(f"{scaling:12s} | {n_steps:>7} | {tau:>9.5f} | {err:>12.3e} | {order:>7}")
            prev_err, prev_tau = err, tau
        print()


if __name__ == "__main__":
    run(d=2, n=81, half_width=6.0, T=0.2, rho=0.6, a=(0.5, 0.4))
