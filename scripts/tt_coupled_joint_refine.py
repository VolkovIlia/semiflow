#!/usr/bin/env python3
"""tt_coupled_joint_refine.py — JOINT (tau, dx) refinement convergence test for §52.9.

Matrix-FREE faithful replica of the Rust CoupledTtChernoff::step (d=2) so we can
reach fine grids cheaply (no n^d x n^d dense matrix).

Background: fixed-dx tau-refinement (tt_coupled_crossterm_verdict.py) plateaus ~0.31
for ALL cross-term scalings, because the integer shift index round(2 sqrt(a tau)/dx)
collapses to 0 once h < dx/2 — the DIAGONAL diffusion sweep silently becomes the
identity (documented sub-grid floor). So the integer-shift scheme is NOT convergent
under tau-only refinement at fixed dx.

The correct refinement for an integer-shift scheme keeps the integer ratio
    s = round(2 sqrt(a tau)/dx) >= 1
bounded and >= 1 while BOTH tau, dx -> 0, i.e. parabolic joint scaling tau ~ C dx^2.
This script measures the error of the faithful scheme vs the analytic correlated-
Gaussian semigroup under joint refinement, for the three cross-term scalings:

  current   : scale = tau rho sqrt(a_j a_k)
  fix_dx2   : scale = tau rho sqrt(a_j a_k) / (dx_j dx_k)
  fix_dx2_x2: scale = 2 tau rho sqrt(a_j a_k) / (dx_j dx_k)

PROCEED if a corrected scaling converges (error -> 0); BOUNDARY if all plateau.

Run:  python3 scripts/tt_coupled_joint_refine.py
"""
import numpy as np


def compute_shift_index(h, dx):
    if dx <= 0:
        return 0
    return int(np.floor(abs(h / dx) + 0.5))


def shift_1d(vec_axis, s, n):
    """3-branch (1/4,1/4,1/2) periodic shift by s along the LAST axis of a 2D array
    of shape (rows, n).  s=0 -> identity."""
    s = s % n
    if s == 0:
        return vec_axis
    fwd = np.roll(vec_axis, -s, axis=-1)
    bwd = np.roll(vec_axis, +s, axis=-1)
    return 0.25 * fwd + 0.25 * bwd + 0.5 * vec_axis


def d1_unnorm_1d(arr, axis, n):
    """Un-normalised central diff [-1/2,0,1/2] along `axis` (matches Rust apply_d1)."""
    fwd = np.roll(arr, -1, axis=axis)
    bwd = np.roll(arr, +1, axis=axis)
    return 0.5 * fwd - 0.5 * bwd


def coupled_step_2d(u, n, dx, a, rho, tau, scaling):
    """One faithful Rust CoupledTtChernoff step on a 2D array u (n x n).

    1. diagonal per-axis shift sweep (both axes, integer 3-branch shift)
    2. coupling sweep: u <- u + scale * (D1_0 (x) D1_1) u  (additive)
    3. reaction (c=0 -> factor 1)
    """
    # Step 1: per-axis shift.
    for axis in (0, 1):
        h = 2.0 * np.sqrt(a[axis] * tau)
        s = compute_shift_index(h, dx)
        # apply 3-branch shift along `axis`
        ss = s % n
        if ss != 0:
            fwd = np.roll(u, -ss, axis=axis)
            bwd = np.roll(u, +ss, axis=axis)
            u = 0.25 * fwd + 0.25 * bwd + 0.5 * u
    # Step 2: coupling (additive cross operator).
    base = tau * rho * np.sqrt(a[0] * a[1])
    if scaling == "current":
        scale = base
    elif scaling == "fix_dx2":
        scale = base / (dx * dx)
    elif scaling == "fix_dx2_x2":
        scale = 2.0 * base / (dx * dx)
    else:
        raise ValueError(scaling)
    cross = d1_unnorm_1d(d1_unnorm_1d(u, 0, n), 1, n)
    u = u + scale * cross
    return u


def gaussian_2d(xs, Sinv):
    X, Y = np.meshgrid(xs[0], xs[1], indexing="ij")
    q = Sinv[0, 0] * X * X + 2 * Sinv[0, 1] * X * Y + Sinv[1, 1] * Y * Y
    return np.exp(-0.5 * q)


def run(half_width=6.0, T=0.2, rho=0.6, a=(0.5, 0.4)):
    a = np.array(a, dtype=float)
    Sigma = np.array([[a[0], rho * np.sqrt(a[0] * a[1])],
                      [rho * np.sqrt(a[0] * a[1]), a[1]]])
    s0 = 0.6
    S0 = (s0 ** 2) * np.eye(2)

    print(f"=== JOINT (tau,dx) refinement, d=2: T={T}, rho={rho}, a={a.tolist()} ===")
    print("parabolic scaling tau ~ C dx^2 keeps integer shift_idx >= 1\n")
    print("Sigma (diffusion matrix incl cross):")
    print(Sigma)
    print(f"S_T = S0 + 2T*Sigma:\n{S0 + 2*T*Sigma}\n")

    # ladder: keep h/dx ~ 1.35 (shift_idx=1) by tau ~ (dx/(2 sqrt a_max))^2 * 1.35^2
    ladder = [81, 121, 161, 241, 321, 481, 641]
    results = {sc: [] for sc in ("current", "fix_dx2", "fix_dx2_x2")}
    dxs = []
    for scaling in ("current", "fix_dx2", "fix_dx2_x2"):
        print(f"{'scaling':12s} | {'n':>4} | {'nsteps':>6} | {'dx':>7} | {'h/dx':>6} | "
              f"{'s':>2} | {'rel_L2_err':>12} | {'order(dx)':>9}")
        print("-" * 88)
        prev_err = prev_dx = None
        for n in ladder:
            L = 2.0 * half_width
            dx = L / n
            xs = [(-half_width + dx * np.arange(n)) for _ in range(2)]
            # choose n_steps so h/dx ~ 1.35 (shift_idx rounds to 1), parabolic.
            target_ratio = 1.35
            tau = (target_ratio * dx / (2.0 * np.sqrt(a.max()))) ** 2
            n_steps = max(int(round(T / tau)), 1)
            tau = T / n_steps
            h = 2.0 * np.sqrt(a.max() * tau)
            s = compute_shift_index(h, dx)
            u0 = gaussian_2d(xs, np.linalg.inv(S0))
            u0 = u0 / (u0.sum() * dx * dx)
            uT = gaussian_2d(xs, np.linalg.inv(S0 + 2 * T * Sigma))
            uT = uT / (uT.sum() * dx * dx)
            u = u0.copy()
            for _ in range(n_steps):
                u = coupled_step_2d(u, n, dx, a, rho, tau, scaling)
            err = np.linalg.norm(u - uT) / np.linalg.norm(uT)
            order = ""
            if prev_err is not None and err > 0 < prev_err:
                order = f"{np.log(prev_err / err) / np.log(prev_dx / dx):+.3f}"
            print(f"{scaling:12s} | {n:>4} | {n_steps:>6} | {dx:>7.4f} | {h/dx:>6.3f} | "
                  f"{s:>2} | {err:>12.3e} | {order:>9}")
            results[scaling].append(err)
            if scaling == "current":
                dxs.append(dx)
            prev_err, prev_dx = err, dx
        print()

    print("SUMMARY (does error -> 0 under joint refinement?):")
    for sc in ("current", "fix_dx2", "fix_dx2_x2"):
        errs = results[sc]
        verdict = "CONVERGES" if errs[-1] < 0.5 * errs[0] and errs[-1] < 5e-2 else "PLATEAUS"
        print(f"  {sc:12s}: first={errs[0]:.3e} last={errs[-1]:.3e}  -> {verdict}")


if __name__ == "__main__":
    run()
