#!/usr/bin/env python3
"""tt_coupled_baseline_check.py — control experiments isolating WHERE the floor is.

Three controls under joint parabolic (tau ~ dx^2, shift_idx=1) refinement, d=2:

  (1) SEPARABLE baseline (rho=0): diagonal integer-shift sweep ONLY, vs the
      uncoupled analytic semigroup S_T = S0 + 2T diag(a).  Does the integer-shift
      diagonal scheme itself converge? If it PLATEAUS, the floor is the integer-shift
      quantization (shift_idx=1 is a fixed, non-vanishing displacement ~dx) and the
      coupling question is moot — the whole scheme has an O(1) floor.

  (2) The SAME diagonal scheme but with a FRACTIONAL (linear-interpolated) shift
      so h is applied exactly (not rounded to integer grid steps). Does THIS converge?
      This isolates the integer-rounding as the culprit.

  (3) A reference EXPLICIT-EULER coupled scheme with 1/dx^2-normalised D2 + cross
      (the tt_coupled_evolver_probe.py operator) vs the analytic coupled semigroup,
      to confirm the analytic truth + the CONTINUOUS generator are correct (this MUST
      converge — it is a standard consistent FD scheme).

Run:  python3 scripts/tt_coupled_baseline_check.py
"""
import numpy as np


def compute_shift_index(h, dx):
    if dx <= 0:
        return 0
    return int(np.floor(abs(h / dx) + 0.5))


def gaussian_2d(xs, Sinv):
    X, Y = np.meshgrid(xs[0], xs[1], indexing="ij")
    q = Sinv[0, 0] * X * X + 2 * Sinv[0, 1] * X * Y + Sinv[1, 1] * Y * Y
    return np.exp(-0.5 * q)


def shift_int(u, axis, s, n):
    s = s % n
    if s == 0:
        return u
    fwd = np.roll(u, -s, axis=axis)
    bwd = np.roll(u, +s, axis=axis)
    return 0.25 * fwd + 0.25 * bwd + 0.5 * u


def shift_frac(u, axis, h, dx, n):
    """Fractional 3-branch shift: shift by the EXACT real displacement h/dx grid
    units via linear interpolation (continuous shift, no integer rounding)."""
    g = h / dx
    s0 = int(np.floor(g))
    frac = g - s0
    def lin_roll(arr, shift_real):
        s_lo = int(np.floor(shift_real))
        f = shift_real - s_lo
        lo = np.roll(arr, -s_lo, axis=axis)
        hi = np.roll(arr, -(s_lo + 1), axis=axis)
        return (1 - f) * lo + f * hi
    fwd = lin_roll(u, g)
    bwd = lin_roll(u, -g)
    return 0.25 * fwd + 0.25 * bwd + 0.5 * u


def run():
    a = np.array([0.5, 0.4])
    rho = 0.6
    T, half_width = 0.2, 6.0
    s0 = 0.6
    S0 = (s0 ** 2) * np.eye(2)
    Sigma_diag = np.diag(a)
    Sigma_coup = np.array([[a[0], rho * np.sqrt(a[0] * a[1])],
                           [rho * np.sqrt(a[0] * a[1]), a[1]]])
    ladder = [81, 121, 161, 241, 321, 481]

    def joint_tau(dx):
        tau = (1.35 * dx / (2 * np.sqrt(a.max()))) ** 2
        n_steps = max(int(round(T / tau)), 1)
        return T / n_steps, n_steps

    print("=== CONTROL (1): SEPARABLE integer-shift diagonal sweep, rho=0 ===")
    print("vs uncoupled analytic S_T = S0 + 2T diag(a). joint parabolic refine.\n")
    prev = None
    for n in ladder:
        dx = 2 * half_width / n
        xs = [(-half_width + dx * np.arange(n)) for _ in range(2)]
        tau, ns = joint_tau(dx)
        u0 = gaussian_2d(xs, np.linalg.inv(S0)); u0 /= u0.sum() * dx * dx
        uT = gaussian_2d(xs, np.linalg.inv(S0 + 2 * T * Sigma_diag)); uT /= uT.sum() * dx * dx
        u = u0.copy()
        for _ in range(ns):
            for axis in (0, 1):
                h = 2 * np.sqrt(a[axis] * tau)
                u = shift_int(u, axis, compute_shift_index(h, dx), n)
        err = np.linalg.norm(u - uT) / np.linalg.norm(uT)
        order = "" if prev is None else f"{np.log(prev[0]/err)/np.log(prev[1]/dx):+.3f}"
        print(f"  n={n:>4} dx={dx:.4f} ns={ns:>4}  rel_err={err:.3e}  order={order}")
        prev = (err, dx)

    print("\n=== CONTROL (2): SEPARABLE FRACTIONAL-shift diagonal sweep, rho=0 ===")
    print("exact real displacement h (linear interp), no integer rounding.\n")
    prev = None
    for n in ladder:
        dx = 2 * half_width / n
        xs = [(-half_width + dx * np.arange(n)) for _ in range(2)]
        tau, ns = joint_tau(dx)
        u0 = gaussian_2d(xs, np.linalg.inv(S0)); u0 /= u0.sum() * dx * dx
        uT = gaussian_2d(xs, np.linalg.inv(S0 + 2 * T * Sigma_diag)); uT /= uT.sum() * dx * dx
        u = u0.copy()
        for _ in range(ns):
            for axis in (0, 1):
                h = 2 * np.sqrt(a[axis] * tau)
                u = shift_frac(u, axis, h, dx, n)
        err = np.linalg.norm(u - uT) / np.linalg.norm(uT)
        order = "" if prev is None else f"{np.log(prev[0]/err)/np.log(prev[1]/dx):+.3f}"
        print(f"  n={n:>4} dx={dx:.4f} ns={ns:>4}  rel_err={err:.3e}  order={order}")
        prev = (err, dx)

    print("\n=== CONTROL (3): EXPLICIT-EULER consistent coupled FD (1/dx^2 D2 + cross) ===")
    print("vs coupled analytic S_T = S0 + 2T Sigma_coup. confirms truth+generator.\n")
    prev = None
    for n in [81, 121, 161, 241]:
        dx = 2 * half_width / n
        xs = [(-half_width + dx * np.arange(n)) for _ in range(2)]
        tau = 0.20 * dx ** 2 / a.max()          # CFL-stable explicit
        ns = max(int(round(T / tau)), 1); tau = T / ns
        u0 = gaussian_2d(xs, np.linalg.inv(S0)); u0 /= u0.sum() * dx * dx
        uT = gaussian_2d(xs, np.linalg.inv(S0 + 2 * T * Sigma_coup)); uT /= uT.sum() * dx * dx
        u = u0.copy()
        def d2(arr, ax):
            return (np.roll(arr, -1, ax) - 2 * arr + np.roll(arr, +1, ax)) / dx ** 2
        def d1(arr, ax):
            return (np.roll(arr, -1, ax) - np.roll(arr, +1, ax)) / (2 * dx)
        cross = rho * np.sqrt(a[0] * a[1])
        for _ in range(ns):
            Lu = a[0] * d2(u, 0) + a[1] * d2(u, 1) + 2 * cross * d1(d1(u, 0), 1)
            u = u + tau * Lu
        err = np.linalg.norm(u - uT) / np.linalg.norm(uT)
        order = "" if prev is None else f"{np.log(prev[0]/err)/np.log(prev[1]/dx):+.3f}"
        print(f"  n={n:>4} dx={dx:.4f} ns={ns:>5}  rel_err={err:.3e}  order={order}")
        prev = (err, dx)


if __name__ == "__main__":
    run()
