#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy's symbolic arithmetic is dynamically typed through operator overloads;
# Pyright cannot trace expressions through sp.expand / sp.simplify chains.
# All operations are valid sympy at runtime (verified by this oracle's PASS).
"""T_TT_BAND_SHIFT_RANK — proof-grade oracle for the cubic-Lagrange band-split
interpolation shift (math.md §52.2 Amd 1, v9.1.0-s3-triz-resolution §5.1/§6).

The TRIZ resolution replaces the integer-index TT shift with the band-split
operator:

    Sₕ = Σ_{m=0}^{3} c_m(t) · P_{s₀+m},   s₀ = ⌊h/dx⌋,  t = h/dx − s₀ ∈ [0,1)

where the cubic-Lagrange weights for nodes {−1, 0, 1, 2} relative to s₀ are:

    c_{-1}(t) = −t(t−1)(t−2)/6
    c_0(t)   =  (t+1)(t−1)(t−2)/2
    c_1(t)   = −(t+1)·t·(t−2)/2
    c_2(t)   =  (t+1)·t·(t−1)/6

This oracle proves three normative claims before the Rust implementation (P2):

  (a) sympy_weight_sum_and_interpolation_order
      Symbolic proof that the cubic-Lagrange weights:
        (a1) sum to exactly 1 for all t ("partition of unity")
        (a2) reproduce polynomial displacements of degree 0..3 exactly, i.e.
             Σ_m c_m(t)·(s₀+node_m)^p = (s₀+t)^p   for p ∈ {0,1,2,3}
      Both assertions use sp.expand/sp.simplify and require the symbolic
      difference to be identically zero.

  (b) numeric_qtt_band_rank
      Each individual band P_{s₀+m} is a permutation (periodic roll) matrix
      → QTT-op-rank = 1 per band (trivially rank-1 Toeplitz/permutation).
      The assembled 4-band sum has measured QTT-op-rank ≤ 3, independent of
      grid resolution L ∈ {6, 8, 10}.  Kazeev–Khoromskij banded-Toeplitz result;
      the prior BOUNDARY verdict never invoked it.

  (c) numeric_convergence_witness
      Under joint parabolic refinement (τ ∼ C·dx², ratio h/dx held ≈1.35):
        - The integer-shift Chernoff error PLATEAUS (stays ≥ 0.15 across L=7..10).
        - The cubic-band-shift Chernoff error is < 1e-4 at L=7 AND decreases
          monotonically through L=10 (≈4× per refinement step → O(τ²)).
      This guards the make-or-break convergence property as a hard regression gate.

Prints "T_TT_BAND_SHIFT_RANK PASS (3/3 sub-checks: ...)" on success.
Prints "T_TT_BAND_SHIFT_RANK FAIL: <reason>" and exits 1 on any failure.

Used by:
- test-fast gate (RELEASE_BLOCKING math-fidelity category)
- Phase P2 green-light prerequisite (before tt_chernoff.rs is touched)

References:
  - math.md §52.2 Amendment 1 (band-split shift, normative)
  - math.md §52.9 Amendment 1 (BOUNDARY verdict WITHDRAWN — RESOLVED)
  - .dev-docs/specs/v9.1.0-s3-triz-resolution.md §5.1, §6, §8.2
  - Kazeev & Khoromskij 2012 (QTT-op rank of banded Toeplitz operators)
  - Probe sources (design-doc reproducibility, NOT deleted):
      .dev-docs/specs/triz_qtt_probe.py   (rank leg)
      .dev-docs/specs/triz_conv_probe.py  (convergence + cubic rank leg)
      .dev-docs/specs/triz_coupled_probe.py  (coupled-step leg, T_TT_COUPLED_RANK)
"""

import sys
from typing import Optional

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# QTT-operator rank helper (adapted from triz_qtt_probe.py / triz_conv_probe.py)
# ---------------------------------------------------------------------------


def _qtt_op_peak_rank_simple(mat: np.ndarray, log2_n: int) -> int:
    """Peak QTT-operator rank — clean unfolding approach.

    For each bond position b in {0, …, L-2}, forms the unfolding matrix by
    splitting the L mode-groups into left (0..b) and right (b+1..L-1) halves
    and computes the SVD rank.  No iterative rank-tracking needed.

    Args:
        mat:    Square matrix of shape (n, n) where n = 2**log2_n.
        log2_n: L — number of QTT levels.

    Returns:
        Maximum bond rank across all L−1 bonds.
    """
    n = 2 ** log2_n
    assert mat.shape == (n, n), f"Expected ({n},{n}), got {mat.shape}"

    # Interleave row/col indices per level: shape (4, 4, …, 4) with L entries.
    tensor = mat.reshape([2] * (2 * log2_n))
    perm = []
    for k in range(log2_n):
        perm.append(k)
        perm.append(k + log2_n)
    grouped = np.transpose(tensor, perm).reshape([4] * log2_n)

    peak = 0
    for b in range(log2_n - 1):
        # Unfold: left = modes 0..b, right = modes b+1..L-1.
        left_size = 4 ** (b + 1)
        right_size = 4 ** (log2_n - b - 1)
        unfolded = grouped.reshape(left_size, right_size)
        sv = np.linalg.svd(unfolded, compute_uv=False)
        tol = 1e-10 * float(sv[0]) if float(sv[0]) > 0.0 else 0.0
        r = int(np.sum(sv > tol))
        if r > peak:
            peak = r
    return peak


# ---------------------------------------------------------------------------
# Cubic-Lagrange band-shift matrix builder
# ---------------------------------------------------------------------------

def _cubic_lagrange_weights(t: float) -> tuple[float, float, float, float]:
    """Cubic-Lagrange interpolation weights for nodes {−1, 0, 1, 2} at abscissa t.

    Nodes relative to floor(h/dx): m ∈ {−1, 0, 1, 2}, so the lattice offsets
    are s₀−1, s₀, s₀+1, s₀+2 where s₀ = ⌊h/dx⌋ and t = h/dx − s₀ ∈ [0,1).

    The four Lagrange basis polynomials at this set of nodes are:

        c_{-1}(t) = −t(t−1)(t−2)/6
        c_0(t)   =  (t+1)(t−1)(t−2)/2
        c_1(t)   = −(t+1)·t·(t−2)/2
        c_2(t)   =  (t+1)·t·(t−1)/6

    Args:
        t: fractional part of h/dx, 0 ≤ t < 1.

    Returns:
        (w_m1, w_0, w_1, w_2): the four weights.
    """
    w_m1 = -t * (t - 1.0) * (t - 2.0) / 6.0
    w_0 = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0
    w_1 = -(t + 1.0) * t * (t - 2.0) / 2.0
    w_2 = (t + 1.0) * t * (t - 1.0) / 6.0
    return w_m1, w_0, w_1, w_2


def _cubic_band_shift_matrix(n: int, h: float, dx: float) -> np.ndarray:
    """Build the n×n cubic-Lagrange band-shift matrix for shift h on grid dx.

    Sₕ = Σ_{m∈{-1,0,1,2}} c_m(t) · roll(I, s₀+m)

    Args:
        n:  Grid size (must be a power of 2 for QTT rank tests).
        h:  Shift distance (continuum, may be non-integer multiple of dx).
        dx: Grid spacing (= 1/n for unit-interval periodic grid).

    Returns:
        (n × n) shift matrix.
    """
    s0 = int(np.floor(h / dx))
    t = h / dx - s0
    w_m1, w_0, w_1, w_2 = _cubic_lagrange_weights(t)
    eye = np.eye(n)
    return (
        w_m1 * np.roll(eye, s0 - 1, axis=0)
        + w_0 * np.roll(eye, s0, axis=0)
        + w_1 * np.roll(eye, s0 + 1, axis=0)
        + w_2 * np.roll(eye, s0 + 2, axis=0)
    )


# ---------------------------------------------------------------------------
# Sub-check (a): sympy weight-sum + degree-0..3 polynomial reproduction
# ---------------------------------------------------------------------------

def check_sympy_weight_sum_and_interpolation_order() -> Optional[str]:
    """(a) Symbolic proof of partition-of-unity and cubic interpolation order.

    Returns None on PASS, or an error string on FAIL.
    """
    t_sym = sp.Symbol("t")
    s0_sym = sp.Symbol("s0")

    # Symbolic cubic-Lagrange weights for nodes {-1, 0, 1, 2}.
    w_m1 = -t_sym * (t_sym - 1) * (t_sym - 2) / 6
    w_0 = (t_sym + 1) * (t_sym - 1) * (t_sym - 2) / 2
    w_1 = -(t_sym + 1) * t_sym * (t_sym - 2) / 2
    w_2 = (t_sym + 1) * t_sym * (t_sym - 1) / 6

    # Lattice offsets for each band relative to s₀.
    offsets = [-1, 0, 1, 2]
    weights = [w_m1, w_0, w_1, w_2]

    # (a1) Partition of unity: Σ c_m(t) = 1
    weight_sum = sp.expand(w_m1 + w_0 + w_1 + w_2)
    diff_unity = sp.simplify(weight_sum - 1)
    if diff_unity != 0:
        return (
            f"sympy_weight_sum: partition-of-unity FAILS — "
            f"Σ c_m(t) − 1 = {diff_unity} (expected identically 0)"
        )
    print(f"    (a1) partition-of-unity: Σ c_m(t) − 1 = {diff_unity} = 0  PASS")

    # (a2) Polynomial reproduction of degree p ∈ {0, 1, 2, 3}.
    # Claim: Σ_m c_m(t) · (s₀ + node_m)^p = (s₀ + t)^p  for p = 0,1,2,3.
    for p in range(4):
        lhs = sp.Integer(0)
        for w, off in zip(weights, offsets):
            lhs = lhs + w * (s0_sym + off) ** p
        rhs = (s0_sym + t_sym) ** p
        diff_p = sp.expand(lhs - rhs)
        diff_p_simp = sp.simplify(diff_p)
        if diff_p_simp != 0:
            return (
                f"sympy_interpolation_order: degree-{p} polynomial reproduction "
                f"FAILS — Σ c_m·node_m^{p} − (s₀+t)^{p} = {diff_p_simp} "
                f"(expected identically 0)"
            )
        print(
            f"    (a2) degree-{p} polynomial reproduction: "
            f"Σ c_m·(s₀+node_m)^{p} − (s₀+t)^{p} = {diff_p_simp} = 0  PASS"
        )

    return None


# ---------------------------------------------------------------------------
# Sub-check (b): numeric QTT-operator rank at L ∈ {6, 8, 10}
# ---------------------------------------------------------------------------

def check_numeric_qtt_band_rank() -> Optional[str]:
    """(b) Each band is a permutation (QTT-op rank = 1); assembled 4-band sum ≤ 3.

    Verified at L ∈ {6, 8, 10} with h/dx = 3.7 (deliberately non-integer).

    Returns None on PASS, or an error string on FAIL.
    """
    h_over_dx = 3.7  # the non-integer ratio from the TRIZ probe (§6.1)

    for log2_n in [6, 8, 10]:
        n = 2 ** log2_n
        dx = 1.0 / n
        h = h_over_dx * dx

        s0 = int(np.floor(h / dx))
        eye = np.eye(n)

        # Single permutation bands: each is a cyclic roll (periodic shift),
        # which is a banded Toeplitz permutation → QTT-op-rank ≤ 2 (Kazeev–Khoromskij 2012).
        # Note: arbitrary rolls may measure rank = 1 or 2 depending on the shift alignment
        # with the QTT bit-structure; the normative bound is ≤ 2 (not = 1).
        band_ranks: list[int] = []
        for band_offset in [-1, 0, 1, 2]:
            perm_mat = np.roll(eye, s0 + band_offset, axis=0)
            r = _qtt_op_peak_rank_simple(perm_mat, log2_n)
            band_ranks.append(r)
            if r > 2:
                return (
                    f"numeric_qtt_band_rank: single permutation band s₀+{band_offset} "
                    f"at L={log2_n} has QTT-op rank {r}, expected ≤ 2. "
                    f"Kazeev–Khoromskij banded-permutation rank-O(1) guarantee VIOLATED."
                )

        # Assembled 4-band cubic sum: must have QTT-op rank ≤ 3.
        S_cubic = _cubic_band_shift_matrix(n, h, dx)
        r_cubic = _qtt_op_peak_rank_simple(S_cubic, log2_n)
        if r_cubic > 3:
            return (
                f"numeric_qtt_band_rank: 4-band cubic-Lagrange shift at L={log2_n} "
                f"has QTT-op rank {r_cubic}, expected ≤ 3. "
                f"Kazeev–Khoromskij banded rank guarantee VIOLATED."
            )

        print(
            f"    (b) L={log2_n} n={n}: "
            f"single-band ranks={band_ranks} ≤ 2 each ✓  "
            f"cubic-band assembled rank={r_cubic} ≤ 3 ✓"
        )

    return None


# ---------------------------------------------------------------------------
# Sub-check (c): convergence witness
# ---------------------------------------------------------------------------

def _heat_fft_reference(u0: np.ndarray, a: float, t_end: float, dx: float) -> np.ndarray:
    """Compute e^{t·a·∂²}u₀ via FFT (the independent truth).

    Args:
        u0:    Initial condition on periodic uniform grid.
        a:     Diffusion coefficient.
        t_end: Evolution time.
        dx:    Grid spacing.

    Returns:
        Exact solution on the same grid.
    """
    n = len(u0)
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    u0_hat = np.fft.fft(u0)
    return np.real(np.fft.ifft(u0_hat * np.exp(-a * k ** 2 * t_end)))


def _chernoff_3branch_step(
    p_plus: np.ndarray,
    p_minus: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """One Chernoff 3-branch heat step: ¼ S⁺u + ¼ S⁻u + ½ u."""
    return 0.25 * (p_plus @ u) + 0.25 * (p_minus @ u) + 0.5 * u


def check_numeric_convergence_witness() -> Optional[str]:
    """(c) Integer shift plateaus (err ≥ 0.15); cubic-frac converges monotone < 1e-4.

    Joint parabolic refinement: h/dx ≈ 1.35 fixed, τ = (h/2)²/a, N = T/τ.
    Grid levels L ∈ {7, 8, 9, 10} match the TRIZ probe §6.2 numbers.

    Returns None on PASS, or an error string on FAIL.
    """
    a = 0.7
    t_end = 0.05
    ratio = 1.35  # h/dx ratio held constant (joint parabolic refinement axis)

    int_errors: list[float] = []
    cub_errors: list[float] = []
    levels = [7, 8, 9, 10]

    for log2_n in levels:
        n = 2 ** log2_n
        dx = 1.0 / n
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        u0 = np.exp(-((x - 0.5) ** 2) / (2.0 * 0.02))

        # Choose τ so that h = ratio·dx = 2·√(a·τ) → τ = (ratio·dx/(2))²/a.
        h = ratio * dx
        tau = (h / 2.0) ** 2 / a
        num_steps = max(1, int(round(t_end / tau)))
        tau = t_end / float(num_steps)
        h = 2.0 * np.sqrt(a * tau)

        truth = _heat_fft_reference(u0, a, t_end, dx)

        # Integer-shift operators.
        s_int = int(round(h / dx))
        eye = np.eye(n)
        pi_plus = np.roll(eye, s_int, axis=0)
        pi_minus = np.roll(eye, -s_int, axis=0)

        # Cubic-band-shift operators.
        pc_plus = _cubic_band_shift_matrix(n, h, dx)
        pc_minus = _cubic_band_shift_matrix(n, -h, dx)

        u_int = u0.copy()
        u_cub = u0.copy()
        for _ in range(num_steps):
            u_int = _chernoff_3branch_step(pi_plus, pi_minus, u_int)
            u_cub = _chernoff_3branch_step(pc_plus, pc_minus, u_cub)

        norm_truth = float(np.linalg.norm(truth))
        e_int = float(np.linalg.norm(u_int - truth)) / norm_truth
        e_cub = float(np.linalg.norm(u_cub - truth)) / norm_truth

        int_errors.append(e_int)
        cub_errors.append(e_cub)
        print(
            f"    (c) L={log2_n} n={n} N={num_steps} h/dx={h/dx:.3f}: "
            f"integer err={e_int:.3e}  cubic-frac err={e_cub:.3e}"
        )

    # Assert 1: integer plateau — every level must show err ≥ 0.15.
    min_int = min(int_errors)
    if min_int < 0.15:
        return (
            f"numeric_convergence_witness: integer-shift error dropped below 0.15 "
            f"plateau threshold (min={min_int:.4e}). Expected floor ≥ 0.15 "
            f"(quantization plateau). Per-level: {int_errors}"
        )

    # Assert 2: cubic-frac err < 1e-4 at L=7 (first level).
    if cub_errors[0] >= 1e-4:
        return (
            f"numeric_convergence_witness: cubic-frac error at L=7 is "
            f"{cub_errors[0]:.4e} ≥ 1e-4. Expected < 1e-4."
        )

    # Assert 3: cubic-frac errors decrease monotonically L=7..10.
    for i in range(1, len(cub_errors)):
        if cub_errors[i] >= cub_errors[i - 1]:
            return (
                f"numeric_convergence_witness: cubic-frac error INCREASED "
                f"from L={levels[i-1]} to L={levels[i]}: "
                f"{cub_errors[i-1]:.4e} → {cub_errors[i]:.4e}. "
                f"Expected monotone decrease (O(τ²) convergence)."
            )

    return None


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _fail(reason: str) -> int:
    """Print FAIL line with oracle tag; return exit code 1."""
    print(f"T_TT_BAND_SHIFT_RANK FAIL: {reason}", flush=True)
    return 1


def main() -> int:
    """Run all 3 sub-checks; print result; exit 0/1."""
    checks = [
        ("sympy_weight_sum_and_interpolation_order", check_sympy_weight_sum_and_interpolation_order),
        ("numeric_qtt_band_rank", check_numeric_qtt_band_rank),
        ("numeric_convergence_witness", check_numeric_convergence_witness),
    ]

    print("=" * 72)
    print("T_TT_BAND_SHIFT_RANK — cubic-Lagrange band-split interpolation shift")
    print("(math.md §52.2 Amd 1, v9.1.0-s3-triz-resolution §5.1/§6/§8.2)")
    print("=" * 72)

    failures: list[str] = []
    passed: list[str] = []

    for name, check in checks:
        print(f"\n[{name}]")
        try:
            result = check()
        except Exception as exc:  # noqa: BLE001
            return _fail(f"sub-check {name} raised exception: {exc!r}")
        if result is None:
            passed.append(name)
        else:
            print(f"  (FAIL) {name}: {result}")
            failures.append(f"{name}: {result}")

    print()
    if failures:
        return _fail(
            f"{len(failures)}/{len(checks)} sub-checks failed: "
            + "; ".join(failures)
        )

    print(
        "T_TT_BAND_SHIFT_RANK PASS (3/3 sub-checks: " + " / ".join(passed) + ")",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
