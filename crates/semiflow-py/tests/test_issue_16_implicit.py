"""Tests for issue #16 — path="implicit" (PCG backward-Euler) on SymmetricOperator.

Test structure
--------------
Fast (default run):
  test_implicit_path_accepted           — path param wires through (no error)
  test_implicit_path_bad_string_raises  — path='bogus' raises SemiflowError (Unsupported)
  test_implicit_well_conditioned_cv     — agrees with chebyshev + scipy dense ≤ 5e-7
  test_implicit_stiff_neumann_accuracy  — stiff Neumann surviving mode ≤ 1e-6
  test_mass_lumped_implicit_accepted    — mass_lumped_evolve accepts path='implicit'

Slow (marked with @pytest.mark.slow; run via -m slow):
  test_implicit_stiff_explicit_timeout_neumann  — chebyshev times out
  test_implicit_stiff_scipy_timeout_neumann     — scipy expm_multiply times out

Fixed (§59.6):
  When the input vector lies in the null space of A (e.g. v = ones for a
  Neumann Laplacian), pcg_shifted now checks ‖r‖ ≤ tol before entering the CG
  loop and returns Ok(0) immediately — the warm-start x already solves the system.
  Regression gate: crates/semiflow/tests/pcg_null_space_guard.rs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import scipy.linalg
    import scipy.sparse

    HAS_SCIPY = True
except ImportError:
    # Provide a bound name so Pyright does not report 'possibly-unbound'
    # in code paths guarded by HAS_SCIPY.  Tests that use scipy are always
    # decorated with @pytest.mark.skipif(not HAS_SCIPY, ...) and will be
    # skipped at runtime when the real module is absent.
    import types as _t
    scipy = _t.SimpleNamespace(  # type: ignore[assignment]
        linalg=_t.SimpleNamespace(),
        sparse=_t.SimpleNamespace(),
    )
    HAS_SCIPY = False

import semiflow

# ---------------------------------------------------------------------------
# CSR helpers
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent.parent.parent.parent / "benchmarks" / "issue16"


def _neumann_csr(n: int, scale: float = 1.0):
    """Singular Neumann Laplacian: row sums = 0, constant vector ∈ ker(A)."""
    indptr = np.zeros(n + 1, dtype=np.int64)
    idx: list[int] = []
    dat: list[float] = []
    for i in range(n):
        is_bd = i == 0 or i == n - 1
        diag = scale if is_bd else 2.0 * scale
        if i > 0:
            idx.append(i - 1)
            dat.append(-scale)
        idx.append(i)
        dat.append(diag)
        if i < n - 1:
            idx.append(i + 1)
            dat.append(-scale)
        indptr[i + 1] = len(idx)
    return indptr, np.array(idx, dtype=np.int32), np.array(dat)


def _make_op(n: int, scale: float = 1.0) -> semiflow.SymmetricOperator:
    """Build SymmetricOperator for the 1-D Neumann Laplacian."""
    indptr, indices, data = _neumann_csr(n, scale)
    sym_tol = 1e-6 if scale >= 1e6 else 1e-10
    return semiflow.SymmetricOperator.from_csr(indptr, indices, data, n, sym_tol)


# ---------------------------------------------------------------------------
# Fast tests
# ---------------------------------------------------------------------------


def test_implicit_path_accepted():
    """path='implicit' is a recognised string and does not raise ValueError."""
    op = _make_op(n=20)
    v = np.linspace(0.1, 1.0, 20).reshape(20, 1)
    out = op.evolve_batched(t=0.001, v_nc=v, path="implicit", n_steps=10)
    assert out.shape == (20, 1)
    assert np.all(np.isfinite(out))


def test_implicit_path_bad_string_raises():
    """path='bogus' must raise SemiflowError (Unsupported)."""
    op = _make_op(n=8)
    v = np.ones((8, 1))
    with pytest.raises(semiflow.SemiflowError):
        op.evolve_batched(t=0.01, v_nc=v, path="bogus")


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_implicit_well_conditioned_cv():
    """Non-stiff N=200 Neumann: implicit ≈ chebyshev ≈ scipy.linalg.expm (dense).

    Tolerances:
      implicit vs chebyshev  ≤ 5e-7   (BE order-1 with 500 steps at t=0.01)
      implicit vs scipy dense ≤ 5e-7
      chebyshev vs scipy dense ≤ 1e-9  (high-accuracy explicit path)
    """
    n = 200
    t = 0.01
    op = _make_op(n=n, scale=1.0)

    # Non-constant initial condition — avoids the zero-residual null-space case
    v = np.sin(np.linspace(0, np.pi, n))

    out_implicit = op.evolve_batched(
        t=t, v_nc=v.reshape(n, 1), path="implicit", n_steps=500
    ).ravel()
    out_cheby = op.evolve_batched(
        t=t, v_nc=v.reshape(n, 1), path="chebyshev"
    ).ravel()

    # Dense scipy reference
    indptr, indices, data = _neumann_csr(n, scale=1.0)
    A_sp = scipy.sparse.csr_matrix(
        (data, indices, indptr.astype(np.int32)), shape=(n, n)
    )
    ref = scipy.linalg.expm(-t * A_sp.toarray()) @ v

    err_impl_cheby = np.max(np.abs(out_implicit - out_cheby))
    err_impl_scipy = np.max(np.abs(out_implicit - ref))
    err_cheby_scipy = np.max(np.abs(out_cheby - ref))

    assert err_impl_cheby <= 5e-7, (
        f"implicit vs chebyshev: {err_impl_cheby:.2e} > 5e-7"
    )
    assert err_impl_scipy <= 5e-7, (
        f"implicit vs scipy dense: {err_impl_scipy:.2e} > 5e-7"
    )
    assert err_cheby_scipy <= 1e-9, (
        f"chebyshev vs scipy dense: {err_cheby_scipy:.2e} > 1e-9"
    )


def test_implicit_stiff_neumann_accuracy():
    """Stiff Neumann ×1e7 t=1: surviving mode oracle mean(v)·1, sup_error ≤ 1e-6.

    This is the PRIMARY accuracy assertion for issue #16.
    The non-constant v = linspace ensures the oracle is non-trivial (mean≈0.50).
    """
    n = 400
    op = _make_op(n=n, scale=1e7)
    v = np.linspace(1.0 / n, 1.0, n).reshape(n, 1)
    oracle = np.full((n, 1), v.mean())

    out = op.evolve_batched(t=1.0, v_nc=v, path="implicit", n_steps=100)
    sup_err = float(np.max(np.abs(out - oracle)))

    assert oracle[0, 0] > 0.1, "non-vacuity: oracle must be O(1)"
    assert sup_err <= 1e-6, (
        f"stiff Neumann sup_error={sup_err:.2e} > 1e-6 vs "
        f"surviving-mode oracle (mean≈{oracle[0,0]:.4f})"
    )


def test_mass_lumped_implicit_accepted():
    """mass_lumped_evolve also accepts path='implicit'."""
    n = 30
    indptr, indices, data = _neumann_csr(n)
    op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
    m_diag = np.ones(n) * 1.5
    v = np.linspace(0.1, 1.0, n).reshape(n, 1)
    out = semiflow.mass_lumped_evolve(
        op, m_diag, t=0.001, v_nc=v, path="implicit", n_steps=10
    )
    assert out.shape == (n, 1)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Slow / subprocess timeout tests
# ---------------------------------------------------------------------------


def _worker_path() -> str:
    return str(BENCH_DIR / "worker_semiflow.py")


def _run_in_subprocess(args: list[str], timeout_s: float) -> tuple[bool, str]:
    """Return (timed_out, stderr) for a subprocess call."""
    try:
        proc = subprocess.run(
            [sys.executable] + args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return False, proc.stderr
    except subprocess.TimeoutExpired:
        return True, ""


@pytest.mark.slow
def test_implicit_stiff_explicit_timeout_neumann():
    """chebyshev times out on stiff Neumann (N=400, scale=1e7, t=1.0).

    Confirms the capability claim: explicit path is intractable where
    implicit succeeds.  Runs as a subprocess with a 10 s hard timeout;
    on benchmarked hardware chebyshev takes ~14 s (3500× slower than
    implicit).  The Rust gate g_symop_implicit_stiff provides
    hardware-independent proof via matvec counts (12 924× fewer).
    """
    worker = _worker_path()
    if not Path(worker).exists():
        pytest.skip("worker_semiflow.py not found in benchmarks/issue16/")
    timed_out, _ = _run_in_subprocess(
        [worker, "neumann", "400", "1e7", "1.0", "chebyshev"], timeout_s=10
    )
    assert timed_out, (
        "Expected chebyshev to TIME OUT within 10 s on stiff N=400×1e7 "
        "(benchmarked at ~14 s on i7-12700K). "
        "If hardware is ≥10× faster this gate needs re-calibration."
    )


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_implicit_stiff_scipy_timeout_neumann():
    """scipy.sparse.linalg.expm_multiply times out on stiff Neumann.

    Confirms that scipy is also intractable in this regime.
    Uses a 90 s timeout — scipy did not return within 90 s on benchmarked
    hardware.
    """
    worker = str(BENCH_DIR / "worker_scipy.py")
    if not Path(worker).exists():
        pytest.skip("worker_scipy.py not found in benchmarks/issue16/")
    timed_out, _ = _run_in_subprocess(
        [worker, "neumann", "400", "1e7", "1.0"], timeout_s=90
    )
    assert timed_out, (
        "Expected scipy expm_multiply to TIME OUT within 90 s on stiff "
        "N=400×1e7.  If scipy has improved significantly, re-evaluate."
    )
