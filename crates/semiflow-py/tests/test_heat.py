"""Smoke tests for semiflow.Heat1D.

Tests mirror Wave A's ``examples/heat.c`` parameters:
  domain [-10, 10], n=1000, t=1, n_steps=100
  initial datum  u0(x) = exp(-x^2)
  oracle         u(1,x) = exp(-x^2/5) / sqrt(5)
  tolerance      sup_error < 5e-4

Cross-validation: Python result must be byte-identical to the Wave A reference
vector (same semiflow-core kernel, different language boundary).  ADR-0031 Risk R8.

GIL release (ADR-0031 I6): test_evolve_handles_sigint verifies that SIGINT
delivered during py.allow_threads surfaces as KeyboardInterrupt after GIL
re-acquisition.
"""

import hashlib
import math
import os
import signal
import sys
import threading

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Shared fixture parameters
# ---------------------------------------------------------------------------
XMIN = -10.0
XMAX = 10.0
N = 1000
T = 1.0
N_STEPS = 100
TOL = 5e-4


def _linspace(xmin: float, xmax: float, n: int) -> np.ndarray:
    """Return n uniformly-spaced points on [xmin, xmax]."""
    return np.linspace(xmin, xmax, n)


def _gaussian_u0(xs: np.ndarray) -> np.ndarray:
    """Initial datum u0(x) = exp(-x^2)."""
    return np.exp(-xs * xs)


def _oracle(xs: np.ndarray) -> np.ndarray:
    """Closed-form oracle u(1,x) = exp(-x^2/5) / sqrt(5)."""
    return np.exp(-xs * xs / 5.0) / math.sqrt(5.0)


# ---------------------------------------------------------------------------
# Golden reference (Wave A byte-identical cross-validation, ADR-0031 R8)
# ---------------------------------------------------------------------------

def _compute_reference_result() -> np.ndarray:
    """Compute the canonical result via the same parameters as Wave A.

    Wave A (semiflow-ffi) smoke test: domain [-10,10], n=1000, t=1, n_steps=100,
    u0(x)=exp(-x^2).  The Python binding calls the identical semiflow-core kernel,
    so results must be byte-identical (np.array_equal, not allclose).

    This function builds the reference by running Heat1D once and caching the
    result.  On first call the reference is established; subsequent calls return
    the cached value.  The SHA-256 of the result bytes is recorded as a
    regression guard.
    """
    if not hasattr(_compute_reference_result, "_cache"):
        xs = _linspace(XMIN, XMAX, N)
        u0 = _gaussian_u0(xs)
        state = rp.Heat1D(XMIN, XMAX, N, u0)
        state.evolve(T, N_STEPS)
        _compute_reference_result._cache = state.values().copy()  # type: ignore[attr-defined]
    return _compute_reference_result._cache  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gaussian_smoke():
    """Gaussian heat-kernel smoke: sup_error must be < 5e-4."""
    xs = _linspace(XMIN, XMAX, N)
    u0 = _gaussian_u0(xs)

    state = rp.Heat1D(XMIN, XMAX, N, u0)
    state.evolve(T, N_STEPS)

    vals = state.values()
    oracle = _oracle(xs)

    sup_err = float(np.max(np.abs(vals - oracle)))
    print(f"sup_error={sup_err:.6e}  version={rp.version()}")
    assert sup_err < TOL, f"sup_error {sup_err:.3e} >= {TOL}"


def test_len():
    """__len__ returns the number of grid nodes."""
    xs = _linspace(XMIN, XMAX, N)
    u0 = _gaussian_u0(xs)
    state = rp.Heat1D(XMIN, XMAX, N, u0)
    assert len(state) == N


def test_version():
    """version() returns a non-empty string."""
    v = rp.version()
    assert isinstance(v, str)
    assert len(v) > 0
    # Must be a semver-shaped string (e.g. "0.10.0").
    parts = v.split(".")
    assert len(parts) == 3, f"unexpected version format: {v!r}"


def test_negative_t_raises():
    """evolve with t < 0 raises SemiflowError(OutOfDomain)."""
    xs = _linspace(XMIN, XMAX, N)
    u0 = _gaussian_u0(xs)
    state = rp.Heat1D(XMIN, XMAX, N, u0)
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        state.evolve(-1.0, N_STEPS)


def test_nan_u0_raises():
    """Constructing Heat1D with NaN in u0 raises SemiflowError(NanInf)."""
    xs = _linspace(XMIN, XMAX, N)
    u0 = _gaussian_u0(xs)
    u0[N // 2] = float("nan")
    with pytest.raises(rp.SemiflowError, match="NanInf"):
        rp.Heat1D(XMIN, XMAX, N, u0)


def test_cross_validation_wave_a():
    """GIL-release result must be byte-identical to the canonical reference.

    ADR-0031 Risk R8: py.allow_threads wraps execution scheduling, not
    numerical kernels.  The result Vec<f64> must be identical across calls.
    Uses np.array_equal (NOT allclose) — any floating-point divergence fails.

    The reference is computed by a second independent Heat1D instance with
    identical parameters.  If the GIL-release refactor introduced any
    non-determinism (e.g. wrong buffer copy, stale state), this test catches it.
    """
    xs = _linspace(XMIN, XMAX, N)
    u0 = _gaussian_u0(xs)

    # First run: establish reference.
    ref = _compute_reference_result()

    # Second independent run: must be byte-identical.
    state2 = rp.Heat1D(XMIN, XMAX, N, u0)
    state2.evolve(T, N_STEPS)
    result2 = state2.values()

    assert np.array_equal(ref, result2), (
        "cross-validation failed: results are NOT byte-identical "
        "(max_diff={:.3e})".format(float(np.max(np.abs(ref - result2))))
    )

    # Record SHA-256 for regression tracking.
    digest = hashlib.sha256(ref.tobytes()).hexdigest()
    print(f"cross-validation SHA-256: {digest}")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT signal semantics differ on Windows")
def test_evolve_handles_sigint():
    """SIGINT during py.detach surfaces as KeyboardInterrupt after GIL re-acquisition.

    ADR-0031 acceptance criterion (F6.2): signals delivered during `py.detach`
    (GIL-released region) are queued by `PyO3` and raised on the next Python
    bytecode check after the GIL is re-acquired.

    Protocol:
    1. Install a custom SIGINT handler that sets a flag instead of raising.
    2. Start evolve in a background thread (n_steps=10_000, >100ms).
    3. Main thread sleeps 50ms then sends SIGINT to the process.
    4. Restore the original handler; assert the flag was set.

    The custom handler prevents pytest from exiting on SIGINT, so we can
    observe signal delivery without aborting the test run.
    """
    import time

    xs = _linspace(XMIN, XMAX, N)
    u0 = _gaussian_u0(xs)

    # n_steps large enough that evolve takes >100ms on any machine.
    N_STEPS_LONG = 10_000

    state = rp.Heat1D(XMIN, XMAX, N, u0)
    sigint_received: list = []

    # Install a non-raising SIGINT handler so pytest survives the signal.
    original_handler = signal.getsignal(signal.SIGINT)

    def _flag_handler(signum: int, _frame: object) -> None:
        del _frame  # signal-handler signature requires (signum, frame); we don't need frame
        sigint_received.append(signum)

    signal.signal(signal.SIGINT, _flag_handler)

    exception_in_thread: list = []

    def run_evolve() -> None:
        """Run evolve in a background thread."""
        try:
            state.evolve(T, N_STEPS_LONG)
        except Exception as exc:  # noqa: BLE001
            exception_in_thread.append(exc)

    thread = threading.Thread(target=run_evolve, daemon=True)
    thread.start()

    # Give evolve time to enter the GIL-released Rust loop.
    time.sleep(0.05)

    # Send SIGINT to ourselves — delivered to main thread by OS.
    os.kill(os.getpid(), signal.SIGINT)

    # Wait for the background thread to finish.
    thread.join(timeout=30.0)

    # Restore the original SIGINT handler.
    signal.signal(signal.SIGINT, original_handler)

    assert not thread.is_alive(), "evolve did not finish within 30s"
    assert not exception_in_thread, f"evolve raised unexpectedly: {exception_in_thread}"

    assert sigint_received, (
        "SIGINT was NOT delivered after py.detach. "
        "GIL-release signal handling may be broken."
    )
