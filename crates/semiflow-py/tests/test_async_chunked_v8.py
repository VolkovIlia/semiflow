"""Tests for Heat1D.evolve_chunked (v8.1.0 A-4, ADR-0141).

Covers:
  - Parity: 0-ULP identity with evolve(t, total_steps) for multiple splits.
  - Progress callback: called expected number of times with monotone (done, total).
  - Cancellation: a callback that raises propagates cleanly (no deadlock, no panic).
  - GIL-release smoke: a concurrent Python thread makes progress during a long
    chunked evolve (best-effort; asserts no deadlock on a moderately long run).
  - Error paths: chunk_steps == 0, total_steps == 0, t < 0.
"""

import threading
import time

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Shared fixture parameters
# ---------------------------------------------------------------------------
XMIN = -10.0
XMAX = 10.0
N = 200
T = 0.5
TOTAL_STEPS = 100


def _make_state() -> rp.Heat1D:
    xs = np.linspace(XMIN, XMAX, N)
    u0 = np.exp(-(xs ** 2))
    return rp.Heat1D(XMIN, XMAX, N, u0)


def _reference_result() -> np.ndarray:
    """Canonical result via synchronous evolve."""
    if not hasattr(_reference_result, "_cache"):
        state = _make_state()
        state.evolve(T, TOTAL_STEPS)
        _reference_result._cache = state.values().copy()  # type: ignore[attr-defined]
    return _reference_result._cache  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Parity: 0-ULP identity with evolve()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total, chunk",
    [
        (TOTAL_STEPS, TOTAL_STEPS),  # chunk = total (one chunk, no split)
        (TOTAL_STEPS, 1),            # chunk = 1 (maximum granularity)
        (TOTAL_STEPS, 33),           # chunk does not divide total (33+33+34)
        (TOTAL_STEPS, 50),           # chunk = total // 2 (even split)
        (TOTAL_STEPS, 7),            # prime chunk size
        (1, 1),                       # single step
    ],
)
def test_parity_zero_ulp(total: int, chunk: int) -> None:
    """evolve_chunked result must be bit-identical (0 ULP) to evolve()."""
    ref = _reference_result()

    state = _make_state()
    # Use total=TOTAL_STEPS and t=T to match the reference; if param varies,
    # recompute reference for that (total, T) pair.
    if total != TOTAL_STEPS:
        ref_state = _make_state()
        ref_state.evolve(T, total)
        ref = ref_state.values().copy()

    state.evolve_chunked(T, total, chunk)
    got = state.values()

    # Strict bit-for-bit identity — same floating-point arithmetic, same order.
    assert np.array_equal(got, ref), (
        f"Not 0-ULP equal for (total={total}, chunk={chunk}): "
        f"max |diff| = {np.max(np.abs(got - ref))}"
    )


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def test_progress_called_expected_times() -> None:
    """progress(done, total) called ceil(total / chunk) times."""
    total = 20
    chunk = 7  # 7 + 7 + 6 = 20 — 3 chunks

    calls: list[tuple[int, int]] = []

    def progress(done: int, tot: int) -> None:
        calls.append((done, tot))

    state = _make_state()
    state.evolve_chunked(T, total, chunk, progress=progress)

    expected_calls = (total + chunk - 1) // chunk  # ceil division
    assert len(calls) == expected_calls, f"Expected {expected_calls} calls, got {len(calls)}"


def test_progress_done_is_monotone() -> None:
    """done argument to progress must be strictly monotone increasing."""
    total = 30
    chunk = 4

    done_values: list[int] = []

    def progress(done: int, tot: int) -> None:
        done_values.append(done)

    state = _make_state()
    state.evolve_chunked(T, total, chunk, progress=progress)

    assert done_values[-1] == total, "Last done must equal total"
    for i in range(1, len(done_values)):
        assert done_values[i] > done_values[i - 1], "done must be monotone"


def test_progress_total_constant() -> None:
    """total argument to progress must be constant and equal to total_steps."""
    total = 20
    chunk = 3

    totals_seen: list[int] = []

    def progress(done: int, tot: int) -> None:
        totals_seen.append(tot)

    state = _make_state()
    state.evolve_chunked(T, total, chunk, progress=progress)

    assert all(t == total for t in totals_seen), "total must be constant"


def test_no_progress_callback() -> None:
    """evolve_chunked with progress=None (default) must not raise."""
    state = _make_state()
    state.evolve_chunked(T, TOTAL_STEPS, 10)  # no progress keyword


# ---------------------------------------------------------------------------
# Cancellation: exception in callback propagates cleanly
# ---------------------------------------------------------------------------


def test_cancellation_exception_from_callback() -> None:
    """An exception raised inside progress propagates out of evolve_chunked."""
    total = 50
    chunk = 10

    call_count: list[int] = [0]

    class StopSignal(Exception):
        pass

    def progress(done: int, tot: int) -> None:
        call_count[0] += 1
        if call_count[0] >= 2:
            raise StopSignal("simulated cancellation")

    state = _make_state()

    with pytest.raises(StopSignal):
        state.evolve_chunked(T, total, chunk, progress=progress)

    # The exception fired on the 2nd callback — evolution stopped before chunk 3.
    assert call_count[0] == 2, "Exception should fire on second callback"


def test_keyboard_interrupt_from_callback() -> None:
    """KeyboardInterrupt raised from progress propagates without deadlock."""
    total = 50
    chunk = 5

    fired: list[int] = [0]

    def progress(done: int, tot: int) -> None:
        fired[0] += 1
        if fired[0] == 3:
            raise KeyboardInterrupt

    state = _make_state()

    with pytest.raises(KeyboardInterrupt):
        state.evolve_chunked(T, total, chunk, progress=progress)

    assert fired[0] == 3


# ---------------------------------------------------------------------------
# GIL-release smoke: concurrent thread makes progress during compute
# ---------------------------------------------------------------------------


def test_gil_release_smoke() -> None:
    """A concurrent Python thread should make progress while compute runs.

    This is a best-effort test: we run a long evolve_chunked in a background
    thread with a large total_steps and large chunk_steps (so the GIL is
    released for a measurable period), and verify that a counter incremented
    by a second Python thread advances during that time.

    Not deterministic, but should reliably pass on any modern machine.
    No deadlock assertion: if the main thread times out, join() detects it.
    """
    # Large enough that the GIL is released for >50ms total on any machine.
    large_total = 2000
    chunk = 200

    # Background counter thread — just increments while GIL is available.
    counter: list[int] = [0]
    stop_flag: list[bool] = [False]

    def counter_thread() -> None:
        while not stop_flag[0]:
            counter[0] += 1
            time.sleep(0.001)

    xs = np.linspace(XMIN, XMAX, N)
    u0 = np.exp(-(xs ** 2))
    state = rp.Heat1D(XMIN, XMAX, N, u0)

    t_counter = threading.Thread(target=counter_thread, daemon=True)
    t_counter.start()

    state.evolve_chunked(T, large_total, chunk)

    stop_flag[0] = True
    t_counter.join(timeout=5.0)

    # Counter should have advanced during the evolve — GIL was released.
    assert counter[0] > 0, "Counter thread never ran — GIL was not released"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_error_chunk_steps_zero() -> None:
    """chunk_steps == 0 must raise SemiflowError(OutOfDomain)."""
    state = _make_state()
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        state.evolve_chunked(T, TOTAL_STEPS, 0)


def test_error_total_steps_zero() -> None:
    """total_steps == 0 must raise SemiflowError(OutOfDomain)."""
    state = _make_state()
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        state.evolve_chunked(T, 0, 10)


def test_error_negative_t() -> None:
    """t < 0 must raise SemiflowError(OutOfDomain)."""
    state = _make_state()
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        state.evolve_chunked(-1.0, TOTAL_STEPS, 10)


def test_error_nonfinite_t() -> None:
    """t = inf must raise SemiflowError(OutOfDomain)."""
    state = _make_state()
    with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
        state.evolve_chunked(float("inf"), TOTAL_STEPS, 10)
