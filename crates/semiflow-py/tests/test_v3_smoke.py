"""Smoke tests for semiflow v3.0 surface (ADR-0076, Wave E).

Tests cover:
  - EvolverHeat1DUnitV3 constructor (happy path + error paths)
  - GrowthV3 namedtuple-like access (.multiplier, .omega)
  - size() and n_chernoff() introspection
  - values() output as numpy array
  - evolve_into numerical accuracy against the v2 Heat1D oracle
  - G_binding_parity sub-test 3 (PyO3 v3 ⇔ v2): bit-identical to Heat1D.evolve

Parameters match the standard Wave-A/B heat smoke:
  domain [-10, 10], n=1000, t=1, n_steps=100
  u0(x) = exp(-x²)
  oracle: u(1,x) = exp(-x²/5) / sqrt(5),  sup_error < 5e-4

Cross-validation:
  EvolverHeat1DUnitV3.evolve_into result MUST equal Heat1D result bit-for-bit
  (same semiflow-core kernel, different binding surface).
"""

import math
import threading

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 1000
XMIN, XMAX = -10.0, 10.0
T = 1.0
N_CHERNOFF = 100


@pytest.fixture
def xs() -> np.ndarray:
    return np.linspace(XMIN, XMAX, N)


@pytest.fixture
def u0(xs: np.ndarray) -> np.ndarray:
    return np.exp(-(xs**2))


@pytest.fixture
def oracle(xs: np.ndarray) -> np.ndarray:
    return np.exp(-(xs**2) / 5.0) / math.sqrt(5.0)


@pytest.fixture
def evolver_v3(u0: np.ndarray) -> rp.EvolverHeat1DUnitV3:
    return rp.EvolverHeat1DUnitV3(XMIN, XMAX, N, u0, N_CHERNOFF)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_ctor_happy_path(u0: np.ndarray) -> None:
    ev = rp.EvolverHeat1DUnitV3(XMIN, XMAX, N, u0, N_CHERNOFF)
    assert ev.size() == N
    assert ev.n_chernoff() == N_CHERNOFF


def test_ctor_bad_n_grid(u0: np.ndarray) -> None:
    with pytest.raises(rp.SemiflowError):
        rp.EvolverHeat1DUnitV3(XMIN, XMAX, 2, u0[:2], N_CHERNOFF)


def test_ctor_bad_n_chernoff(u0: np.ndarray) -> None:
    with pytest.raises(rp.SemiflowError):
        rp.EvolverHeat1DUnitV3(XMIN, XMAX, N, u0, 0)


def test_ctor_nan_in_u0(xs: np.ndarray) -> None:
    bad = np.exp(-(xs**2)).copy()
    bad[5] = float("nan")
    with pytest.raises(rp.SemiflowError):
        rp.EvolverHeat1DUnitV3(XMIN, XMAX, N, bad, N_CHERNOFF)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_size(evolver_v3: rp.EvolverHeat1DUnitV3) -> None:
    assert evolver_v3.size() == N


def test_len(evolver_v3: rp.EvolverHeat1DUnitV3) -> None:
    assert len(evolver_v3) == N


def test_n_chernoff(evolver_v3: rp.EvolverHeat1DUnitV3) -> None:
    assert evolver_v3.n_chernoff() == N_CHERNOFF


def test_values_initial(evolver_v3: rp.EvolverHeat1DUnitV3, u0: np.ndarray) -> None:
    vals = evolver_v3.values()
    assert isinstance(vals, np.ndarray)
    assert vals.shape == (N,)
    assert np.array_equal(vals, u0)


# ---------------------------------------------------------------------------
# GrowthV3
# ---------------------------------------------------------------------------


def test_growth_unit_diffusion(evolver_v3: rp.EvolverHeat1DUnitV3) -> None:
    """G_binding_parity sub-test 3 pre-requisite: growth bound attributes."""
    g = evolver_v3.growth()
    assert isinstance(g, rp.GrowthV3)
    assert g.multiplier == 1.0
    assert g.omega == 0.0


def test_growth_repr(evolver_v3: rp.EvolverHeat1DUnitV3) -> None:
    g = evolver_v3.growth()
    r = repr(g)
    assert "GrowthV3" in r
    assert "multiplier" in r
    assert "omega" in r


# ---------------------------------------------------------------------------
# Numerical accuracy
# ---------------------------------------------------------------------------


def test_evolve_into_accuracy(
    evolver_v3: rp.EvolverHeat1DUnitV3,
    oracle: np.ndarray,
) -> None:
    """EvolverHeat1DUnitV3.evolve_into should match oracle within 5e-4."""
    buf = np.empty(N, dtype=np.float64)
    evolver_v3.evolve_into(T, buf)
    sup_error = np.max(np.abs(buf - oracle))
    assert sup_error < 5e-4, f"sup_error={sup_error:.3e} >= 5e-4"


def test_evolve_into_updates_values(evolver_v3: rp.EvolverHeat1DUnitV3) -> None:
    """After evolve_into, values() should reflect the evolved state."""
    buf = np.empty(N, dtype=np.float64)
    evolver_v3.evolve_into(T, buf)
    vals = evolver_v3.values()
    assert np.array_equal(buf, vals), "values() diverged from evolve_into output"


# ---------------------------------------------------------------------------
# G_binding_parity sub-test 3 (PyO3 v3 ⇔ v2 Heat1D bit-identical)
# ---------------------------------------------------------------------------


def test_g_binding_parity_sub3_v3_equals_v2(u0: np.ndarray) -> None:
    """G_binding_parity sub-test 3: EvolverHeat1DUnitV3 must be bit-identical
    to the v2 Heat1D.evolve result on the same kernel.

    Both call DiffusionChernoff<f64> with unit a=1.0 on the same grid and
    n_chernoff.  The v3 binding is pure pass-through; zero ULP difference is
    achievable and required (ADR-0076 §G_binding_parity gate).
    """
    # v3 path
    ev3 = rp.EvolverHeat1DUnitV3(XMIN, XMAX, N, u0, N_CHERNOFF)
    buf_v3 = np.empty(N, dtype=np.float64)
    ev3.evolve_into(T, buf_v3)

    # v2 path
    heat_v2 = rp.Heat1D(XMIN, XMAX, N, u0)
    heat_v2.evolve(T, N_CHERNOFF)
    buf_v2 = heat_v2.values()

    assert np.array_equal(buf_v3, buf_v2), (
        f"G_binding_parity sub-test 3 FAILED: v3 != v2; "
        f"max diff = {np.max(np.abs(buf_v3 - buf_v2)):.3e}"
    )


# ---------------------------------------------------------------------------
# GIL release (threading smoke)
# ---------------------------------------------------------------------------


def test_gil_release_threading(u0: np.ndarray) -> None:
    """evolve_into releases GIL; a second thread can run concurrently."""
    results: list[float] = []
    errors: list[Exception] = []

    def run_evolver() -> None:
        try:
            ev = rp.EvolverHeat1DUnitV3(XMIN, XMAX, N, u0.copy(), N_CHERNOFF)
            buf = np.empty(N, dtype=np.float64)
            ev.evolve_into(T, buf)
            results.append(float(np.max(np.abs(buf))))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run_evolver) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 4
    for r in results:
        assert r > 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_evolve_into_bad_t(evolver_v3: rp.EvolverHeat1DUnitV3) -> None:
    buf = np.empty(N, dtype=np.float64)
    with pytest.raises(rp.SemiflowError):
        evolver_v3.evolve_into(-1.0, buf)


def test_evolve_into_wrong_buf_size(evolver_v3: rp.EvolverHeat1DUnitV3) -> None:
    buf = np.empty(N // 2, dtype=np.float64)
    with pytest.raises(rp.SemiflowError):
        evolver_v3.evolve_into(T, buf)
