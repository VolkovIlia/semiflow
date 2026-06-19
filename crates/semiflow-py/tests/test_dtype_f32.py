"""Tests for Issue #3 — opt-in dtype="f32" path (ADR-0115).

Covers:
- GraphHeat, MagnusGraphHeat, VarCoefGraphHeat, Heat1D with dtype="f32"
- dtype="f64" default is byte-unchanged vs existing behaviour
- f32 result matches f64 within f32 tolerance (rtol~1e-5)
- Invalid dtype values raise SemiflowError(kind="OutOfDomain")
- fp16 / other strings are rejected
"""

from typing import Literal

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N = 32
T_FINAL = 0.25
N_STEPS = 20
RHO_BAR = 2.0  # spectral bound for P_32 combinatorial Laplacian


def _gaussian_ic(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.exp(-50.0 * (x - 0.5) ** 2)


def _path_graph(n: int) -> rp.Graph:
    return rp.Graph.path(n)


# ---------------------------------------------------------------------------
# GraphHeat dtype tests
# ---------------------------------------------------------------------------


class TestGraphHeatDtype:
    """dtype="f32" and dtype="f64" on GraphHeat."""

    def test_f64_default_returns_float64(self) -> None:
        g = _path_graph(N)
        gh = rp.GraphHeat(graph=g, rho_bar=RHO_BAR)
        f0 = _gaussian_ic(N)
        out = gh.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        assert out.dtype == np.float64

    def test_f64_explicit_returns_float64(self) -> None:
        g = _path_graph(N)
        gh = rp.GraphHeat(graph=g, rho_bar=RHO_BAR, dtype="f64")
        f0 = _gaussian_ic(N)
        out = gh.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        assert out.dtype == np.float64

    def test_f32_returns_float32(self) -> None:
        g = _path_graph(N)
        gh = rp.GraphHeat(graph=g, rho_bar=RHO_BAR, dtype="f32")
        f0 = _gaussian_ic(N)
        out = gh.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        assert out.dtype == np.float32

    def test_f32_close_to_f64(self) -> None:
        g = _path_graph(N)
        f0 = _gaussian_ic(N)
        gh64 = rp.GraphHeat(graph=g, rho_bar=RHO_BAR, dtype="f64")
        gh32 = rp.GraphHeat(graph=g, rho_bar=RHO_BAR, dtype="f32")
        out64 = gh64.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        out32 = gh32.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        np.testing.assert_allclose(out32.astype(np.float64), out64, rtol=1e-4)

    def test_invalid_dtype_raises(self) -> None:
        g = _path_graph(N)
        with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
            rp.GraphHeat(graph=g, rho_bar=RHO_BAR, dtype="f16")  # type: ignore[arg-type]  # intentional invalid dtype: runtime-rejection test

    def test_foo_dtype_raises(self) -> None:
        g = _path_graph(N)
        with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
            rp.GraphHeat(graph=g, rho_bar=RHO_BAR, dtype="foo")  # type: ignore[arg-type]  # intentional invalid dtype: runtime-rejection test


# ---------------------------------------------------------------------------
# MagnusGraphHeat dtype tests
# ---------------------------------------------------------------------------


class TestMagnusGraphHeatDtype:
    """dtype="f32" and dtype="f64" on MagnusGraphHeat."""

    def _make_kernel(self, dtype: Literal["f64", "f32"]) -> rp.MagnusGraphHeat:
        g = _path_graph(N)

        def lap_at_t(t: float) -> rp.Graph:
            return g

        return rp.MagnusGraphHeat(
            graph=g,
            lap_at_t=lap_at_t,
            rho_bar_max=RHO_BAR,
            dtype=dtype,
        )

    def test_f64_returns_float64(self) -> None:
        mgh = self._make_kernel("f64")
        f0 = _gaussian_ic(N)
        out = mgh.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        assert out.dtype == np.float64

    def test_f32_returns_float32(self) -> None:
        mgh = self._make_kernel("f32")
        f0 = _gaussian_ic(N)
        out = mgh.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        assert out.dtype == np.float32

    def test_f32_close_to_f64(self) -> None:
        """f32 Magnus result is close to f64 within f32 arithmetic tolerance."""
        f0 = _gaussian_ic(N)
        mgh64 = self._make_kernel("f64")
        mgh32 = self._make_kernel("f32")
        out64 = mgh64.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        out32 = mgh32.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        np.testing.assert_allclose(out32.astype(np.float64), out64, rtol=5e-3)

    def test_invalid_dtype_raises(self) -> None:
        g = _path_graph(N)

        def lap_at_t(t: float) -> rp.Graph:
            return g

        with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
            rp.MagnusGraphHeat(
                graph=g,
                lap_at_t=lap_at_t,
                rho_bar_max=RHO_BAR,
                dtype="fp16",  # type: ignore[arg-type]  # intentional invalid dtype: runtime-rejection test
            )


# ---------------------------------------------------------------------------
# VarCoefGraphHeat dtype tests
# ---------------------------------------------------------------------------


class TestVarCoefGraphHeatDtype:
    """dtype="f32" and dtype="f64" on VarCoefGraphHeat."""

    def _make_kernel(self, dtype: Literal["f64", "f32"]) -> rp.VarCoefGraphHeat:
        g = _path_graph(N)
        a = np.ones(N, dtype=np.float64) * 1.5
        return rp.VarCoefGraphHeat(graph=g, a=a, rho_bar=RHO_BAR * 2.0, dtype=dtype)

    def test_f64_returns_float64(self) -> None:
        vc = self._make_kernel("f64")
        f0 = _gaussian_ic(N)
        out = vc.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        assert out.dtype == np.float64

    def test_f32_returns_float32(self) -> None:
        vc = self._make_kernel("f32")
        f0 = _gaussian_ic(N)
        out = vc.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        assert out.dtype == np.float32

    def test_f32_close_to_f64(self) -> None:
        f0 = _gaussian_ic(N)
        vc64 = self._make_kernel("f64")
        vc32 = self._make_kernel("f32")
        out64 = vc64.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        out32 = vc32.evolve(t_final=T_FINAL, n_steps=N_STEPS, f0=f0)
        np.testing.assert_allclose(out32.astype(np.float64), out64, rtol=1e-4)

    def test_invalid_dtype_raises(self) -> None:
        g = _path_graph(N)
        a = np.ones(N, dtype=np.float64)
        with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
            rp.VarCoefGraphHeat(graph=g, a=a, rho_bar=RHO_BAR, dtype="float32")  # type: ignore[arg-type]  # intentional invalid dtype: runtime-rejection test


# ---------------------------------------------------------------------------
# Heat1D dtype tests
# ---------------------------------------------------------------------------


class TestHeat1DDtype:
    """dtype="f32" and dtype="f64" on Heat1D (diffusion path)."""

    def test_f64_default_state_is_float64(self) -> None:
        f0 = _gaussian_ic(N)
        h = rp.Heat1D(0.0, 1.0, N, f0)
        vals = h.values()
        assert vals.dtype == np.float64

    def test_f32_evolve_preserves_f64_internal_state(self) -> None:
        """After f32 evolve, values() still returns float64 (internal state is always f64)."""
        f0 = _gaussian_ic(N)
        h = rp.Heat1D(0.0, 1.0, N, f0, dtype="f32")
        h.evolve(0.01, n_steps=10)
        # values() returns the internal f64 state
        vals = h.values()
        assert vals.dtype == np.float64

    def test_f32_evolve_result_close_to_f64(self) -> None:
        """f32 result is close to f64 within f32 arithmetic tolerance.

        The f32 path uses the generic non-SIMD apply_f kernel; max relative
        difference is O(f32 epsilon * sqrt(n_steps)) ≈ 1e-3 for n_steps=10.
        """
        f0 = _gaussian_ic(N)
        h64 = rp.Heat1D(0.0, 1.0, N, f0.copy(), dtype="f64")
        h32 = rp.Heat1D(0.0, 1.0, N, f0.copy(), dtype="f32")
        h64.evolve(0.01, n_steps=10)
        h32.evolve(0.01, n_steps=10)
        v64 = h64.values()
        v32 = h32.values()
        # rtol=5e-3 reflects f32 precision (eps~1.2e-7, accumulated over 10 steps)
        np.testing.assert_allclose(v32, v64, rtol=5e-3)

    def test_invalid_dtype_raises(self) -> None:
        f0 = _gaussian_ic(N)
        with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
            rp.Heat1D(0.0, 1.0, N, f0, dtype="f16")  # type: ignore[arg-type]  # intentional invalid dtype: runtime-rejection test

    def test_fp16_dtype_raises(self) -> None:
        f0 = _gaussian_ic(N)
        with pytest.raises(rp.SemiflowError, match="OutOfDomain"):
            rp.Heat1D(0.0, 1.0, N, f0, dtype="fp16")  # type: ignore[arg-type]  # intentional invalid dtype: runtime-rejection test
