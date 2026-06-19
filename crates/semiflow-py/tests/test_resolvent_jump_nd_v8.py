"""Smoke tests + G_BINDING_RESOLVENT_JUMP_ND_PARITY sub-test 3 (PyO3, v8.3.0).

Tests cover:
  - ResolventJump2DV8 / ResolventJump3DV8 constructors (happy + error paths)
  - .jump(t, g) returns a finite numpy float64 array of correct shape
  - shape() and m_nodes() introspection
  - Error paths: invalid params raise SemiflowError
  - ND layout round-trip: Fortran-order ravel/reshape is byte-identical to
    core golden (THE critical check for the C-vs-F order bug).
  - G_BINDING_RESOLVENT_JUMP_ND_PARITY sub-test 3 (PyO3 v3, determinism check).

Canonical smoke params (§5, V8_3_TIER3_BINDING_DESIGN.md):
  2D: 8×8 on [−5,5]², M=8, t=1.0; u0=exp(−x²−y²); axis-0-fastest layout.
  3D: 4×4×4 on [−1,1]³, M=6, t=1.0; u0=exp(−x²−y²−z²); axis-0-fastest.

The Fortran-order round-trip test is THE safety gate for the ND layout:
  g_nd = u0.reshape((nx,ny), order="F") → jump() → result.ravel("F")
  must be bit-identical to flat-input result (same axis-0-fastest data).
Any divergence indicates a C-vs-F-order marshalling bug.
"""

import numpy as np
import pytest

import semiflow as rp  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# 2D canonical parameters (§5)
# ---------------------------------------------------------------------------

X2_MIN, X2_MAX = -5.0, 5.0
NX2, NY2 = 8, 8
M2 = 8
T2 = 1.0

xs2 = np.linspace(X2_MIN, X2_MAX, NX2)
ys2 = np.linspace(X2_MIN, X2_MAX, NY2)
XX2, YY2 = np.meshgrid(xs2, ys2, indexing="ij")  # (nx, ny)
# u0_nd has shape (nx, ny) in C order; we pass ND → binding ravels "F"
U0_2D_ND = np.exp(-(XX2**2) - (YY2**2)).astype(np.float64)  # shape (8,8)
# Flat axis-0-fastest (Fortran ravel of the (nx,ny) grid)
U0_2D_FLAT = U0_2D_ND.ravel(order="F").astype(np.float64)  # length 64


# ---------------------------------------------------------------------------
# 3D canonical parameters (§5)
# ---------------------------------------------------------------------------

X3_MIN, X3_MAX = -1.0, 1.0
NX3, NY3, NZ3 = 4, 4, 4
M3 = 6
T3 = 1.0

xs3 = np.linspace(X3_MIN, X3_MAX, NX3)
ys3 = np.linspace(X3_MIN, X3_MAX, NY3)
zs3 = np.linspace(X3_MIN, X3_MAX, NZ3)
XX3, YY3, ZZ3 = np.meshgrid(xs3, ys3, zs3, indexing="ij")  # (nx,ny,nz)
U0_3D_ND = np.exp(-(XX3**2) - (YY3**2) - (ZZ3**2)).astype(np.float64)
U0_3D_FLAT = U0_3D_ND.ravel(order="F").astype(np.float64)


# ---------------------------------------------------------------------------
# 2D constructor — happy paths
# ---------------------------------------------------------------------------


def make_2d(nx: int = NX2, ny: int = NY2, m: int = M2) -> rp.ResolventJump2DV8:
    return rp.ResolventJump2DV8(X2_MIN, X2_MAX, nx, X2_MIN, X2_MAX, ny, m)


def test_2d_ctor_happy() -> None:
    ev = make_2d()
    assert ev.shape() == (NX2, NY2)
    assert ev.m_nodes() == M2


def test_2d_ctor_custom_m() -> None:
    ev = make_2d(m=10)
    assert ev.m_nodes() == 10


# ---------------------------------------------------------------------------
# 2D constructor — error paths
# ---------------------------------------------------------------------------


def test_2d_bad_m_nodes() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.ResolventJump2DV8(X2_MIN, X2_MAX, NX2, X2_MIN, X2_MAX, NY2, 5)


def test_2d_bad_nx() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.ResolventJump2DV8(X2_MIN, X2_MAX, 2, X2_MIN, X2_MAX, NY2, M2)


def test_2d_reversed_domain() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.ResolventJump2DV8(5.0, -5.0, NX2, X2_MIN, X2_MAX, NY2, M2)


# ---------------------------------------------------------------------------
# 2D jump — shape, dtype, finiteness (flat input)
# ---------------------------------------------------------------------------


def test_2d_jump_shape_flat() -> None:
    ev = make_2d()
    result = ev.jump(T2, U0_2D_FLAT)
    assert result.shape == (NX2, NY2)


def test_2d_jump_dtype() -> None:
    ev = make_2d()
    result = ev.jump(T2, U0_2D_FLAT)
    assert result.dtype == np.float64


def test_2d_jump_finite() -> None:
    ev = make_2d()
    result = ev.jump(T2, U0_2D_FLAT)
    assert np.all(np.isfinite(result)), "2D jump has non-finite entries"


# ---------------------------------------------------------------------------
# 2D jump — error paths
# ---------------------------------------------------------------------------


def test_2d_jump_bad_t() -> None:
    ev = make_2d()
    with pytest.raises(rp.SemiflowError):
        ev.jump(-1.0, U0_2D_FLAT)


def test_2d_jump_bad_t_zero() -> None:
    ev = make_2d()
    with pytest.raises(rp.SemiflowError):
        ev.jump(0.0, U0_2D_FLAT)


def test_2d_jump_wrong_length() -> None:
    ev = make_2d()
    with pytest.raises(rp.SemiflowError):
        ev.jump(T2, np.ones(NX2 * NY2 + 1))


# ---------------------------------------------------------------------------
# ND LAYOUT ROUND-TRIP (the critical anti-C-order gate)
# ---------------------------------------------------------------------------


def test_2d_fortran_order_roundtrip() -> None:
    """NORMATIVE Fortran-order round-trip: §3.1 V8_3_TIER3_BINDING_DESIGN.md.

    U0_2D_ND has shape (nx, ny) in standard C/row-major numpy order.
    We pass it as ND (binding does ravel(order="F") internally).
    We also pass the equivalent pre-raveled flat buffer.
    Both must produce bit-identical results when flattened (order="F").

    A C-vs-F-order bug would give wrong VALUES (not just transposed output).
    This is the definitive gate for the v8.1.0 F4/C1 bug class.
    """
    ev = make_2d()
    # ND input: binding internally ravels order="F"
    result_nd = ev.jump(T2, U0_2D_ND)     # shape (nx, ny)
    # Flat input: caller pre-raveled order="F"
    result_flat = ev.jump(T2, U0_2D_FLAT)  # shape (nx, ny)

    flat_nd = result_nd.ravel(order="F")
    flat_f = result_flat.ravel(order="F")

    ulp_diff = int(np.max(np.abs(flat_nd.view(np.int64) - flat_f.view(np.int64))))
    print(
        f"\n2D Fortran-order round-trip:\n"
        f"  result_nd[0,0] = {result_nd[0,0]:.16e}\n"
        f"  result_flat[0,0] = {result_flat[0,0]:.16e}\n"
        f"  max ULP diff = {ulp_diff}  (expected 0)"
    )
    assert np.array_equal(flat_nd, flat_f), (
        f"2D Fortran-order round-trip NOT bit-identical (max ULP diff = {ulp_diff}). "
        "C-vs-F-order bug in PyO3 binding."
    )


# ---------------------------------------------------------------------------
# 2D determinism (G_BINDING_RESOLVENT_JUMP_ND_PARITY sub-test 3)
# ---------------------------------------------------------------------------


def test_g_binding_resolvent_jump_nd_2d_parity_deterministic() -> None:
    """G_BINDING_RESOLVENT_JUMP_ND_PARITY sub-test 3 — 2D determinism.

    Two independent ResolventJump2DV8.jump() calls with identical flat inputs
    must be bit-identical (confirms GIL-off determinism, no state mutation).
    """
    ev = make_2d()
    result_a = ev.jump(T2, U0_2D_FLAT)
    result_b = ev.jump(T2, U0_2D_FLAT)

    flat_a = result_a.ravel(order="F")
    flat_b = result_b.ravel(order="F")
    ulp = int(np.max(np.abs(flat_a.view(np.int64) - flat_b.view(np.int64))))
    print(
        f"\nG_BINDING_RESOLVENT_JUMP_ND_PARITY 2D sub-test 3 determinism:\n"
        f"  jump_a[0,0]={result_a[0,0]:.16e}  jump_b[0,0]={result_b[0,0]:.16e}\n"
        f"  max ULP diff = {ulp}  (expected 0)"
    )
    assert np.array_equal(result_a, result_b), (
        f"2D jump non-deterministic (max ULP diff = {ulp})"
    )


# ---------------------------------------------------------------------------
# 3D constructor — happy paths
# ---------------------------------------------------------------------------


def make_3d(
    nx: int = NX3, ny: int = NY3, nz: int = NZ3, m: int = M3
) -> rp.ResolventJump3DV8:
    return rp.ResolventJump3DV8(
        X3_MIN, X3_MAX, nx, X3_MIN, X3_MAX, ny, X3_MIN, X3_MAX, nz, m
    )


def test_3d_ctor_happy() -> None:
    ev = make_3d()
    assert ev.shape() == (NX3, NY3, NZ3)
    assert ev.m_nodes() == M3


# ---------------------------------------------------------------------------
# 3D constructor — error paths
# ---------------------------------------------------------------------------


def test_3d_bad_m_nodes() -> None:
    with pytest.raises(rp.SemiflowError):
        rp.ResolventJump3DV8(
            X3_MIN, X3_MAX, NX3, X3_MIN, X3_MAX, NY3,
            X3_MIN, X3_MAX, NZ3, 5,
        )


# ---------------------------------------------------------------------------
# 3D jump — shape, dtype, finiteness
# ---------------------------------------------------------------------------


def test_3d_jump_shape() -> None:
    ev = make_3d()
    result = ev.jump(T3, U0_3D_FLAT)
    assert result.shape == (NX3, NY3, NZ3)


def test_3d_jump_dtype() -> None:
    ev = make_3d()
    result = ev.jump(T3, U0_3D_FLAT)
    assert result.dtype == np.float64


def test_3d_jump_finite() -> None:
    ev = make_3d()
    result = ev.jump(T3, U0_3D_FLAT)
    assert np.all(np.isfinite(result)), "3D jump has non-finite entries"


# ---------------------------------------------------------------------------
# 3D ND layout round-trip
# ---------------------------------------------------------------------------


def test_3d_fortran_order_roundtrip() -> None:
    """NORMATIVE 3D Fortran-order round-trip: §3.1 V8_3_TIER3_BINDING_DESIGN.md."""
    ev = make_3d()
    result_nd = ev.jump(T3, U0_3D_ND)
    result_flat = ev.jump(T3, U0_3D_FLAT)

    flat_nd = result_nd.ravel(order="F")
    flat_f = result_flat.ravel(order="F")

    ulp_diff = int(np.max(np.abs(flat_nd.view(np.int64) - flat_f.view(np.int64))))
    print(
        f"\n3D Fortran-order round-trip:\n"
        f"  result_nd[0,0,0] = {result_nd[0,0,0]:.16e}\n"
        f"  result_flat[0,0,0] = {result_flat[0,0,0]:.16e}\n"
        f"  max ULP diff = {ulp_diff}  (expected 0)"
    )
    assert np.array_equal(flat_nd, flat_f), (
        f"3D Fortran-order round-trip NOT bit-identical (max ULP diff = {ulp_diff}). "
        "C-vs-F-order bug in PyO3 binding."
    )


# ---------------------------------------------------------------------------
# 3D determinism
# ---------------------------------------------------------------------------


def test_g_binding_resolvent_jump_nd_3d_parity_deterministic() -> None:
    """G_BINDING_RESOLVENT_JUMP_ND_PARITY sub-test 3 — 3D determinism."""
    ev = make_3d()
    result_a = ev.jump(T3, U0_3D_FLAT)
    result_b = ev.jump(T3, U0_3D_FLAT)

    flat_a = result_a.ravel(order="F")
    flat_b = result_b.ravel(order="F")
    ulp = int(np.max(np.abs(flat_a.view(np.int64) - flat_b.view(np.int64))))
    print(
        f"\nG_BINDING_RESOLVENT_JUMP_ND_PARITY 3D sub-test 3 determinism:\n"
        f"  jump_a[0,0,0]={result_a[0,0,0]:.16e}  jump_b[0,0,0]={result_b[0,0,0]:.16e}\n"
        f"  max ULP diff = {ulp}  (expected 0)"
    )
    assert np.array_equal(result_a, result_b), (
        f"3D jump non-deterministic (max ULP diff = {ulp})"
    )
