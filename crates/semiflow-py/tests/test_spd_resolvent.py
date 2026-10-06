"""ADR-0202 Wave 3: SPD resolvent / steady solve and operator composition.

Covers (contracts/semiflow-core.spd-resolvent-api.md §2): SymmetricOperator.resolvent
-> SpdResolvent (solve, solve_batched, solve_info, method, n) and
SymmetricOperator.with_diagonal / to_csr / lumped_congruence. The φ-function tests
live in test_phi_combination.py.

Oracles: scipy.sparse.linalg.spsolve (tests skip without scipy) plus numpy-only dense
and long-double checks that always run. Binding parity pins the FNV-1a digest that
``crates/semiflow/tests/spdr_binding_parity.rs`` pins for the core (same dyadic inputs).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import semiflow
from _spd_common import (
    HAS_SCIPY,
    _csr_from_rows,
    _dense,
    _fnv,
    _grid_op,
    _op,
    _op_1d,
    _path_rows,
    _rel,
    _rhs,
    _to_scipy,
    needs_scipy,
    scipy,
)

# ---------------------------------------------------------------------------
# Solve vs scipy.sparse.linalg.spsolve
# ---------------------------------------------------------------------------


@needs_scipy
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_tridiagonal_vs_spsolve(lam: float) -> None:
    # reaction 10 keeps kappa(S) ~ 3e5: spsolve's own error (~kappa*eps*0.1) stays under
    # the 1e-12 gate. At reaction 1 (kappa ~ 3e6) spsolve itself is off by 1.6e-12 while
    # the LDL^T result is within 1.5e-13 of the long-double truth (next test).
    n = 641
    op = _op_1d(n, reaction=10.0)
    res = op.resolvent(lam=lam)
    assert res.method == "tridiagonal"
    assert res.n == n
    b = _rhs(n)
    s = (_to_scipy(op) + lam * scipy.sparse.identity(n)).tocsc()
    want = np.asarray(scipy.sparse.linalg.spsolve(s, b))
    assert _rel(res.solve(b), want) <= 1e-12


@pytest.mark.skipif(
    bool(np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps),
    reason="np.longdouble is not wider than float64 on this platform",
)
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_tridiagonal_vs_longdouble_thomas(lam: float) -> None:
    """kappa ~ 3e6 case against an x87 long-double Thomas solve (no scipy needed)."""
    n = 641
    op = _op_1d(n)
    b = _rhs(n)
    a = _dense(op)
    d = np.diag(a).astype(np.longdouble) + lam
    off = np.array([a[i, i + 1] for i in range(n - 1)], dtype=np.longdouble)
    bb = b.astype(np.longdouble)
    cp = np.zeros(n - 1, dtype=np.longdouble)
    dp = np.zeros(n, dtype=np.longdouble)
    cp[0], dp[0] = off[0] / d[0], bb[0] / d[0]
    for i in range(1, n):
        piv = d[i] - off[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = off[i] / piv
        dp[i] = (bb[i] - off[i - 1] * dp[i - 1]) / piv
    x = np.zeros(n, dtype=np.longdouble)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    got = op.resolvent(lam=lam).solve(b).astype(np.longdouble)
    assert float(np.max(np.abs(got - x)) / np.max(np.abs(x))) <= 1e-12


@needs_scipy
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_pcg_2d_vs_spsolve(lam: float) -> None:
    side = 30
    op = _grid_op(side).with_diagonal(np.full(side * side, 0.5))
    res = op.resolvent(lam=lam, tol=1e-12)
    assert res.method == "pcg-ic0"
    b = _rhs(side * side)
    s = (_to_scipy(op) + lam * scipy.sparse.identity(side * side)).tocsc()
    want = np.asarray(scipy.sparse.linalg.spsolve(s, b))
    assert _rel(res.solve(b), want) <= 1e-9


@needs_scipy
def test_mass_vs_spsolve() -> None:
    n = 200
    op = _op_1d(n)
    mass = 1.0 + 49.0 * np.linspace(0.0, 1.0, n) ** 2
    res = op.resolvent(lam=2.5, mass=mass)
    b = _rhs(n)
    s = (_to_scipy(op) + 2.5 * scipy.sparse.diags(mass)).tocsc()
    want = np.asarray(scipy.sparse.linalg.spsolve(s, b))
    assert _rel(res.solve(b), want) <= 1e-12


def test_dense_numpy_oracle_both_paths() -> None:
    """Always-on (no scipy): both algorithms vs numpy.linalg.solve."""
    op = _grid_op(6).with_diagonal(np.full(36, 0.25))
    b = _rhs(36)
    want = np.linalg.solve(_dense(op) + 0.75 * np.eye(36), b)
    for solver, method in [("auto", "pcg-ic0"), ("pcg", "pcg-ic0")]:
        res = op.resolvent(lam=0.75, solver=solver, tol=1e-12)
        assert res.method == method
        assert _rel(res.solve(b), want) <= 1e-9
    jac = op.resolvent(lam=0.75, solver="pcg", precond="jacobi", tol=1e-12)
    assert jac.method == "pcg-jacobi"
    assert _rel(jac.solve(b), want) <= 1e-9
    tri = _op_1d(30)
    want_t = np.linalg.solve(_dense(tri), b[:30])
    assert _rel(tri.resolvent(lam=0.0).solve(b[:30]), want_t) <= 1e-12


def test_solve_batched_and_info() -> None:
    n, nc = 120, 5
    op = _op_1d(n)
    res = op.resolvent(lam=0.5)
    cols = np.random.default_rng(9).uniform(-1.0, 1.0, (n, nc))
    out = res.solve_batched(cols)
    assert out.shape == (n, nc)
    for c in range(nc):
        assert np.array_equal(out[:, c], res.solve(np.ascontiguousarray(cols[:, c])))
    x, iters, rel_res = res.solve_info(cols[:, 0].copy())
    assert np.array_equal(x, out[:, 0])
    assert iters == 0
    assert rel_res <= 1e-12
    pcg = _grid_op(8).with_diagonal(np.full(64, 0.3)).resolvent(lam=0.0, tol=1e-12)
    _, iters, rel_res = pcg.solve_info(_rhs(64))
    assert iters > 0
    assert rel_res <= 1e-10


def test_zero_rhs_gives_zero() -> None:
    res = _op_1d(20).resolvent(lam=1.0)
    assert np.array_equal(res.solve(np.zeros(20)), np.zeros(20))


# ---------------------------------------------------------------------------
# Binding parity (pinned against crates/semiflow/tests/spdr_binding_parity.rs)
# ---------------------------------------------------------------------------

PARITY_TRIDIAG = 0x0CFD7D6490FE5AC2
PARITY_PCG = 0xDEB83D114B384D33


def _parity_rhs(n: int) -> np.ndarray:
    return np.array([((i * 7) % 11) / 4.0 - 1.25 for i in range(n)])


def test_binding_parity() -> None:
    n = 32
    cond = lambda i: 1.0 + 0.25 * (i % 5)  # noqa: E731
    op = _op(*_csr_from_rows(_path_rows(n, cond)))
    op = op.with_diagonal(np.array([0.5 + 0.125 * (i % 3) for i in range(n)]))
    mass = np.array([1.0 + 0.5 * (i % 4) for i in range(n)])
    res = op.resolvent(lam=0.5, mass=mass, tol=1e-12)
    assert _fnv(res.solve(_parity_rhs(n))) == PARITY_TRIDIAG

    grid = _grid_op(6).with_diagonal(np.full(36, 0.25))
    pcg = grid.resolvent(lam=0.0, solver="pcg", precond="ic0", tol=1e-12)
    assert _fnv(pcg.solve(_parity_rhs(36))) == PARITY_PCG


# ---------------------------------------------------------------------------
# Operator composition
# ---------------------------------------------------------------------------


def test_to_csr_round_trip() -> None:
    op = _op_1d(40)
    indptr, indices, data = op.to_csr()
    assert indptr.dtype == np.int64
    assert indices.dtype == np.int32
    assert data.dtype == np.float64
    again = semiflow.SymmetricOperator.from_csr(indptr, indices, data, op.n())
    for a, b in zip(again.to_csr(), (indptr, indices, data)):
        assert np.array_equal(a, b)


def test_with_diagonal_matches_dense_and_inserts_missing() -> None:
    # No stored diagonal: with_diagonal must insert it, columns stay sorted.
    rows = [[(1, -0.5)], [(0, -0.5), (2, -0.25)], [(1, -0.25)]]
    op = _op(*_csr_from_rows(rows))
    c = np.array([0.5, 1.0, 2.0])
    got = op.with_diagonal(c)
    assert np.array_equal(_dense(got), _dense(op) + np.diag(c))
    indptr, indices, _ = got.to_csr()
    for i in range(3):
        cols = indices[indptr[i] : indptr[i + 1]]
        assert list(cols) == sorted(cols)
    with pytest.raises(semiflow.SemiflowError) as err:
        op.with_diagonal(np.array([1.0, -1.0, 0.0]))
    assert err.value.kind == "OutOfDomain"
    with pytest.raises(semiflow.SemiflowError):
        op.with_diagonal(np.ones(4))


@needs_scipy
def test_with_diagonal_vs_scipy() -> None:
    op = _grid_op(10)
    c = np.linspace(0.1, 2.0, 100)
    want = (_to_scipy(op) + scipy.sparse.diags(c)).toarray()
    assert np.array_equal(_dense(op.with_diagonal(c)), want)


def test_lumped_congruence_matches_dense() -> None:
    op = _op_1d(25)
    masses = 1.0 + np.linspace(0.0, 3.0, 25)
    d = 1.0 / np.sqrt(masses)
    want = d[:, None] * _dense(op) * d[None, :]
    got = _dense(op.lumped_congruence(masses))
    assert np.max(np.abs(got - want)) <= 1e-14 * np.max(np.abs(want))



# ---------------------------------------------------------------------------
# Rejections (Python mirror of G_SPDR_REJECT)
# ---------------------------------------------------------------------------


def _kind(fn, *args, **kwargs) -> str:
    with pytest.raises(semiflow.SemiflowError) as err:
        fn(*args, **kwargs)
    return err.value.kind


def test_reject_kinds() -> None:
    n = 20
    op = _op_1d(n)
    neumann = semiflow.assemble_conservative_csr_1d(n, 0.0, 1.0, np.ones(n))
    grid = _grid_op(4).with_diagonal(np.full(16, 0.5))
    # Singular: pure Neumann at lam = 0 (constant vector in the null space).
    assert _kind(neumann.resolvent, lam=0.0) == "OutOfDomain"
    # Negative / non-finite shifts and bad mass.
    assert _kind(op.resolvent, lam=-1.0) == "OutOfDomain"
    assert _kind(op.resolvent, lam=float("nan")) == "NanInf"
    assert _kind(op.resolvent, lam=1.0, mass=np.zeros(n)) == "OutOfDomain"
    assert _kind(op.resolvent, lam=1.0, mass=np.ones(n + 1)) == "GridMismatch"
    assert _kind(op.resolvent, lam=1.0, mass=np.full(n, np.nan)) == "NanInf"
    assert _kind(op.resolvent, lam=1.0, tol=0.0) == "OutOfDomain"
    assert _kind(op.resolvent, lam=1.0, tol=1.0) == "OutOfDomain"
    # Forced tridiagonal on a 2-D pattern.
    assert _kind(grid.resolvent, lam=1.0, solver="tridiagonal") == "Unsupported"
    # Unknown menu strings.
    assert _kind(op.resolvent, lam=1.0, solver="lu") == "OutOfDomain"
    assert _kind(op.resolvent, lam=1.0, solver="pcg", precond="ilu") == "OutOfDomain"
    # n = 0 operator.
    empty = np.zeros(1, dtype=np.int64), np.zeros(0, np.int32), np.zeros(0)
    # Relies on the core message "operator dimension n must be >= 1" reaching the shared
    # binding classifier, whose "n must" heuristic maps it to GridMismatch.
    zero = semiflow.SymmetricOperator.from_csr(*empty, 0)
    assert _kind(zero.resolvent, lam=1.0) == "GridMismatch"


def test_reject_solve_inputs() -> None:
    n = 20
    res = _op_1d(n).resolvent(lam=1.0)
    assert _kind(res.solve, np.ones(n + 1)) == "GridMismatch"
    bad = np.ones(n)
    bad[3] = np.nan
    assert _kind(res.solve, bad) == "NanInf"
    assert _kind(res.solve_batched, np.ones((n + 1, 2))) == "GridMismatch"
    # Overflow: the solution is not representable (1x1 operator 1e-200, b = 1e200).
    tiny = semiflow.SymmetricOperator.from_csr(
        np.array([0, 1], dtype=np.int64), np.zeros(1, np.int32), np.array([1e-200]), 1
    )
    assert _kind(tiny.resolvent(lam=0.0).solve, np.array([1e200])) == "NanInf"


def test_pcg_iteration_cap_is_convergence_failed() -> None:
    op = _grid_op(10).with_diagonal(np.full(100, 0.01))
    res = op.resolvent(lam=0.0, solver="pcg", tol=1e-12, max_iter=1)
    assert _kind(res.solve, _rhs(100)) == "ConvergenceFailed"


# ---------------------------------------------------------------------------
# GIL release (ADR-0031)
# ---------------------------------------------------------------------------


def test_gil_released_during_solve() -> None:
    # PCG on a 300x300 grid: ~0.35 s of Rust work for only a few MB of arrays.
    side, nc = 300, 2
    n = side * side
    op = _grid_op(side).with_diagonal(np.full(n, 0.01))
    res = op.resolvent(lam=0.0, tol=1e-12)
    cols = np.random.default_rng(2).uniform(-1.0, 1.0, (n, nc))
    started = threading.Event()
    done: list[np.ndarray] = []

    def work() -> None:
        started.set()
        done.append(res.solve_batched(cols))

    t = threading.Thread(target=work)
    t.start()
    started.wait()
    ticks = 0
    deadline = time.perf_counter() + 0.05
    while time.perf_counter() < deadline:
        ticks += 1
    still_running = t.is_alive()
    t.join()
    assert done and done[0].shape == (n, nc)
    # If the GIL were held through the solve, this thread could not have spun.
    assert ticks > 1000
    assert still_running, "solve finished before the probe window; enlarge the problem"


def test_two_threads_same_result() -> None:
    op = _op_1d(5000)
    res = op.resolvent(lam=0.25)
    b = _rhs(5000)
    want = res.solve(b)
    results: list[np.ndarray] = []

    def work() -> None:
        results.append(res.solve(b))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 2
    assert all(np.array_equal(r, want) for r in results)
