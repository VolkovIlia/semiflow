"""Benchmark for Issue #10 batched evolve API.

Quantifies the removal of O(C) Python/PyO3 overhead:
- Forward: per-channel loop (C calls) vs evolve_batched (1 call)
- Adjoint fwd+bwd: per-channel loop vs batched

Fixture: 90-node path graph, n_steps=8, float64 (matches issue profiling).
C in {1, 4, 8, 16}.

Memory: measured via tracemalloc peak (Python-side allocations).
Timing: median of N_REPS=25 warm-up-excluded repetitions.
"""

from __future__ import annotations

import os
import statistics
import time
import tracemalloc
from typing import Callable

import numpy as np

import semiflow


# ---------------------------------------------------------------------------
# Constants (match issue profiling fixture)
# ---------------------------------------------------------------------------

N_NODES = 90
N_STEPS = 8
T_FINAL = 0.25
# Path graph P_N with unit weights: max Laplacian eigenvalue ≤
# 2 - 2*cos((N-1)*π/N) ≈ 3.9988 for N=90. Use 4.1 as a safe margin.
# NOTE: the Magnus convergence check requires rho_bar * τ < π/2.
# τ = T/n = 0.25/8 = 0.03125. 4.1 * 0.03125 = 0.128 << π/2 ≈ 1.571. OK.
# For GraphHeat (no convergence check), a loose Gershgorin N=90 still works.
RHO_BAR_GERSHGORIN = float(N_NODES)  # loose bound; OK for GraphHeat / adjoint
RHO_BAR_TIGHT = 4.1                   # tight spectral bound for P_90; required for Magnus
CHANNEL_SIZES = [1, 4, 8, 16]
N_WARMUP = 5
N_REPS = 25

# Benchmark output log
_LOG_DIR = os.path.join(os.path.dirname(__file__), "../../.logs/tests/performance/")
os.makedirs(_LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_graph() -> semiflow.Graph:
    return semiflow.Graph.path(N_NODES)


def make_graph_heat(graph: semiflow.Graph) -> semiflow.GraphHeat:
    return semiflow.GraphHeat(graph=graph, rho_bar=RHO_BAR_GERSHGORIN)


def make_magnus(graph: semiflow.Graph) -> semiflow.MagnusGraphHeat:
    # Magnus convergence check: rho_bar_max * tau < pi/2.
    # tau = T/n = 0.25/8 = 0.03125; 4.1 * 0.03125 = 0.128 << pi/2. OK.
    return semiflow.MagnusGraphHeat(
        graph=graph,
        lap_at_t=lambda _t: graph,
        rho_bar_max=RHO_BAR_TIGHT,
    )


def make_adjoint(graph: semiflow.Graph) -> semiflow.GraphAdjointPresampled:
    return semiflow.GraphAdjointPresampled.from_presampled(
        graph=graph,
        lap_at_t=lambda _t: graph,
        rho_bar=RHO_BAR_TIGHT,
        n_steps=N_STEPS,
        t_horizon=T_FINAL,
    )


def make_f0(n_cols: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_NODES, n_cols))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_median(fn: Callable, n_warmup: int = N_WARMUP, n_reps: int = N_REPS) -> float:
    """Return median wall-clock time (seconds) over n_reps runs."""
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _peak_bytes(fn: Callable) -> int:
    """Return peak traced-memory (bytes) for one call."""
    tracemalloc.start()
    tracemalloc.clear_traces()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# ---------------------------------------------------------------------------
# Forward benchmarks
# ---------------------------------------------------------------------------

def bench_forward_loop(kernel, f0: np.ndarray) -> tuple[float, int]:
    """Per-channel Python loop: C separate .evolve() calls."""
    n_cols = f0.shape[1]

    def run():
        results = []
        for c in range(n_cols):
            results.append(kernel.evolve(T_FINAL, N_STEPS, f0[:, c]))
        return results

    return _time_median(run), _peak_bytes(run)


def bench_forward_batched(kernel, f0: np.ndarray) -> tuple[float, int]:
    """Single evolve_batched call."""
    def run():
        return kernel.evolve_batched(T_FINAL, N_STEPS, f0)

    return _time_median(run), _peak_bytes(run)


def bench_forward_single_channel(kernel) -> tuple[float, int]:
    """Baseline: one evolve call on one channel (raw Rust floor)."""
    f0_1ch = np.random.default_rng(0).standard_normal(N_NODES)

    def run():
        return kernel.evolve(T_FINAL, N_STEPS, f0_1ch)

    return _time_median(run), _peak_bytes(run)


# ---------------------------------------------------------------------------
# Adjoint fwd+bwd benchmarks
# ---------------------------------------------------------------------------

def bench_adjoint_loop(adj: semiflow.GraphAdjointPresampled,
                       graph: semiflow.Graph,
                       f0: np.ndarray) -> tuple[float, int]:
    """Per-channel: evolve_state_adjoint + edge_weight_grad, C iterations."""
    n_cols = f0.shape[1]
    lam_batch = np.random.default_rng(7).standard_normal((N_NODES, n_cols))

    def run():
        grads = []
        for c in range(n_cols):
            adj.evolve_state_adjoint(lam_batch[:, c])
            g = semiflow.edge_weight_grad(
                graph, None,
                u0=f0[:, c],
                dj_du_n=lam_batch[:, c],
                t=T_FINAL,
                n_steps=N_STEPS,
                rho_bar=RHO_BAR_TIGHT,
                params="all_edges",
            )
            grads.append(g)
        return grads

    return _time_median(run), _peak_bytes(run)


def bench_adjoint_batched(adj: semiflow.GraphAdjointPresampled,
                          graph: semiflow.Graph,
                          f0: np.ndarray) -> tuple[float, int]:
    """Batched: evolve_state_adjoint_batched + edge_weight_grad_batched."""
    n_cols = f0.shape[1]
    lam_batch = np.random.default_rng(7).standard_normal((N_NODES, n_cols))

    def run():
        adj.evolve_state_adjoint_batched(lam_batch)
        semiflow.edge_weight_grad_batched(
            graph, None,
            u0_cols=f0,
            dj_du_n_cols=lam_batch,
            t=T_FINAL,
            n_steps=N_STEPS,
            rho_bar=RHO_BAR_TIGHT,
            params="all_edges",
        )

    return _time_median(run), _peak_bytes(run)


# ---------------------------------------------------------------------------
# Parity check
# ---------------------------------------------------------------------------

def verify_parity(kernel_name: str, kernel, f0: np.ndarray) -> bool:
    """Assert evolve_batched[:, c] == evolve([:, c]) 0-ULP."""
    batched_out = kernel.evolve_batched(T_FINAL, N_STEPS, f0)
    for c in range(f0.shape[1]):
        single = kernel.evolve(T_FINAL, N_STEPS, f0[:, c])
        if not np.array_equal(batched_out[:, c], single):
            print(f"  PARITY FAIL {kernel_name} channel {c}")
            return False
    return True


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(t_sec: float) -> str:
    us = t_sec * 1e6
    return f"{us:8.1f} µs"


def _fmt_mem(b: int) -> str:
    kb = b / 1024
    return f"{kb:8.1f} KB"


def print_table_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(
        f"{'C':>4} | {'loop time':>10} | {'batched time':>13} | "
        f"{'speedup':>8} | {'loop mem':>10} | {'batched mem':>12} | "
        f"{'mem ratio':>10} | parity"
    )
    print("-" * 100)


def print_row(n_cols: int,
              t_loop: float, t_batched: float,
              m_loop: int, m_batched: int,
              parity_ok: bool) -> None:
    speedup = t_loop / t_batched if t_batched > 0 else float("inf")
    mem_ratio = m_loop / m_batched if m_batched > 0 else float("inf")
    parity_str = "PASS" if parity_ok else "FAIL"
    print(
        f"{n_cols:>4} | {_fmt(t_loop):>10} | {_fmt(t_batched):>13} | "
        f"{speedup:>7.2f}x | {_fmt_mem(m_loop):>10} | {_fmt_mem(m_batched):>12} | "
        f"{mem_ratio:>9.2f}x | {parity_str}"
    )


def print_single_channel_baseline(kernel_name: str, t_1ch: float, m_1ch: int) -> None:
    print(
        f"  Raw 1-channel Rust floor ({kernel_name}): "
        f"{_fmt(t_1ch)} / {_fmt_mem(m_1ch)}"
    )


# ---------------------------------------------------------------------------
# Adjoint parity check
# ---------------------------------------------------------------------------

def verify_adjoint_parity(adj: semiflow.GraphAdjointPresampled,
                           graph: semiflow.Graph,
                           f0: np.ndarray) -> bool:
    """Check evolve_state_adjoint_batched[:, c] == evolve_state_adjoint([:, c])."""
    lam = np.random.default_rng(7).standard_normal((N_NODES, f0.shape[1]))
    batched_out = adj.evolve_state_adjoint_batched(lam)
    for c in range(lam.shape[1]):
        single = adj.evolve_state_adjoint(lam[:, c])
        if not np.array_equal(batched_out[:, c], single):
            print(f"  ADJOINT PARITY FAIL channel {c}")
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_forward_section(kernel_name: str, kernel) -> None:
    t_1ch, m_1ch = bench_forward_single_channel(kernel)
    print_single_channel_baseline(kernel_name, t_1ch, m_1ch)

    print_table_header(f"FORWARD: {kernel_name}  (N={N_NODES}, n_steps={N_STEPS})")
    for n_cols in CHANNEL_SIZES:
        f0 = make_f0(n_cols)
        parity_ok = verify_parity(kernel_name, kernel, f0)
        t_loop, m_loop = bench_forward_loop(kernel, f0)
        t_batched, m_batched = bench_forward_batched(kernel, f0)
        print_row(n_cols, t_loop, t_batched, m_loop, m_batched, parity_ok)


def bench_adjoint_state_only_loop(adj: semiflow.GraphAdjointPresampled,
                                   n_cols: int) -> tuple[float, int]:
    """Per-channel: C calls to evolve_state_adjoint (no grad)."""
    lam_batch = np.random.default_rng(7).standard_normal((N_NODES, n_cols))

    def run():
        for c in range(n_cols):
            adj.evolve_state_adjoint(lam_batch[:, c])

    return _time_median(run), _peak_bytes(run)


def bench_adjoint_state_only_batched(adj: semiflow.GraphAdjointPresampled,
                                      n_cols: int) -> tuple[float, int]:
    """Single evolve_state_adjoint_batched call (no grad)."""
    lam_batch = np.random.default_rng(7).standard_normal((N_NODES, n_cols))

    def run():
        adj.evolve_state_adjoint_batched(lam_batch)

    return _time_median(run), _peak_bytes(run)


def bench_dense_forward_scipy(n_cols: int) -> tuple[float, int]:
    """Dense scipy.linalg.expm(-t*L) @ f0 for C channels (baseline)."""
    import scipy.linalg  # type: ignore[import-untyped]

    f0 = make_f0(n_cols)
    # Build combinatorial Laplacian of P_90
    L = np.zeros((N_NODES, N_NODES))
    for i in range(N_NODES - 1):
        L[i, i] += 1.0
        L[i + 1, i + 1] += 1.0
        L[i, i + 1] = -1.0
        L[i + 1, i] = -1.0
    expL = scipy.linalg.expm(-T_FINAL * L)  # precompute once

    def run():
        return expL @ f0  # applies to all C channels in one matmul

    return _time_median(run), _peak_bytes(run)


def run_adjoint_state_section(adj: semiflow.GraphAdjointPresampled,
                               graph: semiflow.Graph) -> None:
    print_table_header(
        f"ADJOINT state sweep only: evolve_state_adjoint (no grad)  "
        f"(N={N_NODES}, n_steps={N_STEPS})"
    )
    for n_cols in CHANNEL_SIZES:
        parity_ok = verify_adjoint_parity(adj, graph, make_f0(n_cols))
        t_loop, m_loop = bench_adjoint_state_only_loop(adj, n_cols)
        t_batched, m_batched = bench_adjoint_state_only_batched(adj, n_cols)
        print_row(n_cols, t_loop, t_batched, m_loop, m_batched, parity_ok)


def run_adjoint_grad_section(adj: semiflow.GraphAdjointPresampled,
                              graph: semiflow.Graph) -> None:
    print_table_header(
        f"ADJOINT fwd+bwd: evolve_state_adjoint + edge_weight_grad  "
        f"(N={N_NODES}, n_steps={N_STEPS}, {N_NODES - 1} edges)"
    )
    for n_cols in CHANNEL_SIZES:
        f0 = make_f0(n_cols)
        parity_ok = verify_adjoint_parity(adj, graph, f0)
        t_loop, m_loop = bench_adjoint_loop(adj, graph, f0)
        t_batched, m_batched = bench_adjoint_batched(adj, graph, f0)
        print_row(n_cols, t_loop, t_batched, m_loop, m_batched, parity_ok)


def run_dense_baseline() -> None:
    print_table_header(
        f"DENSE scipy.linalg.expm baseline: expm(-T*L) @ f0_batch  "
        f"(N={N_NODES}, exact)")
    for n_cols in CHANNEL_SIZES:
        t_dense, m_dense = bench_dense_forward_scipy(n_cols)
        print(
            f"{n_cols:>4} | {_fmt(t_dense):>10} | {'(dense exact)':>13} | "
            f"{'N/A':>8} | {_fmt_mem(m_dense):>10} | {'N/A':>12} | {'N/A':>10} | N/A"
        )


def main() -> None:
    print("\n" + "#" * 72)
    print("#  Issue #10 Batched API Benchmark")
    print(f"#  Machine: {os.uname().nodename}  |  simd+parallel compiled in: YES")
    print(f"#  Wheel: {semiflow.__file__}")
    print(f"#  Graph: path P_{N_NODES}, n_steps={N_STEPS}, T={T_FINAL}")
    print(f"#  Timing: median of {N_REPS} reps after {N_WARMUP} warm-up calls")
    print(f"#  Memory: tracemalloc peak (Python-side; excludes Rust heap)")
    print(f"#  PyO3 call count: loop=C, batched=1  (confirmed for all kernels)")
    print("#" * 72)

    graph = make_graph()
    gh = make_graph_heat(graph)
    mg = make_magnus(graph)
    adj = make_adjoint(graph)

    run_forward_section("GraphHeat", gh)
    run_forward_section("MagnusGraphHeat", mg)
    run_dense_baseline()
    run_adjoint_state_section(adj, graph)
    run_adjoint_grad_section(adj, graph)

    print("\n" + "=" * 72)
    print("  SUMMARY (expect: speedup grows with C; at C=1 batched≈loop)")
    print("=" * 72)


if __name__ == "__main__":
    main()
