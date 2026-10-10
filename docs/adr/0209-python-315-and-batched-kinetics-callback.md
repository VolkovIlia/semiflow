# ADR-0209 — Python 3.15, PyO3 0.29, and a batched per-stage kinetics callback

- **Status**: Accepted
- **Date**: 2026-10-10
- **Supersedes / amends**: ADR-0179 (callback ABI) — amended for the PyO3 surface;
  ADR-0028 (binding split) unchanged.
- **Contract**: `crates/semiflow-py/tests/test_reaction_diffusion.py`
  (`G_PY_SEMILIN`); CI `py-smoke` matrix.

## Context

- `pyproject.toml` pinned `requires-python < 3.15` and the wheel claimed
  3.10–3.13; the owner's environment runs CPython 3.15. The wheel is built
  against the stable ABI (`abi3-py310`), so one binary loads on every newer
  CPython; what was missing was a PyO3 that knows 3.15, the metadata, and CI.
- ADR-0179 forbids a *per-node* host callback in the integrator hot loop
  (`O(N·steps)` crossings, GIL contention) and blesses a *batched* sampler at
  setup time. ADR-0208's reaction–diffusion systems need user kinetics
  `f(t, x, u)` that depend on the evolving state, so they cannot be sampled
  once at setup.

## Decision

1. **PyO3 / numpy 0.28 → 0.29** (`pyo3 = 0.29.3`, `numpy = 0.29.0`). The
   migration was mechanical: no source change, clippy clean.
2. **Metadata and CI**: `requires-python = ">=3.10,<3.16"`, classifiers 3.14 and
   3.15; the `py-smoke` matrix runs 3.10 / 3.13 / 3.15 (pre-releases allowed until
   `setup-python` ships the final). The wheel stays abi3-py310: one build per
   platform serves 3.10–3.15.
3. **Batched per-stage callback (amends ADR-0179 for PyO3).** A Python kinetics
   callable is invoked ONCE per Runge–Kutta stage with ALL nodes:
   `f(t, x[(dim, N)], u[(K, N)]) -> du[(K, N)]`. Crossings are `O(4r·steps)`,
   independent of `N`; NumPy does the per-node work. The GIL is released for
   the whole evolution and re-acquired only inside the callback; a Python
   exception is stored and re-raised verbatim after the Rust loop stops, and the
   Python object's state is then left unchanged. The per-node prohibition of
   ADR-0179 stands.

Rejected: staying on PyO3 0.28 (works through abi3, but 0.29 is the version
that targets 3.15 and the bump was free); per-version (non-abi3) wheels
(more CI for no runtime gain); a per-node callback (ADR-0179).

## Consequences

Tested locally: the full fast Python suite on CPython 3.13.16 and on CPython
3.15.0b4 (numpy 2.5.4), including `test_reaction_diffusion.py`. A Gray–Scott
system on 64² nodes, 200 steps: 0.94 s; a callable `K = 2` system on 401 nodes
with two Richardson levels: 22 ms.

## Honest limits

- Python 3.15 was tested on 3.15.0b4 here; the final release is covered by
  the CI matrix entry.
- The callback path is still Python-speed per stage; for very small grids the
  crossing overhead (`~10 µs`) dominates and the built-in kinetics are faster.
