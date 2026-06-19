# ADR-0031 — PyO3 GIL-release boundary in `Heat1D::evolve`

**Status**: Accepted (v0.11.0 contract, item I6)
**Date**: 2026-05-09
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0028 (Wave B — `semiflow-py` baseline, profile-release-ffi
panic-unwind invariant), ADR-0029 (v0.11.0 milestone — I6 MUST), constitution
§"Project-Specific Principles" #3 (SIMD bit-equality release-blocking)

## Context

`semiflow-py` v0.10.0 holds the GIL through the entire body of `Heat1D.evolve()`,
freezing concurrent Python threads (Jupyter UI callbacks, ThreadPoolExecutor
workers) for the duration of a long simulation (`n_steps ≥ 10⁴`). Persona P2
(`acceptance.md` §1) needs the inner Rust evolve loop to release the GIL so
other Python threads make progress. Three CRITICAL findings from
`clarity-scan.md` shape the design: F1.2 (release scope is the **inner pure-Rust
loop only**, NOT NumPy reads/writes — resolved), F4.1 (≤2% single-thread
regression budget vs v0.10.0 baseline measured by criterion — resolved),
F6.2 (GIL release × Python signal handler interaction must be verified, with
a Ctrl+C-during-evolve test case — resolved). Additionally, the v0.10.0
cross-validation invariant (Wave B sup_error 1.46e-6 matches Wave A at 3-digit
precision, Δ = 3.02e-10 sub-ULP — recorded in ADR-0028 Amendment 1 / project
memory `v0.10.0 Wave B PyO3 shipped`) MUST remain byte-identical (Risk R8):
GIL release wraps execution scheduling, not numerical kernels.

## Decision

`Heat1D::evolve` in `semiflow-py` is restructured into three phases bounded by
a single `py.allow_threads(|| { ... })` block: **(1) pre-flight under GIL**
— extract the Rust-owned `ChernoffSemigroup` state from the `Bound<'py, PyAny>`,
copy/borrow the input NumPy buffer into an owned `Vec<f64>` (`PyArrayMethods::to_vec`
or `as_slice` if read-only and contiguous), validate `n_steps`, `t`; **(2)
GIL-released compute** — call the pure-Rust `ChernoffSemigroup::evolve(state,
t, n_steps)` inside `py.allow_threads`, returning an owned `Vec<f64>` of
identical layout to the input; **(3) post-flight under GIL** — convert the
result `Vec<f64>` back into a `Bound<PyArray1<f64>>` and return. The
`ChernoffSemigroup` and its constituent state (already `Send + Sync` per
generic-over-Float refactor ADR-0026 — verified at PR time by reviewer-suckless
running `cargo build -p semiflow-py --features no-gil-test`; if not auto-`Send`,
the engineer adds an explicit `unsafe impl Send` with a rustdoc citing the
"no interior mutability, no GIL-bound resource" invariant) cross the
`allow_threads` boundary. Signal handling: PyO3 documents that signals
delivered during `allow_threads` are queued and surface on GIL re-acquisition;
the test suite gains `tests/test_heat.py::test_evolve_handles_sigint` which
spawns a long evolve, sends `SIGINT` mid-flight, and asserts a `KeyboardInterrupt`
is raised after the call returns to GIL-held code. **Performance budget**:
`cargo run -p xtask -- py-bench` (added in v0.11.0) compares single-thread
`evolve` runtime under GIL-release vs the v0.10.0 baseline; regression
> 2% blocks merge (clarity-scan F4.1). **Correctness budget**: existing
sup_error gate (1.46e-6) and cross-validation Δ (3.02e-10 vs Wave A) MUST
remain byte-identical — `tests/test_heat.py::test_cross_validation_wave_a`
asserts `np.array_equal` (not `allclose`) against the unchanged Wave A
reference vector.

## Consequences

- **Pro**: persona P2 unblocked — concurrent Python threads make progress
  during long evolve calls; aligns `semiflow-py` with the standard PyO3
  pattern for CPU-bound work.
- **Pro**: numerical kernels untouched — sup_error and cross-validation Δ
  preserved by construction (the Rust call signature and body are
  unchanged; only the surrounding GIL-management changes).
- **Pro**: signal-handling test surfaces a class of pre-existing bugs
  (Ctrl+C swallowed during long evolve in v0.10.0) — incidentally improves
  v0.11.0's interactive UX beyond the strict GIL goal.
- **Con**: pre-flight buffer copy adds one allocation + memcpy on every
  `evolve` call; for short calls (`n_steps < 10`), the copy may exceed
  the compute time. Acceptable: the 2% regression budget is measured at
  the persona P2 use-case scale (`n_steps ≥ 10⁴`).
- **Follow-up**: v0.12.0 may extend the pattern to 2D/3D bindings (I4/I5)
  trivially — the `allow_threads` block grows to cover the larger evolve;
  no new design.
- **Follow-up**: if F4.1 budget is exceeded (>2% regression), engineer
  scopes-back the `allow_threads` window or ships I6 behind a feature flag
  for v0.11.0; do not relax the budget without a follow-up ADR.

## Alternatives Considered

- **Wrap the entire `evolve()` body (including NumPy I/O) in `allow_threads`**
  — rejected (F1.2): NumPy buffer access from a non-GIL-holding thread is
  unsound and triggers PyO3 runtime panics; only Rust-owned data may cross
  the `allow_threads` boundary.
- **Async `evolve()` using `pyo3-async-runtimes`** — rejected (deferred to
  v0.12.0 I14 per `acceptance.md` §6): adds a tokio dep to `semiflow-py`,
  changes the calling convention from sync to `await`-required, breaks
  v0.10.0 user code. Re-evaluate after I6 telemetry.
- **In-place evolve (no copy of input array)** — rejected: requires `unsafe`
  pointer arithmetic into NumPy storage from a non-GIL thread; the safety
  proof is brittle (NumPy may reallocate under GC pressure) and not worth
  the saved allocation for our `n_steps ≥ 10⁴` target use case.
- **Release the GIL but require user to call `Heat1D.evolve_threadsafe()`
  explicitly** — rejected: doubles the API surface for a behaviour that
  should be transparent to the user; contradicts persona P2's IDE-autocomplete
  goal (I7).
