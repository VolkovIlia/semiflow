# ADR-0061 — Python parity expansion (v2.3)

- **Status**: ACCEPTED (v2.3 wave)
- **Date**: 2026-05-22
- **Wave**: v2.3 feat/python-parity-v2.3
- **Authors**: agentic-engineer
- **Depends on**: ADR-0028 (FFI/PyO3/WASM v0.10.0 binding infrastructure),
  ADR-0031 (PyO3 GIL-release three-phase pattern), ADR-0034 (closure-based
  variable-a API), ADR-0035 (lockstep SemVer), ADR-0044 (generic
  `AdaptivePI<C,F,K>` step controller), ADR-0045 (zero-copy bindings),
  ADR-0055 (`AdjointChernoff` wrapper), ADR-0057 (Schrödinger real-only
  Option A), ADR-0059 (graph bindings, v2.2).
- **Supersedes / amends**: nothing. Extends ADR-0059 coverage to the full
  Rust core surface.
- **Mathematical foundation**: math.md §17 (Schrödinger real-only Option A,
  palindromic Strang), §11 (Strang composition and 2D operators), §14 (variable
  graph topology and variable-a graph kernel). No new math; bindings are
  surface-level.

## Context

Prior to v2.3, `semiflow-py` exposed seven public symbols (`Heat1D`, `Heat2D`,
`Heat3D`, `GraphPath`, `GraphHeat`, `MagnusGraphHeat`, `version`). The Rust core
offered ≈20 Chernoff kernels, four boundary policies, closure-based variable
coefficients, `SchrodingerChernoff`, `AdjointChernoff`, `AdaptivePI`, and an
expanded graph surface (`GraphHeat4thChernoff`, `MagnusGraphHeat6thChernoff`,
`VarCoefGraphHeatChernoff`, `Graph.cycle`, `Graph.from_edges`, `Graph.erdos_renyi`,
`Laplacian.normalized`). Python users could not access any of these features
without dropping into Rust, creating a significant friction barrier for the
primary target audience (data scientists and computational physicists).

## Decision

Bring `semiflow-py` to full parity with the Rust core surface in a single
feature wave on branch `feat/python-parity-v2.3`, organised into five phases:

1. **Boundary kwarg + pre-sampled coefficients** — thread a `boundary` string
   kwarg (`'reflect'`/`'periodic'`/`'zero'`/`'linear'`, default `'reflect'`) into
   all existing 1D/2D/3D constructors via a single `parse_boundary()` function in
   `boundary.rs`. Add `Heat1D.with_a_array(a_values)` as a pre-sampled coefficient
   staticmethod: the numpy array is cloned to an owned `Arc<Vec<f64>>` once, then
   wrapped in a cubic-Hermite Rust closure with zero GIL re-acquires during
   `evolve`. The `with_a_function` callback API is preserved (backwards compat).

2. **1D kernel parity** — expose `Heat1D4th`, `Heat1D6th` (4th/6th-order spatial,
   ADR-0014/0015), `DriftReaction1D` (`∂_t u = b(x)·∂_x u + c(x)·u`), and `Shift1D`
   (universal `a(x)∂²+b(x)∂+c(x)`). Each gets a scalar-default constructor and a
   `with_arrays` staticmethod for pre-sampled coefficients. New Rust-core
   constructor overloads added additively per ADR-0034 pattern:
   `Diffusion4thChernoff::with_closure`, `Diffusion6thChernoff::with_closure`,
   `DriftReactionChernoff::with_closure`.

3. **Schrödinger** — expose `Schrodinger1D` wrapping `SchrodingerChernoff<f64>` +
   `SchrodingerState<f64>` (ADR-0057, math.md §17). Four constructors handle
   scalar V, pre-sampled V, and complex initial state via real+imag parts.

4. **Composition and adaptive wrappers** — expose `NonSeparable2D` (wraps
   `NonSeparableMixedChernoff`, ADR-0058), `Adjoint` (wraps `AdjointChernoff`,
   ADR-0055), and `AdaptivePI` (wraps `AdaptivePI<C,f64,ClassicalPI>`, ADR-0044).
   Both `Adjoint` and `AdaptivePI` use a 5-variant enum dispatch in Rust over
   `(Heat1D, Heat1D4th, Heat1D6th, DriftReaction1D, Shift1D)` to avoid exposing
   Rust generics to Python.

5. **Graph expansion** — add `Graph.cycle(n)`, `Graph.from_edges(n_nodes, edges)`,
   `Graph.erdos_renyi(n, p, seed)` factories; add `Laplacian.normalized(graph)`;
   expose `GraphHeat4th` (ADR-0051), `VarCoefGraphHeat` (ADR-0053),
   `MagnusGraphHeat6` (ADR-0056). Existing `GraphHeat` and `MagnusGraphHeat`
   extended to accept a `Laplacian` directly. `GraphPath` retained as deprecated
   alias for `Graph.path(n)`.

The lockstep version bump (all four crates to 2.3.0) is deferred to Phase 7
(final acceptance) per ADR-0035.

## Mathematical Foundation

No new mathematics is introduced. The Python surface re-exposes existing
Rust implementations:

- Schrödinger palindromic Strang: math.md §17, ADR-0057.
- 2D anisotropic non-separable operator: math.md §10.7-ter, ADR-0058.
- Adjoint backward semigroup: math.md §15 (CITATION: Pazy 1983 §1.10), ADR-0055.
- Graph PDE 4th-order / Magnus K=6: math.md §12.7 / §16, ADR-0051 / ADR-0056.
- Variable-a graph Laplacian: math.md §14, ADR-0053.

## Consequences

**Positive**

- Python users reach full Rust-core parity; no feature is Rust-only except
  `TruncatedExp*`, `StrangSplitGraph`, `GraphTraj`, and the `f32` path.
- Pre-sampled coefficient path is approximately 10× faster than the Python
  callback path (zero GIL re-acquires; measured in Phase 2 benchmarks).
- String boundary API requires no type imports; `boundary='periodic'` is
  discoverable from autocomplete and error messages list valid values.
- Documentation surface (coverage matrix, audit findings, ADR) is aligned with
  the implementation in the same PR, satisfying the "documentation in lockstep"
  principle.

**Negative**

- The 5-variant enum dispatch in `Adjoint` and `AdaptivePI` is rigid: adding a
  sixth inner kernel (e.g. `Schrodinger1D`) requires touching the enum in
  `adjoint.rs` and `adaptive.rs`. This is a bounded cost (≤10 LoC per new
  variant) and was accepted over the alternative of a dynamic dispatch trait
  object, which would complicate the GIL-release three-phase pattern.
- Phase 5 graph expansion does not expose `GraphTraj` (see Gaps section of
  `docs/python-coverage.md`); callers who need trajectory-driven graph PDE
  must use Rust directly.

**Neutral**

- Backwards compatibility is fully preserved. The `boundary` kwarg has a default
  (`'reflect'`), so existing `Heat1D(xmin, xmax, n, u0)` call sites continue to
  compile and produce identical results. `GraphPath` is deprecated but still
  functional.
- The lockstep version bump to 2.3.0 is version-bump-only for FFI and WASM
  (no surface changes in those crates in this wave).

## Alternatives Considered

- **Callable-only variable-coefficient API** — accept a Python callable instead
  of a numpy array. Rejected: each Chernoff step re-acquires the GIL to call
  Python, producing approximately 2–5 µs overhead per step (measured in Phase 1
  experiments). The pre-sampled array path avoids this entirely.
- **Per-kernel `Adjoint` / `AdaptivePI` subclasses** — expose `AdjointHeat1D`,
  `AdjointHeat1D4th`, etc. as separate pyclasses. Rejected: combinatorial
  explosion (5 × 2 = 10 wrapper classes for `Adjoint` alone); enum dispatch
  achieves the same coverage in ≤50 LoC.
- **Drop legacy `GraphPath` immediately** — remove `GraphPath` in favour of
  `Graph.path(n)`. Rejected: breaks existing user code with no migration period;
  deprecation plus removal at v3.0 is the correct SemVer cycle per
  `docs/api-stability.md` §5.

## Acceptance Gates

- `cargo run -p xtask -- py-smoke` — 129 Python tests pass (Phase 0: 19 → Phase
  5: 129; +110 tests over the branch).
- `cargo run -p xtask -- check-lints` — no suckless violations (≤500 LoC per
  file, ≤50 LoC per function; no new GRANDFATHERED entries).
- Cross-binding parity gate (to be verified in Phase 7): Python vs FFI sup-error
  ≤ 3 ULP for matching inputs per ADR-0059 §"Cross-binding sup-error gate".
- Unitarity gate (`Schrodinger1D`): `‖ψ‖²/‖ψ₀‖² − 1 < 1e-6` over 500 steps.
- Convergence-rate gates: 4th-order slope ≤ −3.5, 6th-order slope ≤ −5.5
  (Phase 2 gates; passing at dc45181).
