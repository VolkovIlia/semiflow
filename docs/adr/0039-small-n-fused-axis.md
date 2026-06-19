# ADR-0039 — Small-N fused-axis fast path for Strang2D / Strang3D

**Status**: ABORTED 2026-05-20 — fused path is O(τ¹), NOT O(τ²). See §Empirical Finding.

**Context**: Iter-3 bench F5 (2D separable heat) showed scipy-mol-2d 7.3x faster
than Strang2D at N=64 grid sizes. F7 (3D tensor heat) showed scipy-mol-3d 77x
faster at N=32. Root cause: scipy method-of-lines does NOT do full Strang-split on
small grids — it does direct MOL with a single combined-axis stencil iteration. RC's
full Strang composition pays 3 (2D) or 5 (3D) kernel-launch overhead which dominates
at small grid sizes where per-step arithmetic is short. Researcher analysis 2026-05-19
confirmed. Reference: Hundsdorfer & Verwer (2003) §IV.5 MOL vs operator splitting.

**Proposed Decision (v0.13.0 Wave C)**: Add `apply_fused()` private method to Strang2D
/ Strang3D that performs fewer axis passes than the palindromic Strang composition:

- **Strang2D fused**: `Y(τ) ∘ X(τ)` — 2 passes instead of 3-pass palindromic Strang.
- **Strang3D fused**: `Z(τ) ∘ Y(τ) ∘ X(τ)` — 3 passes instead of 5-pass palindromic.

Dispatch in `apply_serial()` was to select fused vs full-Strang by grid node count:
- 2D: `nx * ny < FUSED_THRESHOLD_2D` (= 4096).
- 3D: `nx * ny * nz < FUSED_THRESHOLD_3D` (= 4096).

**Mathematical justification (theoretical)**: For separable generators `L = L_x + L_y`
where `[L_x, L_y] = 0`, the Baker-Campbell-Hausdorff residue is exactly zero:

    e^{τ(L_x + L_y)} = e^{τL_x} e^{τL_y}   (exactly, for all τ)

Both the palindromic Strang and the direct product `X(τ) Y(τ)` should be O(τ²)-accurate
Chernoff approximations of `e^{τL}` for commuting separable operators.

---

## Empirical Finding — ABORT

**Invariant gate `STRANG_FUSED_TAU2_PRESERVATION` FAILED.**

Measured `‖apply_fused(τ,f) − apply_strang2d_full(τ,f)‖_∞` vs τ on N=16 grid (256
nodes, 2D) and N=8 grid (512 nodes, 3D):

| Dimension | Measured slope | Required ≤ | Result  |
|-----------|----------------|------------|---------|
| 2D        | +1.137         | −1.8       | ABORT   |
| 3D        | +1.261         | −1.8       | ABORT   |

The difference grows as O(τ^1.1)–O(τ^1.3), confirming the fused path is **first-order
accurate relative to the palindromic Strang**, not second-order.

**Root cause**: Although `[L_x, L_y] = 0` ensures the exact semigroups commute, the
DiffusionChernoff Chernoff approximation `Φ_x(τ)` is NOT a perfect commutative
approximation to `e^{τL_x}`. The palindromic structure `X(τ/2) ∘ Y(τ) ∘ X(τ/2)`
cancels the leading BCH-like error at the level of the approximation operators; the
direct product `X(τ) ∘ Y(τ)` does not. The BCH cancellation for the exact
semigroup does not transfer to the approximate operators at O(τ²).

---

## Consequences of Abort

- **No fused dispatch in `apply_serial()`** — both Strang2D and Strang3D always
  use the palindromic path regardless of grid size.
- **`apply_fused()` retained as private `#[allow(dead_code)]` method** to document
  the τ-first-order path and allow the regression gate tests to run.
- **Gate tests converted to negative confirmation**:
  - `strang_fused_order_confirmation_2d`: asserts slope > −1.0 (confirms first-order).
  - `strang_fused_order_confirmation_3d`: asserts slope > −1.0 (confirms first-order).
  These gates will FAIL if the fused path ever becomes τ²-accurate, prompting
  re-evaluation of the dispatch.
- **F5/F7 performance gap remains open** — fused-axis approach cannot close it
  without degrading accuracy. Separate approach required (e.g., multi-step batching,
  native MOL mode for small grids, or architecture-level caching).
- **No FUSED_THRESHOLD_* constants** — removed to avoid dead-code confusion.
- **C1 (configurable threshold) shipped unchanged** — `min_points_per_thread()` with
  `REMIZOV_PARALLEL_THRESHOLD` env var override is independent of C2.
- **C3 SKIPPED** — `y_scratch` and `z_scratch` (`Vec::with_capacity`) already present
  in `strang3d_parallel.rs` `apply_parallel`. Verified 2026-05-20.

**v1.0.0 constraint**: `apply` signature and semantics are unchanged; the abort is
internal. Existing slope gates (G3-2D, G3⁴-2D, G3⁶-2D, G5_3D) are unaffected.
