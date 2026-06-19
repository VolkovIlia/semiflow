# ADR-0056 — Order-6 Magnus expansion `MagnusGraphHeat6thChernoff`

- **Status**: ACCEPTED (v2.2 Wave B)
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave B
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0051 (Magnus K=4 graph), ADR-0026 (Generic-over-Float),
  ADR-0046 (precision-policy bands), ADR-0048 (CSR storage).
- **Mathematical foundation**: math.md §16 (CITATION: Blanes-Casas-Oteo-Ros
  2009 *Phys. Rep.* §3-§5 Table 6; Iserles+ 2000 *Acta Numerica* §6;
  NORMATIVE: order-6 abscissae + commutator coefficient table).

## Context

ADR-0051 ships order-4 Magnus on time-dependent graph Laplacian
`L_G(t)`. Order-6 doubles per-step cost but enables ε ≲ 1e-8 accuracy
in 10× fewer steps. Use cases: pedestal-stiff systems (network with
some edges much heavier than others), validation against ode45-style
high-precision ODE benchmarks, long-time horizon (`T_horizon ≫ 1`)
where K=4 accumulates round-off.

The classical order-6 Magnus uses **3-point Gauss-Legendre quadrature**
(GL₆) with abscissae

```text
c₁ = (5 − √15) / 10,    c₂ = 1 / 2,    c₃ = (5 + √15) / 10
b₁ = 5 / 18,            b₂ = 8 / 18,   b₃ = 5 / 18
```

and ONE nested second-order commutator `[A₃, [A₂, A₁]]` (Blanes+ 2009
Table 6; Iserles+ 2000 Table III). The Ω₆ operator is

```text
Ω₆(τ) = τ · (b₁·A₁ + b₂·A₂ + b₃·A₃)
      + (√15 · τ² / 12) · ([A₃, A₂] + [A₂, A₁])
      − (τ² / 12) · [A₃, A₁]
      + (τ³ / 12) · [A₂, [A₃, A₁]]
      + O(τ⁵).
```

The exact coefficient table (Blanes+ 2009 Table 6 — see math.md §16)
includes 3 single-commutator terms and 1 nested commutator.

## Decision

Ship `MagnusGraphHeat6thChernoff<F: SemiflowFloat = f64>` in a NEW
module `crates/semiflow-core/src/magnus6_graph.rs`. Implements
`ChernoffFunction<F, S = GraphSignal<F>>` with `order() == 6`. Its
`apply_into`:

1. **Sample** `L_G(t_*)` at the three GL₆ Gauss-Legendre quadrature
   points `t_i* = τ · c_i` via the caller-supplied closure
   `Box<dyn Fn(F) -> Arc<Laplacian<F>>>`.
2. **Assemble** Ω₆ via 3 weighted accumulation steps + 4 commutator
   evaluations (each commutator = 2 sparse mat-vecs).
3. **Apply** `exp(Ω₆)` to `src` via degree-6 Taylor truncation
   `Σ_{k=0}^{6} Ω₆ᵏ / k!`.
4. **Check** `ρ̄(τ/2) · τ < π/2` (same as K=4; Magnus convergence
   radius is order-independent on bounded generators).

```rust
pub struct MagnusGraphHeat6thChernoff<F: SemiflowFloat = f64> {
    graph: Arc<Graph<F>>,
    lap_at_t: LaplacianAtTime<F>,
    rho_bar_max: F,
    convergence_radius_check: bool,
}

impl<F: SemiflowFloat> MagnusGraphHeat6thChernoff<F> {
    pub fn new(/* same shape as ADR-0051 */) -> Result<Self, SemiflowError>;
    pub fn apply_into_at(/* same shape */) -> Result<(), SemiflowError>;
}

// f64 only: see §"f32 instability rationale" below.
impl ChernoffFunction<f64> for MagnusGraphHeat6thChernoff<f64> {
    type S = GraphSignal<f64>;
    fn order(&self) -> u32 { 6 }
    /* … */
}
```

**No `impl ChernoffFunction<f32> for MagnusGraphHeat6thChernoff<f32>` block.**
Building `MagnusGraphHeat6thChernoff::<f32>::new(...)` compiles (the type
is generic over `F`), but using it as a `ChernoffFunction<f32>` does not
compile — the type-error message refers callers to ADR-0056.

### f32 instability rationale (NORMATIVE)

The Ω₆ coefficient `τ⁴ · √15 / 12 · ‖[A₃, A₂]‖ · ‖[A₂, A₁]‖` is at
`τ = 1e-3`, `‖A_i‖ = O(1)`: ~3.2e-13. The f32 epsilon is 1.2e-7. The
correction term is ~1e6 times smaller than the floating-point
discretisation — it contributes pure noise. Per ADR-0026 precedent
(SIMD f64-only) and ADR-0046 (precision-policy bands), `f32` is OUT of
scope for order-6 Magnus.

Customers wanting f32 should use `MagnusGraphHeatChernoff::<f32>`
(order-4 K=4, slope band ≤ −3.50 per ADR-0046) — that suffices for
ε ~ 1e-6.

## Rationale

- **Order-6 on smooth-time generators.** O(τ⁶) global error vs K=4's
  O(τ⁴) — useful when ε ≲ 1e-8 is required (HPC validation suites,
  high-accuracy network ODE benchmarks).
- **Bounded generator simplifies analysis.** Same as K=4: matrix
  Magnus convergence radius `∫₀ᵗ ‖L_G(s)‖₂ ds < π` is automatic on
  finite-edge-weight graphs. No new convergence theory needed.
- **Reuses sparse mat-vec hot path.** Each commutator-vector product
  `[A_j, A_i] · v = A_j·(A_i·v) − A_i·(A_j·v)` is 2 sparse mat-vecs;
  no dense intermediate. Total per step: 3 + 4·2 + 7 (Taylor degree-6
  on Ω₆·v) = ~18 sparse mat-vecs.
  Compare K=4: 2 + 1·2 + 5 = ~9. **K=6 is ~2× cost per step**.
- **Zero new dependencies.** Reuses `Laplacian::apply_into_slice` and
  scratch ping-pong from ADR-0042.

## Consequences

- `src/magnus6_graph.rs` projected ~580 LoC. **Exceeds 500-LoC cap** —
  joins Override #1 file-list at v2.2.0 cut (math-co-location:
  rustdoc carries Iserles+/Blanes+ Table 6 coefficients inline; the
  K=6 coefficient table is integral to the implementation, splitting
  would scatter the math/code correspondence). Override #1
  EXPANSION; override count stays 3 ≤ 3 per constitution v1.4.0.
- Public surface +1 type. Additive minor bump.

## Acceptance gates

- **G17 slope gate** (NORMATIVE). Time-dependent path graph `P_64`,
  `w(t) = 1 + 0.3·sin(πt)`, `t_final = 0.5`, `n_steps ∈ {5, 10, 20, 40,
  80}`. Self-convergence at 2× refinement. Threshold:
  slope ≤ −5.85 (f64; targeted gate from theoretical O(τ⁶), 0.15 margin).
  **f32 NOT GATED — compile-error path.**
- **T14N_magnus6_residual sympy gate** (NORMATIVE). Verify the GL₆
  abscissae `c_i ∈ {(5 ± √15)/10, 1/2}` and weights from
  Gauss-Legendre on `[0, 1]`. Verify Ω₆ coefficient table matches
  Blanes+ 2009 Table 6 line-by-line on a 4×4 path Laplacian with
  `w(t) = 1 + 0.3·sin(πt)`. Verify
  `‖Ω₆ − Ω_true(τ)‖_F = O(τ⁷)` (matching through τ⁶) via sympy
  series expansion. Pure symbolic; no library runtime.

## Out of scope (v2.2)

- **Order-8 Magnus.** GL₈ + 11 commutators (Blanes+ 2009 Table 7).
  Cost/benefit poor: 3× cost vs K=6 for only 2 orders better at
  ε ~ 1e-12 (unrealistic on f64 round-off). Deferred indefinitely.
- **Adaptive K=4/K=6 switching.** Per-step Lipschitz estimate decides
  K. Worth doing in `AdaptivePI` integration — deferred to v2.3+.
- **f32 K=6 with extended-precision compensated summation.** Algorithmic
  work to recover from coefficient underflow (Kahan summation, etc.) —
  deferred indefinitely (would complicate the K=6 hot path).

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | f32 users attempt K=6, get cryptic compile error | rustdoc on `MagnusGraphHeat6thChernoff` includes `// f64 ONLY — see ADR-0056` banner; type-error message cites ADR. |
| R2 | K=6 with non-smooth `lap_at_t` (caller violates `C³` contract) silently undershoots order | GL₆ requires `C³` data; documented; caller responsibility. |
| R3 | Total commutator-vector cost (4 commutators × 2 mat-vec = 8) hits scratch-pool capacity | Pre-allocate 4 work-buffers in constructor; size `N · sizeof(F)`; zero allocations per step. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/magnus6_graph.rs` | ~580 (Override #1 file-list extension) |
| `tests/g17_magnus6_slope.rs` | ~180 |
| `.dev-docs/verification/scripts/verify_v2_2_magnus6_residual.py` | ~210 |
| math.md §16 | ~150 |
| ADR-0056 (this) | ~200 |
| **Total** | **~1320** |

## References

- A. Iserles, H. Z. Munthe-Kaas, S. P. Nørsett, A. Zanna, *Acta Numerica*
  **9** (2000) §6 — order-6 Magnus expansion via GL₆.
- S. Blanes, F. Casas, J. A. Oteo, J. Ros, *Phys. Rep.* **470** (2009)
  Tables 5–6 — order-4/6/8 Magnus weights.
- M. Hochbruck, A. Ostermann, *Acta Numerica* **19** (2010) §3 —
  evaluation of `exp(Ω)·v` on bounded operators.
- ADR-0051 (Magnus K=4 graph) — predecessor; same convergence-radius
  policy.
