# ADR-0188 — Implicit / unconditionally-stable stiff high-contrast multilayer diffusion (per-layer ρc mass-weighting + stack assembly)

- **Status**: Proposed (design only — Issue #14; branch `issue-14-implicit-stiff-multilayer`)
- **Date**: 2026-06-27
- **Supersedes**: none — purely ADDITIVE. Reuses ADR-0187/§56
  (`assemble_conservative_csr_1d`, harmonic-mean faces → symmetric `A=−L_k`),
  ADR-0186/§55 (`SymmetricOperator::from_csr`, `mass_lumped_evolve` — the lumped
  `(M,K)` path), ADR-0185/§54 (A1 Krylov exact action `e^{−τA}v`, now with the
  Chebyshev-substep + fail-loud stiff fix, commit `9e5f557`), and ADR-0125/§45
  (`mat_exp_pade13` dense oracle). No existing kernel, gate, public signature, or
  0-ULP scope is changed.
- **Contract**: `contracts/semiflow-core.math.md` §57 (new NORMATIVE section);
  gates `G_TPS_MASS_WEIGHT`, `G_TPS_UNITMASS_FAILS`, `G_TPS_STACK_ACCEPTANCE`,
  `G_TPS_STIFF_STEPCOUNT`.

## Hypothesis-test result (FRONT AND CENTRE)

The issue suggests building a new Cayley/CN variable-coefficient solver. **The
hypothesis test refutes that need.** The inherited machinery already provides an
EXACT, unconditionally-stable, variable-coefficient path; the only real gap is
**per-layer ρc mass-weighting** (volumetric heat capacity), which the §55 lumped
`(M,K)` path already consumes. Tested by analysis of the actual assembly
convention plus a throwaway numerical probe on the real LI-900/SIP/RTV/Al-2024
Shuttle-TPS stack (`dx=0.5 mm`, `N=153`, 2500 s, k-contrast ≈ 3025×, ρc-contrast
≈ 27×, λ_max(M⁻¹A) ≈ 794 s⁻¹):

| Question | Finding (measured) |
|----------|--------------------|
| Does `ConservativeDiffusionChernoff` (#11) handle per-layer ρc? | **NO — unit mass only.** It propagates `∂ₜu = L_k u`. Run as-is on the stack it saturates to the 1500 K hot BC (T_Al = **1500 K** vs correct 336.7 K, rel-err **3169 %**). Mass-weighting is unavoidable. |
| Does the assembled `A=−L_k` + per-node ρc reproduce the physics? | **YES, exactly.** `A` is node-centered (`A[i,i±1]=−T_{i±½}/dx`, `T=k_harm/dx`), so `ρc·∂ₜT=∂ₓ(k∂ₓT)` ⇒ `∂ₜT = −diag(ρc)⁻¹A·T`. This is **literally** `mass_lumped_evolve(K=A, masses=ρc_node)` (§55.3). Small-N congruence sup-err = **6.1e-16**. |
| Does the stable path match a CN reference within 2–3 %? | **YES, far inside.** vs a fine independent CN reference: `mass_lumped_evolve` (one Krylov action over the whole 2500 s) rel-err **6e-10**; coarse 400-step mass-weighted CN rel-err **9e-7**. |
| Does it beat explicit CFL (overcome the stiffness)? | **YES, by ~10³×.** Explicit-Euler stability needs `⌈τλ_max⌉ ≈ 1.98 M` steps; Lanczos needs ≈ `√(τλ_max)` ≈ **1409 matvecs** (1409× fewer); A-stable CN needs ≈ **400** accuracy steps (4962× fewer). |

**Conclusion — chosen path:** *no new numerical method.* #14 = (i) reuse #11's
assembler + #13's `mass_lumped_evolve` with `masses = ρc_node` (PRIMARY, exact,
depth-flat — one action for the entire re-entry), with a thin **multilayer stack
assembly helper** mapping per-layer `(thickness, k, ρc)` → node arrays; (ii) the
TPS acceptance + stiffness gates. An optional mass-weighted A-stable CN convenience
covers the no-Krylov-memory regime (tiny: add `diag(ρc)` to #11's tridiagonal). The
issue's suggested Cayley solver is **not built** (§57.6).

## Context

`Heat1D.with_a_array` (explicit ζ-A) is CFL-limited by the stiffest layer; the
`AdaptivePI`/resolvent kernels are constant-coefficient only. Multilayer
conduction is `∂ₜ(ρc·T) = ∂ₓ(k∂ₓT)`, i.e. `∂ₜT = M⁻¹K T` with `K = −L_k`
(harmonic-mean stiffness) and `M = diag(ρc)` (lumped node capacity). Both
operators already exist: #11 ships the symmetric `K`, #13 ships `e^{−τM⁻¹K}v` for
diagonal `M`. The missing piece is purely the *plumbing* — turning a physical
layer list into the `(grid, k_nodes, ρc_nodes)` triple and choosing the evolver.

## TRIZ note (the stiffness contradiction is resolved by reuse, not by a new solver)

АП: a sharp high-contrast stack is **stiff** (`λ_max(M⁻¹A) ∝ k_max/(ρc·dx²)`).
ТП: an explicit step is cheap/suckless but CFL-unstable on the stiff stack /
the issue's new implicit Cayley solver is unconditionally stable but is new heavy
machinery (violates suckless). ФП: the propagator must be **new** (to be
stable/exact) AND **not-new** (to stay suckless). **Resolution by super-system
resource, not compromise:** the exact unconditionally-stable action `e^{−τM⁻¹K}v`
and the A-stable CN single step **already exist in the topology** (§54/§55/§56,
just shipped by #13/#11). The stiffness is absorbed by the Krylov spectrum
adaptation (`√(τλ_max)` matvecs, *flat in depth*) — not by a τ-stability limit and
not by new code. The system holds **both** properties at once (exact+stable AND
suckless/no-new-method) by exploiting a resource already present. The Cayley
solver the issue proposes would re-derive what `mass_lumped_evolve` does exactly.

## Decision

Three additive pieces; **`k>0`, `ρc>0` required; 1-D first (separable N-D via the
existing N-D assembler + lumped mass is a trivial extension, deferred).**

### D1 — `MultilayerStack`: physical layers → node arrays (the only genuinely new core code)

```rust
// crates/semiflow/src/multilayer.rs   (new, ≤ ~180 lines)

/// One physical layer of a 1-D conduction stack (SI units).
#[derive(Clone, Copy)]
pub struct Layer<F: SemiflowFloat = f64> {
    pub thickness: F,  // m,               > 0
    pub k: F,          // W/(m·K),         > 0
    pub rho_c: F,      // ρ·c J/(m³·K),    > 0   (volumetric heat capacity)
}

/// Discretised stack on a single uniform-dx node grid, with per-node conductivity
/// and per-node lumped mass `ρc` (the diagonal of `M`). Ready for either evolver.
pub struct MultilayerStack<F: SemiflowFloat = f64> {
    pub grid: Grid1D<F>,
    pub k_nodes: alloc::vec::Vec<F>,      // length grid.n  (faces use harmonic mean)
    pub rho_c_nodes: alloc::vec::Vec<F>,  // length grid.n  (M = diag(ρc), §57.2)
}

impl<F: SemiflowFloat> MultilayerStack<F> {
    /// Build from layers with one global `dx ≈ target_dx` (snapped so each layer
    /// gets ≥1 cell and `Σ cells·dx = Σ thickness`). Interface nodes are assigned
    /// the material to their left; the harmonic-mean face between unlike `k`
    /// (computed by the §56 assembler) carries the series resistance exactly.
    ///
    /// # Errors
    /// `DomainViolation` if `layers` empty, any `thickness/k/ρc ≤ 0`, or `n < 2`.
    pub fn from_layers(layers: &[Layer<F>], target_dx: F) -> Result<Self, SemiflowError>;

    /// PRIMARY bridge — exact unconditionally-stable Krylov path (§57.3).
    /// Returns the symmetric PSD stiffness carrier `A = −L_k` (§56) and the lumped
    /// mass vector `ρc`. Propagate with `mass_lumped_evolve(&a, &masses, …)`.
    pub fn to_stiffness_and_mass(&self, boundary: BoundaryPolicy<F>)
        -> Result<(SymmetricOperator<F>, alloc::vec::Vec<F>), SemiflowError>;
}

/// One-shot EXACT stack evolution `u(τ) = e^{−τ diag(ρc)⁻¹ A} u₀` (depth-flat
/// Krylov; the whole re-entry in one action). Thin wrapper over `mass_lumped_evolve`.
///
/// # Errors  Propagated from assembly / Krylov; `Unsupported` if the operator is
/// too stiff for the requested `tol` without substepping (fail-loud, §54 fix).
#[allow(clippy::too_many_arguments)]
pub fn multilayer_evolve<F: SemiflowFloat>(
    stack: &MultilayerStack<F>, boundary: BoundaryPolicy<F>,
    tau: F, u0: &[F], out: &mut [F],
    path: KrylovPath, tol: F, scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>;
```

**Recommended evolver:** `multilayer_evolve` (Krylov, `KrylovPath::Lanczos` for
the stiff/tight-tolerance regime) — exact, unconditionally stable, depth-flat. Use
the optional CN convenience (D2) only when O(1)-in-`m` working memory is required.

### D2 — `MassWeightedConservativeChernoff` (OPTIONAL convenience; NOT required for acceptance)

A-stable CN on `∂ₜT=−M⁻¹A T`: `(M+½τA)Tⁿ⁺¹=(M−½τA)Tⁿ`. `M` diagonal + `A`
tridiagonal ⇒ **still one O(n) Thomas solve** — #11's `cn_step` with `diag(ρc)`
added to the system diagonal. Sibling struct (does not touch #11's frozen signature).

```rust
// crates/semiflow/src/multilayer.rs (same file)
/// Mass-weighted conservative diffusion, order-2 A-stable CN, state `GridFn1D<F>`.
pub struct MassWeightedConservativeChernoff<F: SemiflowFloat = f64> { /* faces, ρc, grid, boundary */ }
impl ChernoffFunction<f64> for MassWeightedConservativeChernoff<f64> {
    type S = GridFn1D<f64>;
    fn apply_into(&self, tau: f64, src: &Self::S, dst: &mut Self::S,
                  scratch: &mut ScratchPool<f64>) -> Result<(), SemiflowError>; // mass-CN Thomas
    fn order(&self) -> u32 { 2 }
    fn growth(&self) -> Growth<f64> { Growth::contraction() }
}
// f32 mirror (ADR-0025).
```

### D3 — Bindings (deferred; `bindings` label)

PyO3: `Heat1D.with_multilayer(layers, target_dx, u0, boundary)` → calls
`multilayer_evolve`; `layers` a numpy `(L,3)` array of `(thickness,k,ρc)`. Reuses
the §55 carrier-handle / `sym_op_evolve` path. C-ABI follows the carrier-handle
precedent, deferred — not required to close #14's gates.

### D4 — Acceptance gates (NORMATIVE; all RELEASE_BLOCKING, `feature_gate: slow-tests`)

| Gate | Definition | Threshold | Oracle (REUSE) |
|------|-----------|-----------|----------------|
| `G_TPS_MASS_WEIGHT` | `mass_lumped_evolve(A, ρc)` vs dense `expm(−τ·diag(ρc)⁻¹A)v` on a 2-material stack, `N≤12`, ρc-contrast `≥2`, `τ‖M⁻¹A‖≥10` | `sup_error ≤ 1e-10` | dense `mat_exp_pade13` (§45/§55.5; **no sympy**) |
| `G_TPS_UNITMASS_FAILS` (**TEETH**) | SAME stack + SAME CN oracle, evolved with #11's UNIT-mass `ConservativeDiffusionChernoff` (no ρc); assert it MISSES | T_Al rel-err **`≥ 0.5`** — assertion of FAILURE (measured 31.7) | the `G_TPS_STACK_ACCEPTANCE` CN reference |
| `G_TPS_STACK_ACCEPTANCE` | Build LI-900/SIP/RTV/Al-2024 (real α/k/ρc, k-contrast `≥100`), hot Dirichlet outer face + insulated Al backface, evolve 2500 s via `multilayer_evolve`; compare T_Al(t), T_bondline(t) at `t∈{500,1500,2500}s` vs an INDEPENDENT test-local dense CN reference | per-probe rel-err **`≤ 2e-2`** (issue allows 2–3 %) | test-local dense Crank–Nicolson, fine steps (**no sympy**) |
| `G_TPS_STIFF_STEPCOUNT` (**TEETH**) | Instrument the Krylov matvec counter (reuse §54.5 `DEPTH_FLAT` counter); compute `X=⌈τ·λ_max(M⁻¹A)⌉` (explicit-CFL count) and the stable-path count `Y` | assert `X/Y ≥ 100` AND `Y ≤ 2·√(τλ_max)` (measured X/Y ≈ 1409) | structural/perf; no oracle |

**Non-vacuity / teeth (asserted INSIDE each gate).** `G_TPS_MASS_WEIGHT`/
`G_TPS_UNITMASS_FAILS` assert ρc-contrast `≥2` (so a unit-mass shortcut FAILS) and
`G_TPS_UNITMASS_FAILS` asserts the failure margin `≥50 %` — proving the
mass-weighting is NECESSARY, not incidental (it is #14's entire delta).
`G_TPS_STACK_ACCEPTANCE` asserts a REAL multilayer (k-contrast `≥100`, harmonic
face at the tile/Al interface differs from the arithmetic mean by `≥10 %`) AND a
genuine mid-transient (`‖u(τ)−u₀‖>0` AND `u(τ)≠u_steady` — a "return input" or
"return steady" shortcut fails). `G_TPS_STIFF_STEPCOUNT` asserts the explicit run
WOULD need `≥X` steps while the stable path uses `≤Y≪X`. Gates live as inline
`#[cfg(test)]` modules (`multilayer_tests.rs`, `include!`-d). **No new sympy.**

## Consequences

- **Reuse, near-zero disturbance.** New behaviour = one small new file
  (`multilayer.rs` ≤180) + one test file (`multilayer_tests.rs` ≤300) + `mod`/
  re-export lines in `lib.rs`. The §56 assembler, §55 `mass_lumped_evolve`, §54
  Krylov, and §45 oracle are consumed verbatim. No existing kernel, gate, public
  signature, or 0-ULP scope changes.
- **The stiffness contradiction is resolved by reuse** (TRIZ above): exact
  unconditionally-stable propagation comes from §54/§55 already in the topology;
  the suggested Cayley solver is not built.
- **Honest boundaries (documented, §57.6, not hidden):** `k>0, ρc>0`; node-centered
  lumped mass (`M=diag(ρc)`) — consistent (non-diagonal) mass is not needed for
  conduction and would use the §55 `MassKOperator` path; material interfaces lie on
  faces (series resistance exact there, spatial order 1 near a jump — FV intrinsic);
  one global `dx` (per-layer non-uniform grids OUT OF SCOPE — choose `dx` to resolve
  the thinnest layer); full-tensor non-separable OUT OF SCOPE.
- **Suckless:** zero new dependencies; no LAPACK; new file ≤ limits; all fns ≤50
  lines; one build path unchanged; deterministic assembly; identity-compatible
  (`ρc≡const` recovers the unit-mass §56 operator exactly).

### Implementation ordering (for the engineer)

1. **`MultilayerStack::from_layers`** — layer→node `dx`-snapping + per-node `k`/`ρc`.
   Unit test: node counts, total thickness, interface material assignment.
2. **`to_stiffness_and_mass` + `multilayer_evolve`** — delegate to §56
   `assemble_conservative_csr_1d` and §55 `mass_lumped_evolve(masses=ρc)`.
   Gate: **`G_TPS_MASS_WEIGHT`** (small-N exactness vs dense `expm`).
3. **TEETH** — same small stack via #11 unit-mass; assert `≥50 %` miss.
   Gate: **`G_TPS_UNITMASS_FAILS`**.
4. **TPS acceptance** — full LI-900/SIP/RTV/Al stack, 2500 s, vs test-local dense CN.
   Gate: **`G_TPS_STACK_ACCEPTANCE`** (with the mid-transient + k-contrast asserts).
5. **Stiffness step-count** — instrument the §54.5 matvec counter; assert `X/Y≥100`.
   Gate: **`G_TPS_STIFF_STEPCOUNT`**.
6. **(Optional) `MassWeightedConservativeChernoff`** — mass-CN Thomas (f64+f32),
   `order()=2`, contraction. Convenience for the O(1)-memory regime; no new gate
   beyond reusing `G_TPS_STACK_ACCEPTANCE` against the CN path.
7. **Bindings (`bindings` label, deferred)** — `Heat1D.with_multilayer`.

## References

- S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere 1980, §4.2-4
  — harmonic-mean interface conductivity (consumed via §56).
- J. H. Lienhard IV & V, *A Heat Transfer Textbook*, 5th ed. — series thermal
  resistance / multilayer conduction; lumped capacity `ρc`.
- D. E. Glass, *Ceramic Matrix Composite (CMC) TPS*, NASA TM (2008) and Shuttle
  Orbiter TPS material data (LI-900, SIP, RTV-560, Al-2024) — representative α/k/ρc.
- ADR-0187 / §56 — `assemble_conservative_csr_1d` (the consumed stiffness assembly).
- ADR-0186 / §55 — `SymmetricOperator`, `mass_lumped_evolve` (the consumed lumped
  `(M,K)` exact propagator — the #11→#13→#14 bridge).
- ADR-0185 / §54 — A1 Krylov exact action + stiff substep fix (`9e5f557`).
- ADR-0125 / §45 — `mat_exp_pade13` (reused dense gate oracle).
