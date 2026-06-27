# ADR-0187 — Conservative (divergence-form) variable-coefficient diffusion with harmonic-mean faces and symmetric carrier

- **Status**: Proposed (design only — Issue #11; branch `issue-11-conservative-divergence`)
- **Date**: 2026-06-27
- **Supersedes**: none — purely ADDITIVE. Reuses ADR-0186/§55
  (`SymmetricOperator::from_csr`), ADR-0185/§54 (A1 Krylov exact action `e^{−τA}v`),
  ADR-0125/§45 (`mat_exp_pade13` dense oracle), and ADR-0008/§9.2.3
  (`DiffusionChernoff` — the non-conservative contrast). No existing kernel, gate,
  public signature, or 0-ULP scope is changed.
- **Contract**: `contracts/semiflow-core.math.md` §56 (new NORMATIVE section);
  gates `G_CONS_SERIES`, `G_CONS_NONCONS_FAILS`, `G_CONS_SYMOP`, `G_CONS_ORDER`,
  `G_CONS_CONTACT`.

## Context

The crate already solves variable-coefficient 1-D diffusion via `DiffusionChernoff`
(§9.2.3, ADR-0008): the ζ-A kernel for `A f = ∂_x(a(x)·∂_x f)` **expanded pointwise**
into `a·f'' + a'·f'`. The PyO3 surface exposes it as `Heat1D.with_a_array(a, a_prime,
a_double_prime, …)`. That expansion is mathematically correct ONLY for smooth `a ∈ C³`
— it requires the caller to supply `a'`, `a''`. At a **sharp material interface** (e.g.
`k = 1 / 100 / 1` across three bonded layers) `a` is discontinuous, `a'` is a δ-function
the discretisation cannot represent, and the pointwise stencil samples a single-side
node value of the coefficient. The consequence is concrete and severe: with the honest
input `a' ≡ 0` (piecewise-constant `a`), the steady operator gives `a·u_xx = 0 ⇒ u_xx = 0`
**globally** ⇒ a single straight temperature line ⇒ EQUAL `ΔT` per equal-thickness
layer regardless of `k`, and a **discontinuous heat flux at the interface**. This
violates the elementary series-resistance physics `ΔT_i = q·R_i, R_i = t_i/k_i`, which
is exactly Issue #11's acceptance criterion (<1% on a sharp 3–4-layer stack).

The fix is the textbook **conservative / divergence-form finite-volume** scheme with
**harmonic-mean face conductivities** `k_{i+½} = 2k_ik_{i+1}/(k_i+k_{i+1})` (Patankar
1980). It is discretely flux-conserving by construction (the same face flux leaves cell
`i` and enters cell `i+1`), so it reproduces the series-resistance network at machine
precision even across a 100:1 jump. Issue #11 asks for this in 1-D, its separable
2-D/3-D analogue, an optional thin **contact-resistance** `R_c` interface (bonded
joints), staying strictly linear / structure-preserving / differentiable.

The existing `apply_div_form_fn` (`approximation.rs`) is NOT this scheme: it is an
internal *jet* helper for the K=2 `ApproximationSubspace` machinery that samples the
coefficient FUNCTION at `x_i ± dx/2` (smooth-`a` only; at a jump it arbitrarily picks
one side, not the harmonic mean) and is not a stable public kernel. So #11 is genuinely
new assembly.

**Design tension (resolved, not compromised).** A sharp `k`-jump makes `L_k` **stiff**
(`λ_max(L_k) ∝ k_max/dx²`). We want the time-stepper to be CHEAP/EXPLICIT (a simple
linear order-2 `ChernoffFunction`, suckless, no new machinery) AND UNCONDITIONALLY
STABLE on the stiff multilayer stack (large steps, no CFL). АРИЗ chain: АП = "a sharp
jump makes `L_k` stiff." ТП = explicit step `I+τL_k` is cheap but CFL-unstable on a
stiff stack / exact step `exp(τL_k)` is unconditionally stable but seemingly needs
expensive new machinery. ФП = the propagator must be NEW machinery (to be exact/stable)
AND NOT-new (to stay suckless). **Resolution by separation in structure + super-system:
split the role of "the step."** Issue #11 builds ONLY what it is uniquely good at — the
**symmetric flux-continuous assembly** (`L_k` with harmonic-mean faces is symmetric and
negative-semidefinite, so `A = −L_k` is a symmetric PSD CSR with `diag ≥ 0`). The exact,
unconditionally-stable propagation `e^{−τA}v = e^{τL_k}v` is then provided **for free**
by the §55 `SymmetricOperator` + §54 Krylov machinery that Issue #13 JUST SHIPPED and is
already in the topology — the resource is reused, not rebuilt. The stiff jump turns from
a problem into a non-issue (Krylov adapts to the spectrum, not to a τ-stability limit).
No property is split down the middle: #11 assembles, #13 propagates exactly & stably,
#14 (stiff multilayer) is the application. For the standard semigroup path under
moderate stiffness, a genuinely order-2 A-stable single step (Crank–Nicolson on the
tridiagonal `L_k`, one `O(n)` Thomas solve) serves without any CFL limit either — the
second free resource is the tridiagonal structure of `L_k` itself.

## Decision

Four additive pieces: (D1) the harmonic-mean assembly + symmetric carrier (the #13
bridge), (D2) the `ConservativeDiffusionChernoff` order-2 `ChernoffFunction`, (D3) the
separable 2-D/3-D analogue, (D4) deferred bindings. **`k > 0` required; symmetric NSD
operator only; full-tensor non-separable OUT OF SCOPE** (§56.8).

### D1 — Harmonic-mean assembly → `SymmetricOperator` (the #11 → #13 → #14 bridge)

`assemble_conservative_csr_1d` builds `A = −L_k` (symmetric PSD CSR, `diag ≥ 0`, sorted
columns) directly consumable by §55 `SymmetricOperator::from_csr`, after which §54
Krylov gives the exact, unconditionally-stable action `e^{−τA}v = e^{τL_k}v`. No new
stiff-ODE machinery is introduced.

```rust
// crates/semiflow/src/conservative_assemble.rs   (new, ≤ ~300 lines; include!-split helpers)

/// Assemble the symmetric PSD carrier `A = −L_k` (CSR, diag ≥ 0) for the conservative
/// 1-D operator `L_k u = ∂_x(k(x) ∂_x u)` with harmonic-mean faces (§56.1-56.2).
/// Symmetry and diag-≥0 hold by construction and are re-validated by `from_csr`.
/// This is the Issue #11 → #13 → #14 bridge: assemble here, propagate via §54 Krylov.
///
/// # Errors
/// `DomainViolation` if any `k_i ≤ 0`, `k.len() != grid.n()`,
/// `r_contact` length ≠ `n−1`, or any entry non-finite.
pub fn assemble_conservative_csr_1d<F: SemiflowFloat>(
    grid: Grid1D<F>,
    k_nodes: &[F],
    r_contact: Option<&[F]>,          // per-face R_c ≥ 0; None ⇒ perfect contact
    boundary: BoundaryPolicy<F>,
) -> Result<SymmetricOperator<F>, SemiflowError>;

/// Separable N-D analogue: `A = −Σ_d L_{k_d}` on the tensor grid, 5-pt (2-D) / 7-pt
/// (3-D) CSR with per-axis harmonic-mean faces; columns emitted SORTED (§55.1 I3).
///
/// # Errors
/// As above, per axis; plus `Unsupported` if `axes ∉ {2, 3}`.
pub fn assemble_conservative_csr_nd<F: SemiflowFloat>(
    grid: &GridNd<F>,                 // existing N-D tensor grid
    k_nodes_per_axis: &[&[F]],        // one k array per axis (separable)
    boundary: BoundaryPolicy<F>,
) -> Result<SymmetricOperator<F>, SemiflowError>;
```

```rust
// crates/semiflow/src/conservative_helpers.rs   (new, ≤ ~120 lines; include!-d into the two above)

fn harmonic_mean<F: SemiflowFloat>(k_l: F, k_r: F) -> F;                 // 2·k_l·k_r/(k_l+k_r)
fn face_transmissibility<F: SemiflowFloat>(k_harm: F, dx: F, r_c: F) -> F; // 1/(dx/k_harm + r_c)
fn build_faces<F: SemiflowFloat>(                                        // validates k>0; returns T_{i+½}
    k_nodes: &[F], dx: F, r_contact: Option<&[F]>,
) -> Result<alloc::vec::Vec<F>, SemiflowError>;
```

### D2 — `ConservativeDiffusionChernoff`: order-2 `ChernoffFunction` (standard semigroup path)

Primary input is the **node-sampled `k` array** (not a closure): at a sharp interface
`k` is naturally per-cell material data, the harmonic mean is a function of adjacent
NODE values `k_i, k_{i+1}`, and this mirrors the `with_a_array` precedent. A `from_k_closure`
convenience samples nodes then delegates. The recommended order-2 step is the A-stable
trapezoidal/Crank–Nicolson `S(τ) = (I − ½τL_k)^{-1}(I + ½τL_k)` — one `O(n)` tridiagonal
Thomas solve per step, unconditionally stable, contraction growth, no new dependency
(CN precedent: `matrix_system_complex_cn.rs`).

```rust
// crates/semiflow/src/conservative.rs   (new, ≤ ~280 lines; struct + 2 ctors + f64/f32 impls)

/// Conservative (divergence-form) variable-coefficient diffusion generator
/// `L_k u = ∂_x(k(x) ∂_x u)`, harmonic-mean faces (§56). Order-2 `ChernoffFunction`,
/// state `GridFn1D<F>`. Identity-compatible: `k ≡ const, R_c ≡ 0` recovers heat.
pub struct ConservativeDiffusionChernoff<F: SemiflowFloat = f64> {
    faces: alloc::sync::Arc<[F]>,     // pre-computed T_{i+½}, length n−1 (§56.1.b)
    grid: Grid1D<F>,
    boundary: BoundaryPolicy<F>,
}

impl<F: SemiflowFloat> ConservativeDiffusionChernoff<F> {
    /// PRIMARY constructor — node-sampled conductivities (length n). Harmonic-mean
    /// faces + optional per-face contact resistance pre-computed once.
    ///
    /// # Errors
    /// `DomainViolation` if any `k_i ≤ 0`, length mismatch, or non-finite.
    pub fn from_k_array(
        grid: Grid1D<F>,
        k_nodes: &[F],
        r_contact: Option<&[F]>,
        boundary: BoundaryPolicy<F>,
    ) -> Result<Self, SemiflowError>;

    /// Convenience — smooth `k(x)` via closure (samples at nodes, delegates).
    pub fn from_k_closure<C: Fn(F) -> F>(
        grid: Grid1D<F>, k: C, boundary: BoundaryPolicy<F>,
    ) -> Result<Self, SemiflowError>;

    /// Build the symmetric PSD carrier `A = −L_k` (delegates to D1) — the #13 bridge.
    pub fn to_symmetric_operator(&self) -> Result<SymmetricOperator<F>, SemiflowError>;
}

impl ChernoffFunction<f64> for ConservativeDiffusionChernoff<f64> {
    type S = GridFn1D<f64>;
    fn apply_into(                       // CN step: (I − ½τL_k)^{-1}(I + ½τL_k) src → dst
        &self, tau: f64, src: &Self::S, dst: &mut Self::S, scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError>;
    fn order(&self) -> u32 { 2 }
    fn growth(&self) -> Growth<f64> { Growth::contraction() }
}
// f32 mirror impl (same shape; ADR-0025 generic-over-Float convention).
```

Separable 2-D/3-D semigroup path reuses the EXISTING `Strang2D`/`Strang3D` +
`AxisLift` with per-axis `ConservativeDiffusionChernoff` — no new struct (§56.5).

### D3 — Acceptance gates (NORMATIVE; all RELEASE_BLOCKING, `feature_gate: slow-tests`)

| Gate | Definition | Threshold | Oracle (REUSE) |
|------|-----------|-----------|----------------|
| `G_CONS_SERIES` | Conservative steady **Dirichlet** solve on a SHARP 3-layer stack `k=[1,100,1]` (equal thickness, NO smoothing); per-layer `ΔT_i` + constant interface flux `q` vs analytic series-resistance (§56.4) | per-layer `ΔT` rel-err `≤ 1e-2` AND face-flux spread `≤ 1e-2` | analytic series-resistance (§56.4) + test-local Thomas / `matrix_inv`; **no sympy** |
| `G_CONS_NONCONS_FAILS` (**TEETH**) | SAME sharp stack + SAME oracle, solved with the NON-conservative `DiffusionChernoff` node operator (`a=k` piecewise-const, `a'≡0`, §9.2.3); assert it MISSES the network | per-layer `ΔT` rel-err **`≥ 0.5`** — assertion of FAILURE | same analytic oracle (§56.4) |
| `G_CONS_SYMOP` | Assemble `A=−L_k` (sharp stack, `N≤12`) → `SymmetricOperator::from_csr` (must ACCEPT: symmetry + diag-≥0); Krylov `e^{−τA}v` vs dense `mat_exp_pade13(−τA)`, `τ‖A‖ ≥ 10` regime | `sup_error ≤ 1e-10` | dense `mat_exp_pade13` (§45/§55.5; **no sympy**) |
| `G_CONS_ORDER` | `ConservativeDiffusionChernoff` temporal order 2 on SMOOTH `k(x)=1+½·sin(x)`: self-convergence slope (probe vs `2N−1`, Richardson) | slope `≤ −1.95` | self-convergence (Richardson pattern; **no sympy**) |
| `G_CONS_CONTACT` | Steady solve with contact joint `R_c>0` on a 2-layer bonded stack; interface jump `ΔT_c` vs `q·R_c` (§56.3) | rel-err `≤ 1e-2` | analytic series + contact (§56.4); test-local Thomas; **no sympy** |

**Non-vacuity / teeth (asserted INSIDE each gate — a degenerate shortcut cannot pass).**
`G_CONS_SERIES`/`G_CONS_SYMOP`/`G_CONS_NONCONS_FAILS` assert a REAL jump:
`max k_i / min k_i ≥ 50` AND `k_i ≠ k_{i+1}` at the interface face (so the harmonic mean
is strictly between and differs from the arithmetic mean — a smoothed coefficient fails
the assertion). `G_CONS_NONCONS_FAILS` is the **teeth**: it asserts the existing
non-conservative form FAILS the same `<1%` test by `≥50%`, so passing `G_CONS_SERIES`
proves the conservative scheme is NECESSARY, not incidental. `G_CONS_SYMOP` also asserts
`from_csr` accepted a non-trivial `A` (`diag > 0`, ≥1 off-diagonal `< 0`, ≥1 face with
`k_L ≠ k_R`). `G_CONS_ORDER` asserts a genuine multi-`τ` slope (not a single point).
Gates live as inline `#[cfg(test)]` modules (`conservative_tests.rs`, `include!`-d); the
steady Dirichlet solve is a test-local Thomas tridiagonal solve (≤40 lines) or existing
`matrix_inv` for `N ≤ 12`. **No new sympy oracle.**

### D4 — Bindings (deferred; `bindings` label)

Additive PyO3 surface, deferred (not required to close #11's math/core gates):
`Heat1D.with_k_conservative(xmin, xmax, n, k_array, u0, r_contact=None, boundary="neumann")`
(semigroup path) and a `sym_op_conservative(...)` that assembles `A=−L_k` and reuses the
§55 carrier-handle / `sym_op_evolve` path (exact Krylov). C-ABI follows the
carrier-handle precedent and is deferred.

## Consequences

- **Reuse, near-zero disturbance.** New behaviour = three small new files
  (`conservative.rs` ≤280, `conservative_assemble.rs` ≤300, `conservative_helpers.rs`
  ≤120) + one test file (`conservative_tests.rs` ≤300) + `mod`/re-export lines in
  `lib.rs`. No existing kernel, gate, public signature, `LaplacianKind`, or 0-ULP scope
  changes; the §55 `SymmetricOperator`, §54 Krylov, and §45 oracle are consumed verbatim.
- **The stiffness contradiction is resolved structurally** (D-context АРИЗ): #11 supplies
  only the symmetric flux-continuous assembly; exact unconditional-stable propagation is
  the reused §55/§54 machinery — and the CN single step is A-stable via one `O(n)` Thomas
  solve. No new stiff integrator, no CFL limit, no new dependency.
- **Honest boundaries (documented, §56.8, not hidden):** `k > 0` required (harmonic mean
  + PSD); symmetric NSD by construction; material interfaces MUST lie on faces (each node
  carries its own `k`); discontinuous-`k` spatial order is 1 near the jump (FV intrinsic)
  while the steady series-resistance network is reproduced EXACTLY on aligned faces (the
  acceptance metric); contact resistance is purely resistive (no interfacial capacitance);
  full-tensor non-separable `∂_x(k∂_y)` is OUT OF SCOPE (separable axes only).
- **Suckless:** zero new dependencies; no LAPACK; all new files ≤ limits; all fns ≤50
  lines; one build path unchanged; bit-stable deterministic assembly; identity-compatible
  (`k≡const, R_c≡0` recovers the constant-coefficient heat operator exactly).

### Implementation ordering (for the engineer)

1. **Helpers + 1-D assembler** — `harmonic_mean`, `face_transmissibility`, `build_faces`
   (validates `k>0`); `assemble_conservative_csr_1d` → `SymmetricOperator::from_csr`.
   Gate: **`G_CONS_SYMOP`** (carrier parity vs `mat_exp_pade13`; assert `from_csr` accepts).
2. **Steady series-resistance gate** — test-local Thomas Dirichlet solve + analytic oracle
   (§56.4) on the sharp `[1,100,1]` stack, with the non-vacuity jump assertion.
   Gate: **`G_CONS_SERIES`**.
3. **TEETH gate** — same stack, solve the non-conservative `DiffusionChernoff` node
   operator (`a=k`, `a'≡0`), assert per-layer `ΔT` rel-err `≥ 50%`.
   Gate: **`G_CONS_NONCONS_FAILS`**.
4. **`ConservativeDiffusionChernoff`** — `from_k_array` / `from_k_closure` /
   `to_symmetric_operator`; CN trapezoidal `apply_into` (f64 + f32), `order()=2`,
   `Growth::contraction()`. Gate: **`G_CONS_ORDER`** (smooth-`k` self-convergence slope).
5. **Contact resistance** — wire `r_contact` through `build_faces`; 2-layer bonded steady
   solve. Gate: **`G_CONS_CONTACT`**.
6. **Separable 2-D/3-D** — `assemble_conservative_csr_nd` (5-pt/7-pt, sorted columns) +
   compose existing `Strang2D`/`Strang3D` + `AxisLift` over per-axis 1-D kernels.
   (Optional parity gate vs dense `mat_exp_pade13` for small `N`.)
7. **Bindings (`bindings` label, deferred)** — `Heat1D.with_k_conservative` +
   `sym_op_conservative` over the f64 surface (numpy in/out; reuse §55 carrier path).

## References

- S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere 1980, §4.2-4 —
  harmonic-mean interface conductivity for discontinuous/conjugate conductivity.
- H. K. Versteeg, W. Malalasekera, *An Introduction to CFD: The Finite Volume Method*,
  2nd ed., Pearson 2007 — conservative FV, face fluxes, series-resistance interface.
- ADR-0186 / §55 — `SymmetricOperator::from_csr` (the consumed symmetric PSD carrier).
- ADR-0185 / §54 — A1 Krylov exact action `e^{−τA}v` (the unconditionally-stable
  propagator reused by the #11 → #13 → #14 bridge).
- ADR-0125 / §45 — `mat_exp_pade13` (reused dense gate oracle).
- ADR-0008 / §9.2.3 — `DiffusionChernoff` ζ-A (the NON-conservative contrast; the teeth).
- ADR-0025 — generic-over-Float convention (f64/f32 mirror impls).
