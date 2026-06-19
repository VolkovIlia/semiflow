# ADR-0057 — Schrödinger unitary semigroup (real-only `Δ + V` scope)

- **Status**: ACCEPTED + Amendment 1 (2026-05-21)
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave B
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0026 (Generic-over-Float), ADR-0043 (HilbertState),
  ADR-0006 (`DiffusionChernoff` τ²-correction).
- **Decision pending**: this ADR picks OPTION A (real-only) over OPTION B
  (introduce `SemiflowComplex` trait). Locks v2.2 surface; v2.3+ may
  revisit for full complex amplitudes.
- **Mathematical foundation**: math.md §17 (CITATION: Engel-Nagel 2000
  *One-Parameter Semigroups* §IV.6 — strongly continuous unitary group
  on Hilbert space; NORMATIVE: real-only `(ψ_re, ψ_im) ∈ ℝ^N × ℝ^N`
  splitting policy).

## Context

The Schrödinger equation `i·∂_t ψ = H·ψ` with self-adjoint Hamiltonian
`H = −Δ + V(x)` (potential `V: ℝ^N → ℝ`) describes time evolution of
quantum-mechanical wavefunctions on a real Hilbert space (after the
`i` factor is absorbed). The semigroup `U(t) = exp(−i·t·H)` is
UNITARY: `U(t)*U(t) = I` for all `t`, equivalent to
`‖U(t)·ψ‖₂ = ‖ψ‖₂` (energy conservation).

For `semiflow-core`, this is the FIRST unitary group — all prior
`ChernoffFunction` impls cover dissipative (heat) or
quasi-contractive (drift-reaction) semigroups where `‖S(τ)‖ ≤ M·exp(ω·τ)`
with `ω > 0` is acceptable. Unitary requires `ω = 0` and orthogonal
spectrum (no eigenvalue dies out).

**Complex-number question**: a faithful implementation would use
`Complex<F>` as the state element type. This requires:

1. A new sealed trait `SemiflowComplex: ... + Mul<Self, Output = Self>`
   alongside `SemiflowFloat`.
2. All 25+ `ChernoffFunction<F>` impls need disambiguation
   (`F: SemiflowFloat` vs `F: SemiflowComplex`).
3. The `State<F>` trait hierarchy doubles in width.
4. SIMD path (currently f64-only) doesn't have native complex
   intrinsics — would need bespoke implementation.

## Decision

**OPTION A**: ship Schrödinger as a **real-only** type
`SchrodingerChernoff<F: SemiflowFloat = f64>` operating on a state
`SchrodingerState<F>` that stores `(ψ_re, ψ_im) ∈ ℝ^N × ℝ^N` as TWO
disjoint `GridFn1D<F>` (or `GridFn2D<F>` / `GraphSignal<F>` in
future v2.3) buffers.

```rust
//! crates/semiflow-core/src/schrodinger.rs (NEW FILE, ~420 LoC)

#[derive(Clone, Debug)]
pub struct SchrodingerState<F: SemiflowFloat = f64> {
    pub psi_re: GridFn1D<F>,
    pub psi_im: GridFn1D<F>,
}

impl<F: SemiflowFloat> State<F> for SchrodingerState<F> {
    fn len(&self) -> usize { 2 * self.psi_re.len() }
    fn axpy_into(&mut self, alpha: F, src: &Self) {
        self.psi_re.axpy_into(alpha, &src.psi_re);
        self.psi_im.axpy_into(alpha, &src.psi_im);
    }
    fn copy_from(&mut self, src: &Self) {
        self.psi_re.copy_from(&src.psi_re);
        self.psi_im.copy_from(&src.psi_im);
    }
    fn zero_into(&mut self) {
        self.psi_re.zero_into();
        self.psi_im.zero_into();
    }
    fn norm_sup(&self) -> F {
        // ‖ψ‖_∞ = sup_x √(ψ_re² + ψ_im²)  — use vec_ext + sqrt
        compute_sup_complex_modulus(&self.psi_re, &self.psi_im)
    }
}

impl<F: SemiflowFloat> HilbertState<F> for SchrodingerState<F> {
    fn dot(&self, other: &Self) -> F {
        // Re⟨ψ, φ⟩ = ψ_re·φ_re + ψ_im·φ_im  (real-valued; imaginary part
        // would be ψ_re·φ_im − ψ_im·φ_re — not stored in this scope)
        self.psi_re.dot(&other.psi_re) + self.psi_im.dot(&other.psi_im)
    }
}

#[derive(Clone, Debug)]
pub struct SchrodingerChernoff<F: SemiflowFloat = f64> {
    /// Kinetic-energy operator on real grid (re-uses `Diffusion4thChernoff`
    /// for `−Δ` discretisation).
    kinetic: Diffusion4thChernoff<F>,
    /// Potential `V: ℝ → ℝ`. Length N, evaluated on grid nodes once at construct.
    v_at_node: Vec<F>,
}

impl<F: SemiflowFloat> SchrodingerChernoff<F> {
    pub fn new(
        kinetic: Diffusion4thChernoff<F>,
        v: impl Fn(F) -> F,
    ) -> Result<Self, SemiflowError>;
}

impl<F: SemiflowFloat> ChernoffFunction<F> for SchrodingerChernoff<F> {
    type S = SchrodingerState<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        // Strang splitting: V-rotation · K(τ) · V-rotation
        // Step 1 of 3: half-step potential rotation
        apply_v_rotation(self, tau * half::<F>(), src, dst)?;
        // Step 2 of 3: full-step kinetic (re-uses Diffusion4thChernoff)
        let mut tmp = dst.clone(); // TODO: zero-alloc via scratch buffer
        self.kinetic.apply_into(tau, &tmp.psi_re, &mut dst.psi_re, scratch)?;
        self.kinetic.apply_into(tau, &tmp.psi_im, &mut dst.psi_im, scratch)?;
        // Step 3 of 3: half-step potential rotation
        apply_v_rotation(self, tau * half::<F>(), &dst.clone(), dst)?;
        Ok(())
    }

    fn order(&self) -> u32 {
        // Strang composition of order-2 V-rotation and order-4 kinetic
        // is globally order-2 (Strang composition theorem).
        2
    }

    fn growth(&self) -> (f64, f64) {
        // Unitary group: ‖U(τ)‖ = 1; (M, ω) = (1, 0).
        (1.0, 0.0)
    }
}
```

The `apply_v_rotation` helper (math.md §17.2):

```text
For each node i:
  α = V(x_i) · τ
  ψ_re_new[i] = cos(α) · ψ_re[i] + sin(α) · ψ_im[i]
  ψ_im_new[i] = − sin(α) · ψ_re[i] + cos(α) · ψ_im[i]
```

This is the 2×2 rotation matrix `[[cos(α), sin(α)], [−sin(α), cos(α)]]`
acting on `(ψ_re, ψ_im)` — the real representation of multiplication by
`exp(−i · V · τ)`. **Unitary by construction** (rotation preserves
modulus).

## Rationale

- **No `SemiflowComplex` ecosystem pressure.** Single use case
  (Schrödinger) doesn't justify doubling the `State<F>` trait
  hierarchy and re-spec'ing all 25+ existing impls. Per
  constitution v1.4.0 principle 4 ("`no_std + alloc` budget is
  sacred"), avoiding a full complex-type plumbing keeps the core
  crate clean.
- **Math is identical**. The 2N-dim real representation is the
  standard textbook approach (Trotter splitting for quantum systems;
  Sanz-Serna-Calvo 1994 §4.5; Blanes-Casas-Murua 2006 *J. Chem. Phys.*).
- **Reuses `Diffusion4thChernoff`**. The kinetic operator `−Δ` already
  has a high-quality order-4 Chernoff implementation. Strang composition
  with the V-rotation gives global order 2 — matches typical
  Schrödinger benchmark accuracy (Sanz-Serna's S2 split is order 2).
- **Backward-compatible**. No change to `SemiflowFloat`, `State<F>`,
  `ChernoffFunction<F>` signatures.

## Consequences

- New module `src/schrodinger.rs` (~420 LoC); under file cap.
- Public surface +2 types (`SchrodingerState<F>`, `SchrodingerChernoff<F>`).
- `lib.rs` re-export adds both.
- Engineers who later need full complex amplitudes (time-dependent V,
  non-Hermitian H for open quantum systems) MUST wait for v2.3+
  `SemiflowComplex` (not in scope here).

## Acceptance gates

- **G18 unitarity gate** (NORMATIVE). For arbitrary initial state
  `ψ₀ = Gaussian wavepacket on `[-5, 5]`, N=64`, `V(x) = ½·x²` (harmonic
  oscillator), `t_final = 1.0`, n_steps = 100. Verify
  `|‖ψ(t_final)‖₂² − ‖ψ₀‖₂²| < 1e-12` (f64) or `< 1e-6` (f32).
  Slope test on `n_steps ∈ {10, 20, 40, 80, 160}`: slope ≤ −1.95 (f64),
  ≤ −1.50 (f32) — order-2 Strang.
- **G19 harmonic-oscillator oracle gate** (NORMATIVE). Initial
  Gaussian centred at `x = 1`, σ = 0.5. After period `T = 2π` of
  harmonic oscillator, ψ should return to itself (up to phase). Verify
  `‖ψ(T) − e^{i·ϕ} · ψ₀‖₂ < 1e-3` (f64) for some real phase ϕ.
- **T15N_schrodinger_unitarity sympy gate** (NORMATIVE). Verify the
  V-rotation 2×2 matrix on symbolic `α` is exactly the rotation matrix
  `[[cos(α), sin(α)], [−sin(α), cos(α)]]` and that its determinant is
  exactly 1 (unitarity = orthogonal in real representation). Pure
  symbolic.

## Out of scope (v2.2)

- **Time-dependent `V(x, t)`.** Strang splitting with frozen V at
  midpoint gives order-2. Order-4 with time-dependent V requires
  Magnus on the V-rotation — couples to ADR-0056 K=6 Magnus pattern.
  Deferred to v2.3+ once K=6 ships.
- **Complex `V(x)` (non-Hermitian H)**. Open quantum systems with
  absorbing boundary, decay channels. Requires `SemiflowComplex` trait.
  Deferred to v2.3+.
- **Graph Schrödinger.** `SchrodingerGraphChernoff` with
  `state = (psi_re: GraphSignal, psi_im: GraphSignal)`. Trivial
  extension once Wave A ships; deferred to v2.3+ unless customer
  demand emerges.
- **3D Schrödinger.** Trivial via `GridFn3D<F>` substitution. Same
  deferral.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Real-only scope regretted when v2.3 needs complex | Migration path: `SchrodingerState<F>` becomes a constructor adapter for `SchrodingerStateComplex<C: SemiflowComplex>`; existing API not removed. Future-compat preserved. |
| R2 | Strang splitting with V-rotation gives order 2 — slower than expected for HPC | Document in rustdoc; users wanting order-4 must wait for v2.3 K=4 Magnus-on-V. |
| R3 | f32 unitarity check `|‖ψ‖² − ‖ψ₀‖²| < 1e-6` insufficient over long horizons | Document the f32 envelope: long-time horizons (T > 10 / ‖H‖_2) require f64; ADR-0046 precision-policy guidance applies. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/schrodinger.rs` | ~420 |
| `tests/g18_unitarity.rs` | ~140 |
| `tests/g19_harmonic_oscillator.rs` | ~160 |
| `.dev-docs/verification/scripts/verify_v2_2_schrodinger_unitarity.py` | ~130 |
| math.md §17 | ~140 |
| ADR-0057 (this) | ~210 |
| **Total** | **~1200** |

## Amendment 1 — Crank-Nicolson kinetic step (2026-05-21)

The original Decision specified Strang splitting `V(τ/2)·K(τ)·V(τ/2)` where `K(τ)`
is a kinetic propagator `exp(τΔ/2)` approximated via `Diffusion4thChernoff`.
Wave 2.2B implementation phase discovered that `Diffusion4thChernoff` approximates
`exp(τaΔ)` (a contractive heat semigroup with positive `a`), NOT the unitary
kinetic propagator `exp(iτΔ/(2m))`. Composing it with the V-rotation broke
unitarity globally (G18 unitarity error ~4e-1).

**Resolution**: replace the kinetic step with Crank-Nicolson (Cayley map):

```
  K_CN(τ) = (I − iτΔ/4m)⁻¹ · (I + iτΔ/4m)
```

implemented via banded LU on the pentadiagonal `(I + A²)` where `A = τΔ/(4m)`.
This is exactly unitary in exact arithmetic; numerical roundoff residual is
O(ε_machine) per step. With mixed-precision strategy (all arithmetic in f64
regardless of `F`), G18 passes both f64 (1.07e-14) and f32 (9.54e-7).

Strang composition unchanged: `V(τ/2)·K_CN(τ)·V(τ/2)`, palindromic.
Order remains 2.
The V-rotation Option A representation `(ψ_re, ψ_im) ∈ ℝ^N × ℝ^N` is preserved.

Trade-off: Crank-Nicolson requires solving a pentadiagonal linear system per
step, vs explicit `Diffusion4thChernoff` apply. Banded LU is O(N) per solve.
Mixed-precision f64 inside f32-public-API increases internal memory by 2× but
preserves the cross-binding compatibility.

**R4 zero-alloc (Amendment 1 adds)**: the 12 f64 working buffers for the
Crank-Nicolson step are pre-allocated in `SchrodingerChernoff::new()` and stored
in a `RefCell<[Vec<f64>; 12]>` field. They are reused on every `apply_into` call
via `borrow_mut()`. After warm-up, `apply_into` allocates zero heap bytes (gate:
`tests/schrodinger_zero_alloc.rs`).

This amendment SUPERSEDES the original Decision §"Kinetic step". The §"Strang
splitting" and §"State representation" sections remain in force.

## References

- K.-J. Engel, R. Nagel, *One-Parameter Semigroups for Linear Evolution
  Equations* (Springer 2000), §IV.6 — strongly continuous unitary group.
- J. M. Sanz-Serna, M. P. Calvo, *Numerical Hamiltonian Problems*
  (Chapman & Hall 1994), §4.5 — Strang-splitting Schrödinger.
- S. Blanes, F. Casas, A. Murua, *J. Chem. Phys.* **124** (2006) 234105.
- A. Iserles, *A First Course in the Numerical Analysis of Differential
  Equations* (Cambridge 2009) §8 — Cayley map for unitary group SU(N).
- ADR-0006 (`DiffusionChernoff` τ²-correction) — kinetic operator reuse.
- ADR-0026 (Generic-over-Float) — f32 supported within precision band.
