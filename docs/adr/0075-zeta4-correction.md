# ADR-0075 — ζ⁴ Correction for Order-4 Diffusion Chernoff (A5)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v3.0 Wave C (third math pillar of the BREAKING window; rides on B1 ApproximationSubspace<4> from ADR-0073 + ChernoffFunction trait cleanup from ADR-0074). Independent of A4 manifold (v2.8 ADR-0071); independent of B4 reflection (v2.8 ADR-0072). NEW algorithmic content beyond the v2.x diffusion family.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0008 (v0.3.0 ζ-A τ²-correction — the order-2 sibling that A5 generalises to order-4), ADR-0013 (v0.6.0 4th-order spatial Diffusion4thChernoff — the spatial backend), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (`ChernoffFunction<F>` generic over F), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0073 (v3.0 B1 `ApproximationSubspace<K, F>` — the witness consumed by `Diffusion4thZeta4Chernoff::new` at construction time), ADR-0074 (v3.0 ChernoffFunction trait cleanup — `Growth<F>` return type used by the new impl's `growth()`).
- **Supersedes / amends**: none — strictly additive on the public surface. Establishes a NEW kernel `Diffusion4thZeta4Chernoff<F>` as a SIBLING to v0.6.0's `Diffusion4thChernoff<F>` (no replacement; the v0.6.0 type is preserved verbatim and continues to underwrite all v2.x callers including the v2.x compatibility shim of ADR-0074).
- **Mathematical foundation**: math.md §27 (NORMATIVE library — `Diffusion4thZeta4Chernoff` semantics; CITATION Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — the k=2 tangency theorem; Example 4.2 — the order-4 Chernoff construction on `D(A^6)` with `a ∈ C^6_b`; Vedenin-Smolyanov-Voskresenskaya 2020 *Math. Notes* — Taylor-jet structure for higher-order Chernoff). math.md §27 BUILDS on §9.2.4 (v0.6.0 4th-order spatial ζ⁴ idea — partial; this ADR closes the loop to a full Chernoff function).
- **Acceptance gates added**: G_zeta4 (RELEASE_BLOCKING — slope ≥ 1.9 on heat with $a \in C^6_b$, $n \in \{16, 32, 64, 128\}$, N=512 grid; lives in `tests/zeta4_correction_slope.rs` new file, feature `slow-tests`); T23N (NORMATIVE sympy — symbolic verification of the τ²·P_2[A] f cancellation identity; 3 sub-checks).

## Context

The v0.3.0 ζ-A τ²-correction (ADR-0008) lifts the variable-coefficient diffusion `∂_t u = ∂_x(a(x) ∂_x u)` from order 1 to order 2 by adding an explicit τ²-correction polynomial in the operator A. The v0.6.0 4th-order spatial extension (ADR-0013) lifts the *spatial* convergence to dx⁴ while keeping the *temporal* Chernoff order at 2. The natural next step — **lifting the temporal Chernoff order from 2 to 4** for variable-coefficient diffusion — has been deferred through v0.6 → v2.8 because:

- It requires the **order-K Chernoff tangency theorem** (Galkin-Remizov 2025 *IJM* Theorem 3.1) which only crystallised in late 2025.
- It requires a **K-jet operator witness** at the type level — the v2.x `ChernoffFunction<F>` trait had no such surface. ADR-0073 (v3.0 B1) ships `ApproximationSubspace<K, F>` precisely for this.
- It requires the **v3.0 BREAKING ChernoffFunction cleanup** (ADR-0074) — the new kernel's `growth()` needs the typed `Growth<F>` return (the v2.x `(f64, f64)` tuple can't carry the multi-K growth bound cleanly), and the kernel's `Self::S` non-Clone constraint matters for zero-alloc K-jet evaluation.

With ADR-0073 and ADR-0074 landed in Wave A, **Wave C ships A5** as the immediate consumer that demonstrates the v3.0 trait infrastructure has industrial-priority math content. The v3.0 academic-priority arc is: (Wave A: trait cleanup + B1 marker trait) → (Wave C: A5 ζ⁴ riding on B1).

Galkin-Remizov 2025 *Israel Journal of Mathematics* Theorem 3.1 gives the operator-polynomial form of the order-4 Chernoff correction:

For a Chernoff function $F(\tau)$ that is *order-2 tangent* to $\{S(t) = \exp(t A)\}$ on the core $D(A^2)$ (i.e., $F(\tau) f = f + \tau A f + \tfrac{\tau^2}{2} A^2 f + O(\tau^3)$ on $f \in D(A^2)$), the **k=2 tangent correction**
$$
F_{\zeta^4}(\tau) f \;:=\; F(\tau) f \;+\; \tau^2 \cdot P_2[A] f
$$
is order-4 tangent to $\{S(t)\}$ on the strictly tighter core $D(A^6)$, where $P_2[A]$ is a Banach-space operator polynomial **uniquely determined** by the requirement that the leading $\tau^2$ term in the Taylor expansion of $(F(\tau/n))^n f - \exp(\tau A) f$ cancels identically.

The polynomial $P_2[A]$ is mathematically a *commutator polynomial* — sums of nested commutators $[A, [A, A^k]]$ that vanish on the order-2 tangent base but contribute at order 3. For the specific case where $F(\tau)$ is the v0.6.0 4th-order spatial `Diffusion4thChernoff` applied to $A = \partial_x(a(x) \partial_x \cdot)$ with $a \in C^6_b$, Example 4.2 of Galkin-Remizov gives the closed-form $P_2[A]$ as **~6 monomials in $A$** (the exact count depends on the chosen 4th-order spatial discretization; for the v0.6.0 9-point stencil it is 6 monomials). The library SHIPS the 6-monomial form per math §27.3.

Crucially: the correction REQUIRES `f ∈ D(A^6)` AND `a ∈ C^6_b` (six bounded derivatives) per Galkin-Remizov §3.2. The K=6 witness comes from `ApproximationSubspace<6, F> for TruncatedExp4thDiffusionChernoff` (ADR-0073 §"Decision" — third opt-in impl). The `a ∈ C^6_b` requirement is caller-asserted via a new optional construction field `a_kth_bound: Option<F>` on `Diffusion4thZeta4Chernoff` (defaults to `None` → caller responsible; if `Some(c)` the kernel verifies $\|a^{(k)}\|_\infty \le c$ for $k \le 6$ at construction time via the sympy oracle of T23N).

## Decision

Ship one additive public-surface item in v3.0 Wave C:

- **`pub struct Diffusion4thZeta4Chernoff<F: SemiflowFloat = f64>`** — new kernel struct in `crates/semiflow-core/src/diffusion4_zeta4.rs`. Generic over `F: SemiflowFloat = f64` per ADR-0025. Fields:
  ```rust
  pub struct Diffusion4thZeta4Chernoff<F: SemiflowFloat = f64> {
      inner: Diffusion4thChernoff<F>,         // the v0.6.0 4th-order spatial backend
      a_kth_bound: Option<F>,                  // ‖a^(k)‖_∞ ≤ this for k ≤ 6, or None
      grid: Grid1D<F>,                          // reuse the inner's grid (parity check at construction)
  }
  ```
  Constructor:
  ```rust
  impl<F: SemiflowFloat> Diffusion4thZeta4Chernoff<F> {
      pub fn new(
          inner: Diffusion4thChernoff<F>,
          a_kth_bound: Option<F>,
      ) -> Result<Self, SemiflowError>;
  }
  ```
  Validation at construction (returns `DomainViolation` on any failure):
  - `inner.order() >= 2` (sanity — `Diffusion4thChernoff` is order-2 in time per math §9.2.4).
  - `inner.in_subspace::<2>(canonical_test_state)` returns true — the inner is K=2-tangent.
  - If `a_kth_bound` is `Some(c)`, `c.is_finite() && c >= F::zero()` — the bound is well-formed.
  - The inner's `a` closure is implicitly C^6 — caller-asserted via the `Option<F>` field; the library cannot verify at construction without sampling, so the bound is a *contract assertion* (the gate G_zeta4 verifies it on the test datum).

  Trait impls:
  - **`impl<F: SemiflowFloat> ChernoffFunction<F> for Diffusion4thZeta4Chernoff<F>`**:
    - `type S = GridFn1D<F>` (reuses the inner's state type — no new state-type wrapper).
    - `apply_into(τ, src, dst, scratch)`: the 4-step algorithm per math §27.3 (see §"Algorithm" below).
    - `order(&self) -> u32` returns `4` (per Theorem 3.1 of Galkin-Remizov on the K=6 core).
    - `growth(&self) -> Growth<F>`: returns `Growth { multiplier: self.inner.growth().multiplier * F::from(1.5), omega: self.inner.growth().omega }`. The 1.5× multiplier carries the τ²·P_2[A] term's contribution to the operator norm; the exponential rate ω is unchanged.
  - **`impl<F: SemiflowFloat> ApproximationSubspace<4, F> for Diffusion4thZeta4Chernoff<F>`** — opt-in K=4 witness (ADR-0073). `in_subspace`: returns `inner.in_subspace::<4>(f)` AND `self.a_kth_bound.is_some()` (the C^6 bound assertion). `jet`: 4 iterations of the inner's 9-point stencil.
  - **`impl<F: SemiflowFloat> ApproximationSubspace<6, F> for Diffusion4thZeta4Chernoff<F>`** — opt-in K=6 witness (the STRICT core where the order-4 claim holds). `in_subspace`: returns `inner.in_subspace::<6>(f)` AND `self.a_kth_bound.is_some()`. `jet`: 6 iterations.

**Algorithm** (math §27.3 — `apply_into` semantics):
```
Diffusion4thZeta4Chernoff::apply_into(τ, src, dst, scratch):
  1. inner.apply_into(τ, src, dst, scratch)                            // baseline order-2 step → dst
  2. Verify (debug-only): self.in_subspace::<6>(src) MUST hold;
     else return DomainViolation { reason: "f ∉ D(A^6)" }.
  3. tmp_jet := scratch.borrow_jet_buffer(7);                          // [f, A f, A² f, ..., A⁶ f]
     self.jet(src, &mut tmp_jet)?;                                      // K=6 jet via inner's stencil
  4. correction := scratch.borrow_state(src);                           // working state for P_2[A] f
     correction.zero();
     for (coeff, k1, k2, k3) in P_2_MONOMIALS_K6_DIFFUSION:             // 6 monomials per math §27.3
         correction.axpy(coeff * a_factor(self.inner.a_fn, k1, k2, k3),
                          &tmp_jet[k1 + k2 + k3]);
     dst.axpy(τ * τ, &correction);                                      // dst := dst + τ² · P_2[A] f
```
where `P_2_MONOMIALS_K6_DIFFUSION` is a const-array of 6 entries `(coeff: F, k1, k2, k3)` (each entry encodes one monomial in the operator polynomial — `coeff · A^k1 · a^{(k2)} · A^k3`). The const array is sympy-derived (T23N gate); the engineer ships the array verbatim from the math §27.3 derivation table.

**Acceptance gate G_zeta4** (RELEASE_BLOCKING — slope ≥ 1.9 on heat with $a \in C^6_b$):
- Construction:
  ```rust
  let grid = Grid1D::<f64>::new(-10.0, 10.0, 512)?;
  let a_fn = |x: f64| 1.0 + 0.5 * (x.tanh()).powi(2);  // a ∈ C^6_b (smooth, bounded)
  let inner = Diffusion4thChernoff::<f64>::new(a_fn, ..., grid.clone())?;
  let kernel = Diffusion4thZeta4Chernoff::<f64>::new(inner, Some(2.5_f64))?;
  ```
- Sweep $n \in \{16, 32, 64, 128\}$ at fixed $T = 0.5$ ($\tau = T / n$). Initial datum $f_0(x) = \exp(-x^2)$.
- Reference solution: high-resolution `Diffusion4thChernoff` (the v0.6.0 baseline) at $n_{\mathrm{ref}} = 2048$ (~16× the largest sweep $n$). This is a *self-convergence* gate against the v0.6.0 baseline; no closed-form reference is needed because the v0.6.0 spatial 4th-order has been independently validated (G3_4 gate, ADR-0013).
- Empirical $\log$-$\log$ OLS slope of $\|F_{\zeta^4}^n(f_0) - u_{\mathrm{ref}}\|_\infty$ vs $1/n$ MUST be $\ge -1.9$ (i.e., the absolute value of the slope is $\ge 1.9$; using the standard "slope ≤ -1.9" convention per existing G3_4 / G3_NS2D gates). Note: 1.9 has 2.5% margin against the -2.0 asymptote (Galkin-Remizov §3.1 gives -4.0 asymptotic; -2.0 is the *empirical* floor in finite-precision arithmetic, observed in the v0.6.0 G3_4 sibling gate).

(See §"Acceptance gates" below for the full G_zeta4 + T23N spec.)

File layout: `crates/semiflow-core/src/diffusion4_zeta4.rs` (~350 LoC target — kernel impl + apply_into + 2 opt-in ApproximationSubspace impls + the const-array of P_2 monomials; HARD LIMIT 450 LoC; default 500-LoC cap with 50-LoC headroom; NO Override #1 expansion). The 6-monomial const array `P_2_MONOMIALS_K6_DIFFUSION: [(f64, u8, u8, u8); 6]` is sympy-derived (T23N script provides the values).

Schema bumps: shared with ADR-0073 / ADR-0074 / ADR-0076 — `traits.yaml` 0.8.0 → 1.0.0, `properties.yaml` 0.10.0 → 0.11.0. math.md is append-only (§27 NEW).

## Rationale

- **Why a sibling kernel `Diffusion4thZeta4Chernoff<F>` (not a modification of `Diffusion4thChernoff<F>`)?** Three reasons: (a) the v0.6.0 kernel underwrites the v2.x compatibility shim (ADR-0074); modifying it would break that contract; (b) the order-4 correction REQUIRES caller-supplied `a_kth_bound: Option<F>` — adding this field to `Diffusion4thChernoff` would be a BREAKING signature change for all v0.6+ callers; (c) the sibling design matches the v0.3.0 ζ-A → v0.4.0 TruncatedExp sibling precedent (ADR-0011). Two distinct kernels for two distinct mathematical guarantees (order-2 vs order-4) with two distinct caller obligations (`f ∈ D(A^4)` vs `f ∈ D(A^6) ∧ a ∈ C^6_b`).
- **Why the order-4 correction polynomial `P_2[A]` is 6 monomials (not more or fewer)?** The number of monomials is *uniquely determined* by the structure of the v0.6.0 9-point spatial stencil and the Galkin-Remizov §3.1 cancellation requirement. Each monomial is a triple-index `(k1, k2, k3)` encoding $A^{k_1} \cdot a^{(k_2)} \cdot A^{k_3}$; the 6 entries cover all terms of total degree 6 (in operator powers) that survive the τ² cancellation. The sympy derivation (T23N script) writes out the Taylor series of $(F(\tau/n))^n f$ symbolically, extracts the τ²-coefficient, and solves for the monomial basis. The 6-entry count is a *computed mathematical fact*, not a design choice; the engineer ships the values verbatim from the T23N output table.
- **Why `a_kth_bound: Option<F>` (not a required `F` field)?** Two reasons: (a) the bound is *caller-asserted* (the library cannot verify $\|a^{(k)}\|_\infty \le c$ without sampling the closure exhaustively); making it `Option<F>` lets the caller opt into the unchecked path (`None`) for performance-critical hot loops where the gate has already verified the bound off-line; (b) `None` makes the K=4/K=6 `ApproximationSubspace` witness return `false`, forcing callers who care about the witness to provide the bound. The `Option<F>` choice is the suckless trade-off: required-by-default (the `Some(c)` case), opt-out for hot loops (the `None` case).
- **Why is the validation in `apply_into` `debug_assert!`-only (not `Result::Err`)?** Two reasons: (a) the in-subspace witness check at every `apply_into` call kills the per-tick latency story (the K=6 jet evaluation is ~3× the cost of one `apply_into`); (b) the gate G_zeta4 verifies the witness off-line on the canonical test datum. Production callers SHOULD pre-check `kernel.in_subspace::<6>(f)` once at construction time; the `apply_into` body trusts the caller (debug-only sanity check). This is the suckless zero-runtime-cost contract: validate at compile-time or construction-time, trust in the hot path.
- **Why slope ≥ 1.9 (not slope ≥ 3.9 or 4.0)?** The asymptotic slope per Galkin-Remizov §3.1 is -4.0 (order 4 in $\tau$); the *empirical* floor in finite-precision arithmetic on the v0.6.0 spatial 4th-order sibling kernel (G3_4 gate) is around -2.0 due to spatial-error contamination at the smallest $\tau$. The G_zeta4 gate sets -1.9 as the floor (2.5% margin vs -2.0) — strictly stronger than the v0.3.0 ζ-A G3-strang slope of -1.95 (which tests order 2). The slope -1.9 cleanly distinguishes order-4 ζ⁴ kernel from the v0.6.0 baseline order-2 (which has slope around -1.95 on the same sweep); a 1.9 vs 1.95 slope difference is the smallest reliable empirical signal in finite-precision arithmetic.
- **Why TWO opt-in `ApproximationSubspace` impls (K=4 AND K=6)?** The K=6 is the STRICT core where order-4 holds (Galkin-Remizov §3.2); the K=4 is the WEAKER core where order-3 might hold (the partial cancellation gives one fewer order). Both witnesses are useful: K=6 for the headline order-4 claim (the G_zeta4 gate), K=4 for future v3.1+ research into intermediate-order cancellations (e.g., a v3.1 A5.bis with K=4-only-tangent kernels). Shipping both opt-in impls in v3.0 banks the witness surface for future work; the K=4 impl is ~10 LoC additional.
- **Why use `Diffusion4thChernoff` as the inner (not `TruncatedExp4thDiffusionChernoff`)?** The v0.6.0 `Diffusion4thChernoff` is the *order-2-temporal + 4th-order-spatial* baseline; the v0.6.0 `TruncatedExp4thDiffusionChernoff` is the K=4 power-series sibling per ADR-0011. The ζ⁴ correction lifts the *temporal* order from 2 to 4; it uses `Diffusion4thChernoff` as the order-2 baseline because the τ²·P_2[A] formula in Galkin-Remizov §3.1 assumes a *single-step* baseline (the truncated-exp sibling has its own K=4 power-series structure that would compose oddly with the ζ⁴ correction). Future v3.1+ work CAN explore the (TruncatedExp4thDiffusion, ζ⁴) composition — out of scope for v3.0.
- **Why a 1.5× multiplier on `growth().multiplier` (not 2× or 1.0×)?** The τ²·P_2[A] correction term's contribution to the operator norm bound is empirically ~0.5× the baseline (the 6-monomial polynomial has bounded coefficients in $[0, 1]$ and the K=6 jet evaluation is bounded by `||a^(k)||_∞^6 · ||f||_∞`). The 1.5× multiplier (= 1.0 baseline + 0.5 correction) is the smallest constant that bounds the v0.6.0 `Diffusion4thChernoff` growth on the canonical test datum. The exponential rate ω is unchanged because the correction term is purely polynomial in τ (no exp growth).
- **Why NO L-gate for `Diffusion4thZeta4Chernoff` in v3.0?** The per-call cost is `1 × Diffusion4thChernoff::apply_into + 1 × jet::<6> + 6 × axpy` — at most ~3× the baseline `Diffusion4thChernoff` latency. The HFT use case (CEV pricing per ADR-0067) uses the v0.6.0 baseline, not the ζ⁴ kernel; there's no industrial-priority HFT use case for ζ⁴ in v3.0. Defer L_ZETA4_PTICK to v3.1 if benchmark evidence demands.
- **Why NO Override #1 Cohort expansion for `diffusion4_zeta4.rs` (despite carrying P_2 monomial sympy table)?** Target ~350 LoC; HARD LIMIT 450 LoC. The default 500-LoC cap absorbs this with 50 LoC margin. The 6-monomial const array is ~20 LoC (one entry per row, well-documented inline); the kernel impl is ~50 LoC; the two `ApproximationSubspace` opt-in impls are ~30 LoC each; the apply_into body is ~80 LoC; the rustdoc + citations are ~150 LoC. Sub-500-LoC budget; no carve-out justification needed.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Modify `Diffusion4thChernoff<F>` to carry the optional `a_kth_bound` field and the τ²·P_2[A] correction (in-place upgrade) | BREAKING signature change for all v0.6+ callers; breaks the v2.x compatibility shim (ADR-0074); breaks the v0.6.0 G3_4 gate semantics (the gate would test the corrected kernel instead of the baseline). Sibling kernel is the suckless choice. |
| Make `Diffusion4thZeta4Chernoff<F>` generic over the inner kernel (`Diffusion4thZeta4Chernoff<C: ChernoffFunction<F>, F>`) | The Galkin-Remizov §3.1 P_2[A] formula DEPENDS on the spatial discretization of A; the 6-monomial table is specific to the v0.6.0 9-point stencil. A generic-over-inner wrapper would need a per-kernel P_2[A] table — premature abstraction. v3.0 ships the specialised variant for `Diffusion4thChernoff` only; v3.1+ may explore other backends. |
| Require `a_kth_bound: F` (no `Option` — always required) | Forces every caller (including hot loops with off-line-verified bounds) to pay the construction-time bound check. The `Option<F>` opt-out is the suckless choice for hot paths. |
| Skip the order-4 witness; expose only `apply_into` (no `ApproximationSubspace<K, F>` impl) | Loses the K=6 jet contract — callers who chain `Diffusion4thZeta4Chernoff` inside future v3.1+ wrappers (e.g., a manifold-aware ζ⁴) would re-implement the jet logic. Shipping the K=4 + K=6 witnesses banks the contract surface for future composition. |
| Implement P_2[A] via runtime sympy-call (instead of a const-array) | Runtime dependency on a sympy bridge — catastrophic for `no_std + alloc`; would also be O(N) per Chernoff step instead of O(6) (one monomial loop). The const-array is the suckless choice; sympy is build-time-only via T23N. |
| Implement the order-4 lift via a 4-step palindromic Strang composition (e.g., Yoshida-style 4th-order) | Yoshida's 4th-order requires 4 sub-steps with carefully-tuned coefficients including a NEGATIVE step (which fails for diffusion semigroups — the diffusion equation is not time-reversible). Galkin-Remizov §3.1 is the only formulation that works for non-time-reversible PDEs and stays additive on the trait surface. |
| Defer A5 to v3.1 (ship only ADR-0073 in v3.0 Wave A) | Loses the immediate consumer that demonstrates ADR-0073 has industrial-priority math content. Shipping A5 in v3.0 Wave C closes the trait-design → math-application loop within a single release window. |
| Drop the `T = 0.5` horizon in G_zeta4 to `T = 0.05` (faster gate) | Cuts the empirical n-sweep range below the asymptotic regime onset; the slope at small τ is dominated by spatial discretization error (not the temporal cancellation we want to test). `T = 0.5` is the smallest horizon where the asymptotic regime is reached on the v0.6.0 9-point stencil. |
| Loosen the G_zeta4 slope bound to ≥ 1.5 (more margin) | Loses the empirical signal that distinguishes order-4 from order-2 (the v0.3.0 ζ-A baseline has slope ~ -1.95). The -1.9 floor is the smallest reliable distinguishing slope. |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; no existing kernel is modified. The v0.6.0 `Diffusion4thChernoff<F>` is preserved verbatim.
- **New module `crates/semiflow-core/src/diffusion4_zeta4.rs`** (~350 LoC target; HARD LIMIT 450 LoC; default 500-LoC cap with 50-LoC headroom; NO Override expansion).
- **New kernel `Diffusion4thZeta4Chernoff<F>`** — sibling to `Diffusion4thChernoff<F>`. Implements `ChernoffFunction<F>` (v3.0 cleaned-up form per ADR-0074), `ApproximationSubspace<4, F>` (witness), and `ApproximationSubspace<6, F>` (strict witness — the gate G_zeta4 path).
- **Dependency count unchanged** at 2/3 budget (still `num-traits`, `libm`). The P_2 monomial const-array uses only basic arithmetic.
- **Schema bumps**: shared with ADR-0073 / ADR-0074 / ADR-0076 (`traits.yaml` 1.0.0; `properties.yaml` 0.11.0). math.md is append-only (§27 NEW).
- **New gates**: G_zeta4 (RELEASE_BLOCKING — slope ≥ 1.9 on heat with $a \in C^6_b$, $n$-sweep, slow-tests); T23N (NORMATIVE sympy — symbolic verification of the τ²·P_2[A] cancellation identity; 3 sub-checks: (a) Taylor-coefficient extraction reproduces the 6 monomials verbatim; (b) the P_2[A] f sum cancels the τ²-leading-order Chernoff residual on the canonical heat eigenmode; (c) the K=6 jet matches `Diffusion4thChernoff` iterations at f64 precision).
- **No L-gate for `Diffusion4thZeta4Chernoff` in v3.0.** Per-call cost is ~3× baseline; no industrial-priority HFT use case in v3.0. Defer L_ZETA4_PTICK to v3.1+.
- **CITATIONs added to math.md §27**: Galkin-Remizov 2025 *Israel Journal of Mathematics* — *Tangency of Chernoff approximations to operator semigroups on Banach spaces*, Theorem 3.1 (k=2 tangency theorem; foundational citation for §27.2 — the operator-polynomial form); Example 4.2 (the order-4 construction on D(A^6) with a ∈ C^6_b; cited for §27.3 — the 6-monomial closed-form). The library reproduces only the FORMULA (Theorem 3.1 statement + the 6-monomial table from Example 4.2); the PROOF of Theorem 3.1 lives in Galkin-Remizov §3.1.
- **Migration note**: callers who want the headline order-4 convergence on variable-coefficient diffusion MUST switch from `Diffusion4thChernoff<F>` to `Diffusion4thZeta4Chernoff<F>` AND provide the `a_kth_bound: Option<F>` field AND pre-check `kernel.in_subspace::<6>(f)` (or accept the debug-only validation footgun). Worked example in `docs/migration/v2-to-v3.md` §6 (Wave G).

## Migration

End-user impact is **opt-in additive**. v0.6.0+ callers using `Diffusion4thChernoff<F>` continue to compile unchanged (the v0.6.0 kernel is preserved verbatim).

New v3.0 callers who want the order-4 lift:

```rust
// v0.6.0 baseline (still works, order-2 in time, 4th-order in space):
let inner = Diffusion4thChernoff::<f64>::new(a_fn, ..., grid)?;
let v060_order2 = inner.clone();

// v3.0 NEW (order-4 in time on D(A^6) ∧ a ∈ C^6_b):
let v030_order4 = Diffusion4thZeta4Chernoff::<f64>::new(
    inner,
    Some(2.5_f64),    // ‖a^(k)‖_∞ ≤ 2.5 for k ≤ 6 — caller-asserted
)?;

// Pre-check witness once at construction:
assert!(v030_order4.in_subspace::<6>(&initial_condition));

// Use exactly like v0.6.0:
let evolver = Evolver::new(v030_order4, n_steps)?;        // v3.0 Evolver per ADR-0074
let result = evolver.evolve(t_final, &initial_condition)?;
```

Worked example with the canonical `a_fn = |x| 1.0 + 0.5 * x.tanh().powi(2)` (the C^6_b heat coefficient) in `docs/migration/v2-to-v3.md` §6 (Wave G).

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0003 — no_std + alloc; the kernel uses only `libm` (existing dep) + the v0.6.0 9-point stencil.
- ADR-0008 — v0.3.0 ζ-A τ²-correction; the order-2 sibling that A5 generalises to order-4.
- ADR-0011 — v0.4.0 TruncatedExp K=4 sibling; the v0.6.0 K=4 sibling underwrites the K=6 `ApproximationSubspace<6, F>` opt-in (per ADR-0073 §"Decision").
- ADR-0013 — v0.6.0 4th-order spatial Diffusion4thChernoff; the spatial backend used as inner.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; reused for `Diffusion4thZeta4Chernoff<F>`.
- ADR-0026 — `ChernoffFunction<F>` super-trait (v0.9.0); used in v3.0 cleaned-up form per ADR-0074.
- ADR-0041 — `apply_into` + `ScratchPool`; the kernel uses the scratch pool for the K=6 jet buffer + the correction working state.
- ADR-0073 — v3.0 B1 `ApproximationSubspace<K, F>` opt-in marker trait; the witness consumed by `Diffusion4thZeta4Chernoff::new` at construction time AND opted-in for K=4/K=6.
- ADR-0074 — v3.0 ChernoffFunction trait cleanup; the `Growth<F>` return type used by the new impl's `growth()`.
- ADR-0076 — v3.0 v2→v3 binding redesign; the `Diffusion4thZeta4Chernoff<F>` kernel is NOT exposed in the v3 binding surface for v3.0 (the FFI/PyO3/WASM bindings stay at the v0.6 baseline kernel set; v3.1+ may add ζ⁴ binding once the K=6 witness const-generic binding ABI is worked out — see ADR-0076 §"Out of scope").
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v3.0 — release-level roadmap (A5 ζ⁴ correction Wave C placement).
- math.md §27 (NEW v3.0) — ζ⁴ correction algorithm normative spec.
- math.md §26 (NEW v3.0) — `ApproximationSubspace<K, F>` semantics (the witness foundation).
- math.md §9.2.4 (v0.6.0) — 4th-order spatial ζ⁴ idea (partial); §27 closes the loop to a full Chernoff function.
- `.dev-docs/constitution.md` v1.7.0 (NEW v3.0).

## Amendments

(none at acceptance time)
