# ADR-0189 — ETD / φ-function actions and ETDRK4 semilinear driver (identity-compatible)

- **Status**: Proposed (design only — Issue #12; worktree `sf-issue-12`,
  branch `issue-12-etd-phi-functions`). No `.rs` bodies; no version assigned.
- **Date**: 2026-06-27
- **Authors**: ai-solutions-architect
- **Supersedes**: none — purely **ADDITIVE**. Reuses ADR-0121/§45
  (`DiffusionExpmvChernoff` truncated-Taylor **matvec-only** action — the 1-D
  engine), ADR-0185/§54 (A1 `GraphKrylovChernoff` Chebyshev/Lanczos action,
  with the stiff Chebyshev-substep fix), ADR-0186/§55 (`SymmetricLinearOp` /
  `SymmetricOperator` generic operator entry point), ADR-0187/§56
  (conservative divergence-form carrier), ADR-0188/§57 (stiff multilayer),
  ADR-0125 (`mat_exp_pade13` dense oracle), ADR-0179 (callback ABI:
  batched **sample-once-at-construction** pattern), ADR-0034 (per-node
  callback 200× / GIL-defeat measurement), ADR-0031 (`py.detach`).
- **Contract**: `contracts/semiflow-core.math.md` §58 (new NORMATIVE section);
  gates `G_PHI_AUG_DENSE`, `G_ETDRK4_ORDER`, `G_ETD_ADJOINT_FD`.

## Context — the semilinear gap and why ETD is the identity-compatible extension

The library's moat is the **exact linear semigroup** `e^{τL}` (Chernoff /
Krylov / Padé actions). Every kernel to date is **linear**: `∂ₜu = Lu`. The
one structural gap is **semilinear** evolution `∂ₜu = Lu + N(u)` — the form of
Allen–Cahn, Kuramoto–Sivashinsky, Burgers, Gray–Scott, Fisher–KPP, and the
reaction-diffusion / phase-field family. Issue #12 asks for the **one extension
that does not replace `e^{τL}`** but composes with it.

Exponential time differencing (ETD) is exactly that. The exact variation-of-
constants solution over one step `h`,
`u(t+h) = e^{hL}u(t) + ∫₀^h e^{(h−s)L} N(u(t+s)) ds`,
treats `L` **exactly** (our existing action) and only **quadratures the
nonlinear Duhamel term** via the φ-functions. The competing routes — IMEX,
operator splitting, fully implicit — either re-discretise `L` (destroying the
exact-action moat) or split it (introducing splitting error on the linear part
we already nail). ETD is the unique route that keeps `e^{τL}` as the load-
bearing core and bolts the nonlinearity on top. This is the issue's TRIZ
framing: do not weaken the strong sub-system; add the new function in the
super-system around it.

### Grounding report (worktree code read, not gitnexus)

`gitnexus` was **not** used (no resolved index in this worktree session); all
findings are from direct `Read`/`grep` of the worktree. Key facts:

1. **`expmv.rs` (`DiffusionExpmvChernoff`)** realises `e^{τA}·v` by a **scaled
   truncated-Taylor Horner-on-vector** loop (`horner_step`), `(s,m)` chosen by
   `select_s_m` against the Al-Mohy–Higham `THETA_M` table. It is **matvec-only
   and symmetry-agnostic** — it already runs on the **non-symmetric** variable-
   `a(x)` divergence-form generator (`apply_div_form`). This is the engine φ_k
   reuses (see Decision).
2. **`graph_krylov.rs` (`GraphKrylovChernoff`)** has Chebyshev (Bessel
   coefficients `e^{−z}I_k(z)`, O(1) vectors) and Lanczos paths — **both
   require a symmetric operator** (`SymmetricLinearOp`, real spectrum on
   `[0,λ_max]`). Mirror `THETA_M`; stiff Chebyshev substepping `s=⌈z/Z_SAFE⌉`.
3. **DISCREPANCY (must be stated):** math §54.5 describes A2's Fréchet via an
   **augmented block-triangular operator** `[[−tL,−tE],[0,−tL]]` whose top-
   right block is the Fréchet derivative — *conceptually* the augmented
   mechanism the issue refers to. **But the shipped `graph_frechet.rs` does NOT
   build that operator**: it realises the same quantity by **8-point Gauss-
   Legendre quadrature of the Duhamel integral**. So the augmented *identity* is
   normative (§54.5) but **no callable augmented-action helper exists in code**.
   φ_k therefore does **not** call an existing routine; it **realises the
   augmented identity for the first time**, reusing the §45 matvec engine.
4. **Callback ABI (ADR-0179):** the blessed cross-language surface is a
   **batched sampler invoked once at construction**; **per-step** host callbacks
   (the `VarCoefMagnusGraph` time-dependent case) are an explicit
   **architectural non-goal** (per-step crossing reintroduces the ADR-0034 200×
   / GIL-defeat hazards). This directly constrains how `N(u)` may cross to
   Python/JS (see Decision C1).

## Decision

### D1 — φ_k via ONE augmented matvec-only Taylor action (reuse §45 engine)

For operator `A` (= the divergence-form generator for 1-D, or `−L` for graph),
build the **augmented block-upper-triangular** operator

```text
Ã = [[ τA,  v·e₁ᵀ ],   ∈ R^{(n+p)×(n+p)},   J ∈ R^{p×p} unit super-diagonal
     [  0,    J   ]]    (Jₖ,ₖ₊₁ = 1, else 0; nilpotent, Jᵖ = 0)
```

(v occupies the **first** augmented column only; τ is folded **inside** A).
Then, by the Duhamel/integral representation `φⱼ(B) = ∫₀¹ e^{(1−s)B} sʲ⁻¹/(j−1)! ds`,

```text
exp(Ã)[1:n, 1:n]    = φ₀(τA) = e^{τA}            (top-left block)
exp(Ã)[1:n, n+j]    = φⱼ(τA)·v ,   j = 1 … p     (top-right columns; NO τ power)
```

**ONE action of `exp(Ã)` on the augmented basis yields φ₀…φ_p·v
simultaneously.** Verified numerically to rel-err 5e-16 for p=3 (throwaway
probe, this ADR). The augmented operator is **non-symmetric** but
**block-triangular with equal symmetric diagonal blocks ⇒ its spectrum = A's,
entirely real**. The action is computed by the **matvec-only truncated-Taylor
Horner** path (the `expmv.rs` engine, symmetry-agnostic, already proven on the
non-symmetric 1-D generator), giving **one code path for both 1-D and graph**.
`Ã`'s matvec is one `A`-matvec + O(p) bookkeeping; the norm bound is
`‖Ã‖ ≤ ‖A‖ + ‖v‖ + 1`, so the existing `select_s_m` / `THETA_M` substepping
applies unchanged. **No Padé-on-φ, no contour integral, no new coefficient
mathematics.** (Resolution of contradiction C2 below; see §"TRIZ".)

### D2 — ETDRK4 driver (Cox–Matthews 2002 / Kassam–Trefethen 2005)

Per step `h`, in φ-form (all `e` and φ act on the **exact** `L`):

```text
a       = e^{hL/2} u_n + (h/2) φ₁(hL/2) N(u_n)
b       = e^{hL/2} u_n + (h/2) φ₁(hL/2) N(a)
c       = e^{hL/2} a   + (h/2) φ₁(hL/2) (2 N(b) − N(u_n))
u_{n+1} = e^{hL} u_n
        + h(φ₁ − 3φ₂ + 4φ₃)(hL) · N(u_n)
        + h(2φ₂ − 4φ₃)(hL)      · (N(a) + N(b))
        + h(4φ₃ − φ₂)(hL)       · N(c)
```

`L` is **never** re-discretised, inverted, or split — it stays the exact action.
The `(h/2)φ₁(hL/2)` coefficient is the identity-compatible form of
`L⁻¹(e^{hL/2}−I)` (the Kassam–Trefethen contour form is **not** needed because
the augmented action computes φ₁ directly, free of the cancellation KT's contour
integral was invented to dodge). Observed order **3.99 / 4.01 / 4.23** on
Allen–Cahn 1-D (throwaway probe). One step needs: `φ₀,φ₁(hL/2)` (one augmented
action per distinct argument) and `φ₀…φ₃(hL)` (one augmented action) — a handful
of augmented actions, all on the cached `L`.

Exponential Rosenbrock (Hochbruck–Ostermann 2010) is **deferred** (named wall):
it needs the Jacobian `J_N` as part of the *linear* operator each step, i.e. a
**time-/state-dependent** augmented `A = L + J_N(u_n)` — which forfeits the
"L cached once" property and lands in the per-step-varying-operator regime
ADR-0179 walls off. ETDRK4 (fixed `L`) is the phase-1 driver.

### D3 — N(u) as data, not per-step foreign code (resolution of C1)

`N(u)` is supplied as a **declarative spec evaluated natively in the loop**, never
as a per-step host callback:

- **Core (Rust):** `trait Nonlinearity<F>` (`eval(u, out)`), implemented either
  natively or as a small **opcode list** `Vec<NlOp>` (`Pow(k)`, `Mul`, `Add`,
  `Scale(c)`, `SpectralDeriv`, `Identity`, const pool) walked in pure Rust each
  step — `Send+Sync`, zero crossing, differentiable (each opcode carries its
  JVP/VJP).
- **Bindings (PyO3/WASM) phase-1:** a **fixed enum menu**
  (`AllenCahn{}`, `KuramotoSivashinsky`, `Burgers`, `GrayScott{}`) crossed
  **once at construction** (ADR-0179 sample-once lifetime) — **no per-step GIL
  crossing**, `py.detach` preserved byte-for-byte.
- **Arbitrary per-step Python/JS `N`:** explicit **deferred non-goal** (the
  ADR-0179 time-dependent-callback wall); the C `fn`-ptr / native-Rust trait is
  the only arbitrary-N path, and its per-step-crossing cost is documented as
  inherent. No silent fallback — absence is compile-time.

### D4 — Adjoint survives one semilinear step (LINEAR-in-v φ_k)

`φ_k(τL)·v` is **linear in v** ⇒ differentiable; its transpose-action is
`φ_k(τLᵀ)` (for symmetric graph `L`, `Lᵀ=L`, free; for 1-D, the transpose of
the augmented matvec — `Aᵀ` block). The ETDRK4 step is a composition of these
linear φ-actions with `N`. Given a **differentiable N** (the `Nonlinearity` JVP
/ VJP, or `NonlinearityDiff` trait), the whole step is differentiable by the
chain rule; the gradient `∂J/∂param` flows end-to-end through one step. Verified
by the FD gate (`G_ETD_ADJOINT_FD`), reusing the §43.6 finite-difference oracle
discipline — **no new sympy**.

## TRIZ — contradiction resolutions (mandatory contradiction-scan gate)

Two genuine design contradictions were found and **resolved** (not compromised):

- **C1 — flexible per-step N vs no per-step foreign crossing.** ФП: the N-spec
  must be *foreign/expressive* AND *native/crossing-free*. **Resolved in time +
  structure**: move N from *code* to *data* — the host declares N once (enum
  menu / opcode AST), the loop interprets it natively. Both properties held at
  once: arbitrary semilinear N within a closed differentiable algebra, zero
  per-step crossing. (D3.)
- **C2 — all φ_k in one action (augmented ⇒ non-symmetric) vs keep the fast
  symmetric kernel.** ФП: the operator must be *augmented* AND *symmetric-
  compatible*. **Resolved in structure** via the decisive resource — the
  augmented block has **real spectrum** and the φ are **functions** separable
  from the operator. Primary: the **matvec-only Taylor action never required
  symmetry** (route used; one path for 1-D + graph). Deferred optimisation:
  compute φ_k on symmetric graph `L` by swapping the **Chebyshev scalar
  coefficients** (`exp`→`φ_k`) — keeps the symmetric O(1)-memory kernel
  untouched. Neither halves the trade-off. (D1.)

## Consequences

- **Positive:** first semilinear capability; reuses the §45 matvec engine, the
  `THETA_M` substepping, and the `mat_exp_pade13` oracle with **zero new deps**
  and **zero new coefficient math**; φ_k usable standalone (exponential
  integrators generally) and inside ETDRK4; adjoint preserved; one unified
  augmented path for 1-D and graph operators.
- **Negative / honest boundaries:**
  - The augmented operator is **non-normal** (nilpotent rank-`p` coupling), so
    Taylor-action error bounds use the field-of-values, slightly weaker than the
    symmetric bound — **mitigated**: `p ≤ 3`, and `G_PHI_AUG_DENSE` verifies
    accuracy directly against dense `mat_exp_pade13`.
  - **ETDRK4 order reduction** is real for very stiff/non-smooth regimes
    (documented in the literature); the order gate fixes a smooth regime where
    the order-4 term dominates and uses a **two-sided** band so a degenerate
    regime cannot pass (§"Gates").
  - **Stiff N** (N itself stiff, e.g. fast reaction) is out of ETDRK4's comfort
    zone → deferred to exponential Rosenbrock (named wall, D2).
  - **Operators supported:** 1-D divergence-form (`Diffusion*`/`expmv` carrier)
    and symmetric graph / `SymmetricOperator`. Non-symmetric graphs, 2-D/3-D
    tensor operators → deferred.
  - **Adjoint requires a differentiable N** (JVP/VJP provided); a black-box
    non-differentiable N gets the forward step only.
  - **Per-step arbitrary Python/JS N** is not shipped (D3 wall).

## API sketch (Rust signatures only — NO bodies)

```rust
// ── Generator adapter (Hurd-translator-style): the ONE thing φ_k needs ────────
/// Matvec of the linear generator A (= div-form op for 1-D, = −L for graph)
/// plus a spectral-norm bound. Both existing operator classes get a thin impl;
/// the augmented action depends ONLY on this trait (replaceable / mountable).
pub trait GeneratorAction<F: SemiflowFloat>: Send + Sync {
    fn dim(&self) -> usize;
    /// dst ← A · src   (len == dim()).
    fn apply_generator(&self, src: &[F], dst: &mut [F]);
    /// Upper bound ρ̄ ≥ ‖A‖ (reuse a_norm_bound / spectral_radius_bound).
    fn norm_bound(&self) -> F;
    /// dst ← Aᵀ · src   (adjoint; symmetric ops may delegate to apply_generator).
    fn apply_generator_transpose(&self, src: &[F], dst: &mut [F]);
}

// Thin adapters (no new math): impl GeneratorAction for the divergence-form
// carrier (apply_div_form, a_norm_bound) and for any SymmetricLinearOp (−L).

// ── φ-function actions (D1) ───────────────────────────────────────────────────
pub const PHI_MAX: usize = 3;

/// φ₀…φ_p (τA)·v in ONE augmented matvec-only Taylor action.
/// `out` row-major (p+1)×n: out[k*n .. (k+1)*n] = φ_k(τA)·v,  k = 0..=p.
pub fn phi_action_batched<F, Op>(
    op: &Op, p: usize, tau: F, v: &[F],
    out: &mut [F], scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where F: SemiflowFloat, Op: GeneratorAction<F>;

/// Single-φ convenience: out ← φ_k(τA)·v  (k ≤ PHI_MAX).
pub fn phi_action<F, Op>(
    op: &Op, k: usize, tau: F, v: &[F],
    out: &mut [F], scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where F: SemiflowFloat, Op: GeneratorAction<F>;

// ── Nonlinearity (D3) ─────────────────────────────────────────────────────────
pub trait Nonlinearity<F: SemiflowFloat>: Send + Sync {
    fn eval(&self, u: &[F], n_out: &mut [F]) -> Result<(), SemiflowError>;
}
/// Differentiable extension required for the adjoint (D4).
pub trait NonlinearityDiff<F: SemiflowFloat>: Nonlinearity<F> {
    /// out ← J_N(u) · du           (forward / tangent)
    fn jvp(&self, u: &[F], du: &[F], out: &mut [F]) -> Result<(), SemiflowError>;
    /// out += J_N(u)ᵀ · w          (reverse / adjoint, accumulates)
    fn vjp(&self, u: &[F], w: &[F], out: &mut [F]) -> Result<(), SemiflowError>;
}

// ── ETDRK4 driver (D2) ────────────────────────────────────────────────────────
pub struct Etdrk4<F: SemiflowFloat, Op: GeneratorAction<F>, Nl: Nonlinearity<F>> {
    /* op (cached L), nl, step h, cached φ-substepping params */
}
impl<F, Op, Nl> Etdrk4<F, Op, Nl>
where F: SemiflowFloat, Op: GeneratorAction<F>, Nl: Nonlinearity<F> {
    pub fn new(op: Op, nl: Nl, h: F) -> Result<Self, SemiflowError>;
    /// One Cox–Matthews ETDRK4 step.
    pub fn step(&self, u: &[F], u_next: &mut [F],
                scratch: &mut ScratchPool<F>) -> Result<(), SemiflowError>;
    /// n_steps fixed-h integration into `out`.
    pub fn integrate(&self, u0: &[F], n_steps: usize, out: &mut [F],
                     scratch: &mut ScratchPool<F>) -> Result<(), SemiflowError>;
}
```

PyO3 note: `Op` and `Nl` are constructed under the GIL once, then `py.detach`
for the loop (ADR-0031 preserved); `Nl` for bindings is the fixed enum menu
(D3), never a per-step Python callable.

## Gates (RELEASE_BLOCKING, `feature_gate: slow-tests`) — NON-VACUOUS

| Gate | Definition | Threshold (two-sided where applicable) | Oracle / teeth |
|------|-----------|----------------------------------------|----------------|
| `G_PHI_AUG_DENSE` | For **each** `k ∈ {0,1,2,3}`: `phi_action(op,k,τ,v)` vs **independent** dense reference = top-left / top-right block of `mat_exp_pade13` on the small dense augmented `Ã` (N ≤ 12). | `sup_error ≤ 1e-10` AND assert `k` spans `0..=3` AND `z = τ·ρ̄ ∈ [0.5, 5]` (non-trivial: φ_k must differ materially from their `z→0` limit `1/k!`). | dense `mat_exp_pade13` (REUSE, no sympy). **Teeth:** the `z∈[0.5,5]` assertion forbids the `z→0` regime where every φ_k≈1/k! and a wrong kernel would still pass. |
| `G_ETDRK4_ORDER` | Allen–Cahn 1-D (`∂ₜu = ε u_xx + u − u³`, periodic, ε,h,IC in the smooth order-4-dominant regime) self-convergence vs fine reference; fit log-log slope over ≥4 step sizes. | slope ∈ **`[3.7, 4.3]`** (TWO-SIDED). | fine-reference / Richardson (no sympy). **Teeth:** two-sided band rejects BOTH order-reduction (<3.7, stiff/degenerate) AND a flat roundoff plateau (slope→0); regime chosen so the order-4 term dominates (probe: 3.99/4.01/4.23). |
| `G_ETD_ADJOINT_FD` | `∂J/∂param` through **ONE** `Etdrk4::step` (via `NonlinearityDiff` adjoint) vs central FD `(J(p+δ)−J(p−δ))/(2δ)`, with a **non-trivial N** (Allen–Cahn `u−u³`, Jacobian `1−3u²` state-dependent) and `param` flowing **through N**. | rel-err `≤ 1e-6`. | central FD (REUSE §43.6 discipline, no new oracle). **Teeth:** `param` must enter the **nonlinear** term so `J_N`'s contribution to the gradient is exercised; a linear-only adjoint bug fails. |

Oracle reuse only; **no new sympy required** — the φ_k identity is verified
against dense `mat_exp_pade13`, order against a PDE benchmark, adjoint against
FD. Flag: if a symbolic confirmation of the augmented-block column mapping is
ever wanted it is a one-liner, but it is redundant with `G_PHI_AUG_DENSE`.

## Suckless / minimalism

- **Zero new dependencies** (reuses `expmv` Horner, `THETA_M`, `mat_exp_pade13`,
  `ScratchPool`).
- New files, each ≤500 LoC, fns ≤50 LoC, `include!`-split per house style:
  `phi_action.rs` (+ `phi_action_helpers.rs` augmented matvec + `phi_action_tests_mod.rs`),
  `etdrk4.rs` (+ `etdrk4_helpers.rs` stage assembly + `etdrk4_tests_mod.rs`),
  `nonlinearity.rs` (trait + opcode interpreter + enum menu),
  `generator_action.rs` (the two thin adapters).
- One augmented action path serves 1-D and graph → minimal surface.

## Implementation ordering (for the engineer; contract-first)

1. `GeneratorAction<F>` trait + two thin adapters (div-form carrier; `−L` over
   `SymmetricLinearOp`). No math, pure plumbing.
2. Augmented matvec + `phi_action_batched` / `phi_action` reusing `select_s_m` /
   Horner. **Gate `G_PHI_AUG_DENSE` first** — nothing proceeds until φ_k matches
   dense Padé ≤1e-10.
3. `Nonlinearity` trait + opcode interpreter + enum menu (Allen–Cahn, KS,
   Burgers, Gray–Scott) with per-opcode JVP/VJP.
4. `Etdrk4::{new,step,integrate}` assembling D2 stages from `phi_action_batched`.
   **Gate `G_ETDRK4_ORDER`** (two-sided [3.7,4.3]).
5. `NonlinearityDiff` adjoint wiring through one step. **Gate
   `G_ETD_ADJOINT_FD`** (≤1e-6, N-flowing param).
6. (Deferred, separate ADRs) Chebyshev-φ-coefficient symmetric optimisation;
   exponential Rosenbrock; PyO3/WASM enum-menu surface; 2-D/3-D operators.

## References

- S. M. Cox, P. C. Matthews (2002), *Exponential time differencing for stiff
  systems*, J. Comput. Phys. 176(2):430–455, DOI 10.1006/jcph.2002.6995.
- A.-K. Kassam, L. N. Trefethen (2005), *Fourth-order time-stepping for stiff
  PDEs*, SIAM J. Sci. Comput. 26(4):1214–1233, DOI 10.1137/S1064827502410633.
- M. Hochbruck, A. Ostermann (2010), *Exponential integrators*, Acta Numerica
  19:209–286, DOI 10.1017/S0962492910000048 (φ-functions, ETDRK4 table,
  exp. Rosenbrock).
- A. H. Al-Mohy, N. J. Higham (2011), *Computing the action of the matrix
  exponential…*, SIAM J. Sci. Comput. 33(2):488–511, DOI 10.1137/100788860
  (augmented-matrix φ construction §4; `expmv` `THETA_M`).
- R. B. Sidje (1998), *Expokit*, ACM TOMS 24(1):130–156 (augmented `phiv`).
- §45 (1-D `expmv`), §54 (graph Krylov action + §54.5 augmented Fréchet),
  §55 (`SymmetricLinearOp`), §43.6 (FD adjoint oracle); ADR-0121/0179/0185/0186.
