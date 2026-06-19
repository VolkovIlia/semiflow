# ADR-0016 — v0.7.0 Non-separable 2D operator (NonSeparable2DChernoff, Block C)

**Status**: Accepted
**Date**: 2026-05-02
**Authors**: ai-solutions-architect
**Supersedes**: none. Implements Block C of the approved plan
`/home/volk/.claude/plans/eager-greeting-gizmo.md` (v0.7.0). Partially
lifts the §10.7 deferral "non-separable 2D operators with cross-derivatives".
**Cross-refs**: ADR-0012 (tensor-product 2D — direct predecessor;
`Strang2D<X, Y>` is the $c \equiv 0$ degeneration), ADR-0011 (TruncatedExp
v0.4.0 — `SemiflowError::CflViolated` variant reused with semantic field
overload), ADR-0013 (4th-order spatial v0.6.0 — `Diffusion4thChernoff`
inner type compatibility), ADR-0014 (Adaptive PI controller — sibling
v0.6.0 ADR — `NonSeparable2DChernoff` is generic enough to nest inside
`AdaptivePI<C>`), `contracts/semiflow-core.math.md` §10.7-bis (NORMATIVE
for v0.7.0), `.dev-docs/verification/scripts/verify_v0_7_0_nonseparable.py`
(reproducible, exit 0), `.dev-docs/research/non-separable-2d-bch.md`
(researcher math-fidelity analysis with full BCH derivation and sympy
canonicalisation Appendix). Cites Hairer, Lubich, Wanner (2006) §III.5.2
and §V.2.1, McLachlan & Quispel (2002) on operator splitting,
Spiteri et al. (2023) arXiv:2302.08034 on 3-operator second-order
Strang generalisation, and Blanes, Casas, Murua (2024) Acta Numerica
on palindromic order conditions.

Adopt **`NonSeparable2DChernoff<X, Y>`** as a new public type next to
`Strang2D<X, Y>` (additive sibling, no replacement; v0.5.0 / v0.6.0 /
v0.7.0 callers using `Strang2D` remain bit-equal). The generator
covered is $L = L_x \otimes I + I \otimes L_y + c(x, y) \cdot
\partial_x \partial_y$ on a 2D tensor-product domain (Option γ in the
plan: scalar mixed-term coefficient $c$, simplest API; full
$\beta(x, y)$ second-derivative paths and 4th-cross-derivative legs
deferred to v0.8+). The composition is the **5-leg palindromic Strang**
$S_5(\tau) = e^{\tau A / 2}\,e^{\tau B / 2}\,e^{\tau M}\,e^{\tau B / 2}\,
e^{\tau A / 2}$ with $A := L_x \otimes I$, $B := I \otimes L_y$, $M :=
c \cdot \partial_x \partial_y$. The reduction theorem ($[A, B] = 0$
because per-axis lifts act on disjoint tensor factors per Lemma 10.1)
collapses the 5-leg form **exactly** (no $\tau$-truncation) to the
standard 2-operator Strang $e^{\tau (A + B)/2}\,e^{\tau M}\,e^{\tau (A + B)/2}$,
to which the textbook BCH machinery (HLW 2006 §III.5.2 + Theorem 2.2
of §V.2.1 for palindromic-symmetry odd-function logarithm) applies
verbatim. Local truncation residue is $\tau^3 \cdot Y_3 + O(\tau^5)$
with **NORMATIVE Y₃ formula**
$Y_3 = -\tfrac{1}{24}([A, [A, M]] + 2 [A, [B, M]] + [B, [B, M]]) +
\tfrac{1}{12}([M, [M, A]] + [M, [M, B]])$
(math.md eq. 10.7-bis.6, sympy gate T7N_τ³). Global error: **$O(\tau^2)$
— order 2** over $[0, T]$ with $N = T / \tau$ steps. The mixed leg uses
the **K=2 truncated Taylor expansion** $\Phi_M(\tau) = I + \tau M +
(\tau^2 / 2) M^2$ rather than the analytic $e^{\tau M}$ (which would
require solving a 2D PDE on its own — defeating the splitting). K=2 is
**mandatory**: K=1 drops local error to $O(\tau^2)$ → global $O(\tau^1)$
(order 1, NOT order 2); K=3 does not improve over K=2 (still $O(\tau^2)$
capped by Strang) but slightly reduces the leading constant. Spatial
discretisation of $\partial_x \partial_y$ uses the **4-point centred
cross-stencil** $(f_{i+1, j+1} - f_{i+1, j-1} - f_{i-1, j+1} +
f_{i-1, j-1}) / (4 \cdot \mathrm{dx} \cdot \mathrm{dy})$ with leading
truncation $O(\mathrm{dx}^2 + \mathrm{dy}^2)$ — chosen for minimal
stencil span (3×3 box) and clean per-axis `BoundaryPolicy` composition
at edge cells (apply X-axis `bc_value` for $i \pm 1$, then Y-axis
`bc_value` for $j \pm 1$, agreeing with `Reflect` and `ZeroExtend` on
$C^2$-regular extensions and exact for affine ghost data under
`LinearExtrapolate`). The $M^2$ term in $\Phi_M$ requires applying the
4-point cross-stencil twice (5×5 box support effectively). Stability
of $\Phi_M$ requires the **CFL gate** $4 \cdot \tau \cdot \|c\|_\infty
< \mathrm{dx} \cdot \mathrm{dy}$ ($\theta = 1/4$ choice — 2× safety
margin below the analytic limit $\theta < 1/2$ at the cross-stencil
Fourier symbol bound; matches the v0.4.0 `TruncatedExpDiffusionChernoff`
CFL choice). Violation is reported as
**`SemiflowError::CflViolated`** (reused from ADR-0011 — **+0 new error
variants**); rustdoc on the variant is amended to document the
semantic field overload for the non-separable case (`dx_squared` field
holds $\mathrm{dx} \cdot \mathrm{dy}$ and `a_norm_bound` holds
$\|c\|_\infty$ — names kept stable for non-breaking-change reasons).
Caller invariants on $c$: **$c \in C^3(\mathbb{R}^2)$** (NORMATIVE —
needed so $[A, [A, M]]$ containing $\partial_{xx} c$ is densely
defined); **$\|c\|_\infty < \infty$** on the grid domain; **ellipticity
$c^2 \le 4\,a\,b$** when $L_x, L_y$ are second-order parabolic with
coefficients $a(x), b(y)$ (doc-only, no runtime check — analogous to
the existing $a > 0$ caller invariant on `DiffusionChernoff`). The
`order()` method returns **2** (τ-axis Chernoff consistency, per
math.md §11.1.bis NORMATIVE — same `p = 2` rule that fixed the v0.6.1
D1 `Diffusion4thChernoff::order()` defect). Spatial accuracy is
$O(\mathrm{dx}^2 + \mathrm{dy}^2)$ globally (cross-stencil floor caps
higher-order per-axis legs); observed only via the slope gate
**G3_NS2D** (≤ -1.95 on $N \in \{32, 64, 128, 256\}$, $n = 200$,
$T = 0.5$, $a_{xx} = a_{yy} = 0.1$, $c = 0.05$) and the variable-$c$
sibling **G3_NS2D_var** ($c(x, y) = 0.05 \cdot \tanh(x + y)$ — exercises
the $[A, [B, M]]$ commutator that vanishes for constant $c$). Closed-form
oracle: **rotated 2D Gaussian** $u(t, x, y) = (\det \Sigma_t)^{-1/2}
\exp(-\tfrac{1}{2}\,\xi^\top \Sigma_t^{-1} \xi)$ with $\Sigma_t = \Sigma_0 +
2 t D$, $D = \bigl[\begin{smallmatrix} a_{xx} & c/2 \\ c/2 & a_{yy}
\end{smallmatrix}\bigr]$ (math.md eq. 10.7-bis.13; sympy gate
T7N_oracle confirms it solves the PDE identically). When $c \equiv 0$
the new type **MUST** detect this at construction time and branch to
the existing `Strang2D::apply` path (no $\Phi_M$-times-identity
multiplication, which would still differ in floating-point arithmetic;
sympy gate T7N_zero-c codifies the bit-equal degeneration). Rejected
alternatives:
**(a) Genuine $e^{\tau M}$ via 2D Krylov / Padé expansion**: would
preserve order 4 of palindromic Strang for the smooth-$c$ case but
costs an inner Krylov solve per step (2-3× slower per-step at
production grid sizes); deferred to v0.8+ pending profiling evidence
that order-2 is the bottleneck at the production scale where v0.7.0
operates.
**(b) K=3 truncated Taylor for the mixed leg**: same global order ($O(\tau^2)$
capped by Strang BCH); 50% more cost per step on the M leg; rejected
as ceremonial cost without observable benefit.
**(c) K=1 truncated Taylor**: drops global order to 1 (math.md
§10.7-bis.3 NORMATIVE proof); rejected — would require a separate
type and a forward-incompatible naming scheme.
**(d) Forsythe / Trotter (non-palindromic) splitting**: also order 1
in the non-commuting case; same rejection.
**(e) Pre-computing $c \cdot \partial_x \partial_y$ matrix coefficients
once at construction**: would amortise stencil cost at the price of
$O(\mathrm{dx}^{-2})$ memory per node; rejected as premature
optimisation (the cross-stencil itself is 4 multiply-adds per node;
production grids fit in cache).
**(f) Exposing $\theta$ as a runtime parameter**: rejected per
suckless-conventions guardrail (config in source, not API surface);
the choice $\theta = 1/4$ is documented and stable.
**(g) New `SemiflowError::CrossStencilCfl` variant**: rejected per
suckless-conventions guardrail (one error variant per *kind* of
failure; CFL stability of a polynomial-in-$M$ truncation is the same
*kind* as the v0.4.0 case — the rustdoc amendment is the right granularity).
Consequences: **+1 public type** (`NonSeparable2DChernoff`); **+0
dependencies**; **+0 `SemiflowError` variants** (CflViolated reused
with rustdoc-amendment for semantic field overload); **+0 changes** to
existing source files (`strang2d.rs`, `axis.rs`, `grid_fn2d.rs`,
`diffusion*.rs`, `truncated_exp*.rs`, `chernoff.rs`, `error.rs`); +1
new module `nonseparable2d.rs` of budget ~350 LoC. v0.5.0 / v0.6.0 /
v0.7.0 Block A / Block B callers using `Strang2D` and the 4th- /
6th-order types remain bit-equal — the new type is a strict addition.
The `c ≡ 0` zero-detection branch is at construction time only (a
single $f64$ comparison via a `c_norm_bound == 0.0` test on the
caller-supplied bound — **NOT** a per-step grid scan), satisfying the
suckless ≤50-line `apply` budget. **Public API exposure**:
`pub mod nonseparable2d; pub use crate::nonseparable2d::NonSeparable2DChernoff;`
in `lib.rs` (one new line plus one `pub mod` declaration; same shape
as ADR-0015's `Diffusion6thChernoff` introduction).

**Mechanism note (NORMATIVE, ratified by gate suite)**: the order-2
claim emerges JOINTLY from three components: **(a)** the palindromic
5-leg structure, which (combined with $[A, B] = 0$) reduces to standard
2-operator Strang with $\tau^2$ palindromic cancellation in
$\log(S_5)$; **(b)** the K=2 truncated Taylor mixed leg, which
matches the BCH residual order $O(\tau^3)$ locally; **(c)** the
4-point centred cross-stencil for $\partial_x \partial_y$, which on
$C^4$-smooth data delivers $O(\mathrm{dx}^2 + \mathrm{dy}^2)$ spatial
accuracy. Removing component (a) (e.g. switching to non-palindromic
Trotter) drops the order to 1; removing component (b) (e.g. K=1)
drops the order to 1; replacing component (c) with a 9-point biased
stencil would add cost without observable benefit at the order-2
ceiling. This decomposition is **provable** (math.md §10.7-bis.2 +
§10.7-bis.3) and is the mechanism the empirical G3_NS2D / G3_NS2D_var
slope gates verify.

**Forward compatibility**:

- **v0.8+** MAY add `NonSeparable2DAnisotropicChernoff<X, Y>` for the
  full-tensor $\beta(x, y) \cdot \partial_x \partial_y$ shape (Option α
  in the plan) without changes to the v0.7.0 type — additive over
  v0.7.0 shape, no API break.
- **v0.8+** MAY add a Krylov-based `NonSeparable2DKrylovChernoff` that
  evaluates the M leg via the matrix exponential to lift the global
  order to 4 — additive sibling, no replacement of v0.7.0.
- The v0.7.0 `NonSeparable2DChernoff::new` constructor is stable: any
  v0.5.0+ inner type satisfying `ChernoffFunction<S = GridFn1D> + Copy`
  may be plugged into the X or Y slot. Higher-order per-axis legs
  (e.g. `Diffusion6thChernoff`) compile and run correctly today; their
  spatial slope is gated by the cross-stencil $O(\mathrm{dx}^2)$ floor
  globally, but the per-axis half-step accuracy is preserved on the
  X-only and Y-only computations of the rotated Gaussian (so the
  diagonal of $\Sigma_t$ is recovered to higher accuracy than the
  off-diagonal — observable via a cross-section convergence test, not
  the global G3_NS2D gate).
- The `c_norm_bound` constructor parameter is a caller-supplied
  $\|c\|_\infty$ bound — analogous to the existing `a_norm_bound`
  parameter on `TruncatedExpDiffusionChernoff` (ADR-0011). v0.8+ MAY
  add an automatic-bound estimator (sample $c$ on the grid and take the
  max), but this is NOT v0.7.0 scope (suckless: explicit input over
  implicit estimator).

---

## Implementation note (v0.7.0 ship)

The `apply` algorithm follows the math.md §10.7-bis.2 sequence
literally: (1) CFL gate; (2) X half-step via `AxisLift::X` on $L_x$;
(3) Y half-step via `AxisLift::Y` on $L_y$; (4) mixed K=2 Taylor step
using the 4-point cross-stencil applied twice (for the $M^2 / 2$
term); (5) Y half-step; (6) X half-step; return. Each numbered step is
a separate ≤50-line helper function in `nonseparable2d.rs` to satisfy
the suckless function-size guardrail. The existing `axis.rs` and
`strang2d.rs` are NOT modified; the cross-stencil is implemented
locally in `nonseparable2d.rs` using `pub(crate) bc_value` from
`grid.rs` (existing helper, not exported publicly). The
`c: fn(f64, f64) -> f64` field is a function-pointer type (NOT
`Box<dyn Fn>`) to satisfy `Copy + Send + Sync` for nesting inside
`AdaptivePI<NonSeparable2DChernoff<X, Y>>` — same constraint as the
existing `DiffusionChernoff::a, b, c` field types. The
`c_norm_bound: f64` is the caller's $\|c\|_\infty$ estimate; the
`new` constructor validates `c_norm_bound >= 0.0` and finite, but
does NOT scan the grid (per suckless: explicit input over implicit
estimator; analogous to `TruncatedExpDiffusionChernoff::a_norm_bound`).
