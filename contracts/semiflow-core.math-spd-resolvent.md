# semiflow-core — Math Specification §62 (SPD resolvent / φ-combination, ADR-0202)

NORMATIVE. This file **is** §62 of the math contract `contracts/semiflow-core.math.md`,
split out verbatim for size (the pre-commit git-guard caps a single file at 1024 KB).
Section numbering, sub-section anchors (§62.1 …) and cross-references to other §§
refer to `contracts/semiflow-core.math.md`; citations of "§62" / "§62.x" resolve here.

## §62 — SPD resolvent / steady solve, operator composition, and the φ-combination (ADR-0202, NORMATIVE library; CITATION mathematics)

> **Scope.** The linear resolvent `(λM + A)⁻¹` of an externally assembled SPD sparse
> operator (§55), for `λ ≥ 0` with a positive diagonal mass `M`. `λ = 0` is the steady
> state. Also: diagonal composition `A + diag(c)`; φ-functions with a diagonal mass
> and on the non-symmetric §55.6/ADR-0195 `GeneralOperator`; the one-sweep
> φ-combination; and the removal of the `‖v‖`-dependence of the §58.2 cost.
> No new numerical *method* class. LDLᵀ, CG, IC(0) and the augmented-matrix
> φ-action are textbook (citations inline). What this section makes NORMATIVE is
> the dispatch, the certificates, and the gates.

### §62.1 — Setting and the resolvent identity (NORMATIVE)

Let `A ∈ ℝ^{n×n}` be symmetric positive semidefinite (the §55 `SymmetricOperator`
convention: evolution `e^{−tA}`). Let `M = diag(m)` with `mᵢ > 0`, and define the
generator `G = −M⁻¹A`. `G` is self-adjoint and dissipative in the `M`-inner product.
For `λ ≥ 0` with `λM + A` positive definite:

```text
R(λ; G) v := (λI − G)⁻¹ v = (λM + A)⁻¹ M v                                   [§62.1.a]
          = ∫₀^∞ e^{−λt} e^{tG} v dt        (Hille–Yosida; Pazy 1983 Thm 1.5.3)  [§62.1.b]
```

The library primitive solves `S x = b` with `S := λM + A`. Then `R(λ;G)v` is `solve(M v)`.
The **steady state** of `M u' = −A u + s` is `u* = A⁻¹ s` (`λ = 0`, `b = s`).
It is the `τ → ∞` limit of the affine evolution (§62.5):

```text
τ·φ₁(τG)·M⁻¹s = (−G)⁻¹(I − e^{τG})M⁻¹s = (I − e^{τG}) u* ,   ‖e^{τG}u*‖_M ≤ e^{−τλ_min}‖u*‖_M   [§62.1.c]
```

This is the identity behind gate `G_SPDR_PHI1_LIMIT`. It explains why the
polynomial route works but costs `O(κ)` matvecs. A polynomial `p` of degree `d`
with `|p(x) − 1/x| ≤ ε/x` on `[λ_min, λ_max]` needs `d ≳ ½√κ ln(2/ε)` (Chebyshev
optimum; CG attains it). Truncated Taylor with `s`-scaling needs `d ∝ τλ_max ∝ κ`.
Only a rational method (a solve) is independent of `κ` in 1-D. **Why this is in
scope** (ROADMAP amended): §62.1.b is the semigroup's own Laplace transform, already
used by §22 (`λ > 0`), §47 (contour nodes) and §59 (`λ = 1/Δt`). A steady solve is
`R(0)`, not a time-stepping scheme.

**Singular case.** `S` is singular iff `λ = 0` and `ker A ≠ {0}` (for example a pure
Neumann Laplacian without reaction: `A·1 = 0`). Rejection is NORMATIVE:
(i) a priori, `λ = 0` and `Σⱼ aᵢⱼ = 0` for every `i` ⇒ `DomainViolation`;
(ii) on the tridiagonal path, the pivot certificate of §62.2.a. On the PCG path a
singular-but-consistent `b` may converge (to some solution), and an inconsistent
`b` ends in `ConvergenceFailed`. That is documented, not hidden.

### §62.2 — Algorithms and dispatch (NORMATIVE)

**Dispatch (`SpdSolver::Auto`).** If every stored entry `aᵢⱼ` has `|i − j| ≤ 1`
(a structural test on CSR), use §62.2.a. Otherwise use §62.2.b with IC(0).

**§62.2.a — Symmetric tridiagonal LDLᵀ (exact, O(n)).** With `δᵢ = λmᵢ + aᵢᵢ` and
`βᵢ = a_{i,i+1}`:

```text
d₀ = δ₀;   for i = 1..n−1:  ℓᵢ = β_{i−1}/d_{i−1},  dᵢ = δᵢ − ℓᵢ β_{i−1}          (factor)
y₀ = b₀;   yᵢ = bᵢ − ℓᵢ y_{i−1};   x_{n−1} = y_{n−1}/d_{n−1};
           xᵢ = yᵢ/dᵢ − ℓ_{i+1} x_{i+1}                                         (solve)
```

*Certificate.* `S` is SPD ⟺ all `dᵢ > 0`, because `dᵢ` is a ratio of leading
principal minors (Sylvester). NORMATIVE rejection: `dᵢ ≤ n·ε·maxⱼ|δⱼ|` ⇒
`DomainViolation` ("not positive definite"). *Stability.* No pivoting is needed.
For SPD tridiagonal matrices Gaussian elimination is backward stable,
`(S + ΔS)x̂ = b` with `|ΔS| ≤ c·u·|S|` (Higham, *Accuracy and Stability*, 2nd ed.,
2002, §9.6), so `‖x − x̂‖/‖x‖ ≤ c·n·u·κ(S)`. The factor `(d, ℓ)` is built once in
`new` and reused by every `solve_into`, so the cost is `3n` flops per solve.

**§62.2.b — PCG with IC(0) or Jacobi.** Conjugate gradients on `S` (Hestenes–Stiefel
1952) with preconditioner `P`. Stop when `‖r_k‖₂² ≤ tol²·‖b‖₂²` (squared norms,
no `sqrt` in the loop). Warm start `x₀ = 0`. Cap `2n + 16` unless overridden.
`b = 0` gives `x = 0` with `iterations = 0`. Jacobi: `P = diag(S)`. IC(0): `P = L̃L̃ᵀ`
with `L̃` on exactly the lower pattern of `S` (Meijerink–van der Vorst 1977). The
factorisation exists with positive pivots whenever `S` is an M-matrix, which
holds for every FV/FD diffusion stencil with harmonic faces plus `λM + diag(c)`,
`c ≥ 0`. On a non-positive pivot, the §59.2 rule applies: fall back to Jacobi and
report it in `method()`.

**§62.2.c — Determinism.** Both paths use only `+ − × ÷`, plus `sqrt` (IC(0)
diagonal, the reported residual), routed through `SemiflowFloat` → `libm`
(ADR-0200). Summation order is fixed: rows ascending, then CSR order within a row;
no threads. Hence std and no_std results are bit-identical, which the nostd-check
digests verify.

### §62.3 — Composition and generators (NORMATIVE)

- `with_diagonal(c)`: `A' = A + diag(c)` with `cᵢ ≥ 0`. This keeps symmetry and PSD,
  and turns `ker A = span{1}` into `{0}` as soon as some `cᵢ > 0` (connected
  pattern). Semantics on the §56 assembled carrier, whose entries are `T/dx` with
  `T = k/dx`, i.e. the FD scaling of `−∂ₓ(k∂ₓ·)`: `A + diag(c)` discretises
  `−∂ₓ(k∂ₓu) + c u` pointwise. With half-cell FV mass `m = [½,1,…,1,½]` (units of
  `dx`), the conservative FV system is `(A + diag(c⊙m))u = m⊙s`. That is the form
  used by `G_SPDR_STEADY_MMS`.
- `CsrGenerator`: `G = −M⁻¹A`, with `apply_generator(v) = −(A v)⊘m` and an exact
  transpose `Gᵀv = −Aᵀ(v⊘m)`. Norm bound: `mass = None` reuses the operator's own
  bound (bit-identity with `NegLaplacianGenerator`, §55). Otherwise the row-wise
  Gershgorin bound `maxᵢ Σⱼ|aᵢⱼ|/mᵢ ≥ ‖G‖_∞`. It is the same φ-function of the
  same generator as the §55.3 congruence route
  (`φ(τG) = M^{−½}φ(−τÂ)M^{½}`, `Â = M^{−½}AM^{−½}`) and needs no square roots.

### §62.4 — φ-combination and the v-independent cost (NORMATIVE; amends §58.2)

**Theorem (Al-Mohy & Higham 2011, SIAM J. Sci. Comput. 33:488, Thm 2.1; stated
in the implemented, τ-absorbed form).** Let `p ≥ 1`, let `J ∈ ℝ^{p×p}` be the
unit super-diagonal nilpotent (`(Jc)_i = c_{i+1}` for `i < p−1`, `(Jc)_{p−1} = 0`;
`‖J‖_∞ = 1`, **not** scaled by `τ`), and let `W_τ ∈ ℝ^{n×p}` have column
`j = 0, …, p−1` equal to `τ^{p−j}·w_{p−j}` (so `W_τ = [τ^p w_p, …, τ w_1]`). With

```text
B̃ = [[τG, W_τ],[0, J]],   [I_n 0]·exp(B̃)·[w₀; e_p]  =  Σ_{k=0}^{p} τ^k φ_k(τG) w_k ,     [§62.4.a]
```

where `e_p` is the last unit vector of `ℝ^p` (augmented slot `n+p−1`). *Proof.*
`c(t) = e^{tJ}e_p` has entry `j` equal to `t^{k−1}/(k−1)!` with `k = p−j`, so the
top block solves `u' = τG u + Σ_{k≥1} τ^k w_k t^{k−1}/(k−1)!`, `u(0) = w₀`, and
`u(1) = e^{τG}w₀ + Σ_k τ^k ∫₀¹ e^{(1−t)τG} t^{k−1}/(k−1)! dt · w_k`, which is the
right-hand side by the integral definition of `φ_k`. ∎ Note `τ` sits inside the
`G`-block and the coupling weights, never on `J`; the exponent is `exp(B̃)`, not
`exp(τB̃)`.

**η-scaling (the cost fix).** For any `η > 0`, `D = diag(I_n, η⁻¹I_p)` gives
`D B̃ D⁻¹ = B̃_η := [[τG, ηW_τ],[0, J]]` and `D[w₀; e_p] = [w₀; e_p/η]`, so the
top block is unchanged; the initial vector carries `1/η` in slot `n+p−1`.
NORMATIVE choice (code: `weighted_columns`, `eta_scaling`):

```text
M = Σ_{k=1}^{p} τ^k ‖w_k‖_∞     (τ-WEIGHTED, SUMMED over columns; zero columns skipped)
η = 2^{−e},  e = ⌈log₂ M⌉ clamped to [−1000, 1000];   η = 1 if M = 0 or M non-finite.
```

`M` non-finite (`τ^k‖w_k‖_∞` overflow) is a typed `DomainViolation`, not a
fallback. **Norm bound.** Row `i < n` of `ηW_τ` has absolute sum
`η Σ_k τ^k |w_{k,i}| ≤ η M ≤ 1` (since `2^e ≥ M`), so `‖ηW_τ‖_∞ ≤ 1` and
`‖B̃_η‖_∞ ≤ ‖τG‖_∞ + ‖ηW_τ‖_∞ + ‖J‖_∞ ≤ τ‖G‖_∞ + 2` for any number `q ≤ p` of
non-zero columns and every `τ ≥ 0`. (An unweighted `max_k ‖w_k‖_∞` would leave `‖ηW_τ‖_∞` up to
`Σ_k τ^k`, i.e. unbounded for `τ > 1` and above 1 for `q ≥ 2` non-zero columns.)
The bound needs `2^e ≥ M`, i.e. `M ≤ 2^{1000}`; above the clamp, and in a float
type where `2^{±e}` is not finite (`η = 1` fallback), the bound — and hence (i)
below — is not guaranteed (accuracy is unaffected, only cost). `(s, m)` is selected
from `PHI_NORM_TIGHTEN·(τ‖G‖ + 2)` (§58.2 constant kept). `w₀` never enters the
matrix. `phi_action(k)` and `phi_action_batched` are the special case
`p = PHI_MAX`, one non-zero column `W = [v, 0, 0]` (weight 1, no `τ^k`), η from
`M = ‖v‖_∞`, initial vector `e_k/η` in slot `n+k−1` (`k ≥ 1`) or `[v; 0]` (`k = 0`);
the top block is then `φ_k(τG)v`. **Consequences:** (i) cost
`(s, m) = phi_cost_probe(‖G‖, τ)` is independent of all input vectors;
(ii) powers of two are exact, so `φ(2^j v) = 2^j φ(v)` **bit for bit** within the
clamp range. That is gate `G_PHI_COST_V_INVARIANT`. The pre-ADR-0202 bound
`τ‖A‖ + ‖v‖_∞ + 1` made the substep count grow linearly in `‖v‖_∞`: measured
1000× cost for a 10⁶× change of scale at `τ = 10⁻⁶`.

**One sweep.** `phi_combination` evaluates §62.4.a by ONE scaled-Horner pass over
the `(n+p)`-augmented vector. The cost equals one `phi_action`, not `p+1` of them.

### §62.5 — What the combination covers (NORMATIVE recipes, documented in rustdoc and the Python docstring)

With `G = −M⁻¹A`, `N(·)` evaluated by the caller (no callbacks, ADR-0189 D3), and
`C(h, [w₀..w_p]) := Σ h^k φ_k(hG) w_k`:

```text
affine (G1):   M u' = −A u + s        u(h) = C(h, [u₀, M⁻¹s])          (exact)
ETD1:          u⁺ = C(h, [u, N(u)])
ETD-RK2 (Cox–Matthews):  a = C(h,[u, N_u]);   u⁺ = C(h, [u, N_u, (N_a − N_u)/h])
               embedded error estimate (free):  e = u⁺ − a = h·φ₂(hG)(N_a − N_u)
ETDRK4 (Cox–Matthews 2002):
   a = C(h/2,[u, N_u]);  b = C(h/2,[u, N_a]);  c = C(h/2,[a, 2N_b − N_u])
   u⁺ = C(h, [u, N_u, (−3N_u + 2N_a + 2N_b − N_c)/h, 4(N_u − N_a − N_b + N_c)/h²])
```

**Stiffness guidance (NORMATIVE in docs).** Every linear part of `N` that the user
can express as a matrix belongs in the operator. Local symmetric parts go through
`with_diagonal`. Nonlocal or non-symmetric parts go through `GeneralOperator`,
which is φ-capable via `CsrGenerator::from_general`. Only the weak remainder
stays explicit. The ETD local error constant scales with `Lip(N)²` (order 2) and
`Lip(N)⁴` (order 4), so moving a near-cancelling linear term into `G` reduces the
error by that power of the Lipschitz ratio.

### §62.6 — Acceptance gates (NORMATIVE; full entries in properties.yaml)

| Gate | Claim | Threshold | Severity |
|---|---|---|---|
| `G_SPDR_TRIDIAG_DENSE` | LDLᵀ vs dense LU, λ ∈ {0 (c>0), 0.5, 1e3}, with/without mass, n=12 | rel ≤ 1e-12; residual ≤ 1e-14 | RELEASE_BLOCKING |
| `G_SPDR_PCG_DENSE` | PCG IC(0) and Jacobi, 2-D 4×3 + reaction, λ ∈ {0, 1} | rel ≤ 1e-10 (tol 1e-12) | RELEASE_BLOCKING |
| `G_SPDR_STEADY_MMS` | FV steady `−(ku')'+cu=s`, Neumann, manufactured | OLS slope ∈ [−2.2, −1.8] over n ∈ {65,129,257,513}; err(513) ≤ 1e-5 | RELEASE_BLOCKING |
| `G_SPDR_PHI1_LIMIT` | `R(0)s` vs `τφ₁(τG)M⁻¹s`, τ = 40/λ_min (§62.1.c) | rel ≤ 1e-10 | RELEASE_BLOCKING |
| `G_SPDR_IMPLICIT_CROSS` | `λ(λI+A)⁻¹v` vs `KrylovPath::ImplicitEuler{1}` at Δt = 1/λ | rel ≤ 1e-10 | RELEASE_BLOCKING |
| `G_SPDR_REJECT` | singular / negative λ / bad mass / forced-tridiagonal misuse → typed errors | exact variant match | RELEASE_BLOCKING |
| `G_SPDR_LINEAR_COST` | tridiagonal wall time `t(2²⁰)/t(2¹⁶)` | ≤ 32 | ADVISORY |
| `G_SYMOP_COMPOSE_EXACT` | `with_diagonal` = dense `A + diag(c)` entrywise; `csr()`/`from_csr` round trip | bitwise | RELEASE_BLOCKING |
| `G_PHI_COST_V_INVARIANT` | `(s,m)` independent of `‖v‖`; `φ(2⁴⁰v) = 2⁴⁰φ(v)` | bitwise | RELEASE_BLOCKING |
| `G_PHI_COMBINATION_DENSE` | §62.4.a vs DST-eigen oracle (the §58 6-node one) and vs separate calls | ≤ 1e-12 / ≤ 1e-13 | RELEASE_BLOCKING |
| `G_PHI_MASS_DENSE` | mass φ vs test-local dense Taylor scaling-and-squaring ref (`tests/phi_dense`); vs §55.3 congruence; `mass=None` ≡ `NegLaplacianGenerator` | ≤ 1e-12 / ≤ 1e-13 / bitwise | RELEASE_BLOCKING |
| `G_PHI_GENERAL_DENSE` | φ_k on upwind drift–diffusion `GeneralOperator` vs test-local dense Taylor ref; transpose vs dense | ≤ 1e-10 / ≤ 1e-14 | RELEASE_BLOCKING |
| `G_ETD_AFFINE_EXACT` | `C(h,[u₀,M⁻¹s])` vs `u* + e^{hG}(u₀−u*)`; ETDRK4 recipe vs `Etdrk4` driver | ≤ 1e-12 / ≤ 1e-11 | RELEASE_BLOCKING |

Plus the ADR-0200 CI digests: new nostd-check scenarios `spdr_tridiag`, `spdr_pcg_ic0`
and `phi_combination` must reproduce the committed FNV-1a table under default,
`std-ref`, `x86-64-v3`, NEON and QEMU.

### §62.7 — Honest limits

- PCG is `O(√κ·nnz)` per solve. There is no multigrid and no sparse direct
  Cholesky; 1-D tridiagonal is the only exact-direct path.
- φ cost stays `∝ τ‖G‖`. A rational (TWS contour, reusing the §47 contour with
  `s^{−k}` weights and §62.2.a solves on complex shifts) φ-action is DEFERRED
  (ADR-0202 honest limits).
- `λ < 0` and indefinite `S` are out of scope.

**Cross-references.** §22 (Laplace–Chernoff resolvent, `λ > 0`), §47 (contour
resolvent), §55 (operator, congruence), §56 (assembly), §58 (φ-functions, amended
by §62.4), §59 (PCG/IC(0) rule, reused). Citations: Pazy 1983; Higham 2002 §9.6;
Hestenes–Stiefel 1952; Meijerink–van der Vorst 1977, Math. Comp. 31:148;
Al-Mohy–Higham 2011; Cox–Matthews 2002, J. Comput. Phys. 176:430.
