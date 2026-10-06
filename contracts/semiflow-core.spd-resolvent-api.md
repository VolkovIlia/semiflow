# API spec — SPD resolvent / steady solve, operator composition, φ-combination (ADR-0202, §62)

Math authority: §62 (all `§62.x` citations below) lives in
`contracts/semiflow-core.math-spd-resolvent.md` (split out of `contracts/semiflow-core.math.md` for size).

NORMATIVE surface the engineer implements against. **Additive only**: no existing
public signature changes. The one behavioural change is the `(s, m)` selection
inside `phi_action` and `phi_action_batched` (§62.4). Errors are values
(`SemiflowError`), and nothing in the library panics. `no_std + alloc`, 0 new
dependencies, functions ≤ 50 lines, files ≤ 500 LoC, `unsafe` forbidden in core
(`unsafe_code = "deny"`).

Sign convention (unchanged from §55/§58): `SymmetricOperator` and `GeneralOperator`
hold a PSD/accretive `A` whose evolution is `e^{−tA}`. The generator is
`G = −M⁻¹A`, where `M = diag(mass)` and `M = I` by default.

## 1. Rust core

### 1.1 `crates/semiflow/src/spd_resolvent.rs` (NEW, ≤ 500 LoC)

```rust
/// Which algorithm `SpdResolvent::new` uses (§62.2).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SpdSolver {
    /// Tridiagonal LDLᵀ if the CSR pattern is tridiagonal, else PCG with IC(0).
    #[default]
    Auto,
    /// Force the tridiagonal LDLᵀ path; `Unsupported` if the pattern is not tridiagonal.
    Tridiagonal,
    /// Force preconditioned CG.
    Pcg {
        precond: Precond,
        /// CG iteration cap. `None` → `2·n + 16` (§62.2.c): at λ = 0 there is no cheap
        /// lower bound on λ_min(A), so the §59.4 √κ budget is not computable; CG
        /// terminates in ≤ n steps in exact arithmetic, the factor 2 absorbs loss of
        /// orthogonality. Convergence normally stops far earlier (tol test).
        max_iter: Option<usize>,
    },
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Precond {
    /// IC(0), zero fill-in. On a non-positive pivot it falls back to Jacobi (§59.2).
    /// The fallback shows up in `SpdResolvent::method()`.
    #[default]
    Ic0,
    Jacobi,
}

/// Method actually in use (after `Auto` dispatch and any IC(0)→Jacobi fallback).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResolventMethod { Tridiagonal, PcgIc0, PcgJacobi }

/// Per-solve diagnostics. Direct path: `iterations = 0`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolveReport {
    pub iterations: usize,
    /// ‖b − (λM+A)x‖₂ / ‖b‖₂ (0 when b = 0). TRUE residual, recomputed after the solve
    /// on both paths (on the prescaled system, §62.2.c, so the value does not depend
    /// on ‖b‖). On the PCG path it may exceed `tol` near the ε·κ(S) attainable-accuracy
    /// floor, which is not an error. It is never above max(1e3·tol, 1e-3) in an `Ok`
    /// (§62.2.b gross-residual guard).
    pub rel_residual: f64,
}

/// Factor-once / solve-many resolvent of an SPD sparse operator:
/// `x = (λ·M + A)⁻¹ b`, `λ ≥ 0`, `M = diag(mass) > 0` (default I).
#[derive(Clone)]
pub struct SpdResolvent<F: SemiflowFloat = f64> { /* private: op Arc, λ, mass, factor/precond, tol */ }

impl<F: SemiflowFloat> SpdResolvent<F> {
    /// # Errors
    /// `DomainViolation`: n = 0; λ < 0 or non-finite; `mass.len() != n`; mass ≤ 0 or
    ///   non-finite; `tol ∉ (0, 1)`; λ = 0 and every row satisfies
    ///   |Σⱼ aᵢⱼ| ≤ 8ε·Σⱼ|aᵢⱼ| ("numerically singular: all row sums <= 8*eps*row
    ///   magnitude (constant vector in the null space)"; §62.1 (i));
    ///   tridiagonal LDLᵀ pivot ≤ n·ε·max|aᵢᵢ + λmᵢ| (not positive definite).
    /// `Unsupported { feature: "spd_resolvent: tridiagonal solver on non-tridiagonal operator" }`.
    pub fn new(
        op: &SymmetricOperator<F>,
        lambda: F,
        mass: Option<&[F]>,
        solver: SpdSolver,
        tol: F,              // PCG relative residual target; ignored by the direct path
    ) -> Result<Self, SemiflowError>;

    pub fn n(&self) -> usize;
    pub fn method(&self) -> ResolventMethod;

    /// `x ← (λM+A)⁻¹ b`. `x` is overwritten; it is not used as a warm start.
    /// `b` is prescaled by an exact power of two (‖b‖∞ ∈ [1,2)), so the result is
    /// scale-equivariant bitwise (§62.2.c). On the PCG path `Ok` means the recursive
    /// residual met `tol`; read `SolveReport::rel_residual` for the true one (§62.2.b).
    /// # Errors
    /// `DomainViolation` (`b.len()`/`x.len() != n`, non-finite `b`);
    /// `ConvergenceFailed { last_residual, max_iter }`, PCG only, when either
    /// (a) the iteration cap is reached, or (b) the recomputed true residual is NaN
    /// or > max(1e3·tol, 1e-3) (gross-residual guard: catches CG stagnation on a
    /// singular system). For both triggers, `max_iter` is the configured cap (override
    /// or 2n+16), not the iteration count. `last_residual` is the relative TRUE
    /// residual ‖b−Sx‖₂/‖b‖₂ of the returned iterate (NaN if non-finite). The direct
    /// path never returns it.
    /// Near-singular band (§62.2.b): for κ(S) ≳ 1e12–1e13 (f64; ≳ 1e4 in f32), but
    /// not caught by the null-space check, PCG may return ConvergenceFailed where
    /// LDLᵀ returns Ok with a comparable residual. This is documented and accepted.
    /// Use the direct path or regularise.
    pub fn solve_into(
        &self, b: &[F], x: &mut [F], scratch: &mut ScratchPool<F>,
    ) -> Result<SolveReport, SemiflowError>;
}
```

Convenience on the operator (in `symmetric_operator.rs`, a one-line delegation):

```rust
impl<F: SemiflowFloat> SymmetricOperator<F> {
    pub fn resolvent(&self, lambda: F, mass: Option<&[F]>, solver: SpdSolver, tol: F)
        -> Result<SpdResolvent<F>, SemiflowError>;
}
```

Complexity: tridiagonal: factor `O(n)` (2n memory: pivots `d`, multipliers `ℓ`),
solve `O(n)`, no allocation per solve. PCG: setup `O(nnz)` (IC(0)) or `O(n)` (Jacobi),
per solve `O(iters·nnz)`, 4 scratch vectors from `ScratchPool`. Summation order is
fixed (row-sequential, no threads), which makes std and no_std results bit-identical.

### 1.2 `pcg.rs` (AMEND, stays ≤ 500 LoC)

- Add a second shifted operator `S = λ·diag(m) + A` (Jacobi diag `λmᵢ + aᵢᵢ`)
  beside the existing `I + Δt·Â`. `ImplicitEuler` keeps its code path untouched,
  so its numerics stay **bit-identical**. Rewriting it as `λ = 1/Δt` would change
  rounding and is forbidden. Both share `cg_loop` through a small private
  "shifted matvec" abstraction (a trait or enum, the engineer's choice) and must
  not duplicate the loop.
- Add `Ic0<F>` implementing `Preconditioner<F>`: lower factor `L̃` on the exact lower
  pattern of the shifted matrix. `build` returns `Option<Self>` (`None` on pivot
  `≤ 0`; the caller falls back to Jacobi). `apply` = forward plus backward sweep.
- `Preconditioner`, `Jacobi` and `Ic0` stay `pub(crate)`. Users reach them only
  through `SpdResolvent`.

### 1.3 `symmetric_operator.rs` (AMEND)

```rust
impl<F: SemiflowFloat> SymmetricOperator<F> {
    /// A + diag(c). Inserts structurally missing diagonal entries; columns stay sorted.
    /// # Errors  `DomainViolation`: c.len() != n, any cᵢ < 0 or non-finite, or an
    ///   assembled entry A+diag(c) overflows (non-finite).
    pub fn with_diagonal(&self, c: &[F]) -> Result<Self, SemiflowError>;

    /// Borrowed CSR view (row_ptr, col_idx, vals). Round-trip:
    /// `from_csr(n, csr().0, csr().1, csr().2, 0)` reproduces identical arrays.
    #[must_use]
    pub fn csr(&self) -> (&[usize], &[u32], &[F]);
}
```

### 1.4 `generator_action.rs` (AMEND): `CsrGenerator<F>`

```rust
/// G = −M⁻¹A over a CSR operator (§62.3). Norm bound computed once (see below).
/// Exact transpose: Gᵀ = −AᵀM⁻¹.
pub struct CsrGenerator<F: SemiflowFloat = f64> { /* Arc CSR (+ transpose for general), inv_mass */ }

impl<F: SemiflowFloat> CsrGenerator<F> {
    /// # Errors `DomainViolation`: mass length/positivity/finiteness; with a mass, any
    ///   quotient |aᵢⱼ|/mᵢ or the norm bound maxᵢ Σⱼ |aᵢⱼ|/mᵢ is non-finite (both
    ///   constructors).
    pub fn from_symmetric(op: &SymmetricOperator<F>, mass: Option<&[F]>) -> Result<Self, SemiflowError>;
    pub fn from_general(op: &GeneralOperator<F>, mass: Option<&[F]>) -> Result<Self, SemiflowError>;
}
impl<F: SemiflowFloat> GeneratorAction<F> for CsrGenerator<F> { /* dim, apply_generator,
    norm_bound, apply_generator_transpose (real transpose) */ }
```

`NegLaplacianGenerator` is unchanged. For `mass = None` on a symmetric operator,
`CsrGenerator` uses `norm_bound = op.lambda_max_bound()` and the same CSR matvec,
so its results are **bit-identical** to `NegLaplacianGenerator` (checked in
`G_PHI_MASS_DENSE`). With `mass = Some(m)` it stores the row-normalised
`Q = M⁻¹A` once (each entry `aᵢⱼ/mᵢ`), applies `G v = −Q v`, and uses the row-wise bound
`maxᵢ Σⱼ(|aᵢⱼ|/mᵢ)` (divide-then-sum; non-finite → `DomainViolation`). For `from_general`, `mass = None` uses
`op.norm_inf_bound()`.

### 1.5 `phi_action.rs` (AMEND) + `phi_combination` (NEW; split into `phi_combination.rs` if > 500 LoC)

```rust
/// out ← Σ_{k=0}^{p} τ^k φ_k(τG) w_k,  p = w.len() − 1 ≤ PHI_MAX (= 3)   (§62.4)
/// One augmented Horner sweep (Al-Mohy–Higham 2011 Thm 2.1, η-scaled).
/// Augmented matrix B̃_η = [[τG, ηW_τ],[0, J]], column j of W_τ = τ^{p−j}·w_{p−j},
/// J unscaled; η = 2^{−⌈log₂ Σ_k τ^k‖w_k‖∞⌉} (§62.4), initial slot n+p−1 = 1/η.
/// # Errors `DomainViolation`: w empty or w.len() > PHI_MAX+1; any w_k.len() != n
///   or non-finite entry; τ < 0 or non-finite; out.len() != n; τ^k‖w_k‖∞ overflow.
pub fn phi_combination<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op, tau: F, w: &[&[F]], out: &mut [F], scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>;

/// (s, m) used by every φ entry point for ‖G‖-bound `norm_g` and step τ.
/// Independent of the input vectors by construction (§62.4, G_PHI_COST_V_INVARIANT).
#[must_use]
pub fn phi_cost_probe(norm_g: f64, tau: f64) -> (u32, u32);
```

`phi_action` and `phi_action_batched` keep their signatures. Their internal norm
becomes `τ‖G‖ + 2`, and the η power-of-two scaling is applied (§62.4).

### 1.6 Re-exports (`lib.rs`)

`SpdResolvent, SpdSolver, Precond, ResolventMethod, SolveReport, CsrGenerator,
phi_combination, phi_cost_probe`. Add each to `test_public_exports.py`'s Rust-side
counterpart, if one exists.

## 2. Python (`crates/semiflow-py`)

New file `spd_resolvent_py.rs` (≤ 500 LoC). Amend `symmetric_op_py.rs` and
`phi_etdrk4_py.rs`. GIL: ADR-0031 three-phase (validate → `py.detach` → scatter).
Errors map through the existing `from_core` (`OutOfDomain`, `Unsupported`,
`ConvergenceFailed` → `SemiflowError(.kind)`).

```python
# --- operator composition (SymmetricOperator methods) ---
op2 = op.with_diagonal(c)                 # c: float64 (n,), c >= 0
indptr, indices, data = op.to_csr()       # int64, int32, float64  (from_csr round-trips)
op_hat = op.lumped_congruence(masses)     # exposes existing Rust

# --- resolvent / steady solve ---
R = op.resolvent(lam=0.0, mass=None, solver="auto", precond="ic0",
                 tol=1e-12, max_iter=None)        # -> SpdResolvent
x  = R.solve(b)                           # (n,) -> (n,)
X  = R.solve_batched(B)                   # (n, nc) -> (n, nc)  (channel layout as evolve_batched)
x, iters, rel_res = R.solve_info(b)
R.method   # "tridiagonal" | "pcg-ic0" | "pcg-jacobi"
R.n

# --- φ-functions (op: SymmetricOperator | GeneralOperator) ---
y   = semiflow.phi_combination(op, tau, w, mass=None)  # w: (p+1, n), rows w_0..w_p, p <= 3
                                                       # y = Σ τ^k φ_k(−τ M⁻¹A) w_k
y1  = semiflow.phi_action(op, k, tau, v, mass=None)          # mass kwarg + GeneralOperator: NEW
Y   = semiflow.phi_action_batched(op, p, tau, v, mass=None)  # idem
```

String arguments are validated against fixed menus. An unknown value raises
`SemiflowError(kind="OutOfDomain")`. `solver ∈ {"auto","tridiagonal","pcg"}`,
`precond ∈ {"ic0","jacobi"}`. Under `solver="auto"` the binding first tries the
core `Tridiagonal` path. If the core returns `Unsupported` (the pattern is not
tridiagonal), it rebuilds with `Pcg{precond, max_iter}`, so `precond` and
`max_iter` are honoured under `"auto"` too. They are ignored only when the
tridiagonal path is taken. The core `SpdSolver::Auto` is unchanged
(ADR-0202 Amendment 3). Inputs must be contiguous float64, and lengths are
checked before `py.detach`. Add the type stubs to the package `.pyi`.

**FFI / WASM**: DEFERRED (ADR-0202 D5; ADR-0186/0195 asymmetry: neither binding has
`SymmetricOperator` or `GeneralOperator`).

## 3. Errors (errors.yaml method entries)

| Method | Emits |
|---|---|
| `SpdResolvent::new` | `DomainViolation`, `Unsupported` |
| `SpdResolvent::solve_into` | `DomainViolation`, `ConvergenceFailed` |
| `SymmetricOperator::with_diagonal` | `DomainViolation` |
| `CsrGenerator::from_symmetric` / `from_general` | `DomainViolation` |
| `phi_combination` | `DomainViolation` |

No new variants. `ConvergenceFailed` gets its first production emitter here, from
two conditions: the cap, and the gross-residual guard of §62.2.b (Amendment 2).
Its rustdoc line "Reserved for v0.3+ resolvent; never returned in v0.1.0" must be
updated, because `pcg.rs` already returns it.

## 4. File / size budget

| File | Change | Budget |
|---|---|---|
| `spd_resolvent.rs` | NEW | ≤ 500 |
| `pcg.rs` (359) | + IC(0), generalised shift | ≤ 500 |
| `symmetric_operator.rs` (286) | + `with_diagonal`, `csr`, `resolvent` | ≤ 500 |
| `generator_action.rs` | + `CsrGenerator` | ≤ 500 |
| `phi_action.rs` (277) | η-scaling, probe; `phi_combination` (or new file) | ≤ 500 |
| `semiflow-py/src/spd_resolvent_py.rs` | NEW | ≤ 500 |
| `crates/semiflow-nostd-check/src/resolvent.rs` | NEW digest scenarios | ≤ 500 |
