# API spec — implicit / shift-invert stiff action for `SymmetricOperator` (Issue #16, ADR-0190, §59)

NORMATIVE surface the engineer implements against. Additive only; follows the
existing channel-major `py.detach` batching convention and the `SemiflowError`
kind taxonomy exactly. No existing signature changes.

## 1. Rust-core entry points

### 1.1 New `KrylovPath` variant (`crates/semiflow/src/graph_krylov.rs`)

```rust
pub enum KrylovPath {
    Chebyshev,                       // unchanged
    Lanczos { m_max: usize },        // unchanged
    /// Implicit backward-Euler shift-invert (§59). `n_steps` sub-steps of
    /// size Δt = τ/n_steps; each solves (I + Δt·Â)x = b by preconditioned CG.
    ImplicitEuler {
        n_steps: usize,
        /// Optional CG iteration cap per sub-step.
        /// `None` uses `ceil(√κ(S)·ln(2/tol))` (§59.4, fix #18); `Some(m)` overrides.
        cg_max_iter: Option<usize>,    // NEW (issue #18)
    },
}
```

Adding the variant makes every `match KrylovPath` exhaustive site fail to compile
until it gains an arm — these are the ONLY WILL-BREAK (d=1) sites and MUST be
updated (fail-loud, intended):

- `GraphKrylovChernoff::action(...)` — dispatch the new arm to `implicit_euler_action`.
- the `(s, m)` substep/degree selector — the `ImplicitEuler` arm returns
  `(n_steps, 0)` (no Chebyshev/Lanczos degree; the explicit substep math is NOT reused).
- PyO3 `krylov_path(path, m_max)` — map `"implicit"` (see §2).

### 1.2 New solver module (`crates/semiflow/src/pcg.rs`, ≤500 LoC; functions ≤50 LoC)

```rust
/// Preconditioner over a symmetric operator (§59.2). Built once per (Â, Δt).
pub(crate) trait Preconditioner<F: SemiflowFloat> {
    fn apply(&self, r: &[F], z: &mut [F]);   // z ← P⁻¹ r
}

/// Jacobi preconditioner P = diag(I + Δt·Â) (v1 default).
pub(crate) struct Jacobi<F> { inv_diag: Vec<F> }

/// Solve (I + dt·op)·x = b by preconditioned CG (§59.4). Reuses op.apply_into_slice.
/// Returns Ok(iters) or Err(ConvergenceFailed). `x` is warm-started by the caller.
pub(crate) fn pcg_shifted<F: SemiflowFloat>(
    op: &dyn SymmetricLinearOp<F>,
    dt: F,
    b: &[F],
    x: &mut [F],                 // in: initial guess (warm start); out: solution
    precond: &dyn Preconditioner<F>,
    tol_cg: F,
    max_iter: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<usize, SemiflowError>;
```

### 1.3 Implicit action (in `graph_krylov.rs`, called by `action`)

```rust
/// Backward-Euler action w ← (I + Δt·Â)^{−n_steps} · w0  (§59.1).
/// Builds the preconditioner ONCE (§59.2), then loops n_steps PCG solves.
/// Reused per channel by the existing graph_batched::evolve_batched driver.
fn implicit_euler_action<F: SemiflowFloat>(
    op: &dyn SymmetricLinearOp<F>,
    src: &[F], dst: &mut [F],
    tau: F, n_steps: usize, tol: F,
    max_iter_override: Option<usize>,  // NEW (issue #18): None → auto-computed formula
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>;
```

Contract: `Δt = tau / n_steps`;
`max_iter = min(N, max(16, ceil( sqrt(1 + Δt·op.lambda_max_bound()) · ln(2/tol_cg) )))` (§59.4, issue-#18 correction; old formula `ceil(4·√κ)` omitted `ln(2/tol_cg)` factor);
`tol_cg = max(tol, 1e-12)`; warm-start each sub-step with the previous `u_k`.
The lumped/consistent √μ / R pre-/post-scale is applied by the EXISTING §55 wrappers
(`mass_lumped_evolve` / `MassKOperator::evolve`) — `implicit_euler_action` operates on
the already-congruent symmetric `Â`.

## 2. PyO3 surface (`crates/semiflow-py/src/{symmetric_op_py,mass_op_py}.rs`)

Additive `path="implicit"` string on the EXISTING methods; no signature break. The
`n_steps` parameter is added with a default (explicit paths ignore it).

### 2.1 `SymmetricOperator.evolve_batched`

```python
# existing (unchanged) — explicit Krylov:
out = op.evolve_batched(t, V_nc, path="chebyshev", tol=1e-10, m_max=18)

# NEW — implicit stiff path (§59):
out = op.evolve_batched(t, V_nc, path="implicit", tol=1e-8, n_steps=100)

# NEW — override CG cap (issue #18 escape hatch):
out = op.evolve_batched(t, V_nc, path="implicit", tol=1e-8, n_steps=100, cg_max_iter=50)
```

Signature (NORMATIVE):

```
evolve_batched(t: float, v_nc: ndarray[f64], path: str = "chebyshev",
               tol: float = 1e-10, m_max: int = 18, n_steps: int = 100,
               cg_max_iter: int | None = None) -> ndarray[f64]
```

- `path="implicit"` routes to `KrylovPath::ImplicitEuler { n_steps, cg_max_iter }`; `m_max` is ignored.
- `n_steps` MUST be `≥ 1` (else `ValueError`/`SemiflowError(kind="OutOfDomain")`).
- `cg_max_iter=None` (default) uses `ceil(√κ·ln(2/tol))` (§59.4, fix #18);
  pass an explicit integer to override, e.g. when the Gershgorin `λ_max` bound is loose.
- Channel-major `py.detach` batching UNCHANGED (validate → detach → scatter, ADR-0031).

### 2.2 `mass_lumped_evolve` (free function)

```
mass_lumped_evolve(k_op, m_diag, t, v_nc, path="chebyshev",
                   tol=1e-10, m_max=18, n_steps=100) -> ndarray[f64]
```

`path="implicit"` applies the §55.3 √μ pre-/post-scale (UNCHANGED) around the implicit
action on `Â`. This is the Issue #16 primary path.

### 2.3 `MassKOperator.evolve` (consistent mass)

Same additive `path="implicit", n_steps=...` extension; reuses the §55.2 R-solve matvec
chain. Scoped as follow-on (design-complete; the solver is identical).

### 2.4 `krylov_path` dispatcher

```rust
fn krylov_path(path: &str, m_max: u32, n_steps: usize) -> PyResult<KrylovPath> {
    match path {
        "chebyshev" => Ok(KrylovPath::Chebyshev),
        "lanczos"   => Ok(KrylovPath::Lanczos { m_max: m_max as usize }),
        "implicit"  => Ok(KrylovPath::ImplicitEuler { n_steps }),   // NEW
        other => Err(/* Unsupported: path must be chebyshev|lanczos|implicit */),
    }
}
```

## 3. Error contract (reuse — NO new kind)

| Condition | Core `SemiflowError` | PyO3 `kind` |
|-----------|----------------------|-------------|
| CG exceeds `max_iter` without reaching `tol_cg` | `ConvergenceFailed { .. }` | `ConvergenceFailed` |
| `n_steps < 1`, or non-finite `t`/`tol` | `DomainViolation { .. }` | `OutOfDomain` (or `NanInf` if value non-finite) |
| Indefinite `Â` → CG breakdown (division by ≤0 curvature) | `ConvergenceFailed { .. }` | `ConvergenceFailed` |
| `path` string not in {chebyshev,lanczos,implicit} | `Unsupported { .. }` | `Unsupported` |
| IC(0) non-positive pivot | *no error* — silent fallback to Jacobi (§59.2) | — |

Errors-as-values: on failure NO partial output is written; the `Result` propagates
through `py.detach` and is mapped by the existing `error.rs::from_core`.

## 4. Defaults and rationale

- `n_steps = 100` default: O(Δt) backward-Euler needs enough steps for low-mode
  accuracy; 100 clears the `G_SYMOP_IMPLICIT_DENSE` 1e-6 gate on moderate operators
  and the stiff repro. Caller tunes up for tighter tolerance (or awaits SDIRK2, §59.5).
- `tol = 1e-8` recommended for the implicit path (looser than the explicit `1e-10`
  because backward-Euler truncation O(Δt) dominates the CG residual floor).
- Preconditioner: Jacobi (v1). IC(0) is a drop-in `Preconditioner` impl (§59.2),
  selectable later WITHOUT any API change.

## 5. Cross-references

- Math: `contracts/semiflow-core.math.md` §59 (NORMATIVE algorithm, gates §59.7).
- Decision/governance: `docs/adr/0191-implicit-stiff-symmetric-operator.md`.
- Reused: §55 (`SymmetricOperator`, `SymmetricLinearOp`, congruence), ADR-0031
  (3-phase GIL batching), `crates/semiflow-py/src/error.rs` (kind mapping).
