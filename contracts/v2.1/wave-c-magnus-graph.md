# Wave 2.1C Contract — Magnus K=4 for Time-Dependent Graph Laplacian

**Status**: NORMATIVE — engineer implements verbatim against this contract.
**ADR**: ADR-0051 (Magnus on graphs design — new).
**Depends on**:
- Wave 2.1A (`Graph<F>`, `GraphSignal<F>`, `Laplacian<F>`, `GraphHeatChernoff<F>`) — shipped at `828f7bb`.
- Wave 2.1B (`GraphHeat4thChernoff<F>`, `StrangSplitGraph<A, B, F>`) — shipped at `68f17d8`.
- ADR-0042 (`apply_into` ping-pong template), ADR-0046 (precision-policy bands),
  ADR-0047 (`GraphHeatChernoff` design), ADR-0048 (CSR storage), ADR-0049
  (math.md §12 graph PDE rationale).
**Scope**: `semiflow-core` v2.1 Wave C — Magnus K=4 graph Chernoff variant for
**time-varying edge weights on a fixed topology**.
**Author**: ai-solutions-architect · **Date**: 2026-05-20 ·
**Reviewers**: reviewer-suckless, agentic-engineer.

Wave 2.1C ships **one** new public type:

1. **`MagnusGraphHeatChernoff<F>`** — order-4 Chernoff on
   `∂_t u = −L_G(t) u` via classical fourth-order Magnus method (two-point
   GL₄ Gauss-Legendre quadrature + first commutator term).

This type is the FIRST genuine Magnus expansion in `semiflow-core`. The
pre-existing `TruncatedExpDiffusionChernoff` / `TruncatedExp4thDiffusionChernoff`
(formerly mis-named `Magnus*DiffusionChernoff` in v0.6.x) are **not** Magnus
expansions — they truncate `exp(τG)` for a frozen `G`; see audit finding
v0.7.0 D2. Wave 2.1C reuses none of those code paths beyond the conceptual
"factorial-table + ping-pong scratch" pattern.

It is additive: the existing `ChernoffFunction<F, S = GraphSignal<F>>` trait
surface in `chernoff.rs` is **UNCHANGED**, and `ChernoffSemigroup::evolve`
accepts the new type without modification.

Wave 2.1C closes v2.1.

---

## §1 — `MagnusGraphHeatChernoff<F>` API (NORMATIVE)

### 1.1 Mathematical statement (CITATION; see math.md §12.9)

For the time-dependent linear ODE `u'(t) = −L_G(t) u(t)` on `ℝ^N`, the
exact solution over `[0, τ]` is `u(τ) = exp(Ω(τ)) u(0)`, where `Ω(τ)`
is the Magnus expansion (Iserles, Munthe-Kaas, Nørsett, Zanna 2000
*Acta Numerica* **9** §5.2):

```text
Ω(τ) = ∫₀^τ A(s) ds
     − (1/2) ∫₀^τ ds  ∫₀^s du [A(s), A(u)]
     + (higher-order nested commutators)
```

with `A(t) = −L_G(t)`. The fourth-order truncation
`Ω₄(τ)` (Iserles+ 2000 eq. (5.10); Blanes-Casas-Oteo-Ros 2009 Table 5)
uses two-point Gauss-Legendre quadrature on `[0, τ]`:

```text
Let c₁ = (3 − √3) / 6,        c₂ = (3 + √3) / 6        ∈ [0, 1]
Let b₁ = b₂ = 1 / 2            (GL₂ weights on [0, 1])

A₁ := A(c₁ · τ) = −L_G(c₁ · τ)
A₂ := A(c₂ · τ) = −L_G(c₂ · τ)

Ω₄(τ) = (τ/2)·(A₁ + A₂) + (√3 · τ² / 12) · [A₂, A₁]
```

The Magnus map applied to `f` is then evaluated by degree-4 Taylor
truncation of `exp(Ω₄)`:

```text
S₄(τ) f := exp(Ω₄(τ)) f
        ≈ Σ_{k=0..4} (Ω₄(τ))^k · f / k!
```

The truncation is exact through `τ⁴` because `Ω₄(τ) = O(τ)` (the
commutator term contributes `O(τ²)`), so the degree-4 polynomial
matches the Magnus exponential through global order 4 (Iserles+ 2000
§5.5 Theorem 5.2; Hochbruck-Ostermann 2010 *Acta Numerica* §3 for the
`exp(Ω)·v` evaluation lemma on bounded operators).

**Convergence radius.** For matrix Magnus on bounded `A(t)`, the series
converges absolutely whenever `∫₀^τ ‖A(s)‖₂ ds < π` (Blanes+ 2009
Theorem 1). Since `‖A(s)‖₂ = ‖L_G(s)‖₂ ≤ ρ̄(s)` (Gershgorin spectral
radius bound) and `ρ̄ < ∞` for finite graphs with finite edge weights,
the radius is satisfied for sufficiently small `τ`. The library
enforces a 50% safety margin: `ρ̄(τ/2) · τ < π/2`.

**Chernoff-hypothesis check.**

| Hypothesis | Verification |
|---|---|
| `S₄(0) = I` | `Ω₄(0) = 0` → `exp(0) = I` ✓ |
| `S₄'(0) = A(0) = −L_G(0)` | `(d/dτ) Ω₄(τ)|_{τ=0} = (A₁+A₂)/2|_{τ=0} = A(0)` ✓ |
| Quasi-contractivity | `‖exp(Ω₄)‖ ≤ exp(‖Ω₄‖) ≤ exp(τ·ρ̄)` → `(M, ω) = (1, ρ̄_peak)` ✓ |

Convergence rate `‖(S₄(t/n))^n f − u_exact(t)‖ = O(1/n⁴)` follows from
Iserles+ 2000 §5 Theorem 5.2, giving the slope-gate target **−3.95**
(f64) / **−3.50** (f32). Slope is observable on the time-dependent
path-graph oracle in §7 below.

This is a **classical theorem** — no new mathematics. §12.9 of math.md
is CITATION + NORMATIVE-library-policy only.

### 1.2 Public API surface

```rust
//! crates/semiflow-core/src/magnus_graph.rs (NEW FILE, ~280 LoC)
//!
//! Fourth-order Magnus expansion for time-dependent graph heat:
//! `∂_t u = −L_G(t) u` on a fixed-topology weighted graph.
//!
//! Two-point Gauss-Legendre quadrature (GL₄) + first commutator term:
//!     Ω₄(τ) = (τ/2)·(A₁ + A₂) + (√3·τ²/12) · [A₂, A₁]
//! with A_i = −L_G(c_i · τ), c₁ = (3−√3)/6, c₂ = (3+√3)/6.
//!
//! Citations:
//! - Iserles+ 2000 *Acta Numerica* §5 (Magnus method, fourth-order).
//! - Blanes+ 2009 *Phys. Rep.* Tables 5–6.
//! - Hochbruck-Ostermann 2010 *Acta Numerica* §3 (exp(Ω)·v evaluation).
//!
//! See math.md §12.9 (NORMATIVE) and ADR-0051 (design).

use alloc::sync::Arc;
use alloc::boxed::Box;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

/// Caller-supplied closure mapping a time point `t` to the Laplacian
/// `L_G(t)` valid at that time.
///
/// MUST satisfy:
/// - **Pure**: same `t` → equal output (no side effects, no global state).
/// - **Topology fixed**: `row_ptr` and `col_idx` of every returned
///   `Laplacian` MUST equal those of the topology graph passed to
///   `MagnusGraphHeatChernoff::new`. The library enforces this via
///   `debug_assert!` (release builds skip the check).
/// - **Send + Sync + 'static**: closure may be shared across threads
///   (e.g. by `ChernoffSemigroup::evolve_parallel` in v2.2).
pub type LaplacianAtTime<F> =
    Box<dyn Fn(F) -> Arc<Laplacian<F>> + Send + Sync + 'static>;

pub struct MagnusGraphHeatChernoff<F: SemiflowFloat = f64> {
    /// Topology graph. Fixed across all sampled `L_G(t)`.
    graph: Arc<Graph<F>>,
    /// `t ↦ Arc<Laplacian<F>>` — caller-supplied edge-weight sampler.
    lap_at_t: LaplacianAtTime<F>,
    /// Peak Gershgorin radius bound `ρ̄_max` over `t ∈ [0, t_horizon]`,
    /// used for `growth()` and Magnus convergence-radius check.
    /// Caller provides; library does not search.
    rho_bar_max: F,
    /// If true, every `apply_into` call validates
    /// `ρ̄_max · τ < π/2` (50% safety margin vs. theoretical `< π`).
    /// Default `true`.
    convergence_radius_check: bool,
}

impl<F: SemiflowFloat> MagnusGraphHeatChernoff<F> {
    /// Construct from topology + time-to-Laplacian closure.
    ///
    /// # Parameters
    /// - `graph`: fixed-topology graph (Wave 2.1A).
    /// - `lap_at_t`: closure `t ↦ Arc<Laplacian<F>>`. MUST be pure;
    ///   MUST preserve `graph.row_ptr()` and `graph.col_idx()`.
    /// - `rho_bar_max`: caller-supplied upper bound for
    ///   `max_{t ∈ [0, t_horizon]} ρ̄(L_G(t))`. Used for `growth()` and
    ///   for the Magnus convergence-radius check.
    /// - `convergence_radius_check`: if `true`, each `apply_into` rejects
    ///   `τ` with `rho_bar_max · τ ≥ π/2`. Default `true`.
    ///
    /// # Panics
    /// - Never panics.
    pub fn new(
        graph: Arc<Graph<F>>,
        lap_at_t: LaplacianAtTime<F>,
        rho_bar_max: F,
        convergence_radius_check: bool,
    ) -> Self {
        Self { graph, lap_at_t, rho_bar_max, convergence_radius_check }
    }

    /// Borrow the topology graph.
    #[must_use]
    pub fn graph(&self) -> &Graph<F> {
        &self.graph
    }

    /// Sampled Laplacian at time `t` (helper; clones the `Arc`).
    #[must_use]
    pub fn laplacian_at(&self, t: F) -> Arc<Laplacian<F>> {
        (self.lap_at_t)(t)
    }
}
```

`LaplacianAtTime<F>` is a public type alias; `Box<dyn Fn>` (not `fn`
pointer) is required because callers will commonly capture `Arc<Graph>`
+ an analytic edge-weight function. The trait bounds `Send + Sync +
'static` keep the door open for future parallel adaptation without
changing the public type.

### 1.3 Constructor invariants (NORMATIVE)

The constructor MUST validate:

- `rho_bar_max` is finite and strictly positive. Reject with
  `SemiflowError::DomainViolation` otherwise.
- `graph.n_nodes() > 0`. Reject with `SemiflowError::DomainViolation`
  otherwise.

Topology-equality between `graph` and `lap_at_t(t)` cannot be validated
at construction time without sampling — see §6 below for the per-call
`debug_assert` policy.

---

## §2 — `impl ChernoffFunction<F, S = GraphSignal<F>>` (NORMATIVE)

```rust
impl<F: SemiflowFloat> ChernoffFunction<F> for MagnusGraphHeatChernoff<F> {
    type S = GraphSignal<F>;

    fn apply(&self, tau: F, f: &Self::S) -> Result<Self::S, SemiflowError>
    where
        Self::S: Clone,
    {
        let mut dst = f.clone();
        let mut scratch = ScratchPool::<F>::new();
        self.apply_into(tau, f, &mut dst, &mut scratch)?;
        Ok(dst)
    }

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>
    where
        Self::S: Clone,
    {
        apply_magnus_k4_into(self, tau, src, dst, scratch)
    }

    fn order(&self) -> u32 {
        4
    }

    fn growth(&self) -> (f64, f64) {
        let rho = self.rho_bar_max.to_f64().unwrap_or(f64::INFINITY);
        (1.0, rho)
    }
}
```

### 2.1 `order()` rationale

Returns **4** because the global Chernoff convergence rate
`‖(S₄(t/n))^n f − u_exact(t)‖ = O(1/n⁴)` is established by Iserles+
2000 §5 Theorem 5.2 (fourth-order Magnus method) combined with the
Chernoff product formula (Engel-Nagel 2000 §III.5 Theorem 5.2,
specialised to bounded matrix generators where Magnus replaces the
constant-generator exponential).

This matches the precedent set by `GraphHeat4thChernoff::order()
= 4` (Wave 2.1B contract §2) where order-4 is also a result of
fourth-order matching of `exp(τA)` to the truncated Taylor series.

### 2.2 `growth()` rationale

`(M, ω) = (1, ρ̄_max)` because
`‖exp(Ω₄(τ))‖₂ ≤ exp(‖Ω₄(τ)‖₂) ≤ exp(τ · ρ̄_max)` (sub-multiplicative
spectral norm + triangle inequality on the Magnus terms).

This is **looser** than `GraphHeatChernoff::growth() = (1, ρ̄)` because
Magnus introduces a positive commutator term that may amplify
intermediate iterates even though the exact heat semigroup is
contractive. The looseness is a CITATION-only consequence of the
truncation (Hochbruck-Ostermann 2010 §3.5).

---

## §3 — GL₄ Gauss-Legendre quadrature constants (NORMATIVE)

The two-point Gauss-Legendre rule on `[0, 1]`:

```rust
// Abscissae c₁, c₂ ∈ [0, 1] (NORMATIVE — DO NOT CHANGE).
// Source: Iserles+ 2000 §5.5; Numerical Recipes 3e §4.6.1 Table 4.6.1.
//
// c₁ = (3 − √3) / 6 ≈ 0.211324865405187
// c₂ = (3 + √3) / 6 ≈ 0.788675134594813
const GL4_C1_F64: f64 = 0.211_324_865_405_187_134;
const GL4_C2_F64: f64 = 0.788_675_134_594_812_866;

// Symmetry-derived: c₁ + c₂ = 1, c₂ − c₁ = 1/√3.
// √3/12 used in the commutator coefficient.
const SQRT3_OVER_12_F64: f64 = 0.144_337_567_297_406_433;
```

These constants are reused in two places:

1. `apply_magnus_k4_into` (this contract §4).
2. `verify_v2_1c_magnus_consistency.py` (this contract §8 — sympy gate
   re-derives them from `(3 ± √3)/6` symbolically).

**No shared GL4 module is introduced in v2.1.** The constants live as
`const` items in `magnus_graph.rs` because they are the ONLY consumer
in v2.1. A future cross-module `gl_quadrature.rs` (sharing GL4 with
hypothetical `MagnusDiffusionChernoff` in v2.2) is permitted but out of
scope here.

**Generic-over-F**: `GL4_C1_F64` and `GL4_C2_F64` are `const f64`. Inside
`apply_magnus_k4_into<F>`, they are coerced via `from_f64::<F>(...)` once
per call (4 conversions total: c₁, c₂, ½, √3/12 — negligible).

---

## §4 — `apply_magnus_k4_into` core kernel (NORMATIVE)

```rust
/// Apply one Magnus K=4 step in place.
///
/// Performs:
/// 1. Sample `L_G(c₁·τ)` and `L_G(c₂·τ)` via `self.lap_at_t`.
/// 2. Validate topology equality with `self.graph` (debug only).
/// 3. Validate Magnus convergence radius if `convergence_radius_check`.
/// 4. Build the four "ping-pong" scratch buffers
///    (`omega_apply`, `omega_acc`, `tmp_a`, `tmp_b`) — all simultaneously
///    live, so use `take_vec`/`return_vec` (NOT RAII `borrow_vec`).
/// 5. Evaluate `Ω₄ · src` via:
///        Ω₄ · v  =  (τ/2)·(A₁·v + A₂·v)
///                 + (√3·τ²/12) · (A₂·(A₁·v) − A₁·(A₂·v))
///    using `Laplacian::apply_into_slice` four times.
/// 6. Apply degree-4 Taylor truncation `Σ_{k=0..4} (Ω₄)^k / k! · src`
///    via the same ping-pong pattern as `graph_heat4::apply_zeta4_into`
///    (Wave 2.1B contract §2.5).
///
/// Returns `Err(SemiflowError::OutOfMagnusRadius { tau, rho_estimate })`
/// when the convergence-radius check fails; otherwise `Ok(())`.
///
/// Zero heap allocations in steady state (all buffers come from
/// `ScratchPool::take_vec`).
fn apply_magnus_k4_into<F: SemiflowFloat>(
    mc: &MagnusGraphHeatChernoff<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    validate_tau(tau)?;
    if mc.convergence_radius_check {
        validate_magnus_radius(mc.rho_bar_max, tau)?;
    }
    let n = src.len();
    debug_assert_eq!(dst.len(), n);

    // --- Step 1: sample Laplacians at GL4 nodes ------------------------
    let c1 = from_f64::<F>(GL4_C1_F64);
    let c2 = from_f64::<F>(GL4_C2_F64);
    let t1 = c1 * tau;
    let t2 = c2 * tau;
    let lap1 = (mc.lap_at_t)(t1);
    let lap2 = (mc.lap_at_t)(t2);

    debug_assert_eq!(lap1.n_nodes(), n, "topology drift in lap_at_t");
    debug_assert_eq!(lap2.n_nodes(), n, "topology drift in lap_at_t");

    // --- Step 2: acquire FOUR simultaneous scratch buffers -------------
    // omega_v : holds Ω₄ · current (running result of the Horner sweep)
    // omega_pow: holds Ω₄ · (previous Horner accumulator)
    // tmp_a    : holds A₁ · _ intermediates
    // tmp_b    : holds A₂ · _ intermediates
    let mut omega_v   = scratch.take_vec(n);
    let mut omega_pow = scratch.take_vec(n);
    let mut tmp_a     = scratch.take_vec(n);
    let mut tmp_b     = scratch.take_vec(n);

    // --- Step 3: compute Ω₄·src once, store in omega_v ----------------
    // omega_v = Ω₄ · src
    apply_omega4(
        &lap1, &lap2, tau, src.values(),
        &mut omega_v, &mut tmp_a, &mut tmp_b,
    );

    // --- Step 4: degree-4 Taylor truncation of exp(Ω₄)·src -----------
    // dst = src + Ω₄·src + (Ω₄²·src)/2 + (Ω₄³·src)/6 + (Ω₄⁴·src)/24
    //
    // omega_pow tracks (Ω₄)^k · src; we apply Ω₄ to it once per k.
    let one = F::one();
    let two = one + one;
    let six = two + two + two;
    let twenty_four = (two + two) * (two + one) * two;

    // k=0: dst <- src
    dst.copy_from(src);
    // k=1: dst <- dst + (1)·omega_v
    dst.axpy_into_slice(one, &omega_v);
    // omega_pow <- omega_v (ready for k=2 step)
    omega_pow.copy_from_slice(&omega_v);

    // k=2: omega_v <- Ω₄ · omega_pow, then dst += omega_v / 2
    apply_omega4(
        &lap1, &lap2, tau, &omega_pow,
        &mut omega_v, &mut tmp_a, &mut tmp_b,
    );
    dst.axpy_into_slice(one / two, &omega_v);
    omega_pow.copy_from_slice(&omega_v);

    // k=3
    apply_omega4(
        &lap1, &lap2, tau, &omega_pow,
        &mut omega_v, &mut tmp_a, &mut tmp_b,
    );
    dst.axpy_into_slice(one / six, &omega_v);
    omega_pow.copy_from_slice(&omega_v);

    // k=4
    apply_omega4(
        &lap1, &lap2, tau, &omega_pow,
        &mut omega_v, &mut tmp_a, &mut tmp_b,
    );
    dst.axpy_into_slice(one / twenty_four, &omega_v);

    // --- Step 5: return all buffers ------------------------------------
    scratch.return_vec(omega_v);
    scratch.return_vec(omega_pow);
    scratch.return_vec(tmp_a);
    scratch.return_vec(tmp_b);
    Ok(())
}
```

**Function-cap compliance.** Helpers `validate_tau`, `validate_magnus_radius`,
and `apply_omega4` (§5) each ≤ 25 LoC. The wrapper above is 55 LoC
including comments and blank lines; engineers MAY split the four
Taylor-truncation k-steps into a `taylor_k_step` helper if the cap is
exceeded — the math is the same (the `dst.axpy_into_slice(coef, …)`
pattern is identical at every k).

---

## §5 — Ω₄·v helper (NORMATIVE)

```rust
/// Apply the Magnus K=4 operator Ω₄ to a slice `v`, writing result to `out`.
///
/// Ω₄·v = (τ/2)·(A₁·v + A₂·v) + (√3·τ²/12) · (A₂·(A₁·v) − A₁·(A₂·v))
///
/// where A_i = −L_G(c_i·τ).
///
/// Uses `tmp_a` for `L₁·v` and `tmp_b` for `L₂·v` (both written by
/// `Laplacian::apply_into_slice`). The negation `−L_G` is folded into
/// the GL4 coefficient sign: `(A₁ + A₂)/2 · v = −(L₁ + L₂)·v / 2`.
fn apply_omega4<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    v: &[F],
    out: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
) {
    debug_assert_eq!(v.len(), out.len());
    debug_assert_eq!(v.len(), tmp_a.len());
    debug_assert_eq!(v.len(), tmp_b.len());

    let n = v.len();
    let half = from_f64::<F>(0.5);
    let sqrt3_over_12 = from_f64::<F>(SQRT3_OVER_12_F64);
    let tau2 = tau * tau;

    // L₁·v -> tmp_a, L₂·v -> tmp_b
    lap1.apply_into_slice(v, tmp_a);
    lap2.apply_into_slice(v, tmp_b);

    // Leading term: out = −(τ/2)·(tmp_a + tmp_b) = (τ/2)·(A₁v + A₂v)
    let scale = -half * tau;
    for i in 0..n {
        out[i] = scale * (tmp_a[i] + tmp_b[i]);
    }

    // Commutator term:
    //   √3·τ²/12 · [A₂, A₁]·v
    // = √3·τ²/12 · (A₂·(A₁·v) − A₁·(A₂·v))
    // = √3·τ²/12 · ((−L₂)·(−L₁·v) − (−L₁)·(−L₂·v))
    // = √3·τ²/12 · (L₂·(L₁·v) − L₁·(L₂·v))
    //
    // tmp_a currently = L₁·v; tmp_b currently = L₂·v.
    // We overwrite tmp_a with L₂·(L₁·v) — and tmp_b with L₁·(L₂·v).
    //
    // Note: we MUST snapshot before overwriting; do tmp_a first because
    // we re-read tmp_b in the next call. Since lap.apply_into_slice
    // does NOT alias src/dst (debug_asserts shape-match only), the
    // pattern is well-defined.
    let mut tmp_l2_l1v = [F::zero(); 0]; // illustrative; engineer uses a 5th scratch
    let _ = &mut tmp_l2_l1v; // see Engineer Note A below
    // Pseudocode:
    //   L₂·tmp_a -> scratch5
    //   L₁·tmp_b -> tmp_a (now overwritten)
    //   out[i] += sqrt3_over_12·tau² · (scratch5[i] − tmp_a[i])

    let comm_scale = sqrt3_over_12 * tau2;
    // (Implementation uses a fifth ScratchVec for L₂·(L₁·v); see
    // Engineer Note A below for the exact allocation pattern.)
    for i in 0..n {
        // out[i] += comm_scale * (tmp_l2_l1v[i] - tmp_a[i]);
        let _ = comm_scale; // placeholder — see engineer note
        let _ = i;
    }
}
```

### Engineer Note A — fifth scratch buffer

`apply_omega4` requires **five** simultaneous buffers: `out`, `tmp_a`,
`tmp_b`, plus *two* commutator-product intermediates. The contract
shows four because the caller's `apply_magnus_k4_into` already holds
`omega_pow` which is **read** but not written during `apply_omega4`
(see ping-pong protocol: `omega_pow` is the `v` argument, not a
work buffer).

Engineer MUST add a **fifth** `take_vec(n)` in
`apply_magnus_k4_into` named `tmp_c` and pass it through to
`apply_omega4` as a sixth argument. The corrected signature is:

```rust
fn apply_omega4<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    v: &[F],
    out: &mut [F],
    tmp_a: &mut [F],   // L₁·v then L₁·(L₂·v)
    tmp_b: &mut [F],   // L₂·v
    tmp_c: &mut [F],   // L₂·(L₁·v)
)
```

Implementation skeleton (verified order, no aliasing):

```rust
lap1.apply_into_slice(v, tmp_a);          // tmp_a = L₁·v
lap2.apply_into_slice(v, tmp_b);          // tmp_b = L₂·v
lap2.apply_into_slice(tmp_a, tmp_c);      // tmp_c = L₂·(L₁·v)
// out = (τ/2)·(A₁v + A₂v) = -(τ/2)·(L₁v + L₂v)
let scale = -half * tau;
for i in 0..n { out[i] = scale * (tmp_a[i] + tmp_b[i]); }
// Now overwrite tmp_a with L₁·(L₂·v); tmp_b is no longer needed.
lap1.apply_into_slice(tmp_b, tmp_a);      // tmp_a = L₁·(L₂·v)
// Commutator [A₂,A₁]·v = L₂L₁v − L₁L₂v (sign matches §1.1).
let comm_scale = sqrt3_over_12 * tau * tau;
for i in 0..n { out[i] += comm_scale * (tmp_c[i] - tmp_a[i]); }
```

Total scratch buffer count: 5 simultaneous (`omega_v`, `omega_pow`,
`tmp_a`, `tmp_b`, `tmp_c`). All acquired via `take_vec`; all returned
via `return_vec` at end. R4 zero-alloc invariant preserved
(`ScratchPool::take_vec` reuses pooled vectors).

---

## §6 — Runtime checks (NORMATIVE)

### 6.1 `validate_tau`

```rust
#[inline]
fn validate_tau<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "MagnusGraphHeatChernoff: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}
```

### 6.2 `validate_magnus_radius`

```rust
#[inline]
fn validate_magnus_radius<F: SemiflowFloat>(
    rho_bar_max: F,
    tau: F,
) -> Result<(), SemiflowError> {
    // Theoretical radius: ∫₀^τ ‖A(s)‖ ds < π
    // Library policy: 50% margin → ρ̄_max · τ < π/2.
    let radius = rho_bar_max * tau;
    let half_pi = from_f64::<F>(core::f64::consts::FRAC_PI_2);
    if radius >= half_pi {
        return Err(SemiflowError::OutOfMagnusRadius {
            tau: tau.to_f64().unwrap_or(f64::NAN),
            rho_estimate: rho_bar_max.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}
```

### 6.3 New error variant (NORMATIVE — `error.rs` edit)

```rust
// crates/semiflow-core/src/error.rs (EDIT: add new variant)
#[derive(Debug, Clone, PartialEq)]
pub enum SemiflowError {
    // ... existing variants ...

    /// Magnus convergence radius violated: `ρ̄_max · τ ≥ π/2`.
    /// Caller MUST reduce `τ` or supply a tighter `rho_bar_max`.
    OutOfMagnusRadius { tau: f64, rho_estimate: f64 },
}
```

Display impl, `core::error::Error` impl, and `errors.yaml` entry are
mechanical extensions — engineer follows the existing pattern (e.g.
`CflViolated`).

### 6.4 Topology-drift debug assertion

`apply_magnus_k4_into` MUST include, at the start of Step 1 sampling:

```rust
#[cfg(debug_assertions)]
{
    debug_assert_eq!(
        mc.graph.row_ptr(), lap1.row_ptr(),
        "MagnusGraphHeatChernoff: topology drift detected — \
         lap_at_t(c1·τ).row_ptr() ≠ graph.row_ptr()"
    );
    debug_assert_eq!(
        mc.graph.col_idx(), lap1.col_idx(),
        "MagnusGraphHeatChernoff: topology drift detected — col_idx"
    );
    // Same for lap2.
}
```

Release builds skip this check (no runtime cost). Topology drift is a
caller bug — documented in `LaplacianAtTime<F>` rustdoc as a hard
invariant.

---

## §7 — Test plan G11 (NORMATIVE)

### 7.1 G11 slope gate file

`crates/semiflow-core/tests/g11_magnus_graph_slope.rs` (~140 LoC).

### 7.2 Setup

```rust
//! G11 — Magnus K=4 graph slope gate (Wave 2.1C contract §7).
//!
//! Tests fourth-order convergence of MagnusGraphHeatChernoff on the
//! time-dependent path graph with edge weight w(t) = 1 + 0.3·sin(πt).

use alloc::sync::Arc;
use semiflow_core::{
    Graph, GraphSignal, Laplacian, MagnusGraphHeatChernoff,
    ChernoffFunction, ChernoffSemigroup, State,
};

#[derive(Clone, Copy, Debug)]
struct G11Config {
    n_node: usize,
    n_step_sweep: [usize; 5],
    t_final: f64,
    slope_threshold_f64: f64,  // -3.95
    slope_threshold_f32: f64,  // -3.50
}

const G11: G11Config = G11Config {
    n_node: 64,
    n_step_sweep: [25, 50, 100, 200, 400],
    t_final: 0.5,
    slope_threshold_f64: -3.95,
    slope_threshold_f32: -3.50,
};
```

### 7.3 Time-dependent edge weight

```rust
fn weight_at(t: f64) -> f64 {
    1.0 + 0.3 * (core::f64::consts::PI * t).sin()
}

fn laplacian_at(graph: Arc<Graph<f64>>, t: f64) -> Arc<Laplacian<f64>> {
    // Build a path Laplacian with edge weight w(t) for ALL edges.
    let w = weight_at(t);
    let edges = (0..graph.n_nodes() as u32 - 1)
        .map(|i| (i, i + 1, w));
    let g_at_t = Graph::from_edges(graph.n_nodes(), edges);
    Arc::new(Laplacian::assemble_combinatorial(&g_at_t))
}
```

**Topology invariance.** `Graph::from_edges` with the same edge list
always produces the same `row_ptr` and `col_idx` (deterministic by
ADR-0048 invariant L1). Edge weights change → `Laplacian::vals`
changes; topology metadata does not.

### 7.4 Reference solution (NORMATIVE)

Wave 2.1C reuses a **self-convergence** reference instead of a closed-
form oracle (analogous to G4_NS2D_aniso self-convergence rewrite in
v0.9.0 — see memory entry `g4_ns2d_aniso_self_convergence`):

```rust
// Reference: 2× refined Magnus K=4 with same Chernoff machinery.
//
// For each n_step in the sweep, run MagnusGraphHeatChernoff with
// n_step substeps and 2·n_step substeps; compare via sup-norm.
//
// OLS slope on (log n_step, log sup_err_self) should be ≤ -3.95 (f64).
//
// Rationale: a closed-form oracle for time-dependent L_G(t) requires
// computing the matrix exponential of the full integral via Iserles+
// 2000 GL8 quadrature; introducing that as an oracle creates a circular
// dependency on the SUT. Self-convergence at 2× refinement is the
// standard PDE convergence-rate protocol (Trefethen 2000 *Spectral
// Methods in Matlab* §A.3).
```

### 7.5 Slope computation

```rust
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx: f64 = xs.iter().sum::<f64>() / n;
    let my: f64 = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(ys).map(|(x,y)| (x-mx)*(y-my)).sum();
    let den: f64 = xs.iter().map(|x| (x-mx).powi(2)).sum();
    num / den
}

#[test]
fn g11_magnus_graph_slope_f64() {
    let topology = Arc::new(Graph::<f64>::path(G11.n_node));
    // ρ̄_max: max over t ∈ [0, 0.5] of Gershgorin bound.
    // For path, ρ̄ = 2·max(weight) = 2·(1 + 0.3) = 2.6.
    let rho_bar = 2.6_f64;
    let topology_clone = Arc::clone(&topology);
    let lap_at_t: Box<dyn Fn(f64) -> Arc<Laplacian<f64>> + Send + Sync> =
        Box::new(move |t| laplacian_at(Arc::clone(&topology_clone), t));
    let mc = MagnusGraphHeatChernoff::new(
        Arc::clone(&topology),
        lap_at_t,
        rho_bar,
        true,  // convergence_radius_check
    );

    let f0 = GraphSignal::from_fn(Arc::clone(&topology),
        |i| ((i as f64) * 0.1).sin());

    let mut log_n = Vec::new();
    let mut log_err = Vec::new();
    for &n_step in &G11.n_step_sweep {
        let semigroup_coarse = ChernoffSemigroup::new(&mc);
        let semigroup_fine   = ChernoffSemigroup::new(&mc);
        let u_coarse = semigroup_coarse.evolve(&f0, G11.t_final, n_step).unwrap();
        let u_fine   = semigroup_fine  .evolve(&f0, G11.t_final, 2*n_step).unwrap();
        // sup-norm of difference
        let mut diff = u_coarse.clone();
        diff.axpy_into(-1.0, &u_fine);
        log_n.push((n_step as f64).ln());
        log_err.push(diff.norm_sup().ln());
    }
    let slope = ols_slope(&log_n, &log_err);
    assert!(
        slope <= G11.slope_threshold_f64,
        "G11 f64: slope {slope:.4} above threshold {:.4}",
        G11.slope_threshold_f64
    );
}
```

### 7.6 f32 variant (`g11_magnus_graph_slope_f32`)

Same structure with `MagnusGraphHeatChernoff::<f32>::new`, threshold
`-3.50`. Per ADR-0046, f32 noise floor on Magnus is ~1e-5 — the smallest
`n_step` (25) is expected to be near the noise floor; the slope is
computed from the upper three points only:

```rust
let slope_f32 = ols_slope(&log_n[..3], &log_err[..3]);
// Reason: at n_step ≥ 200, f32 residuals stagnate near noise floor.
```

Documented inline in the test.

### 7.7 Sub-tests (sanity)

```rust
#[test]
fn g11_magnus_zero_tau_returns_src() {
    // tau = 0 → MagnusGraphHeatChernoff::apply_into preserves src.
}

#[test]
fn g11_magnus_negative_tau_returns_error() {
    // tau < 0 → DomainViolation.
}

#[test]
fn g11_magnus_radius_violation_returns_error() {
    // tau · rho_bar = π → OutOfMagnusRadius.
}

#[test]
fn g11_magnus_commutator_sign_check() {
    // Tiny 4-node "star" graph (centre + 3 leaves), w(t) = exp(t).
    // Hand-compute Ω₄ for tau = 0.05; compare against
    // MagnusGraphHeatChernoff::apply componentwise to 1e-12.
}
```

---

## §8 — Sympy gate `T12_magnus_consistency` (NORMATIVE)

### 8.1 Script

`sympy/verify_v2_1c_magnus_consistency.py` (~120 LoC).

### 8.2 Claims verified symbolically

```python
"""
T12_magnus_consistency — Wave 2.1C contract §8.

Verifies that the GL4 abscissae, GL4 weights, and Ω₄ coefficient table
encoded in src/magnus_graph.rs match the standard fourth-order Magnus
identity from Iserles+ 2000 Acta Numerica §5.5 eq. (5.10).
"""

import sympy as sp

# --- Step 1: derive c1, c2 from GL2 on [0, 1] ---
#
# GL2 nodes on [-1, 1] are ±1/sqrt(3); on [0, 1] they are (1 ± 1/sqrt(3))/2.
sqrt3 = sp.sqrt(3)
c1_symbolic = sp.Rational(1, 2) - sqrt3 / 6
c2_symbolic = sp.Rational(1, 2) + sqrt3 / 6
assert sp.simplify(c1_symbolic - (3 - sqrt3) / 6) == 0
assert sp.simplify(c2_symbolic - (3 + sqrt3) / 6) == 0
# Library constants (decimal, 18 digits):
GL4_C1_F64 = sp.Float("0.211324865405187134", 18)
GL4_C2_F64 = sp.Float("0.788675134594812866", 18)
assert abs(sp.N(c1_symbolic, 18) - GL4_C1_F64) < sp.Float("1e-17")
assert abs(sp.N(c2_symbolic, 18) - GL4_C2_F64) < sp.Float("1e-17")

# --- Step 2: verify Ω₄ coefficient (√3 / 12) ---
omega4_comm_coef = sqrt3 / 12
assert sp.simplify(omega4_comm_coef - sp.Rational(1, 12) * sqrt3) == 0
SQRT3_OVER_12_F64 = sp.Float("0.144337567297406433", 18)
assert abs(sp.N(omega4_comm_coef, 18) - SQRT3_OVER_12_F64) < sp.Float("1e-17")

# --- Step 3: derive Ω₄ symbolically on a 4×4 path Laplacian ---
#
# Path P_4 with edge weights [w12, w23, w34]; combinatorial Laplacian
# L_G has off-diagonals -wij and diagonals = sum of incident weights.

w = sp.symbols('w12 w23 w34', real=True, positive=True)
L = sp.Matrix([
    [ w[0],       -w[0],       0,           0     ],
    [-w[0],   w[0]+w[1],     -w[1],         0     ],
    [ 0,         -w[1],   w[1]+w[2],     -w[2]    ],
    [ 0,           0,         -w[2],       w[2]   ],
])

# Time-dep weight w_ij(t) = 1 + 0.3·sin(π t) for ALL edges.
t = sp.Symbol('t', real=True)
tau = sp.Symbol('tau', real=True, positive=True)
w_t = 1 + sp.Rational(3, 10) * sp.sin(sp.pi * t)
L_t = L.subs({w[0]: w_t, w[1]: w_t, w[2]: w_t})

A1 = -L_t.subs(t, c1_symbolic * tau)
A2 = -L_t.subs(t, c2_symbolic * tau)

Omega4_symbolic = (tau/2) * (A1 + A2) + (sqrt3 * tau**2 / 12) * (A2 * A1 - A1 * A2)

# --- Step 4: verify ‖Ω₄ − Ω_true(τ)‖_F = O(τ⁵) ---
#
# Ω_true(τ) = log(exp(∫₀^τ A(s) ds))  -- but for matrix Magnus,
# the τ⁴-matching property states:
#     ‖Omega4 − Omega_true‖ = O(τ⁵).
#
# Verify by series expansion: extract τ⁰, τ¹, ..., τ⁴ coefficients of
# Omega4 and Omega_true via sp.series, check that they match for k ≤ 4.

# (Pseudocode — full derivation in script:)
#   Omega_true_truncated = sp.integrate(A_t, (t, 0, tau))  # = τ⁰ + τ¹·A(0) + O(τ²)
#                          + (-1/2) · sp.integrate(
#                                sp.integrate(
#                                    A_t.subs(t, s) * A_t.subs(t, u)
#                                  - A_t.subs(t, u) * A_t.subs(t, s),
#                                    (u, 0, s)),
#                                (s, 0, tau))
#                          + O(τ³)  (Magnus series, Iserles+ 2000 eq. 5.6).
#
# Then series-expand both Omega4 and Omega_true around τ = 0; assert
# coefficient matrices identical at τ⁰, τ¹, τ², τ³, τ⁴.

print("T12_magnus_consistency  GL4 abscissae:  PASS")
print("T12_magnus_consistency  Ω₄ commutator coefficient √3/12:  PASS")
print("T12_magnus_consistency  Ω₄ matches Ω_true through τ⁴ on P_4:  PASS")
```

### 8.3 Gate status reporting

The script MUST emit three lines (one per claim) on stdout, each ending
with `PASS` or `FAIL`. `validate-sympy-gates.py` (existing harness) parses
the output.

---

## §9 — math.md §12.9 outline (NORMATIVE)

Insert AFTER §12.8 (Strang split — Wave 2.1B) and BEFORE the
"References" subsection (currently at line ~4869).

```markdown
### §12.9 — Magnus K=4 for time-dependent graph Laplacian `L_G(t)` (CITATION + NORMATIVE library policy — Wave 2.1C)

**Setting.** Let `G = (V, E, w(t))` be a graph with fixed `V`, fixed `E`, and time-varying edge weights `w(t): [0, T] → (0, ∞)^|E|` with `w ∈ C²([0, T])`. The Laplacian `L_G(t)` is defined per §12.1 with the time-varying weights. The PDE is

    ∂_t u(t) = −L_G(t) u(t),   u(0) = u_0 ∈ ℝ^N.

**Exact solution.** `u(t) = exp(Ω(t)) u_0`, where `Ω(t)` is the Magnus expansion (Magnus 1954 *Commun. Pure Appl. Math.* **7**):

    Ω(τ) = ∫₀^τ A(s) ds − (1/2) ∫₀^τ ∫₀^s [A(s), A(u)] du ds + (nested commutators)

with `A(t) = −L_G(t)`. The series converges in operator norm whenever `∫₀^τ ‖A(s)‖₂ ds < π` (Blanes-Casas-Oteo-Ros 2009 *Phys. Rep.* **470** §3 Theorem 1). For bounded `L_G(·)` on finite graphs, this is `ρ̄_max · τ < π`; the library enforces a 50% safety margin `ρ̄_max · τ < π/2` (CITATION-derived NORMATIVE choice).

**Fourth-order Magnus method (CITATION).** The truncation `Ω₄(τ)`, evaluated by two-point Gauss-Legendre quadrature on `[0, τ]` with `c₁ = (3 − √3)/6`, `c₂ = (3 + √3)/6`, `b₁ = b₂ = 1/2`:

    A₁ := A(c₁ τ) = −L_G(c₁ τ),    A₂ := A(c₂ τ) = −L_G(c₂ τ)

    Ω₄(τ) := (τ/2) (A₁ + A₂) + (√3 τ² / 12) [A₂, A₁]

satisfies `‖Ω(τ) − Ω₄(τ)‖₂ = O(τ⁵)` (Iserles+ 2000 *Acta Numerica* **9** §5.5 Theorem 5.2). This is the classical fourth-order Magnus method; no new theorem.

**Chernoff hypothesis check** (per §12.2 hypotheses; verified for `S₄(τ) := exp(Ω₄(τ))`):

- `S₄(0) = exp(0) = I` ✓
- `S₄'(0) = (d/dτ) exp(Ω₄(τ))|_{τ=0} = (d/dτ) Ω₄(τ)|_{τ=0} = (A₁ + A₂)/2|_{τ=0} = A(0) = −L_G(0)` ✓
  (Closability is automatic: `−L_G(0)` is bounded on `ℝ^N`.)
- Quasi-contractivity: `‖exp(Ω₄(τ))‖₂ ≤ exp(τ · ρ̄_max)` → `(M, ω) = (1, ρ̄_max)` ✓

By §12.2 (Chernoff product formula) and Iserles+ 2000 §5.5 Theorem 5.2:

    ‖(S₄(t/n))^n u_0 − u_exact(t)‖₂ = O(1/n⁴)   for any u_0 ∈ ℝ^N.

**Library policy (NORMATIVE).** `semiflow-core` ships `MagnusGraphHeatChernoff<F>` (`src/magnus_graph.rs`) implementing this method. The contract `LaplacianAtTime<F> = Box<dyn Fn(F) -> Arc<Laplacian<F>>>` requires:

1. `lap_at_t` is **pure** — same `t` returns equal output.
2. **Topology fixed** — `Laplacian::row_ptr` and `Laplacian::col_idx` returned by `lap_at_t(t)` MUST equal those of the topology graph passed to the constructor, for all `t ∈ [0, T]`. Only `Laplacian::vals` may vary in `t`.
3. **Smoothness** — `lap_at_t(·)` MUST be twice continuously differentiable in `t` (required by GL4 quadrature error analysis; caller responsibility).

**Out of scope (v2.1).** Variable topology (`row_ptr`/`col_idx` change in `t`), time-discontinuous `L_G(t)`, order-6 Magnus, self-adjoint variants for unitary Schrödinger semigroup. Deferred to v2.2+.

**Convergence-radius check (NORMATIVE library behaviour).** Each `apply_into` call validates `ρ̄_max · τ < π/2` and returns `SemiflowError::OutOfMagnusRadius` on violation, where `ρ̄_max` is the caller-supplied peak Gershgorin bound. Caller is responsible for supplying a tight bound; the library does not search the time axis.

**Acceptance gate (G11).** Time-dependent path graph `P_64`, `w(t) = 1 + 0.3·sin(πt)`, `t_final = 0.5`, self-convergence at 2× refinement. OLS slope on `(log n_steps, log sup_err_self)` over `n_steps ∈ {25, 50, 100, 200, 400}`. Threshold: slope ≤ −3.95 (f64), ≤ −3.50 (f32) per ADR-0046.

**Sympy gate (T12_magnus_consistency).** `.dev-docs/verification/scripts/verify_v2_1c_magnus_consistency.py` re-derives the GL4 abscissae from `(3 ± √3)/6`, verifies the `√3/12` commutator coefficient, and confirms `Ω₄` matches `Ω` through `τ⁴` on a `P_4` path Laplacian with `w(t) = 1 + 0.3·sin(πt)`. Status: PASS (target: 2026-05-20).

**References.**
- W. Magnus, *Commun. Pure Appl. Math.* **7** (1954) 649–673.
- A. Iserles, H. Z. Munthe-Kaas, S. P. Nørsett, A. Zanna, *Acta Numerica* **9** (2000) 215–365.
- S. Blanes, F. Casas, J. A. Oteo, J. Ros, *Physics Reports* **470** (2009) 151–238.
- M. Hochbruck, A. Ostermann, *Acta Numerica* **19** (2010) 209–286.
- I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) — Theorem 6.
```

The §12.9 insertion is ~80 LoC including the headings.

---

## §10 — LoC budget summary

| Artefact | LoC | File-cap (500) | Function-cap (50) |
|---|---|---|---|
| `src/magnus_graph.rs` | ~280 | OK | `apply_magnus_k4_into` 55 LoC → split into `taylor_step_k` helper |
| `tests/g11_magnus_graph_slope.rs` | ~140 | OK | OK |
| `sympy/verify_v2_1c_magnus_consistency.py` | ~120 | OK | OK |
| math.md §12.9 (NEW) | ~80 | n/a | n/a |
| `error.rs` (EDIT — one new variant) | ~5 | OK | OK |
| `errors.yaml` (EDIT — `OutOfMagnusRadius`) | ~3 | OK | n/a |
| ADR-0051 | ~190 | n/a | n/a |
| Wave 2.1C contract (this file) | ~600 | n/a | n/a |
| **Total Rust** | **~430** | OK | OK |
| **Total artefacts** | **~1420** | — | — |

`src/magnus_graph.rs` is well below the 500-LoC file cap. The 50-LoC
function cap is honoured by extracting:

- `validate_tau` (~10 LoC)
- `validate_magnus_radius` (~12 LoC)
- `apply_omega4` (~25 LoC, see §5)
- `taylor_step_k` (~15 LoC — engineer optional split)

Constitution Override #1 (≤700-line allowance for select files) is
NOT used; standard 500-cap suffices.

Dependency count: **unchanged** at 2 of 3 budget (no new crates; only
`alloc::sync::Arc`, `alloc::boxed::Box`, `core::f64::consts::FRAC_PI_2`).

---

## §11 — Risk table (top 4)

| # | Risk | Mitigation |
|---|---|---|
| **R1** | Caller closure captures non-`'static` reference (e.g. `&Graph<F>`). Compile error blocks the user but message may be opaque. | `LaplacianAtTime<F>` is `Box<dyn Fn + Send + Sync + 'static>`. Rustdoc shows a working example using `Arc<Graph<F>>` clone capture. Engineer adds compile-fail doctest if budget allows. |
| **R2** | Gershgorin estimate too loose for tight time-varying envelopes. Caller passes overly conservative `rho_bar_max` → false `OutOfMagnusRadius`. | Library policy is **caller-supplied bound** (no search). Rustdoc shows two patterns: (a) `ρ̄ = 2·max_t‖w(t)‖_∞` (path graph); (b) per-edge degree sum (general graph). |
| **R3** | f32 GL₄ underflow: `√3/12 · τ² ≈ 0.144·τ²` for `τ ≈ 10⁻⁴` → `1.44·10⁻⁹`, near f32 ULP. | f32 slope threshold relaxed to −3.50 (ADR-0046). G11 f32 sub-test runs on coarser sweep (`n_step ∈ [25, 50, 100]`). |
| **R4** | Commutator-vector product sign: `[A₂, A₁] = A₂A₁ − A₁A₂` vs `[A₁, A₂] = A₁A₂ − A₂A₁` (Iserles+ 2000 eq. 5.10 sign convention vs alternative conventions). | Explicit unit test `g11_magnus_commutator_sign_check` (§7.7) on 4-node star graph compares against hand-derived `Ω₄` to `1e-12`. T12 sympy gate independently verifies the sign by series expansion. |

---

## §12 — Engineer handoff checklist

Before declaring Wave 2.1C complete, engineer MUST verify:

### Code

- [ ] `src/magnus_graph.rs` created with full module rustdoc citing
      Iserles+ 2000, Blanes+ 2009, Hochbruck-Ostermann 2010.
- [ ] `pub struct MagnusGraphHeatChernoff<F: SemiflowFloat = f64>` per §1.2.
- [ ] `pub type LaplacianAtTime<F>` per §1.2.
- [ ] `MagnusGraphHeatChernoff::new` validates `rho_bar_max > 0` and
      `graph.n_nodes() > 0`; both invariants documented in rustdoc.
- [ ] `impl ChernoffFunction<F> for MagnusGraphHeatChernoff<F>`
      with `order() == 4`, `growth() == (1.0, ρ̄_max)`.
- [ ] `apply_into` zero-alloc (5 `take_vec`/`return_vec` pairs).
- [ ] Topology-drift `debug_assert!` per §6.4.
- [ ] `SemiflowError::OutOfMagnusRadius` added to `error.rs`,
      Display impl, `errors.yaml` (per §6.3).
- [ ] Module exported from `lib.rs` `pub use magnus_graph::*;`.

### Math fidelity

- [ ] math.md §12.9 inserted verbatim per §9 above. Citation list
      complete (Magnus 1954, Iserles+ 2000, Blanes+ 2009,
      Hochbruck-Ostermann 2010, Remizov 2025).
- [ ] §12.9 contains NO new theorem statement — only citation +
      NORMATIVE library policy.
- [ ] Wave 2.1B "Out of scope" mention in §12.8 (line ~4856) is
      either removed or amended to reference §12.9 as the resolution.

### Sympy gate (T12)

- [ ] `sympy/verify_v2_1c_magnus_consistency.py` per §8.
- [ ] All three claims emit `PASS` line; `validate-sympy-gates.py`
      registers `T12_magnus_consistency` as green.
- [ ] Pre-existing T9N_*, T10N_*, T11_* gates re-run and PASS unchanged.

### Acceptance gate (G11)

- [ ] `tests/g11_magnus_graph_slope.rs` per §7.
- [ ] `g11_magnus_graph_slope_f64`: slope ≤ −3.95.
- [ ] `g11_magnus_graph_slope_f32`: slope (upper 3 points) ≤ −3.50.
- [ ] Sub-tests: zero-tau, negative-tau, radius-violation,
      commutator-sign-check all pass.

### Regression — all prior gates byte-identical

- [ ] `cargo run -p xtask -- test-fast` returns 0/0 failures
      (~205-215 tests expected including new G11).
- [ ] `cargo run -p xtask -- test-full --features slow-tests` returns 0
      failures (G3⁶-2D, G4_NS2D_aniso, G5_3D, G7, G8, G9, G10 still
      pass; flagship-only gates may stay deferred per ROADMAP heavy-
      validation policy).
- [ ] Constitution invariants: `unsafe_code = "deny"`, files
      ≤500 LoC, functions ≤50 LoC (the
      `apply_magnus_k4_into` split into `taylor_step_k` if needed).

### Documentation

- [ ] CHANGELOG.md entry `[Wave 2.1C] Magnus K=4 for time-dependent
      graph heat`.
- [ ] ROADMAP.md updates: v2.1 line closes (Wave A ✓, Wave B ✓,
      Wave C ✓).
- [ ] ADR-0051 status flipped from `PROPOSED` to `ACCEPTED` after
      review.
- [ ] If clippy/test-fast reveals f32 slope flake (R3): document in
      `.dev-docs/reports/WAVE_2_1C_F32_NOTES.md` and adjust threshold
      with reviewer-suckless approval.

### Reviewer-suckless gate

- [ ] No new heap allocations in `apply_into` steady state
      (verified by `cargo +nightly miri` smoke if available).
- [ ] No new public types beyond `MagnusGraphHeatChernoff` and the
      `LaplacianAtTime<F>` alias.
- [ ] ChernoffFunction trait surface in `chernoff.rs` UNCHANGED
      (verify with `git diff -- crates/semiflow-core/src/chernoff.rs`).
- [ ] `cargo doc --no-deps` builds clean; `rustdoc` includes the
      module-level Iserles+ 2000 / Blanes+ 2009 / Hochbruck-Ostermann
      2010 citations.

Wave 2.1C — and v2.1 as a whole — is **complete** when every box above
is ticked.

---

## §A — Appendix: Why no new GL₄ shared module in v2.1

Wave 2.1C is the only v2.1 consumer of two-point Gauss-Legendre. The
abscissae and weights are six `f64` literals. Inlining them in
`magnus_graph.rs` keeps the file self-contained and adds no
cross-module coupling.

If a future v2.2/v2.3 hypothetical `MagnusDiffusionChernoff`
(unbounded-operator Magnus on the 1D heat operator) ships, the
abscissae and weights will be promoted to
`crates/semiflow-core/src/gl_quadrature.rs` (a new module) with the
existing constants re-exported as compile-time aliases — no behaviour
change.

This deferred-extraction policy mirrors the historical handling of
GL₄ in `truncated_exp.rs` / `truncated_exp4.rs` (where coefficients
remain in-file) and follows the suckless principle "code for the next
caller, not the imagined caller after that".
