# Wave 2.2B Contract — Advanced Semigroups

**Status**: NORMATIVE — engineer implements verbatim against this contract.
**ADRs**: 0055 (`AdjointChernoff`), 0056 (`MagnusGraphHeat6thChernoff`), 0057
(`SchrodingerChernoff`).
**Depends on**: v2.2 Wave A — `VarCoefGraphHeatChernoff`, `GraphTraj<F>`,
`MagnusGraphHeatChernoff::evolve_with_traj` (all shipped at Wave A tag).
**Math**: contracts/semiflow-core.math.md §15 (Adjoint), §16 (Order-6),
§17 (Schrödinger).
**Sympy gates**: T13N (adjoint consistency), T14N (Magnus K=6), T15N
(Schrödinger unitarity) — all NEW NORMATIVE.
**Slope gates**: G15 (adjoint self-adjoint identity), G16 (dual-pairing),
G17 (Magnus K=6), G18 (Schrödinger unitarity), G19 (harmonic-osc oracle).
**Author**: ai-solutions-architect · **Date**: 2026-05-21.

This wave ships THREE new public types:

1. `AdjointChernoff<C, F>` — wrapper for backward semigroup over any
   `ChernoffFunction<F, S: HilbertState<F>>` (ADR-0055).
2. `MagnusGraphHeat6thChernoff<F>` — order-6 Magnus on time-dependent
   graph Laplacian (ADR-0056). **f64 ONLY**.
3. `SchrodingerChernoff<F>` + `SchrodingerState<F>` — unitary semigroup
   for real Schrödinger `i ψ_t = (−Δ + V) ψ` via 2N-dim real
   representation (ADR-0057). **Real-only scope** — Option A picked.

---

## §1 — `AdjointChernoff<C, F>` (NORMATIVE — ADR-0055)

### 1.1 File location

```text
crates/semiflow-core/src/adjoint.rs    (NEW FILE, ≤ 500 LoC)
```

### 1.2 Public API

```rust
//! crates/semiflow-core/src/adjoint.rs

use core::marker::PhantomData;
use crate::{HilbertState, SemiflowError, ScratchPool};
use crate::chernoff::ChernoffFunction;
use crate::float::SemiflowFloat;

/// Adjoint (backward) semigroup wrapper. See math.md §15.
///
/// For self-adjoint inner generators (combinatorial graph Laplacian,
/// non-mixed isotropic 2D/3D diffusion, etc.) this is a thin re-export.
/// For non-self-adjoint inners (drift-reaction with `b ≠ 0`, anisotropic
/// non-separable 2D) this provides the genuine dual evolution.
#[derive(Clone, Debug)]
pub struct AdjointChernoff<C, F: SemiflowFloat = f64>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    inner: C,
    is_self_adjoint: bool,
    _f: PhantomData<F>,
}

impl<C, F: SemiflowFloat> AdjointChernoff<C, F>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    /// Construct for non-self-adjoint inner generator.
    /// Cost: 1 `apply_into` + 1 `HilbertState::dot` per step.
    /// Order: min(inner.order(), 2).
    pub fn new_general(inner: C) -> Self;

    /// Construct for known-self-adjoint inner.
    /// Cost: 1 `apply_into` per step (delegates directly).
    /// Order: inner.order().
    ///
    /// **Caller assertion**: library does NOT verify self-adjointness.
    /// Misuse leads to incorrect results, not crashes.
    pub fn new_self_adjoint(inner: C) -> Self;

    pub fn inner(&self) -> &C;
    pub fn is_self_adjoint(&self) -> bool;

    /// Optional developer tool — probabilistic self-adjointness check.
    /// Samples `n_samples` random states; verifies `|⟨A f, g⟩ - ⟨f, A g⟩| < tol`.
    /// NOT used in hot path.
    pub fn detect_self_adjointness(
        inner: &C,
        n_samples: usize,
        tol: F,
    ) -> Result<bool, SemiflowError>
    where
        C::S: Clone;
}

impl<C, F: SemiflowFloat> ChernoffFunction<F> for AdjointChernoff<C, F>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    type S = C::S;

    fn apply(
        &self,
        tau: F,
        f: &Self::S,
    ) -> Result<Self::S, SemiflowError>
    where
        Self::S: Clone;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;

    fn order(&self) -> u32 {
        if self.is_self_adjoint {
            self.inner.order()
        } else {
            core::cmp::min(self.inner.order(), 2)
        }
    }

    fn growth(&self) -> (f64, f64) {
        self.inner.growth()  // ‖U‖ = ‖U*‖ on Hilbert space.
    }
}
```

### 1.3 Algorithm (NORMATIVE)

`apply_into` (general case):

1. Call `self.inner.apply_into(tau, src, dst, scratch)?` to get forward evolution.
2. Compute the dual-pairing correction: scratch-allocate one work-buffer
   `g_basis` of length `src.len()`. Initialise `g_basis = src` (the
   identity component).
3. Apply `correction[i] = - tau * (src.dot(g_basis)) / src.len()` (the
   `O(τ²)` bounded-perturbation closure from math.md §15.1).
   **Note**: this is a simplified closure; the full bounded-perturbation
   expansion (math.md §15.1) has explicit form `(τ² / 2) · (A_sym B − B^T A_sym + ...)`.
   v2.2 ships the leading-order term only; higher-order is open for v2.3+.
4. Write `dst[i] += correction[i]` (axpy_into).

Self-adjoint case: delegate to `self.inner.apply_into` directly.

### 1.4 R4 zero-alloc invariant

All work-buffers MUST come from `ScratchPool`. Zero allocation per step.

### 1.5 Generic-over-F coverage

`F: SemiflowFloat`. f32 G16 dual-pairing tolerance relaxed to `1e-6`
(ADR-0055 R2).

---

## §2 — `MagnusGraphHeat6thChernoff<F>` (NORMATIVE — ADR-0056)

### 2.1 File location

```text
crates/semiflow-core/src/magnus6_graph.rs    (NEW FILE, ~580 LoC; Override #1 carve-out)
```

### 2.2 Public API

```rust
//! crates/semiflow-core/src/magnus6_graph.rs

use alloc::sync::Arc;
use crate::{Graph, GraphSignal, Laplacian, SemiflowError, ScratchPool};
use crate::chernoff::ChernoffFunction;
use crate::float::SemiflowFloat;
use crate::magnus_graph::LaplacianAtTime;

/// GL₆ quadrature abscissae (closed-form constants).
pub const GL6_C1: f64 = (5.0 - 15.0_f64.sqrt()) / 10.0;
pub const GL6_C2: f64 = 0.5;
pub const GL6_C3: f64 = (5.0 + 15.0_f64.sqrt()) / 10.0;

pub const GL6_B1: f64 = 5.0 / 18.0;
pub const GL6_B2: f64 = 8.0 / 18.0;
pub const GL6_B3: f64 = 5.0 / 18.0;

/// Order-6 Magnus expansion on time-dependent graph Laplacian.
///
/// **f64 ONLY** — f32 underflow on `τ⁵ · √15 / 1080` coefficient. The
/// `ChernoffFunction<f32>` impl is intentionally missing; compile-time gated.
///
/// See math.md §16 (NORMATIVE) and ADR-0056 (design).
pub struct MagnusGraphHeat6thChernoff<F: SemiflowFloat = f64> {
    graph: Arc<Graph<F>>,
    lap_at_t: LaplacianAtTime<F>,
    rho_bar_max: F,
    convergence_radius_check: bool,
}

impl<F: SemiflowFloat> MagnusGraphHeat6thChernoff<F> {
    pub fn new(
        graph: Arc<Graph<F>>,
        lap_at_t: LaplacianAtTime<F>,
        rho_bar_max: F,
        convergence_radius_check: bool,
    ) -> Result<Self, SemiflowError>;

    pub fn graph(&self) -> &Graph<F>;
    pub fn laplacian_at(&self, t: F) -> Arc<Laplacian<F>>;

    pub fn apply_into_at(
        &mut self,
        t_start: F,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
}

// f64 ONLY — no f32 impl (see §"f32 instability rationale" in ADR-0056).
impl ChernoffFunction<f64> for MagnusGraphHeat6thChernoff<f64> {
    type S = GraphSignal<f64>;

    fn apply(
        &self,
        tau: f64,
        f: &GraphSignal<f64>,
    ) -> Result<GraphSignal<f64>, SemiflowError>
    where GraphSignal<f64>: Clone;

    fn apply_into(
        &self,
        tau: f64,
        src: &GraphSignal<f64>,
        dst: &mut GraphSignal<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError>;

    fn order(&self) -> u32 { 6 }

    fn growth(&self) -> (f64, f64) {
        (1.0, self.rho_bar_max)
    }
}
```

### 2.3 Algorithm (NORMATIVE — math.md §16.2)

`apply_into_at(t_start, tau, src, dst, scratch)`:

1. Convergence-radius check: if `convergence_radius_check && self.rho_bar_max * tau >= π / 2` →
   `Err(OutOfMagnusRadius { tau, rho_estimate: self.rho_bar_max * tau })`.
2. Sample `A_1 = -L_G(t_start + GL6_C1 * tau)`, `A_2 = -L_G(t_start + GL6_C2 * tau)`,
   `A_3 = -L_G(t_start + GL6_C3 * tau)` via `self.lap_at_t(...)`.
3. Compute the weighted sum: `omega_linear[i] = tau * (GL6_B1 * A_1[i] + GL6_B2 * A_2[i] + GL6_B3 * A_3[i])`
   (where `A_k[i]` denotes `A_k * src` row-i value — via sparse mat-vec).
4. Compute the four commutator-vector products (each 2 sparse mat-vecs):
   - `c32 = [A_3, A_2] * src` = `A_3 (A_2 src) - A_2 (A_3 src)`.
   - `c21 = [A_2, A_1] * src` = `A_2 (A_1 src) - A_1 (A_2 src)`.
   - `c31 = [A_3, A_1] * src` = `A_3 (A_1 src) - A_1 (A_3 src)`.
   - `cn1 = [A_2, [A_3, A_1]] * src` = `A_2 c31 - c31_via_A2(...)` (verify
     symbolically that this is `A_2 (c31) - c31_first_then_A2(...)` — see math.md §16.2).
5. Assemble `Ω₆ * src` into `omega_v` work-buffer:
   ```text
   omega_v[i] = omega_linear[i]
              + (sqrt(15) * tau² / 12) * (c32[i] + c21[i])
              - (tau² / 12) * c31[i]
              + (tau³ / 12) * cn1[i]
   ```
6. Apply `exp(Ω₆) src` via degree-6 Taylor truncation:
   ```text
   dst = src + omega_v
              + omega_v² / 2
              + omega_v³ / 6
              + ...
              + omega_v⁶ / 720
   ```
   Each `omega_v^k * src` is via repeated sparse mat-vec.

### 2.4 R4 zero-alloc invariant

Pre-allocate 7 work-buffers in constructor (each length N):
`omega_linear`, `c32`, `c21`, `c31`, `cn1`, `tmp_a`, `tmp_b`. Zero
allocation per step.

### 2.5 Generic-over-F coverage

`F: SemiflowFloat` — type allows f32, but `impl ChernoffFunction<f32>` is
MISSING. Building `MagnusGraphHeat6thChernoff::<f32>::new(...)` compiles
(the type is generic), but using it as a `ChernoffFunction<f32>` fails at
trait-bound check. The error message MUST cite ADR-0056.

---

## §3 — `SchrodingerChernoff<F>` (NORMATIVE — ADR-0057, Option A real-only)

### 3.1 File location

```text
crates/semiflow-core/src/schrodinger.rs    (NEW FILE, ≤ 500 LoC)
```

### 3.2 Public API

```rust
//! crates/semiflow-core/src/schrodinger.rs

use crate::{GridFn1D, Grid1D, SemiflowError, ScratchPool};
use crate::chernoff::ChernoffFunction;
use crate::diffusion4::Diffusion4thChernoff;
use crate::float::SemiflowFloat;
use crate::state::{HilbertState, State};

/// Real representation of complex wavefunction `ψ = ψ_re + i ψ_im`.
/// 2N-dim total state.
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
        // ‖ψ‖_∞ = sup_i √(ψ_re[i]² + ψ_im[i]²)
        // Inline iter over both vectors.
        // ... (see ADR-0057 for stub)
    }
}

impl<F: SemiflowFloat> HilbertState<F> for SchrodingerState<F> {
    fn dot(&self, other: &Self) -> F {
        // Re ⟨ψ, φ⟩ = ψ_re·φ_re + ψ_im·φ_im
        self.psi_re.dot(&other.psi_re) + self.psi_im.dot(&other.psi_im)
    }
}

/// Schrödinger Chernoff `i ψ_t = (-Δ + V(x)) ψ` via Strang splitting.
///
/// Order 2 globally. Unitarity by construction.
///
/// See math.md §17 (NORMATIVE) and ADR-0057 (Option A real-only).
#[derive(Clone, Debug)]
pub struct SchrodingerChernoff<F: SemiflowFloat = f64> {
    kinetic: Diffusion4thChernoff<F>,
    v_at_node: alloc::vec::Vec<F>,
}

impl<F: SemiflowFloat> SchrodingerChernoff<F> {
    /// Construct from kinetic operator (typically `Diffusion4thChernoff::new(...)`)
    /// and potential `V: ℝ → ℝ`.
    pub fn new(
        kinetic: Diffusion4thChernoff<F>,
        v: impl Fn(F) -> F,
    ) -> Result<Self, SemiflowError>;

    pub fn kinetic(&self) -> &Diffusion4thChernoff<F>;
    pub fn v_at_node(&self) -> &[F];
}

impl<F: SemiflowFloat> ChernoffFunction<F> for SchrodingerChernoff<F> {
    type S = SchrodingerState<F>;

    fn apply(/* … */) -> Result<Self::S, SemiflowError>;
    fn apply_into(/* … */) -> Result<(), SemiflowError>;
    fn order(&self) -> u32 { 2 }
    fn growth(&self) -> (f64, f64) { (1.0, 0.0) }  // unitary
}
```

### 3.3 Algorithm (NORMATIVE — math.md §17.3)

`apply_into(tau, src, dst, scratch)`:

1. **Half-step potential rotation**: for each node `i`:
   ```text
   alpha = self.v_at_node[i] * tau * 0.5
   c = alpha.cos()
   s = alpha.sin()
   dst.psi_re[i] = c * src.psi_re[i] + s * src.psi_im[i]
   dst.psi_im[i] = -s * src.psi_re[i] + c * src.psi_im[i]
   ```
2. **Full-step kinetic** (re-uses `Diffusion4thChernoff`):
   ```text
   // psi_re and psi_im evolve independently under K = -Δ.
   tmp_re ← dst.psi_re.clone()   // scratch
   tmp_im ← dst.psi_im.clone()
   self.kinetic.apply_into(tau, &tmp_re, &mut dst.psi_re, scratch)?
   self.kinetic.apply_into(tau, &tmp_im, &mut dst.psi_im, scratch)?
   ```
3. **Half-step potential rotation** (same as step 1, in-place).

### 3.4 R4 zero-alloc invariant

Scratch buffers `tmp_re`, `tmp_im` from `ScratchPool`. Zero allocation
per step. The V-rotation is per-node, no temp allocation needed.

### 3.5 Generic-over-F coverage

`F: SemiflowFloat`. f32 G18 unitarity threshold relaxed to `1e-6`
(ADR-0057 R3).

---

## §4 — Sympy gates (NORMATIVE)

### 4.1 T13N_adjoint_consistency (NEW — ADR-0055)

Path: `.dev-docs/verification/scripts/verify_v2_2_adjoint_consistency.py`

Symbolic 4-node path P_4 with edge-asymmetric weights (`w(0→1) = w₁`,
`w(1→0) = w₂`, with `w₁ ≠ w₂` — non-symmetric Laplacian). Verify that
`AdjointChernoff::apply_into(tau, f)` computes the matrix-adjoint
exponential through τ¹ (the order=2 case → match through τ² in matrix
Frobenius norm).

### 4.2 T14N_magnus6_residual (NEW — ADR-0056)

Path: `.dev-docs/verification/scripts/verify_v2_2_magnus6_residual.py`

Verify GL₆ abscissae `(5 ± √15)/10` and weights `5/18, 8/18, 5/18` derive
from Gauss-Legendre on `[0, 1]`. Verify Ω₆ coefficient table matches
Blanes+ 2009 Table 6 (4 commutator terms with coefficients
`√15·τ²/12`, `−τ²/12`, `−τ²/12`, `τ³/12`). Verify
`‖Ω₆ − Ω_true(τ)‖_F = O(τ⁷)` via sympy series expansion through τ⁶ on
a `P_4` Laplacian with `w(t) = 1 + 0.3·sin(πt)`.

### 4.3 T15N_schrodinger_unitarity (NEW — ADR-0057)

Path: `.dev-docs/verification/scripts/verify_v2_2_schrodinger_unitarity.py`

Verify the V-rotation 2×2 matrix on symbolic `α` is exactly
`[[cos α, sin α], [-sin α, cos α]]`. Verify determinant = 1 (unitary in
real representation). Verify composition with itself: rotation by `α`
then by `β` = rotation by `α + β` (additivity of unitary group on the
1-parameter subgroup).

---

## §5 — Slope gates (NORMATIVE — tests/g{15..19}_*.rs)

### 5.1 G15 adjoint self-adjoint identity (`tests/g15_adjoint_self_adjoint.rs`)

For `inner = GraphHeatChernoff` (symmetric combinatorial Laplacian),
verify `AdjointChernoff::new_self_adjoint(inner).apply_into(τ, f)` is
**bit-equal** (0 ULP) to `inner.apply_into(τ, f)` on both f64 and f32.

### 5.2 G16 dual-pairing gate (`tests/g16_dual_pairing.rs`)

For `inner = DriftReactionChernoff` (`b = 0.5`), `τ ∈ {0.01, 0.05}`,
random `f`, `g`: `|⟨S(τ)·f, g⟩ − ⟨f, S*(τ)·g⟩| < 1e-12` (f64) /
`< 1e-6` (f32). Slope on `n_steps`: ≤ −1.95 (f64) (order 2 cap from
ADR-0055 §"Order-preservation rule").

### 5.3 G17 Magnus K=6 slope (`tests/g17_magnus6_slope.rs`)

Time-dep `P_64`, `w(t) = 1 + 0.3·sin(πt)`, `t_final = 0.5`,
`n_steps ∈ {5, 10, 20, 40, 80}`. Self-convergence. **f64 only.**
Pass: `slope ≤ -5.85`.

### 5.4 G18 Schrödinger unitarity (`tests/g18_unitarity.rs`)

Gaussian wavepacket on `[-5, 5]`, N=64, `V(x) = ½x²`, `t_final = 1.0`,
`n_steps = 100`. `|‖ψ(t_final)‖² − ‖ψ_0‖²| < 1e-12` (f64) / `< 1e-6`
(f32). Slope on `n_steps ∈ {10, 20, 40, 80, 160}`: ≤ −1.95 (f64) /
≤ −1.50 (f32).

### 5.5 G19 harmonic-osc oracle (`tests/g19_harmonic_oscillator.rs`)

Gaussian centred at x=1, σ=0.5. After period `T = 2π`,
`‖ψ(T) − e^{iϕ} ψ_0‖_2 < 1e-3` for some real ϕ. f64 only.

---

## §6 — Capability / security (NORMATIVE)

No new capability boundaries. All three new types are additive Rust APIs.
STRIDE applies the same as v2.1 (in-process, no IPC, no privilege boundary).

`MagnusGraphHeat6thChernoff` adds 7 scratch buffers of length N — at N=1M
this is 56 MB per instance. Cap with conservative `ScratchPool` quotas
in caller code (no library-side check; resource cap is caller responsibility).

---

## §7 — Build/run path (NORMATIVE — unchanged from v2.1)

```bash
cargo run -p xtask -- test-fast
cargo run -p xtask -- test-full
cargo run -p xtask -- test-flagship
```

No new feature flags. (Future v2.3 may add `feature = "schrodinger"` if
the kinetic operator dependency on `Diffusion4thChernoff` becomes
optional — out of v2.2 scope.)

---

## §8 — Engineer pickup ordering (NORMATIVE)

Step 1: Read ADR-0055, ADR-0056, ADR-0057 + math.md §15, §16, §17.

Step 2: Implement `adjoint.rs` (ADR-0055). Add `tests/g15_*.rs` +
`tests/g16_*.rs`. Sympy T13N.

Step 3: Implement `magnus6_graph.rs` (ADR-0056). Add `tests/g17_*.rs`.
Sympy T14N. Note Override #1 carve-out (file projected ≤ 580 LoC).

Step 4: Implement `schrodinger.rs` (ADR-0057) + `SchrodingerState<F>`.
Add `tests/g18_*.rs` + `tests/g19_*.rs`. Sympy T15N.

Step 5: Update `lib.rs` re-exports: `AdjointChernoff`,
`MagnusGraphHeat6thChernoff`, `SchrodingerChernoff`, `SchrodingerState`.

Step 6: Update CHANGELOG.md with Wave 2.2B entry; re-cite ADRs.

Step 7: Update constitution v1.4.0 → v1.5.0 IF Override #1 file-list
needs `magnus6_graph.rs` addition (anticipated at v2.2.0 cut). Confirm
override count stays 3 ≤ 3 (this is an EXPANSION of #1, not a new override).

Step 8: Handoff to git-workflow for Wave 2.2B commit. Trailer:
`Agent: agentic-engineer`, `Task-ID: v2.2-wave-b-advanced-semigroups`.
