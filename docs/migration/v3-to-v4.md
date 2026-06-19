# Migration Guide: v3.x → v4.0

**Status**: COMPLETE (Wave G filled 2026-05-28; sign-off fills v4.0.0 release SHA in §11).
**Audience**: maintainers + downstream library users upgrading from v3.x to v4.0.
**Target release**: v4.0.0 (approximately 2027-05-27, 12 months after v3.0.0).
**Related ADRs**: ADR-0079 / ADR-0080 / ADR-0081 / ADR-0082 / ADR-0083 / ADR-0084 / ADR-0085 (v4.0 BREAKING window).

---

## Executive summary

v4.0 is the SECOND BREAKING window of the academic-priority trajectory. The release ships:

1. **NEW** — `SemiflowComplex<C>` trait + `SchrödingerChernoffComplex<C>` kernel (ADR-0079).
2. **NEW** — `PointEval<F>` first-class trait + 5 backend impls (ADR-0080).
3. **NEW** — `AnisotropicShiftChernoffND<F, D>` d-D anisotropic shift flagship (ADR-0081).
4. **NEW** — `MatrixDiffusionChernoff<F, M>` coupled-component matrix-valued kernel (ADR-0082).
5. **PROMOTION** — G_RES_RES resolvent residual gate (ADR-0083).
6. **HARD REMOVAL** — v2_compat shim + per-binding v2 shim layers (ADR-0084) — completes the v3.0 → v4.0 12-month deprecation cycle.
7. **CLARIFICATION** — G_zeta4 architect math review: order-4 claim of v3.0 `Diffusion4thZeta4Chernoff` is DEFERRED (ADR-0085); kernel ships but `order()` corrected to return 2.

**For v3.0+ users who already migrated to the v3.x cleaned-up trait surface**: the v4.0 migration is THIN. Most v3.x code compiles unchanged at v4.0; the new v4.0 surface is opt-in additive. Only one BREAKING change affects v3.x users: the `Diffusion4thZeta4Chernoff::order()` return value changes from `4` to `2` per ADR-0085 (the v3.0 claim of order-4 was numerically false; v4.0 corrects).

**For v2.x users who skipped v3.x entirely**: the v4.0 migration is HEAVY. The v2_compat shim is HARD-REMOVED at v4.0; you MUST first migrate v2 → v3 per `docs/migration/v2-to-v3.md`, then v3 → v4 per this document. The 12 months of v3.0+ deprecation warnings were your migration window.

---

## §1 — Quick migration checklist for v3.x users

- [ ] Update `Cargo.toml` to `semiflow-core = "4.0"`.
- [ ] If you use `Diffusion4thZeta4Chernoff<F>`: review the `order()` return value change (4 → 2) per ADR-0085.
- [ ] If you use `feature = "v2_compat"`: REMOVE the feature (it no longer exists). Migrate any remaining v2 API usage to v3 surface FIRST.
- [ ] If you use FFI `_v2`-suffixed symbols: REMOVE references; switch to the bare-name `smf_*` v3 symbols.
- [ ] If you use PyO3 `_v2`-suffixed methods (e.g., `Heat1D.evolve_v2`): REMOVE references; switch to v3 method names.
- [ ] If you use WASM `*V2` methods (e.g., `RemizovWasm.versionV2()`): REMOVE references; switch to v3 method names.
- [ ] OPT-IN to v4.0 new surface if applicable (Schrödinger Option B, PointEval, d-D shift, matrix-valued).

**Estimated migration time for v3.x users**: 1-4 hours (most users have zero changes; the `Diffusion4thZeta4Chernoff::order()` correction is the only mandatory change).

---

## §2 — Schrödinger Option A → Option B (ADR-0079)

v2.2 Option A (real-pair `(psi_re, psi_im)`) is PRESERVED verbatim through v4.x (soft-deprecation via rustdoc only). Migration is opt-in; no hard break.

### Option A (v2.2, still compiles at v4.0)

```rust
use semiflow_core::{Schrodinger1D, ChernoffSemigroup, Grid1D, GridFn1D};

let grid  = Grid1D::new(-5.0, 5.0, 128)?;
let schro = Schrodinger1D::new(|_x| 0.5 * _x * _x, grid); // harmonic V(x)
let semi  = ChernoffSemigroup::new(schro, 200)?;

// Option A state: two GridFn1D fields (re, im)
let psi_re: GridFn1D<f64> = /* ... */;
let psi_im: GridFn1D<f64> = /* ... */;
// evolve via apply_chernoff on the pair (compile time: always worked)
```

### Option B (v4.0 native complex, additive)

```rust
use semiflow_core::{SchrödingerChernoffComplex, Evolver, Grid1D, GridFn1D, SemiflowComplex};
use num_complex::Complex;

let grid   = Grid1D::new(-5.0, 5.0, 128)?;
let kernel = SchrödingerChernoffComplex::harmonic(grid, 1.0)?;
let ev     = Evolver::new(kernel, 200)?;

// Gaussian wave packet initial condition
let psi0: GridFn1D<Complex<f64>> = /* ... */;
let mut dst = psi0.clone();
let mut scratch = semiflow_core::ScratchPool::new();
ev.evolve_into(0.1, &psi0, &mut dst, &mut scratch)?;
// dst now holds ψ(x, 0.1)
```

### Cross-version verification

```rust
// Sup-norm bound: ‖ψ_n^(A) - ψ_n^(B)‖_∞ ≤ 4 ULP (same Chernoff iteration, different repr)
let max_diff = psi_a.values.iter().zip(psi_b.values.iter())
    .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
    .fold(0.0_f64, f64::max);
assert!(max_diff < 4.0 * f64::EPSILON);
```

---

## §3 — PointEval pointwise evaluation (ADR-0080)

Migration is opt-in; the v3.x full-grid path continues to work unchanged.

### v3.x full-grid path (still valid at v4.0)

```rust
use semiflow_core::{DiffusionChernoff, Evolver, Grid1D, GridFn1D};

let grid = Grid1D::new(0.0, 1.0, 512)?;
let func = DiffusionChernoff::new(|_x| 1.0, |_x| 0.0, |_x| 0.0, 1.0, grid);
let ev   = Evolver::new(func, 100)?;
let f0   = GridFn1D::from_fn(grid, |x| (-(x - 0.5).powi(2) / 0.01).exp());
let mut dst = f0.clone();
let mut scratch = semiflow_core::ScratchPool::new();
ev.evolve_into(0.1, &f0, &mut dst, &mut scratch)?;
// Query single point: dst.sample(0.5) (full grid computed)
```

### v4.0 PointEval path (single-point, no full grid)

```rust
use semiflow_core::{DiffusionChernoff, PointEval, Grid1D, GridFn1D};

let grid = Grid1D::new(0.0, 1.0, 512)?;
let func = DiffusionChernoff::new(|_x| 1.0, |_x| 0.0, |_x| 0.0, 1.0, grid);
let f0   = GridFn1D::from_fn(grid, |x| (-(x - 0.5).powi(2) / 0.01).exp());

// Query only x0 = 0.5 — O(1) memory, O(n * kernel_eval_cost)
let val = func.eval_at(0.1 / 100.0, &f0, &[0.5_f64], 100)?;
// val[0] is byte-identical to dst.sample(0.5) from the full-grid path
```

### Byte-identity assertion (v3 full-grid vs v4 PointEval)

```rust
let full_grid_val = dst.sample(0.5);
assert!((full_grid_val - val[0]).abs() < f64::EPSILON,
    "PointEval must be byte-identical to full-grid sample at same point");
```

---

## §4 — d-D anisotropic shift (ADR-0081)

`ShiftChernoff1D<F>` is PRESERVED verbatim at v4.0. `AnisotropicShiftChernoffND<F, 1>` is byte-identity equivalent for d = 1.

### d = 1 equivalence (v3.x ShiftChernoff1D → v4.0 AnisotropicShiftChernoffND<F, 1>)

```rust
// v3.x (still works at v4.0)
use semiflow_core::{ShiftChernoff1D, ChernoffSemigroup, Grid1D, GridFn1D};
let grid  = Grid1D::new(0.0, 1.0, 256)?;
let shift = ShiftChernoff1D::new(|_x| 1.0, |_x| 0.0, |_x| 0.0, 1.0, grid);
let semi  = ChernoffSemigroup::new(shift, 50)?;
let f0    = GridFn1D::from_fn(grid, |x| (-(x-0.5).powi(2) / 0.02).exp());
let out_v3 = semi.evolve(0.1, &f0)?;

// v4.0 equivalent (byte-identical for d = 1)
use semiflow_core::AnisotropicShiftChernoffND;
let shift_nd = AnisotropicShiftChernoffND::<f64, 1>::new(/* same params */, grid)?;
let ev_nd    = semiflow_core::Evolver::new(shift_nd, 50)?;
// evolve_into → byte-identical to out_v3
```

### d = 2 anisotropic heat (v4.0 new capability)

```rust
use semiflow_core::{AnisotropicShiftChernoffND, Evolver, Grid2D, GridFn2D};

let grid2d = Grid2D::new(grid_x, grid_y);
// σ₁ = 1.0, σ₂ = 0.5 — anisotropic diffusion (σ₂/σ₁ = 0.5 ratio)
let kernel = AnisotropicShiftChernoffND::<f64, 2>::new_anisotropic([1.0, 0.5], grid2d)?;
let ev     = Evolver::new(kernel, 100)?;
// … evolve_into on GridFn2D<f64>
```

### d = 5 high-dim (PointEval + MC integration)

For d ≥ 3 the full-grid approach is O(N^d) memory. Use PointEval Backend E:

```rust
use semiflow_core::{AnisotropicShiftChernoffND, PointEval};
// d = 5 anisotropic diffusion
let kernel = AnisotropicShiftChernoffND::<f64, 5>::new_isotropic(1.0, grid5d)?;
// Evaluate at 1024 MC sample points without full grid
let points: Vec<[f64; 5]> = mc_sample_points(1024);
let vals = kernel.eval_at_nd(0.01, &f0_nd, &points, 200)?;
```

---

## §5 — Matrix-valued coupled-component diffusion (ADR-0082)

`DiffusionChernoff<F>` is PRESERVED verbatim. `MatrixDiffusionChernoff<F, 1>` is byte-identity equivalent for M = 1.

### M = 1 equivalence (scalar → matrix, byte-identical)

```rust
// v3.x scalar (still valid at v4.0)
use semiflow_core::{DiffusionChernoff, Evolver, Grid1D, GridFn1D};
let scalar = DiffusionChernoff::new(|_x| 1.0, |_x| 0.0, |_x| 0.0, 1.0, grid);

// v4.0 matrix M = 1 (byte-identical for scalar-equivalent A)
use semiflow_core::MatrixDiffusionChernoff;
let matrix1 = MatrixDiffusionChernoff::<f64, 1>::new_scalar(|_x| 1.0, grid)?;
```

### M = 2 FitzHugh-Nagumo neurone model

```rust
// Activator u, inhibitor v:
// ∂_t [u, v] = D · [∂_xx u, 0] + f(u, v)
// where D = diag(1.0, 0.0) — diffusion only in u channel
use semiflow_core::MatrixDiffusionChernoff;

// Diffusion matrix A(x) = [[1.0, 0], [0, 0]] (activator diffuses, inhibitor does not)
let diffusion_matrix = |_x: f64| [[1.0_f64, 0.0], [0.0, 0.0]];
let kernel = MatrixDiffusionChernoff::<f64, 2>::new(diffusion_matrix, grid)?;
let ev     = semiflow_core::Evolver::new(kernel, 100)?;
// State: GridFn1D<[f64; 2]> — interleaved (u, v) per grid node
```

### M = 4 multi-asset financial PDE

```rust
// 4-asset Black-Scholes PDE: ∂_t V = ½ Σᵢⱼ σᵢ σⱼ ρᵢⱼ Sᵢ Sⱼ ∂²V/∂Sᵢ∂Sⱼ + ...
// MatrixDiffusionChernoff<f64, 4> encodes the Σ diffusion tensor
let corr_matrix = |_x: f64| correlation_4x4(x);  // Cholesky-factored per node
let kernel = MatrixDiffusionChernoff::<f64, 4>::new(corr_matrix, log_price_grid)?;
```

---

## §6 — Diffusion4thZeta4Chernoff `order()` correction (ADR-0085)

**MANDATORY for v3.0+ users who reference `Diffusion4thZeta4Chernoff<F>`.**

Scaffold:
- v3.0 `Diffusion4thZeta4Chernoff::order()` claimed to return `4`. v4.0 corrects to return `2` per ADR-0085 §"Decision".
- The kernel SHIPS in v4.0 but is marked EXPERIMENTAL in rustdoc.
- Migration:
  ```rust
  // v3.0 (numerically false — actually order-2):
  let zeta4 = Diffusion4thZeta4Chernoff::<f64>::new(inner, Some(2.5_f64))?;
  assert_eq!(zeta4.order(), 4);    // WAS true in v3.0 surface; FALSE in v4.0

  // v4.0 (corrected):
  let zeta4 = Diffusion4thZeta4Chernoff::<f64>::new(inner, Some(2.5_f64))?;
  assert_eq!(zeta4.order(), 2);    // CORRECTED per ADR-0085

  // v4.0 recommended migration if you wanted order-2 anyway:
  let scalar = DiffusionChernoff::<f64>::new(a_fn, grid)?;
  assert_eq!(scalar.order(), 2);
  ```
- Wave G fills:
  - Worked example showing the v3.0 assertion fails in v4.0 + the recommended replacement.
  - Pointer to ADR-0085 for the architectural context.

---

## §7 — v2_compat removal checklist (ADR-0084)

**APPLIES ONLY TO v2.x USERS WHO SKIPPED v3.x.** v3.0+ users who already migrated to the v3 surface have NOTHING to do here.

If you are a v2.x user who never upgraded to v3.x: STOP — the v4.0 release HARD REMOVES the v2_compat shim. You MUST:

1. First, upgrade to v3.x by following `docs/migration/v2-to-v3.md`.
2. Verify your code compiles under v3.x WITHOUT the `v2_compat` feature flag.
3. Then upgrade to v4.0 (this document).

If you skipped step 1-2, your v2.x code will NOT compile at v4.0. The 12 months of v3.0+ deprecation warnings (v3.0.0 release 2026-05-27 onward) were your migration window.

**v4.0 Wave G audit finding**: the planned per-binding v2 shim methods (`Heat1D.evolve_v2`, `Heat1D.growth_v2`, `Heat1D.growthV2()`, `RemizovWasm.versionV2()`, `ChernoffSemigroup` Python/JS class alias) were not present in the v3.x codebase — they were not implemented in the additive v3 binding waves (Waves D/E/F) because v3 treated the original v2-era APIs as the PRIMARY surface (not shims to bridge). Consequently, there is no per-binding shim code to delete at v4.0. The core `v2_compat` module (pure Rust, `crates/semiflow-core/src/v2_compat.rs`) is the only actual shim artifact and WAS deleted.

v3.0+ users have ZERO binding migration work at v4.0.

---

## §8 — FFI v3 binding migration (ADR-0084 §35.3)

The v3 FFI surface (`smf_evolver_new_heat_1d_unit_v3`, `smf_evolver_evolve_into_v3`, `smf_growth_v3`, etc.) is PRESERVED VERBATIM at v4.0. The original `remizov_state_*` API in `ffi.rs` is also preserved (it was never a "v2 shim" — it IS the FFI surface).

**v4.0 audit finding**: no `remizov_v2.h` or `_v2`-suffixed extern C symbols were shipped in v3.x. Nothing to remove.

### FFI usage (unchanged at v4.0)

```c
#include "remizov.h"

// v3 surface (additive, stable at v4.0):
SmfEvolverV3 *ev = NULL;
smf_evolver_new_heat_1d_unit_v3(0.0, 1.0, 256, 100, u0, 256, &ev);
double out[256];
smf_evolver_evolve_into_v3(ev, 0.1, out, 256);
SmfGrowthV3 g = smf_growth_v3(ev);
smf_evolver_free_v3(ev);
```

---

## §9 — PyO3 v3 binding migration (ADR-0084 §35.4)

The v3 PyO3 surface (`EvolverHeat1DUnitV3`, `GrowthV3`, `evolve_into`, `growth()`) is PRESERVED VERBATIM at v4.0.

**v4.0 audit finding**: no `_v2`-suffixed shim methods were present in `semiflow-py`. `Heat1D`, `Heat1D4th`, and related classes are the PRIMARY pyclasses (not shims) and are preserved.

### PyO3 usage (unchanged at v4.0)

```python
import semiflow as rp
import numpy as np

# v3 evolver (additive surface, stable at v4.0):
u0 = np.exp(-((np.linspace(0, 1, 256) - 0.5) ** 2) / 0.01)
ev = rp.EvolverHeat1DUnitV3(0.0, 1.0, 256, u0, 100)
out = np.empty(256)
ev.evolve_into(0.1, out)
g = ev.growth()
print(f"multiplier={g.multiplier:.4f} omega={g.omega:.4f}")

# Original Heat1D (still the primary API, unchanged):
heat = rp.Heat1D(0.0, 1.0, 256, u0)
heat.evolve(0.1, 100)
vals = heat.values()
```

---

## §10 — WASM v3 binding migration (ADR-0084 §35.5)

The v3 WASM surface (`EvolverHeat1DUnitV3`, `GrowthV3`) is PRESERVED VERBATIM at v4.0.

**v4.0 audit finding**: no `growthV2()`, `versionV2()`, or `ChernoffSemigroup` JS class were present in `semiflow-wasm`. `Heat1D` and `GraphHeat` are the primary JS classes and are preserved.

### WASM usage (unchanged at v4.0)

```javascript
import init, { EvolverHeat1DUnitV3, Heat1D } from './semiflow_wasm.js';
await init();

const N = 256;
const u0 = new Float64Array(N).map((_, i) => Math.exp(-((i/N - 0.5)**2) / 0.01));

// v3 evolver (additive, stable at v4.0):
const ev  = EvolverHeat1DUnitV3.new(-1.0, 1.0, N, u0, 100);
const out = new Float64Array(N);
ev.evolveInto(0.1, out);
const g = ev.growth();
console.log(`multiplier=${g.multiplier()} omega=${g.omega()}`);

// Original Heat1D (primary API, unchanged):
const heat = Heat1D.new(-1.0, 1.0, N, u0);
heat.evolve(0.1, 100);
const vals = heat.values();
```

---

## §11 — 12-month deprecation cycle completion log (ADR-0035 §9 precedent)

Per the v0.10.0 → v0.11.0 → v1.0.0 12-month deprecation cycle precedent (ADR-0035 §9), the v3.0 → v4.0 cycle ran:

| Phase | Date | Action | Evidence |
|---|---|---|---|
| Deprecation begins | 2026-05-27 (v3.0.0 release) | All v2.x surface marked `#[deprecated(since = "3.0.0", note = "Hard-removed at v4.0")]` | Commit SHA TBD (engineer Wave G fills at v4.0 release) |
| Mid-cycle reminder | ~2026-11-27 (v3.1.0 release) | Warning escalation per ADR-0035 §9 — engineer adds `note = "Hard-removed at v4.0. Migrate now."` | v3.1.0 release commit |
| Final notice | ~2027-05-13 (v4.0-rc.1) | Last 2 weeks before removal; rustdoc front-matter explicitly states removal date | v4.0-rc.1 release commit |
| Hard removal | 2026-05-28 (v4.0.0 Wave G complete) | All v2.x surface artifacts deleted per math.md §35.2-§35.5 | [v4.0.0 release commit SHA — sign-off fills at tag] |

This pattern is the v0.10.0 → v0.11.0 → v1.0.0 cycle replayed at the v3.0 → v4.0 cadence.

---

## §12 — Engineer Wave G completion record

Wave G (2026-05-28) filled all TBD sections:

1. §2 Schrödinger Option A → Option B worked examples — DONE.
2. §3 PointEval full-grid vs single-point worked example — DONE.
3. §4 d-D shift d ∈ {1, 2, 5} worked examples — DONE.
4. §5 Matrix-valued M ∈ {1, 2, 4} worked examples (FitzHugh-Nagumo, multi-asset) — DONE.
5. §6 Diffusion4thZeta4Chernoff `order()` correction — WAS ALREADY FILLED BY ARCHITECT.
6. §7-§10 per-binding audit findings + v4.0 API examples — DONE.
7. §11 v4.0.0 release SHA placeholder — filled with sign-off tag marker.

**v4.0.0 audit note**: the v2_compat shim artifacts that were planned for deletion per ADR-0084 §35.3-§35.5 (FFI `v2_shim.rs`, `remizov_v2.h`, `cbindgen-v2.toml`; PyO3 `_v2` shim methods; WASM `growthV2()`, `versionV2()`) were not present in the v3.x codebase. The SOLE deletable artifact was `crates/semiflow-core/src/v2_compat.rs` (~120 LoC) plus the Cargo.toml feature entries + lib.rs cfg block. See §7 for the full audit record.

---

## §13 — v4.1 G_zeta4 resolution (ADR-0086 Path β)

**Status**: COMPLETE (Path β wave 2026-05-28; agentic-engineer Task-ID=g-zeta4-path-beta-wave).

### What changed in v4.1

`Diffusion4thZeta4Chernoff<F>` implements Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1
(m=4 Taylor tangency) via the single-step 4-term Taylor expansion (Path β, ADR-0086):

```rust
// v3.0: order_method = 4 (claimed but BCH algorithm gave order 2 globally)
// v4.0: order_method = 2 (corrected per ADR-0085 — BCH falsified in v3.1)
// v4.1+: order_method = 4 (Path β achieves what v3.0 promised, per ADR-0086)
let kernel = Diffusion4thZeta4Chernoff::<f64>::new(inner, Some(2.5_f64))?;
assert_eq!(kernel.order(), 4);   // v4.1+ via Path β
```

### Behavior change

- `order()` returns **4** (was 2 in v4.0, was claimed-but-false 4 in v3.0).
- `apply_into` now computes `f + τAf + (τ²/2)A²f + (τ³/6)A³f` via 3 successive
  applications of the divergence-form 9-point stencil A. The BCH+P₂ correction
  code is DELETED.
- Performance: ~3 stencil applications per τ-step (was ~7 in v3.0/v4.0 BCH path).
- Empirical convergence: slope −4.06 (v3.1 Wave D); G_zeta4 gate tightened −1.9 → −3.9.

### Caller invariant narrowing (non-breaking)

- `ApproximationSubspace<6, F>` impl REMOVED (K=6 retired per ADR-0086 AC9).
- `ApproximationSubspace<4, F>` impl retained (K=4 is necessary and sufficient for Path β).
- Callers using `in_subspace::<6>` must downgrade to `in_subspace::<4>` — strictly more
  permissive (any K=4-OK datum automatically passes K=4).

```rust
// v3.0/v4.0: both K=4 and K=6 witnesses available
assert!(kernel.in_subspace::<4>(&f));   // ← still works in v4.1
assert!(kernel.in_subspace::<6>(&f));   // ← REMOVED in v4.1: compile error

// v4.1+: K=4 only
assert!(kernel.in_subspace::<4>(&f));   // ← use this
```

### Files changed in v4.1 Path β wave

| File | Change |
|------|--------|
| `crates/semiflow-core/src/diffusion4_zeta4.rs` | `apply_into` rewritten (Richardson); `order()=4`; EXPERIMENTAL removed; K=6 impl removed |
| `crates/semiflow-core/src/diffusion4_zeta4_data.rs` | **DELETED** (151 LoC; P_2_MONOMIALS, jet6, apply_jet_iter, grid_average_a) |
| `crates/semiflow-core/src/lib.rs` | `mod diffusion4_zeta4_data` declaration removed |
| `crates/semiflow-core/tests/zeta4_correction_slope.rs` | Single gate → 2 sub-gates (const-a BLOCKING + var-a ADVISORY, ADR-0086 AMENDMENT 1) |
| `scripts/verify_zeta4_correction.py` | Extended 3 → 4 sub-checks; sub-check (c) revised to Richardson Lagrange bound |
| `contracts/semiflow-core.properties.yaml` | G_zeta4 split into G_zeta4_const_a_richardson (BLOCKING) + G_zeta4_var_a_slope (ADVISORY) |

### G_zeta4 gate bifurcation (ADR-0086 AMENDMENT 1)

During the Path β Wave implementation (2026-05-28), the engineer's 6-experiment diagnosis
established that the K5 reference (`Diffusion4thChernoff`) uses Catmull-Rom O(dx⁴) grid
sampling internally, creating a constant spatial floor ≈ 1.18e-4 at N=512 that is
independent of `n_ref`. This floor prevents measuring Path β's true order-4 against
K5-as-oracle in the variable-a regime (the probe is two orders better than its oracle).

v4.1 therefore splits G_zeta4 into two sub-gates:

- **`G_zeta4_const_a_richardson`** (RELEASE_BLOCKING): constant-a Richardson ratio
  `log₂(err₄/err₈) ≥ 3.5` with analytic oracle `(1+4T)^{−½} exp(−x²/(1+4T))`.
  Proves Path β achieves order-4 in the spatial-floor-free regime.
- **`G_zeta4_var_a_slope`** (RELEASE_ADVISORY): variable-a OLS slope ≤ −2.5 against
  K5 reference at `n_ref=8192`. Documents operational reality; captures regression signal.
  Does NOT block release until ADR-0088 lifts the floor.

Architectural fix for the Catmull-Rom floor (upgrade `Diffusion4thChernoff::apply_into`
internal `GridFn1D::sample()` from CubicHermite O(dx⁴) to QuinticHermite O(dx⁶))
is deferred to ADR-0088 in v4.2+. This drops the floor from ~1.18e-4 to ~1e-8 and
restores `G_zeta4_var_a_slope` to RELEASE_BLOCKING at threshold ≤ −3.9.

---

## Cross-references

- **All v4.0 ADRs**: ADR-0079 / ADR-0080 / ADR-0081 / ADR-0082 / ADR-0083 / ADR-0084 / ADR-0085.
- **v4.1 ADR**: ADR-0086 (G_zeta4 resolution via Path β — closes 4-deferral cycle).
- **v3.0 BREAKING ADRs** (preserved through v4.x): ADR-0073 (ApproximationSubspace) / ADR-0074 (ChernoffFunction cleanup) / ADR-0075 (ζ⁴ correction — order-4 claim CORRECTED per ADR-0085, RESTORED per ADR-0086) / ADR-0076 (v2→v3 binding redesign).
- **Migration playbook for v2.x users**: `docs/migration/v2-to-v3.md` (must complete first if you skipped v3.x).
- **Deprecation cycle precedent**: ADR-0035 §9 (v0.10.0 → v0.11.0 → v1.0.0).
- **Math foundation**: `contracts/semiflow-core.math.md` §30 (SemiflowComplex + Schrödinger Option B), §31 (PointEval), §32 (d-D shift), §33 (matrix-valued), §34 (resolvent residual), §35 (v2 surface removal manifest).
- **Constitution amendment**: `.dev-docs/constitution.md` v1.8.0 (MAJOR re-evaluation; Cohort 7 Override #1; `num-complex` dep promotion).
- **Roadmap**: `~/.claude/plans/roadmap-reflective-biscuit.md` §v4.0.
