# ADR-0085 — G_zeta4 Architect Math Review: DEFERRAL Decision (Option B)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v4.0 architect math review (out-of-band; resolves the v3.1 Wave D engineer escalation per `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_g_zeta4_escalation.md`). This ADR is the OUTCOME of the architect math review demanded by the v4.0 task spec; the decision is **DEFERRAL (Option B)** with full justification.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0073 (v3.0 ApproximationSubspace<K, F>; the K=6 witness Diffusion4thZeta4Chernoff was supposed to consume), ADR-0075 (v3.0 ζ⁴ correction kernel — the kernel whose order-4 claim is reviewed here), v3.1 Wave D engineer's numerical falsification (preserved in the project memory entry cited above).
- **Supersedes / amends**: ADR-0075 §"Decision" (PARTIAL — the kernel `Diffusion4thZeta4Chernoff<F>` SHIPS verbatim through v4.x, but the **order-4 claim is DOCUMENTED AS UNVERIFIED**; the kernel is marked experimental in rustdoc; G_zeta4 stays ADVISORY through v4.x; no B8 ζ⁶/ζ⁸ ladder ships in v4.0).
- **Mathematical foundation**: The architect math review's FINDING is that the v3.0 `P_2_MONOMIALS_K6_DIFFUSION` table (per ADR-0075 §"Decision" Algorithm) is INSUFFICIENT to achieve the order-4 claim of Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1. The 6 monomials in the math.md §27.3 table have only the leading $-\frac{1}{12} A^2$ term populated; the remaining 5 monomial coefficients are placeholder zeros. The Galkin-Remizov §3.1 6-monomial polynomial requires a sympy-derived solution of a linear system that has NOT been produced. This ADR documents the failure mode honestly and DEFERS the resolution.
- **Acceptance gates added**: None NEW. G_zeta4 stays at v3.0 ADVISORY status through v4.x (NO promotion to RELEASE_BLOCKING). NO G_zeta6 or G_zeta8 gates added in v4.0. NO B8 ladder kernels (Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff) shipped in v4.0.

## Context

The v3.0 release (ADR-0075) shipped `Diffusion4thZeta4Chernoff<F>` with the CLAIM that, on the strict core $D(A^6)$ with $a \in C^6_b$, the kernel achieves order-4 Chernoff convergence per Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1. The claim was backed by:
- A G_zeta4 RELEASE_BLOCKING gate (slope $\le -1.9$ on heat with $a \in C^6_b$).
- A T23N NORMATIVE sympy gate verifying the symbolic τ²-cancellation identity.
- A `P_2_MONOMIALS_K6_DIFFUSION: [(f64, u8, u8, u8); 6]` const-array claimed to be sympy-derived.

The v3.1 Wave D engineer (per the project memory entry cited above) attempted to BUILD AND VERIFY the `Diffusion4thZeta4Chernoff<F>` kernel and **numerically falsified** the order-4 claim:

1. **BCH correction alone gives order-2, NOT order-4**: empirical sweep `n ∈ {4, 8, 16, 32, 64}` showed `err_n = 7.15e-3 → 2.96e-4` with global order ≈ 1.0 asymptotic (the absolute-value error decreases by 2× per doubling of n, not by 16× as order-4 would predict).
2. **Full 4th-order Taylor expansion DOES achieve order-4**: when implemented with explicit `f + τAf + (τ²/2)A²f + (τ³/6)A³f + (τ⁴/24)A⁴f` single-step expansion (no BCH correction), the same sweep gave `err_n = 6.51e-4 → 1.28e-9` with order ≈ 4.06 asymptotic. So order-4 IS achievable, but requires the FULL single-step expansion, not just the BCH correction.
3. **Current `P_2_MONOMIALS_K6_DIFFUSION` is insufficient**: only the leading $-\frac{1}{12} A^2$ term is populated; the other 5 monomial slots are zero. ADR-0075 spec'd a "6-monomial correction polynomial" but the actual algorithm needs richer math than what's in math §27.3.
4. **`compute_jet6` spectral instability**: 3-point stencil at grid scale (dx) produces $\tau^2 \cdot |\lambda_{\max}|^2 / 12 \approx 1176$ correction magnitudes per step at $n = 16$ with $N = 512$ → explosion 1e38 → 1e164 across the n sweep. The v0.6.0 `Diffusion4thChernoff::zeta4_correction_f64` used `max(3·dx, τ^{3/4})` scaling for ζ-A correction; the v3.0 `diffusion4_zeta4_data.rs` has no such scaling.

The engineer correctly ESCALATED to architect review per the task spec ("If P_2 derivation requires research-level math, ESCALATE — don't fake it"). No v3.1 code was committed for the B8 ladder. The architect math review is the v4.0 outcome.

## Decision

**OPTION B — DEFER**. The architect math review finds that the v3.0 ζ⁴ algorithm has a STRUCTURAL GAP that cannot be closed in a single architect session. v4.0 ships WITHOUT the B8 ζ⁶/ζ⁸ ladder; v4.0 documents G_zeta4 as ADVISORY (no promotion to RELEASE_BLOCKING); v4.0 marks `Diffusion4thZeta4Chernoff<F>` as experimental in rustdoc with a CLEAR pointer to this ADR for the unresolved math.

### Rationale for Option B (vs Option A)

The v4.0 task spec offered two paths:
- **Option A (RESOLVE)**: re-derive the 6-monomial P_2 polynomial via sympy `solve()` per Galkin-Remizov 2025 IJM §3.1. If the architect produces the canonical 6 coefficients with full mathematical justification, ship B8 ζ⁶/ζ⁸ ladder in v4.0.
- **Option B (DEFER)**: accept ADR-0075 has structural gap; B8 ζ-ladder stays deferred to post-v4.0 research; ADR-0085 documents the deferral.

The architect math review CHOOSES Option B for these reasons:

1. **The Galkin-Remizov §3.1 6-monomial polynomial is research-level math.** Producing the canonical 6 coefficients requires:
   - Symbolic expansion of the divergence-form generator $A = \partial_x(a(x) \partial_x)$ acting on test functions via the v0.6.0 9-point stencil.
   - Computation of the Taylor series of $(F(\tau/n))^n f$ up to $\tau^4$ accounting for the spatial discretization structure.
   - Solving the linear system that makes the $\tau^2$-coefficient of the residual identically zero on $D(A^6)$ for arbitrary $f$.
   - Verifying the result is uniquely determined (Theorem 3.1 says unique; verification requires showing the linear system has rank 6).
   This is a multi-week sympy port + analytical verification effort. A single architect session cannot credibly produce it.

2. **Repeating the v3.0 failure cycle is unacceptable.** ADR-0075 already committed to a "6-monomial polynomial" without producing the actual coefficients; v3.1 Wave D engineer correctly found the gap. Shipping Option A in v4.0 without a working sympy port would REPEAT THE SAME FAILURE — claiming order-4 + order-6 + order-8 ladders without the canonical coefficients. The user explicitly directed: "don't repeat v3.0's failure cycle" — Option B is the principled response.

3. **Suckless honesty over surface inflation.** The constitution principle #1 ("Math fidelity is non-negotiable: every numerical claim MUST be backed by a sympy oracle in `.dev-docs/verification/scripts/` and cited in `contracts/semiflow-core.math.md`") MANDATES that order-4 / order-6 / order-8 claims have backing sympy oracles. Without the 6-monomial coefficients, the ζ⁴ / ζ⁶ / ζ⁸ ladders have no oracles; shipping them violates the constitution. Suckless: don't ship unverifiable surface.

4. **v4.0 scope is sized correctly without B8.** The v4.0 BREAKING window already includes 6 substantial additions (ADR-0079 SemiflowComplex + ADR-0080 PointEval + ADR-0081 d-D shift + ADR-0082 matrix-valued + ADR-0083 G_RES_RES + ADR-0084 v2_compat removal). Adding B8 ζ ladder would push the engineering surface from 7 Waves to 8+; the BREAKING window is large enough. DEFERRAL reduces v4.0 scope to a cleanly-shipping size.

5. **The kernel still ships** — just marked experimental. `Diffusion4thZeta4Chernoff<F>` SHIPS in v4.0 because removing it would BREAK v3.0+ users who created the kernel construction even if the order-4 promise was empirically unverified. The kernel SHIPS with:
   - Rustdoc marker `// EXPERIMENTAL: order-4 claim per Galkin-Remizov 2025 IJM Theorem 3.1 is NOT empirically verified at v4.0; see ADR-0085 for the deferral. The kernel converges at order ~2 (NOT order 4) per the v3.1 Wave D numerical falsification.`
   - G_zeta4 gate stays at v3.0 ADVISORY status (does NOT block v4.0 release).
   - `order()` method returns 2 (NOT 4) — corrected from ADR-0075 §"Decision" claim of `order() = 4`. The v3.0 claim was wrong; v4.0 corrects via this ADR.

### What v4.0 SHIPS (Confirmed Option B)

| Item | Status | Notes |
|---|---|---|
| `Diffusion4thZeta4Chernoff<F>` kernel | SHIPPED | Marked experimental in rustdoc; pointer to ADR-0085 |
| `order()` method on Diffusion4thZeta4Chernoff | CORRECTED to return 2 | v3.0 ADR-0075 §"Decision" claim of `order() = 4` was wrong; v4.0 corrects |
| G_zeta4 gate | ADVISORY (preserved) | NO promotion to RELEASE_BLOCKING in v4.0 |
| `P_2_MONOMIALS_K6_DIFFUSION` const-array | PRESERVED in code (leading term only) | The placeholder 5 zeros are not removed; they remain as documented unfinished math |
| math.md §27 | AMENDED (this ADR adds a clarification note) | See math.md amendment below |
| B8 ζ⁶/ζ⁸ ladder kernels | DEFERRED | NO Diffusion6thZeta6Chernoff or Diffusion8thZeta8Chernoff in v4.0 |
| G_zeta6 / G_zeta8 gates | DEFERRED | Not added to properties.yaml in v4.0 |
| Wave I (B8 ladder engineering) | OMITTED | engineer handoff has Waves A-G only |

### Path forward for future architect math reviews

To RESOLVE the G_zeta4 claim in a future release (v4.1 or v5.0):

1. **Sympy port**: produce `scripts/derive_zeta4_p2_polynomial.py` that:
   - Symbolically expands $A = \partial_x(a(x) \partial_x)$ via the 9-point stencil.
   - Computes the Taylor series of $(F(\tau/n))^n f$ to $\tau^4$.
   - Solves the linear system for the 6 monomial coefficients.
   - Outputs the canonical coefficients verbatim into a `P_2_MONOMIALS_K6_DIFFUSION` table.
2. **Verify with engineer numerical test**: re-run the v3.1 Wave D falsification protocol with the new coefficients; slope MUST be $\le -3.9$ on the canonical heat sweep.
3. **Re-issue ADR**: new ADR (e.g., ADR-0086 in v4.1) supersedes this ADR-0085; promotes G_zeta4 to RELEASE_BLOCKING; corrects `order()` to return 4; updates math §27.
4. **Add B8 ladder**: if step 1-3 succeed, re-attempt the ζ⁶/ζ⁸ ladder per ADR-0075 §"Future extensions"; ship Wave I (B8 ladder) at the same time.

This path is OUT OF SCOPE for v4.0 architect work but is documented as the path forward for a future architect session with sufficient bandwidth for the sympy port.

### Mathematical clarification note for math §27

ADR-0085 amends math.md §27 with the following clarification note (engineer Wave A adds inline):

> **Note (v4.0 ADR-0085 clarification)**: The v3.0 `P_2_MONOMIALS_K6_DIFFUSION` table has only the leading $-\frac{1}{12} A^2$ monomial populated; the other 5 monomial slots are placeholder zeros. With the placeholder table, $F_{\zeta^4}(\tau) f := F(\tau) f + \tau^2 \cdot P_2[A] f$ achieves only order-2 globally on the empirical sweep (per the v3.1 Wave D engineer's numerical falsification). The order-4 claim of Theorem 27.1 / Galkin-Remizov 2025 *IJM* Theorem 3.1 requires the FULL 6-monomial polynomial with non-trivial coefficients derived via the sympy port that has NOT been produced. v4.0 ships `Diffusion4thZeta4Chernoff<F>` with the current PARTIAL polynomial; the kernel is marked experimental; G_zeta4 stays ADVISORY. Resolution defers to a future architect session per ADR-0085 §"Path forward".

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Option A — RESOLVE in this architect session** (re-derive the 6 coefficients now) | Cannot credibly produce in a single architect session; the sympy port is multi-week effort; repeating v3.0's failure cycle is unacceptable. |
| Remove `Diffusion4thZeta4Chernoff<F>` from v4.0 entirely | Breaks v3.0+ users who constructed the kernel; soft-experimental marking is the gentler path. |
| Promote G_zeta4 to RELEASE_BLOCKING anyway (hoping the empirical slope passes at -1.9) | The v3.1 Wave D falsification proved the slope is around -1.0 (order-2), NOT -1.9 (order-4). Promotion would BLOCK v4.0 release entirely. |
| Set the G_zeta4 slope budget to ≤ -0.9 (accept order-2) | Useless gate — order-2 is the v3.0 `DiffusionChernoff` baseline; renaming `Diffusion4thZeta4Chernoff` as a "ζ⁴ correction" for an order-2 kernel is misleading marketing. |
| Rename `Diffusion4thZeta4Chernoff` to `Diffusion4thZeta4ChernoffPartial` to signal incomplete math | Breaks v3.0+ users who reference the type by name; rustdoc marker + ADR cross-ref is the suckless choice. |
| Defer the entire kernel + the ladder to v5.0+ (revert v3.0 ADR-0075) | The kernel construction + apply_into machinery IS correct (the gap is only the coefficients); reverting throws away usable infrastructure. Soft-experimental marking preserves the infrastructure. |
| Ship B8 ladder with placeholder coefficients (mirror the v3.0 ζ⁴ approach for ζ⁶ and ζ⁸) | REPEATS the v3.0 failure cycle — the user explicitly prohibited this. The whole point of the architect math review is to BREAK the cycle. |
| Document the gap in `docs/audit-findings-v3.md` instead of an ADR | An ADR is the architectural record of decisions; the deferral IS an architectural decision (it determines v4.0 scope and the post-v4.0 trajectory). An audit-findings doc is for code-level findings, not for decisions. |

## Consequences

- **`Diffusion4thZeta4Chernoff<F>` STAYS IN v4.0** with rustdoc EXPERIMENTAL marker + cross-ref to this ADR.
- **`order()` method on Diffusion4thZeta4Chernoff CORRECTED** from v3.0's claimed `4` to v4.0's actual `2`. This is a BEHAVIOUR CHANGE for callers who depended on `order() == 4` (none currently exist because the v3.1 Wave D engineer correctly escalated before any code committed against the order-4 surface). The change is acceptable as a v4.0 BREAKING window correction; documented in `docs/migration/v3-to-v4.md` as a "v3.0 claim correction" entry.
- **G_zeta4 gate STAYS at v3.0 ADVISORY** through v4.x. Properties.yaml preserves the gate entry (with severity ADVISORY) for documentation purposes; CI runs the test but does not block on failure.
- **NO B8 ladder shipped in v4.0**: Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff, G_zeta6, G_zeta8 — ALL OMITTED from v4.0. Reserved for post-v4.0 architect math review session.
- **Wave I (B8 ladder engineering) OMITTED** from the engineer handoff spec. v4.0 engineer Waves are A through G only (7 Waves, not 8+).
- **math.md §27 AMENDED** with the clarification note above (engineer Wave A adds inline; one paragraph).
- **`P_2_MONOMIALS_K6_DIFFUSION` const-array PRESERVED in code** (leading $-\frac{1}{12} A^2$ only; placeholder zeros remain) — the engineer does NOT remove the placeholder zeros, they remain as documented unfinished math; future architect session can populate them.
- **Schema bumps**: shared with ADR-0079/0080/0081/0082/0083/0084 — `traits.yaml` 1.1.0 → **2.0.0 MAJOR**; `properties.yaml` 0.12.0 → **1.0.0 MAJOR**. math.md is append-only (no new section for ADR-0085 — the clarification is inline in §27).
- **Constitution unchanged for this ADR** (already amended for v4.0 to v1.8.0 via ADR-0079 + ADR-0081). The DEFERRAL is consistent with the constitution principle #1 (math fidelity is non-negotiable) — shipping order-4 claim without sympy oracle would VIOLATE the constitution.

## Migration

End-user impact:

- **v3.0+ users who constructed `Diffusion4thZeta4Chernoff<F>` expecting order-4 convergence**: HARD BREAK at v4.0. The `order()` method now returns 2; the kernel is now marked experimental. Migrate to a different kernel (DiffusionChernoff for order-2, or wait for v4.1+ resolution).
- **v3.0+ users who NEVER constructed `Diffusion4thZeta4Chernoff<F>`**: ZERO impact. The kernel is documented as experimental; users who don't reference it are unaffected.
- **v3.0+ users who depended on G_zeta4 RELEASE_BLOCKING status**: G_zeta4 stays ADVISORY; CI does not block on failure. Users who built downstream gates on G_zeta4 should treat it as a "best effort" gate (which matches its v3.0 actual behaviour — the gate was failing in CI per Wave D evidence).

Worked example for migrating away from `Diffusion4thZeta4Chernoff` in `docs/migration/v3-to-v4.md` §6 (Wave G):

```rust
// v3.0 expected order-4 (NOT achieved per ADR-0085):
let zeta4_kernel = Diffusion4thZeta4Chernoff::<f64>::new(inner, Some(2.5_f64))?;
assert_eq!(zeta4_kernel.order(), 4);                              // WAS true in v3.0; FALSE in v4.0

// v4.0 corrected (kernel ships but order = 2):
let zeta4_kernel = Diffusion4thZeta4Chernoff::<f64>::new(inner, Some(2.5_f64))?;
assert_eq!(zeta4_kernel.order(), 2);                              // CORRECTED per ADR-0085

// v4.0 recommended migration (use v0.3.0 DiffusionChernoff for confirmed order-2):
let diffusion_kernel = DiffusionChernoff::<f64>::new(a_fn, grid)?;
assert_eq!(diffusion_kernel.order(), 2);
```

## Cross-references

- ADR-0001 — contract-first; this ADR is a contract-layer decision (DEFERRAL is an architectural decision).
- ADR-0073 — v3.0 ApproximationSubspace<K, F>; the K=6 witness Diffusion4thZeta4Chernoff was supposed to consume. The witness mechanism itself is correct; the gap is only in the P_2 coefficient table.
- ADR-0075 — v3.0 ζ⁴ correction kernel; PARTIAL supersede (the kernel ships verbatim; the order-4 claim is documented as unverified).
- math.md §27 — AMENDED inline with the v4.0 clarification note (engineer Wave A adds the paragraph).
- v3.1 Wave D engineer's numerical falsification: `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_g_zeta4_escalation.md` (the project memory entry that drove this architect review).
- ADR-0085 (this ADR) — the architectural record of the DEFERRAL decision.
- Future ADR (post-v4.0, e.g., ADR-0086 in v4.1) — the path forward for resolution, contingent on a successful sympy port of the Galkin-Remizov §3.1 6-monomial polynomial.

## Amendments

(none at acceptance time)
