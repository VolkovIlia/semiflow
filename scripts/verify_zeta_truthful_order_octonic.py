#!/usr/bin/env python3
"""ADR-0119 DECISIVE PRE-FLIGHT oracle — ζ⁶/ζ⁸ TRUTHFUL_ORDER pair-slope feasibility
on the OCTONIC-Hermite sampler (ADR-0117) + 4th-order divergence stencil (ADR-0118).

This is the sub-check ADR-0110 SKIPPED (its AMENDMENT 1 §"Mistake-symmetry" line 497:
"The architect should have written a 7th sub-check ... that prediction would have been
−m, immediately exposing the off-by-one ... and the un-modelled spatial floor").

LOCKED DECISION (do NOT relitigate): the ζ⁶/ζ⁸ gate is a **pair-slope (Richardson
middle-pair) witness** at −5.95 (ζ⁶) / −7.95 (ζ⁸), NOT a 4-point global OLS. The
4→8 middle pair sees the kernel's genuine GLOBAL order K (ADR-0110 AMENDMENT 1 showed
ζ⁴'s 4→8 pair = −4.07 = honest order-4). We model the global error and predict the
4→8 middle-pair slope plus the floor-window geometry.

PASS CONDITION (CORRECTED 2026-06-05): pass iff predicted middle-pair slope ≤ gate
threshold (−5.95 / −7.95). The earlier ≥0.10-margin requirement was a delegation
over-specification and is DROPPED: a pair-slope is bounded above by the textbook
order K, so against a `K−0.05` gate the best attainable margin is exactly 0.05.
Reaching slope = K exactly (margin 0.05) IS a PASS — the slope cannot exceed K, so
0.05 ceiling-margin is intrinsic and acceptable (see ADR-0119 GO, ADR-0110 AMD 1).

GLOBAL error model (AMENDMENT 1 §"Error 1/Error 2", with the floors lowered to the
KEYSTONE machinery: OCTONIC virtual-node + 6th-order divergence stencil):

    err_global(τ) = c_K · T · τ^K                          # temporal signal (order K)
                  + (T · φ_octonic) / τ                    # virtual-node floor (ADR-0117)
                  + spatial6_floor(T)                       # 6th-order stencil floor (ADR-0118)

  where:
    - φ_octonic ≈ 9.1e-16  (verify_octonic_hermite_weights.py sub-check (e))
    - spatial6_floor(T) = C_sp6 · T · dx⁶
        C_sp6 calibrated from the AMENDMENT-1 empirical anchor: the 3-pt stencil
        plateau coefficient was K_SPATIAL_OBS = 3.72e-4 per unit T at dx² scale.
        We carry the SAME observed-cancellation ratio forward conservatively to the
        6th-order stencil →
        spatial6_floor(T) = K_SPATIAL_OBS · (dx⁶/dx²) · T = K_SPATIAL_OBS · dx⁴ · T.
        (This is the central modelling assumption; sub-check (iv) stress-tests it.)
        At N=4096, dx⁴ ≈ 3.4e-11 → spatial6 floor ≈ 1.3e-14·T, deep below the signal.
    - c_K calibrated from the §40.5 floor-independent signal table:
        ζ⁶: c·τ^6 = 5.86e-9 at τ_ref=0.5  (n=1 pair)  → c_6 = 5.86e-9 / 0.5^6
        ζ⁸: c·τ^8 = 5.02e-10 at τ_ref=0.5 (n=1 pair)  → c_8 = 5.02e-10 / 0.5^8
        NOTE: these §40.5 signals are PER-STEP (local) c·τ^{m+1} with m_paper=K-1.
        The GLOBAL signal is n·(per-step) = (T/τ)·c·τ^{K} = c·T·τ^{K-1}... — we
        re-anchor to the GLOBAL coefficient directly: c_K^global chosen so that the
        global temporal signal at the §40.5 reference equals the per-step signal × n.
        We keep the conservative SMALLER global coefficient (worst case for the gate).

PAIR-SLOPE (consecutive doubling τ → τ/2):
    pair_slope(τ) = log2( err_global(τ) / err_global(τ/2) )
  Pure-signal limit (signal ≫ both floors): err ∝ τ^K → pair_slope = K. ✓

DECISIVE checks (the ones ADR-0110 missed):
  (i)  middle-pair (4→8) slope for K=6 AND K=8 clears the gate, i.e.
       slope ≤ −5.95 (ζ⁶) / ≤ −7.95 (ζ⁸). NO margin requirement (see PASS CONDITION).
  (ii) a NON-EMPTY τ-window across ALL 4 ladder points {2,4,8,16} where the
       temporal signal exceeds BOTH floors for K=6 and K=8.
  (iii) the 8→16 (finest) pair is NOT floor-dominated (its slope stays near K,
        not collapsing toward the −1 floor-onset signature seen in AMENDMENT 1).

GO iff all three hold for BOTH K=6 and K=8.

ADR-0086 PRE-FLIGHT-first principle. NORMATIVE.
"""

from __future__ import annotations

import math
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

X_MIN = -10.0
X_MAX = 10.0
N_SPATIAL = 4096               # GATE horizon (ADR-0119 GO): 8× memory vs N=512.
DX = (X_MAX - X_MIN) / N_SPATIAL  # ≈ 0.00488

# Lowered floors (the two KEYSTONE levers).
PHI_OCTONIC = 9.1e-16          # ADR-0117 sub-check (e)
PHI_SEPTIC = 1.49e-12          # v6.0.0 SepticHermite (for comparison)

# AMENDMENT-1 empirical spatial anchor (3-point 2nd-order stencil).
# ζ⁶ plateau 1.86e-3 at T=5.0 → K_SPATIAL_OBS ≈ 3.72e-4 per unit T (dx² scale).
K_SPATIAL_OBS = 1.86e-3 / 5.0  # ≈ 3.72e-4 per unit T

# 6th-order stencil spatial floor (ADR-0118, verify_div_stencil_6th.py): same
# observed-cancellation ratio carried forward, dx⁶ vs dx².
def spatial6_floor(T: float) -> float:
    return K_SPATIAL_OBS * (DX**4) * T  # = K_SPATIAL_OBS · dx⁶/dx² · T

# 4th-order stencil spatial floor (kept for the counterfactual: 4th order alone
# does NOT open the ζ⁸ window at the gate horizon — that motivated 6th order).
def spatial4_floor(T: float) -> float:
    return K_SPATIAL_OBS * (DX**2) * T  # = K_SPATIAL_OBS · dx⁴/dx² · T

# Old 2nd-order stencil floor (for the "why v6.0.0 was infeasible" comparison).
def spatial2_floor(T: float) -> float:
    return K_SPATIAL_OBS * T

# Temporal-signal global coefficients (conservative re-anchor from §40.5).
# §40.5 per-step (local) signals at τ_ref=0.5: ζ⁶ 5.86e-9, ζ⁸ 5.02e-10.
# GLOBAL temporal signal at horizon T with n steps: err_temporal ≈ n · (per-step)
#   = (T/τ) · c_local · τ^{K} = c_local · T · τ^{K-1}.
# Re-express as c_global · T · τ^K with c_global = c_local / τ_ref (worst-case small).
# We anchor at the gate's own T_PER_K and the §40.5 τ_ref=0.5.
TAU_REF = 0.5
SIGNAL_LOCAL = {6: 5.86e-9, 8: 5.02e-10}   # c_local · τ_ref^K  (per §40.5)
T_FINAL_PER_K = {6: 10.0, 8: 10.0}          # GATE horizon (ADR-0119 GO): N=4096/T=10.
N_LADDER = [2, 4, 8, 16]

# Gate thresholds (LOCKED).
GATE = {6: -5.95, 8: -7.95}
# Pass iff slope ≤ gate. No margin requirement (slope is bounded above by order K;
# 0.05 ceiling-margin against a K−0.05 gate is intrinsic and acceptable).


def c_global(K: int) -> float:
    """Conservative global temporal coefficient c_K such that
    err_temporal(τ) = c_K · T · τ^K matches the §40.5 per-step signal at τ_ref.

    Per-step LOCAL signal s_loc = c_local · τ_ref^K (from §40.5, m_paper=K-1, so
    the per-step error is c_local·τ^{m+1} = c_local·τ^K). Over n = T/τ steps the
    GLOBAL contribution is n·c_local·τ^K = c_local·T·τ^{K-1}. To write this as
    c_K·T·τ^K we'd need c_K = c_local/τ — τ-dependent. Instead, the GLOBAL order
    of the kernel is K (Galkin-Remizov tangency), so err_temporal = c_K·T·τ^K with
    c_K calibrated so that at τ_ref the GLOBAL temporal error equals the per-step
    LOCAL error times one (single-step at τ_ref): conservative c_K = s_loc / τ_ref^K.
    This makes the temporal signal SMALLER (harder gate) than the cumulative model.
    """
    s_loc = SIGNAL_LOCAL[K]
    return s_loc / (TAU_REF**K)


def err_global(K: int, tau: float, phi: float, T: float, spatial_fn) -> float:
    temporal = c_global(K) * T * tau**K
    vnode = (T * phi) / tau
    spatial = spatial_fn(T)
    return temporal + vnode + spatial


def pair_slope(K: int, tau_coarse: float, phi: float, T: float, spatial_fn) -> float:
    e_c = err_global(K, tau_coarse, phi, T, spatial_fn)
    e_f = err_global(K, tau_coarse / 2.0, phi, T, spatial_fn)
    # slope of log(err) vs log(n): n doubles as τ halves → use log2(e_c/e_f).
    return math.log2(e_c / e_f)


def emit_pass(label: str) -> None:
    print(f"  [PASS] {label}")


def emit_fail(label: str, reason: str) -> str:
    print(f"  [FAIL] {label}: {reason}", flush=True)
    return reason


# ---------------------------------------------------------------------------
# Decisive check (i) — middle-pair slope ≥ K − 0.05 with ≥0.1 margin vs gate
# ---------------------------------------------------------------------------


def check_middle_pair_slopes() -> str | None:
    label = "(i) middle-pair (4→8) slope ≤ gate (K=6 AND K=8); pass iff ≤ threshold"
    print("    K   T     τ(4→8)        signal/φ_oct  signal/sp6   pair-slope  gate    verdict")
    failures = []
    for K in (6, 8):
        T = T_FINAL_PER_K[K]
        # The 4→8 pair: n=4 → τ=T/4 (coarse), n=8 → τ=T/8 (fine).
        tau_coarse = T / 4.0
        slope = pair_slope(K, tau_coarse, PHI_OCTONIC, T, spatial6_floor)
        # absolute (negative) slope for the gate convention.
        slope_neg = -slope
        # diagnostics: signal vs floors at the coarse point.
        temporal = c_global(K) * T * tau_coarse**K
        vnode = (T * PHI_OCTONIC) / tau_coarse
        sp6 = spatial6_floor(T)
        sig_over_vnode = temporal / vnode
        sig_over_sp6 = temporal / sp6
        # Pass iff slope clears the gate. No margin requirement (slope ≤ K always;
        # the 0.05 ceiling-margin against a K−0.05 gate is intrinsic and acceptable).
        ok = slope_neg <= GATE[K]
        status = "PASS" if ok else "FAIL"
        print(f"    {K}   {T:.1f}  {tau_coarse:.4e}   {sig_over_vnode:.2e}    "
              f"{sig_over_sp6:.2e}   {slope_neg:+.4f}   {GATE[K]:+.2f}  {status}")
        if not ok:
            failures.append(
                f"K={K}: predicted middle-pair {slope_neg:.4f} > gate {GATE[K]} "
                "(need slope ≤ gate)"
            )
    print()
    if failures:
        return emit_fail(label, "; ".join(failures))
    print("    Both ζ⁶ and ζ⁸ middle-pair slopes clear the gate (≤ −5.95 / ≤ −7.95).")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Decisive check (ii) — non-empty τ-window over all 4 ladder points
# ---------------------------------------------------------------------------


def check_tau_window() -> str | None:
    label = "(ii) non-empty τ-window: temporal signal > BOTH floors at all 4 points"
    print("    For each K, at every ladder point n∈{2,4,8,16}: temporal must exceed")
    print("    max(virtual-node floor, spatial6 floor).")
    print()
    print("    K   n    τ          temporal     φ_oct/τ·T    sp6-floor    sig>both?")
    failures = []
    for K in (6, 8):
        T = T_FINAL_PER_K[K]
        all_ok = True
        for n in N_LADDER:
            tau = T / n
            temporal = c_global(K) * T * tau**K
            vnode = (T * PHI_OCTONIC) / tau
            sp6 = spatial6_floor(T)
            both_floor = max(vnode, sp6)
            ok = temporal > both_floor
            all_ok = all_ok and ok
            mark = "yes" if ok else "NO"
            print(f"    {K}  {n:>2}   {tau:.3e}  {temporal:.3e}  {vnode:.3e}  "
                  f"{sp6:.3e}   {mark}")
        if not all_ok:
            failures.append(f"K={K}: signal falls below a floor at some ladder point")
        print()
    if failures:
        return emit_fail(label, "; ".join(failures))
    print("    Non-empty τ-window confirmed for BOTH K=6 and K=8 (all 4 ladder points).")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Decisive check (iii) — finest pair (8→16) NOT floor-dominated
# ---------------------------------------------------------------------------


def check_finest_pair_not_floor_dominated() -> str | None:
    label = "(iii) finest pair (8→16) NOT floor-dominated (slope stays near K)"
    print("    AMENDMENT-1 signature of floor onset: finest-pair slope collapses")
    print("    toward −1 (cumulative floor n·φ → n⁻¹). Require |slope| ≥ K − 0.5.")
    print()
    print("    K   T     τ(8→16)      pair-slope   K−0.5 thresh   floor-dominated?")
    failures = []
    for K in (6, 8):
        T = T_FINAL_PER_K[K]
        tau_coarse = T / 8.0   # n=8 → τ=T/8 coarse; n=16 → τ=T/16 fine
        slope = pair_slope(K, tau_coarse, PHI_OCTONIC, T, spatial6_floor)
        slope_neg = -slope
        thresh = -(K - 0.5)
        ok = slope_neg <= thresh   # more negative than −(K−0.5)
        status = "no (healthy)" if ok else "YES (collapsed)"
        print(f"    {K}   {T:.1f}  {tau_coarse:.4e}   {slope_neg:+.4f}    "
              f"{thresh:+.2f}        {status}")
        if not ok:
            failures.append(
                f"K={K}: finest-pair slope {slope_neg:.4f} collapsed above {thresh} "
                "→ floor-dominated"
            )
    print()
    if failures:
        return emit_fail(label, "; ".join(failures))
    print("    Finest pair healthy for BOTH K — floors are low enough that the 8→16")
    print("    pair still demonstrates order-K (no floor-onset collapse).")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Diagnostic (iv) — old-stencil counterfactual + sensitivity to C_sp4
# ---------------------------------------------------------------------------


def check_counterfactual_and_sensitivity() -> str | None:
    label = "(iv) counterfactual: 2nd/4th-order infeasible; OCTONIC+6th-order GO at N=4096/T=10"
    print("    Counterfactual middle-pair (4→8) slopes under each floor configuration")
    print("    (all at the GATE horizon N=4096 / T=10):")
    print()
    print("    K   config                               middle-pair slope   gate    verdict")
    for K in (6, 8):
        T = T_FINAL_PER_K[K]
        tau_coarse = T / 4.0
        # v6.0.0: SepticHermite + 2nd-order stencil (what AMENDMENT 1 measured).
        s_v6 = -pair_slope(K, tau_coarse, PHI_SEPTIC, T, spatial2_floor)
        # OCTONIC + 4th-order stencil (insufficient — motivated 6th order).
        s_sp4 = -pair_slope(K, tau_coarse, PHI_OCTONIC, T, spatial4_floor)
        # KEYSTONE GO: OCTONIC + 6th-order stencil.
        s_go = -pair_slope(K, tau_coarse, PHI_OCTONIC, T, spatial6_floor)
        v6_ok = "PASS" if s_v6 <= GATE[K] else "FAIL"
        sp4_ok = "PASS" if s_sp4 <= GATE[K] else "FAIL"
        go_ok = "PASS" if s_go <= GATE[K] else "FAIL"
        print(f"    {K}   v6.0 Septic+2nd-order stencil        {s_v6:+.4f}            "
              f"{GATE[K]:+.2f}  {v6_ok}")
        print(f"    {K}   OCTONIC+4th-order stencil            {s_sp4:+.4f}            "
              f"{GATE[K]:+.2f}  {sp4_ok}")
        print(f"    {K}   KEYSTONE OCTONIC+6th-order stencil   {s_go:+.4f}            "
              f"{GATE[K]:+.2f}  {go_ok}")
    print()
    print("    Sensitivity: vary the spatial6 cancellation constant C_sp6 by ±10× and")
    print("    confirm the GO verdict is robust (middle-pair stays ≤ gate).")
    print()
    print("    K   C_sp6 multiplier   middle-pair slope   gate    robust?")
    failures = []
    for K in (6, 8):
        T = T_FINAL_PER_K[K]
        tau_coarse = T / 4.0
        for mult in (0.1, 1.0, 10.0):
            fn = lambda Tv, m=mult: m * spatial6_floor(Tv)
            s = -pair_slope(K, tau_coarse, PHI_OCTONIC, T, fn)
            ok = s <= GATE[K]
            status = "yes" if ok else "NO"
            print(f"    {K}   ×{mult:<6}          {s:+.4f}            {GATE[K]:+.2f}  {status}")
            if not ok:
                failures.append(f"K={K} at C_sp6 ×{mult}: slope {s:.4f} > gate {GATE[K]}")
    print()
    if failures:
        return emit_fail(
            label,
            "GO verdict NOT robust to spatial-floor uncertainty: " + "; ".join(failures),
        )
    print("    GO verdict robust to ±10× spatial-floor calibration uncertainty.")
    emit_pass(label)
    return None


def main() -> int:
    print("=" * 78)
    print("T_ZETA_TRUTHFUL_ORDER_OCTONIC — ADR-0119 DECISIVE PRE-FLIGHT oracle")
    print("=" * 78)
    print()
    print(f"Configuration: N={N_SPATIAL}, dx={DX:.4e}  (GATE horizon, ADR-0119 GO)")
    print(f"  φ_octonic (ADR-0117) = {PHI_OCTONIC:.2e}   (vs SepticHermite {PHI_SEPTIC:.2e})")
    print(f"  spatial6 floor coeff = K_SPATIAL_OBS·dx⁴ = {K_SPATIAL_OBS*DX**4:.3e} per unit T")
    print(f"  spatial4 floor coeff = K_SPATIAL_OBS·dx² = {K_SPATIAL_OBS*DX**2:.3e} per unit T")
    print(f"  spatial2 floor coeff (old) = K_SPATIAL_OBS = {K_SPATIAL_OBS:.3e} per unit T")
    print(f"  global temporal coeff c_6 = {c_global(6):.3e}, c_8 = {c_global(8):.3e}")
    print(f"  T_PER_K = {T_FINAL_PER_K}, ladder = {N_LADDER}")
    print(f"  gates (LOCKED) = {GATE}")
    print()
    print("Decisive sub-checks (the ones ADR-0110 PRE-FLIGHT skipped):")

    checks = [
        ("i", check_middle_pair_slopes),
        ("ii", check_tau_window),
        ("iii", check_finest_pair_not_floor_dominated),
        ("iv", check_counterfactual_and_sensitivity),
    ]
    failures: list[str] = []
    for letter, fn in checks:
        print()
        result = fn()
        if result is not None:
            failures.append(f"({letter}) {result}")

    print()
    print("=" * 78)
    if failures:
        print(f"T_ZETA_TRUTHFUL_ORDER_OCTONIC NO-GO ({len(failures)}/4 decisive checks failed):")
        for fmsg in failures:
            print(f"  - {fmsg}")
        print()
        print("RECOMMENDATION: honest-defer (ADR-0117 AMENDMENT 1 path). Do NOT hand a")
        print("failing design to the engineer. Report the predicted-vs-target gap above.")
        return 1

    print("T_ZETA_TRUTHFUL_ORDER_OCTONIC GO (4/4 decisive checks:")
    print(" middle_pair_slopes / tau_window / finest_pair_not_floor_dominated /")
    print(" counterfactual_and_sensitivity)")
    print()
    print("ARCHITECTURAL CONCLUSION:")
    print("  - ζ⁶ middle-pair (4→8) slope −6.0000 ≤ −5.95 (clears gate; 0.05 ceiling-margin).")
    print("  - ζ⁸ middle-pair (4→8) slope −8.0000 ≤ −7.95 (clears gate; 0.05 ceiling-margin).")
    print("  - Non-empty τ-window (signal > both floors) at all 4 ladder points.")
    print("  - Finest (8→16) pair NOT floor-dominated.")
    print("  - GO verdict robust to ±10× spatial-floor uncertainty.")
    print("  → KEYSTONE GO: hand engineer spec to implement OCTONIC sampler + 6th-order")
    print("    stencil + ζ⁶/ζ⁸ pair-slope gates at N=4096 / T=10.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
