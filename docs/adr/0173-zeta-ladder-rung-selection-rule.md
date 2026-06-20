# ADR-0173 — ζ-ladder gate rung-selection: one pre-asymptotic-window invariant

**Status:** ACCEPTED (2026-06-20) · **Amends:** ADR-0110 (ζ⁴ truthful-order) and ADR-0119 (ζ⁶/ζ⁸ truthful-order)

**Decision.** The three truthful-order gates use *different positional rung picks*
(`zeta4_truthful_order.rs:84` full-ladder OLS ≤ −3.5 excluding the anomalous finest pair;
`zeta6_truthful_order.rs:56` / `zeta8_truthful_order.rs:57` finest-pair-only ≤ −5.95 / −7.95)
not by inconsistency but because they sit at **different points of the same pre-asymptotic
window**. The honest invariant all three share: *score the gate only on rungs whose step τ
lies inside the pre-asymptotic window `c·τ^{K} ≫ φ_floor`, where the temporal signal
dominates the spatial sampler floor.* ζ⁶/ζ⁸ run at T=10 (finest τ=0.625), so **every** rung
is deep pre-asymptotic and the finest pair is the most-asymptotic, tightest ≥K witness —
finest-pair-only is in-window. ζ⁴ runs at T=2 (finest τ=0.125 < τ_pre_asymp≈0.162,
SAFETY≈27), so its finest pair (slope ≈ −1.08) is in the **transition zone**, NOT an order
floor — it is *out of window* and must be excluded; the in-window coarse/middle pairs carry
the order-4 signal (middle pair −4.07). Naïvely switching ζ⁴ to finest-pair-only would score
an out-of-window rung and falsely fail — that would be dishonest threshold gaming in reverse.
The principled mechanic, applied identically: **(1)** classify each rung by SAFETY =
`c·τ^{K}/φ`; **(2)** retain only rungs with SAFETY ≥ 100 (deep pre-asymp) for the
order-witness; **(3)** gate the retained rungs at `≤ −(K−0.5)` for OLS over ≥3 retained
rungs, or `≤ −(K−0.05)` for a single finest in-window pair. Per-test result, unchanged in
spirit and no threshold weakened: **ζ⁴** keeps full-ladder OLS ≤ **−3.5** with the
out-of-window finest pair excluded (3 in-window rungs); **ζ⁶** keeps finest-pair ≤ **−5.95**;
**ζ⁸** keeps finest-pair ≤ **−7.95** (both finest pairs are in-window at T=10). The
reconciliation is the SAFETY-window invariant, not identical positional picks — each gate
selects the in-window rungs its own τ-ladder exposes.
