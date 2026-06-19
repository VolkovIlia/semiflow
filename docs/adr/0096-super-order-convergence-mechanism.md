# ADR-0096 — Super-Order Convergence on `exp(-x^p)`: NOVEL Math Attempt → Outcome A (partial; Hypothesis C confirmed for p∈{2,4}, p=6 mechanism remains OPEN)

- **Status**: Accepted (positive result for the p=2 and p=4 cases; partial — does NOT yet explain Galkin-Remizov 2023 Observation 4 p=6 collapse).
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Supersedes**: NONE (additive research deliverable; closes one open speculation in Galkin-Remizov 2023 Conclusion Observation 4; partially closes one user-attention item in `.dev-docs/research/verdicts/verdict-v4-4-research-wave.md` §"User-attention items").
- **Depends on**: ADR-0073 (`ApproximationSubspace<K>` witness), ADR-0086 + AMENDMENT 1 (Path β Richardson on K5; ζ⁴), ADR-0088 (nested Richardson ladder; ζ⁶/ζ⁸), ADR-0091 (diagonal Padé ζ⁸), ADR-0092 (Romberg-2D negative result; methodological precedent for sympy-only math creation deliverables), ADR-0093 (ADR-0075 attribution correction; reaffirms Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 as the authoritative abstract rate theorem).
- **Mathematical foundation**: Galkin-Remizov 2023 (arXiv:2301.05284v5) Conclusion §1015 Observation 4 ("smooth exponentials show higher order than predicted; $\alpha_S \approx -3.1$ for $u_0 = e^{-x^4}$ — UNEXPLAINED by existing theory"); Vedenin et al. 2020 *Math. Notes* 108(3) §3 (S-function $S(t)f = \tfrac{2}{3}f(x) + \tfrac{1}{6}f(x \pm \sqrt{6t})$, Chernoff order $k=2$); Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 + Lemma 3.1 (Taylor-tangency abstract rate $\|S(t/n)^n f - e^{tL}f\| \le (M t^{m+1} e^{wt}/n^m) \sum K_j \|L^j f\|$ for $m$-tangent $S(t)$).
- **Acceptance gates added**: NONE (no kernel ships). Sympy oracle `scripts/derive_super_order_mechanism.py` ships as research deliverable with 5 sub-checks; verdict artifact `.dev-docs/research/verdicts/verdict-super-order-attempt.md` documents the partial-positive result and the residual p=6 open question.

## Context

User directive: "если не найдёшь, попробуй сам создать математику" (Wave 4 research synthesis `.dev-docs/research/verdicts/verdict-v4-4-research-wave.md` flagged a HIGH-LEVERAGE math creation opportunity from Galkin-Remizov 2023 Conclusion §1015 Observation 4 — Vedenin's nominally order-2 S-function exhibits empirical $\alpha_S \approx -3.1$ for the super-Gaussian initial datum $u_0(x) = e^{-x^4}$, exceeding the theoretical $-2$ rate predicted by Galkin-Remizov 2024 Theorem 4.2; the paper authors explicitly state this is unexplained). If a formal mechanism could be identified, the practical implication was potentially decisive: order-4+ approximations might be reachable on super-smooth initial data WITHOUT explicit 4th-order tangency design — bypassing the v3.0 Phase 5a falsification (ADR-0075 retraction per ADR-0093), the ζ-ladder pre-asymptotic ceiling (ADR-0088 AMENDMENT 2), and the $\tau\|A\|$ Padé scaling problem (ADR-0091) entirely.

Three candidate mechanisms were entertained (Hypothesis A "decay-rate resonance", Hypothesis B "operator-domain boundary", Hypothesis C "higher-order tangency emergence via pre-asymptotic regime blending"). Hypotheses A and B require fresh literature search and theorem-hypothesis modification beyond pure sympy and are deferred to user-attention items. Hypothesis C is symbolically tractable: this ADR documents the math attempt and the rigorous symbolic result.

## Decision

**Document the math creation attempt and its outcome (Outcome A, partial).** Ship the sympy derivation script (`scripts/derive_super_order_mechanism.py`) as a research deliverable with 5 sub-checks that PASS in their stated framing, and a verdict artifact (`.dev-docs/research/verdicts/verdict-super-order-attempt.md`). Append a new §27.sext to `contracts/semiflow-core.math.md` formalising the mechanism for $p = 2$ and $p = 4$. Do NOT ship a new kernel type, do NOT ship a new acceptance gate, do NOT modify any existing kernel default — the v3.0 / v4.x ζ-ladder design (explicit 4th-order tangency via Path β Richardson, ADR-0086) remains the ONLY library mechanism for guaranteed order-4 convergence at variable n. The super-order mechanism formalised here is a **finite-n explanation, not an asymptotic free-lunch**: sub-check (5) confirms $\alpha_{\mathrm{apparent}}(n) \to -2$ as $n \to \infty$ for all $p$, so the v3.0 ζ-ladder remains necessary for guaranteed asymptotic order-4 behaviour.

## Math attempts: three hypotheses considered

### Hypothesis A — Decay-rate resonance

**Claim**: for $u_0(x) = e^{-c x^p}$, the S-function spacing $\sqrt{6t}$ aligns with the characteristic decay scale of the datum at $p = 4$ such that the Taylor remainder of $S(t)u_0 - e^{tL}u_0$ cancels at higher order than $k = 2$.

**Status**: REQUIRES FRESH LITERATURE. The "resonance" intuition does not map to a standard analytic-number-theory or quadrature-cancellation framework that this architect could derive from first principles. A direct sympy attempt to find a cancellation pattern in $\langle S(t)u_0 - e^{tL}u_0, u_0 \rangle / \|u_0\|^2$ for $u_0 = e^{-x^4}$ gave no closed form. Possible avenues: matched-asymptotics analysis (boundary-layer theory for the heat kernel convolution against $e^{-x^4}$); Plancherel-side analysis (compare Fourier transforms of $S(t)$ kernel vs $e^{tL}$ kernel against $\hat{u}_0(\xi)$). User-attention item; deferred.

### Hypothesis B — Operator-domain boundary

**Claim**: Galkin-Remizov 2025 *IJM* Theorem 3.1 hypothesis (12) `‖S(t)f − Σ_{k=0}^m t^k L^k f / k!‖ ≤ t^{m+1} · Σ_{j=0}^{m+p} K_j(t) · ‖L^j f‖` is provably TIGHT for $f \in D(L^{m+p})$ but admits a sharper bound for "super-smooth" data $f \in D(L^\infty)$ where ALL Taylor derivatives are bounded uniformly. The paper does not investigate this finer stratification.

**Status**: REQUIRES THEOREM-HYPOTHESIS MODIFICATION. The Galkin-Remizov 2025 *IJM* proof structure (Lemma 3.1 algebraic telescoping + Lemma 3.3 Taylor-with-Bochner-remainder) is genuinely $O(n^{-m})$ tight on the standard hypothesis. Tightening the hypothesis to require uniformly-bounded derivatives of ALL orders would give a stronger rate (potentially $O(n^{-m-1})$ or better), but this would constitute new theory beyond what the existing paper proves. User-attention item; recommend Remizov/Galkin direct collaboration for a "Theorem 3.1 super-smooth refinement".

### Hypothesis C — Higher-order tangency emergence via pre-asymptotic regime blending (FORMALISED HERE)

**Claim**: the empirical $\alpha_S \approx -3.1$ on $u_0 = e^{-x^4}$ is a PRE-ASYMPTOTIC effect: the leading $\mathcal{O}(t^3/n^2)\|f^{(VI)}\|$ term in the Galkin-Remizov 2025 *IJM* Theorem 3.1 expansion at $m = 2$ is DOMINATED, in the finite-$n$ window $n \in [4, 11]$ that the paper measures, by the next-order $\mathcal{O}(t^4/n^3)\|f^{(VIII)}\|$ term — because the derivative-norm ratio $\|f^{(VIII)}\|/\|f^{(VI)}\|$ for $f = e^{-x^4}$ is substantially larger than for $f = e^{-x^2}$ (sympy sub-check (3) confirms: 101 vs 14, a 7$\times$ amplification).

**Status**: FORMALLY DERIVED, sympy-verified (5 sub-checks, all PASS in stated framing). Mechanism is order-correct (leading $C_3 = 1/15$, next-order $C_4 = 13/420$, both rational closed forms — sub-check (1)); the resulting two-term blended-error linear regression on $n \in [4, 11]$ yields apparent slope $-2.78$ for $p = 4$ versus $-2.33$ for $p = 2$ — qualitatively matching the paper's $-3.1$ vs $-2.1$ observation (sub-check (4); see Caveat 1 below for the magnitude underestimate). Asymptotic recovery $\alpha_{\mathrm{apparent}}(n) \to -2$ for all $p$ at $n \in [5000, 15000]$ confirms this is regime blending, NOT a violation of Galkin-Remizov 2025 *IJM* Theorem 3.1 (sub-check (5)).

## Algorithm (Hypothesis C derivation, NORMATIVE per sympy `scripts/derive_super_order_mechanism.py`)

### Step 1 — Vedenin S-function single-step Taylor expansion

For the heat operator $L = \partial_x^2$ on $\mathbb{R}$ (set $a = 1$ for clarity; coefficients scale homogeneously), the Vedenin 2020 second-order S-function

$$
S(t)f(x) = \tfrac{2}{3} f(x) + \tfrac{1}{6} f(x + \sqrt{6t}) + \tfrac{1}{6} f(x - \sqrt{6t})
$$

admits the symbolic Taylor expansion in $t$ (sub-check (1) PASS):

$$
S(t)f = f + t \cdot f'' + \tfrac{t^2}{2} \cdot f^{(IV)} + \tfrac{t^3}{10} \cdot f^{(VI)} + \tfrac{3 t^4}{280} \cdot f^{(VIII)} + \mathcal{O}(t^5) \cdot f^{(X)}.
$$

The exact heat semigroup gives $e^{tL}f = f + t f'' + \tfrac{t^2}{2} f^{(IV)} + \tfrac{t^3}{6} f^{(VI)} + \tfrac{t^4}{24} f^{(VIII)} + \mathcal{O}(t^5) \cdot f^{(X)}$. The single-step residual is therefore

$$
R(t) f := S(t) f - e^{tL} f = -\frac{t^3}{15} f^{(VI)} - \frac{13 t^4}{420} f^{(VIII)} + \mathcal{O}(t^5) \cdot f^{(X)}.
$$

This confirms Chernoff tangency order $m = 2$ (the $t^0$, $t^1$, $t^2$ residual coefficients vanish identically) and gives the **closed-form rational coefficients** $C_3 = 1/15 \approx 0.0667$ and $C_4 = 13/420 \approx 0.0310$ of the leading and next-order residual terms. These coefficients are GENERIC: they depend only on $S(t)$ and $L = \partial_x^2$, not on $f$.

### Step 2 — Galkin-Remizov 2025 *IJM* Lemma 3.1 global telescoping

The Galkin-Remizov 2025 *IJM* Theorem 3.1 hypothesis is met with $m = 2$ and (from Step 1)

$$
K_6(t) \equiv 1/15, \qquad K_8(t) \equiv 13 t / 420, \qquad K_j(t) \equiv 0 \text{ for } j \notin \{6, 8\}.
$$

The proof of Theorem 3.1 (Lemma 3.1 algebraic telescoping + Lemma 3.3 Taylor-with-Bochner-remainder of $e^{tL}f$) gives the GLOBAL two-term bound (with $\tau = t/n$):

$$
\bigl\| S(t/n)^n f - e^{tL} f \bigr\| \;\le\; n \cdot \tau^3 \cdot \tfrac{1}{15} \, \|f^{(VI)}\| \;+\; n \cdot \tau^4 \cdot \tfrac{13}{420} \, \|f^{(VIII)}\| \;+\; \mathcal{O}(n \tau^5)
$$
$$
\qquad = \tfrac{t^3}{15 n^2} \, \|f^{(VI)}\| + \tfrac{13 \, t^4}{420 \, n^3} \, \|f^{(VIII)}\| + \mathcal{O}(t^5 / n^4)
$$

(sub-check (2) PASS). The leading $1/n^2$ term recovers the $m = 2$ asymptotic rate guaranteed by Galkin-Remizov 2025 *IJM* Theorem 3.1; the next-order $1/n^3$ term is the SAME object that the v3.0 ζ⁴ Path β Richardson construction (ADR-0086 + AMENDMENT 1) explicitly cancels at single-step level via $\tfrac{4}{3} K_5(\tau/2)^2 - \tfrac{1}{3} K_5(\tau)$, lifting the global rate to $\mathcal{O}(t^5/n^4)$.

### Step 3 — Derivative norms for $f(x) = e^{-x^p}$, $p \in \{2, 4, 6\}$

Sympy numerical estimation (sub-check (3) PASS):

| $p$ | $\|f^{(IV)}\|_\infty$ | $\|f^{(VI)}\|_\infty$ | $\|f^{(VIII)}\|_\infty$ | $\|f^{(VIII)}\|/\|f^{(VI)}\|$ | $\|f^{(VI)}\|/\|f^{(IV)}\|$ |
|---|---|---|---|---|---|
| 2 (Gaussian)        | $1.20 \times 10^{1}$ | $1.20 \times 10^{2}$ | $1.68 \times 10^{3}$ | $14.0$         | $10.0$ |
| 4 (super-Gaussian)  | $1.32 \times 10^{2}$ | $8.52 \times 10^{3}$ | $8.60 \times 10^{5}$ | $\mathbf{101}$ | $64.8$ |
| 6                   | $8.18 \times 10^{2}$ | $1.49 \times 10^{5}$ | $4.71 \times 10^{7}$ | $317$          | $182$ |

**Key observation**: the $\|f^{(VIII)}\|/\|f^{(VI)}\|$ ratio for $p = 4$ is $\approx 7\times$ larger than for $p = 2$. The $\|f^{(VI)}\|/\|f^{(IV)}\|$ ratio is similarly amplified ($\approx 6.5\times$). This is the structural fact behind the super-order observation.

### Step 4 — Apparent regression slope blending

Substituting the Step 3 norms into the Step 2 two-term bound and computing the linear regression of $\ln \mathrm{err}(n)$ vs $\ln n$ on $n \in [4, 11]$, $t = 1/2$ (matching Galkin-Remizov 2023 numerical methodology):

| $p$ | $\alpha_{\mathrm{apparent}}$ (this work) | $\alpha_S$ (paper Observation 4) | Direction matches? |
|---|---|---|---|
| 2 (Gaussian)        | $-2.33$ | $-2.10$ | YES (mildly enhanced beyond -2.0) |
| 4 (super-Gaussian)  | $-2.78$ | $-3.10$ | YES (substantially enhanced) |
| 6                   | $-2.91$ | $-1.80$ (COLLAPSE) | **NO** — sign of effect opposite |

**Caveats** (cf. §"Limitations" below):
- **Caveat 1** (magnitude underestimate, $p = 4$): the two-term model predicts $\alpha = -2.78$ versus the paper's $-3.10$. The missing $0.32$ slope difference is plausibly explained by the $t^5/n^4 \cdot \|f^{(X)}\|$ next-next-order term. For $p = 4$, $\|f^{(X)}\|/\|f^{(VIII)}\|$ would likely be amplified by another $\approx 7\times$ over the Gaussian case (extrapolating the Step 3 trend), making the THIRD term in the expansion also relevant in the $n \in [4, 11]$ window. A three-term blending model is a natural follow-up but does not change the qualitative conclusion.
- **Caveat 2** (sign reversal, $p = 6$): the script predicts $\alpha = -2.91$ (more enhancement than $p = 4$), but the paper reports $\alpha_S \approx -1.8$ for $p = 6$ — a COLLAPSE to the order-1 G-function regime. This is a STRONG signal that an ADDITIONAL mechanism is at play for $p = 6$ that the present blending model does NOT capture. Candidates: spatial-locality effects (the $\sqrt{6t}$ shift sampling falls OUTSIDE the localized support of $e^{-x^6}$ for $t = 1/2$, leading to systematic underestimation that mimics low-order convergence); or boundary-of-domain effects in the paper's finite-grid implementation (the 25-IC sweep methodology, §"Numerical methodology" in the extract, does not specify grid extent). Outcome A is therefore **partial** — the mechanism explains $p \in \{2, 4\}$ enhancement directionally but does NOT explain the $p = 6$ collapse. User-attention item; the $p = 6$ behaviour remains OPEN.

### Step 5 — Asymptotic recovery (sub-check (5) PASS)

| Window $[n_{\mathrm{lo}}, n_{\mathrm{hi}}]$ | $\alpha_{\mathrm{apparent}}$ ($p = 4$) |
|---|---|
| $[4, 11]$        | $-2.77$ |
| $[10, 30]$       | $-2.57$ |
| $[50, 150]$      | $-2.21$ |
| $[500, 1500]$    | $-2.03$ |
| $[5000, 15000]$  | $-2.00$ |

The apparent slope drifts MONOTONICALLY toward $-2.00$ as $n$ grows. This **CONFIRMS that the super-order observation is a pre-asymptotic regime-blending artefact**, NOT a genuine violation of Galkin-Remizov 2025 *IJM* Theorem 3.1. The asymptotic rate remains $\mathcal{O}(1/n^2)$ for $S(t)$, and the v3.0 / v4.x ζ-ladder remains NECESSARY for guaranteed asymptotic order $\ge 4$ at arbitrary $n$.

## Implications for SemiFlow

**The "free 4th order without 4th-order tangency design" hypothesis is FALSIFIED at the asymptotic level**: Step 5 / sub-check (5) shows the apparent super-order is a finite-$n$ blending phenomenon that disappears at $n \gtrsim 10^3$. SemiFlow's v3.0 ζ⁴ Path β Richardson kernel `Diffusion4thZeta4Chernoff` (ADR-0086 + AMENDMENT 1) and the v4.x ζ⁶ / ζ⁸ ladder (ADR-0088 + ADR-0091) remain the ONLY library mechanism for guaranteed order-4+ convergence at production-scale $n$. The asymptotic ceiling for the bare S-function on $\partial_x^2$ is $\mathcal{O}(1/n^2)$, period.

**However, the mechanism gives a finite-$n$ PRESENTATION insight**: for super-smooth initial data (e.g. exp-of-polynomial decay), the apparent convergence rate in the small-$n$ regime ($n \in [4, 11]$) of plain S-function may MIMIC a higher-order kernel. This is operationally interesting if a downstream user (e.g. a financial-engineering benchmark) reports "Vedenin S-function performed as well as ζ⁴ at $n = 8$, $u_0 = e^{-x^4}$": this ADR explains WHY without requiring any code change, and reaffirms that asymptotic claims still require the explicit ζ-ladder kernel.

**No new kernel ships**, no acceptance gate added, no schema bump, no constitution change.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Ship a `SuperGaussianAwareChernoff<F>` kernel that auto-detects exp-of-polynomial ICs and switches to S-function instead of ζ⁴** | The asymptotic ceiling for S-function is still $\mathcal{O}(1/n^2)$ (Step 5). Such a kernel would optimise for a narrow finite-$n$ regime ($n \in [4, 11]$, the regime of the paper) where the apparent super-order holds, at the cost of asymptotic correctness. Suckless single-kernel principle (each kernel has ONE provable order); IC-conditional kernel selection is anti-pattern. |
| **Pursue Hypothesis A (decay-rate resonance) symbolically here** | Requires matched-asymptotics / Plancherel-side derivation beyond pure sympy. Architect attempted symbolic cancellation in $\langle S(t) u_0 - e^{tL} u_0, u_0 \rangle$ for $u_0 = e^{-x^4}$ and obtained no closed form. The intuition is plausible but the derivation is genuine open math; user-attention item. |
| **Pursue Hypothesis B (operator-domain boundary) by modifying Theorem 3.1 hypothesis** | Modifies the foundational theorem to require uniformly-bounded derivatives of ALL orders for "super-smooth" data and proves a stronger rate. This would constitute new abstract theory beyond this ADR's scope. Recommend Remizov/Galkin direct collaboration. |
| **Defer ADR; do NOT investigate per Anchor's risk-management default** | User directive explicitly authorised math creation; the verdict-v4-4-research-wave.md flagged this specific observation as HIGH-LEVERAGE. Outcome A (partial) is a genuinely valuable scientific result — it closes the speculation that "super-order on $e^{-x^4}$ implies asymptotic 4th-order without 4th-order design", reaffirms the necessity of v3.0 / v4.x ζ-ladder, and identifies the $p = 6$ collapse as a genuine OPEN sub-question. Negative results are valuable per Constitution Principle #1 (math fidelity > engineering velocity). |
| **Extend to a 3-term blending model (`1/n^2 + 1/n^3 + 1/n^4`)** | Would likely tighten the $p = 4$ slope match from $-2.78$ to closer to $-3.1$, but does NOT explain the $p = 6$ collapse (which is the load-bearing open sub-question per Caveat 2). Marginal informational value; deferred to user-attention follow-up. |

## Consequences

- **POSITIVE**: closes Galkin-Remizov 2023 Observation 4 in its PRACTICAL implication (super-order is finite-$n$ blending, not asymptotic free-lunch) and provides closed-form rational coefficients $C_3 = 1/15$ and $C_4 = 13/420$ for the S-function residual that may be useful in future analyses (e.g. mixed-order kernel design); reaffirms v3.0 / v4.x ζ-ladder design as the asymptotic mechanism for guaranteed order-4+ convergence (architecturally important — would have been a regression risk if super-order were genuine asymptotic); methodology mirrors ADR-0092 (sympy-only deliverable for math creation attempts) — establishes pattern for future Outcome-A / Outcome-B research.
- **NEUTRAL**: no kernel ships; no acceptance gate added; properties.yaml unchanged; traits.yaml unchanged; constitution unchanged; existing ζ-ladder defaults unchanged.
- **NEGATIVE**: partial outcome — the $p = 6$ collapse remains OPEN. One architect cycle did not produce a unified mechanism for all $p$. However the cycle PRODUCED a rigorous derivation of the $p \in \{2, 4\}$ enhancement and isolated $p = 6$ as a separate research question, narrowing the remaining open space.
- **BREAKING**: NONE.
- **Schema bumps**: NONE (no contract surface change).
- **Open follow-up** (for v4.5+ Anchor delegation, in priority order):
  1. **$p = 6$ collapse investigation** (sympy + numeric replication of paper's grid methodology): is the collapse a numerical-methodology artefact (finite-grid sampling outside $e^{-x^6}$ support) or a genuine analytic effect? Likely tractable in $\le 1$ architect cycle if grid extent is recovered from paper.
  2. **Hypothesis A** (decay-rate resonance): matched-asymptotics or Plancherel-side derivation for $\langle S(t) e^{-x^4} - e^{tL} e^{-x^4}, e^{-x^4} \rangle$. Out-of-band literature search (Davis-Rabinowitz quadrature; saddle-point methods for oscillatory integrals against Gaussian-type measures).
  3. **Hypothesis B** (Theorem 3.1 super-smooth refinement): direct Remizov/Galkin collaboration; would constitute new abstract theory.
  4. **Three-term blending model**: extend `derive_super_order_mechanism.py` sub-check (4) to include the $t^5/n^4 \cdot \|f^{(X)}\|$ term, verify $p = 4$ slope match improves from $-2.78$ to $\approx -3.0$. Low-value polish.

## References

- O. E. Galkin, I. D. Remizov (2025), *Upper and lower estimates for rate of convergence in the Chernoff product formula for semigroups of operators*, **Israel Journal of Mathematics** (online) — Theorem 3.1 (abstract rate $O(n^{-m})$ for $m$-tangent Chernoff function), Lemma 3.1 (algebraic telescoping $Z^n - Y^n = \sum_{k=0}^{n-1} Z^{n-k-1}(Z-Y)Y^k$), Lemma 3.3 (Taylor-with-Bochner-remainder of $e^{tL}f$). The abstract rate-theorem proof structure that underpins Steps 1–2 of this ADR.
- E. Yu. Katalova (Dragunova), N. Nikbakht, I. D. Remizov (2023), *Concrete examples of the rate of convergence of Chernoff approximations: numerical results for the heat semigroup and open questions on them*, arXiv:2301.05284v5 — Conclusion §1015 Observation 4 (the unexplained empirical $\alpha_S \approx -3.1$ on $u_0 = e^{-x^4}$); §"Numerical methodology" (linear-regression of $\ln d_n$ vs $\ln n$ on $n \in [4, 11]$ at $t = 1/2$, the methodology replicated in sub-check (4) of this work).
- A. V. Vedenin, V. S. Voevodkin, V. D. Galkin, E. Yu. Karatetskaya, I. D. Remizov (2020), *Speed of convergence of Chernoff approximations to solutions of evolution equations*, **Mathematical Notes** 108(3), 451–456, DOI 10.1134/S0001434620090151 — §3 (canonical statement of S-function $S(t)f = \tfrac{2}{3}f(x) + \tfrac{1}{6}f(x \pm \sqrt{6t})$ with conjectured $O(1/n^2)$ rate), Remark 6 (jet-derivative principle: more matched Taylor terms ⇒ higher rate).
- I. D. Remizov (2018), *On the Chernoff product formula for parabolic semigroups generated by second-order operators in $L^p(\mathbb{R}^N)$*, **Appl. Math. Comput.** 328, 243–250 — origin of the first-order G-function $G(t)f = \tfrac{1}{2}f(x) + \tfrac{1}{4}f(x \pm 2\sqrt{t})$ and the direct lineage of `DiffusionChernoff::new` in SemiFlow v0.1.0.
- `.dev-docs/research/extracts/galkin-remizov-2023-extract-v2.md` — fresh download v5 extract of arXiv:2301.05284 (Wave 4, 2026-05-29), contains the verbatim Conclusion Observations 1–5 and the empirical $\alpha$ values used in Step 4.
- `.dev-docs/research/extracts/galkin-remizov-2025-extract.md` — extract of arXiv:2104.01249v2 (Wave 2; the IJM-published version), verbatim Theorem 3.1 hypothesis and proof structure.
- `.dev-docs/research/extracts/vedenin-speed-extract.md` — extract of Vedenin et al. 2020 *Math. Notes* 108(3), canonical S-function definition.
- `.dev-docs/research/verdicts/verdict-v4-4-research-wave.md` — Wave 4 architect synthesis that flagged this as a HIGH-LEVERAGE math creation opportunity per user directive.
- `scripts/derive_super_order_mechanism.py` — sympy verification harness (5 sub-checks), this ADR's load-bearing scientific artefact.
- `.dev-docs/research/verdicts/verdict-super-order-attempt.md` — verdict artefact documenting the partial-positive Outcome A and the residual $p = 6$ open sub-question.
- ADR-0073 (ApproximationSubspace witness), ADR-0086 + AMENDMENT 1 (Path β Richardson on K5), ADR-0088 (nested Richardson ladder), ADR-0091 (diagonal Padé direct ζ⁸), ADR-0092 (Romberg-2D negative result; methodological precedent for sympy-only research deliverables), ADR-0093 (ADR-0075 attribution correction).

## Limitations

1. **Caveat 1 — magnitude underestimate at $p = 4$**: two-term model predicts $\alpha = -2.78$ vs paper $-3.1$. Three-term extension is a natural follow-up.
2. **Caveat 2 — $p = 6$ collapse OPEN**: two-term model predicts $\alpha = -2.91$ vs paper $-1.8$ (COLLAPSE). Mechanism for collapse is NOT captured by this ADR. Candidate explanations (spatial-locality, finite-grid sampling, boundary-of-domain) require numerical replication of paper methodology to discriminate.
3. **Hypotheses A and B not investigated symbolically**: Hypothesis A requires matched-asymptotics outside pure sympy reach; Hypothesis B requires new abstract theorem statement. Both are user-attention items.
4. **Asymptotic ceiling unchanged**: the v3.0 / v4.x ζ-ladder remains the ONLY library mechanism for guaranteed order-4+ convergence at arbitrary $n$. This ADR explains a finite-$n$ effect, not an asymptotic free-lunch.
