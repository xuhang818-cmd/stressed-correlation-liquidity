# Figure Guide — Stressed Correlation & Liquidity

Each figure below: **what it measures**, **method**, **role in the study**, and a
**conclusion (DRAFT)**. The conclusions are first-pass readings of the 10-asset run
— treat them as drafts to confirm or rewrite, not final claims.

Basket (PCA-pruned, ~4 effective bets): SPY, EFA, VNQ, HYG, DBC, TLT, TIP, LQD, GLD, UUP.
Stress proxy: VIX. Sample: 2007–present.

Two caveats that apply throughout:
- **Trend confound.** Average correlation drifts upward over the sample, and the
  largest VIX spikes (2008, 2020) sit in the low-correlation early years. A pooled
  cross-section of "high-VIX vs low-VIX" can therefore mix the time trend into the
  stress effect. Control for a time trend (or use event windows / IRF) before any
  causal reading.
- **Naive standard errors.** The trend-line t-stat (figure 1) uses OLS SE and
  ignores autocorrelation; it is optimistic. Use HAC/Newey-West for a defensible t.

---

## 1. corr_vs_liquidity.png
**Measures.** Time-varying average pairwise correlation (navy) against the VIX stress
proxy (red), with crisis windows shaded, plus an OLS trend line on the correlation.
**Method.** DCC-GARCH conditional correlation, averaged over off-diagonal pairs each
day; dashed line is OLS of correlation on time (slope reported per year, with t).
**Role.** The orienting picture — is co-movement visually tied to stress, and is there
a secular drift the rest of the analysis must account for?
**Conclusion (DRAFT).** Correlation rises markedly over the sample; the VIX spikes do
not line up with correlation spikes. The secular trend dominates the stress signal —
which is exactly why the later tests must control for it.

## 2. regime.png
**Measures.** Mean correlation on normal vs worst-VIX-decile days.
**Method.** Split the sample at the 90th percentile of VIX; Welch t-test on the means.
**Role.** The regime test — does extreme stress change average co-movement?
**Conclusion (DRAFT).** Worst-decile correlation is *lower* than normal
(diff ≈ −0.02, t ≈ −8): statistically significant but small. The naive "correlation
spikes in a crisis" does not hold for the average of this diversified basket.

## 3. dose_response.png
**Measures.** Average correlation across VIX deciles (calm → stressed).
**Method.** Bin days into VIX deciles, mean correlation per bin; title carries the
Fisher-z OLS slope (HAC SE).
**Role.** The continuous, monotone-shape version of the regime test.
**Conclusion (DRAFT).** Roughly flat with a slight decline (≈0.14 → 0.13); slope
negative but tiny. Consistent with figure 2 — no dose-response rise in stress.

## 4. corr_matrices_regime.png
**Measures.** The full return-correlation matrix, normal vs worst-VIX-decile.
**Method.** Pearson correlation on each subsample.
**Role.** The mechanism — *which* pairs move when stress hits, behind the flat average.
**Conclusion (DRAFT).** The cross-section reorganizes even though the average barely
moves: the risk block (SPY/EFA/VNQ/HYG/DBC) converges sharply — commodities (DBC) and
IG credit (LQD) get pulled toward the risk cluster — while Treasuries (TLT) decouple
and turn more negative, and the rates block fragments (TLT–LQD ≈ 0.80 → 0.21). Risk
assets converge; havens diverge; the two offset in the signed average.

## 5. diversification_regime.png
**Measures.** Effective number of bets and PC1 variance share, normal vs stress.
**Method.** Eigenvalues of the correlation matrix per regime: participation ratio
1/Σpᵢ² (effective bets) and the top-eigenvalue share (PC1).
**Role.** The sign-robust lens. Signed correlation can stay flat while variance still
concentrates onto one factor; this figure captures that.
**Conclusion (DRAFT).** Effective bets fall (≈4.5 → 4.1) and PC1 share rises
(≈34% → 42%) in stress: diversification erodes and systemic co-movement strengthens —
modestly but in the predicted direction. This reconciles the flat signed-average:
"correlations → 1 in a crisis" is about factor concentration, not the signed mean.

## 6. rolling_concentration.png
**Measures.** Factor concentration through time.
**Method.** Rolling 126-day (~6-month) window; PC1 variance share of the correlation
matrix in each window; crises shaded.
**Role.** Separates the secular trend from acute episodes, and shows concentration is
regime-dependent rather than a fixed property of the basket.
**Conclusion (DRAFT).** PC1 share swings widely (≈30%–63%), with peaks around 2012 and
2022–23 rather than only at the 2008/2020 VIX spikes — concentration tracks macro
regimes (rates/inflation), not just equity-vol events.

---

### Suggested overall takeaway (DRAFT — your call)
In a properly diversified, PCA-pruned basket, the naive claim that *average* cross-asset
correlation spikes in stress is weak-to-absent. The real effect is structural and
sign-robust: in stress the cross-section concentrates onto fewer effective factors
(PC1 share up, effective bets down) as risk assets converge and havens decouple. The
magnitude is modest and is partly entangled with a strong secular rise in correlation,
so causal claims need a time-trend control or an event-window (IRF) design.
