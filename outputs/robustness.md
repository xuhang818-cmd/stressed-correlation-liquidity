# Robustness checks

Block bootstrap (B=1000, block=21 days) preserves daily autocorrelation / volatility clustering.

## 1. Concentration shift (stress - normal), VIX @ 90th pct  (1000 valid resamples)

| metric | mean Δ | 95% CI | bootstrap p |
|---|---|---|---|
| effective bets | -0.47 | [-0.77, -0.17] | 0.000 |
| PC1 share | +8.1pp | [+5.1, +11.1]pp | 0.000 |

Both Δ's exclude 0 (95% CI) => the diversification erosion is statistically real, not sampling noise.

## 2. Cluster stability under VIX stress

See `cluster_stability_VIX.png`. Selected pair co-occurrence (P same group):

- HYG-LQD (credit merge): 0.07
- SPY-EFA (core risk): 1.00
- SPY-TLT (risk vs haven): 0.00
- DBC-SPY (commodity into risk): 0.76

## 2. Cluster stability under FUNDING stress

See `cluster_stability_FUNDING.png`. Selected pair co-occurrence (P same group):

- HYG-LQD (credit merge): 0.49
- SPY-EFA (core risk): 1.00
- SPY-TLT (risk vs haven): 0.00
- DBC-SPY (commodity into risk): 0.01

## 3. Threshold sensitivity (85 / 90 / 95th pct)

| q | proxy | regime Δρ | Δ eff bets | Δ PC1 |
|---|---|---|---|---|
| 0.85 | VIX | -0.002 | -0.40 | +7.6pp |
| 0.85 | FUNDING | -0.032 | +0.30 | -0.3pp |
| 0.90 | VIX | +0.001 | -0.41 | +7.9pp |
| 0.90 | FUNDING | -0.029 | +0.26 | +0.1pp |
| 0.95 | VIX | -0.003 | -0.47 | +7.9pp |
| 0.95 | FUNDING | -0.037 | +0.01 | +2.2pp |

## 4. VIX vs FUNDING stress-day overlap by threshold

| q | Jaccard overlap |
|---|---|
| 0.85 | 0.31 |
| 0.90 | 0.20 |
| 0.95 | 0.16 |

If overlap stays low as q loosens, the two proxies are genuinely different gauges, not a tail artefact.
