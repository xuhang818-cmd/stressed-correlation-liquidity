# Stressed Correlation & Liquidity

**Does cross-asset correlation rise with liquidity stress — and does liquidity *lead* it?**

Folk wisdom says that in a crisis "everything moves together," so a diversified book
offers no protection when you most need it. The usual story is a liquidity one: funding
shocks force deleveraging and fire sales, so unrelated assets sell off at once
(Brunnermeier & Pedersen, 2009). This project measures time-varying correlation with
**DCC-GARCH** and tests, carefully, how much of it actually lines up with — and is led
by — liquidity stress, separating *market* stress (VIX) from *funding* stress (a TED→SOFR
spliced spread).

The headline finding is **not** the folk wisdom, and that is the point.

## Headline results

1. **The signed average barely moves.** For a genuinely diversified 10-asset basket the
   average pairwise correlation is essentially flat — if anything slightly *lower* on the
   worst-VIX days (≈0.15 normal vs ≈0.13 worst-decile; regime diff −0.023, t≈−7.7). The
   "correlations spike to 1 in a crisis" story is rejected at the basket level.

2. **Because the cross-section splits, it doesn't rise uniformly.** Risk assets
   (equities, credit, commodities, REITs) converge; Treasuries and gold decouple
   (flight-to-quality). The positive and negative moves cancel in the signed mean.

3. **Diversification still erodes — as *factor concentration*, not average correlation.**
   On worst-VIX-decile days effective bets fall **4.5 → ~4.0** and PC1 variance share
   rises **34% → 42%**. A 1,000-rep block bootstrap puts the shift at −0.47 bets
   (95% CI [−0.77, −0.17]) and +8pp PC1 (CI [+5, +11]), both p<0.001 — **statistically
   firm but economically moderate.** PC1 is a *bipolar* risk-on/off axis (risk assets +,
   havens −): rising concentration means the basket is increasingly a bet on one factor,
   not that all assets fall together.

4. **It's a market-stress phenomenon, not a funding one.** Re-run on the FUNDING proxy,
   the concentration shift vanishes (≈0 or slightly positive), and VIX-stress vs
   FUNDING-stress days overlap by only **Jaccard ≈ 0.20**. Two largely independent stress
   gauges → the same flat-average result, but different transmission: under **VIX** stress
   commodities get pulled into the risk bloc; under **FUNDING** stress HY and IG credit
   move toward each other (suggestive, co-occurrence ≈0.49). "Diversification breaks down"
   is several mechanisms, not one.

![correlation vs liquidity](outputs/corr_vs_liquidity_VIX.png)

## What it means

- Holding more tickers is not holding more bets — 10 assets here are ~4.5 independent
  bets; the screen shows 13 candidates collapse to ~4 effective bets.
- Effective diversification has to be across *factors*, not asset labels: a hedge must be
  roughly orthogonal to PC1 (near-zero loading), not just "a different ticker."
- The cushion thins exactly in the regime you bought it for — but as factor concentration,
  not the (here false) "average correlation spikes."

## Method

1. **PCA pre-screen** (`pca_screen.py`) — cluster candidates by correlation, drop
   near-duplicates (IWM~SPY, EEM~EFA, IEF~TLT), keep one per cluster + an interactive
   `basket_explorer.html`.
2. **Univariate GARCH(1,1)** per asset → standardized residuals.
3. **DCC** (Engle 2002) → time-varying average pairwise correlation `ρ_t`.
4. **Fisher-z regression** of `ρ_t` on liquidity (+lags), HAC SE, with and without a
   secular time-trend control.
5. **Regime test** (Welch t) + **dose-response** across stress deciles.
6. **Factor-concentration lens** — effective bets (participation ratio) and PC1 share,
   normal vs stress, plus PCA loadings to name the factors.
7. **VAR / IRF** — does a liquidity shock precede a rise in `ρ_t`? (Cholesky-ordered.)
8. **Stress-regime clustering** (`stress_clustering.py`) — how assets regroup under
   stress, VIX vs FUNDING, with a stress-day overlap timeline.
9. **Robustness** (`robustness.py`) — block-bootstrap CIs, cluster-stability
   co-occurrence matrices, and 85/90/95 threshold sensitivity.

## Data

Assets via yfinance (broad/intl equity, REITs, HY/IG credit, commodities, long &
inflation-linked Treasuries, gold, USD). Market stress = VIX (yfinance). Funding stress =
a public-FRED splice: TED spread (TEDRATE, pre-2022) + CP−SOFR (DCPF3M − SOFR), joined on
a z-score scale (their 2018–22 overlap fits at only R²≈0.38, so an affine splice is *not*
used). Configure in `config.py`; the splice is built from three raw FRED CSVs in `data/`.

## Run

```bash
pip install -r requirements.txt
python pca_screen.py            # basket screen + interactive explorer
python main.py --simulate       # offline demo on synthetic data
python main.py                  # real data (per-proxy figures + tables)
python main.py --diagnose-funding   # check the TED/CP-SOFR overlap R^2
python stress_clustering.py     # stress-regime grouping + overlap timeline
python robustness.py            # bootstrap CIs, cluster stability, sensitivity
```

## Structure

```
config.py            tickers, proxies, dates, crises, funding splice
pca_screen.py        basket screen + interactive HTML explorer
stress_clustering.py stress-regime asset grouping + VIX/FUNDING overlap
robustness.py        block-bootstrap CIs, cluster stability, threshold sensitivity
src/data_loader.py   yfinance loader, CSV liquidity, TED->SOFR funding splice
src/dcc_garch.py     univariate GARCH + DCC correlation
src/analysis.py      regression, regime, concentration, PCA, VAR/IRF
src/viz.py           figures (verified writes, per-proxy)
src/export.py        results -> markdown + CSV
main.py              end-to-end pipeline (one figure set per liquidity proxy)
THESIS.md            the writeup as a logical argument
THEORY.md            full math appendix (GARCH/DCC/Fisher-z/PCA/VAR)
FIGURE_GUIDE.md      what each figure measures and how to read it
```

## Limitations

DCC-GARCH measures correlation, not liquidity; co-movement that tracks and is led by
stress is strong circumstantial evidence for the funding mechanism, not proof of
causation. The IRF depends on the Cholesky ordering. The funding splice is numerically,
not economically, continuous (TED carries bank credit risk; SOFR is secured) — a
robustness lens, not a primary series. Results are sensitive to basket composition.

## References

Engle (2002), *Dynamic Conditional Correlation*. Brunnermeier & Pedersen (2009), *Market
& Funding Liquidity*. Longin & Solnik (2001), correlation in bear markets. Forbes &
Rigobon (2002), *No Contagion, Only Interdependence*. Meucci (2009), *Managing
Diversification* (effective number of bets).
