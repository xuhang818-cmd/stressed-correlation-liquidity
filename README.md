# Stressed Correlation & Liquidity

**Does cross-asset correlation rise with funding/liquidity stress — and does liquidity *lead* correlation?**

Asset correlations are not constant: in a crisis "everything moves together." A
large part of that is a liquidity mechanism — funding shocks force deleveraging
and fire sales, so otherwise-unrelated assets sell off at once (Brunnermeier &
Pedersen, 2009). This project measures time-varying correlation with a
**DCC-GARCH** model and tests how much of its variation lines up with, and is led
by, **funding- and market-liquidity** conditions.

## Method

1. **Univariate GARCH(1,1)** on each asset → standardized residuals (strips out
   volatility so what remains is co-movement).
2. **DCC** (Engle, 2002) on those residuals → a time-varying correlation matrix;
   I track the average pairwise correlation `rho_t`.
3. **Fisher-z regression**: `arctanh(rho_t)` on liquidity proxies (+ lags),
   Newey-West/HAC standard errors (because `rho_t` is highly autocorrelated).
4. **Regime test**: average correlation on the worst-liquidity decile of days
   vs. the rest (Welch t-test).
5. **VAR / IRF**: does a liquidity shock precede a rise in correlation?
   (Direction via Cholesky ordering — see *Limitations*.)

## Data

- **Assets** (yfinance): SPY, TLT, LQD, HYG, GLD — a diversified cross-asset basket.
- **Liquidity proxies** (FRED): VIX (`VIXCLS`), HY OAS (`BAMLH0A0HYM2`),
  TED spread (`TEDRATE`, historical). Edit in `config.py`.

## Key result

![correlation vs liquidity](outputs/corr_vs_liquidity.png)

Average pairwise correlation spikes during liquidity-stress episodes; the regime
test confirms the difference is large and significant.

## Limitations (read this — it's the point)

DCC-GARCH measures **correlation, not liquidity**. A rise in `rho_t` is consistent
with liquidity-driven forced selling, but could also reflect a common fundamental
shock. This project shows correlation **co-moves with** and is **led by** liquidity
stress — strong circumstantial evidence — but it does **not** prove causation.
A clean causal claim would need an *identified* exogenous liquidity shock. The
VAR/IRF result also depends on the Cholesky ordering (liquidity ordered first as
the more exogenous variable); the sign is robust to that choice but the magnitude
is not.

## Run

```bash
pip install -r requirements.txt
python main.py --simulate   # offline demo on synthetic data (link known by design)
python main.py              # real data: yfinance + FRED
```

## Structure

```
config.py            tickers, FRED series, dates, crisis windows
src/data_loader.py   real (yfinance + FRED) and simulated data
src/dcc_garch.py     univariate GARCH + DCC correlation recursion
src/analysis.py      Fisher-z regression, regime test, VAR/IRF
src/viz.py           figures
main.py              end-to-end pipeline
```

## References

- Engle (2002), *Dynamic Conditional Correlation*, JBES.
- Brunnermeier & Pedersen (2009), *Market Liquidity and Funding Liquidity*, RFS.
- Cappiello, Engle & Sheppard (2006), asymmetric dynamics in correlations.
