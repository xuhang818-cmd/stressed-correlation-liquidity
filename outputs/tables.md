# Results tables

Auto-generated from the run. CSV versions of the larger tables are alongside.

## Fisher-z regression: VIX coefficient

| Spec | slope | t (HAC) | p |
|---|---|---|---|
| baseline | -0.0009 | -2.50 | 1.2e-02 |
| + time trend | -0.0004 | -1.52 | 1.3e-01 |

## Regime (Welch t)

| group | mean corr |
|---|---|
| normal | 0.1518 |
| worst decile | 0.1289 |
| diff (t=-7.7, p=8.7e-14) | -0.0229 |

## Diversification / concentration

| metric | normal | stress |
|---|---|---|
| effective bets | 4.54 | 4.13 |
| PC1 share | 34% | 42% |

## Dose-response (mean corr by VIX decile)

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean corr | 0.144 | 0.162 | 0.159 | 0.156 | 0.155 | 0.154 | 0.147 | 0.143 | 0.146 | 0.129 |

## Largest correlation changes (stress - normal)

| up most | dRho | down most | dRho |
|---|---|---|---|
| VNQ-DBC | +0.35 | TLT-LQD | -0.59 |
| SPY-DBC | +0.30 | TIP-LQD | -0.43 |
| DBC-LQD | +0.30 | VNQ-TLT | -0.31 |
| EFA-LQD | +0.25 | HYG-TLT | -0.27 |
| EFA-DBC | +0.24 | EFA-TLT | -0.19 |
| SPY-LQD | +0.24 | SPY-TLT | -0.18 |

## IRF (corr response to +1 SD VIX shock)

- VAR lag = 4 (differenced)
- peak response = +0.0004; cumulative = +0.0007

## PCA loadings

| asset | PC1 | PC2 | PC3 | PC4 |
|---|---|---|---|---|
| SPY | +0.47 | -0.13 | -0.16 | -0.02 |
| EFA | +0.48 | -0.08 | -0.01 | -0.17 |
| VNQ | +0.41 | -0.08 | -0.20 | -0.21 |
| HYG | +0.41 | +0.04 | -0.26 | +0.17 |
| DBC | +0.30 | -0.00 | +0.41 | +0.68 |
| TLT | -0.12 | +0.56 | -0.18 | -0.09 |
| TIP | +0.02 | +0.56 | -0.06 | +0.14 |
| LQD | +0.18 | +0.47 | -0.32 | +0.11 |
| GLD | +0.13 | +0.28 | +0.59 | +0.06 |
| UUP | -0.23 | -0.21 | -0.46 | +0.63 |

_variance: PC1 37%, PC2 24%, PC3 13%, PC4 7%_
