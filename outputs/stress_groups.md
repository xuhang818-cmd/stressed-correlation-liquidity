# Stress-regime asset grouping

## Stress-day overlap (VIX vs FUNDING)

- VIX stress days: 483
- FUNDING stress days: 484
- shared: 163  |  Jaccard = 0.20

## Groups under VIX stress  (merge if corr > 0.50)

**Normal days:**
- group 1: SPY, EFA, VNQ, HYG
- group 2: TLT, TIP, LQD
- group 3: DBC
- group 4: GLD
- group 5: UUP

**Worst-9% VIX days:**
- group 1: SPY, EFA, VNQ, HYG, DBC
- group 2: TLT, TIP
- group 3: LQD
- group 4: GLD
- group 5: UUP

## Groups under FUNDING stress  (merge if corr > 0.50)

**Normal days:**
- group 1: SPY, EFA, VNQ, HYG
- group 2: TLT, TIP, LQD
- group 3: DBC
- group 4: GLD
- group 5: UUP

**Worst-9% FUNDING days:**
- group 1: SPY, EFA, VNQ
- group 2: TLT, TIP
- group 3: HYG, LQD
- group 4: DBC
- group 5: GLD
- group 6: UUP
