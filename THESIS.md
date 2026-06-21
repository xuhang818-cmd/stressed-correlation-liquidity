# Stressed Correlation & Liquidity
### Does cross-asset diversification hold up when markets are stressed — and does liquidity stress lead correlation?

Math derivations and citations: see `THEORY.md`. Per-figure documentation: see
`FIGURE_GUIDE.md`. Figures referenced below are in `outputs/`.

> **The argument, in order.** (1) State the question and why it matters. (2) Show why
> you cannot answer it by eyeballing prices — you need a *time-varying* correlation, so
> we build one (GARCH → DCC). (3) Test whether that correlation moves with stress
> (regression, dose-response, regime). (4) The answer is surprising — the average
> barely moves — so we open the matrix to see *why* (structure). (5) Reconcile the flat
> average with a sign-robust factor lens (PCA: concentration). (6) Ask about direction
> (VAR/IRF). (7) State limitations honestly. Each step exists because the previous one
> left a question open.

---

## 1. Question and motivation

Practitioners assume a diversified book protects them in a crisis. The folk claim is
the opposite — "in a crash, all correlations go to 1" — which would mean diversification
fails exactly when it is needed. We test this on a broad multi-asset basket, and also
ask the directional question: does **funding/liquidity stress lead** the rise in
co-movement, or follow it?

The mechanism behind a stress-driven rise is the funding↔market-liquidity spiral
(Brunnermeier & Pedersen 2009): tightening funding forces synchronized de-risking. The
measurement hazard is conditioning bias (Forbes & Rigobon 2002): correlations computed
only on turbulent days are mechanically inflated, so the claim must be tested with care.

---

## 2. Data and basket selection

**Assets (10):** SPY, EFA, VNQ, HYG, DBC, TLT, TIP, LQD, GLD, UUP — US/intl equity,
REITs, HY & IG credit, commodities, Treasuries, TIPS, gold, USD. Daily, 2007–present
(common-sample start set by the youngest ETFs). Stress proxy: VIX. Source: yfinance.

**Why these 10 — a PCA screen.** We began with a wider candidate set and pruned with
correlation clustering + PCA (`pca_screen.py`). Near-duplicate twins collapsed —
SPY≈IWM, EFA≈EEM, TLT≈IEF — leaving 10 economically distinct sleeves. Crucially, even
13 candidates carried only **~4 effective bets** (participation ratio of the correlation
eigenvalues): most apparent diversification is illusory, a result we report rather than
"fix" (over-pruning would remove the co-movement structure we want to study).

> *Now produced:* `pc_loadings.png` (and `pc_loadings.csv`) — the eigenvector weights of
> each asset on PC1–PC4 with the variance share of each, so you can name the factors
> (expected: PC1 risk-on/off, PC2 rates/duration, PC3 inflation/commodity, PC4 USD).

---

## 3. Estimating a time-varying correlation (GARCH → DCC)

You cannot read correlation off overlaid price lines, and a single full-sample number
hides all the dynamics. We need `ρ_t`, a correlation that updates daily.

Each asset's volatility clusters, so we first fit a **univariate GARCH(1,1)** per asset
and take standardized residuals `z_t = ε_t/σ_t` — this strips out each asset's own
volatility so that what remains is comparable "shocks." We then fit **DCC** (Engle 2002)
on the `z_t` to get the time-varying correlation matrix `R_t`; we track its average
off-diagonal `ρ_t`. (Why GARCH must come first, why DCC's two-step works, and why `R_t`
is always a valid correlation matrix: `THEORY.md` Parts 1–2.)

→ Figure: `corr_vs_liquidity.png` (ρ_t vs VIX, with an OLS time trend).

---

## 4. Does correlation move with stress?

Three complementary tests on the 10-asset basket:

- **Regression (Fisher-z + HAC).** Regress `arctanh(ρ_t)` on VIX. Slope is *negative*
  and significant: `≈ −0.001` (HAC t ≈ −2.5, p ≈ 0.01). Higher VIX → slightly *lower*
  average correlation.
- **Dose-response** (`dose_response.png`). Mean ρ across VIX deciles is essentially
  flat, edging down from ≈0.14 (calmest) to ≈0.13 (most stressed).
- **Regime test (Welch t)** (`regime.png`). Normal 0.152 vs worst-VIX-decile 0.129;
  diff −0.023, t = −7.7, p ≈ 8e-14 — significant but **small**.

All three agree: for a properly diversified basket, the naive "average correlation
spikes in stress" does **not** hold. If anything it dips, modestly.

> *Now in the pipeline:* a trend-controlled version of the regression runs alongside
> the baseline; `tables.md` reports the VIX coefficient both ways (baseline vs + time
> trend). Read whether the negative VIX coefficient survives the trend control — if it
> shrinks toward zero / loses significance, the Section-4 effect was largely the secular
> trend. The trend-line slope/HAC-t on `corr_vs_liquidity.png` is also shown on the figure.

---

## 5. Why is the average flat? Open the matrix

A flat average can hide large, offsetting moves. `corr_matrices_regime.png` (normal vs
worst-decile correlation matrices) and the per-pair change `Δρ = ρ_stress − ρ_normal`
show it is the latter — the cross-section **reorganizes**:

| Pair group | mean Δρ | reading |
|---|---|---|
| Risk–risk (SPY/EFA/VNQ/HYG/DBC) | **+0.17** | risk assets converge (contagion) |
| Risk – TLT (Treasuries) | **−0.22** | flight-to-quality: havens decouple/turn negative |
| Rates block internal (TLT–LQD, TIP–LQD, TLT–TIP) | **−0.39** | rates block fragments |
| LQD – risk assets | **+0.20** | IG credit turns "risk-like" in stress |

Largest single movers: up — VNQ–DBC +0.35, SPY–DBC +0.30, SPY/EFA–LQD +0.24; down —
**TLT–LQD 0.80 → 0.21 (−0.59)**, TIP–LQD −0.43, VNQ–TLT −0.31, SPY–TLT −0.18.

So it is **not** "all correlations rise." Risk assets clump toward +1 while Treasuries
(and to a lesser extent the dollar) move the other way; the two roughly cancel in the
signed average. The flat number is the fingerprint of hedges *working*.

### 5b. Who lines up with whom: clustering the stress regime

`Δρ` tells us pairs move; clustering tells us into **which blocs**. On normal vs
worst-decile days we build the return-correlation matrix, cluster the assets
(1−corr distance, average linkage), and cut where corr > 0.50
(`stress_clustering_<proxy>.png`, membership in `stress_groups.md`). Three findings, and
the third is the most useful:

1. **The split is real and stable.** Under VIX stress the risk bloc consolidates —
   **SPY, EFA, VNQ, HYG and DBC** join one cluster (commodities, loosely attached in
   calm, get pulled in), while **TLT/TIP** stay a separate haven bloc and **GLD, UUP**
   stand alone. The dendrogram on stress days has a deep risk cluster that simply is not
   there on normal days.

2. **The two proxies flag almost *different* days.** VIX-stress days and FUNDING-stress
   days overlap by only **Jaccard ≈ 0.20** (163 of ~483 shared). This *strengthens* the
   robustness claim from Section 8: it is not that one stress definition is used twice —
   two largely independent stress definitions (equity-vol vs funding) both yield the same
   flat-average result. VIX is a market-stress gauge, FUNDING a financing one, and the
   data confirm they are not the same thing.

3. **Different stress sources take different transmission paths.** The blocs are not
   identical across proxies, and each split is economically coherent:
   - under **VIX (equity) stress**, *commodities (DBC)* are pulled into the risk bloc
     (bootstrap co-occurrence with SPY ≈ 0.76 — fairly stable), while IG credit (LQD)
     breaks out on its own — equity-vol contagion is a risk-asset phenomenon;
   - under **FUNDING stress**, *HY and IG credit (HYG, LQD) move toward each other*
     (co-occurrence ≈ 0.49 vs only ≈ 0.07 under VIX) — funding strain hits credit
     spreads directly. The direction is clear, but at 0.49 the grouping is **suggestive,
     not stable**: we report it as a tendency, not a firm cluster.

   This is the payoff of separating market- from funding-liquidity: "diversification
   breaks down" is not one mechanism but several, and which assets converge depends on
   *which* stress is hitting. Cluster stability (co-occurrence over 1,000 block
   bootstraps, `cluster_stability_<proxy>.png`) is reported so robust blocs (the core
   risk cluster, SPY–EFA ≈ 1.00) are not confused with fragile ones.

---

## 6. Reconcile with a sign-robust lens (factor concentration)

The signed average washes out opposing moves. A correlation matrix's **factor
concentration** does not: a strongly *negatively* loaded haven still loads on the same
factor. `diversification_regime.png` shows that, on the worst-VIX-decile days,

- effective bets fall **4.5 → 4.1**, and
- PC1 variance share rises **34% → 42%**.

So systemic co-movement *does* strengthen in stress — modestly, and visible only in the
factor lens, not the signed mean. `rolling_concentration.png` shows PC1 share is
strongly regime-dependent over time (range ≈30–63%, peaks around 2012 and 2022–23),
tracking macro regimes more than isolated VIX spikes. (Math: `THEORY.md` Part 5.)

**What PC1 actually is — a bipolar axis, not "everything falls together."** PC1 is the
risk-on/risk-off direction: risk assets load *positively*, havens (TLT, GLD, often UUP)
load *negatively* — read the first column of `pc_loadings.png`. A rising PC1 share does
not mean all assets drop in unison; it means more of the total variance is governed by
this *single* axis, with the basket splitting to its two poles (risk end down together,
haven end up together). The signed average correlation misses this because the positive
(risk–risk) and negative (risk–haven) pairs cancel; PC1 does not, because variance uses
the *square* of the loading, so a strongly negative haven still adds to concentration.
That is the mathematical reason the two metrics give opposite-looking pictures and both
are right.

**Is the shift real? (block bootstrap.)** A 1,000-replication block bootstrap (21-day
blocks, preserving volatility clustering; `robustness.py`) puts the stress−normal shift
at **−0.47 effective bets, 95% CI [−0.77, −0.17]** and **+8pp PC1 share, CI [+5, +11]**,
both with bootstrap p < 0.001. The direction is highly robust — but the magnitude is
*moderate* (≈4.5 → ≈4.0 bets): statistically firm, economically modest. The split is
also stable to the stress cutoff (85/95th pct give the same picture as 90th).

**Important scope condition.** This concentration result is a *market*-stress (VIX)
phenomenon. Re-run on the FUNDING proxy, the effective-bets change is ~0 or slightly
positive — funding-stress days do *not* show the same factor concentration. So
"diversification erodes in stress" holds for equity-vol stress, not for funding stress;
the two liquidity types are different mechanisms, not one effect seen twice.

**This reconciles the project:** "correlations → 1 in a crisis" is true as *factor
concentration*, not as *signed average correlation*; it is statistically real but
moderate, specific to market-stress, and sensitive to basket composition.

---

## 6b. So what? Why this matters for a portfolio

The practical payoff is a way to tell *real* diversification from *nominal*
diversification:

- **Holding more tickers is not holding more bets.** Ten assets here are worth only
  ~4.5 independent bets (the participation ratio). Adding another broad-equity ETF lows
  the count further toward redundancy, not safety — it doubles down on a bet you already
  hold.
- **Diversification has to be across *factors*, not asset labels.** Since stress is
  governed by one risk-on/off axis, an effective hedge must be roughly *orthogonal* to
  PC1 — an asset whose PC1 loading is near zero (GLD/UUP are the candidates here), not
  just "a different ticker."
- **The cushion thins exactly when needed.** Effective bets fall and PC1 share rises in
  market stress — diversification is weakest in the regime you bought it for. Note this
  is *factor concentration*, not the usual (and, here, false) "average correlation spikes"
  story.
- **Credit-desk hook.** If IG and HY are driven by the same spread factor in funding
  stress (the suggestive HYG–LQD signal, Section 5b), "I hold different ratings, so I'm
  diversified" can be an illusion. The framework turns "am I actually diversified?" into
  two measurable numbers — effective bets and PC1 share — and shows how they move by
  regime.

---

## 7. Direction: does liquidity stress lead correlation?

A reduced-form **VAR** on `(ρ_t, VIX)` with an orthogonalized **impulse response** asks
whether a VIX shock is followed by a rise in `ρ_t` over the next ~20 days. Identification
is by Cholesky ordering, which is itself an assumption — so we report robustness to the
ordering and read direction, not strict causality. (Math: `THEORY.md` Part 6.)

> *Now produced:* `irf_liquidity_to_corr.png` (orthogonalized IRF of `ρ_t` to a +1 SD
> VIX shock, 95% band, VIX ordered first). Read after your run: a **positive hump in the
> first days** = liquidity stress leads higher correlation; check robustness by flipping
> the Cholesky order. This is the cleanest test that side-steps the Section-4 trend
> confound, because it is a within-window dynamic. *[fill the sign/peak from your run]*

---

## 8. Robustness: a funding-liquidity proxy

VIX is a *market*-stress proxy — it blends volatility fear with genuine funding strain.
The Brunnermeier–Pedersen mechanism is specifically a *funding* one, so as a robustness
check we rebuild a continuous funding-stress series and re-run the whole pipeline on it.

The series is spliced from public FRED data: the legacy leg is the TED spread (TEDRATE,
which ends in Jan 2022 when USD LIBOR was retired) and the current leg is a CP−SOFR
spread (3-month financial commercial paper minus SOFR — unsecured minus secured, the
closest modern analogue to TED). Because the two legs differ in level and scale, they are
joined on a *standardized* (z-score) scale rather than spliced raw.

Two findings, both reported rather than hidden:

- **The two regimes are not the same series, numerically.** Over their 2018–2022 overlap,
  a linear fit of TED on CP−SOFR gives only **R² ≈ 0.38**. That is *why* the splice uses
  z-score, not an affine overlap map: LIBOR-era and SOFR-era funding stress are related
  but not interchangeable. The low R² is itself a result worth stating.
- **The conclusion is robust to the proxy.** Run on FUNDING, the headline picture is
  unchanged: the signed-average regime difference stays small/negative, and in the joint
  regression the FUNDING coefficient is insignificant (≈ −0.005, p ≈ 0.33). Crucially,
  this is *not* because the two proxies pick the same days — they overlap by only
  **Jaccard ≈ 0.20** (Section 5b). Two largely independent stress definitions — equity
  volatility and funding strain — independently produce the same flat-average result,
  which is stronger evidence than a single proxy could give. The pipeline emits a
  parallel set of figures per proxy (`*_VIX.png` and `*_FUNDING.png`) for side-by-side
  reading; the `corr_vs_liquidity_*` and `irf_*` pairs are where the proxies differ most.

> *Caveat:* numerically continuous ≠ economically identical (TED carries bank credit risk;
> SOFR is secured). FUNDING is a robustness lens, not the primary series. A still cleaner
> funding gauge (e.g. FRA-OIS or bank CDS) is a natural next refinement.

---

## 9. Limitations / identification

- **Secular trend confound.** `ρ_t` trends up over the sample and the biggest VIX spikes
  (2008, 2020) sit in the low-correlation early years, so the pooled Section-4 result
  may mix the trend into the stress effect. Add a time-trend control, or rely on the
  within-window IRF (Section 7).
- **Conditioning bias** (Forbes & Rigobon 2002): correlations on selected high-vol days
  are mechanically inflated — to correct or to state.
- **Correlation ≠ causation.** Same-sign + lead + regime dependence is suggestive; a
  clean causal claim needs an identified shock.
- **Proxy/basket sensitivity.** VIX vs credit-spread proxies, and the signed-average vs
  factor-concentration lenses, can disagree — a feature to report, not hide.

---

## 10. Conclusion *(DRAFT — your call)*

In a PCA-pruned, genuinely diversified basket, the textbook claim that *average*
cross-asset correlation spikes in stress is weak-to-absent (regime diff ≈ −0.02). The
real effect is structural and sign-robust: under stress the cross-section concentrates
onto fewer effective factors (PC1 34%→42%, effective bets 4.5→4.1) as risk assets
converge and Treasuries decouple. Diversification does not vanish, but it thins; and the
effect is modest and partly entangled with a strong secular rise in correlation, so the
causal/lead-lag reading awaits the IRF and a trend-controlled regression.

---

### Build status
- Done: PCA screen; GARCH→DCC ρ_t; Fisher-z/HAC regression (+ trend-controlled version);
  dose-response; Welch regime; correlation-matrix diagnostic; effective-bets/PC1 regime
  bars; rolling PC1; IRF figure; PC loadings figure; auto-exported data tables (markdown
  + CSV).
- Done (Day 2): spliced continuous funding proxy (`load_spliced_funding`, z-score or
  overlap-affine, from local FRED CSVs; built from three raw files via `current_spread`;
  activate via `config.FUNDING_SPLICE`; overlap R² diagnostic) and a self-contained
  interactive basket explorer (`outputs/basket_explorer.html`).
- Done: full pipeline re-run per liquidity proxy (VIX and FUNDING), parallel figure sets
  `*_VIX.png` / `*_FUNDING.png`; conclusion confirmed robust to the proxy.
- Done: explicit stress-regime asset grouping (`stress_clustering.py`) — VIX vs FUNDING
  stress-day overlap (Jaccard ≈ 0.20) timeline, and normal-vs-stress dendrograms + group
  tables per proxy; surfaced the two transmission paths (VIX→commodities into risk bloc;
  FUNDING→HY+IG credit merge).
- Done: robustness suite (`robustness.py`) — block-bootstrap CIs for the concentration
  shift (−0.47 eff bets, +8pp PC1, both p<0.001), cluster-stability co-occurrence
  matrices, and 85/90/95 threshold sensitivity incl. Jaccard; established the result is
  significant-but-moderate and market-stress-specific.
- Optional: name PC1–PC4 from the loadings; flip the IRF Cholesky order; FDR-correct the
  per-pair Δρ; a cleaner funding gauge (FRA-OIS / bank CDS); literature references.
