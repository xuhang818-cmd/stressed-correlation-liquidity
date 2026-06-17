# THEORY — Stressed Correlation & Liquidity

Mathematical appendix for the project. Each part: what the method does, the model,
the derivation of the key results, the estimator, and the assumptions / failure
modes. Primary academic sources are cited inline and collected at the end.

Notation: `r_t` return, `ε_t` mean-zero innovation, `σ_t²` conditional variance,
`z_t = ε_t/σ_t` standardized residual, `F_{t-1}` information through `t−1`.
`E[·]` expectation, `'`/`ᵀ` transpose, bold lowercase = vectors, uppercase = matrices.

---

## Part 0 — Research question and economic framing

We ask whether **cross-asset correlation rises with market/funding stress**, and
whether **liquidity stress leads correlation** (a lead–lag, not a causal, claim).

The economic motivation is the funding↔market-liquidity spiral of Brunnermeier &
Pedersen (2009): when funding tightens, leveraged players de-risk together, which
can raise co-movement. The empirical hazard is the conditioning bias of Forbes &
Rigobon (2002): correlations measured only on high-volatility days are mechanically
inflated, so "correlations rise in crises" must be tested carefully (Longin & Solnik
2001; Ang & Chen 2002 document the down-market asymmetry).

---

## Part 1 — Univariate GARCH(1,1)

**Purpose.** Estimate each asset's time-varying volatility so it can be stripped
out, leaving standardized residuals whose co-movement is comparable across assets.

**Model.** Decompose the return into a conditional mean plus an innovation:

    r_t = μ_t + ε_t,   μ_t = E[r_t | F_{t-1}],   ε_t = σ_t · z_t,
    z_t iid, E[z_t]=0, Var(z_t)=1.

So `σ_t² = Var(ε_t | F_{t-1})` is the conditional variance, specified as

    σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²,    ω>0, α≥0, β≥0.

`α·ε_{t-1}²` is the ARCH term — reaction to the latest squared shock (Engle 1982);
`β·σ_{t-1}²` is the GARCH term — persistence of variance (Bollerslev 1986).

**Stationarity & unconditional variance (derivation).** Take unconditional
expectations and write `σ² = E[σ_t²] = E[ε_t²]` (constant under covariance
stationarity). By the law of iterated expectations,
`E[ε_{t-1}²] = E[ E[ε_{t-1}²|F_{t-2}] ] = E[σ_{t-1}²] = σ²`. Hence

    σ² = ω + α·σ² + β·σ²   ⟹   σ² = ω / (1 − α − β).

This is finite and positive **iff `α + β < 1`** — the covariance-stationarity
condition. `α+β` is the persistence; the half-life of a variance shock is
`ln(0.5)/ln(α+β)`. `α+β=1` is IGARCH (shocks never die); `>1` diverges.

**Estimation — (Q)MLE.** With `z_t ~ N(0,1)`, `ε_t | F_{t-1} ~ N(0, σ_t²)`, so the
conditional log-likelihood (given initial values) is

    ℓ(θ) = Σ_t [ −½ln(2π) − ½·ln σ_t²(θ) − ε_t² / (2·σ_t²(θ)) ],   θ=(ω,α,β,…).

`σ_t²(θ)` is built recursively, so ℓ is nonlinear in θ and is maximized numerically
(BFGS/BHHH). Even if `z` is not truly Gaussian, maximizing the Gaussian likelihood
yields consistent estimates — **Quasi-MLE** (Bollerslev & Wooldridge 1992) — used
with robust (sandwich) standard errors; fat tails are often handled with a Student-t `z`.

**Output → DCC.** Standardized residuals `ẑ_t = ε_t / σ̂_t` should be ≈ iid(0,1) with
no remaining ARCH (check Ljung–Box on `ẑ_t²`). These feed the correlation step.

**Extensions.** Asymmetry (leverage effect): GJR-GARCH (Glosten, Jagannathan &
Runkle 1993), EGARCH (Nelson 1991).

---

## Part 2 — Dynamic Conditional Correlation (DCC)

**Purpose.** Turn the per-asset volatilities and standardized residuals into a
**time-varying correlation matrix** `R_t`, tracking its average off-diagonal `ρ_t`.

**Decomposition.** Stack `z_t = (z_{1,t},…,z_{N,t})'`. The conditional covariance is

    H_t = D_t · R_t · D_t,    D_t = diag(σ_{1,t},…,σ_{N,t}).

`D_t` (volatilities) comes from N univariate GARCHs; `R_t` (correlations) is modeled
separately. This split is the origin of the "two-step" estimator and is the key idea
of Engle (2002), generalizing Bollerslev's (1990) Constant Conditional Correlation.

**Dynamics.** `R_t` cannot be recursed directly (it must remain a valid correlation
matrix), so an auxiliary matrix `Q_t` follows a GARCH-type recursion:

    Q_t = (1 − a − b)·Q̄ + a·(z_{t-1} z_{t-1}ᵀ) + b·Q_{t-1},   a,b ≥ 0, a+b < 1,

- `Q̄ = E[z_t z_tᵀ] ≈ (1/T)Σ z_t z_tᵀ` — unconditional (≈correlation) matrix of the
  standardized residuals; the mean-reversion anchor ("correlation targeting").
- `z_{t-1} z_{t-1}ᵀ` — outer product; entry (i,j) is `z_{i,t-1} z_{j,t-1}`, the
  realized co-movement "news" (ARCH analog, weight `a`).
- `Q_{t-1}` — persistence (GARCH analog, weight `b`).
- `a, b` are scalars: all pairs share one set of dynamics → only 2 parameters.

Targeting is consistent: taking expectations, `E[Q_t] = (1−a−b)Q̄ + aQ̄ + bQ̄ = Q̄`,
so `Q_t` mean-reverts to `Q̄` (cf. GARCH reverting to `ω/(1−α−β)`).

**Normalization `Q_t → R_t`.** `Q_t`'s diagonal is not exactly 1, so rescale:

    R_t = (diag Q_t)^{−1/2} · Q_t · (diag Q_t)^{−1/2},
    i.e.  ρ_{ij,t} = q_{ij,t} / √(q_{ii,t}·q_{jj,t}).

This forces unit diagonal and off-diagonals in [−1,1].

**Why `R_t` is positive-definite (the guarantee).**
1. `Q_t` is a nonnegative combination of (semi-)definite matrices: `(1−a−b)Q̄` is PD
   (Q̄ PD, `1−a−b>0`), `a·z z'` is rank-1 PSD, `b·Q_{t-1}` is PD by induction
   (initialize `Q_0 = Q̄`). PD + PSD + PD ⟹ `Q_t` PD.
2. Normalization is a **congruence** `R_t = D̃ Q_t D̃` with `D̃ = (diag Q_t)^{−1/2}`
   diagonal and positive; congruence by an invertible matrix preserves PD.

So `R_t` is a valid (PD, unit-diagonal) correlation matrix at every `t`,
automatically — two scalars buy a guaranteed-valid time-varying correlation.

**Estimation — two-step QMLE.** The Gaussian log-likelihood (mean-zero `r_t`)

    L = −½ Σ_t [ N·ln(2π) + ln|H_t| + r_tᵀ H_t^{−1} r_t ]

separates because `ln|H_t| = 2·ln|D_t| + ln|R_t|` and
`r_tᵀ H_t^{−1} r_t = (D_t^{−1}r_t)ᵀ R_t^{−1} (D_t^{−1}r_t) = z_tᵀ R_t^{−1} z_t`:

- **Step 1:** estimate each univariate GARCH → `D_t`, `z_t`.
- **Step 2:** treating `z_t` as data, maximize the correlation block
  `L_c = −½ Σ_t [ ln|R_t| + z_tᵀ R_t^{−1} z_t − z_tᵀ z_t ]` over `(a,b)`.

Two-step is **consistent but not fully efficient**; standard errors need the
Engle–Sheppard (2001) / sandwich correction. The payoff is scalability: N univariate
GARCHs + a 2-parameter correlation step, versus full multivariate GARCH (BEKK; Engle
& Kroner 1995) whose parameter count is `O(N²)`–`O(N⁴)`.

**Assumptions / failure modes.** Scalar `a,b` impose identical dynamics on all pairs;
relax with asymmetric/generalized DCC (Cappiello, Engle & Sheppard 2006), which also
lets correlations rise more in down-markets. Targeting needs `Q̄` well estimated
(biased when `N` large / `T` small). Gaussian QMLE is consistent under non-normality
but use robust SEs; fat tails → Student-t DCC.

---

## Part 3 — Fisher z-transform regression with HAC standard errors

**Purpose.** Regress the DCC correlation `ρ_t` on a liquidity-stress proxy (and its
lags) to measure the sign/size of the relationship.

**Why transform `ρ_t`.** A correlation is bounded in [−1,1], and for bivariate-normal
data the sample correlation has `Var(r) ≈ (1−ρ²)²/n` — its variance depends on `ρ`
(heteroskedastic) and it is skewed near ±1. The Fisher (1915, 1921) transform

    z = arctanh(r) = ½·ln[(1+r)/(1−r)]

fixes both. By the delta method with `g(ρ)=arctanh(ρ)`, `g'(ρ)=1/(1−ρ²)`:

    Var(z) ≈ [g'(ρ)]² · Var(r) = [1/(1−ρ²)]² · (1−ρ²)²/n = 1/n   (≈ 1/(n−3)).

The `ρ`-dependence cancels — the transform is **variance-stabilizing** and maps
[−1,1] onto ℝ, so a linear model is appropriate (and won't predict outside [−1,1]).
We regress `arctanh(ρ_t)` on the proxy.

**Why HAC standard errors.** For OLS `y_t = x_t'β + u_t`, `β̂` is consistent, but its
true covariance is the sandwich `(X'X)^{−1}(X'ΩX)(X'X)^{−1}` with `Ω = Var(u)`. When
`u_t` is autocorrelated and/or heteroskedastic — which it is here, since `ρ_t` is
highly persistent — the textbook `σ²(X'X)^{−1}` is wrong (too small ⟹ t-stats too
large ⟹ false significance). The Newey & West (1987) HAC estimator of the middle term

    S = Γ₀ + Σ_{l=1}^{L} w_l (Γ_l + Γ_l'),   Γ_l = (1/T)Σ_t (u_t u_{t−l}) x_t x_{t−l}',
    w_l = 1 − l/(L+1)   (Bartlett kernel, guarantees PSD),

with bandwidth `L`, yields heteroskedasticity- and autocorrelation-consistent SEs
(bandwidth choice: Andrews 1991). Always report the HAC t, not the OLS t, for `ρ_t`.

---

## Part 4 — Regime test (Welch's t)

**Purpose.** Compare mean correlation on "normal" days vs the worst-liquidity decile.

Split the sample at the 90th percentile of the stress proxy. With group means
`x̄₁,x̄₂`, variances `s₁²,s₂²`, sizes `n₁,n₂`, test `H₀: μ₁ = μ₂` with **Welch's
(1947)** statistic

    t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂),

using the Welch–Satterthwaite degrees of freedom. Welch (not pooled Student's t)
because the two regimes have unequal variances and sizes. Caveat: pooling across the
whole sample mixes in any secular trend (see Limitations).

---

## Part 5 — PCA and the effective number of bets

**Purpose.** Decompose returns into orthogonal factors to (i) check how many
independent risk dimensions the basket really has, and (ii) measure factor
concentration in a sign-robust way.

**Construction.** Standardize each asset and form the correlation matrix `C`. The
variance of the data projected onto a unit direction `w` is `wᵀCw`. Maximizing
subject to `‖w‖=1` (Lagrangian) gives the eigenvalue equation

    C w = λ w.

So the principal components are the **eigenvectors** of `C` (their entries are the
loadings), and each **eigenvalue `λ`** is the variance along that direction. Order by
eigenvalue: PC1 = largest. Since `C` is a correlation matrix, `Σλ = trace(C) = N`, so
PC1's variance share is `λ₁/N`. (Pearson 1901; Hotelling 1933.)

**Effective number of bets.** With `pᵢ = λᵢ/Σλ`, the code uses the participation
ratio (inverse Herfindahl)

    N_eff = 1 / Σ pᵢ²,

which equals `N` when all factors are equal and ≈1 when one dominates. A related
diversification measure is Meucci's (2009) entropy form `exp(−Σ pᵢ ln pᵢ)`; both
capture concentration. Rising PC1 share / falling `N_eff` in stress is the
sign-robust statement of "correlations → 1" — it counts a strongly *negatively*
loaded haven as still being driven by the same factor.

---

## Part 6 — VAR / IRF (lead–lag and identification)

**Purpose.** Ask whether a liquidity shock *leads* a rise in correlation.

**Reduced-form VAR(p)** (Sims 1980), with `y_t = (ρ_t-transform, proxy)'`:

    y_t = c + Σ_{i=1}^p A_i y_{t−i} + u_t,   Cov(u_t) = Σ_u.

`u_t` are reduced-form residuals; `Σ_u` is generally **not diagonal**, so a "shock to
one variable" is not ceteris paribus — the system must be identified.

**Identification (Cholesky).** Write `u_t = B ε_t` with orthogonal structural shocks
`Cov(ε_t)=I`. The Cholesky factorization `Σ_u = P P'` (lower-triangular `P`) gives
`ε_t = P^{−1} u_t`. This imposes a **recursive contemporaneous ordering**: the first
variable can affect all others within the period, the last affects none
contemporaneously. **The ordering is an identifying assumption** — the IRF can change
with it, so report robustness to the ordering (and/or use it only for direction).

**Impulse response.** Invert the VAR to its Wold/MA(∞) form `y_t = Σ_h Θ_h u_{t−h}`;
the orthogonalized IRF at horizon `h` is `Θ_h P`, giving the path of `ρ_t` after a
one-standard-deviation liquidity shock. For the pure predictive (not structural)
lead–lag question, Granger (1969) causality is the lighter-weight complement.

Textbook reference for VAR/IRF mechanics: Lütkepohl (2005).

---

## Limitations / identification

- **Secular trend confound.** Average correlation trends up over the sample, and the
  largest stress spikes sit early; a pooled cross-section can mix the trend into the
  stress effect. Control with a time trend, or use event-window/IRF designs.
- **Conditioning bias.** Correlations on selected (high-vol) subsamples are
  mechanically inflated (Forbes & Rigobon 2002) — correct or state it.
- **Correlation ≠ causation.** Same-direction + lead + regime dependence is
  suggestive; a clean causal claim needs an identified shock.
- **Proxy and basket sensitivity.** Results depend on the stress proxy (VIX vs credit
  spreads) and on basket composition (signed-average vs factor-concentration lenses
  can disagree).

---

## References

- Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation consistent
  covariance matrix estimation. *Econometrica*, 59(3), 817–858.
- Ang, A. & Chen, J. (2002). Asymmetric correlations of equity portfolios.
  *Journal of Financial Economics*, 63(3), 443–494.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
  *Journal of Econometrics*, 31(3), 307–327.
- Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange rates:
  a multivariate generalized ARCH model (CCC). *Review of Economics and Statistics*,
  72(3), 498–505.
- Bollerslev, T. & Wooldridge, J.M. (1992). Quasi-maximum likelihood estimation and
  inference in dynamic models with time-varying covariances. *Econometric Reviews*,
  11(2), 143–172.
- Brunnermeier, M.K. & Pedersen, L.H. (2009). Market liquidity and funding liquidity.
  *Review of Financial Studies*, 22(6), 2201–2238.
- Cappiello, L., Engle, R.F. & Sheppard, K. (2006). Asymmetric dynamics in the
  correlations of global equity and bond returns. *Journal of Financial
  Econometrics*, 4(4), 537–572.
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates of
  the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- Engle, R.F. (2002). Dynamic conditional correlation. *Journal of Business &
  Economic Statistics*, 20(3), 339–350.
- Engle, R.F. & Kroner, K.F. (1995). Multivariate simultaneous generalized ARCH
  (BEKK). *Econometric Theory*, 11(1), 122–150.
- Engle, R.F. & Sheppard, K. (2001). Theoretical and empirical properties of dynamic
  conditional correlation multivariate GARCH. *NBER Working Paper 8554*.
- Fisher, R.A. (1915). Frequency distribution of the values of the correlation
  coefficient in samples from an indefinitely large population. *Biometrika*, 10(4),
  507–521. (z-transform developed further in Fisher 1921, *Metron* 1, 3–32.)
- Forbes, K.J. & Rigobon, R. (2002). No contagion, only interdependence: measuring
  stock market comovements. *Journal of Finance*, 57(5), 2223–2261.
- Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993). On the relation between the
  expected value and the volatility of the nominal excess return on stocks (GJR).
  *Journal of Finance*, 48(5), 1779–1801.
- Granger, C.W.J. (1969). Investigating causal relations by econometric models and
  cross-spectral methods. *Econometrica*, 37(3), 424–438.
- Hotelling, H. (1933). Analysis of a complex of statistical variables into principal
  components. *Journal of Educational Psychology*, 24, 417–441, 498–520.
- Longin, F. & Solnik, B. (2001). Extreme correlation of international equity markets.
  *Journal of Finance*, 56(2), 649–676.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Meucci, A. (2009). Managing diversification. *Risk*, 22(5), 74–79.
- Nelson, D.B. (1991). Conditional heteroskedasticity in asset returns: a new approach
  (EGARCH). *Econometrica*, 59(2), 347–370.
- Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55(3), 703–708.
- Pearson, K. (1901). On lines and planes of closest fit to systems of points in
  space. *Philosophical Magazine*, 2(11), 559–572.
- Sims, C.A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1–48.
- Welch, B.L. (1947). The generalization of "Student's" problem when several different
  population variances are involved. *Biometrika*, 34(1–2), 28–35.
