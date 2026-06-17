"""
analysis.py
-----------
Linking the DCC correlation to liquidity. Three layers, increasing strength:

  1. Regression   Fisher-z(rho_t) on liquidity proxies (+ lags), HAC/Newey-West SE.
                  z = arctanh(rho) maps (-1,1) -> R so OLS is well behaved.
  2. Regime       Mean correlation on worst-liquidity days vs the rest (Welch t-test).
  3. VAR / IRF    Does a liquidity shock LEAD a rise in correlation? (direction,
                  not proof of causation -- depends on Cholesky ordering.)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def fisher_z(rho: pd.Series) -> pd.Series:
    r = rho.clip(-0.999, 0.999)
    return np.arctanh(r).rename("z")


def regress_on_liquidity(rho: pd.Series, liq: pd.DataFrame, lags=1, hac_lags=10,
                         add_trend=False):
    """Fisher-z(rho) on liquidity proxies (+lags), HAC SE. If add_trend, include a
    normalized linear time trend as a control -- this strips out the secular rise in
    correlation so the liquidity coefficient reflects within-time variation only."""
    z = fisher_z(rho)
    X = liq.copy()
    for L in range(1, lags + 1):
        for c in liq.columns:
            X[f"{c}_lag{L}"] = liq[c].shift(L)
    df = pd.concat([z, X], axis=1).dropna()
    y = df["z"]
    Xd = df.drop(columns="z")
    if add_trend:
        Xd = Xd.copy()
        Xd["trend"] = np.linspace(0.0, 1.0, len(Xd))   # normalized time index
    Xd = sm.add_constant(Xd)
    model = sm.OLS(y, Xd).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model


def pca_loadings(rets: pd.DataFrame, k=4):
    """Eigen-decomposition of the return-correlation matrix.
    Returns (loadings DataFrame [assets x PC1..PCk], explained-variance-ratio array).
    Each PC's sign is fixed so its largest-magnitude loading is positive (readability)."""
    C = rets.corr()
    vals, vecs = np.linalg.eigh(C.values)            # ascending
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    evr = vals / vals.sum()
    k = min(k, len(vals))
    L = vecs[:, :k].copy()
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] = -L[:, j]
    load = pd.DataFrame(L, index=C.columns, columns=[f"PC{i+1}" for i in range(k)])
    return load, evr[:k]


def regime_test(rho: pd.Series, liq_series: pd.Series, q=0.90):
    df = pd.concat([rho.rename("rho"), liq_series.rename("liq")], axis=1).dropna()
    thr = df["liq"].quantile(q)
    high = df.loc[df["liq"] >= thr, "rho"]      # worst-liquidity days
    low = df.loc[df["liq"] < thr, "rho"]
    t, p = stats.ttest_ind(high, low, equal_var=False)
    return {"threshold": thr, "mean_stress": high.mean(), "mean_normal": low.mean(),
            "diff": high.mean() - low.mean(), "t": t, "p": p,
            "n_stress": len(high), "n_normal": len(low)}


def var_irf(rho: pd.Series, liq_series: pd.Series, maxlags=5, periods=20):
    """VAR on [liquidity, correlation]; orthogonalized IRF of correlation to a
    one-SD liquidity shock. Cholesky ordering puts liquidity FIRST (treated as more
    exogenous) -- this ordering is an identifying assumption; report robustness.

    Returns a tidy dict so plotting code never touches statsmodels internals:
      horizons, resp (response of rho to a liq shock), lower/upper (95% band),
      cum (cumulative), lag, differenced, plus the raw res/irf objects.
    """
    from statsmodels.tsa.api import VAR
    df = pd.concat([liq_series.rename("liq"), rho.rename("rho")], axis=1).dropna()
    differenced = df["liq"].std() > 5          # difference if on a raw VIX-like scale
    if differenced:
        df = df.diff().dropna()
    res = VAR(df).fit(maxlags=maxlags, ic="aic")
    irf = res.irf(periods)
    resp = np.asarray(irf.orth_irfs)[:, 1, 0]  # response of rho (1) to liq shock (0)
    lower = upper = None
    try:
        se = np.asarray(irf.stderr(orth=True))[:, 1, 0]
        lower, upper = resp - 1.96 * se, resp + 1.96 * se
    except Exception:
        pass
    return {"res": res, "irf": irf, "horizons": np.arange(len(resp)),
            "resp": resp, "lower": lower, "upper": upper, "cum": np.cumsum(resp),
            "lag": int(res.k_ar), "differenced": bool(differenced)}


def effective_bets(corr: pd.DataFrame) -> float:
    """Effective number of independent bets = participation ratio of the
    correlation eigenvalues, 1 / sum(p_i^2) where p_i are normalized eigenvalues.
    Equals N if all assets are independent, ~1 if everything is one factor."""
    eig = np.linalg.eigvalsh(corr.values)
    eig = eig[eig > 0]
    p = eig / eig.sum()
    return float(1.0 / np.sum(p ** 2))


def pc1_share(corr: pd.DataFrame) -> float:
    """Share of total variance explained by the top principal component."""
    eig = np.sort(np.linalg.eigvalsh(corr.values))[::-1]
    return float(eig[0] / eig.sum())


def concentration_regime(rets: pd.DataFrame, liq_series: pd.Series, q=0.90) -> dict:
    """Effective bets & PC1 share on normal days vs the worst-liquidity decile.
    The sign-robust version of the study: if diversification breaks down in
    stress, effective bets FALLS and PC1 share RISES on the worst-liquidity days."""
    df = rets.join(liq_series.rename("liq"), how="inner").dropna()
    thr = df["liq"].quantile(q)
    cols = list(rets.columns)
    Cn = df.loc[df["liq"] < thr, cols].corr()
    Cs = df.loc[df["liq"] >= thr, cols].corr()
    return {"eff_normal": effective_bets(Cn), "eff_stress": effective_bets(Cs),
            "pc1_normal": pc1_share(Cn), "pc1_stress": pc1_share(Cs),
            "n_assets": len(cols)}


def rolling_pc1_share(rets: pd.DataFrame, window=126) -> pd.Series:
    """Rolling PC1 variance share (correlation within each trailing window).
    window=126 ~ 6 months of trading days. Spikes mark systemic-risk episodes."""
    arr = rets.values
    idx = rets.index
    out_idx, out_val = [], []
    for end in range(window, len(arr) + 1):
        C = np.corrcoef(arr[end - window:end], rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(C))[::-1]
        out_idx.append(idx[end - 1])
        out_val.append(eig[0] / eig.sum())
    return pd.Series(out_val, index=pd.DatetimeIndex(out_idx), name="pc1_share")
