"""
data_loader.py
--------------
Two ways to get data:
  load_real()      -> pulls asset prices (yfinance) + liquidity proxies (FRED).
  simulate()       -> generates correlated GARCH returns with a regime-dependent
                      correlation plus a liquidity index, so the whole pipeline
                      can be demonstrated offline and the link is known by design.

Returns are LOG returns. Everything is aligned on a common daily index.
"""
import numpy as np
import pandas as pd


def _load_csv_series(path):
    """Load a local two-column [date, value] CSV as a daily Series.

    For proxies with no free programmatic feed (e.g. ICE BofA HY OAS exported
    from Bloomberg). '.' / blank values coerce to NaN.
    """
    raw = pd.read_csv(path)
    date_col, val_col = raw.columns[0], raw.columns[1]
    idx = pd.to_datetime(raw[date_col])
    s = pd.Series(pd.to_numeric(raw[val_col], errors="coerce").values, index=idx, name=val_col)
    return s.sort_index()


def load_real(assets, liq_yf, liq_csv, start, end=None):
    import yfinance as yf

    print("Downloading prices from Yahoo Finance ...", flush=True)
    px = yf.download(assets, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all").ffill().dropna()
    rets = np.log(px).diff().dropna()

    liq_cols = {}

    if liq_yf:
        tickers = list(liq_yf.values())
        print(f"Downloading market-stress proxies from Yahoo Finance: {tickers} ...", flush=True)
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):           # single ticker -> Series
            raw = raw.to_frame(tickers[0])
        for name, tk in liq_yf.items():
            liq_cols[name] = raw[tk]

    for name, path in (liq_csv or {}).items():
        print(f"Loading local proxy '{name}' from {path} ...", flush=True)
        liq_cols[name] = _load_csv_series(path)

    liq = pd.DataFrame(liq_cols).reindex(rets.index).ffill()

    df = rets.join(liq, how="inner").dropna()
    print(f"Loaded {df.shape[0]} aligned days | proxies: {list(liq.columns)}", flush=True)
    return df[assets], df[list(liq_cols.keys())]


def simulate(n_assets=5, T=2500, seed=7):
    """
    Build returns whose pairwise correlation is HIGH in a 'stress' regime and
    LOW otherwise, driven by a single common factor whose loading rises in stress.
    The liquidity proxy is high (bad) precisely during the stress regime, so a
    correct pipeline must recover a positive corr <-> liquidity relationship.
    """
    rng = np.random.default_rng(seed)

    # Slow-moving stress state (blocks of calm punctuated by stress episodes).
    stress = np.zeros(T)
    t = 0
    while t < T:
        calm = rng.integers(150, 400)
        t += calm
        if t >= T:
            break
        episode = rng.integers(30, 120)
        stress[t:min(t + episode, T)] = 1.0
        t += episode

    # Common-factor loading: low in calm (~0.2), high in stress (~0.8).
    load = np.where(stress > 0, 0.80, 0.20)

    # GARCH(1,1) volatility for the common factor and idiosyncratics.
    def garch_path(omega, alpha, beta, T):
        eps = np.zeros(T)
        sig2 = np.zeros(T)
        sig2[0] = omega / (1 - alpha - beta)
        for i in range(1, T):
            sig2[i] = omega + alpha * eps[i - 1] ** 2 + beta * sig2[i - 1]
            eps[i] = np.sqrt(sig2[i]) * rng.standard_normal()
        eps[0] = np.sqrt(sig2[0]) * rng.standard_normal()
        return eps

    f = garch_path(0.02, 0.08, 0.90, T)               # common factor
    rets = np.zeros((T, n_assets))
    for j in range(n_assets):
        idio = garch_path(0.02, 0.05, 0.92, T)
        rets[:, j] = load * f + np.sqrt(np.clip(1 - load ** 2, 0, None)) * idio
    rets = rets / 100.0                                # to ~daily-return scale

    idx = pd.bdate_range("2010-01-01", periods=T)
    cols = [f"A{j+1}" for j in range(n_assets)]
    rets = pd.DataFrame(rets, index=idx, columns=cols)

    # Liquidity proxy: baseline + jump in stress + noise (higher = worse liquidity).
    liq = 1.0 + 4.0 * stress + 0.5 * np.abs(rng.standard_normal(T))
    liq = pd.Series(liq, index=idx, name="LIQ_STRESS").rolling(5).mean().bfill()
    liq = liq.to_frame()
    return rets, liq


# ===================== Day 2: spliced funding proxy =====================
# A continuous funding-stress proxy from local FRED CSVs. Legacy leg: TED spread
# (TEDRATE, pre-2022). Current leg: either a ready-made spread, or built here from
# two raw files via current_spread=(DCPF3M.csv, SOFR.csv) -> CP minus SOFR (unsecured
# minus secured), the closest modern analogue to TED. You only download the raw FRED
# files; the subtraction and date-alignment happen in code. Splice on SCALE, then
# inject as a "FUNDING" column -- analysis.py treats every liquidity column
# generically. Numerically continuous != economically identical (TED carries bank
# credit risk; SOFR is secured), so use it as a robustness proxy. Run
# diagnose_funding_overlap() first to pick method ('overlap' vs 'zscore').
def _build_current(current_csv=None, current_spread=None):
    """Build the post-2022 'current' funding leg.

    current_csv     : one ready-made spread CSV; OR
    current_spread  : (minuend_csv, subtrahend_csv) -> the code aligns them on
                      common dates and subtracts, e.g. ("DCPF3M.csv", "SOFR.csv")
                      gives the CP - SOFR spread (unsecured minus secured). You only
                      download the two raw FRED files; no manual subtraction needed.
    """
    if (current_csv is None) == (current_spread is None):
        raise ValueError("pass exactly one of current_csv or current_spread")
    if current_csv is not None:
        return _load_csv_series(current_csv).sort_index()
    minuend, subtrahend = current_spread
    a = _load_csv_series(minuend)
    b = _load_csv_series(subtrahend)
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()   # align on common dates
    spread = (j["a"] - j["b"]).sort_index()
    spread.name = "CURRENT_SPREAD"
    return spread


def diagnose_funding_overlap(legacy_csv, current_csv=None, current_spread=None,
                             overlap=("2018-04-01", "2022-01-21")):
    """Check whether the two legs line up linearly over their overlap window.
    Prints n / R^2 / slope / intercept and returns them. High R^2 -> method='overlap'
    is justified; low R^2 -> the legs behave differently, use method='zscore'."""
    legacy = _load_csv_series(legacy_csv).sort_index()
    current = _build_current(current_csv, current_spread)
    j = pd.concat([legacy.rename("L"), current.rename("C")], axis=1).loc[
        overlap[0]:overlap[1]].dropna()
    n = len(j)
    if n < 30:
        print(f"Overlap diagnostic: only {n} common days in {overlap} -- too short; "
              f"use method='zscore'.")
        return {"n": n, "r2": float("nan"), "slope": float("nan"), "intercept": float("nan")}
    a, b = np.polyfit(j["C"].values, j["L"].values, 1)
    pred = a * j["C"].values + b
    ss_res = ((j["L"].values - pred) ** 2).sum()
    ss_tot = ((j["L"].values - j["L"].values.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"Overlap {overlap}: n={n}, R^2={r2:.3f}, legacy = {a:+.3f}*current {b:+.3f}")
    print("  -> R^2 high (>~0.6): method='overlap' OK.  R^2 low: use method='zscore'.")
    return {"n": n, "r2": float(r2), "slope": float(a), "intercept": float(b)}


def load_spliced_funding(legacy_csv, current_csv=None, current_spread=None,
                         split="2022-01-21", method="zscore",
                         overlap=("2018-04-01", "2022-01-21")):
    """Return one daily 'FUNDING' Series spliced from a legacy leg and a current leg.

    legacy_csv      : the pre-2022 leg, e.g. "data/TEDRATE.csv".
    current_csv     : a ready-made post-2022 spread CSV; OR
    current_spread  : (minuend_csv, subtrahend_csv), e.g.
                      ("data/DCPF3M.csv", "data/SOFR.csv") -> CP - SOFR is built here.

    method='zscore'  : standardize each leg by its own mean/sd, then join at `split`
                       (puts both on a 'standard deviations of stress' scale).
    method='overlap' : fit an affine map y=a*x+b of the current leg onto the legacy
                       leg over the overlap window, apply it, then join at `split`
                       (keeps the legacy leg's units).
    """
    legacy = _load_csv_series(legacy_csv).sort_index()
    current = _build_current(current_csv, current_spread)

    if method == "zscore":
        lz = (legacy - legacy.mean()) / legacy.std()
        cz = (current - current.mean()) / current.std()
        out = pd.concat([lz.loc[:split], cz.loc[split:]])
    elif method == "overlap":
        j = pd.concat([legacy.rename("L"), current.rename("C")], axis=1).loc[
            overlap[0]:overlap[1]].dropna()
        if len(j) < 30:
            raise ValueError("overlap window too short to calibrate; use method='zscore'")
        a, b = np.polyfit(j["C"].values, j["L"].values, 1)   # map current -> legacy scale
        out = pd.concat([legacy.loc[:split], (a * current + b).loc[split:]])
    else:
        raise ValueError("method must be 'zscore' or 'overlap'")

    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.name = "FUNDING"
    return out


def inject_funding(liq_df, funding_series):
    """Add the spliced FUNDING column to the liquidity frame.

    Usage (configured via config.FUNDING_SPLICE):
        rets, liq = load_real(ASSETS, LIQUIDITY_YF, LIQUIDITY_CSV, START, END)
        liq = inject_funding(liq, load_spliced_funding(**C.FUNDING_SPLICE))
    """
    out = liq_df.copy()
    out["FUNDING"] = funding_series.reindex(liq_df.index).ffill()
    return out
# ======================================================================
