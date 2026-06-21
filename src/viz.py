"""viz.py -- the figures that carry the story (each with a caption strip)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY, RED, BLUE = "#1f3864", "#c0504d", "#7f9bbf"


def _ensure_writable(outpath):
    """Anchor a relative outpath to the project root (parent of this src/ folder) so
    output location does NOT depend on the current working directory, make sure the
    directory exists, and clear a read-only flag on an existing target. If the
    directory cannot be written, RAISE a clear error instead of silently writing
    elsewhere -- a silent fallback is exactly what hid stale PNGs before."""
    import stat
    if not os.path.isabs(outpath):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outpath = os.path.join(root, outpath)
    d = os.path.dirname(outpath) or "."
    os.makedirs(d, exist_ok=True)
    probe = os.path.join(d, ".write_test")
    try:
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as e:
        raise RuntimeError(
            f"Cannot write to '{d}'. This usually means OneDrive 'controlled folder "
            f"access' or a sync lock is blocking it. Move the project out of OneDrive "
            f"(e.g. C:\\dev\\scl) or allow Python through controlled-folder-access. "
            f"Underlying error: {e}")
    if os.path.exists(outpath):
        try:
            os.chmod(outpath, stat.S_IWRITE)
            os.remove(outpath)          # delete stale file so a failed write is visible
        except OSError:
            pass
    return outpath


def _savefig(fig, outpath):
    """Save and then VERIFY the file actually landed; raise loudly if it did not."""
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    if not os.path.exists(outpath):
        raise RuntimeError(f"savefig reported success but '{outpath}' is missing.")


def _save(fig, outpath, caption):
    """Add a one-line caption strip under the figure, then save (verified)."""
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=7.5,
             color="#555", wrap=True)
    _savefig(fig, outpath)


def plot_corr_vs_liquidity(avg_corr, liq_series, crises=None, outpath="outputs/corr_vs_liquidity.png"):
    """Headline chart: time-varying avg correlation vs a liquidity-stress proxy,
    plus an OLS trend line on the correlation series (the secular drift)."""
    outpath = _ensure_writable(outpath)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(avg_corr.index, avg_corr.values, color=NAVY, lw=1.2, label="Avg pairwise corr (DCC)")
    ax1.set_ylabel("Average pairwise correlation", color=NAVY)
    ax1.tick_params(axis="y", labelcolor=NAVY)

    # OLS time trend on the correlation series (naive t -- ignores autocorrelation)
    x = np.arange(len(avg_corr), dtype=float)
    y = avg_corr.values.astype(float)
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b0 + b1 * x
    n = len(x)
    s2 = float((y - yhat) @ (y - yhat)) / (n - 2)
    se = np.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    t_b1 = b1 / se
    per_yr = b1 * 252
    ax1.plot(avg_corr.index, yhat, color="black", ls="--", lw=1.6,
             label=f"OLS trend: {per_yr:+.3f}/yr (t={t_b1:.0f})")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(liq_series.index, liq_series.values, color=RED, lw=1.0, alpha=0.6,
             label=f"Liquidity stress ({liq_series.name})")
    ax2.set_ylabel(f"Liquidity stress ({liq_series.name})", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)

    if crises:
        for s, e in crises:
            ax1.axvspan(pd.to_datetime(s), pd.to_datetime(e), color="grey", alpha=0.15)

    ax1.set_title(f"Average pairwise DCC correlation vs liquidity stress ({liq_series.name})")
    cap = ("MEASURES: time-varying average pairwise correlation (navy, left axis) against the VIX stress proxy "
           "(red, right), crises shaded.  METHOD: DCC-GARCH conditional correlation averaged over off-diagonal "
           "pairs; dashed line is an OLS fit on time (slope per year + naive t).  ROLE: visual of co-movement vs "
           "stress and of the secular trend.")
    _save(fig, outpath, cap)
    return outpath


def plot_regime(regime, outpath="outputs/regime.png"):
    outpath = _ensure_writable(outpath)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.bar(["Normal", "Worst-liquidity\ndecile"],
           [regime["mean_normal"], regime["mean_stress"]], color=[BLUE, RED])
    ax.set_ylabel("Average pairwise correlation")
    ax.set_title(f"Correlation by liquidity regime\n(diff={regime['diff']:.3f}, "
                 f"t={regime['t']:.1f}, p={regime['p']:.1e})")
    cap = ("MEASURES: mean correlation on normal vs worst-VIX-decile days.  METHOD: split the sample at the 90th "
           "percentile of VIX; Welch t-test on the two means.  ROLE: the regime test -- does extreme stress raise "
           "(or lower) co-movement?")
    _save(fig, outpath, cap)
    return outpath


def plot_dose_response(avg_corr, liq_series, n_bins=10, stats=None,
                       outpath="outputs/dose_response.png"):
    """Mean correlation within liquidity-stress deciles: a dose-response view."""
    outpath = _ensure_writable(outpath)
    df = pd.concat([avg_corr.rename("rho"), liq_series.rename("liq")], axis=1).dropna()
    df["bin"] = pd.qcut(df["liq"], q=n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin")["rho"].mean()
    k = len(g)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(k), g.values, color=plt.cm.OrRd(np.linspace(0.30, 0.92, k)))
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(k))
    ax.set_xticklabels([str(i + 1) for i in range(k)])
    ax.set_xlabel(f"{liq_series.name} bin  (1 = calmest  ->  {k} = most stressed)")
    ax.set_ylabel("Mean pairwise correlation")
    direction = "rises with" if g.iloc[-1] >= g.iloc[0] else "FALLS with"
    title = f"Average correlation {direction} liquidity stress  ({g.iloc[0]:.2f} -> {g.iloc[-1]:.2f})"
    if stats:
        title += (f"\nFisher-z regression: slope={stats['slope']:+.3f}, "
                  f"t={stats['t']:.1f}, p={stats['p']:.1e}")
    ax.set_title(title)
    cap = ("MEASURES: average correlation across VIX deciles (the 'dose-response').  METHOD: bin days into VIX "
           "deciles, mean correlation per bin; title carries the Fisher-z OLS slope with HAC SE.  ROLE: the "
           "monotone-shape check -- continuous version of the regime test.")
    _save(fig, outpath, cap)
    return outpath


def plot_corr_regime_matrices(rets, liq_series, q=0.90,
                              outpath="outputs/corr_matrices_regime.png"):
    """Return-correlation heatmaps, normal vs worst-liquidity decile."""
    outpath = _ensure_writable(outpath)
    df = rets.join(liq_series.rename("liq"), how="inner").dropna()
    thr = df["liq"].quantile(q)
    cols = list(rets.columns)
    C_norm = df.loc[df["liq"] < thr, cols].corr()
    C_strs = df.loc[df["liq"] >= thr, cols].corr()

    def offdiag_mean(C):
        M = C.values
        return M[~np.eye(M.shape[0], dtype=bool)].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    im = None
    for ax, C, name in zip(axes, [C_norm, C_strs],
                           ["Normal days", f"Worst-liquidity decile (q>={q:.2f})"]):
        M = C.values
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols); ax.set_yticklabels(cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.55 else "black")
        ax.set_title(f"{name}\nall-pair avg corr = {offdiag_mean(C):+.2f}", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="correlation")
    fig.suptitle("Correlation structure: normal vs stress -- which pairs move the average",
                 fontsize=11, y=1.0)
    cap = ("MEASURES: the full return-correlation matrix on normal vs worst-VIX-decile days.  METHOD: Pearson "
           "correlation on each subsample.  ROLE: diagnoses WHICH pairs drive the average -- risk block converging "
           "(reds) while havens decouple/turn negative (blues).")
    fig.text(0.5, 0.01, cap, ha="center", va="bottom", fontsize=7.5, color="#555", wrap=True)
    _savefig(fig, outpath)
    return outpath


def plot_diversification_regime(stats, outpath="outputs/diversification_regime.png"):
    """Effective bets and PC1 share, normal vs worst-liquidity decile."""
    outpath = _ensure_writable(outpath)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.8))
    labels = ["Normal", "Worst\ndecile"]
    eff = [stats["eff_normal"], stats["eff_stress"]]
    pc1 = [stats["pc1_normal"] * 100, stats["pc1_stress"] * 100]
    a1.bar(labels, eff, color=[BLUE, RED]); a1.set_ylabel("Effective # of bets")
    a1.set_title("Diversification")
    for i, v in enumerate(eff):
        a1.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    a2.bar(labels, pc1, color=[BLUE, RED]); a2.set_ylabel("PC1 variance share (%)")
    a2.set_title("Factor concentration")
    for i, v in enumerate(pc1):
        a2.text(i, v, f"{v:.0f}%", ha="center", va="bottom")
    fig.suptitle(f"Does diversification hold up in stress?  ({stats['n_assets']} assets)")
    cap = ("MEASURES: effective number of independent bets and the PC1 variance share, normal vs stress.  METHOD: "
           "eigenvalues of the correlation matrix per regime -- participation ratio 1/sum(p^2) and top-eigenvalue "
           "share.  ROLE: the sign-robust lens -- if diversification breaks down, bets fall and PC1 share rises.")
    _save(fig, outpath, cap)
    return outpath


def plot_rolling_concentration(pc1_series, crises=None,
                               outpath="outputs/rolling_concentration.png"):
    """Rolling PC1 variance share over time."""
    outpath = _ensure_writable(outpath)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(pc1_series.index, pc1_series.values * 100, color=NAVY, lw=0.9)
    if crises:
        for s, e in crises:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.15)
    ax.set_ylabel("Rolling PC1 share (%)")
    ax.set_title("Factor concentration over time (rolling PC1 share) -- spikes mark systemic stress")
    cap = ("MEASURES: factor concentration through time.  METHOD: rolling 126-day (~6-month) window; in each window, "
           "PC1 variance share of the return-correlation matrix; crises shaded.  ROLE: shows that concentration is "
           "regime-dependent, separating the secular trend from acute episodes.")
    _save(fig, outpath, cap)
    return outpath


def plot_irf(irf_data, proxy="VIX", outpath="outputs/irf_liquidity_to_corr.png"):
    """Impulse response of average correlation to a one-SD liquidity shock.
    A positive hump after the shock = liquidity stress LEADS higher correlation."""
    outpath = _ensure_writable(outpath)
    h = irf_data["horizons"]
    resp = irf_data["resp"]
    fig, ax = plt.subplots(figsize=(9, 5))
    if irf_data.get("lower") is not None:
        ax.fill_between(h, irf_data["lower"], irf_data["upper"], color=BLUE,
                        alpha=0.25, label="95% CI")
    ax.plot(h, resp, color=NAVY, lw=1.8, marker="o", ms=3, label="IRF")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"days after a +1 SD {proxy} shock")
    ax.set_ylabel("response of average correlation")
    note = "differenced series" if irf_data.get("differenced") else "levels"
    ax.set_title(f"Does liquidity lead correlation?  IRF of rho to a {proxy} shock  "
                 f"(VAR lag={irf_data['lag']}, {note})")
    ax.legend()
    cap = (f"MEASURES: response of average correlation over ~20 days to a one-SD {proxy} shock.  METHOD: reduced-form "
           f"VAR on [{proxy}, rho], orthogonalized (Cholesky) IRF with {proxy} ordered first; 95% asymptotic band.  "
           f"ROLE: the lead-lag / direction test -- a positive hump means liquidity stress leads higher correlation. "
           f" CAVEAT: the sign of the response can depend on the Cholesky ordering.")
    _save(fig, outpath, cap)
    return outpath


def plot_pc_loadings(loadings, evr, outpath="outputs/pc_loadings.png"):
    """Heatmap of PCA loadings (assets x PC1..PCk), % variance under each PC.
    Read each column to name the factor: who loads heavily, and with which sign."""
    outpath = _ensure_writable(outpath)
    M = loadings.values
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([f"{c}\n{evr[i]*100:.0f}%" for i, c in enumerate(loadings.columns)])
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(loadings.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="loading")
    ax.set_title("PCA loadings -- name each factor by who loads on it")
    cap = ("MEASURES: each asset's weight (eigenvector) on PC1..PCk, with % variance under each PC.  METHOD: "
           "eigen-decomposition of the return-correlation matrix; sign fixed so the largest loading is positive.  "
           "ROLE: names the risk dimensions -- e.g. PC1 = risk-on/off (risk assets + vs havens -), PC2 = "
           "rates/duration, etc.")
    fig.text(0.5, 0.005, cap, ha="center", va="bottom", fontsize=7.5, color="#555", wrap=True)
    _savefig(fig, outpath)
    return outpath
