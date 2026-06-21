"""
robustness.py -- does the story survive proper uncertainty checks?

Three checks, all preserving the time structure of daily returns (block bootstrap,
so volatility clustering / autocorrelation is not destroyed):

  1. Block-bootstrap CIs for the concentration shift (effective bets and PC1 share,
     stress - normal): is 4.5->4.1 / 34%->42% real or sampling noise?
  2. Cluster stability: bootstrap the stress-regime clustering and record how often
     each asset pair lands in the SAME group (co-occurrence matrix). Tells which blocs
     are robust (e.g. the risk cluster) and which are fragile (e.g. HYG-LQD).
  3. Threshold sensitivity: redo the headline numbers at the 85/90/95th-pct stress
     cutoff, AND report the VIX-vs-FUNDING stress-day overlap (Jaccard) at each -- does
     the low overlap survive a looser threshold?

Run:
    python robustness.py
Outputs (./outputs): robustness.md, cluster_stability_<proxy>.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import config as C
from src.viz import _ensure_writable, _savefig
from stress_clustering import load_all   # same loader (VIX + injected FUNDING)

B = 1000          # bootstrap replications
BLOCK = 21        # block length in days (~1 month; preserves vol clustering)
CUT_CORR = 0.50   # clustering cut (same as stress_clustering)
QS = [0.85, 0.90, 0.95]
SEED = 7


# ----------------------------------------------------------------- primitives
def _conc(R):
    """effective bets and PC1 share from a returns block (rows=days, cols=assets)."""
    Cm = np.corrcoef(R, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(Cm))[::-1]
    evr = ev / ev.sum()
    return 1.0 / np.sum(evr ** 2), evr[0]


def _offdiag_mean(R):
    Cm = np.corrcoef(R, rowvar=False)
    return Cm[~np.eye(Cm.shape[0], dtype=bool)].mean()


def _block_idx(n, L, rng):
    out = []
    while len(out) < n:
        s = rng.integers(0, n - L + 1)
        out.extend(range(s, s + L))
    return np.array(out[:n])


def _ci(x):
    x = np.asarray(x)
    lo, hi = np.percentile(x, [2.5, 97.5])
    frac_below = np.mean(x < 0)
    p = 2 * min(frac_below, 1 - frac_below)          # two-sided bootstrap p
    return x.mean(), lo, hi, p


# ----------------------------------------------------------------- check 1
def bootstrap_concentration(rets, proxy, q=0.90, B=B, L=BLOCK):
    """Block-resample the whole timeline, re-split by the q-th pct of `proxy`,
    and record (stress - normal) for effective bets and PC1 share."""
    R = rets.values
    p = proxy.values
    T, k = R.shape
    rng = np.random.default_rng(SEED)
    d_eff, d_pc1 = [], []
    for _ in range(B):
        ix = _block_idx(T, L, rng)
        Rb, pb = R[ix], p[ix]
        thr = np.quantile(pb, q)
        Rn, Rs = Rb[pb < thr], Rb[pb >= thr]
        if len(Rn) < k + 2 or len(Rs) < k + 2:
            continue
        en, p1n = _conc(Rn)
        es, p1s = _conc(Rs)
        d_eff.append(es - en)
        d_pc1.append(p1s - p1n)
    return {"eff": _ci(d_eff), "pc1": _ci(d_pc1), "n": len(d_eff)}


# ----------------------------------------------------------------- check 2
def cluster_stability(rets, proxy, proxy_name, q=0.90, B=B, L=BLOCK, cut=CUT_CORR):
    """Bootstrap the stress-regime clustering; co[i,j] = P(i,j in same group)."""
    cols = list(rets.columns)
    R, p = rets.values, proxy.values
    T, k = R.shape
    rng = np.random.default_rng(SEED + 1)
    co = np.zeros((k, k))
    cnt = 0
    for _ in range(B):
        ix = _block_idx(T, L, rng)
        Rb, pb = R[ix], p[ix]
        S = Rb[pb >= np.quantile(pb, q)]
        if len(S) < k + 2:
            continue
        Cm = np.corrcoef(S, rowvar=False)
        D = 1.0 - Cm
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        lab = fcluster(Z, t=(1 - cut), criterion="distance")
        same = (lab[:, None] == lab[None, :]).astype(float)
        co += same
        cnt += 1
    co /= max(cnt, 1)

    order = np.argsort(-co.sum(axis=1))               # group stable assets together
    co_o = co[np.ix_(order, order)]
    lbl = [cols[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(co_o, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_xticklabels(lbl, rotation=90)
    ax.set_yticks(range(k)); ax.set_yticklabels(lbl)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{co_o[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if co_o[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(same group)")
    ax.set_title(f"Cluster stability under {proxy_name} stress  ({cnt} bootstraps)")
    cap = (f"MEASURES: probability each asset pair lands in the same cluster across {cnt} block-bootstrap resamples "
           f"of the worst-{int((1-q)*100)}% {proxy_name} days.  METHOD: block bootstrap (L={L}d) -> stress subsample "
           f"-> 1-corr average-linkage cut at corr>{cut:.2f}.  ROLE: dark blocks = robust groupings; pale = fragile.")
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, cap, ha="center", va="bottom", fontsize=7.3, color="#555", wrap=True)
    _savefig(fig, _ensure_writable(f"outputs/cluster_stability_{proxy_name}.png"))
    return co, cols


# ----------------------------------------------------------------- check 3+4
def threshold_sensitivity(rets, liq, proxies, qs=QS):
    rows = []
    for q in qs:
        for p in proxies:
            s = liq[p]
            thr = s.quantile(q)
            R = rets.join(s.rename("liq"), how="inner").dropna()
            Rn = R.loc[R["liq"] < thr, rets.columns].values
            Rs = R.loc[R["liq"] >= thr, rets.columns].values
            en, p1n = _conc(Rn)
            es, p1s = _conc(Rs)
            rows.append({"q": q, "proxy": p,
                         "regime_diff": _offdiag_mean(Rs) - _offdiag_mean(Rn),
                         "d_eff": es - en, "d_pc1": p1s - p1n})
    # Jaccard overlap of stress-day sets at each q (needs >=2 proxies)
    jacc = {}
    if len(proxies) >= 2:
        a, b = proxies[0], proxies[1]
        for q in qs:
            sa = set(liq.index[liq[a] >= liq[a].quantile(q)])
            sb = set(liq.index[liq[b] >= liq[b].quantile(q)])
            jacc[q] = len(sa & sb) / len(sa | sb) if (sa | sb) else float("nan")
    return pd.DataFrame(rows), jacc


# ----------------------------------------------------------------- report
def main():
    rets, liq = load_all()
    proxies = list(liq.columns)
    headline = C.HEADLINE_PROXY
    print(f"Loaded {rets.shape[0]} days x {rets.shape[1]} assets | proxies: {proxies}")
    print(f"Bootstrap: B={B}, block={BLOCK}d")

    md = ["# Robustness checks", "",
          f"Block bootstrap (B={B}, block={BLOCK} days) preserves daily autocorrelation / "
          "volatility clustering.", ""]

    # check 1
    print("\n[1/3] block-bootstrap concentration CIs ...")
    bc = bootstrap_concentration(rets, liq[headline])
    em, elo, ehi, ep = bc["eff"]
    pm, plo, phi, pp = bc["pc1"]
    eff_excludes_0 = (elo > 0) or (ehi < 0)
    verdict = ("Both Δ's exclude 0 (95% CI) => the diversification erosion is statistically "
               "real, not sampling noise." if eff_excludes_0 else
               "The CI includes 0 => cannot reject sampling noise at 5%.")
    md += [f"## 1. Concentration shift (stress - normal), {headline} @ 90th pct  "
           f"({bc['n']} valid resamples)", "",
           "| metric | mean Δ | 95% CI | bootstrap p |", "|---|---|---|---|",
           f"| effective bets | {em:+.2f} | [{elo:+.2f}, {ehi:+.2f}] | {ep:.3f} |",
           f"| PC1 share | {pm*100:+.1f}pp | [{plo*100:+.1f}, {phi*100:+.1f}]pp | {pp:.3f} |", "",
           verdict, ""]

    # check 2
    print("[2/3] cluster stability ...")
    for p in proxies:
        co, cols = cluster_stability(rets, liq[p], p)
        # report a few telling pairs
        def pair(i, j):
            return co[cols.index(i), cols.index(j)] if i in cols and j in cols else float("nan")
        md += [f"## 2. Cluster stability under {p} stress", "",
               f"See `cluster_stability_{p}.png`. Selected pair co-occurrence "
               f"(P same group):", "",
               f"- HYG-LQD (credit merge): {pair('HYG','LQD'):.2f}",
               f"- SPY-EFA (core risk): {pair('SPY','EFA'):.2f}",
               f"- SPY-TLT (risk vs haven): {pair('SPY','TLT'):.2f}",
               f"- DBC-SPY (commodity into risk): {pair('DBC','SPY'):.2f}", ""]

    # check 3+4
    print("[3/3] threshold sensitivity + Jaccard ...")
    tab, jacc = threshold_sensitivity(rets, liq, proxies)
    md += ["## 3. Threshold sensitivity (85 / 90 / 95th pct)", "",
           "| q | proxy | regime Δρ | Δ eff bets | Δ PC1 |", "|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        md.append(f"| {r['q']:.2f} | {r['proxy']} | {r['regime_diff']:+.3f} | "
                  f"{r['d_eff']:+.2f} | {r['d_pc1']*100:+.1f}pp |")
    md += [""]
    if jacc:
        md += ["## 4. VIX vs FUNDING stress-day overlap by threshold", "",
               "| q | Jaccard overlap |", "|---|---|"]
        for q in QS:
            md.append(f"| {q:.2f} | {jacc[q]:.2f} |")
        md += ["", "If overlap stays low as q loosens, the two proxies are genuinely "
               "different gauges, not a tail artefact.", ""]

    out = _ensure_writable("outputs/robustness.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nWrote {out} + cluster_stability figures.")


if __name__ == "__main__":
    main()
