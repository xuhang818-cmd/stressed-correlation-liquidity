"""
stress_clustering.py -- how assets regroup under stress, and whether VIX-stress
and FUNDING-stress pick out the same days.

Run:
    python stress_clustering.py

Step 1 (diagnostic): for each proxy, flag the worst-decile (>=90th pct) days as
  "stress". Quantify how much the VIX-stress set and FUNDING-stress set overlap
  (Jaccard) and draw a timeline marking each set's spikes + their overlap.
Step 2 (clustering): for each proxy, build the return-correlation matrix on normal
  vs stress days, cluster assets (1-corr distance, average linkage), and show the
  two trees side by side + a group-membership table (who converges, who decouples).

Outputs (./outputs):
    stress_overlap.png            timeline of VIX vs FUNDING stress days + overlap
    stress_clustering_<proxy>.png normal vs stress dendrograms
    stress_groups.md / *.csv      group membership, normal vs stress, per proxy
Nothing here touches the main pipeline.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import config as C
from src import data_loader
from src.viz import _ensure_writable, _savefig

Q = 0.90              # worst-decile threshold (same as the regime test)
CUT_CORR = 0.50       # group assets whose correlation exceeds this (cut the tree here)
NAVY, RED, BLUE, PURPLE = "#1f3864", "#c0504d", "#7f9bbf", "#7048a8"


def load_all():
    rets, liq = data_loader.load_real(C.ASSETS, C.LIQUIDITY_YF, C.LIQUIDITY_CSV, C.START, C.END)
    if getattr(C, "FUNDING_SPLICE", None):
        funding = data_loader.load_spliced_funding(**C.FUNDING_SPLICE)
        liq = data_loader.inject_funding(liq, funding)
    liq = liq.reindex(rets.index).ffill()
    df = rets.join(liq, how="inner").dropna()
    return df[C.ASSETS], df[list(liq.columns)]


def stress_mask(liq_series, q=Q):
    return liq_series >= liq_series.quantile(q)


# ----------------------------------------------------------------------- step 1
def overlap_diagnostic(liq, proxies):
    masks = {p: stress_mask(liq[p]) for p in proxies}
    days = {p: set(liq.index[masks[p]]) for p in proxies}
    a, b = proxies[0], proxies[1]
    inter = days[a] & days[b]
    union = days[a] | days[b]
    jacc = len(inter) / len(union) if union else float("nan")
    print(f"\nStress-day overlap ({a} vs {b}, worst {int((1-Q)*100)}% each):")
    print(f"  {a}: {len(days[a])} days | {b}: {len(days[b])} days | both: {len(inter)}")
    print(f"  Jaccard overlap = {jacc:.2f} | "
          f"{len(inter)/len(days[a])*100:.0f}% of {a}-stress days are also {b}-stress")

    fig, ax = plt.subplots(figsize=(13, 3.4))
    idx = liq.index
    ax.vlines(idx[masks[a]], 0.7, 1.0, color=RED, lw=0.6, alpha=0.7)
    ax.vlines(idx[masks[b]], 0.35, 0.65, color=BLUE, lw=0.6, alpha=0.7)
    both = masks[a] & masks[b]
    ax.vlines(idx[both], 0.0, 0.30, color=PURPLE, lw=0.6, alpha=0.85)
    for s, e in (C.CRISES or []):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.12)
    ax.set_yticks([0.85, 0.5, 0.15])
    ax.set_yticklabels([f"{a} stress", f"{b} stress", "both (overlap)"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"When does each proxy flag stress?  Jaccard overlap = {jacc:.2f}  "
                 f"({len(inter)} shared of {len(union)} stress days)")
    cap = (f"MEASURES: days each proxy puts in its worst {int((1-Q)*100)}% (red={a}, blue={b}), and where they "
           f"coincide (purple).  METHOD: 90th-pct threshold per proxy; Jaccard = |overlap|/|union|.  ROLE: shows "
           f"whether the two proxies flag the SAME crises -- high overlap explains why they give the same result; "
           f"proxy-only spikes (e.g. funding-only) are events one gauge sees and the other misses.")
    fig.subplots_adjust(bottom=0.30)
    fig.text(0.5, 0.02, cap, ha="center", va="bottom", fontsize=7.5, color="#555", wrap=True)
    _savefig(fig, _ensure_writable("outputs/stress_overlap.png"))
    return {"jaccard": jacc, "n_a": len(days[a]), "n_b": len(days[b]), "n_both": len(inter)}


# ----------------------------------------------------------------------- step 2
def _groups(corr, cols, cut_corr=CUT_CORR):
    """Cluster on 1-corr distance; cut so pairs with corr>cut_corr share a group."""
    dist = 1.0 - corr.values
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=(1 - cut_corr), criterion="distance")
    groups = []
    for c in sorted(set(labels)):
        members = [cols[i] for i in range(len(cols)) if labels[i] == c]
        groups.append(members)
    groups.sort(key=len, reverse=True)
    return Z, groups, labels


def cluster_proxy(rets, liq_series, proxy):
    cols = list(rets.columns)
    df = rets.join(liq_series.rename("liq"), how="inner").dropna()
    thr = df["liq"].quantile(Q)
    Cn = df.loc[df["liq"] < thr, cols].corr()
    Cs = df.loc[df["liq"] >= thr, cols].corr()
    Zn, gn, _ = _groups(Cn, cols)
    Zs, gs, _ = _groups(Cs, cols)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, Z, name in zip(axes, [Zn, Zs],
                           [f"Normal days", f"Worst-{int((1-Q)*100)}% {proxy} days"]):
        dendrogram(Z, labels=cols, ax=ax, color_threshold=(1 - CUT_CORR))
        ax.axhline(1 - CUT_CORR, color="grey", ls="--", lw=1)
        ax.set_title(name)
        ax.set_ylabel("distance = 1 - correlation")
    fig.suptitle(f"How assets regroup under {proxy} stress  "
                 f"(dashed line = merge if corr > {CUT_CORR:.2f})", y=1.02)
    cap = (f"MEASURES: hierarchical clustering of assets, normal vs worst-{int((1-Q)*100)}% {proxy} days.  METHOD: "
           f"1-correlation distance, average linkage; branches joining BELOW the dashed line are correlated above "
           f"{CUT_CORR:.2f}.  ROLE: shows the cross-sectional split -- which bloc converges (joins low) and which "
           f"decouples in stress.")
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, cap, ha="center", va="bottom", fontsize=7.5, color="#555", wrap=True)
    _savefig(fig, _ensure_writable(f"outputs/stress_clustering_{proxy}.png"))

    # membership CSV: asset -> normal group id, stress group id
    def gid(groups):
        m = {}
        for i, grp in enumerate(groups):
            for a in grp:
                m[a] = i + 1
        return m
    mn, ms = gid(gn), gid(gs)
    pd.DataFrame({"asset": cols,
                 "normal_group": [mn[a] for a in cols],
                 "stress_group": [ms[a] for a in cols]}).to_csv(
        _ensure_writable(f"outputs/stress_groups_{proxy}.csv"), index=False)
    return gn, gs


def main():
    rets, liq = load_all()
    proxies = list(liq.columns)
    print(f"Loaded {rets.shape[0]} days x {rets.shape[1]} assets | proxies: {proxies}")

    md = ["# Stress-regime asset grouping", ""]

    if len(proxies) >= 2:
        ov = overlap_diagnostic(liq, proxies[:2])
        md += [f"## Stress-day overlap ({proxies[0]} vs {proxies[1]})", "",
               f"- {proxies[0]} stress days: {ov['n_a']}",
               f"- {proxies[1]} stress days: {ov['n_b']}",
               f"- shared: {ov['n_both']}  |  Jaccard = {ov['jaccard']:.2f}", ""]
    else:
        print("Only one proxy present; skipping overlap diagnostic.")

    for p in proxies:
        gn, gs = cluster_proxy(rets, liq[p], p)
        md += [f"## Groups under {p} stress  (merge if corr > {CUT_CORR:.2f})", "",
               "**Normal days:**"]
        md += [f"- group {i+1}: {', '.join(g)}" for i, g in enumerate(gn)]
        md += ["", f"**Worst-{int((1-Q)*100)}% {p} days:**"]
        md += [f"- group {i+1}: {', '.join(g)}" for i, g in enumerate(gs)]
        md += [""]

    out = _ensure_writable("outputs/stress_groups.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nWrote {out} and per-proxy figures + CSVs.")


if __name__ == "__main__":
    main()
