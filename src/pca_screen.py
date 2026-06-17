"""
pca_screen.py -- pick a parsimonious, genuinely-diversified basket.

Idea: throw in a WIDE candidate set spanning the main macro factors, then use
return-correlation clustering + PCA to find which assets are near-duplicates and
prune to one representative per cluster.

Run:
    python pca_screen.py

Outputs (in ./outputs, with temp-dir fallback):
    pca_clustermap.png   correlation heatmap reordered by similarity (blocks = redundant groups)
    pca_dendrogram.png   similarity tree -- cut low to merge near-duplicates
    pca_scree.png        PCA variance spectrum + 'effective number of bets'
Prints the redundant clusters and a suggested pruned basket.

Nothing here touches the main pipeline; once you've chosen, copy the winners
into config.ASSETS and re-run main.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from src.viz import _ensure_writable  # reuse the writable-path fallback

# Wide candidate basket spanning macro factors. A few are deliberately near
# duplicates (IEF~TLT, EFA/EEM/IWM~SPY) so the screen has something to prune.
CANDIDATES = {
    "SPY": "US large-cap equity",
    "IWM": "US small-cap equity",
    "EFA": "Intl developed equity",
    "EEM": "EM equity",
    "TLT": "Long Treasuries",
    "IEF": "Intermediate Treasuries",
    "LQD": "IG credit",
    "HYG": "HY credit",
    "TIP": "TIPS (inflation-linked)",
    "DBC": "Broad commodities",
    "GLD": "Gold",
    "VNQ": "US REITs",
    "UUP": "US dollar index",
}

START = "2007-01-01"          # UUP/HYG inception ~2007 bounds the common sample
CORR_REDUNDANT = 0.85         # pairs above this are treated as near-duplicates

# Anchor outputs to THIS file's folder, so it doesn't matter what directory you
# launch from (e.g. running from C:\Program Files\... would otherwise try to
# write 'outputs' there and hit WinError 5).
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "outputs")


def download_candidates(tickers, start=START):
    import yfinance as yf
    print(f"Downloading {len(tickers)} candidates from Yahoo Finance ...", flush=True)
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    px = px.ffill().dropna()
    rets = np.log(px).diff().dropna()
    return rets[[t for t in tickers if t in rets.columns]]


def screen(rets, corr_redundant=CORR_REDUNDANT):
    cols = list(rets.columns)
    print(f"Common sample: {rets.index.min().date()} -> {rets.index.max().date()} "
          f"({len(rets)} days, {len(cols)} assets)", flush=True)

    corr = rets.corr()
    dist = 1.0 - corr                                   # correlation distance
    Z = linkage(squareform(dist.values, checks=False), method="average")

    # ---- clustered correlation heatmap ----
    order = dendrogram(Z, no_plot=True, labels=cols)["ivl"]
    Cf = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(Cf.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(order)):
            v = Cf.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="correlation")
    ax.set_title("Return correlation, clustered  (tight blocks = redundant groups)")
    fig.tight_layout(); fig.savefig(_ensure_writable(os.path.join(OUTDIR, "pca_clustermap.png")), dpi=130)
    plt.close(fig)

    # ---- dendrogram ----
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=cols, ax=ax, color_threshold=(1 - corr_redundant))
    ax.axhline(1 - corr_redundant, color="grey", ls="--", lw=1,
               label=f"merge if corr > {corr_redundant:.2f}")
    ax.set_ylabel("distance = 1 - correlation"); ax.legend()
    ax.set_title("Asset similarity tree  (branches joining below the line are near-duplicates)")
    fig.tight_layout(); fig.savefig(_ensure_writable(os.path.join(OUTDIR, "pca_dendrogram.png")), dpi=130)
    plt.close(fig)

    # ---- PCA scree + effective number of bets ----
    eig = np.sort(np.linalg.eigvalsh(corr.values))[::-1]
    evr = eig / eig.sum()
    eff_bets = 1.0 / np.sum(evr ** 2)                   # participation ratio
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, len(evr) + 1), evr * 100, color="#1f3864")
    ax.set_xlabel("principal component"); ax.set_ylabel("% of variance")
    ax.set_title(f"PCA scree -- PC1 = {evr[0]*100:.0f}%,  "
                 f"effective # of bets = {eff_bets:.1f} of {len(cols)}")
    fig.tight_layout(); fig.savefig(_ensure_writable(os.path.join(OUTDIR, "pca_scree.png")), dpi=130)
    plt.close(fig)

    # ---- prune: cut tree so anything with corr > threshold merges; keep 1 per cluster ----
    labels = fcluster(Z, t=(1 - corr_redundant), criterion="distance")
    keep = []
    print("\nRedundancy clusters (corr > "
          f"{corr_redundant:.2f} = near-duplicate):")
    for c in sorted(set(labels)):
        members = [cols[i] for i in range(len(cols)) if labels[i] == c]
        if len(members) == 1:
            rep = members[0]
            print(f"  unique     : {rep}  ({CANDIDATES.get(rep, '')})")
        else:
            sub = corr.loc[members, members]
            rep = sub.mean().idxmax()                   # most central member
            dropped = [m for m in members if m != rep]
            print(f"  redundant  : {members}  -> KEEP {rep}, drop {dropped}")
        keep.append(rep)

    print(f"\nEffective bets in full set: {eff_bets:.1f} (of {len(cols)})")
    print(f"Suggested pruned basket ({len(keep)}): {keep}")
    print("Tune CORR_REDUNDANT to prune more/less aggressively.")

    # Day 2: self-contained interactive explorer (open in any browser, no server)
    html = export_basket_explorer(Z, cols, corr.values, corr_redundant)
    print(f"Interactive explorer: {html}")
    return keep


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rets = download_candidates(list(CANDIDATES))
    screen(rets)


if __name__ == "__main__":
    main()


# ===================== Day 2: interactive explorer =====================
# Self-contained HTML (no server, no extra deps): embeds the linkage tree Z, the
# labels, and the correlation matrix, then re-cuts the tree in JavaScript as you drag
# a "merge if corr >" slider. The clustering is computed once in Python; dragging only
# re-cuts, so it feels instant. Open outputs/basket_explorer.html in any browser.
def export_basket_explorer(Z, labels, corr_values, default_thr=CORR_REDUNDANT):
    import json
    Z3 = [[int(round(r[0])), int(round(r[1])), float(r[2])] for r in Z]
    names = [CANDIDATES.get(t, "") for t in labels]
    payload = json.dumps({"Z": Z3, "labels": list(labels), "names": names,
                          "corr": [[float(v) for v in row] for row in corr_values],
                          "thr": float(default_thr)})
    html = _EXPLORER_HTML.replace("/*DATA*/", payload)
    out = _ensure_writable(os.path.join(OUTDIR, "basket_explorer.html"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


_EXPLORER_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Basket redundancy explorer</title><style>
body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:#1f2933}
h1{font-size:20px;margin:0 0 4px} .sub{color:#667;margin:0 0 20px}
.ctl{background:#f4f6f9;border:1px solid #e2e8f0;border-radius:10px;padding:16px 18px;margin-bottom:18px}
input[type=range]{width:100%} .big{font-size:30px;font-weight:700;color:#1f3864}
.kept{font-family:ui-monospace,Menlo,monospace;background:#eef2f8;border-radius:6px;padding:8px 10px;display:inline-block;margin-top:6px}
.cl{border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;margin:8px 0}
.rep{font-weight:700;color:#1f3864} .drop{color:#99a;text-decoration:line-through}
.mono{font-family:ui-monospace,Menlo,monospace} .tag{color:#8893a5;font-size:13px}
</style></head><body>
<h1>Basket redundancy explorer</h1>
<p class="sub">Drag the threshold: any two assets whose correlation exceeds it are treated as
near-duplicates and merged; one representative per cluster is kept. The tree is fixed &mdash;
you are just choosing where to cut it.</p>
<div class="ctl">
  <label>merge if correlation &gt; <b id="tval"></b></label>
  <input type="range" id="sld" min="0.50" max="0.99" step="0.01">
  <div style="margin-top:10px"><span class="big" id="cnt"></span> <span class="tag">assets remain (of <span id="tot"></span>)</span></div>
  <div class="kept mono" id="kept"></div>
</div>
<div id="clusters"></div>
<script>
const D = /*DATA*/;
const Z=D.Z, LBL=D.labels, NM=D.names, C=D.corr, n=LBL.length;
const sld=document.getElementById("sld"); sld.value=D.thr;
function find(p,x){while(p[x]!==x){p[x]=p[p[x]];x=p[x];}return x;}
function clustersAt(h){
  let p=[...Array(n).keys()], rep={}; for(let i=0;i<n;i++)rep[i]=i;
  for(let k=0;k<Z.length;k++){let a=Z[k][0],b=Z[k][1],d=Z[k][2];
    if(d<=h){let fa=find(p,rep[a]),fb=find(p,rep[b]); if(fa!==fb)p[fa]=fb;}
    rep[n+k]=rep[a];}
  let g={}; for(let i=0;i<n;i++){let r=find(p,i);(g[r]=g[r]||[]).push(i);} return Object.values(g);
}
function central(ms){if(ms.length===1)return ms[0];let best=ms[0],bs=-2;
  for(const m of ms){let s=0;for(const o of ms)if(o!==m)s+=C[m][o];s/=(ms.length-1);if(s>bs){bs=s;best=m;}}return best;}
function render(){
  const t=parseFloat(sld.value), h=1-t;
  const cl=clustersAt(h).sort((a,b)=>b.length-a.length);
  const reps=cl.map(central);
  document.getElementById("tval").textContent=t.toFixed(2);
  document.getElementById("cnt").textContent=cl.length;
  document.getElementById("tot").textContent=n;
  document.getElementById("kept").textContent=reps.map(i=>LBL[i]).sort().join("  ");
  let html="";
  for(let c=0;c<cl.length;c++){const ms=cl[c],r=central(ms);
    if(ms.length===1){html+=`<div class="cl"><span class="rep">${LBL[ms[0]]}</span> <span class="tag">${NM[ms[0]]} &mdash; unique</span></div>`;}
    else{let parts=ms.map(i=>i===r?`<span class="rep">${LBL[i]}</span>`:`<span class="drop">${LBL[i]}</span>`).join(", ");
      html+=`<div class="cl">${parts} <span class="tag">&rarr; keep ${LBL[r]}</span></div>`;}}
  document.getElementById("clusters").innerHTML=html;
}
sld.addEventListener("input",render); render();
</script></body></html>"""
# ======================================================================
