"""export.py -- write the key numerical results to outputs/ as markdown + CSV,
so the figures and the tables stay in sync. Called once at the end of main()."""
import os
import tempfile
import numpy as np
import pandas as pd


def _resolve_outdir(outdir):
    """Anchor a relative outdir to the project root (parent of this src/ folder) and
    make sure it is writable; fall back to a temp dir otherwise."""
    if not os.path.isabs(outdir):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outdir = os.path.join(root, outdir)
    try:
        os.makedirs(outdir, exist_ok=True)
        probe = os.path.join(outdir, ".w")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        return outdir
    except OSError:
        fb = os.path.join(tempfile.gettempdir(), "scl_outputs")
        os.makedirs(fb, exist_ok=True)
        print(f"  [export] '{outdir}' not writable; using {fb}", flush=True)
        return fb


def _coef(model, name):
    try:
        return float(model.params[name]), float(model.tvalues[name]), float(model.pvalues[name])
    except Exception:
        return float("nan"), float("nan"), float("nan")


def write_tables(outdir, headline, rets, avg_corr, liq_series, q,
                 model, model_trend, regime, conc, irf_data, loadings, evr):
    outdir = _resolve_outdir(outdir)
    md = ["# Results tables", "",
          "Auto-generated from the run. CSV versions of the larger tables are alongside.", ""]

    # --- regression: baseline vs trend-controlled ---
    b0, t0, p0 = _coef(model, headline)
    b1, t1, p1 = _coef(model_trend, headline)
    md += ["## Fisher-z regression: VIX coefficient", "",
           "| Spec | slope | t (HAC) | p |", "|---|---|---|---|",
           f"| baseline | {b0:+.4f} | {t0:.2f} | {p0:.1e} |",
           f"| + time trend | {b1:+.4f} | {t1:.2f} | {p1:.1e} |", ""]

    # --- regime ---
    md += ["## Regime (Welch t)", "",
           "| group | mean corr |", "|---|---|",
           f"| normal | {regime['mean_normal']:.4f} |",
           f"| worst decile | {regime['mean_stress']:.4f} |",
           f"| diff (t={regime['t']:.1f}, p={regime['p']:.1e}) | {regime['diff']:+.4f} |", ""]

    # --- concentration ---
    md += ["## Diversification / concentration", "",
           "| metric | normal | stress |", "|---|---|---|",
           f"| effective bets | {conc['eff_normal']:.2f} | {conc['eff_stress']:.2f} |",
           f"| PC1 share | {conc['pc1_normal']*100:.0f}% | {conc['pc1_stress']*100:.0f}% |", ""]

    # --- dose-response deciles ---
    dfb = pd.concat([avg_corr.rename("rho"), liq_series.rename("liq")], axis=1).dropna()
    dfb["bin"] = pd.qcut(dfb["liq"], 10, labels=False, duplicates="drop")
    dose = dfb.groupby("bin")["rho"].mean()
    dose.to_csv(os.path.join(outdir, "dose_response.csv"))
    md += ["## Dose-response (mean corr by VIX decile)", "",
           "| decile | " + " | ".join(str(i + 1) for i in range(len(dose))) + " |",
           "|" + "---|" * (len(dose) + 1),
           "| mean corr | " + " | ".join(f"{v:.3f}" for v in dose.values) + " |", ""]

    # --- per-pair delta-rho ---
    thr = dfb["liq"].quantile(q)
    df2 = rets.join(liq_series.rename("liq"), how="inner").dropna()
    cols = list(rets.columns)
    Cn = df2.loc[df2["liq"] < thr, cols].corr()
    Cs = df2.loc[df2["liq"] >= thr, cols].corr()
    Cn.to_csv(os.path.join(outdir, "corr_normal.csv"))
    Cs.to_csv(os.path.join(outdir, "corr_stress.csv"))
    d = (Cs - Cn).values
    pairs = [(f"{cols[i]}-{cols[j]}", float(d[i, j]))
             for i in range(len(cols)) for j in range(i + 1, len(cols))]
    pairs.sort(key=lambda x: x[1])
    pd.DataFrame(pairs, columns=["pair", "delta_rho"]).to_csv(
        os.path.join(outdir, "delta_rho.csv"), index=False)
    top_up, top_dn = pairs[-6:][::-1], pairs[:6]
    md += ["## Largest correlation changes (stress - normal)", "",
           "| up most | dRho | down most | dRho |", "|---|---|---|---|"]
    for (pu, vu), (pdn, vd) in zip(top_up, top_dn):
        md.append(f"| {pu} | {vu:+.2f} | {pdn} | {vd:+.2f} |")
    md.append("")

    # --- IRF ---
    if irf_data is not None:
        peak = float(np.nanmax(irf_data["resp"][1:]))
        md += ["## IRF (corr response to +1 SD VIX shock)", "",
               f"- VAR lag = {irf_data['lag']} "
               f"({'differenced' if irf_data['differenced'] else 'levels'})",
               f"- peak response = {peak:+.4f}; cumulative = {irf_data['cum'][-1]:+.4f}", ""]

    # --- PC loadings ---
    loadings.to_csv(os.path.join(outdir, "pc_loadings.csv"))
    md += ["## PCA loadings", "",
           "| asset | " + " | ".join(loadings.columns) + " |",
           "|" + "---|" * (len(loadings.columns) + 1)]
    for asset, row in loadings.iterrows():
        md.append("| " + str(asset) + " | " + " | ".join(f"{v:+.2f}" for v in row.values) + " |")
    md += ["", "_variance: " + ", ".join(f"{c} {e*100:.0f}%"
           for c, e in zip(loadings.columns, evr)) + "_", ""]

    path = os.path.join(outdir, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path
