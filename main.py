"""
main.py -- run the whole pipeline end to end.

    python main.py --simulate     # offline demo on synthetic data
    python main.py                # real data: yfinance prices + FRED liquidity

Steps: load -> univariate GARCH -> DCC correlation -> Fisher-z regression
       -> regime test -> (optional) VAR/IRF -> figures.
"""
import argparse
import numpy as np
import pandas as pd

import config as C
from src import data_loader, dcc_garch, analysis, viz, export


def main(simulate: bool, diagnose_funding: bool = False):
    # Quick command-line diagnostic: check the funding splice overlap, print R^2, exit.
    if diagnose_funding:
        fs = dict(getattr(C, "FUNDING_SPLICE", None) or {})
        if not fs:
            print("No FUNDING_SPLICE configured in config.py.")
            return
        fs.pop("method", None)
        fs.pop("split", None)
        print("Funding splice overlap diagnostic:")
        data_loader.diagnose_funding_overlap(**fs)
        return

    if simulate:
        rets, liq = data_loader.simulate()
        headline = liq.columns[0]
    else:
        rets, liq = data_loader.load_real(C.ASSETS, C.LIQUIDITY_YF, C.LIQUIDITY_CSV, C.START, C.END)
        headline = C.HEADLINE_PROXY

    print(f"Returns: {rets.shape[0]} days x {rets.shape[1]} assets | "
          f"liquidity proxies: {list(liq.columns)}")

    # Day 2 (optional): inject a spliced continuous funding proxy from local CSVs
    if not simulate and getattr(C, "FUNDING_SPLICE", None):
        try:
            funding = data_loader.load_spliced_funding(**C.FUNDING_SPLICE)
            liq = data_loader.inject_funding(liq, funding)
            print(f"Injected spliced FUNDING proxy ({funding.dropna().shape[0]} obs) "
                  f"-> proxies now {list(liq.columns)}")
        except Exception as e:
            print(f"FUNDING splice skipped: {e}")

    # 1. Univariate GARCH -> standardized residuals
    Z, V = dcc_garch.fit_univariate_garch(rets)

    # 2. DCC -> time-varying average pairwise correlation
    dcc = dcc_garch.fit_dcc(Z)
    avg_corr = dcc["avg_corr"]
    print(f"DCC params: a={dcc['a']:.4f}, b={dcc['b']:.4f}, a+b={dcc['a']+dcc['b']:.4f}")

    # Align correlation with liquidity
    liq = liq.reindex(avg_corr.index).ffill().dropna()
    avg_corr = avg_corr.reindex(liq.index)

    # 3. Fisher-z regression on liquidity proxies
    model = analysis.regress_on_liquidity(avg_corr, liq, lags=1)
    print("\n=== Fisher-z(correlation) ~ liquidity (HAC SE) ===")
    print(model.summary().tables[1])
    print(f"R^2 = {model.rsquared:.3f}")

    # 3b. Same regression, controlling for a secular time trend
    model_trend = analysis.regress_on_liquidity(avg_corr, liq, lags=1, add_trend=True)
    print(f"\n{headline} coefficient -- baseline: {model.params[headline]:+.4f} "
          f"(t={model.tvalues[headline]:.2f})  |  +trend control: "
          f"{model_trend.params[headline]:+.4f} (t={model_trend.tvalues[headline]:.2f})")

    # 4-6. Per-proxy analysis + figures (VIX and, if injected, FUNDING -- each gets its
    #      own set of figures with a _{proxy} filename suffix so nothing is overwritten).
    crises = None if simulate else C.CRISES
    proxies = list(liq.columns)
    per_proxy = {}
    figs = []

    for proxy in proxies:
        print(f"\n========== proxy: {proxy} ==========")
        reg = analysis.regime_test(avg_corr, liq[proxy], q=0.90)
        print(f"Regime test: stress={reg['mean_stress']:.3f}  normal={reg['mean_normal']:.3f}  "
              f"diff={reg['diff']:.3f}  t={reg['t']:.2f}  p={reg['p']:.2e}")

        irf_data = None
        try:
            irf_data = analysis.var_irf(avg_corr, liq[proxy])
            print(f"VAR (lag={irf_data['lag']}). Peak IRF of corr to a {proxy} shock: "
                  f"{irf_data['resp'][1:].max():+.4f} (cumulative {irf_data['cum'][-1]:+.4f}).")
        except Exception as e:
            print(f"VAR/IRF skipped: {e}")

        conc = analysis.concentration_regime(rets, liq[proxy], q=0.90)
        print(f"Diversification -- effective bets: normal {conc['eff_normal']:.1f} -> "
              f"stress {conc['eff_stress']:.1f}  |  PC1 share: {conc['pc1_normal']*100:.0f}% -> "
              f"{conc['pc1_stress']*100:.0f}%")

        stats_d = {"slope": model.params.get(proxy, float('nan')),
                   "t": model.tvalues.get(proxy, float('nan')),
                   "p": model.pvalues.get(proxy, float('nan'))}

        figs += [
            viz.plot_corr_vs_liquidity(avg_corr, liq[proxy].rename(proxy), crises,
                                       outpath=f"outputs/corr_vs_liquidity_{proxy}.png"),
            viz.plot_regime(reg, outpath=f"outputs/regime_{proxy}.png"),
            viz.plot_dose_response(avg_corr, liq[proxy].rename(proxy), stats=stats_d,
                                   outpath=f"outputs/dose_response_{proxy}.png"),
            viz.plot_corr_regime_matrices(rets, liq[proxy], q=0.90,
                                          outpath=f"outputs/corr_matrices_regime_{proxy}.png"),
            viz.plot_diversification_regime(conc,
                                            outpath=f"outputs/diversification_regime_{proxy}.png"),
        ]
        if irf_data is not None:
            figs.append(viz.plot_irf(irf_data, proxy=proxy,
                                     outpath=f"outputs/irf_{proxy}.png"))

        per_proxy[proxy] = {"reg": reg, "conc": conc, "irf": irf_data}

    # Proxy-independent figures (returns only)
    figs.append(viz.plot_rolling_concentration(
        analysis.rolling_pc1_share(rets, window=126), crises=crises))
    loadings, evr = analysis.pca_loadings(rets, k=4)
    figs.append(viz.plot_pc_loadings(loadings, evr))

    print("\nFigures:")
    for f in figs:
        print(f"  {f}")

    # 7. Export tables for the headline proxy
    h = per_proxy[headline]
    tbl = export.write_tables("outputs", headline, rets, avg_corr, liq[headline], 0.90,
                              model, model_trend, h["reg"], h["conc"], h["irf"], loadings, evr)
    print(f"\nTables: {tbl}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true", help="run offline on synthetic data")
    ap.add_argument("--diagnose-funding", action="store_true",
                    help="check the funding splice overlap (R^2) and exit")
    args = ap.parse_args()
    main(args.simulate, args.diagnose_funding)
