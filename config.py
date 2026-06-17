"""Central config. Edit tickers / proxies / dates here."""

# Cross-asset basket (yfinance). PCA-pruned "traditional diversified" set:
# from a wider candidate list, near-duplicate twins (IWM~SPY, EEM~EFA, IEF~TLT)
# were dropped via correlation clustering (see pca_screen.py). ~4 effective bets.
#   SPY equity | EFA intl equity | VNQ REITs | HYG HY credit | DBC commodities
#   TLT long Treasuries | TIP TIPS | LQD IG credit | GLD gold | UUP US dollar
ASSETS = ["SPY", "EFA", "VNQ", "HYG", "DBC", "TLT", "TIP", "LQD", "GLD", "UUP"]

# Liquidity / market-stress proxies.
#
# LIQUIDITY_YF: pulled from yfinance, i.e. the SAME source as prices, so as
#   reliable as the price download (no FRED). Higher = more stress.
#     ^VIX : CBOE Volatility Index (equity implied vol / risk aversion)
#   You can add other tradable stress indices here, e.g. "^OVX" (oil vol),
#   "^MOVE" (Treasury vol, when available), "^SKEW" (tail risk).
LIQUIDITY_YF = {
    "VIX": "^VIX",
}

# LIQUIDITY_CSV: optional extra proxies loaded from local CSV files (so you can
#   bring in series that have no free programmatic feed -- e.g. ICE BofA HY OAS
#   exported from Bloomberg Terminal). Each CSV = two columns [date, value],
#   higher = more stress. Leave empty to run on VIX alone. Example:
#     LIQUIDITY_CSV = {"HY_OAS": "data/hy_oas.csv"}
LIQUIDITY_CSV = {}

# Headline series used in the main chart / regime split. Must be a key that
# exists in LIQUIDITY_YF or LIQUIDITY_CSV.
HEADLINE_PROXY = "VIX"

# Day 2 (optional): spliced continuous funding-stress proxy from two local CSVs.
# Set to a dict to activate; injected as a "FUNDING" liquidity column. Example:
#   FUNDING_SPLICE = {"legacy_csv": "data/ted.csv",
#                     "current_csv": "data/sofr_spread.csv",
#                     "split": "2022-01-01", "method": "zscore"}
FUNDING_SPLICE = None

START = "2007-01-01"  # UUP/HYG inception ~2007 bounds the common sample for this basket
END = None  # None -> today

# Crisis windows shaded on the main chart (GFC, COVID).
CRISES = [("2008-08-01", "2009-06-30"), ("2020-02-15", "2020-05-31")]

OUTPUT_DIR = "outputs"
