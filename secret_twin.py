"""Find unexpected stocks whose daily returns track NVDA's closely.

Usage: python secret_twin.py [--period 1y] [--top 15]
"""
import argparse

import pandas as pd
import yfinance as yf

TARGET = "NVDA"

# Obvious tech names to exclude so the "secret twin" is a surprise.
OBVIOUS = {"AAPL", "MSFT", "AMD", "GOOGL", "GOOG", "AMZN", "META", "AVGO",
           "TSM", "INTC", "QCOM", "MU", "TSLA", "ORCL", "SMCI", "ARM"}

# Candidates in "unseen infrastructure" sectors, grouped by theme.
UNIVERSE = {
    "Utilities / Power": ["DUK", "D", "SO", "NEE", "AEP", "ETR", "PEG", "EXC",
                          "XEL", "VST", "CEG", "NRG", "TLN", "PCG", "SRE"],
    "Electrical / Cooling": ["VRT", "ETN", "HUBB", "GEV", "PWR", "EME", "FIX",
                             "AYI", "NVT", "JCI", "TT", "CARR", "LII", "MOD"],
    "Data center REITs": ["EQIX", "DLR", "AMT", "IRM", "CCI", "SBAC"],
    "Chemicals / Materials": ["LIN", "APD", "DD", "DOW", "ECL", "SHW", "ALB",
                              "FCX", "SCCO", "CLF", "MP"],
    "Industrial / Construction": ["CAT", "DE", "URI", "MTZ", "EMR", "ROK",
                                  "PH", "GWW", "FAST", "MLM", "VMC", "J"],
    "Cables / Components": ["APH", "GLW", "TEL", "COHR", "CIEN", "ANET",
                            "FN", "BDC", "WIRE"],
    "Semicap / Materials": ["AMAT", "LRCX", "KLAC", "ASML", "ENTG", "ONTO",
                            "TER", "MKSI"],
    "Energy / Gas": ["WMB", "KMI", "OKE", "LNG", "EQT", "XOM", "CVX"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="1y")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--include-obvious", action="store_true")
    args = ap.parse_args()

    sector_of = {t: s for s, ts in UNIVERSE.items() for t in ts}
    tickers = sorted(t for t in sector_of
                     if args.include_obvious or t not in OBVIOUS)

    prices = yf.download([TARGET] + tickers, period=args.period,
                         auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.9))
    rets = prices.pct_change().dropna(how="all")

    nvda = rets[TARGET]
    rows = []
    for t in rets.columns.drop(TARGET):
        pair = pd.concat([nvda, rets[t]], axis=1).dropna()
        if len(pair) < 60:
            continue
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        beta = pair.cov().iloc[0, 1] / pair.iloc[:, 0].var()  # sensitivity to NVDA
        rows.append((t, sector_of.get(t, "?"), corr, beta, len(pair)))

    df = (pd.DataFrame(rows, columns=["ticker", "sector", "pearson", "beta_to_nvda", "days"])
          .sort_values("pearson", ascending=False).reset_index(drop=True))

    pd.set_option("display.width", 120)
    print(f"\nTop {args.top} correlations with {TARGET} ({args.period}, daily returns)\n")
    print(df.head(args.top).round(3).to_string())

    print("\nBest per sector:")
    print(df.groupby("sector").head(1).round(3).to_string(index=False))

    hits = df[df.pearson > 0.85]
    print(f"\nStocks above 0.85 correlation: {len(hits)}")
    if hits.empty:
        print(f"None; the strongest 'secret twin' is {df.iloc[0].ticker} "
              f"({df.iloc[0].sector}) at r={df.iloc[0].pearson:.2f}.")
    df.to_csv("nvda_correlations.csv", index=False)


if __name__ == "__main__":
    main()
