"""Market-wide scan for stocks whose daily returns track NVDA.

Universe: S&P 500 + S&P 400 + S&P 600 + Nasdaq-100 (from Wikipedia).
Usage: python secret_twin_full.py [--top 30] [--include-obvious]
"""
import argparse
import io

import pandas as pd
import requests
import yfinance as yf

TARGET = "NVDA"
HEADERS = {"User-Agent": "Mozilla/5.0"}
WINDOWS = {"6mo": 126, "1y": 252, "2y": 504}

OBVIOUS = {"AAPL", "MSFT", "AMD", "GOOGL", "GOOG", "AMZN", "META", "AVGO",
           "INTC", "QCOM", "MU", "TSLA", "ORCL", "SMCI", "ARM", "MRVL",
           "AMAT", "LRCX", "KLAC", "TXN", "ADI", "NXPI", "MCHP", "ON", "MPWR"}

SOURCES = {
    "S&P500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "GICS Sector"),
    "S&P400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol", "GICS Sector"),
    "S&P600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol", "GICS Sector"),
    "NDX100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker", "GICS Sector"),
}


def load_universe():
    info = {}
    for name, (url, tcol, scol) in SOURCES.items():
        try:
            html = requests.get(url, headers=HEADERS, timeout=30).text
            for t in pd.read_html(io.StringIO(html)):
                if tcol in t.columns:
                    for _, r in t.iterrows():
                        sym = str(r[tcol]).replace(".", "-").strip()
                        info.setdefault(sym, (r.get(scol, "?"), name))
                    break
            print(f"loaded {name}")
        except Exception as e:
            print(f"skip {name}: {e}")
    return info


def download(tickers, period="2y", chunk=200):
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        d = yf.download(part, period=period, auto_adjust=True,
                        progress=False, threads=True)["Close"]
        if isinstance(d, pd.Series):
            d = d.to_frame(part[0])
        frames.append(d)
        print(f"downloaded {min(i + chunk, len(tickers))}/{len(tickers)}")
    return pd.concat(frames, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--include-obvious", action="store_true")
    args = ap.parse_args()

    info = load_universe()
    tickers = sorted(t for t in info
                     if t != TARGET and (args.include_obvious or t not in OBVIOUS))
    print(f"{len(tickers)} candidates")

    prices = download([TARGET] + tickers)
    prices = prices.loc[:, prices.notna().sum() > 120]
    rets = prices.pct_change().iloc[1:]
    nvda = rets[TARGET]

    rows = []
    for t in rets.columns.drop(TARGET):
        row = {"ticker": t, "sector": info[t][0], "index": info[t][1]}
        for label, n in WINDOWS.items():
            pair = pd.concat([nvda.tail(n), rets[t].tail(n)], axis=1).dropna()
            row[f"corr_{label}"] = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) > 60 else None
            if label == "1y" and len(pair) > 60:
                row["beta_1y"] = pair.cov().iloc[0, 1] / pair.iloc[:, 0].var()
        rows.append(row)

    df = pd.DataFrame(rows).dropna(subset=["corr_1y"])
    corr_cols = [f"corr_{k}" for k in WINDOWS]
    df["min_corr"] = df[corr_cols].min(axis=1)  # stable across windows
    df = df.sort_values("corr_1y", ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(f"\nTop {args.top} by 1y correlation with {TARGET}\n")
    print(df.head(args.top).round(3).to_string())

    print("\nMost consistent across 6mo/1y/2y (by min correlation):")
    print(df.sort_values("min_corr", ascending=False).head(10).round(3).to_string())

    print("\nBest per sector:")
    print(df.groupby("sector").head(1).round(3).to_string(index=False))

    n = (df.corr_1y > 0.85).sum()
    print(f"\nStocks above 0.85: {n}; above 0.7: {(df.corr_1y > 0.7).sum()}; above 0.6: {(df.corr_1y > 0.6).sum()}")
    df.to_csv("nvda_correlations_full.csv", index=False)
    print("saved nvda_correlations_full.csv")


if __name__ == "__main__":
    main()
