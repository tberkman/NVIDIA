"""Huge scan: every US-listed common stock / ADR vs NVDA (daily returns).

Universe comes from the Nasdaq Trader symbol directory (NASDAQ, NYSE, AMEX, ARCA).
ETFs, test issues, warrants, rights, units and preferreds are dropped.
Prices are cached in prices_cache.pkl; use --refresh to re-download.

Usage: python secret_twin_huge.py [--top 40] [--min-days 240] [--refresh]
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TARGET = "NVDA"
CACHE = "prices_cache.pkl"
URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]
BAD_NAME = re.compile(r"\b(warrant|right|unit|preferred|note|debenture|trust|depositary share"
                      r"|acquisition|spac|etn|fund)\b", re.I)
# Obvious chip names; hidden when --hide-obvious is used.
OBVIOUS = {"AAPL", "MSFT", "AMD", "GOOGL", "GOOG", "AMZN", "META", "AVGO", "INTC", "QCOM",
           "MU", "TSLA", "ORCL", "SMCI", "ARM", "MRVL", "AMAT", "LRCX", "KLAC", "TSM", "ASML"}


def load_universe():
    rows = []
    for url in URLS:
        text = requests.get(url, timeout=60).text.splitlines()
        hdr = text[0].split("|")
        for line in text[1:]:
            if line.startswith("File Creation"):
                continue
            d = dict(zip(hdr, line.split("|")))
            sym = d.get("Symbol") or d.get("ACT Symbol")
            if not sym or d.get("ETF") == "Y" or d.get("Test Issue") == "Y":
                continue
            name = d["Security Name"]
            if BAD_NAME.search(name) and "common" not in name.lower() and "ordinary" not in name.lower():
                continue
            if not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", sym):
                continue
            rows.append((sym.replace(".", "-"), name))
    return dict(rows)


def fetch(part):
    d = yf.download(part, period="2y", auto_adjust=True, progress=False,
                    threads=True)["Close"]
    if isinstance(d, pd.Series):
        d = d.to_frame(part[0])
    return d.dropna(axis=1, how="all")


def download(tickers, rounds=4):
    """Download in small chunks, retrying symbols missed due to rate limits."""
    got = {}
    pending = list(tickers)
    for rnd in range(rounds):
        chunk = max(50, 150 // (rnd + 1))
        for i in range(0, len(pending), chunk):
            part = pending[i:i + chunk]
            try:
                d = fetch(part)
                for c in d.columns:
                    got[c] = d[c]
            except Exception as e:
                print("chunk failed:", e)
            time.sleep(1.0 + rnd)
            print(f"round {rnd + 1}: {min(i + chunk, len(pending))}/{len(pending)} "
                  f"(have {len(got)})", flush=True)
        pending = [t for t in tickers if t not in got]
        # Symbols still missing after round 1 are mostly dead tickers; keep retrying a few times.
        if not pending or (rnd >= 1 and len(pending) < 50):
            break
        time.sleep(15)
    return pd.DataFrame(got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-days", type=int, default=240)
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--hide-obvious", action="store_true")
    args = ap.parse_args()

    names = load_universe()
    print(f"{len(names)} symbols in universe")

    if os.path.exists(CACHE) and not args.refresh:
        prices = pd.read_pickle(CACHE)
    else:
        prices = download(sorted(set(names) | {TARGET}))
        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices.to_pickle(CACHE)
    print(f"prices for {prices.shape[1]} tickers, {len(prices)} days")

    # Drop thin / penny / stale series: they create noisy or fake correlations.
    ok = (prices.notna().sum() >= args.min_days) & (prices.median() >= args.min_price)
    prices = prices.loc[:, ok | (prices.columns == TARGET)]
    rets = prices.pct_change(fill_method=None)
    # Zero-return days (no trading) inflate nothing but dilute; drop stocks that rarely move.
    active = (rets != 0).sum() >= args.min_days * 0.9
    rets = rets.loc[:, active | (rets.columns == TARGET)]
    print(f"{rets.shape[1] - 1} stocks pass quality filters")

    nvda = rets[TARGET]
    others = rets.drop(columns=TARGET)
    if args.hide_obvious:
        others = others.drop(columns=[c for c in OBVIOUS if c in others], errors="ignore")

    def corr_window(n, lag=0):
        x = nvda.shift(lag).tail(n)
        sub = others.tail(n)
        cnt = sub.notna().sum()
        return sub.corrwith(x).where(cnt >= int(n * 0.8))

    df = pd.DataFrame({
        "name": [names.get(t, "")[:45] for t in others.columns],
        "corr_6mo": corr_window(126).values,
        "corr_1y": corr_window(252).values,
        "corr_2y": corr_window(504).values,
        "lag_nvda_leads_1d": corr_window(252, lag=1).values,   # stock today vs NVDA yesterday
        "lag_stock_leads_1d": corr_window(252, lag=-1).values,  # stock today vs NVDA tomorrow
    }, index=others.columns)
    df.index.name = "ticker"

    # beta of each stock to NVDA over 1y
    n = 252
    x = nvda.tail(n)
    df["beta_1y"] = [others[t].tail(n).cov(x) / x.var() for t in df.index]
    df["min_corr"] = df[["corr_6mo", "corr_1y", "corr_2y"]].min(axis=1)
    df = df.dropna(subset=["corr_1y"]).sort_values("corr_1y", ascending=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nTop {args.top} by 1y correlation with {TARGET}\n")
    print(df.head(args.top).round(3).to_string())
    print("\nTop 15 most stable across 6mo/1y/2y (min corr):")
    print(df.sort_values("min_corr", ascending=False).head(15).round(3).to_string())
    print("\nTop 10 negatively correlated (hedges):")
    print(df.tail(10).round(3).to_string())

    for th in (0.85, 0.7, 0.6, 0.5):
        print(f"above {th}: {(df.corr_1y > th).sum()}")
    df.to_csv("nvda_correlations_huge.csv")
    print("saved nvda_correlations_huge.csv")


if __name__ == "__main__":
    main()
