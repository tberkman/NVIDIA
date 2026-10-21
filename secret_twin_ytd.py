"""Year-to-date scan vs NVDA using cached prices (US + international).

Windows: YTD (since Jan 1 of the current year), last 6mo, last 3mo.
Reports daily and weekly correlation, best of lag 0/1 for Asian listings.
Usage: python secret_twin_ytd.py [--top 40] [--year 2026]
"""
import argparse
import os
import re

import pandas as pd

import secret_twin_huge as base

TARGET = "NVDA"
FUND_RE = re.compile(r"\b(fund|etn|etf|trust|income|equity|term|leveraged|index|portfolio|"
                     r"enhanced|tax-managed|artificial intelligence & technology)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--year", type=int, default=pd.Timestamp.now().year)
    ap.add_argument("--min-frac", type=float, default=0.9, help="min fraction of window days with data")
    args = ap.parse_args()

    names = base.load_universe()
    us = pd.read_pickle(base.CACHE)
    intl = pd.read_pickle("intl_cache.pkl") if os.path.exists("intl_cache.pkl") else pd.DataFrame()
    if not intl.empty:
        intl = intl.reindex(us.index).ffill(limit=3)  # align local holidays to the US calendar
    px = pd.concat([us.drop(columns=TARGET), intl], axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    nv = us[TARGET].dropna()
    start = pd.Timestamp(f"{args.year}-01-01")
    print(f"data through {nv.index[-1].date()}, YTD days: {(nv.index >= start).sum()}")

    # Penny / stale filter on the YTD slice
    ytd = px[px.index >= start]
    ok = (ytd.median() >= 2) | ytd.columns.str.contains(r"\.")
    px = px.loc[:, ok]

    nd = nv.pct_change()
    results = {}
    windows = {"ytd": start, "6mo": nv.index[-1] - pd.DateOffset(months=6),
               "3mo": nv.index[-1] - pd.DateOffset(months=3)}
    for label, st in windows.items():
        sub = px[px.index >= st]
        r = sub.pct_change(fill_method=None)
        need = int(len(r) * args.min_frac)
        valid = (r.notna().sum() >= need) & ((r != 0).sum() >= need * 0.8)
        r = r.loc[:, valid]
        x0 = nd.reindex(r.index)
        x1 = nd.shift(1).reindex(r.index)  # previous US session (Asia trades first)
        results[f"d_{label}"] = pd.concat([r.corrwith(x0), r.corrwith(x1)], axis=1).max(axis=1)
        results[f"lag1_{label}"] = r.corrwith(x1) > r.corrwith(x0)
        wk = sub.resample("W-FRI").last().pct_change(fill_method=None).iloc[1:]
        wk = wk.loc[:, wk.notna().sum() >= len(wk) * 0.9]
        nw = nv.resample("W-FRI").last().pct_change().reindex(wk.index)
        results[f"w_{label}"] = wk.corrwith(nw)
        if label == "ytd":
            x = nd.reindex(r.index)
            results["beta_ytd"] = pd.Series({t: r[t].cov(x) / x.var() for t in r.columns})
    df = pd.DataFrame(results)
    df["name"] = [names.get(t, "")[:38] for t in df.index]
    df["market"] = ["INTL" if "." in t else "US" for t in df.index]
    df = df[~df["name"].str.contains(FUND_RE)].dropna(subset=["d_ytd"])
    df["consistency"] = df[["d_ytd", "d_6mo", "d_3mo"]].min(axis=1)
    df["blend"] = df[["d_ytd", "w_ytd"]].mean(axis=1)

    cols = ["market", "name", "d_ytd", "w_ytd", "d_6mo", "d_3mo", "beta_ytd"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nScanned {len(df)} stocks (funds removed)")
    print(f"\nTop {args.top} by YTD daily correlation\n")
    print(df.sort_values("d_ytd", ascending=False)[cols].head(args.top).round(2).to_string())
    print("\nTop 20 by YTD weekly correlation\n")
    print(df.sort_values("w_ytd", ascending=False)[cols].head(20).round(2).to_string())
    print("\nTop 15 most consistent (min of YTD/6mo/3mo daily)\n")
    print(df.sort_values("consistency", ascending=False)[cols + ["consistency"]].head(15).round(2).to_string())
    print("\nTop 10 international\n")
    print(df[df.market == "INTL"].sort_values("d_ytd", ascending=False)[cols].head(10).round(2).to_string())
    for th in (0.85, 0.7, 0.6, 0.5):
        print(f"daily>{th}: {(df.d_ytd > th).sum()}   weekly>{th}: {(df.w_ytd > th).sum()}")
    df.to_csv("nvda_correlations_ytd.csv")


if __name__ == "__main__":
    main()
