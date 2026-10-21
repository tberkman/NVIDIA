"""Global scan: US (cached) + Japan + Taiwan + Hong Kong vs NVDA.

Asian exchanges close before the US opens, so each stock is tested against NVDA
on the same date (lag 0) and against NVDA's previous US session (lag 1); the best
of the two is reported. Weekly-return correlations are also computed because
daily noise hides real co-movement.

Usage:
    python secret_twin_global.py --build-intl      # download international prices (slow, resumable)
    python secret_twin_global.py                   # analyze using caches
"""
import argparse
import os
import re
import time

import pandas as pd

import secret_twin_huge as base

TARGET = "NVDA"
INTL_CACHE = "intl_cache.pkl"
FUND_RE = re.compile(r"\b(fund|etn|etf|trust|income|equity|term|leveraged|index|portfolio)\b", re.I)


def intl_symbols():
    syms = [f"{c}.T" for c in range(1300, 10000)]                                  # Tokyo
    syms += [f"{c}.TW" for c in list(range(1100, 7000)) + list(range(8000, 9000))]  # Taiwan
    syms += [f"{c:04d}.HK" for c in range(1, 4000)]                                 # Hong Kong
    return syms


def build_intl():
    have = pd.read_pickle(INTL_CACHE) if os.path.exists(INTL_CACHE) else pd.DataFrame()
    done = set(open("intl_tried.txt").read().split()) if os.path.exists("intl_tried.txt") else set()
    todo = [s for s in intl_symbols() if s not in done and s not in have.columns]
    print(f"{len(todo)} international symbols to try", flush=True)
    chunk = 80
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        try:
            d = base.fetch(part)
            have = pd.concat([have, d.loc[:, ~d.columns.isin(have.columns)]], axis=1)
            with open("intl_tried.txt", "a") as f:
                f.write("\n".join(part) + "\n")
        except Exception as e:
            print("chunk failed:", e)
        time.sleep(3)
        if (i // chunk) % 5 == 0:
            have.to_pickle(INTL_CACHE)
        print(f"{min(i + chunk, len(todo))}/{len(todo)} tried, {have.shape[1]} found", flush=True)
    have.to_pickle(INTL_CACHE)


def corr_stats(stock_rets, nvda, n, lag):
    x = nvda.shift(lag).reindex(stock_rets.index).tail(n)
    sub = stock_rets.tail(n)
    return sub.corrwith(x).where(sub.notna().sum() >= int(n * 0.7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-intl", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    if args.build_intl:
        return build_intl()

    names = base.load_universe()
    us = pd.read_pickle(base.CACHE)
    intl = pd.read_pickle(INTL_CACHE) if os.path.exists(INTL_CACHE) else pd.DataFrame()
    intl = intl.loc[:, intl.notna().sum() >= 200]
    us = us.loc[:, (us.notna().sum() >= 240) & ((us.median() >= 2) | (us.columns == TARGET))]
    print(f"US {us.shape[1] - 1}, international {intl.shape[1]} stocks", flush=True)

    nvda_px = us[TARGET].dropna()
    nvda_d = nvda_px.pct_change()
    nvda_w = nvda_px.resample("W-FRI").last().pct_change()

    rows = []
    for market, px in (("US", us.drop(columns=TARGET)), ("INTL", intl)):
        if px.empty:
            continue
        rd = px.pct_change(fill_method=None)
        rd = rd.loc[:, (rd != 0).sum() >= 150]
        rw = px.resample("W-FRI").last().pct_change(fill_method=None)[rd.columns]
        out = pd.DataFrame(index=rd.columns)
        out["market"] = market
        out["d_lag0_1y"] = corr_stats(rd, nvda_d, 252, 0)
        out["d_lag1_1y"] = corr_stats(rd, nvda_d, 252, 1)
        out["d_lag0_2y"] = corr_stats(rd, nvda_d, 504, 0)
        out["d_lag1_2y"] = corr_stats(rd, nvda_d, 504, 1)
        out["w_1y"] = rw.tail(52).corrwith(nvda_w.reindex(rw.index).tail(52))
        out["w_2y"] = rw.tail(104).corrwith(nvda_w.reindex(rw.index).tail(104))
        rows.append(out)
    df = pd.concat(rows)
    df.index.name = "ticker"
    df["name"] = [names.get(t, "")[:40] for t in df.index]
    df = df[~df["name"].str.contains(FUND_RE)]  # drop funds / ETNs that merely hold NVDA

    df["daily_1y"] = df[["d_lag0_1y", "d_lag1_1y"]].max(axis=1)
    df["daily_2y"] = df[["d_lag0_2y", "d_lag1_2y"]].max(axis=1)
    df["best_lag"] = (df.d_lag1_1y > df.d_lag0_1y).astype(int)
    df["score"] = df[["daily_1y", "w_1y"]].mean(axis=1)  # blended daily/weekly rank

    cols = ["market", "name", "daily_1y", "daily_2y", "best_lag", "w_1y", "w_2y"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nTotal scanned: {len(df)}")
    print(f"\nTop {args.top} by DAILY 1y correlation\n")
    print(df.sort_values("daily_1y", ascending=False)[cols].head(args.top).round(2).to_string())
    print("\nTop 25 by WEEKLY 1y correlation\n")
    print(df.sort_values("w_1y", ascending=False)[cols].head(25).round(2).to_string())
    print("\nTop 15 international (daily)\n")
    print(df[df.market == "INTL"].sort_values("daily_1y", ascending=False)[cols].head(15).round(2).to_string())
    for th in (0.85, 0.7, 0.6, 0.5):
        print(f"daily>{th}: {(df.daily_1y > th).sum()}   weekly>{th}: {(df.w_1y > th).sum()}")
    df.to_csv("nvda_correlations_global.csv")


if __name__ == "__main__":
    main()
