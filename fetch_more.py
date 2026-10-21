"""Resumable download of extra stock universes into more_cache.pkl.

Sets: us_missing (US symbols with no cached prices), kr (KOSPI/KOSDAQ codes),
      intl_retry (Tokyo/Taiwan/HK symbols not yet found).
Usage: python fetch_more.py us_missing|kr|intl_retry
"""
import os
import sys
import time

import pandas as pd

import secret_twin_huge as base
import secret_twin_global as g

CACHE = "more_cache.pkl"
TRIED = "more_tried.txt"


def symbols(kind):
    if kind == "us_missing":
        have = set(pd.read_pickle(base.CACHE).columns)
        return [s for s in base.load_universe() if s not in have]
    if kind == "idx":
        import intl_indexes
        return list(intl_indexes.build())
    if kind == "kr":
        codes = [f"{c:06d}" for c in range(10, 400000, 10)]
        return [f"{c}.KS" for c in codes] + [f"{c}.KQ" for c in codes]
    if kind == "intl_retry":
        have = set(pd.read_pickle("intl_cache.pkl").columns)
        return [s for s in g.intl_symbols() if s not in have]
    raise SystemExit("unknown set")


def main():
    kind = sys.argv[1]
    have = pd.read_pickle(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    tried = set(open(TRIED).read().split()) if os.path.exists(TRIED) else set()
    todo = [s for s in symbols(kind) if s not in tried and s not in have.columns]
    print(f"{kind}: {len(todo)} to try", flush=True)
    chunk, pause = 100, 2.5
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        try:
            d = base.fetch(part)
            d = d.loc[:, d.notna().sum() >= 100]
            have = pd.concat([have, d.loc[:, ~d.columns.isin(have.columns)]], axis=1)
            with open(TRIED, "a") as f:
                f.write("\n".join(part) + "\n")
        except Exception as e:
            print("chunk failed:", e, flush=True)
        time.sleep(pause)
        if (i // chunk) % 5 == 0:
            have.to_pickle(CACHE)
        print(f"{min(i + chunk, len(todo))}/{len(todo)} tried, {have.shape[1]} found", flush=True)
    have.to_pickle(CACHE)


if __name__ == "__main__":
    main()
