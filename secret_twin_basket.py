"""Basket of stocks that tracks NVDA's daily returns.

1. Equal-weight basket of hand-picked names.
2. Greedy forward selection over the whole cached universe (train on first
   ~70% of the last 2y, evaluate on the held-out most recent ~30%).
Usage: python secret_twin_basket.py [--k 10]
"""
import argparse

import numpy as np
import pandas as pd

import secret_twin_huge as base

TARGET = "NVDA"
PICKS = ["VRT", "TSM", "AVGO", "ENTG", "NVMI"]
EXCLUDE = {"AIO", "ETV", "ETY", "EOS", "FFA", "NIE", "JCE", "BSTZ", "FNGO"}  # funds holding NVDA


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=300, help="candidates pre-ranked by train correlation")
    args = ap.parse_args()

    px = pd.read_pickle(base.CACHE)
    px = px.loc[:, (px.notna().sum() >= 480) & ((px.median() >= 2) | (px.columns == TARGET))]
    px = px.drop(columns=[c for c in EXCLUDE if c in px], errors="ignore")
    r = px.pct_change(fill_method=None).iloc[1:].dropna()  # common full-history window
    y = r[TARGET]
    X = r.drop(columns=TARGET)
    cut = int(len(r) * 0.7)
    tr, te = slice(0, cut), slice(cut, None)
    print(f"{X.shape[1]} stocks, {len(r)} days; train {cut}, test {len(r) - cut}")

    eq = X[PICKS].mean(axis=1)
    print(f"\nEqual-weight {PICKS}:")
    print(f"  corr train {corr(eq[tr], y[tr]):.3f}   test {corr(eq[te], y[te]):.3f}")

    # Greedy forward selection using OLS fit on train; score on test.
    pool = X.iloc[tr].corrwith(y.iloc[tr]).sort_values(ascending=False).index[:args.pool]
    chosen = []
    print(f"\nGreedy basket (pool={len(pool)}), OLS weights fit on train only:")
    print(f"{'k':>2} {'added':<8} {'train':>6} {'test':>6}")
    for k in range(1, args.k + 1):
        best, best_c = None, -1
        for t in pool:
            if t in chosen:
                continue
            cols = chosen + [t]
            A = np.c_[X.iloc[tr][cols].values, np.ones(cut)]
            w = np.linalg.lstsq(A, y.iloc[tr].values, rcond=None)[0]
            c = corr(A @ w, y.iloc[tr].values)
            if c > best_c:
                best, best_c = t, c
        chosen.append(best)
        A = np.c_[X.iloc[tr][chosen].values, np.ones(cut)]
        w = np.linalg.lstsq(A, y.iloc[tr].values, rcond=None)[0]
        pred_te = np.c_[X.iloc[te][chosen].values, np.ones(len(r) - cut)] @ w
        print(f"{k:>2} {best:<8} {best_c:6.3f} {corr(pred_te, y.iloc[te].values):6.3f}")

    print("\nFinal weights (per $1 of NVDA daily move):")
    for t, wt in zip(chosen, w[:-1]):
        print(f"  {t:<6} {wt:+.3f}")

    # Larger baskets, ridge-free: how far can test correlation go?
    print("\nTop-N equal-weight basket by train correlation (no fitting):")
    ranked = X.iloc[tr].corrwith(y.iloc[tr]).sort_values(ascending=False).index
    for n in (5, 10, 20, 40, 80):
        b = X[ranked[:n]].mean(axis=1)
        print(f"  top {n:<3} train {corr(b[tr], y[tr]):.3f}  test {corr(b[te], y[te]):.3f}")


if __name__ == "__main__":
    main()
