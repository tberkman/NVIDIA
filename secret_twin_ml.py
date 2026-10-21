"""Walk-forward regularized regression: how well can the whole market explain NVDA?

Every reported correlation is out-of-sample: models are fit on past days only
and predict the next block of days.
Usage: python secret_twin_ml.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA

import secret_twin_huge as base

TARGET = "NVDA"
EXCLUDE = {"AIO", "ETV", "ETY", "EOS", "FFA", "NIE", "JCE", "BSTZ", "FNGO"}


def walk_forward(X, y, make_model, train_min=250, step=21):
    preds = pd.Series(index=y.index, dtype=float)
    for s in range(train_min, len(y), step):
        e = min(s + step, len(y))
        m = make_model()
        m.fit(X.iloc[:s], y.iloc[:s])
        preds.iloc[s:e] = m.predict(X.iloc[s:e])
    ok = preds.notna()
    return preds[ok], float(np.corrcoef(preds[ok], y[ok])[0, 1])


def main():
    px = pd.read_pickle(base.CACHE)
    px = px.loc[:, (px.notna().sum() >= 480) & ((px.median() >= 2) | (px.columns == TARGET))]
    px = px.drop(columns=[c for c in EXCLUDE if c in px], errors="ignore")
    r = px.pct_change(fill_method=None).iloc[1:].dropna()
    r = r.clip(-0.5, 0.5)
    y = r[TARGET]
    X = r.drop(columns=TARGET)
    X = (X - X.mean()) / X.std()
    print(f"{X.shape[1]} stocks, {len(r)} days, evaluating on last {len(r) - 250} days\n")

    # Pre-filter to the most correlated names using only the first 250 days (no lookahead).
    for pool in (50, 200, 1000):
        top = X.iloc[:250].corrwith(y.iloc[:250]).abs().sort_values(ascending=False).index[:pool]
        Xp = X[top]
        _, c = walk_forward(Xp, y, lambda: RidgeCV(alphas=np.logspace(0, 4, 12)))
        print(f"Ridge,  pool={pool:<5} OOS corr {c:.3f}")
        if pool <= 200:
            _, c = walk_forward(Xp, y, lambda: LassoCV(cv=3, n_alphas=20, max_iter=3000), step=42)
            print(f"Lasso,  pool={pool:<5} OOS corr {c:.3f}")

    # PCA factors of the full market + ridge
    class PcaRidge:
        def __init__(self, k=20):
            self.p, self.m = PCA(k), RidgeCV(alphas=np.logspace(-1, 3, 10))
        def fit(self, A, b):
            self.m.fit(self.p.fit_transform(A), b)
        def predict(self, A):
            return self.m.predict(self.p.transform(A))

    for k in (5, 10, 20, 40):
        _, c = walk_forward(X, y, lambda: PcaRidge(k))
        print(f"PCA({k}) + ridge, all stocks   OOS corr {c:.3f}")

    # Weekly version of the best simple approach
    wk = px.resample("W-FRI").last().pct_change(fill_method=None).iloc[1:].dropna(axis=1)
    wk = wk.clip(-0.5, 0.5)
    yw = wk[TARGET]
    Xw = wk.drop(columns=TARGET)
    Xw = (Xw - Xw.mean()) / Xw.std()
    top = Xw.iloc[:52].corrwith(yw.iloc[:52]).abs().sort_values(ascending=False).index[:100]
    _, c = walk_forward(Xw[top], yw, lambda: RidgeCV(alphas=np.logspace(0, 4, 12)), train_min=52, step=4)
    print(f"\nWeekly ridge pool=100 OOS corr {c:.3f}  ({len(yw) - 52} test weeks)")


if __name__ == "__main__":
    main()
