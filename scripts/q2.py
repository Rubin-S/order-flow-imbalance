"""q2.py - Q2: do deeper book levels help explain the SAME-bucket price move?

Raw OFI10 did not beat OFI1 in Q1. The reason: 1 BTC of flow at level 8 means
far less than 1 BTC at the touch, because the deep book is thicker and mostly
cancel-and-replace noise. So, as pre-registered:

  1. Normalise. For each level m, divide its bucket OFI by D_m, the average
     depth of levels 1..m (bid and ask) over the PREVIOUS bucket. Using the
     previous bucket means nothing from inside the bucket leaks into the scale.
       n_m   = e_m summed over the bucket / D_m             (one level)
       N1, N5, N10 = (e_1 + ... + e_m) / D_m                (cumulative)
  2. Integrate. PCA on the 10 series n_1..n_10, pooled over TRAINING days only
     (the development days here). The first component's weights, scaled to sum
     to 1, give one number per bucket: integrated OFI  I = sum_m w_m * n_m.
  3. Fit dP ~ X (OLS, HAC errors) for raw OFI1/5/10, normalised N1/N5/N10, and I.

Still ACCOUNTING, not prediction. Q3/Q4 use the same features, but lagged.

usage: python scripts/q2.py                 (development days only)
       python scripts/q2.py --holdout       (adds Aug-Oct, ONLY for the final run;
                                             PCA weights stay fit on dev days)
writes results/q2_fits.csv and results/pca_weights.csv
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

DEV = ["20260401", "20260501", "20260601", "20260701"]
HOLDOUT = ["20260801", "20260901", "20261001"]
DTS_MS = [100, 1000, 10000, 60000]
TICK = 0.1
LV = range(1, 11)
SPECS = {                                     # name -> one regressor column
    "OFI1": "raw1", "OFI5": "raw5", "OFI10": "raw10",
    "N1": "N1", "N5": "N5", "N10": "N10", "I": "I",
}


def buckets(df, day_start_us, dt_ms):
    """One row per clock bucket: per-level flow, depth scales, end mid, keep flag."""
    b = (df["ts_exch_us"] - day_start_us) // (dt_ms * 1000)

    # average depth of levels 1..m, per row: (sum of bid+ask sizes) / (2m)
    cols = {"b": b, "mid": df["mid"], "bad": ~df["keep"]}
    cum = 0
    for m in LV:
        cum = cum + df[f"bq{m}"] + df[f"aq{m}"]
        cols[f"D{m}"] = cum / (2 * m)
        cols[f"e{m}"] = df[f"e{m}"]
    agg = {"mid": ("mid", "last"), "bad": ("bad", "any")}
    agg.update({f"e{m}": (f"e{m}", "sum") for m in LV})
    agg.update({f"D{m}": (f"D{m}", "mean") for m in LV})
    g = pd.DataFrame(cols).groupby("b").agg(**agg)

    full = pd.RangeIndex(g.index.min(), g.index.max() + 1)      # empty buckets still exist
    g = g.reindex(full)
    ecols, dcols = [f"e{m}" for m in LV], [f"D{m}" for m in LV]
    g[ecols] = g[ecols].fillna(0.0)
    g["bad"] = g["bad"].fillna(False).astype(bool)
    g["mid"] = g["mid"].ffill()
    g[dcols] = g[dcols].ffill().shift(1)                          # PREVIOUS bucket's depth

    cume = g[ecols].cumsum(axis=1)                                # e1, e1+e2, ...
    for m in (1, 5, 10):
        g[f"raw{m}"] = cume[f"e{m}"]
        g[f"N{m}"] = cume[f"e{m}"] / g[f"D{m}"]
    for m in LV:
        g[f"n{m}"] = g[f"e{m}"] / g[f"D{m}"]

    g["dp"] = g["mid"].diff() / TICK
    # drop: masked rows inside, no dP, or no previous depth (first bucket / a reset just before)
    g["keep"] = ~g["bad"] & ~g["bad"].shift(1, fill_value=True) & g["dp"].notna() \
                & (g[dcols] > 0).all(axis=1)
    return g[g["keep"]].copy()


def ols(y, X):
    lags = int(4 * (len(y) / 100) ** (2 / 9))                    # Newey-West rule of thumb
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def pca_weights(frames):
    """First principal component of n1..n10, pooled; sign fixed, scaled to sum to 1."""
    X = pd.concat(frames)[[f"n{m}" for m in LV]].to_numpy()
    X = X - X.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))        # ascending eigenvalues
    w = vecs[:, -1]
    w = w / w.sum()                                              # sum to 1 (also fixes the sign)
    return w, vals[-1] / vals.sum()


holdout = "--holdout" in sys.argv
days = DEV + (HOLDOUT if holdout else [])
feats = {tag: pd.read_parquet(f"data/feat_{tag}.parquet") for tag in days}
starts = {tag: pd.Timestamp(tag, tz="UTC").value // 1000 for tag in days}

fits, wrows = [], []
for dt in DTS_MS:
    g = {tag: buckets(feats[tag], starts[tag], dt) for tag in days}
    w, share = pca_weights([g[t] for t in DEV])                  # TRAINING days only
    wrows.append({"dt_ms": dt, "pc1_share": share, **{f"w{m}": w[m - 1] for m in LV}})
    for tag in days:
        h = g[tag]
        h["I"] = h[[f"n{m}" for m in LV]].to_numpy() @ w
        for name, col in SPECS.items():
            r = ols(h["dp"], h[[col]])
            fits.append({"day": tag, "set": "dev" if tag in DEV else "holdout", "dt_ms": dt,
                         "spec": name, "n": len(h), "beta": r.params[col], "t": r.tvalues[col],
                         "r2": r.rsquared})
    print(f"dt={dt} ms done")

fits = pd.DataFrame(fits)
fits.to_csv("results/q2_fits.csv", index=False)
W = pd.DataFrame(wrows)
W.to_csv("results/pca_weights.csv", index=False)

print("\nSame-bucket R^2, mean over development days")
print(fits.query("set == 'dev'").pivot_table(index="dt_ms", columns="spec", values="r2",
      aggfunc="mean")[list(SPECS)].round(3).to_string())
print("\nPCA weights on n1..n10 (fit on development days), and PC1's share of variance")
print(W.round(3).to_string(index=False))
