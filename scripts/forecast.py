"""forecast.py - Q3 + Q4: can flow we ALREADY KNOW predict the NEXT price move?

Q1/Q2 explained a bucket's move with that same bucket's flow (accounting).
Here every feature is known at the END of bucket t, we skip bucket t+1
(the embargo: time to compute and send an order), and we predict the move
over bucket t+2. Nothing from the future touches a feature.

  features at end of bucket t
    state : qi1 (level-1 queue imbalance), imb5 (levels 1-5), micro (microprice - mid,
            ticks), dp (this bucket's move, ticks)
    flow  : OFI1, OFI5, OFI10 (raw, summed over bucket t), I (integrated OFI:
            depth-normalised levels, PCA weights fit on the TRAINING days of each fold)
  target  : mid change over bucket t+2, in ticks
  latency : Q4 also shifts the target window later by L ms (L = 0 means just the embargo)

Walk-forward (development days only): train Apr -> test May, Apr-May -> Jun, Apr-Jun -> Jul.
Models: zero forecast (the baseline), recent dP only, OLS on each feature set, ridge on
state+flow (alpha picked on the last training day; with one training day, its last quarter).
Score: out-of-sample R^2 = 1 - SSE(model) / SSE(zero). Zero or below = no edge.

usage: python scripts/forecast.py               (development walk-forward)
       python scripts/forecast.py --holdout     (train Apr-Jul, test Aug/Sep/Oct, ONLY for the final run)
writes results/forecast_oos.csv
"""
import sys
import numpy as np
import pandas as pd

DEV = ["20260401", "20260501", "20260601", "20260701"]
HOLDOUT = ["20260801", "20260901", "20261001"]
DTS_MS = [100, 1000, 10000, 60000]
LAT_MS = [0, 10, 25, 50, 100]
TICK = 0.1
LV = range(1, 11)
STATE = ["qi1", "imb5", "micro", "dp"]
SETS = {                                       # OLS feature sets (all with an intercept)
    "recent_dp": ["dp"],
    "OFI1": ["ofi1"], "OFI10": ["ofi10"], "I": ["I"],
    "state": STATE,
    "state+flow": STATE + ["ofi1", "ofi5", "ofi10", "I"],
}
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
MIN_N = 30                                     # fewer test buckets than this: skip, don't guess


def build(tag, dt_ms):
    """Per-bucket features (known at bucket end) and targets for every latency."""
    df = pd.read_parquet(f"data/feat_{tag}.parquet")
    start = pd.Timestamp(tag, tz="UTC").value // 1000
    dt = dt_ms * 1000
    b = (df["ts_exch_us"] - start) // dt

    cols = {"b": b, "mid": df["mid"], "spread": df["spread"], "bad": ~df["keep"]}
    cum = 0
    for m in LV:
        cum = cum + df[f"bq{m}"] + df[f"aq{m}"]
        cols[f"D{m}"] = cum / (2 * m)
        cols[f"e{m}"] = df[f"e{m}"]
    for m in range(1, 6):
        cols[f"bq{m}"], cols[f"aq{m}"] = df[f"bq{m}"], df[f"aq{m}"]
    agg = {"mid": ("mid", "last"), "spread": ("spread", "last"), "bad": ("bad", "any")}
    agg.update({f"e{m}": (f"e{m}", "sum") for m in LV})
    agg.update({f"D{m}": (f"D{m}", "mean") for m in LV})
    agg.update({f"{s}q{m}": (f"{s}q{m}", "last") for s in "ba" for m in range(1, 6)})
    g = pd.DataFrame(cols).groupby("b").agg(**agg)

    g = g.reindex(pd.RangeIndex(g.index.min(), g.index.max() + 1))     # empty buckets exist
    ecols, dcols = [f"e{m}" for m in LV], [f"D{m}" for m in LV]
    g[ecols] = g[ecols].fillna(0.0)
    g["bad"] = g["bad"].fillna(False).astype(bool)
    g = g.ffill()                                                       # book state carries over
    g[dcols] = g[dcols].shift(1)                                        # PREVIOUS bucket's depth

    cume = g[ecols].cumsum(axis=1)
    for m in (1, 5, 10):
        g[f"ofi{m}"] = cume[f"e{m}"]
    for m in LV:
        g[f"n{m}"] = g[f"e{m}"] / g[f"D{m}"]
    g["qi1"] = (g["bq1"] - g["aq1"]) / (g["bq1"] + g["aq1"])
    B5 = sum(g[f"bq{m}"] for m in range(1, 6)); A5 = sum(g[f"aq{m}"] for m in range(1, 6))
    g["imb5"] = (B5 - A5) / (B5 + A5)
    g["micro"] = g["spread"] / 2 * g["qi1"] / TICK                      # microprice - mid, ticks
    g["dp"] = g["mid"].diff() / TICK
    ok = ~g["bad"] & ~g["bad"].shift(1, fill_value=True) & g["dp"].notna() & (g[dcols] > 0).all(axis=1)

    # targets from the raw message stream: mid at any time = last mid at or before it
    ts, mid = df["ts_exch_us"].to_numpy(), df["mid"].to_numpy()
    badcum = np.cumsum(~df["keep"].to_numpy())
    for L in LAT_MS:
        s = start + (g.index.to_numpy() + 2) * dt + L * 1000            # embargo: skip bucket t+1
        i0 = np.searchsorted(ts, s, "right") - 1
        i1 = np.searchsorted(ts, s + dt, "right") - 1
        good = (i0 >= 0) & (i1 < len(ts) - 1)                           # window inside the data
        i0c, i1c = np.clip(i0, 0, None), np.clip(i1, 0, None)
        good &= badcum[i1c] == badcum[i0c]                              # no masked row in the window
        g[f"y{L}"] = np.where(good, (mid[i1c] - mid[i0c]) / TICK, np.nan)
    return g[ok].copy()


def pca_w(frames):
    X = pd.concat(frames)[[f"n{m}" for m in LV]].to_numpy()
    vals, vecs = np.linalg.eigh(np.cov(X - X.mean(axis=0), rowvar=False))
    w = vecs[:, -1]
    return w / w.sum()


def ols_fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def ols_pred(c, X):
    return c[0] + X @ c[1:]


def ridge_fit(X, y, a):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12                      # scale with TRAIN stats
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, y.mean(), beta


def ridge_pred(p, X):
    mu, sd, y0, beta = p
    return y0 + ((X - mu) / sd) @ beta


def r2_oos(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum(y ** 2)                 # vs the zero forecast


holdout = "--holdout" in sys.argv
if holdout:
    folds = [(DEV, h) for h in HOLDOUT]
else:
    folds = [(DEV[:k], DEV[k]) for k in range(1, len(DEV))]
days = sorted({d for tr, te in folds for d in tr + [te]})

rows = []
for dt in DTS_MS:
    G = {d: build(d, dt) for d in days}
    for train, test in folds:
        w = pca_w([G[d] for d in train])                                # TRAINING days only
        for d in train + [test]:
            G[d]["I"] = G[d][[f"n{m}" for m in LV]].to_numpy() @ w
        for L in LAT_MS:
            tr = pd.concat([G[d] for d in train]).dropna(subset=[f"y{L}"])
            te = G[test].dropna(subset=[f"y{L}"])
            ytr, yte = tr[f"y{L}"].to_numpy(), te[f"y{L}"].to_numpy()
            if len(yte) < MIN_N or len(ytr) < 2 * MIN_N:                 # 60 s buckets on short data
                print(f"  skip dt={dt} L={L} test={test}: too few buckets ({len(ytr)} train, {len(yte)} test)")
                continue
            out = {"zero": np.zeros(len(yte))}
            for name, cols in SETS.items():
                out[name] = ols_pred(ols_fit(tr[cols].to_numpy(), ytr), te[cols].to_numpy())

            # ridge alpha: validate on the last training day (or the last quarter of a single day)
            cols = SETS["state+flow"]
            if len(train) > 1:
                fit_part = pd.concat([G[d] for d in train[:-1]]).dropna(subset=[f"y{L}"])
                val_part = G[train[-1]].dropna(subset=[f"y{L}"])
            else:
                cut = int(len(tr) * 0.75)
                fit_part, val_part = tr.iloc[:cut], tr.iloc[cut:]
            best = max(ALPHAS, key=lambda a: r2_oos(val_part[f"y{L}"].to_numpy(), ridge_pred(
                ridge_fit(fit_part[cols].to_numpy(), fit_part[f"y{L}"].to_numpy(), a),
                val_part[cols].to_numpy())))
            out["ridge"] = ridge_pred(ridge_fit(tr[cols].to_numpy(), ytr, best), te[cols].to_numpy())

            for name, pred in out.items():
                rows.append({"train": "+".join(train), "test": test, "dt_ms": dt, "latency_ms": L,
                             "model": name, "n_train": len(ytr), "n_test": len(yte),
                             "alpha": best if name == "ridge" else np.nan,
                             "r2_oos": r2_oos(yte, pred)})
    print(f"dt={dt} ms done")

res = pd.DataFrame(rows)
res.to_csv("results/forecast_oos.csv", index=False)
order = ["zero", "recent_dp", "OFI1", "OFI10", "I", "state", "state+flow", "ridge"]

print("\nOut-of-sample R^2 (%), latency 0, mean over test days (rows: horizon, cols: model)")
print((100 * res.query("latency_ms == 0").pivot_table(index="dt_ms", columns="model",
       values="r2_oos", aggfunc="mean")[order]).round(2).to_string())
print("\nOut-of-sample R^2 (%) by test day, state+flow OLS, latency 0")
print((100 * res.query("latency_ms == 0 and model == 'state+flow'").pivot_table(
       index="dt_ms", columns="test", values="r2_oos")).round(2).to_string())
print("\nQ4 latency: OOS R^2 (%), state+flow OLS, mean over test days (rows: horizon, cols: latency ms)")
print((100 * res.query("model == 'state+flow'").pivot_table(index="dt_ms", columns="latency_ms",
       values="r2_oos", aggfunc="mean")).round(2).to_string())
