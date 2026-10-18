"""bootstrap.py - the pre-registered pass rule: moving-block bootstrap CIs.

For each frozen comparison "model A vs model B" it asks: is
R^2_oos(A) - R^2_oos(B) really different from zero, or could it be luck?

How: on each test day, cut the buckets into overlapping 5-minute blocks.
A resample draws blocks (with replacement) until it has as many buckets as the
day, does that for every test day, and recomputes the difference. 2,000
resamples give a distribution; its 2.5% and 97.5% points are the 95% CI.
Blocks (not single buckets) keep the autocorrelation inside each 5 minutes.

Trick that makes it fast: R^2_oos only needs three sums (SSE of A, SSE of B,
sum of y^2), and the sum over any block is a difference of cumulative sums.
So a resample is a few thousand additions, not a refit.

Also bootstraps the Q5 headline: mean gross bps per trade for 1 s, q90, latency 0,
compared with the 4 bps maker round-trip fee.

usage: python scripts/bootstrap.py             (dev walk-forward test days, data/preds.parquet)
       python scripts/bootstrap.py --holdout   (ONLY for the final run)
writes results/bootstrap.csv (or results/bootstrap_holdout.csv)
"""
import sys
import numpy as np
import pandas as pd

BLOCK_S = 300                   # 5-minute blocks
N_BOOT = 2000
SEED = 20261018                 # fixed, so the CI is reproducible
TICK = 0.1
MAKER_RT_BPS = 4.0
# frozen comparisons: (name, dt_ms, model A, model B). "zero" = the no-move forecast
TESTS = [
    ("edge: ridge vs zero", 100, "ridge", "zero"),
    ("edge: ridge vs zero", 1000, "ridge", "zero"),
    ("Q3: state+flow vs state", 100, "state+flow", "state"),
    ("Q3: state+flow vs state", 1000, "state+flow", "state"),
    ("Q2: OFI10 vs OFI1", 1000, "OFI10", "OFI1"),
    ("Q2: I vs OFI1", 1000, "I", "OFI1"),
]
ECON = (1000, "q90")            # the frozen Q5 headline config, latency 0

holdout = "--holdout" in sys.argv
p = pd.read_parquet("data/preds_holdout.parquet" if holdout else "data/preds.parquet")
p = p[p["latency_ms"] == 0]
rng = np.random.default_rng(SEED)


def block_sums(cols, n_block):
    """For one day: sums of each column over every possible block start."""
    cs = np.vstack([np.zeros(cols.shape[1]), np.cumsum(cols, axis=0)])
    return cs[n_block:] - cs[:-n_block]                     # (n - n_block + 1, n_cols)


def resample(frame, cols, dt):
    """Point estimate and N_BOOT resampled totals of `cols`, blocks drawn within each day."""
    n_block = max(1, BLOCK_S * 1000 // dt)
    point = np.zeros(len(cols)); boot = np.zeros((N_BOOT, len(cols)))
    for _, d in frame.sort_values("bucket").groupby("test"):
        X = d[cols].to_numpy(dtype=float)
        point += X.sum(axis=0)
        nb = min(n_block, len(X))
        S = block_sums(X, nb)
        k = int(np.ceil(len(X) / nb))                        # blocks per resampled day
        starts = rng.integers(0, len(S), size=(N_BOOT, k))
        boot += S[starts].sum(axis=1)
    return point, boot


rows = []
for name, dt, a, b in TESTS:
    d = p[p["dt_ms"] == dt].copy()
    d["ea"] = (d["y"] - d[f"p_{a}"]) ** 2
    d["eb"] = (d["y"] - d[f"p_{b}"]) ** 2
    d["y2"] = d["y"] ** 2
    pt, bt = resample(d, ["ea", "eb", "y2"], dt)
    diff = lambda s: (s[..., 1] - s[..., 0]) / s[..., 2]  # R2(A) - R2(B) = (SSE_B - SSE_A) / SST0
    est, dist = diff(pt), diff(bt)
    lo, hi = np.quantile(dist, [0.025, 0.975])
    rows.append({"test": name, "dt_ms": dt, "metric": "R2 diff (pp)", "estimate": 100 * est,
                 "ci_lo": 100 * lo, "ci_hi": 100 * hi, "share_le_0": (dist <= 0).mean(),
                 "pass": bool(lo > 0), "n_buckets": len(d)})

dt, thr = ECON
d = p[p["dt_ms"] == dt].copy()
d["trade"] = ((d["pred"].abs() > d[thr]) & (d["pred"] != 0)).astype(float)
d["gross"] = d["trade"] * np.sign(d["pred"]) * d["y"] * TICK / d["mid"] * 1e4
pt, bt = resample(d, ["gross", "trade"], dt)
dist = bt[:, 0] / np.maximum(bt[:, 1], 1)
lo, hi = np.quantile(dist, [0.025, 0.975])
rows.append({"test": f"Q5: gross bps, {thr}", "dt_ms": dt, "metric": "bps per trade",
             "estimate": pt[0] / pt[1], "ci_lo": lo, "ci_hi": hi, "share_le_0": (dist <= 0).mean(),
             "pass": bool(lo > MAKER_RT_BPS), "n_buckets": int(pt[1])})   # pass = beats maker fees

res = pd.DataFrame(rows)
res.to_csv("results/bootstrap_holdout.csv" if holdout else "results/bootstrap.csv", index=False)
pd.set_option("display.width", 140)
print(f"Moving-block bootstrap: {BLOCK_S} s blocks within each test day, {N_BOOT} resamples, seed {SEED}")
print(res.round({"estimate": 3, "ci_lo": 3, "ci_hi": 3, "share_le_0": 3}).to_string(index=False))
print("\npass = 95% CI lower bound > 0 (R2 rows) or > 4 bps maker fee (Q5 row); n_buckets = trades on the Q5 row")
