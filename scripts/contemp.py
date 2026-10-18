"""contemp.py - Q1: does the price move with order flow in the SAME time bucket?

For each day and each bucket size dt in {100 ms, 1 s, 10 s, 60 s}:
  1. cut the 2-hour window into fixed clock buckets (exchange time)
  2. per bucket: OFI_1, OFI_5, OFI_10 (sum of e1..em), trade imbalance TI
     (buy volume - sell volume, from the trades in the .evt), and
     dP = change in mid over the bucket, in ticks
  3. drop any bucket that touches a masked row (reset or funding burn-in)
  4. OLS  dP ~ X  with Newey-West (HAC) errors, for 5 specs

Also the CKS depth test: at dt = 1 s, fit dP ~ OFI_1 in each 30-minute
piece of each day and compare beta with the average level-1 depth there.
CKS predict beta ~ 1/depth, i.e. a slope near -1 on a log-log plot.

This is ACCOUNTING, not prediction: the same orders build OFI and move the
price inside one bucket. Forecasting comes later.

usage: python scripts/contemp.py                    (development days only)
       python scripts/contemp.py --holdout          (all 7, ONLY for the final run)
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

DEV = ["20260401", "20260501", "20260601", "20260701"]
HOLDOUT = ["20260801", "20260901", "20261001"]
DTS_MS = [100, 1000, 10000, 60000]
TICK = 0.1
SPECS = {                                   # name -> regressors
    "OFI1":    ["ofi1"],
    "TI":      ["ti"],
    "OFI1+TI": ["ofi1", "ti"],
    "OFI5":    ["ofi5"],
    "OFI10":   ["ofi10"],
}

# the engine's 32-byte Event record (include/mdr/event.hpp), read straight into numpy
EVT = np.dtype([("ts_exch", "<i8"), ("ts_local", "<i8"), ("px", "<i8"), ("qty", "<i4"),
                ("side", "u1"), ("kind", "u1"), ("flags", "u1"), ("res", "u1")])
HDR = np.dtype([("magic", "S8"), ("version", "<u2"), ("res", "<u2"), ("px_scale", "<u4"),
                ("qty_scale", "<u4"), ("count", "<u4"), ("min_px", "<i8")])


def load_trades(tag):
    """Signed trade volume in BTC: + when a buyer lifts the ask, - when a seller hits the bid."""
    hdr = np.fromfile(f"data/{tag}.evt", dtype=HDR, count=1)[0]
    ev = np.fromfile(f"data/{tag}.evt", dtype=EVT, offset=HDR.itemsize)
    assert len(ev) == hdr["count"], "event count does not match header"
    tr = ev[ev["kind"] == 1]
    sign = np.where(tr["side"] == 0, 1.0, -1.0)            # side = aggressor; Bid(0) = buyer
    return pd.DataFrame({"ts_exch_us": tr["ts_exch"], "ti": sign * tr["qty"] / hdr["qty_scale"]})


def buckets(df, tr, day_start_us, dt_ms):
    """One row per clock bucket: summed flow, end-of-bucket mid, and a keep flag."""
    dt_us = dt_ms * 1000
    b = (df["ts_exch_us"] - day_start_us) // dt_us
    e = df[[f"e{m}" for m in range(1, 11)]]
    g = pd.DataFrame({
        "b": b,
        "ofi1": e["e1"],
        "ofi5": e.iloc[:, :5].sum(axis=1),
        "ofi10": e.sum(axis=1),
        "mid": df["mid"],
        "bad": ~df["keep"],
        "depth1": (df["bq1"] + df["aq1"]) / 2,
    }).groupby("b").agg(ofi1=("ofi1", "sum"), ofi5=("ofi5", "sum"), ofi10=("ofi10", "sum"),
                        mid=("mid", "last"), bad=("bad", "any"), depth1=("depth1", "mean"))

    tb = (tr["ts_exch_us"] - day_start_us) // dt_us
    g["ti"] = tr.groupby(tb)["ti"].sum()

    # full grid: a bucket with no book message still exists (flow 0, mid unchanged)
    full = pd.RangeIndex(g.index.min(), g.index.max() + 1)
    g = g.reindex(full)
    g[["ofi1", "ofi5", "ofi10", "ti"]] = g[["ofi1", "ofi5", "ofi10", "ti"]].fillna(0.0)
    g["bad"] = g["bad"].fillna(False).astype(bool)
    g["mid"] = g["mid"].ffill()
    g["depth1"] = g["depth1"].ffill()

    g["dp"] = g["mid"].diff() / TICK                     # price move over the bucket, ticks
    g["keep"] = ~g["bad"] & g["dp"].notna()              # a reset inside a bucket ruins its dP
    return g[g["keep"]]


def ols(y, X):
    lags = int(4 * (len(y) / 100) ** (2 / 9))           # Newey-West rule of thumb
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


days = DEV + (HOLDOUT if "--holdout" in sys.argv else [])
fits, depth_rows = [], []
for tag in days:
    df = pd.read_parquet(f"data/feat_{tag}.parquet")
    tr = load_trades(tag)
    day_start_us = pd.Timestamp(tag, tz="UTC").value // 1000
    regime = "52ms" if tag in ("20260401", "20260501") else "26ms"

    for dt in DTS_MS:
        g = buckets(df, tr, day_start_us, dt)
        for name, cols in SPECS.items():
            r = ols(g["dp"], g[cols])
            fits.append({"day": tag, "regime": regime, "dt_ms": dt, "spec": name, "n": len(g),
                         "beta": r.params[cols[0]], "t": r.tvalues[cols[0]], "r2": r.rsquared})

        if dt == 1000:                                   # CKS depth test, 30-minute pieces
            piece = (g.index * dt // 1000) // 1800
            for p, h in g.groupby(piece):
                if len(h) < 600:                         # skip short pieces (< 10 min of data)
                    continue
                r = ols(h["dp"], h[["ofi1"]])
                depth_rows.append({"day": tag, "piece": int(p), "beta": r.params["ofi1"],
                                   "depth1": h["depth1"].mean()})
    print(f"{tag} done")

fits = pd.DataFrame(fits)
fits.to_csv("results/contemp_fits.csv", index=False)

print("\nSame-bucket R^2, mean over days (rows: bucket size, cols: spec)")
print(fits.pivot_table(index="dt_ms", columns="spec", values="r2", aggfunc="mean")
          [list(SPECS)].round(3).to_string())

print("\nOFI1 beta (ticks per BTC) and HAC t-stat at 1 s, per day")
print(fits.query("dt_ms == 1000 and spec == 'OFI1'")[["day", "regime", "n", "beta", "t", "r2"]]
          .round(4).to_string(index=False))

dep = pd.DataFrame(depth_rows)
dep.to_csv("results/depth_scaling.csv", index=False)
ok = dep["beta"] > 0
if ok.sum() < 3:
    sys.exit("CKS depth test: fewer than 3 usable pieces, skipped")
slope = np.polyfit(np.log(dep.loc[ok, "depth1"]), np.log(dep.loc[ok, "beta"]), 1)[0]
print(f"\nCKS depth test: {ok.sum()} pieces, log(beta) vs log(depth) slope = {slope:.2f} (CKS: about -1)")
