"""
POST-HOC DIAGNOSTIC, written AFTER the holdout run (tag holdout-v1). Exploratory only.
It changes no model, feature, threshold or headline. It only explains the frozen numbers:

  Why is the pooled bootstrap edge positive while the mean-over-days R^2 is negative?
  Why did 2026-08-01 fail?  Wrong direction (low corr), or right direction but wrong size/bias?

usage: python scripts/diag_holdout.py   ->  results/holdout_diag.csv
"""
import numpy as np
import pandas as pd

p = pd.read_parquet("data/preds_holdout.parquet").query("latency_ms == 0 and dt_ms in (100, 1000)")
models = [c for c in p.columns if c.startswith("p_")]

rows = []
for (dt, test), d in p.groupby(["dt_ms", "test"]):
    y = d["y"].to_numpy()
    sst = (y ** 2).sum()                                   # SSE of the zero forecast
    r = {"dt_ms": dt, "test": test, "n": len(d), "sst": sst,
         "mean_y": y.mean(), "sd_y": y.std(),
         "mean_p_ridge": d["p_ridge"].mean(), "sd_p_ridge": d["p_ridge"].std(),
         "corr_ridge": np.corrcoef(y, d["p_ridge"])[0, 1]}
    for m in models:                                       # per-day R^2 (%) of every model
        r[m[2:]] = 100 * (1 - ((y - d[m]) ** 2).sum() / sst)
    rows.append(r)

res = pd.DataFrame(rows)
res["sst_share"] = res["sst"] / res.groupby("dt_ms")["sst"].transform("sum")   # pooled weight of the day

for dt, d in p.groupby("dt_ms"):                           # pooled row = what bootstrap.py scores
    y = d["y"].to_numpy(); sst = (y ** 2).sum()
    r = {"dt_ms": dt, "test": "POOLED", "n": len(d), "sst": sst, "sst_share": 1.0}
    for m in models:
        r[m[2:]] = 100 * (1 - ((y - d[m]) ** 2).sum() / sst)
    rows.append(r)
res = pd.concat([res, pd.DataFrame(rows[len(res):])], ignore_index=True)

res.to_csv("results/holdout_diag.csv", index=False)
show = ["dt_ms", "test", "sst_share", "corr_ridge", "mean_y", "mean_p_ridge", "sd_y", "sd_p_ridge"] \
       + [m[2:] for m in models]
pd.set_option("display.width", 200)
print(res[show].round(3).to_string(index=False))
