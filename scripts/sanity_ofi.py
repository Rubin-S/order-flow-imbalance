"""sanity_ofi.py - first sanity check of mdr_ofi output.

Does buying pressure (OFI > 0) go together with the mid price going up?
If not, there's a sign bug, and every later result would be backwards.

usage: python scripts/sanity_ofi.py data/ofi_20261001.csv
"""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
df["dmid"] = df["mid"].diff()              # mid change caused by this message
df = df[df["reset_flag"] == 0].copy()      # drop rows across a book rebuild

print("rows used:", len(df))
print("event-level corr(e1, dmid):", round(df["e1"].corr(df["dmid"]), 3))

# Add up every message inside each 1-second bucket (exchange clock)
df["sec"] = df["ts_exch_us"] // 1_000_000
cols = ["dmid"] + [f"e{m}" for m in range(1, 11)]
g = df.groupby("sec")[cols].sum()
g["L1"] = g["e1"]
g["L5"] = g[[f"e{m}" for m in range(1, 6)]].sum(axis=1)
g["L10"] = g[[f"e{m}" for m in range(1, 11)]].sum(axis=1)

print("1-second buckets:", len(g))
for c in ["L1", "L5", "L10"]:
    r2 = g[c].corr(g["dmid"]) ** 2
    print(f"same-second R2  {c:>3}: {r2:.3f}")
