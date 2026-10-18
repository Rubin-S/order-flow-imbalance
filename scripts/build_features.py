"""build_features.py - turn each day's mdr_ofi csv into a clean parquet table.

What it does, per day:
  1. reads data/ofi_YYYYMMDD.csv
  2. marks rows to KEEP for research:
       - not a reset row (a book rebuild is not order flow)
       - not in the first BURN_IN_S seconds after 00:00 UTC (funding settlement)
  3. adds a few columns the notebooks will need (time, tick spread)
  4. writes data/feat_YYYYMMDD.parquet (fast, typed, ~5x smaller than csv)
and writes ONE committed audit table, results/day_summary.csv, so anyone can
see each day's size, cadence, spread and how much was masked.

usage: python scripts/build_features.py            (every data/ofi_*.csv)
"""
import glob
import os
import pandas as pd

BURN_IN_S = 60          # decided before looking at results; see README pre-registration
TICK = 0.1              # BTCUSDT perp tick size (USDT)

rows = []
for path in sorted(glob.glob("data/ofi_*.csv")):
    tag = os.path.basename(path)[4:12]                       # ofi_20261001.csv -> 20261001
    df = pd.read_csv(path)

    day_start_us = pd.Timestamp(tag, tz="UTC").value // 1000  # 00:00 UTC in microseconds
    df["t_s"] = (df["ts_exch_us"] - day_start_us) / 1e6       # seconds since 00:00 UTC
    df["spread_ticks"] = (df["spread"] / TICK).round().astype("int32")
    df["keep"] = (df["reset_flag"] == 0) & (df["t_s"] >= BURN_IN_S)

    df.to_parquet(f"data/feat_{tag}.parquet", index=False)

    gap_ms = df["ts_exch_us"].diff().median() / 1000          # typical time between messages
    rows.append({
        "day": tag,
        "rows": len(df),
        "kept": int(df["keep"].sum()),
        "resets": int(df["reset_flag"].sum()),
        "median_msg_gap_ms": round(gap_ms, 1),
        "mid_start": round(df["mid"].iloc[0], 1),
        "mid_end": round(df["mid"].iloc[-1], 1),
        "pct_1tick_spread": round(100 * (df["spread_ticks"] == 1).mean(), 1),
    })
    print(f"{tag}: {len(df):>7} rows, kept {rows[-1]['kept']:>7} -> data/feat_{tag}.parquet")

summary = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
summary.to_csv("results/day_summary.csv", index=False)
print()
print(summary.to_string(index=False))
