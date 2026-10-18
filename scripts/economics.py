"""economics.py - Q5: would the best forecast make money after costs?

Takes the ridge (state+flow) forecasts that forecast.py saved for each TEST day
and turns them into trades:
  - trade a bucket only when |forecast| beats a threshold. The thresholds are the
    50th / 90th / 99th percentile of |forecast| on the TRAINING days (plus "all")
  - go long if the forecast is up, short if down; hold for the target bucket
  - gross P&L = sign(forecast) * actual mid move, turned into basis points of price

Costs per ROUND TRIP (in and out), Binance USD-M VIP 0, as pre-registered:
  taker: 2 x 0.05% = 10 bps fees, and you cross the spread (1 tick total)
  maker: 2 x 0.02% =  4 bps fees, and you EARN the spread, but only if both orders
         fill. That's optimistic: no missed fills and no adverse selection modelled.

The break-even cost is just the gross edge per trade: any round-trip cost above it
loses money. Compare it with 10 bps (taker) and 4 bps (maker).

usage: python scripts/economics.py             (development forecasts, data/preds.parquet)
       python scripts/economics.py --holdout   (data/preds_holdout.parquet, ONLY for the final run)
writes results/economics.csv
"""
import sys
import numpy as np
import pandas as pd

TICK = 0.1
TAKER_RT_BPS = 2 * 5.0          # 0.05% per side
MAKER_RT_BPS = 2 * 2.0          # 0.02% per side
MIN_TRADES = 100                # a "best" row needs at least this many trades

holdout = "--holdout" in sys.argv
p = pd.read_parquet("data/preds_holdout.parquet" if holdout else "data/preds.parquet")
p["tick_bps"] = TICK / p["mid"] * 1e4                       # one tick in bps of price
p["gross_bps"] = np.sign(p["pred"]) * p["y"] * p["tick_bps"]

rows = []
for (dt, L), g in p.groupby(["dt_ms", "latency_ms"]):
    for name in ["all", "q50", "q90", "q99"]:
        t = g if name == "all" else g[g["pred"].abs() > g[name]]
        t = t[t["pred"] != 0]
        if len(t) == 0:
            continue
        gross = t["gross_bps"].mean()
        spread = t["tick_bps"].mean()                       # one full spread = 1 tick here
        rows.append({
            "dt_ms": dt, "latency_ms": L, "threshold": name, "trades": len(t),
            "flat_share": (t["y"] == 0).mean(),            # mid didn't move at all
            "hit_rate": (t["gross_bps"] > 0).sum() / max((t["gross_bps"] != 0).sum(), 1),
            "gross_bps": gross,                             # = break-even round-trip cost
            "taker_net_bps": gross - spread - TAKER_RT_BPS,
            "maker_net_bps": gross + spread - MAKER_RT_BPS,  # optimistic: both legs fill
            "tick_bps": spread,
        })

res = pd.DataFrame(rows)
res.to_csv("results/economics.csv", index=False)
pd.set_option("display.width", 140)
print("Per-trade economics, ridge state+flow forecasts on test days (bps of price)")
print(res.round({"flat_share": 3, "hit_rate": 3, "gross_bps": 4, "taker_net_bps": 3, "maker_net_bps": 3,
                 "tick_bps": 4}).to_string(index=False))
ok = res[res["trades"] >= MIN_TRADES]
best = ok.loc[ok["gross_bps"].idxmax()]
print("\n(hit_rate = share of winning trades among those where the mid moved)")
print(f"\nBest break-even (rows with >= {MIN_TRADES} trades) round-trip cost: {best['gross_bps']:.4f} bps "
      f"(dt={best['dt_ms']} ms, latency={best['latency_ms']} ms, {best['threshold']}, "
      f"{best['trades']} trades)")
print(f"Fees to beat: taker {TAKER_RT_BPS:.0f} bps, maker {MAKER_RT_BPS:.0f} bps per round trip "
      f"= {MAKER_RT_BPS / best['tick_bps']:.0f} / {TAKER_RT_BPS / best['tick_bps']:.0f} ticks"
      f"\n-> best edge covers {100 * best['gross_bps'] / MAKER_RT_BPS:.1f}% of even the maker fee")
