"""figures.py - draw the README plots from the committed results/*.csv files.

Reads only results/ (never the raw data), so anyone who clones the repo can
rebuild every figure in a second and check it against the tables.

  fig1_r2_by_horizon.png   same-bucket R^2 (accounting) next to out-of-sample R^2 (forecasting)
  fig2_latency.png         how the forecast's OOS R^2 fades as latency grows
  fig3_breakeven.png       gross edge per trade vs Binance VIP 0 round-trip fees (log scale)
  fig4_depth_scaling.png   CKS check: price impact beta vs level-1 depth (log-log)

usage: python scripts/figures.py          (or: make figures)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # no screen needed
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

INK, BG = "#1d1d1f", "#e8e8ea"
C = ["#b1bcd2", "#607868", "#dbb691", "#a79fb8", "#d9c489", "#7e9c9b", "#c9a3a8"]
ACCENT = "#ff4d00"
H_LABEL = {100: "100 ms", 1000: "1 s", 10000: "10 s", 60000: "60 s"}
MAKER_RT_BPS, TAKER_RT_BPS = 4.0, 10.0

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": (0.11, 0.11, 0.12, 0.5),
    "xtick.color": (0.11, 0.11, 0.12, 0.5), "ytick.color": (0.11, 0.11, 0.12, 0.5),
    "axes.edgecolor": (0.11, 0.11, 0.12, 0.3), "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": (0.11, 0.11, 0.12, 0.07),
    "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 11, "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
})
os.makedirs("figures", exist_ok=True)


def title(ax, main, sub):
    ax.set_title(main, loc="left", fontsize=13, fontweight="bold", pad=22)
    ax.text(0, 1.02, transform=ax.transAxes, s=sub, fontsize=10, alpha=0.6)


def plain_log(axis):
    """Log axis labelled 0.01, 0.1, 1 instead of 10^-2 ..., with 1-2-5 steps."""
    axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}", dpi=200)
    plt.close(fig)
    print(f"figures/{name}")


# ---- 1. accounting vs forecasting, by horizon ----
ct = pd.read_csv("results/contemp_fits.csv")
fo = pd.read_csv("results/forecast_oos.csv")
hz = sorted(ct["dt_ms"].unique())
x = np.arange(len(hz))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for i, spec in enumerate(["OFI1+TI", "OFI1", "TI"]):
    m = ct[ct["spec"] == spec].groupby("dt_ms")["r2"].mean().reindex(hz)
    a1.plot(x, m.values, color=[INK, C[0], C[2]][i], marker="o", ms=4, label=spec)
a1.set_xticks(x, [H_LABEL[h] for h in hz]); a1.set_ylim(0, 1)
a1.set_ylabel("R\u00b2 (same bucket)"); a1.legend(frameon=False, loc="lower right")
title(a1, "Flow explains the move it happens in", "Same-bucket R\u00b2, mean of dev days (accounting)")

f0 = fo[fo["latency_ms"] == 0]
for i, mdl in enumerate(["ridge", "state", "OFI1"]):
    m = 100 * f0[f0["model"] == mdl].groupby("dt_ms")["r2_oos"].mean().reindex(hz)
    a2.plot(x, m.values, color=[INK, C[1], C[0]][i], marker="o", ms=4, label=mdl)
a2.axhline(0, color=INK, alpha=0.14, ls="--", lw=1)
a2.text(len(hz) - 1, 0, " zero forecast", va="bottom", ha="right", fontsize=9, alpha=0.6)
a2.set_xticks(x, [H_LABEL[h] for h in hz]); a2.set_ylabel("Out-of-sample R\u00b2 (%)")
lo = float(np.nanmin(100 * f0[f0["model"].isin(["ridge", "state", "OFI1"])]["r2_oos"]))
a2.set_ylim(max(lo * 1.1, -25), None)            # clip a huge negative tail so the edge stays visible
a2.legend(frameon=False, loc="lower left")
title(a2, "...but barely predicts the next one", "Walk-forward OOS R\u00b2 vs zero forecast, 1-bucket embargo")
save(fig, "fig1_r2_by_horizon.png")

# ---- 2. latency decay ----
fl = fo[(fo["model"] == "state+flow") & (fo["dt_ms"].isin([100, 1000]))]
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, dt in enumerate([100, 1000]):
    m = 100 * fl[fl["dt_ms"] == dt].groupby("latency_ms")["r2_oos"].mean()
    ax.plot(m.index, m.values, color=[INK, C[0]][i], marker="o", ms=4)
    ax.text(m.index[-1] + 2, m.values[-1], H_LABEL[dt], va="center", color=INK, fontsize=10)
ax.set_xlabel("Extra latency after the embargo (ms)"); ax.set_ylabel("Out-of-sample R\u00b2 (%)")
ax.set_ylim(0, None); ax.set_xlim(None, fl["latency_ms"].max() + 14)
title(ax, "The 100 ms edge fades with latency", "state+flow OLS, mean over walk-forward test days")
save(fig, "fig2_latency.png")

# ---- 3. break-even vs fees ----
# only the horizons with a positive OOS edge (100 ms, 1 s); 10 s / 60 s lose before costs
# and are in results/economics.csv
ec = pd.read_csv("results/economics.csv")
ec = ec[(ec["latency_ms"] == 0) & (ec["dt_ms"].isin([100, 1000]))].reset_index(drop=True)
ec["label"] = ec["dt_ms"].map(H_LABEL) + " " + ec["threshold"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(ec.index, ec["gross_bps"].clip(lower=1e-3), color=[C[0] if d == 100 else C[1] for d in ec["dt_ms"]],
       width=0.65)
for i, r in ec.iterrows():
    ax.text(i, max(r["gross_bps"], 1e-3) * 1.15, f"{r['gross_bps']:.3f}", ha="center", fontsize=8)
ax.set_yscale("log"); ax.set_ylim(1e-3, 30); plain_log(ax.yaxis)
ax.set_xticks(ec.index, ec["label"], fontsize=9)
for y, name in [(MAKER_RT_BPS, "maker fees 4 bps"), (TAKER_RT_BPS, "taker fees 10 bps")]:
    ax.axhline(y, color=ACCENT if y == MAKER_RT_BPS else INK, alpha=1 if y == MAKER_RT_BPS else 0.3,
               ls="--", lw=1)
    ax.text(len(ec) - 0.6, y * 1.15, name, ha="right", fontsize=9)
ax.set_ylabel("Gross edge per round trip (bps, log)")
best = ec["gross_bps"].max()
title(ax, f"Best gross edge is {best:.2f} bps: {100 * best / MAKER_RT_BPS:.1f}% of the maker fee",
      "Ridge forecasts on test days, latency 0; 10 s and 60 s lose before costs (see economics.csv)")
save(fig, "fig3_breakeven.png")

# ---- 4. CKS depth scaling ----
dp = pd.read_csv("results/depth_scaling.csv")
dp = dp[dp["beta"] > 0]
lx, ly = np.log(dp["depth1"]), np.log(dp["beta"])
k, c0 = np.polyfit(lx, ly, 1)
fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.scatter(dp["depth1"], dp["beta"], color=C[0], edgecolor=INK, lw=0.5, s=40, alpha=0.9)
xs = np.linspace(lx.min(), lx.max(), 50)
ax.plot(np.exp(xs), np.exp(c0 + k * xs), color=INK)
ax.plot(np.exp(xs), np.exp(ly.mean() - (xs - lx.mean())), color=INK,   # slope -1 through the centre
        alpha=0.3, ls="--", lw=1)
ax.text(np.exp(xs[-1]), np.exp(c0 + k * xs[-1]), f"  fit {k:.2f}", va="center", fontsize=10)
ax.set_xscale("log"); ax.set_yscale("log"); plain_log(ax.xaxis); plain_log(ax.yaxis)
ax.set_xlabel("Mean level-1 depth (BTC, log)"); ax.set_ylabel("OFI1 beta (ticks per BTC, log)")
ax.grid(True, axis="both")
title(ax, f"Impact falls with depth: slope {k:.2f}", "30-min pieces of dev days at 1 s; dashed = CKS slope \u22121")
save(fig, "fig4_depth_scaling.png")
