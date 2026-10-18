# order-flow-imbalance

Event-level multi-level order flow imbalance on Binance BTCUSDT perps, built on [l2-orderbook-replay](https://github.com/Rubin-S/l2-orderbook-replay).

> **Status:** work in progress. Engine, tests and data pipeline are done. The research questions, data split and pass rules below were committed **before** any forecasting model was fitted or any holdout day was analysed.

## Reproduce

```bash
git clone --recurse-submodules https://github.com/Rubin-S/order-flow-imbalance.git
cd order-flow-imbalance
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make data features      # builds the pinned engine, streams 7 days, writes data/feat_*.parquet
```

`results/data_checksums.txt` lists the sha256 of every `.evt` file. `results/day_summary.csv` is the per-day data audit.

## Pre-registration

Fixed on 2026-10-18, before fitting any forecasting model. The git timestamp of this commit (tag `prereg-v1`) is the proof.

### Research questions

- **Q1. Contemporaneous impact.** Does ΔP ≈ β·OFI hold on BTC perps (Cont, Kukanov & Stoikov 2014), and does β fall as average depth rises?
- **Q2. Depth.** Do deeper levels (L1 → L5 → L10 → PCA-integrated OFI) raise contemporaneous R², and do they raise out-of-sample forecast R²?
- **Q3. Flow vs state.** Does lagged OFI add forecasting power on top of queue imbalance and microprice?
- **Q4. Decay.** How fast does forecasting power fade with horizon (100 ms → 60 s) and with latency (0 → 100 ms)?
- **Q5. Economics.** What cost per round trip would the best forecast break even at, and is that above or below Binance fees for a taker and for a maker?

### My priors (written down so I can be wrong in public)

1. β > 0 on every day.
2. Contemporaneous R² rises with depth. Out-of-sample gains from deeper levels are small, because BTCUSDT is a big-tick book (the spread is one tick 99.6 to 100% of the time), so the next move is decided at level 1.
3. Lagged OFI adds little over queue imbalance at ≤ 1 s.
4. Forecast R² is close to zero by 60 s.
5. A taker cannot beat fees. Any edge belongs to a maker.

### Data

- **Source:** Tardis free first-of-month files, `binance-futures`, `incremental_book_L2` + `trades`, BTCUSDT.
- **Days:** the 1st of each month, Apr to Oct 2026, window **00:00 to 02:00 UTC** every day.
- **Development days:** Apr, May, Jun, Jul. **Holdout days:** Aug, Sep, Oct. Every holdout number gets reported, whatever it says, from a single run of frozen code.
- **Masks:** rows flagged `reset_flag` (book rebuilt after a reconnect) and the first **60 s** after 00:00 UTC (funding settlement). Flow is never computed across a reset.
- **Feed regime:** Binance pushed book updates about every **52 ms** on Apr and May and every **26 ms** from June on (`results/day_summary.csv`). All horizons are in clock time, never "next N messages", and results are reported per regime as well as pooled.

### Method

- **Buckets:** Δt ∈ {100 ms, 1 s, 10 s, 60 s}. OFI is summed over each bucket; mid is the last value at or before the bucket end; ΔP is measured in **ticks** (0.1 USDT).
- **Clocks:** the exchange timestamp for contemporaneous accounting, and the local (capture) timestamp for anything predictive: a feature may only use rows with `ts_local` ≤ the decision time.
- **Depth normalisation:** OFI at depth m is divided by the average depth of levels 1..m over the **previous** bucket, so no same-bucket information leaks in.
- **Integrated OFI:** PCA on normalised L1..L10 OFI, fit on **training days only**; weights normalised to sum to 1.
- **Contemporaneous specs:** OLS of ΔP on OFI₁, trade imbalance, OFI₁ + trade imbalance, OFI₅, OFI₁₀ and integrated OFI, with Newey–West (HAC) errors.
- **Forecast target:** ΔP over the next bucket, skipping one bucket as an embargo.
- **Forecast features:** lagged OFI (L1, L5, L10, integrated), level-1 queue imbalance, `imbalance5`, microprice − mid, recent ΔP.
- **Baselines (reported first):** a zero forecast, and recent ΔP only.
- **Models:** OLS and ridge (α chosen on the last training day only).
- **Out-of-sample R²:** 1 − SSE(model) / SSE(zero forecast). Zero means no better than predicting "no move".
- **Walk-forward on dev:** train Apr → test May; Apr–May → Jun; Apr–Jun → Jul.
- **Holdout:** train on all 4 dev days, test Aug, Sep, Oct, once.

### Pass rule

Model A beats model B only if the 95% confidence interval of **R²_oos(A) − R²_oos(B)** excludes zero. The interval comes from a moving-block bootstrap on the holdout days, with 5-minute blocks resampled within each day and 2,000 resamples. Blocks keep the autocorrelation that a plain row bootstrap would ignore. Three holdout days are too few blocks on their own.

### Costs

Binance USDⓈ-M VIP 0 fees: **0.02% maker, 0.05% taker** per side (checked 2026-10-18), i.e. 4 bps maker and 10 bps taker per round trip, before spread. Economics are reported gross, after the spread, and after fees, together with the break-even cost at each horizon.

### Disclosures

- The Oct 1, 00:00–00:39 UTC sample was used to build and test the engine before this split was fixed. It produced only contemporaneous sanity numbers (same-second R² 0.48–0.54), not forecasts.
- The holdout files were downloaded and audited for data quality only (row counts, resets, message cadence, spread). No OFI-to-return statistics have been computed on Aug, Sep or Oct.

### Addendum: frozen before the holdout (tag `freeze-v1`)

Added after the development analysis and before any model touched Aug, Sep or Oct.
It fixes choices the original pre-registration left open. Nothing below changes after the holdout run.

- **Frozen code:** `forecast.py`, `economics.py`, `bootstrap.py` and `figures.py` at this tag. The holdout is one run of `forecast.py --holdout`, `economics.py --holdout` and `bootstrap.py --holdout`.
- **Comparisons tested** (pass rule above, 95% CI of the R² difference excludes 0):
  1. Edge: ridge vs zero forecast at 100 ms and 1 s.
  2. Q3: state+flow vs state (OLS) at 100 ms and 1 s.
  3. Q2 out-of-sample: OFI10 vs OFI1, and integrated OFI vs OFI1, at 1 s.
- **Q5 headline:** ridge, 1 s, latency 0, trading only when |forecast| beats the training q90. It was chosen on dev because it is the best row with over 1,000 trades. It "pays" only if its bootstrap CI lower bound beats the 4 bps maker round trip. Every other economics row is exploratory.
- **Multiple comparisons:** six R² tests plus one economics test, reported without correction; read a single marginal pass with that in mind.
- **Disclosure:** for the first dev fold (Apr → May) there is no separate validation day, so ridge α is chosen on the chronologically last 25% of Apr 1. Every other fold, and the holdout, uses the last training day.
- **Figure 3** shows only horizons with positive OOS R² (100 ms, 1 s); every row is in `results/economics.csv`.
