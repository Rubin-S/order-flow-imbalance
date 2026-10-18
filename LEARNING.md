# LEARNING.md: what this project teaches, in plain words

Read this before an interview. Every section ends with the one sentence to say out loud.

## 1. What OFI is, and why it moves the price (Cont, Kukanov & Stoikov 2014)

At level 1 the book is two queues: the best bid with size q_b, and the best ask with size q_a. Each book update changes them. OFI adds up the pressure from one update to the next:

- e_b = (bid size added at or above the old best bid) − (bid size removed at or below it)
- e_a = the same for the ask, with the sign flipped
- OFI = e_b − e_a, summed over a bucket

In concrete cases: if the bid price rises, e_b = the new bid size. If it falls, e_b = −(the old bid size). If it stays the same, e_b = the change in size. The ask side mirrors this.

**Why ΔP ≈ β·OFI:** in a one-tick book the mid moves only when a queue empties. If the queue at the best price holds about D shares, net pressure of OFI shares moves the price about OFI / D ticks. So β ≈ 1 / depth, and that's the depth-scaling test: log β vs log depth should have slope −1. We got **−0.90**.

> *Say:* "Net order flow at the touch, divided by how deep the touch is, tells you how many ticks the mid moved."

## 2. Explains is not predicts

The same-bucket R² of 0.59 uses flow that's only known **at the end** of the bucket, after the move. That's accounting: you're explaining a price change with the orders that caused it. Forecasting uses flow from **before** the bucket, with a one-bucket embargo and local timestamps, and gets about 1%. The gap between 59% and 1% is the main result.

> *Say:* "Same-window R² is bookkeeping. Out-of-sample R² is the only number that's a forecast."

## 3. Big-tick books, and why depth didn't help

BTCUSDT has a one-tick spread 99.6–100% of the time. The tick (0.1 USDT) is large compared with how much the price moves, so huge queues sit at the touch, and the next move is decided by whether level 1 empties. Levels 5 to 10 are far from the action. They mostly add noise: cancellations that never matter. In small-tick stocks (wide spread in ticks), deeper OFI helps more, which is why the paper's multi-level result doesn't transfer here.

> *Say:* "In a big-tick book, the queue at the touch is the state of the world."

## 4. Out-of-sample R², and why it can be negative

R²_oos = 1 − SSE(model) / SSE(zero forecast). Zero forecast means "the mid won't move", and that's hard to beat at 100 ms. R² below zero means the model did worse than saying nothing.

The decomposition that explained Aug 1: with ρ = corr(forecast, move) and k = sd(forecast) / sd(move),

  R² ≈ 2ρk − k²

- The best k is ρ, giving R² = ρ². With ρ ≈ 0.1, the ceiling is about 1%.
- If k is too big (an overconfident forecast), R² goes negative **even when the direction is right**.
- On Aug 1: ρ = 0.09, k = 0.39, so R² ≈ −7.9%. A quiet day plus forecasts that didn't shrink.

> *Say:* "Negative R² with a positive correlation means a scale problem, not a signal problem."

## 5. Pooled vs mean of days

- **Pooled:** 1 − ΣSSE / Σy² over every bucket on every day. A volatile day weighs more, because it has more Σy².
- **Mean of days:** the average of each day's R². Every day weighs a third.

Both are legitimate. You **must** pick one before seeing the data and report the other too. Ours: pooled was frozen. Aug 1 had 8–9% of the pooled weight, so pooled passed and the mean didn't.

> *Say:* "I pre-registered pooled, and I report the day-mean beside it because they disagree."

## 6. Walk-forward, not shuffled

Shuffled k-fold puts 00:15 in the test set and 00:14 and 00:16 in the training set. Order flow is autocorrelated, so the model has effectively seen the answer. Walk-forward trains only on the past (Apr → May, Apr–May → Jun, …), and adds a one-bucket gap so no target overlaps a feature.

> *Say:* "Time series get split by time, with an embargo, or the R² is a leak."

## 7. The block bootstrap

R² differences need a confidence interval, but buckets aren't independent. A 100 ms bucket looks like its neighbours. Resampling single rows would make the interval far too narrow. Instead we resample 5-minute blocks within each day, which keeps the autocorrelation inside each block. It's also paired: both models are scored on the same resampled buckets, so common noise cancels.

What it **leaves out**: refitting (β, α and PCA uncertainty) and other regimes. Three days are three days.

> *Say:* "Moving blocks keep the autocorrelation. Pairing cancels the noise both models share."

## 8. Pre-registration: why it's worth more than a better R²

Tags: `prereg-v1` (questions and rules), then `freeze-v1` (code), then predictions committed, then `holdout-v1` (one run). The git timestamps prove the order. Without it, any researcher can try 32 configs and report the best. Our economics table had 32 rows, and its maximum (0.49 bps on dev) was 114 trades at a 50% hit rate: a coin flip.

> *Say:* "The holdout was run once, on frozen code, and I reported the day it failed."

## 9. Economics: fees, ticks, and adverse selection

- The edge is **0.085–0.148 bps** per trade (holdout and dev). Fees are 4 bps (maker) or 10 bps (taker) round trip. The edge covers 2–4% of the cheapest fee.
- Hit rate is misleading. 86% "right" with tiny gains still loses to fees.
- **Adverse selection:** a resting maker bid fills exactly when sellers push through it, just before the price drops. So a maker's real edge is worse than the forecast says. Fills you miss are the ones where the price ran your way.

> *Say:* "A 1% R² at 1 s is real information and zero dollars at retail fees."

## 10. Interview Q&A

**Q1. Walk me through the OFI formula on a whiteboard.**
Draw a bid queue and an ask queue. Do the three cases (price up, same, down) for e_b, then mirror them for the ask. Then ΔP ≈ OFI / depth, which is why β ~ 1/depth.

**Q2. Your same-bucket R² is 59%. Why not trade it?**
Because that flow is only known after the move. Out of sample it's about 1%, and that's the honest number.

**Q3. Why did deeper levels hurt?**
Big-tick book: level 1 decides the move, and deeper levels add cancellation noise. On dev, OFI₁ beat OFI₁₀ significantly.

**Q4. Your holdout day-average R² is negative. Does your model fail?**
The frozen test was pooled, and it passed at 100 ms and 1 s. One of three days failed: Aug 1 was 2.5× quieter, the direction was still right (ρ = +0.09), but the forecasts didn't shrink with volatility, and R² ≈ 2ρk − k² punishes that. So it's a calibration failure, with N = 3 days. I pre-registered a vol-scaled v2 on new days instead of patching it on the holdout.

**Q5. Could a maker make money on this?**
No. Even before adverse selection, the edge is 2–4% of the 4 bps maker fee. With adverse selection, fills cluster just before bad moves. You'd need VIP-level rebates and a queue-position model to even ask the question.

**Q6. What would you do with one more week?**
Run v2 (vol scaling) on Nov–Jan, add a fill simulator with queue position, and test a second asset with a smaller tick, where deeper OFI should matter more.

## Further reading
- Cont, Kukanov & Stoikov (2014), *The Price Impact of Order Book Events*, Journal of Financial Econometrics.
- Xu, Gould & Howison (2019), *Multi-level order-flow imbalance in a limit order book*, Market Microstructure and Liquidity.
- Cont, Cucuringu & Zhang (2023), *Cross-impact of order flow imbalance in equity markets*, Quantitative Finance: where integrated (PCA) OFI comes from.
- Kolm, Turiel & Westray (2023), *Deep order flow imbalance*, Mathematical Finance: multi-level OFI fed to deep nets, at many horizons.
- Bouchaud, Bonart, Donier & Gould (2018), *Trades, Quotes and Prices* (Cambridge UP), chapters on queues and impact.
