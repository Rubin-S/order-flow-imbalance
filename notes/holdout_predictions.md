# Holdout Predictions (Pre-Holdout Preregistration)

Written and committed before executing `scripts/forecast.py --holdout`, `scripts/economics.py --holdout`, and `scripts/bootstrap.py --holdout` on August 1, September 1, and October 1, 2026.

## Pre-Holdout Predictions for the 7 Bootstrap Tests

| # | Test | dt | Metric | Prediction | Rationale |
|---|---|---|---|---|---|
| 1 | `edge: ridge vs zero` | 100 ms | R² diff (pp) | **Pass** | The 100 ms edge was positive on dev (1.28 pp, CI [0.21, 2.11]); holdout days are in the fast 26 ms feed regime where queue imbalance and microprice maintain predictive signal over 100 ms. |
| 2 | `edge: ridge vs zero` | 1000 ms | R² diff (pp) | **Pass** | The 1 s predictive edge was strong and consistent across dev days (1.16 pp, CI [0.78, 1.61]) with 0% of resamples ≤ 0; training on all 4 dev days gives a well-regularized model. |
| 3 | `Q3: state+flow vs state` | 100 ms | R² diff (pp) | **Fail** | On dev, state alone accounted for most of the predictive power (0.72% vs 0.84% for state+flow), and the dev CI straddled zero ([-0.16, 0.30]); flow is largely already reflected in queue state. |
| 4 | `Q3: state+flow vs state` | 1000 ms | R² diff (pp) | **Fail** | At 1 s on dev, state alone outperformed state+flow OLS (1.82% vs 1.42%), and the dev CI was mostly negative ([-0.26, 0.04]); adding flow regressors increases estimation variance without incremental signal. |
| 5 | `Q2: OFI10 vs OFI1` | 1000 ms | R² diff (pp) | **Fail** | Pass requires CI lower bound > 0 (OFI10 beating OFI1); on dev, deeper flow added cancel-and-replace noise so OFI10 significantly underperformed OFI1 (CI [-0.19, -0.04] < 0). |
| 6 | `Q2: I vs OFI1` | 1000 ms | R² diff (pp) | **Fail** | Integrated OFI underperformed OFI1 on dev (CI [-0.35, -0.05] < 0) because PCA weights chase high-variance deep-book churn rather than price informativeness. |
| 7 | `Q5: gross bps, q90` | 1000 ms | bps per trade | **Fail** | Pass requires the 95% CI lower bound to exceed the 4.0 bps maker fee; the dev point estimate was 0.148 bps with CI [0.117, 0.182], covering under 5% of the fee hurdle. |
