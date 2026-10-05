# Validation record

Validated locally on 5 October 2026 using Python 3.12.14 on Windows. All numerical claims here refer to observed runs, not planned results.

## Automated checks

**71 tests passed**, with no failures or skipped tests, both in the build environment and a separate fresh virtual environment installed from `requirements-lock.txt`.

After simplifying implementation names, formatting and the run workflow, all 71 tests passed again. Exact before/after comparisons matched all ten result tables on both synthetic and market data; the market tables also matched the saved CSV results. Pytest reported one local cache-write warning, which did not affect the tests or outputs.

| Area | Checks | Main evidence |
| --- | ---: | --- |
| Data/configuration | 10 | Seeded fixture, return compounding, CSV round trip, invalid prices/dates/weights rejected |
| VaR/ES | 14 | Gaussian closed forms, fractional empirical tails, ties, signed loss thresholds, exposure scaling, seeded MC convergence |
| Backtesting | 10 | Hand-calculated likelihoods and transitions, boundary counts, clustering, conditional-coverage identity, current/future-data leakage test |
| Stress | 7 | Zero/uniform loss reconciliation, PSD covariance stress, unchanged marginal variances, zero-volatility asset, distinct risk/scenario outputs |
| Options | 14 | Textbook prices, put-call parity, bounds, finite-difference Greeks, expiry/zero-volatility limits, stress direction and invalid inputs |
| FRTB-style | 13 | Actual 10-day compounding, all five supported horizons, nested subsets, independent stress-window enumeration, nonnegative charge floor |
| Integrated workflow | 3 | Repeated seeded run gives identical CSV checksums; outputs reconcile; labels/plots exist; missing/wrong-checksum cache rejected |

The complete workflow ran on both the real ETF snapshot and the default 1,751-price offline fixture. Each produced ten CSV tables, six plots, a source/versions manifest, a generated case study. Reproducibility is numerical: timestamps differ between runs by design.

Six real-data plots were visually inspected for legible labels, units and appropriate source/horizon labeling. The options plot explicitly states that it uses assumed parameters and no market option quotes.

## Real-data execution

- Source: Yahoo Finance adjusted daily prices via yfinance; checksum in `data/source.json` and the market-run manifest.
- 2,198 common price observations: 2 January 2018 through 30 September 2026.
- 2,197 simple returns; 1,947 out-of-sample forecasts per method/confidence.
- Historical 95%: 95 exceptions, 4.8793%; coverage does not reject, independence rejects at 5% significance.
- Historical 99%: 32 exceptions, 1.64355%; coverage rejects at 5% significance.
- Parametric 99%: 48 exceptions, 2.46533%; coverage rejects at 5% significance.
- Stressed ES calibration selected 31 May 2019 through 27 May 2020 from 1,948 candidate windows.
- Stressed liquidity-adjusted 97.5% ES / illustrative capital proxy: USD 225,811.258293, for the stated USD 1 million reference exposure.

These are findings for one fixed portfolio and data snapshot, not broad claims about model superiority. Exact values and p-values are in the exported tables.

## Limits of validation

This validates the prototype's documented mathematical and workflow behavior. It does not establish market-data licensing, vendor data accuracy, regulatory compliance, model approval or production suitability. Remote GitHub Actions has not run because no repository was published. Optional GARCH, option-inclusive VaR, notebook and separate historical-scenario features were not implemented or claimed.
