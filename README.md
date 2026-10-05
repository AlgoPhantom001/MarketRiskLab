# MarketRiskLab | Market Risk & Derivatives Analytics

Python · pandas · NumPy · SciPy · Matplotlib · yfinance

MarketRiskLab answers three questions: **How much could a portfolio lose? Do the risk forecasts match observed losses? How do market shocks affect risk and option prices?**

The example portfolio uses SPY 40%, QQQ 35% and IWM 25%, with USD 1 million reference exposure. It has no web application, database or trading system.

## What it does

| Part | Purpose |
| --- | --- |
| VaR and Expected Shortfall | Estimate a loss threshold and the average loss in the worst tail using historical, Gaussian and Monte Carlo methods |
| Backtesting | Compare forecasts with the following day's loss, track exceptions and test their frequency and clustering |
| Stress testing | Apply equity falls, higher volatility and higher correlation |
| Options | Price European calls/puts, calculate four Greeks and compare full repricing with a Greek approximation |
| FRTB-style ES | Demonstrate 97.5% ES, stressed calibration and longer liquidity horizons |

**Start with [How it works](docs/HOW_IT_WORKS.md)** for a short explanation of the calculations and code.

## Run it

Use Python 3.12. Create and activate an environment:

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`

macOS/Linux: `source .venv/bin/activate`

Then install, test and run:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pytest -q
python -m marketrisklab.cli run --data synthetic
```

The offline demo produces ten CSV tables, six plots, a manifest and `reports/synthetic/case_study.md`. Its data is **labeled synthetic equity proxies, not real ETF observations**.

For fresh market data:

```bash
python -m marketrisklab.cli fetch
python -m marketrisklab.cli run --data market
```

The raw Yahoo price cache is excluded from the repository. Fetching needs network access; revised provider data can change results. `config.json` holds the portfolio, dates and simulation settings.

## Results and validation

The included [real-data case study](reports/market/case_study.md) covers January 2018 through September 2026, with **1,947 out-of-sample forecasts per model/confidence**. Historical 95% VaR recorded 95 exceptions (4.88%): its coverage test did not reject, but independence rejected, indicating clustering. Historical and Gaussian 99% coverage both rejected.

**71 tests pass**, covering calculation benchmarks, look-ahead leakage, statistical tests, pricing/Greeks and liquidity aggregation. See the [validation record](docs/VALIDATION.md). GitHub Actions runs the offline suite on pushes and pull requests.

![Rolling VaR and exceptions](reports/market/figures/rolling_var_exceptions.png)

## Assumptions to understand

- A forecast uses only the preceding 250 returns.
- Monte Carlo and parametric models share the Gaussian assumption.
- Options use illustrative inputs and are separate from portfolio VaR.
- FRTB-style calculations are an **educational extension, not full regulatory compliance**. The capital proxy is stressed liquidity-adjusted ES only.

The [methodology](docs/METHODOLOGY.md) contains the formulas and limitations; [references](docs/REFERENCES.md) identify the conceptual sources.

## Files

```text
src/marketrisklab/      calculations and the run command
tests/                 validation checks
data/                  synthetic sample
reports/market/        saved results and plots
docs/                  explanations and validation evidence
.github/workflows/     automated tests
config.json            example portfolio settings
requirements-lock.txt  recorded dependency versions
```
