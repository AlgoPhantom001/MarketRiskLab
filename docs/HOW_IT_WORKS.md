# How MarketRiskLab works

## Follow this one workflow

**Prices → daily returns → risk estimates → backtests and shocks → tables and plots.**

1. Load adjusted prices for three equity ETFs and calculate each day's percentage return.
2. Combine asset returns with weights 40%, 35% and 25%. Multiply by USD 1 million and reverse the sign to obtain losses. A negative loss is a gain.
3. Use the previous 250 returns to estimate the next day's VaR and ES at 95% and 99% confidence.
4. Compare each forecast with the next day's actual hypothetical loss. Record exceptions and test their frequency and clustering.
5. Apply stress assumptions, price example European options, and calculate the simplified FRTB-style measures.
6. Export the calculations, plots, assumptions and results.

## Explain the methods in plain language

| Term | Meaning in this project |
| --- | --- |
| VaR | A loss threshold expected to be exceeded on 5% or 1% of days, depending on confidence |
| Expected Shortfall | The average loss in that worst 5% or 1% tail |
| Historical method | Use the losses actually present in the calibration window |
| Parametric method | Fit a normal distribution using the portfolio mean and covariance |
| Monte Carlo method | Simulate losses from that fitted Gaussian distribution and calculate tail measures |
| Exception | The next day's loss is greater than the predicted VaR |
| Kupiec test | Check whether the total exception frequency matches the expected probability |
| Independence test | Check whether one exception is associated with an increased chance of another |
| Conditional coverage | Combine the frequency and independence checks |

These tests can reject a model. A rejection is a finding to discuss, not something to hide. Failure to reject is not proof that the model is correct.

## Stress scenarios

- **Equity -10%/-20%:** calculate the loss if every equity price falls by that amount.
- **Volatility x1.5:** increase the variability of daily returns and recalculate Gaussian risk.
- **High correlation:** make assets move more closely together while keeping their individual volatilities unchanged.
- **Synchronized sell-off:** apply a downside move of twice each asset's daily standard deviation simultaneously.

Deterministic shock losses and distribution-based VaR estimates are reported separately.

## Options and Greeks

The option example uses S=100, K=100, one year to expiry, 5% interest, 20% annual volatility and no dividend yield. These are assumptions, not live option quotes.

- **Delta:** option price sensitivity to the underlying price.
- **Gamma:** how Delta changes as the underlying moves.
- **Vega:** option price sensitivity to one volatility percentage point.
- **Theta:** option price time decay per calendar day.

For a shock, the engine calculates the option price again. It compares this full repricing with `Delta*dS + 0.5*Gamma*dS² + Vega*(dVol/0.01)`. The difference shows the limits of a local sensitivity approximation. Options do not enter the ETF portfolio's VaR calculation.

## The FRTB-style extension

It uses **97.5% ES** and actual compounded **10-day asset changes**. It calculates the current risk, then finds the 250-observation period with the largest liquidity-adjusted ES.

The demonstration assigns SPY/QQQ to a 10-day horizon and IWM to a 20-day horizon. Longer-horizon factors enter additional nested ES calculations. The implementation supports 10/20/40/60/120-day horizons; the example has exposure in the first two.

The capital proxy equals stressed liquidity-adjusted ES. ETF mappings and aggregation are simplified; the project does not implement every FRTB requirement.

## Where to read the code

Start with `cli.py`, whose `run()` function calls the following modules:

The run function has four steps: load inputs, calculate results, export tables, and write plots/report metadata. Input loading and metadata recording have their own small helper functions.

| Module | Responsibility |
| --- | --- |
| `data.py` | Load/check prices and calculate returns |
| `risk.py` | Historical, parametric and Monte Carlo VaR/ES |
| `backtesting.py` | Prior-window forecasts and exception tests |
| `stress.py` | Price, volatility and correlation shocks |
| `options.py` | European prices, Greeks and option shocks |
| `frtb.py` | Ten-day changes, stressed calibration and horizon aggregation |
| `reporting.py` | Turn calculated outputs into plots and a case study |

`config.json` contains the example settings. Tests check the mathematics and ensure today's or future returns cannot leak into today's forecast.

For the implementation, read `empirical_risk()` and `gaussian_risk()` first, then `rolling_backtest()`. Follow with `black_scholes()` and `liquidity_es()`. The calculations use ordinary functions, named intermediate values and direct formulas. Comments explain conventions such as loss signs, fractional ES tails, Greek units and nested liquidity horizons.

## A short explanation you can give

"I built a Python engine to estimate portfolio tail losses and check those estimates against subsequent observations. I compared historical, Gaussian and Monte Carlo VaR/ES, then used coverage and independence tests to examine exceptions. I added market shocks and European option pricing with Greeks. Finally, I implemented a simplified FRTB-style ES extension with stressed periods and liquidity horizons. The calculations are tested and the results are reproducible; the regulatory extension is educational."
