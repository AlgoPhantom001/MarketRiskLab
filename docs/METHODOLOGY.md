# Methodology and assumptions

## Data and hypothetical P&L

Daily adjusted close snapshots from Yahoo Finance/yfinance are used as total-return-style proxies. `auto_adjust=True`, no silent price repair and no forward filling. Interior missing observations, duplicate dates, nonpositive prices and insufficient history are rejected. Regular exchange holidays are absent from the exchange's own daily series; the loader does not independently certify an exchange calendar or detect every omitted business day.

Simple asset return is `r_i,t = P_i,t / P_i,t-1 - 1`. One-day loss is `L_t = -N * sum(w_i * r_i,t)` with constant N and fixed reference weights reset for each one-day hypothetical observation. This is not a buy-and-hold account or a realized trading strategy. No FX conversion, costs, slippage, borrow or look-through holdings are modeled.

Downloaded prices are a single contemporary snapshot, not point-in-time historical vintages. Adjustments can change following corporate actions or vendor revisions. The manifest records the snapshot checksum, retrieval time, versions and configuration. Synthetic data uses seeded multivariate Student-t innovations (six degrees of freedom) and preset volatility regimes; it never represents an actual historical event.

## VaR and ES

For confidence c, VaR is the c-quantile of the **loss** distribution. Negative risk values are retained and signify gain thresholds.

Historical VaR uses NumPy's `inverted_cdf` quantile. ES is the quantile integral over the upper `(1-c)` probability tail. For n empirical losses sorted from worst to best, sum the largest floor(n*(1-c)) losses and the fractional share of the next loss, then divide by n*(1-c). This handles ties and avoids changing the nominal tail mass. At n=250 and c=99%, the tail mass is 2.5 observations, so the estimate is noisy.

Gaussian portfolio mean return is `mu_p = w' mu`; variance is `sigma_p^2 = w' Sigma w`, using sample covariance with ddof=1. Loss mean is `m=-N*mu_p` and standard deviation `s=N*sigma_p`.

- `VaR_c = m + s*Phi^-1(c)`
- `ES_c = m + s*phi(Phi^-1(c))/(1-c)`

Monte Carlo draws Gaussian portfolio losses with these fitted moments and computes empirical VaR/ES. This projection is mathematically equivalent to drawing correlated Gaussian asset returns then weighting them for this linear portfolio. It is a numerical implementation comparison, not a new heavy-tail model. Latest-window draws=100,000; rolling draws=20,000. Snapshot seed is 42; each forecast uses seed 42 plus its position in the input return series. Inserting earlier data changes simulation streams; changing future returns does not.

## Out-of-sample tests

Forecast at date t uses returns `[t-250:t]` and is evaluated against return t. A breach is **strictly** `realized_loss > VaR`. All estimators share forecast dates and observations. ES forecasts are exported, but no formal ES backtest is claimed.

Kupiec tests the exception probability `p=1-c`. Its statistic is twice the log-likelihood improvement from p to observed breach rate x/T. `xlogy`/`xlog1py` preserve the conventions for zero counts. The asymptotic reference is chi-square(1).

Christoffersen independence counts adjacent indicators n00, n01, n10, n11. It compares a pooled next-state probability against separate probabilities conditional on the previous state. The asymptotic reference is chi-square(1). If a previous-state group has no observations, the test is reported as **not identifiable**, not a passing test.

Conditional coverage is `LR_CC=LR_UC+LR_IND`, compared with chi-square(2), only when independence is identifiable. The full OOS sample and latest 250 forecasts are reported separately. Low breach counts, particularly at 99%, limit asymptotic inference and power. Tests are two-sided adequacy diagnostics; too few exceptions can also reject coverage. A rejection is not an implementation failure, and failure to reject is not model approval. No cause of an exception is asserted solely from these statistics.

## Stress

Uniform equity moves and synchronized per-asset moves create deterministic frozen-exposure losses. Volatility x1.5 and correlation stress generate **new Gaussian VaR/ES**, not deterministic cash losses. These are different quantities and remain in separate table columns.

The correlation matrix is replaced by `0.1*R+0.9*J`, where J is the all-ones correlation matrix. This convex combination is PSD and keeps marginal volatilities unchanged. The synchronized scenario applies `-2*sigma_i` simultaneously to each equity. No scenario is assigned a probability or represented as a forecast.

## Options

Black-Scholes-Merton includes continuously compounded r and dividend yield q. Prices and Greeks are per option unit; illustrative stress P&L uses 100 option units, not an inferred exchange contract multiplier. Inputs: S=K=100, T=1 year, r=5%, sigma=20%, q=0. These are illustrative assumptions, not quoted ETF-option prices or estimated implied volatilities.

Vega is USD per +1 percentage point of annual volatility. Theta is USD per calendar day: the annual time-decay derivative divided by 365. Gamma is delta change per USD of underlying movement. Expiry Greeks are undefined; zero-volatility forward kinks are explicitly undefined. Prices still return intrinsic/deterministic limits.

Full revaluation keeps maturity, strike, rates and dividend yield fixed and bumps spot and/or annual volatility. The approximation is `Delta*dS + 0.5*Gamma*dS^2 + Vega*(dSigma/0.01)`. It omits cross-Greeks and volatility curvature. Approximation errors are reported rather than hidden. Options are not incorporated into ETF VaR or FRTB-style capital.

## FRTB-style extension

The conceptual source is Basel MAR33, particularly 33.3 (97.5% ES), 33.4 (10-day base and liquidity horizons) and stressed calibration. This module does **not** implement the full Basel methodology.

For each asset, actual 10-observation changes are compounded: `prod(1+r_i)-1`. Scenario losses use frozen initial dollar exposures, not daily rebalancing throughout the 10-day horizon. Each 250-return window supplies 241 overlapping scenarios; none crosses outside its window. One-day square-root-of-time scaling is not used for the base ES.

Educational mapping: SPY/QQQ proxy large-cap equity price factors at 10 days; IWM proxies small-cap equity price at 20 days. ETF look-through and individual factor eligibility are not assessed. The aggregator supports 10/20/40/60/120 days, but the demonstration has nonzero exposures only at 10/20.

Let `E_j` be 97.5% ES of 10-day losses from only factors assigned at least LH_j, holding others fixed. For horizons [10,20,40,60,120] and LH_0=0:

`ES_LH = sqrt(sum_j max(E_j,0)^2 * (LH_j-LH_j-1)/10)`.

The first term is all-factor base ES squared. Nonnegative charge floors prevent squaring a negative expected gain into a positive capital charge; raw ES is retained in the bucket table. At positive ES and a single common horizon H, this reduces to base 10-day ES times sqrt(H/10). This identity is a horizon-aggregation check, not permission to scale the one-day base to ten days.

The stressed period maximizes this liquidity-adjusted ES across all available rolling 250-return windows. Exact ties choose the earliest window. The current window is included, so stressed ES cannot be below current ES. Selection uses available history retrospectively and is separate from the no-leakage VaR backtest. It is not the regulatory search since 2005, and it omits approved reduced-factor stress scaling.

The **illustrative capital proxy** equals stressed liquidity-adjusted ES, with no extra multiplier. No claim is made for regulatory capital, IMCC, RWA, economic capital, capital adequacy, standardized sensitivities charges, SIMM, NMRF, DRC, PLA, desk approval or full FRTB compliance. Comparisons always label confidence and horizon: 99% one-day VaR and 97.5% 10-day ES are not like-for-like measures.
