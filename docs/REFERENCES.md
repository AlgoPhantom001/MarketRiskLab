# References

The implementation was written specifically for MarketRiskLab. No third-party repository implementation code or sample data was copied. The following projects informed the organization and scope:

- [himalaybambhaniya/var-engine](https://github.com/himalaybambhaniya/var-engine)
- [AnasOujja/black-scholes-toolkit](https://github.com/AnasOujja/black-scholes-toolkit)
- [tancredeg/var-backtesting](https://github.com/tancredeg/var-backtesting)
- [Aneesh2409/garch-risk-analytics](https://github.com/Aneesh2409/garch-risk-analytics)
- [ShrishDhuria/market-risk-engine](https://github.com/ShrishDhuria/market-risk-engine)

## Mathematical and data sources

- [Basel market-risk framework, including MAR33](https://www.bis.org/committees/bcbs/basel-framework/standard/mar?allChapters=true): ES confidence, ten-day base and liquidity horizons.
- [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html): adjusted-price and end-date conventions.
- Kupiec (1995), *Techniques for Verifying the Accuracy of Risk Measurement Models*: exception coverage.
- Christoffersen (1998), *Evaluating Interval Forecasts*: independence and conditional coverage.
- Black and Scholes (1973), *The Pricing of Options and Corporate Liabilities*, and Merton (1973), *Theory of Rational Option Pricing*: European option pricing.
