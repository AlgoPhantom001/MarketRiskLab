"""Small, explicit adjusted-price ingestion and labeled offline fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = np.asarray(config["weights"], dtype=float)
    tickers = config["tickers"]
    horizons = config["liquidity_horizons"]
    if len(tickers) != len(weights) or len(set(tickers)) != len(tickers):
        raise ValueError("Unique tickers and one weight per ticker are required")
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Long-only finite weights must sum to one")
    if len(horizons) != len(tickers) or any(
        h not in (10, 20, 40, 60, 120) for h in horizons
    ):
        raise ValueError(
            "One supported educational liquidity horizon per ticker is required"
        )
    if config["window"] != 250:
        raise ValueError("This demonstration uses a 250-observation estimation window")
    if not np.isfinite(config["notional"]) or config["notional"] <= 0:
        raise ValueError("Reference exposure must be positive")
    if config["confidence_levels"] != [0.95, 0.99]:
        raise ValueError("This demonstration requires confidence levels [0.95, 0.99]")
    if config["currency"] != "USD":
        raise ValueError(
            "This demonstration reports USD exposures; currency conversion is not implemented"
        )
    if any(config[k] < 1000 for k in ("snapshot_draws", "rolling_draws")):
        raise ValueError("Simulation requires at least 1,000 draws")
    return config


def validate_prices(
    prices: pd.DataFrame, tickers: list[str], minimum: int = 501
) -> pd.DataFrame:
    """Reject incomplete observations rather than bridge gaps or fill returns."""
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Prices require valid dates")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    if list(prices.columns) != tickers:
        raise ValueError("Price columns must exactly match configured ticker order")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "All adjusted prices must be finite and positive; no filling is allowed"
        )
    if len(prices) < minimum:
        raise ValueError(f"Need at least {minimum} common price observations")
    return prices.astype(float)


def read_prices(path: str | Path, tickers: list[str]) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return validate_prices(prices, tickers)


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Invalid returns")
    return returns


def synthetic_prices(config: dict, observations: int = 1751) -> pd.DataFrame:
    """Generic equity proxies with seeded volatility regimes and Student-t shocks.

    Columns follow requested ticker identifiers for interface compatibility only.
    These are NOT real SPY/QQQ/IWM observations or a historical crisis replay.
    """
    random_generator = np.random.default_rng(config["seed"])
    asset_count = len(config["tickers"])
    correlation = np.full((asset_count, asset_count), 0.7)
    np.fill_diagonal(correlation, 1.0)
    correlated_normals = (
        random_generator.standard_normal((observations - 1, asset_count))
        @ np.linalg.cholesky(correlation).T
    )
    # A common chi-square divisor preserves multivariate Student-t dependence.
    standardized_shocks = (
        correlated_normals
        / np.sqrt(random_generator.chisquare(6, size=(observations - 1, 1)) / 6)
        * np.sqrt(4 / 6)
    )
    daily_volatility = np.linspace(0.010, 0.015, asset_count)
    volatility_regime = np.ones(observations - 1)
    volatility_regime[500:650] = 2.4
    volatility_regime[1100:1200] = 1.8
    returns = (
        0.00025 + standardized_shocks * daily_volatility * volatility_regime[:, None]
    )
    if (returns <= -1).any():
        raise ValueError("Fixture produced an invalid simple return")
    prices = np.vstack(
        [np.full(asset_count, 100.0), 100.0 * np.cumprod(1 + returns, axis=0)]
    )
    dates = pd.bdate_range(config["start"], periods=observations)
    price_frame = pd.DataFrame(prices, index=dates, columns=config["tickers"])
    price_frame.index.name = "date"
    return validate_prices(price_frame, config["tickers"])


def download_prices(config: dict, cache_dir: Path) -> pd.DataFrame:
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    downloaded_data = yf.download(
        config["tickers"],
        start=config["start"],
        end=config["end_exclusive"],
        interval="1d",
        auto_adjust=True,
        repair=False,
        keepna=True,
        progress=False,
        threads=False,
        timeout=20,
    )
    if downloaded_data is None or downloaded_data.empty:
        raise ValueError("Provider returned no prices")
    prices = downloaded_data["Close"].reindex(columns=config["tickers"])
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "date"
    # Reject ticker gaps: dropping an interior row can create a multi-day return.
    return validate_prices(prices, config["tickers"])
