"""Strictly prior-window forecasts and stable exception likelihood tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy, xlog1py
from scipy.stats import chi2

from .risk import (
    check_confidence,
    empirical_risk,
    gaussian_risk,
    moments,
    monte_carlo_losses,
)


def _indicators(exceptions: np.ndarray) -> np.ndarray:
    exception_flags = np.asarray(exceptions)
    if (
        exception_flags.ndim != 1
        or len(exception_flags) == 0
        or not np.isin(exception_flags, [0, 1]).all()
    ):
        raise ValueError(
            "A nonempty one-dimensional binary exception sequence is required"
        )
    return exception_flags.astype(int)


def _bernoulli_log_likelihood(zeros: int, ones: int, probability: float) -> float:
    return float(xlog1py(zeros, -probability) + xlogy(ones, probability))


def kupiec(exceptions: np.ndarray, confidence: float) -> dict:
    """Test whether the exception rate matches 1 - confidence."""
    check_confidence(confidence)
    exception_flags = _indicators(exceptions)
    exception_count = int(exception_flags.sum())
    forecast_count = len(exception_flags)
    null_log_likelihood = _bernoulli_log_likelihood(
        forecast_count - exception_count, exception_count, 1 - confidence
    )
    fitted_log_likelihood = _bernoulli_log_likelihood(
        forecast_count - exception_count,
        exception_count,
        exception_count / forecast_count,
    )
    likelihood_ratio = max(0.0, -2 * (null_log_likelihood - fitted_log_likelihood))
    return {
        "lr": likelihood_ratio,
        "p_value": float(chi2.sf(likelihood_ratio, 1)),
        "status": "available",
    }


def christoffersen_independence(exceptions: np.ndarray) -> dict:
    """Test whether a breach depends on the previous day's breach status."""
    exception_flags = _indicators(exceptions)
    unavailable = {"lr": None, "p_value": None, "status": "not_identifiable"}
    if len(exception_flags) < 2:
        return unavailable
    previous, current = exception_flags[:-1], exception_flags[1:]
    # n01 counts a non-breach followed by a breach; likewise for 00, 10, 11.
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    transitions = dict(n00=n00, n01=n01, n10=n10, n11=n11)
    if n00 + n01 == 0 or n10 + n11 == 0:
        return {**unavailable, **transitions}
    pooled_probability = (n01 + n11) / (len(exception_flags) - 1)
    probability_after_no_exception = n01 / (n00 + n01)
    probability_after_exception = n11 / (n10 + n11)
    null_log_likelihood = _bernoulli_log_likelihood(
        n00 + n10, n01 + n11, pooled_probability
    )
    fitted_log_likelihood = _bernoulli_log_likelihood(
        n00, n01, probability_after_no_exception
    ) + _bernoulli_log_likelihood(n10, n11, probability_after_exception)
    likelihood_ratio = max(0.0, -2 * (null_log_likelihood - fitted_log_likelihood))
    return {
        "lr": likelihood_ratio,
        "p_value": float(chi2.sf(likelihood_ratio, 1)),
        "status": "available",
        **transitions,
    }


def coverage_tests(exceptions: np.ndarray, confidence: float) -> dict:
    coverage_result = kupiec(exceptions, confidence)
    independence = christoffersen_independence(exceptions)
    conditional_coverage = {"lr": None, "p_value": None, "status": "not_identifiable"}
    if independence["lr"] is not None:
        conditional_lr = coverage_result["lr"] + independence["lr"]
        conditional_coverage = {
            "lr": conditional_lr,
            "p_value": float(chi2.sf(conditional_lr, 2)),
            "status": "available",
        }
    return {
        "kupiec": coverage_result,
        "independence": independence,
        "conditional_coverage": conditional_coverage,
    }


def rolling_backtest(returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit on the prior window, forecast today, then record today's loss."""
    weights = np.asarray(config["weights"], dtype=float)
    asset_returns = returns.to_numpy(dtype=float)
    window = config["window"]
    if len(asset_returns) <= window or not np.isfinite(asset_returns).all():
        raise ValueError("Insufficient or nonfinite returns for backtesting")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Backtest dates must be unique and increasing")
    forecast_rows = []
    for forecast_position in range(window, len(asset_returns)):
        training_returns = asset_returns[
            forecast_position - window : forecast_position
        ]  # never contains forecast-day return
        training_losses = -config["notional"] * (training_returns @ weights)
        realized_loss = -config["notional"] * float(
            asset_returns[forecast_position] @ weights
        )
        mean_loss, sigma_loss = moments(training_returns, weights, config["notional"])
        simulated_losses = monte_carlo_losses(
            mean_loss,
            sigma_loss,
            config["rolling_draws"],
            config["seed"] + forecast_position,
        )
        for confidence in config["confidence_levels"]:
            predictions = {
                "historical": empirical_risk(training_losses, confidence),
                "parametric": gaussian_risk(mean_loss, sigma_loss, confidence),
                "monte_carlo": empirical_risk(simulated_losses, confidence),
            }
            for method, (value_at_risk, expected_shortfall) in predictions.items():
                forecast_rows.append(
                    dict(
                        date=returns.index[forecast_position],
                        method=method,
                        confidence=confidence,
                        training_start=returns.index[forecast_position - window],
                        training_end=returns.index[forecast_position - 1],
                        training_observations=window,
                        realized_loss_usd=realized_loss,
                        var_usd=value_at_risk,
                        es_usd=expected_shortfall,
                        exception=bool(realized_loss > value_at_risk),
                        exceedance_usd=max(0.0, realized_loss - value_at_risk),
                    )
                )
    return pd.DataFrame(forecast_rows)


def summarize_backtest(daily: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (method, confidence), model_forecasts in daily.groupby(
        ["method", "confidence"], sort=True
    ):
        model_forecasts = model_forecasts.sort_values("date")
        evaluation_samples = {"full_oos": model_forecasts}
        if len(model_forecasts) >= 250:
            evaluation_samples["latest_250_forecasts"] = model_forecasts.iloc[-250:]
        for scope, evaluated_forecasts in evaluation_samples.items():
            exception_flags = evaluated_forecasts["exception"].to_numpy(dtype=int)
            statistical_tests = coverage_tests(exception_flags, confidence)
            summary_row = dict(
                method=method,
                confidence=confidence,
                evaluation=scope,
                forecasts=len(evaluated_forecasts),
                start=str(pd.Timestamp(evaluated_forecasts.iloc[0]["date"]).date()),
                end=str(pd.Timestamp(evaluated_forecasts.iloc[-1]["date"]).date()),
                exceptions=int(exception_flags.sum()),
                exception_rate=float(exception_flags.mean()),
                expected_rate=1 - confidence,
                expected_exceptions=len(exception_flags) * (1 - confidence),
                observed_to_expected=float(exception_flags.mean() / (1 - confidence)),
                sample_warning="Asymptotic tests; few 99% exceptions limit power",
            )
            for test_name, test_result in statistical_tests.items():
                for key in ("lr", "p_value", "status"):
                    summary_row[f"{test_name}_{key}"] = test_result[key]
                summary_row[f"{test_name}_reject_5pct"] = (
                    None
                    if test_result["p_value"] is None
                    else test_result["p_value"] < 0.05
                )
            summary_rows.append(summary_row)
    return pd.DataFrame(summary_rows)
