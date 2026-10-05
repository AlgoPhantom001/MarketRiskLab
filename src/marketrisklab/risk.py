"""One-day loss-distribution estimators with explicit confidence and units.

Risk is a signed loss quantile: a negative result denotes a gain threshold.
Values are never floored to zero. Gaussian MC projects the fitted multivariate
distribution onto portfolio weights; this is exactly equivalent for linear P&L.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def check_confidence(confidence: float) -> None:
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")


def empirical_risk(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    """Inverse empirical CDF VaR and quantile-integrated ES.

    ES uses exactly n*(1-confidence) observations' probability mass, including
    fractional boundary weight, rather than all observations >= VaR.
    """
    check_confidence(confidence)
    loss_values = np.asarray(losses, dtype=float)
    if (
        loss_values.ndim != 1
        or len(loss_values) == 0
        or not np.isfinite(loss_values).all()
    ):
        raise ValueError("Need a nonempty finite one-dimensional loss sample")
    value_at_risk = float(np.quantile(loss_values, confidence, method="inverted_cdf"))
    tail_mass = len(loss_values) * (1 - confidence)
    descending_losses = np.sort(loss_values)[::-1]
    whole_observations = int(np.floor(tail_mass))
    fractional_observation = tail_mass - whole_observations
    weighted_tail_loss = descending_losses[:whole_observations].sum()
    if fractional_observation > 0 and whole_observations < len(loss_values):
        weighted_tail_loss += (
            fractional_observation * descending_losses[whole_observations]
        )
    expected_shortfall = float(weighted_tail_loss / tail_mass)
    return value_at_risk, expected_shortfall


def gaussian_risk(
    mean_loss: float, sigma_loss: float, confidence: float
) -> tuple[float, float]:
    """Normal-distribution VaR and average loss beyond that quantile."""
    check_confidence(confidence)
    if not np.isfinite([mean_loss, sigma_loss]).all() or sigma_loss < 0:
        raise ValueError("Finite mean and nonnegative standard deviation required")
    z_score = norm.ppf(confidence)
    value_at_risk = float(mean_loss + sigma_loss * z_score)
    expected_shortfall = float(
        mean_loss + sigma_loss * norm.pdf(z_score) / (1 - confidence)
    )
    return value_at_risk, expected_shortfall


def moments(
    returns: np.ndarray, weights: np.ndarray, notional: float = 1.0
) -> tuple[float, float]:
    """Sample mean and sample covariance (ddof=1), converted to loss units."""
    asset_returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        asset_returns.ndim != 2
        or len(asset_returns) < 2
        or asset_returns.shape[1] != len(weights)
    ):
        raise ValueError("Need at least two observations and matching asset weights")
    if not np.isfinite(asset_returns).all() or not np.isfinite(weights).all():
        raise ValueError("Returns and weights must be finite")
    if not np.isfinite(notional) or notional <= 0:
        raise ValueError("Notional must be positive")
    mean_portfolio_return = float(asset_returns.mean(axis=0) @ weights)
    centered_returns = asset_returns - asset_returns.mean(axis=0)
    covariance_matrix = centered_returns.T @ centered_returns / (len(asset_returns) - 1)
    portfolio_variance = float(weights @ covariance_matrix @ weights)
    # Positive portfolio returns become negative losses.
    mean_loss = -notional * mean_portfolio_return
    sigma_loss = notional * np.sqrt(max(portfolio_variance, 0.0))
    return mean_loss, sigma_loss


def monte_carlo_losses(
    mean_loss: float, sigma_loss: float, draws: int, seed: int
) -> np.ndarray:
    if draws < 1000 or not isinstance(draws, (int, np.integer)):
        raise ValueError("At least 1,000 integer draws required")
    if not np.isfinite([mean_loss, sigma_loss]).all() or sigma_loss < 0:
        raise ValueError("Invalid Gaussian parameters")
    random_generator = np.random.default_rng(seed)
    return random_generator.normal(mean_loss, sigma_loss, draws)


def compare_models(returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = np.asarray(config["weights"])
    notional = config["notional"]
    calibration_returns = returns.iloc[-config["window"] :]
    if len(calibration_returns) != config["window"]:
        raise ValueError("Insufficient calibration observations")
    losses = -notional * (calibration_returns.to_numpy() @ weights)
    mean_loss, sigma_loss = moments(calibration_returns.to_numpy(), weights, notional)
    simulated_losses = monte_carlo_losses(
        mean_loss, sigma_loss, config["snapshot_draws"], config["seed"]
    )
    results = []
    for confidence in config["confidence_levels"]:
        estimates = {
            "historical": empirical_risk(losses, confidence),
            "parametric": gaussian_risk(mean_loss, sigma_loss, confidence),
            "monte_carlo": empirical_risk(simulated_losses, confidence),
        }
        for method, (value_at_risk, expected_shortfall) in estimates.items():
            results.append(
                dict(
                    method=method,
                    confidence=confidence,
                    horizon_trading_days=1,
                    var_usd=value_at_risk,
                    es_usd=expected_shortfall,
                    var_pct=100 * value_at_risk / notional,
                    es_pct=100 * expected_shortfall / notional,
                    calibration_observations=len(calibration_returns),
                    calibration_start=str(calibration_returns.index[0].date()),
                    calibration_end=str(calibration_returns.index[-1].date()),
                )
            )
    return pd.DataFrame(results)
