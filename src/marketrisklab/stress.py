"""Hypothetical price losses and separate covariance stress risk measures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import gaussian_risk, moments


def scenario_loss(
    weights: np.ndarray, asset_shocks: np.ndarray, notional: float
) -> float:
    weights = np.asarray(weights, dtype=float)
    asset_shocks = np.asarray(asset_shocks, dtype=float)
    if (
        weights.ndim != 1
        or weights.shape != asset_shocks.shape
        or not np.isfinite([*weights, *asset_shocks, notional]).all()
    ):
        raise ValueError("Finite matching weights, shocks and notional required")
    if notional <= 0 or (asset_shocks < -1).any():
        raise ValueError("Positive notional and shocks no worse than -100% required")
    return float(-notional * (weights @ asset_shocks))


def correlation_stress(covariance: np.ndarray, strength: float = 0.9) -> np.ndarray:
    """Convex combination with a perfectly synchronized correlation matrix.

    Valid baseline covariance -> valid stressed covariance, unchanged marginal
    volatilities. Zero-volatility assets retain zero covariance throughout.
    """
    covariance = np.asarray(covariance, dtype=float)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.isfinite(covariance).all()
        or not 0 <= strength <= 1
    ):
        raise ValueError("Finite square covariance and strength in [0,1] required")
    if (
        not np.allclose(covariance, covariance.T)
        or np.linalg.eigvalsh(covariance).min() < -1e-12
    ):
        raise ValueError("Baseline covariance must be symmetric positive semidefinite")
    asset_volatilities = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    volatility_products = np.outer(asset_volatilities, asset_volatilities)
    correlation = np.divide(
        covariance,
        volatility_products,
        out=np.zeros_like(covariance),
        where=volatility_products > 0,
    )
    np.fill_diagonal(correlation, 1.0)
    stressed_correlation = (1 - strength) * correlation + strength * np.ones_like(
        correlation
    )
    return stressed_correlation * volatility_products


def stress_table(returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    calibration_returns = returns.iloc[-config["window"] :].to_numpy()
    weights = np.asarray(config["weights"], dtype=float)
    notional = config["notional"]
    covariance_matrix = np.atleast_2d(np.cov(calibration_returns, rowvar=False, ddof=1))
    asset_daily_volatility = np.sqrt(np.diag(covariance_matrix))
    mean_loss, portfolio_sigma_loss = moments(calibration_returns, weights, notional)
    stressed_covariance = correlation_stress(covariance_matrix)
    stressed_sigma_loss = notional * np.sqrt(
        max(float(weights @ stressed_covariance @ weights), 0.0)
    )
    stress_rows = []
    risk_scenarios = {
        "baseline": (portfolio_sigma_loss, "Latest 250-day covariance"),
        "volatility_x1.5": (
            1.5 * portfolio_sigma_loss,
            "Each marginal daily volatility x1.5; correlations and mean unchanged",
        ),
        "high_correlation": (
            stressed_sigma_loss,
            "R_stress = 0.1 R + 0.9 J; marginal volatilities unchanged",
        ),
    }
    for scenario, (scenario_sigma_loss, description) in risk_scenarios.items():
        for confidence in config["confidence_levels"]:
            value_at_risk, expected_shortfall = gaussian_risk(
                mean_loss, scenario_sigma_loss, confidence
            )
            stress_rows.append(
                dict(
                    scenario=scenario,
                    result_type="distribution_risk",
                    confidence=confidence,
                    deterministic_loss_usd=None,
                    var_usd=value_at_risk,
                    es_usd=expected_shortfall,
                    portfolio_daily_sigma_usd=scenario_sigma_loss,
                    description=description,
                )
            )
    for scenario, shocks, description in [
        (
            "equity_minus10pct",
            np.full(len(weights), -0.10),
            "All equity prices fall 10%",
        ),
        (
            "equity_minus20pct",
            np.full(len(weights), -0.20),
            "All equity prices fall 20%",
        ),
        (
            "synchronized_minus2sigma",
            -2 * asset_daily_volatility,
            "Each equity falls twice its own daily standard deviation simultaneously",
        ),
    ]:
        deterministic_loss = scenario_loss(weights, shocks, notional)
        stress_rows.append(
            dict(
                scenario=scenario,
                result_type="deterministic_scenario",
                confidence=None,
                deterministic_loss_usd=deterministic_loss,
                var_usd=None,
                es_usd=None,
                portfolio_daily_sigma_usd=None,
                description=description,
            )
        )
    return pd.DataFrame(stress_rows)
