"""Educational FRTB-style ES / liquidity treatment, NOT regulatory capital.

Uses actual overlapping 10-day changes, frozen exposures, historical ES and
nested horizon subsets. No one-day sqrt(10) approximation. No reduced-factor
stress scaling, regulatory multipliers, NMRF, DRC, PLA or supervisory approval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import empirical_risk

HORIZONS = (10, 20, 40, 60, 120)


def ten_day_changes(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound overlapping ten-day returns separately for each asset."""
    asset_returns = daily_returns.to_numpy(dtype=float)
    if (
        len(asset_returns) < 10
        or not np.isfinite(asset_returns).all()
        or (asset_returns <= -1).any()
    ):
        raise ValueError(
            "At least ten finite daily simple returns greater than -1 required"
        )
    # Log returns turn multiplication into addition. Subtract running totals
    # ten rows apart, then convert back to a compounded simple return.
    cumulative_log_returns = np.vstack(
        [np.zeros(asset_returns.shape[1]), np.cumsum(np.log1p(asset_returns), axis=0)]
    )
    compounded_changes = np.expm1(
        cumulative_log_returns[10:] - cumulative_log_returns[:-10]
    )
    return pd.DataFrame(
        compounded_changes, index=daily_returns.index[9:], columns=daily_returns.columns
    )


def liquidity_es(
    daily_returns: pd.DataFrame,
    weights: np.ndarray,
    horizons: np.ndarray,
    notional: float,
) -> tuple[float, pd.DataFrame]:
    weights, horizons = np.asarray(weights, dtype=float), np.asarray(horizons)
    if len(weights) != daily_returns.shape[1] or len(horizons) != len(weights):
        raise ValueError("Matching weights and liquidity horizons required")
    if (
        not np.isin(horizons, HORIZONS).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(notional)
        or notional <= 0
    ):
        raise ValueError("Finite exposures and supported liquidity horizons required")
    changes = ten_day_changes(daily_returns)
    bucket_rows, previous_horizon = [], 0
    for horizon in HORIZONS:
        # Longer horizons retain only factors that cannot be exited sooner.
        included_factors = horizons >= horizon
        if included_factors.any():
            losses = -notional * (
                changes.to_numpy()[:, included_factors] @ weights[included_factors]
            )
            ten_day_var, ten_day_es = empirical_risk(losses, 0.975)
        else:
            ten_day_var, ten_day_es = 0.0, 0.0
        # Charges are nonnegative; retain raw ES so an all-gain case is not
        # silently converted into a positive risk charge by squaring it.
        nonnegative_es = max(ten_day_es, 0.0)
        horizon_increment = (horizon - previous_horizon) / 10
        bucket_rows.append(
            dict(
                liquidity_horizon=horizon,
                factor_subset=",".join(daily_returns.columns[included_factors]),
                factor_count=int(included_factors.sum()),
                ten_day_scenarios=len(changes),
                var10_raw_usd=ten_day_var,
                es10_raw_usd=ten_day_es,
                es10_charge_usd=nonnegative_es,
                horizon_increment_over10=horizon_increment,
                squared_contribution=nonnegative_es**2 * horizon_increment,
            )
        )
        previous_horizon = horizon
    bucket_table = pd.DataFrame(bucket_rows)
    return float(np.sqrt(bucket_table.squared_contribution.sum())), bucket_table


def select_stressed_window(
    daily_returns: pd.DataFrame, config: dict
) -> tuple[dict, pd.DataFrame]:
    window_size = config["window"]
    if len(daily_returns) < window_size:
        raise ValueError("Insufficient history to select stressed calibration")
    window_results = []
    for start_position in range(len(daily_returns) - window_size + 1):
        calibration_returns = daily_returns.iloc[
            start_position : start_position + window_size
        ]
        liquidity_adjusted_es, _ = liquidity_es(
            calibration_returns,
            config["weights"],
            config["liquidity_horizons"],
            config["notional"],
        )
        window_results.append(
            dict(
                start_position=start_position,
                start=str(calibration_returns.index[0].date()),
                end=str(calibration_returns.index[-1].date()),
                liquidity_adjusted_es_usd=liquidity_adjusted_es,
            )
        )
    candidate_windows = pd.DataFrame(window_results)
    # Earliest window wins an exact tie. Selection is retrospective descriptive
    # calibration; it never enters out-of-sample VaR forecasts.
    stressed_window = candidate_windows.iloc[
        int(np.argmax(candidate_windows.liquidity_adjusted_es_usd.to_numpy()))
    ].to_dict()
    return stressed_window, candidate_windows


def frtb_tables(
    daily_returns: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stressed_window, candidate_windows = select_stressed_window(daily_returns, config)
    stressed_start = int(stressed_window["start_position"])
    calibration_samples = {
        "current": daily_returns.iloc[-config["window"] :],
        "stressed": daily_returns.iloc[
            stressed_start : stressed_start + config["window"]
        ],
    }
    comparison_rows, bucket_tables = [], []
    for calibration, calibration_returns in calibration_samples.items():
        one_day_losses = -config["notional"] * (
            calibration_returns.to_numpy() @ np.asarray(config["weights"])
        )
        one_day_var_99, _ = empirical_risk(one_day_losses, 0.99)
        _, one_day_es_975 = empirical_risk(one_day_losses, 0.975)
        liquidity_adjusted_es, bucket_table = liquidity_es(
            calibration_returns,
            config["weights"],
            config["liquidity_horizons"],
            config["notional"],
        )
        all_factor_bucket = bucket_table.iloc[0]
        comparison_rows.append(
            dict(
                calibration=calibration,
                start=str(calibration_returns.index[0].date()),
                end=str(calibration_returns.index[-1].date()),
                daily_observations=len(calibration_returns),
                ten_day_scenarios=int(all_factor_bucket.ten_day_scenarios),
                var99_1day_usd=one_day_var_99,
                es975_1day_usd=one_day_es_975,
                var975_10day_usd=all_factor_bucket.var10_raw_usd,
                es975_10day_usd=all_factor_bucket.es10_raw_usd,
                liquidity_adjusted_es975_usd=liquidity_adjusted_es,
                illustrative_capital_proxy_usd=(
                    liquidity_adjusted_es if calibration == "stressed" else None
                ),
                regulatory_compliance=False,
                proxy_definition="stressed_LH_ES_only_no_regulatory_multiplier",
            )
        )
        bucket_tables.append(bucket_table.assign(calibration=calibration))
    return (
        pd.DataFrame(comparison_rows),
        pd.concat(bucket_tables, ignore_index=True),
        candidate_windows,
    )
