import numpy as np
import pandas as pd
import pytest

from marketrisklab.stress import correlation_stress, scenario_loss, stress_table


def test_zero_and_uniform_shocks_reconcile():
    weights = np.array([0.4, 0.35, 0.25])
    assert scenario_loss(weights, np.zeros(3), 1e6) == 0
    assert scenario_loss(weights, np.full(3, -0.1), 1e6) == pytest.approx(100000)
    assert scenario_loss(weights, np.full(3, -0.2), 1e6) == pytest.approx(200000)


def test_correlation_stress_psd_diagonal_and_direction():
    covariance = np.array([[0.0001, 0.00004], [0.00004, 0.0004]])
    stressed = correlation_stress(covariance)
    np.testing.assert_allclose(np.diag(stressed), np.diag(covariance))
    assert np.linalg.eigvalsh(stressed).min() >= -1e-12
    weights = np.array([0.6, 0.4])
    assert weights @ stressed @ weights >= weights @ covariance @ weights
    np.testing.assert_allclose(correlation_stress(covariance, 0), covariance)


def test_zero_volatility_asset():
    stressed = correlation_stress(np.array([[0.0, 0.0], [0.0, 0.04]]))
    np.testing.assert_allclose(stressed, [[0, 0], [0, 0.04]])


def test_stress_table_risk_and_losses_remain_distinct():
    returns = pd.DataFrame(np.random.default_rng(4).normal(0, 0.01, (250, 3)))
    config = {
        "window": 250,
        "weights": [0.4, 0.35, 0.25],
        "notional": 1e6,
        "confidence_levels": [0.95, 0.99],
    }
    table = stress_table(returns, config)
    risks = table.loc[table.result_type == "distribution_risk"]
    losses = table.loc[table.result_type == "deterministic_scenario"]
    assert risks.deterministic_loss_usd.isna().all()
    assert losses.var_usd.isna().all()
    baseline = risks.loc[risks.scenario == "baseline"].iloc[0]
    stressed = risks.loc[risks.scenario == "volatility_x1.5"].iloc[0]
    assert stressed.portfolio_daily_sigma_usd == pytest.approx(
        1.5 * baseline.portfolio_daily_sigma_usd
    )
    assert stressed.var_usd > baseline.var_usd


@pytest.mark.parametrize(
    "matrix",
    [np.array([[1, 2], [2, 1]]), np.array([[1, 0.5], [0, 1]]), np.array([[np.nan]])],
)
def test_bad_covariance(matrix):
    with pytest.raises(ValueError):
        correlation_stress(matrix)
