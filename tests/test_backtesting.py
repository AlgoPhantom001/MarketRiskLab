import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from marketrisklab.backtesting import (
    christoffersen_independence,
    coverage_tests,
    kupiec,
    rolling_backtest,
    summarize_backtest,
)


def test_kupiec_hand_calculation_and_boundary_counts():
    flags = np.array([1] * 8 + [0] * 92)
    actual = kupiec(flags, 0.95)
    expected = 2 * (92 * np.log(0.92 / 0.95) + 8 * np.log(0.08 / 0.05))
    assert actual["lr"] == pytest.approx(expected, abs=1e-12)
    assert actual["p_value"] == pytest.approx(chi2.sf(expected, 1))
    assert kupiec(np.array([1] * 50 + [0] * 950), 0.95)["lr"] == pytest.approx(
        0, abs=1e-10
    )
    for flags in (np.zeros(250), np.ones(250)):
        result = kupiec(flags, 0.99)
        assert np.isfinite(result["lr"])


def test_independence_transition_likelihood():
    flags = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    result = christoffersen_independence(flags)
    assert [result[k] for k in ("n00", "n01", "n10", "n11")] == [3, 3, 2, 1]
    # independent null p=4/9, conditional probabilities 3/6 and 1/3
    null = 5 * np.log(5 / 9) + 4 * np.log(4 / 9)
    fitted = 3 * np.log(0.5) + 3 * np.log(0.5) + 2 * np.log(2 / 3) + np.log(1 / 3)
    assert result["lr"] == pytest.approx(-2 * (null - fitted), abs=1e-12)


def test_clustering_and_conditional_decomposition():
    clustered = np.concatenate([np.zeros(180), np.ones(20)] * 5)
    result = coverage_tests(clustered, 0.95)
    assert result["independence"]["p_value"] < 0.001
    assert result["conditional_coverage"]["lr"] == pytest.approx(
        result["kupiec"]["lr"] + result["independence"]["lr"]
    )
    independent = (np.random.default_rng(42).random(3000) < 0.05).astype(int)
    assert christoffersen_independence(independent)["p_value"] > 0.05


@pytest.mark.parametrize("flags", [np.zeros(250), np.ones(250), np.array([0])])
def test_unidentifiable_independence_is_explicit(flags):
    result = coverage_tests(flags, 0.99)
    assert result["independence"]["p_value"] is None
    assert result["conditional_coverage"]["status"] == "not_identifiable"


def test_forecasts_exclude_current_and_future_observations():
    config = {
        "weights": [0.6, 0.4],
        "notional": 1e6,
        "window": 250,
        "rolling_draws": 2000,
        "seed": 42,
        "confidence_levels": [0.95, 0.99],
    }
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (260, 2)), index=pd.bdate_range("2020-01-01", periods=260)
    )
    original = rolling_backtest(returns, config)
    altered = returns.copy()
    altered.iloc[254:] = 0.50
    changed = rolling_backtest(altered, config)
    dates = original["date"] <= returns.index[254]
    for col in ("var_usd", "es_usd", "training_start", "training_end"):
        pd.testing.assert_series_equal(
            original.loc[dates, col], changed.loc[dates, col]
        )
    assert (original["training_end"] < original["date"]).all()
    assert len(original) == 10 * 6
    first_loss = -1e6 * float(returns.iloc[250].to_numpy() @ np.array([0.6, 0.4]))
    assert original.iloc[0]["realized_loss_usd"] == pytest.approx(first_loss)
    assert np.array_equal(
        original["exception"], original["realized_loss_usd"] > original["var_usd"]
    )
    summary = summarize_backtest(original)
    assert len(summary) == 6
    assert (summary["forecasts"] == 10).all()


@pytest.mark.parametrize("flags", [[], [0, 2], [np.nan]])
def test_invalid_exception_sequence(flags):
    with pytest.raises(ValueError):
        kupiec(np.array(flags), 0.95)
