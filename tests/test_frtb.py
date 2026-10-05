import numpy as np
import pandas as pd
import pytest

from marketrisklab.frtb import (
    frtb_tables,
    liquidity_es,
    select_stressed_window,
    ten_day_changes,
)
from marketrisklab.risk import empirical_risk


def fixture(n=250):
    return pd.DataFrame(
        np.random.default_rng(8).normal(0, 0.015, (n, 2)),
        index=pd.bdate_range("2019-01-01", periods=n),
        columns=["A", "B"],
    )


def test_actual_ten_day_compounding_and_alignment():
    returns = fixture(20)
    changes = ten_day_changes(returns)
    assert len(changes) == 11
    np.testing.assert_allclose(
        changes.iloc[0], (1 + returns.iloc[:10]).prod() - 1, atol=1e-14
    )
    np.testing.assert_allclose(
        changes.iloc[-1], (1 + returns.iloc[-10:]).prod() - 1, atol=1e-14
    )
    assert changes.index[0] == returns.index[9]


@pytest.mark.parametrize("horizon", [10, 20, 40, 60, 120])
def test_single_horizon_aggregation_identity(horizon):
    returns = fixture()
    weights = np.array([0.6, 0.4])
    base = empirical_risk(
        -1e6 * (ten_day_changes(returns).to_numpy() @ weights), 0.975
    )[1]
    aggregated, buckets = liquidity_es(returns, weights, [horizon, horizon], 1e6)
    assert aggregated == pytest.approx(base * np.sqrt(horizon / 10), abs=1e-8)
    assert buckets.loc[buckets.liquidity_horizon > horizon, "factor_count"].sum() == 0


def test_nested_mixed_horizons_and_longer_horizon():
    returns = fixture()
    w = np.array([0.6, 0.4])
    actual, buckets = liquidity_es(returns, w, [10, 20], 1e6)
    base = empirical_risk(-1e6 * (ten_day_changes(returns).to_numpy() @ w), 0.975)[1]
    subset = empirical_risk(-1e6 * 0.4 * ten_day_changes(returns).B.to_numpy(), 0.975)[
        1
    ]
    assert actual == pytest.approx(np.sqrt(base**2 + subset**2))
    assert buckets.iloc[1].factor_subset == "B"
    all10, _ = liquidity_es(returns, w, [10, 10], 1e6)
    assert actual >= all10


def test_selection_matches_independent_small_enumeration():
    returns = fixture(35)
    config = {
        "window": 20,
        "weights": [0.6, 0.4],
        "liquidity_horizons": [10, 20],
        "notional": 1e6,
    }
    values = []
    for start in range(16):
        sample = returns.iloc[start : start + 20]
        # Independently compound each 10-day block, not the production routine.
        blocks = np.array(
            [(1 + sample.iloc[i : i + 10]).prod().to_numpy() - 1 for i in range(11)]
        )
        base = empirical_risk(-1e6 * (blocks @ np.array([0.6, 0.4])), 0.975)[1]
        subset = empirical_risk(-1e6 * 0.4 * blocks[:, 1], 0.975)[1]
        values.append(np.sqrt(max(base, 0) ** 2 + max(subset, 0) ** 2))
    best, candidates = select_stressed_window(returns, config)
    assert best["start_position"] == int(np.argmax(values))
    np.testing.assert_allclose(candidates.liquidity_adjusted_es_usd, values, atol=1e-8)
    summary, buckets, _ = frtb_tables(returns, config)
    assert (
        summary.iloc[1].liquidity_adjusted_es975_usd
        >= summary.iloc[0].liquidity_adjusted_es975_usd
    )
    assert (summary.regulatory_compliance == False).all()


def test_all_gain_case_preserves_raw_es_but_no_charge():
    returns = pd.DataFrame(np.full((30, 1), 0.01), columns=["A"])
    charge, buckets = liquidity_es(returns, [1], [10], 1e6)
    assert buckets.iloc[0].es10_raw_usd < 0
    assert charge == 0


@pytest.mark.parametrize(
    "issue", ["too_short", "return_below_minus1", "nan", "bad_horizon"]
)
def test_bad_horizon_inputs(issue):
    returns = fixture()
    if issue == "too_short":
        returns = returns.iloc[:9]
    elif issue == "return_below_minus1":
        returns.iloc[3, 0] = -1.1
    elif issue == "nan":
        returns.iloc[3, 0] = np.nan
    with pytest.raises(ValueError):
        liquidity_es(
            returns, [0.6, 0.4], [10, 15] if issue == "bad_horizon" else [10, 20], 1e6
        )
