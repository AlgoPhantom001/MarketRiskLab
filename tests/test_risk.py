import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from marketrisklab.risk import (
    empirical_risk,
    gaussian_risk,
    moments,
    monte_carlo_losses,
)


@pytest.mark.parametrize("confidence", [0.95, 0.99, 0.975])
def test_normal_closed_form(confidence):
    var, es = gaussian_risk(0.0, 1.0, confidence)
    assert var == pytest.approx(norm.ppf(confidence), abs=1e-12)
    assert es == pytest.approx(
        norm.pdf(norm.ppf(confidence)) / (1 - confidence), abs=1e-12
    )
    assert es >= var


def test_exact_fractional_tail_and_ties():
    # 250 observations; 99% ES includes worst 2.5 observations' mass.
    losses = np.arange(250, dtype=float)
    var, es = empirical_risk(losses, 0.99)
    assert var == 247
    assert es == pytest.approx((249 + 248 + 0.5 * 247) / 2.5)
    assert empirical_risk(np.full(250, 7.0), 0.99) == pytest.approx((7, 7))
    assert empirical_risk(np.array([1, 2, 3, 4]), 0.75) == (3, 4)


def test_signed_gain_threshold_is_not_clipped():
    var, es = empirical_risk(np.array([-4.0, -3.0, -2.0, -1.0]), 0.75)
    assert (var, es) == (-2.0, -1.0)


def test_portfolio_moments_and_exposure_scaling():
    returns = np.array([[0.01, -0.02], [-0.01, 0.01], [0.02, 0.03], [0.03, -0.01]])
    w = np.array([0.6, 0.4])
    mean, sigma = moments(returns, w)
    assert mean == pytest.approx(-np.mean(returns @ w))
    assert sigma == pytest.approx(np.std(returns @ w, ddof=1))
    np.testing.assert_allclose(moments(returns, w, 100), np.array([mean, sigma]) * 100)


@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_monte_carlo_convergence_and_reproducibility(confidence):
    losses = monte_carlo_losses(0.1, 2.0, 400000, 42)
    np.testing.assert_array_equal(losses, monte_carlo_losses(0.1, 2.0, 400000, 42))
    actual = empirical_risk(losses, confidence)
    expected = gaussian_risk(0.1, 2.0, confidence)
    np.testing.assert_allclose(actual, expected, atol=0.025, rtol=0)


@pytest.mark.parametrize(
    "losses,c",
    [([], 0.95), ([np.nan], 0.95), ([1, 2], 1), ([1, 2], 0), ([[1, 2]], 0.95)],
)
def test_bad_empirical_inputs(losses, c):
    with pytest.raises(ValueError):
        empirical_risk(np.array(losses), c)


def test_zero_volatility_and_confidence_order():
    assert gaussian_risk(-10, 0, 0.99) == (-10.0, -10.0)
    values = np.random.default_rng(22).standard_t(5, 1000)
    for confidence in [0.95, 0.975, 0.99]:
        var, es = empirical_risk(values, confidence)
        assert es >= var
    assert empirical_risk(values, 0.99)[0] >= empirical_risk(values, 0.95)[0]
