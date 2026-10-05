import numpy as np
import pytest

from marketrisklab.options import black_scholes, DEMO_OPTION, option_tables


def test_textbook_prices():
    call = black_scholes(**DEMO_OPTION, kind="call")
    put = black_scholes(**DEMO_OPTION, kind="put")
    assert call["price"] == pytest.approx(10.450583572185565, abs=1e-10)
    assert put["price"] == pytest.approx(5.573526022256971, abs=1e-10)


@pytest.mark.parametrize(
    "spot,strike,t,r,sigma,q",
    [
        (100, 100, 1, 0.05, 0.2, 0),
        (80, 110, 0.5, 0.02, 0.35, 0.01),
        (150, 100, 2, -0.01, 0.15, 0.03),
    ],
)
def test_put_call_parity_and_bounds(spot, strike, t, r, sigma, q):
    call = black_scholes(spot, strike, t, r, sigma, "call", q)["price"]
    put = black_scholes(spot, strike, t, r, sigma, "put", q)["price"]
    dq, dr = np.exp(-q * t), np.exp(-r * t)
    assert call - put == pytest.approx(spot * dq - strike * dr, abs=1e-10)
    assert max(0, spot * dq - strike * dr) - 1e-12 <= call <= spot * dq
    assert max(0, strike * dr - spot * dq) - 1e-12 <= put <= strike * dr


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_against_independent_finite_differences(kind):
    params = {**DEMO_OPTION, "dividend": 0.02}
    base = black_scholes(**params, kind=kind)

    def price(**changes):
        return black_scholes(**{**params, **changes}, kind=kind)["price"]

    h = 0.01
    delta = (price(spot=params["spot"] + h) - price(spot=params["spot"] - h)) / (2 * h)
    gamma = (
        price(spot=params["spot"] + h)
        - 2 * base["price"]
        + price(spot=params["spot"] - h)
    ) / h**2
    v = 1e-5
    vega = (
        (
            price(volatility=params["volatility"] + v)
            - price(volatility=params["volatility"] - v)
        )
        / (2 * v)
        * 0.01
    )
    t = 1e-5
    theta = (
        -(
            price(maturity=params["maturity"] + t)
            - price(maturity=params["maturity"] - t)
        )
        / (2 * t)
        / 365
    )
    assert base["delta"] == pytest.approx(delta, abs=1e-7)
    assert base["gamma"] == pytest.approx(gamma, abs=1e-7)
    assert base["vega"] == pytest.approx(vega, abs=1e-7)
    assert base["theta"] == pytest.approx(theta, abs=1e-8)


def test_expiry_and_zero_volatility():
    expired = black_scholes(110, 100, 0, 0.05, 0.2)
    assert expired["price"] == 10
    assert np.isnan(expired["delta"])
    assert expired["greek_status"] == "undefined_at_expiry"
    deterministic = black_scholes(110, 100, 1, 0.05, 0)
    assert deterministic["price"] == pytest.approx(110 - 100 * np.exp(-0.05))
    assert deterministic["delta"] == 1
    kink = black_scholes(100, 100, 1, 0, 0)
    assert np.isnan(kink["gamma"])


def test_stress_zero_shock_and_sensitivity():
    prices, stress = option_tables()
    zero = stress.loc[stress.scenario == "zero_shock"]
    assert (zero.full_repricing_pnl_usd == 0).all()
    vol = stress.loc[stress.scenario == "volatility_plus10points"]
    assert (vol.full_repricing_pnl_usd > 0).all()
    assert (prices.gamma > 0).all() and (prices.vega > 0).all()
    fall = stress.loc[stress.scenario == "equity_minus20pct"]
    assert fall.loc[fall.kind == "call", "full_repricing_loss_usd"].iloc[0] > 0
    assert fall.loc[fall.kind == "put", "full_repricing_loss_usd"].iloc[0] < 0


@pytest.mark.parametrize(
    "changes",
    [
        {"spot": 0},
        {"strike": -1},
        {"maturity": -1},
        {"volatility": -0.1},
        {"kind": "american"},
        {"rate": np.nan},
    ],
)
def test_invalid_option_inputs(changes):
    with pytest.raises(ValueError):
        black_scholes(**{**DEMO_OPTION, **changes})
