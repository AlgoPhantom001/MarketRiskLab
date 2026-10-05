"""European Black-Scholes-Merton prices and explicitly scaled Greeks.

Time is years (ACT/365 for calendar-day maturity inputs), volatility is annual
decimal volatility, r/q continuously compounded. Vega is per +0.01 volatility;
theta is per calendar day. Each price is per underlying share/option unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

DEMO_OPTION = dict(
    spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.20, dividend=0.0
)
DEMO_UNITS = 100


def black_scholes(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    kind: str = "call",
    dividend: float = 0.0,
) -> dict:
    """Price one European option and return Greeks in the units above."""
    if not np.isfinite([spot, strike, maturity, rate, volatility, dividend]).all():
        raise ValueError("All option inputs must be finite")
    if (
        spot <= 0
        or strike <= 0
        or maturity < 0
        or volatility < 0
        or kind not in ("call", "put")
    ):
        raise ValueError(
            "Positive spot/strike, nonnegative maturity/volatility and call/put required"
        )
    option_sign = 1 if kind == "call" else -1
    if maturity == 0:
        return dict(
            price=max(option_sign * (spot - strike), 0.0),
            delta=np.nan,
            gamma=np.nan,
            vega=np.nan,
            theta=np.nan,
            greek_status="undefined_at_expiry",
        )
    dividend_discount, rate_discount = np.exp(-dividend * maturity), np.exp(
        -rate * maturity
    )
    if volatility == 0:
        discounted_forward_payoff = spot * dividend_discount - strike * rate_discount
        option_price = max(option_sign * discounted_forward_payoff, 0.0)
        if np.isclose(
            discounted_forward_payoff, 0.0, rtol=0, atol=1e-12 * max(spot, strike)
        ):
            return dict(
                price=option_price,
                delta=np.nan,
                gamma=np.nan,
                vega=np.nan,
                theta=np.nan,
                greek_status="undefined_at_zero_volatility_kink",
            )
        in_the_money = option_sign * discounted_forward_payoff > 0
        return dict(
            price=option_price,
            delta=option_sign * dividend_discount if in_the_money else 0.0,
            gamma=0.0,
            vega=0.0,
            theta=(
                option_sign
                * (dividend * spot * dividend_discount - rate * strike * rate_discount)
                / 365
                if in_the_money
                else 0.0
            ),
            greek_status="available_zero_volatility_limit",
        )
    # Standard Black-Scholes terms: d1 for the asset, d2 for the strike.
    sqrt_maturity = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_maturity)
    d2 = d1 - volatility * sqrt_maturity
    d1_density = norm.pdf(d1)
    option_price = option_sign * (
        spot * dividend_discount * norm.cdf(option_sign * d1)
        - strike * rate_discount * norm.cdf(option_sign * d2)
    )
    delta = option_sign * dividend_discount * norm.cdf(option_sign * d1)
    gamma = dividend_discount * d1_density / (spot * volatility * sqrt_maturity)
    # Convert Vega to one volatility percentage point, and Theta to one day.
    vega = spot * dividend_discount * d1_density * sqrt_maturity * 0.01
    annual_theta = (
        -spot * dividend_discount * d1_density * volatility / (2 * sqrt_maturity)
        - option_sign * rate * strike * rate_discount * norm.cdf(option_sign * d2)
        + option_sign * dividend * spot * dividend_discount * norm.cdf(option_sign * d1)
    )
    return dict(
        price=float(option_price),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(annual_theta / 365),
        greek_status="available",
    )


def option_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Illustrative instruments, separate from ETF VaR and capital calculations."""
    price_rows, stress_rows = [], []
    scenarios = [
        ("zero_shock", 0.0, 0.0),
        ("equity_minus10pct", -0.10, 0.0),
        ("equity_minus20pct", -0.20, 0.0),
        ("volatility_plus10points", 0.0, 0.10),
        ("equity_minus20pct_vol_plus10points", -0.20, 0.10),
    ]
    for kind in ("call", "put"):
        baseline_result = black_scholes(**DEMO_OPTION, kind=kind)
        price_rows.append(
            dict(
                kind=kind,
                **DEMO_OPTION,
                **baseline_result,
                option_units=DEMO_UNITS,
                source="illustrative_inputs_not_market_option_quotes",
            )
        )
        for name, spot_shock, volatility_shock in scenarios:
            shocked_inputs = {
                **DEMO_OPTION,
                "spot": DEMO_OPTION["spot"] * (1 + spot_shock),
                "volatility": DEMO_OPTION["volatility"] + volatility_shock,
            }
            shocked_result = black_scholes(**shocked_inputs, kind=kind)
            spot_change = shocked_inputs["spot"] - DEMO_OPTION["spot"]
            approximate_change_per_unit = (
                baseline_result["delta"] * spot_change
                + 0.5 * baseline_result["gamma"] * spot_change**2
                + baseline_result["vega"] * (volatility_shock / 0.01)
            )
            full_repricing_pnl = DEMO_UNITS * (
                shocked_result["price"] - baseline_result["price"]
            )
            approximate_pnl = DEMO_UNITS * approximate_change_per_unit
            stress_rows.append(
                dict(
                    kind=kind,
                    scenario=name,
                    spot_shock_pct=100 * spot_shock,
                    volatility_shock_percentage_points=100 * volatility_shock,
                    option_units=DEMO_UNITS,
                    baseline_price=baseline_result["price"],
                    stressed_price=shocked_result["price"],
                    full_repricing_pnl_usd=full_repricing_pnl,
                    full_repricing_loss_usd=-full_repricing_pnl,
                    dgamma_vega_pnl_usd=approximate_pnl,
                    approximation_error_usd=approximate_pnl - full_repricing_pnl,
                )
            )
    return pd.DataFrame(price_rows), pd.DataFrame(stress_rows)
