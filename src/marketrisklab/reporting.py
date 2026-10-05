"""Tables, report narrative and Matplotlib plots driven only by engine outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .options import black_scholes, DEMO_OPTION

COLORS = {"historical": "#176b87", "parametric": "#e49b2e", "monte_carlo": "#7757a5"}


def markdown_table(frame: pd.DataFrame) -> str:
    """No additional table-rendering dependency; unavailable values stay explicit."""

    def fmt(value):
        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            return "unavailable"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "/")

    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(fmt(x) for x in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def plot_outputs(
    returns: pd.DataFrame,
    config: dict,
    tables: dict[str, pd.DataFrame],
    output: Path,
    source_label: str,
) -> list[str]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    generated = []

    def save(fig, name, title, subtitle=None):
        fig.suptitle(title + "\n" + (subtitle or source_label), fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        fig.savefig(figures / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated.append("figures/" + name)

    n = config["notional"]
    losses = -n * (
        returns.iloc[-config["window"] :].to_numpy() @ np.asarray(config["weights"])
    )
    historical = (
        tables["risk_comparison"]
        .query("method == 'historical' and confidence == 0.99")
        .iloc[0]
    )
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(losses / 1000, bins=35, color="#76a8b9", edgecolor="white")
    ax.axvline(historical.var_usd / 1000, color="#e49b2e", label="Historical 99% VaR")
    ax.axvline(
        historical.es_usd / 1000,
        color="#a13d4c",
        linestyle="--",
        label="Historical 99% ES",
    )
    ax.set(
        xlabel="One-day loss (USD thousands; gains are negative)", ylabel="Observations"
    )
    ax.legend()
    save(fig, "loss_distribution.png", "Latest 250-day empirical loss distribution")

    daily = tables["backtest_daily"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, confidence in zip(axes, [0.95, 0.99]):
        sample = daily.loc[
            (daily.method == "historical") & (daily.confidence == confidence)
        ]
        ax.plot(
            sample.date,
            sample.realized_loss_usd / 1000,
            color="#c6ced2",
            lw=0.8,
            label="Realized loss",
        )
        for method, color in COLORS.items():
            forecast = daily.loc[
                (daily.method == method) & (daily.confidence == confidence)
            ]
            ax.plot(
                forecast.date,
                forecast.var_usd / 1000,
                color=color,
                lw=0.85,
                label=method + " VaR",
            )
        breaches = sample.loc[sample.exception]
        ax.scatter(
            breaches.date,
            breaches.realized_loss_usd / 1000,
            color="#ae3041",
            s=13,
            zorder=4,
            label="Historical exceptions",
        )
        ax.set(
            ylabel="USD thousands",
            title=f"{confidence:.0%} confidence; prior 250 observations",
        )
        ax.grid(alpha=0.15)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlim(sample.date.iloc[0], sample.date.iloc[-1])
    save(fig, "rolling_var_exceptions.png", "Out-of-sample one-day VaR and exceptions")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, confidence in zip(axes, [0.95, 0.99]):
        comparison = tables["risk_comparison"].loc[
            tables["risk_comparison"].confidence == confidence
        ]
        x = np.arange(len(comparison))
        ax.bar(x - 0.18, comparison.var_usd / 1000, 0.36, color="#176b87", label="VaR")
        ax.bar(x + 0.18, comparison.es_usd / 1000, 0.36, color="#e49b2e", label="ES")
        ax.set(
            xticks=x,
            xticklabels=comparison.method,
            title=f"{confidence:.0%} confidence",
        )
        ax.tick_params(axis="x", labelrotation=15)
    axes[0].set_ylabel("One-day loss (USD thousands)")
    axes[0].legend()
    save(fig, "model_comparison.png", "Latest-window risk estimates")

    stress = tables["stress_results"]
    deterministic = stress.loc[stress.result_type == "deterministic_scenario"]
    risk = stress.loc[
        (stress.result_type == "distribution_risk") & (stress.confidence == 0.99)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(
        ["Equity -10%", "Equity -20%", "Synchronized\n-2 sigma"],
        deterministic.deterministic_loss_usd / 1000,
        color="#176b87",
    )
    axes[0].set(title="Deterministic scenario losses", ylabel="USD thousands")
    axes[1].bar(
        ["Baseline", "Volatility x1.5", "High correlation"],
        risk.var_usd / 1000,
        color="#e49b2e",
    )
    axes[1].set(
        title="99% Gaussian VaR under covariance stress", ylabel="USD thousands"
    )
    axes[1].tick_params(axis="x", labelrotation=15)
    save(
        fig,
        "stress_comparison.png",
        "Price scenarios and distribution risk are separate measures",
    )

    spots = np.linspace(60, 140, 121)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, key, label in zip(
        axes.flat,
        ["price", "delta", "gamma", "vega", "theta"],
        [
            "Price (USD/unit)",
            "Delta",
            "Gamma (per USD)",
            "Vega (USD / +1 vol point)",
            "Theta (USD / calendar day)",
        ],
    ):
        for kind, color in [("call", "#176b87"), ("put", "#e49b2e")]:
            values = [
                black_scholes(**{**DEMO_OPTION, "spot": s}, kind=kind)[key]
                for s in spots
            ]
            ax.plot(spots, values, color=color, label=kind)
        ax.set(xlabel="Underlying spot (USD)", ylabel=label)
        ax.grid(alpha=0.15)
    axes[0, 0].legend()
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.04,
        0.8,
        "Illustrative European options\nK=100, T=1 year, r=5%\nAnnual volatility=20%, q=0\n\nIndependent of ETF portfolio VaR",
        va="top",
        linespacing=1.6,
    )
    save(
        fig,
        "option_sensitivities.png",
        "Black-Scholes-Merton prices and Greeks (illustrative inputs)",
        "Assumed parameters; no market option quotes",
    )

    frtb = tables["frtb_comparison"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    metrics = [
        "var99_1day_usd",
        "es975_1day_usd",
        "es975_10day_usd",
        "liquidity_adjusted_es975_usd",
    ]
    x = np.arange(4)
    for offset, calibration, color in [
        (-0.18, "current", "#176b87"),
        (0.18, "stressed", "#a13d4c"),
    ]:
        row = frtb.loc[frtb.calibration == calibration].iloc[0]
        ax.bar(
            x + offset,
            np.array([row[k] for k in metrics]) / 1000,
            0.36,
            label=calibration,
            color=color,
        )
    ax.set(
        xticks=x,
        xticklabels=[
            "99% VaR\n1 day",
            "97.5% ES\n1 day",
            "97.5% ES\n10 days",
            "97.5% LH ES\n10-day base",
        ],
        ylabel="USD thousands",
    )
    ax.legend()
    save(
        fig,
        "frtb_comparison.png",
        "FRTB-style comparison: confidence and horizons differ; not regulatory capital",
    )
    return generated


def write_case_study(
    tables: dict[str, pd.DataFrame], manifest: dict, output: Path
) -> None:
    source = manifest["source"]
    label = (
        "SYNTHETIC EQUITY PROXIES - NOT MARKET OBSERVATIONS"
        if source["synthetic"]
        else source["description"]
    )
    backtest_summary = tables["backtest_summary"]
    full_period_results = backtest_summary.loc[
        backtest_summary.evaluation == "full_oos"
    ]
    historical_95 = full_period_results.loc[
        (full_period_results.method == "historical")
        & (full_period_results.confidence == 0.95)
    ].iloc[0]
    historical_99 = full_period_results.loc[
        (full_period_results.method == "historical")
        & (full_period_results.confidence == 0.99)
    ].iloc[0]
    parametric_99 = full_period_results.loc[
        (full_period_results.method == "parametric")
        & (full_period_results.confidence == 0.99)
    ].iloc[0]
    frtb_comparison = tables["frtb_comparison"]
    stressed = frtb_comparison.loc[frtb_comparison.calibration == "stressed"].iloc[0]
    current = frtb_comparison.loc[frtb_comparison.calibration == "current"].iloc[0]
    status = lambda p: (
        "is unavailable" if pd.isna(p) else "rejects" if p < 0.05 else "does not reject"
    )
    ptext = lambda p: "unavailable" if pd.isna(p) else f"{p:.6g}"
    lines = [
        "# MarketRiskLab - generated case study",
        "",
        f"**Data: {label}.**",
        f"Price period: {manifest['prices']['start']} to {manifest['prices']['end']}; "
        f"{manifest['prices']['observations']:,} prices and {manifest['returns']['observations']:,} returns. "
        f"Reference exposure: USD {manifest['config']['notional']:,.0f}; weights: {manifest['config']['weights']}.",
        "",
        "## Latest-window VaR and ES",
        "",
        markdown_table(
            tables["risk_comparison"][
                [
                    "method",
                    "confidence",
                    "var_usd",
                    "es_usd",
                    "calibration_start",
                    "calibration_end",
                ]
            ]
        ),
        "",
        "![Loss distribution](figures/loss_distribution.png)",
        "![Model comparison](figures/model_comparison.png)",
        "",
        "## Out-of-sample validation",
        "",
        f"Each model has {int(historical_95.forecasts):,} forecasts from {historical_95.start} to {historical_95.end}. "
        "Every forecast uses the preceding 250 observations; the realized return is excluded.",
        "",
        markdown_table(
            full_period_results[
                [
                    "method",
                    "confidence",
                    "forecasts",
                    "exceptions",
                    "exception_rate",
                    "kupiec_p_value",
                    "independence_p_value",
                    "conditional_coverage_p_value",
                ]
            ]
        ),
        "",
        f"Historical 95% VaR has {int(historical_95.exceptions)} exceptions ({historical_95.exception_rate:.2%}; expected 5%). "
        f"Kupiec {status(historical_95.kupiec_p_value)} at 5% significance (p={historical_95.kupiec_p_value:.6g}); "
        f"independence {status(historical_95.independence_p_value)} (p={ptext(historical_95.independence_p_value)}).",
        f"Historical 99% VaR has {int(historical_99.exceptions)} exceptions ({historical_99.exception_rate:.2%}; expected 1%). "
        f"Kupiec {status(historical_99.kupiec_p_value)} (p={historical_99.kupiec_p_value:.6g}). "
        f"Gaussian 99% VaR has {int(parametric_99.exceptions)} exceptions ({parametric_99.exception_rate:.2%}); "
        f"Kupiec {status(parametric_99.kupiec_p_value)} (p={parametric_99.kupiec_p_value:.6g}).",
        "A rejection is a sample-specific model diagnostic. Failure to reject does not prove accuracy. "
        "These tests cannot establish the cause of a breach or distinguish every source of model error.",
        "",
        "![Backtest](figures/rolling_var_exceptions.png)",
        "",
        "### Latest 250 forecasts",
        "",
        markdown_table(
            backtest_summary.loc[
                backtest_summary.evaluation == "latest_250_forecasts",
                [
                    "method",
                    "confidence",
                    "forecasts",
                    "exceptions",
                    "exception_rate",
                    "kupiec_p_value",
                    "independence_p_value",
                ],
            ]
        ),
        "",
        "### Largest realized losses that breached historical 99% VaR",
        "",
        markdown_table(
            tables["exceptions"]
            .loc[
                (tables["exceptions"].method == "historical")
                & (tables["exceptions"].confidence == 0.99)
            ]
            .nlargest(5, "realized_loss_usd")[
                ["date", "realized_loss_usd", "var_usd", "exceedance_usd"]
            ]
        ),
        "",
        "## Stress and derivatives",
        "",
        markdown_table(
            tables["stress_results"][
                [
                    "scenario",
                    "confidence",
                    "deterministic_loss_usd",
                    "var_usd",
                    "es_usd",
                ]
            ]
        ),
        "Blank measures are unavailable by design: deterministic losses are not confidence-level VaR estimates.",
        "",
        "![Stress](figures/stress_comparison.png)",
        "",
        "Options use illustrative inputs rather than market quotes; they are excluded from ETF VaR and capital calculations.",
        "",
        markdown_table(
            tables["option_prices_greeks"][
                ["kind", "price", "delta", "gamma", "vega", "theta"]
            ]
        ),
        "",
        markdown_table(
            tables["option_stress"][
                [
                    "kind",
                    "scenario",
                    "full_repricing_loss_usd",
                    "approximation_error_usd",
                ]
            ]
        ),
        "",
        "![Options](figures/option_sensitivities.png)",
        "",
        "## FRTB-style extension",
        "",
        f"Evaluated {len(tables['stress_window_candidates']):,} candidate 250-observation windows. "
        f"The selected window is {stressed.start} to {stressed.end}. "
        f"Current liquidity-adjusted ES is USD {current.liquidity_adjusted_es975_usd:,.2f}; "
        f"stressed liquidity-adjusted ES and the illustrative capital proxy are USD {stressed.liquidity_adjusted_es975_usd:,.2f}.",
        "The same 250-observation calibration produces 241 overlapping 10-day scenarios. "
        "The stressed window is selected retrospectively and is not used in historical VaR forecasts.",
        "",
        markdown_table(
            frtb_comparison[
                [
                    "calibration",
                    "start",
                    "end",
                    "var99_1day_usd",
                    "es975_1day_usd",
                    "es975_10day_usd",
                    "liquidity_adjusted_es975_usd",
                ]
            ]
        ),
        "",
        "![FRTB-style](figures/frtb_comparison.png)",
        "**This is not full FRTB or regulatory capital.** ETF liquidity mappings are educational proxies. "
        "There is no reduced-factor calibration, regulatory multiplier, NMRF, default-risk charge, desk eligibility or PLA assessment.",
        "",
        "## Main limitations",
        "",
        "- Historical tails are noisy: 99% ES on 250 observations uses only 2.5 observations' probability mass.",
        "- Gaussian parametric and Monte Carlo share normality and constant-window volatility assumptions.",
        "- Adjusted-price snapshots may be revised; this is not a point-in-time vendor archive.",
        "- ETF exposures overlap; there is no look-through, intraday, transaction-cost, FX or liquidity-cost model.",
        "- Constant reference exposures produce hypothetical losses, not an actual trading-account backtest.",
        "- Overlapping 10-day scenarios are dependent; ES and retrospective stress selection have estimation uncertainty.",
        "- Black-Scholes assumes European exercise, constant rates/volatility and continuous diffusion.",
        "",
        "All numerical statements in this report come from the exported tables. "
        "See run_manifest.json for input checksum, source, versions and conventions.",
        "",
    ]
    (output / "case_study.md").write_text("\n\n".join(lines), encoding="utf-8")
