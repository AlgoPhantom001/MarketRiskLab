"""Reproducible fetch/run commands; offline mode never silently changes source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from .data import (
    download_prices,
    load_config,
    read_prices,
    simple_returns,
    synthetic_prices,
)
from .risk import compare_models
from .backtesting import rolling_backtest, summarize_backtest
from .stress import stress_table
from .options import option_tables
from .frtb import frtb_tables
from .reporting import plot_outputs, write_case_study


def fetch(config: dict, data_dir: Path) -> Path:
    prices = download_prices(config, data_dir.parent / ".yfinance-cache")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "market_prices.csv"
    prices.to_csv(path, float_format="%.12g")
    source = dict(
        description="Yahoo Finance adjusted prices via yfinance",
        synthetic=False,
        provider="Yahoo Finance",
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        auto_adjust=True,
        start_requested=config["start"],
        end_exclusive=config["end_exclusive"],
        tickers=config["tickers"],
        currency=config["currency"],
        checksum_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (data_dir / "source.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    return path


def _load_prices_and_source(config, data_mode, data_dir, csv_path):
    """Load the chosen input and validate its provenance before calculations."""
    if data_mode == "synthetic":
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "synthetic_prices.csv"
        synthetic_prices(config).to_csv(path, float_format="%.12g")
        source = dict(
            description="Seeded multivariate Student-t equity proxies; NOT actual ETF observations",
            synthetic=True,
            provider="MarketRiskLab fixture",
            seed=config["seed"],
        )
    elif data_mode == "csv":
        if csv_path is None:
            raise ValueError("--prices is required for CSV mode")
        path = csv_path
        source = dict(
            description="User CSV; external provider provenance not verified",
            synthetic=False,
            provider="user_csv",
        )
    else:
        path = data_dir / "market_prices.csv"
        metadata = data_dir / "source.json"
        if not path.exists() or not metadata.exists():
            raise ValueError(
                "Run 'marketrisklab fetch' first, or choose --data synthetic explicitly"
            )
        source = json.loads(metadata.read_text(encoding="utf-8"))
        source.setdefault(
            "description", source.get("source", "Cached market-price snapshot")
        )
        if source.get("synthetic"):
            raise ValueError("Market mode cannot use a synthetic source record")
        if source.get("tickers", config["tickers"]) != config["tickers"]:
            raise ValueError("Cached source tickers do not match configuration")
        recorded = source.get("checksum_sha256")
        if recorded and recorded != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Cached price checksum does not match source metadata")
    prices = read_prices(path, config["tickers"])
    return prices, path, source


def _build_run_manifest(config, prices, returns, path, source):
    """Record inputs, conventions and dependency versions for reproducibility."""
    manifest = dict(
        project="MarketRiskLab",
        project_version=version("marketrisklab"),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        source=source,
        input_file=path.name,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config=config,
        python=platform.python_version(),
        versions={
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "matplotlib", "yfinance")
        },
        prices=dict(
            observations=len(prices),
            start=str(prices.index[0].date()),
            end=str(prices.index[-1].date()),
        ),
        returns=dict(
            observations=len(returns),
            start=str(returns.index[0].date()),
            end=str(returns.index[-1].date()),
        ),
        conventions=dict(
            loss="minus_constant_notional_times_weighted_simple_return",
            quantile="inverted_cdf",
            es="exact_fractional_tail_mass",
            sample_covariance_ddof=1,
            forecast_information="250 observations strictly before forecast date",
            option_volatility="annual_decimal",
            vega="per_0.01_volatility",
            theta="per_calendar_day",
            frtb="educational_proxy_not_regulatory_compliance",
        ),
    )
    return manifest


def run(
    config: dict,
    data_mode: str,
    data_dir: Path,
    output: Path,
    csv_path: Path | None = None,
) -> dict:
    # 1. Load validated prices and calculate daily simple returns.
    prices, path, source = _load_prices_and_source(
        config, data_mode, data_dir, csv_path
    )
    returns = simple_returns(prices)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Data: {source['description']}; {len(prices):,} prices")
    # 2. Calculate risk forecasts, option analytics and FRTB-style measures.
    daily = rolling_backtest(returns, config)
    options, option_stress = option_tables()
    frtb, buckets, candidates = frtb_tables(returns, config)
    tables = {
        "risk_comparison": compare_models(returns, config),
        "backtest_daily": daily,
        "backtest_summary": summarize_backtest(daily),
        "exceptions": daily.loc[daily.exception].copy(),
        "stress_results": stress_table(returns, config),
        "option_prices_greeks": options,
        "option_stress": option_stress,
        "frtb_comparison": frtb,
        "liquidity_buckets": buckets,
        "stress_window_candidates": candidates,
    }
    # 3. Export the ten result tables with consistent precision.
    for name, frame in tables.items():
        frame.to_csv(output / (name + ".csv"), index=False, float_format="%.12g")
    # 4. Save plots, provenance and a short case study.
    manifest = _build_run_manifest(config, prices, returns, path, source)
    if source["synthetic"]:
        source_label = "SYNTHETIC PROXIES - NOT MARKET DATA"
    elif data_mode == "market":
        source_label = "Adjusted-price ETF snapshot"
    else:
        source_label = "User-supplied CSV"
    manifest["figures"] = plot_outputs(returns, config, tables, output, source_label)
    manifest["output_sha256"] = {}
    for table_name in tables:
        filename = table_name + ".csv"
        manifest["output_sha256"][filename] = hashlib.sha256(
            (output / filename).read_bytes()
        ).hexdigest()
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_case_study(tables, manifest, output)
    print(
        f"Completed: {len(tables)} tables, {len(manifest['figures'])} plots and case study -> {output}"
    )
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="MarketRiskLab educational market-risk methodology engine"
    )
    parser.add_argument("command", choices=["fetch", "run"])
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--data", choices=["synthetic", "market", "csv"], default="synthetic"
    )
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "fetch":
            path = fetch(config, args.data_dir)
            print(f"Validated market snapshot saved to {path}")
        else:
            output = args.output or Path("reports") / args.data
            if args.prices and args.data != "csv":
                raise ValueError("--prices requires --data csv")
            run(config, args.data, args.data_dir, output, args.prices)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"MarketRiskLab: {error}\n")


if __name__ == "__main__":
    main()
