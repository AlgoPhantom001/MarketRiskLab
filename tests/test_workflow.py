import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketrisklab import cli
from marketrisklab.data import load_config, synthetic_prices
from marketrisklab.reporting import markdown_table


def test_offline_workflow_reconciles_and_reproduces(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[1] / "config.json")
    config["snapshot_draws"] = 2000
    config["rolling_draws"] = 2000
    # Small but >500-price fixture exercises latest-250 reporting too.
    monkeypatch.setattr(cli, "synthetic_prices", lambda c: synthetic_prices(c, 511))
    one = cli.run(config, "synthetic", tmp_path / "data", tmp_path / "first")
    two = cli.run(config, "synthetic", tmp_path / "data", tmp_path / "second")
    assert one["source"]["synthetic"]
    assert one["output_sha256"] == two["output_sha256"]
    assert one["input_sha256"] == two["input_sha256"]
    assert len(one["figures"]) == 6
    for name in one["figures"]:
        assert (tmp_path / "first" / name).stat().st_size > 1000
    root = tmp_path / "first"
    daily = pd.read_csv(root / "backtest_daily.csv")
    summary = pd.read_csv(root / "backtest_summary.csv")
    exceptions = pd.read_csv(root / "exceptions.csv")
    assert len(exceptions) == int(daily.exception.sum())
    for row in summary.itertuples():
        sample = daily.loc[
            (daily.method == row.method) & (daily.confidence == row.confidence)
        ]
        if row.evaluation == "latest_250_forecasts":
            sample = sample.iloc[-250:]
        assert row.exceptions == int(sample.exception.sum())
        assert row.exception_rate == pytest.approx(sample.exception.mean(), abs=1e-12)
    risk = pd.read_csv(root / "risk_comparison.csv")
    np.testing.assert_allclose(
        risk.var_usd / config["notional"] * 100, risk.var_pct, rtol=1e-10
    )
    frtb = pd.read_csv(root / "frtb_comparison.csv")
    buckets = pd.read_csv(root / "liquidity_buckets.csv")
    for row in frtb.itertuples():
        b = buckets.loc[buckets.calibration == row.calibration]
        assert row.liquidity_adjusted_es975_usd == pytest.approx(
            np.sqrt(b.squared_contribution.sum()), rel=1e-10
        )
    manifest = json.loads((root / "run_manifest.json").read_text())
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    report = (root / "case_study.md").read_text(encoding="utf-8")
    assert "SYNTHETIC EQUITY PROXIES" in report
    assert "not full FRTB" in report
    assert not (root / "resume_evidence.md").exists()


def test_market_mode_rejects_missing_cache_and_checksum_mismatch(tmp_path):
    config = load_config(Path(__file__).parents[1] / "config.json")
    data = tmp_path / "data"
    with pytest.raises(ValueError, match="fetch"):
        cli.run(config, "market", data, tmp_path / "out")
    data.mkdir()
    synthetic_prices(config, 511).to_csv(data / "market_prices.csv")
    (data / "source.json").write_text(
        json.dumps(
            {"synthetic": False, "description": "test", "checksum_sha256": "wrong"}
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        cli.run(config, "market", data, tmp_path / "out")


def test_unavailable_table_values_are_explicit():
    assert "unavailable" in markdown_table(pd.DataFrame({"p_value": [np.nan]}))
