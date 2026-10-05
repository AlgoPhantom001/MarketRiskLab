import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketrisklab.data import (
    load_config,
    read_prices,
    simple_returns,
    synthetic_prices,
    validate_prices,
)


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "config.json")


def test_seeded_fixture_and_returns(config, tmp_path):
    first = synthetic_prices(config)
    pd.testing.assert_frame_equal(first, synthetic_prices(config))
    path = tmp_path / "prices.csv"
    first.to_csv(path)
    read = read_prices(path, config["tickers"])
    np.testing.assert_allclose(read, first)
    returns = simple_returns(first)
    np.testing.assert_allclose(
        (1 + returns).prod().to_numpy(), (first.iloc[-1] / first.iloc[0]).to_numpy()
    )


@pytest.mark.parametrize(
    "issue",
    [
        "duplicate",
        "descending",
        "nan",
        "zero",
        "negative",
        "infinity",
        "columns",
        "short",
    ],
)
def test_bad_prices_rejected(config, issue):
    frame = synthetic_prices(config)
    if issue == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0]] + list(frame.index[:-1]))
    elif issue == "descending":
        frame = frame.iloc[::-1]
    elif issue == "columns":
        frame = frame.iloc[:, ::-1]
    elif issue == "short":
        frame = frame.iloc[:250]
    else:
        frame.iloc[10, 0] = {
            "nan": np.nan,
            "zero": 0.0,
            "negative": -1.0,
            "infinity": np.inf,
        }[issue]
    with pytest.raises(ValueError):
        validate_prices(frame, config["tickers"])


def test_bad_weights_rejected(config, tmp_path):
    config["weights"] = [0.4, 0.4, 0.4]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)
