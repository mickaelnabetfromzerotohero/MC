import json

import pandas as pd

from mcrisk.cli import main
from mcrisk.data import demo_params, load_prices_csv, synthetic_prices


def test_demo_run_writes_summary_and_charts(tmp_path, capsys):
    code = main(["--demo", "--sims", "2000", "--out", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Value at risk (95%)" in out
    assert "Expected shortfall" in out
    for name in ("summary.json", "1_inputs.png", "2_futures.png", "3_var_es.png", "4_correlation.png"):
        assert (tmp_path / name).stat().st_size > 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["tickers"] == ["NVDA", "AAPL", "TSLA"]
    assert summary["moving_together"]["n_sims"] == 2000
    assert summary["var_ratio"] > 1


def test_csv_run_with_custom_weights(tmp_path, capsys):
    csv = tmp_path / "prices.csv"
    synthetic_prices(demo_params(), days=600, seed=9).to_csv(csv)
    code = main(["--csv", str(csv), "--weights", "2", "1", "1", "--sims", "1000",
                 "--out", str(tmp_path / "out"), "--no-plots"])
    assert code == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["weights"] == [0.5, 0.25, 0.25]
    assert summary["history"]["days"] == 601
    assert not (tmp_path / "out" / "2_futures.png").exists()


def test_csv_column_selection(tmp_path):
    csv = tmp_path / "prices.csv"
    synthetic_prices(demo_params(), days=100, seed=9).to_csv(csv)
    prices = load_prices_csv(csv, ["TSLA", "NVDA"])
    assert list(prices.columns) == ["TSLA", "NVDA"]
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_bad_csv_returns_an_error_code(tmp_path, capsys):
    csv = tmp_path / "prices.csv"
    csv.write_text("date,A\n2024-01-01,1\n2024-01-02,2\n")
    assert main(["--csv", str(csv), "--out", str(tmp_path)]) == 1
    assert "error" in capsys.readouterr().out


def test_download_parses_yfinance_columns(monkeypatch):
    """yfinance returns (Price, Ticker) MultiIndex columns; only Close is kept, in ticker order."""
    import sys
    import types

    import numpy as np

    from mcrisk import data

    idx = pd.bdate_range("2024-01-01", periods=60)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAPL", "NVDA"]], names=["Price", "Ticker"])
    frame = pd.DataFrame(np.linspace(100, 120, 60)[:, None].repeat(4, axis=1), index=idx, columns=cols)
    calls = {}

    def fake_download(tickers, **kwargs):
        calls.update(kwargs, tickers=tickers)
        return frame

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    prices = data.download_prices(["NVDA", "AAPL"], years=1, end="2024-03-22")
    assert list(prices.columns) == ["NVDA", "AAPL"]
    assert len(prices) == 60
    assert calls["start"] == "2023-03-22" and calls["auto_adjust"] is True
