"""Drive the real Streamlit script with fake market data (no network)."""
import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.test_engine import make_prices

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_signal_app.py")


@pytest.fixture
def app(monkeypatch, tmp_path):
    import yfinance as yf

    calls = []

    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, **kw):
            calls.append((self.sym, kw))
            if self.sym == "NOPE":
                return pd.DataFrame()
            return make_prices(1500).tz_localize("UTC")

    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setenv("SIGNALS_DATA_DIR", str(tmp_path / "out"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.calls = calls
    at.out = tmp_path / "out"
    return at


def run_with(at, ticker):
    at.run()
    at.sidebar.text_input[0].set_value(ticker)
    at.sidebar.button[0].click()
    at.run()
    return at


def test_happy_path_renders_and_saves_files(app):
    at = run_with(app, "aapl")
    assert not at.exception, at.exception
    assert any("Complete Trade History" in m.value for m in at.markdown)
    assert any(f.name.startswith("AAPL_") for f in app.out.iterdir())
    assert "start" in app.calls[0][1]  # explicit window, not the old period_map


def test_results_survive_widget_interaction(app):
    at = run_with(app, "AAPL")
    sort_box = next(b for b in at.selectbox if "Sort table by" in b.label)
    sort_box.select("Win Rate %").run()
    assert not at.exception, at.exception
    assert any("Complete Trade History" in m.value for m in at.markdown)


def test_unknown_ticker_shows_error_not_traceback(app):
    at = run_with(app, "NOPE")
    assert not at.exception
    assert any("No data found" in e.value for e in at.error)


def test_path_traversal_ticker_rejected_and_nothing_written(app):
    at = run_with(app, "../../evil")
    assert not at.exception
    assert any("Ticker may only contain" in e.value for e in at.error)
    assert not app.out.exists() or not any(app.out.iterdir())
    assert app.calls == []
