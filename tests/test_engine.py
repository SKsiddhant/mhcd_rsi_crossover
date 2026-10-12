import numpy as np
import pandas as pd
import pytest

from signal_engine import (
    add_indicators, detect_signals, safe_filename_part, simulate_trades,
    sweep_rsi, trim_to_years, validate_ticker,
)


def make_prices(n=600, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.1, 2, n)) + 10 * np.sin(np.arange(n) / 15)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000},
        index=idx,
    )


def manual_frame(closes, signals):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Close": closes, "RSI": 60.0, "MACD": 0.0, "signal": signals}, index=idx)


def test_validate_ticker_accepts_real_symbols():
    for t in ["aapl", "RELIANCE.NS", "^NSEI", "BRK-B", "EURUSD=X"]:
        assert validate_ticker(t) == t.upper()


@pytest.mark.parametrize("bad", ["", "  ", "../../etc/passwd", "a/b", "A B", "x" * 30, "A;B", "<script>"])
def test_validate_ticker_rejects_unsafe_values(bad):
    with pytest.raises(ValueError):
        validate_ticker(bad)


def test_safe_filename_part_has_no_separators():
    assert safe_filename_part("^NSEI") == "NSEI"
    assert safe_filename_part("TCS.NS") == "TCS_NS"
    assert "/" not in safe_filename_part("a/b\\c")


def test_trim_to_years_really_returns_requested_window():
    df = make_prices(1500)
    three = trim_to_years(df, 3)
    five = trim_to_years(df, 5)
    assert (three.index[-1] - three.index[0]).days <= 3 * 366
    assert len(three) < len(five)  # 3 years used to silently fetch 5 years


def test_indicators_and_signals_shape():
    df = detect_signals(add_indicators(make_prices()), 50)
    assert {"MACD", "MACD_signal", "RSI", "signal"} <= set(df.columns)
    assert not df[["MACD", "RSI"]].isna().any().any()
    assert set(df["signal"].unique()) <= {"BUY", "SELL", ""}
    # a bullish and a bearish crossover can never occur on the same bar
    assert not (df["bull_cross"] & df["bear_cross"]).any()


def test_higher_rsi_threshold_never_adds_buys():
    base = add_indicators(make_prices())
    low = (detect_signals(base, 30)["signal"] == "BUY").sum()
    high = (detect_signals(base, 80)["signal"] == "BUY").sum()
    assert high <= low


def test_simulate_macd_exit_and_open_position():
    df = manual_frame([10, 11, 12, 11, 15, 16], ["BUY", "", "SELL", "BUY", "", ""])
    trades, open_pos = simulate_trades(df, 0)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert (t["Entry Price"], t["Exit Price"], t["P&L"], t["Result"]) == (10.0, 12.0, 2.0, "PROFIT")
    assert t["Sell Reason"] == "MACD Crossover"
    assert open_pos["entry_price"] == 11


def test_stop_loss_exits_before_macd():
    df = manual_frame([100, 99, 94, 120], ["BUY", "", "", "SELL"])
    trades, open_pos = simulate_trades(df, 5)
    assert len(trades) == 1 and open_pos is None
    assert trades.iloc[0]["Exit Price"] == 94
    assert trades.iloc[0]["Sell Reason"].startswith("Stop-Loss")


def test_stop_loss_disabled_with_zero():
    df = manual_frame([100, 50, 120], ["BUY", "", "SELL"])
    trades, _ = simulate_trades(df, 0)
    assert trades.iloc[0]["Exit Price"] == 120


def test_flat_trade_counts_as_loss_consistently():
    df = manual_frame([100, 100], ["BUY", "SELL"])
    trades, _ = simulate_trades(df, 0)
    assert trades.iloc[0]["Result"] == "LOSS"


def test_no_trades_returns_typed_empty_frame():
    trades, open_pos = simulate_trades(manual_frame([1, 2], ["", ""]), 0)
    assert trades.empty and open_pos is None
    assert "P&L" in trades.columns


def test_sweep_covers_every_threshold_and_is_consistent():
    df = detect_signals(add_indicators(make_prices()), 50)
    sweep = sweep_rsi(df, 5)
    assert list(sweep["RSI Threshold"]) == list(range(1, 101))
    assert (sweep["Wins"] + sweep["Losses"] == sweep["Trades"]).all()
    # sweep row for the same threshold must equal a direct run
    trades, _ = simulate_trades(detect_signals(df, 50.0), 5)
    row = sweep[sweep["RSI Threshold"] == 50].iloc[0]
    assert row["Trades"] == len(trades)
    assert row["Net P&L"] == pytest.approx(round(trades["P&L"].sum(), 2))
