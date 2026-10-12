# MACD + RSI Stock Signal Analyzer

A Streamlit app that back-tests a simple long-only strategy on daily Yahoo Finance data.

| | Rule |
|---|---|
| **Buy** | MACD(12,26,9) crosses above its signal line **and** RSI(14) ≥ your threshold |
| **Sell** | MACD crosses below its signal line **or** the close is ≥ *stop-loss %* below the entry price |

It also sweeps every RSI threshold from 1 to 100 so you can see how sensitive the result is.

## Run

```bash
pip install -r requirements.txt
streamlit run stock_signal_app.py
```

Indian stocks need a suffix (`RELIANCE.NS`, `RELIANCE.BO`); US stocks do not (`AAPL`).

## How the numbers are produced

* Trades enter and exit at the **close of the signal day**; the stop-loss is also checked against the close, so a gap-down can fill below the stop price.
* No fees, taxes, slippage or position sizing: P&L is **per share**, long only, one position at a time.
* Prices are dividend/split adjusted (`auto_adjust=True`). Today's unfinished bar is dropped.
* Indicators are calculated on extra history before the chosen window, so the first day of the window already has valid MACD/RSI values.
* The "best RSI threshold" in the sweep is picked **in-sample on the same data** — it is a description of the past, not evidence the setting will work going forward.

Each run also writes `stock_signals_data/*_{summary.json,trades.csv,indicators.csv}` (git-ignored; override the folder with `SIGNALS_DATA_DIR`).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The strategy logic lives in `signal_engine.py`; the tests use synthetic prices and a fake `yfinance`, so they need no network.

Not financial advice.
