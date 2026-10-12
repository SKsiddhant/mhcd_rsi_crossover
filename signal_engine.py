"""Pure signal / trade-simulation logic for the MACD + RSI analyzer.

Kept free of Streamlit so it can be unit-tested without a browser or network.
"""
import re

import numpy as np
import pandas as pd
import ta

# Extra history fetched before the requested window so MACD(26) / RSI(14) are
# already "warmed up" on the first day the user asked for.
WARMUP_DAYS = 120

_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^_]{0,19}$")


def validate_ticker(raw: str) -> str:
    """Return the cleaned ticker or raise ValueError. Blocks path separators etc."""
    t = (raw or "").strip().upper()
    if not _TICKER_RE.match(t):
        raise ValueError(
            "Ticker may only contain letters, digits and . - ^ = _ (max 20 characters)."
        )
    return t


def safe_filename_part(ticker: str) -> str:
    """Filesystem-safe version of a ticker, e.g. '^NSEI' -> 'NSEI', 'TCS.NS' -> 'TCS_NS'."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", ticker.replace("^", ""))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].squeeze().astype(float)
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd_obj.macd().values
    df["MACD_signal"] = macd_obj.macd_signal().values
    df["MACD_hist"] = macd_obj.macd_diff().values
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi().values
    return df.dropna()


def trim_to_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Keep only the last `years` calendar years (drops the warm-up history)."""
    if df.empty:
        return df
    cutoff = df.index[-1] - pd.DateOffset(years=years)
    return df[df.index >= cutoff]


def detect_signals(df: pd.DataFrame, rsi_buy: float) -> pd.DataFrame:
    """BUY: MACD bullish crossover AND RSI >= rsi_buy.  SELL: MACD bearish crossover."""
    df = df.copy()
    macd = df["MACD"].astype(float)
    sig_ln = df["MACD_signal"].astype(float)
    rsi = df["RSI"].astype(float)

    prev_macd = macd.shift(1)
    prev_sig = sig_ln.shift(1)
    bull_cross = (prev_macd < prev_sig) & (macd >= sig_ln)
    bear_cross = (prev_macd > prev_sig) & (macd <= sig_ln)

    buy_signal = bull_cross & (rsi >= rsi_buy)
    df["signal"] = np.where(buy_signal, "BUY", np.where(bear_cross, "SELL", ""))
    df["bull_cross"] = bull_cross
    df["bear_cross"] = bear_cross
    return df


TRADE_COLUMNS = [
    "Trade #", "Entry Date", "Entry Price", "RSI at Buy", "Exit Date", "Exit Price",
    "RSI at Sell", "Days Held", "Sell Reason", "P&L", "P&L %", "Result",
]


def simulate_trades(df: pd.DataFrame, stop_loss_pct: float = 0.0):
    """Walk the signals and return (trades DataFrame, open position or None).

    Entry and exit both happen at the close of the signal bar. The stop-loss is
    checked against the close too (not the intraday low), so a stop can be
    filled below the stop price in a gap-down.
    """
    trades = []
    position = None

    for idx, row in df.iterrows():
        sig = str(row["signal"]).strip()
        price = float(row["Close"])

        if sig == "BUY" and position is None:
            position = {
                "entry_date": idx.date(),
                "entry_price": price,
                "entry_rsi": round(float(row["RSI"]), 2),
                "entry_macd": round(float(row["MACD"]), 4),
            }
        elif position is not None:
            macd_sell = sig == "SELL"
            stop_hit = stop_loss_pct > 0 and price <= position["entry_price"] * (
                1 - stop_loss_pct / 100
            )
            if macd_sell or stop_hit:
                pnl = price - position["entry_price"]
                pnl_pct = pnl / position["entry_price"] * 100
                if macd_sell and stop_hit:
                    reason = "MACD + Stop-Loss"
                elif stop_hit:
                    reason = f"Stop-Loss ({stop_loss_pct}%)"
                else:
                    reason = "MACD Crossover"
                trades.append({
                    "Trade #": len(trades) + 1,
                    "Entry Date": str(position["entry_date"]),
                    "Entry Price": round(position["entry_price"], 2),
                    "RSI at Buy": position["entry_rsi"],
                    "Exit Date": str(idx.date()),
                    "Exit Price": round(price, 2),
                    "RSI at Sell": round(float(row["RSI"]), 2),
                    "Days Held": (idx.date() - position["entry_date"]).days,
                    "Sell Reason": reason,
                    "P&L": round(pnl, 2),
                    "P&L %": round(pnl_pct, 2),
                    # Win/loss use the same rule everywhere: a flat trade is not a win.
                    "Result": "PROFIT" if pnl > 0 else "LOSS",
                })
                position = None

    return pd.DataFrame(trades, columns=TRADE_COLUMNS), position


def sweep_rsi(df: pd.DataFrame, stop_loss_pct: float, lo: int = 1, hi: int = 100) -> pd.DataFrame:
    """Re-run the strategy for every RSI threshold in [lo, hi]."""
    rows = []
    for rsi_val in range(lo, hi + 1):
        t_df, _ = simulate_trades(detect_signals(df, float(rsi_val)), stop_loss_pct)
        n = len(t_df)
        wins = int((t_df["P&L"] > 0).sum()) if n else 0
        losses = n - wins
        rows.append({
            "RSI Threshold": rsi_val,
            "Trades": n,
            "Wins": wins,
            "Losses": losses,
            "Win Rate %": round(wins / n * 100, 1) if n else 0.0,
            "Net P&L": round(t_df["P&L"].sum(), 2) if n else 0.0,
            "Avg Win": round(t_df.loc[t_df["P&L"] > 0, "P&L"].mean(), 2) if wins else 0.0,
            "Avg Loss": round(t_df.loc[t_df["P&L"] <= 0, "P&L"].mean(), 2) if losses else 0.0,
        })
    return pd.DataFrame(rows)
