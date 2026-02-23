"""
╔══════════════════════════════════════════════════════════════╗
║        MACD + RSI  Stock Signal Analyzer  —  Streamlit       ║
║                                                              ║
║  BUY  → MACD bullish crossover  AND  RSI ≥ user threshold   ║
║  SELL → MACD bearish crossover                               ║
║      OR price drops ≥ user stop-loss % below entry          ║
║                                                              ║
║  Run:  streamlit run stock_signal_app.py                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, json
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MACD + RSI Signal Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"]          { background-color: #161b22; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 10px;
        text-align: center;
    }
    .metric-card.green  { border-left-color: #00c853; }
    .metric-card.red    { border-left-color: #ff1744; }
    .metric-card.yellow { border-left-color: #ffd600; }
    .metric-card.purple { border-left-color: #ce93d8; }

    .metric-label { font-size: 11px; color: #8899aa; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.8px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #e8eaf6; margin-top: 6px; }

    .open-position-box {
        background: linear-gradient(135deg, #2a2000, #332a00);
        border: 1px solid #ffd600;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 14px 0;
    }
    .logic-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
        font-size: 13px;
        line-height: 1.8;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f8ef7, #7c4dff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        font-weight: 600;
        margin-top: 8px;
        cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.86; }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
DATA_DIR = "stock_signals_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Data Fetching ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, years: int) -> pd.DataFrame:
    try:
        period_map = {1: "1y", 2: "2y", 3: "5y", 5: "5y", 10: "10y"}
        period = period_map.get(years, "5y")

        t  = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df   = df[keep].copy()
        df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Only keep up to yesterday's close
        yesterday = (datetime.today() - timedelta(days=1)).date()
        df = df[df.index.date <= yesterday]

        return df

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# ── Indicator Calculation ────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Always squeeze + .values to ensure plain 1-D numpy arrays — fixes yfinance multi-index issues
    close = df["Close"].squeeze()

    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd_obj.macd().squeeze().values
    df["MACD_signal"] = macd_obj.macd_signal().squeeze().values
    df["MACD_hist"]   = macd_obj.macd_diff().squeeze().values
    df["RSI"]         = ta.momentum.RSIIndicator(close, window=14).rsi().squeeze().values

    return df.dropna()

# ── Signal Detection ─────────────────────────────────────────
def detect_signals(df: pd.DataFrame, rsi_buy: float) -> pd.DataFrame:
    """
    BUY  : MACD bullish crossover  AND  RSI >= rsi_buy
    SELL : MACD bearish crossover  (RSI is completely ignored)
    """
    import numpy as np

    df = df.copy()

    # Cast explicitly to float Series — prevents silent DataFrame wrapping bugs
    macd   = df["MACD"].astype(float)
    sig_ln = df["MACD_signal"].astype(float)
    rsi    = df["RSI"].astype(float)

    prev_macd = macd.shift(1)
    prev_sig  = sig_ln.shift(1)

    # Bullish: was below signal yesterday, at/above today
    bull_cross = (prev_macd < prev_sig) & (macd >= sig_ln)
    # Bearish: was above signal yesterday, at/below today
    bear_cross = (prev_macd > prev_sig) & (macd <= sig_ln)

    buy_signal  = bull_cross & (rsi >= rsi_buy)
    sell_signal = bear_cross   # RSI completely ignored for SELL

    # numpy.where avoids pandas object-dtype .loc assignment bugs entirely
    df["signal"]     = np.where(buy_signal, "BUY", np.where(sell_signal, "SELL", ""))
    df["bull_cross"] = bull_cross
    df["bear_cross"] = bear_cross

    return df

# ── Trade Simulation ─────────────────────────────────────────
def simulate_trades(df: pd.DataFrame, stop_loss_pct: float = 0.0):
    """
    BUY  : MACD bullish crossover AND RSI >= threshold
    SELL : MACD bearish crossover  OR  price drops >= stop_loss_pct% below entry
           (whichever triggers first — OR condition)
    """
    trades   = []
    position = None

    for idx, row in df.iterrows():
        sig   = str(row["signal"]).strip()
        price = float(row["Close"])

        if sig == "BUY" and position is None:
            position = {
                "entry_date"  : idx.date(),
                "entry_price" : price,
                "entry_rsi"   : round(float(row["RSI"]), 2),
                "entry_macd"  : round(float(row["MACD"]), 4),
            }

        elif position is not None:
            # Condition 1: MACD bearish crossover
            macd_sell = (sig == "SELL")

            # Condition 2: Stop-loss — price dropped >= stop_loss_pct% below entry
            stop_loss_triggered = (
                stop_loss_pct > 0
                and price <= position["entry_price"] * (1 - stop_loss_pct / 100)
            )

            if macd_sell or stop_loss_triggered:
                pnl     = price - position["entry_price"]
                pnl_pct = (pnl / position["entry_price"]) * 100

                if macd_sell and stop_loss_triggered:
                    sell_reason = "MACD + Stop-Loss"
                elif stop_loss_triggered:
                    sell_reason = f"Stop-Loss ({stop_loss_pct}%)"
                else:
                    sell_reason = "MACD Crossover"

                days_held = (idx.date() - position["entry_date"]).days

                trades.append({
                    "Trade #"    : len(trades) + 1,
                    "Entry Date" : str(position["entry_date"]),
                    "Entry Price": round(position["entry_price"], 2),
                    "RSI at Buy" : position["entry_rsi"],
                    "Exit Date"  : str(idx.date()),
                    "Exit Price" : round(price, 2),
                    "RSI at Sell": round(float(row["RSI"]), 2),
                    "Days Held"  : days_held,
                    "Sell Reason": sell_reason,
                    "P&L"        : round(pnl, 2),
                    "P&L %"      : round(pnl_pct, 2),
                    "Result"     : "PROFIT" if pnl >= 0 else "LOSS",
                })
                position = None

    return pd.DataFrame(trades), position

# ── RSI Sweep ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def rsi_sweep(df_with_indicators: pd.DataFrame, stop_loss_pct: float) -> pd.DataFrame:
    """Run RSI threshold from 1 to 100 and collect stats for each."""
    rows = []
    for rsi_val in range(1, 101):
        swept    = detect_signals(df_with_indicators.copy(), float(rsi_val))
        t_df, _  = simulate_trades(swept, stop_loss_pct)

        n        = len(t_df)
        wins     = int((t_df["P&L"] > 0).sum()) if n else 0
        losses   = n - wins
        net      = round(t_df["P&L"].sum(), 2)      if n else 0.0
        wr       = round(wins / n * 100, 1)          if n else 0.0
        avg_w    = round(t_df.loc[t_df["P&L"] > 0, "P&L"].mean(), 2) if wins   else 0.0
        avg_l    = round(t_df.loc[t_df["P&L"] < 0, "P&L"].mean(), 2) if losses else 0.0

        rows.append({
            "RSI Threshold" : rsi_val,
            "Trades"        : n,
            "Wins"          : wins,
            "Losses"        : losses,
            "Win Rate %"    : wr,
            "Net P&L"       : net,
            "Avg Win"       : avg_w,
            "Avg Loss"      : avg_l,
        })
    return pd.DataFrame(rows)
def save_results(ticker, rsi_buy, stop_loss_pct, trades_df, open_pos, df):
    yesterday = str(df.index[-1].date())
    safe      = ticker.replace(".", "_").replace("^", "")
    base      = os.path.join(DATA_DIR, f"{safe}_{yesterday}")

    total   = len(trades_df)
    winners = int((trades_df["P&L"] > 0).sum()) if total else 0

    sl_desc = f"stop-loss at -{stop_loss_pct}% below entry" if stop_loss_pct > 0 else "disabled"
    summary = {
        "ticker"         : ticker,
        "data_as_of"     : yesterday,
        "rsi_buy_thresh" : rsi_buy,
        "sell_condition" : f"MACD bearish crossover OR {sl_desc}",
        "total_trades"   : total,
        "profitable"     : winners,
        "loss_trades"    : total - winners,
        "net_pnl"        : round(trades_df["P&L"].sum(), 2) if total else 0,
        "open_position"  : open_pos,
        "generated_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(base + "_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)

    if not trades_df.empty:
        trades_df.to_csv(base + "_trades.csv", index=False)

    df[["Open", "High", "Low", "Close", "Volume",
        "MACD", "MACD_signal", "MACD_hist", "RSI", "signal"]].to_csv(
        base + "_indicators.csv", index_label="Date")

    return base

# ── Interactive Chart ─────────────────────────────────────────
def build_chart(df: pd.DataFrame, ticker: str, rsi_buy: float) -> go.Figure:
    buys  = df[df["signal"].astype(str) == "BUY"]
    sells = df[df["signal"].astype(str) == "SELL"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.025,
        subplot_titles=(
            f"  {ticker} — Price  ▲ BUY  ▼ SELL",
            "  MACD  (12, 26, 9)",
            "  RSI  (14)",
        ),
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a",  decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",   decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # BUY markers
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Low"] * 0.993,
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=13,
                    color="#00e676", line=dict(width=1, color="#fff")),
        text=["B"] * len(buys), textposition="bottom center",
        textfont=dict(color="#00e676", size=9),
        name="BUY",
    ), row=1, col=1)

    # SELL markers
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["High"] * 1.007,
        mode="markers+text",
        marker=dict(symbol="triangle-down", size=13,
                    color="#ff5252", line=dict(width=1, color="#fff")),
        text=["S"] * len(sells), textposition="top center",
        textfont=dict(color="#ff5252", size=9),
        name="SELL",
    ), row=1, col=1)

    # MACD histogram
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_hist"],
        name="MACD Hist", marker_color=hist_colors, opacity=0.55,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        line=dict(color="#4f8ef7", width=1.6), name="MACD",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_signal"],
        line=dict(color="#ff9800", width=1.6, dash="dot"), name="Signal Line",
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        line=dict(color="#ce93d8", width=1.6), name="RSI",
        fill="tozeroy", fillcolor="rgba(206,147,216,0.07)",
    ), row=3, col=1)

    for level, color, label in [
        (rsi_buy, "#00e676", f"Buy RSI {int(rsi_buy)}"),
        (70,      "#ff5252", "Overbought 70"),
        (30,      "#26a69a", "Oversold 30"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                      opacity=0.55, row=3, col=1,
                      annotation_text=label,
                      annotation_position="right",
                      annotation_font_color=color,
                      annotation_font_size=10)

    fig.update_layout(
        height=800,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#c9d1d9", family="Inter, sans-serif"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, x=0, y=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=80, t=50, b=10),
        hovermode="x unified",
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1e2130", zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor="#1e2130", zeroline=False, row=i, col=1)

    return fig

# ── Metric Card Helper ────────────────────────────────────────
def metric_card(col, label: str, value, css_class: str = ""):
    col.markdown(f"""
    <div class="metric-card {css_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Signal Settings")
    st.markdown("---")

    ticker_input = st.text_input(
        "🔍 Stock Ticker Symbol",
        placeholder="e.g. RELIANCE.NS  /  AAPL  /  TCS.NS",
        help="Indian stocks: add .NS (NSE) or .BO (BSE)\nUS stocks: no suffix needed",
    ).strip().upper()

    rsi_buy = st.slider(
        "📊 RSI Threshold for BUY",
        min_value=40, max_value=85, value=60, step=1,
        help="BUY signal fires only when RSI is at or above this value at the crossover",
    )

    stop_loss_pct = st.slider(
        "🛑 Stop-Loss % Below Entry Price",
        min_value=0, max_value=30, value=5, step=1,
        help="SELL triggers if price drops this % below your buy price. Set 0 to disable.",
        format="%d%%",
    )

    lookback = st.selectbox(
        "📅 Historical Data Range",
        options=[1, 2, 3, 5, 10],
        index=3,
        format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}",
    )

    st.markdown("---")
    sl_label = f"Price drops ≥ {stop_loss_pct}% below entry" if stop_loss_pct > 0 else "Stop-loss: disabled (set to 0%)"
    st.markdown(f"""
    <div class="logic-box">
    <b>🟢 BUY Signal</b><br>
    MACD line crosses <b>above</b> Signal line<br>
    <i>+ RSI must be ≥ {rsi_buy}</i>
    <br><br>
    <b>🔴 SELL Signal</b> &nbsp;<span style="color:#ff9800">(OR condition)</span><br>
    ① MACD line crosses <b>below</b> Signal line<br>
    ② {sl_label}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("💾 Results auto-saved to `stock_signals_data/`")
    run = st.button("🚀  Run Analysis", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
st.markdown("# 📈 MACD + RSI Stock Signal Analyzer")
st.caption("Yahoo Finance data  ·  MACD (12,26,9)  ·  RSI (14)  ·  Data up to yesterday's close")

if not run:
    st.info("👈  Enter a **stock ticker** in the sidebar, set your **RSI threshold**, then click **Run Analysis**.")
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **🟢 BUY Condition**
        - MACD line crosses **above** the Signal line (bullish crossover)
        - AND RSI must be **≥ your threshold** (e.g. 60 or 70)
        - Both conditions must be true at the same time
        """)
    with col2:
        st.error("""
        **🔴 SELL Condition  (OR)**
        - ① MACD line crosses **below** the Signal line (bearish crossover)
        - ② Price drops ≥ your **stop-loss %** below entry price
        - **Either** condition alone is enough to exit the trade
        """)
    st.stop()

if not ticker_input:
    st.error("⚠️ Please enter a stock ticker symbol in the sidebar.")
    st.stop()

# ── Fetch ─────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker_input}** …"):
    raw_df = fetch_data(ticker_input, lookback)

if raw_df is None or raw_df.empty:
    st.error(f"""
    ❌ No data found for **{ticker_input}**.

    **Common fixes:**
    - Indian NSE stocks → `RELIANCE.NS` / `TCS.NS` / `INFY.NS` / `HDFCBANK.NS`
    - Indian BSE stocks → `RELIANCE.BO`
    - US stocks → `AAPL` / `TSLA` / `MSFT` / `GOOGL`
    - Verify on finance.yahoo.com that the ticker is correct
    """)
    st.stop()

# ── Process ───────────────────────────────────────────────────
df                  = add_indicators(raw_df.copy())
df                  = detect_signals(df, rsi_buy)
trades_df, open_pos = simulate_trades(df, stop_loss_pct)
save_results(ticker_input, rsi_buy, stop_loss_pct, trades_df, open_pos, df)

# ── Derived stats ─────────────────────────────────────────────
yesterday = df.index[-1].date()
ltp       = round(float(df["Close"].iloc[-1]), 2)
total     = len(trades_df)
winners   = int((trades_df["P&L"] > 0).sum()) if total else 0
losers    = total - winners
net_pnl   = round(trades_df["P&L"].sum(), 2) if total else 0
win_rate  = round((winners / total) * 100, 1) if total else 0
avg_win   = round(trades_df.loc[trades_df["P&L"] > 0, "P&L"].mean(), 2) if winners else 0
avg_loss  = round(trades_df.loc[trades_df["P&L"] < 0, "P&L"].mean(), 2) if losers  else 0
best      = round(trades_df["P&L"].max(), 2) if total else 0
worst     = round(trades_df["P&L"].min(), 2) if total else 0

# ── Header ────────────────────────────────────────────────────
st.markdown(f"### {ticker_input} &nbsp;·&nbsp; Data as of `{yesterday}` &nbsp;·&nbsp; LTP: `{ltp}`")

# ── Metric Cards Row ──────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
metric_card(c1, "Last Close",      f"{ltp}",                    "")
metric_card(c2, "Total Trades",    total,                        "purple")
metric_card(c3, "Profitable",      winners,                      "green")
metric_card(c4, "Loss Trades",     losers,                       "red")
metric_card(c5, "Win Rate",        f"{win_rate}%",               "green" if win_rate >= 50 else "red")
metric_card(c6, "Net P&L / share", f"{net_pnl}",                 "green" if net_pnl >= 0 else "red")
metric_card(c7, "RSI Buy ≥",       f"{int(rsi_buy)}",            "yellow")
metric_card(c8, "Stop-Loss",       f"{stop_loss_pct}%" if stop_loss_pct > 0 else "OFF", "red" if stop_loss_pct > 0 else "")

# ── Open Position Banner ──────────────────────────────────────
if open_pos:
    unr      = round(ltp - open_pos["entry_price"], 2)
    unr_pct  = round((unr / open_pos["entry_price"]) * 100, 2)
    color    = "#00e676" if unr >= 0 else "#ff5252"
    days_open = (yesterday - open_pos["entry_date"]).days
    st.markdown(f"""
    <div class="open-position-box">
        ⚠️ <strong style="color:#ffd600">OPEN POSITION — Trade not yet closed</strong><br>
        &nbsp;&nbsp;Bought @ <strong>{open_pos['entry_price']}</strong>
        on <strong>{open_pos['entry_date']}</strong>
        &nbsp;|&nbsp; RSI at entry: <strong>{open_pos['entry_rsi']}</strong>
        &nbsp;|&nbsp; Days Running: <strong style="color:#ffd600">{days_open}d</strong>
        &nbsp;|&nbsp; Current LTP: <strong>{ltp}</strong>
        &nbsp;|&nbsp; Unrealised P&L:
        <strong style="color:{color}">{unr} ({unr_pct}%)</strong>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Chart ─────────────────────────────────────────────────────
st.plotly_chart(build_chart(df, ticker_input, rsi_buy), use_container_width=True)

st.markdown("---")

# ── Trade History Table ───────────────────────────────────────
st.markdown("### 📋 Complete Trade History")

if trades_df.empty:
    st.warning("⚠️ No completed trades found. Try lowering the RSI threshold or increasing the data range.")
else:
    def color_result(val):
        return "color: #00e676; font-weight:700;" if val == "PROFIT" else "color: #ff5252; font-weight:700;"

    def color_pnl(val):
        try:
            return "color: #00e676; font-weight:700;" if float(val) >= 0 else "color: #ff5252; font-weight:700;"
        except:
            return ""

    styled = (
        trades_df.style
        .applymap(color_result, subset=["Result"])
        .applymap(color_pnl,    subset=["P&L", "P&L %"])
        .format({
            "Entry Price" : "{:.2f}",
            "Exit Price"  : "{:.2f}",
            "RSI at Buy"  : "{:.1f}",
            "RSI at Sell" : "{:.1f}",
            "Days Held"   : "{:d}d",
            "P&L"         : "{:.2f}",
            "P&L %"       : "{:.2f}",
        })
        .set_properties(**{"text-align": "center"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Performance Stats ──────────────────────────────────────
    st.markdown("### 📊 Performance Statistics")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Total Trades",     total)
    s2.metric("✅ Profitable",    winners)
    s3.metric("❌ Loss Trades",   losers)
    s4.metric("Avg Profit",       f"{avg_win}")
    s5.metric("Avg Loss",         f"{avg_loss}")
    s6.metric("Best / Worst",     f"{best} / {worst}")

    # ── Equity Curve ───────────────────────────────────────────
    if total >= 2:
        st.markdown("### 📈 Cumulative P&L Curve")
        cum_pnl = trades_df["P&L"].cumsum().reset_index(drop=True)
        eq_fig  = go.Figure()
        eq_fig.add_trace(go.Scatter(
            x=list(range(1, len(cum_pnl) + 1)),
            y=cum_pnl,
            mode="lines+markers",
            line=dict(color="#4f8ef7", width=2),
            marker=dict(
                color=["#00e676" if v >= 0 else "#ff5252" for v in cum_pnl],
                size=7,
            ),
            fill="tozeroy",
            fillcolor="rgba(79,142,247,0.10)",
            name="Cumulative P&L",
        ))
        eq_fig.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.5)
        eq_fig.update_layout(
            height=300,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(title="Trade #", gridcolor="#1e2130"),
            yaxis=dict(title="Cumulative P&L (per share)", gridcolor="#1e2130"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(eq_fig, use_container_width=True)

st.markdown("---")

# ── RSI Sweep Table ───────────────────────────────────────────
st.markdown("### 🔬 RSI Threshold Sweep  (1 → 100)")
st.caption("Net P&L and key stats for every possible RSI buy threshold, using the same stop-loss setting.")

with st.spinner("Running RSI sweep across all 100 thresholds …"):
    sweep_df = rsi_sweep(df, stop_loss_pct)

sw_col1, sw_col2 = st.columns([2, 1])

with sw_col1:
    sort_by = st.selectbox(
        "📊 Sort table by",
        options=["RSI Threshold", "Net P&L", "Win Rate %", "Trades", "Avg Win", "Avg Loss"],
        index=1,
    )

with sw_col2:
    sort_order = st.radio(
        "Order",
        options=["⬆️ Ascending", "⬇️ Descending"],
        index=1,
        horizontal=True,
    )

ascending = (sort_order == "⬆️ Ascending")
sweep_sorted = sweep_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

# highlight best RSI row (highest net P&L)
best_rsi_row = sweep_df.loc[sweep_df["Net P&L"].idxmax(), "RSI Threshold"]

def color_net_pnl(val):
    try:
        v = float(val)
        if v > 0:   return "color: #00e676; font-weight: 700;"
        elif v < 0: return "color: #ff5252; font-weight: 700;"
    except:
        pass
    return ""

def color_winrate(val):
    try:
        v = float(val)
        if v >= 60:  return "color: #00e676;"
        elif v >= 45: return "color: #ffd600;"
        else:         return "color: #ff5252;"
    except:
        pass
    return ""

def highlight_best(row):
    if row["RSI Threshold"] == best_rsi_row:
        return ["background-color: #1a2e1a; border-left: 3px solid #00e676;"] * len(row)
    return [""] * len(row)

sweep_styled = (
    sweep_sorted.style
    .apply(highlight_best, axis=1)
    .applymap(color_net_pnl,  subset=["Net P&L", "Avg Win", "Avg Loss"])
    .applymap(color_winrate,  subset=["Win Rate %"])
    .format({
        "RSI Threshold" : "{:d}",
        "Trades"        : "{:d}",
        "Wins"          : "{:d}",
        "Losses"        : "{:d}",
        "Win Rate %"    : "{:.1f}%",
        "Net P&L"       : "{:.2f}",
        "Avg Win"       : "{:.2f}",
        "Avg Loss"      : "{:.2f}",
    })
    .set_properties(**{"text-align": "center"})
)

st.dataframe(sweep_styled, use_container_width=True, hide_index=True, height=420)

# Summary callout
best_net  = round(sweep_df["Net P&L"].max(), 2)
best_rsi  = int(sweep_df.loc[sweep_df["Net P&L"].idxmax(), "RSI Threshold"])
worst_net = round(sweep_df["Net P&L"].min(), 2)
worst_rsi = int(sweep_df.loc[sweep_df["Net P&L"].idxmin(), "RSI Threshold"])

sc1, sc2 = st.columns(2)
sc1.success(f"🏆 **Best RSI threshold: {best_rsi}** → Net P&L: **{best_net}**")
sc2.error(  f"💀 **Worst RSI threshold: {worst_rsi}** → Net P&L: **{worst_net}**")

# Download sweep table
st.download_button(
    "⬇️ Download RSI Sweep CSV",
    data=sweep_sorted.to_csv(index=False),
    file_name=f"{ticker_input}_rsi_sweep_{yesterday}.csv",
    mime="text/csv",
)

st.markdown("---")

# ── Downloads ─────────────────────────────────────────────────
st.markdown("### 💾 Download Results")
d1, d2, d3 = st.columns(3)

with d1:
    if not trades_df.empty:
        st.download_button(
            "⬇️ Trades CSV",
            data=trades_df.to_csv(index=False),
            file_name=f"{ticker_input}_trades_{yesterday}.csv",
            mime="text/csv",
            use_container_width=True,
        )

with d2:
    ind_cols = ["Open","High","Low","Close","Volume","MACD","MACD_signal","MACD_hist","RSI","signal"]
    st.download_button(
        "⬇️ Indicators CSV",
        data=df[ind_cols].to_csv(index_label="Date"),
        file_name=f"{ticker_input}_indicators_{yesterday}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with d3:
    summary_json = {
        "ticker"          : ticker_input,
        "data_as_of"      : str(yesterday),
        "rsi_buy_thresh"  : rsi_buy,
        "stop_loss_pct"   : stop_loss_pct,
        "sell_condition"  : f"MACD bearish crossover OR stop-loss -{stop_loss_pct}% (whichever first)",
        "total_trades"    : total,
        "profitable"      : winners,
        "loss_trades"     : losers,
        "win_rate_pct"    : win_rate,
        "net_pnl"         : net_pnl,
        "avg_win"         : avg_win,
        "avg_loss"        : avg_loss,
        "best_trade"      : best,
        "worst_trade"     : worst,
        "open_position"   : str(open_pos) if open_pos else None,
        "generated_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.download_button(
        "⬇️ Summary JSON",
        data=json.dumps(summary_json, indent=4, default=str),
        file_name=f"{ticker_input}_summary_{yesterday}.json",
        mime="application/json",
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    f"📁 Auto-saved to `stock_signals_data/`  ·  "
    f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  "
    f"Ticker: {ticker_input}  ·  RSI Buy ≥ {rsi_buy}  ·  "
    f"Sell: MACD bearish crossover OR stop-loss -{stop_loss_pct}%"
)
