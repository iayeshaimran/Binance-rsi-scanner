import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Binance RSI + EMA + HA Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Binance RSI + EMA + Heikin Ashi Scanner")

st.caption(
    "Binance USDT Spot Scanner — RSI + EMA 9/33 + Multi-Timeframe Heikin Ashi"
)


# =========================================================
# RSI SETTINGS
# =========================================================

st.header("📊 RSI Scanner")

RSI_RANGES = ["All", "40 - 50", "50 - 55", "55 - 60", "60 - 70", "70+"]

primary_timeframe = st.selectbox(
    "Select Primary Timeframe",
    ["5m", "15m", "1h", "4h"],
    index=1
)

confirmation_timeframe = st.selectbox(
    "Select Confirmation Timeframe",
    ["5m", "15m", "1h", "4h"],
    index=0
)

primary_rsi_range = st.selectbox(
    "Select Primary RSI Range",
    RSI_RANGES,
    index=2
)

confirmation_rsi_range = st.selectbox(
    "Confirmation RSI Range",
    RSI_RANGES,
    index=0
)

direction_filter = st.selectbox(
    "RSI Direction",
    ["All", "Rising", "Falling"],
    index=0
)

use_closed_candles = st.checkbox(
    "Use closed candles only (recommended)",
    value=True
)


# =========================================================
# EMA SETTINGS
# =========================================================

st.header("🟢 EMA 9 / EMA 33")

ema_timeframes = st.multiselect(
    "EMA Bullish Cross Timeframes",
    ["5m", "15m", "1h", "4h"],
    default=["5m", "15m", "1h", "4h"]
)

st.caption(
    "Only a fresh bullish cross is shown: "
    "EMA 9 was below/equal to EMA 33 and then crossed above it."
)


# =========================================================
# HEIKIN ASHI SETTINGS
# =========================================================

st.header("🕯️ Multi-Timeframe Heikin Ashi")

ha_timeframes = st.multiselect(
    "Heikin Ashi Timeframes",
    ["3m", "5m", "15m", "1h", "4h"],
    default=["3m", "5m", "15m", "1h", "4h"]
)

ha_lower_wick_limit = st.slider(
    "Maximum HA lower wick / body",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

ha_min_alignment = st.selectbox(
    "Minimum Bullish HA Timeframes Required",
    [1, 2, 3, 4, 5],
    index=2
)

st.caption(
    "Bullish HA = last 2 closed HA candles are green, "
    "second candle has higher close and higher high, "
    "and lower wicks are small."
)


# =========================================================
# REFRESH SETTINGS
# =========================================================

st.header("🔄 Scanner Refresh")

auto_refresh = st.checkbox("Auto Refresh")

refresh_minutes = st.selectbox(
    "Refresh every",
    [1, 5, 10, 15],
    index=1,
    disabled=not auto_refresh
)

scan_clicked = st.button("🔍 Scan Now", type="primary")


# =========================================================
# THREAD LOCAL SESSION
# =========================================================

thread_local = threading.local()


def get_session():

    if not hasattr(thread_local, "session"):

        session = requests.Session()

        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=20,
            pool_maxsize=20
        )

        session.mount("https://", adapter)

        thread_local.session = session

    return thread_local.session


# =========================================================
# RSI (Wilder smoothing — matches Binance / TradingView)
# =========================================================

def calculate_rsi(closes, period=14):

    delta = closes.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # avg_loss == 0 -> RSI 100
    rsi = rsi.where(avg_loss != 0, 100)

    return rsi


# =========================================================
# EMA
# =========================================================

def calculate_ema(closes, period):

    return closes.ewm(span=period, adjust=False).mean()


# =========================================================
# BINANCE SYMBOLS
# =========================================================

@st.cache_data(ttl=300)
def get_symbols():

    url = "https://data-api.binance.vision/api/v3/exchangeInfo"

    response = get_session().get(url, timeout=(10, 30))
    response.raise_for_status()

    data = response.json()

    symbols = []

    for item in data["symbols"]:

        if (
            item["quoteAsset"] == "USDT"
            and item["status"] == "TRADING"
            and item["isSpotTradingAllowed"]
        ):
            symbols.append(item["symbol"])

    return symbols


# =========================================================
# RSI RANGE
# =========================================================

def is_in_range(rsi, selected_range):

    if selected_range == "All":
        return True

    if selected_range == "40 - 50":
        return 40 <= rsi < 50

    if selected_range == "50 - 55":
        return 50 <= rsi < 55

    if selected_range == "55 - 60":
        return 55 <= rsi < 60

    if selected_range == "60 - 70":
        return 60 <= rsi < 70

    if selected_range == "70+":
        return rsi >= 70

    return False


# =========================================================
# GET KLINES
# =========================================================

def get_klines(symbol, interval, closed_only, limit=200):

    url = "https://data-api.binance.vision/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:

        response = get_session().get(url, params=params, timeout=(5, 12))

        if response.status_code != 200:
            return None

        candles = response.json()

        if len(candles) < 40:
            return None

        if closed_only:
            candles = candles[:-1]

        return candles

    except Exception:

        return None


# =========================================================
# HEIKIN ASHI
# =========================================================

def calculate_heikin_ashi(candles):

    opens = pd.Series([float(c[1]) for c in candles])
    highs = pd.Series([float(c[2]) for c in candles])
    lows = pd.Series([float(c[3]) for c in candles])
    closes = pd.Series([float(c[4]) for c in candles])

    ha_close = (opens + highs + lows + closes) / 4

    ha_open = pd.Series(index=opens.index, dtype=float)

    ha_open.iloc[0] = (opens.iloc[0] + closes.iloc[0]) / 2

    for i in range(1, len(candles)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    ha_high = pd.concat([highs, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([lows, ha_open, ha_close], axis=1).min(axis=1)

    return ha_open, ha_high, ha_low, ha_close


# =========================================================
# ANALYZE ONE COIN
# =========================================================

def analyze_coin(symbol, interval, closed_only, wick_limit):

    candles = get_klines(symbol, interval, closed_only, 200)

    if candles is None:
        return None

    try:

        closes = pd.Series([float(c[4]) for c in candles])

        # ---------------- RSI ----------------

        rsi = calculate_rsi(closes, 14)

        current_rsi = rsi.iloc[-1]
        previous_rsi = rsi.iloc[-2]

        if pd.isna(current_rsi) or pd.isna(previous_rsi):

            current_rsi = None
            previous_rsi = None
            direction = "⚪ Flat"

        else:

            current_rsi = round(float(current_rsi), 2)
            previous_rsi = round(float(previous_rsi), 2)

            if current_rsi > previous_rsi:
                direction = "🟢 Rising"
            elif current_rsi < previous_rsi:
                direction = "🔴 Falling"
            else:
                direction = "⚪ Flat"

        # ---------------- EMA ----------------

        ema9 = calculate_ema(closes, 9)
        ema33 = calculate_ema(closes, 33)

        previous_ema9 = float(ema9.iloc[-2])
        previous_ema33 = float(ema33.iloc[-2])
        current_ema9 = float(ema9.iloc[-1])
        current_ema33 = float(ema33.iloc[-1])

        ema_bullish_cross = (
            previous_ema9 <= previous_ema33
            and current_ema9 > current_ema33
        )

        # ---------------- HEIKIN ASHI ----------------

        ha_open, ha_high, ha_low, ha_close = calculate_heikin_ashi(candles)

        first = -2
        second = -1

        first_green = ha_close.iloc[first] > ha_open.iloc[first]
        second_green = ha_close.iloc[second] > ha_open.iloc[second]

        higher_close = ha_close.iloc[second] > ha_close.iloc[first]
        higher_high = ha_high.iloc[second] > ha_high.iloc[first]

        first_body = abs(ha_close.iloc[first] - ha_open.iloc[first])
        second_body = abs(ha_close.iloc[second] - ha_open.iloc[second])

        first_lower_wick = (
            min(ha_open.iloc[first], ha_close.iloc[first])
            - ha_low.iloc[first]
        )

        second_lower_wick = (
            min(ha_open.iloc[second], ha_close.iloc[second])
            - ha_low.iloc[second]
        )

        first_wick_ok = (
            first_body > 0
            and first_lower_wick <= first_body * wick_limit
        )

        second_wick_ok = (
            second_body > 0
            and second_lower_wick <= second_body * wick_limit
        )

        ha_bullish_signal = bool(
            first_green
            and second_green
            and higher_close
            and higher_high
            and first_wick_ok
            and second_wick_ok
        )

        return {
            "symbol": symbol,
            "interval": interval,
            "price": float(closes.iloc[-1]),

            "rsi": current_rsi,
            "previous_rsi": previous_rsi,
            "direction": direction,

            "ema9": current_ema9,
            "ema33": current_ema33,
            "ema_bullish_cross": ema_bullish_cross,

            "ha_bullish_signal": ha_bullish_signal,
            "ha_close": float(ha_close.iloc[-1])
        }

    except Exception:

        return None


# =========================================================
# TRADINGVIEW
# =========================================================

def get_tradingview_interval(interval):

    mapping = {
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240"
    }

    return mapping.get(interval, "15")


def get_tradingview_url(symbol, interval):

    tv_interval = get_tradingview_interval(interval)

    return (
        "https://www.tradingview.com/chart/"
        f"?symbol=BINANCE:{symbol}"
        f"&interval={tv_interval}"
    )


# =========================================================
# SCAN MARKET
# =========================================================

def scan_market(symbols, required_timeframes, closed_only, wick_limit):

    market_data = {}

    total = len(symbols) * len(required_timeframes)

    if total == 0:
        return market_data

    completed = 0

    progress = st.progress(0)
    status = st.empty()

    for interval in required_timeframes:

        results = []

        with ThreadPoolExecutor(max_workers=20) as executor:

            futures = {
                executor.submit(
                    analyze_coin,
                    symbol,
                    interval,
                    closed_only,
                    wick_limit
                ): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):

                try:
                    data = future.result()
                    if data is not None:
                        results.append(data)
                except Exception:
                    pass

                completed += 1

                progress.progress(completed / total)

                status.write(f"Scanning {interval}: {completed}/{total}")

        market_data[interval] = results

    progress.empty()
    status.empty()

    return market_data


# =========================================================
# BUILD RSI RESULTS
# =========================================================

def build_rsi_results(market_data):

    primary = market_data.get(primary_timeframe, [])
    confirmation = market_data.get(confirmation_timeframe, [])

    confirmation_map = {item["symbol"]: item for item in confirmation}

    results = []

    for item in primary:

        symbol = item["symbol"]
        rsi = item["rsi"]

        if rsi is None:
            continue

        if not is_in_range(rsi, primary_rsi_range):
            continue

        if direction_filter == "Rising" and item["direction"] != "🟢 Rising":
            continue

        if direction_filter == "Falling" and item["direction"] != "🔴 Falling":
            continue

        confirm = confirmation_map.get(symbol)

        if confirm is None:
            continue

        confirmation_rsi = confirm["rsi"]

        if confirmation_rsi is None:
            continue

        if not is_in_range(confirmation_rsi, confirmation_rsi_range):
            continue

        results.append({
            "Coin": symbol,
            "Primary RSI": rsi,
            "Previous RSI": item["previous_rsi"],
            "Direction": item["direction"],
            "Confirmation RSI": confirmation_rsi,
            "TradingView": get_tradingview_url(symbol, primary_timeframe)
        })

    return results


# =========================================================
# BUILD EMA RESULTS
# =========================================================

def build_ema_results(market_data):

    results = []

    for interval in ema_timeframes:

        for item in market_data.get(interval, []):

            if not item["ema_bullish_cross"]:
                continue

            results.append({
                "Coin": item["symbol"],
                "Timeframe": interval,
                "Signal": "🟢 EMA 9 → EMA 33",
                "Price": item["price"],
                "EMA 9": item["ema9"],
                "EMA 33": item["ema33"],
                "RSI": item["rsi"],
                "TradingView": get_tradingview_url(item["symbol"], interval)
            })

    return results


# =========================================================
# BUILD MULTI-TIMEFRAME HA RESULTS
# =========================================================

def build_ha_results(market_data):

    # Index each timeframe by symbol once (much faster than nested loops)
    lookup = {
        interval: {item["symbol"]: item for item in market_data.get(interval, [])}
        for interval in ha_timeframes
    }

    symbols = set()

    for interval in ha_timeframes:
        symbols.update(lookup[interval].keys())

    results = []

    for symbol in symbols:

        row = {"Coin": symbol}

        bullish_count = 0
        bullish_timeframes = []

        for interval in ha_timeframes:

            matching = lookup[interval].get(symbol)

            if matching is not None and matching["ha_bullish_signal"]:

                row[interval] = "🟢"
                bullish_count += 1
                bullish_timeframes.append(interval)

            else:

                row[interval] = "⚪"

        if bullish_count < ha_min_alignment:
            continue

        row["Bullish Count"] = bullish_count
        row["Aligned"] = ", ".join(bullish_timeframes)

        # Link to the highest bullish timeframe
        order = ["3m", "5m", "15m", "1h", "4h"]
        best_tf = max(bullish_timeframes, key=order.index)

        row["TradingView"] = get_tradingview_url(symbol, best_tf)

        results.append(row)

    results.sort(key=lambda r: (-r["Bullish Count"], r["Coin"]))

    return results


# =========================================================
# DISPLAY
# =========================================================

def show_table(rows, empty_message):

    if not rows:
        st.info(empty_message)
        return

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "TradingView": st.column_config.LinkColumn(
                "TradingView",
                display_text="Open chart"
            )
        }
    )

    st.caption(f"{len(df)} result(s)")


# =========================================================
# MAIN
# =========================================================

def run_scan():

    try:
        symbols = get_symbols()
    except Exception as e:
        st.error(f"Could not load Binance symbols: {e}")
        return

    required = []

    for tf in (
        [primary_timeframe, confirmation_timeframe]
        + list(ema_timeframes)
        + list(ha_timeframes)
    ):
        if tf not in required:
            required.append(tf)

    market_data = scan_market(
        symbols,
        required,
        use_closed_candles,
        ha_lower_wick_limit
    )

    st.session_state["rsi_results"] = build_rsi_results(market_data)
    st.session_state["ema_results"] = build_ema_results(market_data)
    st.session_state["ha_results"] = build_ha_results(market_data)
    st.session_state["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if scan_clicked or (auto_refresh and "last_scan" not in st.session_state):
    run_scan()

if "last_scan" in st.session_state:

    st.success(f"Last scan: {st.session_state['last_scan']}")

    tab_rsi, tab_ema, tab_ha = st.tabs(
        ["📊 RSI", "🟢 EMA Cross", "🕯️ Heikin Ashi"]
    )

    with tab_rsi:
        show_table(
            st.session_state["rsi_results"],
            "No coins match the RSI filters."
        )

    with tab_ema:
        show_table(
            st.session_state["ema_results"],
            "No fresh EMA 9/33 bullish crosses found."
        )

    with tab_ha:
        show_table(
            st.session_state["ha_results"],
            "No coins meet the Heikin Ashi alignment requirement."
        )

else:
    st.info("Press **Scan Now** to start scanning.")


# =========================================================
# AUTO REFRESH
# =========================================================

if auto_refresh:

    st.caption(f"Next refresh in {refresh_minutes} minute(s)...")

    time.sleep(refresh_minutes * 60)

    run_scan()
    st.rerun() 
