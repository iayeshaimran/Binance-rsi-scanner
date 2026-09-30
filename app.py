import streamlit as st
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Binance RSI + EMA Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Binance RSI + EMA Scanner")

st.caption(
    "Binance USDT Spot Scanner — RSI + EMA 9/33 Bullish Cross"
)


# =========================================================
# RSI SETTINGS
# =========================================================

st.header("📊 RSI Scanner")

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
    [
        "All",
        "40 - 50",
        "50 - 55",
        "55 - 60",
        "60 - 70",
        "70+"
    ],
    index=2
)

confirmation_rsi_range = st.selectbox(
    "Confirmation RSI Range",
    [
        "All",
        "40 - 50",
        "50 - 55",
        "55 - 60",
        "60 - 70",
        "70+"
    ],
    index=0
)

direction_filter = st.selectbox(
    "RSI Direction (primary)",
    [
        "All",
        "Rising",
        "Falling"
    ],
    index=0
)

use_closed_candles = st.checkbox(
    "Use closed candles only (recommended)",
    value=True
)


# =========================================================
# EMA SETTINGS
# =========================================================

st.header("🟢 EMA 9 / EMA 33 Bullish Cross")

ema_timeframes = st.multiselect(
    "EMA Bullish Cross Timeframes",
    ["5m", "15m", "1h", "4h"],
    default=["5m", "15m", "1h", "4h"]
)

st.caption(
    "Only a fresh bullish cross is shown: "
    "EMA 9 was below/equal to EMA 33 on the previous candle "
    "and is above EMA 33 on the latest closed candle."
)


# =========================================================
# REFRESH SETTINGS
# =========================================================

st.header("🔄 Scanner Refresh")

auto_refresh = st.checkbox(
    "Auto Refresh"
)

refresh_minutes = st.selectbox(
    "Refresh every",
    [1, 5, 10, 15],
    index=1,
    disabled=not auto_refresh
)


# =========================================================
# THREAD LOCAL REQUEST SESSION
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
            status_forcelist=[
                429,
                500,
                502,
                503,
                504
            ],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(
            max_retries=retry
        )

        session.mount(
            "https://",
            adapter
        )

        thread_local.session = session

    return thread_local.session


# =========================================================
# RSI CALCULATION
# =========================================================

def calculate_rsi(
    closes,
    period=14
):

    delta = closes.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# =========================================================
# EMA CALCULATION
# =========================================================

def calculate_ema(
    closes,
    period
):

    return closes.ewm(
        span=period,
        adjust=False
    ).mean()


# =========================================================
# RSI RANGE FILTER
# =========================================================

def is_in_range(
    rsi,
    selected_range
):

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
# GET BINANCE SYMBOLS
# =========================================================

@st.cache_data(ttl=300)
def get_symbols():

    url = (
        "https://data-api.binance.vision"
        "/api/v3/exchangeInfo"
    )

    session = get_session()

    response = session.get(
        url,
        timeout=(10, 30)
    )

    response.raise_for_status()

    data = response.json()

    symbols = []

    for item in data["symbols"]:

        if (
            item["quoteAsset"] == "USDT"
            and item["status"] == "TRADING"
            and item["isSpotTradingAllowed"]
        ):

            symbols.append(
                item["symbol"]
            )

    return symbols


# =========================================================
# GET KLINES
# =========================================================

def get_klines(
    symbol,
    interval,
    limit=100
):

    url = (
        "https://data-api.binance.vision"
        "/api/v3/klines"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:

        session = get_session()

        response = session.get(
            url,
            params=params,
            timeout=(5, 12)
        )

        if response.status_code != 200:
            return None

        candles = response.json()

        if len(candles) < 40:
            return None

        # Remove currently forming candle
        if use_closed_candles:
            candles = candles[:-1]

        return candles

    except Exception:

        return None


# =========================================================
# ANALYZE ONE COIN / ONE TIMEFRAME
# =========================================================

def analyze_coin(
    symbol,
    interval
):

    candles = get_klines(
        symbol,
        interval,
        100
    )

    if candles is None:
        return None

    try:

        closes = pd.Series(
            [
                float(candle[4])
                for candle in candles
            ]
        )

        if len(closes) < 35:
            return None

        # -----------------------------
        # RSI
        # -----------------------------

        rsi = calculate_rsi(
            closes,
            14
        )

        current_rsi = rsi.iloc[-1]

        previous_rsi = rsi.iloc[-2]

        if (
            pd.isna(current_rsi)
            or pd.isna(previous_rsi)
        ):

            current_rsi = None
            previous_rsi = None
            direction = "⚪ Flat"

        else:

            current_rsi = round(
                float(current_rsi),
                2
            )

            previous_rsi = round(
                float(previous_rsi),
                2
            )

            if current_rsi > previous_rsi:

                direction = "🟢 Rising"

            elif current_rsi < previous_rsi:

                direction = "🔴 Falling"

            else:

                direction = "⚪ Flat"

        # -----------------------------
        # EMA 9
        # -----------------------------

        ema9 = calculate_ema(
            closes,
            9
        )

        # -----------------------------
        # EMA 33
        # -----------------------------

        ema33 = calculate_ema(
            closes,
            33
        )

        previous_ema9 = float(
            ema9.iloc[-2]
        )

        previous_ema33 = float(
            ema33.iloc[-2]
        )

        current_ema9 = float(
            ema9.iloc[-1]
        )

        current_ema33 = float(
            ema33.iloc[-1]
        )

        # =================================================
        # FRESH BULLISH CROSS
        # =================================================
        #
        # Previous candle:
        # EMA 9 <= EMA 33
        #
        # Latest closed candle:
        # EMA 9 > EMA 33
        #
        # =================================================

        bullish_cross = (
            previous_ema9 <= previous_ema33
            and
            current_ema9 > current_ema33
        )

        current_price = float(
            closes.iloc[-1]
        )

        return {
            "symbol": symbol,
            "interval": interval,

            "price": current_price,

            "rsi": current_rsi,
            "previous_rsi": previous_rsi,
            "direction": direction,

            "ema9": current_ema9,
            "ema33": current_ema33,

            "bullish_cross": bullish_cross
        }

    except Exception:

        return None


# =========================================================
# TRADINGVIEW URL
# =========================================================

def get_tradingview_interval(
    interval
):

    mapping = {
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240"
    }

    return mapping.get(
        interval,
        "15"
    )


def get_tradingview_url(
    symbol,
    interval
):

    tv_interval = get_tradingview_interval(
        interval
    )

    # Colon is intentionally left unencoded
    # so the coin name can be displayed
    # by Streamlit LinkColumn.

    return (
        "https://www.tradingview.com/chart/"
        f"?symbol=BINANCE:{symbol}"
        f"&interval={tv_interval}"
    )


# =========================================================
# SCAN SYMBOLS FOR A TIMEFRAME
# =========================================================

def scan_timeframe(
    symbols,
    interval,
    progress_info=None
):

    results = []

    total = len(symbols)

    completed = 0

    with ThreadPoolExecutor(
        max_workers=20
    ) as executor:

        futures = {

            executor.submit(
                analyze_coin,
                symbol,
                interval
            ): symbol

            for symbol in symbols
        }

        for future in as_completed(
            futures
        ):

            try:

                data = future.result()

                if data is not None:

                    results.append(
                        data
                    )

            except Exception:

                pass

            completed += 1

            if progress_info is not None:

                progress_info(
                    completed,
                    total,
                    interval
                )

    return results


# =========================================================
# SCAN ALL REQUIRED TIMEFRAMES
# =========================================================

def scan_market_data(
    symbols,
    required_timeframes
):

    all_data = {}

    progress = st.progress(0)

    status = st.empty()

    total_work = (
        len(required_timeframes)
        *
        len(symbols)
    )

    completed_total = 0

    for interval in required_timeframes:

        interval_results = []

        with ThreadPoolExecutor(
            max_workers=20
        ) as executor:

            futures = {

                executor.submit(
                    analyze_coin,
                    symbol,
                    interval
                ): symbol

                for symbol in symbols
            }

            for future in as_completed(
                futures
            ):

                try:

                    data = future.result()

                    if data is not None:

                        interval_results.append(
                            data
                        )

                except Exception:

                    pass

                completed_total += 1

                progress.progress(
                    completed_total
                    / total_work
                )

                status.write(
                    f"Scanning {interval}: "
                    f"{completed_total}/"
                    f"{total_work}"
                )

        all_data[interval] = (
            interval_results
        )

    progress.empty()

    status.empty()

    return all_data


# =========================================================
# RSI RESULTS
# =========================================================

def build_rsi_results(
    market_data
):

    primary_data = market_data.get(
        primary_timeframe,
        []
    )

    confirmation_data = market_data.get(
        confirmation_timeframe,
        []
    )

    confirmation_map = {
        item["symbol"]: item
        for item in confirmation_data
    }

    results = []

    for item in primary_data:

        symbol = item["symbol"]

        current_rsi = item["rsi"]

        if current_rsi is None:
            continue

        # Primary RSI filter
        if not is_in_range(
            current_rsi,
            primary_rsi_range
        ):
            continue

        # Direction filter
        if (
            direction_filter == "Rising"
            and item["direction"]
            != "🟢 Rising"
        ):
            continue

        if (
            direction_filter == "Falling"
            and item["direction"]
            != "🔴 Falling"
        ):
            continue

        confirmation = (
            confirmation_map.get(symbol)
        )

        if confirmation is None:
            continue

        confirmation_rsi = (
            confirmation["rsi"]
        )

        if confirmation_rsi is None:
            continue

        if not is_in_range(
            confirmation_rsi,
            confirmation_rsi_range
        ):
            continue

        results.append(
            {
                "Coin": symbol,

                "Primary RSI":
                    item["rsi"],

                "Previous RSI":
                    item["previous_rsi"],

                "Direction":
                    item["direction"],

                "Confirmation RSI":
                    confirmation_rsi,

                "TradingView":
                    get_tradingview_url(
                        symbol,
                        primary_timeframe
                    )
            }
        )

    return results


# =========================================================
# EMA BULLISH CROSS RESULTS
# =========================================================

def build_ema_results(
    market_data
):

    results = []

    for interval in ema_timeframes:

        timeframe_data = market_data.get(
            interval,
            []
        )

        for item in timeframe_data:

            # ONLY FRESH BULLISH CROSS
            if not item["bullish_cross"]:
                continue

            results.append(
                {
                    "Coin":
                        item["symbol"],

                    "Timeframe":
                        interval,

                    "Signal":
                        "🟢 Bullish EMA Cross",

                    "Price":
                        item["price"],

                    "EMA 9":
                        item["ema9"],

                    "EMA 33":
                        item["ema33"],

                    "RSI":
                        item["rsi"],

                    "TradingView":
                        get_tradingview_url(
                            item["symbol"],
                            interval
                        )
                }
            )

    return results


# =========================================================
# SHOW RSI RESULTS
# =========================================================

def show_rsi_results(
    results
):

    st.subheader(
        "📊 RSI Results"
    )

    df = pd.DataFrame(
        results
    )

    if df.empty:

        st.info(
            "No coins matched the selected RSI filters."
        )

        return

    df = df.sort_values(
        "Primary RSI",
        ascending=False
    )

    st.success(
        f"RSI scan completed — "
        f"{len(df)} coins found."
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Coin":
                st.column_config.LinkColumn(
                    "Coin",
                    display_text=(
                        r".*symbol=BINANCE:(.*?)&interval=.*"
                    ),
                    pinned=True
                ),

            "Primary RSI":
                st.column_config.NumberColumn(
                    f"{primary_timeframe} RSI",
                    format="%.2f"
                ),

            "Previous RSI":
                st.column_config.NumberColumn(
                    "Previous RSI",
                    format="%.2f"
                ),

            "Direction":
                st.column_config.TextColumn(
                    "RSI Direction"
                ),

            "Confirmation RSI":
                st.column_config.NumberColumn(
                    f"{confirmation_timeframe} RSI",
                    format="%.2f"
                ),

            "TradingView":
                st.column_config.LinkColumn(
                    "TradingView",
                    display_text="Open Chart 🔗"
                )
        }
    )


# =========================================================
# SHOW EMA RESULTS
# =========================================================

def show_ema_results(
    results
):

    st.subheader(
        "🟢 EMA 9 / EMA 33 Bullish Cross Alerts"
    )

    df = pd.DataFrame(
        results
    )

    if df.empty:

        st.info(
            "No fresh bullish EMA 9/33 cross "
            "found in the selected timeframes."
        )

        return

    df = df.sort_values(
        [
            "Timeframe",
            "Coin"
        ]
    )

    st.success(
        f"Fresh bullish cross found: "
        f"{len(df)}"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Coin":
                st.column_config.LinkColumn(
                    "Coin",
                    help=(
                        "Click the coin name "
                        "to open TradingView."
                    ),
                    display_text=(
                        r".*symbol=BINANCE:(.*?)&interval=.*"
                    ),
                    pinned=True
                ),

            "Timeframe":
                st.column_config.TextColumn(
                    "Timeframe"
                ),

            "Signal":
                st.column_config.TextColumn(
                    "Signal"
                ),

            "Price":
                st.column_config.NumberColumn(
                    "Current Price",
                    format="%.8f"
                ),

            "EMA 9":
                st.column_config.NumberColumn(
                    "EMA 9",
                    format="%.8f"
                ),

            "EMA 33":
                st.column_config.NumberColumn(
                    "EMA 33",
                    format="%.8f"
                ),

            "RSI":
                st.column_config.NumberColumn(
                    "RSI",
                    format="%.2f"
                ),

            "TradingView":
                st.column_config.LinkColumn(
                    "TradingView",
                    display_text="Open Chart 🔗"
                )
        }
