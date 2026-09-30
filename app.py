import streamlit as st
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from urllib.parse import quote

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
    "Binance USDT Spot Scanner — RSI Multi-Timeframe + EMA 9/33 Bullish Cross"
)

# =========================================================
# RSI SETTINGS
# =========================================================

st.header("📊 RSI Scanner")

timeframe = st.selectbox(
    "Select Primary Timeframe",
    ["5m", "15m", "1h", "4h"],
    index=1,
    key="rsi_primary_tf"
)

confirmation_timeframe = st.selectbox(
    "Select Confirmation Timeframe",
    ["5m", "15m", "1h", "4h"],
    index=0,
    key="rsi_confirmation_tf"
)

rsi_range = st.selectbox(
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
    "Only fresh bullish crosses are shown: "
    "EMA 9 crosses from below EMA 33 to above EMA 33."
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

def calculate_rsi(closes, period=14):

    delta = closes.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()

    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================================================
# EMA CALCULATION
# =========================================================

def calculate_ema(closes, period):

    return closes.ewm(
        span=period,
        adjust=False
    ).mean()


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
# RSI RANGE
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

        # Remove current unfinished candle
        if use_closed_candles:
            candles = candles[:-1]

        return candles

    except Exception:

        return None


# =========================================================
# GET RSI
# =========================================================

def get_coin_rsi(
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

        if len(closes) < 16:
            return None

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
            return None

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

        return {
            "current": current_rsi,
            "previous": previous_rsi,
            "direction": direction
        }

    except Exception:

        return None


# =========================================================
# EMA BULLISH CROSS
# =========================================================

def get_ema_bullish_cross(
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

        ema9 = calculate_ema(
            closes,
            9
        )

        ema33 = calculate_ema(
            closes,
            33
        )

        # Previous closed candle
        previous_ema9 = float(
            ema9.iloc[-2]
        )

        previous_ema33 = float(
            ema33.iloc[-2]
        )

        # Latest closed candle
        current_ema9 = float(
            ema9.iloc[-1]
        )

        current_ema33 = float(
            ema33.iloc[-1]
        )

        # Fresh bullish crossover:
        #
        # Previous:
        # EMA9 <= EMA33
        #
        # Current:
        # EMA9 > EMA33

        bullish_cross = (
            previous_ema9 <= previous_ema33
            and
            current_ema9 > current_ema33
        )

        if not bullish_cross:
            return None

        current_price = float(
            closes.iloc[-1]
        )

        return {
            "price": current_price,
            "ema9": current_ema9,
            "ema33": current_ema33
        }

    except Exception:

        return None


# =========================================================
# TRADINGVIEW
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

    encoded_symbol = quote(
        f"BINANCE:{symbol}",
        safe=""
    )

    return (
        "https://www.tradingview.com/chart/"
        f"?symbol={encoded_symbol}"
        f"&interval={tv_interval}"
    )


# =========================================================
# PRIMARY RSI SCAN
# =========================================================

def scan_primary(
    symbols
):

    results = []

    progress = st.progress(0)

    status = st.empty()

    completed = 0

    total = len(symbols)

    with ThreadPoolExecutor(
        max_workers=20
    ) as executor:

        futures = {

            executor.submit(
                get_coin_rsi,
                symbol,
                timeframe
            ): symbol

            for symbol in symbols
        }

        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]

            try:

                data = future.result()

                if data is not None:

                    if not is_in_range(
                        data["current"],
                        rsi_range
                    ):

                        completed += 1

                        progress.progress(
                            completed / total
                        )

                        continue

                    if (
                        direction_filter
                        == "Rising"
                        and
                        data["direction"]
                        != "🟢 Rising"
                    ):

                        completed += 1

                        progress.progress(
                            completed / total
                        )

                        continue

                    if (
                        direction_filter
                        == "Falling"
                        and
                        data["direction"]
                        != "🔴 Falling"
                    ):

                        completed += 1

                        progress.progress(
                            completed / total
                        )

                        continue

                    results.append(
                        {
                            "Coin":
                                symbol,

                            "Primary RSI":
                                data["current"],

                            "Previous RSI":
                                data["previous"],

                            "Direction":
                                data["direction"]
                        }
                    )

            except Exception:

                pass

            completed += 1

            progress.progress(
                completed / total
            )

            status.write(
                f"Primary scan: "
                f"{completed}/{total}"
            )

    progress.empty()

    status.empty()

    return results


# =========================================================
# CONFIRMATION RSI SCAN
# =========================================================

def scan_confirmation(
    primary_results
):

    if not primary_results:
        return []

    results = []

    progress = st.progress(0)

    status = st.empty()

    completed = 0

    total = len(primary_results)

    with ThreadPoolExecutor(
        max_workers=20
    ) as executor:

        futures = {

            executor.submit(
                get_coin_rsi,
                item["Coin"],
                confirmation_timeframe
            ): item

            for item in primary_results
        }

        for future in as_completed(
            futures
        ):

            item = futures[
                future
            ]

            try:

                data = future.result()

                if data is not None:

                    if not is_in_range(
                        data["current"],
                        confirmation_rsi_range
                    ):

                        completed += 1

                        progress.progress(
                            completed / total
                        )

                        continue

                    symbol = item[
                        "Coin"
                    ]

                    results.append(
                        {
                            "Coin":
                                symbol,

                            "Primary RSI":
                                item[
                                    "Primary RSI"
                                ],

                            "Previous RSI":
                                item[
                                    "Previous RSI"
                                ],

                            "Direction":
                                item[
                                    "Direction"
                                ],

                            "Confirmation RSI":
                                data[
                                    "current"
                                ],

                            "TradingView":
                                get_tradingview_url(
                                    symbol,
                                    timeframe
                                )
                        }
                    )

            except Exception:

                pass

            completed += 1

            progress.progress(
                completed / total
            )

            status.write(
                f"Confirmation scan: "
                f"{completed}/{total}"
            )

    progress.empty()

    status.empty()

    return results


# =========================================================
# EMA MULTI-TIMEFRAME SCAN
# =========================================================

def scan_ema_bullish(
    symbols,
    selected_timeframes
):

    all_results = []

    if not selected_timeframes:

        return all_results

    total_tasks = (
        len(symbols)
        *
        len(selected_timeframes)
    )

    progress = st.progress(0)

    status = st.empty()

    completed = 0

    with ThreadPoolExecutor(
        max_workers=20
    ) as executor:

        futures = {}

        for interval in selected_timeframes:

            for symbol in symbols:

                future = executor.submit(
                    get_ema_bullish_cross,
                    symbol,
                    interval
                )

                futures[
                    future
                ] = (
                    symbol,
                    interval
                )

        for future in as_completed(
            futures
        ):

            symbol, interval = futures[
                future
            ]

            try:

                data = future.result()

                if data is not None:

                    # Get RSI for the same timeframe
                    rsi_data = get_coin_rsi(
                        symbol,
                        interval
                    )

                    if rsi_data is not None:

                        rsi_value = (
                            rsi_data["current"]
                        )

                    else:

                        rsi_value = None

                    all_results.append(
                        {
                            "Coin":
                                symbol,

                            "Timeframe":
                                interval,

                            "Signal":
                                "🟢 Bullish EMA Cross",

                            "Price":
                                round(
                                    data[
                                        "price"
                                    ],
                                    8
                                ),

                            "EMA 9":
                                round(
                                    data[
                                        "ema9"
                                    ],
                                    8
                                ),

                            "EMA 33":
                                round(
                                    data[
                                        "ema33"
                                    ],
                                    8
                                ),

                            "RSI":
                                rsi_value,

                            "TradingView":
                                get_tradingview_url(
                                    symbol,
                                    interval
                                )
                        }
                    )

            except Exception:

                pass

            completed += 1

            progress.progress(
                completed / total_tasks
            )

            status.write(
                f"EMA scan: "
                f"{completed}/{total_tasks}"
            )

    progress.empty()

    status.empty()

    return all_results


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

        st.warning(
            "No RSI coins found."
        )

        return

    df = df.sort_values(
        "Primary RSI",
        ascending=False
    )

    # Save original coin names
    coin_names = df["Coin"].copy()

    # Convert to TradingView URLs
    df["Coin"] = [
        get_tradingview_url(
            symbol,
            timeframe
        )
        for symbol in coin_names
    ]

    st.success(
        f"RSI scan completed! "
        f"Found {len(df)} coins."
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
                        r".*symbol=BINANCE%3A(.*?)&interval=.*"
                    ),
                    pinned=True
                ),
