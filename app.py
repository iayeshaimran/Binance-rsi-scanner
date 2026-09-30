import streamlit as st
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Binance RSI + EMA + Heikin Ashi Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Binance RSI + EMA + Heikin Ashi Scanner")

st.caption(
    "Binance USDT Spot Scanner"
)


# =========================================================
# RSI SETTINGS
# =========================================================

st.header("📊 RSI Scanner")

primary_timeframe = st.selectbox(
    "Primary Timeframe",
    ["5m", "15m", "1h", "4h"],
    index=1
)

confirmation_timeframe = st.selectbox(
    "Confirmation Timeframe",
    ["5m", "15m", "1h", "4h"],
    index=0
)

primary_rsi_range = st.selectbox(
    "Primary RSI Range",
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
    "RSI Direction",
    [
        "All",
        "Rising",
        "Falling"
    ],
    index=0
)

use_closed_candles = st.checkbox(
    "Use closed candles only",
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
    "Only a fresh bullish cross is shown."
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
    "Maximum lower wick / candle body",
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
    "HA signal = 2 consecutive green candles + "
    "higher close + higher high + small lower wicks."
)


# =========================================================
# REFRESH SETTINGS
# =========================================================

st.header("🔄 Refresh Settings")

auto_refresh = st.checkbox(
    "Auto Refresh",
    value=False
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
# RSI
# =========================================================

def calculate_rsi(closes, period=14):

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

    return 100 - (
        100 / (1 + rs)
    )


# =========================================================
# EMA
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
# BINANCE SYMBOLS
# =========================================================

@st.cache_data(ttl=300)
def get_symbols():

    url = (
        "https://data-api.binance.vision"
        "/api/v3/exchangeInfo"
    )

    response = get_session().get(
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
# BINANCE KLINES
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

        response = get_session().get(
            url,
            params=params,
            timeout=(5, 15)
        )

        if response.status_code != 200:
            return None

        candles = response.json()

        if len(candles) < 40:
            return None

        if use_closed_candles:
            candles = candles[:-1]

        return candles

    except Exception:

        return None


# =========================================================
# HEIKIN ASHI
# =========================================================

def calculate_heikin_ashi(
    candles
):

    opens = pd.Series(
        [float(c[1]) for c in candles]
    )

    highs = pd.Series(
        [float(c[2]) for c in candles]
    )

    lows = pd.Series(
        [float(c[3]) for c in candles]
    )

    closes = pd.Series(
        [float(c[4]) for c in candles]
    )

    ha_close = (
        opens
        + highs
        + lows
        + closes
    ) / 4

    ha_open = pd.Series(
        index=opens.index,
        dtype=float
    )

    ha_open.iloc[0] = (
        opens.iloc[0]
        + closes.iloc[0]
    ) / 2

    for i in range(
        1,
        len(candles)
    ):

        ha_open.iloc[i] = (
            ha_open.iloc[i - 1]
            + ha_close.iloc[i - 1]
        ) / 2

    ha_high = pd.concat(
        [
            highs,
            ha_open,
            ha_close
        ],
        axis=1
    ).max(axis=1)

    ha_low = pd.concat(
        [
            lows,
            ha_open,
            ha_close
        ],
        axis=1
    ).min(axis=1)

    return (
        ha_open,
        ha_high,
        ha_low,
        ha_close
    )


# =========================================================
# ANALYZE ONE COIN
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
            [float(c[4]) for c in candles]
        )

        # =================================================
        # RSI
        # =================================================

        rsi_values = calculate_rsi(
            closes,
            14
        )

        current_rsi = rsi_values.iloc[-1]

        previous_rsi = rsi_values.iloc[-2]

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

        # =================================================
        # EMA
        # =================================================

        ema9 = calculate_ema(
            closes,
            9
        )

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

        # Fresh bullish cross
        ema_bullish_cross = (
            previous_ema9 <= previous_ema33
            and
            current_ema9 > current_ema33
        )

        # =================================================
        # HEIKIN ASHI
        # =================================================

        (
            ha_open,
            ha_high,
            ha_low,
            ha_close
        ) = calculate_heikin_ashi(
            candles
        )

        first = -2
        second = -1

        # Two green candles
        first_green = (
            ha_close.iloc[first]
            > ha_open.iloc[first]
        )

        second_green = (
            ha_close.iloc[second]
            > ha_open.iloc[second]
        )

        # Higher close
        higher_close = (
            ha_close.iloc[second]
            > ha_close.iloc[first]
        )

        # Higher high
        higher_high = (
            ha_high.iloc[second]
            > ha_high.iloc[first]
        )

        # =================================================
        # LOWER WICK
        # =================================================

        first_body = abs(
            ha_close.iloc[first]
            - ha_open.iloc[first]
        )

        second_body = abs(
            ha_close.iloc[second]
            - ha_open.iloc[second]
        )

        first_lower_wick = (
            min(
                ha_open.iloc[first],
                ha_close.iloc[first]
            )
            - ha_low.iloc[first]
        )

        second_lower_wick = (
            min(
                ha_open.iloc[second],
                ha_close.iloc[second]
            )
            - ha_low.iloc[second]
        )

        if first_body > 0:

            first_wick_ok = (
                first_lower_wick
                <=
                first_body
                * ha_lower_wick_limit
            )

        else:

            first_wick_ok = False

        if second_body > 0:

            second_wick_ok = (
                second_lower_wick
                <=
                second_body
                * ha_lower_wick_limit
            )

        else:

            second_wick_ok = False

        # =================================================
        # FINAL HA SIGNAL
        # =================================================

        ha_bullish_signal = (
            first_green
            and
            second_green
            and
            higher_close
            and
            higher_high
            and
            first_wick_ok
            and
            second_wick_ok
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

            "ema_bullish_cross":
                ema_bullish_cross,

            "ha_bullish_signal":
                ha_bullish_signal
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
        "3m": "3",
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

    return (
        "https://www.tradingview.com/chart/"
        f"?symbol=BINANCE:{symbol}"
        f"&interval={tv_interval}"
    )


# =========================================================
# SCAN MARKET
# =========================================================

def scan_market(
    symbols,
    required_timeframes
):

    market_data = {}

    total = (
        len(symbols)
        *
        len(required_timeframes)
    )

    completed = 0

    progress = st.progress(0)

    status = st.empty()

    for interval in required_timeframes:

        results = []

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

                    result = future.result()

                    if result is not None:

                        results.append(
                            result
                        )

                except Exception:

                    pass

                completed += 1

                progress.progress(
                    completed / total
                )

                status.write(
                    f"Scanning {interval} "
                    f"— {completed}/{total}"
                )

        market_data[interval] = results

    progress.empty()

    status.empty()

    return market_data


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

        rsi = item["rsi"]

        if rsi is None:
            continue

        if not is_in_range(
            rsi,
            primary_rsi_range
        ):
            continue

        if (
            direction_filter == "Rising"
            and
            item["direction"] != "🟢 Rising"
        ):
            continue

        if (
            direction_filter == "Falling"
            and
            item["direction"] != "🔴 Falling"
        ):
            continue

        confirmation = confirmation_map.get(
            symbol
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
                "Coin":
                    get_tradingview_url(
                        symbol,
                        primary_timeframe
                    ),

                "Primary RSI":
                    rsi,

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
# EMA RESULTS
# =========================================================

def build_ema_results(
    market_data
):

    results = []

    for interval in ema_timeframes:

        for item in market_data.get(
            interval,
            []
        ):

            if not item[
                "ema_bullish_cross"
            ]:
                continue

            symbol = item["symbol"]

            results.append(
                {
                    "Coin":
                        get_tradingview_url(
                            symbol,
                            interval
                        ),

                    "Timeframe":
                        interval,

                    "Signal":
                        "🟢 EMA 9 → EMA 33",

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
                            symbol,
                            interval
                        )
                }
            )

    return results


# =========================================================
# HEIKIN ASHI RESULTS
# =========================================================

def build_ha_results(
    market_data
):

    ha_map = {}

    # Create symbol/timeframe map
    for interval in ha_timeframes:

        for item in market_data.get(
            interval,
            []
        ):

            symbol = item["symbol"]

            if symbol not in ha_map:

                ha_map[symbol] = {}

            ha_map[symbol][interval] = item

    results = []

    for symbol, timeframe_data in ha_map.items():

        bullish_count = 0

        bullish_timeframes = []

        row = {}

        # ---------------------------------------------
        # Check all selected HA timeframes
        # ---------------------------------------------

        for interval in ha_timeframes:

            item = timeframe_data.get(
                interval
            )

            if (
                item is not None
                and
                item["ha_bullish_signal"]
            ):

                row[interval] = "🟢"

                bullish_count += 1

                bullish_timeframes.append(
                    interval
                )

            else:

                row[interval] = "⚪"

        # ---------------------------------------------
        # Alignment filter
        # ---------------------------------------------

        if bullish_count < ha_min_alignment:

            continue

        # ---------------------------------------------
        # Coin clickable
        # ---------------------------------------------

        if bullish_timeframes:

            chart_tf = (
                bullish_timeframes[0]
            )

        else:

            chart_tf = "15m"

        row["Coin"] = (
            get_tradingview_url(
                symbol,
                chart_tf
            )
        )

        row["Bullish TFs"] = (
            bullish_count
        )

        row["Alignment"] = (
            f"{bullish_count}/"
            f"{len(ha_timeframes)}"
        )

        row["Bullish Timeframes"] = (
            ", ".join(
                bullish_timeframes
            )
        )

        # ---------------------------------------------
        # Price / RSI
        # ---------------------------------------------

        price_item = None

        preferred_timeframes = [
            primary_timeframe,
            "15m",
            "5m",
            "1h",
            "4h"
        ]

        for tf in preferred_timeframes:

            if tf in timeframe_data:

                price_item = (
                    timeframe_data[tf]
                )

                break

        if price_item is not None:

            row["Price"] = (
                price_item["price"]
            )

            row["RSI"] = (
                price_item["rsi"]
            )

        else:

            row["Price"] = None
            row["RSI"] = None

        row["TradingView"] = (
            get_tradingview_url(
                symbol,
                chart_tf
            )
        )

        results.append(row)

    return results


# =========================================================
# DISPLAY RSI
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
            "No coins matched the RSI filters."
        )

        return

    df = df.sort_values(
        "Primary RSI",
        ascending=False
    )

    st.success(
        f"RSI Results: {len(df)} coins"
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
# DISPLAY EMA
# =========================================================

def show_ema_results(
    results
):

    st.subheader(
        "🟢 EMA 9 / EMA 33 Fresh Bullish Cross"
    )

    df = pd.DataFrame(
        results
    )

    if df.empty:

        st.info(
            "No fresh EMA 9/33 bullish cross found."
        )

        return

    st.success(
        f"Fresh EMA Crosses: {len(df)}"
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
    )


# =========================================================
# DISPLAY HEIKIN ASHI
# =========================================================

def show_ha_results(
    results
):

    st.subheader(
        "🕯️ Multi-Timeframe Heikin Ashi Bullish Alignment"
    )

    df = pd.DataFrame(
        results
    )

    if df.empty:

        st.info(
            "No coins currently meet the selected "
            "HA alignment."
        )

        return

    df = df.sort_values(
        "Bullish TFs",
        ascending=False
    )

    st.success(
        f"HA Results: {len(df)} coins"
    )

    column_config = {

        "Coin":
            st.column_config.LinkColumn(
                "Coin",
                display_text=(
                    r".*symbol=BINANCE:(.*?)&interval=.*"
                ),
                pinned=True
            ),

        "Bullish TFs":
            st.column_config.NumberColumn(
                "Bullish TFs"
            ),

        "Alignment":
            st.column_config.TextColumn(
                "Alignment"
            ),

        "Bullish Timeframes":
            st.column_config.TextColumn(
                "Bullish Timeframes"
            ),

        "Price":
            st.column_config.NumberColumn(
                "Current Price",
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

    # Add HA timeframe columns
    for interval in ha_timeframes:

        column_config[interval] = (
            st.column_config.TextColumn(
                interval,
                help="🟢 = Bullish HA signal"
            )
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )


# =========================================================
# MAIN SCANNER
# =========================================================

def run_scanner():

    try:

        # ---------------------------------------------
        # Get Binance symbols
        # ---------------------------------------------

        symbols = get_symbols()

        st.info(
            f"Found {len(symbols)} "
            f"Binance USDT Spot pairs."
        )

        # ---------------------------------------------
        # Required timeframes
        # ---------------------------------------------

        required = set()

        required.add(
            primary_timeframe
        )

        required.add(
            confirmation_timeframe
        )

        for tf in ema_timeframes:

            required.add(tf)

        for tf in ha_timeframes:

            required.add(tf)

        timeframe_order = [
            "3m",
            "5m",
            "15m",
            "1h",
            "4h"
        ]

        required = sorted(
            required,
            key=lambda x:
            timeframe_order.index(x)
        )

        # ---------------------------------------------
        # Scan Binance
        # ---------------------------------------------

        market_data = scan_market(
            symbols,
            required
        )

        # ---------------------------------------------
        # RSI
        # ---------------------------------------------

        rsi_results = build_rsi_results(
            market_data
        )

        show_rsi_results(
            rsi_results
        )

        # ---------------------------------------------
        # EMA
        # ---------------------------------------------

        ema_results = build_ema_results(
            market_data
        )

        show_ema_results(
            ema_results
        )

        # ---------------------------------------------
        # HEIKIN ASHI
        # ---------------------------------------------

        if ha_timeframes:

            ha_results = build_ha_results(
                market_data
            )

            show_ha_results(
                ha_results
            )

        else:

            st.warning(
                "Select at least one "
                "Heikin Ashi timeframe."
            )

        # ---------------------------------------------
        # Scan time
        # ---------------------------------------------

        st.caption(
            "Last scan: "
            + datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    except requests.RequestException:

        st.error(
            "Binance connection error. "
            "Please try again."
        )

    except Exception as error:

        st.error(
            f"Scanner error: {error}"
        )


# =========================================================
# ALWAYS VISIBLE SCAN BUTTON
# =========================================================

st.divider()

scan_now = st.button(
    "🔍 SCAN ALL NOW",
    type="primary",
    use_container_width=True
)

if scan_now:

    run_scanner()


# =========================================================
# AUTO REFRESH
# =========================================================

if auto_refresh:

    st.info(
        f"🔄 Auto Refresh ON — "
        f"Automatic scan every {refresh_minutes} minutes."
    )

    @st.fragment(
        run_every=f"{refresh_minutes}m"
    )
    def automatic_scanner():

        run_scanner()

    automatic_scanner()

