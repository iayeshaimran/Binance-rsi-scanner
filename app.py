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
    page_title="Binance RSI Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Binance RSI Scanner")
st.caption(
    "Binance USDT Spot RSI Scanner with multi-timeframe confirmation"
)

# =========================================================
# FILTERS
# =========================================================

timeframe = st.selectbox(
    "Select Primary Timeframe",
    ["5m", "15m", "1h", "4h"],
    index=1
)

confirmation_timeframe = st.selectbox(
    "Select Confirmation Timeframe",
    ["5m", "15m", "1h", "4h"],
    index=0
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

auto_refresh = st.checkbox(
    "🔄 Auto Refresh"
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
# RSI RANGE FILTER
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
# GET COIN RSI
# =========================================================

def get_coin_rsi(symbol, interval):

    url = (
        "https://data-api.binance.vision"
        "/api/v3/klines"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 100
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

        if len(candles) < 16:
            return None

        # Use only closed candles
        if use_closed_candles:
            candles = candles[:-1]

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
# TRADINGVIEW URL
# =========================================================

def get_tradingview_interval(interval):

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


def get_tradingview_url(symbol):

    tv_interval = get_tradingview_interval(
        timeframe
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
# PRIMARY SCAN
# =========================================================

def scan_primary(symbols):

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

        for future in as_completed(futures):

            symbol = futures[future]

            try:

                data = future.result()

                if data is not None:

                    # RSI range filter
                    if not is_in_range(
                        data["current"],
                        rsi_range
                    ):

                        completed += 1

                        progress.progress(
                            completed / total
                        )

                        status.write(
                            f"Primary scan: "
                            f"{completed}/{total}"
                        )

                        continue

                    # Direction filter
                    if (
                        direction_filter == "Rising"
                        and data["direction"]
                        != "🟢 Rising"
                    ):

                        completed += 1

                        progress.progress(
                            completed / total
                        )

                        status.write(
                            f"Primary scan: "
                            f"{completed}/{total}"
                        )

                        continue

                    if (
                        direction_filter == "Falling"
                        and data["direction"]
                        != "🔴 Falling"
                    ):

                        completed += 1

                        progress.progress(
                            completed / total
                        )

                        status.write(
                            f"Primary scan: "
                            f"{completed}/{total}"
                        )

                        continue

                    results.append(
                        {
                            "Coin": symbol,
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
# CONFIRMATION SCAN
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

        for future in as_completed(futures):

            item = futures[future]

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

                        status.write(
                            f"Confirmation scan: "
                            f"{completed}/{total}"
                        )

                        continue

                    symbol = item["Coin"]

                    tradingview_url = (
                        get_tradingview_url(
                            symbol
                        )
                    )

                    results.append(
                        {
                            "Coin": symbol,

                            "Primary RSI":
                                item["Primary RSI"],

                            "Previous RSI":
                                item["Previous RSI"],

                            "Direction":
                                item["Direction"],

                            "Confirmation RSI":
                                data["current"],

                            "TradingView":
                                tradingview_url
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
# COMPLETE BINANCE SCAN
# =========================================================

def scan_binance():

    symbols = get_symbols()

    st.info(
        f"Found {len(symbols)} "
        f"Binance USDT Spot pairs."
    )

    primary_results = scan_primary(
        symbols
    )

    st.info(
        f"Primary RSI matched "
        f"{len(primary_results)} coins."
    )

    results = scan_confirmation(
        primary_results
    )

    return results


# =========================================================
# SHOW RESULTS
# =========================================================

def show_results(results):

    df = pd.DataFrame(results)

    if df.empty:

        st.warning(
            "No coins found."
        )

        return

    # Sort by primary RSI
    df = df.sort_values(
        "Primary RSI",
        ascending=False
    )

    # Convert coin to TradingView URL
    df["Coin"] = df.apply(
        lambda row:
        get_tradingview_url(
            row["Coin"]
        ),
        axis=1
    )

    st.success(
        f"Scan completed! "
        f"Found {len(df)} coins."
    )

    st.caption(
        "Last updated: "
        + datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    st.dataframe(

        df,

        use_container_width=True,

        hide_index=True,

        column_config={

            # =================================================
            # COIN NAME WILL BE SHOWN
            # AND WILL BE CLICKABLE
            # =================================================

            "Coin":
                st.column_config.LinkColumn(
                    "Coin",
                    help=(
                        "Click coin name "
                        "to open TradingView"
                    ),

                    display_text=(
                        r".*symbol=BINANCE%3A(.*?)&interval=.*"
                    ),

                    pinned=True
                ),

            "Primary RSI":
                st.column_config.NumberColumn(
                    f"{timeframe} RSI",
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
# MANUAL SCAN
# =========================================================

if not auto_refresh:

    if st.button(
        "🔍 Scan Binance",
        type="primary"
    ):

        try:

            results = scan_binance()

            show_results(
                results
            )

        except requests.RequestException:

            st.error(
                "Binance connection timed out. "
                "Please try again."
            )

        except Exception as error:

            st.error(
                f"Scanner error: {error}"
            )


# =========================================================
# AUTO REFRESH
# =========================================================

if auto_refresh:

    st.info(
        f"🔄 Auto Refresh ON — "
        f"scanner refreshes every "
        f"{refresh_minutes} minutes."
    )

    @st.fragment(
        run_every=f"{refresh_minutes}m"
    )
    def automatic_scanner():

        try:

            results = scan_binance()

            show_results(
                results
            )

        except requests.RequestException:

            st.warning(
                "Binance temporarily timed out. "
                "The next scan will retry."
            )

        except Exception as error:

            st.warning(
                f"Temporary scanner issue: "
                f"{error}"
            )

    automatic_scanner()
