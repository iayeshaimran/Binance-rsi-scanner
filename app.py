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
st.caption("Binance USDT Spot Scanner — RSI + EMA 9/33 Bullish Cross")

TIMEFRAMES = ["5m", "15m", "1h", "4h"]
RSI_RANGES = ["All", "40 - 50", "50 - 55", "55 - 60", "60 - 70", "70+"]

KLINE_LIMIT = 300      # longer history -> Wilder RSI converges better
MIN_CANDLES = 100      # minimum candles required from Binance
MAX_WORKERS = 20


# =========================================================
# RSI SETTINGS
# =========================================================

st.header("📊 RSI Scanner")

primary_timeframe = st.selectbox(
    "Select Primary Timeframe", TIMEFRAMES, index=1
)

confirmation_timeframe = st.selectbox(
    "Select Confirmation Timeframe", TIMEFRAMES, index=0
)

primary_rsi_range = st.selectbox(
    "Select Primary RSI Range", RSI_RANGES, index=2
)

confirmation_rsi_range = st.selectbox(
    "Confirmation RSI Range", RSI_RANGES, index=0
)

direction_filter = st.selectbox(
    "RSI Direction (primary)", ["All", "Rising", "Falling"], index=0
)

use_closed_candles = st.checkbox(
    "Use closed candles only (recommended)", value=True
)


# =========================================================
# EMA SETTINGS
# =========================================================

st.header("🟢 EMA 9 / EMA 33 Bullish Cross")

ema_timeframes = st.multiselect(
    "EMA Bullish Cross Timeframes",
    TIMEFRAMES,
    default=TIMEFRAMES
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

auto_refresh = st.checkbox("Auto Refresh")

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
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)

        thread_local.session = session

    return thread_local.session


# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(closes, period=14):
    """RSI with Wilder's smoothing (matches TradingView)."""

    delta = closes.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Not enough data for the first `period` candles
    rsi.iloc[:period] = float("nan")

    return rsi


def calculate_ema(closes, period):

    return closes.ewm(span=period, adjust=False).mean()


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
# GET BINANCE SYMBOLS
# =========================================================

@st.cache_data(ttl=300)
def get_symbols():

    url = "https://data-api.binance.vision/api/v3/exchangeInfo"

    session = get_session()

    response = session.get(url, timeout=(10, 30))
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
# GET KLINES
# =========================================================

def get_klines(symbol, interval, closed_only, limit=KLINE_LIMIT):

    url = "https://data-api.binance.vision/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:

        session = get_session()

        response = session.get(url, params=params, timeout=(5, 12))

        if response.status_code != 200:
            return None

        candles = response.json()

        # Remove currently forming candle
        if closed_only:
            candles = candles[:-1]

        if len(candles) < MIN_CANDLES:
            return None

        return candles

    except Exception:

        return None


# =========================================================
# ANALYZE ONE COIN / ONE TIMEFRAME
# =========================================================

def analyze_coin(symbol, interval, closed_only):

    candles = get_klines(symbol, interval, closed_only)

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
            direction = "flat"

        else:

            current_rsi = round(float(current_rsi), 2)
            previous_rsi = round(float(previous_rsi), 2)

            if current_rsi > previous_rsi:
                direction = "rising"
            elif current_rsi < previous_rsi:
                direction = "falling"
            else:
                direction = "flat"

        # ---------------- EMA ----------------

        ema9 = calculate_ema(closes, 9)
        ema33 = calculate_ema(closes, 33)

        previous_ema9 = float(ema9.iloc[-2])
        previous_ema33 = float(ema33.iloc[-2])
        current_ema9 = float(ema9.iloc[-1])
        current_ema33 = float(ema33.iloc[-1])

        # Fresh bullish cross:
        # previous candle EMA9 <= EMA33, latest candle EMA9 > EMA33
        bullish_cross = (
            previous_ema9 <= previous_ema33
            and current_ema9 > current_ema33
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
            "bullish_cross": bullish_cross
        }

    except Exception:

        return None


# =========================================================
# TRADINGVIEW URL
# =========================================================

def get_tradingview_url(symbol, interval):

    mapping = {"5m": "5", "15m": "15", "1h": "60", "4h": "240"}

    tv_interval = mapping.get(interval, "15")

    return (
        "https://www.tradingview.com/chart/"
        f"?symbol=BINANCE:{symbol}"
        f"&interval={tv_interval}"
    )


DIRECTION_LABELS = {
    "rising": "🟢 Rising",
    "falling": "🔴 Falling",
    "flat": "⚪ Flat"
}


# =========================================================
# SCAN ONE TIMEFRAME
# =========================================================

def scan_timeframe(symbols, interval, closed_only, on_progress=None):

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(analyze_coin, symbol, interval, closed_only)
            for symbol in symbols
        ]

        for future in as_completed(futures):

            try:
                data = future.result()
                if data is not None:
                    results.append(data)
            except Exception:
                pass

            if on_progress is not None:
                on_progress()

    return results


# =========================================================
# SCAN ALL REQUIRED TIMEFRAMES
# =========================================================

def get_required_timeframes():

    required = {primary_timeframe, confirmation_timeframe}
    required.update(ema_timeframes)

    return sorted(required, key=TIMEFRAMES.index)


def scan_market_data(symbols, required_timeframes, closed_only):

    all_data = {}

    progress = st.progress(0)
    status = st.empty()

    total_work = len(required_timeframes) * len(symbols)
    done = {"count": 0}

    for interval in required_timeframes:

        def on_progress(interval=interval):
            done["count"] += 1
            progress.progress(done["count"] / total_work)
            status.write(
                f"Scanning {interval}: {done['count']}/{total_work}"
            )

        all_data[interval] = scan_timeframe(
            symbols, interval, closed_only, on_progress
        )

    progress.empty()
    status.empty()

    return all_data


# =========================================================
# BUILD RESULTS
# =========================================================

def build_rsi_results(market_data):

    primary_data = market_data.get(primary_timeframe, [])
    confirmation_data = market_data.get(confirmation_timeframe, [])

    confirmation_map = {item["symbol"]: item for item in confirmation_data}

    results = []

    for item in primary_data:

        current_rsi = item["rsi"]

        if current_rsi is None:
            continue

        if not is_in_range(current_rsi, primary_rsi_range):
            continue

        if direction_filter == "Rising" and item["direction"] != "rising":
            continue

        if direction_filter == "Falling" and item["direction"] != "falling":
            continue

        confirmation = confirmation_map.get(item["symbol"])

        if confirmation is None or confirmation["rsi"] is None:
            continue

        if not is_in_range(confirmation["rsi"], confirmation_rsi_range):
            continue

        results.append(
            {
                "Coin": item["symbol"],
                "Primary RSI": item["rsi"],
                "Previous RSI": item["previous_rsi"],
                "Direction": DIRECTION_LABELS[item["direction"]],
                "Confirmation RSI": confirmation["rsi"],
                "TradingView": get_tradingview_url(
                    item["symbol"], primary_timeframe
                )
            }
        )

    return results


def build_ema_results(market_data):

    results = []

    for interval in ema_timeframes:

        for item in market_data.get(interval, []):

            if not item["bullish_cross"]:
                continue

            results.append(
                {
                    "Coin": item["symbol"],
                    "Timeframe": interval,
                    "Signal": "🟢 Bullish EMA Cross",
                    "Price": item["price"],
                    "EMA 9": item["ema9"],
                    "EMA 33": item["ema33"],
                    "RSI": item["rsi"],
                    "TradingView": get_tradingview_url(
                        item["symbol"], interval
                    )
                }
            )

    return results


# =========================================================
# SHOW RESULTS
# =========================================================

COIN_LINK_REGEX = r".*symbol=BINANCE:(.*?)&interval=.*"


def show_rsi_results(results):

    st.subheader("📊 RSI Results")

    df = pd.DataFrame(results)

    if df.empty:
        st.info("No coins matched the selected RSI filters.")
        return

    df = df.sort_values("Primary RSI", ascending=False)

    st.success(f"RSI scan completed — {len(df)} coins found.")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Coin": st.column_config.LinkColumn(
                "Coin",
                display_text=COIN_LINK_REGEX,
                pinned=True
            ),
            "Primary RSI": st.column_config.NumberColumn(
                f"{primary_timeframe} RSI", format="%.2f"
            ),
            "Previous RSI": st.column_config.NumberColumn(
                "Previous RSI", format="%.2f"
            ),
            "Direction": st.column_config.TextColumn("RSI Direction"),
            "Confirmation RSI": st.column_config.NumberColumn(
                f"{confirmation_timeframe} RSI", format="%.2f"
            ),
            "TradingView": st.column_config.LinkColumn(
                "TradingView", display_text="Open Chart 🔗"
            )
        }
    )


def show_ema_results(results):

    st.subheader("🟢 EMA 9 / EMA 33 Bullish Cross Alerts")

    df = pd.DataFrame(results)

    if df.empty:
        st.info(
            "No fresh bullish EMA 9/33 cross "
            "found in the selected timeframes."
        )
        return

    df = df.sort_values(["Timeframe", "Coin"])

    st.success(f"Fresh bullish cross found: {len(df)}")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Coin": st.column_config.LinkColumn(
                "Coin",
                help="Click the coin name to open TradingView.",
                display_text=COIN_LINK_REGEX,
                pinned=True
            ),
            "Timeframe": st.column_config.TextColumn("Timeframe"),
            "Signal": st.column_config.TextColumn("Signal"),
            "Price": st.column_config.NumberColumn(
                "Current Price", format="%.8f"
            ),
            "EMA 9": st.column_config.NumberColumn("EMA 9", format="%.8f"),
            "EMA 33": st.column_config.NumberColumn("EMA 33", format="%.8f"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.2f"),
            "TradingView": st.column_config.LinkColumn(
                "TradingView", display_text="Open Chart 🔗"
            )
        }
    )


def render_results():
    """Show results from the last scan. Filters re-apply without rescanning."""

    market_data = st.session_state.get("market_data")

    if market_data is None:
        return

    missing = [
        tf for tf in get_required_timeframes() if tf not in market_data
    ]

    if missing:
        st.warning(
            "You selected timeframes that were not part of the last scan "
            f"({', '.join(missing)}). Please scan again."
        )
        return

    show_rsi_results(build_rsi_results(market_data))
    show_ema_results(build_ema_results(market_data))

    st.caption(
        "Last updated: " + st.session_state.get("last_update", "-")
    )


# =========================================================
# RUN SCANNER
# =========================================================

def run_scanner():

    try:

        symbols = get_symbols()

        st.info(f"Found {len(symbols)} Binance USDT Spot pairs.")

        market_data = scan_market_data(
            symbols,
            get_required_timeframes(),
            use_closed_candles
        )

        st.session_state["market_data"] = market_data
        st.session_state["last_update"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    except requests.RequestException:

        st.error("Binance connection timed out. Please try again.")

    except Exception as error:

        st.error(f"Scanner error: {error}")


# =========================================================
# MANUAL SCAN
# =========================================================

if not auto_refresh:

    if st.button("🔍 Scan Binance", type="primary"):
        run_scanner()

    render_results()


# =========================================================
# AUTO REFRESH
# =========================================================

if auto_refresh:

    st.info(
        f"🔄 Auto Refresh ON — "
        f"scanner refreshes every {refresh_minutes} minutes."
    )

    @st.fragment(run_every=f"{refresh_minutes}m")
    def automatic_scanner():

        run_scanner()
        render_results()

    automatic_scanner()
