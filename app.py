import streamlit as st
import requests
import pandas as pd
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Crypto Signal Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #080d16;
    color: white;
}

.block-container {
    max-width: 1500px;
    padding-top: 1rem;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
}

.subtitle {
    color: #8b96a8;
    font-size: 14px;
}

.card {
    background: linear-gradient(145deg,#111a29,#0b121e);
    border: 1px solid #26344b;
    border-radius: 14px;
    padding: 16px;
}

.card-title {
    font-size: 14px;
    color: #8b96a8;
}

.card-value {
    font-size: 27px;
    font-weight: 800;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    margin-top: 12px;
}

.section-subtitle {
    color: #8995a7;
    margin-bottom: 15px;
}

.signal-green {
    color: #00e676;
    font-weight: 800;
}

.signal-red {
    color: #ff5252;
    font-weight: 800;
}

.signal-yellow {
    color: #ffd740;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "rsi_results": [],
    "ema_results": [],
    "ha_results": [],
    "last_rsi_scan": "",
    "last_ema_scan": "",
    "last_ha_scan": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CONSTANTS
# =========================================================

BASE_URL = "https://data-api.binance.vision"

# 20 workers can trigger Binance 429 rate limits, 12 is safer
MAX_WORKERS = 12

TV_INTERVAL = {
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240"
}

RSI_RANGES = ["All", "40 - 50", "50 - 55", "55 - 60", "60 - 70", "70+"]

COIN_LINK = st.column_config.LinkColumn(
    "Coin",
    display_text=r".*symbol=BINANCE:(.*?)&interval=.*"
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
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry)

        session.mount("https://", adapter)

        thread_local.session = session

    return thread_local.session


# =========================================================
# TRADINGVIEW
# =========================================================

def tradingview_url(symbol, interval):

    tv_interval = TV_INTERVAL.get(interval, "15")

    return (
        "https://www.tradingview.com/chart/"
        f"?symbol=BINANCE:{symbol}"
        f"&interval={tv_interval}"
    )


# =========================================================
# SYMBOLS
# =========================================================

@st.cache_data(ttl=300)
def get_symbols():

    try:

        response = get_session().get(
            f"{BASE_URL}/api/v3/exchangeInfo",
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
                symbols.append(item["symbol"])

        return sorted(symbols)

    except Exception:

        return []


# =========================================================
# KLINES
# =========================================================

def get_klines(symbol, interval, limit=100, closed_only=True):

    try:

        response = get_session().get(
            f"{BASE_URL}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            },
            timeout=(5, 15)
        )

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
# PARALLEL RUNNER (progress bar + "scanned X/Y" text)
# =========================================================

def run_parallel(symbols, worker, label):

    results = []
    total = len(symbols)

    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [executor.submit(worker, s) for s in symbols]

        for done, future in enumerate(as_completed(futures), 1):

            try:
                result = future.result()
            except Exception:
                result = None

            if result is not None:
                results.append(result)

            progress.progress(done / max(total, 1))
            status.caption(f"{label} {done}/{total}")

    progress.empty()
    status.empty()

    return results


# =========================================================
# RSI  (Wilder smoothing, same as TradingView)
# =========================================================

def calculate_rsi(closes, period=14):

    delta = closes.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)

    return 100 - (100 / (1 + rs))


def rsi_range_match(value, selected):

    if selected == "All":
        return True

    if selected == "40 - 50":
        return 40 <= value < 50

    if selected == "50 - 55":
        return 50 <= value < 55

    if selected == "55 - 60":
        return 55 <= value < 60

    if selected == "60 - 70":
        return 60 <= value < 70

    if selected == "70+":
        return value >= 70

    return False


# =========================================================
# EMA
# =========================================================

def calculate_ema(closes, period):

    return closes.ewm(span=period, adjust=False).mean()


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
        ha_open.iloc[i] = (
            ha_open.iloc[i - 1] + ha_close.iloc[i - 1]
        ) / 2

    ha_high = pd.concat(
        [highs, ha_open, ha_close], axis=1
    ).max(axis=1)

    ha_low = pd.concat(
        [lows, ha_open, ha_close], axis=1
    ).min(axis=1)

    return ha_open, ha_high, ha_low, ha_close


def ha_bullish_signal(candles, wick_limit):

    try:

        ha_open, ha_high, ha_low, ha_close = calculate_heikin_ashi(candles)

        a = -2
        b = -1

        first_green = ha_close.iloc[a] > ha_open.iloc[a]
        second_green = ha_close.iloc[b] > ha_open.iloc[b]

        higher_close = ha_close.iloc[b] > ha_close.iloc[a]
        higher_high = ha_high.iloc[b] > ha_high.iloc[a]

        body_a = abs(ha_close.iloc[a] - ha_open.iloc[a])
        body_b = abs(ha_close.iloc[b] - ha_open.iloc[b])

        if body_a <= 0 or body_b <= 0:
            return False

        wick_a = min(ha_open.iloc[a], ha_close.iloc[a]) - ha_low.iloc[a]
        wick_b = min(ha_open.iloc[b], ha_close.iloc[b]) - ha_low.iloc[b]

        wick_a_ok = wick_a <= body_a * wick_limit
        wick_b_ok = wick_b <= body_b * wick_limit

        return (
            first_green
            and second_green
            and higher_close
            and higher_high
            and wick_a_ok
            and wick_b_ok
        )

    except Exception:

        return False


# =========================================================
# RSI SCANNER
# =========================================================

def scan_rsi(
    primary_tf,
    confirmation_tf,
    primary_range,
    confirmation_range,
    direction,
    closed_only
):

    symbols = get_symbols()

    # ---------- Confirmation timeframe ----------

    def confirmation_worker(symbol):

        candles = get_klines(symbol, confirmation_tf, 100, closed_only)

        if candles is None:
            return None

        closes = pd.Series([float(c[4]) for c in candles])

        rsi = calculate_rsi(closes).iloc[-1]

        if pd.isna(rsi):
            return None

        return symbol, float(rsi)

    confirmation_data = dict(
        run_parallel(symbols, confirmation_worker, "RSI confirmation")
    )

    # ---------- Primary timeframe ----------

    def primary_worker(symbol):

        confirmation = confirmation_data.get(symbol)

        if confirmation is None:
            return None

        if not rsi_range_match(confirmation, confirmation_range):
            return None

        candles = get_klines(symbol, primary_tf, 100, closed_only)

        if candles is None:
            return None

        closes = pd.Series([float(c[4]) for c in candles])

        rsi_values = calculate_rsi(closes)

        current = rsi_values.iloc[-1]
        previous = rsi_values.iloc[-2]

        if pd.isna(current) or pd.isna(previous):
            return None

        current = float(current)
        previous = float(previous)

        if not rsi_range_match(current, primary_range):
            return None

        if current > previous:
            direction_text = "🟢 Rising"
        elif current < previous:
            direction_text = "🔴 Falling"
        else:
            direction_text = "⚪ Flat"

        if direction == "Rising" and direction_text != "🟢 Rising":
            return None

        if direction == "Falling" and direction_text != "🔴 Falling":
            return None

        return {
            "Coin": tradingview_url(symbol, primary_tf),
            "RSI": round(current, 2),
            "Previous RSI": round(previous, 2),
            "Direction": direction_text,
            "Confirm RSI": round(confirmation, 2),
            "Price": float(closes.iloc[-1])
        }

    results = run_parallel(symbols, primary_worker, "RSI primary")

    results.sort(key=lambda x: x["RSI"], reverse=True)

    return results


# =========================================================
# EMA SCANNER
# =========================================================

def scan_ema(timeframes, closed_only):

    symbols = get_symbols()

    results = []

    for timeframe in timeframes:

        def worker(symbol, timeframe=timeframe):

            candles = get_klines(symbol, timeframe, 100, closed_only)

            if candles is None:
                return None

            closes = pd.Series([float(c[4]) for c in candles])

            ema9 = calculate_ema(closes, 9)
            ema33 = calculate_ema(closes, 33)

            previous_9 = float(ema9.iloc[-2])
            previous_33 = float(ema33.iloc[-2])

            current_9 = float(ema9.iloc[-1])
            current_33 = float(ema33.iloc[-1])

            # Fresh bullish crossover
            fresh_cross = (
                previous_9 <= previous_33
                and current_9 > current_33
            )

            if not fresh_cross:
                return None

            return {
                "Coin": tradingview_url(symbol, timeframe),
                "Timeframe": timeframe,
                "Signal": "🟢 Fresh Bullish Cross",
                "Price": float(closes.iloc[-1]),
                "EMA 9": current_9,
                "EMA 33": current_33
            }

        results.extend(
            run_parallel(symbols, worker, f"EMA {timeframe}")
        )

    return results


# =========================================================
# HEIKIN ASHI SCANNER
# =========================================================

def scan_heikin_ashi(timeframes, minimum_alignment, wick_limit, closed_only):

    symbols = get_symbols()

    signal_map = {symbol: {} for symbol in symbols}

    for timeframe in timeframes:

        def worker(symbol, timeframe=timeframe):

            candles = get_klines(symbol, timeframe, 100, closed_only)

            if candles is None:
                return symbol, False

            return symbol, ha_bullish_signal(candles, wick_limit)

        for symbol, signal in run_parallel(
            symbols, worker, f"HA {timeframe}"
        ):
            signal_map[symbol][timeframe] = signal

    results = []

    for symbol in symbols:

        data = signal_map[symbol]

        bullish_tfs = [tf for tf in timeframes if data.get(tf, False)]

        count = len(bullish_tfs)

        if count < minimum_alignment:
            continue

        row = {
            "Coin": tradingview_url(
                symbol,
                bullish_tfs[0] if bullish_tfs else timeframes[0]
            )
        }

        for tf in timeframes:
            row[tf] = "🟢" if data.get(tf, False) else "⚪"

        row["Alignment"] = f"{count}/{len(timeframes)}"
        row["Bullish Timeframes"] = ", ".join(bullish_tfs)

        results.append(row)

    results.sort(
        key=lambda x: int(x["Alignment"].split("/")[0]),
        reverse=True
    )

    return results


# =========================================================
# DISPLAY HELPERS
# =========================================================

def show_table(results):

    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True,
        hide_index=True,
        column_config={"Coin": COIN_LINK}
    )


def display_rsi_heatmap(results):

    if not results:
        return

    df = pd.DataFrame(results)

    zones = [
        ("40-50", 40, 50),
        ("50-55", 50, 55),
        ("55-60", 55, 60),
        ("60-70", 60, 70),
        ("70+", 70, 101)
    ]

    cols = st.columns(len(zones))

    for col, (name, low, high) in zip(cols, zones):

        count = len(df[(df["RSI"] >= low) & (df["RSI"] < high)])

        with col:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-value">{count}</div>
                    <div class="card-title">RSI {name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def display_top_movers():

    symbols = get_symbols()

    st.markdown("### 🚀 Top Movers")

    st.caption("Latest 24h Binance USDT Spot movers.")

    try:

        response = get_session().get(
            f"{BASE_URL}/api/v3/ticker/24hr",
            timeout=(10, 20)
        )

        data = response.json()

        rows = []

        allowed = set(symbols)

        for item in data:

            symbol = item.get("symbol", "")

            if symbol not in allowed:
                continue

            try:

                rows.append({
                    "Coin": tradingview_url(symbol, "15m"),
                    "24h %": float(item["priceChangePercent"]),
                    "Price": float(item["lastPrice"]),
                    "Volume": float(item["quoteVolume"])
                })

            except Exception:
                continue

        rows.sort(key=lambda x: x["24h %"], reverse=True)

        top = rows[:10]

        if top:

            st.dataframe(
                pd.DataFrame(top),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Coin": COIN_LINK,
                    "24h %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Price": st.column_config.NumberColumn(format="%.8f"),
                    "Volume": st.column_config.NumberColumn(format="%.0f")
                }
            )

    except Exception:

        st.warning("Unable to load Top Movers.")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">📊 Crypto Signal Scanner</div>
    <div class="subtitle">
        Binance USDT Spot • RSI • EMA 9/33 • Heikin Ashi
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# DASHBOARD CARDS
# =========================================================

symbols = get_symbols()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-value">{len(symbols)}</div>
            <div class="card-title">Binance USDT Pairs</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="card">
            <div class="card-value signal-green">● LIVE</div>
            <div class="card-title">Binance Spot API</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-value">{len(st.session_state.rsi_results)}</div>
            <div class="card-title">RSI Signals</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-value">{len(st.session_state.ema_results)}</div>
            <div class="card-title">EMA Signals</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

display_top_movers()

st.divider()


# =========================================================
# RSI SCANNER UI
# =========================================================

st.markdown(
    '<div class="section-title">📊 RSI Scanner</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Independent RSI scanner</div>',
    unsafe_allow_html=True
)

r1, r2, r3 = st.columns(3)

with r1:
    rsi_primary = st.selectbox(
        "Primary Timeframe",
        ["5m", "15m", "1h", "4h"],
        index=1,
        key="rsi_primary"
    )

with r2:
    rsi_confirmation = st.selectbox(
        "Confirmation Timeframe",
        ["5m", "15m", "1h", "4h"],
        index=0,
        key="rsi_confirmation"
    )

with r3:
    rsi_range = st.selectbox(
        "Primary RSI Range",
        RSI_RANGES,
        index=2,
        key="rsi_range"
    )

r4, r5, r6 = st.columns(3)

with r4:
    rsi_confirm_range = st.selectbox(
        "Confirmation RSI Range",
        RSI_RANGES,
        index=0,
        key="rsi_confirm_range"
    )

with r5:
    rsi_direction = st.selectbox(
        "RSI Direction",
        ["All", "Rising", "Falling"],
        key="rsi_direction"
    )

with r6:
    rsi_closed = st.checkbox(
        "Closed candles only",
        True,
        key="rsi_closed"
    )

r7, r8 = st.columns(2)

with r7:
    rsi_auto = st.checkbox(
        "🔄 Auto Refresh RSI",
        False,
        key="rsi_auto"
    )

with r8:
    rsi_refresh = st.selectbox(
        "RSI Refresh Time",
        [1, 2, 5, 10, 15, 30],
        index=2,
        format_func=lambda x: f"Every {x} minute",
        disabled=not rsi_auto,
        key="rsi_refresh"
    )

scan_rsi_button = st.button(
    "🔍 SCAN RSI",
    type="primary",
    use_container_width=True,
    key="scan_rsi_button"
)


def perform_rsi_scan():

    with st.spinner("Scanning RSI..."):

        st.session_state.rsi_results = scan_rsi(
            rsi_primary,
            rsi_confirmation,
            rsi_range,
            rsi_confirm_range,
            rsi_direction,
            rsi_closed
        )

        st.session_state.last_rsi_scan = time.strftime("%H:%M:%S")


def display_rsi():

    if st.session_state.rsi_results:
        display_rsi_heatmap(st.session_state.rsi_results)
        show_table(st.session_state.rsi_results)


if scan_rsi_button:
    perform_rsi_scan()

if rsi_auto:

    @st.fragment(run_every=f"{rsi_refresh}m")
    def rsi_auto_fragment():
        perform_rsi_scan()
        display_rsi()

    rsi_auto_fragment()

else:

    display_rsi()

if st.session_state.last_rsi_scan:
    st.caption(f"Last RSI scan: {st.session_state.last_rsi_scan}")

st.divider()


# =========================================================
# EMA SCANNER UI
# =========================================================

st.markdown(
    '<div class="section-title">📈 EMA 9 / EMA 33 Scanner</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Fresh bullish EMA 9 crossing EMA 33 only.'
    '</div>',
    unsafe_allow_html=True
)

ema_timeframes = st.multiselect(
    "EMA Timeframes",
    ["5m", "15m", "1h", "4h"],
    default=["5m", "15m", "1h", "4h"],
    key="ema_timeframes"
)

e1, e2 = st.columns(2)

with e1:
    ema_closed = st.checkbox(
        "Closed candles only",
        True,
        key="ema_closed"
    )

with e2:
    ema_auto = st.checkbox(
        "🔄 Auto Refresh EMA",
        False,
        key="ema_auto"
    )

ema_refresh = st.selectbox(
    "EMA Refresh Time",
    [1, 2, 5, 10, 15, 30],
    index=2,
    format_func=lambda x: f"Every {x} minute",
    disabled=not ema_auto,
    key="ema_refresh"
)

scan_ema_button = st.button(
    "🔍 SCAN EMA 9/33",
    type="primary",
    use_container_width=True,
    key="scan_ema_button"
)


def perform_ema_scan():

    if not ema_timeframes:
        st.warning("Select at least one EMA timeframe.")
        return

    with st.spinner("Scanning EMA 9/33..."):

        st.session_state.ema_results = scan_ema(
            ema_timeframes,
            ema_closed
        )

        st.session_state.last_ema_scan = time.strftime("%H:%M:%S")


def display_ema():

    if st.session_state.ema_results:
        show_table(st.session_state.ema_results)


if scan_ema_button:
    perform_ema_scan()

if ema_auto and ema_timeframes:

    @st.fragment(run_every=f"{ema_refresh}m")
    def ema_auto_fragment():
        perform_ema_scan()
        display_ema()

    ema_auto_fragment()

else:

    display_ema()

if st.session_state.last_ema_scan:
    st.caption(f"Last EMA scan: {st.session_state.last_ema_scan}")

st.divider()


# =========================================================
# HEIKIN ASHI UI
# =========================================================

st.markdown(
    '<div class="section-title">🕯️ Heikin Ashi Scanner</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Multi-timeframe bullish Heikin Ashi alignment.'
    '</div>',
    unsafe_allow_html=True
)

ha_timeframes = st.multiselect(
    "HA Timeframes",
    ["3m", "5m", "15m", "1h", "4h"],
    default=["3m", "5m", "15m", "1h", "4h"],
    key="ha_timeframes"
)

h1, h2, h3 = st.columns(3)

with h1:
    ha_wick = st.slider(
        "Maximum lower wick / body",
        0.0,
        1.0,
        0.25,
        0.05,
        key="ha_wick"
    )

with h2:
    max_alignment = max(len(ha_timeframes), 1)

    ha_min = st.selectbox(
        "Minimum Bullish Alignment",
        list(range(1, max_alignment + 1)),
        index=min(2, max_alignment - 1),
        key="ha_min"
    )

with h3:
    ha_closed = st.checkbox(
        "Closed candles only",
        True,
        key="ha_closed"
    )

ha_auto = st.checkbox(
    "🔄 Auto Refresh Heikin Ashi",
    False,
    key="ha_auto"
)

ha_refresh = st.selectbox(
    "HA Refresh Time",
    [1, 2, 5, 10, 15, 30],
    index=0,
    format_func=lambda x: f"Every {x} minute",
    disabled=not ha_auto,
    key="ha_refresh"
)

scan_ha_button = st.button(
    "🔍 SCAN HEIKIN ASHI",
    type="primary",
    use_container_width=True,
    key="scan_ha_button"
)


def perform_ha_scan():

    if not ha_timeframes:
        st.warning("Select at least one HA timeframe.")
        return

    with st.spinner("Scanning Heikin Ashi..."):

        st.session_state.ha_results = scan_heikin_ashi(
            ha_timeframes,
            ha_min,
            ha_wick,
            ha_closed
        )

        st.session_state.last_ha_scan = time.strftime("%H:%M:%S")


def display_ha():

    if st.session_state.ha_results:
        show_table(st.session_state.ha_results)
    else:
        st.info("No Heikin Ashi signals yet.")


if scan_ha_button:
    perform_ha_scan()

if ha_auto and ha_timeframes:

    @st.fragment(run_every=f"{ha_refresh}m")
    def ha_auto_fragment():
        perform_ha_scan()
        display_ha()

    ha_auto_fragment()

else:

    display_ha()

if st.session_state.last_ha_scan:
    st.caption(f"Last HA scan: {st.session_state.last_ha_scan}")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Crypto Signal Scanner • Binance public Spot API • "
    "Technical signals only."
)

