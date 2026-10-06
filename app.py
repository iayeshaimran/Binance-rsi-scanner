import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import html
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Crypto Signal Scanner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.stApp{
    background:
      radial-gradient(circle at 20% 0%,rgba(35,80,130,.16),transparent 28%),
      radial-gradient(circle at 80% 0%,rgba(100,45,150,.12),transparent 28%),
      #080b12;
}
.block-container{max-width:1500px;padding-top:1rem;padding-bottom:3rem}
.hero,.panel,.stat{
    background:linear-gradient(145deg,#111927,#0b111b);
    border:1px solid #223047;
    border-radius:16px;
    box-shadow:0 8px 28px rgba(0,0,0,.20);
}
.hero{padding:20px;margin-bottom:16px}
.hero-title{font-size:31px;font-weight:850}
.muted{color:#8d99ab}
.live{color:#3ee98a;font-weight:800}
.stat{padding:15px;min-height:95px}
.stat-value{font-size:25px;font-weight:800}
.stat-label{font-size:12px;color:#8c98aa;margin-top:4px}
.section{font-size:22px;font-weight:850;margin-top:26px;margin-bottom:4px}
.sub{color:#7e8a9d;font-size:13px;margin-bottom:12px}
.coin-grid{display:grid;grid-template-columns:repeat(7,minmax(100px,1fr));gap:7px}
.coin{
    display:block;
    padding:10px 7px;
    border-radius:10px;
    text-align:center;
    color:#fff !important;
    text-decoration:none !important;
    border:1px solid rgba(255,255,255,.10);
    min-height:62px;
}
.coin:hover{transform:translateY(-2px);border-color:#65a9ff}
.red{background:linear-gradient(145deg,#3b211d,#251512)}
.orange{background:linear-gradient(145deg,#332e18,#1d1a10)}
.green{background:linear-gradient(145deg,#123a28,#0c241a)}
.mover{
    display:block;color:#fff !important;text-decoration:none !important;
    background:#101824;border:1px solid #243248;border-radius:11px;
    padding:11px;margin-bottom:7px;
}
.mover:hover{border-color:#65a9ff}
.good{color:#3ee98a;font-weight:800}
.warn{color:#ffd166;font-weight:800}
.bad{color:#ff7868;font-weight:800}
.signal{
    background:linear-gradient(145deg,#121d2b,#0b1119);
    border:1px solid #26364c;border-radius:14px;padding:13px;margin-bottom:8px
}
.score{font-size:24px;font-weight:850}
.score90{color:#3ee98a}.score80{color:#9be564}.score70{color:#ffd166}
.small{font-size:11px;color:#8b97a9}
@media(max-width:1000px){.coin-grid{grid-template-columns:repeat(4,1fr)}}
@media(max-width:600px){.coin-grid{grid-template-columns:repeat(2,1fr)}.hero-title{font-size:24px}}
div.stButton>button{border-radius:10px;font-weight:750}
</style>
""", unsafe_allow_html=True)

# ============================================================
# STATE
# ============================================================

defaults = {
    "rsi_results": [],
    "ema_results": [],
    "ha_results": [],
    "confluence_results": [],
    "heatmap": [],
    "last_signal": None,
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# BINANCE
# ============================================================

BASE = "https://data-api.binance.vision"

@st.cache_data(ttl=300)
def get_symbols():
    try:
        d = requests.get(BASE+"/api/v3/exchangeInfo",timeout=15).json()
        return sorted([
            x["symbol"] for x in d["symbols"]
            if x.get("status")=="TRADING"
            and x.get("quoteAsset")=="USDT"
            and x.get("isSpotTradingAllowed",True)
        ])
    except:
        return []

def get_klines(symbol, tf, limit=100):
    try:
        d = requests.get(
            BASE+"/api/v3/klines",
            params={"symbol":symbol,"interval":tf,"limit":limit},
            timeout=10
        ).json()
        if not isinstance(d,list) or len(d)<35:
            return None
        df=pd.DataFrame(d,columns=[
            "time","open","high","low","close","volume",
            "close_time","quote_volume","trades",
            "tb","tq","ignore"
        ])
        for c in ["open","high","low","close","volume"]:
            df[c]=pd.to_numeric(df[c],errors="coerce")
        return df
    except:
        return None

@st.cache_data(ttl=30)
def get_tickers():
    try:
        d=requests.get(BASE+"/api/v3/ticker/24hr",timeout=15).json()
        rows=[]
        for x in d:
            if not x.get("symbol","").endswith("USDT"):
                continue
            try:
                rows.append({
                    "symbol":x["symbol"],
                    "price":float(x["lastPrice"]),
                    "change":float(x["priceChangePercent"]),
                    "volume":float(x["quoteVolume"])
                })
            except:
                pass
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

# ============================================================
# INDICATORS
# ============================================================

def rsi_series(close,period=14):
    delta=close.diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/period,adjust=False).mean()
    al=loss.ewm(alpha=1/period,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return (100-100/(1+rs)).fillna(50)

def indicators(df):
    x=df.iloc[:-1].copy()
    r=rsi_series(x["close"])
    e9=x["close"].ewm(span=9,adjust=False).mean()
    e33=x["close"].ewm(span=33,adjust=False).mean()
    e200=x["close"].ewm(span=200,adjust=False).mean()
    avgvol=x["volume"].rolling(20).mean()
    vol_ratio=float(x["volume"].iloc[-1]/avgvol.iloc[-1]) if avgvol.iloc[-1] else 0
    high20=float(x["high"].iloc[-21:-1].max())
    price=float(x["close"].iloc[-1])
    return {
        "price":price,
        "rsi":float(r.iloc[-1]),
        "rsi_prev":float(r.iloc[-2]),
        "ema9":float(e9.iloc[-1]),
        "ema33":float(e33.iloc[-1]),
        "ema200":float(e200.iloc[-1]),
        "vol_ratio":vol_ratio,
        "resistance":high20,
        "fresh_cross":bool(e9.iloc[-2]<=e33.iloc[-2] and e9.iloc[-1]>e33.iloc[-1]),
        "trend":price>float(e200.iloc[-1]),
        "breakout":price>high20 and vol_ratio>=1.2,
    }

def ha_signal(df,wick_limit=.25):
    x=df.iloc[:-1].copy()
    hc=(x["open"]+x["high"]+x["low"]+x["close"])/4
    ho=np.zeros(len(x))
    ho[0]=(x["open"].iloc[0]+x["close"].iloc[0])/2
    for i in range(1,len(x)):
        ho[i]=(ho[i-1]+hc.iloc[i-1])/2
    hh=np.maximum.reduce([x["high"].to_numpy(),ho,hc.to_numpy()])
    hl=np.minimum.reduce([x["low"].to_numpy(),ho,hc.to_numpy()])
    a,b=-2,-1
    if not(hc.iloc[a]>ho[a] and hc.iloc[b]>ho[b]):
        return False
    if hc.iloc[b]<=hc.iloc[a] or hh[b]<=hh[a]:
        return False
    ba=abs(hc.iloc[a]-ho[a]);bb=abs(hc.iloc[b]-ho[b])
    if ba==0 or bb==0:return False
    wa=(min(ho[a],hc.iloc[a])-hl[a])/ba
    wb=(min(ho[b],hc.iloc[b])-hl[b])/bb
    return wa<=wick_limit and wb<=wick_limit

def tv(symbol,tf):
    m={"3m":"3","5m":"5","15m":"15","1h":"60","4h":"240"}
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval={m.get(tf,'15')}"

# ============================================================
# CONFLUENCE
# ============================================================

def score_symbol(symbol, tf, use_ema200, use_cross, use_rsi,
                 use_volume, use_breakout, use_ha, wick_limit):
    df=get_klines(symbol,tf,120)
    if df is None:return None
    try:
        x=indicators(df)
        score=0
        checks=[]
        if use_ema200:
            ok=x["trend"]
            score+=20 if ok else 0
            checks.append(("EMA 200",ok))
        if use_cross:
            ok=x["fresh_cross"] or x["ema9"]>x["ema33"]
            score+=20 if ok else 0
            checks.append(("EMA 9/33",ok))
        if use_rsi:
            ok=50<=x["rsi"]<=65 and x["rsi"]>x["rsi_prev"]
            score+=15 if ok else 0
            checks.append(("RSI momentum",ok))
        if use_volume:
            ok=x["vol_ratio"]>=1.5
            score+=20 if ok else 0
            checks.append(("Volume spike",ok))
        if use_breakout:
            ok=x["breakout"]
            score+=15 if ok else 0
            checks.append(("Breakout",ok))
        if use_ha:
            ok=ha_signal(df,wick_limit)
            score+=10 if ok else 0
            checks.append(("Heikin Ashi",ok))
        if score==0:return None
        label="WATCH"
        if score>=90:label="STRONG BUY"
        elif score>=80:label="BUY"
        elif score>=70:label="WATCH"
        return {
            "Coin":symbol,"Score":score,"Signal":label,
            "Price":x["price"],"RSI":x["rsi"],
            "Volume x":x["vol_ratio"],
            "EMA9":x["ema9"],"EMA33":x["ema33"],
            "EMA200":x["ema200"],
            "Breakout":x["breakout"],
            "Checks":checks,
            "TradingView":tv(symbol,tf)
        }
    except:
        return None

def scan_confluence(symbols,tf,minimum_score,settings,wick):
    out=[]
    def one(s):
        z=score_symbol(s,tf,*settings,wick)
        return z if z and z["Score"]>=minimum_score else None
    with ThreadPoolExecutor(max_workers=8) as ex:
        fs=[ex.submit(one,s) for s in symbols]
        for f in as_completed(fs):
            try:
                z=f.result()
                if z:out.append(z)
            except:pass
    return sorted(out,key=lambda x:x["Score"],reverse=True)

# ============================================================
# DASHBOARD
# ============================================================

symbols=get_symbols()
tickers=get_tickers()

st.markdown("""
<div class="hero">
 <div class="hero-title">🚀 Crypto Signal Scanner</div>
 <div class="muted">
 Binance USDT Spot • Confluence • RSI • EMA • Volume • Breakout • HA
 &nbsp;&nbsp;<span class="live">● LIVE</span>
 </div>
</div>
""",unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)
c1.markdown(f'<div class="stat"><div class="stat-value">{len(symbols)}</div><div class="stat-label">USDT Pairs</div></div>',unsafe_allow_html=True)
c2.markdown('<div class="stat"><div class="stat-value live">● LIVE</div><div class="stat-label">Binance API</div></div>',unsafe_allow_html=True)
c3.markdown(f'<div class="stat"><div class="stat-value">{len(st.session_state.confluence_results)}</div><div class="stat-label">Strong Signals</div></div>',unsafe_allow_html=True)
c4.markdown(f'<div class="stat"><div class="stat-value">{len(st.session_state.heatmap)}</div><div class="stat-label">Heatmap Coins</div></div>',unsafe_allow_html=True)

# ============================================================
# TOP MOVERS
# ============================================================

st.markdown('<div class="section">🚀 Top Movers</div><div class="sub">24h Binance USDT Spot movement</div>',unsafe_allow_html=True)

if not tickers.empty:
    top=tickers.sort_values("change",ascending=False).head(8)
    cols=st.columns(4)
    for i,(_,r) in enumerate(top.iterrows()):
        with cols[i%4]:
            st.markdown(
                f'<a class="mover" href="{tv(r["symbol"],"15m")}" target="_blank">'
                f'<b>{r["symbol"]}</b><br><span class="good">+{r["change"]:.2f}%</span><br>'
                f'<span class="small">${r["price"]:.8f}</span></a>',
                unsafe_allow_html=True
            )

# ============================================================
# CONFLUENCE CONTROLS
# ============================================================

st.markdown('<div class="section">🔥 Confluence Signal Scanner</div>',unsafe_allow_html=True)
st.markdown('<div class="sub">Turn indicators ON/OFF and scan only the conditions you want.</div>',unsafe_allow_html=True)

a,b,c=st.columns(3)
with a:
    conf_tf=st.selectbox("Signal timeframe",["5m","15m","1h","4h"],index=1)
    minimum=st.slider("Minimum score",40,100,75,5)
with b:
    use_ema200=st.checkbox("EMA 200 Trend",True)
    use_cross=st.checkbox("EMA 9/33",True)
    use_rsi=st.checkbox("RSI Momentum",True)
with c:
    use_volume=st.checkbox("Volume Spike",True)
    use_breakout=st.checkbox("Breakout",True)
    use_ha=st.checkbox("Heikin Ashi",True)

wick=st.slider("HA maximum lower wick / body",0.0,1.0,.25,.05)

if st.button("🔥 SCAN STRONG SIGNALS",use_container_width=True):
    settings=(use_ema200,use_cross,use_rsi,use_volume,use_breakout,use_ha)
    with st.spinner(f"Scanning {len(symbols)} coins..."):
        st.session_state.confluence_results=scan_confluence(
            symbols,conf_tf,minimum,settings,wick
        )
    st.session_state.last_signal=time.strftime("%H:%M:%S")

# ============================================================
# SIGNAL RESULTS
# ============================================================

results=st.session_state.confluence_results

if results:
    st.success(f"{len(results)} signals found • Last scan {st.session_state.last_signal}")
    for r in results[:20]:
        score=r["Score"]
        cls="score90" if score>=90 else "score80" if score>=80 else "score70"
        checks=" ".join(
            f"{'✅' if ok else '❌'} {name}"
            for name,ok in r["Checks"]
        )
        st.markdown(f"""
        <a class="signal" href="{r["TradingView"]}" target="_blank">
          <b>{r["Coin"]}</b>
          <span class="{cls}" style="float:right">{score}/100</span><br>
          <b>{r["Signal"]}</b>
          &nbsp; RSI {r["RSI"]:.1f} ↑
          &nbsp; Volume {r["Volume x"]:.1f}x<br>
          <span class="small">{checks}</span>
        </a>
        """,unsafe_allow_html=True)

    st.dataframe(
        pd.DataFrame([{
            "Coin":r["Coin"],
            "Score":r["Score"],
            "Signal":r["Signal"],
            "Price":r["Price"],
            "RSI":round(r["RSI"],2),
            "Volume x":round(r["Volume x"],2),
            "Breakout":r["Breakout"],
            "TradingView":r["TradingView"]
        } for r in results]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "TradingView":st.column_config.LinkColumn(
                "TradingView",display_text="Open Chart ↗"
            )
        }
    )
elif st.session_state.last_signal:
    st.info("No signals met the selected score.")

# ============================================================
# RSI HEATMAP
# ============================================================

st.markdown('<div class="section">📊 RSI Heatmap</div><div class="sub">Quick market overview</div>',unsafe_allow_html=True)

heat_tf=st.selectbox("Heatmap timeframe",["5m","15m","1h","4h"],index=1,key="heat_tf")

if st.button("🔄 UPDATE RSI HEATMAP",use_container_width=True):
    vals=[]
    with st.spinner("Building heatmap..."):
        with ThreadPoolExecutor(max_workers=8) as ex:
            fs={ex.submit(get_klines,s,heat_tf,60):s for s in symbols[:100]}
            for f in as_completed(fs):
                try:
                    d=f.result()
                    if d is not None:vals.append((fs[f],rsi_series(d.iloc[:-1]["close"]).iloc[-1]))
                except:pass
    st.session_state.heatmap=sorted(vals,key=lambda x:x[1])

heat=st.session_state.heatmap
if heat:
    zones=[
        ("🔴 Oversold",[x for x in heat if x[1]<30],"red"),
        ("🟠 Neutral",[x for x in heat if 30<=x[1]<60],"orange"),
        ("🟢 Strong/Gainers",[x for x in heat if x[1]>=60],"green")
    ]
    for title,items,cl in zones:
        st.markdown(f"**{title} — {len(items)} coins**")
        cards=[]
        for s,v in items:
            cards.append(
                f'<a class="coin {cl}" href="{tv(s,heat_tf)}" target="_blank">'
                f'<b>{s.replace("USDT","")}</b><br><small>RSI {v:.1f}</small></a>'
            )
        st.markdown('<div class="coin-grid">'+''.join(cards)+'</div>',unsafe_allow_html=True)
        st.write("")
else:
    st.info("Click UPDATE RSI HEATMAP.")

# ============================================================
# LEGACY SCANNERS / QUICK SCANS
# ============================================================

st.markdown('<div class="section">📈 Quick EMA 9/33 Scanner</div>',unsafe_allow_html=True)
ema_tf=st.selectbox("EMA timeframe",["5m","15m","1h","4h"],index=1,key="quick_ema_tf")

def quick_ema_scan(ss,tf):
    out=[]
    def one(s):
        d=get_klines(s,tf,100)
        if d is None:return None
        x=d.iloc[:-1]
        e9=x["close"].ewm(span=9,adjust=False).mean()
        e33=x["close"].ewm(span=33,adjust=False).mean()
        if e9.iloc[-2]<=e33.iloc[-2] and e9.iloc[-1]>e33.iloc[-1]:
            return {"Coin":s,"EMA9":e9.iloc[-1],"EMA33":e33.iloc[-1],"TradingView":tv(s,tf)}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z:out.append(z)
            except:pass
    return out

if st.button("🔍 SCAN EMA 9/33",use_container_width=True):
    with st.spinner("Scanning EMA crossovers..."):
        st.session_state.ema_results=quick_ema_scan(symbols,ema_tf)

if st.session_state.ema_results:
    st.dataframe(pd.DataFrame(st.session_state.ema_results),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})

st.markdown('<div class="section">🕯️ Quick Heikin Ashi Scanner</div>',unsafe_allow_html=True)
ha_tfs=st.multiselect("HA timeframes",["3m","5m","15m","1h","4h"],["3m","5m","15m","1h","4h"],key="ha_tfs")
ha_min=st.slider("Minimum bullish HA timeframes",1,5,3,key="ha_min")

def quick_ha_scan(ss,tfs,minimum,wick):
    out=[]
    def one(s):
        good=[]
        for tf in tfs:
            d=get_klines(s,tf,100)
            if d is not None and ha_signal(d,wick):good.append(tf)
        if len(good)>=minimum:
            return {"Coin":s,"Alignment":f"{len(good)}/{len(tfs)}","Bullish Timeframes":", ".join(good),"TradingView":tv(s,good[0])}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z:out.append(z)
            except:pass
    return out

if st.button("🔍 SCAN HEIKIN ASHI",use_container_width=True):
    with st.spinner("Scanning Heikin Ashi..."):
        st.session_state.ha_results=quick_ha_scan(symbols,ha_tfs,ha_min,wick)

if st.session_state.ha_results:
    st.dataframe(pd.DataFrame(st.session_state.ha_results),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("🚀 Crypto Signal Scanner • Binance Public Spot API • Technical scanner only • Not financial advice")




