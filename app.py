import streamlit as st
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

st.set_page_config(page_title="COCO Pulse", page_icon="🟢", layout="wide", initial_sidebar_state="expanded")

BASE = "https://data-api.binance.vision"
TIMEFRAMES = ["3m", "5m", "15m", "1h", "4h"]
RSI_TFS = ["5m", "15m", "1h", "4h"]

# ----------------------------- UI -----------------------------
st.markdown(r"""

<style>
:root{--bg:#070b13;--panel:#0d1422;--line:#1e2b40;--text:#eef4ff;--muted:#8ea0b8;--green:#37e58b;--red:#ff6577;--amber:#ffc857;--blue:#4db7ff}
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 50% -15%,#17263c 0,#0a101b 42%,#070b13 100%)!important;color:var(--text)}
[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:1500px!important;padding:28px 34px 70px!important}
.hero{padding:26px 30px 22px;border:1px solid #23334b;border-radius:24px;background:linear-gradient(135deg,rgba(18,31,50,.96),rgba(10,17,29,.96));box-shadow:0 18px 55px rgba(0,0,0,.28);margin-bottom:20px}
.hero-top{display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{font-size:31px;font-weight:950;line-height:1}.brand span{color:var(--blue)}.hero-sub{color:var(--muted);font-size:13px;margin-top:9px}
.live-dot{display:inline-flex;align-items:center;gap:7px;padding:8px 13px;border-radius:999px;background:rgba(55,229,139,.10);border:1px solid rgba(55,229,139,.28);color:var(--green);font-weight:850}.live-dot i{width:8px;height:8px;background:var(--green);border-radius:50%;box-shadow:0 0 12px var(--green)}
.panel{background:linear-gradient(145deg,rgba(15,24,39,.97),rgba(9,15,26,.97));border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:0 14px 38px rgba(0,0,0,.22);margin-bottom:20px}
.panel-title{font-size:20px;font-weight:900;margin-bottom:6px}.panel-sub{color:var(--muted);font-size:12px;margin-bottom:18px}
.stat{min-height:105px;padding:18px 20px;border-radius:17px;background:rgba(14,24,40,.9);border:1px solid #20304a}.stat-k{font-size:11px;font-weight:850;color:#8293aa;letter-spacing:.12em}.stat-v{font-size:27px;font-weight:950;margin-top:7px}
.green{color:var(--green)!important}.redtxt{color:var(--red)!important}.blue{color:var(--blue)!important}.amber{color:var(--amber)!important}
.market-banner{padding:25px 28px;border-radius:20px;border:1px solid #38465e;background:linear-gradient(135deg,rgba(22,34,52,.98),rgba(12,20,33,.98));text-align:center;min-height:132px;display:flex;flex-direction:column;justify-content:center}.market-banner.bull{border-color:rgba(55,229,139,.42)}.market-banner.bear{border-color:rgba(255,101,119,.42)}.market-label{font-size:11px;letter-spacing:.16em;color:#8da0b7;font-weight:900}.market-state{font-size:30px;font-weight:950;margin:5px 0}.market-note{font-size:12px;color:#a6b4c7}
.side-card{background:linear-gradient(145deg,#101a2b,#0b1320);border:1px solid var(--line);border-radius:18px;padding:19px;margin-bottom:16px}.side-title{font-size:16px;font-weight:900;margin-bottom:13px}.side-row{display:flex;justify-content:space-between;gap:12px;padding:11px 0;border-bottom:1px solid rgba(255,255,255,.055);font-size:13px}.side-row:last-child{border-bottom:0}.active{color:var(--green);font-weight:850}.idle{color:#718198}
.mover{display:flex;align-items:center;justify-content:space-between;text-decoration:none!important;color:var(--text)!important;padding:13px 14px;margin:7px 0;border-radius:12px;background:#0c1524;border:1px solid #1b2940;transition:.16s}.mover:hover{border-color:#385273;transform:translateY(-1px);background:#111e31}
.zone-filter{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:16px 0}.zone-filter button{min-height:43px!important;border-radius:12px!important;font-size:12px!important;font-weight:850!important}
.zone-head{font-size:18px;font-weight:950;margin:18px 0 13px;display:flex;align-items:center;gap:8px}.zone-count{font-size:12px;color:#7e91aa;font-weight:700}
.heat-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:11px}.coin{min-height:82px;padding:14px 12px;border-radius:14px;text-align:center;text-decoration:none!important;color:var(--text)!important;background:linear-gradient(145deg,#111d30,#0c1625);border:1px solid #253650;transition:.16s;display:block}.coin:hover{transform:translateY(-3px);border-color:#4b6b94;box-shadow:0 9px 25px rgba(0,0,0,.28)}.coin b{font-size:14px}.coin small{color:#aab8ca;font-size:11px}.coin .chg{font-size:11px;font-weight:850;color:var(--green)}.coin.red{border-color:#633340;background:linear-gradient(145deg,#27151e,#15111a)}.coin.warn{border-color:#68522d;background:linear-gradient(145deg,#292114,#17151a)}.coin.bluezone{border-color:#285477;background:linear-gradient(145deg,#11263a,#101925)}.coin.hot{border-color:#285b49;background:linear-gradient(145deg,#10251f,#0e1919)}
.signal{padding:19px 21px;margin:11px 0;border:1px solid #20304a;border-radius:16px;background:linear-gradient(145deg,#101a2a,#0b1422)}.signal-top{display:flex;align-items:center;justify-content:space-between;gap:16px}.score{font-size:28px;font-weight:950}.badge{display:inline-block;padding:4px 8px;border-radius:7px;font-size:10px;font-weight:900;margin-left:6px}.badge.buy{background:rgba(55,229,139,.12);color:var(--green);border:1px solid rgba(55,229,139,.25)}.badge.watch{background:rgba(255,200,87,.12);color:var(--amber);border:1px solid rgba(255,200,87,.25)}.checks{font-size:11px;color:#9aabc0;margin-top:8px;line-height:1.8}.small-note,.muted{color:var(--muted);font-size:12px}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{border-radius:11px!important;border:1px solid #2a3d59!important;background:#111f33!important;color:#edf4ff!important;font-weight:850!important;min-height:42px}[data-testid="stButton"] button:hover{border-color:#4c719f!important;background:#15263d!important}
[data-baseweb="select"]>div{background:#0f1a2b!important;border-color:#263951!important;color:#eef4ff!important;border-radius:11px!important}
[data-testid="stSidebar"]{background:#080e18!important;border-right:1px solid #172438!important}
@media(max-width:1100px){.heat-grid{grid-template-columns:repeat(4,minmax(0,1fr))}.zone-filter{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:800px){.block-container{padding:18px 14px 50px!important}.hero-top{align-items:flex-start;flex-direction:column}.brand{font-size:27px}.heat-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.zone-filter{grid-template-columns:repeat(2,minmax(0,1fr))}}

.clock-box{display:flex;align-items:center;gap:12px;padding:9px 14px;border:1px solid #263951;border-radius:14px;background:rgba(8,15,26,.72);min-width:185px}
.clock-icon{font-size:18px}.clock-time{font-size:17px;font-weight:950;line-height:1}.clock-date{font-size:10px;color:#8293aa;margin-top:4px}
</style>

""", unsafe_allow_html=True)

# ----------------------------- Data -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_symbols():
    try:
        d=requests.get(BASE+"/api/v3/exchangeInfo",timeout=15).json()
        return sorted([x["symbol"] for x in d["symbols"] if x["status"]=="TRADING" and x["quoteAsset"]=="USDT" and x.get("isSpotTradingAllowed",True)])
    except Exception:
        return []

@st.cache_data(ttl=30, show_spinner=False)
def get_tickers():
    try:
        d=requests.get(BASE+"/api/v3/ticker/24hr",timeout=20).json()
        if not isinstance(d,list): return {}
        return {x["symbol"]:x for x in d if x.get("symbol","" ).endswith("USDT")}
    except Exception:
        return {}

def active_symbols(limit=120):
    # Prioritize liquid/active pairs instead of scanning hundreds of low-volume coins.
    ranked=[]
    for s in ss_global:
        t=tickers_global.get(s,{})
        try: q=float(t.get("quoteVolume",0))
        except Exception: q=0
        ranked.append((s,q))
    ranked.sort(key=lambda x:x[1],reverse=True)
    return [s for s,_ in ranked[:limit]]

@st.cache_data(ttl=15, show_spinner=False)
def klines(symbol, tf, limit=90):
    try:
        d=requests.get(BASE+"/api/v3/klines",params={"symbol":symbol,"interval":tf,"limit":limit},timeout=10).json()
        if not isinstance(d,list) or len(d)<35:return None
        cols=["t","o","h","l","c","v","ct","qv","n","tb","tq","i"]
        x=pd.DataFrame(d,columns=cols)
        for c in ["o","h","l","c","v"]: x[c]=pd.to_numeric(x[c],errors="coerce")
        return x
    except Exception:return None

def rsi(df):
    d=df["c"].diff(); g=d.clip(lower=0); loss=-d.clip(upper=0)
    ag=g.ewm(alpha=1/14,adjust=False).mean(); al=loss.ewm(alpha=1/14,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return float((100-100/(1+rs)).fillna(50).iloc[-2])

def rsi_prev(df):
    return rsi(df.iloc[:-1]) if len(df)>40 else rsi(df)

def tv(symbol,tf):
    m={"3m":"3","5m":"5","15m":"15","1h":"60","4h":"240"}
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval={m.get(tf,'15')}"

def ema_vals(df):
    x=df.iloc[:-1]
    return x["c"].ewm(span=9,adjust=False).mean(),x["c"].ewm(span=33,adjust=False).mean(),x["c"].ewm(span=200,adjust=False).mean()

def ema_signal(df):
    e9,e33,e200=ema_vals(df)
    fresh=e9.iloc[-2]<=e33.iloc[-2] and e9.iloc[-1]>e33.iloc[-1]
    return fresh,float(e9.iloc[-1]),float(e33.iloc[-1]),float(e200.iloc[-1])

def heikin(df):
    x=df.iloc[:-1]
    hc=(x.o+x.h+x.l+x.c)/4
    ho=np.zeros(len(x));ho[0]=(x.o.iloc[0]+x.c.iloc[0])/2
    for i in range(1,len(x)):ho[i]=(ho[i-1]+hc.iloc[i-1])/2
    hh=np.maximum.reduce([x.h.to_numpy(),ho,hc.to_numpy()]); hl=np.minimum.reduce([x.l.to_numpy(),ho,hc.to_numpy()])
    return ho,hc,hh,hl

def ha_signal(df,wick=.25):
    try:
        ho,hc,hh,hl=heikin(df);a,b=-2,-1
        if not(hc[a]>ho[a] and hc[b]>ho[b] and hc[b]>hc[a] and hh[b]>hh[a]):return False
        ba=abs(hc[a]-ho[a]);bb=abs(hc[b]-ho[b])
        if ba==0 or bb==0:return False
        return (min(ho[a],hc[a])-hl[a])/ba<=wick and (min(ho[b],hc[b])-hl[b])/bb<=wick
    except Exception:return False

def safe_change(t):
    try:return float(t.get("priceChangePercent",0))
    except:return 0.0

def scan_confluence(ss,tf,minimum,settings,wick):
    out=[]
    def one(s):
        d=klines(s,tf)
        if d is None:return None
        score=0;checks=[];close=float(d["c"].iloc[-2])
        e9,e33,e200=ema_vals(d)
        if settings["ema200"]:
            ok=close>e200.iloc[-1];score+=20 if ok else 0;checks.append(("EMA200",ok))
        if settings["ema933"]:
            ok=e9.iloc[-1]>e33.iloc[-1] or (e9.iloc[-2]<=e33.iloc[-2] and e9.iloc[-1]>e33.iloc[-1]);score+=20 if ok else 0;checks.append(("EMA9/33",ok))
        rv=rsi(d); rp=rsi_prev(d)
        if settings["rsi"]:
            ok=50<=rv<=65 and rv>rp;score+=15 if ok else 0;checks.append(("RSI momentum",ok))
        vol=float(d["v"].iloc[-2]);avg=float(d["v"].iloc[-22:-2].mean()) if len(d)>24 else vol
        vr=vol/avg if avg else 1
        if settings["volume"]:
            ok=vr>=1.5;score+=20 if ok else 0;checks.append(("Volume spike",ok))
        prior_high=float(d["h"].iloc[-22:-2].max())
        if settings["breakout"]:
            ok=close>prior_high and vr>=1.2;score+=15 if ok else 0;checks.append(("20-bar breakout",ok))
        if settings["ha"]:
            ok=ha_signal(d,wick);score+=10 if ok else 0;checks.append(("Heikin Ashi",ok))
        if score<minimum:return None
        label="STRONG BUY" if score>=90 else "BUY" if score>=80 else "WATCH"
        return {"Coin":s,"Score":score,"Signal":label,"RSI":round(rv,1),"Vol x":round(vr,2),"TradingView":tv(s,tf),"Checks":checks}
    with ThreadPoolExecutor(max_workers=16) as ex:
        fs=[ex.submit(one,s) for s in ss]
        for f in as_completed(fs):
            try:
                z=f.result()
                if z:out.append(z)
            except Exception:pass
    return sorted(out,key=lambda x:(x["Score"],x["RSI"]),reverse=True)

def scan_rsi(ss,tf,confirm,rr,cr,direction):
    def okrange(v,r):
        return r=="All" or {"40 - 50":40<=v<50,"50 - 55":50<=v<55,"55 - 60":55<=v<60,"60 - 70":60<=v<70,"70+":v>=70}[r]
    out=[]
    def one(s):
        a=klines(s,tf);b=klines(s,confirm)
        if a is None or b is None:return None
        x=rsi(a);y=rsi(b);p=rsi_prev(a);dr="Rising" if x>p else "Falling" if x<p else "Flat"
        if not okrange(x,rr) or not okrange(y,cr) or (direction!="All" and dr!=direction):return None
        return {"Coin":s,"RSI":round(x,2),"Confirm RSI":round(y,2),"Direction":dr,"TradingView":tv(s,tf)}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z:out.append(z)
            except Exception:pass
    return sorted(out,key=lambda x:x["RSI"],reverse=True)

def scan_ema(ss,tf):
    out=[]
    def one(s):
        d=klines(s,tf)
        if d is None:return None
        ok,e9,e33,e200=ema_signal(d)
        if not ok:return None
        return {"Coin":s,"EMA 9":round(e9,8),"EMA 33":round(e33,8),"Signal":"Fresh Bullish Cross","TradingView":tv(s,tf)}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z:out.append(z)
            except Exception:pass
    return out

def scan_ha(ss,tfs,wick,minimum):
    out=[]
    def one(s):
        good=[]
        for tf in tfs:
            d=klines(s,tf)
            if d is not None and ha_signal(d,wick):good.append(tf)
        if len(good)<minimum:return None
        return {"Coin":s,"Bullish Timeframes":", ".join(good),"Alignment":f"{len(good)}/{len(tfs)}","TradingView":tv(s,good[0])}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z:out.append(z)
            except Exception:pass
    return out

# ----------------------------- helpers -----------------------------
def card_link(symbol,tf,change,zone):
    cls="hot" if change>=0 else "red"
    return f'<a class="coin {cls}" href="{tv(symbol,tf)}" target="_blank" rel="noopener noreferrer"><b>{symbol.replace("USDT","")}</b><br><small>RSI: {zone["rsi"]:.1f}</small><br><span class="chg">{change:+.1f}%</span></a>'

def render_heatmap(rows,tf):
    groups=[
        ("🔴 10–20 Extreme",[x for x in rows if 10<=x["rsi"]<20],"red"),
        ("🟠 20–30 Oversold",[x for x in rows if 20<=x["rsi"]<30],"warn"),
        ("🔵 30–40 Hidden Bull & ICT",[x for x in rows if 30<=x["rsi"]<40],"bluezone"),
        ("⚪ 40–50 Neutral",[x for x in rows if 40<=x["rsi"]<50],""),
        ("🟢 50–60 V1 Zone",[x for x in rows if 50<=x["rsi"]<60],"hot"),
        ("🚀 60+ Gainers",[x for x in rows if x["rsi"]>=60],"hot"),
    ]

    labels=["10–20 Extreme","20–30 Oversold","30–40 Hidden Bull & ICT",
            "40–50 Neutral","50–60 V1 Zone","60+ Gainers"]
    keys=["extreme","oversold","hidden","neutral","v1","gainers"]
    colors=["redp","amberp","bluep","","greenp","greenp"]

    if "heat_zone" not in st.session_state:
        st.session_state.heat_zone="gainers"

    st.markdown('<div class="zone-filter"><span class="zone-label">RSI ZONES</span>',unsafe_allow_html=True)
    cols=st.columns(6)
    for i,(label,key) in enumerate(zip(labels,keys)):
        with cols[i]:
            if st.button(label,use_container_width=True,key=f"heat_zone_{key}"):
                st.session_state.heat_zone=key
    st.markdown('</div>',unsafe_allow_html=True)

    idx=keys.index(st.session_state.heat_zone)
    title,items,cls=groups[idx]

    # Small live highlight: strongest rising RSI in the 52–55 band.
    rising=[x for x in rows if 52<=x["rsi"]<55 and x.get("change",0)>0]
    if rising:
        hot=max(rising,key=lambda x:x["rsi"])
        st.markdown(
            f'<div class="small-note">⬜ <b>RSI 52–55 Rising:</b> '
            f'<span class="amber">{hot["symbol"].replace("USDT","")} RSI:{hot["rsi"]:.1f}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="zone-head">{title} <span class="zone-count">{len(items)} coins</span></div>',
        unsafe_allow_html=True
    )

    if not items:
        st.markdown('<div class="small-note">No coins in this zone right now.</div>',unsafe_allow_html=True)
        return

    html='<div class="heat-grid">'
    for z in items:
        html+=(
            f'<a class="coin {cls}" href="{tv(z["symbol"],tf)}" '
            f'target="_blank" rel="noopener noreferrer">'
            f'<b>{z["symbol"].replace("USDT","")}</b><br>'
            f'<small>RSI {z["rsi"]:.1f}</small><br>'
            f'<span class="chg">{z["change"]:+.1f}%</span></a>'
        )
    st.markdown(html+'</div>',unsafe_allow_html=True)


@st.fragment(run_every=5)
def dashboard_heatmap():
    st.markdown(
        '<div class="panel"><div class="panel-title">📊 RSI Heatmap '
        '<span class="live">LIVE</span></div>',
        unsafe_allow_html=True
    )

    h1,h2,h3,h4=st.columns([1.15,0.9,0.95,1.15])
    with h1:
        ht=st.selectbox("Heatmap timeframe",RSI_TFS,index=1,key="dash_heat_tf")
    with h2:
        limit_heat=st.selectbox("Scan universe",[60,100,150,200],index=1,key="heat_universe")
    with h3:
        auto_heat=st.toggle("🔄 Auto Update",value=True,key="heat_auto")
    with h4:
        refresh_heat=st.button("⚡ UPDATE NOW",use_container_width=True,key="dash_heat_btn")

    import time as _time
    if "heat_last_update" not in st.session_state:
        st.session_state.heat_last_update=0.0
    if "heat_next_update" not in st.session_state:
        st.session_state.heat_next_update=0.0

    now=_time.time()
    due=now >= st.session_state.heat_next_update
    should_update=refresh_heat or (auto_heat and due)

    if should_update:
        with st.spinner(f"Updating RSI heatmap • {limit_heat} active coins..."):
            rows=[]
            candidates=active_symbols(limit_heat)
            with ThreadPoolExecutor(max_workers=16) as ex:
                fs={ex.submit(klines,s,ht):s for s in candidates}
                for f in as_completed(fs):
                    try:
                        d=f.result()
                        if d is not None:
                            rows.append({
                                "symbol":fs[f],
                                "rsi":rsi(d),
                                "change":safe_change(tickers.get(fs[f],{}))
                            })
                    except Exception:
                        pass
            st.session_state.heat=sorted(rows,key=lambda x:x["rsi"],reverse=True)
            st.session_state.heat_last_update=_time.time()
            st.session_state.heat_next_update=st.session_state.heat_last_update+60

    if st.session_state.heat:
        remaining=max(0,int(st.session_state.heat_next_update-_time.time()))
        if auto_heat:
            st.caption(
                f"🟢 Auto update ON • Updated: "
                f"{_time.strftime('%H:%M:%S',_time.localtime(st.session_state.heat_last_update))}"
                f" • Next update in: {remaining}s"
            )
        else:
            st.caption(
                f"⚪ Auto update OFF • Updated: "
                f"{_time.strftime('%H:%M:%S',_time.localtime(st.session_state.heat_last_update))}"
            )
        render_heatmap(st.session_state.heat,ht)
    else:
        st.info("Heatmap is loading automatically. You can also press UPDATE NOW.")

    st.markdown('</div>',unsafe_allow_html=True)

# ----------------------------- state -----------------------------
ss=get_symbols(); tickers=get_tickers()
ss_global=ss; tickers_global=tickers
if "heat" not in st.session_state:st.session_state.heat=[]
if "r" not in st.session_state:st.session_state.r=[]
if "e" not in st.session_state:st.session_state.e=[]
if "h" not in st.session_state:st.session_state.h=[]
if "c" not in st.session_state:st.session_state.c=[]

# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🟢 COCO Pulse")
    st.caption("Professional Binance signal dashboard")
    st.markdown("### Navigation")
    section=st.radio("",["Dashboard","RSI Scanner","Confluence Scanner","EMA 9/33","Heikin Ashi"],label_visibility="collapsed")
    st.markdown("### Chart Settings")
    chart_tf=st.selectbox("TradingView timeframe",RSI_TFS,index=2)
    search=st.text_input("🔎 Find coin",placeholder="BTC, XRP, DOGE...")
    st.markdown("### Scanner Status")
    st.markdown('<div class="side-row"><span>RSI Heatmap</span><span class="active">● LIVE</span></div><div class="side-row"><span>EMA Crossover</span><span class="idle">Ready</span></div><div class="side-row"><span>Heikin Ashi</span><span class="idle">Ready</span></div><div class="side-row"><span>Confluence</span><span class="idle">Ready</span></div>',unsafe_allow_html=True)
    st.caption("⚡ Fast mode: scans the most active pairs first")

# ----------------------------- header -----------------------------
now=datetime.now().strftime("%H:%M:%S")
st.markdown(f'<div class="hero"><div class="brand"><div class="brand-badge">🟢</div><div><h1>COCO Pulse</h1><div class="hero-sub">Crypto Signal Scanner • Binance USDT Spot • <span class="live"><span class="live-dot"></span>LIVE</span> • Updated {now}</div></div></div></div>',unsafe_allow_html=True)

st.markdown('<div class="small-note" style="margin:4px 2px 12px">⚡ Scans are optimized for speed by prioritizing high-volume active pairs. Increase the scan universe only when you need wider coverage.</div>',unsafe_allow_html=True)

@st.fragment(run_every=1)
def render_live_header():
    now = datetime.now()
    clock = now.strftime("%I:%M:%S %p")
    date_txt = now.strftime("%d %b %Y")
    st.markdown(
        f'<div class="hero"><div class="hero-top">'
        f'<div><div class="brand">🚀 COCO <span>Pulse</span></div>'
        f'<div class="hero-sub">Professional Binance USDT Market Intelligence • RSI Heatmap • Confluence Signals • EMA • Heikin Ashi</div></div>'
        f'<div style="display:flex;align-items:center;gap:12px">'
        f'<div class="live-dot"><i></i> LIVE • Binance Spot</div>'
        f'<div class="clock-box"><div class="clock-icon">🕐</div>'
        f'<div><div class="clock-time">{clock}</div><div class="clock-date">🇵🇰 PKT • {date_txt}</div></div></div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

render_live_header()

# ----------------------------- top stats -----------------------------
total=len(ss); moves=[safe_change(tickers.get(s,{})) for s in ss]; up=sum(x>0 for x in moves); down=sum(x<0 for x in moves)
market="BULLISH" if up>down else "BEARISH" if down>up else "NEUTRAL"; breadth=(up/total*100 if total else 0)
cols=st.columns(5)
vals=[("USDT PAIRS",total,""),("GAINERS",up,"green"),("LOSERS",down,"redtxt"),("MARKET",market,"green" if market=="BULLISH" else "redtxt"),("BREADTH",f"{breadth:.0f}%","blue")]
for col,(k,v,c) in zip(cols,vals):
    with col:
        if k=="MARKET":
            state_class="bull" if market=="BULLISH" else "bear" if market=="BEARISH" else ""
            arrow="↗" if market=="BULLISH" else "↘" if market=="BEARISH" else "→"
            st.markdown(f'<div class="market-banner {state_class}"><div class="market-label">OVERALL MARKET</div><div class="market-state {c}">{arrow} {market}</div><div class="market-note">{up/total*100:.0f}% rising • {down/total*100:.0f}% falling (24h)</div></div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="stat"><div class="stat-k">{k}</div><div class="stat-v {c}">{v}</div></div>',unsafe_allow_html=True)

# ----------------------------- movers + main content -----------------------------
left,right=st.columns([4.8,1.35],gap="large")
with right:
    st.markdown('<div class="panel"><div class="panel-title">🔥 Top Movers</div>',unsafe_allow_html=True)
    movers=sorted([(s,safe_change(tickers.get(s,{}))) for s in ss],key=lambda x:x[1],reverse=True)[:7]
    for s,ch in movers:
        st.markdown(f'<a class="mover" href="{tv(s,chart_tf)}" target="_blank"><b>{s.replace("USDT","")}</b><span class="green">{ch:+.2f}%</span></a>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="side-card"><div class="side-title">⚙ Active Scanners</div><div class="side-row"><span>RSI Heatmap</span><span class="active">LIVE</span></div><div class="side-row"><span>Confluence</span><span class="active" if="" else "idle">READY</span></div><div class="side-row"><span>EMA 9/33</span><span class="idle">READY</span></div><div class="side-row"><span>Heikin Ashi</span><span class="idle">READY</span></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="side-card"><div class="side-title">📈 Quick Guide</div><div class="small-note">90+ Strong Buy<br>80–89 Buy<br>70–79 Watch<br>Fresh EMA9/33 cross = new bullish event<br>Cards open TradingView.</div></div>',unsafe_allow_html=True)

with left:
    if section=="Dashboard":
        dashboard_heatmap()

        st.markdown('<div class="panel"><div class="panel-title">🎯 Confluence Signal Scanner</div><div class="muted">Combine EMA trend + EMA9/33 + RSI momentum + volume + breakout + Heikin Ashi into one score.</div>',unsafe_allow_html=True)
        a1,a2,a3=st.columns(3)
        with a1:ctf=st.selectbox("Signal timeframe",RSI_TFS,index=1,key="conf_tf");minscore=st.slider("Minimum score",40,100,75,5,key="conf_score");conf_universe=st.selectbox("Scan universe",[60,100,150,200],index=1,key="conf_universe")
        with a2:ema200=st.checkbox("EMA 200 Trend",True,key="c_200");ema933=st.checkbox("EMA 9/33",True,key="c_933");rsim=st.checkbox("RSI Momentum",True,key="c_rsi")
        with a3:vol=st.checkbox("Volume Spike",True,key="c_vol");br=st.checkbox("Breakout",True,key="c_break");hac=st.checkbox("Heikin Ashi",True,key="c_ha")
        wick=st.slider("HA max lower-wick / body",0.0,1.0,.25,.05,key="c_wick")
        if st.button("🚀 SCAN CONFLUENCE",use_container_width=True,key="conf_scan"):
            settings={"ema200":ema200,"ema933":ema933,"rsi":rsim,"volume":vol,"breakout":br,"ha":hac}
            candidates=active_symbols(conf_universe)
            with st.spinner(f"Fast scan: {len(candidates)} active pairs..."):st.session_state.c=scan_confluence(candidates,ctf,minscore,settings,wick)
        if st.session_state.c:
            for z in st.session_state.c[:12]:
                badge="buy" if z["Score"]>=80 else "watch"; checks=" • ".join([("✓ " if ok else "○ ")+name for name,ok in z["Checks"]])
                st.markdown(f'<div class="signal"><div class="signal-top"><div><b>{z["Coin"].replace("USDT","")}</b> <span class="badge {badge}">{z["Signal"]}</span></div><div class="score green">{z["Score"]}</div></div><div class="muted">RSI {z["RSI"]} • Volume {z["Vol x"]}x • <a href="{z["TradingView"]}" target="_blank">TradingView ↗</a></div><div class="checks">{checks}</div></div>',unsafe_allow_html=True)
        else:st.caption("No confluence signals loaded yet.")
        st.markdown('</div>',unsafe_allow_html=True)

    elif section=="RSI Scanner":
        st.markdown('<div class="panel"><div class="panel-title">🔍 RSI Scanner</div>',unsafe_allow_html=True)
        x1,x2,x3=st.columns(3)
        with x1:pt=st.selectbox("Primary timeframe",RSI_TFS,index=1);pr=st.selectbox("Primary RSI range",["All","40 - 50","50 - 55","55 - 60","60 - 70","70+"])
        with x2:ct=st.selectbox("Confirmation timeframe",RSI_TFS,index=2);cr=st.selectbox("Confirmation RSI range",["All","40 - 50","50 - 55","55 - 60","60 - 70","70+"])
        with x3:di=st.selectbox("RSI direction",["All","Rising","Falling"]);rsi_universe=st.selectbox("Scan universe",[60,100,150,200],index=1,key="rsi_universe");st.checkbox("Closed candles only",True)
        if st.button("🔍 SCAN RSI",use_container_width=True,key="rsi_page_scan"):
            candidates=active_symbols(rsi_universe)
            with st.spinner(f"Fast RSI scan: {len(candidates)} active pairs..."):st.session_state.r=scan_rsi(candidates,pt,ct,pr,cr,di)
        if search:st.session_state.r=[z for z in st.session_state.r if search.upper() in z["Coin"]]
        if st.session_state.r:st.dataframe(pd.DataFrame(st.session_state.r),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})
        else:st.info("Run RSI scan to see matching coins.")
        st.markdown('</div>',unsafe_allow_html=True)

    elif section=="Confluence Scanner":
        st.markdown('<div class="panel"><div class="panel-title">🎯 Full Confluence Scanner</div>',unsafe_allow_html=True)
        tf=st.selectbox("Timeframe",RSI_TFS,index=1,key="full_conf_tf");score=st.slider("Minimum score",40,100,75,5,key="full_score");full_universe=st.selectbox("Scan universe",[100,150,200,300],index=0,key="full_universe")
        q=st.button("🚀 RUN FULL CONFLUENCE SCAN",use_container_width=True,key="full_conf_scan")
        if q:
            settings={"ema200":True,"ema933":True,"rsi":True,"volume":True,"breakout":True,"ha":True}
            candidates=active_symbols(full_universe)
            with st.spinner(f"Scanning {len(candidates)} active pairs..."):st.session_state.c=scan_confluence(candidates,tf,score,settings,.25)
        if st.session_state.c:
            st.dataframe(pd.DataFrame([{k:v for k,v in z.items() if k!="Checks"} for z in st.session_state.c]),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})
        else:st.info("Run the scanner to find multi-factor signals.")
        st.markdown('</div>',unsafe_allow_html=True)

    elif section=="EMA 9/33":
        st.markdown('<div class="panel"><div class="panel-title">📈 EMA 9 / 33 Fresh Bullish Cross</div>',unsafe_allow_html=True)
        tf=st.selectbox("EMA timeframe",RSI_TFS,index=1,key="ema_page_tf");ema_universe=st.selectbox("Scan universe",[60,100,150,200],index=1,key="ema_universe")
        if st.button("🔍 SCAN EMA 9/33",use_container_width=True,key="ema_page_scan"):
            candidates=active_symbols(ema_universe)
            with st.spinner(f"Checking {len(candidates)} active pairs..."):st.session_state.e=scan_ema(candidates,tf)
        if st.session_state.e:st.dataframe(pd.DataFrame(st.session_state.e),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})
        else:st.info("No fresh EMA9/33 bullish cross loaded.")
        st.markdown('</div>',unsafe_allow_html=True)

    elif section=="Heikin Ashi":
        st.markdown('<div class="panel"><div class="panel-title">🕯️ Heikin Ashi Bullish Scanner</div>',unsafe_allow_html=True)
        tfs=st.multiselect("HA timeframes",TIMEFRAMES,["3m","5m","15m","1h","4h"],key="ha_tfs")
        w=st.slider("Maximum lower wick / body",0.0,1.0,.25,.05,key="ha_wick")
        minimum=st.slider("Minimum bullish HA timeframes",1,5,3,key="ha_min");ha_universe=st.selectbox("Scan universe",[40,60,80,100],index=1,key="ha_universe")
        if st.button("🔍 SCAN HEIKIN ASHI",use_container_width=True,key="ha_page_scan"):
            candidates=active_symbols(ha_universe)
            with st.spinner(f"Checking {len(candidates)} active pairs across {len(tfs)} timeframes..."):st.session_state.h=scan_ha(candidates,tfs,w,minimum)
        if st.session_state.h:st.dataframe(pd.DataFrame(st.session_state.h),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})
        else:st.info("No HA alignment loaded.")
        st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<div class="small-note" style="text-align:center;margin-top:18px">COCO Pulse • Technical scanner only • Binance Public Spot API • TradingView links open charts • Not financial advice</div>',unsafe_allow_html=True)






