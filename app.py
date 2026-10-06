import streamlit as st
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import time

st.set_page_config(page_title="COCO Nexus", page_icon="🟢", layout="wide", initial_sidebar_state="expanded")

BASE = "https://data-api.binance.vision"
TIMEFRAMES = ["3m", "5m", "15m", "1h", "4h"]
RSI_TFS = ["5m", "15m", "1h", "4h"]

# ----------------------------- UI -----------------------------
st.markdown(r"""
<style>
:root{--bg:#070b12;--panel:#0d1521;--panel2:#111c2b;--line:#223249;--text:#edf3fb;--muted:#8190a5;--green:#49f08b;--red:#ff6376;--amber:#ffc84a;--blue:#66b7ff;--cyan:#62e5ff;}
.stApp{background:radial-gradient(circle at 10% 0%,#102238 0,#070b12 35%),#070b12;color:var(--text)}
.block-container{max-width:1700px;padding-top:1rem;padding-bottom:3rem}
section[data-testid="stSidebar"]{background:#080e18;border-right:1px solid #1b293c}
section[data-testid="stSidebar"] .block-container{padding-top:1rem}
.hero{background:linear-gradient(135deg,#101e31,#0b121e 65%);border:1px solid #263a54;border-radius:20px;padding:22px 24px;margin-bottom:14px;box-shadow:0 12px 35px rgba(0,0,0,.22)}
.brand{display:flex;align-items:center;gap:12px}.brand-badge{width:48px;height:48px;border-radius:15px;display:grid;place-items:center;background:linear-gradient(135deg,#43f18a,#1b8f63);font-size:24px;box-shadow:0 0 28px rgba(73,240,139,.18)}
.hero h1{margin:0;font-size:30px;letter-spacing:-.7px}.hero-sub{color:var(--muted);margin-top:5px}.live{color:var(--green);font-weight:800}.live-dot{display:inline-block;width:8px;height:8px;background:var(--green);border-radius:50%;margin-right:5px;box-shadow:0 0 10px var(--green)}
.panel{background:rgba(13,21,33,.93);border:1px solid var(--line);border-radius:18px;padding:15px;margin:0 0 14px 0;box-shadow:0 8px 25px rgba(0,0,0,.13)}
.panel-title{font-size:18px;font-weight:800;margin-bottom:10px}.muted{color:var(--muted)}
.stat{background:linear-gradient(180deg,#101b2b,#0c1420);border:1px solid #21334a;border-radius:15px;padding:13px 15px;min-height:88px}.stat-k{color:#7f91aa;font-size:12px;text-transform:uppercase;letter-spacing:.8px}.stat-v{font-size:25px;font-weight:900;margin-top:4px}.market-card{border-color:#6b3946;box-shadow:0 0 22px rgba(255,99,118,.08)}.green{color:var(--green)}.redtxt{color:var(--red)}.amber{color:var(--amber)}.blue{color:var(--blue)}
.mover{display:flex;justify-content:space-between;align-items:center;background:#0e1927;border:1px solid #203149;border-radius:11px;padding:9px 11px;margin:6px 0;text-decoration:none;color:#fff}.mover:hover{border-color:#49f08b;transform:translateX(2px)}
.zone-head{display:flex;align-items:center;gap:8px;font-size:16px;font-weight:900;margin:12px 2px 7px}.zone-count{font-size:12px;color:#8494aa;font-weight:600}.heat-grid{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px}.coin{display:block;text-decoration:none;color:#fff;text-align:center;border-radius:11px;padding:9px 5px;background:linear-gradient(180deg,#172338,#111b2b);border:1px solid #30445f;transition:.14s;min-height:55px}.coin:hover{transform:translateY(-2px);border-color:#68c9ff;box-shadow:0 5px 18px rgba(74,183,255,.14)}.coin b{font-size:12px}.coin small{font-size:10px;color:#aab7c9}.coin .chg{font-size:10px;font-weight:800;margin-left:2px}.coin.hot{background:linear-gradient(180deg,#183c2b,#10251d);border-color:#2c704d}.coin.warn{background:linear-gradient(180deg,#3a3118,#211e13);border-color:#77612a}.coin.red{background:linear-gradient(180deg,#3a2027,#21151a);border-color:#78414c}.coin.bluezone{background:linear-gradient(180deg,#17334b,#13243a);border-color:#3b6c92}
.pill-row{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:8px}.zone-filter{display:flex;flex-wrap:wrap;gap:9px;margin:8px 0 12px}.zone-label{font-size:12px;color:#8798ad;font-weight:800;align-self:center}.pill{border:1px solid #31435a;background:#101a29;border-radius:999px;padding:6px 10px;font-size:11px;font-weight:800}.pill.greenp{border-color:#27704b;color:#6ff1a2}.pill.bluep{border-color:#356b95;color:#82caff}.pill.amberp{border-color:#7b6428;color:#ffd66c}.pill.redp{border-color:#7d3e4b;color:#ff8997}
.signal{border:1px solid #243750;background:linear-gradient(145deg,#111d2d,#0b131f);border-radius:15px;padding:12px;margin:7px 0}.signal-top{display:flex;justify-content:space-between;align-items:center}.score{font-size:22px;font-weight:900}.badge{font-size:10px;font-weight:900;border-radius:999px;padding:5px 8px}.buy{background:#153e2a;color:#72f2a5;border:1px solid #2c8156}.watch{background:#3b3216;color:#ffd66a;border:1px solid #7b652a}.checks{color:#aebaca;font-size:11px;line-height:1.65;margin-top:6px}
.side-card{background:#0e1826;border:1px solid #213149;border-radius:14px;padding:12px;margin-bottom:10px}.side-title{font-weight:900;font-size:15px;margin-bottom:8px}.side-row{display:flex;justify-content:space-between;padding:7px 8px;border-radius:9px;background:#101d2d;margin:5px 0;color:#c5cfdd;font-size:12px}.idle{color:#7e8da0}.active{color:var(--green);font-weight:800}.danger{color:var(--red);font-weight:800}
div[data-testid="stButton"] button{background:#142236!important;color:#f5f8fc!important;border:1px solid #31506e!important;border-radius:10px!important;font-weight:800!important;min-height:42px!important}div[data-testid="stButton"] button:hover{background:#18342d!important;border-color:#49f08b!important}
div[data-testid="stMetric"]{background:#0e1826;border:1px solid #213149;padding:10px;border-radius:12px}
[data-testid="stDataFrame"]{border:1px solid #213149;border-radius:12px}
.small-note{font-size:11px;color:#718096}.section-space{height:3px}
@media(max-width:1200px){.heat-grid{grid-template-columns:repeat(4,minmax(110px,1fr))}}
@media(max-width:850px){.heat-grid{grid-template-columns:repeat(3,minmax(95px,1fr))}.hero h1{font-size:24px}}
@media(max-width:520px){.heat-grid{grid-template-columns:repeat(2,minmax(100px,1fr))}.block-container{padding-left:.65rem;padding-right:.65rem}}
.levels{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}.lvl{padding:6px 9px;border-radius:8px;font-size:11px;font-weight:800;border:1px solid #2b3b52}.lvl.entry{color:#dbe6f5;background:rgba(120,140,165,.10)}.lvl.tp{color:var(--green);background:rgba(55,229,139,.08);border-color:rgba(55,229,139,.25)}.lvl.sl{color:var(--red);background:rgba(255,101,119,.08);border-color:rgba(255,101,119,.25)}.trade-levels{display:flex;gap:10px;align-items:center;margin-top:12px;margin-bottom:8px}</style>
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

def structure_metrics(df, lookback=30):
    """Balanced market-structure model using confirmed candle closes."""
    x=df.iloc[:-1].copy()
    if len(x)<12:
        return {"bos":False,"choch":False,"hh":False,"hl":False,"sweep":False,"last_high":None,"last_low":None}
    n=min(lookback,len(x)-3)
    highs=x["h"].iloc[-n:]
    lows=x["l"].iloc[-n:]
    # Most recent meaningful swing levels. Exclude the latest 2 candles from the reference.
    ref=x.iloc[-(n+2):-2]
    if len(ref)<5:
        ref=x.iloc[:-2]
    last_high=float(ref["h"].max()); last_low=float(ref["l"].min())
    close=float(x["c"].iloc[-1]); prev_close=float(x["c"].iloc[-2])
    prior_high=float(x["h"].iloc[-8:-2].max()) if len(x)>=10 else last_high
    prior_low=float(x["l"].iloc[-8:-2].min()) if len(x)>=10 else last_low
    # Simple swing direction: compare two recent windows.
    mid=max(3,min(7,len(x)//5))
    a=x.iloc[-2*mid:-mid]; b=x.iloc[-mid:]
    ah=float(a["h"].max()); al=float(a["l"].min()); bh=float(b["h"].max()); bl=float(b["l"].min())
    hh=bh>ah and bh>=prior_high
    hl=bl>al
    ll=bl<al
    # BOS: close through a meaningful prior high. CHoCH: bullish close through a prior lower structure.
    bos=close>last_high or close>prior_high
    choch=close>prior_high and al<ah
    # Bullish liquidity sweep: wick below a recent low, then close back above it.
    recent_low=float(x["l"].iloc[-8:-2].min()) if len(x)>=10 else last_low
    sweep=float(x["l"].iloc[-1])<recent_low and close>recent_low
    return {"bos":bool(bos),"choch":bool(choch),"hh":bool(hh),"hl":bool(hl),"sweep":bool(sweep),"last_high":last_high,"last_low":last_low}

def ha_components(df, include_live=False):
    x=df.copy() if include_live else df.iloc[:-1].copy()
    if len(x)<4:return None
    hc=(x.o+x.h+x.l+x.c)/4
    ho=np.zeros(len(x)); ho[0]=(x.o.iloc[0]+x.c.iloc[0])/2
    for i in range(1,len(x)): ho[i]=(ho[i-1]+hc.iloc[i-1])/2
    hh=np.maximum.reduce([x.h.to_numpy(),ho,hc.to_numpy()])
    hl=np.minimum.reduce([x.l.to_numpy(),ho,hc.to_numpy()])
    return ho,hc,hh,hl

def ha_state(df, wick=.25, include_live=False):
    try:
        z=ha_components(df,include_live)
        if z is None:return {"early":False,"confirmed":False,"count":0,"no_wick":False}
        ho,hc,hh,hl=z
        a,b=-2,-1
        green_a=hc[a]>ho[a]; green_b=hc[b]>ho[b]
        higher_close=hc[b]>hc[a]; higher_high=hh[b]>hh[a]
        ba=abs(hc[a]-ho[a]); bb=abs(hc[b]-ho[b])
        wa=((min(ho[a],hc[a])-hl[a])/ba) if ba else 99
        wb=((min(ho[b],hc[b])-hl[b])/bb) if bb else 99
        no_wick=wa<=wick and wb<=wick
        confirmed=green_a and green_b and higher_close and higher_high and no_wick
        # Early: previous closed HA candle is bullish and the current candle is forming bullish.
        early=green_a and green_b and higher_close and (wb<=wick*1.35 if wick>0 else wb<=0.03)
        return {"early":bool(early),"confirmed":bool(confirmed),"count":2 if green_a and green_b else 1 if green_b else 0,"no_wick":bool(no_wick),"lower_wick":round(wb,3)}
    except Exception:
        return {"early":False,"confirmed":False,"count":0,"no_wick":False}

def scan_smart_confluence(ss,tf,minimum,settings,wick):
    out=[]
    def one(s):
        d=klines(s,tf)
        if d is None:return None
        closed=d.iloc[:-1]
        if len(closed)<35:return None
        close=float(closed["c"].iloc[-1])
        e9,e33,e200=ema_vals(d)
        rv=rsi(d); rp=rsi_prev(d)
        vol=float(closed["v"].iloc[-1]); avg=float(closed["v"].iloc[-22:-1].mean()) if len(closed)>23 else vol
        vr=vol/avg if avg else 1
        stx=structure_metrics(d)
        ha_live=ha_state(d,wick,True); ha_closed=ha_state(d,wick,False)
        checks=[]; score=0
        def add(name,ok,pts):
            nonlocal score
            score+=pts if ok else 0; checks.append((name,bool(ok)))
        add("EMA200 trend",close>e200.iloc[-1],15)
        add("EMA9 > EMA33",e9.iloc[-1]>e33.iloc[-1],15)
        add("RSI rising",50<=rv<=68 and rv>rp,10)
        add("Volume",vr>=1.25,10)
        add("BOS",stx["bos"],15)
        add("CHoCH",stx["choch"],10)
        add("HH/HL",stx["hh"] or stx["hl"],8)
        add("Liquidity sweep",stx["sweep"],12)
        add("HA early",ha_live["early"],8)
        add("HA confirmed",ha_closed["confirmed"],10)
        # Early is intentionally selective: at least one structure event plus HA/EMA/RSI/volume confluence.
        early_ok=(ha_live["early"] and rv>rp and vr>=1.15 and (close>=e9.iloc[-1]*0.998) and (stx["sweep"] or stx["choch"] or stx["bos"]))
        confirmed_ok=(score>=minimum and ha_closed["confirmed"] and close>e9.iloc[-1]>e33.iloc[-1] and rv>rp and vr>=1.25 and (stx["bos"] or stx["choch"]))
        if not early_ok and not confirmed_ok:return None
        status="CONFIRMED" if confirmed_ok else "EARLY"
        tp1_pct=float(settings.get("tp1_pct",1.0)); tp2_pct=float(settings.get("tp2_pct",2.0)); sl_pct=float(settings.get("sl_pct",1.0))
        return {"Coin":s,"Score":score,"Status":status,"Signal":"STRONG BUY" if confirmed_ok else "EARLY BUY","Entry":close,"TP1":close*(1+tp1_pct/100),"TP2":close*(1+tp2_pct/100),"SL":close*(1-sl_pct/100),"TP1 %":tp1_pct,"TP2 %":tp2_pct,"SL %":sl_pct,"RSI":round(rv,1),"Vol x":round(vr,2),"BOS":stx["bos"],"CHoCH":stx["choch"],"HH/HL":stx["hh"] or stx["hl"],"Liquidity Sweep":stx["sweep"],"HA":ha_closed["confirmed"],"TradingView":tv(s,tf),"Checks":checks}
    with ThreadPoolExecutor(max_workers=16) as ex:
        fs=[ex.submit(one,s) for s in ss]
        for f in as_completed(fs):
            try:
                z=f.result()
                if z:out.append(z)
            except Exception:pass
    return sorted(out,key=lambda x:(x["Status"]=="CONFIRMED",x["Score"],x["RSI"]),reverse=True)


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

    # Small live highlight: strongest rising RSI in the 52–55 band.
    rising=[x for x in rows if 52<=x["rsi"]<55 and x.get("change",0)>0]
    if rising:
        hot=max(rising,key=lambda x:x["rsi"])
        st.markdown(
            f'<div class="small-note">⬜ <b>RSI 52–55 Rising:</b> '
            f'<span class="amber">{hot["symbol"].replace("USDT","")} RSI:{hot["rsi"]:.1f}</span></div>',
            unsafe_allow_html=True
        )

    # Show every RSI zone as its own section, like a professional market heatmap.
    for title,items,cls in groups:
        items=sorted(items,key=lambda x:x["rsi"],reverse=True)
        st.markdown(
            f'<div class="zone-head">{title} '
            f'<span class="zone-count">{len(items)} coins</span></div>',
            unsafe_allow_html=True
        )

        if not items:
            st.markdown(
                '<div class="small-note">No coins in this zone right now.</div>',
                unsafe_allow_html=True
            )
            continue

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
    st.markdown("## 🟢 COCO Nexus")
    st.caption("Professional Binance signal dashboard")
    st.markdown("### Navigation")
    section=st.radio("",["Dashboard","RSI Scanner","Confluence Scanner","EMA 9/33","Heikin Ashi"],label_visibility="collapsed")
    st.markdown("### Chart Settings")
    chart_tf=st.selectbox("TradingView timeframe",RSI_TFS,index=2)
    search=st.text_input("🔎 Find coin",placeholder="BTC, XRP, DOGE...")
    st.markdown("### Scanner Status")
    st.markdown('<div class="side-row"><span>RSI Heatmap</span><span class="active">● LIVE</span></div><div class="side-row"><span>EMA Crossover</span><span class="idle">Ready</span></div><div class="side-row"><span>Heikin Ashi</span><span class="idle">Ready</span></div><div class="side-row"><span>Smart Entry</span><span class="active">LIVE</span></div><div class="side-row"><span>Confluence</span><span class="idle">Ready</span></div>',unsafe_allow_html=True)
    st.caption("⚡ Fast mode: scans the most active pairs first")

# ----------------------------- live top clock -----------------------------
st.markdown("<style>.top-clock{display:flex;justify-content:flex-end;align-items:center;margin:0 4px 10px}.top-clock-box{background:rgba(13,21,33,.94);border:1px solid #263a54;border-radius:14px;padding:8px 15px;text-align:right;box-shadow:0 8px 22px rgba(0,0,0,.18)}.top-clock-time{font-size:20px;font-weight:900;line-height:1.05;color:#edf3fb}.top-clock-date{font-size:10px;color:#8190a5;margin-top:3px;letter-spacing:.5px}.top-clock-live{color:#49f08b;font-size:10px;font-weight:800;margin-right:5px}</style>",unsafe_allow_html=True)

@st.fragment(run_every=1)
def live_clock():
    pkt=datetime.now(ZoneInfo("Asia/Karachi"))
    st.markdown(f'<div class="top-clock"><div class="top-clock-box"><div class="top-clock-time"><span class="top-clock-live">● LIVE</span>{pkt.strftime("%I:%M:%S %p")}</div><div class="top-clock-date">🇵🇰 PKT • {pkt.strftime("%d %b %Y")}</div></div></div>',unsafe_allow_html=True)

live_clock()

# ----------------------------- header -----------------------------
now=datetime.now(ZoneInfo("Asia/Karachi")).strftime("%I:%M:%S %p")
st.markdown(f'<div class="hero"><div class="brand"><div class="brand-badge">🟢</div><div><h1>COCO Nexus</h1><div class="hero-sub">Crypto Signal Scanner • Binance USDT Spot • <span class="live"><span class="live-dot"></span>LIVE</span> • Updated {now} PKT</div></div></div></div>',unsafe_allow_html=True)

st.markdown('<div class="small-note" style="margin:4px 2px 12px">⚡ Scans are optimized for speed by prioritizing high-volume active pairs. Increase the scan universe only when you need wider coverage.</div>',unsafe_allow_html=True)

# ----------------------------- top stats -----------------------------
total=len(ss); moves=[safe_change(tickers.get(s,{})) for s in ss]; up=sum(x>0 for x in moves); down=sum(x<0 for x in moves)
market="BULLISH" if up>down else "BEARISH" if down>up else "NEUTRAL"; breadth=(up/total*100 if total else 0)
cols=st.columns(5)
vals=[("USDT PAIRS",total,""),("GAINERS",up,"green"),("LOSERS",down,"redtxt"),("MARKET",market,"green" if market=="BULLISH" else "redtxt"),("BREADTH",f"{breadth:.0f}%","blue")]
for col,(k,v,c) in zip(cols,vals):
    with col:st.markdown(f'<div class="stat"><div class="stat-k">{k}</div><div class="stat-v {c}">{v}</div></div>',unsafe_allow_html=True)

# ----------------------------- movers + main content -----------------------------
left,right=st.columns([4.2,1.25],gap="large")
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

        @st.fragment(run_every=15)
        def smart_entry_panel():
            st.markdown('<div class="panel"><div class="panel-title">🧠 Smart Entry Scanner <span class="live">LIVE</span></div><div class="muted">Early + Confirmed price-action confluence. Status automatically upgrades without pressing Scan again.</div>',unsafe_allow_html=True)
            a1,a2,a3=st.columns(3)
            with a1:
                ctf=st.selectbox("Signal timeframe",RSI_TFS,index=1,key="smart_tf")
                minscore=st.slider("Minimum confirmed score",60,100,80,5,key="smart_score")
                conf_universe=st.selectbox("Scan universe",[40,60,100,150,200],index=1,key="smart_universe")
            with a2:
                auto=st.toggle("🔄 Auto monitor",value=True,key="smart_auto")
                show_early=st.checkbox("Show EARLY",True,key="smart_show_early")
                show_confirmed=st.checkbox("Show CONFIRMED",True,key="smart_show_confirmed")
            with a3:
                wick=st.slider("HA max lower-wick / body",0.0,1.0,.20,.05,key="smart_wick")
                tp1_pct=st.number_input("TP1 %",0.1,20.0,1.0,0.1,key="smart_tp1")
                tp2_pct=st.number_input("TP2 %",0.1,50.0,2.0,0.1,key="smart_tp2")
                sl_pct=st.number_input("SL %",0.1,20.0,1.0,0.1,key="smart_sl")
            b1,b2=st.columns(2)
            with b1:manual=st.button("⚡ UPDATE NOW",use_container_width=True,key="smart_update")
            with b2:st.caption("🟡 EARLY → 🟢 CONFIRMED • checks every 15 seconds when Auto monitor is ON")
            if "smart_states" not in st.session_state: st.session_state.smart_states={}
            now_ts=time.time()
            due=now_ts>=st.session_state.get("smart_next",0)
            if manual or (auto and due) or not st.session_state.get("smart_initialized",False):
                settings={"tp1_pct":tp1_pct,"tp2_pct":tp2_pct,"sl_pct":sl_pct}
                candidates=active_symbols(conf_universe)
                with st.spinner(f"Smart scan: {len(candidates)} active pairs..."):
                    fresh=scan_smart_confluence(candidates,ctf,minscore,settings,wick)
                for z in fresh:
                    old=st.session_state.smart_states.get(z["Coin"])
                    if old and old.get("Status")=="EARLY" and z["Status"]=="CONFIRMED":
                        z["UpgradedAt"]=datetime.now(ZoneInfo("Asia/Karachi")).strftime("%H:%M:%S")
                    st.session_state.smart_states[z["Coin"]]=z
                # Remove stale EARLY records after 30 minutes; confirmed records persist for 30 minutes too.
                cutoff=now_ts-1800
                st.session_state.smart_states={k:v for k,v in st.session_state.smart_states.items() if v.get("_ts",now_ts)>=cutoff or v in fresh}
                for z in fresh: st.session_state.smart_states[z["Coin"]]["_ts"]=now_ts
                st.session_state.smart_next=now_ts+15; st.session_state.smart_initialized=True
            rows=list(st.session_state.smart_states.values())
            rows=[r for r in rows if (r.get("Status")=="EARLY" and show_early) or (r.get("Status")=="CONFIRMED" and show_confirmed)]
            rows=sorted(rows,key=lambda x:(x["Status"]=="CONFIRMED",x["Score"]),reverse=True)
            if rows:
                for z in rows[:20]:
                    badge_cls="buy" if z["Status"]=="CONFIRMED" else "watch"
                    checks=" • ".join([("✓ " if ok else "○ ")+name for name,ok in z["Checks"] if ok])
                    levels=f'<span class="lvl entry">Entry {z["Entry"]:.8g}</span><span class="lvl tp">TP1 {z["TP1"]:.8g} (+{z["TP1 %"]:.1f}%)</span><span class="lvl tp">TP2 {z["TP2"]:.8g} (+{z["TP2 %"]:.1f}%)</span><span class="lvl sl">SL {z["SL"]:.8g} (-{z["SL %"]:.1f}%)</span>'
                    pa=f'BOS {"✓" if z["BOS"] else "○"} • CHoCH {"✓" if z["CHoCH"] else "○"} • HH/HL {"✓" if z["HH/HL"] else "○"} • Sweep {"✓" if z["Liquidity Sweep"] else "○"} • HA {"✓" if z["HA"] else "○"}'
                    st.markdown(f'<div class="signal"><div class="signal-top"><div><b>{z["Coin"].replace("USDT","")}</b> <span class="badge {badge_cls}">{z["Status"]}</span></div><div class="score green">{z["Score"]}</div></div><div class="muted">RSI {z["RSI"]} • Volume {z["Vol x"]}x • {pa} • <a href="{z["TradingView"]}" target="_blank">TradingView ↗</a></div><div class="levels">{levels}</div><div class="checks">{checks}</div></div>',unsafe_allow_html=True)
            else:
                st.info("No EARLY/CONFIRMED smart-entry setup right now. Auto monitor will keep checking.")
            st.markdown('</div>',unsafe_allow_html=True)
        smart_entry_panel()

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
        f1,f2,f3,f4,f5,f6=st.columns(6)
        with f1:full_tp1=st.checkbox("Show TP1",True,key="full_tp1")
        with f2:full_tp2=st.checkbox("Show TP2",True,key="full_tp2")
        with f3:full_sl=st.checkbox("Show SL",True,key="full_sl")
        with f4:full_tp1_pct=st.number_input("TP1 %",0.1,20.0,1.0,0.1,key="full_tp1_pct")
        with f5:full_tp2_pct=st.number_input("TP2 %",0.1,50.0,2.0,0.1,key="full_tp2_pct")
        with f6:full_sl_pct=st.number_input("SL %",0.1,20.0,1.0,0.1,key="full_sl_pct")
        q=st.button("🚀 RUN FULL CONFLUENCE SCAN",use_container_width=True,key="full_conf_scan")
        if q:
            settings={"ema200":True,"ema933":True,"rsi":True,"volume":True,"breakout":True,"ha":True,"tp1_pct":full_tp1_pct,"tp2_pct":full_tp2_pct,"sl_pct":full_sl_pct}
            candidates=active_symbols(full_universe)
            with st.spinner(f"Scanning {len(candidates)} active pairs..."):st.session_state.c=scan_confluence(candidates,tf,score,settings,.25)
        if st.session_state.c:
            full_rows=[]
            for z in st.session_state.c:
                row={k:v for k,v in z.items() if k not in ("Checks","TP1 %","TP2 %","SL %","TP1","TP2","SL")}
                if full_tp1: row[f"TP1 (+{full_tp1_pct:.1f}%)"]=z["TP1"]
                if full_tp2: row[f"TP2 (+{full_tp2_pct:.1f}%)"]=z["TP2"]
                if full_sl: row[f"SL (-{full_sl_pct:.1f}%)"]=z["SL"]
                full_rows.append(row)
            st.dataframe(pd.DataFrame(full_rows),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})
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

st.markdown('<div class="small-note" style="text-align:center;margin-top:18px">COCO Nexus • Technical scanner only • Binance Public Spot API • TradingView links open charts • Not financial advice</div>',unsafe_allow_html=True)















