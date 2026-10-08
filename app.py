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


def structure_signal(df):
    x=df.iloc[:-1]
    if len(x)<30:
        return {"bos":False,"choch":False,"hh":False,"hl":False,"sweep":False}
    h=x["h"].to_numpy(); l=x["l"].to_numpy(); c=x["c"].to_numpy()
    # Conservative structure: compare two closed swing windows.
    ph_old=float(max(h[-12:-6])); ph_new=float(max(h[-6:-1]))
    pl_old=float(min(l[-12:-6])); pl_new=float(min(l[-6:-1]))
    hh=ph_new>ph_old
    hl=pl_new>pl_old
    bos= c[-1] > ph_old
    prior_bearish = pl_new < pl_old
    choch = bos and prior_bearish
    # Liquidity sweep: the second-last closed candle takes a prior low
    # and closes back above that prior-low level. The prior-low window
    # deliberately excludes the sweep candle itself.
    prior_low=float(min(l[-8:-2]))
    sweep=float(l[-2]) < prior_low and float(c[-2]) > prior_low
    return {"bos":bos,"choch":choch,"hh":hh,"hl":hl,"sweep":sweep}

def tf_state(symbol,tf,wick=.20):
    d=klines(symbol,tf)
    if d is None:return None
    close=float(d["c"].iloc[-2])
    e9,e33,e200=ema_vals(d)
    rv=rsi(d); rp=rsi_prev(d)
    vol=float(d["v"].iloc[-2])
    avg=float(d["v"].iloc[-22:-2].mean()) if len(d)>24 else vol
    vr=vol/avg if avg else 1
    pa=structure_signal(d)
    return {
        "close":close,"e9":float(e9.iloc[-1]),"e33":float(e33.iloc[-1]),
        "e200":float(e200.iloc[-1]),"rsi":rv,"rsi_prev":rp,"volx":vr,
        "ha":ha_signal(d,wick),"pa":pa
    }

def scan_mtf_early_confirmed(ss,early_min=60,confirm_min=80,wick=.20,
                             tp1_pct=1.0,tp2_pct=2.0,sl_pct=1.0):
    """
    5m = EARLY timing
    15m = ENTRY confirmation
    1h = confirmation
    4h = trend filter
    The scan keeps EARLY rows visible so they can become CONFIRMED
    on the next automatic refresh.
    """
    out=[]
    def one(s):
        a=tf_state(s,"5m",wick); b=tf_state(s,"15m",wick)
        c=tf_state(s,"1h",wick); d=tf_state(s,"4h",wick)
        if not all([a,b,c,d]): return None

        early_checks = [
            a["e9"]>a["e33"],
            a["rsi"]>a["rsi_prev"] and 45<=a["rsi"]<=68,
            a["ha"],
            a["volx"]>=1.20,
            a["pa"]["bos"] or a["pa"]["choch"] or a["pa"]["sweep"],
            b["close"]>=b["e33"]
        ]
        early_score=sum(15 for x in early_checks if x)
        if early_score < early_min:
            return None

        confirm_checks = [
            b["e9"]>b["e33"],
            b["rsi"]>b["rsi_prev"] and 48<=b["rsi"]<=68,
            b["ha"],
            b["volx"]>=1.20,
            b["pa"]["bos"] or b["pa"]["choch"],
            c["e9"]>c["e33"] and c["close"]>c["e200"],
            c["rsi"]>=50,
            d["close"]>d["e200"] and d["e9"]>d["e33"]
        ]
        confirm_score=sum(12.5 for x in confirm_checks if x)
        confirmed=confirm_score>=confirm_min

        status="CONFIRMED" if confirmed else "EARLY"
        stage="5m ✓ → 15m ✓ → 1H ✓ → 4H ✓" if confirmed else "5m ✓ → Waiting for 15m/1H confirmation"
        entry=b["close"] if confirmed else a["close"]

        checks = [
            ("5m EMA9>33",a["e9"]>a["e33"]),
            ("5m RSI rising",a["rsi"]>a["rsi_prev"]),
            ("5m HA",a["ha"]),
            ("5m structure",a["pa"]["bos"] or a["pa"]["choch"] or a["pa"]["sweep"]),
            ("5m HH",a["pa"]["hh"]),
            ("5m HL",a["pa"]["hl"]),
            ("5m Liquidity Sweep",a["pa"]["sweep"]),
            ("15m EMA9>33",b["e9"]>b["e33"]),
            ("15m BOS/CHoCH",b["pa"]["bos"] or b["pa"]["choch"]),
            ("1H bullish",c["close"]>c["e200"] and c["e9"]>c["e33"]),
            ("4H bullish",d["close"]>d["e200"] and d["e9"]>d["e33"])
        ]
        return {
            "Coin":s,"Status":status,"Stage":stage,
            "Score":round(confirm_score if confirmed else early_score,1),
            "Signal":"CONFIRMED BUY" if confirmed else "EARLY BUY",
            "Entry":entry,
            "TP1":entry*(1+tp1_pct/100),"TP2":entry*(1+tp2_pct/100),
            "SL":entry*(1-sl_pct/100),
            "TP1 %":tp1_pct,"TP2 %":tp2_pct,"SL %":sl_pct,
            "5m RSI":round(a["rsi"],1),"15m RSI":round(b["rsi"],1),
            "1H RSI":round(c["rsi"],1),"4H RSI":round(d["rsi"],1),
            "BOS":bool(b["pa"]["bos"]),"CHoCH":bool(b["pa"]["choch"]),
            "HH":bool(b["pa"]["hh"]),"HL":bool(b["pa"]["hl"]),
            "Liquidity Sweep":bool(b["pa"]["sweep"]),
            "TradingView":tv(s,"15m"),"Checks":checks
        }

    with ThreadPoolExecutor(max_workers=12) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z: out.append(z)
            except Exception:
                pass
    return sorted(out,key=lambda z:(1 if z["Status"]=="CONFIRMED" else 0,z["Score"]),reverse=True)

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
        tp1_pct=float(settings.get("tp1_pct",1.0)); tp2_pct=float(settings.get("tp2_pct",2.0)); sl_pct=float(settings.get("sl_pct",1.0))
        return {
            "Coin":s,"Score":score,"Signal":label,"Entry":close,
            "TP1":close*(1+tp1_pct/100),"TP2":close*(1+tp2_pct/100),"SL":close*(1-sl_pct/100),
            "TP1 %":tp1_pct,"TP2 %":tp2_pct,"SL %":sl_pct,
            "RSI":round(rv,1),"Vol x":round(vr,2),"TradingView":tv(s,tf),"Checks":checks
        }
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


















