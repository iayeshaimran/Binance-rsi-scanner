import streamlit as st
import requests, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Crypto Signal Scanner", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.stApp{background:#080b12;color:#f5f7fa}
.block-container{max-width:1500px;padding-top:1rem}
.hero,.card{background:#101824;border:1px solid #243247;border-radius:14px;padding:16px;margin-bottom:12px}
.hero h1{margin:0}.muted{color:#8995a8}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:7px}
.coin{padding:9px;border-radius:9px;text-align:center;background:#123728;border:1px solid #255d43}
.red{background:#39211d;border-color:#704139}.orange{background:#302b18;border-color:#6b5a27}
@media(max-width:900px){.grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:500px){.grid{grid-template-columns:repeat(2,1fr)}}
</style>
""", unsafe_allow_html=True)

BASE="https://data-api.binance.vision"

@st.cache_data(ttl=300)
def symbols():
    try:
        d=requests.get(BASE+"/api/v3/exchangeInfo",timeout=15).json()
        return sorted([x["symbol"] for x in d["symbols"] if x["status"]=="TRADING" and x["quoteAsset"]=="USDT" and x.get("isSpotTradingAllowed",True)])
    except: return []

def klines(s,tf,limit=80):
    try:
        d=requests.get(BASE+"/api/v3/klines",params={"symbol":s,"interval":tf,"limit":limit},timeout=10).json()
        if not isinstance(d,list) or len(d)<30:return None
        x=pd.DataFrame(d,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","i"])
        for c in ["o","h","l","c"]:x[c]=pd.to_numeric(x[c])
        return x
    except:return None

def rsi(df):
    d=df["c"].diff();g=d.clip(lower=0);l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/14,adjust=False).mean();al=l.ewm(alpha=1/14,adjust=False).mean()
    return float((100-100/(1+ag/al.replace(0,np.nan))).fillna(50).iloc[-2])

def tv(s,tf): return f"https://www.tradingview.com/chart/?symbol=BINANCE:{s}&interval="+{"3m":"3","5m":"5","15m":"15","1h":"60","4h":"240"}.get(tf,"15")

def ema_signal(df):
    x=df.iloc[:-1];e9=x.c.ewm(span=9,adjust=False).mean();e33=x.c.ewm(span=33,adjust=False).mean()
    return e9.iloc[-2]<=e33.iloc[-2] and e9.iloc[-1]>e33.iloc[-1],e9.iloc[-1],e33.iloc[-1]

def ha_signal(df,wick=.25):
    x=df.iloc[:-1];hc=(x.o+x.h+x.l+x.c)/4;ho=np.zeros(len(x));ho[0]=(x.o.iloc[0]+x.c.iloc[0])/2
    for i in range(1,len(x)):ho[i]=(ho[i-1]+hc.iloc[i-1])/2
    hh=np.maximum.reduce([x.h.to_numpy(),ho,hc.to_numpy()]);hl=np.minimum.reduce([x.l.to_numpy(),ho,hc.to_numpy()])
    a,b=-2,-1
    if not(hc.iloc[a]>ho[a] and hc.iloc[b]>ho[b] and hc.iloc[b]>hc.iloc[a] and hh[b]>hh[a]):return False
    ba=abs(hc.iloc[a]-ho[a]);bb=abs(hc.iloc[b]-ho[b])
    if ba==0 or bb==0:return False
    return (min(ho[a],hc.iloc[a])-hl[a])/ba<=wick and (min(ho[b],hc.iloc[b])-hl[b])/bb<=wick

def scan_rsi(ss,tf,confirm,rr,cr,direction):
    out=[]
    def one(s):
        a=klines(s,tf);b=klines(s,confirm)
        if a is None or b is None:return None
        x=rsi(a);y=rsi(b)
        if rr!="All" and not({"40 - 50":40<=x<50,"50 - 55":50<=x<55,"55 - 60":55<=x<60,"60 - 70":60<=x<70,"70+":x>=70}[rr]):return None
        if cr!="All" and not({"40 - 50":40<=y<50,"50 - 55":50<=y<55,"55 - 60":55<=y<60,"60 - 70":60<=y<70,"70+":y>=70}[cr]):return None
        prev=rsi(a.iloc[:-1]) if len(a)>35 else x
        dr="Rising" if x>prev else "Falling" if x<prev else "Flat"
        if direction!="All" and dr!=direction:return None
        return {"Coin":s,"RSI":round(x,2),"Confirm RSI":round(y,2),"Direction":dr,"TradingView":tv(s,tf)}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z:out.append(z)
            except:pass
    return out

def scan_ema(ss,tf):
    out=[]
    def one(s):
        d=klines(s,tf)
        if d is None:return None
        ok,e9,e33=ema_signal(d)
        return {"Coin":s,"EMA 9":round(e9,8),"EMA 33":round(e33,8),"Signal":"🟢 Fresh Bullish Cross","TradingView":tv(s,tf)} if ok else None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z:out.append(z)
            except:pass
    return out

def scan_ha(ss,tfs,wick,minimum):
    out=[]
    def one(s):
        good=[tf for tf in tfs if (lambda d: d is not None and ha_signal(d,wick))(klines(s,tf))]
        return {"Coin":s,"Bullish Timeframes":", ".join(good),"Alignment":f"{len(good)}/{len(tfs)}","TradingView":tv(s,good[0])} if len(good)>=minimum else None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(one,s) for s in ss]):
            try:
                z=f.result()
                if z:out.append(z)
            except:pass
    return out

ss=symbols()
st.markdown('<div class="hero"><h1>🚀 Crypto Signal Scanner</h1><div class="muted">Binance USDT Spot • RSI Heatmap • EMA 9/33 • Heikin Ashi • <b style="color:#35e88a">● LIVE</b></div></div>',unsafe_allow_html=True)

a,b,c,d=st.columns(4)
a.metric("USDT Pairs",len(ss));b.metric("API","LIVE");c.metric("RSI Signals",len(st.session_state.get("r",[])));d.metric("EMA Signals",len(st.session_state.get("e",[])))

st.subheader("📊 RSI Heatmap")
ht=st.selectbox("Heatmap timeframe",["5m","15m","1h","4h"],index=1)
if st.button("🔄 UPDATE RSI HEATMAP",use_container_width=True):
    with st.spinner("Loading RSI heatmap..."):
        vals=[]
        with ThreadPoolExecutor(max_workers=8) as ex:
            fs={ex.submit(klines,s,ht):s for s in ss[:80]}
            for f in as_completed(fs):
                try:
                    q=f.result()
                    if q is not None:vals.append((fs[f],rsi(q)))
                except:pass
        st.session_state["heat"]=vals
heat=st.session_state.get("heat",[])
if heat:
    for title,items,cls in [
        ("🔴 Oversold",[x for x in heat if x[1]<30],"red"),
        ("🟠 Neutral",[x for x in heat if 30<=x[1]<60],"orange"),
        ("🟢 Gainers Zone",[x for x in heat if x[1]>=60],"")]:
        st.markdown(f"**{title} — {len(items)} coins**",unsafe_allow_html=True)
        cards = []
        for s, v in items:
            name = s.replace("USDT", "")
            link = tv(s, ht)
            cards.append(
                f'<a href="{link}" target="_blank" class="coin {cls}" '
                f'style="text-decoration:none;color:inherit;display:block;">'
                f'<b>{name}</b><br><small>RSI: {v:.1f}</small>'
                f'</a>'
            )
        html = '<div class="grid">' + ''.join(cards) + '</div>'
        st.markdown(html, unsafe_allow_html=True)
else:st.info("Update the heatmap to load coins.")

st.subheader("🔍 RSI Scanner")
x1,x2,x3=st.columns(3)
with x1:pt=st.selectbox("Primary timeframe",["5m","15m","1h","4h"],index=1);pr=st.selectbox("Primary RSI range",["All","40 - 50","50 - 55","55 - 60","60 - 70","70+"])
with x2:ct=st.selectbox("Confirmation timeframe",["5m","15m","1h","4h"],index=2);cr=st.selectbox("Confirmation RSI range",["All","40 - 50","50 - 55","55 - 60","60 - 70","70+"])
with x3:di=st.selectbox("RSI direction",["All","Rising","Falling"]);st.checkbox("Closed candles only",True)
if st.button("🔍 SCAN RSI",use_container_width=True):
    with st.spinner(f"Scanning {len(ss)} coins..."):st.session_state["r"]=scan_rsi(ss,pt,ct,pr,cr,di)
if st.session_state.get("r"):st.dataframe(pd.DataFrame(st.session_state["r"]),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})

st.subheader("📈 EMA 9 / 33 Scanner")
et=st.selectbox("EMA timeframe",["5m","15m","1h","4h"],index=1)
if st.button("🔍 SCAN EMA 9/33",use_container_width=True):
    with st.spinner(f"Scanning {len(ss)} coins..."):st.session_state["e"]=scan_ema(ss,et)
if st.session_state.get("e"):st.dataframe(pd.DataFrame(st.session_state["e"]),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})

st.subheader("🕯️ Heikin Ashi Scanner")
tfs=st.multiselect("HA timeframes",["3m","5m","15m","1h","4h"],["3m","5m","15m","1h","4h"])
w=st.slider("Maximum lower wick / body",0.0,1.0,.25,.05)
minimum=st.slider("Minimum bullish HA timeframes",1,5,3)
if st.button("🔍 SCAN HEIKIN ASHI",use_container_width=True):
    with st.spinner(f"Scanning {len(ss)} coins..."):st.session_state["h"]=scan_ha(ss,tfs,w,minimum)
if st.session_state.get("h"):st.dataframe(pd.DataFrame(st.session_state["h"]),use_container_width=True,hide_index=True,column_config={"TradingView":st.column_config.LinkColumn("TradingView",display_text="Open Chart ↗")})

st.markdown("---")
st.caption("Technical scanner only • Binance Public Spot API • Not financial advice")



