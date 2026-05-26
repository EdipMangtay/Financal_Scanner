#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SWING TRADE SCANNER  -  Streamlit Web UI (production)
================================================================================
Local:  streamlit run swing_app.py
Docker: docker build -t fin-scanner . && docker run -p 8501:8501 fin-scanner
Railway: connect GitHub repo, auto-deploys from Dockerfile.
================================================================================
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import swing_engine as eng


# --------------------------------------------------------------------------- #
# SAYFA KONFIGURASYONU
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Swing Scanner · Karar Destek",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Multi-factor swing trade scanner. Karar destek aracidir, "
                 "finansal danisman degildir."
    },
)

CUSTOM_CSS = """
<style>
.stApp { background: radial-gradient(ellipse at top, #0b1220 0%, #060912 70%); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #060912 100%);
    border-right: 1px solid #1e293b;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #a855f7 100%);
    padding: 22px 28px; border-radius: 16px; margin-bottom: 18px;
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(255,255,255,0.06);
}
.hero h1 { color:#fff; margin:0; font-size:28px; font-weight:800; letter-spacing:-0.02em; }
.hero p  { color:rgba(255,255,255,0.85); margin:6px 0 0 0; font-size:14px; }
.hero-pill {
    display:inline-block; padding:4px 10px; background:rgba(255,255,255,0.18);
    border-radius:999px; color:#fff; font-size:11px; font-weight:600;
    letter-spacing:0.05em; margin-right:6px;
}

/* Metric kart */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 14px 16px; border-radius: 12px;
    border: 1px solid #1e293b;
    transition: transform 0.15s, border-color 0.15s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #334155; }
.metric-card .label { color:#94a3b8; font-size:11px; font-weight:600;
    letter-spacing:0.06em; text-transform:uppercase; }
.metric-card .value { color:#f1f5f9; font-size:26px; font-weight:800;
    margin-top:4px; line-height:1; }
.metric-card .sub   { color:#64748b; font-size:11px; margin-top:4px; }

/* Pick card */
.pick-card {
    background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
    padding: 16px; border-radius: 14px;
    border: 1px solid #064e3b;
    height: 100%;
}
.pick-card.zirvede { background: linear-gradient(135deg, #78350f 0%, #0f172a 100%);
    border-color:#78350f; }
.pick-card.asiri   { background: linear-gradient(135deg, #7f1d1d 0%, #0f172a 100%);
    border-color:#7f1d1d; }
.pick-card .ticker { color:#fff; font-size:22px; font-weight:800; }
.pick-card .name   { color:#94a3b8; font-size:11px; margin-bottom:8px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pick-card .price  { color:#f1f5f9; font-size:18px; font-weight:700; }
.pick-card .setup  { display:inline-block; padding:2px 8px; border-radius:999px;
    font-size:10px; font-weight:700; letter-spacing:0.05em; margin-top:8px; }

/* Regime */
.regime-risk_on  { color:#10b981 !important; }
.regime-risk_off { color:#ef4444 !important; }
.regime-neutral  { color:#f59e0b !important; }

/* Tablo */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Sidebar baslik */
section[data-testid="stSidebar"] h2 {
    background: linear-gradient(90deg, #10b981, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Footer */
.footer {
    text-align:center; padding:18px; color:#64748b; font-size:12px;
    border-top: 1px solid #1e293b; margin-top:24px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# YARDIMCILAR
# --------------------------------------------------------------------------- #
LABEL_COLORS = {
    "GUCLU AL": "#10b981",
    "AL":       "#22c55e",
    "IZLE":     "#3b82f6",
    "NOTR":     "#64748b",
    "KACIN":    "#ef4444",
}
SETUP_COLORS = {
    "TAZE":       "#10b981",
    "ZIRVEDE":    "#f59e0b",
    "ASIRI ALIM": "#ef4444",
    "":           "#1e293b",
}
SETUP_PILL_BG = {
    "TAZE":       "background:#10b98133; color:#10b981; border:1px solid #10b98155",
    "ZIRVEDE":    "background:#f59e0b33; color:#f59e0b; border:1px solid #f59e0b55",
    "ASIRI ALIM": "background:#ef444433; color:#ef4444; border:1px solid #ef444455",
}


def style_decision_table(df: pd.DataFrame):
    def color_label(val):
        c = LABEL_COLORS.get(val, "#64748b")
        return f"background-color:{c}; color:white; font-weight:700; text-align:center;"

    def color_setup(val):
        c = SETUP_COLORS.get(val, "#1e293b")
        if not val:
            return ""
        return f"background-color:{c}; color:white; font-weight:600; text-align:center; font-size:11px;"

    fmt = {
        "Skor": "{:.2f}", "Fiyat": "{:.2f}",
        "Mom_20g_%": "{:+.1f}", "Pullback_%": "{:+.1f}",
        "RSI": "{:.1f}", "MACD_H": "{:+.3f}",
        "ATR_%": "{:.2f}", "Hacim_x": "{:.2f}",
        "Giris": "{:.2f}", "Stop": "{:.2f}",
        "Hedef": "{:.2f}", "R:R": "{:.2f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    styled = df.style.map(color_label, subset=["KARAR"])
    if "SETUP" in df.columns:
        styled = styled.map(color_setup, subset=["SETUP"])
    return styled.format(fmt)


@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(scope: str, period: str):
    return eng.run_scan(scope=scope, period=period)


def run_scan_with_progress(scope: str, period: str):
    bar = st.progress(0.0)
    txt = st.empty()

    def cb(done, total, msg):
        bar.progress(min(done / max(total, 1), 1.0))
        txt.caption(msg)

    res = eng.run_scan(scope=scope, period=period, progress_callback=cb)
    bar.empty()
    txt.empty()
    return res


def regime_badge(regime: str) -> str:
    label = {"risk_on": "RISK-ON", "risk_off": "RISK-OFF", "neutral": "NOTR"}.get(regime, "?")
    return f'<span class="regime-{regime}">{label}</span>'


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlc(ticker: str, period: str = "1y"):
    """Tek hisse icin OHLC + Hacim — mum grafik icin."""
    import yfinance as yf
    try:
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, threads=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def build_candlestick(ticker: str, action_row, period: str = "6mo"):
    df = fetch_ohlc(ticker, period=period)
    if df.empty:
        return None
    df = df.iloc[-150:]
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()
    ma200_full = fetch_ohlc(ticker, period="1y")
    ma200 = ma200_full["Close"].rolling(200).mean().iloc[-150:] if not ma200_full.empty else None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
        vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker,
        increasing=dict(line=dict(color="#10b981"), fillcolor="#10b981"),
        decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=ma20.index, y=ma20.values, name="MA20",
                             line=dict(color="#22d3ee", width=1.2, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, name="MA50",
                             line=dict(color="#f59e0b", width=1.2, dash="dash")),
                  row=1, col=1)
    if ma200 is not None:
        fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, name="MA200",
                                 line=dict(color="#a855f7", width=1.4)),
                      row=1, col=1)

    if action_row is not None and action_row["KARAR"] in ("GUCLU AL", "AL"):
        for level, color, name in [
            (action_row["Giris"], "#10b981", "Giriş"),
            (action_row["Stop"], "#ef4444", "Stop"),
            (action_row["Hedef"], "#3b82f6", "Hedef"),
        ]:
            fig.add_hline(y=level, line=dict(color=color, width=1.4, dash="dot"),
                          annotation_text=f"{name}: {level:.2f}",
                          annotation_position="right",
                          annotation_font_color=color,
                          row=1, col=1)

    colors = ["#10b981" if c >= o else "#ef4444"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors,
                         name="Hacim", showlegend=False, opacity=0.7),
                  row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=580,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#060912", paper_bgcolor="#060912",
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        xaxis2=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", title="Fiyat"),
        yaxis2=dict(showgrid=False, title="Hacim"),
        legend=dict(orientation="h", y=1.04, yanchor="bottom"),
    )
    return fig


def make_pick_card(row: pd.Series) -> str:
    setup = row.get("SETUP", "") or ""
    klass = setup.lower().replace(" ", "")
    if klass == "asirialim":
        klass = "asiri"
    pill_style = SETUP_PILL_BG.get(setup, "background:#1e293b;color:#94a3b8")
    pull = row.get("Pullback_%", 0)
    rsi = row.get("RSI", 50)
    setup_html = (
        f'<span class="setup" style="{pill_style}">{setup}</span>'
        if setup else ""
    )
    karar_color = LABEL_COLORS.get(row["KARAR"], "#94a3b8")

    return f"""
<div class="pick-card {klass}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div class="ticker">{row['Ticker']}</div>
    <div style="background:{karar_color}; color:white; font-size:10px;
                font-weight:700; padding:3px 8px; border-radius:6px;
                letter-spacing:0.05em;">{row['KARAR']}</div>
  </div>
  <div class="name">{row['Varlik']}</div>
  <div class="price">${row['Fiyat']:,.2f}</div>
  <div style="color:#64748b; font-size:11px; margin-top:6px;">
    Skor <b style="color:#cbd5e1">{row['Skor']:.2f}</b>
    &nbsp;·&nbsp; RSI <b style="color:#cbd5e1">{rsi:.0f}</b>
    &nbsp;·&nbsp; Pullback <b style="color:#cbd5e1">{pull:+.1f}%</b>
  </div>
  <div style="color:#94a3b8; font-size:11px; margin-top:4px;">
    Stop <span style="color:#ef4444">${row['Stop']:.2f}</span>
    &nbsp;·&nbsp; Hedef <span style="color:#3b82f6">${row['Hedef']:.2f}</span>
    &nbsp;·&nbsp; R:R <b style="color:#10b981">{row['R:R']:.2f}</b>
  </div>
  {setup_html}
</div>
"""


# --------------------------------------------------------------------------- #
# SIDEBAR
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚙ Ayarlar")

    scope_options = {
        "curated": "🚀 Hızlı (130 varlık)",
        "nasdaq":  "💻 Nasdaq-100 + ETF (~230)",
        "sp500":   "🇺🇸 S&P 500 + ETF (~530)",
        "sp1500":  "🌐 S&P 1500 + ETF (TÜM, ~1530)",
    }
    scope = st.radio(
        "📡 Tarama kapsamı",
        options=list(scope_options.keys()),
        format_func=lambda k: scope_options[k],
        index=2,
        help="Genişlettikçe ilk indirme uzar (30-90 sn), sonra 15 dk cache.",
    )

    period = st.selectbox("🕐 Veri penceresi", ["1y", "2y", "5y"], index=0)

    st.markdown("---")
    st.markdown("### 🎚 Filtreler")
    top_n = st.slider("Tabloda kaç varlık?", 20, 1500, 200)
    min_score = st.slider("Min. skor", -2.0, 2.0, -2.0, 0.1)
    show_only_buys = st.checkbox("Sadece AL / GÜÇLÜ AL", value=False)
    only_fresh = st.checkbox(
        "Sadece TAZE setup'lar",
        value=False,
        help="Pullback bölgesindeki adayları gösterir, zirvede olanları gizler.",
    )

    st.markdown("---")
    st.markdown("### 💰 Pozisyon Hesaplayıcı")
    portfolio = st.number_input("Portföy ($)", min_value=100.0, value=10000.0,
                                step=500.0, format="%.0f")
    risk_pct = st.slider("İşlem başına risk (%)", 0.25, 5.0, 1.0, 0.25,
                         help="Tek işlemde portföyünün ne kadarını riske atıyorsun.")
    st.caption(
        f"💡 Tek işlem riski: **${portfolio * risk_pct / 100:,.0f}**.\n"
        "Detaylı hesap aşağıdaki seçili varlık panelinde."
    )

    st.markdown("---")
    if st.button("🔄 Veriyi Yenile (cache temizle)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(
        "Bu sistem **AL emri vermez**. Aday gösterir, giriş/stop/hedef önerir. "
        "Tetiği HTF analizinle SEN çek. Finansal danışman değildir."
    )


# --------------------------------------------------------------------------- #
# HERO
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
<div class="hero">
  <span class="hero-pill">📈 SWING TRADE SCANNER</span>
  <span class="hero-pill">v1.0</span>
  <span class="hero-pill">{scope_options[scope]}</span>
  <h1>Karar Destek — Çok Faktörlü Swing Tarama</h1>
  <p>9 faktörlü kompozit skor · ATR-stop'lu giriş/hedef · otomatik rejim tespiti · 
     {datetime.today().strftime('%d %b %Y, %H:%M')}</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# TARAMA
# --------------------------------------------------------------------------- #
cache_key = (scope, period)
if "last_scan" not in st.session_state:
    st.session_state.last_scan = {}

if cache_key in st.session_state.last_scan:
    res = st.session_state.last_scan[cache_key]
else:
    info_box = st.info(
        f"İlk tarama: **{scope_options[scope]}** indiriliyor. "
        "Büyük kapsamda 30-90 saniye sürebilir; sonra 15 dakika önbellektedir."
    )
    res = run_scan_with_progress(scope, period)
    st.session_state.last_scan[cache_key] = res
    info_box.empty()

if not res.get("ok"):
    st.error(f"Veri çekilemedi: {res.get('status', 'unknown')}")
    st.stop()

table: pd.DataFrame = res["table"]
regime: str = res["regime"]
close: pd.DataFrame = res["close"]
last_date = pd.to_datetime(res["last_date"]).strftime("%Y-%m-%d")
n_buy = int((table["KARAR"] == "AL").sum())
n_strong = int((table["KARAR"] == "GUCLU AL").sum())
n_watch = int((table["KARAR"] == "IZLE").sum())
n_avoid = int((table["KARAR"] == "KACIN").sum())
n_fresh = int(((table["KARAR"].isin(["GUCLU AL", "AL"])) & (table["SETUP"] == "TAZE")).sum())
n_total = res.get("universe_size", len(table))

# --------------------------------------------------------------------------- #
# METRIK SATIRI
# --------------------------------------------------------------------------- #
metric_html = """
<div style="display:grid; grid-template-columns: repeat(6, 1fr); gap:12px; margin-bottom:18px;">
  <div class="metric-card"><div class="label">REJİM</div>
    <div class="value">{regime_html}</div><div class="sub">son veri: {last_date}</div></div>
  <div class="metric-card"><div class="label">TARANAN</div>
    <div class="value" style="color:#cbd5e1">{n_total}</div><div class="sub">{status}</div></div>
  <div class="metric-card"><div class="label">GÜÇLÜ AL</div>
    <div class="value" style="color:#10b981">{n_strong}</div><div class="sub">top %8</div></div>
  <div class="metric-card"><div class="label">AL</div>
    <div class="value" style="color:#22c55e">{n_buy}</div><div class="sub">top %22</div></div>
  <div class="metric-card"><div class="label">🟢 TAZE SETUP</div>
    <div class="value" style="color:#06b6d4">{n_fresh}</div><div class="sub">pullback bölgesi</div></div>
  <div class="metric-card"><div class="label">KAÇIN</div>
    <div class="value" style="color:#ef4444">{n_avoid}</div><div class="sub">bottom %20</div></div>
</div>
"""
st.markdown(
    metric_html.format(
        regime_html=regime_badge(regime),
        last_date=last_date,
        n_total=n_total,
        status=res.get("status", ""),
        n_strong=n_strong, n_buy=n_buy, n_fresh=n_fresh, n_avoid=n_avoid,
    ),
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# TOP 5 PICKS KARTLARI (sadece TAZE varsa)
# --------------------------------------------------------------------------- #
fresh_picks = table[
    table["KARAR"].isin(["GUCLU AL", "AL"]) & (table["SETUP"] == "TAZE")
].head(5)

if len(fresh_picks) > 0:
    st.markdown("### 🎯 En İyi Taze Setup'lar (pullback bölgesinde)")
    cols = st.columns(min(5, len(fresh_picks)))
    for i, (_, row) in enumerate(fresh_picks.iterrows()):
        with cols[i]:
            st.markdown(make_pick_card(row), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# TABLAR
# --------------------------------------------------------------------------- #
tab_buys, tab_chart, tab_full, tab_sector, tab_help = st.tabs([
    "🎯 Alım Listesi",
    "📊 Detay Grafik",
    "📋 Tam Tarama",
    "🗺 Sektör Isı Haritası",
    "ℹ Rehber",
])

# --- TAB 1: ALIM LISTESI ---
with tab_buys:
    buys = table[table["KARAR"].isin(["GUCLU AL", "AL"])].reset_index(drop=True)
    if only_fresh:
        buys = buys[buys["SETUP"] == "TAZE"].reset_index(drop=True)
    st.markdown(f"#### {len(buys)} aday")
    if only_fresh:
        st.caption("Filtre: sadece **TAZE** setup'lar gösteriliyor.")
    if buys.empty:
        st.info("Bugün AL kriterini geçen yok. Bu da bir bilgidir — piyasa "
                "pahalı veya zayıf olabilir. Beklemek de bir karardır.")
    else:
        buy_cols = ["Ticker", "Varlik", "KARAR", "SETUP", "Skor", "Yuzde",
                    "Fiyat", "Trend", "Mom_20g_%", "Pullback_%", "RSI",
                    "MACD_H", "ATR_%", "Hacim_x", "52H_Pos_%",
                    "Giris", "Stop", "Hedef", "R:R"]
        buy_cols = [c for c in buy_cols if c in buys.columns]
        st.dataframe(
            style_decision_table(buys[buy_cols]),
            use_container_width=True, hide_index=True,
            height=min(600, 56 + 36 * len(buys)),
        )
        csv = buys[buy_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ CSV indir", data=csv,
            file_name=f"alim_listesi_{scope}_{last_date}.csv", mime="text/csv",
        )

# --- TAB 2: DETAY GRAFIK ---
with tab_chart:
    default_options = (
        fresh_picks["Ticker"].tolist()
        if len(fresh_picks) > 0
        else table["Ticker"].tolist()
    )
    sel_ticker = st.selectbox(
        "Hangi varlığa bakmak istersin?",
        options=table["Ticker"].tolist(),
        index=table["Ticker"].tolist().index(default_options[0]) if default_options else 0,
    )
    action_row = table[table["Ticker"] == sel_ticker].iloc[0]

    with st.spinner(f"{sel_ticker} mum grafiği yükleniyor..."):
        fig = build_candlestick(sel_ticker, action_row)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Fiyat", f"${action_row['Fiyat']:.2f}")
    g2.metric("Giriş", f"${action_row['Giris']:.2f}")
    g3.metric("Stop", f"${action_row['Stop']:.2f}",
              delta=f"{((action_row['Stop']/action_row['Fiyat'])-1)*100:+.1f}%",
              delta_color="inverse")
    g4.metric("Hedef", f"${action_row['Hedef']:.2f}",
              delta=f"{((action_row['Hedef']/action_row['Fiyat'])-1)*100:+.1f}%")
    g5.metric("R:R", f"{action_row['R:R']:.2f}")

    karar = action_row["KARAR"]
    setup = action_row.get("SETUP", "")
    karar_color = LABEL_COLORS.get(karar, "#64748b")
    setup_html = (
        f'<span style="background:{SETUP_COLORS[setup]}33;color:{SETUP_COLORS[setup]};'
        f'border:1px solid {SETUP_COLORS[setup]}55; padding:3px 10px;border-radius:999px;'
        f'font-size:12px;font-weight:700;margin-left:8px;">{setup}</span>'
        if setup else ""
    )
    st.markdown(
        f'<div style="padding:14px 18px; background:{karar_color}22; '
        f'border-left:4px solid {karar_color}; border-radius:8px; margin-top:12px;">'
        f'<b style="color:{karar_color}; font-size:18px;">{karar}</b>{setup_html}'
        f'<br/><span style="color:#94a3b8;font-size:13px;">'
        f'Skor {action_row["Skor"]:.2f} ({action_row["Yuzde"]}. yüzdelik)  '
        f'· Trend {action_row["Trend"]}  · RSI {action_row["RSI"]:.1f}  '
        f'· MACD {action_row["MACD_H"]:+.3f}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # --- POZISYON HESAPLAYICI ---
    if karar in ("GUCLU AL", "AL"):
        st.markdown("---")
        st.markdown("### 💰 Pozisyon Boyutu Hesabı")
        risk_dollar = portfolio * risk_pct / 100
        stop_distance = action_row["Fiyat"] - action_row["Stop"]
        position_dollar = (risk_dollar / stop_distance) * action_row["Fiyat"] if stop_distance > 0 else 0
        shares = int(position_dollar / action_row["Fiyat"]) if action_row["Fiyat"] > 0 else 0

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Risk ($)", f"${risk_dollar:,.0f}", help="Portföy × Risk%")
        pc2.metric("Stop Mesafesi", f"${stop_distance:.2f}",
                   help="Giriş - Stop")
        pc3.metric("Pozisyon Büyüklüğü", f"${position_dollar:,.0f}",
                   help="Yanılırsan tam risk_dollar kaybedersin")
        pc4.metric("Hisse Adedi", f"{shares} adet",
                   help="Tam sayı")

        st.caption(
            f"➡ **${portfolio:,.0f} portföy** ile, **{sel_ticker}** için "
            f"**{shares} adet** (~${position_dollar:,.0f}) al. "
            f"Stop'a takılırsan ${risk_dollar:,.0f} kaybedersin "
            f"(portföyünün %{risk_pct:.2f}'i). "
            f"Hedefe gelirse +${risk_dollar * action_row['R:R']:,.0f} kazanırsın."
        )

# --- TAB 3: TAM TARAMA ---
with tab_full:
    display_df = table.copy()
    display_df = display_df[display_df["Skor"] >= min_score]
    if show_only_buys:
        display_df = display_df[display_df["KARAR"].isin(["GUCLU AL", "AL"])]
    if only_fresh:
        display_df = display_df[display_df["SETUP"] == "TAZE"]
    display_df = display_df.head(top_n)
    st.markdown(f"#### {len(display_df)} varlık (toplam {len(table)})")

    full_cols = ["Ticker", "Varlik", "KARAR", "SETUP", "Skor", "Yuzde",
                 "Fiyat", "Trend", "Mom_20g_%", "Pullback_%", "RSI",
                 "MACD_H", "ATR_%", "Hacim_x", "52H_Pos_%",
                 "Giris", "Stop", "Hedef", "R:R"]
    full_cols = [c for c in full_cols if c in display_df.columns]
    st.dataframe(
        style_decision_table(display_df[full_cols]),
        use_container_width=True, hide_index=True, height=620,
    )
    csv_full = display_df[full_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Tam tabloyu CSV indir", data=csv_full,
        file_name=f"tam_tarama_{scope}_{last_date}.csv", mime="text/csv",
    )

# --- TAB 4: SEKTOR ISI HARITASI ---
with tab_sector:
    sector_etfs = {
        "XLK": "Teknoloji", "XLF": "Finans", "XLE": "Enerji", "XLV": "Sağlık",
        "XLP": "Temel Tüketim", "XLY": "İsteğe Bağlı Tüketim", "XLI": "Sanayi",
        "XLU": "Kamu Hizmetleri", "XLB": "Malzeme", "XLRE": "Gayrimenkul",
        "XLC": "İletişim", "SMH": "Yarı İletken", "IGV": "Yazılım",
        "XBI": "Biyoteknoloji", "KBE": "Bankacılık", "XHB": "Konut",
        "ITA": "Savunma", "XOP": "Petrol", "XME": "Madencilik",
        "TAN": "Güneş", "GLD": "Altın", "SLV": "Gümüş",
        "TLT": "Uzun Tahvil", "BTC-USD": "Bitcoin",
    }
    sector_data = table[table["Ticker"].isin(sector_etfs.keys())].copy()
    if not sector_data.empty:
        sector_data["Sektör"] = sector_data["Ticker"].map(sector_etfs)
        sector_data = sector_data.sort_values("Skor", ascending=True)
        colors = sector_data["Skor"].apply(
            lambda s: "#10b981" if s > 0.5 else
                      "#22c55e" if s > 0.2 else
                      "#64748b" if s > -0.2 else
                      "#f59e0b" if s > -0.5 else "#ef4444"
        ).tolist()

        fig_sector = go.Figure(go.Bar(
            x=sector_data["Skor"],
            y=sector_data["Sektör"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{s:+.2f}" for s in sector_data["Skor"]],
            textposition="outside",
            customdata=sector_data[["KARAR", "SETUP", "Fiyat"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Skor: %{x:.2f}<br>"
                "Karar: %{customdata[0]}<br>"
                "Setup: %{customdata[1]}<br>"
                "Fiyat: $%{customdata[2]:.2f}<extra></extra>"
            ),
        ))
        fig_sector.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="#060912", paper_bgcolor="#060912",
            title="Sektör/Tema Cazibe Skorları",
            xaxis=dict(title="Kompozit Skor (z)", gridcolor="#1e293b", zeroline=True,
                       zerolinecolor="#475569"),
            yaxis=dict(title=None),
            showlegend=False,
        )
        st.plotly_chart(fig_sector, use_container_width=True)
        st.caption(
            "🟢 Yeşil = güçlü trend, 🟡 Sarı = nötr/zayıf, 🔴 Kırmızı = düşüş. "
            "Hangi sektörlerin liderlik yaptığını burada hızlıca görürsün; "
            "alım listendeki hisselerin hangi sektörlerden olduğu da burada belli olur."
        )
    else:
        st.info("Sektör ETF'leri taranan evrende değil. Curated kapsamı seçersen sektör görünür.")

# --- TAB 5: REHBER ---
with tab_help:
    st.markdown(
        """
### 🎓 Sistem Nasıl Okunur?

**Karar etiketleri**
- 🟢 **GÜÇLÜ AL** : Skor en üst %8 + trend tam yukarı + MACD pozitif + RSI 40-72 + çekilme < %15
- 🟢 **AL** : Skor en üst %22 + trend yukarı + RSI sağlıklı
- 🔵 **İZLE** : Potansiyel; gelişimi takip et
- ⚫ **NOTR** : Ortada, aksiyon yok
- 🔴 **KAÇIN** : Skor en alt %20

**Setup etiketleri (geç kaldım mı?)**
- 🟢 **TAZE** : Son 20-gün zirvesinden %1-8 geride → ideal pullback bölgesi
- 🟡 **ZIRVEDE** : Tam zirvede (-1% ile 0%) → pullback bekle
- 🔴 **ASIRI ALIM** : RSI > 72 → kısa vadede geri çekilme riski

**Sütun anlamları**
- **Skor** : Kompozit z-skor (trend + momentum + RSI + MACD + ATR + pullback)
- **Trend** : 4/4 üzerinden (Fiyat > MA20, > MA50, > MA200, MA50 > MA200)
- **Mom_20g_%** : Son 20 günlük getiri
- **Pullback_%** : 20-gün zirvesinden mesafe (-1 ile -8% sağlıklı)
- **RSI** : 14-gün, 45-65 sweet spot
- **MACD_H** : MACD histogram, pozitif = boğa
- **ATR_%** : Günlük ortalama % hareket
- **Hacim_x** : 5g/20g hacim oranı (>1.2 patlama)
- **52H_Pos_%** : 52-haftalık aralıkta pozisyon

**Giriş / Stop / Hedef**
- Giriş = bugünkü fiyat (pullback bekleyebilirsin)
- Stop = Giriş − 2 × ATR
- Hedef = Giriş + 3 × ATR
- R:R = 1.5 (1 birim risk için 1.5 birim kazanç)

**Pozisyon boyutlandırma**
- Tek işleme portföyünün max **%1**'ini riske at.
- Pozisyon $ = (Portföy × Risk%) / (Giriş − Stop) × Giriş
- Sidebar'daki pozisyon hesaplayıcı bunu otomatik yapar.

---

### 📈 Akademik Temeller
- **Momentum** : Jegadeesh & Titman (1993)
- **Düşük Oynaklık** : Frazzini & Pedersen (2014)
- **Trend Takip** : Moskowitz, Ooi, Pedersen (2012)
- **Risk Yönetimi** : Van Tharp position sizing

### ⚠ Uyarılar
- Komisyon/slipaj dahil değil.
- Geçmiş performans gelecek garantisi değildir.
- Sistem aday gösterir, **AL emri vermez**. Tetik kararı senindir.
"""
    )

# --------------------------------------------------------------------------- #
# FOOTER
# --------------------------------------------------------------------------- #
st.markdown(
    f'<div class="footer">Swing Trade Scanner · '
    f'taranan: {n_total} varlık · son veri: {last_date} · '
    f'Karar destek aracı, finansal danışman değildir.</div>',
    unsafe_allow_html=True,
)
