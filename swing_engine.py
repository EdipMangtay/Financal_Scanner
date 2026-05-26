#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SWING TRADE SCANNER ENGINE  -  100+ varlik, cok faktorlu, swing odakli
================================================================================
quant_engine.py uzun-vade rotasyon icindir; bu motor SWING (gunler-haftalar)
icin tasarlandi. Faktorler:

  - Trend hizalanmasi (Price > MA20 > MA50 > MA200)
  - 20-gun momentum
  - 5-gun momentum
  - Pullback kalitesi (son zirveden makul mesafe = saglikli giris)
  - RSI 14 (sweet spot 45-65)
  - MACD histogram (pozitif = boga sinyali)
  - Hacim patlamasi (5g/20g hacim orani)
  - ATR yuzdesi (dusuk = temiz risk-odul)
  - 52-haftalik pozisyon (yuksek = trendde yukari)

Karar etiketleri varlik basina:
  GUCLU AL / AL / IZLE / NOTR / KACIN
ve her AL onerisi icin:
  Giris fiyati, Stop (2*ATR), Hedef (3*ATR), R:R 1.5

Bu sistem yine "AL" emri vermez; aday + giris/stop/hedef onerir. Tetigi sen cek.
================================================================================
"""

from __future__ import annotations

import functools
import json
import os
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

UNIVERSE_CACHE_FILE = "data_cache/universe_cache.json"
UNIVERSE_CACHE_TTL = 86400  # 24 saat
os.makedirs("data_cache", exist_ok=True)

# --------------------------------------------------------------------------- #
# 100+ VARLIK EVRENI
# --------------------------------------------------------------------------- #
UNIVERSE: Dict[str, str] = {
    # === GENIS ENDEKSLER / ETF'ler ===
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones",
    "IWM": "Russell 2000", "MDY": "Mid-Cap 400", "VTI": "Total Market",

    # === ABD SEKTOR ETF'leri ===
    "XLK": "Teknoloji", "XLF": "Finans", "XLE": "Enerji", "XLV": "Saglik",
    "XLP": "Temel Tuketim", "XLY": "Iste'ge bagli Tuketim", "XLI": "Sanayi",
    "XLU": "Kamu Hizmetleri", "XLB": "Malzeme", "XLRE": "Gayrimenkul",
    "XLC": "Iletisim",

    # === ALT SEKTOR / TEMA ETF'leri ===
    "SMH": "Yari Iletken", "SOXX": "Yari Iletken (alt)", "IGV": "Yazilim",
    "XBI": "Biyoteknoloji", "IBB": "Biyotech (alt)", "KBE": "Bankacilik",
    "KRE": "Bolgesel Banka", "XHB": "Konut Insaat", "ITA": "Savunma",
    "XOP": "Petrol Arama", "XME": "Madencilik", "TAN": "Gunes Enerjisi",
    "ARKK": "Yenilik (ARK)", "JETS": "Havayolu",

    # === ULUSLARARASI ===
    "EFA": "Gelismis (US-disi)", "EEM": "Gelismekte Olan",
    "EWJ": "Japonya", "EWZ": "Brezilya", "FXI": "Cin",
    "INDA": "Hindistan", "EWG": "Almanya", "EWU": "Ingiltere",
    "MCHI": "Cin (alt)",

    # === EMTIA / GUVENLI LIMAN ===
    "GLD": "Altin", "SLV": "Gumus", "USO": "Petrol", "UNG": "Dogal Gaz",
    "DBC": "Emtia Sepeti", "DBA": "Tarim", "CPER": "Bakir",

    # === TAHVIL ===
    "TLT": "Uzun Vadeli Tahvil", "IEF": "Orta Vadeli Tahvil",
    "HYG": "Yuksek Getirili Tahvil", "LQD": "Yatirim Notu Tahvili",

    # === KRIPTO ===
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",

    # === FAKTOR ETF'leri ===
    "MTUM": "Momentum Faktoru", "QUAL": "Kalite Faktoru",
    "VLUE": "Deger Faktoru", "USMV": "Dusuk Oynaklik",

    # === MEGA-CAP TEKNOLOJI ===
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
    "GOOGL": "Alphabet (A)", "GOOG": "Alphabet (C)", "META": "Meta",
    "AMZN": "Amazon", "TSLA": "Tesla", "AVGO": "Broadcom",
    "ORCL": "Oracle", "CRM": "Salesforce", "ADBE": "Adobe",
    "AMD": "AMD", "INTC": "Intel", "QCOM": "Qualcomm",
    "TXN": "Texas Instruments", "NFLX": "Netflix", "PYPL": "PayPal",
    "UBER": "Uber", "MU": "Micron", "ASML": "ASML",

    # === FINANS ===
    "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "BLK": "BlackRock",
    "BRK-B": "Berkshire B", "V": "Visa", "MA": "Mastercard",
    "AXP": "American Express",

    # === SAGLIK ===
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "LLY": "Eli Lilly",
    "PFE": "Pfizer", "MRK": "Merck", "ABBV": "AbbVie", "TMO": "Thermo Fisher",
    "ABT": "Abbott",

    # === TUKETIM ===
    "WMT": "Walmart", "COST": "Costco", "HD": "Home Depot",
    "MCD": "McDonald's", "SBUX": "Starbucks", "NKE": "Nike",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "PG": "Procter & Gamble",
    "TGT": "Target",

    # === ENERJI / SANAYI ===
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "CAT": "Caterpillar", "BA": "Boeing", "GE": "GE Aerospace",
    "DE": "Deere", "LMT": "Lockheed Martin", "RTX": "RTX Corp",
    "HON": "Honeywell",

    # === ILETISIM / MEDYA ===
    "DIS": "Disney", "T": "AT&T", "VZ": "Verizon", "CMCSA": "Comcast",

    # === BUYUME / SPEKULATIF ===
    "XYZ": "Block (eski SQ)", "SHOP": "Shopify", "PLTR": "Palantir",
    "COIN": "Coinbase", "RBLX": "Roblox", "SNOW": "Snowflake",
    "DDOG": "Datadog", "CRWD": "CrowdStrike", "NET": "Cloudflare",
}


# --------------------------------------------------------------------------- #
# EVREN GENISLETME — Wikipedia'dan S&P 500 / 400 / 600 / Nasdaq-100 ceker
# --------------------------------------------------------------------------- #
SCOPE_LABELS = {
    "curated": "Secili 130 varlik (hizli, ETF + sektor + mega-cap)",
    "sp500":   "S&P 500 (~500 ABD hisse + tum ETF/emtia/kripto)",
    "sp1500":  "S&P 1500 (kucuk + orta + buyuk cap, ~1500 hisse + ETF)",
    "nasdaq":  "Nasdaq-100 + Mevcut ETF/sektor (~230 varlik)",
}


def _load_universe_cache() -> dict:
    if not os.path.exists(UNIVERSE_CACHE_FILE):
        return {}
    try:
        if time.time() - os.path.getmtime(UNIVERSE_CACHE_FILE) > UNIVERSE_CACHE_TTL:
            return {}
        with open(UNIVERSE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_universe_cache(data: dict) -> None:
    try:
        with open(UNIVERSE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _fetch_wiki_index(url: str, min_size: int = 50) -> Dict[str, str]:
    """Wikipedia tablosundan ticker + sirket adi ceker.
    Yahoo formatina cevirir (BRK.B -> BRK-B).
    NOT: Wikipedia default urllib UA'sini engellediginden requests ile ceriyoruz."""
    try:
        import requests
        from io import StringIO
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return {}
        tables = pd.read_html(StringIO(r.text))
    except Exception:
        return {}

    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        sym_col = None
        for c in cols:
            if c.lower() in ("symbol", "ticker", "ticker symbol", "code"):
                sym_col = c
                break
        if sym_col is None:
            continue
        name_col = None
        for c in cols:
            cl = c.lower()
            if any(k in cl for k in ("security", "company", "constituent", "name", "issuer")):
                name_col = c
                break
        if name_col is None:
            name_col = sym_col

        result: Dict[str, str] = {}
        for _, row in tbl.iterrows():
            sym = str(row[sym_col]).strip().replace(".", "-")
            name = str(row[name_col]).strip()
            if sym and sym.lower() not in ("nan", "none"):
                result[sym] = name if name and name.lower() != "nan" else sym
        if len(result) >= min_size:
            return result
    return {}


@functools.lru_cache(maxsize=4)
def get_sp500_dict() -> Dict[str, str]:
    return _fetch_wiki_index(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", min_size=400
    )


@functools.lru_cache(maxsize=4)
def get_sp400_dict() -> Dict[str, str]:
    return _fetch_wiki_index(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", min_size=300
    )


@functools.lru_cache(maxsize=4)
def get_sp600_dict() -> Dict[str, str]:
    return _fetch_wiki_index(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", min_size=400
    )


@functools.lru_cache(maxsize=4)
def get_nasdaq100_dict() -> Dict[str, str]:
    return _fetch_wiki_index("https://en.wikipedia.org/wiki/Nasdaq-100", min_size=80)


def get_extended_universe(scope: str = "curated") -> Dict[str, str]:
    """
    scope:
      'curated' -> mevcut 130 ticker (en hizli)
      'sp500'   -> S&P 500 + ETF/sektor/emtia/kripto (~530)
      'sp1500'  -> S&P 1500 + ETF (~1530)
      'nasdaq'  -> Nasdaq-100 + ETF (~230)

    24 saatlik disk cache vardir, ilk indirme sonrasi anlik.
    """
    if scope == "curated":
        return dict(UNIVERSE)

    cache = _load_universe_cache()
    if scope in cache and len(cache[scope]) > 50:
        return cache[scope]

    base = dict(UNIVERSE)

    if scope == "nasdaq":
        base.update(get_nasdaq100_dict())
    elif scope == "sp500":
        base.update(get_sp500_dict())
    elif scope == "sp1500":
        base.update(get_sp500_dict())
        base.update(get_sp400_dict())
        base.update(get_sp600_dict())
    else:
        return dict(UNIVERSE)

    cache[scope] = base
    _save_universe_cache(cache)
    return base


# --------------------------------------------------------------------------- #
# VERI CEKME
# --------------------------------------------------------------------------- #
def fetch_data_chunked(
    tickers,
    period: str = "1y",
    chunk_size: int = 80,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
):
    """Buyuk evrenler icin parcali yfinance indirme.
    progress_callback(done, total, msg) UI ilerleme cubugu icin."""
    import yfinance as yf

    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    n = len(chunks)
    close_dfs, vol_dfs = [], []
    failed_chunks = 0

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, n, f"Indiriliyor: {i*chunk_size+1}-{min((i+1)*chunk_size, len(tickers))} / {len(tickers)}")
        try:
            data = yf.download(
                chunk, period=period, auto_adjust=True,
                progress=False, threads=True,
            )
            if data is None or data.empty:
                failed_chunks += 1
                continue
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"] if "Close" in data.columns.get_level_values(0) else pd.DataFrame()
                vol = data["Volume"] if "Volume" in data.columns.get_level_values(0) else pd.DataFrame()
            else:
                close = data[["Close"]].copy()
                close.columns = chunk[:1]
                vol = data[["Volume"]].copy() if "Volume" in data.columns else pd.DataFrame()
                if not vol.empty:
                    vol.columns = chunk[:1]
            if not close.empty:
                close_dfs.append(close)
            if not vol.empty:
                vol_dfs.append(vol)
        except Exception:
            failed_chunks += 1
            continue

    if progress_callback:
        progress_callback(n, n, "Birlestiriliyor...")

    if not close_dfs:
        return pd.DataFrame(), pd.DataFrame(), "Hicbir batch'ten veri alinamadi"

    all_close = pd.concat(close_dfs, axis=1).dropna(how="all").ffill()
    all_vol = pd.concat(vol_dfs, axis=1).dropna(how="all") if vol_dfs else None
    all_close = all_close.loc[:, ~all_close.columns.duplicated()]
    if all_vol is not None:
        all_vol = all_vol.loc[:, ~all_vol.columns.duplicated()]

    status = f"OK ({all_close.shape[1]}/{len(tickers)} varlik, {n} batch"
    if failed_chunks:
        status += f", {failed_chunks} batch basarisiz"
    status += ")"
    return all_close, all_vol, status


def fetch_data(tickers, period: str = "2y", retries: int = 2):
    """yfinance ile fiyat + hacim ceker. Hata olursa retry.
    Doner: (close_df, volume_df, status_str)
    """
    import yfinance as yf

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers, period=period, auto_adjust=True,
                progress=False, threads=True, group_by="column",
            )
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
                volume = data["Volume"] if "Volume" in data.columns.levels[0] else None
            else:
                close = data[["Close"]]
                close.columns = [tickers[0] if isinstance(tickers, list) else tickers]
                volume = data[["Volume"]] if "Volume" in data.columns else None
                if volume is not None:
                    volume.columns = close.columns
            close = close.dropna(how="all").ffill()
            return close, volume, f"OK (deneme {attempt}, {close.shape[1]} varlik)"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * attempt)
    return pd.DataFrame(), pd.DataFrame(), f"HATA: {last_err}"


# --------------------------------------------------------------------------- #
# YARDIMCI FAKTOR HESAPLAMALARI
# --------------------------------------------------------------------------- #
def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _macd_hist(series: pd.Series) -> float:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = (macd_line - signal).iloc[-1]
    return float(hist) if pd.notna(hist) else 0.0


def _atr_pct(series: pd.Series, period: int = 14) -> float:
    """Close-bazli yaklasik ATR yuzdesi (high/low yok varsayim)."""
    rets = series.pct_change().abs()
    atr = rets.rolling(period).mean().iloc[-1]
    return float(atr * 100) if pd.notna(atr) else 0.0


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return s * 0
    return (s - s.mean()) / sd


def _rsi_score(rsi: float) -> float:
    """RSI tatli nokta: 45-65. Asiri alim/asiri satim cezalandirilir."""
    if 50 <= rsi <= 65:
        return 1.0
    if 45 <= rsi < 50:
        return 0.7
    if 65 < rsi <= 72:
        return 0.4
    if 40 <= rsi < 45:
        return 0.4
    if rsi > 75:
        return -0.7
    if rsi < 30:
        return 0.1
    return 0.0


def _pullback_score(pb_pct: float) -> float:
    """Saglikli pullback: -1% ile -8% arasi en iyi.
    Cok yukseginde (zirvede) zayif; cok cukurda (kapana) tehlikeli."""
    if -8 <= pb_pct <= -1:
        return 1.0
    if -12 <= pb_pct < -8:
        return 0.4
    if -1 < pb_pct <= 0:
        return 0.5
    if pb_pct < -12:
        return -0.4
    return 0.0


# --------------------------------------------------------------------------- #
# FAKTOR PANELI
# --------------------------------------------------------------------------- #
def compute_features(close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
    """Her varlik icin swing faktorlerini hesaplar."""
    feats = {}
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < 60:
            continue
        try:
            price = float(s.iloc[-1])
            ma20 = float(s.rolling(20).mean().iloc[-1])
            ma50 = float(s.rolling(50).mean().iloc[-1])
            ma200 = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else ma50

            trend_score = 0
            if price > ma20: trend_score += 1
            if price > ma50: trend_score += 1
            if price > ma200: trend_score += 1
            if ma50 > ma200: trend_score += 1

            mom_5 = (price / float(s.iloc[-5]) - 1) * 100 if len(s) >= 5 else 0.0
            mom_20 = (price / float(s.iloc[-20]) - 1) * 100 if len(s) >= 20 else 0.0
            mom_60 = (price / float(s.iloc[-60]) - 1) * 100 if len(s) >= 60 else 0.0

            high_20 = float(s.iloc[-20:].max())
            pullback_20 = (price / high_20 - 1) * 100

            rsi = _rsi(s)
            macd_h = _macd_hist(s)
            atr_p = _atr_pct(s)

            low_252 = float(s.iloc[-252:].min()) if len(s) >= 252 else float(s.min())
            high_252 = float(s.iloc[-252:].max()) if len(s) >= 252 else float(s.max())
            pos_52w = (price - low_252) / (high_252 - low_252 + 1e-9) * 100

            vol_surge = 1.0
            if volume is not None and t in volume.columns:
                v = volume[t].dropna()
                if len(v) >= 20:
                    v5 = float(v.iloc[-5:].mean())
                    v20 = float(v.iloc[-20:].mean())
                    if v20 > 0:
                        vol_surge = v5 / v20

            feats[t] = {
                "price": price,
                "ma20": ma20, "ma50": ma50, "ma200": ma200,
                "trend_score": trend_score,
                "mom_5": mom_5, "mom_20": mom_20, "mom_60": mom_60,
                "pullback_20": pullback_20,
                "rsi": rsi, "macd_hist": macd_h,
                "atr_pct": atr_p,
                "pos_52w": pos_52w,
                "vol_surge": vol_surge,
            }
        except Exception:
            continue
    return pd.DataFrame(feats).T


# --------------------------------------------------------------------------- #
# REJIM TESPITI (basit, sadece SPY uzerinden)
# --------------------------------------------------------------------------- #
def detect_market_regime(close: pd.DataFrame) -> str:
    """SPY varsa onun trendine bakar, yoksa equal-weight."""
    proxy = close["SPY"] if "SPY" in close.columns else close.mean(axis=1)
    proxy = proxy.dropna()
    if len(proxy) < 200:
        return "neutral"
    price = float(proxy.iloc[-1])
    ma50 = float(proxy.rolling(50).mean().iloc[-1])
    ma200 = float(proxy.rolling(200).mean().iloc[-1])
    vol = proxy.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    vol_hist = proxy.pct_change().rolling(21).std() * np.sqrt(252) * 100
    vol_pct = float((vol_hist <= vol).mean())

    if price > ma50 > ma200 and vol_pct < 0.7:
        return "risk_on"
    if price < ma200 or vol_pct > 0.85:
        return "risk_off"
    return "neutral"


# --------------------------------------------------------------------------- #
# COMPOSITE SKORLAMA
# --------------------------------------------------------------------------- #
SWING_WEIGHTS = {
    "risk_on":  {"trend": 0.20, "mom_20": 0.20, "mom_5": 0.10,
                 "pullback": 0.10, "rsi": 0.15, "macd": 0.10,
                 "lowvol": 0.05, "vol_surge": 0.05, "pos_52w": 0.05},
    "risk_off": {"trend": 0.30, "mom_20": 0.10, "mom_5": 0.05,
                 "pullback": 0.10, "rsi": 0.15, "macd": 0.10,
                 "lowvol": 0.15, "vol_surge": 0.00, "pos_52w": 0.05},
    "neutral":  {"trend": 0.25, "mom_20": 0.15, "mom_5": 0.10,
                 "pullback": 0.10, "rsi": 0.15, "macd": 0.10,
                 "lowvol": 0.10, "vol_surge": 0.05, "pos_52w": 0.00},
}


def composite_swing_score(features: pd.DataFrame, regime: str = "neutral") -> pd.Series:
    w = SWING_WEIGHTS.get(regime, SWING_WEIGHTS["neutral"])
    z = pd.DataFrame(index=features.index)
    z["trend"] = (features["trend_score"] - 2) / 1.5
    z["mom_20"] = _zscore(features["mom_20"])
    z["mom_5"] = _zscore(features["mom_5"])
    z["pullback"] = features["pullback_20"].apply(_pullback_score)
    z["rsi"] = features["rsi"].apply(_rsi_score)
    z["macd"] = _zscore(features["macd_hist"]).clip(-2, 2)
    z["lowvol"] = -_zscore(features["atr_pct"])
    z["vol_surge"] = _zscore(features["vol_surge"]).clip(-2, 2)
    z["pos_52w"] = (features["pos_52w"] - 50) / 25
    score = sum(w[k] * z[k] for k in w)
    return score.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# KARAR + GIRIS/STOP/HEDEF
# --------------------------------------------------------------------------- #
def decide_actions(
    features: pd.DataFrame,
    scores: pd.Series,
    universe_dict: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Her varlik icin etiket + giris/stop/hedef onerisi."""
    if universe_dict is None:
        universe_dict = UNIVERSE
    pct = scores.rank(pct=True)
    rows = []
    for t, sc in scores.items():
        f = features.loc[t]
        p = pct.loc[t]
        price = f["price"]
        atr_dollar = price * (f["atr_pct"] / 100)
        if not np.isfinite(atr_dollar) or atr_dollar <= 0:
            atr_dollar = price * 0.02

        # karar mantigi
        bullish = (price > f["ma50"]) and (f["ma50"] > f["ma200"])
        macd_ok = f["macd_hist"] > 0
        rsi_ok = 40 <= f["rsi"] <= 72
        not_blown = f["pullback_20"] > -15
        fresh_pullback = -8 <= f["pullback_20"] <= -1   # taze pullback bolgesi

        if p >= 0.92 and bullish and macd_ok and rsi_ok and not_blown:
            label = "GUCLU AL"
        elif p >= 0.78 and bullish and rsi_ok and not_blown:
            label = "AL"
        elif p >= 0.55 and price > f["ma50"]:
            label = "IZLE"
        elif p <= 0.20:
            label = "KACIN"
        else:
            label = "NOTR"

        entry = price
        stop = price - 2 * atr_dollar
        target = price + 3 * atr_dollar
        rr = (target - entry) / max(entry - stop, 1e-9)

        # Taze pullback isareti — "geç kaldım mı?" sorusunu yanitlar
        setup = "TAZE" if fresh_pullback and label in ("GUCLU AL", "AL") else ""
        if not setup and label in ("GUCLU AL", "AL"):
            if f["pullback_20"] > -1:
                setup = "ZIRVEDE"
            elif f["rsi"] > 72:
                setup = "ASIRI ALIM"

        rows.append({
            "Ticker": t,
            "Varlik": universe_dict.get(t, t),
            "Skor": round(float(sc), 2),
            "Yuzde": round(float(p) * 100),
            "Fiyat": round(price, 2),
            "Trend": f"{int(f['trend_score'])}/4",
            "Mom_20g_%": round(float(f["mom_20"]), 1),
            "Pullback_%": round(float(f["pullback_20"]), 1),
            "RSI": round(float(f["rsi"]), 1),
            "MACD_H": round(float(f["macd_hist"]), 3),
            "ATR_%": round(float(f["atr_pct"]), 2),
            "Hacim_x": round(float(f["vol_surge"]), 2),
            "52H_Pos_%": round(float(f["pos_52w"])),
            "KARAR": label,
            "SETUP": setup,
            "Giris": round(entry, 2),
            "Stop": round(stop, 2),
            "Hedef": round(target, 2),
            "R:R": round(float(rr), 2),
        })
    df = pd.DataFrame(rows).sort_values("Skor", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# TEK FONKSIYONLA TAM TARAMA
# --------------------------------------------------------------------------- #
def run_scan(
    tickers=None,
    scope: str = "curated",
    period: str = "1y",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
):
    """Tum boru hattini calistir, ozet doner.

    scope: 'curated' | 'sp500' | 'sp1500' | 'nasdaq'  (tickers verilmediyse)
    """
    if tickers is None:
        universe_dict = get_extended_universe(scope)
        tickers = list(universe_dict.keys())
    else:
        universe_dict = {t: UNIVERSE.get(t, t) for t in tickers}

    if len(tickers) > 120:
        close, volume, status = fetch_data_chunked(
            tickers, period=period, progress_callback=progress_callback,
        )
    else:
        close, volume, status = fetch_data(tickers, period=period)

    if close.empty:
        return {"ok": False, "status": status}
    features = compute_features(close, volume)
    if features.empty:
        return {"ok": False, "status": "Hicbir varlik icin yeterli veri yok."}
    regime = detect_market_regime(close)
    scores = composite_swing_score(features, regime)
    table = decide_actions(features, scores, universe_dict=universe_dict)
    return {
        "ok": True,
        "status": status,
        "regime": regime,
        "table": table,
        "close": close,
        "features": features,
        "last_date": close.index.max(),
        "universe_size": len(tickers),
        "scope": scope,
    }


if __name__ == "__main__":
    import sys
    scope = sys.argv[1] if len(sys.argv) > 1 else "curated"
    print(f"Kapsam: {scope}  ({SCOPE_LABELS.get(scope, '?')})")

    def cli_progress(done, total, msg):
        pct = int(done * 100 / max(total, 1))
        print(f"  [{pct:3d}%] {msg}", flush=True)

    res = run_scan(scope=scope, progress_callback=cli_progress)
    if not res["ok"]:
        print(res["status"])
    else:
        print(f"\nRejim: {res['regime']}  |  Son veri: {res['last_date']}  |  {res['status']}")
        buys = res["table"][res["table"]["KARAR"].isin(["GUCLU AL", "AL"])]
        print(f"\nAlim listesi ({len(buys)} aday, en yuksek 20):")
        cols = ["Ticker", "Varlik", "Skor", "KARAR", "SETUP", "Giris", "Stop", "Hedef", "R:R"]
        print(buys[cols].head(20).to_string(index=False))
