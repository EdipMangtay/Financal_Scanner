#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DAILY SECTOR SCANNER  —  canli, dayanikli, gunluk karar listesi
================================================================================
quant_engine.py uzerine kurulu PRODUKSIYON katmani. Her calistirildiginda:
  1) Genis sektor/varlik evrenini CANLI API'lerden ceker (dayanikli),
  2) Faktor + rejim + HRP motorunu calistirir,
  3) Sana net bir KARAR LISTESI + KISA LISTE (HTF icin) cikarir,
  4) Sonucu tarihli olarak diske loglar.

>>> IS BOLUMU (onemli):
    Bu sistem "AL" emri VERMEZ. Sektoru tarar, en saglikli adaylari kisa
    listeye koyar ve "HTF onayi bekliyor" etiketi verir. SEN o adaylara
    kendi HTF (ust zaman dilimi) analizini uygular, tetigi sen cekersin.
    Sistem = filtre/kisa-liste. Karar + zamanlama = sen.

>>> CANLI VERI GERCEGI (durust ol):
    Ucretsiz API'ler "sorunsuz" calismaz; bos doner, rate-limit yer. O yuzden
    GARANTI degil DAYANIKLILIK kuruldu:
      - Birincil kaynak: yfinance (Yahoo)
      - Yedek kaynak   : Stooq (dogrudan CSV)
      - Son care       : disk cache (bayat veri UYARISIYLA)
    Her calismada veri tazeligi raporlanir; bayatsa acikca soyler.

KULLANIM:
  python daily_scanner.py --demo        # internetsiz boru hatti testi
  python daily_scanner.py               # canli veri (yerelde calistir)
  python daily_scanner.py --top 10      # kisa liste boyutu
================================================================================
"""

import argparse
import io
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import quant_engine as qe   # ayni klasorde olmali

CACHE_DIR = "data_cache"
LOG_DIR = "scan_logs"
STALE_DAYS = 4   # veri bu kadar gunden eskiyse "bayat" say


# --------------------------------------------------------------------------- #
# GENIS EVREN — "tum sektoru tara". Buraya istedigin kadar ticker ekle.
# Tek tek hisse de ekleyebilirsin (ornek: "AAPL", "ASELS.IS"). BIST icin .IS
# --------------------------------------------------------------------------- #
UNIVERSE = {
    # Genis endeksler / bolgeler
    "SPY": "ABD S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones",
    "IWM": "ABD kucuk olcek", "EFA": "Gelismis (ABD disi)",
    "EEM": "Gelismekte olan", "EWZ": "Brezilya", "EWJ": "Japonya",
    # ABD ana sektorler
    "XLK": "Teknoloji", "XLF": "Finans", "XLE": "Enerji", "XLV": "Saglik",
    "XLP": "Temel tuketim", "XLY": "Iste'ge bagli tuketim", "XLI": "Sanayi",
    "XLU": "Kamu hizmetleri", "XLB": "Malzeme", "XLRE": "Gayrimenkul",
    "XLC": "Iletisim",
    # Alt-sektorler / temalar (derinlik)
    "SMH": "Yari iletken", "IGV": "Yazilim", "XBI": "Biyoteknoloji",
    "KBE": "Bankacilik", "XHB": "Konut insaat", "ITA": "Savunma/havacilik",
    "XOP": "Petrol arama", "XME": "Madencilik", "TAN": "Gunes enerjisi",
    # Faktor ETF'leri (akademik faktorlerin canli temsilcisi)
    "MTUM": "Momentum faktoru", "QUAL": "Kalite faktoru",
    "VLUE": "Deger faktoru", "USMV": "Dusuk oynaklik faktoru",
    # Emtia / tahvil / kripto
    "GLD": "Altin", "SLV": "Gumus", "USO": "Petrol", "DBC": "Emtia sepeti",
    "TLT": "Uzun vadeli tahvil", "IEF": "Orta vadeli tahvil",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
}


# --------------------------------------------------------------------------- #
# DAYANIKLI VERI KATMANI
# --------------------------------------------------------------------------- #
class DataFeed:
    """Cok kaynakli, retry'li, cache'li fiyat saglayici."""

    def __init__(self, years=12, retries=3, backoff=2.0):
        self.years = years
        self.retries = retries
        self.backoff = backoff
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.status = {}   # kaynak -> durum raporu

    # ---- birincil: yfinance ----
    def _from_yfinance(self, tickers):
        import yfinance as yf
        data = yf.download(tickers, period=f"{self.years}y",
                           auto_adjust=True, progress=False, threads=True)
        px = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        return px.dropna(how="all")

    # ---- yedek: Stooq dogrudan CSV (ticker basina) ----
    def _from_stooq(self, tickers):
        import requests
        frames = {}
        start = (datetime.today() - timedelta(days=365 * self.years)).strftime("%Y%m%d")
        end = datetime.today().strftime("%Y%m%d")
        for t in tickers:
            sym = self._stooq_symbol(t)
            url = (f"https://stooq.com/q/d/l/?s={sym}&d1={start}&d2={end}&i=d")
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200 and "Date" in r.text[:50]:
                    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
                    if not df.empty:
                        frames[t] = df.set_index("Date")["Close"]
            except Exception:
                continue
        return pd.DataFrame(frames) if frames else pd.DataFrame()

    @staticmethod
    def _stooq_symbol(t):
        # Stooq ABD hisseleri ".us" ister; kripto/ozel durumlar farkli
        if t.endswith("-USD"):
            return t.replace("-USD", "").lower() + ".v"   # kripto Stooq'ta sinirli
        if "." in t:
            return t.lower()
        return t.lower() + ".us"

    def _retry(self, fn, name, tickers):
        for attempt in range(1, self.retries + 1):
            try:
                df = fn(tickers)
                if df is not None and not df.empty and df.shape[1] > 0:
                    self.status[name] = f"OK (deneme {attempt}, {df.shape[1]} varlik)"
                    return df
                self.status[name] = f"bos dondu (deneme {attempt})"
            except Exception as e:
                self.status[name] = f"hata: {type(e).__name__} (deneme {attempt})"
            time.sleep(self.backoff * attempt)
        return None

    def _cache_path(self):
        return os.path.join(CACHE_DIR, "prices.csv")

    def _save_cache(self, df):
        df.to_csv(self._cache_path())

    def _load_cache(self):
        p = self._cache_path()
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0, parse_dates=True)
        return None

    def get(self, tickers):
        """Sirayla dener: yfinance -> stooq -> cache. Birlestirip doner."""
        result = None
        try:
            import yfinance  # noqa
            result = self._retry(self._from_yfinance, "yfinance", tickers)
        except ImportError:
            self.status["yfinance"] = "kurulu degil"

        if result is None or result.shape[1] < len(tickers) * 0.6:
            self.status["yfinance"] = self.status.get("yfinance", "") + " -> Stooq yedege geciliyor"
            stq = self._retry(self._from_stooq, "stooq", tickers)
            if stq is not None:
                result = stq if result is None else result.combine_first(stq)

        if result is None or result.empty:
            cache = self._load_cache()
            if cache is not None:
                self.status["cache"] = "CANLI KAYNAKLAR COKTU -> diskteki veri kullaniliyor"
                return cache.ffill(), self._freshness(cache)
            raise RuntimeError("Hicbir kaynaktan veri alinamadi ve cache yok.")

        result = result.ffill().dropna(how="all")
        self._save_cache(result)
        return result, self._freshness(result)

    def _freshness(self, df):
        last = pd.to_datetime(df.index.max())
        age = (datetime.today() - last.to_pydatetime().replace(tzinfo=None)).days
        return {"son_veri_tarihi": last.strftime("%Y-%m-%d"),
                "gun_yas": age, "bayat": age > STALE_DAYS}


# --------------------------------------------------------------------------- #
# KARAR KATMANI — skoru aksiyon alinabilir etikete cevir
# --------------------------------------------------------------------------- #
def decide(px, scores, regime):
    """Her varlik icin net karar etiketi uretir. Kisa listeyi (HTF icin) isaretler."""
    rows = []
    for t, sc in scores.items():
        s = px[t].dropna()
        if len(s) < 200:
            continue
        ma200 = s.iloc[-200:].mean()
        above_trend = s.iloc[-1] > ma200
        mom = s.iloc[-22] / s.iloc[-252] - 1 if len(s) > 252 else 0.0
        ddown = s.iloc[-1] / s.iloc[-252:].max() - 1
        pct = (scores.rank(pct=True))[t]   # skorun kesitsel yuzdeligi

        # Karar mantigi: yuksek skor + trend uzeri + bicagi yakalamiyor
        if pct >= 0.80 and above_trend and mom > -0.05:
            label, action = "GUCLU ADAY", "HTF onayi bekliyor"
        elif pct >= 0.80 and not above_trend:
            label, action = "UCUZ ama DUSUSTE", "izle, acele etme"
        elif pct >= 0.60:
            label, action = "IZLE", "potansiyel, gelisimi takip et"
        elif pct <= 0.25:
            label, action = "KACIN", "gorece pahali/zayif"
        else:
            label, action = "NOTR", "-"

        rows.append({
            "Ticker": t,
            "Varlik": UNIVERSE.get(t, t),
            "Skor": round(float(sc), 2),
            "Yuzdelik": round(float(pct) * 100),
            "Trend": "Uzeri" if above_trend else "Alti",
            "Mom_12_1_%": round(float(mom) * 100, 1),
            "Zirveden_%": round(float(ddown) * 100, 1),
            "KARAR": label,
            "AKSIYON": action,
        })
    df = pd.DataFrame(rows).sort_values("Skor", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# RAPOR + LOG
# --------------------------------------------------------------------------- #
def run(demo=False, top=8, years=12):
    tickers = list(UNIVERSE.keys())
    today = datetime.today().strftime("%Y-%m-%d")
    print("=" * 78)
    print(f"  DAILY SECTOR SCANNER  —  {today}")
    print("=" * 78)

    if demo:
        print("  [DEMO] Sentetik veri (canli API test edilmiyor).\n")
        px = qe.make_demo_prices(tickers, years=years)
        freshness = {"son_veri_tarihi": "DEMO", "gun_yas": 0, "bayat": False}
        feed_status = {"demo": "sentetik veri"}
    else:
        feed = DataFeed(years=years)
        print("  Canli veri cekiliyor (yfinance -> Stooq -> cache)...\n")
        px, freshness = feed.get(tickers)
        feed_status = feed.status

    # --- veri tazeligi raporu ---
    print(">>> VERI TAZELIGI")
    for k, v in feed_status.items():
        print(f"    {k:10s}: {v}")
    flag = "  !!! BAYAT VERI - DIKKAT" if freshness["bayat"] else "  (taze)"
    print(f"    son veri  : {freshness['son_veri_tarihi']} "
          f"({freshness['gun_yas']} gun){flag}\n")

    # --- motor: rejim + skor ---
    regime = qe.detect_regime(px)
    panel = qe.compute_factor_panel(px)
    scores = qe.composite_score(panel, regime)
    print(f">>> PIYASA REJIMI: {regime.upper()}  (faktor egilimi buna gore)\n")

    # --- karar tablosu ---
    table = decide(px, scores, regime)
    pd.set_option("display.max_rows", None, "display.width", 220)
    print(">>> TAM TARAMA — KARAR TABLOSU\n")
    print(table.to_string(index=False))

    # --- kisa liste (HTF icin) + HRP agirliklari ---
    short = table[table["KARAR"] == "GUCLU ADAY"]["Ticker"].tolist()[:top]
    print("\n" + "-" * 78)
    if short:
        print(f">>> KISA LISTE — bunlara HTF analizi uygula ({len(short)} aday)\n")
        rets = px[short].pct_change().iloc[-252:]
        w = qe.hrp_weights(rets).sort_values(ascending=False)
        for t in short:
            print(f"    • {t:8s} {UNIVERSE.get(t, t):24s}  "
                  f"onerilen risk-agirlik: %{w.get(t, 0)*100:4.1f}")
        print("\n    >> Sistem bu adaylari saglikli buldu. Tetigi HTF onayinla SEN cek.")
    else:
        print(">>> KISA LISTE BOS")
        print("    Su an 'guclu aday' kriterini gecen yok. Bu da bir bilgidir:")
        print("    piyasa pahali/zayif olabilir, beklemek de bir karardir.")

    # --- loglama ---
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, f"scan_{today}.csv")
    table.to_csv(csv_path, index=False)
    json_path = os.path.join(LOG_DIR, f"scan_{today}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "tarih": today, "rejim": regime,
            "veri_tazeligi": freshness, "kisa_liste": short,
            "kaynak_durumu": feed_status,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  -> Loglandi: {csv_path}")
    print(f"  -> Loglandi: {json_path}")

    print("\n" + "=" * 78)
    print("  UYARI: Bu bir KISA-LISTE uretir, AL emri vermez. HTF analizi ve")
    print("  final karar SENIN. Finansal danisman degilim.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--years", type=int, default=12)
    args = ap.parse_args()
    run(demo=args.demo, top=args.top, years=args.years)


if __name__ == "__main__":
    main()
