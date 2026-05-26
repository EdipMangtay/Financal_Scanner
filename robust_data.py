#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robust_data.py — "database is locked" ve canli API cokmelerini coozen
dayanikli fiyat saglayici.

SORUN COZUMLERI:
  * "database is locked": yfinance'in SQLite zaman-dilimi cache'i, paralel
    (threaded) indirmede kilitlenir. Cozum: (1) cache'i yazilabilir bir TEMP
    dizine al, (2) threads=False, (3) ticker'lari TEK TEK indir, (4) kilit
    hatasinda kisa bekleyip yeniden dene.
  * Kaynak cokmesi: yfinance -> Stooq -> disk cache sirasiyla yedekleme.
  * Bayat veri: son veri tarihini raporlar, eskiyse acikca uyarir.
"""

import io
import os
import sys
import time
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = "data_cache"
STALE_DAYS = 4


def _setup_yf_cache():
    """yfinance cache'ini yazilabilir temp dizine alir -> kilit cakismasi biter."""
    try:
        import yfinance as yf
        tz_dir = os.path.join(tempfile.gettempdir(), "yf_tz_cache")
        os.makedirs(tz_dir, exist_ok=True)
        # surum farkliliklarina karsi guvenli
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(tz_dir)
    except Exception:
        pass


class RobustFeed:
    def __init__(self, years=12, retries=3, backoff=1.5):
        self.years = years
        self.retries = retries
        self.backoff = backoff
        self.status = {}
        os.makedirs(CACHE_DIR, exist_ok=True)
        _setup_yf_cache()

    # ----------------- yfinance: tek tek, threadsiz, kilit-dayanikli -------- #
    def _yf_single(self, ticker):
        import yfinance as yf
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                df = yf.download(ticker, period=f"{self.years}y",
                                 auto_adjust=True, progress=False,
                                 threads=False)   # <-- kilit cakismasini onler
                if df is not None and not df.empty:
                    col = df["Close"]
                    if isinstance(col, pd.DataFrame):
                        col = col.iloc[:, 0]
                    return col.rename(ticker)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # "database is locked" -> bekle ve yeniden dene
                wait = self.backoff * attempt * (2 if "lock" in msg else 1)
                time.sleep(wait)
        if last_err:
            self.status.setdefault("yfinance_hatalar", []).append(f"{ticker}: {type(last_err).__name__}")
        return None

    def _from_yfinance(self, tickers):
        frames = {}
        ok = 0
        for t in tickers:
            s = self._yf_single(t)
            if s is not None and not s.dropna().empty:
                frames[t] = s
                ok += 1
        self.status["yfinance"] = f"{ok}/{len(tickers)} varlik alindi"
        return pd.DataFrame(frames) if frames else pd.DataFrame()

    # ----------------- Stooq yedek (CSV) ------------------------------------ #
    def _from_stooq(self, tickers):
        import requests
        frames = {}
        start = (datetime.today() - timedelta(days=365 * self.years)).strftime("%Y%m%d")
        end = datetime.today().strftime("%Y%m%d")
        ok = 0
        for t in tickers:
            sym = self._stooq_symbol(t)
            url = f"https://stooq.com/q/d/l/?s={sym}&d1={start}&d2={end}&i=d"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200 and r.text[:4] == "Date":
                    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
                    if not df.empty:
                        frames[t] = df.set_index("Date")["Close"].rename(t)
                        ok += 1
            except Exception:
                continue
        self.status["stooq"] = f"{ok}/{len(tickers)} varlik (yedek)"
        return pd.DataFrame(frames) if frames else pd.DataFrame()

    @staticmethod
    def _stooq_symbol(t):
        if t.endswith("-USD"):
            return t.replace("-USD", "").lower() + ".v"
        if "." in t:
            return t.lower()
        return t.lower() + ".us"

    # ----------------- cache ------------------------------------------------ #
    def _cache_path(self):
        return os.path.join(CACHE_DIR, "prices.csv")

    def _save_cache(self, df):
        try:
            df.to_csv(self._cache_path())
        except Exception:
            pass

    def _load_cache(self):
        p = self._cache_path()
        if os.path.exists(p):
            try:
                return pd.read_csv(p, index_col=0, parse_dates=True)
            except Exception:
                return None
        return None

    # ----------------- ana giris ------------------------------------------- #
    def get(self, tickers):
        result = None
        try:
            import yfinance  # noqa
            result = self._from_yfinance(tickers)
        except ImportError:
            self.status["yfinance"] = "kurulu degil"

        # eksik kalanlari Stooq ile tamamla
        missing = [t for t in tickers if result is None or t not in result.columns]
        if missing:
            stq = self._from_stooq(missing)
            if not stq.empty:
                result = stq if result is None else result.combine_first(stq)

        if result is None or result.empty:
            cache = self._load_cache()
            if cache is not None:
                self.status["cache"] = "CANLI KAYNAKLAR COKTU -> disk cache"
                return cache.ffill(), self._freshness(cache)
            raise RuntimeError("Hicbir kaynaktan veri yok ve cache bos.")

        result = result.sort_index().ffill().dropna(how="all")
        self._save_cache(result)
        return result, self._freshness(result)

    def _freshness(self, df):
        last = pd.to_datetime(df.index.max())
        try:
            age = (datetime.today() - last.to_pydatetime().replace(tzinfo=None)).days
        except Exception:
            age = 0
        return {"son_veri_tarihi": str(last.date()), "gun_yas": int(age),
                "bayat": age > STALE_DAYS}


if __name__ == "__main__":
    feed = RobustFeed(years=3)
    px, fr = feed.get(["SPY", "QQQ", "XLE"])
    print("Durum:", feed.status)
    print("Tazelik:", fr)
    print(px.tail())
