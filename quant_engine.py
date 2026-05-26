#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MULTI-FACTOR · REGIME-AWARE · HRP ALLOCATION ENGINE
Akademik literatura dayali, overfitting'e karsi titiz dogrulamali deger motoru
================================================================================

TASARIM FELSEFESI (en onemli kisim):
  Finansta GUC = karmasiklik DEGIL, SAGLAMLIK + acimasiz dogrulama.
  Devasa kara kutular backtest'te parlar, canlida coker (Bailey & Lopez de
  Prado 2014, "The Probability of Backtest Overfitting"). Bu motor, gucunu
  ogrenilmis agirliklardan degil, on yillarca test edilmis faktorlerden ve
  ileri-yonlu (out-of-sample) dogrulamadan alir.

DORT SUTUN ve AKADEMIK DAYANAKLARI:

  1) COK-FAKTORLU SKORLAMA (kesitsel, cross-sectional)
     - Momentum (12-1 ay): Jegadeesh & Titman (1993); Asness, Moskowitz,
       Pedersen (2013) "Value and Momentum Everywhere".
     - Uzun-vade Deger/Reversal: De Bondt & Thaler (1985).
     - Dusuk-Oynaklik anomalisi: Frazzini & Pedersen (2014) "Betting Against
       Beta"; Baker, Bradley, Wurgler (2011).
     - Zaman-serisi Momentum (trend): Moskowitz, Ooi, Pedersen (2012).
     - Kalite/Trend filtresi: 200-gun trend rejimi.

  2) REJIM TESPITI (Gaussian Mixture / rejim kumeleme)
     - Piyasayi risk-on / risk-off durumlarina ayirir; faktor egilimini
       rejime gore (OLCULU sekilde) kaydirir. scikit-learn GaussianMixture
       kullanilir (HAZIR WHEEL, C++ derleyici GEREKMEZ). Rejim modellemesi
       fikir kokeni: Hamilton (1989). HMM yerine GMM secimi, kurulum
       sorunlarini (hmmlearn derleme) tamamen ortadan kaldirir.

  3) PORTFOY KURULUMU — Hierarchical Risk Parity (HRP)
     - Lopez de Prado (2016) "Building Diversified Portfolios that Outperform
       Out-of-Sample". Gurultulu kovaryans matrisini TERSINE CEVIRMEDEN
       (Markowitz'in en zayif noktasi) hiyerarsik risk-dengeli dagitim yapar.

  4) DOGRULAMA — overfitting'i YAKALAYAN, gizlemeyen
     - Purged & Embargoed Walk-Forward: Lopez de Prado (2018) "Advances in
       Financial Machine Learning". Lookahead bias ve serisel sizinti engellenir.
     - Deflated Sharpe Ratio (DSR): Bailey & Lopez de Prado (2014). Coklu
       deneme sayisini hesaba katarak Sharpe'in "sans eseri" olma olasiligini
       duser.
     - Probability of Backtest Overfitting (PBO): Bailey et al. (2017),
       CSCV yontemi. Stratejinin in-sample sirasinin out-of-sample'da
       korunup korunmadigini olcer.

NE YAPMAZ:
  - "AL" garantisi vermez. Olasilik ve risk-dengeli AGIRLIK onerir.
  - Gelecegi bilmez. Sadece bugunku konumu + tarihsel kenarin GERCEK olup
    olmadigini olcer.

KULLANIM:
  python quant_engine.py --demo                 # internetsiz sentetik test
  python quant_engine.py                         # gercek veri (yerelde)
  python quant_engine.py --validate              # DSR + PBO overfitting testi
  python quant_engine.py --backtest              # purged walk-forward backtest
================================================================================
"""

import argparse
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import norm

warnings.filterwarnings("ignore")

TRADING_DAYS = 252

# --------------------------------------------------------------------------- #
# EVREN
# --------------------------------------------------------------------------- #
UNIVERSE = {
    "SPY": "ABD S&P 500", "QQQ": "Nasdaq 100", "EFA": "Gelismis (ABD disi)",
    "EEM": "Gelismekte olan piyasalar",
    "XLK": "Teknoloji", "XLF": "Finans", "XLE": "Enerji", "XLV": "Saglik",
    "XLP": "Temel tuketim", "XLY": "Iste'ge bagli tuketim", "XLI": "Sanayi",
    "XLU": "Kamu hizmetleri", "XLB": "Malzeme", "XLRE": "Gayrimenkul",
    "GLD": "Altin", "SLV": "Gumus", "TLT": "Uzun vadeli tahvil",
    "BTC-USD": "Bitcoin",
}

# Faktor agirliklari — REJIME GORE. Backtest'e gore SECME (data snooping);
# bunlar literaturden gelen makul on-degerlerdir, sen kendi gorusunle ayarla.
FACTOR_WEIGHTS = {
    "risk_on":  {"mom": 0.40, "ts_mom": 0.25, "value": 0.10, "lowvol": 0.10, "trend": 0.15},
    "risk_off": {"mom": 0.15, "ts_mom": 0.15, "value": 0.25, "lowvol": 0.35, "trend": 0.10},
    "neutral":  {"mom": 0.30, "ts_mom": 0.20, "value": 0.20, "lowvol": 0.20, "trend": 0.10},
}


# --------------------------------------------------------------------------- #
# VERI KATMANI
# --------------------------------------------------------------------------- #
def fetch_prices(tickers, years=12):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance yok: pip install yfinance  (veya --demo)")
    data = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)
    px = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    return px.dropna(how="all").ffill()


def make_demo_prices(tickers, years=12, seed=11):
    """Sentetik ama gercekci: korelasyonlu varliklar + ortak piyasa faktoru +
    rejim degisimleri. Boru hattini internetsiz test eder."""
    rng = np.random.default_rng(seed)
    n = years * TRADING_DAYS
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    # ortak piyasa faktoru (rejimli)
    market = np.zeros(n)
    vol_state = 0.010
    for i in range(1, n):
        if rng.random() < 0.01:  # rejim degisim olasiligi
            vol_state = rng.choice([0.007, 0.012, 0.025])
        market[i] = rng.normal(0.0003, vol_state)
    out = {}
    for t in tickers:
        beta = rng.uniform(0.3, 1.4)
        idio = rng.normal(rng.uniform(-0.0001, 0.0005), rng.uniform(0.006, 0.018), n)
        rets = beta * market + idio
        out[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(out, index=idx)


# --------------------------------------------------------------------------- #
# FAKTOR MOTORU  (her faktor SADECE t anina kadarki veriyle hesaplanir)
# --------------------------------------------------------------------------- #
def _winsorize(s, lo=0.05, hi=0.95):
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)


def _zscore(s):
    s = _winsorize(s.astype(float))
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0


def compute_factor_panel(px):
    """Tum varliklar icin kesitsel faktor degerleri (son gozlem)."""
    rets = px.pct_change()
    feats = {}
    for t in px.columns:
        s = px[t].dropna()
        if len(s) < TRADING_DAYS + 22:
            continue
        f = {}
        # Momentum 12-1 (son ay haric, son 12 ay) — Jegadeesh-Titman
        f["mom"] = s.iloc[-22] / s.iloc[-TRADING_DAYS] - 1
        # Zaman-serisi momentum — Moskowitz-Ooi-Pedersen (trailing 12m isaret+guc)
        f["ts_mom"] = s.iloc[-1] / s.iloc[-TRADING_DAYS] - 1
        # Deger/uzun-vade reversal — De Bondt-Thaler: log-trendin altinda mi?
        logp = np.log(s.values[-min(len(s), 5 * TRADING_DAYS):])
        x = np.arange(len(logp))
        resid = logp - np.polyval(np.polyfit(x, logp, 1), x)
        z = (resid[-1] - resid.mean()) / (resid.std() + 1e-9)
        f["value"] = -z                                  # trendin altinda = deger
        # Dusuk-oynaklik anomalisi — Frazzini-Pedersen: dusuk vol cazip
        f["lowvol"] = -rets[t].iloc[-TRADING_DAYS:].std() * np.sqrt(TRADING_DAYS)
        # Trend kalitesi: 200g ortalamanin uzerinde mi
        ma200 = s.iloc[-200:].mean() if len(s) >= 200 else s.mean()
        f["trend"] = (s.iloc[-1] / ma200) - 1
        feats[t] = f
    return pd.DataFrame(feats).T


def composite_score(panel, regime="neutral"):
    """Faktorleri kesitsel z-skora cevirip rejim agirligiyla birlestirir."""
    w = FACTOR_WEIGHTS[regime]
    z = pd.DataFrame({k: _zscore(panel[k]) for k in w})
    score = sum(w[k] * z[k] for k in w)
    return score.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# REJIM TESPITI — Hidden Markov Model (Hamilton 1989)
# --------------------------------------------------------------------------- #
def detect_regime(px, n_states=3):
    """Piyasa (esit-agirlikli) getiri + oynaklik uzerinden rejim tespiti.
    Doner: 'risk_on' | 'risk_off' | 'neutral' (son gunun rejimi).

    NOT: Birincil yontem scikit-learn GaussianMixture'dir -- HAZIR WHEEL ile
    gelir, C++ derleyici (MSVC) GEREKTIRMEZ. hmmlearn'in Python 3.14'te derleme
    sorununu tamamen ortadan kaldirir. Rejim kumeleme icin Gaussian karisim
    modeli standart bir yaklasimdir. Son care olarak oynaklik esigi kullanilir."""
    mkt = px.pct_change().mean(axis=1).dropna()
    vol = mkt.rolling(21).std()
    X = pd.concat([mkt, vol], axis=1).dropna().values
    if len(X) < 250:
        return "neutral"
    try:
        from sklearn.mixture import GaussianMixture
        # ozellikleri olcekle (getiri ve vol farkli buyukluklerde)
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        gm = GaussianMixture(n_components=n_states, covariance_type="full",
                             n_init=5, random_state=42, max_iter=300)
        states = gm.fit_predict(Xs)
        # her durumun ortalama (olceklenmemis) getirisi
        means = [X[states == k, 0].mean() if (states == k).any() else -np.inf
                 for k in range(n_states)]
        order = np.argsort(means)
        last = states[-1]
        if last == order[-1]:
            return "risk_on"
        if last == order[0]:
            return "risk_off"
        return "neutral"
    except Exception:
        # Son care: oynaklik esigi (derleme/baglilik hicbiri gerekmez)
        recent_vol = vol.iloc[-1]
        if recent_vol > vol.quantile(0.75):
            return "risk_off"
        if recent_vol < vol.quantile(0.35):
            return "risk_on"
        return "neutral"


# --------------------------------------------------------------------------- #
# HIERARCHICAL RISK PARITY — Lopez de Prado (2016)
# --------------------------------------------------------------------------- #
def _ivp(cov):
    """Inverse-variance portfolio agirliklari."""
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def _cluster_var(cov, items):
    c = cov.loc[items, items]
    w = _ivp(c.values).reshape(-1, 1)
    return float((w.T @ c.values @ w).item())


def _quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def hrp_weights(returns):
    """Hierarchical Risk Parity portfoy agirliklari."""
    returns = returns.dropna(axis=1, how="any")
    if returns.shape[1] < 2:
        n = returns.shape[1]
        return pd.Series([1.0 / max(n, 1)] * n, index=returns.columns)
    cov = returns.cov()
    corr = returns.corr()
    dist = np.sqrt(((1 - corr) / 2).clip(0, 1))
    link = linkage(squareform(dist.values, checks=False), method="single")
    sort_ix = _quasi_diag(link)
    ordered = corr.index[sort_ix].tolist()
    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= (1 - alpha)
    return w.reindex(returns.columns).fillna(0)


def build_portfolio(px, top_k=8):
    """Faktor skoru ile en cazip top_k varligi sec, HRP ile agirliklandir."""
    regime = detect_regime(px)
    panel = compute_factor_panel(px)
    scores = composite_score(panel, regime)
    picks = scores.index[:top_k].tolist()
    rets = px[picks].pct_change().iloc[-TRADING_DAYS:]
    w = hrp_weights(rets)
    return regime, scores, w.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# DOGRULAMA 1 — Purged & Embargoed Walk-Forward (Lopez de Prado 2018)
# --------------------------------------------------------------------------- #
def purged_walk_forward(px, train=3 * TRADING_DAYS, test=TRADING_DAYS // 4,
                        embargo=21, top_k=8):
    """Her adimda SADECE gecmis egitim penceresiyle portfoy kur, embargo
    bosluk birak (serisel sizinti engeli), sonra GELECEK test penceresinde
    getiriyi olc."""
    px = px.dropna()
    idx = px.index
    out = []
    i = train
    while i + embargo + test < len(px):
        train_px = px.iloc[:i]
        try:
            _, _, w = build_portfolio(train_px, top_k)
        except Exception:
            i += test
            continue
        a, b = i + embargo, i + embargo + test   # embargo sonrasi test
        seg = px.iloc[a:b][w.index]
        port_ret = (seg.pct_change().fillna(0) @ w.values).add(1).prod() - 1
        bench = (px.iloc[a:b].pct_change().fillna(0).mean(axis=1)).add(1).prod() - 1
        out.append({"tarih": idx[a], "strateji": port_ret, "benchmark": bench})
        i += test
    return pd.DataFrame(out).set_index("tarih") if out else pd.DataFrame()


def sharpe(returns, periods_per_year):
    r = pd.Series(returns).dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# DOGRULAMA 2 — Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# --------------------------------------------------------------------------- #
def deflated_sharpe(returns, n_trials, periods_per_year):
    """Coklu deneme + carpiklik/basiklik etkisini hesaba katarak Sharpe'in
    gercekten 0'dan buyuk olma olasiligini doner [0,1]."""
    r = pd.Series(returns).dropna()
    if len(r) < 5:
        return np.nan, np.nan
    sr = r.mean() / r.std()                       # donemsel (annualize edilmemis)
    T = len(r)
    g = r.skew()
    k = r.kurt() + 3                              # pandas fazlalik basiklik doner
    # cok-deneme altinda beklenen maksimum Sharpe (Bailey-LdP)
    emax = np.sqrt(2 * np.log(max(n_trials, 2)))
    sr0 = (emax * (1 - np.euler_gamma) +
           np.euler_gamma * np.sqrt(2 * np.log(max(n_trials, 2) * np.e))) / np.sqrt(T)
    denom = np.sqrt(1 - g * sr + (k - 1) / 4 * sr ** 2)
    dsr = norm.cdf(((sr - sr0) * np.sqrt(T - 1)) / (denom + 1e-12))
    return float(dsr), float(sr * np.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# DOGRULAMA 3 — Probability of Backtest Overfitting (PBO) via CSCV
# Bailey, Borwein, Lopez de Prado, Zhu (2017)
# --------------------------------------------------------------------------- #
def probability_backtest_overfitting(perf_matrix, n_splits=8):
    """perf_matrix: (T_donem x N_strateji) getiri matrisi.
    In-sample en iyi stratejinin out-of-sample siralamasi medyanin altina
    duser mi? PBO = bu olayin olasiligi. Yuksek PBO = overfit risk yuksek."""
    from itertools import combinations
    M = np.asarray(perf_matrix)
    T, N = M.shape
    if N < 2 or T < n_splits:
        return np.nan
    n_splits = min(n_splits, T // 2 * 2)
    if n_splits < 2:
        return np.nan
    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    logits = []
    for combo in combinations(range(n_splits), half):
        is_idx = np.concatenate([blocks[c] for c in combo])
        oos_idx = np.concatenate([blocks[c] for c in range(n_splits) if c not in combo])
        is_perf = M[is_idx].mean(axis=0)
        oos_perf = M[oos_idx].mean(axis=0)
        best = np.argmax(is_perf)                 # in-sample sampiyon
        # bu stratejinin out-of-sample yuzdelik sirasi
        rank = (oos_perf <= oos_perf[best]).mean()
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())            # medyan altina dusme orani


# --------------------------------------------------------------------------- #
# KOMPLE BACKTEST RAPORU — "ne kadar surede ne kadar kazandirmis, neye karsi"
# --------------------------------------------------------------------------- #
def _max_drawdown(equity):
    """Equity egrisinden maksimum tepe-dip dusus (negatif sayi)."""
    eq = pd.Series(equity)
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())


def backtest_report(px, top_k=8, start_capital=400000.0, periods_per_year=4):
    """Purged walk-forward backtest'ten KOMPLE performans raporu:
    sure, toplam getiri, CAGR, Sharpe, oynaklik, maks dusus, benchmark
    karsilastirmasi ve baslangic sermayesinin ne olduguna donustugu.

    DURUSTLUK: out-of-sample (ileri-yonlu) testtir, lookahead yoktur; yine de
    GECMIS performanstir, gelecegi GARANTI ETMEZ."""
    bt = purged_walk_forward(px, top_k=top_k)
    if bt.empty or len(bt) < 4:
        return {"ok": False, "error": "Backtest icin yeterli veri yok."}

    eq_s = (1 + bt["strateji"]).cumprod()
    eq_b = (1 + bt["benchmark"]).cumprod()
    dates = [pd.Timestamp(d).strftime("%Y-%m") for d in bt.index]

    n_years = max((bt.index[-1] - bt.index[0]).days / 365.25, 0.5)
    cum_s = float(eq_s.iloc[-1] - 1)
    cum_b = float(eq_b.iloc[-1] - 1)
    cagr_s = (1 + cum_s) ** (1 / n_years) - 1
    cagr_b = (1 + cum_b) ** (1 / n_years) - 1

    return {
        "ok": True,
        "baslangic": dates[0], "bitis": dates[-1],
        "yil": round(n_years, 1), "periyot": len(bt),
        "toplam_getiri_strateji_%": round(cum_s * 100, 1),
        "toplam_getiri_benchmark_%": round(cum_b * 100, 1),
        "cagr_strateji_%": round(cagr_s * 100, 1),
        "cagr_benchmark_%": round(cagr_b * 100, 1),
        "sharpe_strateji": round(float(sharpe(bt["strateji"], periods_per_year)), 2),
        "sharpe_benchmark": round(float(sharpe(bt["benchmark"], periods_per_year)), 2),
        "oynaklik_strateji_%": round(float(bt["strateji"].std() * np.sqrt(periods_per_year)) * 100, 1),
        "maks_dusus_strateji_%": round(_max_drawdown(eq_s.values) * 100, 1),
        "maks_dusus_benchmark_%": round(_max_drawdown(eq_b.values) * 100, 1),
        "yenme_orani_%": round(float((bt["strateji"] > bt["benchmark"]).mean()) * 100, 1),
        "baslangic_sermaye": start_capital,
        "son_sermaye_strateji": round(start_capital * (1 + cum_s)),
        "son_sermaye_benchmark": round(start_capital * (1 + cum_b)),
        "tarihler": dates,
        "equity_strateji": [round(float(v), 4) for v in eq_s.values],
        "equity_benchmark": [round(float(v), 4) for v in eq_b.values],
    }


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    tickers = list(UNIVERSE.keys())
    print("=" * 74)
    print("  MULTI-FACTOR · REGIME-AWARE · HRP ALLOCATION ENGINE")
    print("=" * 74)

    if args.demo:
        print("  [DEMO] Sentetik korelasyonlu veri.\n")
        px = make_demo_prices(tickers)
    else:
        print("  Gercek veri cekiliyor (yfinance)...\n")
        px = fetch_prices(tickers)

    # --- Mevcut durum: rejim + skor + HRP agirliklari ---
    regime, scores, weights = build_portfolio(px, args.top_k)
    print(f">>> TESPIT EDILEN PIYASA REJIMI : {regime.upper()}")
    print(f"    (faktor egilimi bu rejime gore ayarlandi)\n")

    rank = pd.DataFrame({
        "Varlik": [UNIVERSE.get(t, t) for t in scores.index],
        "Faktor_Skoru": scores.round(2).values,
    }, index=scores.index)
    print(">>> FAKTOR CAZIBE SIRALAMASI (kesitsel z-skor, rejim-agirlikli)\n")
    print(rank.to_string())

    print("\n>>> ONERILEN HRP PORTFOY AGIRLIKLARI (risk-dengeli)\n")
    wtab = pd.DataFrame({
        "Varlik": [UNIVERSE.get(t, t) for t in weights.index],
        "Agirlik_%": (weights * 100).round(1).values,
    }, index=weights.index)
    print(wtab[wtab["Agirlik_%"] > 0].to_string())
    wtab.to_csv("hrp_portfoy.csv")

    # --- Walk-forward backtest ---
    bt = None
    if args.backtest or args.validate:
        print("\n" + "-" * 74)
        print(">>> PURGED & EMBARGOED WALK-FORWARD BACKTEST\n")
        bt = purged_walk_forward(px, top_k=args.top_k)
        if bt.empty:
            print("  Yeterli veri yok.")
        else:
            ppy = 4   # ceyreklik test pencereleri
            s_sh = sharpe(bt["strateji"], ppy)
            b_sh = sharpe(bt["benchmark"], ppy)
            win = (bt["strateji"] > bt["benchmark"]).mean()
            print(f"  Periyot sayisi        : {len(bt)}")
            print(f"  Strateji Sharpe (yil) : {s_sh:.2f}")
            print(f"  Benchmark Sharpe (yil): {b_sh:.2f}")
            print(f"  Benchmark'i yenme     : {win*100:.1f}%")
            print(f"  Birikimli strateji    : {((1+bt['strateji']).prod()-1)*100:+.1f}%")
            print(f"  Birikimli benchmark   : {((1+bt['benchmark']).prod()-1)*100:+.1f}%")

    # --- Overfitting dogrulamasi ---
    if args.validate and bt is not None and not bt.empty:
        print("\n" + "-" * 74)
        print(">>> OVERFITTING DOGRULAMASI\n")
        # DSR: kac faktor agirlik kombinasyonu "denedik" -> 3 rejim x 5 faktor ~ konservatif 15
        dsr, ann_sr = deflated_sharpe(bt["strateji"].values, n_trials=15, periods_per_year=4)
        print(f"  Deflated Sharpe Ratio : {dsr:.3f}")
        print(f"    -> Sharpe'in gercekten >0 olma olasiligi (coklu-deneme duzeltmeli).")
        print(f"    -> 0.95+ : guclu | 0.5 civari : suphe | dusuk : muhtemelen sans.\n")

        # PBO: faktor-agirlik varyantlarindan bir performans matrisi kur
        variants = []
        for reg in FACTOR_WEIGHTS:
            r = _variant_backtest(px, reg, args.top_k)
            if r is not None and len(r) > 0:
                variants.append(r)
        if len(variants) >= 2:
            L = min(len(v) for v in variants)
            mat = np.column_stack([v[:L] for v in variants])
            pbo = probability_backtest_overfitting(mat)
            print(f"  Probability of Backtest Overfitting (PBO): {pbo:.2f}")
            print(f"    -> 0'a yakin: saglam | 0.5+: secimin overfit olma riski yuksek.")
        else:
            print("  PBO icin yeterli varyant uretilemedi.")

    print("\n" + "=" * 74)
    print("  UYARI: Karar-destek aracidir. Olasilik ve risk-agirlik onerir,")
    print("  'AL' garantisi vermez. Final karar senin. Finansal danisman degilim.")
    print("=" * 74)


def _variant_backtest(px, regime, top_k):
    """Belirli bir rejim-agirlik varyantiyla sabit walk-forward (PBO icin)."""
    px = px.dropna()
    train = 3 * TRADING_DAYS
    test = TRADING_DAYS // 4
    embargo = 21
    out = []
    i = train
    while i + embargo + test < len(px):
        panel = compute_factor_panel(px.iloc[:i])
        if panel.empty:
            i += test
            continue
        scores = composite_score(panel, regime)
        picks = scores.index[:top_k].tolist()
        rets = px[picks].pct_change().iloc[max(0, i - TRADING_DAYS):i]
        w = hrp_weights(rets)
        a, b = i + embargo, i + embargo + test
        seg = px.iloc[a:b][w.index]
        out.append((seg.pct_change().fillna(0) @ w.values).add(1).prod() - 1)
        i += test
    return np.array(out)


if __name__ == "__main__":
    main()
