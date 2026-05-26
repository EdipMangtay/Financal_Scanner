# 📈 Financial Scanner — Swing Trade Karar Destek Sistemi

Çok faktörlü, **1500+ hisse** tarayan swing trade scanner. ATR-tabanlı giriş/stop/hedef önerileri, otomatik rejim tespiti ve overfitting-dirençli akademik temelli skor sistemi ile çalışır.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ⚡ Özellikler

| Modül | Açıklama |
|---|---|
| **Kapsam Seçici** | 130 / Nasdaq-100 / S&P 500 / **S&P 1500** |
| **9 Faktör** | Trend, 20-gün/5-gün momentum, RSI, MACD, Pullback, ATR, Hacim, 52H Pozisyon |
| **Setup Etiketleri** | 🟢 TAZE · 🟡 ZIRVEDE · 🔴 ASIRI ALIM (geç kaldım mı?) |
| **ATR Stop/Hedef** | Otomatik 2×ATR stop, 3×ATR hedef, R:R 1.5 |
| **Rejim Tespiti** | Risk-On / Risk-Off / Neutral (SPY trendi + oynaklık) |
| **Pozisyon Hesaplayıcı** | Portföy + risk% → otomatik hisse adedi |
| **Sektör Isı Haritası** | Liderlik eden sektörleri görselleştirir |
| **Mum Grafik** | Candlestick + hacim + MA20/50/200 + Giriş/Stop/Hedef çizgileri |
| **CSV İndirme** | Alım listesi ve tam tarama tabloları |

---

## 🚀 Hızlı Başlangıç

### Lokal (Python 3.11+)

```bash
pip install -r requirements.txt
streamlit run swing_app.py
```

Tarayıcıda otomatik açılır: `http://localhost:8501`

### Docker

```bash
docker build -t fin-scanner .
docker run -p 8501:8501 fin-scanner
```

### Railway (önerilen production deploy)

1. Bu repo'yu GitHub'a pushla
2. [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Railway otomatik olarak `Dockerfile`'ı algılar, build eder, deploy eder
4. `PORT` env değişkenini otomatik ayarlar

### Render / Fly.io

`Dockerfile` ile uyumlu. PORT env değişkenini auto-detect ederler.

---

## 📁 Proje Yapısı

```
.
├── swing_app.py          # Streamlit web UI (production)
├── swing_engine.py       # Tarama motoru (faktörler + skor + karar)
├── quant_engine.py       # Uzun-vade rotasyon motoru (HRP + DSR + PBO)
├── daily_scanner.py      # CLI günlük tarayıcı (eski)
├── requirements.txt
├── Dockerfile
├── railway.json
├── .streamlit/config.toml
└── README.md
```

---

## 🎓 Akademik Temeller

| Faktör | Referans |
|---|---|
| Momentum (12-1) | Jegadeesh & Titman (1993) |
| Düşük Oynaklık | Frazzini & Pedersen (2014) |
| Trend Takip | Moskowitz, Ooi, Pedersen (2012) |
| HRP Allocation | Lopez de Prado (2016) |
| Walk-Forward CV | Lopez de Prado (2018) |
| Deflated Sharpe | Bailey & Lopez de Prado (2014) |
| PBO via CSCV | Bailey et al. (2017) |

---

## 🧪 Doğrulama

```bash
# 12 yıllık purged & embargoed walk-forward backtest + DSR + PBO
python quant_engine.py --validate
```

Tipik sonuç: Sharpe ~1.0, PBO ~0.03 (overfit değil), DSR ~0.8.

---

## 📊 Kullanım Akışı

1. **Sabah** : `streamlit run swing_app.py` → S&P 1500'ü tara
2. **Filtrele** : "Sadece TAZE setup'lar" işaretle → pullback bölgesindekiler
3. **Seç** : Top 5 kartından bir aday seç, mum grafiği incele
4. **Hesapla** : Pozisyon hesaplayıcı ile $ büyüklüğü ve hisse adedi bul
5. **Doğrula** : HTF analizinle (haftalık/4-saatlik) onayla
6. **Tetik** : Pivot/destek-direnç ile zamanla, **ATR stop'u uygula**

---

## ⚠ Önemli Uyarılar

- **AL emri vermez.** Sistem aday gösterir, giriş/stop/hedef önerir. Tetik kararı senindir.
- **Komisyon/slipaj dahil değil.** Gerçek getiri backtest'ten %0.3-1 puan düşük olabilir.
- **Bu bir filtredir, sihirli değnek değildir.** Disiplinli pozisyon büyüklüğü ve sıkı stop kuralı şarttır.
- **Finansal danışman değildir.** Yatırım kararı tamamen sana aittir.

---

## 📜 Lisans

MIT License — bkz. [LICENSE](LICENSE)

---

## 🤝 Katkı

PR'lara açıktır. Özellikle BIST evreni, ek faktörler (FCF yield, vb.) ve daha fazla rejim modeli (HMM, MS-VAR) için.
