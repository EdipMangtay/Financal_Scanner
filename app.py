#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
QUANT TERMINAL  —  web dashboard
================================================================================
quant_engine + robust_data uzerine kurulu, tarayicida calisan kontrol paneli.

CALISTIRMA:
  pip install flask pandas numpy scipy scikit-learn yfinance requests
  python app.py
  -> tarayici otomatik acilir:  http://127.0.0.1:5000

  python app.py --demo      # internetsiz sentetik veriyle ac
  python app.py --port 8080 # farkli port

Cozulen sorunlar:
  - hmmlearn DERLEME sorunu: rejim artik sklearn GaussianMixture ile (wheel,
    derleyici gerekmez). quant_engine.detect_regime guncellendi.
  - "database is locked": robust_data.RobustFeed (temp cache + threadsiz +
    tek tek + retry) kullaniliyor.
================================================================================
"""

import argparse
import threading
import webbrowser
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response

import quant_engine as qe
from daily_scanner import UNIVERSE, decide
from robust_data import RobustFeed

app = Flask(__name__)
DEMO_MODE = False


# --------------------------------------------------------------------------- #
# TARAMA CEKIRDEGI
# --------------------------------------------------------------------------- #
def scan_core(top=8, demo=False):
    tickers = list(UNIVERSE.keys())
    if demo:
        px = qe.make_demo_prices(tickers, years=12)
        freshness = {"son_veri_tarihi": "DEMO", "gun_yas": 0, "bayat": False}
        status = {"mod": "demo (sentetik veri)"}
    else:
        feed = RobustFeed(years=12)
        px, freshness = feed.get(tickers)
        status = feed.status

    regime = qe.detect_regime(px)
    panel = qe.compute_factor_panel(px)
    scores = qe.composite_score(panel, regime)
    table = decide(px, scores, regime)

    short = table[table["KARAR"] == "GUCLU ADAY"]["Ticker"].tolist()[:top]
    weights = {}
    if short:
        rets = px[short].pct_change().iloc[-252:]
        w = qe.hrp_weights(rets)
        weights = {t: round(float(w.get(t, 0)) * 100, 1) for t in short}

    return {
        "ok": True,
        "tarih": pd.Timestamp.today().strftime("%Y-%m-%d %H:%M"),
        "regime": regime,
        "freshness": freshness,
        "status": {k: (v if not isinstance(v, list) else ", ".join(v))
                   for k, v in status.items()},
        "rows": table.to_dict(orient="records"),
        "shortlist": [{"ticker": t, "name": UNIVERSE.get(t, t),
                       "weight": weights.get(t, 0)} for t in short],
    }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/scan")
def api_scan():
    try:
        top = int(request.args.get("top", 8))
        demo = request.args.get("demo", "0") == "1" or DEMO_MODE
        return jsonify(scan_core(top=top, demo=demo))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc()[-1200:]})


@app.route("/api/backtest")
def api_backtest():
    try:
        demo = request.args.get("demo", "0") == "1" or DEMO_MODE
        capital = float(request.args.get("capital", 400000))
        top = int(request.args.get("top", 8))
        tickers = list(UNIVERSE.keys())
        px = (qe.make_demo_prices(tickers) if demo
              else RobustFeed(years=12).get(tickers)[0])
        return jsonify(qe.backtest_report(px, top_k=top, start_capital=capital))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc()[-1200:]})


@app.route("/api/validate")
def api_validate():
    try:
        demo = request.args.get("demo", "0") == "1" or DEMO_MODE
        tickers = list(UNIVERSE.keys())
        px = (qe.make_demo_prices(tickers) if demo
              else RobustFeed(years=12).get(tickers)[0])
        bt = qe.purged_walk_forward(px, top_k=8)
        if bt.empty:
            return jsonify({"ok": False, "error": "Yeterli veri yok."})
        s_sh = qe.sharpe(bt["strateji"], 4)
        b_sh = qe.sharpe(bt["benchmark"], 4)
        dsr, ann = qe.deflated_sharpe(bt["strateji"].values, 15, 4)
        variants = []
        for reg in qe.FACTOR_WEIGHTS:
            r = qe._variant_backtest(px, reg, 8)
            if r is not None and len(r):
                variants.append(r)
        pbo = None
        if len(variants) >= 2:
            L = min(len(v) for v in variants)
            mat = np.column_stack([v[:L] for v in variants])
            pbo = qe.probability_backtest_overfitting(mat)
        return jsonify({
            "ok": True, "periods": len(bt),
            "strat_sharpe": round(float(s_sh), 2),
            "bench_sharpe": round(float(b_sh), 2),
            "win_rate": round(float((bt["strateji"] > bt["benchmark"]).mean()) * 100, 1),
            "cum_strat": round(float((1 + bt["strateji"]).prod() - 1) * 100, 1),
            "cum_bench": round(float((1 + bt["benchmark"]).prod() - 1) * 100, 1),
            "dsr": round(float(dsr), 3) if dsr == dsr else None,
            "pbo": round(float(pbo), 2) if pbo is not None else None,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc()[-1200:]})


# --------------------------------------------------------------------------- #
# ON YUZ
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html><html lang="tr"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QUANT TERMINAL</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0a0b0d; --panel:#111317; --panel2:#15181d; --line:#23272f;
  --ink:#e9e6dd; --muted:#8b8f98; --gold:#e8b34a; --green:#3fbf7f;
  --red:#e5604d; --blue:#5a9bd8;
  --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
  --serif:'Instrument Serif',Georgia,serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.5;-webkit-font-smoothing:antialiased;
  background-image:radial-gradient(circle at 15% -10%,rgba(232,179,74,.06),transparent 40%),
    radial-gradient(circle at 90% 0%,rgba(90,155,216,.05),transparent 35%);}
.grain{position:fixed;inset:0;pointer-events:none;opacity:.025;z-index:1;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");}
.wrap{max-width:1180px;margin:0 auto;padding:34px 26px 80px;position:relative;z-index:2}
header{display:flex;justify-content:space-between;align-items:flex-end;
  border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:26px}
.logo{font-family:var(--serif);font-size:42px;letter-spacing:.5px;line-height:1}
.logo .i{font-style:italic;color:var(--gold)}
.tag{font-family:var(--mono);font-size:11px;color:var(--muted);
  text-transform:uppercase;letter-spacing:2px;margin-top:6px}
.controls{display:flex;gap:10px;align-items:center}
button{font-family:var(--mono);font-size:12px;letter-spacing:1px;cursor:pointer;
  background:var(--panel);color:var(--ink);border:1px solid var(--line);
  padding:10px 16px;border-radius:7px;transition:.15s;text-transform:uppercase}
button:hover{border-color:var(--gold);color:var(--gold)}
button.primary{background:var(--gold);color:#1a1407;border-color:var(--gold);font-weight:600}
button.primary:hover{filter:brightness(1.08);color:#1a1407}
button:disabled{opacity:.4;cursor:wait}
select{font-family:var(--mono);font-size:12px;background:var(--panel);color:var(--ink);
  border:1px solid var(--line);padding:9px 10px;border-radius:7px}
input#cap{font-family:var(--mono);font-size:12px;background:var(--panel);color:var(--ink);
  border:1px solid var(--line);padding:9px 10px;border-radius:7px;width:104px}
input#cap:focus{outline:none;border-color:var(--gold)}
.btpanel{display:none;margin-top:14px}
.btgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 17px}
.metric .k{margin-bottom:6px}
.metric .v{font-family:var(--mono);font-size:22px;letter-spacing:.5px}
.metric .sub{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:3px}
.capbox{background:linear-gradient(160deg,var(--panel2),var(--panel));
  border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin-bottom:16px}
.capbox .row{display:flex;justify-content:space-between;align-items:baseline;padding:7px 0}
.capbox .lab{font-family:var(--mono);font-size:12px;color:var(--muted)}
.capbox .big{font-family:var(--serif);font-size:30px}
.chartwrap{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px}
.legend{display:flex;gap:20px;font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:6px}
.legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px}
.bar{display:grid;grid-template-columns:1.1fr 1fr;gap:16px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 22px}
.k{font-family:var(--mono);font-size:10.5px;color:var(--muted);
  letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px}
.regime{font-family:var(--serif);font-size:34px;letter-spacing:.5px}
.regime.on{color:var(--green)} .regime.off{color:var(--red)} .regime.neutral{color:var(--gold)}
.fresh{font-family:var(--mono);font-size:13px;color:var(--muted);margin-top:6px}
.stale{color:var(--red);font-weight:600}
.src{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:4px}
h2{font-family:var(--serif);font-size:25px;font-weight:400;margin:30px 0 14px;
  display:flex;align-items:baseline;gap:12px}
h2 .n{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:1px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px}
th{text-align:left;color:var(--muted);font-weight:500;font-size:10.5px;letter-spacing:1px;
  text-transform:uppercase;padding:9px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid #1a1d23}
tr:hover td{background:var(--panel2)}
.tk{color:var(--gold);font-weight:600}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--green)} .neg{color:var(--red)}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.5px;padding:3px 8px;
  border-radius:20px;white-space:nowrap;border:1px solid}
.p-strong{color:var(--green);border-color:rgba(63,191,127,.4);background:rgba(63,191,127,.08)}
.p-cheap{color:var(--gold);border-color:rgba(232,179,74,.4);background:rgba(232,179,74,.08)}
.p-watch{color:var(--blue);border-color:rgba(90,155,216,.35);background:rgba(90,155,216,.07)}
.p-avoid{color:var(--red);border-color:rgba(229,96,77,.35);background:rgba(229,96,77,.07)}
.p-neutral{color:var(--muted);border-color:var(--line)}
.shorts{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
.short{background:linear-gradient(160deg,var(--panel2),var(--panel));
  border:1px solid var(--line);border-left:3px solid var(--green);border-radius:11px;padding:16px 18px}
.short .t{font-family:var(--mono);color:var(--gold);font-size:16px;font-weight:600}
.short .nm{font-size:12.5px;color:var(--muted);margin:2px 0 12px}
.wbar{height:6px;background:#1a1d23;border-radius:4px;overflow:hidden}
.wbar i{display:block;height:100%;background:linear-gradient(90deg,var(--gold),var(--green))}
.wlab{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:6px;display:flex;justify-content:space-between}
.empty{font-family:var(--mono);font-size:13px;color:var(--muted);
  border:1px dashed var(--line);border-radius:11px;padding:26px;text-align:center;line-height:1.7}
.note{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:10px}
.htf{color:var(--gold)}
.foot{margin-top:40px;padding-top:18px;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.8}
.spin{display:inline-block;width:13px;height:13px;border:2px solid var(--line);
  border-top-color:var(--gold);border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px}
@keyframes s{to{transform:rotate(360deg)}}
.vbox{background:var(--panel2);border:1px solid var(--line);border-radius:11px;
  padding:18px 20px;margin-top:14px;font-family:var(--mono);font-size:13px;display:none}
.vrow{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1a1d23}
.vrow:last-child{border:0}
.vgood{color:var(--green)} .vbad{color:var(--red)} .vmid{color:var(--gold)}
.fade{animation:f .5s ease both}
@keyframes f{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
</style></head><body>
<div class="grain"></div>
<div class="wrap">
  <header>
    <div>
      <div class="logo">QUANT<span class="i">·</span>TERMINAL</div>
      <div class="tag">multi-factor · regime-aware · HRP · daily scan</div>
    </div>
    <div class="controls">
      <input id="cap" type="number" value="400000" step="10000" title="baslangic sermayesi (TL)">
      <select id="top">
        <option value="5">top 5</option>
        <option value="8" selected>top 8</option>
        <option value="12">top 12</option>
      </select>
      <button id="btBtn">backtest</button>
      <button id="valBtn">overfitting testi</button>
      <button id="runBtn" class="primary">↻ tarama</button>
    </div>
  </header>

  <div id="content">
    <div class="empty">Yukleniyor <span class="spin"></span></div>
  </div>

  <div class="foot">
    UYARI — Bu arac KISA-LISTE uretir, AL emri vermez. Adaylara HTF analizini SEN
    uygular, tetigi SEN cekersin. Bu bir finansal danismanlik degildir.<br>
    rejim: sklearn GaussianMixture · portfoy: Hierarchical Risk Parity (Lopez de Prado 2016)
    · dogrulama: Deflated Sharpe + PBO (Bailey & Lopez de Prado)
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const DEMO = %DEMO%;

function pill(k){
  const m={'GUCLU ADAY':['p-strong','GUCLU ADAY'],'UCUZ ama DUSUSTE':['p-cheap','UCUZ·DUSUSTE'],
    'IZLE':['p-watch','IZLE'],'KACIN':['p-avoid','KACIN'],'NOTR':['p-neutral','NOTR']};
  const [c,t]=m[k]||['p-neutral',k];return `<span class="pill ${c}">${t}</span>`;
}
const sgn = v => v>0?`<span class="pos">+${v}</span>`:(v<0?`<span class="neg">${v}</span>`:v);

async function run(){
  const btn=$('#runBtn'); btn.disabled=true; btn.innerHTML='<span class="spin"></span> calisiyor';
  $('#content').innerHTML='<div class="empty">Veri cekiliyor, faktorler hesaplaniyor <span class="spin"></span></div>';
  const top=$('#top').value;
  try{
    const r=await fetch(`/api/scan?top=${top}&demo=${DEMO?1:0}`);
    const d=await r.json();
    if(!d.ok){ $('#content').innerHTML=`<div class="empty vbad">Hata: ${d.error}</div>`; }
    else render(d);
  }catch(e){ $('#content').innerHTML=`<div class="empty vbad">Baglanti hatasi: ${e}</div>`; }
  btn.disabled=false; btn.innerHTML='↻ tarama';
}

function render(d){
  const rc = d.regime==='risk_on'?'on':(d.regime==='risk_off'?'off':'neutral');
  const rl = {risk_on:'RISK-ON',risk_off:'RISK-OFF',neutral:'NOTR'}[d.regime];
  const f = d.freshness;
  const freshTxt = f.son_veri_tarihi==='DEMO' ? 'DEMO veri'
    : `son veri: ${f.son_veri_tarihi} (${f.gun_yas} gun)` + (f.bayat?' · <span class="stale">BAYAT</span>':' · taze');
  const src = Object.entries(d.status).map(([k,v])=>`${k}: ${v}`).join(' &nbsp;·&nbsp; ');

  let rows = d.rows.map(x=>`<tr>
    <td class="tk">${x.Ticker}</td><td>${x.Varlik}</td>
    <td class="num">${x.Skor}</td><td class="num">${x.Yuzdelik}</td>
    <td>${x.Trend==='Uzeri'?'<span class="pos">▲</span>':'<span class="neg">▼</span>'} ${x.Trend}</td>
    <td class="num">${sgn(x['Mom_12_1_%'])}</td>
    <td class="num neg">${x['Zirveden_%']}</td>
    <td>${pill(x.KARAR)}</td></tr>`).join('');

  let shorts;
  if(d.shortlist.length){
    shorts = `<div class="shorts">`+d.shortlist.map(s=>`<div class="short fade">
      <div class="t">${s.ticker}</div><div class="nm">${s.name}</div>
      <div class="wbar"><i style="width:${Math.min(s.weight,100)}%"></i></div>
      <div class="wlab"><span>risk-agirlik</span><span>%${s.weight}</span></div>
    </div>`).join('')+`</div>
    <div class="note htf">↳ Sistem bu adaylari saglikli buldu. HTF analizini uygula, tetigi SEN cek.</div>`;
  }else{
    shorts = `<div class="empty">Su an "GUCLU ADAY" kriterini gecen YOK.<br>
      Bu da bir bilgidir: piyasa pahali/zayif olabilir — <span class="htf">beklemek de bir karardir.</span></div>`;
  }

  $('#content').innerHTML = `
    <div class="bar fade">
      <div class="card"><div class="k">Tespit Edilen Piyasa Rejimi</div>
        <div class="regime ${rc}">${rl}</div>
        <div class="note">faktor egilimi bu rejime gore ayarlandi · ${d.tarih}</div></div>
      <div class="card"><div class="k">Veri Tazeligi & Kaynak</div>
        <div class="fresh">${freshTxt}</div><div class="src">${src}</div></div>
    </div>
    <div id="vbox" class="vbox"></div>
    <div id="btpanel" class="btpanel"></div>
    <h2>Kisa Liste <span class="n">— HTF icin</span></h2>${shorts}
    <h2>Tam Tarama <span class="n">— ${d.rows.length} varlik · karar tablosu</span></h2>
    <div class="card" style="padding:6px 10px"><table>
      <thead><tr><th>Ticker</th><th>Varlik</th><th class="num">Skor</th>
        <th class="num">%lik</th><th>Trend</th><th class="num">Mom 12-1</th>
        <th class="num">Zirveden</th><th>Karar</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

const tl = n => '₺'+Number(n).toLocaleString('tr-TR');
const colr = (a,b)=> a>=b?'pos':'neg';

function sparkline(dates, sStrat, sBench){
  const W=900,H=240,P=34;
  const all=sStrat.concat(sBench); const mn=Math.min(...all), mx=Math.max(...all);
  const x=i=>P+(i/(sStrat.length-1))*(W-2*P);
  const y=v=>H-P-((v-mn)/((mx-mn)||1))*(H-2*P);
  const path=arr=>arr.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ');
  // 1.0 referans cizgisi (baslangic)
  const base=y(1.0);
  const ticks=[0,Math.floor(sStrat.length/2),sStrat.length-1]
    .map(i=>`<text x="${x(i)}" y="${H-10}" fill="#8b8f98" font-size="10" font-family="monospace" text-anchor="middle">${dates[i]}</text>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet">
    <line x1="${P}" y1="${base}" x2="${W-P}" y2="${base}" stroke="#23272f" stroke-dasharray="3 4"/>
    <path d="${path(sBench)}" fill="none" stroke="#5a9bd8" stroke-width="2" opacity=".85"/>
    <path d="${path(sStrat)}" fill="none" stroke="#e8b34a" stroke-width="2.4"/>
    ${ticks}
    <text x="${P}" y="${base-6}" fill="#8b8f98" font-size="9.5" font-family="monospace">baslangic (1.0x)</text>
  </svg>`;
}

async function backtest(){
  const b=$('#btBtn'); b.disabled=true; b.innerHTML='<span class="spin"></span> backtest';
  const cap=$('#cap').value||400000, top=$('#top').value;
  const p=$('#btpanel'); p.style.display='block';
  p.innerHTML='<div class="empty">Walk-forward backtest calisiyor (1-2 dk surebilir) <span class="spin"></span></div>';
  try{
    const r=await fetch(`/api/backtest?demo=${DEMO?1:0}&capital=${cap}&top=${top}`);
    const d=await r.json();
    if(!d.ok){ p.innerHTML=`<div class="empty vbad">Backtest hatasi: ${d.error}</div>`; }
    else renderBT(d);
  }catch(e){ p.innerHTML=`<div class="empty vbad">${e}</div>`; }
  b.disabled=false; b.innerHTML='backtest';
}

function renderBT(d){
  const beatCls = d['toplam_getiri_strateji_%']>=d['toplam_getiri_benchmark_%']?'vgood':'vbad';
  $('#btpanel').innerHTML = `
  <h2 class="fade">Backtest Raporu <span class="n">— ${d.baslangic} → ${d.bitis} · ${d.yil} yil · ${d.periyot} periyot (out-of-sample)</span></h2>
  <div class="capbox fade">
    <div class="row"><span class="lab">${tl(d.baslangic_sermaye)} koysaydin — STRATEJI ile bugun</span>
      <span class="big ${beatCls==='vgood'?'pos':''}" style="color:var(--gold)">${tl(d.son_sermaye_strateji)}</span></div>
    <div class="row"><span class="lab">Ayni parayi tum evrene esit dagitsaydin (benchmark)</span>
      <span class="big" style="color:var(--blue)">${tl(d.son_sermaye_benchmark)}</span></div>
  </div>
  <div class="btgrid fade">
    <div class="metric"><div class="k">Toplam Getiri · Strateji</div>
      <div class="v ${colr(d['toplam_getiri_strateji_%'],0)}">${d['toplam_getiri_strateji_%']>0?'+':''}${d['toplam_getiri_strateji_%']}%</div>
      <div class="sub">benchmark: ${d['toplam_getiri_benchmark_%']}%</div></div>
    <div class="metric"><div class="k">Yillik Bilesik (CAGR)</div>
      <div class="v">${d['cagr_strateji_%']}%</div>
      <div class="sub">benchmark: ${d['cagr_benchmark_%']}%</div></div>
    <div class="metric"><div class="k">Sharpe (yil)</div>
      <div class="v">${d.sharpe_strateji}</div>
      <div class="sub">benchmark: ${d.sharpe_benchmark}</div></div>
    <div class="metric"><div class="k">Maks Dusus</div>
      <div class="v neg">${d['maks_dusus_strateji_%']}%</div>
      <div class="sub">benchmark: ${d['maks_dusus_benchmark_%']}%</div></div>
    <div class="metric"><div class="k">Yillik Oynaklik</div>
      <div class="v">${d['oynaklik_strateji_%']}%</div>
      <div class="sub">risk seviyesi</div></div>
    <div class="metric"><div class="k">Benchmark'i Yenme</div>
      <div class="v">${d['yenme_orani_%']}%</div>
      <div class="sub">periyotlarin orani</div></div>
  </div>
  <div class="chartwrap fade">
    <div class="legend"><span><i style="background:#e8b34a"></i>Strateji</span><span><i style="background:#5a9bd8"></i>Benchmark</span>
      <span style="margin-left:auto">${tl(d.baslangic_sermaye)} baslangic = 1.0x</span></div>
    ${sparkline(d.tarihler, d.equity_strateji, d.equity_benchmark)}
  </div>`;
  $('#btpanel').scrollIntoView({behavior:'smooth',block:'start'});
}

async function validate(){
  const b=$('#valBtn'); b.disabled=true; b.innerHTML='<span class="spin"></span> test';
  try{
    const r=await fetch(`/api/validate?demo=${DEMO?1:0}`); const d=await r.json();
    const box=$('#vbox'); box.style.display='block';
    if(!d.ok){ box.innerHTML=`<div class="vbad">Test hatasi: ${d.error}</div>`; }
    else{
      const dsrCls=d.dsr>=.95?'vgood':(d.dsr>=.5?'vmid':'vbad');
      const pboCls=(d.pbo!=null&&d.pbo<=.25)?'vgood':(d.pbo<=.5?'vmid':'vbad');
      box.innerHTML=`
       <div class="vrow"><span>Walk-forward periyot</span><span>${d.periods}</span></div>
       <div class="vrow"><span>Strateji Sharpe (yil)</span><span>${d.strat_sharpe}</span></div>
       <div class="vrow"><span>Benchmark Sharpe (yil)</span><span>${d.bench_sharpe}</span></div>
       <div class="vrow"><span>Benchmark'i yenme</span><span>${d.win_rate}%</span></div>
       <div class="vrow"><span>Deflated Sharpe (gercek mi?)</span><span class="${dsrCls}">${d.dsr ?? '—'}</span></div>
       <div class="vrow"><span>PBO (overfit riski)</span><span class="${pboCls}">${d.pbo ?? '—'}</span></div>`;
    }
  }catch(e){ $('#vbox').style.display='block'; $('#vbox').innerHTML=`<div class="vbad">${e}</div>`; }
  b.disabled=false; b.innerHTML='overfitting testi';
}

$('#runBtn').onclick=run;
$('#btBtn').onclick=backtest;
$('#valBtn').onclick=validate;
$('#top').onchange=run;
run();
</script>
</body></html>"""


@app.route("/")
def index():
    html = PAGE.replace("%DEMO%", "true" if DEMO_MODE else "false")
    return Response(html, mimetype="text/html")


def main():
    global DEMO_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Sentetik veriyle ac")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    DEMO_MODE = args.demo

    url = f"http://127.0.0.1:{args.port}"
    print("=" * 64)
    print("  QUANT TERMINAL calisiyor:", url)
    print("  Mod:", "DEMO (sentetik)" if DEMO_MODE else "CANLI veri")
    print("  Kapatmak icin: Ctrl+C")
    print("=" * 64)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
