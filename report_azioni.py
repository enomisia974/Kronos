import os, smtplib, sys, io, gc, base64
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from model import Kronos, KronosTokenizer, KronosPredictor

FTSE_MIB = (
    "A2A.MI","AMP.MI","AZM.MI","BGN.MI","BMED.MI","BMPS.MI","BAMI.MI","BPE.MI",
    "BC.MI","CPR.MI","DIA.MI","ENEL.MI","ENI.MI","ERG.MI","FBK.MI","G.MI",
    "HER.MI","IG.MI","IP.MI","ISP.MI","IVG.MI","LDO.MI","MB.MI","MONC.MI",
    "NEXI.MI","PIRC.MI","PST.MI","PRY.MI","RACE.MI","REC.MI","SPM.MI","SRG.MI",
    "STLAM.MI","STM.MI","TEN.MI","TIT.MI","TRN.MI","UCG.MI","UNI.MI","US.MI",
)

ASSET_NAMES = {
    "A2A.MI":"A2A","AMP.MI":"Amplifon","AZM.MI":"Azimut","BGN.MI":"B.Generali",
    "BMED.MI":"Mediolanum","BMPS.MI":"M.Paschi","BAMI.MI":"Banco BPM","BPE.MI":"BPER",
    "BC.MI":"Brunello Cuc.","CPR.MI":"Campari","DIA.MI":"Diasorin","ENEL.MI":"Enel",
    "ENI.MI":"Eni","ERG.MI":"Erg","FBK.MI":"Fineco","G.MI":"Generali",
    "HER.MI":"Hera","IG.MI":"Italgas","IP.MI":"Interpump","ISP.MI":"Intesa SP",
    "IVG.MI":"Iveco","LDO.MI":"Leonardo","MB.MI":"Mediobanca","MONC.MI":"Moncler",
    "NEXI.MI":"Nexi","PIRC.MI":"Pirelli","PST.MI":"Poste It.","PRY.MI":"Prysmian",
    "RACE.MI":"Ferrari","REC.MI":"Recordati","SPM.MI":"Saipem","SRG.MI":"Snam",
    "STLAM.MI":"Stellantis","STM.MI":"STMicro","TEN.MI":"Tenaris","TIT.MI":"Tim",
    "TRN.MI":"Terna","UCG.MI":"UniCredit","UNI.MI":"Unipol","US.MI":"UnipolSai",
}

def calcola_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def calcola_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calcola_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def fmt(v):
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.2f}"

def scarica_batch(tickers):
    raw = yf.download(list(tickers), period="1y", interval="1d", progress=False, group_by='ticker')
    result = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy()
            if df.empty:
                continue
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.rename(columns={'Date':'timestamps','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
            df['volume'] = df['volume'].astype(float)
            numeric_cols = ['open','high','low','close','volume']
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            result[t] = df
        except:
            continue
    return result

def analizza_ticker(ticker, df=None):
    if df is None:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty:
                return None
            df = df.reset_index()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df = df.rename(columns={'Date':'timestamps','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
            df['volume'] = df['volume'].astype(float)
        except:
            return None

    if len(df) < 60:
        return None

    p = df['close'].iloc[-1]
    ema10 = calcola_ema(df['close'], 10).iloc[-1]
    ema20 = calcola_ema(df['close'], 20).iloc[-1]
    ema50 = calcola_ema(df['close'], 50).iloc[-1]
    rsi = calcola_rsi(df['close']).iloc[-1]
    atr = calcola_atr(df).iloc[-1]
    vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(10).mean().iloc[-1]

    resistenza = df['high'].iloc[-60:-1].max() if len(df) >= 60 else df['high'].max()
    dist_res = (p - resistenza) / resistenza * 100

    returns_20d = df['close'].pct_change().tail(20).dropna()
    varianza_20d = returns_20d.var()
    media_close_20d = df['close'].tail(20).mean()
    vcp = varianza_20d < (media_close_20d * 0.0008)

    ch30 = ((p - df['close'].iloc[-30]) / df['close'].iloc[-30]) * 100 if len(df) >= 30 else 0
    vola = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100

    score = 0
    filters = []

    if ema10 > ema20 > ema50:
        score += 40
        filters.append(("EMA Trend +40", True))
    else:
        filters.append(("EMA Trend", False))

    if vcp:
        score += 30
        filters.append(("VCP +30", True))
    else:
        filters.append(("VCP", False))

    if p >= resistenza:
        score += 30
        filters.append(("Breakout +30", True))
        resist_status = "BREAKOUT"
    elif -0.20 <= dist_res <= 1.00:
        score += 20
        filters.append(("Ready +20", True))
        resist_status = "READY"
    else:
        filters.append(("Resistenza", False))
        resist_status = "LONTANO"

    if score >= 70:
        action = "BUY"
    elif score >= 60:
        action = "READY"
    else:
        action = "NO"

    sl = ema20
    capitale = 10000
    rischio_pct = 1.50
    perdita_max = capitale * (rischio_pct / 100)
    dist_sl_pct = abs(p - sl) / p if p != sl else 0.01
    size_quote = perdita_max / (abs(p - sl)) if abs(p - sl) > 0 else 0
    size_eur = size_quote * p
    rischio_eur = perdita_max
    tp = p + (p - sl) * 2 if p != sl else p * 1.03
    tp_potenziale = (tp - p) / p * 100

    return {
        'ticker': ticker, 'name': ASSET_NAMES.get(ticker, ticker), 'prezzo': p,
        'ema10': ema10, 'ema20': ema20, 'ema50': ema50,
        'rsi': rsi, 'atr': atr, 'vol_ratio': vol_ratio,
        'resistenza': resistenza, 'dist_res_pct': dist_res, 'resist_status': resist_status,
        'vcp': vcp, 'ch30': ch30, 'vola': vola,
        'score': score, 'action': action, 'filters': filters,
        'sl_ema20': sl, 'dist_sl_pct': dist_sl_pct * 100,
        'size_quote': int(size_quote), 'size_eur': size_eur,
        'tp': tp, 'tp_potenziale': tp_potenziale, 'rischio_eur': rischio_eur,
    }

def invia_email(subject, html_body, img_bytes=None, ticker=""):
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "enomisia974@gmail.com")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_to = os.environ.get("SMTP_TO", "enomisia974@gmail.com")

    if not smtp_password:
        print("   SMTP_PASSWORD non impostata, salto email")
        return

    msg = MIMEMultipart('related')
    msg.preamble = 'This is a multi-part message in MIME format.'
    msg_alt = MIMEMultipart('alternative')
    msg['From'] = smtp_user
    msg['To'] = smtp_to
    msg['Subject'] = subject
    msg_alt.attach(MIMEText(subject, 'plain', 'utf-8'))
    msg_alt.attach(MIMEText(html_body, 'html', 'utf-8'))
    msg.attach(msg_alt)
    if img_bytes:
        img = MIMEImage(img_bytes, name=f'{ticker}_chart.png')
        img.add_header('Content-ID', '<chart>')
        img.add_header('Content-Disposition', 'inline')
        msg.attach(img)
    s = smtplib.SMTP(smtp_server, smtp_port)
    s.starttls()
    s.login(smtp_user, smtp_password)
    s.sendmail(smtp_user, [smtp_to], msg.as_string())
    s.quit()
    print(f"   Email inviata a {smtp_to}")

def genera_html_individuale(s, pred_eur, backtest):
    p_now = s['prezzo']
    ticker = s['ticker']
    now = datetime.now()
    now_str = now.strftime("%d %B %Y \u2022 %H:%M")
    now_dmy = now.strftime("%d/%m/%Y")

    el = "#22c55e" if s['action'] == "BUY" else "#f59e0b" if s['action'] == "READY" else "#64748b"
    tc = "#22c55e" if s['ch30'] >= 0 else "#ef4444"

    tab_pred = ""
    if pred_eur is not None:
        for idx, row in pred_eur.iterrows():
            d = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
            dl = row['close'] - row['open']
            dc = "#22c55e" if dl >= 0 else "#ef4444"
            tab_pred += f"<tr><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{d}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{fmt(row['open'])}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{fmt(row['high'])}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{fmt(row['low'])}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;font-weight:600;'>{fmt(row['close'])}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:{dc};font-weight:600;'>{dl:+,.2f}</td></tr>"

    tab_back = ""
    if backtest:
        for b in backtest:
            tab_back += f"<tr><td style='padding:6px;border-bottom:1px solid #2a2a2a;color:#cbd5e1;'>{b['date']}</td><td style='padding:6px;border-bottom:1px solid #2a2a2a;color:#cbd5e1;'>{fmt(b['predetto'])}</td><td style='padding:6px;border-bottom:1px solid #2a2a2a;color:#cbd5e1;'>{fmt(b['reale'])}</td><td style='padding:6px;border-bottom:1px solid #2a2a2a;font-weight:600;color:{'#ef4444' if b['error_pct']<0 else '#22c55e'}'>{b['error_pct']:+.2f}%</td><td style='padding:6px;border-bottom:1px solid #2a2a2a;color:#cbd5e1;'>{abs(b['error_pct']):.2f}%</td></tr>"

    mae_text = f"MAE: {abs(np.mean([b['error_pct'] for b in backtest])):.2f}%" if backtest else ""

    f_rows = "".join(f"<tr><td style='padding:3px 8px;font-size:11px;color:{'#22c55e' if ok else '#ef4444'};'>{'✓' if ok else '✗'} {n}</td></tr>" for n, ok in s['filters'])

    return f"""<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>{ticker} — Report Kronos</title></head>
<body style="margin:0;padding:0;background:#f4f4f6;color:#1e293b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px;line-height:1.4;">
<div style="width:100%;padding:0;margin:0;">

<div style="background:linear-gradient(135deg,#0b1120,#1e293b);padding:24px 20px 18px;border-radius:10px 10px 0 0;text-align:center;">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1.2px;">Kronos Daily \u2022 {now_dmy} \u2022 {now.strftime('%H:%M')}</div>
  <h1 style="font-size:18px;font-weight:700;color:#fff;margin:6px 0 2px;">{ticker} — {s['name']}</h1>
  <div style="font-size:36px;font-weight:800;color:#d4a853;">{fmt(p_now)} <span style="font-size:14px;color:#64748b;font-weight:400;">EUR</span></div>
</div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Segnale</div>
<div style="background:#fff;border-radius:8px;border:2px solid {el};padding:14px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <div style="background:{el};color:#fff;font-weight:700;font-size:16px;padding:6px 14px;border-radius:4px;">{s['action']}</div>
    <div style="font-size:13px;color:#334155;">Score <b>{s['score']}</b>/100</div>
    <div style="margin-left:auto;font-size:11px;color:#64748b;">{s['resist_status']}</div>
  </div>
  <table width="100%" style="font-size:13px;">
    <tr><td style="padding:3px 8px;color:#64748b;">Entrata</td><td style="padding:3px 8px;font-weight:700;">{fmt(s['prezzo'])} EUR</td>
        <td style="padding:3px 8px;color:#64748b;">Stop Loss</td><td style="padding:3px 8px;font-weight:700;color:#ef4444;">{fmt(s['sl_ema20'])} EUR</td></tr>
    <tr><td style="padding:3px 8px;color:#64748b;">Dimensione</td><td style="padding:3px 8px;font-weight:700;">{s['size_quote']} azioni ({fmt(s['size_eur'])} EUR)</td>
        <td style="padding:3px 8px;color:#64748b;">Max Perdita</td><td style="padding:3px 8px;font-weight:700;">150.00 EUR</td></tr>
  </table>
  <table style="margin-top:6px;">{f_rows}</table>
  <div style="font-size:11px;color:#94a3b8;margin-top:6px;padding-top:6px;border-top:1px solid #f1f5f9;">
    RSI {s['rsi']:.0f} | ATR {s['atr']:.2f} | VolRatio {s['vol_ratio']:.2f}x | Dist.Res {s['dist_res_pct']:+.2f}% | {mae_text}
  </div>
</div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Analisi Grafica</div>
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:10px;"><div style="padding:6px;text-align:center;"><img src="cid:chart" style="max-width:100%;height:auto;display:block;" alt="{ticker} Chart"/></div></div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Previsione Dettaglio</div>
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:10px;overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:12px;">
<thead><tr><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Data</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Apertura</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Max</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Min</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Chiusura</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Delta</th></tr></thead>
<tbody>{tab_pred}</tbody>
</table></div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Backtest</div>
<div style="background:#222;border-radius:8px;border:1px solid #333;overflow:hidden;margin-bottom:10px;overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:11px;">
<thead><tr><th style="background:#1a1a1a;color:#94a3b8;padding:8px 6px;font-weight:600;font-size:9px;border-bottom:1px solid #333;text-align:left;">Data</th><th style="background:#1a1a1a;color:#94a3b8;padding:8px 6px;font-weight:600;font-size:9px;border-bottom:1px solid #333;text-align:left;">Predetto</th><th style="background:#1a1a1a;color:#94a3b8;padding:8px 6px;font-weight:600;font-size:9px;border-bottom:1px solid #333;text-align:left;">Reale</th><th style="background:#1a1a1a;color:#94a3b8;padding:8px 6px;font-weight:600;font-size:9px;border-bottom:1px solid #333;text-align:left;">Errore</th><th style="background:#1a1a1a;color:#94a3b8;padding:8px 6px;font-weight:600;font-size:9px;border-bottom:1px solid #333;text-align:left;">MAE</th></tr></thead>
<tbody>{tab_back}</tbody>
</table></div>

<div style="text-align:center;padding:14px 0;font-size:10px;color:#94a3b8;">
Kronos Quantitative Research \u2022 {now_str}<br>Dati: Yahoo Finance \u2022 Modello: Kronos-base</div>
</div></body></html>"""

def genera_html_watchlist(results):
    now = datetime.now()
    now_dmy = now.strftime("%d/%m/%Y")
    rows = ""
    for r in results:
        ac = "#22c55e" if r['action'] == "BUY" else "#f59e0b" if r['action'] == "READY" else "#64748b"
        bg = "#052e16" if r['action'] == "BUY" else "#1c1917" if r['action'] == "READY" else "#0f172a"
        tc = "#22c55e" if r['ch30'] >= 0 else "#ef4444"
        rows += f"<tr style='background:{bg};'>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:600;font-size:12px;'>{r['name']}</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#94a3b8;font-size:12px;'>{r['ticker'].replace('.MI','')}</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-size:12px;'>{fmt(r['prezzo'])}</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;text-align:center;'><div style='background:{ac};color:#fff;font-weight:700;font-size:10px;padding:1px 7px;border-radius:3px;text-align:center;display:inline-block;'>{r['action']}</div></td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:600;text-align:center;font-size:12px;'>{r['score']}</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#ef4444;text-align:right;font-size:12px;'>{fmt(r['sl_ema20'])}</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#22c55e;text-align:right;font-size:12px;font-weight:600;'>{fmt(r['tp'])}</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#e2e8f0;text-align:right;font-size:12px;'>{r['size_quote']}</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#ef4444;text-align:right;font-size:12px;'>{fmt(r['rischio_eur'])}</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#22c55e;text-align:right;font-size:12px;'>{r['tp_potenziale']:+.1f}%</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:{tc};text-align:right;font-size:12px;'>{r['ch30']:+.1f}%</td>"
        rows += f"<td style='padding:6px;border-bottom:1px solid #1e293b;color:#94a3b8;text-align:center;font-size:12px;'>{r['rsi']:.0f}</td>"
        rows += f"</tr>"

    return f"""<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>FTSE MIB — Watchlist Kronos</title></head>
<body style="margin:0;padding:0;background:#0f172a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;line-height:1.4;">
<div style="width:100%;padding:0;margin:0;">

<div style="background:linear-gradient(135deg,#0b1120,#1e293b);padding:24px 20px 18px;text-align:center;">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1.2px;">Kronos Screening \u2022 {now_dmy}</div>
  <h1 style="font-size:20px;font-weight:700;color:#d4a853;margin:6px 0;">FTSE MIB — Watchlist</h1>
  <div style="font-size:12px;color:#94a3b8;">{len(results)} titoli analizzati \u2022 Score: EMA Trend + VCP + Resistenza</div>
</div>

<div style="padding:10px;overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:11px;">
<thead><tr style="background:#1e293b;">
<th style="padding:6px;color:#64748b;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:left;">Nome</th>
<th style="padding:6px;color:#64748b;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:left;">Tkr</th>
<th style="padding:6px;color:#64748b;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:left;">Acquisto</th>
<th style="padding:6px;color:#64748b;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:center;">Segn</th>
<th style="padding:6px;color:#64748b;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:center;">Scr</th>
<th style="padding:6px;color:#ef4444;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:right;">SL</th>
<th style="padding:6px;color:#22c55e;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:right;">Vendita</th>
<th style="padding:6px;color:#64748b;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:right;">Azioni</th>
<th style="padding:6px;color:#ef4444;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:right;">Rischio</th>
<th style="padding:6px;color:#22c55e;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:right;">TP%</th>
<th style="padding:6px;color:#64748b;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:right;">30gg</th>
<th style="padding:6px;color:#64748b;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.4px;border-bottom:2px solid #334155;text-align:center;">RSI</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
</div>

<div style="text-align:center;padding:14px 0;font-size:10px;color:#64748b;">
Kronos Quantitative Research \u2022 Capitale 10.000 EUR \u2022 Rischio 1.50% \u2022 SL su EMA(20) \u2022 Vendita: 2× rischio
</div>
</div></body></html>"""

def genera_html_riepilogo(results, top_data):
    now = datetime.now()
    now_dmy = now.strftime("%d/%m/%Y")
    bux = sum(1 for r in results if r['action']=='BUY')
    rdy = sum(1 for r in results if r['action']=='READY')

    watch_rows = ""
    for r in results:
        ac = "#22c55e" if r['action'] == "BUY" else "#f59e0b" if r['action'] == "READY" else "#64748b"
        bg = "#052e16" if r['action'] == "BUY" else "#1c1917" if r['action'] == "READY" else "#0f172a"
        tc = "#22c55e" if r['ch30'] >= 0 else "#ef4444"
        watch_rows += f"<tr style='background:{bg};'>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:600;font-size:11px;'>{r['name']}</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#94a3b8;font-size:11px;'>{r['ticker'].replace('.MI','')}</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-size:11px;'>{fmt(r['prezzo'])}</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;text-align:center;font-size:11px;'><div style='background:{ac};color:#fff;font-weight:700;font-size:10px;padding:1px 6px;border-radius:3px;text-align:center;display:inline-block;'>{r['action']}</div></td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:600;text-align:center;font-size:11px;'>{r['score']}</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#94a3b8;text-align:right;font-size:11px;'>{fmt(r['sl_ema20'])}</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#22c55e;text-align:right;font-size:11px;font-weight:600;'>{fmt(r['tp'])}</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#e2e8f0;text-align:right;font-size:11px;'>{r['size_quote']} ({fmt(r['size_eur'])})</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#ef4444;text-align:right;font-size:11px;'>{fmt(r['rischio_eur'])}</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#22c55e;text-align:right;font-size:11px;'>{r['tp_potenziale']:+.1f}%</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:{tc};text-align:right;font-size:11px;'>{r['ch30']:+.1f}%</td>"
        watch_rows += f"<td style='padding:5px 6px;border-bottom:1px solid #1e293b;color:#94a3b8;text-align:center;font-size:11px;'>{r['rsi']:.0f}</td>"
        watch_rows += f"</tr>"

    detail_blocks = ""
    for td in top_data:
        s, pred, backtest, img_b64 = td['s'], td['pred'], td['backtest'], td['img_b64']
        el = "#22c55e" if s['action'] == "BUY" else "#f59e0b"
        mae_val = abs(np.mean([b['error_pct'] for b in backtest])) if backtest else 0
        pred_last = pred['close'].iloc[-1] if pred is not None and not pred.empty else s['prezzo']
        pred_var = (pred_last - s['prezzo']) / s['prezzo'] * 100
        f_ok = sum(1 for _, ok in s['filters'] if ok)
        f_tot = len(s['filters'])
        detail_blocks += f"""
<div style="background:linear-gradient(135deg,#0b1120,#1e293b);padding:12px 14px;border-radius:6px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <div style="background:{el};color:#fff;font-weight:700;font-size:13px;padding:3px 10px;border-radius:3px;">{s['action']}</div>
    <div style="font-size:14px;font-weight:700;color:#d4a853;">{s['name']}</div>
    <div style="font-size:12px;color:#94a3b8;">{s['ticker'].replace('.MI','')}</div>
    <div style="margin-left:auto;font-size:12px;color:#e2e8f0;">Score <b>{s['score']}</b></div>
  </div>
  <table style="width:100%;font-size:11px;border-collapse:collapse;">
    <tr>
      <td style="padding:2px 4px;color:#64748b;">Acquisto</td>
      <td style="padding:2px 4px;color:#e2e8f0;font-weight:600;">{fmt(s['prezzo'])} €</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">Stop Loss</td>
      <td style="padding:2px 4px;color:#ef4444;font-weight:600;">{fmt(s['sl_ema20'])} €</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">Vendita</td>
      <td style="padding:2px 4px;color:#22c55e;font-weight:600;">{fmt(s['tp'])} €</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">R:R</td>
      <td style="padding:2px 4px;color:#e2e8f0;">1:2</td>
    </tr>
    <tr>
      <td style="padding:2px 4px;color:#64748b;">Azioni</td>
      <td style="padding:2px 4px;color:#e2e8f0;font-weight:600;">{s['size_quote']}</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">Esposizione</td>
      <td style="padding:2px 4px;color:#e2e8f0;font-weight:600;">{fmt(s['size_eur'])} €</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">Max Perdita</td>
      <td style="padding:2px 4px;color:#ef4444;font-weight:600;">{fmt(s['rischio_eur'])} €</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">Max Guadagno</td>
      <td style="padding:2px 4px;color:#22c55e;font-weight:600;">{fmt(s['rischio_eur']*2)} €</td>
    </tr>
    <tr>
      <td style="padding:2px 4px;color:#64748b;">RSI</td>
      <td style="padding:2px 4px;color:#e2e8f0;">{s['rsi']:.0f}</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">Resistenza</td>
      <td style="padding:2px 4px;color:#e2e8f0;">{s['dist_res_pct']:+.1f}%</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">Filtri</td>
      <td style="padding:2px 4px;color:#22c55e;">{f_ok}/{f_tot}</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">MAE</td>
      <td style="padding:2px 4px;color:#e2e8f0;">{mae_val:.2f}%</td>
      <td style="padding:2px 8px;color:#64748b;text-align:right;">AI 14gg</td>
      <td style="padding:2px 4px;color:{'#22c55e' if pred_var>=0 else '#ef4444'};">{pred_var:+.1f}%</td>
    </tr>
  </table>
  <div style="margin-top:6px;text-align:center;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%;height:auto;border-radius:4px;" alt="{s['ticker']}"/></div>
</div>"""

    return f"""<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>FTSE MIB Top 10 — Kronos</title></head>
<body style="margin:0;padding:0;background:#0b1120;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;line-height:1.4;">
<div style="max-width:680px;margin:0 auto;padding:0;">

<div style="background:linear-gradient(135deg,#0b1120,#1e293b);padding:22px 18px;text-align:center;border-bottom:2px solid #d4a853;">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1.2px;">Kronos Quantitative Research \u2022 {now_dmy}</div>
  <h1 style="font-size:22px;font-weight:700;color:#d4a853;margin:4px 0;">FTSE MIB — Top Picks</h1>
  <div style="font-size:12px;color:#94a3b8;">{len(results)} titoli analizzati | {bux} BUY | {rdy} READY | {len(top_data)} migliori selezionati</div>
</div>

<div style="padding:12px;">
  <div style="font-size:13px;font-weight:700;color:#d4a853;margin-bottom:6px;">Watchlist Completa</div>
  <div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;">
  <thead><tr style="background:#1e293b;">
    <th style="padding:5px 6px;color:#64748b;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:left;">Nome</th>
    <th style="padding:5px 6px;color:#64748b;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:left;">Tkr</th>
    <th style="padding:5px 6px;color:#64748b;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:left;">Acq</th>
    <th style="padding:5px 6px;color:#64748b;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:center;">Segn</th>
    <th style="padding:5px 6px;color:#64748b;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:center;">Scr</th>
    <th style="padding:5px 6px;color:#ef4444;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:right;">SL</th>
    <th style="padding:5px 6px;color:#22c55e;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:right;">Vendita</th>
    <th style="padding:5px 6px;color:#64748b;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:right;">Azioni (€)</th>
    <th style="padding:5px 6px;color:#ef4444;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:right;">Rischio</th>
    <th style="padding:5px 6px;color:#22c55e;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:right;">TP%</th>
    <th style="padding:5px 6px;color:#64748b;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:right;">30gg</th>
    <th style="padding:5px 6px;color:#64748b;font-weight:600;font-size:9px;letter-spacing:0.4px;text-align:center;">RSI</th>
  </tr></thead>
  <tbody>{watch_rows}</tbody>
  </table>
  </div>
</div>

<div style="padding:0 12px 12px;">
  <div style="font-size:13px;font-weight:700;color:#d4a853;margin-bottom:6px;">Analisi Dettaglio — Top {len(top_data)}</div>
  {detail_blocks}
</div>

<div style="text-align:center;padding:14px;font-size:10px;color:#475569;border-top:1px solid #1e293b;">
Kronos Quantitative Research \u2022 Capitale 10.000 EUR \u2022 Rischio 1.50% \u2022 SL su EMA(20)<br>
Dati: Yahoo Finance \u2022 Modello AI: Kronos-base</div>
</div></body></html>"""

def main():
    if len(sys.argv) > 1 and sys.argv[1].upper() != "MIB":
        ticker = sys.argv[1]
        print(f"Modalita singola: {ticker}")
        s = analizza_ticker(ticker)
        if not s:
            print("Errore analisi ticker")
            return
        print(f"  {s['name']}: {fmt(s['prezzo'])} EUR, Score {s['score']}, {s['action']}")

        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        df = df.reset_index()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.rename(columns={'Date':'timestamps','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        df['volume'] = df['volume'].astype(float)
        df['amount'] = df['close'] * df['volume']

        x_df = df.iloc[-90:].copy()
        df_input = x_df[['open','high','low','close','volume','amount']].copy()
        x_ts = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
        y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1]+timedelta(days=1), periods=14, freq='B'))

        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        predictor = KronosPredictor(tokenizer=tokenizer, model=model)
        pred = predictor.predict(df=df_input, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=14)
        pred_eur = pred.copy()

        backtest = []
        for offset in range(14, 0, -1):
            cut = len(df) - offset
            if cut < 90: continue
            x = df.iloc[cut-90:cut].copy()
            inp = x[['open','high','low','close','volume','amount']].copy()
            xt = pd.Series(pd.to_datetime(x['timestamps']).dt.tz_localize(None))
            yt = pd.Series([xt.iloc[-1]+timedelta(days=1)])
            p = predictor.predict(df=inp, x_timestamp=xt, y_timestamp=yt, pred_len=1)
            err = (p['close'].iloc[0] - df.iloc[cut]['close']) / df.iloc[cut]['close'] * 100
            d = df.iloc[cut]['timestamps']
            backtest.append({'date': str(d.date()) if hasattr(d,'date') else str(d)[:10],
                            'predetto': p['close'].iloc[0], 'reale': df.iloc[cut]['close'], 'error_pct': err})

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=x_ts, open=x_df['open'], high=x_df['high'],
            low=x_df['low'], close=x_df['close'], name="Storico",
            increasing_line_color='#22c55e', decreasing_line_color='#ef4444'))
        fig.add_trace(go.Candlestick(x=y_ts, open=pred['open'], high=pred['high'],
            low=pred['low'], close=pred['close'], name="Previsione",
            increasing_line_color='#3b82f6', decreasing_line_color='#8b5cf6'))
        fig.update_layout(template='none', height=400, margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='Inter,sans-serif', size=11, color='#334155'),
            hovermode='x unified', legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10)),
            xaxis_rangeslider_visible=False)
        fig.update_xaxes(gridcolor='#f1f5f9', zeroline=False)
        fig.update_yaxes(gridcolor='#f1f5f9', zeroline=False, title='Prezzo (EUR)')
        img_bytes = fig.to_image(format='png', width=1000, height=400, scale=1)

        html = genera_html_individuale(s, pred_eur, backtest)
        safe = ticker.replace("=", "").replace("-", "_")
        with open(f"Report_{safe}.html", "w", encoding="utf-8") as f:
            f.write(html)
        invia_email(f"[Kronos Daily] {ticker} \u2022 {s['action']} \u2022 {fmt(s['prezzo'])} EUR", html, img_bytes, ticker)
        return

    print("=== FTSE MIB SCREENING ===")

    print("Download dati in batch...")
    all_data = scarica_batch(FTSE_MIB)
    print(f"  Scaricati {len(all_data)}/{len(FTSE_MIB)} ticker")

    results = []
    for ticker in FTSE_MIB:
        if ticker not in all_data:
            continue
        s = analizza_ticker(ticker, df=all_data[ticker])
        if s:
            print(f"  {ticker}: Score {s['score']} {s['action']}")
            results.append(s)

    results.sort(key=lambda x: x['score'], reverse=True)
    top = [r for r in results if r['action'] in ("BUY", "READY")]

    bux = sum(1 for r in results if r['action']=='BUY')
    rdy = sum(1 for r in results if r['action']=='READY')
    print(f"\n=== RISULTATI: {len(results)} titoli, {bux} BUY, {rdy} READY, {len(results)-bux-rdy} NO ===")
    for r in results[:5]:
        print(f"    {r['name']:20s} Score {r['score']:3d} {r['action']:5s} {fmt(r['prezzo']):>8s} EUR")

    html_wl = genera_html_watchlist(results)
    invia_email(f"[Kronos Watchlist] FTSE MIB \u2022 {bux} BUY \u2022 {rdy} READY", html_wl)

    if not top:
        print("\nNessun segnale BUY/READY, invio solo watchlist.")
        print("\n=== COMPLETATO ===")
        return

    top_n = min(len(top), 10)
    print(f"\nCarico modello Kronos per top {top_n}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)

    top_data = []
    for s in top[:top_n]:
        print(f"\n  Elaborazione: {s['name']} ({s['action']}, Score {s['score']})")
        try:
            df = all_data[s['ticker']].copy()
            df['amount'] = df['close'].fillna(0) * df['volume'].fillna(0)
            df = df.ffill().bfill().fillna(0)

            x_df = df.iloc[-90:].copy()
            df_input = x_df[['open','high','low','close','volume','amount']].copy()
            x_ts = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
            y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1]+timedelta(days=1), periods=14, freq='B'))

            pred = predictor.predict(df=df_input, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=14)

            backtest = []
            for offset in range(14, 0, -1):
                cut = len(df) - offset
                if cut < 90:
                    continue
                x = df.iloc[cut-90:cut].copy()
                inp = x[['open','high','low','close','volume','amount']].copy()
                xt = pd.Series(pd.to_datetime(x['timestamps']).dt.tz_localize(None))
                yt = pd.Series([xt.iloc[-1]+timedelta(days=1)])
                p = predictor.predict(df=inp, x_timestamp=xt, y_timestamp=yt, pred_len=1)
                err = (p['close'].iloc[0] - df.iloc[cut]['close']) / df.iloc[cut]['close'] * 100
                d = df.iloc[cut]['timestamps']
                backtest.append({'date': str(d.date()) if hasattr(d,'date') else str(d)[:10],
                                'predetto': p['close'].iloc[0], 'reale': df.iloc[cut]['close'], 'error_pct': err})

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=x_ts, open=x_df['open'], high=x_df['high'],
                low=x_df['low'], close=x_df['close'], name="Storico",
                increasing_line_color='#22c55e', decreasing_line_color='#ef4444'))
            fig.add_trace(go.Candlestick(x=y_ts, open=pred['open'], high=pred['high'],
                low=pred['low'], close=pred['close'], name="Previsione",
                increasing_line_color='#3b82f6', decreasing_line_color='#8b5cf6'))
            fig.update_layout(template='none', height=200, margin=dict(l=6,r=6,t=6,b=6),
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(family='Inter,sans-serif', size=8, color='#334155'),
                hovermode='x unified', showlegend=False,
                xaxis_rangeslider_visible=False)
            fig.update_xaxes(gridcolor='#f1f5f9', zeroline=False, visible=False)
            fig.update_yaxes(gridcolor='#f1f5f9', zeroline=False, title='', visible=False)
            img_bytes = fig.to_image(format='png', width=600, height=200, scale=1)
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

            top_data.append({'s': s, 'pred': pred, 'backtest': backtest, 'img_b64': img_b64})
            gc.collect()
        except Exception as e:
            print(f"  Errore {s['ticker']}: {e}")

    if top_data:
        html = genera_html_riepilogo(results, top_data)
        invia_email(f"[Kronos Top 10] FTSE MIB \u2022 {bux} BUY \u2022 {rdy} READY", html)
        print(f"\nRiepilogo top {len(top_data)} inviato.")

    print("\n=== COMPLETATO ===")

if __name__ == "__main__":
    main()
