import os, smtplib, sys
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

def calcola_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def calcola_bb(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    std = tp.rolling(period).std()
    bbw = ((ma + 2*std) - (ma - 2*std)) / ma
    return bbw

def get_eur_rate():
    eur = yf.download("EURUSD=X", period="5d", interval="1d")
    if eur.empty:
        return 0.92
    close = eur['Close'].iloc[-1]
    return float(close) if isinstance(close, (int, float)) else float(close.iloc[0])

def is_eur_ticker(ticker):
    return ticker.endswith(".MI") or ticker.endswith(".DE") or ticker.endswith(".PA") or ticker.endswith(".TO")

def fmt(v, ticker):
    if ticker in ("BTC-USD", "ETH-USD", "SOL-USD", "GC=F"):
        return f"{v:,.0f}"
    return f"{v:,.2f}"

def backtest_kronos(df, predictor, eur_rate, n_back=10, lookback=60):
    df_bt = df.copy()
    results = []
    for offset in range(n_back, 0, -1):
        cut = len(df_bt) - offset
        if cut < lookback:
            continue
        x = df_bt.iloc[cut - lookback:cut].copy()
        inp = x[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_ts = pd.Series(pd.to_datetime(x['timestamps']).dt.tz_localize(None))
        y_ts = pd.Series([x_ts.iloc[-1] + timedelta(days=1)])
        p = predictor.predict(df=inp, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=1)
        pred_close_usd = p['close'].iloc[0]
        actual_close_usd = df_bt.iloc[cut]['close']
        err = (pred_close_usd - actual_close_usd) / actual_close_usd * 100
        d = df_bt.iloc[cut]['timestamps']
        date = str(d.date()) if hasattr(d, 'date') else str(d)[:10]
        results.append({
            'date': date,
            'predetto': pred_close_usd / eur_rate,
            'reale': actual_close_usd / eur_rate,
            'error_pct': err,
        })
    return results

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    now = datetime.now()
    now_str = now.strftime("%d %B %Y • %H:%M")
    now_dmy = now.strftime("%d/%m/%Y")

    print(f"1. Download dati {ticker}...")
    df = yf.download(ticker, period="1y", interval="1d")
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.rename(columns={
        'Date': 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    df['volume'] = df['volume'].astype(float)
    df['amount'] = df['close'] * df['volume']

    print(f"2. Tasso EUR/USD...")
    eur_rate = get_eur_rate()
    in_eur = is_eur_ticker(ticker)
    if in_eur:
        print(f"   Ticker in EUR, salto conversione")
    else:
        print(f"   1 EUR = {eur_rate:.4f} USD")

    lookback = 60
    pred_len = 5
    x_df = df.iloc[-lookback:].copy()
    df_input = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    y_ts = pd.Series(pd.date_range(
        start=x_ts.iloc[-1] + pd.Timedelta(days=1), periods=pred_len, freq='B'))

    p_now_raw = df_input['close'].iloc[-1]
    p_now = p_now_raw if in_eur else p_now_raw / eur_rate

    rsi_series = calcola_rsi(df_input['close'])
    rsi = rsi_series.iloc[-1]
    bbw = calcola_bb(x_df).iloc[-1]
    vol_ratio = df_input['volume'].iloc[-1] / df_input['volume'].rolling(10).mean().iloc[-1]
    vola = df_input['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
    ch30 = ((p_now_raw - df_input['close'].iloc[-30]) / df_input['close'].iloc[-30]) * 100

    sym = "EUR"
    print(f"   Prezzo: {fmt(p_now, ticker)} {sym}, RSI: {rsi:.0f}")

    print(f"3. Previsione Kronos...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)
    pred = predictor.predict(
        df=df_input, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len)

    pred_eur = pred.copy()
    if not in_eur:
        for col in ['open', 'high', 'low', 'close']:
            pred_eur[col] = pred[col] / eur_rate

    print(f"4. Grafico...")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x_ts, open=x_df['open']/(1 if in_eur else eur_rate),
        high=x_df['high']/(1 if in_eur else eur_rate),
        low=x_df['low']/(1 if in_eur else eur_rate),
        close=x_df['close']/(1 if in_eur else eur_rate),
        name="Storico", increasing_line_color='#22c55e', decreasing_line_color='#ef4444'))
    fig.add_trace(go.Candlestick(
        x=y_ts, open=pred_eur['open'], high=pred_eur['high'],
        low=pred_eur['low'], close=pred_eur['close'],
        name="Previsione", increasing_line_color='#3b82f6', decreasing_line_color='#8b5cf6'))
    fig.update_layout(
        template='none', height=400, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=11, color='#334155'),
        hovermode='x unified', legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor='#f1f5f9', zeroline=False)
    fig.update_yaxes(gridcolor='#f1f5f9', zeroline=False, title=f'Prezzo ({sym})')

    img_bytes = fig.to_image(format='png', width=1000, height=400, scale=1)

    print(f"5. Backtest storico...")
    backtest = backtest_kronos(df, predictor, 1 if in_eur else eur_rate)
    mae_avg = sum(abs(b['error_pct']) for b in backtest) / len(backtest) if backtest else 0
    print(f"   MAE medio: {mae_avg:.2f}% ({len(backtest)} test)")

    print(f"6. HTML email...")
    rj, rc = ("SOVRACOMPRATO", "#ef4444") if rsi > 70 else ("SOVRAVENDUTO", "#22c55e") if rsi < 30 else ("NEUTRALE", "#3b82f6")

    tab_pred = ""
    for idx, row in pred_eur.iterrows():
        d = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
        dl = row['close'] - row['open']
        dc = "#22c55e" if dl >= 0 else "#ef4444"
        ds = "+" if dl >= 0 else ""
        tab_pred += f"<tr><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{d}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{fmt(row['open'], ticker)}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{fmt(row['high'], ticker)}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{fmt(row['low'], ticker)}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;font-weight:600;'>{fmt(row['close'], ticker)}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:{dc};font-weight:600;'>{ds}{dl:+,.2f}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{ticker} — Report Kronos</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f6;color:#1e293b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px;line-height:1.4;">
<div style="width:100%;padding:0;margin:0;">

<div style="background:linear-gradient(135deg,#0b1120,#1e293b);padding:24px 20px 18px;border-radius:10px 10px 0 0;text-align:center;">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1.2px;">Kronos Daily • {now_dmy} • {now.strftime('%H:%M')}</div>
  <h1 style="font-size:18px;font-weight:700;color:#fff;margin:6px 0 2px;">{ticker} — Previsione AI</h1>
  <div style="font-size:36px;font-weight:800;color:#d4a853;">{fmt(p_now, ticker)} <span style="font-size:14px;color:#64748b;font-weight:400;">{sym}</span></div>
</div>

<table width="100%" cellpadding="0" cellspacing="0" style="margin:12px 0;table-layout:fixed;">
  <col style="width:33%;"><col style="width:33%;"><col style="width:34%;">
  <tr>
    <td style="padding:2px;vertical-align:top;">
      <div style="background:#fff;border-radius:8px;padding:10px 4px;text-align:center;border:1px solid #e2e8f0;">
        <div style="font-size:18px;font-weight:700;color:#d4a853;">KRONOS AI</div>
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;margin-top:3px;">Segnale Predittivo</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">Next {pred_len} days</div>
      </div>
    </td>
    <td style="padding:2px;vertical-align:top;">
      <div style="background:#fff;border-radius:8px;padding:10px 4px;text-align:center;border:1px solid #e2e8f0;">
        <div style="font-size:18px;font-weight:700;color:{rc};">{rj}</div>
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;margin-top:3px;">RSI 14</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{rsi:.0f}</div>
      </div>
    </td>
    <td style="padding:2px;vertical-align:top;">
      <div style="background:#fff;border-radius:8px;padding:10px 4px;text-align:center;border:1px solid #e2e8f0;">
        <div style="font-size:18px;font-weight:700;color:{'#22c55e' if ch30>=0 else '#ef4444'};">{ch30:+.1f}%</div>
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;margin-top:3px;">Performance 30gg</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">Vol {vola:.0f}%</div>
      </div>
    </td>
  </tr>
</table>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Analisi Grafica</div>
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:10px;"><div style="padding:6px;text-align:center;"><img src="cid:chart" style="max-width:100%;height:auto;display:block;" alt="{ticker} Chart"/></div></div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Previsione Dettaglio</div>
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:10px;overflow-x:auto;overflow-y:hidden;">
<table style="width:100%;border-collapse:collapse;font-size:12px;">
  <thead><tr><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Data</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Apertura</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Max</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Min</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Chiusura</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Delta</th></tr></thead>
  <tbody>{tab_pred}</tbody>
</table>
</div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Backtest Previsioni ({len(backtest)} test)</div>
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:10px;overflow-x:auto;overflow-y:hidden;">
<table style="width:100%;border-collapse:collapse;font-size:12px;">
  <thead><tr><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Data</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Predetto</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Reale</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Errore</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">MAE</th></tr></thead>
  <tbody>{"".join(f"<tr><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{b['date']}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{fmt(b['predetto'], ticker)}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{fmt(b['reale'], ticker)}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;font-weight:600;color:{'#ef4444' if b['error_pct']<0 else '#22c55e'};'>{b['error_pct']:+.2f}%</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{abs(b['error_pct']):.2f}%</td></tr>" for b in backtest)}</tbody>
</table>
</div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Contesto</div>
<table width="100%" cellpadding="0" cellspacing="0" style="margin:12px 0;">
  <tr>
    <td width="33%" style="width:33%;min-width:33%;padding:2px;vertical-align:top;">
      <div style="background:#fff;border-radius:8px;padding:10px 4px;text-align:center;border:1px solid #e2e8f0;">
        <div style="font-size:18px;font-weight:700;">{vola:.1f}%</div>
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;margin-top:3px;">Volatilità Annua</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">20gg annualizzata</div>
      </div>
    </td>
    <td width="33%" style="width:33%;min-width:33%;padding:2px;vertical-align:top;">
      <div style="background:#fff;border-radius:8px;padding:10px 4px;text-align:center;border:1px solid #e2e8f0;">
        <div style="font-size:18px;font-weight:700;">{vol_ratio:.2f}x</div>
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;margin-top:3px;">Volume Ratio</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">vs media 10gg</div>
      </div>
    </td>
    <td width="33%" style="width:33%;min-width:33%;padding:2px;vertical-align:top;">
      <div style="background:#fff;border-radius:8px;padding:10px 4px;text-align:center;border:1px solid #e2e8f0;">
        <div style="font-size:18px;font-weight:700;">{bbw:.4f}</div>
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;margin-top:3px;">Bollinger Width</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">Larghezza bande</div>
      </div>
    </td>
  </tr>
</table>

<div style="text-align:center;padding:14px 0;font-size:10px;color:#94a3b8;">
  Kronos Quantitative Research • {now_str}<br>
  Dati: Yahoo Finance • Modello: Kronos-base<br>
  {ticker} • MAE medio: {mae_avg:.2f}% • Tasso EUR/USD: {eur_rate:.4f}
</div>

</div>
</body>
</html>"""

    safe_ticker = ticker.replace("=", "").replace("-", "_")
    output_filename = f"Report_{safe_ticker}.html"
    print(f"7. Salvataggio report...")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   Report salvato: {output_filename}")

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "enomisia974@gmail.com")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_to = os.environ.get("SMTP_TO", "enomisia974@gmail.com")

    if smtp_password:
        msg = MIMEMultipart('related')
        msg.preamble = 'This is a multi-part message in MIME format.'
        msg_alt = MIMEMultipart('alternative')
        msg['From'] = smtp_user
        msg['To'] = smtp_to
        msg['Subject'] = f"[Kronos Daily] {ticker} • Report AI • {now_dmy}"

        body_plain = f"Report Kronos per {ticker} del {now_dmy}. Visualizza con client HTML."
        msg_alt.attach(MIMEText(body_plain, 'plain', 'utf-8'))
        msg_alt.attach(MIMEText(html, 'html', 'utf-8'))
        msg.attach(msg_alt)

        img = MIMEImage(img_bytes, name='chart.png')
        img.add_header('Content-ID', '<chart>')
        img.add_header('Content-Disposition', 'inline')
        msg.attach(img)

        s = smtplib.SMTP(smtp_server, smtp_port)
        s.starttls()
        s.login(smtp_user, smtp_password)
        s.sendmail(smtp_user, [smtp_to], msg.as_string())
        s.quit()
        print(f"   Email inviata a {smtp_to}")
    else:
        print("   SMTP_PASSWORD non impostata, salto invio email.")

if __name__ == "__main__":
    main()
