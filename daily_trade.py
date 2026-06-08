import os, json, warnings, smtplib, logging, argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import torch

from kronos_system.config import ASSETS
from report_complete import (
    genera_html, news_sentiment, train_xgboost, regression_7gg, _currency, ASSET_NAMES,
)
from report_complete import calcola_optimal_trade as calcola_optimal_trade_completo

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("daily_trade")

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

LOOKBACK = 90
PRED_LEN = 14
REPORT_DIR = os.environ.get("REPORT_DIR", "Report_Giornalieri_Btc")

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "enomisia974@gmail.com")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_TO = os.environ.get("SMTP_TO", "enomisia974@gmail.com")


SENT_COLS_UNIFIED = ['sentiment_score', 'sentiment_weighted',
                     'sentiment_positive_ratio', 'sentiment_negative_ratio',
                     'sentiment_neutral_ratio', 'article_count']
SENT_COLS_DAILY = ['sentiment_mean', 'sentiment_weighted',
                   'sentiment_positive_ratio', 'sentiment_negative_ratio',
                   'sentiment_neutral_ratio', 'article_count']

def carica_sentiment_recente(ticker="BTC-EUR"):
    for csv_path, cols in [(f"feature_store/unified_master_{ticker}.csv", SENT_COLS_UNIFIED),
                           (f"feature_store/news_sentiment_{ticker}_daily.csv", SENT_COLS_DAILY)]:
        try:
            df = pd.read_csv(csv_path, parse_dates=['timestamps' if 'unified' in csv_path else 'date'])
            last = df[cols].dropna().iloc[-1]
            return last.values.astype(np.float32)
        except (FileNotFoundError, IndexError, KeyError):
            continue
    return np.zeros(6, dtype=np.float32)


# â”€â”€â”€ DATA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def scarica_dati(ticker="BTC-EUR"):
    df = yf.download(ticker, period="1y", interval="1d")
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.rename(columns={
        'Date': 'timestamps', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df['volume'] = df['volume'].astype(float)
    df['amount'] = df['close'] * df['volume']
    df['timestamps'] = pd.to_datetime(df['timestamps']).dt.tz_localize(None)
    return df


def calcola_indicatori(df):
    df = df.copy()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    d = df['close'].diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.rolling(14).mean()
    al = l.rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + ag / (al + 1e-10)))
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = sma + 2 * std
    df['bb_lower'] = sma - 2 * std
    df['bb_mid'] = sma
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    df['atr_14'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['volume_sma_10'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_10'] + 1e-10)
    df['return_1d'] = df['close'].pct_change()
    df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    df['high_low_pct'] = (df['high'] - df['low']) / df['close']
    return df


# â”€â”€â”€ KRONOS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def estrai_embedding_kronos(tokenizer, model, df, window=90):
    from model.kronos import calc_time_stamps
    price_cols = ['open', 'high', 'low', 'close']
    df = df.reset_index(drop=True)
    x = df[price_cols + ['volume', 'amount']].values.astype(np.float32)
    x_norm = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        local = x[start:i + 1]
        local_mean = np.mean(local, axis=0)
        local_std = np.std(local, axis=0) + 1e-5
        x_norm[i] = (x[i] - local_mean) / local_std
    x = np.clip(x_norm, -5, 5)
    ts = pd.to_datetime(df['timestamps']).dt.tz_localize(None)
    stamp = calc_time_stamps(ts).values.astype(np.float32)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    st = torch.from_numpy(stamp).unsqueeze(0).to(device)
    x_token = tokenizer.encode(xt.clip(-5, 5), half=True)
    with torch.no_grad():
        _, ctx = model.decode_s1(x_token[0], x_token[1], st)
    return ctx[0].cpu().numpy()


def kronos_prediction(df, lookback, pred_len, tokenizer=None, model=None):
    from model import KronosPredictor
    if tokenizer is None or model is None:
        from model import Kronos, KronosTokenizer
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)
    x_df = df.iloc[-lookback:].copy()
    inp = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + timedelta(days=1), periods=pred_len, freq='D'))
    pred = predictor.predict(df=inp, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len)
    return pred, x_df, x_ts, y_ts, tokenizer, model


def backtest_kronos_predizioni(df_tech, tokenizer, model, n_back=14):
    from model import KronosPredictor
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)
    lookback = min(90, len(df_tech) - 1)
    results = []
    for offset in range(n_back, 0, -1):
        cut = len(df_tech) - offset
        if cut < lookback:
            continue
        x_df = df_tech.iloc[cut - lookback:cut].copy()
        inp = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_ts = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
        y_ts = pd.Series([x_ts.iloc[-1] + timedelta(days=1)])
        pred = predictor.predict(df=inp, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=1)
        pred_close = pred['close'].iloc[0]
        actual_close = df_tech.iloc[cut]['close']
        d = df_tech.iloc[cut]['timestamps']
        date = str(d.date()) if hasattr(d, 'date') else str(d)[:10]
        error_pct = (pred_close - actual_close) / actual_close * 100
        results.append({
            'date': date, 'predetto': pred_close,
            'reale': actual_close, 'error_pct': error_pct,
        })
    return results


# â”€â”€â”€ XGBOOST INFERENCE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def carica_modello(ticker="BTC-EUR"):
    import joblib
    base = "feature_store/models"
    model = joblib.load(os.path.join(base, f"xgboost_latest_{ticker}_model.pkl"))
    pca = joblib.load(os.path.join(base, f"xgboost_latest_{ticker}_pca.pkl"))
    info = joblib.load(os.path.join(base, f"xgboost_latest_{ticker}_info.pkl"))
    return model, pca, info["feature_names"], info["metadata"]


def predici_segnale(model, pca, emb_latest, tech_latest, sent_latest):
    emb_pca = pca.transform(emb_latest.reshape(1, -1))
    X = np.concatenate([tech_latest.reshape(1, -1), emb_pca, sent_latest.reshape(1, -1)], axis=1)
    prob = model.predict_proba(X)[0, 1]
    return prob


# â”€â”€â”€ BEST TRADE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def calcola_optimal_trade(pred):
    pred = pred.reset_index()
    idx_entry = pred['low'].idxmin()
    idx_exit = pred.iloc[idx_entry:]['high'].idxmax()
    if idx_entry == idx_exit:
        idx_exit = len(pred) - 1
    entry = pred.iloc[idx_entry]['low']
    exit_ = pred.iloc[idx_exit]['high']
    ret = (exit_ - entry) / entry * 100
    hold = idx_exit - idx_entry
    entry_date = str(pred.iloc[idx_entry]['index'].date()) if hasattr(pred.iloc[idx_entry]['index'], 'date') else str(pred.iloc[idx_entry]['index'])[:10]
    exit_date = str(pred.iloc[idx_exit]['index'].date()) if hasattr(pred.iloc[idx_exit]['index'], 'date') else str(pred.iloc[idx_exit]['index'])[:10]
    return {
        'entry_date': entry_date, 'exit_date': exit_date,
        'entry_price': entry, 'exit_price': exit_,
        'return_pct': ret, 'hold_days': hold,
    }


# ─── EMAIL HTML (template leggero) ─────────────────────────────────

def genera_email_html(ticker, df_oggi, pred, prob, trade, backtest):
    now = datetime.now()
    now_str = now.strftime("%d %B %Y • %H:%M")
    now_dmy = now.strftime("%d/%m/%Y")

    asset_name = ASSET_NAMES.get(ticker, ticker)
    currency = _currency(ticker)

    last = df_oggi.iloc[-1]
    p_now = last['close']
    rsi = last['rsi_14']
    bbw = last['bb_width']
    atr = last['atr_14']
    vr = last['volume_ratio']
    vola = last['volatility_20d'] * 100

    p_30 = df_oggi.iloc[-30]['close']
    ch30 = ((p_now - p_30) / p_30) * 100

    sig = "BUY" if prob > 0.75 else "POSITIVE" if prob > 0.6 else "NEUTRAL" if prob > 0.4 else "CAUTION"

    if rsi > 70:
        rj, rc = "SOVRACOMPRATO", "#ef4444"
    elif rsi < 30:
        rj, rc = "SOVRAVENDUTO", "#22c55e"
    else:
        rj, rc = "NEUTRALE", "#3b82f6"

    p_col = '#22c55e' if prob > 0.75 else '#3b82f6' if prob > 0.6 else '#64748b' if prob > 0.4 else '#f59e0b'

    # Tabella previsioni
    tab_pred = ""
    for idx, row in pred.iterrows():
        d = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
        dl = row['close'] - row['open']
        dc = "#22c55e" if dl >= 0 else "#ef4444"
        ds = "+" if dl >= 0 else ""
        tab_pred += f"<tr><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{d}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{row['open']:,.0f}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{row['high']:,.0f}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{row['low']:,.0f}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;font-weight:600;'>{row['close']:,.0f}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;font-weight:600;color:{dc};'>{ds}{dl:+,.0f}</td></tr>"

    # Tabella backtest
    tab_back = ""
    mae_sum = 0
    mae_n = 0
    if backtest:
        for b in backtest:
            err = b['error_pct']
            mae_sum += abs(err)
            mae_n += 1
            sign = "✓" if abs(err) < 10 else "✗"
            tab_back += f"<tr><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{b['date']}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{b['predetto']:,.0f}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{b['reale']:,.0f}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;font-weight:600;color:{'#ef4444' if err<0 else '#22c55e'};'>{err:+.2f}%</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{sign}</td><td style='padding:7px 6px;border-bottom:1px solid #f1f5f9;color:#334155;'>{abs(err):.2f}%</td></tr>"
        mae_avg = mae_sum / mae_n if mae_n > 0 else 0
    else:
        mae_avg = 0

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{ticker} — Report Giornaliero</title>
</head>
<body style="background:#f4f4f6;color:#1e293b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px;line-height:1.4;margin:0;padding:0;">
<div style="width:100%;padding:0;margin:0;">

<div style="background:linear-gradient(135deg,#0b1120,#1e293b);padding:24px 20px 18px;border-radius:10px 10px 0 0;text-align:center;">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1.2px;">Kronos Daily • {now_dmy} • {now.strftime('%H:%M')}</div>
  <h1 style="font-size:18px;font-weight:700;color:#fff;margin:6px 0 2px;">{asset_name} — Report Giornaliero</h1>
  <div style="font-size:36px;font-weight:800;color:#d4a853;">{p_now:,.0f} <span style="font-size:14px;color:#64748b;font-weight:400;">{currency}</span></div>
</div>

<table width="100%" cellpadding="0" cellspacing="0" style="margin:12px 0;table-layout:fixed;">
  <col style="width:33%;">
  <col style="width:33%;">
  <col style="width:34%;">
  <tr>
    <td style="padding:2px;vertical-align:top;">
      <div style="background:#fff;border-radius:8px;padding:10px 4px;text-align:center;border:1px solid #e2e8f0;">
        <div style="font-size:18px;font-weight:700;color:{p_col};">{sig}</div>
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;margin-top:3px;">Signal XGBoost</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{prob*100:.0f}%</div>
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
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">ATR {atr:,.0f} • Vol {vola:.0f}%</div>
      </div>
    </td>
  </tr>
</table>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Miglior Trade Teorico</div>
<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1px solid #bbf7d0;border-radius:8px;padding:14px 2px;margin-bottom:10px;">
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr>
      <td width="25%" style="width:25%;min-width:25%;text-align:center;padding:0 2px;vertical-align:top;">
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Ingresso</div>
        <div style="font-size:18px;font-weight:700;color:#0f172a;">{trade['entry_price']:,.0f}</div>
        <div style="font-size:11px;color:#64748b;">{trade['entry_date']}</div>
      </td>
      <td width="25%" style="width:25%;min-width:25%;text-align:center;padding:0 2px;vertical-align:top;">
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Uscita</div>
        <div style="font-size:18px;font-weight:700;color:#0f172a;">{trade['exit_price']:,.0f}</div>
        <div style="font-size:11px;color:#64748b;">{trade['exit_date']}</div>
      </td>
      <td width="25%" style="width:25%;min-width:25%;text-align:center;padding:0 2px;vertical-align:top;">
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Rendimento</div>
        <div style="font-size:18px;font-weight:700;color:#22c55e;">+{trade['return_pct']:.2f}%</div>
      </td>
      <td width="25%" style="width:25%;min-width:25%;text-align:center;padding:0 2px;vertical-align:top;">
        <div style="font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Hold</div>
        <div style="font-size:18px;font-weight:700;color:#0f172a;">{trade['hold_days']}g</div>
        <div style="font-size:11px;color:#64748b;">giorni</div>
      </td>
    </tr>
  </table>
</div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Analisi Grafica</div>
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:10px;"><div style="padding:6px;text-align:center;"><img src="cid:chart" style="max-width:100%;height:auto;display:block;" alt="{ticker} Chart"/></div></div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Previsione Dettaglio</div>
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:10px;overflow-x:auto;overflow-y:hidden;">
<table style="width:100%;border-collapse:collapse;font-size:12px;">
  <thead><tr><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Data</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Apertura</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Max</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Min</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Chiusura</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Delta</th></tr></thead>
  <tbody>{tab_pred}</tbody>
</table>
</div>

<div style="font-size:15px;font-weight:700;color:#0f172a;margin:16px 0 8px;padding-bottom:5px;border-bottom:2px solid #d4a853;">Backtest Previsioni</div>
<div style="background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;margin-bottom:10px;overflow-x:auto;overflow-y:hidden;">
<table style="width:100%;border-collapse:collapse;font-size:12px;">
  <thead><tr><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Data</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Predetto</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Reale</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Errore</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">Segno</th><th style="background:#f8fafc;color:#64748b;padding:8px 6px;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.6px;border-bottom:2px solid #e2e8f0;text-align:left;">MAE</th></tr></thead>
  <tbody>{tab_back}</tbody>
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
        <div style="font-size:18px;font-weight:700;">{vr:.2f}x</div>
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
  Dati: Yahoo Finance • Modello: Kronos + XGBoost<br>
  {asset_name} • MAE medio: {mae_avg:.2f}%
</div>

</div>
</body>
</html>"""
    return html


# ─── EMAIL ──────────────────────────────────────────────────────────

def invia_email(ticker, trade, prob, current_price, html_content=None, fig=None):
    sig = "BUY" if prob > 0.75 else "POSITIVE" if prob > 0.6 else "NEUTRAL" if prob > 0.4 else "CAUTION" if prob > 0.25 else "SELL"
    now_str = datetime.now().strftime("%d/%m/%Y")

    msg = MIMEMultipart('related')
    msg.preamble = 'This is a multi-part message in MIME format.'

    msg_alt = MIMEMultipart('alternative')
    msg['From'] = SMTP_USER
    msg['Subject'] = f"[Kronos Daily] {ticker} \u2022 {sig} \u2022 {now_str}"

    recipients = [addr.strip() for addr in SMTP_TO.split(',') if addr.strip()]
    msg['To'] = ', '.join(recipients)

    body_plain = f"""{ticker} \u2022 Report Giornaliero {now_str}

Segnale XGBoost: {sig} ({prob*100:.0f}%)
Prezzo attuale: {current_price:,.0f} EUR

Miglior trade teorico ({trade['hold_days']}g):
  Entrata: {trade['entry_date']} @ {trade['entry_price']:,.0f}
  Uscita:  {trade['exit_date']} @ {trade['exit_price']:,.0f}
  Rendimento: {trade['return_pct']:+.2f}%

--
Kronos Quantitative Research
"""
    msg_alt.attach(MIMEText(body_plain, 'plain', 'utf-8'))

    if html_content:
        import re
        html_clean = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        msg_alt.attach(MIMEText(html_clean, 'html', 'utf-8'))

    msg.attach(msg_alt)

    if fig is not None:
        img_bytes = pio.to_image(fig, format='png', width=1000, height=500, scale=1)
        img = MIMEImage(img_bytes, name='chart.png')
        img.add_header('Content-ID', '<chart>')
        img.add_header('Content-Disposition', 'inline')
        msg.attach(img)

    if not SMTP_PASSWORD:
        logger.warning("   SMTP_PASSWORD non impostata, salto invio email.")
        return False
    try:
        s = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        s.starttls()
        s.login(SMTP_USER, SMTP_PASSWORD)
        s.send_message(msg)
        s.quit()
        logger.info(f"   Email inviata a {len(recipients)} destinatari.")
        return True
    except Exception as e:
        logger.error("   ERRORE email: %s", e)
        return False


# â”€â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def processa_asset(ticker, kronos_tokenizer, kronos_model, send_email=True):
    logger.info("\n" + "=" * 60)
    logger.info(f"KRONOS DAILY TRADE \u2022 {ticker} \u2022 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    logger.info("\n[1] Download dati %s...", ticker)
    df = scarica_dati(ticker)
    df_tech = calcola_indicatori(df)
    logger.info("   %d righe", len(df_tech))

    logger.info("\n[2] Kronos prediction + embeddings...")
    pred, x_df, x_ts, y_ts, _, _ = kronos_prediction(df_tech, LOOKBACK, PRED_LEN, kronos_tokenizer, kronos_model)
    emb = estrai_embedding_kronos(kronos_tokenizer, kronos_model, df_tech.iloc[-len(df_tech):])
    logger.info("   Previsione: %d giorni, embeddings: %dD", len(pred), emb.shape[1])

    logger.info("\n[3] Caricamento XGBoost salvato...")
    xgb_model, pca, feat_names, meta = carica_modello(ticker)
    logger.info("   Modello: %s", meta.get('trained_on', 'N/A'))
    logger.info("   Feature: %d", len(feat_names))
    trade_threshold = meta.get("optimal_threshold", 0.75)
    logger.info("   Soglia ottimale: %.2f", trade_threshold)

    logger.info("\n[4] Predizione segnale...")
    tech_cols = ['ema_5', 'ema_10', 'ema_20', 'rsi_14', 'bb_width', 'atr_14', 'volume_ratio']
    emb_latest = emb[-1]
    tech_latest = df_tech[tech_cols].values[-1]
    sent_latest = carica_sentiment_recente(ticker)
    prob = predici_segnale(xgb_model, pca, emb_latest, tech_latest, sent_latest)
    logger.info("   Probabilit\u00e0 rialzo: %.1f%%", prob*100)
    logger.info("   Sentiment: score=%.4f, articles=%d", sent_latest[0], int(sent_latest[5]))

    logger.info("\n[5] Feature importance (XGBoost locale)...")
    _, _, fi = train_xgboost(df_tech, emb)

    logger.info("\n[6] Retrospettiva walk-forward...")
    reg_df = regression_7gg(df_tech, emb, kronos_tokenizer, kronos_model)
    if len(reg_df) > 0:
        logger.info("   %d predizioni, accuratezza: %.0f%%", len(reg_df), reg_df['correct'].mean() * 100)
    else:
        logger.info("   Dati insufficienti")

    logger.info("\n[7] Migliori trade (Kronos + XGBoost)...")
    best_long, best_short = calcola_optimal_trade_completo(pred)
    logger.info("   Long: +%.2f%% (%dg)", best_long['return_pct'], best_long['hold_days'])
    logger.info("   Short: +%.2f%% (%dg)", best_short['return_pct'], best_short['hold_days'])

    logger.info("\n[8] Sentiment giornaliero...")
    daily = news_sentiment(ticker)

    logger.info("\n[9] Backtest previsioni passate...")
    backtest = backtest_kronos_predizioni(df_tech, kronos_tokenizer, kronos_model, PRED_LEN)
    logger.info("   %d giorni confrontati", len(backtest))

    logger.info("\n[10] Generazione report HTML completo (su disco)...")
    os.makedirs(REPORT_DIR, exist_ok=True)
    html_completo = genera_html(df_tech, x_df, pred, x_ts, y_ts, daily, prob, fi, best_long, best_short, reg_df, ticker=ticker)
    report_file = os.path.join(REPORT_DIR, f"report_{ticker}.html")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_completo)
    logger.info("   Report: %s", report_file)

    logger.info("\n[11] HTML leggero per email...")
    email_html = genera_email_html(ticker, df_tech, pred, prob, best_long, backtest)

    logger.info("\n[12] Grafico statico per email...")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.50, 0.25, 0.25])
    fig.add_trace(go.Candlestick(x=x_ts, open=x_df['open'], high=x_df['high'],
        low=x_df['low'], close=x_df['close'], name=f"{ticker} Storico",
        increasing_line_color='#22c55e', decreasing_line_color='#ef4444'), row=1, col=1)
    fig.add_trace(go.Candlestick(x=y_ts, open=pred['open'], high=pred['high'],
        low=pred['low'], close=pred['close'], name="Previsione",
        increasing_line_color='#3b82f6', decreasing_line_color='#8b5cf6',
        increasing_fillcolor='rgba(59,130,246,0.25)', decreasing_fillcolor='rgba(139,92,246,0.25)'), row=1, col=1)
    colors = ['#22c55e' if x_df['close'].iloc[i] >= x_df['open'].iloc[i] else '#ef4444' for i in range(len(x_df))]
    fig.add_trace(go.Bar(x=x_df['timestamps'], y=x_df['volume'], name='Volume',
        marker_color=colors, opacity=0.4), row=2, col=1)
    rsi_vals = x_df['rsi_14']
    fig.add_trace(go.Scatter(x=x_df['timestamps'], y=rsi_vals, mode='lines',
        name='RSI 14', line=dict(color='#818cf8', width=2), fill='tozeroy',
        fillcolor='rgba(129,140,248,0.15)'), row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(34,197,94,0.08)', line_width=0, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.08)', line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#ef4444', opacity=0.4, row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#22c55e', opacity=0.4, row=3, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#94a3b8', opacity=0.3, row=3, col=1)
    fig.update_layout(template='none', height=500, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=11, color='#334155'),
        hovermode='x unified', legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False)
    for i in range(1, 4):
        fig.update_xaxes(gridcolor='#f1f5f9', zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor='#f1f5f9', zeroline=False, row=i, col=1)
    fig.update_yaxes(title=f'Prezzo ({_currency(ticker)})', row=1, col=1)
    fig.update_yaxes(title='Volume', row=2, col=1)
    fig.update_yaxes(title='RSI', row=3, col=1, range=[0, 100])

    if send_email:
        logger.info("\n[13] Invio email per %s...", ticker)
        invia_email(ticker, best_long, prob, df_tech.iloc[-1]['close'], email_html, fig)

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETATO %s", ticker)
    logger.info("=" * 60)

    return {
        'ticker': ticker, 'prob': prob, 'signal': "BUY" if prob >= trade_threshold else "NEUTRAL",
        'trade': best_long, 'price': df_tech.iloc[-1]['close'],
    }


def main():
    parser = argparse.ArgumentParser(description="Kronos Daily Trade")
    parser.add_argument("--ticker", type=str, default=None, help="Singolo asset (es. AAPL). Default: tutti")
    args = parser.parse_args()

    targets = [args.ticker] if args.ticker else ASSETS

    logger.info("=" * 60)
    logger.info("KRONOS DAILY TRADE â€¢ %s â€¢ %s",
                "Multi-Asset" if not args.ticker else args.ticker,
                datetime.now().strftime('%Y-%m-%d %H:%M'))
    logger.info("Targets: %s", targets)
    logger.info("=" * 60)

    from model import Kronos, KronosTokenizer
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info("Caricamento Kronos (singola istanza per batch)...")
    kronos_tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    kronos_model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    kronos_tokenizer = kronos_tokenizer.to(device)
    kronos_model = kronos_model.to(device)
    kronos_model.eval()

    results = {}
    for ticker in targets:
        try:
            force_email = os.environ.get('FORCE_EMAIL', '0') == '1'
            results[ticker] = processa_asset(ticker, kronos_tokenizer, kronos_model,
                                             send_email=(args.ticker is None) or force_email)
            logger.info("[OK] %s", ticker)
        except Exception as e:
            logger.error("[FAIL] %s: %s", ticker, e)
            results[ticker] = None

    ok = sum(1 for v in results.values() if v is not None)
    logger.info("=" * 60)
    logger.info("COMPLETATO: %d/%d asset", ok, len(targets))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
