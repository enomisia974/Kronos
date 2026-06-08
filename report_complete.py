import os, json, warnings, argparse
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import torch
from sklearn.metrics import accuracy_score

# Modular pipeline imports
from kronos_system.data.database import init_db
from kronos_system.data.ingestion import validate_prices
from kronos_system.features.technical import compute_target, TECH_FEATURES
from kronos_system.features.sentiment import run_sentiment_pipeline
from kronos_system.ml.pca_pipeline import CausalPCA
from kronos_system.config import (
    ASSETS, TARGET_HORIZON_DAYS, TARGET_THRESHOLD_PCT,
    KRONOS_LOOKBACK, KRONOS_PRED_LEN,
)

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

LOOKBACK = KRONOS_LOOKBACK
PRED_LEN = KRONOS_PRED_LEN

ASSET_NAMES = {
    "BTC-EUR": "Bitcoin",
    "ETH-EUR": "Ethereum",
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "SPY": "S&P 500",
    "GLD": "Gold",
}


def _currency(ticker):
    return "EUR"


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
    df['forward_return_pct'] = (df['close'].shift(-3) - df['close']) / df['close'] * 100
    df['target'] = (df['forward_return_pct'] > 2.5).astype(int)
    df['return_1d'] = df['close'].pct_change()
    df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    df['high_low_pct'] = (df['high'] - df['low']) / df['close']
    return df


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
        print("[2] Kronos loading...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)
    x_df = df.iloc[-lookback:].copy()
    inp = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + timedelta(days=1), periods=pred_len, freq='D'))
    pred = predictor.predict(df=inp, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len)
    return pred, x_df, x_ts, y_ts, tokenizer, model


def news_sentiment(ticker="BTC-EUR"):
    print(f"[3] News + FinBERT for {ticker}...")
    daily = run_sentiment_pipeline(ticker)
    if daily.empty:
        print("  No news data available")
    else:
        print(f"  Processed {len(daily)} days of sentiment")
    return daily


def train_xgboost(df, emb):
    tech = TECH_FEATURES
    tech_arr = df[tech].values[-len(emb):]
    close_arr = df['close'].values[-len(emb):]
    # Mask NaN rows: forward return non calcolabile per ultimi TARGET_HORIZON_DAYS
    forward_ret = df['close'].shift(-TARGET_HORIZON_DAYS).values[-len(emb):]
    ok = ~np.isnan(forward_ret)
    tech_arr = tech_arr[ok]
    emb_arr = emb[ok]
    close_arr = close_arr[ok]

    target_arr = compute_target(pd.Series(close_arr), horizon=TARGET_HORIZON_DAYS, threshold_pct=TARGET_THRESHOLD_PCT)

    split = int(len(tech_arr) * 0.8)

    cpca = CausalPCA()
    train_pca = cpca.fit_transform(emb_arr[:split])
    test_pca = cpca.transform(emb_arr[split:])

    train_X = np.concatenate([tech_arr[:split], train_pca], axis=1)
    train_y = target_arr[:split]
    test_X = np.concatenate([tech_arr[split:], test_pca], axis=1)
    test_y = target_arr[split:]

    ok_train = ~np.isnan(train_y)
    ok_test = ~np.isnan(test_y)
    train_X, train_y = train_X[ok_train], train_y[ok_train]
    test_X, test_y = test_X[ok_test], test_y[ok_test]

    scale_pos = (train_y == 0).sum() / max((train_y == 1).sum(), 1)
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.7,
        colsample_bytree=0.7, eval_metric='logloss', verbosity=0, random_state=42,
        scale_pos_weight=scale_pos,
    )
    m.fit(train_X, train_y, eval_set=[(test_X, test_y)], verbose=False)

    # Final prediction on last point
    full_pca = cpca.transform(emb_arr[-1:])
    full_X = np.concatenate([tech_arr[-1:], full_pca], axis=1)
    prob = m.predict_proba(full_X)[0, 1]

    fi = pd.DataFrame({
        'feature': TECH_FEATURES + [f'emb_{i}' for i in range(cpca.get_n_components())],
        'importance': m.feature_importances_,
    })
    return m, prob, fi


def regression_7gg(df, emb, tokenizer=None, model=None):
    tech_cols = TECH_FEATURES
    n = len(emb)
    if 'target' not in df.columns:
        future_close = df['close'].shift(-7)
        forward_return = (future_close - df['close']) / df['close'] * 100
        df = df.copy()
        df['target'] = (forward_return > 2.5).astype(int)
    tech_raw = df[tech_cols].values[-n:]
    close_raw = df['close'].values[-n:]
    ts_raw = pd.to_datetime(df['timestamps'].values[-n:])
    target_raw = df['target'].values[-n:]
    ok = ~np.isnan(target_raw)
    tech_arr = tech_raw[ok]
    close_arr = close_raw[ok]
    ts_arr = ts_raw[ok]
    emb_arr = emb[ok]

    from model import KronosPredictor
    predictor = KronosPredictor(tokenizer=tokenizer, model=model) if tokenizer and model else None

    target_7d = np.zeros(len(tech_arr))
    for j in range(len(tech_arr) - 7):
        pct = (close_arr[j + 7] - close_arr[j]) / close_arr[j] * 100
        target_7d[j] = 1 if pct > 2.5 else 0

    last_7_start = max(0, len(tech_arr) - 21)
    results = []
    import xgboost as xgb

    for i in range(last_7_start, len(tech_arr) - 7):
        if len(emb_arr[:i]) < 20 or len(np.unique(target_7d[:i])) < 2:
            continue

        cpca = CausalPCA()
        train_pca = cpca.fit_transform(emb_arr[:i])
        test_pca = cpca.transform(emb_arr[i:i+1])

        train_X = np.concatenate([tech_arr[:i], train_pca], axis=1)
        train_y = target_7d[:i]
        test_X = np.concatenate([tech_arr[i:i+1], test_pca], axis=1)

        m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, eval_metric='logloss', verbosity=0,
            random_state=42, scale_pos_weight=(train_y==0).sum()/max((train_y==1).sum(),1))
        m.fit(train_X, train_y, verbose=False)

        prob = m.predict_proba(test_X)[0, 1]
        close_now = close_arr[i]
        close_future = close_arr[i + 7]
        actual_pct = (close_future - close_now) / close_now * 100

        kronos_predicted = None
        if predictor is not None and i >= 60:
            lookback = min(90, i)
            slice_start = i - lookback + (len(df) - n)
            slice_end = i + (len(df) - n) + 1
            x_inp = df.iloc[slice_start:slice_end][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            x_ts_local = pd.Series(pd.to_datetime(df['timestamps'].iloc[slice_start:slice_end]).dt.tz_localize(None))
            y_ts_local = pd.Series(pd.date_range(start=x_ts_local.iloc[-1] + timedelta(days=1), periods=7, freq='D'))
            pred_k = predictor.predict(df=x_inp, x_timestamp=x_ts_local, y_timestamp=y_ts_local, pred_len=7)
            kronos_predicted = float(pred_k['close'].iloc[6]) if len(pred_k) > 6 else None

        results.append({
            'date': str(ts_arr[i])[:10],
            'close': close_now,
            'kronos_predicted': kronos_predicted,
            'close_future': close_future,
            'actual_pct': round(actual_pct, 2),
            'prob': prob,
            'actual': int(target_7d[i]),
            'correct': (prob > 0.5) == bool(target_7d[i]),
        })

    return pd.DataFrame(results)


# ─── GRAFICI ─────────────────────────────────────────────

def grafico_principale(x_df, pred, x_ts, y_ts, ticker="BTC-EUR"):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.40, 0.18, 0.22, 0.20],
                        subplot_titles=("", "", "", ""))
    # Candele
    fig.add_trace(go.Candlestick(x=x_ts, open=x_df['open'], high=x_df['high'], low=x_df['low'],
        close=x_df['close'], name=f"{ticker} Storico", increasing_line_color='#22c55e', decreasing_line_color='#ef4444'), row=1, col=1)
    fig.add_trace(go.Candlestick(x=y_ts, open=pred['open'], high=pred['high'], low=pred['low'],
        close=pred['close'], name="Kronos Previsione", increasing_line_color='#3b82f6', decreasing_line_color='#8b5cf6',
        increasing_fillcolor='rgba(59,130,246,0.25)', decreasing_fillcolor='rgba(139,92,246,0.25)'), row=1, col=1)
    # Bande di Bollinger
    bb_x = x_df['timestamps']
    fig.add_trace(go.Scatter(x=bb_x, y=x_df['bb_upper'], mode='lines', name='BB Upper',
        line=dict(color='rgba(99,102,241,0.3)', width=1), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_x, y=x_df['bb_lower'], mode='lines', name='BB Lower',
        line=dict(color='rgba(99,102,241,0.3)', width=1), fill='tonexty',
        fillcolor='rgba(99,102,241,0.06)', showlegend=True), row=1, col=1)
    # EMA
    fig.add_trace(go.Scatter(x=bb_x, y=x_df['ema_5'], mode='lines', name='EMA 5',
        line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_x, y=x_df['ema_20'], mode='lines', name='EMA 20',
        line=dict(color='#ec4899', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb_x, y=x_df['sma_50'], mode='lines', name='SMA 50',
        line=dict(color='#64748b', width=1, dash='dot')), row=1, col=1)

    # Volume
    colors = ['#22c55e' if x_df['close'].iloc[i] >= x_df['open'].iloc[i] else '#ef4444' for i in range(len(x_df))]
    fig.add_trace(go.Bar(x=x_df['timestamps'], y=x_df['volume'], name='Volume',
        marker_color=colors, opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_df['timestamps'], y=x_df['volume_sma_10'], mode='lines',
        name='Media Volume', line=dict(color='#3b82f6', width=1.5)), row=2, col=1)

    # RSI
    rsi = x_df['rsi_14']
    rsi_x = x_df['timestamps']
    fig.add_trace(go.Scatter(x=rsi_x, y=rsi, mode='lines', name='RSI 14',
        line=dict(color='#818cf8', width=2), fill='tozeroy', fillcolor='rgba(129,140,248,0.15)'), row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(34,197,94,0.08)', line_width=0, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.08)', line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#ef4444', opacity=0.4, row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#22c55e', opacity=0.4, row=3, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#94a3b8', opacity=0.3, row=3, col=1)

    # ATR + Volatilita
    fig.add_trace(go.Scatter(x=x_df['timestamps'], y=x_df['atr_14'], mode='lines', name='ATR 14',
        line=dict(color='#f59e0b', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_df['timestamps'], y=x_df['volatility_20d']*100, mode='lines',
        name='Vol. Annua %', line=dict(color='#a78bfa', width=1.5, dash='dot')), row=4, col=1)
    fig.add_trace(go.Bar(x=x_df['timestamps'], y=x_df['high_low_pct']*100, name='Range %%',
        marker_color='#cbd5e1', opacity=0.3), row=4, col=1)

    fig.update_layout(
        template='none', height=900, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=11, color='#334155'),
        hovermode='x unified', legend=dict(orientation='h', y=1.12, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor='#f1f5f9', zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor='#f1f5f9', zeroline=False, row=i, col=1)
    fig.update_yaxes(title=f'Prezzo ({_currency(ticker)})', row=1, col=1)
    fig.update_yaxes(title='Volume', row=2, col=1)
    fig.update_yaxes(title='RSI', row=3, col=1, range=[0, 100])
    fig.update_yaxes(title='ATR / Vol', row=4, col=1)
    fig.update_xaxes(title='', row=1, col=1)
    fig.update_xaxes(title='', row=2, col=1)
    fig.update_xaxes(title='', row=3, col=1)
    fig.update_xaxes(title='Data', row=4, col=1)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def grafico_sentiment(daily):
    fig = go.Figure()
    pos = daily[daily['score'] >= 0]
    neg = daily[daily['score'] < 0]
    fig.add_trace(go.Bar(x=pos['date'], y=pos['score'], name='Positivo',
        marker_color='#22c55e', opacity=0.8))
    fig.add_trace(go.Bar(x=neg['date'], y=neg['score'], name='Negativo',
        marker_color='#ef4444', opacity=0.8))
    fig.add_trace(go.Scatter(x=daily['date'], y=daily['pos_ratio'], mode='lines+markers',
        name='% Positivo', line=dict(color='#3b82f6', width=2), yaxis='y2'))
    fig.update_layout(
        template='none', height=280, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=11, color='#334155'),
        hovermode='x unified', legend=dict(orientation='h', y=1.1, x=0),
        yaxis=dict(title='Sentiment Score', gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#e2e8f0'),
        yaxis2=dict(title='% Positivo', overlaying='y', side='right', range=[0, 1],
                    gridcolor='#f1f5f9', tickformat='.0%'),
        xaxis=dict(gridcolor='#f1f5f9'),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def grafico_feature_importance(fi):
    top20 = fi.sort_values('importance', ascending=True).tail(20)
    colors = ['#3b82f6' if 'emb' in f else '#f59e0b' if f in ['rsi_14','atr_14','bb_width'] else '#64748b' for f in top20['feature']]
    fig = go.Figure(go.Bar(
        x=top20['importance'], y=top20['feature'],
        orientation='h', marker_color=colors, marker_line_width=0,
        text=np.round(top20['importance'], 4), textposition='outside',
    ))
    fig.update_layout(
        template='none', height=400, margin=dict(l=10,r=60,t=10,b=10),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=11, color='#334155'),
        xaxis=dict(title='Importanza', gridcolor='#f1f5f9', zeroline=False),
        yaxis=dict(title='', gridcolor='#f1f5f9'),
        hovermode='y',
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def calcola_optimal_trade(pred):
    """Simula il miglior timing teorico: compra al lowest low, vendi al highest high.
    Mostra il massimo range atteso della previsione — non un segnale eseguibile."""
    pred = pred.reset_index()
    idx_entry = pred['low'].idxmin()
    idx_exit = pred['high'].idxmax()
    if idx_entry > idx_exit:
        # Se il minimo è dopo il massimo, inverti: compra al primo minimo, vendi al massimo successivo
        idx_exit = pred.iloc[idx_entry:]['high'].idxmax()
    if idx_entry == idx_exit:
        idx_exit = len(pred) - 1

    entry = pred.iloc[idx_entry]['low']
    exit_ = pred.iloc[idx_exit]['high']
    ret = (exit_ - entry) / entry * 100
    hold = idx_exit - idx_entry
    entry_date = str(pred.iloc[idx_entry]['index'].date()) if hasattr(pred.iloc[idx_entry]['index'], 'date') else str(pred.iloc[idx_entry]['index'])[:10]
    exit_date = str(pred.iloc[idx_exit]['index'].date()) if hasattr(pred.iloc[idx_exit]['index'], 'date') else str(pred.iloc[idx_exit]['index'])[:10]

    # Long: buy low, sell high
    long_trade = {
        'entry_date': entry_date,
        'exit_date': exit_date,
        'entry_price': entry,
        'exit_price': exit_,
        'return_pct': ret,
        'hold_days': hold,
    }
    # Short: sell high, buy low (invert)
    short_trade = {
        'entry_date': exit_date,
        'exit_date': entry_date,
        'entry_price': exit_,
        'exit_price': entry,
        'return_pct': ret,
        'hold_days': hold,
    }
    return long_trade, short_trade


def grafico_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=prob * 100,
        title=dict(text='Prob. Rialzo 3gg', font=dict(size=16, color='#334155')),
        number=dict(suffix='%', font=dict(size=36, color='#0f172a')),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor='#94a3b8'),
            bar=dict(color='#3b82f6', thickness=0.3),
            bgcolor='white',
            borderwidth=0,
            steps=[
                dict(range=[0, 25], color='rgba(239,68,68,0.15)'),
                dict(range=[25, 45], color='rgba(245,158,11,0.10)'),
                dict(range=[45, 55], color='rgba(148,163,184,0.08)'),
                dict(range=[55, 75], color='rgba(59,130,246,0.10)'),
                dict(range=[75, 100], color='rgba(34,197,94,0.15)'),
            ],
            threshold=dict(line=dict(color='#0f172a', width=4), thickness=0.75, value=prob*100),
        ),
    ))
    fig.update_layout(
        template='none', height=260, margin=dict(l=30,r=30,t=40,b=10),
        paper_bgcolor='white', font=dict(family='Inter, sans-serif'),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def grafico_regressione(x_df, reg_df, ticker="BTC-EUR"):
    if reg_df.empty or x_df is None:
        return "", 0, 0

    acc = reg_df['correct'].mean() * 100
    confident = reg_df[reg_df['prob'] > 0.6]['correct'].mean() * 100 if len(reg_df[reg_df['prob'] > 0.6]) > 0 else 0

    # Last 25 days for candle context
    end = x_df['timestamps'].max()
    start = end - timedelta(days=30)
    sub = x_df[x_df['timestamps'] >= start].copy()
    if len(sub) < 5:
        return "", acc, confident

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=sub['timestamps'], open=sub['open'], high=sub['high'],
        low=sub['low'], close=sub['close'],
        name=ticker, increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444',
    ))

    # Prediction connectors
    for _, r in reg_df.iterrows():
        pred_date = pd.to_datetime(r['date'])
        target_date = pred_date + timedelta(days=7)

        sp = sub[sub['timestamps'].dt.date == pred_date.date()]
        ep = sub[sub['timestamps'].dt.date == target_date.date()]
        if sp.empty or ep.empty:
            continue

        st = sp['timestamps'].iloc[0]
        et = ep['timestamps'].iloc[0]
        close_pred = sp['close'].iloc[0]
        close_actual = ep['close'].iloc[0]

        c = '#22c55e' if r['correct'] else '#ef4444'

        # Connector line from prediction candle to outcome candle
        fig.add_trace(go.Scatter(
            x=[st, et], y=[close_pred, close_actual],
            mode='lines+markers',
            line=dict(color=c, width=2, dash='dot'),
            marker=dict(size=[8, 10], color=[c, c],
                       symbol=['circle', 'diamond']),
            name=f"{r['date']} \u2192 +7gg",
            showlegend=False,
            text=f"Previsto: {r['prob']*100:.0f}% | Reale: {r['actual_pct']:+.2f}%",
            hoverinfo='text',
        ))

        # Probability label above prediction dot
        fig.add_annotation(x=st, y=close_pred,
            text=f"<b>{r['prob']*100:.0f}%</b>",
            showarrow=True, arrowhead=0, arrowsize=1,
            ax=0, ay=-35, font=dict(size=11, color=c, weight=600),
            bgcolor='rgba(255,255,255,0.85)', bordercolor=c,
            borderwidth=1, borderpad=4,
        )

        # Checkmark/X label on outcome dot
        icon = "\u2713" if r['correct'] else "\u2717"
        fig.add_annotation(x=et, y=close_actual,
            text=icon, showarrow=False,
            font=dict(size=16, color=c, weight=700),
        )

    # Legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#22c55e', symbol='circle'),
        name='Previsto correttamente \u2713'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#ef4444', symbol='circle'),
        name='Previsto male \u2717'))

    fig.update_layout(
        template='none', height=450, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=12, color='#334155'),
        hovermode='x unified',
        xaxis=dict(gridcolor='#f1f5f9', zeroline=False, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor='#f1f5f9', title=f'Prezzo {ticker} ({_currency(ticker)})',
                   tickformat=','),
        legend=dict(orientation='h', y=1.15, x=0, font=dict(size=11)),
    )
    chart = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return chart, acc, confident


# ─── REPORT HTML ────────────────────────────────────────

def genera_html(df_oggi, x_df, pred, x_ts, y_ts, daily, prob, fi, best_long, best_short, reg_df, ticker="BTC-EUR", email_chart_html=None):
    last = df_oggi.iloc[-1]
    now = datetime.now().strftime("%d %B %Y \u2022 %H:%M")
    p_now = last['close']
    p_old = df_oggi.iloc[-30]['close']
    ch30 = ((p_now - p_old) / p_old) * 100
    rsi = last['rsi_14']
    bbw = last['bb_width']
    atr = last['atr_14']
    vr = last['volume_ratio']
    vola = last['volatility_20d'] * 100

    k_o = pred['open'].iloc[0]
    k_c = pred['close'].iloc[-1]
    k_ch = ((k_c - k_o) / k_o) * 100
    k_hi = pred['high'].max()
    k_lo = pred['low'].min()

    if len(daily) > 0:
        s = daily.iloc[-1]
        s_s = s['score']
        s_l = "Positivo" if s_s > 0.05 else ("Negativo" if s_s < -0.05 else "Neutrale")
        s_n = int(s['count'])
    else:
        s_s, s_l, s_n = 0, "N/D", 0

    p_col = '#22c55e' if prob > 0.75 else '#3b82f6' if prob > 0.6 else '#64748b' if prob > 0.4 else '#f59e0b' if prob > 0.25 else '#ef4444'
    jdg = "BUY SIGNAL" if prob > 0.75 else "POSITIVE BIAS" if prob > 0.6 else "NEUTRAL" if prob > 0.4 else "CAUTION" if prob > 0.25 else "SELL SIGNAL"
    desc = {
        "BUY SIGNAL": "Configurazione statisticamente favorevole",
        "POSITIVE BIAS": "Leggera inclinazione positiva, attendere conferma",
        "NEUTRAL": "Segnali contrastanti, mercato in equilibrio",
        "CAUTION": "Rischio ribasso percepito",
        "SELL SIGNAL": "Configurazione statisticamente sfavorevole"
    }[jdg]

    if rsi > 70: rj, rc = "SOVRACOMPRATO", "#ef4444"
    elif rsi < 30: rj, rc = "SOVRAVENDUTO", "#22c55e"
    else: rj, rc = "NEUTRALE", "#3b82f6"

    kt, kc = ("BULLISH", "#22c55e") if k_ch > 2 else ("POSITIVO", "#3b82f6") if k_ch > 0 else ("NEGATIVO", "#f59e0b") if k_ch > -2 else ("BEARISH", "#ef4444")

    asset_name = ASSET_NAMES.get(ticker, ticker)
    currency = _currency(ticker)

    ch_main = email_chart_html if email_chart_html else grafico_principale(x_df, pred, x_ts, y_ts, ticker=ticker)
    ch_sent = grafico_sentiment(daily) if len(daily) > 0 else ""
    ch_fi = grafico_feature_importance(fi)
    ch_gauge = grafico_gauge(prob)

    tab_pred = ""
    for idx, row in pred.iterrows():
        d = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
        dl = row['close'] - row['open']
        dc = "#22c55e" if dl >= 0 else "#ef4444"
        ds = "+" if dl >= 0 else ""
        tab_pred += f"<tr><td style='padding:10px 14px;border-bottom:1px solid #f1f5f9;color:#0f172a;font-weight:500;'>{d}</td><td style='padding:10px 14px;border-bottom:1px solid #f1f5f9;'>{row['open']:,.0f}</td><td style='padding:10px 14px;border-bottom:1px solid #f1f5f9;color:#22c55e;'>{row['high']:,.0f}</td><td style='padding:10px 14px;border-bottom:1px solid #f1f5f9;color:#ef4444;'>{row['low']:,.0f}</td><td style='padding:10px 14px;border-bottom:1px solid #f1f5f9;font-weight:600;'>{row['close']:,.0f}</td><td style='padding:10px 14px;border-bottom:1px solid #f1f5f9;color:{dc};font-weight:600;'>{ds}{dl:+,.0f}</td></tr>"

    tab_news = ""
    if len(daily) > 0:
        for _, r in daily.tail(7).iterrows():
            em = "\U0001f7e2" if r['score'] > 0.05 else ("\U0001f534" if r['score'] < -0.05 else "\u26aa")
            tab_news += f"<tr><td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;'>{str(r['date'])[:10]}</td><td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;'>{em} {r['score']:+.3f}</td><td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;'>{int(r['count'])}</td><td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;'>{r['pos_ratio']:.0%} / {r['neg_ratio']:.0%}</td></tr>"

    # Top features per tipo
    fi_grp = fi.copy()
    fi_grp['group'] = fi_grp['feature'].apply(lambda x: 'emb' if 'emb' in x else 'tech')
    grp_imp = fi_grp.groupby('group')['importance'].sum()
    emb_pct = grp_imp.get('emb', 0) / grp_imp.sum() * 100
    tech_pct = grp_imp.get('tech', 0) / grp_imp.sum() * 100

    retros_ok = len(reg_df) > 0 and reg_df['correct'].sum() > 0
    if retros_ok:
        reg_html, acc, confident = grafico_regressione(x_df, reg_df, ticker=ticker)
        tab_reg = ""
        for _, r in reg_df.iterrows():
            bar_w = min(r['prob'] * 100, 100)
            bar_c = "#22c55e" if r['prob'] > 0.5 else "#ef4444"
            var_c = "#22c55e" if r['actual_pct'] > 0 else "#ef4444"
            var_s = "+" if r['actual_pct'] > 0 else ""
            icon = "\U0001f7e2" if r['correct'] else "\U0001f534"
            tab_reg += f"""<tr>
<td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;white-space:nowrap;'>{r['date']}</td>
<td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;font-weight:600;'>{r['close']:,.0f}</td>
<td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;font-weight:600;color:#8b5cf6;'>{r['kronos_predicted']:,.0f}</td>
<td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;font-weight:600;'>{r['close_future']:,.0f}</td>
<td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;font-weight:600;color:{var_c};'>{var_s}{r['actual_pct']:.2f}%</td>
<td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;'><div style='background:#f1f5f9;border-radius:4px;height:18px;width:100px;overflow:hidden;'><div style='height:100%;width:{bar_w:.0f}%;background:{bar_c};border-radius:4px;'></div></div></td>
<td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;font-weight:600;'>{r['prob']*100:.0f}%</td>
<td style='padding:8px 12px;border-bottom:1px solid #f1f5f9;'>{icon}</td>
</tr>"""
    else:
        reg_html = ""
        tab_reg = ""
        acc = 0
        confident = 0

    retros_section = f"""
    <div class="sec-t"><span>Retrospettiva: cosa diceva il modello 7 giorni fa?</span> <span class="tag">WALK-FORWARD</span></div>
    <div class="card" style="margin-top:0;">
      <div class="card-hd"><h2>Candele + Previsioni: la freccia parte dal giorno della predizione e arriva al risultato 7 giorni dopo</h2></div>
      <div class="card-bd" style="padding:8px;">{reg_html if reg_html else '<div style="padding:30px;text-align:center;color:#94a3b8;">Dati insufficienti per la retrospettiva</div>'}</div>
      <div style="overflow-x:auto;border-top:1px solid #f1f5f9;">
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead><tr style="background:#f8fafc;"><th style="padding:10px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;">Data</th><th style="padding:10px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;">Prezzo</th><th style="padding:10px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;">Prezzo Predetto</th><th style="padding:10px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;">Prezzo +7gg</th><th style="padding:10px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;">Variazione</th><th style="padding:10px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;">Confidenza XGBoost</th><th style="padding:10px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;">Prob.</th><th style="padding:10px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;">Accurato?</th></tr></thead>
        <tbody>{tab_reg if tab_reg else '<tr><td colspan="8" style="text-align:center;color:#94a3b8;padding:20px;">Nessun dato</td></tr>'}</tbody>
      </table>
      </div>
      {f'<div style="padding:12px 16px;border-top:1px solid #f1f5f9;font-size:13px;color:#475569;">Accuratezza complessiva: <strong>{acc:.0f}%</strong> &middot; Predizioni sicure (&gt;60%): <strong>{confident:.0f}%</strong></div>' if retros_ok else ''}
    </div>
    """ if retros_ok else ""

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{ticker} \u2022 Quantitative Research Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&family=Playfair+Display:wght@700;900&display=swap" rel="stylesheet">
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#f8f7f4;color:#1e293b;font-family:'Inter',-apple-system,sans-serif;}}
  .hd{{background:linear-gradient(135deg,#0b1120 0%,#162032 50%,#1e293b 100%);padding:48px 0 36px;border-bottom:3px solid #d4a853;}}
  .hd-in{{max-width:1280px;margin:0 auto;padding:0 32px;}}
  .hd-in .meta{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;}}
  .hd-in h1{{font-family:'Playfair Display',serif;font-weight:900;font-size:44px;color:#fff;letter-spacing:-1.5px;line-height:1.1;}}
  .hd-in .sub{{font-size:16px;color:#94a3b8;margin-top:6px;font-weight:300;}}
  .price-hero{{font-family:'Playfair Display',serif;font-size:64px;font-weight:700;color:#d4a853;margin-top:16px;line-height:1;}}
  .price-hero span{{font-size:20px;color:#64748b;font-weight:400;}}
  .price-info{{font-size:14px;margin-top:6px;}}
  .cnt{{max-width:1280px;margin:0 auto;padding:28px 32px;}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
  .card{{background:#fff;border-radius:6px;border:1px solid #e2e8f0;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.03);}}
  .card-hd{{padding:16px 20px;border-bottom:1px solid #f1f5f9;background:#fafbfc;}}
  .card-hd h2{{font-family:'Playfair Display',serif;font-size:16px;font-weight:700;color:#0f172a;}}
  .card-bd{{padding:16px 20px;}}
  .metric{{text-align:center;padding:18px 12px;}}
  .metric .val{{font-family:'Playfair Display',serif;font-size:30px;font-weight:700;}}
  .metric .lbl{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-top:4px;}}
  .metric .dsc{{font-size:12px;color:#94a3b8;margin-top:6px;}}
  .signal{{padding:24px;border-radius:6px;text-align:center;border:2px solid;}}
  .signal .st{{font-family:'Playfair Display',serif;font-size:28px;font-weight:700;letter-spacing:-0.5px;}}
  .signal .sp{{font-size:52px;font-weight:800;margin-top:6px;}}
  .signal .sd{{font-size:14px;margin-top:8px;opacity:0.85;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th{{color:#64748b;padding:10px 14px;font-weight:600;border-bottom:2px solid #e2e8f0;text-transform:uppercase;letter-spacing:0.8px;font-size:10px;text-align:left;}}
  td{{padding:8px 14px;}}
  .sec-t{{font-family:'Playfair Display',serif;font-size:20px;font-weight:700;color:#0f172a;margin:28px 0 14px;padding-bottom:8px;border-bottom:2px solid #d4a853;display:flex;align-items:center;gap:10px;}}
  .sec-t .tag{{font-size:11px;background:#d4a853;color:#0b1120;padding:2px 10px;border-radius:12px;font-weight:600;font-family:'Inter',sans-serif;}}
  .divider{{height:1px;background:linear-gradient(to right,transparent,#d4a853,transparent);margin:28px 0;}}
  .ftr{{max-width:1280px;margin:0 auto;padding:24px 32px;font-size:11px;color:#94a3b8;border-top:1px solid #e2e8f0;text-align:center;}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:10px;font-weight:600;}}
  .chip{{display:inline-block;padding:4px 12px;border-radius:16px;font-size:12px;font-weight:600;background:#f1f5f9;color:#475569;}}
  .sim-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:500px;}}
  .sim-in label{{display:block;font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;}}
  .sim-in input{{width:100%;padding:10px 14px;border:1px solid #e2e8f0;border-radius:6px;font-size:18px;font-weight:600;color:#0f172a;font-family:'Inter',sans-serif;outline:none;}}
  .sim-in input:focus{{border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,0.15);}}
  .sim-ris{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:20px;padding-top:16px;border-top:1px solid #f1f5f9;}}
  .sim-r{{text-align:center;}}
  .sim-r-lbl{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;}}
  .sim-r-val{{font-size:20px;font-weight:700;color:#0f172a;margin-top:4px;}}
  @media(max-width:768px){{.grid-3,.grid-2,.sim-grid,.sim-ris{{grid-template-columns:1fr;}}.hd-in h1{{font-size:28px;}}.price-hero{{font-size:40px;}}}}
</style>
</head>
<body>

<div class="hd">
  <div class="hd-in">
    <div class="meta">Kronos Quantitative Research \u2022 {now}</div>
    <h1>{asset_name} Market Outlook</h1>
    <div class="sub">Analisi multimodale: Kronos AI + FinBERT Sentiment + XGBoost Meta-Learning</div>
    <div class="price-hero">{p_now:,.0f} <span>{currency}</span></div>
    <div class="price-info" style="color:{'#22c55e' if ch30 >=0 else '#ef4444'}">
      <span style="font-weight:600;">{ch30:+.2f}%</span> (30gg) &middot; ATR {atr:,.0f} &middot; RSI {rsi:.0f} &middot; Vol. Annua {vola:.1f}%
    </div>
  </div>
</div>

<div class="cnt">

  <!-- Gauge + 3 Metric Cards Row -->
  <div class="grid-3">
    <div class="card">
      <div class="metric">
        <div class="val" style="color:{kc};">{kt}</div>
        <div class="lbl">Kronos AI &mdash; 14gg</div>
        <div class="dsc">{k_ch:+.2f}% &middot; High {k_hi:,.0f} &middot; Low {k_lo:,.0f}</div>
      </div>
    </div>
    <div class="card">
      <div class="metric">
        <div class="val" style="color:{'#22c55e' if s_s>0.05 else '#ef4444' if s_s<-0.05 else '#64748b'};">{s_l}</div>
        <div class="lbl">News Sentiment (FinBERT)</div>
        <div class="dsc">{s_s:+.3f} &middot; {s_n} articoli oggi</div>
      </div>
    </div>
    <div class="card">
      <div class="metric">
        <div class="val" style="color:{p_col};">{jdg}</div>
        <div class="lbl">XGBoost Meta-Signal</div>
        <div class="dsc">{prob:.1%} prob. rialzo +2.5% in 3gg</div>
      </div>
    </div>
  </div>

  <!-- Signal Box + Gauge -->
  <div class="grid-2" style="margin-top:20px;">
    <div class="card">
      <div class="signal" style="background:{p_col}08;border-color:{p_col};height:100%;display:flex;flex-direction:column;justify-content:center;">
        <div class="st" style="color:{p_col};">{jdg}</div>
        <div class="sp" style="color:{p_col};">{prob*100:.0f}%</div>
        <div class="sd" style="color:#475569;">{desc} &middot; Il meta-modello assegna probabilit&agrave; di rialzo del {prob*100:.0f}% nei prossimi 3 giorni.</div>
      </div>
    </div>
    <div class="card">
      {ch_gauge}
    </div>
  </div>

  <!-- Trading Idea -->
  <div class="sec-t"><span>Migliori Opportunit&agrave; di Trading</span> <span class="tag">KRONOS + XGBOOST</span></div>
  <div class="grid-2" style="margin-top:0;">
    <div class="card" style="border-left:4px solid #22c55e;">
      <div class="card-hd" style="background:#f0fdf4;"><h2 style="color:#15803d;">&uarr; Long &mdash; Migliore Acquisto/Vendita</h2></div>
      <div class="card-bd">
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
          <div><div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Ingresso</div><div style="font-size:22px;font-weight:700;color:#0f172a;margin-top:4px;">{best_long['entry_price']:,.0f}</div><div style="font-size:13px;color:#64748b;">{best_long['entry_date']}</div></div>
          <div><div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Uscita</div><div style="font-size:22px;font-weight:700;color:#0f172a;margin-top:4px;">{best_long['exit_price']:,.0f}</div><div style="font-size:13px;color:#64748b;">{best_long['exit_date']}</div></div>
          <div><div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Rendimento</div><div style="font-size:22px;font-weight:700;color:#22c55e;margin-top:4px;">+{best_long['return_pct']:.2f}%</div></div>
          <div><div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Hold</div><div style="font-size:22px;font-weight:700;color:#0f172a;margin-top:4px;">{best_long['hold_days']}g</div><div style="font-size:13px;color:#64748b;">giorni</div></div>
        </div>
      </div>
    </div>
    <div class="card" style="border-left:4px solid #ef4444;">
      <div class="card-hd" style="background:#fef2f2;"><h2 style="color:#b91c1c;">&darr; Short &mdash; Migliore Vendita/Riacquisto</h2></div>
      <div class="card-bd">
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
          <div><div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Vendita</div><div style="font-size:22px;font-weight:700;color:#0f172a;margin-top:4px;">{best_short['entry_price']:,.0f}</div><div style="font-size:13px;color:#64748b;">{best_short['entry_date']}</div></div>
          <div><div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Riacquisto</div><div style="font-size:22px;font-weight:700;color:#0f172a;margin-top:4px;">{best_short['exit_price']:,.0f}</div><div style="font-size:13px;color:#64748b;">{best_short['exit_date']}</div></div>
          <div><div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Rendimento</div><div style="font-size:22px;font-weight:700;color:#ef4444;margin-top:4px;">{best_short['return_pct']:.2f}%</div></div>
          <div><div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Hold</div><div style="font-size:22px;font-weight:700;color:#0f172a;margin-top:4px;">{best_short['hold_days']}g</div><div style="font-size:13px;color:#64748b;">giorni</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Simulatore Investimento -->
  <div class="sec-t"><span>Calcolatore Rendimento</span> <span class="tag">SIMULATORE</span></div>
  <div class="card">
    <div class="card-hd"><h2>Simula il tuo investimento sulla migliore opportunit&agrave; rilevata</h2></div>
    <div class="card-bd">
      <div class="sim-grid">
        <div class="sim-in">
          <label>Capitale investito ({currency})</label>
          <input type="number" id="capitale" value="1000" step="100" min="0" oninput="calcolaSim()">
        </div>
        <div class="sim-in">
          <label>Commissioni totali ({currency})</label>
          <input type="number" id="commissioni" value="0" step="0.5" min="0" oninput="calcolaSim()">
        </div>
      </div>
      <div class="sim-ris" id="sim-risultati">
        <div class="sim-r">
          <div class="sim-r-lbl">Prezzo ingresso</div>
          <div class="sim-r-val" id="sim-entry">{best_long['entry_price']:,.0f}</div>
        </div>
        <div class="sim-r">
          <div class="sim-r-lbl">Prezzo uscita</div>
          <div class="sim-r-val" id="sim-exit">{best_long['exit_price']:,.0f}</div>
        </div>
        <div class="sim-r">
          <div class="sim-r-lbl">Rendimento lordo</div>
          <div class="sim-r-val" id="sim-gross-pct" style="color:#22c55e;">+{best_long['return_pct']:.2f}%</div>
        </div>
        <div class="sim-r">
          <div class="sim-r-lbl">Profitto lordo</div>
          <div class="sim-r-val" id="sim-gross-eur" style="color:#22c55e;">{currency} <span id="gross-eur-val">{(1000 * best_long['return_pct'] / 100):.2f}</span></div>
        </div>
        <div class="sim-r" style="border-top:2px solid #e2e8f0;padding-top:12px;">
          <div class="sim-r-lbl" style="font-weight:700;">Profitto netto</div>
          <div class="sim-r-val" id="sim-net-eur" style="font-size:24px;font-weight:700;color:#22c55e;">{currency} <span id="net-eur-val">{(1000 * best_long['return_pct'] / 100):.2f}</span></div>
        </div>
        <div class="sim-r" style="border-top:2px solid #e2e8f0;padding-top:12px;">
          <div class="sim-r-lbl" style="font-weight:700;">Rendimento netto</div>
          <div class="sim-r-val" id="sim-net-pct" style="font-size:24px;font-weight:700;color:#22c55e;"><span id="net-pct-val">{best_long['return_pct']:.2f}</span>%</div>
        </div>
      </div>
      <div style="font-size:12px;color:#94a3b8;margin-top:12px;text-align:center;">
        Operativa: {best_long['entry_date']} &rarr; {best_long['exit_date']} &middot; {best_long['hold_days']} giorni hold
      </div>
    </div>
  </div>

  <script>
  function calcolaSim() {{
    var cap = parseFloat(document.getElementById('capitale').value) || 0;
    var fee = parseFloat(document.getElementById('commissioni').value) || 0;
    var entry = {best_long['entry_price']};
    var exit = {best_long['exit_price']};
    var grossPct = {best_long['return_pct']};
    var grossEur = cap * grossPct / 100;
    var netEur = grossEur - fee;
    var netPct = cap > 0 ? (netEur / cap * 100) : 0;
    document.getElementById('gross-eur-val').textContent = grossEur.toFixed(2);
    document.getElementById('net-eur-val').textContent = netEur.toFixed(2);
    document.getElementById('net-pct-val').textContent = netPct.toFixed(2);
    var netColor = netEur >= 0 ? '#22c55e' : '#ef4444';
    document.getElementById('sim-net-eur').style.color = netColor;
    document.getElementById('sim-net-pct').style.color = netColor;
  }}
  calcolaSim();
  </script>

  {retros_section}

  <!-- Main Chart -->
  <div class="sec-t"><span>Analisi Grafica dei Prezzi</span> <span class="tag">INTERATTIVO</span></div>
  <div class="card"><div class="card-bd" style="padding:8px;">{ch_main}</div></div>

  <!-- Kronos Table + Feature Importance -->
  <div class="grid-2" style="margin-top:20px;">
    <div class="card">
      <div class="card-hd"><h2>Proiezione Kronos &mdash; {PRED_LEN} Giorni</h2></div>
      <div style="overflow-x:auto;">
      <table>
        <thead><tr><th>Data</th><th>Apertura</th><th>Max</th><th>Min</th><th>Chiusura</th><th>Delta</th></tr></thead>
        <tbody>{tab_pred}</tbody>
      </table>
      </div>
    </div>
    <div class="card">
      <div class="card-hd"><h2>Feature Importance (XGBoost)</h2></div>
      <div class="card-bd">
        <div style="display:flex;gap:12px;margin-bottom:12px;">
          <span class="chip"><span style="color:#3b82f6;font-weight:700;">&blacksquare;</span> Embeddings Kronos {emb_pct:.0f}%</span>
          <span class="chip"><span style="color:#f59e0b;font-weight:700;">&blacksquare;</span> Tecnici {tech_pct:.0f}%</span>
        </div>
        {ch_fi}
      </div>
    </div>
  </div>

  <!-- Sentiment Chart + Table -->
  <div class="sec-t"><span>Analisi del Sentiment</span> <span class="tag">{'FINBERT' if len(daily)>0 else 'N/D'}</span></div>
  <div class="grid-2">
    <div class="card">
      <div class="card-hd"><h2>Sentiment Score Giornaliero</h2></div>
      <div class="card-bd" style="padding:8px;">{ch_sent if ch_sent else '<div style="padding:40px;text-align:center;color:#94a3b8;">Dati non disponibili</div>'}</div>
    </div>
    <div class="card">
      <div class="card-hd"><h2>Dettaglio Sentiment</h2></div>
      <div style="overflow-x:auto;">
      <table>
        <thead><tr><th>Data</th><th>Score</th><th>Articoli</th><th>Pos/Neg</th></tr></thead>
        <tbody>{tab_news if tab_news else '<tr><td colspan="4" style="text-align:center;color:#94a3b8;padding:20px;">Nessun dato disponibile</td></tr>'}</tbody>
      </table>
      </div>
    </div>
  </div>

  <!-- Market Context -->
  <div class="sec-t"><span>Contesto di Mercato</span></div>
  <div class="grid-3">
    <div class="card">
      <div class="card-hd"><h2>Volatilit&agrave;</h2></div>
      <div class="card-bd">
        <div style="font-size:28px;font-weight:700;color:#0f172a;">{vola:.1f}%</div>
        <div style="font-size:12px;color:#64748b;">Volatilit&agrave; annualizzata (20gg)</div>
        <div style="margin-top:8px;font-size:13px;color:#475569;">ATR: {atr:,.0f} {currency} &middot; Range medio giornaliero: {last['high_low_pct']*100:.2f}%</div>
      </div>
    </div>
    <div class="card">
      <div class="card-hd"><h2>Momentum</h2></div>
      <div class="card-bd">
        <div style="font-size:28px;font-weight:700;color:{rc};">{rj}</div>
        <div style="font-size:12px;color:#64748b;">RSI 14: {rsi:.1f}</div>
        <div style="margin-top:8px;font-size:13px;color:#475569;">EMA 5: {last['ema_5']:,.0f} &middot; EMA 20: {last['ema_20']:,.0f} &middot; SMA 50: {last['sma_50']:,.0f}</div>
      </div>
    </div>
    <div class="card">
      <div class="card-hd"><h2>Volume &amp; Liquidit&agrave;</h2></div>
      <div class="card-bd">
        <div style="font-size:28px;font-weight:700;color:#0f172a;">{vr:.2f}x</div>
        <div style="font-size:12px;color:#64748b;">Volume Ratio (vs media 10gg)</div>
        <div style="margin-top:8px;font-size:13px;color:#475569;">Volume: {last['volume']:,.0f} &middot; Media 10gg: {last['volume_sma_10']:,.0f}</div>
      </div>
    </div>
  </div>

  <!-- Metodologia -->
  <div class="divider"></div>
  <div class="card">
    <div class="card-hd"><h2>Metodologia</h2></div>
    <div class="card-bd" style="font-size:13px;color:#475569;line-height:1.8;">
      <p><strong>1. Kronos (NeoQuasar/Kronos-base):</strong> Modello trasformatore foundation pre-addestrato su 45 exchange globali. Genera previsioni autoregressive a {PRED_LEN} giorni su OHLCV tramite tokenizzazione gerarchica e sampling con temperatura.</p>
      <p><strong>2. FinBERT (ProsusAI):</strong> BERT fine-tuned su testi finanziari (FinancialPhraseBank). Analizza il sentiment dei titoli da feed RSS configurati per l'asset. Output: score continuo [-1, +1] aggregato per giorno.</p>
      <p><strong>3. XGBoost Meta-Learning:</strong> Classificatore gradient-boosted (500 alberi, max_depth=4, lr=0.03). Feature: {len(TECH_FEATURES)} indicatori tecnici + componenti PCA da embeddings Kronos (832D &rarr; ridotti, varianza 90%). Target binario: rialzo &gt;{TARGET_THRESHOLD_PCT}% in {TARGET_HORIZON_DAYS} giorni. Addestrato con TimeSeriesSplit su storia disponibile.</p>
      <p style="margin-top:10px;color:#94a3b8;font-style:italic;">Disclaimer: Report sperimentale generato da sistema di ricerca quantitativa. Non costituisce consulenza finanziaria. Le performance passate non sono garanzia di risultati futuri.</p>
    </div>
  </div>

</div>

<div class="ftr">
  Kronos Quantitative Research System &middot; {now} &middot; Dati: Yahoo Finance &middot; Modelli: NeoQuasar/Kronos &middot; ProsusAI/FinBERT<br>
  <span style="color:#cbd5e1;">Per rigenerare: <code style="background:#f1f5f9;padding:1px 6px;border-radius:3px;">python report_complete.py --ticker {ticker}</code></span>
</div>

</body>
</html>"""
    return html


def processa_asset(ticker, tokenizer=None, model=None, output_dir=None):
    if output_dir is None:
        output_dir = f"report_{ticker}"
    print("=" * 60)
    print(f"KRONOS QUANTITATIVE RESEARCH REPORT \u2022 {ticker}")
    print(f"Orizzonte: {PRED_LEN} giorni")
    print("=" * 60)

    init_db()

    print(f"\n[1] Download dati {ticker}...")
    df = scarica_dati(ticker)
    df_tech = calcola_indicatori(df)
    print(f"   {len(df_tech)} righe")

    print("\n[2] Kronos prediction + embeddings...")
    pred, x_df, x_ts, y_ts, tok, km = kronos_prediction(df_tech, LOOKBACK, PRED_LEN, tokenizer, model)
    emb = estrai_embedding_kronos(tok, km, df_tech.iloc[-len(df_tech):])
    print(f"   Previsione: {len(pred)} giorni, embeddings: {emb.shape[1]}D")

    print("\n[3] News + FinBERT...")
    daily = news_sentiment(ticker)

    print("\n[4] XGBoost training...")
    xgb_model, prob, fi = train_xgboost(df_tech, emb)
    print(f"   Prob. rialzo: {prob:.1%}")

    print("\n[5] Migliori opportunit\u00e0...")
    best_long, best_short = calcola_optimal_trade(pred)
    print(f"   Long: +{best_long['return_pct']:.2f}% ({best_long['hold_days']}g)")
    print(f"   Short: +{best_short['return_pct']:.2f}% ({best_short['hold_days']}g)")

    print("\n[6] Retrospettiva walk-forward...")
    reg_df = regression_7gg(df_tech, emb, tok, km)
    if len(reg_df) > 0:
        acc = reg_df['correct'].mean() * 100
        print(f"   Accuratezza: {acc:.0f}% su {len(reg_df)} predizioni")
    else:
        print("   Dati insufficienti")

    print("\n[7] Generazione report HTML...")
    os.makedirs(output_dir, exist_ok=True)
    html = genera_html(df_tech, x_df, pred, x_ts, y_ts, daily, prob, fi, best_long, best_short, reg_df, ticker=ticker)

    output_file = os.path.join(output_dir, f"Report_{ticker}_Completo.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDone. Output: {os.path.abspath(output_file)}")
    print(f"Segnale XGBoost: {prob:.1%}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Kronos Complete Research Report")
    parser.add_argument("--ticker", type=str, default=None, help="Singolo asset (es. AAPL). Default: tutti")
    args = parser.parse_args()

    targets = [args.ticker] if args.ticker else ASSETS

    print("=" * 60)
    print(f"KRONOS QUANTITATIVE RESEARCH \u2022 {'Multi-Asset' if not args.ticker else args.ticker}")
    print(f"Targets: {targets}")
    print("=" * 60)

    tokenizer, model = None, None
    if len(targets) > 1:
        from model import Kronos, KronosTokenizer
        print("\nCaricamento Kronos (singola istanza per batch)...")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        tokenizer = tokenizer.to(device)
        model = model.to(device)
        model.eval()

    results = {}
    for ticker in targets:
        try:
            out = processa_asset(ticker, tokenizer, model)
            results[ticker] = out
            print(f"[OK] {ticker}: {out}")
        except Exception as e:
            print(f"[FAIL] {ticker}: {e}")
            import traceback
            traceback.print_exc()
            results[ticker] = None

    ok = sum(1 for v in results.values() if v is not None)
    print("\n" + "=" * 60)
    print(f"COMPLETATO: {ok}/{len(targets)} asset")
    print("=" * 60)


if __name__ == "__main__":
    main()
