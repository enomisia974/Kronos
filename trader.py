import os, sys, time
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

from model import Kronos, KronosTokenizer, KronosPredictor

def calcola_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def calcola_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calcola_bb(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    std = tp.rolling(period).std()
    bbw = ((ma + 2*std) - (ma - 2*std)) / ma
    return bbw, ma, std

def backtest_mae(df_full, predictor, n_back=14, lookback=90):
    results = []
    for offset in range(n_back, 0, -1):
        cut = len(df_full) - offset
        if cut < lookback:
            continue
        x = df_full.iloc[cut - lookback:cut].copy()
        inp = x[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_ts = pd.Series(pd.to_datetime(x['timestamps']).dt.tz_localize(None))
        y_ts = pd.Series([x_ts.iloc[-1] + timedelta(days=1)])
        p = predictor.predict(df=inp, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=1)
        pred_close = p['close'].iloc[0]
        actual_close = df_full.iloc[cut]['close']
        err = abs((pred_close - actual_close) / actual_close * 100)
        results.append(err)
    return np.mean(results) if results else 5.0

def get_signal(ticker, pred_len=14):
    df = yf.download(ticker, period="1y", interval="1d")
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.rename(columns={
        'Date': 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    df['volume'] = df['volume'].astype(float)
    df['amount'] = df['close'] * df['volume']

    lookback = 90
    x_df = df.iloc[-lookback:].copy()
    df_input = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    y_ts = pd.Series(pd.date_range(
        start=x_ts.iloc[-1] + timedelta(days=1), periods=pred_len, freq='B'))

    p_now = df_input['close'].iloc[-1]
    rsi = calcola_rsi(df_input['close']).iloc[-1]
    atr = calcola_atr(x_df).iloc[-1]
    bbw, bb_ma, bb_std = calcola_bb(x_df)
    bbw_val = bbw.iloc[-1]

    vol_ratio = df_input['volume'].iloc[-1] / df_input['volume'].rolling(10).mean().iloc[-1]

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)
    pred = predictor.predict(
        df=df_input, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len)

    pred_close_last = pred['close'].iloc[-1]
    ret = (pred_close_last - p_now) / p_now * 100

    mae = backtest_mae(df, predictor)

    # Trend AI (14 giorni)
    if ret > 2:
        trend = 1
    elif ret < -2:
        trend = -1
    else:
        trend = 0

    # RSI signal
    if rsi < 15:
        rsi_signal = 2  # iper-sovravenduto: eccezione mean reversion
    elif rsi < 30:
        rsi_signal = 1
    elif rsi > 85:
        rsi_signal = -2
    elif rsi > 70:
        rsi_signal = -1
    else:
        rsi_signal = 0

    # Mean reversion clause: RSI < 15 relaxa trend
    if rsi < 15:
        effective_trend = trend  # accettiamo anche trend = 0
        mean_rev_mode = True
    else:
        effective_trend = trend
        mean_rev_mode = False

    # Score
    score = effective_trend * 0.6 + rsi_signal * 0.4

    # MAE filter: predicted return must be > MAE * 2
    mae_threshold = mae * 2
    sufficient_return = abs(ret) > mae_threshold

    # Volume filter
    vol_filter = vol_ratio > 0.8

    # Decision
    if mean_rev_mode and trend >= 0 and rsi_signal == 2:
        action = 'BUY'
    elif score > 0.3 and trend == 1 and sufficient_return:
        action = 'BUY'
    elif score < -0.3 and trend == -1 and sufficient_return:
        action = 'SELL'
    else:
        action = 'HOLD'

    return {
        'ticker': ticker,
        'price': p_now,
        'rsi': rsi,
        'atr': atr,
        'bbw': bbw_val,
        'vol_ratio': vol_ratio,
        'mae': mae,
        'pred_return_pct': ret,
        'trend': trend,
        'rsi_signal': rsi_signal,
        'score': score,
        'sufficient_return': sufficient_return,
        'mae_threshold': mae_threshold,
        'mean_rev_mode': mean_rev_mode,
        'action': action,
    }

def execute_trades(client, tickers, pred_len=14):
    account = client.get_account()
    cash = float(account.cash)
    positions = {p.symbol: p for p in client.get_all_positions()}

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        sig = get_signal(ticker, pred_len)

        is_crypto = ticker in ("BTC-USD", "ETH-USD", "SOL-USD")
        base_size = 0.10 if is_crypto else 0.15
        size = base_size

        # Volume filter: riduci del 50% se vol ratio < 0.8
        if sig['vol_ratio'] < 0.8:
            size *= 0.5
            print(f"  Volume Ratio {sig['vol_ratio']:.2f}x < 0.8 -> size ridotta a {size*100:.0f}%")

        # MAE confidence check
        print(f"  Prezzo: {sig['price']:.2f}, RSI: {sig['rsi']:.0f}, ATR: {sig['atr']:.2f}")
        print(f"  BBW: {sig['bbw']:.4f}, VolRatio: {sig['vol_ratio']:.2f}x")
        print(f"  Pred. {pred_len}gg: {sig['pred_return_pct']:.2f}% (MAE: {sig['mae']:.2f}%, soglia: {sig['mae_threshold']:.2f}%)")
        print(f"  Score: {sig['score']:.2f}, MeanRev: {sig['mean_rev_mode']}, ReturnOK: {sig['sufficient_return']}")
        print(f"  Azione: {sig['action']}")

        in_pos = ticker in positions

        if sig['action'] == 'BUY' and not in_pos:
            # Stop loss dinamico basato su ATR
            stop_price = round(sig['price'] - 1.5 * sig['atr'], 2)
            qty = max(1, int(cash * size / sig['price']))
            actual_sl_pct = (sig['price'] - stop_price) / sig['price'] * 100

            print(f"  >> ACQUISTO {qty} @ {sig['price']:.2f}")
            print(f"  Stop Loss dinamico: {stop_price:.2f} ({actual_sl_pct:.1f}%, ATRx1.5={1.5*sig['atr']:.2f})")

            try:
                o = client.submit_order(MarketOrderRequest(
                    symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                print(f"  Ordine: {o.id}")

                time.sleep(1)
                client.submit_order(StopLossRequest(
                    symbol=ticker, qty=qty, side=OrderSide.SELL,
                    type=OrderType.STOP, stop_price=stop_price, time_in_force=TimeInForce.GTC))
                print(f"  Stop loss OK")
                cash -= qty * sig['price']
            except Exception as e:
                print(f"  ERRORE: {e} (ticker non supportato da Alpaca?)")

        elif sig['action'] == 'SELL' and in_pos:
            pos = positions[ticker]
            qty = int(pos.qty)
            print(f"  >> VENDITA {qty} @ {sig['price']:.2f}")
            try:
                client.submit_order(MarketOrderRequest(
                    symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            except Exception as e:
                print(f"  ERRORE: {e}")

        elif in_pos:
            pos = positions[ticker]
            entry = float(pos.avg_entry_price)
            pnl = (sig['price'] - entry) / entry * 100
            print(f"  Posizione: entry {entry:.2f}, P&L {pnl:+.2f}%")

    print("\n=== Riepilogo posizioni ===")
    for p in client.get_all_positions():
        pnl = float(p.unrealized_pl)
        print(f"  {p.symbol}: {p.qty} @ {p.avg_entry_price} | P&L: ${pnl:.2f}")

def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["BAMI.MI", "BMPS.MI", "ISP.MI", "UNI.MI", "LDO.MI"]
    pred_len = 14

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("ERRORE: ALPACA_API_KEY e ALPACA_SECRET_KEY non impostate")
        sys.exit(1)

    client = TradingClient(api_key, secret_key, paper=True)
    execute_trades(client, tickers, pred_len)

if __name__ == "__main__":
    main()
