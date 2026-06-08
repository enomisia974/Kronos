import os, sys, time
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from model import Kronos, KronosTokenizer, KronosPredictor

def calcola_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def get_signal(ticker):
    df = yf.download(ticker, period="1y", interval="1d")
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.rename(columns={
        'Date': 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    df['volume'] = df['volume'].astype(float)
    df['amount'] = df['close'] * df['volume']

    lookback = 60
    pred_len = 5
    x_df = df.iloc[-lookback:].copy()
    df_input = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_ts = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    y_ts = pd.Series(pd.date_range(
        start=x_ts.iloc[-1] + timedelta(days=1), periods=pred_len, freq='B'))

    p_now = df_input['close'].iloc[-1]
    rsi = calcola_rsi(df_input['close']).iloc[-1]

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)
    pred = predictor.predict(
        df=df_input, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len)

    pred_min = pred['low'].min()
    pred_max = pred['high'].max()
    pred_close_last = pred['close'].iloc[-1]
    ret = (pred_close_last - p_now) / p_now * 100

    if rsi < 30:
        rsi_signal = 1
    elif rsi > 70:
        rsi_signal = -1
    else:
        rsi_signal = 0

    if ret > 2:
        trend = 1
    elif ret < -2:
        trend = -1
    else:
        trend = 0

    confidence = min(100, max(0, abs(ret) * 10 + abs(rsi - 50) + 20))
    score = trend * 0.6 + rsi_signal * 0.4

    return {
        'ticker': ticker,
        'price': p_now,
        'rsi': rsi,
        'pred_return_pct': ret,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'trend': trend,
        'rsi_signal': rsi_signal,
        'score': score,
        'confidence': confidence,
        'action': 'BUY' if score > 0.3 and trend == 1 else 'SELL' if score < -0.3 and trend == -1 else 'HOLD',
    }

def execute_trades(client, tickers, stop_loss_pct=3.0, max_positions=5):
    account = client.get_account()
    cash = float(account.cash)
    positions = {p.symbol: p for p in client.get_all_positions()}

    print(f"Cash: ${cash:.2f} | Posizioni aperte: {len(positions)}/{max_positions}")

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        sig = get_signal(ticker)
        print(f"  Prezzo: {sig['price']:.2f}, RSI: {sig['rsi']:.0f}, Score: {sig['score']:.2f}")
        print(f"  Pred. ritorno: {sig['pred_return_pct']:.2f}% → {sig['action']}")

        in_pos = ticker in positions

        if sig['action'] == 'BUY' and not in_pos and len(positions) < max_positions:
            qty = max(1, int(cash * 0.15 / sig['price']))
            stop_price = round(sig['price'] * (1 - stop_loss_pct / 100), 2)
            print(f"  >> ACQUISTO {qty} @ {sig['price']:.2f}, STOP {stop_price:.2f}")

            o = client.submit_order(MarketOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            print(f"  Ordine: {o.id}")

            time.sleep(1)
            client.submit_order(StopLossRequest(
                symbol=ticker, qty=qty, side=OrderSide.SELL,
                type=OrderType.STOP, stop_price=stop_price, time_in_force=TimeInForce.GTC))
            print(f"  Stop loss OK")
            cash -= qty * sig['price']

        elif sig['action'] == 'SELL' and in_pos:
            pos = positions[ticker]
            qty = int(pos.qty)
            print(f"  >> VENDITA {qty} @ {sig['price']:.2f}")
            client.submit_order(MarketOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))

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
    stop_loss_pct = 3.0

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("ERRORE: ALPACA_API_KEY e ALPACA_SECRET_KEY non impostate")
        sys.exit(1)

    client = TradingClient(api_key, secret_key, paper=True)
    execute_trades(client, tickers, stop_loss_pct)

if __name__ == "__main__":
    main()
