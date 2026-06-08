import pandas as pd
import numpy as np
import yfinance as yf
from model import Kronos, KronosTokenizer, KronosPredictor

def main():
    ticker = "AAPL"
    print(f"1. Scaricamento dati storici reali per {ticker} da Yahoo Finance...")
    
    # Scarichiamo l'ultimo anno di dati giornalieri
    df_apple = yf.download(ticker, period="1y", interval="1d")
    
    # Pulizia e formattazione del DataFrame per i requisiti di Kronos
    df_apple = df_apple.reset_index()
    df_apple.columns = [col[0] if isinstance(col, tuple) else col for col in df_apple.columns]
    df_apple = df_apple.rename(columns={
        'Date': 'timestamps', 
        'Open': 'open', 
        'High': 'high', 
        'Low': 'low', 
        'Close': 'close', 
        'Volume': 'volume'
    })
    
    # Allineiamo i volumi e aggiungiamo la colonna 'amount' fittizia se richiesta dal dataset
    df_apple['volume'] = df_apple['volume'].astype(float)
    df_apple['amount'] = df_apple['close'] * df_apple['volume']

    # Configurazione del contesto (Lookback) e dei giorni da prevedere (pred_len)
    # Prendiamo gli ultimi 30 giorni di borsa come storico per prevedere i successivi 5 giorni lavorativi
    lookback = 30  
    steps_da_prevedere = 5   
    
    # Tagliamo il DataFrame per isolare il periodo di lookback
    x_df = df_apple.iloc[-lookback:].copy()
    
    # Estraiamo i dati numerici necessari
    df_input = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    
    # Convertiamo i timestamp storici in una Series pulita (senza fusi orari)
    x_timestamp = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    
    # Generiamo i timestamp futuri (escludendo i weekend tramite freq='B' - Business Days)
    y_timestamp = pd.Series(pd.date_range(
        start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), 
        periods=steps_da_prevedere, 
        freq='B'
    ))

    print(f"\nUltimo prezzo storico di chiusura rilevato ({x_timestamp.iloc[-1].date()}): ${df_input['close'].iloc[-1]:.2f}")
    print(f"Generazione previsioni per le date: {[str(d.date()) for d in y_timestamp]}")

    print("\n2. Caricamento del modello Kronos...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    
    # Inizializziamo il predictor
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)

    print("\n3. Generazione della previsione con Kronos...")
    # Chiamata corretta con la firma esatta identificata nel codice sorgente
    previsioni = predictor.predict(
        df=df_input,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=steps_da_prevedere
    )

    print("\n**************************************************")
    print(f"RISULTATO PREVISIONI DI KRONOS PER APPLE ({ticker}):")
    print(previsioni[['open', 'high', 'low', 'close']])
    print("**************************************************")

if __name__ == "__main__":
    main()