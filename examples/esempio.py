import pandas as pd
import numpy as np
from model import Kronos, KronosTokenizer, KronosPredictor

def main():
    print("1. Caricamento del modello e del tokenizer in corso...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)

    print("\n2. Preparazione dei dati storici (Lookback) con i relativi Timestamp...")
    valori_storici = [10.0, 10.5, 11.0, 10.8, 11.5, 12.0, 11.9, 12.5]
    
    df_input = pd.DataFrame({
        'open': valori_storici,
        'high': valori_storici,
        'low': valori_storici,
        'close': valori_storici,
        'volume': [1000] * len(valori_storici)
    })
    
    # Generiamo i timestamp e li convertiamo SUBITO in una Series per soddisfare il codice dell'autore
    x_timestamp = pd.Series(pd.date_range(end=pd.Timestamp.now(), periods=len(valori_storici), freq='D'))
    
    print(f"Dati storici inseriti:\n{valori_storici}")

    print("\n3. Generazione della previsione...")
    steps_da_prevedere = 3
    
    # Generiamo i timestamp futuri e li convertiamo anch'essi in una Series
    y_timestamp = pd.Series(pd.date_range(start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), periods=steps_da_prevedere, freq='D'))
    
    # Eseguiamo la predizione con le Series temporali corrette
    previsioni = predictor.predict(
        df=df_input,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=steps_da_prevedere
    )

    print("\n**************************************************")
    print("RISULTATO PREVISIONI DI KRONOS:")
    print(previsioni)
    print("**************************************************")

if __name__ == "__main__":
    main()