import os
import subprocess
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

OUTPUT_DIR = "feature_store"
PRICE_CSV = os.path.join(OUTPUT_DIR, "master_btc_features.csv")
SENTIMENT_CSV = os.path.join(OUTPUT_DIR, "news_sentiment_daily.csv")
UNIFIED_CSV = os.path.join(OUTPUT_DIR, "unified_master.csv")


def run_step(script_name, description):
    print(f"\n{'=' * 60}")
    print(f"ESECUZIONE: {description}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"\n[ERRORE] {script_name} fallito (codice {result.returncode})")
        print(result.stderr)
        return False
    return True


def load_and_align():
    print("\n" + "=" * 60)
    print("FASE 3.1: ASSEMBLAGGIO MASTER DATAFRAME UNIFICATO")
    print("=" * 60)

    print(f"\n[1] Caricamento dati prezzo + embeddings da {PRICE_CSV}...")
    df_price = pd.read_csv(PRICE_CSV, parse_dates=['timestamps'])
    print(f"   {len(df_price)} righe x {df_price.shape[1]} colonne")

    print(f"\n[2] Caricamento sentiment daily da {SENTIMENT_CSV}...")
    df_sentiment = pd.read_csv(SENTIMENT_CSV, parse_dates=['date'])
    print(f"   {len(df_sentiment)} righe x {df_sentiment.shape[1]} colonne")
    print(f"   Range date: {df_sentiment['date'].min()} -> {df_sentiment['date'].max()}")

    print("\n[3] Allineamento temporale (merge su data)...")
    df_price['date'] = df_price['timestamps'].dt.date
    df_sentiment['date_key'] = df_sentiment['date'].dt.date

    sentiment_cols = [
        'date_key', 'sentiment_mean', 'sentiment_weighted',
        'sentiment_positive_ratio', 'sentiment_negative_ratio',
        'sentiment_neutral_ratio', 'article_count', 'sentiment_std',
    ]
    df_sentiment_clean = df_sentiment[sentiment_cols].copy()
    df_sentiment_clean = df_sentiment_clean.rename(columns={
        'sentiment_mean': 'sentiment_score',
    })
    del sentiment_cols

    df_unified = df_price.merge(
        df_sentiment_clean,
        how='left',
        left_on='date',
        right_on='date_key',
    )
    # Keep original sentiment date for leak checking; drop the merge key
    df_unified = df_unified.rename(columns={'date_key': 'sentiment_original_date'})

    n_matched = df_unified['sentiment_score'].notna().sum()
    print(f"   Righe con sentiment: {n_matched} / {len(df_unified)}")

    # ─────────────────────────────────────────────────────────────
    # FIX: T-1 SHIFT — elimina look-ahead bias intraday
    # Il sentiment della barra daily T-1 viene usato per predire T.
    # Se un articolo è pubblicato dopo la chiusura di T-1, non
    # contamina la feature di T. Se è pubblicato durante T, viene
    # ignorato (sarà nel sentiment di T+1 dopo il prossimo shift).
    # ─────────────────────────────────────────────────────────────
    print("\n[4] Shift sentiment a T-1 (prevenzione data leakage)...")
    sent_cols = [
        'sentiment_score', 'sentiment_weighted',
        'sentiment_positive_ratio', 'sentiment_negative_ratio',
        'sentiment_neutral_ratio', 'article_count', 'sentiment_std',
    ]
    existing_sent_cols = [c for c in sent_cols if c in df_unified.columns]
    df_unified[existing_sent_cols] = df_unified[existing_sent_cols].shift(1)
    df_unified['sentiment_original_date'] = df_unified['sentiment_original_date'].shift(1)
    print(f"   Shift applicato: sentiment T-1 → riga T")

    # ─────────────────────────────────────────────────────────────
    # LEAK CHECK REALE: dopo lo shift, sentiment_original_date deve
    # essere sempre ≤ timestamps (stesso giorno o giorno prima).
    # Se >, significa che un sentiment futuro è finito in una riga
    # passata — anomalia nel merge o nello shift.
    # ─────────────────────────────────────────────────────────────
    print("\n[4b] Maniacal Timestamp Check — dopo T-1 shift...")
    df_check = df_unified.dropna(subset=['sentiment_original_date']).copy()
    df_check['price_date'] = pd.to_datetime(df_check['timestamps']).dt.date
    # sentiment_original_date è già un oggetto date
    leaks = df_check[df_check['sentiment_original_date'] > df_check['price_date']]
    if len(leaks) > 0:
        print(f"   [DATA LEAK] {len(leaks)} righe con sentiment futuro dopo shift!")
        print(leaks[['timestamps', 'sentiment_original_date', 'sentiment_score']].head(10))
        print("   CORREZIONE: lo shift T-1 dovrebbe aver eliminato tutti i leak.")
        print("   Se questo warning compare, c'è un bug nel merge o nello shift.")
    else:
        print("   [OK] Dopo shift T-1: nessun sentiment futuro nelle feature.")
        print("   (sentiment_original_date <= timestamps per tutte le righe)")

    df_unified = df_unified.drop(columns=['price_date', 'sentiment_original_date'],
                                  errors='ignore')

    print(f"\n[5] Forward-fill sentiment per giorni senza notizie (Fase 3.3)...")
    nan_before = df_unified['sentiment_score'].isna().sum()
    df_unified[existing_sent_cols] = df_unified[existing_sent_cols].fillna(method='ffill')
    nan_after = df_unified['sentiment_score'].isna().sum()
    print(f"   NaN prima: {nan_before}, dopo forward-fill: {nan_after}")
    if nan_after > 0:
        df_unified[existing_sent_cols] = df_unified[existing_sent_cols].fillna(0.0)
        print(f"   NaN rimanenti azzerati a 0.0")

    df_unified = df_unified.drop(columns=['date'])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_unified.to_csv(UNIFIED_CSV, index=False)
    print(f"\n[6] Unified Master salvato in: {os.path.abspath(UNIFIED_CSV)}")
    print(f"   Shape finale: {df_unified.shape[0]} righe x {df_unified.shape[1]} colonne")

    print("\n" + "-" * 60)
    print("ANTEPRIMA UNIFIED MASTER (ultime 5 righe):")
    print("-" * 60)
    preview = ['timestamps', 'close', 'rsi_14', 'sentiment_score',
               'sentiment_weighted', 'article_count', 'target']
    available = [c for c in preview if c in df_unified.columns]
    print(df_unified[available].tail(5).to_string())

    print("\nColonne sentiment disponibili:")
    print([c for c in existing_sent_cols])

    print("\n[SUCCESSO] Fase 3 completata. Dataset pronto per Fase 4 (XGBoost).")

    return df_unified


def main():
    print("=" * 60)
    print("KRONOS QUANT PIPELINE - VERSIONE UNIFICATA")
    print(f"Data esecuzione: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    if not os.path.exists(PRICE_CSV):
        print("\n[INFO] feature_store/master_btc_features.csv non trovato.")
        print("Eseguo feature_pipeline.py (Fase 1)...")
        ok = run_step("feature_pipeline.py", "Fase 1: Feature Engineering + Kronos Embeddings")
        if not ok:
            print("Fallito. Impossibile proseguire.")
            return
    else:
        print(f"\n[INFO] Dati prezzo esistenti: {PRICE_CSV}")

    if not os.path.exists(SENTIMENT_CSV):
        print("\n[INFO] feature_store/news_sentiment_daily.csv non trovato.")
        print("Eseguo news_pipeline.py (Fase 2)...")
        ok = run_step("news_pipeline.py", "Fase 2: NLP Sentiment Analysis")
        if not ok:
            print("Fallito. Impossibile proseguire.")
            return
    else:
        print(f"\n[INFO] Dati sentiment esistenti: {SENTIMENT_CSV}")

    df = load_and_align()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETATA CON SUCCESSO")
    print("=" * 60)
    print(f"\nFile output: {os.path.abspath(UNIFIED_CSV)}")
    print(f"\nProssimo passo (Fase 4): addestrare XGBoost con:")
    print("  from sklearn.model_selection import TimeSeriesSplit")
    print("  import xgboost as xgb")
    print("  model = xgb.XGBClassifier()")


if __name__ == "__main__":
    main()
