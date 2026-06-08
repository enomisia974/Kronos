import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps

warnings.filterwarnings("ignore")


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_bollinger_bands(series, period=20, n_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + n_std * std, sma - n_std * std


def calc_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def add_technical_indicators(df):
    df = df.copy()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['rsi_14'] = calc_rsi(df['close'], 14)
    bb_upper, bb_lower = calc_bollinger_bands(df['close'], 20, 2)
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / df['close']
    df['atr_14'] = calc_atr(df, 14)
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_10'] + 1e-10)
    return df


def create_forward_target(df, horizon=3, threshold_pct=2.5):
    """Compute target. WARNING: NOT to be saved to CSV — recompute inside each fold."""
    df = df.copy()
    future_close = df['close'].shift(-horizon)
    forward_return = (future_close - df['close']) / (df['close'] + 1e-10) * 100
    df['target'] = (forward_return > threshold_pct).astype(int)
    return df


def extract_all_embeddings_fast(tokenizer, model, df_full, device='cpu', clip=5, window=90):
    price_cols = ['open', 'high', 'low', 'close']
    vol_col = 'volume'
    amt_col = 'amount'

    df = df_full.reset_index(drop=True)
    n = len(df)
    if n < 30:
        return pd.DataFrame()

    x = df[price_cols + [vol_col, amt_col]].values.astype(np.float32)

    # Sliding-window causal normalization (no future data leakage)
    x_norm = np.zeros_like(x)
    for i in range(n):
        start = max(0, i - window + 1)
        local = x[start:i + 1]
        local_mean = np.mean(local, axis=0)
        local_std = np.std(local, axis=0) + 1e-5
        x_norm[i] = (x[i] - local_mean) / local_std
    x_norm = np.clip(x_norm, -clip, clip)

    timestamps = pd.to_datetime(df['timestamps']).dt.tz_localize(None)
    time_df = calc_time_stamps(timestamps)
    stamp = time_df.values.astype(np.float32)

    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
    stamp_tensor = torch.from_numpy(stamp).unsqueeze(0).to(device)

    x_token = tokenizer.encode(x_tensor.clip(-clip, clip), half=True)

    with torch.no_grad():
        _, context = model.decode_s1(x_token[0], x_token[1], stamp_tensor)

    context_np = context[0].cpu().numpy()
    d_model = context_np.shape[1]
    emb_df = pd.DataFrame(
        context_np,
        columns=[f'kronos_emb_{j}' for j in range(d_model)]
    )
    return emb_df


def verify_kronos_causality(tokenizer, model, df, device='cpu', test_pos=50):
    """Verifica empiricamente che l'embedding al giorno T non cambi
    se la sequenza viene troncata a T. Se cambia, il modello non è causale."""
    emb_full = extract_all_embeddings_fast(tokenizer, model, df, device=device)
    df_truncated = df.iloc[:test_pos + 1].copy()
    emb_truncated = extract_all_embeddings_fast(tokenizer, model, df_truncated, device=device)
    if emb_full.empty or emb_truncated.empty:
        print(f"   SKIP: dati insufficienti per test causalità (full={len(emb_full)}, trunc={len(emb_truncated)})")
        return False
    vec_full = emb_full.iloc[test_pos].values
    vec_trunc = emb_truncated.iloc[test_pos].values
    diff = np.abs(vec_full - vec_trunc).max()
    is_causal = diff < 1e-4
    print(f"   Differenza max embedding[{test_pos}]: {diff:.2e}")
    print(f"   Kronos causale: {'SI' if is_causal else 'NO — LOOK-AHEAD BIAS PRESENTE'}")
    return is_causal


def main():
    ticker = "BTC-EUR"
    lookback = 90
    horizon = 3
    threshold_pct = 2.5

    print("=" * 60)
    print("FASE 1.1: DATA ENGINEERING - Indicatori tecnici")
    print("=" * 60)

    print(f"\n[1] Download dati {ticker}...")
    df_raw = yf.download(ticker, period="1y", interval="1d")
    df_raw = df_raw.reset_index()
    df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
    df_raw = df_raw.rename(columns={
        'Date': 'timestamps', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df_raw['volume'] = df_raw['volume'].astype(float)
    df_raw['amount'] = df_raw['close'] * df_raw['volume']
    print(f"   Scaricate {len(df_raw)} righe di dati.")

    print("\n[2] Calcolo indicatori tecnici...")
    df_tech = add_technical_indicators(df_raw)
    print(f"   Feature tecniche: {[c for c in df_tech.columns if c not in df_raw.columns]}")

    print("\n[3] Creazione target binario...")
    df_tech = create_forward_target(df_tech, horizon=horizon, threshold_pct=threshold_pct)
    positive_ratio = df_tech['target'].dropna().mean() * 100
    print(f"   Target: 1 se +{threshold_pct}% in {horizon}gg")
    print(f"   Distribuzione target: {positive_ratio:.1f}% positivi")

    print("\n" + "=" * 60)
    print("FASE 1.2: ESTRAZIONE EMBEDDINGS KRONOS")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[4] Caricamento modello Kronos su {device}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    tokenizer = tokenizer.to(device)
    model = model.to(device)
    model.eval()
    print("   Modello caricato.")

    print("\n[4b] Verifica causalità Kronos...")
    test_pos = min(50, len(df_tech) - 5)
    verify_kronos_causality(tokenizer, model, df_tech, device=device, test_pos=test_pos)

    print(f"\n[5] Estrazione embeddings Kronos (forward pass unico su tutta la sequenza)...")
    emb_df = extract_all_embeddings_fast(
        tokenizer, model, df_tech, device=device
    )
    d_model = emb_df.shape[1]
    print(f"   Embeddings estratti: {len(emb_df)} vettori x {d_model} dimensioni")
    print(f"   (Normalizzazione sliding-window causale — no data leakage)")

    print("\n" + "=" * 60)
    print("FASE 1.3: ASSEMBLAGGIO MASTER DATAFRAME")
    print("=" * 60)

    df_master = pd.concat([df_tech, emb_df], axis=1)
    df_master['timestamps'] = pd.to_datetime(df_master['timestamps']).dt.tz_localize(None)

    print(f"\n[6] Master DataFrame creato: {df_master.shape[0]} righe x {df_master.shape[1]} colonne")
    print(f"   Colonne tecniche: {[c for c in df_tech.columns if c not in df_raw.columns]}")
    print(f"   Colonne embedding: {emb_df.shape[1]}")
    print(f"   Target columns: (nessuna — calcolato a runtime in xgboost_pipeline.py)")
    print(f"   Valori NaN totali: {df_master.isna().sum().sum()}")

    output_dir = "feature_store"
    os.makedirs(output_dir, exist_ok=True)

    # Drop forward-looking columns BEFORE saving to CSV
    cols_to_drop = [c for c in ['target', 'forward_return_pct'] if c in df_master.columns]
    if cols_to_drop:
        df_master = df_master.drop(columns=cols_to_drop)
        print(f"   Droppate colonne forward-looking: {cols_to_drop}")

    csv_path = os.path.join(output_dir, "master_btc_features.csv")
    df_master.to_csv(csv_path, index=False)
    print(f"\n[7] Master salvato in: {os.path.abspath(csv_path)}")

    meta = {
        'ticker': ticker,
        'period': '1y',
        'lookback': lookback,
        'horizon': horizon,
        'threshold_pct': threshold_pct,
        'total_rows': len(df_master),
        'total_columns': df_master.shape[1],
        'embedding_dim': d_model,
        'technical_features': [c for c in df_tech.columns if c not in df_raw.columns],
        'positive_ratio_pct': round(positive_ratio, 2),
        'nan_total': int(df_master.isna().sum().sum()),
    }
    meta_path = os.path.join(output_dir, "pipeline_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"   Metadata salvati in: {os.path.abspath(meta_path)}")

    print("\n" + "-" * 60)
    print("ANTEPRIMA MASTER DATAFRAME:")
    print("-" * 60)
    preview_cols = ['timestamps', 'close', 'ema_10', 'rsi_14', 'bb_width']
    preview_cols += [c for c in emb_df.columns if c.startswith('kronos_emb_0')]
    available = [c for c in preview_cols if c in df_master.columns]
    print(df_master[available].tail(10).to_string())
    print("-" * 60)

    print("\n[SUCCESSO] Fase 1 completata. Dataset pronto per Fase 2 (NLP) e Fase 3 (fusione).")
    print(f"Prossimo passo: ingestione notizie con FinBERT e integrazione sentiment.")


if __name__ == "__main__":
    main()
