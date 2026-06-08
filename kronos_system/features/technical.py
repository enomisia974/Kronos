import logging
import numpy as np
import pandas as pd
from kronos_system.config import TECH_FEATURES, TARGET_THRESHOLD_PCT

logger = logging.getLogger(__name__)

TECH_FEATURE_NAMES = TECH_FEATURES


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators WITHOUT forward-looking target."""
    df = df.copy()

    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    d = df["close"].diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.rolling(14).mean()
    al = l.rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + ag / (al + 1e-10)))

    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_mid"] = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    df["atr_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["volume_sma_10"] = df["volume"].rolling(10).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_10"] + 1e-10)

    df["return_1d"] = df["close"].pct_change()
    df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)
    df["high_low_pct"] = (df["high"] - df["low"]) / df["close"]

    n_initial = len(df)
    df = df.dropna(subset=TECH_FEATURE_NAMES)
    logger.debug("compute_indicators: dropped %d initial-NaN rows", n_initial - len(df))
    return df


def compute_target(close_series: pd.Series, horizon: int = 3, threshold_pct: float = 2.5) -> np.ndarray:
    """Compute forward-looking target CAUSALLY.
    
    target[i] = 1 if (close[i+horizon] - close[i]) / close[i] > threshold_pct
    Last `horizon` positions get target=0 (unknown).
    Returns integer array.
    """
    n = len(close_series)
    target = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        ret = (close_series.iloc[i + horizon] - close_series.iloc[i]) / close_series.iloc[i] * 100
        target[i] = 1 if ret > threshold_pct else 0
    return target


def prepare_feature_matrix(
    df_tech: pd.DataFrame,
    embedding_matrix: np.ndarray | None = None,
    sentiment_df: pd.DataFrame | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate tech features + PCA embeddings + sentiment.
    
    Returns (X, y) where X rows align with df_tech[start_idx:end_idx].
    Target is already aligned to each row.
    """
    if end_idx is None:
        end_idx = len(df_tech)

    sub = df_tech.iloc[start_idx:end_idx]
    X_parts = []

    tech_arr = sub[TECH_FEATURE_NAMES].values
    X_parts.append(tech_arr)

    if embedding_matrix is not None:
        emb_sub = embedding_matrix[start_idx:end_idx]
        X_parts.append(emb_sub)

    if sentiment_df is not None:
        merged = sub[["timestamps"]].copy()
        merged["date"] = merged["timestamps"].dt.strftime("%Y-%m-%d")
        merged = merged.merge(sentiment_df, on="date", how="left")
        sent_arr = merged[["score", "count", "pos_ratio"]].fillna(0.0).values
        X_parts.append(sent_arr)

    X = np.concatenate(X_parts, axis=1)
    y = sub["target"].values
    return X, y


def indicators_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert indicator rows to dict for DB insertion."""
    records = []
    for _, r in df.iterrows():
        records.append({
            "date": r["timestamps"].strftime("%Y-%m-%d") if hasattr(r["timestamps"], "strftime") else str(r["timestamps"])[:10],
            "ema_5": float(r.get("ema_5", 0) or 0),
            "ema_10": float(r.get("ema_10", 0) or 0),
            "ema_20": float(r.get("ema_20", 0) or 0),
            "rsi_14": float(r.get("rsi_14", 50) or 50),
            "bb_width": float(r.get("bb_width", 0) or 0),
            "atr_14": float(r.get("atr_14", 0) or 0),
            "volume_ratio": float(r.get("volume_ratio", 1) or 1),
            "target": int(r.get("target", 0)),
        })
    return records
