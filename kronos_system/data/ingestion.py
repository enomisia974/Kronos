import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
from kronos_system.config import YFINANCE_PERIOD, YFINANCE_INTERVAL

logger = logging.getLogger(__name__)


def fetch_prices(asset_id: str) -> pd.DataFrame:
    """Download OHLCV from yfinance. Returns empty DataFrame on failure."""
    try:
        df = yf.download(asset_id, period=YFINANCE_PERIOD, interval=YFINANCE_INTERVAL, progress=False)
        if df is None or df.empty:
            logger.warning("yfinance returned empty DataFrame for %s", asset_id)
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.rename(columns={
            "Date": "timestamps", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df["volume"] = df["volume"].astype(float)
        df["amount"] = df["close"] * df["volume"]
        df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
        logger.info("Fetched %d rows for %s", len(df), asset_id)
        return df
    except Exception as e:
        logger.error("Failed to fetch prices for %s: %s", asset_id, e)
        return pd.DataFrame()


def validate_prices(df: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    """Remove rows with NaN/Inf in OHLCV. Log warnings."""
    before = len(df)
    if df.empty:
        return df
    for col in ["open", "high", "low", "close", "volume"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            logger.warning("%s: %d NaN in %s — dropping rows", asset_id, n_nan, col)
    df = df.replace([float("inf"), -float("inf")], float("nan"))
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    after = len(df)
    if before - after > 0:
        logger.info("%s: dropped %d invalid rows", asset_id, before - after)
    return df


def prices_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert OHLCV DataFrame to list of dict for DB insertion."""
    records = []
    for _, r in df.iterrows():
        records.append({
            "date": r["timestamps"].strftime("%Y-%m-%d"),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]),
        })
    return records
