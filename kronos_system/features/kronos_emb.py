import logging
import numpy as np
import pandas as pd
import torch
from kronos_system.config import KRONOS_LOOKBACK

logger = logging.getLogger(__name__)


def extract_embeddings(tokenizer, model, df: pd.DataFrame, window: int = KRONOS_LOOKBACK) -> np.ndarray:
    """Extract Kronos embeddings with CAUSAL sliding-window normalization."""
    from model.kronos import calc_time_stamps

    price_cols = ["open", "high", "low", "close"]
    df = df.reset_index(drop=True)
    x = df[price_cols + ["volume", "amount"]].values.astype(np.float32)
    x_norm = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        local = x[start:i + 1]
        local_mean = np.mean(local, axis=0)
        local_std = np.std(local, axis=0) + 1e-5
        x_norm[i] = (x[i] - local_mean) / local_std
    x = np.clip(x_norm, -5, 5)

    ts = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    stamp = calc_time_stamps(ts).values.astype(np.float32)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    st = torch.from_numpy(stamp).unsqueeze(0).to(device)
    x_token = tokenizer.encode(xt.clip(-5, 5), half=True)
    with torch.no_grad():
        _, ctx = model.decode_s1(x_token[0], x_token[1], st)
    embeddings = ctx[0].cpu().numpy()
    logger.info("Extracted Kronos embeddings: %s", str(embeddings.shape))
    return embeddings
