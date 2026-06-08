"""FastAPI backend — read-only, no ML logic."""

import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from kronos_system.data.database import (
    read_latest_prediction, read_prediction_history,
    read_rolling_metrics, read_asset_catalog, read_latest_sentiment,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kronos Quantitative Research API",
    version="1.0.0",
    description="Read-only API for pre-computed predictions and metrics",
)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/assets/catalog")
def asset_catalog():
    rows = read_asset_catalog()
    return [dict(r) for r in rows]


@app.get("/predictions/latest/{asset_id}")
def prediction_latest(asset_id: str):
    row = read_latest_prediction(asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for {asset_id}")
    result = dict(row)
    sent = read_latest_sentiment(asset_id)
    result["sentiment"] = dict(sent) if sent else None
    return result


@app.get("/predictions/history/{asset_id}")
def prediction_history(asset_id: str, days: int = 60):
    rows = read_prediction_history(asset_id, days)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No predictions found for {asset_id}")
    return [dict(r) for r in rows]


@app.get("/metrics/rolling/{asset_id}")
def metrics_rolling(asset_id: str):
    rows = read_rolling_metrics(asset_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No metrics found for {asset_id}")
    data = [dict(r) for r in rows]
    accs = [m["accuracy"] for m in data]
    return {
        "asset_id": asset_id,
        "n_folds": len(data),
        "mean_accuracy": round(sum(accs) / len(accs), 4) if accs else 0.0,
        "std_accuracy": round(
            (sum((a - sum(accs) / len(accs)) ** 2 for a in accs) / len(accs)) ** 0.5, 4
        ) if accs else 0.0,
        "folds": data,
    }
