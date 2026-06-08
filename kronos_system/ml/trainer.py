import logging
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from kronos_system.config import (
    MODELS_DIR, ASSETS, TECH_FEATURE_NAMES,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LR, XGB_SUBSAMPLE,
    XGB_COLSAMPLE, XGB_RANDOM_STATE, TARGET_THRESHOLD_PCT,
    TARGET_HORIZON_DAYS, SENTIMENT_FEATURES,
)
from kronos_system.data.database import init_db, write_prediction
from kronos_system.data.ingestion import fetch_prices, validate_prices, prices_to_records
from kronos_system.data.database import write_prices, write_indicators
from kronos_system.features.technical import compute_indicators, prepare_feature_matrix, compute_target, indicators_to_records
from kronos_system.ml.pca_pipeline import CausalPCA
from kronos_system.ml.walkforward import run_walkforward
from kronos_system.features.sentiment import run_sentiment_pipeline
from kronos_system.data.database import write_daily_sentiment, write_sentiment_articles

logger = logging.getLogger(__name__)


def _model_path(asset_id: str) -> str:
    return os.path.join(MODELS_DIR, asset_id, "xgb_model.pkl")


def _pca_path(asset_id: str) -> str:
    return os.path.join(MODELS_DIR, asset_id, "pca_pipeline.pkl")


def train_asset(asset_id: str, df_tech: pd.DataFrame, embedding_matrix: np.ndarray,
                sentiment_df: pd.DataFrame) -> dict:
    """Full training pipeline for one asset: walk-forward, save model, return metrics summary."""
    model_params = {
        "n_estimators": XGB_N_ESTIMATORS,
        "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LR,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE,
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": XGB_RANDOM_STATE,
    }

    wf_results = run_walkforward(
        df_tech=df_tech,
        embedding_matrix=embedding_matrix,
        sentiment_df=sentiment_df,
        model_class=xgb.XGBClassifier,
        model_params=model_params,
        asset_id=asset_id,
    )

    # Train final model on ALL available data
    target_arr = compute_target(
        df_tech["close"], horizon=TARGET_HORIZON_DAYS, threshold_pct=TARGET_THRESHOLD_PCT
    )
    tech_arr = df_tech[TECH_FEATURE_NAMES].values

    cpca = CausalPCA()
    os.makedirs(os.path.dirname(_model_path(asset_id)), exist_ok=True)

    train_parts = [tech_arr]
    if embedding_matrix is not None:
        emb_pca = cpca.fit_transform(embedding_matrix[:len(df_tech)])
        train_parts.append(emb_pca)
        with open(_pca_path(asset_id), "wb") as f:
            pickle.dump(cpca, f)
    if sentiment_df is not None and not sentiment_df.empty:
        sent_aligned = df_tech[["timestamps"]].copy()
        sent_aligned["date"] = sent_aligned["timestamps"].dt.strftime("%Y-%m-%d")
        sent_aligned = sent_aligned.merge(sentiment_df, on="date", how="left")[["score", "count", "pos_ratio"]].fillna(0.0).values
        train_parts.append(sent_aligned)

    X_full = np.concatenate(train_parts, axis=1)
    y_full = target_arr

    ok = ~np.isnan(y_full)
    X_full, y_full = X_full[ok], y_full[ok]

    scale_pos = (y_full == 0).sum() / max((y_full == 1).sum(), 1)
    final_model = xgb.XGBClassifier(
        **model_params, scale_pos_weight=scale_pos,
    )
    final_model.fit(X_full, y_full)
    with open(_model_path(asset_id), "wb") as f:
        pickle.dump(final_model, f)

    logger.info("Saved model for %s to %s", asset_id, _model_path(asset_id))
    logger.info("Final model scale_pos_weight=%.2f, n_features=%d", scale_pos, X_full.shape[1])

    accs = [f["accuracy"] for f in wf_results["folds"]]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "n_folds": len(wf_results["folds"]),
        "n_components_pca": cpca.get_n_components() if embedding_matrix is not None else 0,
    }


def predict_asset(asset_id: str, df_tech: pd.DataFrame, embedding_matrix: np.ndarray,
                  sentiment_df: pd.DataFrame, kronos_proj_ret: float | None = None) -> dict | None:
    """Run inference for one asset using the saved model.
    
    Returns dict with prediction fields, or None if model not found.
    """
    model_path = _model_path(asset_id)
    if not os.path.exists(model_path):
        logger.warning("No saved model for %s at %s", asset_id, model_path)
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    tech_arr = df_tech[TECH_FEATURE_NAMES].values
    parts = [tech_arr[-1:]]  # latest row

    if embedding_matrix is not None and os.path.exists(_pca_path(asset_id)):
        with open(_pca_path(asset_id), "rb") as f:
            cpca = pickle.load(f)
        emb_pca = cpca.transform(embedding_matrix[-1:])
        parts.append(emb_pca)

    if sentiment_df is not None and not sentiment_df.empty:
        latest_date = df_tech["timestamps"].iloc[-1].strftime("%Y-%m-%d")
        sent_row = sentiment_df[sentiment_df["date"] == latest_date]
        if sent_row.empty:
            sent_row = sentiment_df.tail(1)
        if not sent_row.empty:
            sent_vals = np.array([[float(sent_row["score"].iloc[0]),
                                   int(sent_row["count"].iloc[0]),
                                   float(sent_row["pos_ratio"].iloc[0])]])
            parts.append(sent_vals)
        else:
            parts.append(np.array([[0.0, 0, 0.0]]))

    X = np.concatenate(parts, axis=1)

    prob = float(model.predict_proba(X)[0, 1])

    if prob > 0.75:
        signal = "LONG"
    elif prob > 0.6:
        signal = "POSITIVE_BIAS"
    elif prob > 0.4:
        signal = "NEUTRAL"
    elif prob > 0.25:
        signal = "CAUTION"
    else:
        signal = "SELL_SIGNAL"

    confidence = max(prob, 1 - prob)

    pred = {
        "date": df_tech["timestamps"].iloc[-1].strftime("%Y-%m-%d"),
        "probability": round(prob, 4),
        "signal": signal,
        "confidence": round(confidence, 4),
        "model_version": "1.0.0",
        "kronos_proj_ret": kronos_proj_ret,
        "sentiment_score": float(sentiment_df["score"].iloc[-1]) if sentiment_df is not None and not sentiment_df.empty else 0.0,
    }
    write_prediction(asset_id, pred)
    return pred


def run_full_pipeline(asset_id: str) -> dict:
    """Run the end-to-end pipeline for one asset: fetch → features → train → predict.
    
    If a model already exists, skips training and only runs inference.
    """
    logger.info("=== Running pipeline for %s ===", asset_id)
    result = {"asset_id": asset_id, "status": "ok", "prediction": None, "metrics": None}

    try:
        df = fetch_prices(asset_id)
        if df.empty:
            result["status"] = "no_data"
            logger.warning("%s: no price data", asset_id)
            return result
        df = validate_prices(df, asset_id)
        if df.empty:
            result["status"] = "invalid_data"
            return result

        price_records = prices_to_records(df)
        write_prices(asset_id, price_records)

        df_tech = compute_indicators(df)
        if df_tech.empty:
            result["status"] = "indicator_failure"
            return result

        indicator_records = indicators_to_records(df_tech)
        write_indicators(asset_id, indicator_records)

        from model import Kronos, KronosTokenizer
        logger.info("Loading Kronos model for %s...", asset_id)
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        emb = _extract_emb_wrapper(tokenizer, model, df_tech)

        daily_sent = run_sentiment_pipeline(asset_id, asset_keywords=asset_id.split("-")[0].split(".")[0].lower().split())
        if not daily_sent.empty:
            sent_articles = _articles_from_daily(daily_sent, asset_id)
            write_sentiment_articles(asset_id, sent_articles)
            sent_records = daily_sent.rename(columns={"pos_ratio": "pos_ratio", "neg_ratio": "neg_ratio"}).to_dict("records")
            for r in sent_records:
                r["pos_ratio"] = float(r.get("pos_ratio", 0))
                r["neg_ratio"] = float(r.get("neg_ratio", 0))
            write_daily_sentiment(asset_id, sent_records)

        kronos_proj_ret = _compute_kronos_proj_return(model, tokenizer, df_tech)

        import os
        if os.path.exists(_model_path(asset_id)):
            logger.info("Model exists for %s, running inference only", asset_id)
            pred = predict_asset(asset_id, df_tech, emb, daily_sent, kronos_proj_ret)
            result["prediction"] = pred
        else:
            logger.info("No model for %s, training...", asset_id)
            metrics = train_asset(asset_id, df_tech, emb, daily_sent)
            result["metrics"] = metrics
            pred = predict_asset(asset_id, df_tech, emb, daily_sent, kronos_proj_ret)
            result["prediction"] = pred

        return result

    except Exception as e:
        logger.error("Pipeline failed for %s: %s", asset_id, e, exc_info=True)
        result["status"] = f"error: {e}"
        return result


def _extract_emb_wrapper(tokenizer, model, df_tech):
    from kronos_system.features.kronos_emb import extract_embeddings
    return extract_embeddings(tokenizer, model, df_tech)


def _articles_from_daily(daily: pd.DataFrame, asset_id: str) -> list[dict]:
    articles = []
    for _, r in daily.iterrows():
        articles.append({
            "date": r["date"].strftime("%Y-%m-%d") if hasattr(r["date"], "strftime") else str(r["date"])[:10],
            "headline": f"Sentiment aggregate for {asset_id}",
            "score": float(r["score"]),
            "label": "positive" if r["score"] > 0.05 else ("negative" if r["score"] < -0.05 else "neutral"),
            "source": "RSS",
        })
    return articles


def _compute_kronos_proj_return(model, tokenizer, df_tech) -> float | None:
    try:
        from kronos_system.config import KRONOS_LOOKBACK, KRONOS_PRED_LEN
        from model import KronosPredictor
        predictor = KronosPredictor(tokenizer=tokenizer, model=model)
        x_df = df_tech.iloc[-KRONOS_LOOKBACK:]
        inp = x_df[["open", "high", "low", "close", "volume", "amount"]].copy()
        x_ts = pd.Series(pd.to_datetime(x_df["timestamps"]).dt.tz_localize(None))
        y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + pd.Timedelta(days=1), periods=KRONOS_PRED_LEN, freq="D"))
        pred = predictor.predict(df=inp, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=KRONOS_PRED_LEN)
        ret = (pred["close"].iloc[-1] - pred["open"].iloc[0]) / pred["open"].iloc[0] * 100
        return float(ret)
    except Exception as e:
        logger.warning("Kronos prediction failed: %s", e)
        return None
