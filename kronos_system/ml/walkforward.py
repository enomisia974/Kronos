import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from kronos_system.config import (
    WALK_FORWARD_FOLDS, WALK_FORWARD_WINDOW,
    WALK_FORWARD_GAP, TARGET_THRESHOLD_PCT, TARGET_HORIZON_DAYS,
)
from kronos_system.features.technical import TECH_FEATURE_NAMES
from kronos_system.data.database import write_fold, write_metrics
from kronos_system.ml.pca_pipeline import CausalPCA
from kronos_system.features.technical import compute_target

logger = logging.getLogger(__name__)


def generate_folds(n_total: int, n_folds: int = WALK_FORWARD_FOLDS,
                   window: int = WALK_FORWARD_WINDOW, gap: int = WALK_FORWARD_GAP) -> list[dict]:
    """Generate walk-forward fold indices.
    
    Each fold: train on [train_start, train_end], test on [test_start, test_end]
    with gap days between train_end and test_start to prevent target leakage.
    
    Returns list of dict: {train_start, train_end, test_start, test_end,
                           train_slice, test_slice}
    """
    folds = []
    step = max(1, (n_total - window) // n_folds)
    for k in range(n_folds):
        train_end = window + k * step
        if train_end >= n_total - gap - 1:
            break
        train_start = 0
        test_start = train_end + gap
        test_end = min(test_start + step, n_total)
        if test_end - test_start < 2:
            break
        folds.append({
            "fold": k,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_slice": slice(train_start, train_end),
            "test_slice": slice(test_start, test_end),
        })
    logger.info("Generated %d walk-forward folds", len(folds))
    return folds


def run_walkforward(
    df_tech: pd.DataFrame,
    embedding_matrix: np.ndarray,
    sentiment_df: pd.DataFrame | None,
    model_class,
    model_params: dict,
    asset_id: str,
) -> dict:
    """Execute walk-forward validation and return per-fold metrics."""
    from kronos_system.features.technical import TECH_FEATURE_NAMES

    tech_arr = df_tech[TECH_FEATURE_NAMES].values
    close_arr = df_tech["close"].values
    ts_arr = df_tech["timestamps"].values

    target_arr = compute_target(
        df_tech["close"], horizon=TARGET_HORIZON_DAYS, threshold_pct=TARGET_THRESHOLD_PCT
    )

    # Align embeddings + sentiment to same index range
    n = len(df_tech)
    emb = embedding_matrix[:n] if embedding_matrix is not None else None

    sent_aligned = None
    if sentiment_df is not None and not sentiment_df.empty:
        sent_aligned = df_tech[["timestamps"]].copy()
        sent_aligned["date"] = sent_aligned["timestamps"].dt.strftime("%Y-%m-%d")
        sent_aligned = sent_aligned.merge(
            sentiment_df, on="date", how="left"
        )[["score", "count", "pos_ratio"]].fillna(0.0).values

    folds = generate_folds(n)
    results = []

    for fold_info in folds:
        tr_s = fold_info["train_slice"]
        te_s = fold_info["test_slice"]

        # Build train X
        train_parts = [tech_arr[tr_s]]

        if emb is not None:
            cpca = CausalPCA()
            train_emb_pca = cpca.fit_transform(emb[tr_s])
            train_parts.append(train_emb_pca)

        if sent_aligned is not None:
            train_parts.append(sent_aligned[tr_s])

        train_X = np.concatenate(train_parts, axis=1)
        train_y = target_arr[tr_s]

        # Filter NaN
        ok_train = ~np.isnan(train_y)
        train_X, train_y = train_X[ok_train], train_y[ok_train]

        if len(np.unique(train_y)) < 2:
            logger.warning("Fold %d: only one class in train, skipping", fold_info["fold"])
            continue

        model = model_class(**model_params)
        model.fit(train_X, train_y)

        # Build test X
        test_parts = [tech_arr[te_s]]

        if emb is not None:
            test_parts.append(cpca.transform(emb[te_s]))

        if sent_aligned is not None:
            test_parts.append(sent_aligned[te_s])

        test_X = np.concatenate(test_parts, axis=1)
        test_y = target_arr[te_s]

        ok_test = ~np.isnan(test_y)
        test_X, test_y = test_X[ok_test], test_y[ok_test]

        if len(test_y) < 2:
            continue

        preds = model.predict(test_X)
        proba = model.predict_proba(test_X)[:, 1]

        acc = accuracy_score(test_y, preds)
        baseline = max((test_y == 1).mean(), (test_y == 0).mean())

        try:
            prec = precision_score(test_y, preds, zero_division=0)
            rec = recall_score(test_y, preds, zero_division=0)
            f1 = f1_score(test_y, preds, zero_division=0)
        except Exception:
            prec = rec = f1 = 0.0

        fold_result = {
            "fold": fold_info["fold"],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "n_trades": int((preds == 1).sum()),
            "baseline_accuracy": baseline,
            "n_components_pca": cpca.get_n_components() if emb is not None else 0,
            "n_train": len(train_y),
            "n_test": len(test_y),
        }
        results.append(fold_result)

        # Write to DB
        try:
            write_metrics(asset_id, fold_info["fold"], fold_result)
        except Exception as e:
            logger.warning("Failed to write metrics to DB: %s", e)

        try:
            db_fold = {
                "train_start": str(ts_arr[tr_s][0])[:10],
                "train_end": str(ts_arr[tr_s][-1])[:10],
                "test_start": str(ts_arr[te_s][0])[:10],
                "test_end": str(ts_arr[te_s][-1])[:10],
                "n_train": fold_result["n_train"],
                "n_test": fold_result["n_test"],
            }
            write_fold(asset_id, fold_info["fold"], db_fold)
        except Exception as e:
            logger.warning("Failed to write fold to DB: %s", e)

        logger.info(
            "Fold %2d | train %4d | test %4d | acc %.3f | base %.3f | n_comp %d",
            fold_info["fold"], fold_result["n_train"], fold_result["n_test"],
            acc, baseline, fold_result["n_components_pca"]
        )

    return {"folds": results, "target_arr": target_arr}
