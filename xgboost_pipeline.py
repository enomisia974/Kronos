import os, logging, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xgboost_pipeline")

warnings.filterwarnings("ignore")

UNIFIED_CSV = "feature_store/unified_master.csv"

TECHNICAL_FEATURES = [
    'ema_5', 'ema_10', 'ema_20', 'rsi_14', 'bb_width',
    'atr_14', 'volume_ratio',
]

SENTIMENT_FEATURES = [
    'sentiment_score', 'sentiment_weighted',
    'sentiment_positive_ratio', 'sentiment_negative_ratio',
    'sentiment_neutral_ratio', 'article_count',
]

BACKTEST_DAYS = 30
TRADE_THRESHOLD = 0.75
TRADE_AMOUNT = 100.0
FEE = 2.0
TARGET_HORIZON = 3
TARGET_THRESHOLD_PCT = 2.5

# PCA adaptive
PCA_VARIANCE_THRESHOLD = 0.90
PCA_MIN_COMPONENTS = 5
PCA_MAX_COMPONENTS = 50

# CV adattivo
CV_MIN_SPLITS = 5
CV_MAX_SPLITS = 30
CV_TEST_MIN_DAYS = 15

# Model persistence
MODEL_DIR = "feature_store/models"


def compute_target_for_idx(close: np.ndarray, idx: np.ndarray,
                           horizon: int = TARGET_HORIZON,
                           threshold: float = TARGET_THRESHOLD_PCT) -> np.ndarray:
    """Compute binary target using ONLY close prices available within idx.
    
    target[j] = 1 if price rises > threshold% from idx[j] to idx[j]+horizon
    
    Positions where idx[j]+horizon exceeds idx[-1] (i.e. the future
    close needed for the label is outside the fold) are masked with -1.
    This guarantees no look-ahead even in the label construction.
    """
    target = np.full(len(idx), -1, dtype=np.int8)
    last_valid_future = idx[-1]  # max index accessible within this fold
    for j, i in enumerate(idx):
        future_idx = i + horizon
        if future_idx <= last_valid_future:
            ret = (close[future_idx] - close[i]) / close[i] * 100
            target[j] = 1 if ret > threshold else 0
    return target


def load_data():
    logger.info("=" * 60)
    logger.info("FASE 4: META-MODELLO XGBOOST")
    logger.info("=" * 60)

    logger.info(f"[1] Caricamento unified master: %s", UNIFIED_CSV)
    df = pd.read_csv(UNIFIED_CSV, parse_dates=['timestamps'])
    logger.info("   Shape: %d righe x %d colonne", df.shape[0], df.shape[1])
    logger.info("   Range date: %s -> %s", df['timestamps'].min(), df['timestamps'].max())
    logger.info("   NOTA: sentiment già shiftato a T-1 da run_unified.py")

    emb_cols = [c for c in df.columns if c.startswith('kronos_emb_')]
    logger.info("   Embedding Kronos: %d dimensioni", len(emb_cols))
    logger.info("   Feature tecniche: %d", len(TECHNICAL_FEATURES))
    logger.info("   Feature sentiment: %d", len(SENTIMENT_FEATURES))

    df = df.dropna(subset=TECHNICAL_FEATURES).reset_index(drop=True)

    logger.info("   Righe dopo dropna: %d", len(df))

    df = df.sort_values('timestamps').reset_index(drop=True)
    return df, emb_cols


def reduce_embeddings(emb_data, variance_threshold=PCA_VARIANCE_THRESHOLD,
                      min_comp=PCA_MIN_COMPONENTS, max_comp=PCA_MAX_COMPONENTS):
    """Fit PCA on emb_data selecting components to explain `variance_threshold`.
    
    MUST be called inside each fold with train data only.
    Returns (emb_pca, pca_object, explained_variance_ratio, n_components).
    """
    from sklearn.decomposition import PCA
    n_possible = min(max_comp, emb_data.shape[0], emb_data.shape[1])
    if n_possible < 2:
        raise ValueError(f"Too few dimensions for PCA: {n_possible}")

    # Fit once with max possible components to measure explained variance
    pca_full = PCA(n_components=n_possible, random_state=42)
    pca_full.fit(emb_data)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_opt = int(np.searchsorted(cumsum, variance_threshold) + 1)
    low = min(min_comp, n_possible)
    n_opt = min(max(low, n_opt), n_possible)

    pca = PCA(n_components=n_opt, random_state=42)
    emb_pca = pca.fit_transform(emb_data)
    explained = pca.explained_variance_ratio_.sum()
    return emb_pca, pca, explained, n_opt


def train_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight=None):
    import xgboost as xgb
    params = {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': 42,
    }
    if scale_pos_weight:
        params['scale_pos_weight'] = scale_pos_weight

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def evaluate_model(model, X_test, y_test, label="Modello"):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score,
                                 confusion_matrix)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"\n   {label}:")
    logger.info(f"   Accuracy:  {acc:.4f}")
    logger.info(f"   Precision: {prec:.4f}")
    logger.info(f"   Recall:    {rec:.4f}")
    logger.info(f"   F1:        {f1:.4f}")
    logger.info(f"   ROC-AUC:   {auc:.4f}")
    logger.info(f"   Confusion Matrix:")
    logger.info(f"      TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    logger.info(f"      FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    return y_prob, y_pred


def granger_causality_test(df):
    from statsmodels.tsa.stattools import grangercausalitytests

    logger.info("\n" + "=" * 60)
    logger.info("FASE 0: GRANGER CAUSALITY TEST (Sentiment → BTC Return 1d)")
    logger.info("=" * 60)

    data = df[['close'] + SENTIMENT_FEATURES].copy()
    data['return_1d'] = data['close'].pct_change()
    data = data[['return_1d'] + SENTIMENT_FEATURES].dropna()
    logger.info(f"   Righe valide: {len(data)}")

    results = {}
    for col in SENTIMENT_FEATURES:
        try:
            gc = grangercausalitytests(data[['return_1d', col]], maxlag=5, verbose=False)
            pvals = [round(gc[lag][0]['ssr_chi2test'][1], 4) for lag in range(1, 6)]
            results[col] = {'min_p': min(pvals), 'pvals': pvals}
        except Exception as e:
            results[col] = {'min_p': 1.0, 'error': str(e)}

    logger.info(f"\n   {'Feature':<35} {'min p-value':<15} {'Granger-causes close?'}")
    logger.info(f"   {'-'*70}")
    for col, res in results.items():
        sig = "YES" if res['min_p'] < 0.05 else "NO"
        logger.info(f"   {col:<35} {res['min_p']:<15.4f} {sig}")

    n_sig = sum(1 for r in results.values() if r['min_p'] < 0.05)
    logger.info(f"\n   {n_sig}/{len(SENTIMENT_FEATURES)} feature sentiment Granger-causano BTC close (α=0.05)")
    return results


def regime_test(close_prices, emb_data, tech_data, sent_data, timestamps, scale_pos_weight):
    """Train su anno N, test su anno N+1 per rilevare regime change."""
    logger.info("\n" + "=" * 60)
    logger.info("FASE 4.1b: REGIME TEST (train year N → test year N+1)")
    logger.info("=" * 60)

    years = sorted(set(pd.DatetimeIndex(timestamps).year))
    results = []
    for y in years:
        if y + 1 not in years:
            continue
        label = f"{y}→{y+1}"
        train_mask = pd.DatetimeIndex(timestamps).year == y
        test_mask = pd.DatetimeIndex(timestamps).year == y + 1
        if train_mask.sum() < 30 or test_mask.sum() < 10:
            logger.info(f"   {label}: dati insufficienti, skip")
            continue

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        y_train = compute_target_for_idx(close_prices, train_idx)
        y_test = compute_target_for_idx(close_prices, test_idx)
        train_valid = y_train >= 0
        test_valid = y_test >= 0
        y_train, y_test = y_train[train_valid], y_test[test_valid]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.info(f"   {label}: classi insufficienti, skip")
            continue

        emb_pca_train, pca, _, n_opt = reduce_embeddings(emb_data[train_idx][train_valid])
        emb_pca_test = pca.transform(emb_data[test_idx][test_valid])

        X_train = np.concatenate([tech_data[train_idx][train_valid], emb_pca_train, sent_data[train_idx][train_valid]], axis=1)
        X_test = np.concatenate([tech_data[test_idx][test_valid], emb_pca_test, sent_data[test_idx][test_valid]], axis=1)

        model = train_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        results.append({'regime': label, 'n_train': n_train, 'n_test': n_test,
                        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc})

    if results:
        logger.info(f"\n   {'Regime':<15} {'Train':<8} {'Test':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
        logger.info(f"   {'-'*70}")
        for r in results:
            logger.info(f"   {r['regime']:<15} {r['n_train']:<8} {r['n_test']:<8} {r['accuracy']:.4f}  {r['precision']:.4f}  {r['recall']:.4f}  {r['f1']:.4f}  {r['roc_auc']:.4f}")

        if len(results) >= 2:
            acc_delta = results[-1]['accuracy'] - results[0]['accuracy']
            auc_delta = results[-1]['roc_auc'] - results[0]['roc_auc']
            logger.info(f"\n   Delta accuracy ({results[0]['regime']} → {results[-1]['regime']}): {acc_delta:+.4f}")
            logger.info(f"   Delta ROC-AUC: {auc_delta:+.4f}")
            if abs(acc_delta) > 0.15:
                logger.warning(f"   WARNING: accuracy delta > 15% — possibile regime change non catturato")
    else:
        logger.info("   Nessun regime test valido.")
    return results


def timeseries_cv(close_prices, emb_data, tech_data, sent_data, timestamps):
    from sklearn.model_selection import TimeSeriesSplit

    logger.info("\n" + "=" * 60)
    logger.info("FASE 4.1: TIME-SERIES CROSS-VALIDATION (PCA + target per fold)")
    logger.info("=" * 60)

    total = len(tech_data)
    n_splits = min(CV_MAX_SPLITS, max(CV_MIN_SPLITS,
                   (total - CV_TEST_MIN_DAYS) // (CV_TEST_MIN_DAYS // 2)))
    test_size = total // (n_splits + 1)
    logger.info(f"   Total rows: {total}, n_splits: {n_splits}, test_size: ~{test_size}")

    # Compute pos/neg ratio on full data for scale_pos_weight
    full_target = compute_target_for_idx(close_prices, np.arange(total))
    valid_mask = full_target >= 0
    full_target_valid = full_target[valid_mask]
    n_pos = full_target_valid.sum()
    n_neg = len(full_target_valid) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"   Scale pos weight: {scale_pos_weight:.2f} ({n_pos} positivi / {n_neg} negativi)")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    cv_naive = []

    logger.info(f"\n   {'Fold':<6} {'Train':<8} {'Test':<8} {'Score':<10} {'Naive':<10} {'PCA_comp':<10}")
    logger.info(f"   {'-'*55}")

    MIN_TRAIN_SAMPLES = 60

    for i, (train_idx, test_idx) in enumerate(tscv.split(tech_data)):
        # Target computed from close[train_idx] ONLY
        y_train = compute_target_for_idx(close_prices, train_idx)
        y_test = compute_target_for_idx(close_prices, test_idx)
        train_valid = y_train >= 0
        test_valid = y_test >= 0

        if train_valid.sum() < MIN_TRAIN_SAMPLES:
            logger.info(f"   Fold {i+1}: saltato (train troppo piccolo: {train_valid.sum()} < {MIN_TRAIN_SAMPLES})")
            continue

        # PCA fit ONLY on train_idx — adaptive components
        emb_pca_train, pca, explained, n_opt = reduce_embeddings(
            emb_data[train_idx][train_valid]
        )
        emb_pca_test = pca.transform(emb_data[test_idx][test_valid])

        y_train, y_test = y_train[train_valid], y_test[test_valid]

        X_train = np.concatenate([
            tech_data[train_idx][train_valid],
            emb_pca_train,
            sent_data[train_idx][train_valid]
        ], axis=1)
        X_test = np.concatenate([
            tech_data[test_idx][test_valid],
            emb_pca_test,
            sent_data[test_idx][test_valid]
        ], axis=1)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.info(f"   Fold {i+1}: saltato (classi insufficienti dopo masking)")
            continue

        model = train_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight)
        score = model.score(X_test, y_test)
        cv_scores.append(score)

        majority = Counter(y_train).most_common(1)[0][0]
        naive_acc = (y_test == majority).mean()
        cv_naive.append(naive_acc)

        train_start = str(timestamps[train_idx[0]])[:10]
        train_end = str(timestamps[train_idx[-1] - TARGET_HORIZON])[:10]
        test_start = str(timestamps[test_idx[0]])[:10]
        test_end = str(timestamps[test_idx[-1] - TARGET_HORIZON])[:10]

        logger.info(f"   Fold {i+1:<4} {train_valid.sum():<8} {test_valid.sum():<8} {score:.4f}  {naive_acc:.4f}  {n_opt:<8}")
        logger.info(f"         {train_start}->{train_end} | {test_start}->{test_end}  (var={explained:.2%})")

    if cv_scores:
        logger.info(f"\n   CV Score medio:     {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
        logger.info(f"   Baseline naive:     {np.mean(cv_naive):.4f} +/- {np.std(cv_naive):.4f}")
        logger.info(f"   Delta (XGB - naive): {np.mean(cv_scores) - np.mean(cv_naive):+.4f}")
    else:
        logger.info("\n   CV: nessun fold valido.")
    return cv_scores, scale_pos_weight


def train_and_evaluate(close_prices, emb_data, tech_data, sent_data, timestamps,
                        scale_pos_weight, label="Modello"):
    from sklearn.metrics import f1_score

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{label} (PCA adattivo + target per fold)")
    logger.info(f"{'=' * 60}")

    total = len(tech_data)
    train_end = int(total * 0.60)
    val_end = int(total * 0.80)

    train_idx = np.arange(train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, total)

    # PCA on train only — adaptive
    emb_pca_train, pca, explained, n_opt = reduce_embeddings(emb_data[train_idx])

    y_train = compute_target_for_idx(close_prices, train_idx)
    y_val = compute_target_for_idx(close_prices, val_idx)
    y_test = compute_target_for_idx(close_prices, test_idx)
    train_valid = y_train >= 0
    val_valid = y_val >= 0
    test_valid = y_test >= 0
    y_train, y_val, y_test = y_train[train_valid], y_val[val_valid], y_test[test_valid]

    emb_pca_val = pca.transform(emb_data[val_idx][val_valid])
    emb_pca_test = pca.transform(emb_data[test_idx][test_valid])

    X_train = np.concatenate([
        tech_data[train_idx][train_valid],
        emb_pca_train[train_valid],
        sent_data[train_idx][train_valid]
    ], axis=1)
    X_val = np.concatenate([
        tech_data[val_idx][val_valid],
        emb_pca_val,
        sent_data[val_idx][val_valid]
    ], axis=1)
    X_test = np.concatenate([
        tech_data[test_idx][test_valid],
        emb_pca_test,
        sent_data[test_idx][test_valid]
    ], axis=1)

    logger.info(f"\n   Train: {X_train.shape[0]} ({str(timestamps[train_idx[0]])[:10]} -> {str(timestamps[train_idx[train_valid.sum()-1]])[:10]})")
    logger.info(f"   Val:   {X_val.shape[0]} ({str(timestamps[val_idx[0]])[:10]} -> {str(timestamps[val_idx[val_valid.sum()-1]])[:10]})")
    logger.info(f"   Test:  {X_test.shape[0]} ({str(timestamps[test_idx[0]])[:10]} -> {str(timestamps[test_idx[test_valid.sum()-1]])[:10]})")
    logger.info(f"   PCA: {n_opt} componenti (varianza {explained:.2%})")
    logger.info(f"   (ultimi {TARGET_HORIZON} giorni esclusi da ogni set — target non calcolabile)")

    model = train_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight)
    y_prob, y_pred = evaluate_model(model, X_test, y_test, label)

    # Optimal threshold on validation set
    y_prob_val = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.50, 0.90, 0.05)
    best_thresh, best_f1 = 0.75, 0.0
    for t in thresholds:
        y_pred_t = (y_prob_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    logger.info(f"\n   Soglia ottimale (validation F1): {best_thresh:.2f} (F1={best_f1:.4f})")

    return model, pca, X_train, X_test, y_train, y_test, y_prob, timestamps[test_idx][test_valid], n_opt, best_thresh


def shap_analysis(model, X_test, feature_cols):
    logger.info("\n" + "=" * 60)
    logger.info("FASE 4.3: SHAP FEATURE IMPORTANCE")
    logger.info("=" * 60)

    try:
        import shap
        logger.info("\n   Calcolo SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])

        shap_sum = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(shap_sum)[-20:][::-1]

        logger.info(f"\n   Top 20 feature per importanza SHAP:")
        logger.info(f"   {'#':<4} {'Feature':<35} {'|SHAP|':<10} {'Gruppo'}")
        logger.info(f"   {'-'*60}")
        for rank, idx in enumerate(top_idx, 1):
            c = feature_cols[idx]
            if 'kronos_emb' in c or 'emb_pca' in c:
                g = 'embedding'
            elif c in SENTIMENT_FEATURES:
                g = 'sentiment'
            else:
                g = 'tecnica'
            logger.info(f"   {rank:<4} {c:<35} {shap_sum[idx]:<10.6f} {g}")

        kronos_imp = sum(shap_sum[i] for i, c in enumerate(feature_cols)
                         if 'kronos_emb' in c or 'emb_pca' in c)
        tech_imp = sum(shap_sum[i] for i, c in enumerate(feature_cols)
                       if c in TECHNICAL_FEATURES)
        sent_imp = sum(shap_sum[i] for i, c in enumerate(feature_cols)
                       if c in SENTIMENT_FEATURES)
        total = kronos_imp + tech_imp + sent_imp

        if total > 0:
            logger.info(f"\n   Importanza per gruppo:")
            logger.info(f"   Embeddings: {kronos_imp / total * 100:.1f}%")
            logger.info(f"   Tecnici:    {tech_imp / total * 100:.1f}%")
            logger.info(f"   Sentiment:  {sent_imp / total * 100:.1f}%")
    except ImportError:
        logger.info("\n   SHAP non disponibile. Uso importanza nativa XGBoost:")
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-10:][::-1]
        for rank, idx in enumerate(top_idx, 1):
            logger.info(f"   {rank}. {feature_cols[idx]:35} {importance[idx]:.6f}")


def backtest_simulation(close_prices, emb_data, tech_data, sent_data, timestamps,
                        scale_pos_weight, threshold=0.75):
    logger.info("\n" + "=" * 60)
    logger.info("FASE 4.4: BACKTESTING SIMULATO (P&L da prezzi reali)")
    logger.info("=" * 60)

    import xgboost as xgb

    split_idx = max(60, len(tech_data) - BACKTEST_DAYS)
    # Stop TARGET_HORIZON days early so every trade has a real 3-day forward exit
    end_idx = len(tech_data) - TARGET_HORIZON
    n_days = end_idx - split_idx
    logger.info(f"\n   Walk-forward: {n_days} giorni (stop {TARGET_HORIZON}gg prima per exit reale)")
    logger.info(f"   Soglia trade: prob > {threshold}")
    logger.info(f"   Capitale: {TRADE_AMOUNT}€, Commissione: {FEE}€/trade")
    logger.info(f"   Target calcolato dentro ogni fold da close[train_idx]")
    logger.info(f"   P&L calcolato su prezzo reale, non forfettario")

    trades = []
    for i in range(split_idx, end_idx):
        train_idx = np.arange(i)

        # Target from close[train_idx] ONLY
        y_train = compute_target_for_idx(close_prices, train_idx)
        train_valid = y_train >= 0
        y_train = y_train[train_valid]
        train_idx_valid = train_idx[train_valid]

        if len(np.unique(y_train)) < 2:
            continue

        # PCA on train only — adaptive
        emb_pca_train, pca, _, n_opt = reduce_embeddings(
            emb_data[train_idx_valid]
        )
        emb_pca_test = pca.transform(emb_data[i:i+1])

        X_train = np.concatenate([
            tech_data[train_idx_valid], emb_pca_train,
            sent_data[train_idx_valid]
        ], axis=1)

        model_i = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            eval_metric='logloss', verbosity=0, random_state=42,
            scale_pos_weight=scale_pos_weight,
        )
        model_i.fit(X_train, y_train, verbose=False)

        prob = model_i.predict_proba(np.concatenate([
            tech_data[i:i+1], emb_pca_test, sent_data[i:i+1]
        ], axis=1))[0, 1]

        if prob > threshold:
            entry_price = close_prices[i]
            exit_idx = i + TARGET_HORIZON
            exit_price = close_prices[exit_idx]
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_eur = TRADE_AMOUNT * pnl_pct - FEE

            # Ground truth: was the price rise > threshold?
            actual_ret = (close_prices[i + TARGET_HORIZON] - close_prices[i]) / close_prices[i] * 100

            trades.append({
                'date': str(timestamps[i])[:10],
                'close': entry_price,
                'prob': prob,
                'target_actual': 1 if actual_ret > TARGET_THRESHOLD_PCT else 0,
                'profit': round(pnl_eur, 2),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(pnl_pct * 100, 2),
            })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    logger.info(f"\n   Trade eseguiti: {len(trades)}")

    if len(trades) > 0:
        win_rate = (trades_df['profit'] > 0).mean() * 100
        total_pnl = trades_df['profit'].sum()
        logger.info(f"   Win rate: {win_rate:.1f}%")
        logger.info(f"   P&L totale: {total_pnl:+.2f}€")
        logger.info(f"   Avg trade: {trades_df['profit'].mean():+.2f}€")
        logger.info(f"   Max win: {trades_df['profit'].max():+.2f}€ | Max loss: {trades_df['profit'].min():+.2f}€")

        logger.info(f"\n   Ultimi 10 trade:")
        logger.info(f"   {'Data':<14} {'Entry':<10} {'Exit':<10} {'Ret%':<8} {'Prob':<8} {'Profit':<10}")
        logger.info(f"   {'-'*65}")
        for _, t in trades_df.tail(10).iterrows():
            r = 'WIN ' if t['profit'] > 0 else 'LOSS'
            logger.info(f"   {t['date']:<14} {t['entry_price']:<10.2f} {t['exit_price']:<10.2f} "
                  f"{t['return_pct']:<+7.2f}% {t['prob']:<8.3f} {t['profit']:+7.2f}€ ({r})")

        logger.info(f"\n   {'POSITIVO' if total_pnl > 0 else 'NEGATIVO'}: {total_pnl:+.2f}€")
    else:
        logger.info("   Nessun trade (soglia non raggiunta).")

    return trades_df


def save_model_artifacts(model, pca, feature_names, metadata):
    """Serialize model, PCA, and metadata to disk for inference."""
    import joblib, shutil
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(MODEL_DIR, f"xgboost_{timestamp}")
    joblib.dump(model, f"{base}_model.pkl")
    joblib.dump(pca, f"{base}_pca.pkl")
    joblib.dump({"feature_names": feature_names, "metadata": metadata}, f"{base}_info.pkl")
    # Keep a "latest" copy
    for name in ["model", "pca", "info"]:
        src = f"{base}_{name}.pkl"
        dst = os.path.join(MODEL_DIR, f"xgboost_latest_{name}.pkl")
        shutil.copy2(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    logger.info(f"\n   Modello salvato in: {base}_*.pkl")
    logger.info(f"   Latest: {MODEL_DIR}/xgboost_latest_*.pkl")
    return base


def load_model_artifacts(path_prefix=None):
    """Load model, PCA, and info from disk.
    
    Args:
        path_prefix: e.g. "feature_store/models/xgboost_20260530_120000"
                     If None, loads "latest".
    """
    import joblib
    if path_prefix is None:
        base = MODEL_DIR
        model = joblib.load(os.path.join(base, "xgboost_latest_model.pkl"))
        pca = joblib.load(os.path.join(base, "xgboost_latest_pca.pkl"))
        info = joblib.load(os.path.join(base, "xgboost_latest_info.pkl"))
    else:
        model = joblib.load(f"{path_prefix}_model.pkl")
        pca = joblib.load(f"{path_prefix}_pca.pkl")
        info = joblib.load(f"{path_prefix}_info.pkl")
    return model, pca, info["feature_names"], info["metadata"]


def main():
    df, emb_cols = load_data()

    # Estrai arrays numpy una volta
    close_prices = df['close'].values.astype(np.float64)
    timestamps = df['timestamps'].values
    emb_data = df[emb_cols].values.astype(np.float32)
    tech_data = df[TECHNICAL_FEATURES].values.astype(np.float32)
    sent_data = df[[c for c in SENTIMENT_FEATURES if c in df.columns]].fillna(0.0).values.astype(np.float32)

    baseline_cols = [c for c in TECHNICAL_FEATURES + SENTIMENT_FEATURES if c in df.columns]
    logger.info(f"\n   Baseline features (tecnici + sentiment): {len(baseline_cols)}")

    granger_causality_test(df)

    cv_scores, scale_pos_weight = timeseries_cv(
        close_prices, emb_data, tech_data, sent_data, timestamps,
    )

    regime_test(close_prices, emb_data, tech_data, sent_data, timestamps, scale_pos_weight)

    model, pca, X_train, X_test, y_train, y_test, y_prob, test_ts, n_opt, best_thresh = train_and_evaluate(
        close_prices, emb_data, tech_data, sent_data, timestamps,
        scale_pos_weight,
        label="FASE 4.2: TRAINING FINALE XGBOOST (PCA adattivo + target per fold)"
    )
    pca_cols = [f'emb_pca_{j}' for j in range(n_opt)]
    full_feature_names = baseline_cols + pca_cols

    shap_analysis(model, X_test, full_feature_names)

    # Serializza modello, PCA e feature names
    metadata = {
        "trained_on": str(datetime.now()),
        "n_rows": len(df),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "pca_n_components": n_opt,
        "target_horizon": TARGET_HORIZON,
        "target_threshold_pct": TARGET_THRESHOLD_PCT,
        "optimal_threshold": best_thresh,
        "technical_features": TECHNICAL_FEATURES,
        "sentiment_features": SENTIMENT_FEATURES,
        "embedding_dim": len(emb_cols),
    }
    save_model_artifacts(model, pca, full_feature_names, metadata)

    trades_df = backtest_simulation(
        close_prices, emb_data, tech_data, sent_data, timestamps,
        scale_pos_weight, threshold=best_thresh,
    )

    logger.info("\n" + "=" * 60)
    logger.info("FASE 4 COMPLETATA")
    logger.info("=" * 60)
    logger.info(f"\nRiepilogo:")
    logger.info(f"  Feature: {len(full_feature_names)} ({len(TECHNICAL_FEATURES)} tecniche + "
          f"{len(SENTIMENT_FEATURES)} sentiment + {n_opt} embedding PCA)")
    logger.info(f"  CV Score medio: {np.mean(cv_scores):.4f}" if cv_scores else "  CV: nessun fold valido")
    if len(trades_df) > 0:
        logger.info(f"  Trade: {len(trades_df)} | Win rate: {(trades_df['profit']>0).mean()*100:.1f}% | "
              f"P&L: {trades_df['profit'].sum():+.2f}€")


if __name__ == "__main__":
    main()
