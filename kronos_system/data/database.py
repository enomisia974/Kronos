import sqlite3
import logging
from datetime import datetime
from typing import Optional
from kronos_system.config import DB_PATH

logger = logging.getLogger(__name__)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS assets (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            asset_type  TEXT NOT NULL DEFAULT 'crypto',
            is_active   INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS historical_prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id    TEXT NOT NULL REFERENCES assets(id),
            date        TEXT NOT NULL,
            open        REAL NOT NULL,
            high        REAL NOT NULL,
            low         REAL NOT NULL,
            close       REAL NOT NULL,
            volume      REAL NOT NULL,
            UNIQUE(asset_id, date)
        );

        CREATE TABLE IF NOT EXISTS technical_indicators (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id    TEXT NOT NULL REFERENCES assets(id),
            date        TEXT NOT NULL,
            ema_5       REAL,
            ema_10      REAL,
            ema_20      REAL,
            rsi_14      REAL,
            bb_width    REAL,
            atr_14      REAL,
            volume_ratio REAL,
            target      INTEGER,
            UNIQUE(asset_id, date)
        );

        CREATE TABLE IF NOT EXISTS news_sentiment (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id    TEXT NOT NULL REFERENCES assets(id),
            date        TEXT NOT NULL,
            headline    TEXT NOT NULL,
            score       REAL NOT NULL,
            label       TEXT NOT NULL,
            source      TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS daily_sentiment (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id    TEXT NOT NULL REFERENCES assets(id),
            date        TEXT NOT NULL,
            score       REAL NOT NULL DEFAULT 0.0,
            count       INTEGER NOT NULL DEFAULT 0,
            pos_ratio   REAL NOT NULL DEFAULT 0.0,
            neg_ratio   REAL NOT NULL DEFAULT 0.0,
            UNIQUE(asset_id, date)
        );

        CREATE TABLE IF NOT EXISTS model_predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id        TEXT NOT NULL REFERENCES assets(id),
            date            TEXT NOT NULL,
            probability     REAL NOT NULL,
            signal          TEXT NOT NULL,
            confidence      REAL NOT NULL,
            model_version   TEXT NOT NULL,
            kronos_proj_ret REAL,
            sentiment_score REAL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id        TEXT NOT NULL REFERENCES assets(id),
            fold            INTEGER NOT NULL,
            accuracy        REAL NOT NULL,
            precision       REAL,
            recall          REAL,
            f1_score        REAL,
            n_trades        INTEGER NOT NULL DEFAULT 0,
            baseline_accuracy REAL NOT NULL,
            n_components_pca INTEGER,
            timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS walkforward_folds (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id    TEXT NOT NULL REFERENCES assets(id),
            fold        INTEGER NOT NULL,
            train_start TEXT NOT NULL,
            train_end   TEXT NOT NULL,
            test_start  TEXT NOT NULL,
            test_end    TEXT NOT NULL,
            n_train     INTEGER NOT NULL,
            n_test      INTEGER NOT NULL,
            UNIQUE(asset_id, fold)
        );

        CREATE INDEX IF NOT EXISTS idx_hp_asset_date ON historical_prices(asset_id, date);
        CREATE INDEX IF NOT EXISTS idx_ti_asset_date ON technical_indicators(asset_id, date);
        CREATE INDEX IF NOT EXISTS idx_ns_asset_date ON news_sentiment(asset_id, date);
        CREATE INDEX IF NOT EXISTS idx_ds_asset_date ON daily_sentiment(asset_id, date);
        CREATE INDEX IF NOT EXISTS idx_mp_asset_date ON model_predictions(asset_id, date);
        """)
        conn.commit()

        cursor = conn.execute("SELECT COUNT(*) FROM assets")
        if cursor.fetchone()[0] == 0:
            from kronos_system.config import ASSETS
            for aid in ASSETS:
                atype = "crypto" if "EUR" in aid else ("commodity" if aid == "GLD" else "stock")
                conn.execute("INSERT OR IGNORE INTO assets(id, name, asset_type) VALUES (?, ?, ?)",
                             (aid, aid, atype))
            conn.commit()
            logger.info("Populated %d assets into catalog", len(ASSETS))
    finally:
        conn.close()


def write_prices(asset_id: str, rows: list[dict]):
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO historical_prices(asset_id, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(asset_id, r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"]) for r in rows]
        )
        conn.commit()
        logger.info("Wrote %d price rows for %s", len(rows), asset_id)
    finally:
        conn.close()


def write_indicators(asset_id: str, rows: list[dict]):
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO technical_indicators(asset_id, date, ema_5, ema_10, ema_20, "
            "rsi_14, bb_width, atr_14, volume_ratio, target) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(asset_id, r["date"], r.get("ema_5"), r.get("ema_10"), r.get("ema_20"),
              r.get("rsi_14"), r.get("bb_width"), r.get("atr_14"), r.get("volume_ratio"), r.get("target"))
             for r in rows]
        )
        conn.commit()
    finally:
        conn.close()


def write_sentiment_articles(asset_id: str, rows: list[dict]):
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT INTO news_sentiment(asset_id, date, headline, score, label, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(asset_id, r["date"], r["headline"], r["score"], r["label"], r["source"]) for r in rows]
        )
        conn.commit()
    finally:
        conn.close()


def write_daily_sentiment(asset_id: str, rows: list[dict]):
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_sentiment(asset_id, date, score, count, pos_ratio, neg_ratio) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(asset_id, r["date"], r["score"], r["count"], r["pos_ratio"], r["neg_ratio"]) for r in rows]
        )
        conn.commit()
    finally:
        conn.close()


def write_prediction(asset_id: str, pred: dict):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO model_predictions(asset_id, date, probability, signal, confidence, "
            "model_version, kronos_proj_ret, sentiment_score) VALUES (?,?,?,?,?,?,?,?)",
            (asset_id, pred["date"], pred["probability"], pred["signal"], pred["confidence"],
             pred.get("model_version", "1.0.0"), pred.get("kronos_proj_ret"), pred.get("sentiment_score"))
        )
        conn.commit()
    finally:
        conn.close()


def write_metrics(asset_id: str, fold: int, metrics: dict):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO model_metrics(asset_id, fold, accuracy, precision, recall, f1_score, "
            "n_trades, baseline_accuracy, n_components_pca) VALUES (?,?,?,?,?,?,?,?,?)",
            (asset_id, fold, metrics["accuracy"], metrics.get("precision"), metrics.get("recall"),
             metrics.get("f1_score"), metrics["n_trades"], metrics["baseline_accuracy"],
             metrics.get("n_components_pca"))
        )
        conn.commit()
    finally:
        conn.close()


def write_fold(asset_id: str, fold: int, fold_info: dict):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO walkforward_folds(asset_id, fold, train_start, train_end, "
            "test_start, test_end, n_train, n_test) VALUES (?,?,?,?,?,?,?,?)",
            (asset_id, fold, fold_info["train_start"], fold_info["train_end"],
             fold_info["test_start"], fold_info["test_end"],
             fold_info["n_train"], fold_info["n_test"])
        )
        conn.commit()
    finally:
        conn.close()


def read_prices(asset_id: str, days: int = 365) -> list[sqlite3.Row]:
    conn = get_conn()
    try:
        cursor = conn.execute(
            "SELECT * FROM historical_prices WHERE asset_id = ? ORDER BY date DESC LIMIT ?",
            (asset_id, days)
        )
        return cursor.fetchall()
    finally:
        conn.close()


def read_latest_prediction(asset_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    try:
        cursor = conn.execute(
            "SELECT * FROM model_predictions WHERE asset_id = ? ORDER BY date DESC LIMIT 1",
            (asset_id,)
        )
        return cursor.fetchone()
    finally:
        conn.close()


def read_prediction_history(asset_id: str, days: int = 60) -> list[sqlite3.Row]:
    conn = get_conn()
    try:
        cursor = conn.execute(
            "SELECT * FROM model_predictions WHERE asset_id = ? ORDER BY date DESC LIMIT ?",
            (asset_id, days)
        )
        return cursor.fetchall()
    finally:
        conn.close()


def read_rolling_metrics(asset_id: str) -> list[sqlite3.Row]:
    conn = get_conn()
    try:
        cursor = conn.execute(
            "SELECT * FROM model_metrics WHERE asset_id = ? ORDER BY fold ASC", (asset_id,)
        )
        return cursor.fetchall()
    finally:
        conn.close()


def read_asset_catalog() -> list[sqlite3.Row]:
    conn = get_conn()
    try:
        cursor = conn.execute("SELECT * FROM assets WHERE is_active = 1 ORDER BY id")
        return cursor.fetchall()
    finally:
        conn.close()


def read_latest_sentiment(asset_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    try:
        cursor = conn.execute(
            "SELECT * FROM daily_sentiment WHERE asset_id = ? ORDER BY date DESC LIMIT 1",
            (asset_id,)
        )
        return cursor.fetchone()
    finally:
        conn.close()
