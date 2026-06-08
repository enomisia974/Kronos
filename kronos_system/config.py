import os
from datetime import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "kronos_system", "kronos.db")
MODELS_DIR = os.path.join(BASE_DIR, "kronos_system", "models")
ASSETS = ["BTC-EUR", "ETH-EUR", "AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "SPY", "GLD"]

YFINANCE_PERIOD = "2y"
YFINANCE_INTERVAL = "1d"
KRONOS_LOOKBACK = 90
KRONOS_PRED_LEN = 14
TARGET_THRESHOLD_PCT = 2.5
TARGET_HORIZON_DAYS = 3
WALK_FORWARD_FOLDS = 20
WALK_FORWARD_WINDOW = 60
WALK_FORWARD_GAP = 3

PCA_VARIANCE_THRESHOLD = 0.90
PCA_MAX_COMPONENTS = 50
PCA_MIN_COMPONENTS = 5

XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 4
XGB_LR = 0.03
XGB_SUBSAMPLE = 0.7
XGB_COLSAMPLE = 0.7
XGB_RANDOM_STATE = 42

TECH_FEATURES = ["ema_5", "ema_10", "ema_20", "rsi_14", "bb_width", "atr_14", "volume_ratio"]
SENTIMENT_FEATURES = ["sentiment_score", "sentiment_count", "sentiment_pos_ratio"]

CRON_HOUR = 0
CRON_MINUTE = 5
CRON_TIMEZONE = "UTC"

RSS_FEEDS = {
    "CoinDesk": "https://feeds.feedburner.com/CoinDesk",
    "CoinTelegraph": "https://cointelegraph.com/rss",
}

MODEL_VERSION = "1.0.0"
