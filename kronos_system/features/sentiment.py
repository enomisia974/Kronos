import logging
import feedparser
import pandas as pd
from transformers import pipeline

from kronos_system.config import RSS_FEEDS

logger = logging.getLogger(__name__)

_sentiment_pipeline = None


def _get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info("Loading FinBERT pipeline (first call)...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    return _sentiment_pipeline


def fetch_articles(asset_keywords: list[str] | None = None) -> list[dict]:
    """Fetch RSS articles. On failure, returns empty list (never raises)."""
    articles = []
    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:50]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                if asset_keywords:
                    title_lower = title.lower()
                    if not any(kw.lower() in title_lower for kw in asset_keywords):
                        continue
                articles.append({
                    "title": title,
                    "source": name,
                    "date_str": entry.get("published", ""),
                })
            logger.debug("Fetched %d articles from %s", len(feed.entries[:50]), name)
        except Exception as e:
            logger.warning("RSS feed %s failed: %s", name, e)
    if not articles:
        logger.warning("No articles fetched from any RSS source")
    return articles


def score_articles(articles: list[dict], batch_size: int = 32) -> list[dict]:
    """Score articles with FinBERT. Returns articles with 'score' and 'label' attached."""
    if not articles:
        return articles
    pipe = _get_pipeline()
    for i in range(0, len(articles), batch_size):
        batch = [a["title"] for a in articles[i:i + batch_size]]
        try:
            results = pipe(batch)
            for res, article in zip(results, articles[i:i + batch_size]):
                label = res["label"]
                score = res["score"]
                if label == "positive":
                    article["score"] = score
                elif label == "negative":
                    article["score"] = -score
                else:
                    article["score"] = 0.0
                article["label"] = label
        except Exception as e:
            logger.error("FinBERT scoring batch failed: %s", e)
            for article in articles[i:i + batch_size]:
                article["score"] = 0.0
                article["label"] = "neutral"
    return articles


def aggregate_daily(articles: list[dict]) -> pd.DataFrame:
    """Aggregate scored articles into daily DataFrame.
    
    Returns columns: [date, score, count, pos_ratio, neg_ratio]
    If no articles, returns empty DataFrame.
    """
    if not articles:
        return pd.DataFrame()
    d = pd.DataFrame(articles)
    d["date"] = pd.to_datetime(d["date_str"], errors="coerce").dt.date
    d = d.dropna(subset=["date"])
    if d.empty:
        return pd.DataFrame()
    daily = d.groupby("date").agg(
        score=("score", "mean"),
        count=("title", "count"),
        pos_ratio=("label", lambda x: (x == "positive").mean()),
        neg_ratio=("label", lambda x: (x == "negative").mean()),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date")


def run_sentiment_pipeline(asset_id: str, asset_keywords: list[str] | None = None) -> pd.DataFrame:
    """Full sentiment pipeline: fetch → score → aggregate.
    
    Returns daily aggregated DataFrame.
    On complete failure, returns empty DataFrame (never crashes).
    """
    try:
        articles = fetch_articles(asset_keywords)
        articles = score_articles(articles)
        daily = aggregate_daily(articles)
        if daily.empty:
            logger.info("No sentiment data for %s, returning empty", asset_id)
        return daily
    except Exception as e:
        logger.error("Sentiment pipeline failed for %s: %s", asset_id, e, exc_info=True)
        return pd.DataFrame()
