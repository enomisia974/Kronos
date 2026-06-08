import os
import warnings
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import pipeline
import time

warnings.filterwarnings("ignore")

RSS_FEEDS = {
    'CoinDesk': 'https://feeds.feedburner.com/CoinDesk',
    'CoinTelegraph': 'https://cointelegraph.com/rss',
}

SENTIMENT_MODEL = "ProsusAI/finbert"
OUTPUT_DIR = "feature_store"
OUTPUT_FILE = "news_sentiment_daily.csv"


def fetch_news_from_rss(feeds, max_per_feed=100):
    articles = []
    for source_name, feed_url in feeds.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get('title', '').strip()
                if not title:
                    continue
                pub_date = entry.get('published', entry.get('updated', ''))
                articles.append({
                    'title': title,
                    'date_str': pub_date,
                    'source': source_name,
                    'link': entry.get('link', ''),
                })
        except Exception as e:
            print(f"   [WARN] Errore fetch {source_name}: {e}")
    return articles


def parse_news_dates(articles):
    parsed = []
    for art in articles:
        dt = None
        for fmt in [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
        ]:
            try:
                dt = datetime.strptime(art['date_str'], fmt)
                break
            except (ValueError, TypeError):
                continue
        if dt is None:
            dt = datetime.now()
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        art['datetime'] = dt
        art['date'] = dt.date()
        parsed.append(art)
    return parsed


def analyze_sentiment_batch(articles, sentiment_pipeline, batch_size=32):
    titles = [a['title'] for a in articles]
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        try:
            results = sentiment_pipeline(batch)
            for j, res in enumerate(results):
                idx = i + j
                label = res['label']
                score = res['score']
                if label == 'positive':
                    articles[idx]['sentiment_score'] = score
                    articles[idx]['sentiment_label'] = 'positive'
                elif label == 'negative':
                    articles[idx]['sentiment_score'] = -score
                    articles[idx]['sentiment_label'] = 'negative'
                else:
                    articles[idx]['sentiment_score'] = 0.0
                    articles[idx]['sentiment_label'] = 'neutral'
        except Exception as e:
            print(f"   [WARN] Errore sentiment batch {i}: {e}")
            for j in range(len(batch)):
                idx = i + j
                articles[idx]['sentiment_score'] = 0.0
                articles[idx]['sentiment_label'] = 'neutral'
    return articles


def aggregate_daily_sentiment(articles):
    df = pd.DataFrame(articles)
    if df.empty:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby('date').agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        article_count=('title', 'count'),
        pos_count=('sentiment_label', lambda x: (x == 'positive').sum()),
        neg_count=('sentiment_label', lambda x: (x == 'negative').sum()),
        neu_count=('sentiment_label', lambda x: (x == 'neutral').sum()),
    ).reset_index()
    daily['sentiment_positive_ratio'] = daily['pos_count'] / daily['article_count']
    daily['sentiment_negative_ratio'] = daily['neg_count'] / daily['article_count']
    daily['sentiment_neutral_ratio'] = daily['neu_count'] / daily['article_count']
    daily['sentiment_weighted'] = (
        daily['sentiment_positive_ratio'] * 1.0 +
        daily['sentiment_negative_ratio'] * (-1.0)
    )
    daily = daily.drop(columns=['pos_count', 'neg_count', 'neu_count'])
    return daily


def main(max_articles=200):
    print("=" * 60)
    print("FASE 2.1: INGESTIONE NOTIZIE DA RSS")
    print("=" * 60)

    print("\n[1] Fetch notizie da RSS feeds...")
    articles = fetch_news_from_rss(RSS_FEEDS, max_per_feed=max_articles)
    print(f"   Ottenuti {len(articles)} articoli grezzi.")

    print("\n[2] Parsing date...")
    articles = parse_news_dates(articles)

    print("\n[3] Caricamento FinBERT...")
    t0 = time.time()
    sentiment_pipeline = pipeline(
        'sentiment-analysis', model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL
    )
    print(f"   Caricato in {time.time() - t0:.1f}s")

    print(f"\n[4] Analisi sentiment su {len(articles)} titoli...")
    t0 = time.time()
    articles = analyze_sentiment_batch(articles, sentiment_pipeline)
    print(f"   Completato in {time.time() - t0:.1f}s")
    print("\n   Anteprima sentiment:")
    for a in articles[:5]:
        print(f"   {a['sentiment_label']:>8} ({a['sentiment_score']:+.2f}) -> {a['title'][:70]}")

    print("\n" + "=" * 60)
    print("FASE 2.3: AGGREGAZIONE GIORNALIERA")
    print("=" * 60)

    print("\n[5] Aggregazione daily...")
    daily_df = aggregate_daily_sentiment(articles)
    if daily_df.empty:
        print("   Nessun dato di sentiment disponibile.")
        return
    print(f"   {len(daily_df)} giorni con notizie")
    print(f"   Media sentiment: {daily_df['sentiment_mean'].mean():+.3f}")
    print(f"   Giorni positivi: {(daily_df['sentiment_mean'] > 0).sum()}")
    print(f"   Giorni negativi: {(daily_df['sentiment_mean'] < 0).sum()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    daily_df.to_csv(output_path, index=False)
    print(f"\n[6] Sentiment daily salvato in: {os.path.abspath(output_path)}")
    print("\n   Anteprima daily sentiment:")
    print(daily_df.tail(10).to_string())

    print("\n[SUCCESSO] Fase 2 completata.")
    print("Prossimo passo: integrazione nel Master DataFrame (Fase 3).")


if __name__ == "__main__":
    main()
