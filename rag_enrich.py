import requests
import feedparser
import re
from datetime import datetime, timedelta

DEFAULT_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
KEYWORDS = ["injury", "injured", "squad", "suspension", "suspended", "out", "doubt", "retire", "withdraw"]


def fetch_google_news(query, max_items=20):
    url = DEFAULT_GOOGLE_NEWS_RSS.format(query=requests.utils.quote(query))
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_items]:
        title = entry.get('title', '')
        summary = entry.get('summary', '')
        link = entry.get('link', '')
        published = entry.get('published', '')
        items.append({
            'title': title,
            'summary': summary,
            'link': link,
            'published': published,
        })
    return items


def _count_keywords(text, keywords=KEYWORDS):
    text = text.lower()
    counts = {k: len(re.findall(r"\b" + re.escape(k) + r"\b", text)) for k in keywords}
    return counts


def aggregate_article_counts(articles):
    agg = {k: 0 for k in KEYWORDS}
    for art in articles:
        text = (art.get('title','') + ' ' + art.get('summary',''))
        counts = _count_keywords(text)
        for k, v in counts.items():
            agg[k] += v
    return agg


def filter_recent_articles(articles, days=14):
    cutoff = datetime.utcnow() - timedelta(days=days)
    recent = []
    for a in articles:
        pub = a.get('published', '')
        try:
            # feedparser gives structured time sometimes
            if isinstance(pub, tuple):
                pub_dt = datetime(*pub[:6])
            else:
                pub_dt = datetime.strptime(pub, '%a, %d %b %Y %H:%M:%S %Z')
        except Exception:
            # if parsing fails, include conservatively
            recent.append(a)
            continue
        if pub_dt >= cutoff:
            recent.append(a)
    return recent


def enrich_team(team_name, days=14):
    query = f"{team_name} football"
    articles = fetch_google_news(query, max_items=30)
    recent = filter_recent_articles(articles, days=days)
    agg = aggregate_article_counts(recent)
    total_articles = len(recent)
    features = {
        'articles_recent_count': total_articles,
        **{f'kw_{k}': v for k, v in agg.items()}
    }
    return features


def enrich_match(home_team, away_team, days=14):
    home_features = enrich_team(home_team, days=days)
    away_features = enrich_team(away_team, days=days)

    # Simple combined features
    combined = {
        'home_articles': home_features['articles_recent_count'],
        'away_articles': away_features['articles_recent_count'],
    }
    # injury mentions
    combined['home_injury_mentions'] = sum(v for k, v in home_features.items() if k.startswith('kw_injury')) if 'kw_injury' in home_features else 0
    combined['away_injury_mentions'] = sum(v for k, v in away_features.items() if k.startswith('kw_injury')) if 'kw_injury' in away_features else 0

    # also expose granular keyword counts
    for k, v in home_features.items():
        combined[f'home_{k}'] = v
    for k, v in away_features.items():
        combined[f'away_{k}'] = v

    return combined


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='RAG-like enrichment: fetch recent news and extract keyword counts')
    parser.add_argument('home', help='Home team name')
    parser.add_argument('away', help='Away team name')
    parser.add_argument('--days', type=int, default=14, help='Lookback window in days')
    args = parser.parse_args()

    out = enrich_match(args.home, args.away, days=args.days)
    print(json.dumps(out, indent=2))
