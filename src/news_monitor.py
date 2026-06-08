import logging
import defusedxml.ElementTree as ET
import requests
from config.settings import settings

logger = logging.getLogger(__name__)

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_X_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
_REQUEST_TIMEOUT = 8


def _fetch_newsapi(query: str) -> list[str]:
    """Returnează titlurile articolelor recente de la NewsAPI pentru un query."""
    if not settings.NEWS_API_KEY:
        return []

    try:
        response = requests.get(
            _NEWSAPI_URL,
            params={
                "q": query,
                "sortBy": "publishedAt",
                "pageSize": 10,
                "language": "en",
            },
            headers={"X-Api-Key": settings.NEWS_API_KEY},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title")]
    except requests.Timeout:
        logger.warning("[NewsMonitor] NewsAPI timeout.")
        return []
    except requests.RequestException as exc:
        logger.warning("[NewsMonitor] NewsAPI error: %s", exc)
        return []


def _fetch_x_recent(query: str) -> list[str]:
    """Returnează textele tweet-urilor recente de la X API v2 pentru un query."""
    if not settings.X_API_BEARER_TOKEN:
        return []

    try:
        response = requests.get(
            _X_SEARCH_URL,
            params={"query": query, "max_results": 10, "tweet.fields": "text"},
            headers={"Authorization": f"Bearer {settings.X_API_BEARER_TOKEN}"},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        tweets = response.json().get("data", [])
        return [t["text"] for t in tweets if t.get("text")]
    except requests.Timeout:
        logger.warning("[NewsMonitor] X API timeout.")
        return []
    except requests.RequestException as exc:
        logger.warning("[NewsMonitor] X API error: %s", exc)
        return []


def fetch_free_news(query: str) -> list[str]:
    """Caută știri gratuit via Google News RSS, fără API key."""
    try:
        encoded = requests.utils.quote(query[:100])
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        titles = []
        for item in root.findall(".//item")[:10]:
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(title_el.text)
        logger.debug("[NewsMonitor] Free RSS fetch: %d titluri pentru '%s'.", len(titles), query[:50])
        return titles
    except Exception as exc:
        logger.warning("[NewsMonitor] Free news fetch failed: %s", exc)
        return []


def fetch_latest_breaking_news(query: str = "politics election market") -> list[str]:
    """
    Agregă știri din NewsAPI și tweet-uri recente de pe X.
    Returnează o listă de texte brute (titluri + tweet-uri).
    Dacă ambele surse sunt indisponibile sau neconfigurare, returnează [].
    """
    headlines = _fetch_newsapi(query)
    tweets = _fetch_x_recent(query)
    combined = headlines + tweets

    if combined:
        logger.debug("[NewsMonitor] Fetched %d items (news=%d, tweets=%d).",
                     len(combined), len(headlines), len(tweets))
    return combined


def analyze_sentiment(text: str, keywords: list[str]) -> bool:
    """
    Returnează True dacă textul conține cel puțin unul dintre cuvintele cheie.
    Comparația este case-insensitive și caută substring exact.
    """
    if not text or not keywords:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)
