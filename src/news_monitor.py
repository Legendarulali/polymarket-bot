import html
import logging
import re
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


# ---------------------------------------------------------------------------
# Serializare știri → Markdown compact pentru prompt (taie tokeni → evită 429 TPD)
# ---------------------------------------------------------------------------
_MD_MAX_ARTICLES = 5      # maximum articole/snippet-uri păstrate per piață
_MD_MAX_TITLE = 140       # plafon caractere titlu (după curățare)
_MD_MAX_SNIPPET = 200     # plafon caractere snippet (după curățare)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(text, max_len: int) -> str:
    """Curăță un fragment de știre pentru prompt: dez-escapează entitățile HTML, scoate
    tag-urile HTML și orice paranteze unghiulare rămase (anti prompt-injection),
    colapsează spațiile albe și taie la max_len. Reduce tokenii și zgomotul."""
    if not text:
        return ""
    text = html.unescape(str(text))                  # &amp; → &, &#39; → ' etc.
    text = _HTML_TAG_RE.sub(" ", text)               # scoate <tag ...>
    text = text.replace("<", " ").replace(">", " ")  # neutralizează unghiularele rămase
    text = _WS_RE.sub(" ", text).strip()             # colapsează spațiile
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text


def _strip_source_suffix(title: str) -> str:
    """Elimină sufixul „ - Publisher" tipic titlurilor Google News RSS (publisher-ul
    se repetă pe fiecare linie → tokeni irosiți, fără valoare pentru decizie)."""
    idx = title.rfind(" - ")
    if idx > 0:
        suffix = title[idx + 3:]
        if 0 < len(suffix) <= 40 and len(suffix.split()) <= 6:
            return title[:idx].rstrip()
    return title


def _title_and_snippet(item) -> tuple[str, str]:
    """Normalizează un articol brut (string sau dict) la (titlu, snippet).

    - dict (ex. Tavily / NewsAPI): citește title + snippet/description/content/summary;
    - string (ex. titlu Google News RSS „Headline - Sursă"): titlul = headline fără
      sufixul de sursă, fără snippet distinct."""
    if isinstance(item, dict):
        title = item.get("title") or item.get("headline") or ""
        snippet = (item.get("snippet") or item.get("description")
                   or item.get("content") or item.get("summary") or "")
        return str(title), str(snippet)
    return _strip_source_suffix(str(item)), ""


def serialize_news_to_markdown(news_list, market_title: str) -> str:
    """Comprimă lista brută de știri într-un Markdown curat și compact pentru prompt.

    - păstrează maximum _MD_MAX_ARTICLES (5) cele mai relevante articole/snippet-uri;
    - curăță fiecare fragment (entități + tag-uri HTML, spații) → taie tokeni și zgomot;
    - emite structura:
        ### Context News for: <market_title>
        1. **[TITLE] <titlu>**
           - Snippet: <rezumat curățat>
      Linia „Snippet" e omisă când articolul NU are un rezumat distinct de titlu
      (ex. titlurile Google News) — ca să nu dublăm textul și să irosim tokeni.

    Acceptă atât string-uri (titluri brute) cât și dict-uri ({title, snippet, ...}).
    Listă goală → marcaj compact „_No recent news available._"."""
    safe_title = _clean_text(market_title, 120) or "Unknown market"
    lines = [f"### Context News for: {safe_title}"]

    count = 0
    for item in (news_list or []):
        raw_title, raw_snippet = _title_and_snippet(item)
        title = _clean_text(raw_title, _MD_MAX_TITLE)
        if not title:
            continue
        count += 1
        lines.append(f"{count}. **[TITLE] {title}**")
        snippet = _clean_text(raw_snippet, _MD_MAX_SNIPPET)
        if snippet and snippet.lower() != title.lower():
            lines.append(f"   - Snippet: {snippet}")
        if count >= _MD_MAX_ARTICLES:
            break

    if count == 0:
        lines.append("_No recent news available._")
    return "\n".join(lines)


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
