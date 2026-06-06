import json
import logging
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_MIN_CONFIDENCE = 0.0


def _get_groq_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY,
        )
    return _client


def analyse_market(market_title: str, current_price: float, news_items: list[str]) -> dict:
    """
    Trimite datele pieței la Groq LLM și returnează decizia de trading.
    Returnează: {"trade": bool, "outcome": "YES"/"NO", "confidence": float, "reason": str}
    """
    if not settings.GROQ_API_KEY:
        logger.warning("[AIAnalyser] GROQ_API_KEY lipsă — skip analiză.")
        return {"trade": False, "outcome": "NO", "confidence": 0.0, "reason": "No API key configured"}

    news_text = (
        "\n".join(f"- {n}" for n in news_items[:8])
        if news_items
        else "No recent news available."
    )

    prompt = (
        f"You are an AI trading analyst for prediction markets. Analyze the market below and decide.\n\n"
        f"Market: {market_title}\n"
        f"Current YES price: ${current_price:.4f} (implied probability {current_price * 100:.1f}%)\n\n"
        f"Recent news:\n{news_text}\n\n"
        f"Decide:\n"
        f"1. Should we trade? (true/false)\n"
        f"2. Which outcome? (YES or NO)\n"
        f"3. Confidence 0.0-1.0\n"
        f"4. Brief reason (max 100 chars)\n\n"
        f'Respond STRICTLY with valid JSON only, no extra text:\n'
        f'{{"trade": true, "outcome": "YES", "confidence": 0.85, "reason": "short explanation"}}'
    )

    raw = ""
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        parsed = json.loads(raw)
        return {
            "trade": bool(parsed.get("trade", False)),
            "outcome": str(parsed.get("outcome", "NO")).upper(),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", ""))[:200],
        }
    except json.JSONDecodeError as exc:
        logger.error("[AIAnalyser] JSON parse error: %s | raw=%s", exc, raw[:200])
        return {"trade": False, "outcome": "NO", "confidence": 0.0, "reason": "JSON parse error"}
    except Exception as exc:
        logger.error("[AIAnalyser] Groq API error: %s", exc)
        return {"trade": False, "outcome": "NO", "confidence": 0.0, "reason": str(exc)[:100]}
