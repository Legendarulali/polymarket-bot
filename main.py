import time
import sys
import signal
import logging

import httpx
from py_clob_client_v2 import Side

from config.settings import settings
from src.reconciliation import log_reconciliation_report, get_account_state
from src.execution import place_limit_order
from src.clob_client import get_clob_client
from src.news_monitor import fetch_free_news
from src.ai_analyser import analyse_market

logger = logging.getLogger(__name__)

RUNNING = True
_LOOP_INTERVAL = 60  # seconds
_GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
_MIN_BALANCE_PUSD = 5.0
_ORDER_SIZE = 1.0
_MIN_CONFIDENCE = 0.70


def handle_shutdown(signum, frame):
    global RUNNING
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating graceful shutdown...", sig_name)
    RUNNING = False


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def _interruptible_sleep(seconds: int) -> None:
    """Sleep in 1-second increments so SIGTERM is honoured promptly."""
    for _ in range(seconds):
        if not RUNNING:
            break
        time.sleep(1)


def fetch_active_markets(limit: int = 5) -> list[dict]:
    try:
        response = httpx.get(
            _GAMMA_MARKETS_URL,
            params={"active": "true", "closed": "false", "limit": limit},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as exc:
        logger.warning("[Agent] Could not fetch active markets: %s", exc)
    return []


def _token_for_outcome(market: dict, outcome: str) -> str | None:
    for token in market.get("tokens", []):
        if token.get("outcome", "").upper() == outcome.upper():
            return token.get("token_id")
    return None


def _current_price(market: dict, outcome: str) -> float:
    """Returnează prețul YES sau NO din datele Gamma, cu fallback la 0.5."""
    try:
        prices = market.get("outcomePrices", [])
        if outcome.upper() == "YES" and prices:
            return float(prices[0])
        if outcome.upper() == "NO" and len(prices) > 1:
            return float(prices[1])
        ask = market.get("bestAsk")
        if ask is not None:
            return float(ask)
    except (ValueError, TypeError):
        pass
    return 0.5


def process_market(market: dict, account_state: dict) -> None:
    pusd_balance: float = account_state.get("pusd_balance", 0.0)
    active_positions: list[dict] = account_state.get("active_positions", [])

    title = market.get("question") or market.get("title") or "Unknown market"
    market_id = market.get("id", "?")

    logger.info("[Agent] Procesez piața id=%s: %s", market_id, title[:80])

    # Pas 1: Culegere știri gratuite pe baza titlului pieței ca query dinamic
    news = fetch_free_news(query=title[:100])
    logger.info("[Agent] %d știri găsite pentru '%s'.", len(news), title[:50])

    # Pas 2: Analiză AI via Groq
    yes_price = _current_price(market, "YES")
    decision = analyse_market(title, yes_price, news)
    logger.info(
        "[Agent] Decizie AI: trade=%s | outcome=%s | confidence=%.2f | reason=%s",
        decision["trade"],
        decision["outcome"],
        decision["confidence"],
        decision["reason"],
    )

    if not decision["trade"]:
        return

    if decision["confidence"] < _MIN_CONFIDENCE:
        logger.info(
            "[Agent] Confidence %.2f sub pragul %.2f. Ordin blocat.",
            decision["confidence"],
            _MIN_CONFIDENCE,
        )
        return

    # Pas 3: Verificări de risc
    if pusd_balance < _MIN_BALANCE_PUSD:
        logger.warning(
            "[Agent] Fonduri insuficiente (%.4f pUSD < %.1f minim). Ordin blocat.",
            pusd_balance,
            _MIN_BALANCE_PUSD,
        )
        return

    outcome = decision["outcome"]
    token_id = _token_for_outcome(market, outcome)
    if not token_id:
        logger.warning(
            "[Agent] Token negăsit pentru outcome=%s în piața id=%s.", outcome, market_id
        )
        return

    if any(pos.get("token_id") == token_id for pos in active_positions):
        logger.info("[Agent] Poziție deschisă deja pe token=%s. Ordin blocat.", token_id)
        return

    # Pas 4: Plasare ordin
    price = _current_price(market, outcome)
    logger.warning(
        "[TRADE] AI → BUY %s | '%s' | token=%s | price=$%.4f | size=%.1f | confidence=%.2f",
        outcome,
        title[:60],
        token_id,
        price,
        _ORDER_SIZE,
        decision["confidence"],
    )

    result = place_limit_order(
        token_id=token_id,
        price=price,
        size=_ORDER_SIZE,
        side=Side.BUY,
    )

    if result.get("success"):
        logger.info("[Agent] Ordin plasat cu succes | order_id=%s", result.get("order_id"))
    else:
        logger.error("[Agent] Ordin respins de CLOB: %s", result.get("error"))


def main() -> None:
    logger.info(
        "Polymarket AI Agent starting | host=%s | chain_id=%d | funder=%s",
        settings.CLOB_HOST,
        settings.CHAIN_ID,
        settings.FUNDER_ADDRESS,
    )

    try:
        logger.info("Running boot reconciliation...")
        log_reconciliation_report()
    except Exception as exc:
        logger.critical("Boot reconciliation failed — aborting startup: %s", exc)
        sys.exit(1)

    get_clob_client()  # warm up connection
    logger.info(
        "Entering autonomous AI agent loop (interval=%ds). Send SIGTERM/SIGINT to stop.",
        _LOOP_INTERVAL,
    )

    try:
        while RUNNING:
            account_state = get_account_state()

            markets = fetch_active_markets(limit=5)
            if not markets:
                logger.warning("[Agent] Nicio piață activă găsită. Aștept...")
                _interruptible_sleep(_LOOP_INTERVAL)
                continue

            logger.info("[Agent] Procesez %d piețe active.", len(markets))
            for market in markets:
                if not RUNNING:
                    break
                try:
                    process_market(market, account_state)
                except Exception as exc:
                    logger.error(
                        "[Agent] Eroare la procesarea pieței id=%s: %s",
                        market.get("id", "?"),
                        exc,
                        exc_info=True,
                    )
                time.sleep(3)

            _interruptible_sleep(_LOOP_INTERVAL)

    except Exception as exc:
        logger.critical("Unhandled exception in event loop: %s", exc, exc_info=True)
    finally:
        logger.info("Bot stopped. Running final safety reconciliation...")
        try:
            log_reconciliation_report()
        except Exception as exc:
            logger.error("Final reconciliation failed: %s", exc)
        logger.info("Shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
