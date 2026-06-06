import time
import sys
import signal
import logging

import httpx

from config.settings import settings
from src.reconciliation import log_reconciliation_report, get_account_state
from src.execution import place_limit_order, cancel_order  # noqa: F401 — available for strategy use
from src.clob_client import get_clob_client
from src.strategy import MultiOutcomeArbStrategy

logger = logging.getLogger(__name__)

RUNNING = True
_LOOP_INTERVAL = 10  # seconds


def get_active_market_tokens(slug="fed-decision-in-june"):
    try:
        url = f"https://gamma-api.polymarket.com/events?slug={slug}"
        response = httpx.get(url)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0 and "markets" in data[0]:
                market = data[0]["markets"][0]
                tokens = [t["token_id"] for t in market.get("tokens", [])]
                if tokens:
                    return tokens
    except Exception as e:
        print(f"Eroare la preluarea tokenilor dinamici: {e}")
    return ["8441400852834915183759801017793514978104486628517653995211751018945988243154"]


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


def main() -> None:
    logger.info(
        "Polymarket V2 bot starting | host=%s | chain_id=%d | funder=%s",
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

    market_token_ids = get_active_market_tokens()
    logger.info("Market token IDs loaded: %s", market_token_ids)
    strategy = MultiOutcomeArbStrategy(client=get_clob_client(), token_ids=market_token_ids)
    logger.info("Strategy initialised: %s", type(strategy).__name__)
    logger.info("Entering event loop (interval=%ds). Send SIGTERM/SIGINT to stop.", _LOOP_INTERVAL)

    try:
        while RUNNING:
            account_state = get_account_state()
            strategy.tick(account_state)

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
