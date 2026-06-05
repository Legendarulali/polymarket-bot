import time
import sys
import signal
import logging

from config.settings import settings
from src.reconciliation import log_reconciliation_report, get_account_state
from src.execution import place_limit_order, cancel_order  # noqa: F401 — available for strategy use
from src.clob_client import get_clob_client
from src.strategy import MultiOutcomeArbStrategy

logger = logging.getLogger(__name__)

RUNNING = True
_LOOP_INTERVAL = 10  # seconds


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

    # Înlocuiește cu token_ids reale din piața multi-opțiune țintă.
    market_token_ids = [
        "DUMMY_TOKEN_ID_OUTCOME_A",
        "DUMMY_TOKEN_ID_OUTCOME_B",
        "DUMMY_TOKEN_ID_OUTCOME_C",
    ]
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
