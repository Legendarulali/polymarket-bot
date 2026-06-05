import logging
import requests
from py_clob_client_v2 import BalanceAllowanceParams, AssetType
from src.clob_client import get_clob_client
from config.settings import settings

logger = logging.getLogger(__name__)

_POSITIONS_URL = "https://data-api.polymarket.com/positions"
_MICRO_PUSD = 1e6
_ALLOWANCE_WARN_THRESHOLD = 100.0


def get_account_state() -> dict:
    client = get_clob_client()

    bal_data = client.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    )
    pusd_balance = float(bal_data.balance) / _MICRO_PUSD
    allowance = float(bal_data.allowance) / _MICRO_PUSD

    active_positions = []
    try:
        response = requests.get(
            _POSITIONS_URL,
            params={"user": settings.FUNDER_ADDRESS},
            timeout=10,
        )
        response.raise_for_status()
        positions_raw = response.json()

        for pos in positions_raw:
            if float(pos.get("size", 0)) > 0:
                active_positions.append({
                    "asset_name": pos.get("title") or pos.get("asset_name", ""),
                    "token_id":   pos.get("asset") or pos.get("token_id", ""),
                    "side":       pos.get("side", ""),
                    "size":       float(pos["size"]),
                    "avg_price":  float(pos.get("avgPrice") or pos.get("avg_price", 0)),
                })
    except requests.RequestException as exc:
        logger.warning("Could not fetch positions from Data API: %s", exc)

    return {
        "pusd_balance":     pusd_balance,
        "allowance":        allowance,
        "active_positions": active_positions,
    }


def log_reconciliation_report() -> None:
    try:
        state = get_account_state()
    except Exception as exc:
        logger.error("Reconciliation failed: %s", exc)
        return

    pusd_balance = state["pusd_balance"]
    allowance    = state["allowance"]
    positions    = state["active_positions"]

    logger.info("=" * 60)
    logger.info("RECONCILIATION REPORT")
    logger.info("=" * 60)
    logger.info("  Balance  : %.6f pUSD", pusd_balance)
    logger.info("  Allowance: %.6f pUSD", allowance)

    if allowance < _ALLOWANCE_WARN_THRESHOLD:
        logger.warning(
            "Allowance is below %.0f pUSD (current: %.6f). "
            "Consider re-approving the CLOB contract.",
            _ALLOWANCE_WARN_THRESHOLD,
            allowance,
        )

    if not positions:
        logger.info("  Positions: none")
    else:
        logger.info("  Active positions (%d):", len(positions))
        for pos in positions:
            logger.info(
                "    [%s] %s | token=%s | size=%.4f | avg_price=%.6f",
                pos["side"].upper() or "?",
                pos["asset_name"] or "unknown",
                pos["token_id"],
                pos["size"],
                pos["avg_price"],
            )

    logger.info("=" * 60)
