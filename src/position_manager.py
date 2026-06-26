import logging
import time

from config.settings import settings
from src.clob_client import get_clob_client
from src.execution import Side, place_limit_order
from src.notifications import notify_trade

logger = logging.getLogger(__name__)

# Polymarket respinge ordine sub $1 valoare totală.
_MIN_EXIT_VALUE_USD = 1.0
# Nu re-trimite un ordin de exit pentru același token mai des de atât —
# pozițiile rămân vizibile în Data API până când ordinul SELL se umple.
_EXIT_COOLDOWN_SECONDS = 30 * 60

# {token_id: timestamp ultimei încercări de exit}
_exit_attempts: dict[str, float] = {}


def _best_bid(token_id: str) -> float | None:
    """Cel mai bun bid live din orderbook — prețul la care se poate vinde acum."""
    try:
        ob = get_clob_client().get_order_book(token_id=token_id)
        bids = ob.bids if hasattr(ob, "bids") else ob.get("bids", [])
        if not bids:
            return None
        return max(
            float(b.price if hasattr(b, "price") else b["price"]) for b in bids
        )
    except Exception as exc:
        logger.warning("[Exit] Orderbook indisponibil pentru token=%s: %s", token_id, exc)
        return None


def should_attempt_exit(token_id: str) -> bool:
    """True dacă pe acest token NU s-a încercat un exit în ultimele _EXIT_COOLDOWN_SECONDS.

    Cooldown-ul (`_exit_attempts`) e PARTAJAT între TP/SL (`check_exit_conditions`) și
    panic-sell (`main._reevaluate_active_positions`), ca aceeași poziție să nu fie vândută
    de două ori în fereastra de cooldown (ex: stop-loss + știre proastă în aceeași rundă)."""
    return time.time() - _exit_attempts.get(token_id, 0.0) >= _EXIT_COOLDOWN_SECONDS


def record_exit_attempt(token_id: str) -> None:
    """Marchează acum o încercare de exit pe `token_id` (anti-spam, partajat TP/SL + panic)."""
    _exit_attempts[token_id] = time.time()


def check_exit_conditions(account_state: dict) -> None:
    """
    Verifică take-profit / stop-loss pe toate pozițiile deschise și plasează
    ordine SELL la best bid când pragurile sunt atinse.

    Activ doar dacă TAKE_PROFIT_PCT și/sau STOP_LOSS_PCT sunt > 0 — implicit
    botul păstrează pozițiile până la rezoluția pieței (comportament istoric).
    """
    take_profit = settings.TAKE_PROFIT_PCT
    stop_loss = settings.STOP_LOSS_PCT
    if take_profit <= 0 and stop_loss <= 0:
        return

    for pos in account_state.get("active_positions", []):
        token_id = pos.get("token_id") or ""
        avg_price = float(pos.get("avg_price") or 0)
        size = float(pos.get("size") or 0)
        if not token_id or avg_price <= 0 or size <= 0:
            continue

        if not should_attempt_exit(token_id):
            continue

        bid = _best_bid(token_id)
        if bid is None:
            continue

        pnl_pct = (bid - avg_price) / avg_price
        if take_profit > 0 and pnl_pct >= take_profit:
            reason = f"Take Profit atins (+{pnl_pct * 100:.1f}%)"
        elif stop_loss > 0 and pnl_pct <= -stop_loss:
            reason = f"Stop Loss declanșat ({pnl_pct * 100:.1f}%)"
        else:
            continue

        if bid * size < _MIN_EXIT_VALUE_USD:
            logger.warning(
                "[Exit] Poziție sub minimul de $%.2f (token=%s, valoare=$%.2f) — "
                "nu poate fi închisă prin ordin; va rămâne până la rezoluție.",
                _MIN_EXIT_VALUE_USD, token_id, bid * size,
            )
            record_exit_attempt(token_id)
            continue

        logger.warning(
            "[EXIT] %s | '%s' | token=%s | avg=$%.4f | bid=$%.4f | size=%.2f",
            reason, str(pos.get("asset_name", ""))[:60], token_id, avg_price, bid, size,
        )

        record_exit_attempt(token_id)
        result = place_limit_order(
            token_id=token_id,
            price=bid,
            side=Side.SELL,
            size=size,
        )

        if result.get("success"):
            logger.info("[Exit] Ordin SELL plasat | order_id=%s", result.get("order_id"))
            notify_trade(
                "SELL",
                str(pos.get("asset_name", "")) or token_id,
                price=bid,
                size=size,
                total=bid * size,
                reason=reason,
                order_id=result.get("order_id"),
                action="CLOSE POSITION",
            )
        else:
            logger.error("[Exit] Ordin SELL respins: %s", result.get("error"))
