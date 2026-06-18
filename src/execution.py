import logging
import math
from enum import Enum

from py_clob_client_v2 import OrderArgs, OrderType, PartialCreateOrderOptions

from src.clob_client import get_clob_client

logger = logging.getLogger(__name__)


class Side(str, Enum):
    """Direcția ordinului — clientul oficial folosește string-uri simple."""
    BUY = "BUY"
    SELL = "SELL"


_TARGET_ORDER_VALUE = 5.0   # țintă valoare ordin ($). Polymarket acceptă ordine >= $1.
_MIN_SIZE           = 2.0   # floor absolut în shares (safety net, ramura fără buget)
# Buffer de siguranță aplicat de apelant pe balanță ÎNAINTE de dimensionare: nu
# angajăm niciodată întreaga balanță, ca estimarea dinamică de fee a CLOB să nu
# împingă ordinul în „Insufficient Balance". safe_balance = balanță * 0.97.
_BALANCE_SAFETY_BUFFER = 0.97
# Polymarket respinge (400 Bad Request) ordinele sub acest număr de share-uri.
# Verificat de apelant ÎNAINTE de a trimite ordinul, ca să nu lovim API-ul cu
# cereri invalide pe balanțe mici.
_MIN_SHARES = 5.0
_VALID_TICK_SIZES = ("0.1", "0.01", "0.001", "0.0001")
_DEFAULT_TICK_SIZE = "0.01"


def compute_order_size(price: float, budget: float | None = None) -> float:
    """Size dinamic pentru un ordin BUY.

    Țintește o valoare de $5 per ordin. Dacă `budget` (balanța pUSD disponibilă)
    e dată și sub $5, folosește cât mai mult din ea fără a o depăși
    (size * price <= budget), ca un ordin mic să NU fie respins pentru fonduri
    insuficiente. Apelantul garantează budget >= minimul Polymarket de $1.
    """
    if budget is not None:
        value = min(_TARGET_ORDER_VALUE, budget)
        # Trunchiere (floor) la 2 zecimale: size*price nu trebuie să depășească `value`.
        return math.floor((value / price) * 100) / 100
    return round(max(_TARGET_ORDER_VALUE / price, _MIN_SIZE), 2)


def place_limit_order(
    token_id: str,
    price: float,
    side: Side,
    tick_size: str = _DEFAULT_TICK_SIZE,
    size: float | None = None,
) -> dict:
    """
    Plasează un ordin limită GTC. Dacă `size` lipsește, se calculează dinamic
    (target $5) — pentru SELL pe poziții existente trebuie dat explicit, altfel
    s-ar vinde o cantitate calculată din preț, nu cea deținută.
    """
    if tick_size not in _VALID_TICK_SIZES:
        logger.warning(
            "Unknown tick_size %r — falling back to %s", tick_size, _DEFAULT_TICK_SIZE
        )
        tick_size = _DEFAULT_TICK_SIZE

    tick = float(tick_size)
    decimals = len(tick_size.split(".")[1])
    tick_price = round(price, decimals)
    max_price = round(1 - tick, decimals)

    if not (tick <= tick_price <= max_price):
        return {
            "success": False,
            "error": f"Price {tick_price} out of valid range [{tick}, {max_price}]",
        }

    size = round(size, 2) if size is not None else compute_order_size(tick_price)
    if size <= 0:
        return {"success": False, "error": f"Invalid order size {size}"}
    total_value = size * tick_price

    logger.info(
        "Placing %s limit order | token=%s | size=%.2f | price=$%.2f | total=$%.2f",
        side.name,
        token_id,
        size,
        tick_price,
        total_value,
    )

    client = get_clob_client()
    try:
        result = client.create_and_post_order(
            order_args=OrderArgs(
                token_id=token_id,
                price=tick_price,
                size=size,
                side=side.value,
            ),
            options=PartialCreateOrderOptions(tick_size=tick_size),
            order_type=OrderType.GTC,
        )

        order_id = (
            result.get("orderID")
            or result.get("order_id")
            or result.get("id")
            or "unknown"
        )
        logger.info(
            "%s order placed successfully | order_id=%s | token=%s | size=%.2f | price=$%.2f | total=$%.2f",
            side.name,
            order_id,
            token_id,
            size,
            tick_price,
            total_value,
        )
        return {"success": True, "order_id": order_id, "raw": result}

    except Exception as exc:
        error_msg = str(exc)
        logger.error(
            "Failed to place %s order | token=%s | price=$%.2f | error=%s",
            side.name,
            token_id,
            tick_price,
            error_msg,
        )
        return {"success": False, "error": error_msg}


def cancel_order(order_id: str) -> dict:
    logger.info("Cancelling order | order_id=%s", order_id)

    client = get_clob_client()
    try:
        result = client.cancel_order(order_id)

        logger.info("Order cancelled successfully | order_id=%s", order_id)
        return {"success": True, "order_id": order_id, "raw": result}

    except Exception as exc:
        error_msg = str(exc)
        logger.error(
            "Failed to cancel order | order_id=%s | error=%s",
            order_id,
            error_msg,
        )
        return {"success": False, "order_id": order_id, "error": error_msg}
