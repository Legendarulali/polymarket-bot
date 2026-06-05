import logging
from py_clob_client_v2 import OrderArgs, Side, OrderType, PartialCreateOrderOptions
from src.clob_client import get_clob_client

logger = logging.getLogger(__name__)


def place_limit_order(
    token_id: str,
    price: float,
    size: float,
    side: Side,
) -> dict:
    tick_price = round(price, 2)
    size = float(size)

    if not (0.01 <= tick_price <= 0.99):
        return {
            "success": False,
            "error": f"Price {tick_price} out of valid range [0.01, 0.99]",
        }

    logger.info(
        "Placing %s limit order | token=%s | size=%.4f | price=$%.2f",
        side.name,
        token_id,
        size,
        tick_price,
    )

    client = get_clob_client()
    try:
        result = client.create_and_post_order(
            order_args=OrderArgs(
                token_id=token_id,
                price=tick_price,
                size=size,
                side=side,
            ),
            options=PartialCreateOrderOptions(tick_size="0.01"),
            order_type=OrderType.GTC,
        )

        order_id = (
            result.get("orderID")
            or result.get("order_id")
            or result.get("id")
            or "unknown"
        )
        logger.info(
            "%s order placed successfully | order_id=%s | token=%s | size=%.4f | price=$%.2f",
            side.name,
            order_id,
            token_id,
            size,
            tick_price,
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
