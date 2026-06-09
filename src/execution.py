import logging
from py_clob_client_v2 import OrderArgs, Side, OrderType, PartialCreateOrderOptions
from src.clob_client import get_clob_client

logger = logging.getLogger(__name__)


_MIN_ORDER_VALUE = 5.0   # target $5 per order (satisfies Polymarket's $1 minimum)
_MIN_SIZE        = 2.0   # absolute floor in shares (safety net)


def place_limit_order(
    token_id: str,
    price: float,
    side: Side,
) -> dict:
    tick_price = round(price, 2)

    if not (0.01 <= tick_price <= 0.99):
        return {
            "success": False,
            "error": f"Price {tick_price} out of valid range [0.01, 0.99]",
        }

    # Dynamic size: target $5 order value; floor at _MIN_SIZE shares
    size = max(_MIN_ORDER_VALUE / tick_price, _MIN_SIZE)
    size = round(size, 2)
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
