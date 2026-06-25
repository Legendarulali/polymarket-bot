import logging
import time

import requests
from py_clob_client_v2 import BalanceAllowanceParams, AssetType

# PolyApiException e ridicată de SDK la răspunsuri non-2xx (ex. 401 Unauthorized
# dintr-o desincronizare de ceas sau un glitch temporar al API-ului Polymarket).
# Import defensiv: dacă fork-ul nu expune modulul, cădem pe un stub local, ca un
# import greșit să nu doboare botul la pornire (mai rău decât eroarea originală).
try:
    from py_clob_client_v2.exceptions import PolyApiException
except ImportError:  # pragma: no cover
    class PolyApiException(Exception):  # type: ignore[no-redef]
        status_code = None

from src.clob_client import get_clob_client, get_signature_type
from config.settings import settings

logger = logging.getLogger(__name__)

_POSITIONS_URL = "https://data-api.polymarket.com/positions"
_MICRO_PUSD = 1e6
_ALLOWANCE_WARN_THRESHOLD = 100.0
# Retry pe apelul critic de balanță: un glitch tranzitoriu de rețea sau un 401 de o
# secundă (ex. desincronizare de ceas) nu trebuie să propage eroarea din prima spre
# main. Reîncercăm de câteva ori, la interval scurt, înainte de a renunța.
_BALANCE_RETRY_ATTEMPTS = 3
_BALANCE_RETRY_DELAY_SECONDS = 3


def _f(pos: dict, *keys: str, default: float = 0.0) -> float:
    """Primul câmp prezent (non-None) dintre `keys`, convertit la float."""
    for key in keys:
        val = pos.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return default


def _get_balance_allowance_with_retry(client):
    """`client.get_balance_allowance(...)` cu reîncercări pe erori tranzitorii.

    Polymarket poate întoarce sporadic 401 (Unauthorized/Invalid api key) dintr-o
    desincronizare de ceas sau un glitch de API, ori apelul poate pica pe o eroare de
    rețea de o secundă. Reîncercăm de `_BALANCE_RETRY_ATTEMPTS` ori, la interval de
    `_BALANCE_RETRY_DELAY_SECONDS`; dacă tot eșuează, re-aruncăm ultima excepție spre
    apelant (main o prinde și intră în standby, fără să trântească procesul)."""
    params = BalanceAllowanceParams(
        asset_type=AssetType.COLLATERAL, signature_type=get_signature_type()
    )
    last_exc: Exception | None = None
    for attempt in range(1, _BALANCE_RETRY_ATTEMPTS + 1):
        try:
            return client.get_balance_allowance(params)
        except PolyApiException as exc:
            last_exc = exc
            logger.warning(
                "get_balance_allowance a eșuat (încercarea %d/%d): PolyApiException "
                "status_code=%s — %s",
                attempt, _BALANCE_RETRY_ATTEMPTS, getattr(exc, "status_code", None),
                "reîncerc..." if attempt < _BALANCE_RETRY_ATTEMPTS else "renunț și propag eroarea.",
            )
        except Exception as exc:  # erori tranzitorii de rețea din SDK (requests/httpx/etc.)
            last_exc = exc
            logger.warning(
                "get_balance_allowance a eșuat (încercarea %d/%d): %s — %s",
                attempt, _BALANCE_RETRY_ATTEMPTS, exc,
                "reîncerc..." if attempt < _BALANCE_RETRY_ATTEMPTS else "renunț și propag eroarea.",
            )
        if attempt < _BALANCE_RETRY_ATTEMPTS:
            time.sleep(_BALANCE_RETRY_DELAY_SECONDS)
    raise last_exc  # toate încercările au eșuat → propagăm ultima excepție


def get_account_state() -> dict:
    client = get_clob_client()

    bal_data = _get_balance_allowance_with_retry(client)
    if hasattr(bal_data, "balance"):
        raw_balance = bal_data.balance
    elif isinstance(bal_data, dict):
        raw_balance = bal_data.get("balance", 0)
    else:
        raw_balance = 0

    if hasattr(bal_data, "allowance"):
        raw_allowance = bal_data.allowance
    elif isinstance(bal_data, dict):
        raw_allowance = bal_data.get("allowance", 0)
    else:
        raw_allowance = 0

    pusd_balance = float(raw_balance) / _MICRO_PUSD
    allowance = float(raw_allowance) / _MICRO_PUSD

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
                size = float(pos["size"])
                avg_price = _f(pos, "avgPrice", "avg_price")
                cur_price = _f(pos, "curPrice", "cur_price")
                # Data API returnează deja P/L calculat (exact ca în portofoliul
                # Polymarket); fallback la calcul local din preț × size dacă lipsesc.
                current_value = _f(pos, "currentValue", "current_value",
                                   default=(cur_price * size if cur_price else avg_price * size))
                initial_value = _f(pos, "initialValue", "initial_value",
                                   default=avg_price * size)
                cash_pnl = _f(pos, "cashPnl", "cash_pnl",
                              default=current_value - initial_value)
                active_positions.append({
                    "asset_name":    pos.get("title") or pos.get("asset_name", ""),
                    "token_id":      pos.get("asset") or pos.get("token_id", ""),
                    # Outcome-ul (YES/NO sau numele opțiunii) din Data API, propagat
                    # până în notify_positions pentru raportarea inventarului.
                    "outcome":       pos.get("outcome") or pos.get("outcome_name") or "",
                    "side":          pos.get("side", ""),
                    "size":          size,
                    "avg_price":     avg_price,
                    "cur_price":     cur_price,
                    "current_value": current_value,
                    "initial_value": initial_value,
                    "cash_pnl":      cash_pnl,
                    "percent_pnl":   _f(pos, "percentPnl", "percent_pnl"),
                    "realized_pnl":  _f(pos, "realizedPnl", "realized_pnl"),
                    # eventSlug corespunde slug-ului din `events` la piețele Gamma;
                    # fără acest câmp, filtrul de event clustering din main nu are
                    # cu ce să compare și nu limitează nimic.
                    "event_id":      pos.get("eventSlug") or pos.get("eventId") or pos.get("conditionId") or "",
                })
    except requests.RequestException as exc:
        logger.warning("Could not fetch positions from Data API: %s", exc)

    return {
        "pusd_balance":     pusd_balance,
        "allowance":        allowance,
        "active_positions": active_positions,
    }


def compute_portfolio_summary(state: dict) -> dict:
    """
    Agregă starea contului într-un sumar de portofoliu (ca graficul P/L din
    Polymarket): valoare totală = cash + valoarea curentă a pozițiilor, plus
    P/L nerealizat (suma cashPnl pe pozițiile deschise).
    """
    cash = float(state.get("pusd_balance", 0.0))
    positions = state.get("active_positions", [])
    positions_value = sum(p.get("current_value", 0.0) for p in positions)
    unrealized_pnl = sum(p.get("cash_pnl", 0.0) for p in positions)
    cost_basis = sum(p.get("initial_value", 0.0) for p in positions)
    return {
        "cash":            cash,
        "positions_value": positions_value,
        "total_value":     cash + positions_value,
        "unrealized_pnl":  unrealized_pnl,
        "cost_basis":      cost_basis,
        "n_positions":     len(positions),
        "positions":       positions,
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
