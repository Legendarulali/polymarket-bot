"""
Notificări Discord prin webhook: alerte buy/sell și raport orar de portofoliu.

Modulul e proiectat să nu pice niciodată trading-ul: dacă webhook-ul nu e
configurat funcțiile sunt no-op, iar orice eroare de rețea e prinsă și logată.
"""
import logging
from datetime import datetime, timezone

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 10
_GREEN = 0x2ECC71
_RED = 0xE74C3C
_BLUE = 0x3498DB
_FOOTER = {"text": "Polymarket AI Bot"}
_MARKETS_PREVIEW = 10      # câte piețe se afișează înainte de truncare
_MARKETS_TITLE_CAP = 100   # lungime maximă per titlu în listă
_DESC_LIMIT = 2000         # plafon caractere pentru descriere (limita webhook-urilor)


def _enabled() -> bool:
    return bool(settings.DISCORD_WEBHOOK_URL)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _signed_money(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def _signed_pct(x: float) -> str:
    return f"{'+' if x >= 0 else ''}{x:.1f}%"


def _post(embed: dict) -> None:
    """Trimite un embed la webhook. No-op dacă nu e configurat; nu ridică excepții."""
    if not _enabled():
        return
    try:
        resp = requests.post(
            settings.DISCORD_WEBHOOK_URL,
            json={"embeds": [embed]},
            timeout=_TIMEOUT,
        )
        if resp.status_code >= 300:
            logger.warning("[Discord] Webhook a răspuns %s: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("[Discord] Trimitere notificare eșuată: %s", exc)


def notify_trade(
    side: str,
    title: str,
    price: float,
    size: float,
    total: float,
    *,
    outcome: str | None = None,
    confidence: float | None = None,
    reason: str | None = None,
    order_id: str | None = None,
) -> None:
    """Notificare la o tranzacție executată (BUY sau SELL)."""
    if not _enabled():
        return

    is_buy = side.upper() == "BUY"
    emoji = "🟢" if is_buy else "🔴"
    color = _GREEN if is_buy else _RED

    fields = []
    if outcome:
        fields.append({"name": "Outcome", "value": str(outcome)[:64], "inline": True})
    fields += [
        {"name": "Preț", "value": f"${price:.4f}", "inline": True},
        {"name": "Size", "value": f"{size:.2f}", "inline": True},
        {"name": "Total", "value": _money(total), "inline": True},
    ]
    if confidence is not None:
        fields.append({"name": "Confidence", "value": f"{confidence * 100:.0f}%", "inline": True})
    if reason:
        fields.append({"name": "Motiv", "value": str(reason)[:1024], "inline": False})

    embed = {
        "title": f"{emoji} {side.upper()} • {str(title)[:230]}",
        "color": color,
        "fields": fields,
        "timestamp": _now_iso(),
        "footer": _FOOTER,
    }
    if order_id:
        embed["footer"] = {"text": f"Polymarket AI Bot • order {str(order_id)[:40]}"}
    _post(embed)


def notify_portfolio(
    stats: dict,
    *,
    prev_total: float | None = None,
    session_start_total: float | None = None,
) -> None:
    """Mesaj 1 — Status financiar: valoare totală, cash, poziții și P/L nerealizat
    (total, ultima oră, de la pornire). Lista de poziții se trimite separat prin
    `notify_positions`; aici sunt DOAR cifrele financiare. `stats` = sumarul din
    compute_portfolio_summary.
    """
    if not _enabled():
        return

    total = stats.get("total_value", 0.0)
    cash = stats.get("cash", 0.0)
    positions_value = stats.get("positions_value", 0.0)
    unrealized = stats.get("unrealized_pnl", 0.0)
    cost_basis = stats.get("cost_basis", 0.0)
    unreal_pct = (unrealized / cost_basis * 100) if cost_basis else 0.0

    fields = [
        {"name": "💰 Valoare totală", "value": _money(total), "inline": True},
        {"name": "💵 Cash", "value": _money(cash), "inline": True},
        {"name": "📈 În poziții", "value": _money(positions_value), "inline": True},
        {"name": "P/L nerealizat (total)",
         "value": f"{_signed_money(unrealized)} ({_signed_pct(unreal_pct)})", "inline": True},
    ]
    if prev_total is not None:
        fields.append({"name": "P/L ultima oră", "value": _signed_money(total - prev_total), "inline": True})
    if session_start_total is not None:
        sdelta = total - session_start_total
        spct = (sdelta / session_start_total * 100) if session_start_total else 0.0
        fields.append({
            "name": "P/L de la pornire",
            "value": f"{_signed_money(sdelta)} ({_signed_pct(spct)})",
            "inline": True,
        })

    embed = {
        "title": "💰 Status Portofoliu",
        "color": _GREEN if unrealized >= 0 else _RED,
        "fields": fields,
        "timestamp": _now_iso(),
        "footer": _FOOTER,
    }
    _post(embed)


def notify_positions(positions: list) -> None:
    """Mesaj 2 — Inventarul pozițiilor deschise, embed separat de statusul financiar.

    Fiecare linie: titlu | outcome | sumă investită | P/L curent. Listă goală →
    „Nicio poziție deschisă". Plafonat la _DESC_LIMIT caractere (truncat cu „…și încă N").
    Notă: `outcome` apare doar dacă e prezent în dict-ul poziției (Data API nu îl
    expune prin reconciliation.py); altfel se afișează „—".
    """
    if not _enabled():
        return

    if not positions:
        _post({
            "title": "📦 Poziții Active",
            "description": "Nicio poziție deschisă.",
            "color": _BLUE,
            "timestamp": _now_iso(),
            "footer": _FOOTER,
        })
        return

    ordered = sorted(positions, key=lambda p: p.get("current_value", 0.0), reverse=True)
    shown = ordered[:_MARKETS_PREVIEW]
    lines = []
    for p in shown:
        name = str(p.get("asset_name") or "?")[:60]
        outcome = str(p.get("outcome") or "—")
        invested = _money(float(p.get("initial_value", 0.0) or 0.0))
        pnl = float(p.get("cash_pnl", 0.0) or 0.0)
        mark = "🟢" if pnl >= 0 else "🔴"
        lines.append(f"{mark} {name} | {outcome} | {invested} | {_signed_money(pnl)}")
    remaining = len(ordered) - len(shown)
    if remaining > 0:
        lines.append(f"…și încă {remaining} poziții")

    description = "\n".join(lines)
    if len(description) > _DESC_LIMIT:
        description = description[:_DESC_LIMIT - 1] + "…"

    _post({
        "title": f"📦 Poziții Active ({len(ordered)})",
        "description": description,
        "color": _BLUE,
        "timestamp": _now_iso(),
        "footer": _FOOTER,
    })


def notify_markets(markets: list, *, total: int | None = None) -> None:
    """Listă a piețelor active procesate, trimisă separat de raportul de portofoliu.

    Afișează primele _MARKETS_PREVIEW piețe; dacă sunt mai multe, adaugă
    „…și încă N piețe". Descrierea e plafonată la _DESC_LIMIT caractere (limita
    webhook-urilor). Acceptă fie dict-uri de piață (cu 'question'/'title'), fie
    string-uri. Ca toate notificările: no-op fără webhook și nu ridică excepții.
    """
    if not _enabled():
        return

    titles = [
        (str(m.get("question") or m.get("title") or "?") if isinstance(m, dict) else str(m))
        for m in markets
    ]
    count = total if total is not None else len(titles)

    shown = titles[:_MARKETS_PREVIEW]
    lines = [f"{i}. {t[:_MARKETS_TITLE_CAP]}" for i, t in enumerate(shown, 1)]
    remaining = count - len(shown)
    if remaining > 0:
        lines.append(f"…și încă {remaining} piețe")

    description = "\n".join(lines) or "Nicio piață activă."
    if len(description) > _DESC_LIMIT:
        description = description[:_DESC_LIMIT - 1] + "…"

    embed = {
        "title": f"📋 Piețe active procesate ({count})",
        "description": description,
        "color": _BLUE,
        "timestamp": _now_iso(),
        "footer": _FOOTER,
    }
    _post(embed)
