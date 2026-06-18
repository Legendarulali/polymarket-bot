import json
import os
import random
import time
import sys
import signal
import logging
from collections import Counter

import httpx

from config.settings import settings
from src.reconciliation import log_reconciliation_report, get_account_state, compute_portfolio_summary
from src.execution import Side, place_limit_order, compute_order_size, _TARGET_ORDER_VALUE
from src.clob_client import get_clob_client
from src.news_monitor import fetch_free_news
from src.ai_analyser import analyse_market, analyse_match
from src.football import parse_match_market, format_legs, extract_teams
from src.allowance import ensure_collateral_approved
from src.position_manager import check_exit_conditions
from src.notifications import notify_trade, notify_portfolio, notify_positions

logger = logging.getLogger(__name__)

RUNNING = True
_LOOP_INTERVAL = 60          # secunde între iterații normale
_LOW_BALANCE_SLEEP = 15 * 60  # 15 minute sleep când balanța e insuficientă
_CACHE_TTL_SECONDS = 4 * 3600 # cooldown: re-analizează o piață cel mult o dată la 4h (persistat în history.json)
_GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
_GAMMA_PAGE_SIZE = 100                                           # piețe per pagină la paginare Gamma
# Plafon OPȚIONAL de siguranță pe câte piețe se preiau per buclă. 0 (implicit) =
# fără plafon → paginarea dinamică preia TOATE piețele active până când API-ul
# întoarce o pagină goală/parțială. Setează-l (ex. 2000) doar ca să limitezi
# numărul de cereri către Gamma pe rețele lente/cu rate-limit agresiv.
_MARKET_SCAN_LIMIT = int(os.getenv("MARKET_SCAN_LIMIT", "0"))
# Sortare Gamma (câmp + direcție). Gol = ordinea implicită a API-ului. Ex:
# GAMMA_ORDER=volumeNum (lichide întâi) sau GAMMA_ORDER=startDate (cele mai noi —
# utile pentru meciurile create azi). Dacă valoarea e respinsă, se reia fără sortare.
_GAMMA_ORDER = os.getenv("GAMMA_ORDER", "").strip()
_GAMMA_ASCENDING = os.getenv("GAMMA_ASCENDING", "false").strip()
# Deep-scan opțional pe tag-uri Gamma specifice (ex: fotbal / FIFA World Cup).
# Gol = dezactivat. Necesită tag_id-urile NUMERICE reale (vezi .env.example).
_GAMMA_TAG_IDS = [t.strip() for t in os.getenv("GAMMA_TAG_IDS", "").split(",") if t.strip()]
_INTER_ANALYSIS_DELAY = 3                                        # secunde throttle între analize LLM
_MIN_BALANCE_PUSD = 1.0
_MIN_CONFIDENCE = 0.70
_MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "10000"))  # Ignoră piețele sub acest volum ($)
# Fereastră de probabilitate YES UNICĂ pentru toate tipurile de piață (meci,
# outright, general): [0.45, 0.95]. Ține botul concentrat pe meciurile
# competitive (probabilități medii-mari) și ignoră „zgomotul" de probabilitate
# foarte mică, lăsând decizia reală pe seama LLM-ului + pragului de confidence.
# Configurabilă din mediu (PROBABILITY_MIN / PROBABILITY_MAX).
_MIN_PROBABILITY = float(os.getenv("PROBABILITY_MIN", "0.45"))  # Ignoră probabilitățile YES prea mici (zgomot)
_MAX_PROBABILITY = float(os.getenv("PROBABILITY_MAX", "0.95"))  # Ignoră dacă probabilitatea YES e quasi-certă
_MAX_POSITIONS_PER_EVENT = 2     # Max poziții per event_id (piețe generale)
_MAX_OUTRIGHT_PER_EVENT = 1      # Max poziții pe piețe de tip "outright winner"
# Mod universal (IMPLICIT): botul evaluează TOATE piețele active — politice,
# crypto, generale ȘI meciuri de fotbal (acestea din urmă pe ruta dedicată 1X2).
# Opțional, FOOTBALL_ONLY=true restrânge scanarea DOAR la meciuri individuale de
# fotbal (ignoră complet outright/general) — util în timpul unui turneu.
_FOOTBALL_ONLY = os.getenv("FOOTBALL_ONLY", "false").strip().lower() in ("1", "true", "yes", "on")

# Istoric persistent al piețelor analizate: {market_id: timestamp_float}.
# Încărcat din _HISTORY_FILE la pornire; folosit pentru a filtra piețele
# analizate în ultimele _CACHE_TTL_SECONDS înainte de procesare.
# Calea poate fi suprascrisă cu HISTORY_FILE (ex: un volum montat pe Railway,
# unde /app e efemer și root-only); altfel, history.json lângă main.py.
_HISTORY_FILE = os.getenv("HISTORY_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "history.json"
)
_analyzed_history: dict[str, float] = {}

# Stare raport orar P/L pe Discord (ts=0 → primul raport e trimis imediat la pornire)
_last_pnl_report_ts: float = 0.0
_last_report_value: float | None = None
_session_start_value: float | None = None

# Statistici de filtrare per scanare (motiv → câte piețe). Resetat la fiecare buclă;
# logate ca sumar după scanare pentru a vedea de ce sunt „ignorate" piețele.
_scan_stats: Counter = Counter()

_MATCH_KEYWORDS = (" vs ", " vs. ", "match winner")
_MATCH_SLUG_PATTERNS = ("-vs-", "-v-", "-beat-", "match-winner")
# Fraze de tip outright / pariu pe termen lung — NU sunt meciuri 1X2 și se exclud
# din modul focus-fotbal: câștigător de turneu, câștigător/clasare de grupă,
# avansare, golgheter etc. (verificate DUPĂ detecția de meci, care are precedență).
_OUTRIGHT_PHRASES = (
    "win the 2026 fifa world cup",
    "win the fifa world cup",
    "win the world cup",
    "world cup winner",
    "outright winner",
    "championship winner",
    "tournament winner",
    "win group",            # câștigător de grupă (ex: 'Will Brazil win Group A?')
    "win their group",
    "to win the group",
    "group winner",
    "group stage winner",
    "to advance",           # avansare / calificare (prop, nu meci 1X2)
    "advance to",
    "to qualify",
    "reach the final",
    "reach the knockout",
    "top goalscorer",
    "golden boot",
    "top scorer",
)
_TYPE_LABELS = {"match": "Meci", "outright": "Outright", "general": "General"}


def _classify_market(title: str, slug: str = "") -> str:
    """Clasifică piața ca 'match', 'outright' sau 'general' după titlu și slug.

    Ordinea contează: meciul (head-to-head) are PRECEDENȚĂ față de outright, ca
    un titlu de tip 'World Cup Match Winner: USA vs Iran' să fie clasificat MECI,
    nu câștigător de turneu. 'Win the World Cup' (fără head-to-head) rămâne
    outright.

    Detecția de meci e robustă: pe lângă cuvintele-cheie ('vs'/'match winner') și
    slug, folosește `extract_teams`, care prinde și 'A v B', 'A versus B' și
    formatul binar 'Will A beat B?' (curent la meciurile World Cup pe Polymarket).
    """
    lower = title.lower()
    slug_lower = (slug or "").lower()

    # 1) Meci individual (head-to-head) — verificat PRIMUL. Dacă putem extrage
    #    cele două echipe, e cu certitudine un meci, nu un outright.
    if (
        any(kw in lower for kw in _MATCH_KEYWORDS)
        or any(p in slug_lower for p in _MATCH_SLUG_PATTERNS)
        or extract_teams(title, slug) != (None, None)
    ):
        return "match"

    # 2) Outright / câștigător turneu sau grupă (ex: 'Win the World Cup', 'win Group A').
    if any(phrase in lower for phrase in _OUTRIGHT_PHRASES):
        return "outright"

    # 3) Restul (politice, generale).
    return "general"


def _is_match_market(market: dict) -> bool:
    """True dacă piața e de tip meci (head-to-head). Folosit pentru a prioritiza
    meciurile (adesea iminente) la începutul cozii de scanare."""
    return _classify_market(
        market.get("question") or market.get("title") or "", market.get("slug", "")
    ) == "match"


def handle_shutdown(signum, frame):
    global RUNNING
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating graceful shutdown...", sig_name)
    RUNNING = False


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def _interruptible_sleep(seconds: int) -> None:
    """Sleep în incremente de 1s astfel încât SIGTERM e onorat prompt."""
    for _ in range(seconds):
        if not RUNNING:
            break
        time.sleep(1)


def _fetch_markets_page(params: dict) -> list | None:
    """Un singur GET la Gamma /markets. Lista (poate fi goală) sau None la eroare/non-200."""
    try:
        response = httpx.get(_GAMMA_MARKETS_URL, params=params, timeout=10)
        if response.status_code != 200:
            logger.warning("[Agent] Gamma a răspuns %s (params=%s).", response.status_code, params)
            return None
        return response.json()
    except Exception as exc:
        logger.warning("[Agent] Could not fetch markets (params=%s): %s", params, exc)
        return None


def _collect_active_markets(extra_params: dict, seen_ids: set, out: list, cap: int = 0) -> None:
    """Paginează DINAMIC Gamma /markets și adaugă în `out` TOATE piețele noi
    (deduplicate după id), incrementând offset-ul cu _GAMMA_PAGE_SIZE la fiecare
    pas, până când API-ul întoarce o listă goală sau o pagină parțială (mai
    puține rezultate decât o pagină completă). Oprește și la prima eroare.

    Între cereri se inserează o mică pauză (time.sleep) pentru a evita rate-limit
    (HTTP 429). `cap` > 0 = plafon opțional de siguranță pe totalul din `out`
    (0 = fără plafon, preia tot).

    Dacă GAMMA_ORDER e setat dar respins de API (non-200), reia pagina FĂRĂ sortare,
    astfel încât o valoare de `order` greșită să nu golească complet scanarea.
    """
    order_params = {"order": _GAMMA_ORDER, "ascending": _GAMMA_ASCENDING} if _GAMMA_ORDER else {}
    offset = 0
    while True:
        base = {"active": "true", "closed": "false",
                "limit": _GAMMA_PAGE_SIZE, "offset": offset, **extra_params}
        page = _fetch_markets_page({**base, **order_params})
        if page is None and order_params:
            logger.warning("[Agent] Reîncerc fără 'order=%s' (posibil nesuportat de Gamma).", _GAMMA_ORDER)
            page = _fetch_markets_page(base)
        if not page:
            return  # eroare sau listă goală → gata

        for m in page:
            mid = str(m.get("id", ""))
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                out.append(m)

        if cap and len(out) >= cap:
            logger.info("[Agent] Plafon de siguranță MARKET_SCAN_LIMIT=%d atins — opresc paginarea.", cap)
            return
        if len(page) < _GAMMA_PAGE_SIZE:
            return  # ultima pagină (mai puține rezultate decât o pagină completă)
        offset += _GAMMA_PAGE_SIZE
        time.sleep(0.1)  # mică pauză între cereri pentru a evita rate-limit (HTTP 429)


def fetch_active_markets(limit: int = _MARKET_SCAN_LIMIT) -> list[dict]:
    """Preia DINAMIC toate piețele active din Gamma, paginat (offset) și
    deduplicat după id, până când API-ul nu mai întoarce rezultate.

    Paginarea aduce și meciurile individuale (head-to-head), care altfel sunt
    îngropate sub piețele mari de tip outright în primele ~30 de rezultate.
    Opțional, face un deep-scan suplimentar pe tag-urile din GAMMA_TAG_IDS
    (ex: fotbal / FIFA World Cup), ca meciurile de azi să nu fie ratate.
    `limit` > 0 = plafon opțional de siguranță (0 = fără plafon, preia tot).
    """
    seen_ids: set = set()
    markets: list[dict] = []
    _collect_active_markets({}, seen_ids, markets, cap=limit)
    for tag_id in _GAMMA_TAG_IDS:
        _collect_active_markets({"tag_id": tag_id}, seen_ids, markets, cap=limit)
    logger.info("[Agent] Gamma: %d piețe active preluate (paginat dinamic, %d tag-uri deep-scan).",
                len(markets), len(_GAMMA_TAG_IDS))
    return markets


def _token_for_outcome(market: dict, outcome: str) -> str | None:
    """Extrage token_id din clobTokenIds (index 0=YES, 1=NO) returnat de Gamma API ca string JSON."""
    raw = market.get("clobTokenIds")
    if raw:
        try:
            ids: list[str] = json.loads(raw) if isinstance(raw, str) else raw
            if outcome.upper() == "YES" and len(ids) > 0:
                return ids[0]
            if outcome.upper() == "NO" and len(ids) > 1:
                return ids[1]
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("[Agent] Eroare parsare clobTokenIds: %s | raw=%r", exc, raw)

    # Fallback: câmpul tokens cu obiecte {outcome, token_id}
    for token in market.get("tokens", []):
        if token.get("outcome", "").upper() == outcome.upper():
            return token.get("token_id")

    logger.debug(
        "[Agent] Token negăsit pentru outcome=%s | chei piață: %s",
        outcome,
        list(market.keys()),
    )
    return None


def _current_price(market: dict, outcome: str) -> float:
    """Returnează prețul YES sau NO din datele Gamma, cu fallback la 0.5."""
    # Gamma API returnează outcomePrices ca string JSON (ex: '["0.72", "0.28"]'),
    # la fel ca clobTokenIds — trebuie dezserializat înainte de indexare.
    prices = market.get("outcomePrices") or []
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            logger.warning("[Agent] Eroare parsare outcomePrices: %r", prices[:80])
            prices = []

    try:
        if outcome.upper() == "YES" and len(prices) > 0:
            return float(prices[0])
        if outcome.upper() == "NO" and len(prices) > 1:
            return float(prices[1])
    except (ValueError, TypeError):
        pass

    try:
        ask = market.get("bestAsk")
        if ask is not None:
            return float(ask)
    except (ValueError, TypeError):
        pass
    return 0.5


def _live_best_ask(token_id: str) -> float | None:
    """Best ask live din orderbook CLOB, sau None dacă e indisponibil."""
    try:
        ob = get_clob_client().get_order_book(token_id=token_id)
        asks = ob.asks if hasattr(ob, "asks") else ob.get("asks", [])
        if not asks:
            return None
        return min(
            float(a.price if hasattr(a, "price") else a["price"]) for a in asks
        )
    except Exception as exc:
        logger.warning("[Agent] Orderbook indisponibil pentru token=%s: %s", token_id, exc)
        return None


def _load_history() -> None:
    """Încarcă istoricul piețelor analizate din history.json (prima instrucțiune la rulare).

    Intrările mai vechi decât _CACHE_TTL_SECONDS sunt eliminate la încărcare, pentru
    a limita creșterea fișierului. Acceptă atât formatul {id: timestamp} cât și o
    listă simplă de ID-uri (considerate analizate „acum").
    """
    global _analyzed_history
    now = time.time()
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        _analyzed_history = {}
        logger.info("[History] %s inexistent — pornesc cu istoric gol.", _HISTORY_FILE)
        return
    except (json.JSONDecodeError, OSError) as exc:
        _analyzed_history = {}
        logger.warning("[History] Nu am putut citi %s (%s) — pornesc cu istoric gol.", _HISTORY_FILE, exc)
        return

    parsed: dict[str, float] = {}
    if isinstance(raw, dict):
        for mid, ts in raw.items():
            try:
                parsed[str(mid)] = float(ts)
            except (ValueError, TypeError):
                continue
    elif isinstance(raw, list):
        parsed = {str(mid): now for mid in raw}

    _analyzed_history = {mid: ts for mid, ts in parsed.items() if now - ts < _CACHE_TTL_SECONDS}
    expired = len(parsed) - len(_analyzed_history)
    logger.info(
        "[History] Încărcate %d piețe analizate recent din %s (%d expirate, ignorate).",
        len(_analyzed_history), _HISTORY_FILE, expired,
    )
    if expired:
        _save_history()  # rescrie fișierul curățat de intrările expirate


def _save_history() -> None:
    """Persistă istoricul piețelor analizate în history.json (scriere atomică).

    Creează directorul părinte dacă lipsește; erorile de permisiuni/IO sunt
    prinse și logate, fără a opri botul (history e best-effort).
    """
    tmp = f"{_HISTORY_FILE}.tmp"
    try:
        parent = os.path.dirname(_HISTORY_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_analyzed_history, fh)
        os.replace(tmp, _HISTORY_FILE)
    except OSError as exc:
        logger.warning("[History] Nu am putut salva %s: %s", _HISTORY_FILE, exc)


def _is_recently_analyzed(market_id: str) -> bool:
    """True dacă piața a fost analizată în ultimele _CACHE_TTL_SECONDS (cooldown activ)."""
    ts = _analyzed_history.get(market_id)
    return ts is not None and (time.time() - ts) < _CACHE_TTL_SECONDS


def _record_analyzed(market_id: str) -> None:
    """Marchează o piață ca analizată acum și persistă imediat istoricul."""
    _analyzed_history[market_id] = time.time()
    _save_history()


def _extract_event_id(market: dict) -> str:
    """
    Identificatorul evenimentului pieței Gamma. Preferă slug-ul din `events`,
    care corespunde câmpului `eventSlug` din pozițiile Data API — altfel
    clustering-ul pe eveniment nu se poate corela cu pozițiile deschise.
    """
    events = market.get("events") or []
    if isinstance(events, list) and events and isinstance(events[0], dict):
        slug_or_id = events[0].get("slug") or events[0].get("id")
        if slug_or_id:
            return str(slug_or_id)
    return str(
        market.get("eventId")
        or market.get("event_id")
        or market.get("conditionId")
        or ""
    )


def _maybe_send_pnl_report(account_state: dict) -> None:
    """Raportul periodic de portofoliu pe Discord, în 2 mesaje grupate, trimise
    o singură dată la fiecare PNL_REPORT_INTERVAL_HOURS (NU la fiecare buclă):
      1) status financiar (cash, P/L, valoare totală),
      2) inventarul pozițiilor deschise (sau „Niciuna").
    Ambele se trimit la fiecare interval, ca raportarea să fie predictibilă."""
    global _last_pnl_report_ts, _last_report_value, _session_start_value

    interval_h = settings.PNL_REPORT_INTERVAL_HOURS
    if interval_h <= 0 or not settings.DISCORD_WEBHOOK_URL:
        return
    now = time.time()
    if now - _last_pnl_report_ts < interval_h * 3600:
        return

    summary = compute_portfolio_summary(account_state)
    if _session_start_value is None:
        _session_start_value = summary["total_value"]

    logger.info(
        "[P/L] Raport portofoliu: total=$%.2f | cash=$%.2f | poziții=$%.2f | "
        "P/L nerealizat=$%.2f | %d poziții",
        summary["total_value"], summary["cash"], summary["positions_value"],
        summary["unrealized_pnl"], summary["n_positions"],
    )
    # Mesaj 1: status financiar.
    notify_portfolio(
        summary,
        prev_total=_last_report_value,
        session_start_total=_session_start_value,
    )
    # Mesaj 2: inventarul pozițiilor deschise (sau „Niciuna").
    notify_positions(summary.get("positions", []))

    _last_report_value = summary["total_value"]
    _last_pnl_report_ts = now


def process_market(market: dict, account_state: dict) -> None:
    """Orchestrează filtrarea unei piețe și decide ruta de analiză.

    Mod universal (implicit): toate piețele sunt evaluate. Meciurile structurate
    1X2 (Match Odds) merg pe fluxul dedicat `_process_match_1x2` (rezultate mapate
    1/X/2); restul (politice, crypto, generale, outright) merg pe fluxul binar
    YES/NO moștenit. Opțional, FOOTBALL_ONLY=true ignoră tot ce nu e meci."""
    active_positions: list[dict] = account_state.setdefault("active_positions", [])

    title = market.get("question") or market.get("title") or "Unknown market"
    market_id = market.get("id", "?")
    market_type = _classify_market(title, market.get("slug", ""))
    type_label = _TYPE_LABELS.get(market_type, market_type)

    # Log de nivel DEBUG: se emite o linie per FIECARE piață scanată (mii/buclă).
    # La INFO ar inunda Railway (rate-limit 500 loguri/s) — vezi sumarul [Scan] de
    # la finalul buclei pentru imaginea de ansamblu.
    logger.debug("[Agent] Procesez piața id=%s [Tip: %s]: %s", market_id, type_label, title[:80])

    # Limita globală de poziții se re-verifică per piață: în aceeași iterație
    # pot fi deschise mai multe poziții noi, iar verificarea din bucla main
    # vede doar starea de la începutul iterației.
    if (
        settings.MAX_ACTIVE_POSITIONS > 0
        and len(active_positions) >= settings.MAX_ACTIVE_POSITIONS
    ):
        _scan_stats["expunere maximă"] += 1
        logger.debug(
            "[FILTER] Limită de expunere atinsă (%d/%d poziții). Skip piața '%s'.",
            len(active_positions), settings.MAX_ACTIVE_POSITIONS, title[:60],
        )
        return

    # --- FILTRU 0 (OPȚIONAL): mod focus pe fotbal (FOOTBALL_ONLY=true). Implicit
    # dezactivat — botul e universal și procesează toate tipurile de piață. ---
    if _FOOTBALL_ONLY and market_type != "match":
        _scan_stats["non-meci (outright/general)"] += 1
        logger.debug(
            "[FILTER] Ignor piața '%s' [%s] — FOOTBALL_ONLY activ, nu e meci 1X2.",
            title[:60], type_label,
        )
        return

    # --- FILTRU 1: Volum minim ---
    volume = 0.0
    try:
        volume = float(market.get("volume") or market.get("volumeNum") or 0)
    except (ValueError, TypeError):
        pass
    if volume < _MIN_VOLUME_USD:
        _scan_stats["volum mic"] += 1
        logger.debug(
            "[FILTER] Ignor piața '%s' [%s] (Volum prea mic: $%.0f < $%.0f).",
            title[:60], type_label, volume, _MIN_VOLUME_USD,
        )
        return

    event_id = _extract_event_id(market)

    # Meci structurat 1X2 → flux dedicat cu maparea 1/X/2. Dacă piața e un meci
    # dar nu are structura 1X2 (rezultate binare Yes/No), cade pe fluxul moștenit.
    if market_type == "match":
        parsed = parse_match_market(market)
        if parsed:
            _process_match_1x2(market, account_state, parsed, event_id, title, market_id)
            return
        logger.debug(
            "[Meci] '%s' nu e structurat 1X2 (rezultate binare) — folosesc fluxul YES/NO.",
            title[:60],
        )

    _process_legacy_binary(market, account_state, event_id, title, market_id, market_type, type_label)


def _process_match_1x2(
    market: dict,
    account_state: dict,
    parsed: dict,
    event_id: str,
    title: str,
    market_id,
) -> None:
    """Flux de analiză pentru un meci 1X2 (Match Odds).

    Logează numele meciului și cotele 1/X/2, aplică fereastra de probabilitate
    pe fiecare rezultat (YES = prețul rezultatului), apoi cere LLM-ului să aleagă
    un cod (1/X/2) și plasează ordinul pe token-ul rezultatului ales."""
    home, away = parsed["home"], parsed["away"]
    # Cerință de produs: logurile arată clar meciul + cotele 1/X/2.
    logger.info("[Meci] %s | %s", parsed["label"], format_legs(parsed))

    # --- FILTRU 2: Fereastra de probabilitate [0.45, 0.95] pe fiecare leg 1X2 ---
    candidates = [
        leg for leg in parsed["legs"]
        if _MIN_PROBABILITY < leg["price"] < _MAX_PROBABILITY
    ]
    if not candidates:
        _scan_stats["probabilitate în afara ferestrei"] += 1
        logger.debug(
            "[FILTER] Ignor meciul '%s' — niciun rezultat 1X2 în fereastra [%.2f, %.2f] (%s).",
            parsed["label"], _MIN_PROBABILITY, _MAX_PROBABILITY, format_legs(parsed),
        )
        return

    # Culegere știri + analiză 1X2 via Groq.
    news = fetch_free_news(query=title[:100])
    logger.info("[Agent] %d știri găsite pentru '%s'.", len(news), parsed["label"][:50])

    decision = analyse_match(home, away, candidates, news)

    _record_analyzed(str(market_id))
    _scan_stats["analizate (LLM)"] += 1
    logger.debug(
        "[Agent] Piață id=%s marcată ca analizată în history.json (cooldown=%dh).",
        market_id, _CACHE_TTL_SECONDS // 3600,
    )
    logger.info(
        "[Agent] Decizie AI 1X2: trade=%s | outcome=%s | confidence=%.2f | reason=%s",
        decision["trade"], decision["outcome"], decision["confidence"], decision["reason"],
    )

    if not decision["trade"]:
        return
    if decision["confidence"] < _MIN_CONFIDENCE:
        logger.info(
            "[Agent] Confidence %.2f sub pragul %.2f. Ordin blocat.",
            decision["confidence"], _MIN_CONFIDENCE,
        )
        return

    pick = decision["outcome"]
    leg = parsed["by_code"].get(pick)
    if not leg:
        logger.warning("[Agent] Cod 1X2 ales (%s) fără leg corespunzător. Ordin blocat.", pick)
        return

    _execute_trade(
        market, account_state,
        token_id=leg["token_id"],
        price=leg["price"],
        outcome_label=f"{pick} ({leg['name']})",
        decision=decision,
        event_id=event_id,
        title=title,
    )


def _process_legacy_binary(
    market: dict,
    account_state: dict,
    event_id: str,
    title: str,
    market_id,
    market_type: str,
    type_label: str,
) -> None:
    """Flux binar YES/NO moștenit (piețe non-1X2 sau focus pe fotbal dezactivat)."""
    active_positions: list[dict] = account_state.setdefault("active_positions", [])

    # --- FILTRU 2: Fereastra de probabilitate pe YES ---
    yes_price_raw = _current_price(market, "YES")
    if not (_MIN_PROBABILITY < yes_price_raw < _MAX_PROBABILITY):
        _scan_stats["probabilitate în afara ferestrei"] += 1
        logger.debug(
            "[FILTER] Ignor piața '%s' [%s] (Probabilitate YES=%.2f în afara ferestrei [%.2f, %.2f]).",
            title[:60], type_label, yes_price_raw, _MIN_PROBABILITY, _MAX_PROBABILITY,
        )
        return

    # --- FILTRU 3: Event clustering (meciurile nu sunt plafonate pe event_id) ---
    if market_type == "match":
        logger.debug("[FILTER] Piață de tip meci detectată ('%s') — skip event clustering.", title[:60])
    elif event_id:
        limit = _MAX_OUTRIGHT_PER_EVENT if market_type == "outright" else _MAX_POSITIONS_PER_EVENT
        event_positions = sum(1 for pos in active_positions if pos.get("event_id") == event_id)
        if event_positions >= limit:
            _scan_stats["event clustering"] += 1
            logger.debug(
                "[FILTER] Ignor piața '%s' [%s] (eveniment %s: %d/%d poziții).",
                title[:60], type_label, event_id, event_positions, limit,
            )
            return

    news = fetch_free_news(query=title[:100])
    logger.info("[Agent] %d știri găsite pentru '%s'.", len(news), title[:50])

    yes_price = _current_price(market, "YES")
    decision = analyse_market(title, yes_price, news)

    _record_analyzed(str(market_id))
    _scan_stats["analizate (LLM)"] += 1
    logger.debug(
        "[Agent] Piață id=%s marcată ca analizată în history.json (cooldown=%dh).",
        market_id, _CACHE_TTL_SECONDS // 3600,
    )
    logger.info(
        "[Agent] Decizie AI: trade=%s | outcome=%s | confidence=%.2f | reason=%s",
        decision["trade"], decision["outcome"], decision["confidence"], decision["reason"],
    )

    if not decision["trade"]:
        return
    if decision["confidence"] < _MIN_CONFIDENCE:
        logger.info(
            "[Agent] Confidence %.2f sub pragul %.2f. Ordin blocat.",
            decision["confidence"], _MIN_CONFIDENCE,
        )
        return

    outcome = decision["outcome"]
    _execute_trade(
        market, account_state,
        token_id=_token_for_outcome(market, outcome),
        price=_current_price(market, outcome),
        outcome_label=outcome,
        decision=decision,
        event_id=event_id,
        title=title,
    )


def _execute_trade(
    market: dict,
    account_state: dict,
    *,
    token_id: str | None,
    price: float,
    outcome_label: str,
    decision: dict,
    event_id: str,
    title: str,
) -> None:
    """Verificări de risc finale + plasare ordin BUY + actualizare stare locală.

    Comun fluxului 1X2 și celui binar: balanță minimă, token valid, dedup pe
    token, preț valid, protecție anti-slippage față de best ask live și
    dimensionare ordin în limita balanței."""
    pusd_balance: float = account_state.get("pusd_balance", 0.0)
    active_positions: list[dict] = account_state.setdefault("active_positions", [])

    # Polymarket respinge ordinele sub $1 valoare totală: dacă balanța e sub acest
    # minim absolut nu putem plasa niciun ordin valid → blocăm. Între $1 și $5
    # ordinul NU mai e blocat, ci dimensionat dinamic (vezi mai jos).
    if pusd_balance < _MIN_BALANCE_PUSD:
        logger.warning(
            "[Agent] Balanță %.4f pUSD sub minimul Polymarket de $%.2f — ordin imposibil. Blocat.",
            pusd_balance, _MIN_BALANCE_PUSD,
        )
        return

    if not token_id:
        logger.warning("[Agent] Token negăsit pentru outcome=%s (piață '%s').", outcome_label, title[:60])
        return

    # Verificare poziție duplicată
    if any(pos.get("token_id") == token_id for pos in active_positions):
        logger.info(
            "[Agent] Poziție deja existentă pentru token_id %s, se evită deschiderea unei poziții duplicate.",
            token_id,
        )
        return

    if price < 0.01:
        logger.warning("[Agent] Preț invalid (%.4f) pentru token=%s. Ordin blocat.", price, token_id)
        return

    # Protecție anti-slippage: prețul Gamma poate fi stale față de orderbook.
    # Dacă ask-ul real e mult peste prețul Gamma, piața s-a mișcat împotriva
    # noastră — skip. Dacă e sub, cumpărăm la ask (preț mai bun, fill imediat).
    if settings.SLIPPAGE_TOLERANCE > 0:
        best_ask = _live_best_ask(token_id)
        if best_ask is not None:
            if best_ask > price * (1 + settings.SLIPPAGE_TOLERANCE):
                logger.warning(
                    "[Agent] Slippage prea mare: ask live $%.4f > preț Gamma $%.4f "
                    "(+%.0f%% toleranță). Ordin blocat.",
                    best_ask, price, settings.SLIPPAGE_TOLERANCE * 100,
                )
                return
            price = min(price, best_ask)

    # Dimensionare dinamică: țintim o valoare de $5/ordin, dar dacă balanța
    # disponibilă e sub $5 folosim cât mai mult din ea (compute_order_size
    # garantează size*price <= balanță). Balanța sub $1 e deja blocată mai sus,
    # deci aici ordinul e mereu plasabil — nu mai blocăm pentru „cost > balanță".
    estimated_size = compute_order_size(price, budget=pusd_balance)
    order_total = round(estimated_size * price, 4)
    if pusd_balance < _TARGET_ORDER_VALUE:
        logger.info(
            "[Agent] Balanță %.4f pUSD sub ținta de $%.2f — dimensionez ordinul la $%.2f (size≈%.2f).",
            pusd_balance, _TARGET_ORDER_VALUE, order_total, estimated_size,
        )

    logger.warning(
        "[TRADE] AI → BUY %s | '%s' | token=%s | price=$%.4f | size≈%.2f | total≈$%.2f | confidence=%.2f",
        outcome_label, title[:60], token_id, price, estimated_size, order_total,
        decision.get("confidence", 0.0),
    )

    tick_size = str(market.get("orderPriceMinTickSize") or "0.01")

    result = place_limit_order(
        token_id=token_id,
        price=price,
        side=Side.BUY,
        tick_size=tick_size,
        size=estimated_size,
    )

    if result.get("success"):
        logger.info("[Agent] Ordin plasat cu succes | order_id=%s", result.get("order_id"))
        # Actualizează starea locală imediat: altfel piețele următoare din
        # aceeași iterație văd balanța veche (risc de overspend) și nu văd
        # poziția nouă (dedup și event clustering ar fi ocolite).
        account_state["pusd_balance"] = round(pusd_balance - order_total, 6)
        active_positions.append({
            "asset_name":    title,
            "token_id":      token_id,
            "outcome":       outcome_label,
            "side":          "BUY",
            "size":          estimated_size,
            "avg_price":     price,
            "current_value": order_total,
            "initial_value": order_total,
            "cash_pnl":      0.0,
            "event_id":      event_id,
        })
        notify_trade(
            "BUY",
            title,
            price=price,
            size=estimated_size,
            total=order_total,
            outcome=outcome_label,
            confidence=decision.get("confidence"),
            reason=decision.get("reason"),
            order_id=result.get("order_id"),
        )
    else:
        logger.error("[Agent] Ordin respins de CLOB: %s", result.get("error"))


def main() -> None:
    _load_history()  # prima instrucțiune la rulare: încarcă history.json (piețe deja analizate)

    logger.info(
        "Polymarket AI Agent starting | host=%s | chain_id=%d | funder=%s",
        settings.CLOB_HOST,
        settings.CHAIN_ID,
        settings.FUNDER_ADDRESS,
    )
    logger.info(
        "Mod scanare: %s",
        "DOAR meciuri de fotbal (Match Odds / 1X2)" if _FOOTBALL_ONLY
        else "universal — toate piețele (fotbal → 1X2, restul → binar YES/NO)",
    )

    if settings.OVERRIDE_BALANCE > 0:
        logger.warning(
            "OVERRIDE_BALANCE=%.4f este setat — botul va presupune această balanță "
            "când API-ul raportează 0. Dacă fondurile reale s-au epuizat, ordinele "
            "vor fi respinse de CLOB. Dezactivează variabila când nu mai e necesară.",
            settings.OVERRIDE_BALANCE,
        )

    try:
        logger.info("Running boot reconciliation...")
        log_reconciliation_report()
    except Exception as exc:
        logger.critical("Boot reconciliation failed — aborting startup: %s", exc)
        sys.exit(1)

    get_clob_client()  # warm up connection

    logger.info("Verificare și setare allowance USDC pentru CLOB Exchange...")
    try:
        ensure_collateral_approved()
    except Exception as exc:
        logger.error("Allowance check eșuat (bot continuă oricum): %s", exc)

    logger.info(
        "Entering autonomous AI agent loop (interval=%ds). Send SIGTERM/SIGINT to stop.",
        _LOOP_INTERVAL,
    )

    try:
        while RUNNING:
            account_state = get_account_state()
            pusd_balance: float = account_state.get("pusd_balance", 0.0)

            # Conturile proxy returnează 0 de la /balance-allowance chiar dacă au fonduri.
            # OVERRIDE_BALANCE permite specificarea balanței reale cunoscute via env var.
            if settings.OVERRIDE_BALANCE > 0 and pusd_balance == 0.0:
                logger.info(
                    "[Agent] API balanță = 0 (cont proxy) — folosesc OVERRIDE_BALANCE=%.4f pUSD.",
                    settings.OVERRIDE_BALANCE,
                )
                pusd_balance = settings.OVERRIDE_BALANCE
                account_state = {**account_state, "pusd_balance": pusd_balance}

            # Raport orar P/L pe Discord. Rulează înaintea verificării de balanță
            # ca să continue și când fondurile sunt insuficiente (bucla doarme
            # 15 min în acel caz, dar raportul tot iese la timp).
            try:
                _maybe_send_pnl_report(account_state)
            except Exception as exc:
                logger.error("[Agent] Raport P/L eșuat: %s", exc, exc_info=True)

            # Verifică TP/SL pe pozițiile deschise (opt-in via TAKE_PROFIT_PCT /
            # STOP_LOSS_PCT). Rulează înainte de verificarea de balanță: vânzarea
            # trebuie să funcționeze și când nu mai sunt fonduri de cumpărare.
            try:
                check_exit_conditions(account_state)
            except Exception as exc:
                logger.error("[Agent] Exit check eșuat: %s", exc, exc_info=True)

            # Verificare balanță înainte de orice apel LLM sau API Polymarket
            if pusd_balance < _MIN_BALANCE_PUSD:
                logger.warning(
                    "[Agent] Balanță insuficientă (%.4f pUSD < %.1f minim). "
                    "Suspendez analiza pentru %d minute.",
                    pusd_balance,
                    _MIN_BALANCE_PUSD,
                    _LOW_BALANCE_SLEEP // 60,
                )
                _interruptible_sleep(_LOW_BALANCE_SLEEP)
                continue

            # Limită globală de expunere
            n_open = len(account_state.get("active_positions", []))
            if settings.MAX_ACTIVE_POSITIONS > 0 and n_open >= settings.MAX_ACTIVE_POSITIONS:
                logger.info(
                    "[Agent] Limită de expunere atinsă: %d/%d poziții deschise. "
                    "Nu se deschid poziții noi. Aștept...",
                    n_open,
                    settings.MAX_ACTIVE_POSITIONS,
                )
                _interruptible_sleep(_LOOP_INTERVAL)
                continue

            markets = fetch_active_markets()
            if not markets:
                logger.warning("[Agent] Nicio piață activă găsită. Aștept...")
                _interruptible_sleep(_LOOP_INTERVAL)
                continue

            # Filtrare pe baza istoricului persistent (history.json) ÎNAINTE de procesare:
            # ignoră piețele analizate în ultimele _CACHE_TTL_SECONDS (cooldown).
            total_fetched = len(markets)
            markets = [m for m in markets if not _is_recently_analyzed(str(m.get("id", "?")))]
            skipped = total_fetched - len(markets)
            if skipped:
                logger.info(
                    "[History] %d/%d piețe deja analizate recent — filtrate. Rămân %d.",
                    skipped, total_fetched, len(markets),
                )
            if not markets:
                logger.info("[Agent] Toate piețele active au fost analizate recent. Aștept...")
                _interruptible_sleep(_LOOP_INTERVAL)
                continue

            # Amestecă, apoi prioritizează meciurile (head-to-head, adesea iminente):
            # un sort STABIL le aduce în față, păstrând ordinea aleatorie în interiorul
            # fiecărui grup. Astfel meciurile sunt analizate înaintea limitelor de
            # poziții/balanță sau a throttle-ului care s-ar putea atinge în timpul scanării.
            random.shuffle(markets)
            markets.sort(key=lambda m: not _is_match_market(m))
            n_matches = sum(1 for m in markets if _is_match_market(m))
            logger.info(
                "[Agent] Procesez %d piețe active (%d meciuri prioritizate la început).",
                len(markets), n_matches,
            )
            _scan_stats.clear()
            for market in markets:
                if not RUNNING:
                    break
                hist_before = len(_analyzed_history)
                try:
                    process_market(market, account_state)
                except Exception as exc:
                    logger.error(
                        "[Agent] Eroare la procesarea pieței id=%s: %s",
                        market.get("id", "?"),
                        exc,
                        exc_info=True,
                    )
                # Throttle DOAR după o analiză reală (process_market a înregistrat
                # piața în history → a făcut apel LLM). Piețele filtrate ieftin nu
                # dorm, altfel scanarea paginată ar pierde minute pe nimic.
                if len(_analyzed_history) > hist_before:
                    _interruptible_sleep(_INTER_ANALYSIS_DELAY)

            # Sumar de filtrare: ce a oprit piețele. Cheia pentru a vedea DE CE
            # „ignoră" botul tot — ex: {'volum mic': 195, 'analizate (LLM)': 3}.
            # Lista piețelor scanate rămâne STRICT în consolă/loguri (vezi logurile
            # „[Agent] Procesez piața ..." de mai sus) — pe Discord nu se trimite
            # nimic per buclă, doar raportul periodic din _maybe_send_pnl_report.
            logger.info("[Scan] Rezultat filtrare pe %d piețe: %s",
                        len(markets), dict(_scan_stats) or "nimic procesat")

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
