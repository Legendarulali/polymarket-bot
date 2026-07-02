import json
import os
import random
import time
import sys
import signal
import logging
from collections import Counter
from datetime import datetime, timedelta

import httpx

# PolyApiException — ridicată de SDK-ul CLOB la răspunsuri non-2xx (ex. 401
# Unauthorized dintr-un glitch API Polymarket / desincronizare de ceas). O prindem în
# bucla principală ca un astfel de glitch tranzitoriu să nu mai omoare tot procesul.
# Import defensiv: stub local dacă fork-ul nu expune modulul (importul nu trebuie să pice).
try:
    from py_clob_client_v2.exceptions import PolyApiException
except ImportError:  # pragma: no cover
    class PolyApiException(Exception):  # type: ignore[no-redef]
        status_code = None

# openai.RateLimitError — ridicată de SDK-ul OpenAI-compatibil (Groq) la HTTP 429 când
# bugetul zilnic de tokeni (TPD) e epuizat. O prindem EXPLICIT (după TIP) în Panic Sell ca
# să IEȘIM din ciclul de re-evaluare — toate apelurile rămase ar da tot 429, deci a itera
# mai departe ar ține bucla ocupată degeaba. Import defensiv, în aceeași manieră ca
# PolyApiException: dacă `openai` lipsește sau e mock-uit (fără o clasă reală de excepție),
# degradăm la None și ne bazăm pe `_is_rate_limit_error` (detecție pe mesaj), ca importul
# acestui modul să nu pice niciodată.
try:
    from openai import RateLimitError as _OPENAI_RATE_LIMIT  # type: ignore[assignment]
    if not (isinstance(_OPENAI_RATE_LIMIT, type) and issubclass(_OPENAI_RATE_LIMIT, BaseException)):
        _OPENAI_RATE_LIMIT = None  # type: ignore[assignment]
except Exception:  # pragma: no cover — openai absent sau mock-uit
    _OPENAI_RATE_LIMIT = None  # type: ignore[assignment]

from config.settings import settings
from src.reconciliation import log_reconciliation_report, get_account_state, compute_portfolio_summary
from src.execution import (
    Side, place_limit_order, compute_order_size,
    _TARGET_ORDER_VALUE, _BALANCE_SAFETY_BUFFER,
)
from src.clob_client import get_clob_client
from src.news_monitor import fetch_free_news
from src.ai_analyser import analyse_market, analyse_match, analyse_batch, analyse_position_exit
from src.football import parse_match_market, format_legs, extract_teams
from src.allowance import ensure_collateral_approved
from src.position_manager import check_exit_conditions, should_attempt_exit, record_exit_attempt
from src.notifications import notify_trade, notify_portfolio, notify_positions, notify_startup

logger = logging.getLogger(__name__)

RUNNING = True
# Heartbeat-ul buclei (secunde între iterații). RĂMÂNE mic (60s) ca reconcilierea,
# exit-urile TP/SL, Panic Sell, raportul P/L și coada să fie verificate des și să poată
# vinde prompt. Scanarea masivă (costisitoare) e throttle-ată SEPARAT prin
# _SCAN_INTERVAL_SECONDS — deci poți încetini scanarea fără să încetinești protecția pozițiilor.
_LOOP_INTERVAL = float(os.getenv("LOOP_INTERVAL_SECONDS", "60"))
# Cadența SCANĂRII masive (fetch Gamma + analiză Groq pe sute de piețe), DECUPLATĂ de
# heartbeat: scanarea rulează cel mult o dată la acest interval (IMPLICIT 15 min — vezi
# SCAN_INTERVAL_MINUTES), ca să nu lovim Gamma/Groq în fiecare minut, în timp ce restul
# buclei (monitorizarea + vânzarea pozițiilor) rămâne la _LOOP_INTERVAL. 0 = scanează la
# fiecare buclă (vechiul comportament). În DRY_RUN gate-ul e ignorat (stress-test).
_SCAN_INTERVAL_SECONDS = max(0.0, float(os.getenv("SCAN_INTERVAL_MINUTES", "15"))) * 60
# Pauză (secunde) după o excepție neprevăzută prinsă în bucla principală, înainte de a
# relua AUTOMAT iterația (auto-restart). „Câteva secunde" ca un crash tranzitoriu
# (ValueError/TypeError/timeout de rețea etc.) să nu trântească tot procesul; configurabil
# via CRASH_RECOVERY_SLEEP_SECONDS. Minim 1s.
_CRASH_RECOVERY_SLEEP = max(1.0, float(os.getenv("CRASH_RECOVERY_SLEEP_SECONDS", "5")))
_LOW_BALANCE_SLEEP = 15 * 60  # standby 15 min la eșec de reconciliere (401/glitch API, fără stare de cont). NU se mai folosește la cash insuficient — acolo doar dezactivăm cumpărarea (allow_new_trades) și rămânem la 60s.
_CACHE_TTL_SECONDS = 4 * 3600 # cooldown: re-analizează o piață cel mult o dată la 4h (persistat în history.json)
_GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
_GAMMA_PAGE_SIZE = 100                                           # piețe per pagină la paginare Gamma
# Gamma respinge paginarea dincolo de o adâncime maximă (offset prea mare, ex.
# offset=2100) cu HTTP 422. NU e o eroare, ci capătul rezultatelor permise: la 422
# oprim paginarea curat, păstrând tot ce am preluat. `_MAX_PAGINATION_DEPTH` e
# sentinelul intern prin care `_fetch_markets_page` semnalează acest caz buclei.
_GAMMA_MAX_DEPTH_STATUS = 422
_MAX_PAGINATION_DEPTH = object()
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
# Throttle (secunde) DUPĂ fiecare apel Groq — o dată per LOT (sau per piață când
# BATCH_SIZE<=1). Spațiază cererile ca să nu lovim rate-limit-ul Groq când scanăm sute
# de piețe într-o rundă. Configurabil (acceptă și fracțiuni, ex. 1.5).
_INTER_ANALYSIS_DELAY = float(os.getenv("INTER_ANALYSIS_DELAY_SECONDS", "3"))
# Analiză în LOTURI (batching): câte piețe pregătite trimitem la Groq într-UN singur
# apel. Reduce drastic numărul de cereri → permite modelul gpt-oss-20b
# fără a lovi rate-limit-urile (429). Implicit 12 (recomandat 10-15). BATCH_SIZE<=1
# revine la analiza individuală (o cerere per piață, prin process_market).
_BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "12")))
# Backoff defensiv (secunde) când Groq întoarce 429 (rate-limit) la un lot: așteptăm
# înainte de a reîncerca la scanarea următoare, ca să nu ardem bugetul zilnic de tokeni.
_RATE_LIMIT_BACKOFF = int(os.getenv("RATE_LIMIT_BACKOFF_SECONDS", "60"))
# --- Filtru STRICT anti-429 (STEP 2) ---
# Înainte de apelurile Groq reducem setul: păstrăm piețele cu poziție deschisă, cele cu
# volum ≥ _STRICT_MIN_VOLUME_USD și (dacă _STRICT_KEYWORDS e setat) cele al căror titlu/
# slug conține un keyword de interes; apoi sortăm după volum desc.
# _MAX_MARKETS_PER_SCAN plafonează câte piețe ajung la analiză: IMPLICIT 0 = FĂRĂ plafon
# → analizăm TOT ce trece de filtrul de volum (~900 piețe, ca înainte). Protecția anti-429
# se bazează acum pe: filtrul de volum + batching (_BATCH_SIZE) + throttle per lot
# (_INTER_ANALYSIS_DELAY) + cadența de scanare (_SCAN_INTERVAL_SECONDS) + cooldown-ul de 4h
# din history.json. Setează MAX_MARKETS_PER_SCAN>0 (ex. 500) DOAR ca să re-impui un plafon dur.
_STRICT_FILTER = os.getenv("STRICT_FILTER", "true").strip().lower() in ("1", "true", "yes", "on")
_STRICT_MIN_VOLUME_USD = float(os.getenv("STRICT_MIN_VOLUME_USD", "50000"))
_MAX_MARKETS_PER_SCAN = int(os.getenv("MAX_MARKETS_PER_SCAN", "0"))
_STRICT_KEYWORDS = [k.strip().lower() for k in os.getenv("STRICT_KEYWORDS", "").split(",") if k.strip()]
# Mod SIMULARE (stress test Groq). IMPLICIT False. Când e True: botul trece peste
# gate-urile de balanță (1.05 pUSD) și de expunere ca să scaneze și să trimită loturile
# către Groq (testăm rate-limit-urile), dar NU plasează NICIUN ordin real — `_execute_trade`
# e interceptat la intrare și doar loghează „[DRY_RUN] Simulare …". În plus: nu scrie în
# history.json (ca să nu polueze cooldown-ul real) și sare verificările de exit TP/SL (care
# plasează ordine SELL reale prin position_manager). Folosește DRY_RUN=true DOAR pentru teste.
_DRY_RUN = os.getenv("DRY_RUN", "false").strip().lower() in ("1", "true", "yes", "on")
# Mod TEST NOTIFICĂRI (STEP 3). IMPLICIT False. Când e True: după un BUY REAL reușit,
# botul așteaptă exact 1s și plasează imediat un SELL REAL pentru exact câte share-uri a
# cumpărat (la best bid), ca să verifice end-to-end ambele webhook-uri (BUY + SELL) cu
# ordine reale. Suprascrie orice strategie de holding/rezoluție. NU rulează în DRY_RUN
# (acolo nu pleacă niciun ordin real). Loguri „[NOTIF-TEST]". Folosește-l DOAR pentru teste.
_NOTIFICATION_TEST_MODE = os.getenv("NOTIFICATION_TEST_MODE", "false").strip().lower() in ("1", "true", "yes", "on")
# Balanța minimă (pUSD) sub care nu mai analizăm/tranzacționăm. 1.05 (nu 1.00) ca
# să acopere bufferul de siguranță de 3% din dimensionare: în „zona moartă"
# 1.00–1.04, safe_balance = balanță*0.97 produce un ordin sub minimul Polymarket de
# $1 ([Sizing] Skip) FĂRĂ ca balanța să scadă → break-ul nu se declanșează și botul
# ar apela Groq în gol pe toate cele ~2100 de piețe la fiecare buclă. La 1.05,
# safe_balance ≈ 1.0185, deci orice piață care trece de gate produce un ordin
# finanțabil (≥ $1) și balanța chiar scade.
_MIN_BALANCE_PUSD = 1.05
# Valoarea minimă (în $) a unui ordin acceptat de API-ul Polymarket. Verificată
# DUPĂ dimensionarea dinamică. Acum în mare parte defensivă: pragul _MIN_BALANCE_PUSD
# de 1.05 previne deja zona moartă, dar păstrăm verificarea ca plasă de siguranță
# (ex: dacă bufferul de 3% se schimbă) — sărim local în loc să primim un 400.
_MIN_ORDER_VALUE_USD = 1.0
_MIN_CONFIDENCE = 0.70
_MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "10000"))  # Ignoră piețele sub acest volum ($)
# Fereastră de probabilitate YES UNICĂ pentru toate tipurile de piață (meci,
# outright, general): [0.45, 0.95]. Ține botul concentrat pe meciurile
# competitive (probabilități medii-mari) și ignoră „zgomotul" de probabilitate
# foarte mică, lăsând decizia reală pe seama LLM-ului + pragului de confidence.
# Configurabilă din mediu (PROBABILITY_MIN / PROBABILITY_MAX).
_MIN_PROBABILITY = float(os.getenv("PROBABILITY_MIN", "0.45"))  # Ignoră probabilitățile YES prea mici (zgomot)
_MAX_PROBABILITY = float(os.getenv("PROBABILITY_MAX", "0.95"))  # Ignoră dacă probabilitatea YES e quasi-certă
# În NOTIFICATION_TEST_MODE lărgim fereastra de probabilitate la (0.01, 0.99) — vezi
# _prob_window — ca testul de notificări să prindă UȘOR o piață eligibilă. Pragurile reale
# de mai sus rămân neatinse pentru operarea normală.
_NOTIF_TEST_PROB_MIN = 0.01
_NOTIF_TEST_PROB_MAX = 0.99
_MAX_POSITIONS_PER_EVENT = 2     # Max poziții per event_id (piețe generale)
_MAX_OUTRIGHT_PER_EVENT = 1      # Max poziții pe piețe de tip "outright winner"
# Mod universal (IMPLICIT): botul evaluează TOATE piețele active — politice,
# crypto, generale ȘI meciuri de fotbal (acestea din urmă pe ruta dedicată 1X2).
# Opțional, FOOTBALL_ONLY=true restrânge scanarea DOAR la meciuri individuale de
# fotbal (ignoră complet outright/general) — util în timpul unui turneu.
_FOOTBALL_ONLY = os.getenv("FOOTBALL_ONLY", "false").strip().lower() in ("1", "true", "yes", "on")

# Istoric persistent al piețelor analizate: {market_id: timestamp_float}.
# Încărcat din _HISTORY_FILE la pornire; folosit pentru a filtra piețele
# analizate în ultimele _CACHE_TTL_SECONDS înainte de procesare. Fișierul
# history.json ține AMBELE stări (format „învelit"): piețele analizate ȘI
# ultimele decizii AI per poziție (cheia `last_ai_decisions`, vezi _last_ai_decision),
# ca ambele să supraviețuiască unui restart.
# Calea poate fi suprascrisă cu HISTORY_FILE (ex: un volum montat pe Railway,
# unde /app e efemer și root-only); altfel, history.json lângă main.py.
_HISTORY_FILE = os.getenv("HISTORY_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "history.json"
)
_analyzed_history: dict[str, float] = {}

# Coadă de oportunități „next in line" (next_in_line.json): piețe aprobate de AI
# (trade=True, confidence peste prag) dar nefinanțate la momentul scanării (lipsă
# fonduri). Sunt re-evaluate PRIORITAR când revin fondurile, fără a re-scana mii de
# piețe. Calea poate fi suprascrisă cu QUEUE_FILE (ex: volum montat pe Railway);
# altfel next_in_line.json lângă main.py. Best-effort, exact ca history.json.
_QUEUE_FILE = os.getenv("QUEUE_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "next_in_line.json"
)
_QUEUE_TTL_HOURS = 12   # oportunitățile mai vechi sunt eliminate (știri expirate)
_QUEUE_MAX_SIZE = 10    # plafon coadă: păstrăm intrările cu cel mai mare confidence
# Activează coada „Wishlist". IMPLICIT dezactivată (False): coada e complet inertă —
# funcțiile ei (_clean_queue / _process_queue / _enqueue_opportunity) NU sunt apelate.
# Când e ON, coada salvează DOAR piețele aprobate de AI care nu pot fi deschise pentru
# că s-a atins MAX_ACTIVE_POSITIONS (deși balanța e suficientă, ≥ _MIN_BALANCE_PUSD), și
# le re-evaluează prioritar când se eliberează un slot. NU atinge regula de balanță 1.05.
_ACTIVATE_QUEUE = os.getenv("ACTIVATE_QUEUE", "false").strip().lower() in ("1", "true", "yes", "on")

# --- Panic Sell (STEP 4): re-evaluare pe ȘTIRI a pozițiilor DEȚINUTE ---
# IMPLICIT dezactivat (False): e o lichidare REALĂ declanșată de judecata unui LLM, deci
# opt-in explicit (ca DRY_RUN / ACTIVATE_QUEUE). Când e ON, `_reevaluate_active_positions`
# rulează la fiecare heartbeat (independent de cash — vânzarea folosește token-urile deținute),
# dar cheamă Groq DOAR la o scădere bruscă de preț (_PANIC_THRESHOLD_PERCENT) și, dacă
# Groq vede o știre care invalidează catastrofal teza de BUY, lichidează poziția la best bid
# (SELL), ocolind `_execute_trade` (care e doar pentru BUY) și gate-ul de balanță.
_PANIC_SELL_ENABLED = os.getenv("PANIC_SELL_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
# Pragul de confidence peste care acționăm un PANIC_SELL (conservator — nu lichidăm pe semnale slabe).
_PANIC_SELL_MIN_CONFIDENCE = float(os.getenv("PANIC_SELL_MIN_CONFIDENCE", "0.70"))
# TRIGGER MATEMATIC (economie de tokeni / anti-429): prag de SCĂDERE BRUSCĂ a prețului (best bid),
# în PROCENTE, față de scanarea anterioară (heartbeat-ul precedent), peste care declanșăm o
# re-evaluare AI a poziției. La FIECARE heartbeat comparăm prețul curent cu cel memorat (citire
# orderbook ieftină, ZERO tokeni): DACĂ a scăzut ≥ acest prag → întrebăm Groq; dacă e stabil / în
# creștere / scădere mică → doar actualizăm prețul memorat și trecem mai departe (NICIUN apel Groq).
# Așa reacția rămâne promptă (≤ 1 heartbeat) fără să ardem bugetul de tokeni. Implicit 5%.
_PANIC_THRESHOLD_PERCENT = float(os.getenv("PANIC_THRESHOLD_PERCENT", "5"))
# Ultimul best bid observat per token DEȚINUT (memorie in-process, nepersistată) — baseline pentru
# trigger-ul de preț de mai sus, realimentat la fiecare heartbeat. La (re)pornire e gol → prima
# observare a unei poziții doar STABILEȘTE baseline-ul, fără apel AI (nimic cu ce compara încă).
_last_position_price: dict[str, float] = {}

# Ultima decizie AI cunoscută per token DEȚINUT. Cheie: token_id → {"action",
# "confidence", "reason", "ts"}. Alimentată de intrarea în poziție (BUY reușit) și de
# re-evaluarea Panic Sell (HOLD/PANIC_SELL); citită de raportul de status per-heartbeat
# (`_log_position_status`) — pur informativ, NU costă niciun apel Groq. PERSISTATĂ în
# history.json (cheia `last_ai_decisions`): _record_ai_decision o salvează la fiecare
# update, iar _load_history o restaurează la pornire, ca o poziție preexistentă să NU mai
# reapară drept „Pending initial AI sync" după un restart de proces/container. Fără intrare
# pentru un token → raportul afișează „Pending initial AI sync".
_last_ai_decision: dict[str, dict] = {}

# Stare raport orar P/L pe Discord. `_next_pnl_report_ts` = momentul (epoch) al
# următorului raport, ALINIAT la ceas (oră fixă xx:00, ca resetul de metrici Polymarket),
# NU la un interval relativ de pornire. None = încă neprogramat (se setează la prima
# buclă, pe următoarea graniță orară — deci NU se trimite un raport parțial la pornire).
_next_pnl_report_ts: float | None = None
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


def _interruptible_sleep(seconds: float) -> None:
    """Sleep în incremente de ≤1s astfel încât SIGTERM e onorat prompt. Acceptă și
    valori fracționare (ex. throttle de 1.5s) sau 0 (no-op)."""
    remaining = float(seconds)
    while remaining > 0 and RUNNING:
        chunk = 1.0 if remaining > 1.0 else remaining
        time.sleep(chunk)
        remaining -= chunk


def _fetch_markets_page(params: dict) -> list | object | None:
    """Un singur GET la Gamma /markets.

    Întoarce:
      - lista de piețe (poate fi goală) la HTTP 200;
      - `_MAX_PAGINATION_DEPTH` la HTTP 422 — am atins adâncimea maximă de paginare
        permisă de Polymarket (ex. offset=2100). NU e o eroare, ci capătul
        rezultatelor: apelantul oprește paginarea curat, cu ce a strâns;
      - `None` la orice altă eroare / non-200 / excepție de rețea.
    """
    try:
        response = httpx.get(_GAMMA_MARKETS_URL, params=params, timeout=10)
        # 422 = adâncime maximă de paginare atinsă → semnalăm distinct, nu ca eroare.
        if response.status_code == _GAMMA_MAX_DEPTH_STATUS:
            return _MAX_PAGINATION_DEPTH
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
        # Fallback: dacă 'order' e respins (non-200, dar NU 422), reia pagina fără sortare.
        if page is None and order_params:
            logger.warning("[Agent] Reîncerc fără 'order=%s' (posibil nesuportat de Gamma).", _GAMMA_ORDER)
            page = _fetch_markets_page(base)
        # HTTP 422 = adâncimea maximă de paginare permisă de Polymarket (ex. offset=2100):
        # nu e eroare, ci capătul rezultatelor. Oprim curat, păstrând piețele deja strânse.
        if page is _MAX_PAGINATION_DEPTH:
            logger.info(
                "[Agent] Adâncime maximă de paginare atinsă (offset=%d, HTTP 422). "
                "Continui cu cele %d piețe preluate.",
                offset, len(out),
            )
            return
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


def _live_best_bid(token_id: str) -> float | None:
    """Best bid live din orderbook CLOB (prețul la care se poate vinde acum), sau None."""
    try:
        ob = get_clob_client().get_order_book(token_id=token_id)
        bids = ob.bids if hasattr(ob, "bids") else ob.get("bids", [])
        if not bids:
            return None
        return max(
            float(b.price if hasattr(b, "price") else b["price"]) for b in bids
        )
    except Exception as exc:
        logger.warning("[Agent] Orderbook indisponibil pentru token=%s: %s", token_id, exc)
        return None


def _parse_ai_decisions(raw) -> dict[str, dict]:
    """Deserializează blocul `last_ai_decisions` din history.json → {token_id: {...}}.

    Robust la date lipsă/corupte: sare intrările care nu-s dict și normalizează câmpurile.
    `ts` și `confidence` sunt trecute prin _qfloat (deci merg și dacă au fost salvate ca
    string), ca vechimea „(acum Xm)" din raportul de status să nu crape după un restart."""
    out: dict[str, dict] = {}
    if not isinstance(raw, dict):
        return out
    for tid, dec in raw.items():
        if not isinstance(dec, dict):
            continue
        out[str(tid)] = {
            "action":     str(dec.get("action") or "?"),
            "confidence": _qfloat(dec.get("confidence")),
            "reason":     str(dec.get("reason") or "")[:200],
            "ts":         _qfloat(dec.get("ts")),
        }
    return out


def _load_history() -> None:
    """Încarcă istoricul din history.json (prima instrucțiune la rulare).

    Formatul curent e un dict „învelit": {"analyzed": {id: ts}, "last_ai_decisions":
    {token_id: {...}}}, ca AMBELE stări — cooldown-ul de analiză ȘI ultima decizie AI per
    poziție — să supraviețuiască unui restart dintr-un singur fișier. Compatibil înapoi cu
    formatele vechi: un dict plat {id: ts} sau o listă simplă de ID-uri (analizate „acum").
    Intrările de analiză mai vechi decât _CACHE_TTL_SECONDS sunt eliminate la încărcare.
    """
    global _analyzed_history, _last_ai_decision
    now = time.time()
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        _analyzed_history = {}
        _last_ai_decision = {}
        logger.info("[History] %s inexistent — pornesc cu istoric gol.", _HISTORY_FILE)
        return
    except (json.JSONDecodeError, OSError) as exc:
        _analyzed_history = {}
        _last_ai_decision = {}
        logger.warning("[History] Nu am putut citi %s (%s) — pornesc cu istoric gol.", _HISTORY_FILE, exc)
        return

    # Formatul „învelit" nou (dict cu sub-blocul `analyzed` și/sau `last_ai_decisions`) vs.
    # formatele vechi (dict plat {id: ts} sau listă de ID-uri). Detecția e sigură: în formatul
    # vechi valorile sunt timestamp-uri (numere), niciodată dict-uri.
    raw_analyzed = raw
    decisions_raw: dict = {}
    if isinstance(raw, dict) and (
        isinstance(raw.get("analyzed"), dict) or isinstance(raw.get("last_ai_decisions"), dict)
    ):
        raw_analyzed = raw.get("analyzed", {})
        decisions_raw = raw.get("last_ai_decisions", {})

    parsed: dict[str, float] = {}
    if isinstance(raw_analyzed, dict):
        for mid, ts in raw_analyzed.items():
            try:
                parsed[str(mid)] = float(ts)
            except (ValueError, TypeError):
                continue
    elif isinstance(raw_analyzed, list):
        parsed = {str(mid): now for mid in raw_analyzed}

    _analyzed_history = {mid: ts for mid, ts in parsed.items() if now - ts < _CACHE_TTL_SECONDS}
    _last_ai_decision = _parse_ai_decisions(decisions_raw)
    expired = len(parsed) - len(_analyzed_history)
    logger.info(
        "[History] Încărcate %d piețe analizate recent din %s (%d expirate, ignorate); "
        "%d decizii AI restaurate.",
        len(_analyzed_history), _HISTORY_FILE, expired, len(_last_ai_decision),
    )
    if expired:
        _save_history()  # rescrie fișierul curățat de intrările expirate


def _save_history() -> None:
    """Persistă istoricul în history.json (scriere atomică), în formatul „învelit":
    {"analyzed": {id: ts}} (piețele analizate — cooldown) + {"last_ai_decisions":
    {token_id: {...}}} (ultima decizie AI per poziție), ca ambele să supraviețuiască unui
    restart. Creează directorul părinte dacă lipsește; erorile de IO/serializare sunt
    prinse și logate, fără a opri botul (history e best-effort).
    """
    tmp = f"{_HISTORY_FILE}.tmp"
    try:
        parent = os.path.dirname(_HISTORY_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "analyzed": _analyzed_history,
            "last_ai_decisions": _last_ai_decision,
        }
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, _HISTORY_FILE)
    except (OSError, TypeError) as exc:
        logger.warning("[History] Nu am putut salva %s: %s", _HISTORY_FILE, exc)


def _is_recently_analyzed(market_id: str) -> bool:
    """True dacă piața a fost analizată în ultimele _CACHE_TTL_SECONDS (cooldown activ).

    În DRY_RUN întoarce mereu False: ignorăm cooldown-ul ca fiecare iterație să re-scaneze
    TOT și să retrimită loturile către Groq (sarcină susținută pentru stress test)."""
    if _DRY_RUN:
        return False
    ts = _analyzed_history.get(market_id)
    return ts is not None and (time.time() - ts) < _CACHE_TTL_SECONDS


def _record_analyzed(market_id: str) -> None:
    """Marchează o piață ca analizată acum și persistă imediat istoricul.

    În DRY_RUN e no-op: nu persistăm nimic, ca să nu poluăm cooldown-ul real din
    history.json (altfel, după simulare, piețele ar fi blocate 4h în modul live)."""
    if _DRY_RUN:
        return
    _analyzed_history[market_id] = time.time()
    _save_history()


# ---------------------------------------------------------------------------
# Coadă de oportunități „next in line" (next_in_line.json)
# ---------------------------------------------------------------------------
def _qfloat(value, default: float = 0.0) -> float:
    """Conversie tolerantă la float (date posibil corupte/lipsă din JSON-ul cozii)."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _load_queue() -> list[dict]:
    """Citește coada de oportunități din next_in_line.json.

    Întoarce o listă (goală dacă fișierul lipsește, e corupt sau nu e o listă).
    Filtrează intrările fără `market_id` (nereutilizabile). Robust la fișier
    lipsă/corupt — niciodată ridică, ca botul să nu cadă pe o coadă stricată."""
    try:
        with open(_QUEUE_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[Queue] Nu am putut citi %s (%s) — coadă goală.", _QUEUE_FILE, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("[Queue] %s nu conține o listă — coadă goală.", _QUEUE_FILE)
        return []
    return [it for it in raw if isinstance(it, dict) and it.get("market_id")]


def _save_queue(queue: list[dict]) -> None:
    """Persistă coada (scriere atomică). Best-effort: erorile IO sunt logate, nu ridicate."""
    tmp = f"{_QUEUE_FILE}.tmp"
    try:
        parent = os.path.dirname(_QUEUE_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(queue, fh)
        os.replace(tmp, _QUEUE_FILE)
    except OSError as exc:
        logger.warning("[Queue] Nu am putut salva %s: %s", _QUEUE_FILE, exc)


def _enqueue_opportunity(
    market_id,
    token_id: str | None,
    title: str,
    outcome_label: str,
    price: float,
    confidence: float,
    event_id: str = "",
    tick_size: str = "0.01",
) -> None:
    """Adaugă în coadă o oportunitate aprobată de AI dar nefinanțată (lipsă fonduri).

    Deduplică pe `market_id` (intrarea nouă o înlocuiește pe cea veche, cu preț/
    confidence proaspete) și plafonează coada la _QUEUE_MAX_SIZE, păstrând
    intrările cu cel mai mare confidence. Fără `token_id` nu putem reexecuta, deci
    intrarea e ignorată."""
    if not token_id:
        logger.debug("[Queue] '%s' fără token_id — nu se pune în coadă.", str(title)[:50])
        return
    mid = str(market_id)
    queue = [it for it in _load_queue() if str(it.get("market_id")) != mid]
    queue.append({
        "market_id":         mid,
        "token_id":          str(token_id),
        "market_title":      str(title),
        "predicted_outcome": str(outcome_label),
        "original_price":    _qfloat(price),
        "confidence":        _qfloat(confidence),
        "event_id":          str(event_id or ""),
        "tick_size":         str(tick_size or "0.01"),
        "added_at":          time.time(),
    })
    if len(queue) > _QUEUE_MAX_SIZE:
        queue.sort(key=lambda it: _qfloat(it.get("confidence")), reverse=True)
        queue = queue[:_QUEUE_MAX_SIZE]
    _save_queue(queue)
    logger.info(
        "[Queue] Piața '%s' a fost adăugată în lista de așteptare (Confidence: %.2f).",
        str(title)[:60], _qfloat(confidence),
    )


def _clean_queue() -> None:
    """Elimină din coadă oportunitățile mai vechi de _QUEUE_TTL_HOURS (știri expirate).

    Rulează la începutul fiecărei iterații, înainte de procesarea cozii sau a
    piețelor, ca să nu re-evaluăm teze pe baza unor știri perimate."""
    queue = _load_queue()
    if not queue:
        return
    now = time.time()
    ttl = _QUEUE_TTL_HOURS * 3600
    fresh = [it for it in queue if now - _qfloat(it.get("added_at")) < ttl]
    removed = len(queue) - len(fresh)
    if removed:
        _save_queue(fresh)
        logger.info(
            "[Queue] %d oportunități expirate (> %dh) eliminate. Rămân %d în coadă.",
            removed, _QUEUE_TTL_HOURS, len(fresh),
        )


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


def _next_report_time(interval_h: float, now: datetime | None = None) -> datetime:
    """Următoarea graniță de raportare ALINIATĂ la ceas, strict după `now`.

    Granițele sunt ancorate la miezul nopții și pășesc din `interval_h` în `interval_h`
    ore: pentru interval_h=1 → fix la oră întreagă (00:00, 01:00, …, 23:00); pentru 0.5 →
    xx:00 și xx:30; pentru 2 → orele pare. Astfel raportul iese la ORĂ FIXĂ (ca resetul
    orar de metrici/orderbook Polymarket), nu la un interval relativ de pornirea botului.

    Notă fus orar: folosim ceasul local (`datetime.now()`); pe Railway containerul rulează
    în UTC, deci granițele coincid cu vârful orei UTC (când Polymarket își resetează ora)."""
    now = now or datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    interval = timedelta(hours=interval_h)
    # int() == floor aici (diferența e mereu ≥ 0); +1 ⇒ prima graniță STRICT după `now`.
    steps = int((now - midnight) / interval) + 1
    return midnight + steps * interval


def _maybe_send_pnl_report(account_state: dict) -> None:
    """Raportul periodic de portofoliu pe Discord, în 2 mesaje grupate:
      1) status financiar (cash, P/L, valoare totală),
      2) inventarul pozițiilor deschise (sau „Niciuna").
    Trimise o singură dată la fiecare graniță orară ALINIATĂ la ceas (xx:00, pas de
    PNL_REPORT_INTERVAL_HOURS), NU la fiecare buclă și NU la interval relativ de pornire.
    Prima rulare DOAR programează următoarea graniță (fără raport parțial instant)."""
    global _next_pnl_report_ts, _last_report_value, _session_start_value

    interval_h = settings.PNL_REPORT_INTERVAL_HOURS
    if interval_h <= 0 or not settings.DISCORD_WEBHOOK_URL:
        return
    now = time.time()

    # Prima rulare: ancorăm raportul la următoarea oră fixă și reținem valoarea de pornire
    # ca bază pentru P/L-ul „de la pornirea sesiunii". NU trimitem nimic acum — dacă botul
    # pornește la 14:45, primul raport iese la 15:00, nu un raport parțial instant.
    if _next_pnl_report_ts is None:
        _next_pnl_report_ts = _next_report_time(interval_h).timestamp()
        if _session_start_value is None:
            _session_start_value = compute_portfolio_summary(account_state)["total_value"]
        logger.info(
            "[P/L] Primul raport aliniat la ora fixă următoare (în ~%.1f min).",
            max(0.0, (_next_pnl_report_ts - now) / 60),
        )
        return

    if now < _next_pnl_report_ts:
        return

    summary = compute_portfolio_summary(account_state)
    if _session_start_value is None:  # plasă de siguranță
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
    # Reprogramează la următoarea graniță aliniată la ceas. Recalculat din `now`, deci
    # dacă o iterație a întârziat și am sărit o graniță, programăm direct următoarea —
    # fără rafală de rapoarte restante.
    _next_pnl_report_ts = _next_report_time(interval_h).timestamp()


def _record_ai_decision(token_id: str, action: str, confidence: float, reason: str) -> None:
    """Reține ultima decizie AI pentru un token DEȚINUT (ca raportul de status per-heartbeat
    s-o poată afișa FĂRĂ a re-apela Groq) ȘI o persistă imediat în history.json, ca decizia
    să supraviețuiască unui restart — altfel poziția preexistentă ar reapărea drept „Pending
    initial AI sync". Pur informativ, nu influențează tranzacționarea."""
    tid = str(token_id or "")
    if not tid:
        return
    _last_ai_decision[tid] = {
        "action":     str(action or "?"),
        "confidence": _qfloat(confidence),
        "reason":     str(reason or "")[:200],
        "ts":         time.time(),
    }
    _save_history()   # persistă deciziile (+ istoricul de analiză) — supraviețuiesc restartului


def _live_bid_ask(token_id: str) -> tuple[float | None, float | None]:
    """Best bid & best ask live dintr-o SINGURĂ cerere orderbook (vs _live_best_bid +
    _live_best_ask, care fac câte una fiecare). Best-effort și TĂCUT: la orice eroare întoarce
    (None, None) și loghează doar la DEBUG — raportul de status rulează la FIECARE heartbeat,
    deci nu vrem spam de warning-uri dacă orderbook-ul e momentan indisponibil."""
    tid = str(token_id or "")
    if not tid:
        return None, None
    try:
        ob = get_clob_client().get_order_book(token_id=tid)
        bids = ob.bids if hasattr(ob, "bids") else ob.get("bids", [])
        asks = ob.asks if hasattr(ob, "asks") else ob.get("asks", [])
        best_bid = max((float(b.price if hasattr(b, "price") else b["price"]) for b in bids), default=None) if bids else None
        best_ask = min((float(a.price if hasattr(a, "price") else a["price"]) for a in asks), default=None) if asks else None
        return best_bid, best_ask
    except Exception as exc:
        logger.debug("[Status] Orderbook indisponibil pentru token=%s: %s", tid, exc)
        return None, None


def _format_ai_decision(token_id: str) -> str:
    """Ultima decizie AI cunoscută pentru un token (Action, Confidence, Reason + vechime), sau
    „Pending initial AI sync" dacă poziția nu a fost încă evaluată de AI (fără apel Groq)."""
    dec = _last_ai_decision.get(str(token_id or ""))
    if not dec:
        return "AI: Pending initial AI sync"
    age_min = max(0.0, (time.time() - _qfloat(dec.get("ts"))) / 60.0)
    reason = str(dec.get("reason") or "").strip()
    reason_str = f' - "{reason[:70]}"' if reason else ""
    return (
        f"AI: {dec.get('action', '?')} conf {_qfloat(dec.get('confidence')):.2f} "
        f"(acum {age_min:.0f}m){reason_str}"
    )


def _format_position_status(idx: int, pos: dict) -> str:
    """Un RÂND de status pentru o poziție deținută: titlu + outcome, preț de intrare vs preț
    curent (mark din reconciliere + best bid/ask live best-effort), P/L și ultima decizie AI.
    Pur informativ — NU apelează Groq."""
    title = str(pos.get("asset_name") or "Necunoscut").strip()[:55]
    outcome = str(pos.get("outcome") or "?")
    token_id = str(pos.get("token_id") or "")
    entry = _qfloat(pos.get("avg_price"))
    cur = _qfloat(pos.get("cur_price"))
    size = _qfloat(pos.get("size"))
    pnl = _qfloat(pos.get("cash_pnl"))
    pnl_pct = _qfloat(pos.get("percent_pnl"))

    # Best bid/ask live (best-effort). Prețul „curent" preferă mark-ul din reconciliere (deja
    # preluat, mereu prezent); dacă lipsește, cade pe mijlocul bid/ask, dacă îl avem.
    bid, ask = _live_bid_ask(token_id)
    book_str = f" (bid ${bid:.3f} / ask ${ask:.3f})" if (bid is not None and ask is not None) else ""
    if cur <= 0 and bid is not None and ask is not None:
        cur = (bid + ask) / 2.0

    if entry > 0 and cur > 0:
        move = (cur - entry) / entry * 100.0
        arrow = "▲" if cur > entry else ("▼" if cur < entry else "▬")
        price_str = f"intrare ${entry:.3f} → acum ${cur:.3f}{book_str} {arrow} {move:+.1f}%"
    elif cur > 0:
        price_str = f"intrare ${entry:.3f} → acum ${cur:.3f}{book_str}"
    else:
        price_str = f"intrare ${entry:.3f} → preț curent indisponibil"

    return (
        f"  #{idx} {title} · {outcome} | {price_str} | "
        f"size {size:.2f} · P/L ${pnl:+.2f} ({pnl_pct:+.1f}%) | {_format_ai_decision(token_id)}"
    )


def _log_position_status(account_state: dict) -> None:
    """Raport de status per-heartbeat (la fiecare _LOOP_INTERVAL, 60s) al pozițiilor deținute.
    PUR informativ și FĂRĂ apel Groq (nu ardem rate-limit-ul): pentru fiecare poziție arată
    prețul curent (mark + best bid/ask best-effort) vs prețul de intrare, P/L-ul și ULTIMA
    decizie AI cunoscută din memorie. Un rând per poziție; robust: o eroare pe o singură
    poziție e prinsă și NU strică restul raportului (fără traceback-uri repetate în buclă).

    Volum de loguri neglijabil (≤ MAX_ACTIVE_POSITIONS rânduri/minut), deci sigur la INFO
    chiar cu rate-limit-ul Railway (spre deosebire de scanarea a mii de piețe)."""
    positions = account_state.get("active_positions", [])
    if not positions:
        logger.info("[Status] Nicio poziție activă — monitorizez, portofoliu în așteptare.")
        return

    summary = compute_portfolio_summary(account_state)
    logger.info(
        "[Status] Urmăresc %d poziții active | valoare $%.2f · cash $%.2f · P/L nerealizat $%+.2f",
        summary["n_positions"], summary["total_value"], summary["cash"], summary["unrealized_pnl"],
    )
    for idx, pos in enumerate(positions, 1):
        try:
            logger.info("[Status] %s", _format_position_status(idx, pos))
        except Exception as exc:  # niciodată nu stricăm raportul pe o singură poziție
            logger.warning(
                "[Status] Nu am putut construi statusul poziției token=%s: %s",
                pos.get("token_id"), exc,
            )


def _prob_window() -> tuple[float, float]:
    """Fereastra de probabilitate efectivă pentru filtre: lărgită (0.01, 0.99) în
    NOTIFICATION_TEST_MODE ca testul de notificări să găsească ușor o piață; altfel
    pragurile reale PROBABILITY_MIN/MAX. NU afectează calea live normală (flag off)."""
    if _NOTIFICATION_TEST_MODE:
        return _NOTIF_TEST_PROB_MIN, _NOTIF_TEST_PROB_MAX
    return _MIN_PROBABILITY, _MAX_PROBABILITY


def _prepare_market(market: dict, account_state: dict) -> dict | None:
    """Aplică TOATE filtrele ieftine (fără apel LLM) și, dacă piața merită analizată,
    culege știrile și construiește un „item" de analiză. Întoarce None dacă piața e
    filtrată (motivul e contorizat în `_scan_stats`).

    Item-ul rezultat e folosit fie individual (`process_market`, fallback non-batch),
    fie acumulat într-un lot și trimis la Groq într-un singur apel (`_scan_markets`).

    Mod universal (implicit): toate piețele sunt evaluate. Meciurile structurate 1X2
    (Match Odds) produc un item kind='match'; restul (politice, crypto, generale,
    outright) un item kind='binary'. Opțional, FOOTBALL_ONLY=true ignoră tot ce nu e meci."""
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
    #
    # Cu ACTIVATE_QUEUE NU ieșim devreme aici: lăsăm analiza AI să ruleze, iar dacă
    # aprobă, `_execute_trade` salvează piața în wishlist (next_in_line.json) în loc s-o
    # tranzacționeze. Fără ACTIVATE_QUEUE păstrăm comportamentul implicit (skip ieftin,
    # fără apel LLM) ca să nu irosim API calls când suntem la MAX. În DRY_RUN nu ieșim
    # devreme nici la MAX: vrem ca piața să ajungă în lot și să fie trimisă către Groq.
    if (
        not _ACTIVATE_QUEUE
        and not _DRY_RUN
        and settings.MAX_ACTIVE_POSITIONS > 0
        and len(active_positions) >= settings.MAX_ACTIVE_POSITIONS
    ):
        _scan_stats["expunere maximă"] += 1
        logger.debug(
            "[FILTER] Limită de expunere atinsă (%d/%d poziții). Skip piața '%s'.",
            len(active_positions), settings.MAX_ACTIVE_POSITIONS, title[:60],
        )
        return None

    # --- FILTRU 0 (OPȚIONAL): mod focus pe fotbal (FOOTBALL_ONLY=true). Implicit
    # dezactivat — botul e universal și procesează toate tipurile de piață. ---
    if _FOOTBALL_ONLY and market_type != "match":
        _scan_stats["non-meci (outright/general)"] += 1
        logger.debug(
            "[FILTER] Ignor piața '%s' [%s] — FOOTBALL_ONLY activ, nu e meci 1X2.",
            title[:60], type_label,
        )
        return None

    # --- FILTRU 1: Volum minim ---
    volume = _market_volume(market)
    if volume < _MIN_VOLUME_USD:
        _scan_stats["volum mic"] += 1
        logger.debug(
            "[FILTER] Ignor piața '%s' [%s] (Volum prea mic: $%.0f < $%.0f).",
            title[:60], type_label, volume, _MIN_VOLUME_USD,
        )
        return None

    event_id = _extract_event_id(market)

    # Meci structurat 1X2 → flux dedicat cu maparea 1/X/2. Dacă piața e un meci
    # dar nu are structura 1X2 (rezultate binare Yes/No), cade pe fluxul moștenit.
    if market_type == "match":
        parsed = parse_match_market(market)
        if parsed:
            return _prepare_match_1x2(market, parsed, event_id, title, market_id)
        logger.debug(
            "[Meci] '%s' nu e structurat 1X2 (rezultate binare) — folosesc fluxul YES/NO.",
            title[:60],
        )

    return _prepare_legacy_binary(market, account_state, event_id, title, market_id, market_type, type_label)


def _prepare_match_1x2(
    market: dict,
    parsed: dict,
    event_id: str,
    title: str,
    market_id,
) -> dict | None:
    """Filtre ieftine + culegere știri pentru un meci 1X2 → item kind='match' (sau None).

    Logează numele meciului și cotele 1/X/2, aplică fereastra de probabilitate pe
    fiecare rezultat (YES = prețul rezultatului) și culege știrile. Decizia LLM
    (analyse_match / analyse_batch) și execuția se fac mai târziu, pe item-ul întors."""
    home, away = parsed["home"], parsed["away"]
    # Cerință de produs: logurile arată clar meciul + cotele 1/X/2.
    logger.info("[Meci] %s | %s", parsed["label"], format_legs(parsed))

    # --- FILTRU 2: Fereastra de probabilitate pe fiecare leg 1X2 (lărgită în NOTIF-TEST) ---
    prob_min, prob_max = _prob_window()
    candidates = [
        leg for leg in parsed["legs"]
        if prob_min < leg["price"] < prob_max
    ]
    if not candidates:
        _scan_stats["probabilitate în afara ferestrei"] += 1
        logger.debug(
            "[FILTER] Ignor meciul '%s' — niciun rezultat 1X2 în fereastra [%.2f, %.2f] (%s).",
            parsed["label"], prob_min, prob_max, format_legs(parsed),
        )
        return None

    news = fetch_free_news(query=title[:100])
    logger.info("[Agent] %d știri găsite pentru '%s'.", len(news), parsed["label"][:50])

    return {
        "kind": "match",
        "market": market,
        "market_id": market_id,
        "title": title,
        "label": parsed["label"],
        "event_id": event_id,
        "home": home,
        "away": away,
        "candidates": candidates,
        "by_code": parsed["by_code"],
        "news": news,
    }


def _prepare_legacy_binary(
    market: dict,
    account_state: dict,
    event_id: str,
    title: str,
    market_id,
    market_type: str,
    type_label: str,
) -> dict | None:
    """Filtre ieftine + culegere știri pentru o piață binară YES/NO → item kind='binary'
    (sau None). Flux moștenit pentru piețe non-1X2 (sau focus pe fotbal dezactivat)."""
    active_positions: list[dict] = account_state.setdefault("active_positions", [])

    # --- FILTRU 2: Fereastra de probabilitate pe YES (lărgită în NOTIF-TEST) ---
    yes_price_raw = _current_price(market, "YES")
    prob_min, prob_max = _prob_window()
    if not (prob_min < yes_price_raw < prob_max):
        _scan_stats["probabilitate în afara ferestrei"] += 1
        logger.debug(
            "[FILTER] Ignor piața '%s' [%s] (Probabilitate YES=%.2f în afara ferestrei [%.2f, %.2f]).",
            title[:60], type_label, yes_price_raw, prob_min, prob_max,
        )
        return None

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
            return None

    news = fetch_free_news(query=title[:100])
    logger.info("[Agent] %d știri găsite pentru '%s'.", len(news), title[:50])

    return {
        "kind": "binary",
        "market": market,
        "market_id": market_id,
        "title": title,
        "event_id": event_id,
        "yes_price": _current_price(market, "YES"),
        "news": news,
    }


def _analyse_single(item: dict) -> dict:
    """Decizie LLM pentru UN singur item pregătit (cale non-batch: process_market).
    Rutează către analyse_match (1X2) sau analyse_market (binar)."""
    if item["kind"] == "match":
        return analyse_match(item["home"], item["away"], item["candidates"], item["news"])
    return analyse_market(item["title"], item["yes_price"], item["news"])


def _apply_decision(
    item: dict, decision: dict, account_state: dict, allow_new_trades: bool = True
) -> None:
    """Aplică o decizie LLM peste un item pregătit: log + poarta de confidence +
    maparea outcome → token/preț → `_execute_trade` (care conține TOATE barierele de
    risc: gate 1.05 pUSD, dedup, slippage, dimensionare). Comun fluxului individual
    (process_market) și celui în batch (_flush_batch).

    `allow_new_trades` se propagă neschimbat în `_execute_trade`: când e False (cash
    insuficient pentru achiziții noi), orice BUY e blocat strict acolo, fără a atinge
    SELL-urile (TP/SL/exit rulează în afara acestui flux)."""
    market = item["market"]
    title = item["title"]
    event_id = item["event_id"]

    if item["kind"] == "match":
        logger.info(
            "[Agent] Decizie AI 1X2: trade=%s | outcome=%s | confidence=%.2f | reason=%s",
            decision["trade"], decision["outcome"], decision["confidence"], decision["reason"],
        )
    else:
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

    if item["kind"] == "match":
        pick = decision["outcome"]
        leg = item["by_code"].get(pick)
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
            allow_new_trades=allow_new_trades,
        )
    else:
        outcome = decision["outcome"]
        _execute_trade(
            market, account_state,
            token_id=_token_for_outcome(market, outcome),
            price=_current_price(market, outcome),
            outcome_label=outcome,
            decision=decision,
            event_id=event_id,
            title=title,
            allow_new_trades=allow_new_trades,
        )


def process_market(market: dict, account_state: dict, allow_new_trades: bool = True) -> None:
    """Procesează O singură piață end-to-end: filtre ieftine → știri → analiză LLM
    INDIVIDUALĂ → execuție.

    E calea folosită de fallback-ul non-batch (BATCH_SIZE<=1) și de testele unitare;
    bucla principală folosește, implicit, analiza în LOTURI (vezi `_scan_markets`).
    Ambele căi partajează `_prepare_market` (filtrele) și `_apply_decision` (execuția),
    deci barierele de risc din `_execute_trade` sunt identice indiferent de cale."""
    item = _prepare_market(market, account_state)
    if item is None:
        return
    decision = _analyse_single(item)
    _record_analyzed(str(item["market_id"]))
    _scan_stats["analizate (LLM)"] += 1
    logger.debug(
        "[Agent] Piață id=%s marcată ca analizată în history.json (cooldown=%dh).",
        item["market_id"], _CACHE_TTL_SECONDS // 3600,
    )
    _apply_decision(item, decision, account_state, allow_new_trades)


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
    from_queue: bool = False,
    allow_new_trades: bool = True,
) -> None:
    """Verificări de risc finale + plasare ordin BUY + actualizare stare locală.

    Comun fluxului 1X2 și celui binar: balanță minimă, token valid, dedup pe
    token, preț valid, protecție anti-slippage față de best ask live și
    dimensionare ordin în limita balanței.

    `allow_new_trades=False` (cash insuficient pentru achiziții noi) blochează STRICT
    orice BUY chiar la intrare, înaintea oricărei alte verificări — vezi „master-switch
    BUY" mai jos. E independent de gate-ul de balanță și nu afectează SELL-urile.

    Wishlist (doar cu ACTIVATE_QUEUE): dacă balanța e suficientă (≥ _MIN_BALANCE_PUSD)
    dar s-a atins MAX_ACTIVE_POSITIONS, piața aprobată de AI nu se tranzacționează, ci
    se salvează în next_in_line.json pentru re-evaluare prioritară când se eliberează un
    slot. `from_queue=True` (apel din re-procesarea cozii) dezactivează salvarea, ca o
    intrare re-evaluată să nu se reintroducă la nesfârșit."""
    pusd_balance: float = account_state.get("pusd_balance", 0.0)
    active_positions: list[dict] = account_state.setdefault("active_positions", [])

    # DRY_RUN (mod simulare / stress test): la acest punct piața a fost deja APROBATĂ de
    # AI (trade=True, confidence ≥ prag — verificate în `_apply_decision`/`_reevaluate_queued`).
    # Interceptăm AICI, înaintea oricărei verificări sau apel CLOB: NU plasăm niciun ordin
    # real, nu atingem balanța/pozițiile/coada — doar logăm. E plasă de siguranță finală:
    # indiferent ce gate-uri ar trece în amonte, niciun ordin real nu pleacă în DRY_RUN.
    if _DRY_RUN:
        # NOTIFICATION_TEST_MODE peste DRY_RUN: testăm lanțul de notificări Discord FĂRĂ
        # balanță reală și FĂRĂ ordine reale — simulăm un ciclu complet BUY→1s→SELL și
        # declanșăm AMBELE webhook-uri. Niciun apel CLOB; balanța/pozițiile rămân neatinse.
        if _NOTIFICATION_TEST_MODE:
            _simulate_notif_cycle(price, title, outcome_label, decision)
            return
        logger.warning(
            "[DRY_RUN] Simulare: Piața '%s' a fost aprobată de AI (outcome=%s, confidence=%.2f), "
            "dar nu plasăm ordin real.",
            title[:80], outcome_label, decision.get("confidence", 0.0),
        )
        return

    # Master-switch BUY: când modul de cumpărare e dezactivat (cash insuficient pentru
    # achiziții noi — vezi `allow_new_trades` în bucla principală), niciun ordin NOU nu
    # pleacă, indiferent de restul verificărilor de mai jos. E gardul explicit, numit, pe
    # chokepoint-ul prin care trec TOATE cumpărările (scanare + coadă). SELL-urile (TP/SL/
    # exit din position_manager) NU trec pe aici, deci rămân active chiar și cu cash zero.
    if not allow_new_trades:
        logger.info(
            "[Agent] BUY blocat pentru '%s' (%s): modul de cumpărare e dezactivat "
            "(cash insuficient pentru achiziții noi).",
            title[:60], outcome_label,
        )
        return

    # Sub pragul _MIN_BALANCE_PUSD (1.05, peste minimul Polymarket de $1 ca să acopere
    # bufferul de 3% — vezi „zona moartă") nu putem plasa un ordin valid → blocăm.
    # Între prag și $5 ordinul NU e blocat, ci dimensionat dinamic. Regula NU e atinsă
    # de ACTIVATE_QUEUE: wishlist-ul de mai jos rulează DOAR după ce trecem de acest gate.
    if pusd_balance < _MIN_BALANCE_PUSD:
        logger.warning(
            "[Agent] Balanță %.4f pUSD sub pragul de $%.2f (buffer anti-zonă-moartă) — ordin imposibil. Blocat.",
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

    # Wishlist (ACTIVATE_QUEUE): balanță OK, token valid, fără duplicat, dar suntem la
    # MAX_ACTIVE_POSITIONS → nu putem deschide o poziție nouă acum. Salvăm piața aprobată
    # de AI în next_in_line.json pentru când se eliberează un slot, în loc s-o pierdem.
    # Doar la scanare reală (not from_queue), ca o re-evaluare din coadă să nu se re-pună.
    if (
        not from_queue
        and _ACTIVATE_QUEUE
        and settings.MAX_ACTIVE_POSITIONS > 0
        and len(active_positions) >= settings.MAX_ACTIVE_POSITIONS
    ):
        _scan_stats["adăugate în wishlist"] += 1
        _enqueue_opportunity(
            market.get("id", "?"), token_id, title, outcome_label, price,
            decision.get("confidence", 0.0), event_id=event_id,
            tick_size=str(market.get("orderPriceMinTickSize") or "0.01"),
        )
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

    # Buffer de siguranță pentru taxe/slippage dinamice Polymarket: nu angajăm
    # niciodată întreaga balanță, ca estimarea de fee a CLOB să nu împingă ordinul
    # în „Insufficient Balance". safe_balance = balanță * 0.97.
    safe_balance = pusd_balance * _BALANCE_SAFETY_BUFFER

    # Dimensionare DINAMICĂ: cumpărăm câte acțiuni permite bugetul alocat (țintă
    # $5/ordin, plafonat la safe_balance), trunchiat (floor) la 2 zecimale astfel
    # încât valoarea ordinului să nu depășească bugetul. Fără minim rigid de
    # share-uri — singurele bariere sunt cele două verificări de siguranță de mai jos.
    estimated_size = compute_order_size(price, budget=safe_balance)   # acțiuni de cumpărat
    order_total = round(estimated_size * price, 4)                    # valoarea ordinului ($)

    # Verificare 1: la acest preț bugetul nu acoperă nici măcar o acțiune întreagă.
    if estimated_size < 1:
        logger.info(
            "[Sizing] Skip '%s': preț $%.2f prea mare — bugetul de $%.2f cumpără doar "
            "%.2f acțiuni (< 1). Niciun ordin trimis.",
            title[:40], price, safe_balance, estimated_size,
        )
        return

    # Verificare 2: valoarea totală a ordinului e sub minimul de $1.00 impus de
    # API-ul Polymarket → ar fi respins (400). Îl sărim local, fără a lovi API-ul.
    if order_total < _MIN_ORDER_VALUE_USD:
        logger.info(
            "[Sizing] Skip '%s': valoare ordin $%.2f (%.2f acțiuni × $%.2f) sub minimul "
            "Polymarket de $%.2f. Niciun ordin trimis.",
            title[:40], order_total, estimated_size, price, _MIN_ORDER_VALUE_USD,
        )
        return

    if pusd_balance < _TARGET_ORDER_VALUE:
        logger.info(
            "[Agent] Balanță %.4f pUSD sub ținta de $%.2f — dimensionare dinamică: "
            "ordin $%.2f (%.2f acțiuni × $%.2f).",
            pusd_balance, _TARGET_ORDER_VALUE, order_total, estimated_size, price,
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
        # Reține teza de intrare (decizia AI la BUY) pentru raportul de status per-heartbeat,
        # ca poziția proaspăt deschisă să nu apară drept „Pending initial AI sync".
        _record_ai_decision(
            token_id, "BUY (intrare)",
            decision.get("confidence", 0.0), decision.get("reason", ""),
        )
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
        # STEP 3 — NOTIFICATION_TEST_MODE: după un BUY REAL reușit, închide IMEDIAT poziția
        # (SELL real la best bid) ca să declanșeze și webhook-ul de vânzare. Suprascrie
        # orice holding/TP/SL. Nu se atinge în DRY_RUN (ieșit mai sus, fără ordin real).
        if _NOTIFICATION_TEST_MODE:
            _notif_test_close(token_id, estimated_size, tick_size, title, outcome_label, account_state)
    else:
        logger.error("[Agent] Ordin respins de CLOB: %s", result.get("error"))


def _notif_test_close(
    token_id: str,
    size: float,
    tick_size: str,
    title: str,
    outcome_label: str,
    account_state: dict,
) -> None:
    """STEP 3 (NOTIFICATION_TEST_MODE): după un BUY REAL reușit, vinde IMEDIAT exact `size`
    share-uri la best bid (SELL real → „market sell" pe CLOB = ordin limită la cel mai bun
    bid), ca să declanșeze webhook-ul de vânzare și să testăm ciclul complet BUY→SELL cu
    ordine reale. Suprascrie orice strategie de holding/rezoluție. Loguri „[NOTIF-TEST]".

    Caveat: BUY-ul e un ordin limită; dacă nu s-a umplut în 1s, SELL-ul (pe share-uri încă
    nedeținute) poate fi respins de CLOB — îl logăm și lăsăm poziția deschisă pentru verificare."""
    logger.warning(
        "[NOTIF-TEST] BUY reușit pe '%s' (%s). Aștept 1s, apoi vând imediat %.2f share-uri "
        "pentru a testa ciclul complet de notificări.", title[:60], outcome_label, size,
    )
    time.sleep(1)  # cerință: exact 1 secundă între BUY și SELL

    bid = _live_best_bid(token_id)
    if bid is None or bid <= 0:
        logger.error(
            "[NOTIF-TEST] Best bid indisponibil pentru token=%s — SELL de test anulat. "
            "Poziția rămâne deschisă, verific-o manual.", token_id,
        )
        return

    result = place_limit_order(
        token_id=token_id, price=bid, side=Side.SELL, tick_size=tick_size, size=size,
    )
    if result.get("success"):
        logger.warning(
            "[NOTIF-TEST] SELL de test plasat | order_id=%s | size=%.2f | price=$%.4f — "
            "ciclul de notificări BUY→SELL e complet.",
            result.get("order_id"), size, bid,
        )
        # Reflectă închiderea în starea locală: scoate poziția și recreditează (aprox) balanța.
        positions = account_state.get("active_positions", [])
        account_state["active_positions"] = [p for p in positions if p.get("token_id") != token_id]
        account_state["pusd_balance"] = round(account_state.get("pusd_balance", 0.0) + bid * size, 6)
        notify_trade(
            "SELL", title, price=bid, size=size, total=bid * size,
            outcome=outcome_label, reason="NOTIFICATION_TEST_MODE: vânzare imediată de test",
            order_id=result.get("order_id"), action="CLOSE POSITION",
        )
    else:
        logger.error(
            "[NOTIF-TEST] SELL de test RESPINS de CLOB: %s. Poziția rămâne deschisă — "
            "verific-o manual.", result.get("error"),
        )


def _simulate_notif_cycle(price: float, title: str, outcome_label: str, decision: dict) -> None:
    """DRY_RUN + NOTIFICATION_TEST_MODE: simulează un ciclu COMPLET BUY→SELL și declanșează
    AMBELE webhook-uri Discord, FĂRĂ ordine reale și FĂRĂ balanță reală (niciun apel CLOB,
    starea locală neatinsă). Pentru a testa notificările fără fonduri. Loguri „[NOTIF-TEST][SIM]".

    Dimensionarea e nominală ($5 țintă, ca în producție) doar ca cifrele din webhook să arate
    realist; nu se cumpără/vinde nimic real, deci nu hrănim P/L-ul cu un round-trip fictiv."""
    size = compute_order_size(price)            # ~$5 nominal (fără buget/balanță reală)
    total = round(size * price, 4)
    logger.warning(
        "[NOTIF-TEST][SIM] BUY simulat: '%s' (%s) | size≈%.2f | price=$%.4f | total≈$%.2f "
        "→ declanșez webhook BUY.", title[:60], outcome_label, size, price, total,
    )
    notify_trade(
        "BUY", title, price=price, size=size, total=total,
        outcome=outcome_label, confidence=decision.get("confidence"),
        reason="DRY_RUN + NOTIFICATION_TEST_MODE: BUY simulat (fără ordin real)",
        order_id="SIMULATED-BUY",
    )
    time.sleep(1)  # cerință: exact 1 secundă între BUY și SELL
    logger.warning(
        "[NOTIF-TEST][SIM] SELL simulat: '%s' (%s) | size≈%.2f | price=$%.4f "
        "→ declanșez webhook SELL.", title[:60], outcome_label, size, price,
    )
    notify_trade(
        "SELL", title, price=price, size=size, total=total,
        outcome=outcome_label,
        reason="DRY_RUN + NOTIFICATION_TEST_MODE: SELL simulat (fără ordin real)",
        order_id="SIMULATED-SELL", action="CLOSE POSITION",
    )


# ---------------------------------------------------------------------------
# Panic Sell (STEP 4): re-evaluare pe ȘTIRI a pozițiilor DEȚINUTE
# ---------------------------------------------------------------------------
def _reevaluate_active_positions(account_state: dict) -> None:
    """STEP 4 — Panic Sell: monitorizează pozițiile DEȚINUTE la FIECARE heartbeat și, DOAR când
    prețul (best bid) scade brusc față de scanarea anterioară (peste _PANIC_THRESHOLD_PERCENT),
    întreabă Groq dacă o dezvoltare recentă invalidează teza de BUY și, la nevoie, lichidează
    (SELL la best bid). Opt-in via `PANIC_SELL_ENABLED`.

    TRIGGER MATEMATIC (anti-429 / economie de tokeni): prețul e verificat local în fiecare minut
    (citire orderbook — ieftină, ZERO tokeni); apelul Groq se face NUMAI la o scădere bruscă. Dacă
    prețul e stabil / în creștere / scade puțin, doar actualizăm prețul memorat și trecem mai
    departe. Așa reacția rămâne promptă (≤ 1 heartbeat) fără re-evaluări AI inutile.

    Rulează INDEPENDENT de cash (vânzarea folosește token-urile deja deținute) și OCOLEȘTE
    `_execute_trade` (BUY-only) — protejează portofoliul chiar și cu balanță zero."""
    if not _PANIC_SELL_ENABLED:
        return
    positions = account_state.get("active_positions", [])
    if not positions:
        return
    # Curăță baseline-urile prețului pentru token-uri care nu mai sunt deținute (poziții închise),
    # ca dict-ul să nu crească nemărginit și să nu re-declanșeze pe o intrare stală la re-achiziție.
    held = {str(p.get("token_id") or "") for p in positions}
    for tid in [t for t in _last_position_price if t not in held]:
        _last_position_price.pop(tid, None)

    # Iterăm pe un snapshot: _panic_sell_position rescrie lista de poziții la o lichidare reușită.
    for pos in list(positions):
        if not RUNNING:
            break
        try:
            ai_called = _reevaluate_one_position(pos, account_state)
        except Exception as exc:
            # Rate-limit LLM (HTTP 429 — buget zilnic de tokeni epuizat, atât pe Groq cât și pe
            # fallback-ul OpenRouter dacă ambele sunt epuizate): TOATE apelurile rămase din acest
            # ciclu ar eșua identic, deci nu are rost să iterăm mai departe. Prindem
            # `openai.RateLimitError` direct (după TIP — robust chiar dacă mesajul nu conține „429")
            # ȘI, via `_is_rate_limit_error`, cazul învelit de tenacity (RetryError). Logăm clar și
            # IEȘIM din ciclu (break), lăsând bucla principală să continue monitorizarea.
            if (
                (_OPENAI_RATE_LIMIT is not None and isinstance(exc, _OPENAI_RATE_LIMIT))
                or _is_rate_limit_error(exc)
            ):
                logger.warning(
                    "[PanicSell] Groq API rate limit reached. Skipping remaining position "
                    "evaluations for this cycle to allow the core loop to continue."
                )
                break
            logger.error(
                "[PanicSell] Eroare la re-evaluarea poziției token=%s: %s",
                pos.get("token_id"), exc, exc_info=True,
            )
            continue
        # Throttle DOAR după un apel LLM real (o scădere bruscă). Când prețul e stabil (calea
        # obișnuită, fără apel) NU dormim — bucla rămâne promptă și nu irosim heartbeat-ul.
        if ai_called:
            _interruptible_sleep(_INTER_ANALYSIS_DELAY)


def _reevaluate_one_position(pos: dict, account_state: dict) -> bool:
    """Verifică O poziție DEȚINUTĂ. Întâi TRIGGER-ul de preț (ieftin, ZERO tokeni): compară best
    bid-ul curent cu cel memorat; dacă NU a scăzut brusc (< _PANIC_THRESHOLD_PERCENT), doar
    actualizează prețul memorat și iese. La o SCĂDERE bruscă: știri → Groq (`analyse_position_exit`)
    și, dacă verdictul e PANIC_SELL cu confidence ≥ prag, lichidează. Cooldown-ul de exit e partajat
    cu TP/SL (`should_attempt_exit`) ca să nu vindem același token de două ori.

    Întoarce True DOAR dacă a făcut un apel LLM (ca apelantul să aplice throttle-ul), altfel False."""
    token_id = str(pos.get("token_id") or "")
    size = float(pos.get("size") or 0)
    avg_price = float(pos.get("avg_price") or 0)
    title = str(pos.get("asset_name") or "")
    outcome_label = str(pos.get("outcome") or "")
    if not token_id or size <= 0:
        return False

    bid = _live_best_bid(token_id)
    if bid is None or bid <= 0:
        logger.debug("[PanicSell] Best bid indisponibil pentru '%s' — skip.", title[:50])
        return False

    # --- TRIGGER MATEMATIC: comparăm prețul curent cu cel de la scanarea anterioară ---
    prev = _last_position_price.get(token_id)
    _last_position_price[token_id] = bid   # baseline mereu proaspăt (inclusiv la prima observare)
    if prev is None or prev <= 0:
        logger.debug("[PanicSell] '%s' baseline inițial preț=$%.4f (fără apel AI).", title[:50], bid)
        return False
    drop_pct = (prev - bid) / prev * 100.0   # pozitiv = scădere
    if drop_pct < _PANIC_THRESHOLD_PERCENT:
        # Preț stabil / în creștere / scădere sub prag → NICIUN apel AI (calea ieftină, obișnuită).
        logger.debug(
            "[PanicSell] '%s' preț $%.4f→$%.4f (%+.1f%%) sub pragul de %.1f%% — fără apel AI.",
            title[:50], prev, bid, -drop_pct, _PANIC_THRESHOLD_PERCENT,
        )
        return False

    # Scădere bruscă. Respectăm cooldown-ul de exit ÎNAINTE de a cheltui un apel LLM (anti-dublă
    # vânzare: dacă TP/SL sau o rundă anterioară de panic a încercat deja un exit recent, sărim).
    if not should_attempt_exit(token_id):
        logger.debug("[PanicSell] Token %s în cooldown de exit — skip apel AI.", token_id)
        return False

    logger.info(
        "[PanicSell] '%s' (%s) SCĂDERE BRUSCĂ preț $%.4f→$%.4f (−%.1f%% ≥ prag %.1f%%) — "
        "interoghez AI pe baza știrilor.",
        title[:50], outcome_label, prev, bid, drop_pct, _PANIC_THRESHOLD_PERCENT,
    )
    news = fetch_free_news(query=title[:100])
    decision = analyse_position_exit(title, outcome_label, avg_price, bid, news)
    # Reține decizia AI proaspătă pentru raportul de status per-heartbeat (fără re-apel Groq).
    _record_ai_decision(token_id, decision["action"], decision["confidence"], decision.get("reason", ""))
    logger.info(
        "[PanicSell] '%s' (%s) | entry=$%.4f bid=$%.4f → action=%s | confidence=%.2f | reason=%s",
        title[:50], outcome_label, avg_price, bid,
        decision["action"], decision["confidence"], decision["reason"],
    )

    if decision["action"] == "PANIC_SELL" and decision["confidence"] >= _PANIC_SELL_MIN_CONFIDENCE:
        _panic_sell_position(pos, bid, decision, account_state)
    elif decision["action"] == "PANIC_SELL":
        logger.info(
            "[PanicSell] PANIC_SELL pentru '%s' sub pragul de confidence %.2f (%.2f) — păstrez poziția.",
            title[:50], _PANIC_SELL_MIN_CONFIDENCE, decision["confidence"],
        )
    return True


def _panic_sell_position(pos: dict, bid: float, decision: dict, account_state: dict) -> None:
    """Lichidează o poziție DEȚINUTĂ: SELL la best bid (ocolește `_execute_trade` / cash gate),
    apoi actualizează starea locală și notifică pe Discord cu eticheta „PANIC SELL"."""
    token_id = str(pos.get("token_id") or "")
    size = float(pos.get("size") or 0)
    title = str(pos.get("asset_name") or "")
    outcome_label = str(pos.get("outcome") or "")
    reason = f"PANIC SELL (AI): {decision.get('reason', '')}"[:200]

    logger.warning(
        "[PanicSell] LICHIDEZ '%s' (%s) | token=%s | bid=$%.4f | size=%.2f | confidence=%.2f | %s",
        title[:60], outcome_label, token_id, bid, size,
        decision.get("confidence", 0.0), decision.get("reason", ""),
    )
    # Marchează încercarea de exit ÎNAINTE de plasare (cooldown partajat cu TP/SL → fără dublă vânzare).
    record_exit_attempt(token_id)
    result = place_limit_order(token_id=token_id, price=bid, side=Side.SELL, size=size)

    if result.get("success"):
        logger.info("[PanicSell] Ordin SELL plasat | order_id=%s", result.get("order_id"))
        # Reflectă închiderea în starea locală: scoate poziția și recreditează (aprox) balanța,
        # ca restul iterației (dedup, raport) să vadă portofoliul actualizat.
        account_state["active_positions"] = [
            p for p in account_state.get("active_positions", []) if p.get("token_id") != token_id
        ]
        account_state["pusd_balance"] = round(account_state.get("pusd_balance", 0.0) + bid * size, 6)
        notify_trade(
            "SELL", title or token_id, price=bid, size=size, total=bid * size,
            outcome=outcome_label, reason=reason,
            order_id=result.get("order_id"), action="PANIC SELL",
        )
    else:
        logger.error("[PanicSell] Ordin SELL respins de CLOB: %s", result.get("error"))


def _reevaluate_queued(item: dict, account_state: dict, allow_new_trades: bool = True) -> None:
    """Re-evaluează o oportunitate din coadă cu preț proaspăt + re-apel Groq.

    (a) ia best-ask-ul live din orderbook (fallback la prețul salvat); (b) re-cheamă
    `analyse_market` cu titlul + outcome-ul salvat și noul preț; (c) dacă teza
    rămâne validă (trade=True, confidence peste prag, reconfirmare YES), trimite
    ordinul prin `_execute_trade(from_queue=True)` — deci NU se re-pune în coadă."""
    title = str(item.get("market_title") or "Unknown")
    token_id = item.get("token_id")
    outcome_label = str(item.get("predicted_outcome") or "YES")
    event_id = str(item.get("event_id") or "")
    tick_size = str(item.get("tick_size") or "0.01")

    if not token_id:
        logger.info("[Queue] '%s' fără token_id — elimin din coadă.", title[:50])
        return

    # (a) Preț proaspăt din orderbook; fallback la prețul original salvat.
    fresh_price = _live_best_ask(str(token_id))
    if fresh_price is None:
        fresh_price = _qfloat(item.get("original_price"))
    if fresh_price <= 0:
        logger.info("[Queue] Preț indisponibil pentru '%s' — elimin din coadă.", title[:50])
        return

    # (b) Re-apel Groq cu noul preț. Întrebarea YES/NO = „mai cumpărăm acest outcome?".
    news = fetch_free_news(query=title[:100])
    decision = analyse_market(f"{title} ({outcome_label})", fresh_price, news)
    _record_analyzed(str(item.get("market_id")))
    logger.info(
        "[Queue] Reevaluare '%s' (%s) @ $%.4f → trade=%s | confidence=%.2f | reason=%s",
        title[:50], outcome_label, fresh_price,
        decision["trade"], decision["confidence"], decision["reason"],
    )

    # (c) Confirmare: doar dacă AI-ul reconfirmă cumpărarea (YES) cu confidence suficient.
    if not decision["trade"] or decision["confidence"] < _MIN_CONFIDENCE:
        return
    if decision["outcome"] != "YES":
        logger.info(
            "[Queue] '%s': AI nu mai confirmă cumpărarea (outcome=%s) — skip.",
            title[:50], decision["outcome"],
        )
        return

    _execute_trade(
        {"id": item.get("market_id"), "orderPriceMinTickSize": tick_size},
        account_state,
        token_id=str(token_id),
        price=fresh_price,
        outcome_label=outcome_label,
        decision=decision,
        event_id=event_id,
        title=title,
        from_queue=True,
        allow_new_trades=allow_new_trades,
    )


def _process_queue(account_state: dict, allow_new_trades: bool = True) -> None:
    """Procesează PRIORITAR coada next_in_line.json înainte de scanarea masivă.

    Oportunitățile cu confidence mai mare se evaluează întâi. Fiecare e scoasă din
    coadă ÎNAINTE de re-evaluare (coada avansează indiferent de rezultat — cerința
    4d). Se oprește când balanța scade sub minim sau limita de poziții e atinsă,
    lăsând restul intrărilor în fișier pentru runda următoare (cerința 4e)."""
    queue = _load_queue()
    if not queue:
        return
    logger.info(
        "[Queue] %d oportunități în așteptare — procesare prioritară înainte de scanare.",
        len(queue),
    )
    queue.sort(key=lambda it: _qfloat(it.get("confidence")), reverse=True)

    while queue and RUNNING:
        if account_state.get("pusd_balance", 0.0) < _MIN_BALANCE_PUSD:
            logger.info(
                "[Queue] Balanță sub minimul de $%.2f în timpul procesării cozii — "
                "opresc; %d intrări rămân pentru runda următoare.",
                _MIN_BALANCE_PUSD, len(queue),
            )
            break
        n_open = len(account_state.get("active_positions", []))
        if settings.MAX_ACTIVE_POSITIONS > 0 and n_open >= settings.MAX_ACTIVE_POSITIONS:
            logger.info(
                "[Queue] Limită de expunere atinsă (%d/%d) — opresc procesarea cozii.",
                n_open, settings.MAX_ACTIVE_POSITIONS,
            )
            break

        item = queue.pop(0)
        _save_queue(queue)  # eliminare imediată: coada avansează chiar dacă re-evaluarea eșuează
        try:
            _reevaluate_queued(item, account_state, allow_new_trades)
        except Exception as exc:
            logger.error(
                "[Queue] Eroare la reevaluarea '%s': %s",
                str(item.get("market_title"))[:50], exc, exc_info=True,
            )
        _interruptible_sleep(_INTER_ANALYSIS_DELAY)  # throttle după apelul LLM


# ---------------------------------------------------------------------------
# Scanare cu analiză în LOTURI (batching)
# ---------------------------------------------------------------------------
def _batch_request(item: dict) -> dict:
    """Reduce un item pregătit la payload-ul minim trimis lui analyse_batch."""
    if item["kind"] == "match":
        return {
            "market_id": item["market_id"], "kind": "match",
            "title": item["label"], "candidates": item["candidates"], "news": item["news"],
        }
    return {
        "market_id": item["market_id"], "kind": "binary",
        "title": item["title"], "price": item["yes_price"], "news": item["news"],
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    """True dacă excepția pare un rate-limit Groq (HTTP 429). Inspectează mesajul, iar dacă
    e un RetryError tenacity (retry epuizat) și ultima încercare învelită + __cause__, ca să
    prindem 429-ul chiar și când e împachetat."""
    parts = [str(exc)]
    last = getattr(exc, "last_attempt", None)  # tenacity.RetryError
    if last is not None:
        try:
            parts.append(str(last.exception()))
        except Exception:
            pass
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        parts.append(str(cause))
    blob = " ".join(parts).lower()
    return any(s in blob for s in ("429", "rate limit", "rate_limit", "too many requests"))


def _flush_batch(batch: list[dict], account_state: dict, allow_new_trades: bool = True) -> None:
    """Trimite un lot de piețe pregătite la Groq într-UN singur apel (analyse_batch),
    apoi aplică deciziile. Toate piețele din lot sunt marcate ca analizate (apelul LLM
    le-a acoperit pe toate). Un eșec total al apelului (rate-limit/API) NU marchează
    nimic → lotul se reîncearcă la scanarea următoare. Un singur throttle după apel.

    Bariera de balanță rămâne intactă: `_apply_decision` → `_execute_trade` verifică
    gate-ul de 1.05 pUSD per ordin; în plus, oprim aplicarea restului lotului dacă
    balanța scade sub minim în timpul execuției."""
    if not batch:
        return
    requests = [_batch_request(it) for it in batch]
    try:
        decisions = analyse_batch(requests)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            logger.warning(
                "[Batch] Rate-limit Groq (HTTP 429) la %d piețe — backoff %ds înainte de a "
                "reîncerca la scanarea următoare (protejăm bugetul zilnic de tokeni).",
                len(batch), _RATE_LIMIT_BACKOFF,
            )
            _interruptible_sleep(_RATE_LIMIT_BACKOFF)
        else:
            logger.error(
                "[Batch] Analiză batch eșuată (%d piețe) — reîncerc la scanarea următoare: %s",
                len(batch), exc,
            )
        return  # nu marcăm nimic ca analizat → se reîncearcă

    # Apelul LLM a acoperit tot lotul → marcăm fiecare piață ca analizată (cooldown 4h).
    for it in batch:
        _record_analyzed(str(it["market_id"]))
        _scan_stats["analizate (LLM)"] += 1

    for it in batch:
        mid = str(it["market_id"])
        decision = decisions.get(mid) or {
            "trade": False, "outcome": "NONE", "confidence": 0.0, "reason": "lipsă din răspunsul batch",
        }
        try:
            _apply_decision(it, decision, account_state, allow_new_trades)
        except Exception as exc:
            logger.error("[Batch] Eroare la aplicarea deciziei pentru id=%s: %s", mid, exc, exc_info=True)
        if not _DRY_RUN and account_state.get("pusd_balance", 0.0) < _MIN_BALANCE_PUSD:
            logger.info(
                "[Batch] Balanță sub minimul de $%.2f în timpul execuției lotului — "
                "opresc aplicarea restului.", _MIN_BALANCE_PUSD,
            )
            break

    _interruptible_sleep(_INTER_ANALYSIS_DELAY)  # throttle după apelul LLM (per lot)


def _market_volume(market: dict) -> float:
    """Volumul ($) al unei piețe Gamma (`volume` sau `volumeNum`); 0.0 dacă lipsește/invalid."""
    try:
        return float(market.get("volume") or market.get("volumeNum") or 0)
    except (ValueError, TypeError):
        return 0.0


def _market_token_ids(market: dict) -> list[str]:
    """clobTokenIds pieței (string JSON sau listă) → listă de string-uri; [] la lipsă/eroare."""
    raw = market.get("clobTokenIds") or market.get("clob_token_ids") or "[]"
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
        return [str(t) for t in ids]
    except (ValueError, TypeError):
        return []


def _strict_prefilter(markets: list[dict], account_state: dict) -> list[dict]:
    """STEP 2 (anti-429): reduce DRASTIC câte piețe ajung la Groq, ca modelul mare
    (gpt-oss-20b) să nu depășească bugetul zilnic de tokeni.

    Păstrează o piață dacă: avem deja o poziție deschisă pe ea, SAU volumul ≥
    _STRICT_MIN_VOLUME_USD, SAU (când _STRICT_KEYWORDS e setat) titlul/slug-ul conține un
    keyword de interes. Apoi sortează după volum desc și taie la _MAX_MARKETS_PER_SCAN
    (cele mai lichide întâi). STRICT_FILTER=false → no-op (revine la „scanează tot").

    ATENȚIE: filtrarea pe volum poate elimina meciuri de azi (volum mic la creare) —
    coboară STRICT_MIN_VOLUME_USD sau adaugă echipele în STRICT_KEYWORDS ca să le prinzi."""
    if not _STRICT_FILTER:
        return markets

    held = {p.get("token_id") for p in account_state.get("active_positions", [])}
    held.discard(None)

    def _keep(m: dict) -> bool:
        if held and any(t in held for t in _market_token_ids(m)):
            return True
        if _STRICT_KEYWORDS:
            text = f"{m.get('question') or m.get('title') or ''} {m.get('slug', '')}".lower()
            if any(kw in text for kw in _STRICT_KEYWORDS):
                return True
        return _market_volume(m) >= _STRICT_MIN_VOLUME_USD

    kept = [m for m in markets if _keep(m)]
    kept.sort(key=_market_volume, reverse=True)
    capped = kept[:_MAX_MARKETS_PER_SCAN] if _MAX_MARKETS_PER_SCAN > 0 else kept
    cap_label = f"plafon {_MAX_MARKETS_PER_SCAN}" if _MAX_MARKETS_PER_SCAN > 0 else "fără plafon"
    logger.info(
        "[StrictFilter] %d → %d piețe după filtru (volum≥$%.0f / %d keywords / poziții "
        "deschise) → %d trimise spre analiză (%s).",
        len(markets), len(kept), _STRICT_MIN_VOLUME_USD, len(_STRICT_KEYWORDS),
        len(capped), cap_label,
    )
    return capped


def _scan_markets(markets: list[dict], account_state: dict, allow_new_trades: bool = True) -> None:
    """Scanează piețele deja filtrate pe istoric. IMPLICIT acumulează piețele în loturi
    de _BATCH_SIZE și le trimite la Groq într-un SINGUR apel (economisind cereri →
    modelul gpt-oss-20b fără rate-limit). BATCH_SIZE<=1 revine la analiza individuală
    (process_market, o cerere per piață).

    Gate de balanță intra-scanare: dacă un ordin epuizează balanța, abandonăm restul
    rundei (sleep-ul defensiv de la începutul iterației următoare preia)."""
    _scan_stats.clear()

    # Fallback non-batch: o cerere LLM per piață (comportamentul clasic).
    if _BATCH_SIZE <= 1:
        for market in markets:
            if not RUNNING:
                break
            if not _DRY_RUN and account_state.get("pusd_balance", 0.0) < _MIN_BALANCE_PUSD:
                logger.info(
                    "[Agent] Balanța a fost epuizată în timpul scanării curente. "
                    "Abandonez restul piețelor din această rundă pentru a economisi API calls."
                )
                break
            hist_before = len(_analyzed_history)
            try:
                process_market(market, account_state, allow_new_trades)
            except Exception as exc:
                logger.error(
                    "[Agent] Eroare la procesarea pieței id=%s: %s",
                    market.get("id", "?"), exc, exc_info=True,
                )
            # Throttle DOAR după o analiză reală (s-a înregistrat piața în history).
            if len(_analyzed_history) > hist_before:
                _interruptible_sleep(_INTER_ANALYSIS_DELAY)
        return

    # Cale implicită: acumulează loturi de _BATCH_SIZE și trimite-le într-un apel.
    batch: list[dict] = []
    for market in markets:
        if not RUNNING:
            break
        if not _DRY_RUN and account_state.get("pusd_balance", 0.0) < _MIN_BALANCE_PUSD:
            logger.info(
                "[Agent] Balanța a fost epuizată în timpul scanării curente. "
                "Abandonez restul piețelor din această rundă pentru a economisi API calls."
            )
            break
        try:
            item = _prepare_market(market, account_state)
        except Exception as exc:
            logger.error(
                "[Agent] Eroare la pregătirea pieței id=%s: %s",
                market.get("id", "?"), exc, exc_info=True,
            )
            item = None
        if item is None:
            continue

        batch.append(item)
        if len(batch) >= _BATCH_SIZE:
            _flush_batch(batch, account_state, allow_new_trades)
            batch = []
            if not _DRY_RUN and account_state.get("pusd_balance", 0.0) < _MIN_BALANCE_PUSD:
                logger.info("[Agent] Balanța a fost epuizată după un lot. Abandonez restul rundei.")
                break

    # Lotul parțial rămas la finalul scanării.
    if batch and RUNNING:
        _flush_batch(batch, account_state, allow_new_trades)


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
    if _DRY_RUN:
        logger.warning(
            "[DRY_RUN] MOD SIMULARE ACTIV: trec peste gate-urile de balanță ($%.2f) și de "
            "expunere ca să scanez și să trimit loturi către Groq, dar NU plasez NICIUN ordin "
            "real și sar verificările TP/SL. Dezactivează DRY_RUN pentru tranzacționare reală.",
            _MIN_BALANCE_PUSD,
        )

    if _NOTIFICATION_TEST_MODE:
        logger.warning(
            "[NOTIF-TEST] MOD TEST NOTIFICĂRI ACTIV: după fiecare BUY REAL reușit, botul vinde "
            "imediat (după 1s) exact câte share-uri a cumpărat — ordine REALE pe ambele direcții, "
            "ca să verifice webhook-urile BUY+SELL. Suprascrie holding-ul/TP/SL. Dezactivează "
            "NOTIFICATION_TEST_MODE pentru operare normală."
        )

    if _PANIC_SELL_ENABLED:
        logger.warning(
            "[PanicSell] PANIC SELL ACTIV: monitorizez pozițiile deținute la fiecare heartbeat și, "
            "DOAR la o scădere bruscă de preț (≥ %.1f%% față de minutul precedent), întreb Groq și "
            "LICHIDEZ (SELL real la best bid) cele invalidate catastrofal (confidence ≥ %.2f), "
            "independent de cash. Dezactivează PANIC_SELL_ENABLED pentru a opri.",
            _PANIC_THRESHOLD_PERCENT, _PANIC_SELL_MIN_CONFIDENCE,
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
        "Entering autonomous AI agent loop (heartbeat=%.0fs, scanare la fiecare %.0fs). "
        "Send SIGTERM/SIGINT to stop.",
        _LOOP_INTERVAL, _SCAN_INTERVAL_SECONDS,
    )

    # Ping de pornire pe Discord: confirmă INDEPENDENT că webhook-ul e funcțional,
    # înainte de orice tranzacție. No-op fără webhook configurat; nu ridică niciodată
    # (trece prin `_post`, care prinde orice eroare de rețea).
    notify_startup()

    try:
        # Scanarea masivă (fetch Gamma + Groq) e decuplată de heartbeat: reținem când am
        # scanat ultima oară ca să o rulăm cel mult o dată la _SCAN_INTERVAL_SECONDS, în timp
        # ce restul buclei (reconciliere, TP/SL, Panic Sell, raport, coadă) rulează la fiecare
        # _LOOP_INTERVAL. Pornim de la 0.0 → prima iterație scanează imediat.
        last_scan_time = 0.0
        while RUNNING:
            try:
                # Reconciliere stare cont — apel critic de rețea (CLOB get_balance_allowance
                # + Data API positions). Îl izolăm într-un try/except: un glitch temporar al
                # API-ului Polymarket (ex. 401 Unauthorized dintr-o desincronizare de ceas sau
                # o cheie respinsă tranzitoriu) NU trebuie să trântească tot procesul. Funcția
                # are deja un retry intern (3×); dacă tot eșuează, intrăm în standby și reîncercăm
                # runda următoare cu `continue`, în loc să propagăm eroarea în handler-ul fatal.
                try:
                    account_state = get_account_state()
                except PolyApiException as exc:
                    status = getattr(exc, "status_code", None)
                    if status == 401:
                        logger.error(
                            "[Agent] Eroare temporară de autentificare 401 (Unauthorized/"
                            "Invalid api key). Posibil glitch API Polymarket sau desincronizare "
                            "de ceas (clock drift). Reîncerc după standby de %d min...",
                            _LOW_BALANCE_SLEEP // 60,
                        )
                    else:
                        logger.error(
                            "[Agent] Eroare API Polymarket la reconciliere (status=%s): %s. "
                            "Reîncerc după standby de %d min...",
                            status, exc, _LOW_BALANCE_SLEEP // 60,
                        )
                    _interruptible_sleep(_LOW_BALANCE_SLEEP)
                    continue
                except Exception as exc:
                    logger.error(
                        "[Agent] Reconcilierea stării contului a eșuat (eroare tranzitorie de "
                        "rețea?): %s. Reîncerc după standby de %d min...",
                        exc, _LOW_BALANCE_SLEEP // 60, exc_info=True,
                    )
                    _interruptible_sleep(_LOW_BALANCE_SLEEP)
                    continue

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

                # Curățarea cozii (DOAR cu ACTIVATE_QUEUE — altfel coada e complet inertă):
                # elimină oportunitățile expirate (> _QUEUE_TTL_HOURS) la începutul fiecărei
                # iterații, ca să nu re-evaluăm teze pe baza unor știri perimate.
                if _ACTIVATE_QUEUE:
                    try:
                        _clean_queue()
                    except Exception as exc:
                        logger.error("[Queue] Curățarea cozii a eșuat: %s", exc, exc_info=True)

                # Raport orar P/L pe Discord. Rulează înaintea verificării de balanță
                # ca să continue și când fondurile sunt insuficiente (cu cash insuficient
                # se dezactivează doar cumpărarea, dar bucla rulează în continuare la 60s,
                # deci raportul tot iese la timp).
                try:
                    _maybe_send_pnl_report(account_state)
                except Exception as exc:
                    logger.error("[Agent] Raport P/L eșuat: %s", exc, exc_info=True)

                # Verifică TP/SL pe pozițiile deschise (opt-in via TAKE_PROFIT_PCT /
                # STOP_LOSS_PCT). Rulează înainte de verificarea de balanță: vânzarea (SELL)
                # trebuie să funcționeze și când nu mai sunt fonduri de cumpărare — pentru a
                # vinde avem nevoie doar de token-urile deja deținute. Cu cash insuficient
                # bucla NU mai doarme 15 min, deci acest check (protecția pozițiilor) se reia
                # la fiecare 60s, nu o dată pe sfert de oră. În DRY_RUN sărim complet: TP/SL
                # plasează ordine SELL REALE (prin position_manager, nu prin _execute_trade)
                # → nimic nu trebuie să ajungă pe lanț.
                if not _DRY_RUN:
                    try:
                        check_exit_conditions(account_state)
                    except Exception as exc:
                        logger.error("[Agent] Exit check eșuat: %s", exc, exc_info=True)

                # STEP 4 — Panic Sell: monitorizare la FIECARE heartbeat a pozițiilor deținute (opt-in
                # via PANIC_SELL_ENABLED), cu apel Groq DOAR la o scădere bruscă de preț (TRIGGER
                # MATEMATIC, _PANIC_THRESHOLD_PERCENT). La fel ca TP/SL: rulează ÎNAINTE de gate-ul de
                # balanță (SELL nu cere cash) și e sărit în DRY_RUN (plasează SELL-uri REALE prin
                # place_limit_order, nu prin _execute_trade interceptat).
                if not _DRY_RUN:
                    try:
                        _reevaluate_active_positions(account_state)
                    except Exception as exc:
                        logger.error("[PanicSell] Re-evaluarea pozițiilor a eșuat: %s", exc, exc_info=True)

                # Raport de status per-heartbeat: la FIECARE buclă (60s), NU doar la scanare, și
                # NECONDIȚIONAT de balanță/expunere/cadență (rulează înaintea tuturor gate-urilor
                # de mai jos), ca utilizatorul să vadă în fiecare minut că botul își urmărește activ
                # portofoliul. ZERO apeluri Groq — doar starea deja reconciliată + un best bid/ask
                # best-effort. Izolat în try/except ca un eșec de raportare să nu oprească bucla.
                try:
                    _log_position_status(account_state)
                except Exception as exc:
                    logger.error("[Status] Raportul de status al pozițiilor a eșuat: %s", exc, exc_info=True)

                # --- Separare Cumpărare vs. Vânzare la cash insuficient ---
                # Sub pragul minim de cash NU mai putem deschide poziții NOI (BUY) — un ordin
                # ar fi oricum blocat în `_execute_trade`. DAR pozițiile deschise rămân expuse:
                # pentru a le ÎNCHIDE (SELL / TP / SL) avem nevoie doar de token-urile deja
                # deținute, nu de cash. De aceea NU mai intrăm în standby 15 min cu `continue`
                # (ne-ar lăsa pozițiile nemonitorizate un sfert de oră); doar DEZACTIVĂM modul
                # de cumpărare (`allow_new_trades=False`) și păstrăm bucla la cadența normală
                # de 60s, ca reconcilierea + check_exit_conditions (deja rulate mai sus, fără
                # cash) să se reia la fiecare minut.
                #
                # `allow_new_trades` e firul roșu care, dus în _process_queue / _scan_markets /
                # _apply_decision → _execute_trade, blochează STRICT orice BUY, lăsând SELL-urile
                # (check_exit_conditions, în afara acestui flux) neatinse. În DRY_RUN rămâne True
                # (oricum nu pleacă niciun ordin real — _execute_trade e interceptat la intrare).
                allow_new_trades = _DRY_RUN or pusd_balance >= _MIN_BALANCE_PUSD
                if not allow_new_trades:
                    logger.warning(
                        "[Agent] Cash insuficient pentru achiziții noi (%.4f pUSD < $%.2f). "
                        "Modul de cumpărare este dezactivat — continui să monitorizez și pot "
                        "vinde pozițiile existente (TP/SL/exit rămân active).",
                        pusd_balance, _MIN_BALANCE_PUSD,
                    )
                    # Scanarea masivă produce DOAR decizii de cumpărare (BUY) pe piețe noi —
                    # toate ar fi blocate acum → ar irosi degeaba bugetul de tokeni Groq (vezi
                    # filtrul STRICT/batching). O sărim; protecția pozițiilor vine din exit
                    # checks (SELL), rulate deja mai sus și reluate la următoarea iterație (60s).
                    _interruptible_sleep(_LOOP_INTERVAL)
                    continue
                if _DRY_RUN and pusd_balance < _MIN_BALANCE_PUSD:
                    logger.info(
                        "[DRY_RUN] Balanță %.4f sub $%.2f — ignor gate-ul de balanță și "
                        "continui scanarea (simulare).", pusd_balance, _MIN_BALANCE_PUSD,
                    )

                # Limită globală de expunere
                n_open = len(account_state.get("active_positions", []))
                maxed = (
                    settings.MAX_ACTIVE_POSITIONS > 0
                    and n_open >= settings.MAX_ACTIVE_POSITIONS
                )
                # Fără ACTIVATE_QUEUE: MAX poziții = standby (comportament implicit — nu
                # irosim API calls scanând când nu putem deschide nimic). Cu ACTIVATE_QUEUE
                # NU dormim: scanăm ca să construim wishlist-ul (piețele aprobate de AI sunt
                # salvate în next_in_line.json pentru când se eliberează un slot).
                # În DRY_RUN nu dormim nici la MAX poziții: vrem să scanăm și să trimitem loturi
                # către Groq (nu deschidem nimic oricum — _execute_trade e interceptat).
                if maxed and not _ACTIVATE_QUEUE and not _DRY_RUN:
                    logger.info(
                        "[Agent] Limită de expunere atinsă: %d/%d poziții deschise. "
                        "Nu se deschid poziții noi. Aștept...",
                        n_open,
                        settings.MAX_ACTIVE_POSITIONS,
                    )
                    _interruptible_sleep(_LOOP_INTERVAL)
                    continue

                # Procesare PRIORITARĂ a cozii (DOAR cu ACTIVATE_QUEUE și DOAR când avem
                # sloturi libere): re-evaluăm wishlist-ul și deschidem pozițiile aprobate
                # ÎNAINTE de scanarea masivă. La MAX poziții n-am putea deschide nimic, deci
                # sărim execuția și mergem direct la scanare (care doar va alimenta wishlist-ul).
                if _ACTIVATE_QUEUE and not maxed:
                    try:
                        _process_queue(account_state, allow_new_trades=allow_new_trades)
                    except Exception as exc:
                        logger.error("[Queue] Procesarea cozii a eșuat: %s", exc, exc_info=True)

                    # Coada poate epuiza fondurile: dacă balanța a scăzut sub minim, dezactivăm
                    # cumpărarea și sărim scanarea masivă (ar produce doar BUY-uri blocate). NU
                    # dormim 15 min — păstrăm cadența de 60s ca pozițiile să rămână monitorizate
                    # (exit/SELL se reia la următoarea iterație).
                    if not _DRY_RUN and account_state.get("pusd_balance", 0.0) < _MIN_BALANCE_PUSD:
                        allow_new_trades = False
                        logger.warning(
                            "[Agent] Fonduri epuizate la procesarea cozii — modul de cumpărare "
                            "dezactivat; sar scanarea, dar continui monitorizarea pozițiilor.",
                        )
                        _interruptible_sleep(_LOOP_INTERVAL)
                        continue

                # --- Cadența scanării masive (decuplată de heartbeat) ---
                # Reconcilierea + exit-urile (TP/SL/Panic Sell) + coada au rulat deja la acest
                # heartbeat (60s). Scanarea masivă (fetch Gamma + Groq pe sute de piețe) e însă
                # costisitoare → o rulăm cel mult o dată la _SCAN_INTERVAL_SECONDS. În DRY_RUN o
                # rulăm la fiecare buclă (stress-test de rate-limit); 0 = dezactivat (scanează mereu).
                now = time.time()
                scan_due = (
                    _DRY_RUN
                    or _SCAN_INTERVAL_SECONDS <= 0
                    or (now - last_scan_time) >= _SCAN_INTERVAL_SECONDS
                )
                if not scan_due:
                    logger.debug(
                        "[Agent] Scanare amânată: %.0f/%.0fs scurse. Heartbeat la %.0fs — "
                        "monitorizez și pot vinde pozițiile existente între scanări.",
                        now - last_scan_time, _SCAN_INTERVAL_SECONDS, _LOOP_INTERVAL,
                    )
                    _interruptible_sleep(_LOOP_INTERVAL)
                    continue
                last_scan_time = now

                if maxed:  # _ACTIVATE_QUEUE sau _DRY_RUN (altfel am fi făcut deja continue)
                    if _DRY_RUN:
                        logger.info(
                            "[DRY_RUN] Limită de expunere atinsă (%d/%d) — ignor și scanez ca "
                            "să trimit loturi către Groq (simulare).",
                            n_open, settings.MAX_ACTIVE_POSITIONS,
                        )
                    else:
                        logger.info(
                            "[Agent] Limită de expunere atinsă (%d/%d) — ACTIVATE_QUEUE activ: "
                            "scanez pentru wishlist (next_in_line.json), fără a deschide poziții noi.",
                            n_open, settings.MAX_ACTIVE_POSITIONS,
                        )

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

                # STEP 2 (anti-429): filtru STRICT — reduce DRASTIC câte piețe ajung la Groq
                # (poziție deschisă / volum mare / keyword de interes, apoi plafonat). Vezi
                # _strict_prefilter. STRICT_FILTER=false dezactivează complet acest pas.
                markets = _strict_prefilter(markets, account_state)
                if not markets:
                    logger.info("[StrictFilter] Nicio piață nu trece filtrul strict în această rundă. Aștept...")
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
                # Scanare cu analiză în LOTURI (sau individual dacă BATCH_SIZE<=1). Gate-ul
                # de balanță intra-scanare și throttle-ul (un singur sleep per lot/analiză)
                # sunt încapsulate în `_scan_markets`, care resetează și `_scan_stats`.
                # `allow_new_trades` (True aici — altfel am fi făcut deja continue, exceptând
                # DRY_RUN) e dus până la `_execute_trade`, gardul comun de blocare a BUY-urilor.
                _scan_markets(markets, account_state, allow_new_trades=allow_new_trades)

                # Sumar de filtrare: ce a oprit piețele. Cheia pentru a vedea DE CE
                # „ignoră" botul tot — ex: {'volum mic': 195, 'analizate (LLM)': 3}.
                # Lista piețelor scanate rămâne STRICT în consolă/loguri (vezi logurile
                # „[Agent] Procesez piața ..." de mai sus) — pe Discord nu se trimite
                # nimic per buclă, doar raportul periodic din _maybe_send_pnl_report.
                logger.info("[Scan] Rezultat filtrare pe %d piețe: %s",
                            len(markets), dict(_scan_stats) or "nimic procesat")

                _interruptible_sleep(_LOOP_INTERVAL)

            except Exception as exc:
                # Plasă de siguranță per-iterație: prindem ORICE excepție neprevăzută
                # (ValueError/TypeError/timeout de rețea etc.) ca un crash într-o rundă să
                # NU trântească tot procesul. O logăm cu traceback complet, dormim câteva
                # secunde și RELUĂM automat bucla (auto-restart) în loc să ieșim. SIGTERM/
                # SIGINT rămân onorate: _interruptible_sleep verifică RUNNING în timpul pauzei.
                logger.critical(
                    "[Agent] Excepție neprevăzută în bucla principală (%s) — prinsă ca să"
                    " nu cadă tot procesul; reiau automat după %.0fs.",
                    exc, _CRASH_RECOVERY_SLEEP, exc_info=True,
                )
                _interruptible_sleep(_CRASH_RECOVERY_SLEEP)
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
