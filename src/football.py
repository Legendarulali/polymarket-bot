"""
Parsare meciuri de fotbal și maparea pieței 1X2 (Match Odds).

Polymarket reprezintă un meci ca o piață Gamma cu array-uri PARALELE codate ca
string JSON: `outcomes` (numele rezultatelor), `outcomePrices` (probabilitățile
YES) și `clobTokenIds` (token-ul CLOB al fiecărui rezultat). Pentru un meci 1X2:

    outcomes      = ["Arsenal", "Draw", "Chelsea"]
    outcomePrices = ["0.55",    "0.25", "0.20"]
    clobTokenIds  = ["0x..a",   "0x..x", "0x..c"]

Acest modul transformă piața brută într-o structură 1X2 cu rezultatele mapate
strict:
    "1" = victoria primei echipe (gazdă / home)
    "X" = egal (draw / tie)
    "2" = victoria celei de-a doua echipe (oaspete / away)

Prețul fiecărui rezultat ESTE probabilitatea lui YES (sumă ≈ 1.0 pe cele 3
rezultate), deci fereastra de probabilitate [0.45, 0.95] se aplică direct pe
prețul fiecărui „picior" (leg).

Pur Python (json/re/logging) — fără dependențe noi de mockuit în conftest.
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

# Codurile 1X2 în ordine canonică (folosită la logare și la ordonarea candidaților).
HOME, DRAW, AWAY = "1", "X", "2"
ORDERED_CODES = (HOME, DRAW, AWAY)

# Nume care înseamnă „egal" în outcome-urile Polymarket (normalizate, lowercase).
_DRAW_NAMES = {"draw", "tie", "x", "empate", "nul", "egalite", "egal"}

# Separatori titlu „Echipă A vs Echipă B", în ordinea priorității.
_TITLE_SEPARATORS = (" vs. ", " vs ", " v. ", " v ", " versus ")
_SLUG_SEPARATORS = ("-vs-", "-v-", "-beat-")
# Format binar „Will [Echipa A] beat [Echipa B]?": ambii operanzi trebuie să
# înceapă cu MAJUSCULĂ (nume proprii de echipe), ca să NU confundăm fraze gen
# „beat expectations" / „defeat inflation" cu un meci. Articol opțional „the".
_BEAT_RE = re.compile(
    r"\b([A-Z][\w.&'’-]*(?:\s+[A-Z][\w.&'’-]*)*)\s+(?:to\s+)?"
    r"(?:beat|beats|defeat|defeats)\s+(?:the\s+)?"
    r"([A-Z][\w.&'’-]*(?:\s+[A-Z][\w.&'’-]*)*)"
)
# Prefix interogativ eliminat înainte de detecția „beat" (plus articolul „the").
_LEAD_INTERROGATIVE_RE = re.compile(
    r"^\s*(?:will|can|could|would|does|do|is|are)\s+(?:the\s+)?", re.IGNORECASE
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    """Normalizează un nume pentru comparație: lowercase, doar alfanumeric + spațiu."""
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


def _json_array(raw) -> list:
    """Dezserializează un câmp Gamma care e fie listă, fie string JSON (ex: '["a","b"]')."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            logger.warning("[Football] Câmp JSON invalid, ignorat: %r", raw[:80])
    return []


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _strip_prefix(team: str) -> str:
    """Curăță un nume de echipă de prefixe de turneu și semne de întrebare.

    Ex: 'EPL: Arsenal' → 'Arsenal'; '2026 World Cup: France' → 'France';
    'Chelsea?' → 'Chelsea'. Păstrăm partea de după ultimul ':' (numele real).
    """
    cleaned = str(team).strip().strip("?").strip()
    if ":" in cleaned:
        cleaned = cleaned.rsplit(":", 1)[-1].strip()
    return cleaned


def extract_teams(title: str, slug: str = "") -> tuple[str | None, str | None]:
    """Extrage (gazdă, oaspete) dintr-un titlu/slug de meci head-to-head.

    Acoperă: 'A vs B' / 'A v B' / 'A versus B', formatul binar 'Will A beat B?'
    ('beat'/'defeat', cu operanzi cu majusculă) și slug-uri 'a-vs-b' / 'a-beat-b'.
    Întoarce (None, None) dacă nu e un format head-to-head — caz în care apelantul
    cade pe numele rezultatelor (outcomes) pentru a deduce echipele.
    """
    text = str(title or "")
    lowered = text.lower()
    for sep in _TITLE_SEPARATORS:
        idx = lowered.find(sep)
        if idx != -1:
            home = _strip_prefix(text[:idx])
            away = _strip_prefix(text[idx + len(sep):])
            if home and away:
                return home, away

    # Format „beat / defeat" (întrebări binare: „Will France beat Germany?").
    # Eliminăm prefixul interogativ, apoi cerem doi operanzi cu majusculă.
    match = _BEAT_RE.search(_LEAD_INTERROGATIVE_RE.sub("", text))
    if match:
        home = _strip_prefix(match.group(1))
        away = _strip_prefix(match.group(2))
        if home and away:
            return home, away

    slug_lower = (slug or "").lower()
    for sep in _SLUG_SEPARATORS:
        if sep in slug_lower:
            left, _, right = slug_lower.partition(sep)
            home = _strip_prefix(left.replace("-", " ")).title()
            # Slug-ul poate avea sufix de dată după echipa oaspete (ex: '...-2026-06-18').
            right = re.sub(r"[-\s]\d{2,4}([-\s]\d{1,2}){0,2}$", "", right)
            away = _strip_prefix(right.replace("-", " ")).title()
            if home and away:
                return home, away

    return None, None


def _is_draw(name: str) -> bool:
    """True dacă numele rezultatului înseamnă „egal" (Draw/Tie/X/…)."""
    return _norm(name) in _DRAW_NAMES


def _name_score(outcome_norm: str, team_norm: str) -> int:
    """Cât de bine se potrivește numele unui rezultat cu o echipă (0 = deloc)."""
    if not outcome_norm or not team_norm:
        return 0
    if outcome_norm == team_norm:
        return 3
    if outcome_norm in team_norm or team_norm in outcome_norm:
        return 2
    shared = set(outcome_norm.split()) & set(team_norm.split())
    return 1 if shared else 0


def _assign_codes(raw_legs: list[dict], home: str | None, away: str | None) -> dict[str, dict]:
    """Mapează fiecare leg pe codul 1 / X / 2.

    Strategie: (1) recunoaște egalul după nume → X; (2) potrivește restul după
    numele echipelor din titlu (home→1, away→2); (3) completează codurile rămase
    în ordinea de intrare (convenția Polymarket: primul rezultat = gazdă).
    """
    home_n, away_n = _norm(home or ""), _norm(away or "")
    by_code: dict[str, dict] = {}
    used: set[int] = set()

    # 1) Egal → X
    for i, leg in enumerate(raw_legs):
        if _is_draw(leg["name"]):
            leg["code"] = DRAW
            by_code[DRAW] = leg
            used.add(i)
            break  # un singur leg de tip egal

    # 2) Potrivire pe nume cu gazdă (1) / oaspete (2)
    for code, team_n in ((HOME, home_n), (AWAY, away_n)):
        if not team_n:
            continue
        best_i, best_score = None, 0
        for i, leg in enumerate(raw_legs):
            if i in used:
                continue
            score = _name_score(_norm(leg["name"]), team_n)
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score > 0:
            raw_legs[best_i]["code"] = code
            by_code[code] = raw_legs[best_i]
            used.add(best_i)

    # 3) Completează codurile neatribuite (1 apoi 2) în ordinea de intrare.
    free = [c for c in (HOME, AWAY) if c not in by_code]
    for i, leg in enumerate(raw_legs):
        if i in used or not free:
            continue
        code = free.pop(0)
        leg["code"] = code
        by_code[code] = leg
        used.add(i)

    return by_code


def parse_match_market(market: dict) -> dict | None:
    """Construiește structura 1X2 a unui meci dintr-o piață Gamma.

    Întoarce None dacă piața nu e un meci 1X2 mapabil (ex: rezultate Yes/No
    binare, sau lipsesc array-urile outcomes/prices/tokens) — caz în care
    apelantul poate cădea pe fluxul binar moștenit.

    Structura returnată:
        {
          "home": str, "away": str, "label": "Home vs Away",
          "legs": [ {"code","name","price","token_id"}, ... ]  # ordine 1, X, 2
          "by_code": {"1": leg, "X": leg, "2": leg},
          "is_three_way": bool,
        }
    """
    names = _json_array(market.get("outcomes"))
    prices = _json_array(market.get("outcomePrices"))
    tokens = _json_array(market.get("clobTokenIds"))
    n = min(len(names), len(prices), len(tokens))
    if n < 2:
        return None  # nu putem construi nici măcar un 1 vs 2

    raw_legs = [
        {
            "name": str(names[i]).strip(),
            "price": _to_float(prices[i]),
            "token_id": str(tokens[i]),
        }
        for i in range(n)
    ]

    # Piață binară Yes/No → nu e 1X2; lăsăm fluxul moștenit să o trateze.
    lowered = {leg["name"].lower() for leg in raw_legs}
    if lowered <= {"yes", "no"}:
        return None

    home, away = extract_teams(
        market.get("question") or market.get("title") or "",
        market.get("slug", ""),
    )
    by_code = _assign_codes(raw_legs, home, away)
    if HOME not in by_code and AWAY not in by_code:
        return None  # niciun leg de echipă recunoscut

    # Numele finale ale echipelor: preferă numele rezultatului mapat, cu fallback
    # pe ce am extras din titlu (sau placeholder dacă lipsește complet).
    home_name = by_code[HOME]["name"] if HOME in by_code else (home or "Home")
    away_name = by_code[AWAY]["name"] if AWAY in by_code else (away or "Away")

    legs = [by_code[c] for c in ORDERED_CODES if c in by_code]
    return {
        "home": home_name,
        "away": away_name,
        "label": f"{home_name} vs {away_name}",
        "legs": legs,
        "by_code": by_code,
        "is_three_way": DRAW in by_code,
    }


def format_legs(parsed: dict) -> str:
    """Linie de log a cotelor 1X2: '1 (Arsenal)=0.55 | X (Draw)=0.25 | 2 (Chelsea)=0.20'."""
    parts = []
    for code in ORDERED_CODES:
        leg = parsed["by_code"].get(code)
        if leg:
            parts.append(f"{code} ({leg['name']})={leg['price']:.2f}")
    return " | ".join(parts)
