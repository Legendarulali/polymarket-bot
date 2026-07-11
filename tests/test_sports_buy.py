"""
Integration-style tests: scanning & BUYing NEW sports markets (NBA preseason & co).

Validează că motorul UNIVERSAL e pregătit pentru piețele de sport nou-apărute
(pre-sezon NBA, tenis WTA etc.):
  1. detecția sportului și rutarea corectă a payload-urilor Gamma (meci binar fără egal);
  2. prompturile generate: prompturile de BUY pentru meciuri sunt SPORT-AWARE —
     _match_sport_context injectează „[Context Sportiv: …]" cu sportul detectat
     (_detect_sport pe titlul complet al pieței) și, la piețele fără egal
     (is_three_way=False), mesajul explicit că egalul (X) NU e o opțiune de pariere;
     în batch fiecare meci primește propria etichetă de sport. Garanția rămâne dublată
     STRUCTURAL (fără leg „X" → un pick „X" e anulat de validare), iar promptul de
     re-evaluare (_build_position_exit_prompt) își păstrează contextul sport-aware;
  3. parsarea deciziei de BUY în stil OpenRouter (booleene ca STRING, fără JSON Mode);
  4. fluxul complet mockuit: piață NBA → AI zice BUY → _execute_trade → baseline-ul
     stal e curățat din _last_position_price → totul persistat în history.json.

Toate dependențele grele sunt mockuite în conftest.py; funcțiile testate sunt cele
REALE din main / src (nu copii locale). Nu se modifică niciun cod de producție.
"""
import json
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Payload-uri Gamma simulate (date brute de piață, ca de la /markets)
# ---------------------------------------------------------------------------

def _nba_gamma_market(mid="nba-pre-1"):
    """Meci NBA de pre-sezon: piață Gamma cu 2 outcome-uri (moneyline, FĂRĂ egal)."""
    return {
        "id": mid,
        "question": "NBA Preseason: Los Angeles Lakers vs Golden State Warriors",
        "slug": "nba-preseason-los-angeles-lakers-vs-golden-state-warriors",
        "volume": "75000",
        "outcomes": json.dumps(["Lakers", "Warriors"]),
        "outcomePrices": json.dumps(["0.62", "0.38"]),
        "clobTokenIds": json.dumps(["0xLAKERS", "0xGSW"]),
        "events": [{"slug": "nba-preseason-lal-gsw"}],
    }


def _wta_gamma_market(mid="wta-mtl-1"):
    """Meci de tenis WTA în format binar Yes/No (structura curentă Polymarket la tenis)."""
    return {
        "id": mid,
        "question": "WTA Montreal: Will Iga Swiatek beat Aryna Sabalenka?",
        "slug": "wta-montreal-iga-swiatek-beat-aryna-sabalenka",
        "volume": "30000",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.55", "0.45"]),
        "clobTokenIds": json.dumps(["0xIGA", "0xARYNA"]),
        "events": [{"slug": "wta-montreal"}],
    }


def _nba_candidates():
    """Candidații 1X2 eligibili ai unui meci NBA: DOAR 1 și 2 — X nu există structural."""
    return [
        {"code": "1", "name": "Lakers", "price": 0.62},
        {"code": "2", "name": "Warriors", "price": 0.38},
    ]


# ---------------------------------------------------------------------------
# 1. Scanare și detecție corectă (NBA & alte sporturi)
# ---------------------------------------------------------------------------

class TestSportsScanDetection:
    """Detecția sportului din payload-ul Gamma + rutarea corectă în pipeline-ul de scanare:
    un meci NBA (2 outcome-uri) e clasificat MECI și parsat 1X2 FĂRĂ leg de egal; un meci
    de tenis în format binar Yes/No cade curat pe fluxul moștenit YES/NO."""

    def test_nba_preseason_detected_as_basketball_binary(self):
        from src.ai_analyser import _detect_sport
        title = _nba_gamma_market()["question"]
        assert _detect_sport(title) == ("Baschet", False)   # baschet, FĂRĂ egal

    def test_wta_tennis_detected_as_binary(self):
        from src.ai_analyser import _detect_sport
        assert _detect_sport("WTA Montreal: Iga Swiatek vs Aryna Sabalenka") == \
            ("Tenis/Ping-Pong", False)

    def test_nba_market_classified_as_match(self):
        import main
        market = _nba_gamma_market()
        assert main._classify_market(market["question"], market["slug"]) == "match"
        assert main._is_match_market(market) is True        # prioritizat în coada de scanare

    def test_nba_market_parses_1x2_without_draw_leg(self):
        # Structura reală NBA (moneyline, 2 outcome-uri): parse_match_market construiește
        # DOAR legs 1 și 2 — egalul (X) nu există, deci nu poate fi nici măcar candidat.
        from src.football import parse_match_market
        parsed = parse_match_market(_nba_gamma_market())
        assert parsed is not None
        assert parsed["is_three_way"] is False
        assert "X" not in parsed["by_code"]
        assert parsed["by_code"]["1"]["token_id"] == "0xLAKERS"
        assert parsed["by_code"]["2"]["token_id"] == "0xGSW"
        assert parsed["label"] == "Lakers vs Warriors"

    def test_nba_prepare_market_yields_match_item_with_eligible_legs(self):
        # Fereastra [0.45, 0.95]: Lakers (0.62) e eligibil, Warriors (0.38) nu → un
        # singur candidat trimis către AI, cu token-ul corect.
        import main
        state = {"pusd_balance": 20.0, "active_positions": []}
        with patch("main.fetch_free_news", return_value=[]):
            item = main._prepare_market(_nba_gamma_market(), state)
        assert item is not None and item["kind"] == "match"
        assert [c["code"] for c in item["candidates"]] == ["1"]
        assert item["candidates"][0]["token_id"] == "0xLAKERS"

    def test_wta_binary_market_falls_back_to_yes_no_flow(self):
        # Tenisul e reprezentat binar (Yes/No) → parse_match_market întoarce None și
        # piața curge prin fluxul moștenit YES/NO (kind='binary'), tot tranzacționabilă.
        import main
        state = {"pusd_balance": 20.0, "active_positions": []}
        with patch("main.fetch_free_news", return_value=[]):
            item = main._prepare_market(_wta_gamma_market(), state)
        assert item is not None and item["kind"] == "binary"
        assert item["yes_price"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 2. Prompturile generate pe fluxul de BUY + promptul sport-aware
# ---------------------------------------------------------------------------

class TestSportsBuyPromptGeneration:
    """Prompturile de BUY pentru meciuri (_build_match_prompt / _build_batch_prompt) sunt
    SPORT-AWARE: _match_sport_context injectează sportul detectat din titlul pieței
    (_detect_sport) și regula de egal derivată din STRUCTURA reală (is_three_way) — un
    meci NBA/tenis (2 outcome-uri) primește mesajul explicit că egalul (X) NU e o opțiune
    de pariere, un 1X2 de fotbal păstrează egalul ca risc real. Garanția rămâne dublată
    structural: X nu e printre codurile permise, iar un pick „X" e anulat de validare."""

    NO_DRAW_MSG = ("Egalul (X) NU este o opțiune validă de pariere. "
                   "Analizează dinamica specifică acestui sport pentru decizia de BUY.]")

    def test_match_prompt_nba_offers_only_binary_codes(self):
        from src import ai_analyser
        p = ai_analyser._build_match_prompt(
            "Los Angeles Lakers", "Golden State Warriors", _nba_candidates(),
            ["Preseason opener tonight"],
            market_title=_nba_gamma_market()["question"], is_three_way=False,
        )
        assert '"1"' in p and '"2"' in p                    # ambele echipe prezentate
        assert '"X"' not in p                               # egalul NU e un cod ofertat
        assert "egal (draw)" not in p                       # nici eticheta lui
        assert "Lakers" in p and "Warriors" in p
        assert "62.0%" in p and "38.0%" in p                # probabilitățile implicite
        # Contextul sportiv injectat: sportul detectat + mesajul explicit „fără egal".
        assert "[Context Sportiv: Acest meci aparține sportului 'Baschet'." in p
        assert self.NO_DRAW_MSG in p

    def test_match_prompt_soccer_three_way_keeps_draw_as_risk(self):
        # Un 1X2 de fotbal (are leg X în piață) NU primește mesajul „egalul nu există" —
        # chiar dacă X e în afara ferestrei de probabilitate și lipsește dintre candidați,
        # egalul rămâne un REZULTAT posibil al meciului (risc real pentru share-ul „1").
        from src import ai_analyser
        p = ai_analyser._build_match_prompt(
            "Arsenal", "Chelsea",
            [{"code": "1", "name": "Arsenal", "price": 0.55}], [],
            market_title="Premier League: Arsenal vs Chelsea", is_three_way=True,
        )
        assert "[Context Sportiv: Acest meci aparține sportului 'Fotbal'." in p
        assert "NU este o opțiune validă" not in p
        assert "rezultat posibil" in p                      # egalul semnalat ca risc real

    def test_match_prompt_structural_fallback_labels_football(self):
        # Meci three-way FĂRĂ cuvinte-cheie de sport în titlu → fallback-ul structural
        # din _detect_sport (structura 1X2 → Fotbal), refolosit de _match_sport_context.
        from src import ai_analyser
        p = ai_analyser._build_match_prompt(
            "FC Astra", "FC Farul",
            [{"code": "1", "name": "FC Astra", "price": 0.50}], [],
            market_title="FC Astra vs FC Farul", is_three_way=True,
        )
        assert "sportului 'Fotbal'" in p

    def test_batch_prompt_nba_allowed_codes_exclude_draw(self):
        from src import ai_analyser
        req = {"market_id": "nba-pre-1", "kind": "match", "title": "Lakers vs Warriors",
               "market_title": _nba_gamma_market()["question"], "is_three_way": False,
               "candidates": _nba_candidates(), "news": []}
        p = ai_analyser._build_batch_prompt([req])
        assert "TYPE=MATCH" in p
        assert 'allowed outcome: "1", "2", or "NONE"' in p  # X absent din codurile permise
        assert '"X"' not in p
        assert "[Context Sportiv: Acest meci aparține sportului 'Baschet'." in p
        assert self.NO_DRAW_MSG in p

    def test_batch_prompt_differentiates_sports_per_market(self):
        # În același lot, FIECARE meci primește eticheta LUI de sport, ca modelul să
        # diferențieze un meci NBA Preseason de unul de tenis WTA.
        from src import ai_analyser
        reqs = [
            {"market_id": "nba-pre-1", "kind": "match", "title": "Lakers vs Warriors",
             "market_title": "NBA Preseason: Los Angeles Lakers vs Golden State Warriors",
             "is_three_way": False, "candidates": _nba_candidates(), "news": []},
            {"market_id": "wta-mtl-1", "kind": "match", "title": "Swiatek vs Sabalenka",
             "market_title": "WTA Montreal: Iga Swiatek vs Aryna Sabalenka",
             "is_three_way": False,
             "candidates": [{"code": "1", "name": "Swiatek", "price": 0.55},
                            {"code": "2", "name": "Sabalenka", "price": 0.45}],
             "news": []},
        ]
        p = ai_analyser._build_batch_prompt(reqs)
        assert "sportului 'Baschet'" in p
        assert "sportului 'Tenis/Ping-Pong'" in p
        assert p.count("[Context Sportiv:") == 2            # o etichetă per meci

    def test_analyse_match_rejects_draw_pick_on_binary_sport(self):
        # Plasă de siguranță: chiar dacă modelul ar halucina un egal la NBA, codul „X"
        # nu e printre candidați → trade anulat, nu propagat spre execuție.
        from src import ai_analyser
        from config.settings import settings
        raw = '{"trade": true, "outcome": "X", "confidence": 0.9, "reason": "expects draw"}'
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch("src.ai_analyser._call_groq", return_value=raw):
            d = ai_analyser.analyse_match(
                "Lakers", "Warriors", _nba_candidates(), [],
                market_title=_nba_gamma_market()["question"], is_three_way=False,
            )
        assert d["trade"] is False
        assert d["outcome"] == "NONE"
        assert d["confidence"] == pytest.approx(0.0)

    def test_exit_prompt_contains_sport_name_and_no_draw_rule(self):
        # Promptul universal (Panic Sell) pentru poziția NBA: sportul detectat („Baschet")
        # e injectat în context, cu instrucțiunea EXPLICITĂ că egalul nu există.
        from src import ai_analyser
        p = ai_analyser._build_position_exit_prompt(
            _nba_gamma_market()["question"], "1 (Lakers)", 0.62, 0.55, [],
        )
        assert "Sport: Baschet" in p
        assert "egalul (X) NU există" in p                  # nota sportului binar
        assert "PERMITE egal" not in p                      # nota sportului cu remiză, absentă


# ---------------------------------------------------------------------------
# 3. Parsarea deciziei de BUY în stil OpenRouter (booleene ca STRING)
# ---------------------------------------------------------------------------

class TestSportsBuyDecisionParsing:
    """Fallback-ul OpenRouter nu are JSON Mode → `trade` poate veni ca STRING. `_parse_bool`
    trebuie să accepte '"true"' (permite execuția) și să respingă '"false"' (bool("false")
    e True în Python — fără gardă am fi plasat un BUY real pe un artefact de parsare)."""

    def _analyse_market_raw(self, raw):
        from src import ai_analyser
        from config.settings import settings
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch("src.ai_analyser._call_groq", return_value=raw):
            return ai_analyser.analyse_market(
                "NBA Preseason: Los Angeles Lakers vs Golden State Warriors", 0.62, [],
            )

    def test_parse_bool_openrouter_string_forms(self):
        from src.ai_analyser import _parse_bool
        assert _parse_bool("true") is True      # string „true" → True → execuția poate continua
        assert _parse_bool("false") is False    # bool("false") ar fi fost True — bugul evitat

    def test_openrouter_style_payload_string_true_trades(self):
        # Payload-ul exact din cerință (fără cheia „outcome"): trade '"true"' ca STRING
        # e convertit corect la True, confidence și reason ajung intacte în decizie.
        d = self._analyse_market_raw(
            '{"trade": "true", "confidence": 0.88, '
            '"reason": "Lakers joacă cu prima echipă în prima jumătate."}'
        )
        assert d["trade"] is True
        assert d["confidence"] == pytest.approx(0.88)
        assert d["reason"] == "Lakers joacă cu prima echipă în prima jumătate."
        # Fără „outcome" în răspuns, default-ul documentat e „NO" (valid, nu anulează trade-ul).
        assert d["outcome"] == "NO"

    def test_string_true_with_outcome_yes_reaches_execution_shape(self):
        d = self._analyse_market_raw(
            '{"trade": "true", "outcome": "YES", "confidence": 0.88, "reason": "r"}'
        )
        assert d == {"trade": True, "outcome": "YES",
                     "confidence": pytest.approx(0.88), "reason": "r"}

    def test_string_false_never_trades(self):
        d = self._analyse_market_raw(
            '{"trade": "false", "outcome": "YES", "confidence": 0.97, "reason": "r"}'
        )
        assert d["trade"] is False

    def test_analyse_match_string_true_accepted_for_nba_leg(self):
        from src import ai_analyser
        from config.settings import settings
        raw = '{"trade": "true", "outcome": "1", "confidence": 0.88, "reason": "r"}'
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch("src.ai_analyser._call_groq", return_value=raw):
            d = ai_analyser.analyse_match("Lakers", "Warriors", _nba_candidates(), [])
        assert d["trade"] is True and d["outcome"] == "1"


# ---------------------------------------------------------------------------
# 4. Fluxul complet (end-to-end mockuit): piață NBA → BUY → baseline curat → history
# ---------------------------------------------------------------------------

class TestNbaEndToEndBuyFlow:
    """Ciclul complet pe un meci NBA, cu parserul REAL peste un răspuns LLM brut în stil
    OpenRouter (booleene ca string): scanare → decizie BUY → _execute_trade → baseline-ul
    stal e șters din _last_position_price (igiena verificată la auditul Panic Sell) →
    cooldown-ul și decizia AI de intrare sunt persistate în history.json."""

    NBA_ID = "nba-pre-1"
    LAKERS_TOKEN = "0xLAKERS"
    RAW_BUY = ('{"trade": "true", "outcome": "1", "confidence": 0.88, '
               '"reason": "Lakers joacă cu prima echipă în prima jumătate."}')

    def _fresh_main(self, monkeypatch, tmp_path):
        """Stare izolată: history.json pe tmp, memorie goală și un baseline STAL pe token-ul
        Lakers (rest de la o deținere anterioară, vândută între timp la 0.90)."""
        import main
        hist = tmp_path / "history.json"
        monkeypatch.setattr(main, "_HISTORY_FILE", str(hist))
        monkeypatch.setattr(main, "_analyzed_history", {})
        monkeypatch.setattr(main, "_last_ai_decision", {})
        monkeypatch.setattr(main, "_last_position_price", {self.LAKERS_TOKEN: 0.90})
        return main, hist

    def test_full_cycle_single_analysis_path(self, monkeypatch, tmp_path):
        from config.settings import settings
        from src.execution import Side
        main, hist = self._fresh_main(monkeypatch, tmp_path)
        state = {"pusd_balance": 20.0, "active_positions": []}

        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch("src.ai_analyser._call_groq", return_value=self.RAW_BUY) as mock_groq, \
             patch("main.fetch_free_news", return_value=["Lakers full squad tonight"]), \
             patch("main._live_best_ask", return_value=None), \
             patch("main.notify_trade") as mock_notify, \
             patch("main.place_limit_order",
                   return_value={"success": True, "order_id": "oid-nba"}) as mock_order:
            main.process_market(_nba_gamma_market(self.NBA_ID), state)

        # Promptul trimis EFECTIV la LLM conține contextul sportiv — titlul complet și
        # structura meciului au fost propagate corect din main (_analyse_single).
        prompt_sent = mock_groq.call_args.args[1]
        assert "sportului 'Baschet'" in prompt_sent
        assert "Egalul (X) NU este o opțiune validă de pariere." in prompt_sent

        # Ordinul BUY a plecat pe leg-ul „1" (Lakers), la prețul Gamma, în ținta de $5.
        mock_order.assert_called_once()
        kw = mock_order.call_args.kwargs
        assert kw["token_id"] == self.LAKERS_TOKEN
        assert kw["side"] == Side.BUY
        assert kw["price"] == pytest.approx(0.62)
        assert kw["size"] * kw["price"] <= 5.0 + 1e-9

        # Starea locală actualizată imediat (dedup/overspend în aceeași iterație).
        assert len(state["active_positions"]) == 1
        pos = state["active_positions"][0]
        assert pos["token_id"] == self.LAKERS_TOKEN
        assert pos["outcome"] == "1 (Lakers)"
        assert pos["event_id"] == "nba-preseason-lal-gsw"
        assert state["pusd_balance"] < 20.0

        # Igiena baseline-ului: BUY-ul șterge prețul STAL al deținerii anterioare, ca
        # primul heartbeat Panic Sell să nu compare bid-ul cu 0.90 (trigger AI fals).
        assert self.LAKERS_TOKEN not in main._last_position_price

        # Persistență: cooldown-ul de analiză ȘI teza de intrare, în formatul „învelit".
        data = json.loads(hist.read_text(encoding="utf-8"))
        assert self.NBA_ID in data["analyzed"]
        dec = data["last_ai_decisions"][self.LAKERS_TOKEN]
        assert dec["action"] == "BUY (intrare)"
        assert dec["confidence"] == pytest.approx(0.88)
        assert "prima echipă" in dec["reason"]

        # Notificarea Discord de BUY a fost declanșată.
        mock_notify.assert_called_once()
        assert mock_notify.call_args.args[0] == "BUY"

    def test_full_cycle_batch_path(self, monkeypatch, tmp_path):
        # Calea IMPLICITĂ de producție (batch): analyse_batch real peste un array JSON brut
        # cu trade ca STRING → aceleași efecte (ordin, baseline curat, history persistat).
        from config.settings import settings
        main, hist = self._fresh_main(monkeypatch, tmp_path)
        state = {"pusd_balance": 20.0, "active_positions": []}
        raw = ('[{"market_id": "nba-pre-1", "trade": "true", "outcome": "1", '
               '"confidence": 0.88, "reason": "Lakers cu prima echipă"}]')

        with patch.object(main, "_BATCH_SIZE", 5), \
             patch.object(main, "_INTER_ANALYSIS_DELAY", 0), \
             patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch("src.ai_analyser._call_groq", return_value=raw) as mock_groq, \
             patch("main.fetch_free_news", return_value=[]), \
             patch("main._live_best_ask", return_value=None), \
             patch("main.notify_trade"), \
             patch("main.place_limit_order",
                   return_value={"success": True, "order_id": "oid-b"}) as mock_order:
            main._scan_markets([_nba_gamma_market(self.NBA_ID)], state)

        # Contextul sportiv per meci ajunge și în promptul de BATCH (via _batch_request).
        prompt_sent = mock_groq.call_args.args[1]
        assert "sportului 'Baschet'" in prompt_sent
        assert "Egalul (X) NU este o opțiune validă de pariere." in prompt_sent

        mock_order.assert_called_once()
        assert mock_order.call_args.kwargs["token_id"] == self.LAKERS_TOKEN
        assert len(state["active_positions"]) == 1
        assert self.LAKERS_TOKEN not in main._last_position_price
        data = json.loads(hist.read_text(encoding="utf-8"))
        assert self.NBA_ID in data["analyzed"]
        assert data["last_ai_decisions"][self.LAKERS_TOKEN]["action"] == "BUY (intrare)"

    def test_string_false_from_fallback_blocks_buy_but_records_cooldown(self, monkeypatch, tmp_path):
        # Inversul de siguranță: '"trade": "false"' ca STRING (OpenRouter fără JSON Mode)
        # NU plasează ordin (bool("false") ar fi fost True), dar piața tot intră în
        # cooldown-ul de 4h; baseline-ul stal rămâne neatins (nu a existat BUY).
        from config.settings import settings
        main, hist = self._fresh_main(monkeypatch, tmp_path)
        state = {"pusd_balance": 20.0, "active_positions": []}
        raw = ('[{"market_id": "nba-pre-1", "trade": "false", "outcome": "1", '
               '"confidence": 0.97, "reason": "no edge"}]')

        with patch.object(main, "_BATCH_SIZE", 5), \
             patch.object(main, "_INTER_ANALYSIS_DELAY", 0), \
             patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch("src.ai_analyser._call_groq", return_value=raw), \
             patch("main.fetch_free_news", return_value=[]), \
             patch("main._live_best_ask", return_value=None), \
             patch("main.place_limit_order") as mock_order:
            main._scan_markets([_nba_gamma_market(self.NBA_ID)], state)

        mock_order.assert_not_called()
        assert state["active_positions"] == []
        assert state["pusd_balance"] == pytest.approx(20.0)
        assert main._last_position_price == {self.LAKERS_TOKEN: 0.90}
        data = json.loads(hist.read_text(encoding="utf-8"))
        assert self.NBA_ID in data["analyzed"]      # analizată → cooldown 4h înregistrat
