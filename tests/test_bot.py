"""
Unit tests for the Polymarket trading bot.

Covers pure-logic functions and mocked API interactions.
All external dependencies (py_clob_client_v2, openai, web3, httpx, …) are
mocked in conftest.py so no production environment is required.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

# Helpers are imported from main so tests exercise the real production code
# (conftest.py mocks all heavy dependencies before this import).
from main import _token_for_outcome, _current_price, _extract_event_id


# ---------------------------------------------------------------------------
# Tests: _token_for_outcome
# ---------------------------------------------------------------------------

class TestTokenForOutcome:
    def test_json_string_yes(self):
        market = {"clobTokenIds": json.dumps(["0xAAA", "0xBBB"])}
        assert _token_for_outcome(market, "YES") == "0xAAA"

    def test_json_string_no(self):
        market = {"clobTokenIds": json.dumps(["0xAAA", "0xBBB"])}
        assert _token_for_outcome(market, "NO") == "0xBBB"

    def test_already_list_yes(self):
        assert _token_for_outcome({"clobTokenIds": ["0xCCC", "0xDDD"]}, "YES") == "0xCCC"

    def test_already_list_no(self):
        assert _token_for_outcome({"clobTokenIds": ["0xCCC", "0xDDD"]}, "NO") == "0xDDD"

    def test_fallback_tokens_yes(self):
        market = {"tokens": [{"outcome": "Yes", "token_id": "0xY"}, {"outcome": "No", "token_id": "0xN"}]}
        assert _token_for_outcome(market, "YES") == "0xY"

    def test_fallback_tokens_no(self):
        market = {"tokens": [{"outcome": "Yes", "token_id": "0xY"}, {"outcome": "No", "token_id": "0xN"}]}
        assert _token_for_outcome(market, "NO") == "0xN"

    def test_not_found_returns_none(self):
        assert _token_for_outcome({}, "YES") is None

    def test_malformed_json_returns_none(self):
        assert _token_for_outcome({"clobTokenIds": "not-json"}, "YES") is None

    def test_case_insensitive_yes(self):
        market = {"clobTokenIds": json.dumps(["0xAAA", "0xBBB"])}
        assert _token_for_outcome(market, "yes") == "0xAAA"

    def test_case_insensitive_no(self):
        market = {"clobTokenIds": json.dumps(["0xAAA", "0xBBB"])}
        assert _token_for_outcome(market, "No") == "0xBBB"

    def test_single_token_no_returns_none(self):
        assert _token_for_outcome({"clobTokenIds": json.dumps(["0xAAA"])}, "NO") is None


# ---------------------------------------------------------------------------
# Tests: _current_price
# ---------------------------------------------------------------------------

class TestCurrentPrice:
    def test_yes_from_outcome_prices_list(self):
        assert _current_price({"outcomePrices": ["0.72", "0.28"]}, "YES") == pytest.approx(0.72)

    def test_no_from_outcome_prices_list(self):
        assert _current_price({"outcomePrices": ["0.72", "0.28"]}, "NO") == pytest.approx(0.28)

    def test_yes_from_json_string(self):
        # Gamma API returns outcomePrices as a JSON-encoded string
        assert _current_price({"outcomePrices": '["0.72", "0.28"]'}, "YES") == pytest.approx(0.72)

    def test_no_from_json_string(self):
        assert _current_price({"outcomePrices": '["0.72", "0.28"]'}, "NO") == pytest.approx(0.28)

    def test_malformed_json_string_falls_back_to_best_ask(self):
        assert _current_price({"outcomePrices": "not-json", "bestAsk": "0.61"}, "YES") == pytest.approx(0.61)

    def test_malformed_json_string_falls_back_to_0_5(self):
        assert _current_price({"outcomePrices": "not-json"}, "YES") == pytest.approx(0.5)

    def test_fallback_to_best_ask(self):
        assert _current_price({"bestAsk": "0.55"}, "YES") == pytest.approx(0.55)

    def test_fallback_to_0_5_when_empty(self):
        assert _current_price({}, "YES") == pytest.approx(0.5)

    def test_zero_price_returned(self):
        assert _current_price({"outcomePrices": ["0.0", "1.0"]}, "YES") == pytest.approx(0.0)

    def test_invalid_price_string_falls_back(self):
        assert _current_price({"outcomePrices": ["invalid"]}, "YES") == pytest.approx(0.5)

    def test_invalid_price_string_falls_back_to_best_ask(self):
        assert _current_price({"outcomePrices": ["invalid"], "bestAsk": "0.4"}, "YES") == pytest.approx(0.4)

    def test_no_outcome_prices_only_one_element_falls_back(self):
        assert _current_price({"outcomePrices": ["0.6"]}, "NO") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: _extract_event_id
# ---------------------------------------------------------------------------

class TestExtractEventId:
    def test_prefers_events_slug(self):
        market = {"events": [{"slug": "world-cup-2026", "id": "123"}], "conditionId": "0xC"}
        assert _extract_event_id(market) == "world-cup-2026"

    def test_falls_back_to_events_id(self):
        market = {"events": [{"id": "123"}], "conditionId": "0xC"}
        assert _extract_event_id(market) == "123"

    def test_falls_back_to_condition_id(self):
        assert _extract_event_id({"conditionId": "0xC"}) == "0xC"

    def test_empty_when_nothing_available(self):
        assert _extract_event_id({}) == ""


# ---------------------------------------------------------------------------
# Tests: dynamic size calculation and tick size in execution.py
# ---------------------------------------------------------------------------

class TestDynamicSizeCalculation:
    """place_limit_order computes size = max($5/price, 2.0) rounded to 2dp."""

    def _call(self, price, mock_result=None, **kwargs):
        from src.execution import place_limit_order
        from src.execution import Side

        if mock_result is None:
            mock_result = {"orderID": "test-id-123"}

        mock_client = MagicMock()
        mock_client.create_and_post_order.return_value = mock_result

        with patch("src.execution.get_clob_client", return_value=mock_client):
            result = place_limit_order(token_id="0xTOKEN", price=price, side=Side.BUY, **kwargs)

        return result, mock_client

    def _placed_size(self, price):
        _, client = self._call(price)
        return client.create_and_post_order.call_args.kwargs["order_args"].size

    def test_price_0_50_gives_size_10(self):
        assert self._placed_size(0.50) == pytest.approx(10.0)

    def test_price_0_90_gives_approx_5_56(self):
        assert self._placed_size(0.90) == pytest.approx(5.56, abs=0.01)

    def test_price_0_25_gives_size_20(self):
        assert self._placed_size(0.25) == pytest.approx(20.0)

    def test_floor_applied_at_high_price(self):
        # $5 / 0.99 ≈ 5.05 — above floor
        assert self._placed_size(0.99) >= 2.0

    def test_floor_of_2_when_formula_lower(self):
        size = self._placed_size(0.98)
        assert size >= 2.0

    def test_invalid_price_zero_returns_error(self):
        from src.execution import place_limit_order
        from src.execution import Side
        result = place_limit_order(token_id="0xT", price=0.0, side=Side.BUY)
        assert result["success"] is False
        assert "range" in result["error"]

    def test_price_above_0_99_returns_error(self):
        from src.execution import place_limit_order
        from src.execution import Side
        result = place_limit_order(token_id="0xT", price=1.01, side=Side.BUY)
        assert result["success"] is False

    def test_success_result_returned(self):
        result, _ = self._call(0.5, mock_result={"orderID": "abc-123"})
        assert result["success"] is True
        assert result["order_id"] == "abc-123"

    def test_api_exception_returns_failure(self):
        from src.execution import place_limit_order
        from src.execution import Side

        mock_client = MagicMock()
        mock_client.create_and_post_order.side_effect = Exception("500 server error")

        with patch("src.execution.get_clob_client", return_value=mock_client):
            result = place_limit_order(token_id="0xT", price=0.5, side=Side.BUY)

        assert result["success"] is False
        assert "500 server error" in result["error"]

    def test_size_is_rounded_to_2_decimal_places(self):
        size = self._placed_size(0.33)  # 5/0.33 = 15.1515...
        assert size == round(size, 2)

    def test_compute_order_size_matches_placed_size(self):
        from src.execution import compute_order_size
        assert compute_order_size(0.50) == pytest.approx(10.0)
        assert compute_order_size(0.25) == pytest.approx(20.0)

    # --- Tick size handling ---

    def test_default_tick_rounds_to_2_decimals(self):
        _, client = self._call(0.123)
        assert client.create_and_post_order.call_args.kwargs["order_args"].price == pytest.approx(0.12)

    def test_tick_0_001_keeps_3_decimals(self):
        _, client = self._call(0.123, tick_size="0.001")
        assert client.create_and_post_order.call_args.kwargs["order_args"].price == pytest.approx(0.123)

    def test_tick_0_001_allows_price_0_995(self):
        result, _ = self._call(0.995, tick_size="0.001")
        assert result["success"] is True

    def test_default_tick_rejects_price_0_995(self):
        # round(0.995, 2) → 0.99 (valid); but 0.999 rounds to 1.0 → rejected
        result, _ = self._call(0.999)
        assert result["success"] is False

    def test_unknown_tick_size_falls_back_to_default(self):
        _, client = self._call(0.123, tick_size="0.5")
        assert client.create_and_post_order.call_args.kwargs["order_args"].price == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Tests: explicit size for SELL orders (position exits)
# ---------------------------------------------------------------------------

class TestExplicitOrderSize:
    def _call(self, price, size, side_name="SELL"):
        from src.execution import place_limit_order
        from src.execution import Side

        mock_client = MagicMock()
        mock_client.create_and_post_order.return_value = {"orderID": "x"}

        side = Side.SELL if side_name == "SELL" else Side.BUY
        with patch("src.execution.get_clob_client", return_value=mock_client):
            result = place_limit_order(token_id="0xT", price=price, side=side, size=size)
        return result, mock_client

    def test_explicit_size_used_verbatim(self):
        _, client = self._call(0.5, size=7.5)
        assert client.create_and_post_order.call_args.kwargs["order_args"].size == pytest.approx(7.5)

    def test_explicit_size_rounded_to_2dp(self):
        _, client = self._call(0.5, size=7.123)
        assert client.create_and_post_order.call_args.kwargs["order_args"].size == pytest.approx(7.12)

    def test_zero_size_rejected(self):
        result, _ = self._call(0.5, size=0.0)
        assert result["success"] is False
        assert "size" in result["error"].lower()


# ---------------------------------------------------------------------------
# Tests: budget-aware dynamic order sizing (compute_order_size(price, budget))
# ---------------------------------------------------------------------------

class TestBudgetAwareSizing:
    """Țintă $5/ordin; dacă balanța (budget) e sub $5, folosește cât mai mult din
    ea fără s-o depășească. Balanța < $1 e blocată de apelant (vezi _execute_trade)."""

    def test_ample_budget_uses_5usd_target(self):
        from src.execution import compute_order_size
        # budget >> $5 → valoare ordin ~$5 (ținta).
        assert compute_order_size(0.50, budget=20.0) * 0.50 == pytest.approx(5.0, abs=0.01)

    def test_low_budget_uses_available_balance(self):
        from src.execution import compute_order_size
        # $1.96 (< $5, >= $1) → folosește ~toată balanța, fără s-o depășească.
        cost = compute_order_size(0.70, budget=1.9611) * 0.70
        assert cost <= 1.9611                          # nu depășește balanța
        assert cost == pytest.approx(1.96, abs=0.02)   # ~toată balanța

    def test_never_exceeds_budget_across_prices(self):
        from src.execution import compute_order_size
        for price in (0.05, 0.33, 0.50, 0.70, 0.90, 0.99):
            for budget in (1.0, 1.9611, 3.5, 4.99):
                assert compute_order_size(price, budget=budget) * price <= budget + 1e-9

    def test_budget_at_target_uses_5usd(self):
        from src.execution import compute_order_size
        assert compute_order_size(0.50, budget=5.0) * 0.50 == pytest.approx(5.0, abs=0.01)

    def test_no_budget_keeps_5usd_target(self):
        from src.execution import compute_order_size
        # Ramura fără buget (apel din place_limit_order) rămâne neschimbată.
        assert compute_order_size(0.50) == pytest.approx(10.0)
        assert compute_order_size(0.25) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Tests: position_manager exit logic (take-profit / stop-loss)
# ---------------------------------------------------------------------------

class TestExitConditions:
    def _state(self, avg_price=0.50, size=10.0, token_id="0xPOS"):
        return {
            "pusd_balance": 5.0,
            "active_positions": [{
                "asset_name": "Test market",
                "token_id": token_id,
                "side": "BUY",
                "size": size,
                "avg_price": avg_price,
            }],
        }

    def _run(self, state, bid, tp=0.0, sl=0.0, order_result=None):
        from src import position_manager
        from config.settings import settings

        position_manager._exit_attempts.clear()
        order_result = order_result or {"success": True, "order_id": "exit-1"}

        with patch.object(settings, "TAKE_PROFIT_PCT", tp), \
             patch.object(settings, "STOP_LOSS_PCT", sl), \
             patch("src.position_manager._best_bid", return_value=bid), \
             patch("src.position_manager.place_limit_order", return_value=order_result) as mock_order:
            position_manager.check_exit_conditions(state)
        return mock_order

    def test_disabled_when_both_thresholds_zero(self):
        mock_order = self._run(self._state(), bid=0.99, tp=0.0, sl=0.0)
        mock_order.assert_not_called()

    def test_take_profit_triggers_sell(self):
        # avg 0.50, bid 0.65 → +30% ≥ 20% TP
        mock_order = self._run(self._state(avg_price=0.50), bid=0.65, tp=0.20)
        mock_order.assert_called_once()
        kwargs = mock_order.call_args.kwargs
        assert kwargs["token_id"] == "0xPOS"
        assert kwargs["price"] == pytest.approx(0.65)
        assert kwargs["size"] == pytest.approx(10.0)

    def test_take_profit_not_reached_no_order(self):
        # avg 0.50, bid 0.55 → +10% < 20% TP
        mock_order = self._run(self._state(avg_price=0.50), bid=0.55, tp=0.20)
        mock_order.assert_not_called()

    def test_stop_loss_triggers_sell(self):
        # avg 0.50, bid 0.35 → -30% ≤ -20% SL
        mock_order = self._run(self._state(avg_price=0.50), bid=0.35, sl=0.20)
        mock_order.assert_called_once()

    def test_stop_loss_not_reached_no_order(self):
        # avg 0.50, bid 0.45 → -10% > -20% SL
        mock_order = self._run(self._state(avg_price=0.50), bid=0.45, sl=0.20)
        mock_order.assert_not_called()

    def test_no_orderbook_data_no_order(self):
        mock_order = self._run(self._state(), bid=None, tp=0.20, sl=0.20)
        mock_order.assert_not_called()

    def test_tiny_position_below_min_value_skipped(self):
        # 1 share * $0.65 = $0.65 < $1 minimum
        mock_order = self._run(self._state(avg_price=0.50, size=1.0), bid=0.65, tp=0.20)
        mock_order.assert_not_called()

    def test_cooldown_prevents_duplicate_exit(self):
        from src import position_manager
        from config.settings import settings

        position_manager._exit_attempts.clear()
        state = self._state(avg_price=0.50)

        with patch.object(settings, "TAKE_PROFIT_PCT", 0.20), \
             patch.object(settings, "STOP_LOSS_PCT", 0.0), \
             patch("src.position_manager._best_bid", return_value=0.65), \
             patch("src.position_manager.place_limit_order",
                   return_value={"success": True, "order_id": "x"}) as mock_order:
            position_manager.check_exit_conditions(state)
            position_manager.check_exit_conditions(state)

        assert mock_order.call_count == 1


# ---------------------------------------------------------------------------
# Tests: process_market state updates (balance decrement, dedup, clustering)
# ---------------------------------------------------------------------------

class TestProcessMarketStateTracking:
    def _market(self, market_id="m1", token_ids=("0xYES", "0xDRAW", "0xNO"),
                outcomes=("Arsenal", "Draw", "Chelsea"),
                prices=("0.70", "0.20", "0.10"), event_slug="ev-1"):
        # Meci 1X2: outcome '1' (Arsenal) → clobTokenIds[0] = '0xYES', preț 0.70.
        return {
            "id": market_id,
            "question": "Arsenal vs Chelsea",
            "volume": "50000",
            "outcomes": json.dumps(list(outcomes)),
            "outcomePrices": json.dumps(list(prices)),
            "clobTokenIds": json.dumps(list(token_ids)),
            "events": [{"slug": event_slug}],
        }

    def _run(self, market, account_state, decision=None, order_result=None, live_ask=None):
        import main
        # Decizia 1X2 implicită: pariem '1' (gazda, Arsenal).
        decision = decision or {"trade": True, "outcome": "1", "confidence": 0.9, "reason": "test"}
        order_result = order_result or {"success": True, "order_id": "oid-1"}

        # _record_analyzed is patched out so process_market does not touch history.json;
        # history persistence is covered separately by TestAnalyzedHistory.
        with patch("main.fetch_free_news", return_value=[]), \
             patch("main.analyse_match", return_value=decision), \
             patch("main._record_analyzed"), \
             patch("main._live_best_ask", return_value=live_ask), \
             patch("main.place_limit_order", return_value=order_result) as mock_order:
            main.process_market(market, account_state)
        return mock_order

    def test_successful_order_decrements_balance(self):
        state = {"pusd_balance": 20.0, "active_positions": []}
        self._run(self._market(), state)
        # size = 5/0.7 ≈ 7.14, total ≈ 5.0
        assert state["pusd_balance"] < 20.0

    def test_successful_order_appends_position(self):
        state = {"pusd_balance": 20.0, "active_positions": []}
        self._run(self._market(), state)
        assert len(state["active_positions"]) == 1
        assert state["active_positions"][0]["token_id"] == "0xYES"
        assert state["active_positions"][0]["event_id"] == "ev-1"

    def test_failed_order_does_not_change_state(self):
        state = {"pusd_balance": 20.0, "active_positions": []}
        self._run(self._market(), state, order_result={"success": False, "error": "rejected"})
        assert state["pusd_balance"] == pytest.approx(20.0)
        assert state["active_positions"] == []

    def test_duplicate_token_skipped(self):
        state = {
            "pusd_balance": 20.0,
            "active_positions": [{"token_id": "0xYES", "event_id": "other"}],
        }
        mock_order = self._run(self._market(), state)
        mock_order.assert_not_called()

    def test_max_active_positions_blocks_new_orders(self):
        from config.settings import settings
        positions = [{"token_id": f"0x{i}", "event_id": ""} for i in range(settings.MAX_ACTIVE_POSITIONS)]
        state = {"pusd_balance": 20.0, "active_positions": positions}
        mock_order = self._run(self._market(), state)
        mock_order.assert_not_called()

    def test_low_volume_market_skipped(self):
        market = self._market()
        market["volume"] = "500"
        state = {"pusd_balance": 20.0, "active_positions": []}
        mock_order = self._run(market, state)
        mock_order.assert_not_called()

    def test_probability_window_enforced_with_json_string_prices(self):
        # Leg '1' (Arsenal)=0.95 sits on the upper bound of the [0.45, 0.95]
        # window (exclusive) and the other legs are below 0.45 → no eligible 1X2
        # outcome → filtered. Prices must be read from the JSON string.
        market = self._market(prices=("0.95", "0.03", "0.02"))
        state = {"pusd_balance": 20.0, "active_positions": []}
        mock_order = self._run(market, state)
        mock_order.assert_not_called()

    def test_insufficient_balance_blocks_order(self):
        # < $1 → blocat (minim absolut Polymarket).
        state = {"pusd_balance": 0.5, "active_positions": []}
        mock_order = self._run(self._market(), state)
        mock_order.assert_not_called()

    def test_low_balance_places_dynamic_order(self):
        # $1.96 la preț 0.70: safe_balance ≈ 1.90 → 2.71 acțiuni, valoare ordin ≈ $1.90.
        # Sub vechea regulă de „minim 5 share-uri" era sărit; acum, fiind > 1 acțiune
        # și peste $1.00, ordinul DINAMIC se plasează cu cantitatea calculată.
        state = {"pusd_balance": 1.9611, "active_positions": []}
        mock_order = self._run(self._market(), state)
        mock_order.assert_called_once()
        kwargs = mock_order.call_args.kwargs
        assert kwargs["size"] >= 1.0                                    # cel puțin o acțiune
        assert kwargs["size"] * kwargs["price"] >= 1.0                  # peste minimul de $1.00
        assert kwargs["size"] * kwargs["price"] <= 1.9611 * 0.97 + 1e-9  # nu depășește balanța sigură
        assert state["pusd_balance"] < 1.9611                          # balanța decrementată cu costul real
        assert len(state["active_positions"]) == 1

    def test_low_balance_places_buffered_order(self):
        # $4.00 la preț 0.70: safe_balance = 3.88 → ordin mic (sub ținta de $5),
        # dimensionat dinamic în limita balanței SIGURE (buffer 3%).
        state = {"pusd_balance": 4.0, "active_positions": []}
        mock_order = self._run(self._market(), state)
        mock_order.assert_called_once()
        kwargs = mock_order.call_args.kwargs
        assert kwargs["size"] >= 1.0                                  # cel puțin o acțiune (fără minim rigid)
        assert kwargs["size"] * kwargs["price"] >= 1.0               # peste minimul de $1.00
        assert kwargs["size"] * kwargs["price"] <= 4.0 * 0.97 + 1e-9  # buffer 3% respectat
        assert state["pusd_balance"] < 4.0                           # balanța decrementată cu costul real

    def test_sub_dollar_order_value_skipped(self):
        # Balanță $1.00 (trece de gate-ul de $1) dar bufferul 3% → safe_balance $0.97.
        # La preț 0.46: 2.10 acțiuni × $0.46 = $0.97 valoare ordin < $1.00 → sărit
        # LOCAL (ar fi respins de API), fără să se trimită cererea.
        market = self._market(prices=("0.46", "0.30", "0.20"))
        state = {"pusd_balance": 1.0, "active_positions": []}
        mock_order = self._run(market, state)
        mock_order.assert_not_called()
        assert state["pusd_balance"] == pytest.approx(1.0)   # balanța neatinsă
        assert state["active_positions"] == []

    def test_price_too_high_for_single_share_skipped(self):
        # Gardă directă pe _execute_trade: la balanță $1.00 (safe $0.97) și preț 0.98,
        # bugetul nu acoperă nici măcar o acțiune întreagă (0.98 < 1) → sărit, fără ordin.
        import main
        state = {"pusd_balance": 1.0, "active_positions": []}
        decision = {"trade": True, "outcome": "1", "confidence": 0.9, "reason": "t"}
        with patch("main._live_best_ask", return_value=None), \
             patch("main.place_limit_order") as mock_order:
            main._execute_trade(
                self._market(), state,
                token_id="0xHI", price=0.98, outcome_label="1",
                decision=decision, event_id="", title="High price market",
            )
        mock_order.assert_not_called()
        assert state["pusd_balance"] == pytest.approx(1.0)
        assert state["active_positions"] == []

    # --- Slippage guard ---

    def test_stale_price_with_high_live_ask_blocks_order(self):
        # Gamma price 0.70, live ask 0.80 → +14% > 5% tolerance
        state = {"pusd_balance": 20.0, "active_positions": []}
        mock_order = self._run(self._market(), state, live_ask=0.80)
        mock_order.assert_not_called()

    def test_lower_live_ask_improves_buy_price(self):
        # Gamma price 0.70, live ask 0.68 → buy at 0.68 (better, fills now)
        state = {"pusd_balance": 20.0, "active_positions": []}
        mock_order = self._run(self._market(), state, live_ask=0.68)
        mock_order.assert_called_once()
        assert mock_order.call_args.kwargs["price"] == pytest.approx(0.68)

    def test_no_orderbook_data_falls_back_to_gamma_price(self):
        state = {"pusd_balance": 20.0, "active_positions": []}
        mock_order = self._run(self._market(), state, live_ask=None)
        mock_order.assert_called_once()
        assert mock_order.call_args.kwargs["price"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Tests: reconciliation balance parsing
# ---------------------------------------------------------------------------

class TestBalanceParsing:
    def _get_state(self, bal_data, positions=None):
        from src.reconciliation import get_account_state

        mock_client = MagicMock()
        mock_client.get_balance_allowance.return_value = bal_data

        mock_resp = MagicMock()
        mock_resp.json.return_value = positions or []
        mock_resp.raise_for_status.return_value = None

        with patch("src.reconciliation.get_clob_client", return_value=mock_client), \
             patch("src.reconciliation.requests.get", return_value=mock_resp):
            return get_account_state()

    def test_balance_from_dict(self):
        state = self._get_state({"balance": 15_640_000, "allowance": 100_000_000})
        assert state["pusd_balance"] == pytest.approx(15.64)
        assert state["allowance"] == pytest.approx(100.0)

    def test_balance_from_object_attributes(self):
        bal = MagicMock()
        bal.balance = 5_000_000
        bal.allowance = 50_000_000
        state = self._get_state(bal)
        assert state["pusd_balance"] == pytest.approx(5.0)
        assert state["allowance"] == pytest.approx(50.0)

    def test_zero_balance(self):
        state = self._get_state({"balance": 0, "allowance": 0})
        assert state["pusd_balance"] == pytest.approx(0.0)

    def test_missing_keys_default_to_zero(self):
        state = self._get_state({})
        assert state["pusd_balance"] == pytest.approx(0.0)
        assert state["allowance"] == pytest.approx(0.0)

    def test_active_position_included(self):
        positions = [{"asset": "0xTOK", "size": "3.5", "side": "BUY",
                      "avgPrice": "0.6", "title": "Test market"}]
        state = self._get_state({"balance": 0, "allowance": 0}, positions=positions)
        assert len(state["active_positions"]) == 1
        assert state["active_positions"][0]["size"] == pytest.approx(3.5)
        assert state["active_positions"][0]["token_id"] == "0xTOK"

    def test_position_includes_event_id_from_event_slug(self):
        positions = [{"asset": "0xTOK", "size": "1.0", "side": "BUY",
                      "avgPrice": "0.6", "eventSlug": "world-cup-2026"}]
        state = self._get_state({"balance": 0, "allowance": 0}, positions=positions)
        assert state["active_positions"][0]["event_id"] == "world-cup-2026"

    def test_position_event_id_falls_back_to_condition_id(self):
        positions = [{"asset": "0xTOK", "size": "1.0", "side": "BUY",
                      "avgPrice": "0.6", "conditionId": "0xCOND"}]
        state = self._get_state({"balance": 0, "allowance": 0}, positions=positions)
        assert state["active_positions"][0]["event_id"] == "0xCOND"

    def test_zero_size_position_excluded(self):
        positions = [{"asset": "0xTOK", "size": "0", "side": "BUY", "avgPrice": "0.6"}]
        state = self._get_state({"balance": 0, "allowance": 0}, positions=positions)
        assert len(state["active_positions"]) == 0

    def test_multiple_positions(self):
        positions = [
            {"asset": "0xA", "size": "1.0", "side": "BUY", "avgPrice": "0.5"},
            {"asset": "0xB", "size": "2.0", "side": "BUY", "avgPrice": "0.7"},
        ]
        state = self._get_state({"balance": 0, "allowance": 0}, positions=positions)
        assert len(state["active_positions"]) == 2


# ---------------------------------------------------------------------------
# Tests: settings loading
# ---------------------------------------------------------------------------

class TestSettings:
    def test_signature_type_is_3(self):
        # POLY_1271 (EIP-1271) — tipul de semnătură al fork-ului py-clob-client-v2,
        # singurul confirmat funcțional pentru acest cont proxy.
        from config.settings import settings
        assert settings.SIGNATURE_TYPE == 3

    def test_chain_id_is_polygon(self):
        from config.settings import settings
        assert settings.CHAIN_ID == 137

    def test_clob_host_contains_polymarket(self):
        from config.settings import settings
        assert "polymarket" in settings.CLOB_HOST

    def test_override_balance_defaults_to_zero(self):
        from config.settings import settings
        assert settings.OVERRIDE_BALANCE == pytest.approx(0.0)

    def test_max_active_positions_default_is_5(self):
        from config.settings import settings
        assert settings.MAX_ACTIVE_POSITIONS == 5

    def test_usdc_approval_defaults_to_unlimited(self):
        from config.settings import settings
        assert settings.USDC_APPROVAL_USD == pytest.approx(0.0)

    def test_max_gas_price_default_is_500_gwei(self):
        from config.settings import settings
        assert settings.MAX_GAS_PRICE_GWEI == pytest.approx(500.0)

    def test_take_profit_disabled_by_default(self):
        from config.settings import settings
        assert settings.TAKE_PROFIT_PCT == pytest.approx(0.0)

    def test_stop_loss_default_is_30_pct(self):
        from config.settings import settings
        assert settings.STOP_LOSS_PCT == pytest.approx(0.30)

    def test_min_balance_constant_in_main(self):
        import main
        assert main._MIN_BALANCE_PUSD == pytest.approx(1.0)

    def test_min_confidence_constant_in_main(self):
        import main
        assert main._MIN_CONFIDENCE == pytest.approx(0.70)

    def test_discord_webhook_defaults_empty(self):
        from config.settings import settings
        assert settings.DISCORD_WEBHOOK_URL == ""

    def test_pnl_report_interval_default_is_1h(self):
        from config.settings import settings
        assert settings.PNL_REPORT_INTERVAL_HOURS == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: position P/L enrichment from the Data API
# ---------------------------------------------------------------------------

class TestPositionPnLParsing:
    def _get_state(self, positions):
        from src.reconciliation import get_account_state

        mock_client = MagicMock()
        mock_client.get_balance_allowance.return_value = {"balance": 0, "allowance": 0}

        mock_resp = MagicMock()
        mock_resp.json.return_value = positions
        mock_resp.raise_for_status.return_value = None

        with patch("src.reconciliation.get_clob_client", return_value=mock_client), \
             patch("src.reconciliation.requests.get", return_value=mock_resp):
            return get_account_state()

    def test_pnl_fields_from_data_api(self):
        positions = [{
            "asset": "0xTOK", "size": "10", "side": "BUY", "avgPrice": "0.50",
            "curPrice": "0.65", "currentValue": "6.5", "initialValue": "5.0",
            "cashPnl": "1.5", "percentPnl": "30", "realizedPnl": "0",
        }]
        pos = self._get_state(positions)["active_positions"][0]
        assert pos["cur_price"] == pytest.approx(0.65)
        assert pos["current_value"] == pytest.approx(6.5)
        assert pos["cash_pnl"] == pytest.approx(1.5)
        assert pos["percent_pnl"] == pytest.approx(30.0)

    def test_pnl_falls_back_to_local_computation(self):
        # No P/L fields from API → compute from price × size
        positions = [{"asset": "0xTOK", "size": "10", "side": "BUY",
                      "avgPrice": "0.50", "curPrice": "0.60"}]
        pos = self._get_state(positions)["active_positions"][0]
        assert pos["current_value"] == pytest.approx(6.0)   # 0.60 * 10
        assert pos["initial_value"] == pytest.approx(5.0)   # 0.50 * 10
        assert pos["cash_pnl"] == pytest.approx(1.0)        # 6.0 - 5.0

    def test_cash_pnl_zero_is_respected_not_recomputed(self):
        # cashPnl explicitly 0 must NOT trigger the fallback (falsy-zero bug guard)
        positions = [{"asset": "0xTOK", "size": "10", "side": "BUY",
                      "avgPrice": "0.50", "currentValue": "9.0", "cashPnl": "0"}]
        pos = self._get_state(positions)["active_positions"][0]
        assert pos["cash_pnl"] == pytest.approx(0.0)

    def test_outcome_extracted_from_data_api(self):
        positions = [{"asset": "0xTOK", "size": "10", "side": "BUY",
                      "avgPrice": "0.50", "outcome": "YES"}]
        pos = self._get_state(positions)["active_positions"][0]
        assert pos["outcome"] == "YES"

    def test_outcome_propagates_to_positions_notification(self):
        # End-to-end: Data API outcome="YES" → reconciliation → notify_positions embed.
        from src import notifications
        from config.settings import settings
        positions = [{"asset": "0xTOK", "size": "10", "side": "BUY",
                      "avgPrice": "0.50", "outcome": "YES", "title": "USA vs Iran"}]
        state = self._get_state(positions)
        assert state["active_positions"][0]["outcome"] == "YES"
        with patch.object(settings, "DISCORD_WEBHOOK_URL", "https://discord.test/hook"), \
             patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_positions(state["active_positions"])
        desc = mock_post.call_args.kwargs["json"]["embeds"][0]["description"]
        assert "YES" in desc


# ---------------------------------------------------------------------------
# Tests: portfolio summary aggregation
# ---------------------------------------------------------------------------

class TestPortfolioSummary:
    def test_aggregates_cash_and_positions(self):
        from src.reconciliation import compute_portfolio_summary
        state = {
            "pusd_balance": 3.0,
            "active_positions": [
                {"current_value": 6.5, "initial_value": 5.0, "cash_pnl": 1.5},
                {"current_value": 2.0, "initial_value": 3.0, "cash_pnl": -1.0},
            ],
        }
        s = compute_portfolio_summary(state)
        assert s["cash"] == pytest.approx(3.0)
        assert s["positions_value"] == pytest.approx(8.5)
        assert s["total_value"] == pytest.approx(11.5)
        assert s["unrealized_pnl"] == pytest.approx(0.5)
        assert s["cost_basis"] == pytest.approx(8.0)
        assert s["n_positions"] == 2

    def test_empty_portfolio(self):
        from src.reconciliation import compute_portfolio_summary
        s = compute_portfolio_summary({"pusd_balance": 5.0, "active_positions": []})
        assert s["total_value"] == pytest.approx(5.0)
        assert s["unrealized_pnl"] == pytest.approx(0.0)
        assert s["n_positions"] == 0


# ---------------------------------------------------------------------------
# Tests: Discord notifications
# ---------------------------------------------------------------------------

class TestNotifications:
    def _patch(self, webhook="https://discord.test/webhook"):
        from config.settings import settings
        return patch.object(settings, "DISCORD_WEBHOOK_URL", webhook)

    def test_no_webhook_is_noop(self):
        from src import notifications
        with self._patch(webhook=""), \
             patch("src.notifications.requests.post") as mock_post:
            notifications.notify_trade("BUY", "Market", 0.8, 6.0, 5.0)
            notifications.notify_portfolio({"total_value": 10.0})
        mock_post.assert_not_called()

    def test_buy_notification_payload(self):
        from src import notifications
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_trade(
                "BUY", "Will X win?", 0.81, 6.17, 5.0,
                confidence=0.85, reason="strong news",
            )
        mock_post.assert_called_once()
        embed = mock_post.call_args.kwargs["json"]["embeds"][0]
        assert "BUY" in embed["title"]
        assert embed["color"] == 0x2ECC71  # green
        names = [f["name"] for f in embed["fields"]]
        assert "Preț" in names and "Total" in names and "Confidence" in names

    def test_sell_notification_is_red(self):
        from src import notifications
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_trade("SELL", "Market", 0.4, 10.0, 4.0, reason="stop-loss (-30%)")
        embed = mock_post.call_args.kwargs["json"]["embeds"][0]
        assert "SELL" in embed["title"]
        assert embed["color"] == 0xE74C3C  # red

    def test_portfolio_shows_financial_stats_without_position_list(self):
        from src import notifications
        stats = {
            "total_value": 12.0, "cash": 2.0, "positions_value": 10.0,
            "unrealized_pnl": 1.0, "cost_basis": 9.0, "n_positions": 2,
            "positions": [{"asset_name": "Market A", "current_value": 6.0, "cash_pnl": 1.5}],
        }
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_portfolio(stats, prev_total=11.0, session_start_total=10.0)
        embed = mock_post.call_args.kwargs["json"]["embeds"][0]
        assert "Status Financiar" in embed["title"]
        names = [f["name"] for f in embed["fields"]]
        assert "P/L ultima oră" in names
        assert "P/L de la pornire" in names
        assert "Top poziții" not in names   # positions moved to notify_positions

    def test_network_error_is_swallowed(self):
        from src import notifications
        with self._patch(), \
             patch("src.notifications.requests.post", side_effect=Exception("boom")):
            # Must not raise — trading should never break on a Discord failure
            notifications.notify_trade("BUY", "Market", 0.8, 6.0, 5.0)

    def test_trade_includes_outcome_field(self):
        from src import notifications
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_trade("BUY", "USA vs Iran", 0.5, 10.0, 5.0, outcome="YES")
        embed = mock_post.call_args.kwargs["json"]["embeds"][0]
        vals = {f["name"]: f["value"] for f in embed["fields"]}
        assert vals.get("Outcome") == "YES"

    def test_trade_without_outcome_has_no_outcome_field(self):
        from src import notifications
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_trade("BUY", "Market", 0.5, 10.0, 5.0)
        names = [f["name"] for f in mock_post.call_args.kwargs["json"]["embeds"][0]["fields"]]
        assert "Outcome" not in names


# ---------------------------------------------------------------------------
# Tests: persistent analyzed-markets history (history.json)
# ---------------------------------------------------------------------------

class TestAnalyzedHistory:
    """_load_history / _save_history / _record_analyzed / _is_recently_analyzed."""

    def _fresh_main(self, monkeypatch, tmp_path):
        import importlib, sys
        if "main" not in sys.modules:
            importlib.import_module("main")
        import main
        hist_file = tmp_path / "history.json"
        monkeypatch.setattr(main, "_HISTORY_FILE", str(hist_file))
        monkeypatch.setattr(main, "_analyzed_history", {})
        return main, hist_file

    def test_record_then_recently_analyzed(self, monkeypatch, tmp_path):
        main, hist_file = self._fresh_main(monkeypatch, tmp_path)
        assert main._is_recently_analyzed("m1") is False
        main._record_analyzed("m1")
        assert main._is_recently_analyzed("m1") is True
        assert hist_file.exists()  # persisted to disk

    def test_save_load_roundtrip_survives_restart(self, monkeypatch, tmp_path):
        main, _ = self._fresh_main(monkeypatch, tmp_path)
        main._record_analyzed("m1")
        main._record_analyzed("m2")
        monkeypatch.setattr(main, "_analyzed_history", {})  # simulate process restart
        main._load_history()
        assert main._is_recently_analyzed("m1") is True
        assert main._is_recently_analyzed("m2") is True

    def test_expired_entries_pruned_on_load(self, monkeypatch, tmp_path):
        import json, time
        main, hist_file = self._fresh_main(monkeypatch, tmp_path)
        now = time.time()
        hist_file.write_text(json.dumps({
            "old": now - main._CACHE_TTL_SECONDS - 10,
            "fresh": now,
        }), encoding="utf-8")
        main._load_history()
        assert main._is_recently_analyzed("fresh") is True
        assert main._is_recently_analyzed("old") is False
        assert "old" not in main._analyzed_history

    def test_list_format_supported(self, monkeypatch, tmp_path):
        import json
        main, hist_file = self._fresh_main(monkeypatch, tmp_path)
        hist_file.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        main._load_history()
        assert main._is_recently_analyzed("a") is True
        assert main._is_recently_analyzed("b") is True

    def test_missing_file_starts_empty(self, monkeypatch, tmp_path):
        main, _ = self._fresh_main(monkeypatch, tmp_path)
        main._load_history()
        assert main._analyzed_history == {}

    def test_corrupt_file_starts_empty(self, monkeypatch, tmp_path):
        main, hist_file = self._fresh_main(monkeypatch, tmp_path)
        hist_file.write_text("{not valid json", encoding="utf-8")
        main._load_history()
        assert main._analyzed_history == {}


# ---------------------------------------------------------------------------
# Tests: match-market classification (title + slug)
# ---------------------------------------------------------------------------

class TestClassifyMarket:
    def test_match_by_title_vs(self):
        import main
        assert main._classify_market("USA vs Iran", "") == "match"

    def test_match_by_slug_vs(self):
        import main
        assert main._classify_market("Group stage result", "usa-vs-iran-2026") == "match"

    def test_match_by_match_winner_phrase(self):
        import main
        assert main._classify_market("Brazil Match Winner", "") == "match"

    def test_outright_still_detected(self):
        import main
        assert main._classify_market("Win the 2026 FIFA World Cup", "argentina-to-win") == "outright"

    def test_general_default(self):
        import main
        assert main._classify_market("Will it rain in NYC?", "rain-nyc") == "general"

    def test_beat_format_is_match(self):
        # Formatul binar World Cup „Will [A] beat [B]?" trebuie detectat ca meci.
        import main
        assert main._classify_market("Will France beat Germany?", "") == "match"

    def test_defeat_format_is_match(self):
        import main
        assert main._classify_market("Will Argentina defeat Brazil?", "") == "match"

    def test_single_v_separator_is_match(self):
        import main
        assert main._classify_market("England v Croatia", "") == "match"

    def test_beat_slug_is_match(self):
        import main
        assert main._classify_market("Match result", "france-beat-germany") == "match"

    def test_world_cup_winner_question_is_outright(self):
        # Outright pe termen lung — NU meci.
        import main
        assert main._classify_market("Will Germany win the World Cup?", "") == "outright"

    def test_group_winner_is_outright(self):
        import main
        assert main._classify_market("Will Brazil win Group A?", "") == "outright"

    def test_advance_prop_is_outright(self):
        import main
        assert main._classify_market("Will the USA advance to the knockouts?", "") == "outright"

    def test_beat_with_lowercase_operands_not_match(self):
        # „beat expectations" / „defeat inflation" NU sunt meciuri (fără nume proprii).
        import main
        assert main._classify_market("Will inflation beat expectations?", "") == "general"

    def test_world_cup_winner_is_outright(self):
        import main
        assert main._classify_market("Will Brazil win the 2026 FIFA World Cup?", "") == "outright"

    def test_world_cup_match_winner_is_match(self):
        import main
        assert main._classify_market("World Cup Match Winner: USA vs Iran", "wc-usa-vs-iran") == "match"

    def test_world_cup_match_vs_takes_precedence_over_outright(self):
        import main
        # 'vs' present → must classify as match, not outright (precedence)
        assert main._classify_market("2026 World Cup: France vs Croatia", "") == "match"

    def test_is_match_market_true_for_vs(self):
        import main
        assert main._is_match_market({"question": "England vs Croatia"}) is True

    def test_is_match_market_false_for_general(self):
        import main
        assert main._is_match_market({"question": "Will it rain in NYC?"}) is False

    def test_match_markets_sort_to_front(self):
        # Logica de prioritizare din bucla de scanare: sort stabil cu cheia
        # `not _is_match_market` aduce meciurile în față, restul după.
        import main
        markets = [
            {"question": "Will it rain in NYC?"},
            {"question": "England vs Croatia"},
            {"question": "Win the 2026 FIFA World Cup"},
            {"question": "France vs Spain"},
        ]
        markets.sort(key=lambda m: not main._is_match_market(m))
        assert main._is_match_market(markets[0]) and main._is_match_market(markets[1])
        assert not main._is_match_market(markets[2]) and not main._is_match_market(markets[3])


# ---------------------------------------------------------------------------
# Tests: probability window — unified [0.45, 0.95] for every market type
# ---------------------------------------------------------------------------

class TestProbabilityWindow:
    """Fereastra de probabilitate [0.45, 0.95] se aplică pe fiecare rezultat 1X2
    (YES = prețul rezultatului): pariem doar pe outcome-uri în fereastră, iar dacă
    niciunul nu califică, meciul e ignorat."""

    def _market(self, question, outcomes, prices):
        return {
            "id": "mx", "question": question, "volume": "50000",
            "outcomes": json.dumps(outcomes),
            "outcomePrices": json.dumps(prices),
            "clobTokenIds": json.dumps([f"0x{i}" for i in range(len(outcomes))]),
            "events": [{"slug": "ev"}],
        }

    def _run(self, market, outcome="1"):
        import main
        decision = {"trade": True, "outcome": outcome, "confidence": 0.9, "reason": "t"}
        state = {"pusd_balance": 20.0, "active_positions": []}
        with patch("main.fetch_free_news", return_value=[]), \
             patch("main.analyse_match", return_value=decision), \
             patch("main._record_analyzed"), \
             patch("main._live_best_ask", return_value=None), \
             patch("main.place_limit_order",
                   return_value={"success": True, "order_id": "o"}) as mock_order:
            main.process_market(market, state)
        return mock_order

    def test_match_midrange_allowed(self):
        # Leg '1' (USA) la 0.50 — în fereastra [0.45, 0.95].
        self._run(self._market("USA vs Iran", ["USA", "Draw", "Iran"],
                               ["0.50", "0.30", "0.20"])).assert_called_once()

    def test_low_probability_noise_blocked(self):
        # Niciun leg ≥ 0.45 → meci „necompetitiv pentru bot", filtrat.
        self._run(self._market("USA vs Iran", ["USA", "Draw", "Iran"],
                               ["0.30", "0.30", "0.40"])).assert_not_called()

    def test_extreme_high_probability_blocked(self):
        # 0.97 > 0.95 (favorit quasi-cert), restul sub 0.45 → niciun leg eligibil.
        self._run(self._market("USA vs Iran", ["USA", "Draw", "Iran"],
                               ["0.97", "0.02", "0.01"])).assert_not_called()

    def test_extreme_low_probability_blocked(self):
        # Meci 2-way (fără egal): 0.005 / 0.995 → ambele în afara ferestrei.
        self._run(self._market("USA vs Iran", ["USA", "Iran"],
                               ["0.005", "0.995"])).assert_not_called()

    def test_volume_filter_increments_scan_stats(self):
        import main
        main._scan_stats.clear()
        m = self._market("X vs Y", ["X", "Draw", "Y"], ["0.50", "0.30", "0.20"])
        m["volume"] = "100"   # below MIN_VOLUME_USD → filtered before any LLM call
        main.process_market(m, {"pusd_balance": 20.0, "active_positions": []})
        assert main._scan_stats["volum mic"] == 1


# ---------------------------------------------------------------------------
# Tests: fetch_active_markets pagination + dedup
# ---------------------------------------------------------------------------

class TestFetchActiveMarkets:
    def _resp(self, payload, status=200):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload
        return r

    def test_paginates_and_dedupes(self, monkeypatch):
        import main
        page1 = [{"id": str(i)} for i in range(100)]       # ids 0..99
        page2 = [{"id": str(i)} for i in range(99, 149)]   # ids 99..148: 99 overlaps, 50 < page_size → stop
        responses = [self._resp(page1), self._resp(page2)]
        calls = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            i = calls["n"]
            calls["n"] += 1
            return responses[i] if i < len(responses) else self._resp([])

        monkeypatch.setattr(main.httpx, "get", fake_get)
        out = main.fetch_active_markets(limit=200)
        ids = [m["id"] for m in out]
        assert len(out) == 149                  # 100 + 49 new (one duplicate dropped)
        assert len(ids) == len(set(ids))        # all unique

    def test_stops_on_non_200(self, monkeypatch):
        import main
        monkeypatch.setattr(main.httpx, "get", lambda *a, **k: self._resp([], status=500))
        assert main.fetch_active_markets(limit=50) == []

    def test_stops_on_exception(self, monkeypatch):
        import main

        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(main.httpx, "get", boom)
        assert main.fetch_active_markets(limit=50) == []

    def test_order_param_falls_back_on_non_200(self, monkeypatch):
        import main
        monkeypatch.setattr(main, "_GAMMA_ORDER", "volumeNum")
        seen = []

        def fake_get(url, params=None, timeout=None):
            seen.append(dict(params))
            if "order" in params:
                return self._resp(None, status=500)   # ordered request rejected
            return self._resp([{"id": "1"}])           # unordered works

        monkeypatch.setattr(main.httpx, "get", fake_get)
        out = main.fetch_active_markets(limit=5)
        assert any("order" in p for p in seen)         # tried with order first
        assert any("order" not in p for p in seen)     # then retried without it
        assert [m["id"] for m in out] == ["1"]

    def test_paginates_dynamically_beyond_200(self, monkeypatch):
        # Fără plafon (implicit), paginarea continuă dincolo de vechiul cap de 200:
        # 4 pagini (100+100+100+30 = 330) până la pagina parțială.
        import main
        sleeps = []
        monkeypatch.setattr(main.time, "sleep", lambda s: sleeps.append(s))
        pages = [
            [{"id": str(i)} for i in range(0, 100)],
            [{"id": str(i)} for i in range(100, 200)],
            [{"id": str(i)} for i in range(200, 300)],
            [{"id": str(i)} for i in range(300, 330)],
        ]
        seen_offsets = []

        def fake_get(url, params=None, timeout=None):
            seen_offsets.append(params["offset"])
            idx = len(seen_offsets) - 1
            return self._resp(pages[idx]) if idx < len(pages) else self._resp([])

        monkeypatch.setattr(main.httpx, "get", fake_get)
        out = main.fetch_active_markets()              # limit=0 → fără plafon
        assert len(out) == 330                          # toate cele 4 pagini (> 200)
        assert seen_offsets == [0, 100, 200, 300]       # offset += 100 pe iterație
        assert sleeps == [0.1, 0.1, 0.1]                # pauză între cereri, nu după ultima

    def test_safety_cap_stops_pagination_early(self, monkeypatch):
        # MARKET_SCAN_LIMIT > 0 = plafon opțional: oprește după pagina care îl atinge.
        import main
        monkeypatch.setattr(main.time, "sleep", lambda *a: None)
        pages = [
            [{"id": str(i)} for i in range(0, 100)],
            [{"id": str(i)} for i in range(100, 200)],
            [{"id": str(i)} for i in range(200, 300)],
        ]
        calls = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            i = calls["n"]
            calls["n"] += 1
            return self._resp(pages[i]) if i < len(pages) else self._resp([])

        monkeypatch.setattr(main.httpx, "get", fake_get)
        out = main.fetch_active_markets(limit=150)     # plafon de siguranță
        assert len(out) == 200                          # se oprește la pagina care atinge ≥150
        assert calls["n"] == 2                          # doar 2 cereri, nu a treia


# ---------------------------------------------------------------------------
# Tests: positions inventory webhook (Mesaj 2)
# ---------------------------------------------------------------------------

class TestPositionsNotification:
    def _patch(self, webhook="https://discord.test/webhook"):
        from config.settings import settings
        return patch.object(settings, "DISCORD_WEBHOOK_URL", webhook)

    def _embed(self, mock_post):
        return mock_post.call_args.kwargs["json"]["embeds"][0]

    def test_empty_positions_message(self):
        from src import notifications
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_positions([])
        embed = self._embed(mock_post)
        assert "Poziții Active" in embed["title"]
        assert "Niciuna." in embed["description"]

    def test_lists_position_fields(self):
        from src import notifications
        positions = [{"asset_name": "USA vs Iran", "outcome": "YES",
                      "initial_value": 5.0, "cash_pnl": 1.25, "current_value": 6.25}]
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_positions(positions)
        desc = self._embed(mock_post)["description"]
        assert "USA vs Iran" in desc and "YES" in desc
        assert "$5.00" in desc       # sumă investită
        assert "+$1.25" in desc      # P/L curent

    def test_missing_outcome_falls_back(self):
        from src import notifications
        positions = [{"asset_name": "Market", "initial_value": 5.0, "cash_pnl": 0.0}]
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_positions(positions)
        assert "—" in self._embed(mock_post)["description"]

    def test_no_webhook_is_noop(self):
        from src import notifications
        with self._patch(webhook=""), patch("src.notifications.requests.post") as mock_post:
            notifications.notify_positions([{"asset_name": "x"}])
        mock_post.assert_not_called()

    def test_truncates_over_preview(self):
        from src import notifications
        positions = [{"asset_name": f"M{i}", "outcome": "YES", "initial_value": 1.0,
                      "cash_pnl": 0.0, "current_value": float(i)} for i in range(25)]
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_positions(positions)
        assert "și încă 15 poziții" in self._embed(mock_post)["description"]


# ---------------------------------------------------------------------------
# Tests: football 1X2 market parsing (src/football.py)
# ---------------------------------------------------------------------------

class TestFootballParsing:
    def _market(self, question, outcomes, prices, tokens, slug=""):
        return {
            "question": question, "slug": slug,
            "outcomes": json.dumps(outcomes),
            "outcomePrices": json.dumps(prices),
            "clobTokenIds": json.dumps(tokens),
        }

    def test_three_way_mapped_to_1x2(self):
        from src.football import parse_match_market
        p = parse_match_market(self._market(
            "Arsenal vs. Chelsea", ["Arsenal", "Draw", "Chelsea"],
            ["0.55", "0.25", "0.20"], ["0xA", "0xX", "0xC"]))
        assert p["home"] == "Arsenal" and p["away"] == "Chelsea"
        assert p["is_three_way"] is True
        assert p["by_code"]["1"]["token_id"] == "0xA"
        assert p["by_code"]["X"]["price"] == pytest.approx(0.25)
        assert p["by_code"]["2"]["name"] == "Chelsea"
        assert [leg["code"] for leg in p["legs"]] == ["1", "X", "2"]

    def test_outcome_order_irrelevant_maps_by_name(self):
        from src.football import parse_match_market
        # Outcomes listed away-first/draw-last: mapping must follow names, not order.
        p = parse_match_market(self._market(
            "Arsenal vs Chelsea", ["Chelsea", "Arsenal", "Draw"],
            ["0.20", "0.55", "0.25"], ["0xC", "0xA", "0xX"]))
        assert p["by_code"]["1"]["token_id"] == "0xA"   # Arsenal = home
        assert p["by_code"]["2"]["token_id"] == "0xC"   # Chelsea = away
        assert p["by_code"]["X"]["token_id"] == "0xX"

    def test_two_way_match_has_no_draw(self):
        from src.football import parse_match_market
        p = parse_match_market(self._market(
            "USA vs Iran", ["USA", "Iran"], ["0.60", "0.40"], ["0xU", "0xI"]))
        assert p["is_three_way"] is False
        assert "X" not in p["by_code"]
        assert p["by_code"]["1"]["name"] == "USA"
        assert p["by_code"]["2"]["name"] == "Iran"

    def test_draw_synonym_tie_detected(self):
        from src.football import parse_match_market
        p = parse_match_market(self._market(
            "A vs B", ["A", "Tie", "B"], ["0.5", "0.3", "0.2"], ["0xA", "0xT", "0xB"]))
        assert p["by_code"]["X"]["token_id"] == "0xT"

    def test_binary_yes_no_is_not_1x2(self):
        from src.football import parse_match_market
        assert parse_match_market(self._market(
            "Brazil Match Winner", ["Yes", "No"], ["0.6", "0.4"], ["0xY", "0xN"])) is None

    def test_missing_arrays_returns_none(self):
        from src.football import parse_match_market
        assert parse_match_market({"question": "A vs B"}) is None

    def test_extract_teams_from_title_with_prefix(self):
        from src.football import extract_teams
        assert extract_teams("EPL: Arsenal vs. Chelsea") == ("Arsenal", "Chelsea")
        assert extract_teams("France vs Croatia") == ("France", "Croatia")

    def test_extract_teams_none_when_no_separator(self):
        from src.football import extract_teams
        assert extract_teams("Will it rain in NYC?") == (None, None)

    def test_extract_teams_beat_format(self):
        from src.football import extract_teams
        assert extract_teams("Will France beat Germany?") == ("France", "Germany")

    def test_extract_teams_beat_with_articles_and_multiword(self):
        from src.football import extract_teams
        assert extract_teams("Will the Netherlands beat the USA?") == ("Netherlands", "USA")
        assert extract_teams("Costa Rica defeat Brazil") == ("Costa Rica", "Brazil")

    def test_extract_teams_beat_lowercase_not_teams(self):
        from src.football import extract_teams
        assert extract_teams("Will inflation beat expectations?") == (None, None)

    def test_extract_teams_from_slug(self):
        from src.football import extract_teams
        home, away = extract_teams("", "arsenal-vs-chelsea-2026-06-18")
        assert home.lower() == "arsenal" and away.lower() == "chelsea"

    def test_format_legs_renders_codes_and_prices(self):
        from src.football import parse_match_market, format_legs
        p = parse_match_market(self._market(
            "Arsenal vs Chelsea", ["Arsenal", "Draw", "Chelsea"],
            ["0.55", "0.25", "0.20"], ["0xA", "0xX", "0xC"]))
        s = format_legs(p)
        assert "1 (Arsenal)=0.55" in s
        assert "X (Draw)=0.25" in s
        assert "2 (Chelsea)=0.20" in s


# ---------------------------------------------------------------------------
# Tests: market routing (universal default + opt-in football-only)
# ---------------------------------------------------------------------------

class TestMarketRouting:
    """Modul universal (implicit): toate piețele sunt evaluate — fotbalul pe ruta
    1X2, restul (politice/crypto/generale) pe ruta binară YES/NO. FOOTBALL_ONLY
    e opt-in (off by default) și, când e activ, ignoră tot ce nu e meci."""

    def _match_1x2(self):
        return {
            "id": "m1", "question": "Arsenal vs Chelsea", "volume": "50000",
            "outcomes": json.dumps(["Arsenal", "Draw", "Chelsea"]),
            "outcomePrices": json.dumps(["0.55", "0.25", "0.20"]),
            "clobTokenIds": json.dumps(["0xA", "0xX", "0xC"]),
            "events": [{"slug": "epl-ars-che"}],
        }

    def test_non_match_processed_by_default(self):
        # IMPLICIT (universal): o piață politică NU mai e ignorată — merge pe fluxul
        # binar YES/NO și ajunge la analiză. (_FOOTBALL_ONLY = False by default.)
        import main
        market = {
            "id": "p1", "question": "Will the Democrats win the 2028 election?", "volume": "50000",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0.55", "0.45"]),
            "clobTokenIds": json.dumps(["0xY", "0xN"]), "events": [{"slug": "us-2028"}],
        }
        with patch("main.fetch_free_news", return_value=[]), \
             patch("main._record_analyzed"), \
             patch("main.analyse_match") as am, \
             patch("main.analyse_market",
                   return_value={"trade": False, "outcome": "NO",
                                 "confidence": 0.0, "reason": "x"}) as ab:
            main.process_market(market, {"pusd_balance": 20.0, "active_positions": []})
        ab.assert_called_once()   # ruta binară (universal) rulează pe piața non-fotbal
        am.assert_not_called()    # nu e meci 1X2

    def test_outright_skipped_only_when_football_only_enabled(self, monkeypatch):
        # Opt-in FOOTBALL_ONLY=true: outright-urile sunt ignorate complet.
        import main
        monkeypatch.setattr(main, "_FOOTBALL_ONLY", True)
        main._scan_stats.clear()
        market = {
            "id": "o1", "question": "Win the 2026 FIFA World Cup", "volume": "999999",
            "outcomes": json.dumps(["Brazil", "Argentina"]),
            "outcomePrices": json.dumps(["0.20", "0.18"]),
            "clobTokenIds": json.dumps(["0xB", "0xAr"]), "events": [{"slug": "wc-2026"}],
        }
        with patch("main.analyse_match") as am, patch("main.analyse_market") as ab, \
             patch("main.place_limit_order") as mo:
            main.process_market(market, {"pusd_balance": 20.0, "active_positions": []})
        am.assert_not_called()
        ab.assert_not_called()
        mo.assert_not_called()
        assert main._scan_stats["non-meci (outright/general)"] == 1

    def test_outright_processed_when_football_only_disabled(self):
        # IMPLICIT: același outright e acum procesat (ruta binară), nu aruncat.
        # Preț 0.55 în fereastra [0.45, 0.95] ca să ajungă la analiză.
        import main
        market = {
            "id": "o2", "question": "Will Brazil win the 2026 FIFA World Cup?", "volume": "999999",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0.55", "0.45"]),
            "clobTokenIds": json.dumps(["0xY", "0xN"]), "events": [{"slug": "wc-2026"}],
        }
        with patch("main.fetch_free_news", return_value=[]), \
             patch("main._record_analyzed"), \
             patch("main.analyse_market",
                   return_value={"trade": False, "outcome": "NO",
                                 "confidence": 0.0, "reason": "x"}) as ab:
            main.process_market(market, {"pusd_balance": 20.0, "active_positions": []})
        ab.assert_called_once()

    def test_match_reaches_1x2_analysis(self):
        import main
        with patch("main.fetch_free_news", return_value=[]), \
             patch("main._record_analyzed"), \
             patch("main.analyse_match",
                   return_value={"trade": False, "outcome": "NONE",
                                 "confidence": 0.0, "reason": "x"}) as am:
            main.process_market(self._match_1x2(), {"pusd_balance": 20.0, "active_positions": []})
        am.assert_called_once()
        # LLM-ul primește exact echipele + candidații, nu outright-uri.
        assert am.call_args.args[0] == "Arsenal"
        assert am.call_args.args[1] == "Chelsea"

    def test_beat_format_binary_match_reaches_analysis(self):
        # „Will France beat Germany?" (Yes/No) nu mai e ignorat: e clasificat meci
        # și rulează pe fluxul binar moștenit (nu e 1X2, deci nu analyse_match).
        import main
        market = {
            "id": "b1", "question": "Will France beat Germany?", "volume": "50000",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0.60", "0.40"]),
            "clobTokenIds": json.dumps(["0xY", "0xN"]), "events": [{"slug": "wc-fra-ger"}],
        }
        with patch("main.fetch_free_news", return_value=[]), \
             patch("main._record_analyzed"), \
             patch("main.analyse_match") as am, \
             patch("main.analyse_market",
                   return_value={"trade": False, "outcome": "NO",
                                 "confidence": 0.0, "reason": "x"}) as ab:
            main.process_market(market, {"pusd_balance": 20.0, "active_positions": []})
        ab.assert_called_once()   # ruta binară moștenită rulează (meci, dar Yes/No)
        am.assert_not_called()    # nu e 1X2 → nu folosește analyse_match


# ---------------------------------------------------------------------------
# Tests: analyse_match 1X2 decision validation (src/ai_analyser.py)
# ---------------------------------------------------------------------------

class TestAnalyseMatch:
    def _candidates(self):
        return [
            {"code": "1", "name": "Arsenal", "price": 0.55},
            {"code": "X", "name": "Draw", "price": 0.25},
        ]

    def _run(self, raw):
        from src import ai_analyser
        from config.settings import settings
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch("src.ai_analyser._call_groq", return_value=raw):
            return ai_analyser.analyse_match("Arsenal", "Chelsea", self._candidates(), [])

    def test_valid_home_pick(self):
        d = self._run('{"trade": true, "outcome": "1", "confidence": 0.8, "reason": "r"}')
        assert d["trade"] is True and d["outcome"] == "1" and d["confidence"] == pytest.approx(0.8)

    def test_draw_pick_allowed(self):
        d = self._run('{"trade": true, "outcome": "X", "confidence": 0.7, "reason": "r"}')
        assert d["trade"] is True and d["outcome"] == "X"

    def test_pick_outside_candidates_cancels_trade(self):
        # "2" (away) nu e printre candidați → trade anulat, outcome NONE.
        d = self._run('{"trade": true, "outcome": "2", "confidence": 0.9, "reason": "r"}')
        assert d["trade"] is False and d["outcome"] == "NONE"

    def test_none_pick_no_trade(self):
        d = self._run('{"trade": false, "outcome": "NONE", "confidence": 0.0, "reason": "r"}')
        assert d["trade"] is False and d["outcome"] == "NONE"

    def test_no_api_key_skips(self):
        from src import ai_analyser
        from config.settings import settings
        with patch.object(settings, "GROQ_API_KEY", ""):
            d = ai_analyser.analyse_match("A", "B", self._candidates(), [])
        assert d["trade"] is False and d["outcome"] == "NONE"
