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
    def _market(self, market_id="m1", token_ids=("0xYES", "0xNO"), event_slug="ev-1"):
        return {
            "id": market_id,
            "question": "Will team X win the final?",
            "volume": "50000",
            "outcomePrices": json.dumps(["0.70", "0.30"]),
            "clobTokenIds": json.dumps(list(token_ids)),
            "events": [{"slug": event_slug}],
        }

    def _run(self, market, account_state, decision=None, order_result=None, live_ask=None):
        import main
        main._analysis_cache.clear()
        decision = decision or {"trade": True, "outcome": "YES", "confidence": 0.9, "reason": "test"}
        order_result = order_result or {"success": True, "order_id": "oid-1"}

        with patch("main.fetch_free_news", return_value=[]), \
             patch("main.analyse_market", return_value=decision), \
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
        # YES=0.95 is outside the (0.65, 0.90) window — and must be read
        # correctly from the JSON string, not silently default to 0.5.
        market = self._market()
        market["outcomePrices"] = json.dumps(["0.95", "0.05"])
        state = {"pusd_balance": 20.0, "active_positions": []}
        mock_order = self._run(market, state)
        mock_order.assert_not_called()

    def test_insufficient_balance_blocks_order(self):
        state = {"pusd_balance": 0.5, "active_positions": []}
        mock_order = self._run(self._market(), state)
        mock_order.assert_not_called()

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
