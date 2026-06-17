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
        decision = decision or {"trade": True, "outcome": "YES", "confidence": 0.9, "reason": "test"}
        order_result = order_result or {"success": True, "order_id": "oid-1"}

        # _record_analyzed is patched out so process_market does not touch history.json;
        # history persistence is covered separately by TestAnalyzedHistory.
        with patch("main.fetch_free_news", return_value=[]), \
             patch("main.analyse_market", return_value=decision), \
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
        assert "Status Portofoliu" in embed["title"]
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


# ---------------------------------------------------------------------------
# Tests: probability window — relaxed for match markets, strict otherwise
# ---------------------------------------------------------------------------

class TestMatchProbabilityWindow:
    def _market(self, question, prices):
        return {
            "id": "mx", "question": question, "volume": "50000",
            "outcomePrices": json.dumps(prices),
            "clobTokenIds": json.dumps(["0xYES", "0xNO"]),
            "events": [{"slug": "ev"}],
        }

    def _run(self, market):
        import main
        decision = {"trade": True, "outcome": "YES", "confidence": 0.9, "reason": "t"}
        state = {"pusd_balance": 20.0, "active_positions": []}
        with patch("main.fetch_free_news", return_value=[]), \
             patch("main.analyse_market", return_value=decision), \
             patch("main._record_analyzed"), \
             patch("main._live_best_ask", return_value=None), \
             patch("main.place_limit_order",
                   return_value={"success": True, "order_id": "o"}) as mock_order:
            main.process_market(market, state)
        return mock_order

    def test_match_midrange_probability_allowed(self):
        # 0.50 is outside the strict (0.65, 0.90) window but inside the match window
        self._run(self._market("USA vs Iran", ["0.50", "0.50"])).assert_called_once()

    def test_general_midrange_probability_blocked(self):
        self._run(self._market("Will the team win the final?", ["0.50", "0.50"])).assert_not_called()

    def test_match_extreme_probability_still_blocked(self):
        # 0.97 is outside even the relaxed (0.45, 0.95) match window
        self._run(self._market("USA vs Iran", ["0.97", "0.03"])).assert_not_called()

    def test_volume_filter_increments_scan_stats(self):
        import main
        main._scan_stats.clear()
        m = self._market("X vs Y", ["0.50", "0.50"])
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


# ---------------------------------------------------------------------------
# Tests: markets-list webhook + trade outcome field
# ---------------------------------------------------------------------------

class TestMarketsNotification:
    def _patch(self, webhook="https://discord.test/webhook"):
        from config.settings import settings
        return patch.object(settings, "DISCORD_WEBHOOK_URL", webhook)

    def _desc(self, mock_post):
        return mock_post.call_args.kwargs["json"]["embeds"][0]["description"]

    def test_no_webhook_is_noop(self):
        from src import notifications
        with self._patch(webhook=""), patch("src.notifications.requests.post") as mock_post:
            notifications.notify_markets([{"question": "m"}])
        mock_post.assert_not_called()

    def test_lists_market_titles(self):
        from src import notifications
        markets = [{"question": "USA vs Iran"}, {"title": "Brazil vs Spain"}]
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_markets(markets)
        desc = self._desc(mock_post)
        assert "USA vs Iran" in desc and "Brazil vs Spain" in desc

    def test_truncates_over_10(self):
        from src import notifications
        markets = [{"question": f"Market {i}"} for i in range(25)]
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_markets(markets)
        desc = self._desc(mock_post)
        assert "și încă 15 piețe" in desc
        assert desc.count("\n") == 10  # 10 market lines + the "…și încă" line

    def test_respects_2000_char_limit(self):
        from src import notifications
        markets = [{"question": "X" * 400} for _ in range(50)]
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_markets(markets)
        assert len(self._desc(mock_post)) <= 2000

    def test_accepts_plain_strings(self):
        from src import notifications
        with self._patch(), patch("src.notifications.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            notifications.notify_markets(["Market A", "Market B"])
        assert "Market A" in self._desc(mock_post)

    def test_network_error_is_swallowed(self):
        from src import notifications
        with self._patch(), \
             patch("src.notifications.requests.post", side_effect=Exception("boom")):
            # Must not raise — a Discord failure must never break the bot loop.
            notifications.notify_markets([{"question": "m"}])

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
# Tests: positions inventory webhook (Mesaj 2) + change-detection signature
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
        assert "Nicio poziție deschisă" in embed["description"]

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


class TestPositionsSignature:
    def test_order_independent(self):
        import main
        a = [{"token_id": "t1", "size": 5.0}, {"token_id": "t2", "size": 3.0}]
        b = [{"token_id": "t2", "size": 3.0}, {"token_id": "t1", "size": 5.0}]
        assert main._positions_signature(a) == main._positions_signature(b)

    def test_size_change_changes_signature(self):
        import main
        assert main._positions_signature([{"token_id": "t1", "size": 5.0}]) != \
               main._positions_signature([{"token_id": "t1", "size": 6.0}])

    def test_empty_is_stable(self):
        import main
        assert main._positions_signature([]) == ""
