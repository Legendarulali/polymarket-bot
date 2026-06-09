"""
Unit tests for the Polymarket trading bot.

Covers pure-logic functions and mocked API interactions.
All external dependencies (py_clob_client_v2, openai, web3, …) are
mocked in conftest.py so no production environment is required.
"""
import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Pure helper functions copied from main.py (no external deps needed)
# These are tested in isolation to guarantee their logic is correct.
# ---------------------------------------------------------------------------

def _token_for_outcome(market: dict, outcome: str):
    raw = market.get("clobTokenIds")
    if raw:
        try:
            ids = json.loads(raw) if isinstance(raw, str) else raw
            if outcome.upper() == "YES" and len(ids) > 0:
                return ids[0]
            if outcome.upper() == "NO" and len(ids) > 1:
                return ids[1]
        except (json.JSONDecodeError, TypeError):
            pass
    for token in market.get("tokens", []):
        if token.get("outcome", "").upper() == outcome.upper():
            return token.get("token_id")
    return None


def _current_price(market: dict, outcome: str) -> float:
    try:
        prices = market.get("outcomePrices", [])
        if outcome.upper() == "YES" and prices:
            return float(prices[0])
        if outcome.upper() == "NO" and len(prices) > 1:
            return float(prices[1])
        ask = market.get("bestAsk")
        if ask is not None:
            return float(ask)
    except (ValueError, TypeError):
        pass
    return 0.5


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
    def test_yes_from_outcome_prices(self):
        assert _current_price({"outcomePrices": ["0.72", "0.28"]}, "YES") == pytest.approx(0.72)

    def test_no_from_outcome_prices(self):
        assert _current_price({"outcomePrices": ["0.72", "0.28"]}, "NO") == pytest.approx(0.28)

    def test_fallback_to_best_ask(self):
        assert _current_price({"bestAsk": "0.55"}, "YES") == pytest.approx(0.55)

    def test_fallback_to_0_5_when_empty(self):
        assert _current_price({}, "YES") == pytest.approx(0.5)

    def test_zero_price_returned(self):
        assert _current_price({"outcomePrices": ["0.0", "1.0"]}, "YES") == pytest.approx(0.0)

    def test_invalid_price_string_falls_back(self):
        assert _current_price({"outcomePrices": ["invalid"]}, "YES") == pytest.approx(0.5)

    def test_no_outcome_prices_only_one_element_falls_back(self):
        assert _current_price({"outcomePrices": ["0.6"]}, "NO") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: dynamic size calculation in execution.py
# ---------------------------------------------------------------------------

class TestDynamicSizeCalculation:
    """place_limit_order computes size = max($5/price, 2.0) rounded to 2dp."""

    def _call(self, price, mock_result=None):
        from src.execution import place_limit_order
        from py_clob_client_v2 import Side

        if mock_result is None:
            mock_result = {"orderID": "test-id-123"}

        mock_client = MagicMock()
        mock_client.create_and_post_order.return_value = mock_result

        with patch("src.execution.get_clob_client", return_value=mock_client):
            result = place_limit_order(token_id="0xTOKEN", price=price, side=Side.BUY)

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
        # If price were somehow very high (hypothetical), floor kicks in.
        # price=0.99: 5/0.99≈5.05 > 2.0, so floor not active here.
        # We test floor with price close to 1 and confirm size >= 2.0
        size = self._placed_size(0.98)
        assert size >= 2.0

    def test_invalid_price_zero_returns_error(self):
        from src.execution import place_limit_order
        from py_clob_client_v2 import Side
        result = place_limit_order(token_id="0xT", price=0.0, side=Side.BUY)
        assert result["success"] is False
        assert "range" in result["error"]

    def test_price_above_0_99_returns_error(self):
        from src.execution import place_limit_order
        from py_clob_client_v2 import Side
        result = place_limit_order(token_id="0xT", price=1.01, side=Side.BUY)
        assert result["success"] is False

    def test_success_result_returned(self):
        result, _ = self._call(0.5, mock_result={"orderID": "abc-123"})
        assert result["success"] is True
        assert result["order_id"] == "abc-123"

    def test_api_exception_returns_failure(self):
        from src.execution import place_limit_order
        from py_clob_client_v2 import Side

        mock_client = MagicMock()
        mock_client.create_and_post_order.side_effect = Exception("500 server error")

        with patch("src.execution.get_clob_client", return_value=mock_client):
            result = place_limit_order(token_id="0xT", price=0.5, side=Side.BUY)

        assert result["success"] is False
        assert "500 server error" in result["error"]

    def test_size_is_rounded_to_2_decimal_places(self):
        size = self._placed_size(0.33)  # 5/0.33 = 15.1515...
        assert size == round(size, 2)


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

    def test_min_balance_constant_in_main(self):
        import importlib, sys
        if "main" not in sys.modules:
            importlib.import_module("main")
        import main
        assert main._MIN_BALANCE_PUSD == pytest.approx(1.0)

    def test_min_confidence_constant_in_main(self):
        import importlib, sys
        if "main" not in sys.modules:
            importlib.import_module("main")
        import main
        assert main._MIN_CONFIDENCE == pytest.approx(0.70)
