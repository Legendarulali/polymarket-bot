"""
Global pytest configuration.
Sets required env vars and mocks all packages that are not installed locally
(py_clob_client_v2, openai, web3, tenacity, defusedxml, httpx) so unit tests
can import bot modules without a full production environment. When a package
IS installed, the real one is used.
"""
import os
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Required environment variables (must come before any project import)
# ---------------------------------------------------------------------------
os.environ.setdefault("POLYMARKET_PRIVATE_KEY", "0x" + "a" * 64)
os.environ.setdefault("FUNDER_ADDRESS", "0x" + "b" * 40)
os.environ.setdefault("CLOB_HOST", "https://clob.polymarket.com")
os.environ.setdefault("CHAIN_ID", "137")
os.environ.setdefault("SIGNATURE_TYPE", "3")
os.environ.setdefault("POLYGON_RPC_URL", "https://polygon-rpc.com")

# ---------------------------------------------------------------------------
# Mock py_clob_client_v2 — only if not installed
# ---------------------------------------------------------------------------
try:
    import py_clob_client_v2  # noqa: F401
except ImportError:
    @dataclass
    class OrderArgs:
        token_id: str
        price: float
        size: float
        side: str
        expiration: int = 0

    class OrderType:
        GTC = "GTC"
        FOK = "FOK"

    class AssetType:
        COLLATERAL = "COLLATERAL"
        CONDITIONAL = "CONDITIONAL"

    _mock_clob = MagicMock()
    _mock_clob.OrderArgs = OrderArgs
    _mock_clob.OrderType = OrderType
    _mock_clob.AssetType = AssetType
    _mock_clob.PartialCreateOrderOptions = MagicMock(return_value=MagicMock())
    _mock_clob.ClobClient = MagicMock()
    _mock_clob.BalanceAllowanceParams = MagicMock(return_value=MagicMock())

    _mock_clob_types = MagicMock()
    _mock_clob_types.ApiCreds = MagicMock()

    class PolyApiException(Exception):
        """Stub al excepției SDK pentru teste — păstrează `status_code`/`error_msg` ca
        în fork-ul real (forma string e cea pe care o loghează botul la un 401)."""

        def __init__(self, status_code=None, error_msg=None):
            self.status_code = status_code
            self.error_msg = error_msg
            super().__init__(
                f"PolyApiException[status_code={status_code}, error_message={error_msg}]"
            )

    _mock_exceptions = MagicMock()
    _mock_exceptions.PolyApiException = PolyApiException

    sys.modules["py_clob_client_v2"] = _mock_clob
    sys.modules["py_clob_client_v2.clob_types"] = _mock_clob_types
    sys.modules["py_clob_client_v2.exceptions"] = _mock_exceptions

# ---------------------------------------------------------------------------
# Mock openai (used by ai_analyser.py)
# ---------------------------------------------------------------------------
try:
    import openai  # noqa: F401
except ImportError:
    sys.modules["openai"] = MagicMock()

# ---------------------------------------------------------------------------
# Mock tenacity (used by ai_analyser.py)
# ---------------------------------------------------------------------------
try:
    import tenacity  # noqa: F401
except ImportError:
    _mock_tenacity = MagicMock()
    _mock_tenacity.retry = lambda **kw: (lambda f: f)  # pass-through decorator
    _mock_tenacity.wait_exponential = MagicMock()
    _mock_tenacity.stop_after_attempt = MagicMock()
    _mock_tenacity.RetryError = Exception
    sys.modules["tenacity"] = _mock_tenacity

# ---------------------------------------------------------------------------
# Mock web3 / eth_account (used by allowance.py)
# ---------------------------------------------------------------------------
try:
    import web3  # noqa: F401
except ImportError:
    _mock_web3_mod = MagicMock()
    _mock_web3_cls = MagicMock()
    _mock_web3_cls.to_checksum_address = staticmethod(lambda addr: addr)
    _mock_web3_cls.HTTPProvider = MagicMock()
    _mock_web3_mod.Web3 = _mock_web3_cls
    sys.modules["web3"] = _mock_web3_mod
    sys.modules["web3.middleware"] = MagicMock()
    sys.modules["eth_account"] = MagicMock()

# ---------------------------------------------------------------------------
# Mock httpx (used by main.py)
# ---------------------------------------------------------------------------
try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules["httpx"] = MagicMock()

# ---------------------------------------------------------------------------
# Mock defusedxml (used by news_monitor.py)
# ---------------------------------------------------------------------------
try:
    import defusedxml  # noqa: F401
except ImportError:
    _mock_defusedxml_et = MagicMock()
    _mock_defusedxml = MagicMock()
    _mock_defusedxml.ElementTree = _mock_defusedxml_et
    sys.modules["defusedxml"] = _mock_defusedxml
    sys.modules["defusedxml.ElementTree"] = _mock_defusedxml_et


# ---------------------------------------------------------------------------
# Izolare fișier persistent: history.json → director temporar per test.
# Necesar acum că _record_ai_decision persistă deciziile AI (împreună cu istoricul
# de analiză) la fiecare update. Fără redirecționare, testele care ating
# _record_ai_decision (panic-sell, raportul de status) ar suprascrie history.json
# real din repo. Rulează pentru fiecare test; testele care își setează propriul
# _HISTORY_FILE îl suprascriu în continuare (monkeypatch, ultima setare câștigă).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_history_file(monkeypatch, tmp_path):
    main = sys.modules.get("main")
    if main is not None:
        monkeypatch.setattr(main, "_HISTORY_FILE", str(tmp_path / "history.json"), raising=False)
    yield
