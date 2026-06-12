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

    sys.modules["py_clob_client_v2"] = _mock_clob
    sys.modules["py_clob_client_v2.clob_types"] = _mock_clob_types

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
