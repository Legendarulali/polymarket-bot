import os
import logging
from dotenv import load_dotenv

load_dotenv()

def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value

class Settings:
    PRIVATE_KEY: str = _require("POLYMARKET_PRIVATE_KEY")
    FUNDER_ADDRESS: str = _require("FUNDER_ADDRESS")
    CLOB_HOST: str = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
    CHAIN_ID: int = int(os.getenv("CHAIN_ID", "137"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    X_API_BEARER_TOKEN: str = os.getenv("X_API_BEARER_TOKEN", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # 0=EOA, 3=POLY_1271/EIP-1271 (Gnosis Safe / Magic Wallet proxy accounts)
    SIGNATURE_TYPE: int = int(os.getenv("SIGNATURE_TYPE", "3"))
    POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
    # Chei API pre-generate din Polymarket UI (opționale — dacă sunt prezente, ocolesc derivarea)
    POLYMARKET_API_KEY: str = os.getenv("POLYMARKET_API_KEY", "")
    POLYMARKET_API_SECRET: str = os.getenv("POLYMARKET_API_SECRET", "")
    POLYMARKET_API_PASSPHRASE: str = os.getenv("POLYMARKET_API_PASSPHRASE", "")
    # Balanță reală cunoscută (pUSD) — folosită când /balance-allowance returnează 0 pentru conturi proxy
    OVERRIDE_BALANCE: float = float(os.getenv("OVERRIDE_BALANCE", "0"))
    # Număr maxim de poziții deschise simultan (0 = fără limită)
    MAX_ACTIVE_POSITIONS: int = int(os.getenv("MAX_ACTIVE_POSITIONS", "5"))
    # Plafon USDC aprobat pentru CLOB Exchange, în USD (0 = aprobare nelimitată).
    # Aprobarea nelimitată expune tot soldul dacă exchange-ul e compromis;
    # o valoare mărginită necesită re-aprobare periodică (cost de gas).
    USDC_APPROVAL_USD: float = float(os.getenv("USDC_APPROVAL_USD", "0"))
    # Plafon preț gas pe Polygon (gwei). Tranzacțiile on-chain sunt amânate
    # când rețeaua depășește acest prag (0 = fără plafon).
    MAX_GAS_PRICE_GWEI: float = float(os.getenv("MAX_GAS_PRICE_GWEI", "500"))
    # Take-profit / stop-loss ca fracție din avg_price (ex: 0.20 = ±20%).
    # 0 = dezactivat — botul păstrează pozițiile până la rezoluția pieței.
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0"))
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0"))
    # Toleranță față de best ask live la BUY: dacă ask-ul real depășește prețul
    # Gamma cu mai mult de această fracție, ordinul e blocat (0 = dezactivat).
    SLIPPAGE_TOLERANCE: float = float(os.getenv("SLIPPAGE_TOLERANCE", "0.05"))

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
