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
    # 0=EOA, 1=POLY_PROXY (funder≠signer, standard Polymarket setup), 2=POLY_GNOSIS_SAFE
    SIGNATURE_TYPE: int = int(os.getenv("SIGNATURE_TYPE", "1"))

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
