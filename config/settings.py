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
    # FAILSAFE ROUTING: cheie OpenRouter (gol = dezactivat). Când Groq eșuează (în special
    # 429 rate-limit / TPD epuizat), apelul LLM se rutează SILENȚIOS către OpenRouter ca
    # backup (vezi src/ai_analyser._call_groq). Fără cheie → niciun fallback (comportament
    # identic cu cel dinainte). OPENROUTER_MODEL: model gratuit/ieftin pe OpenRouter.
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    # Tipuri de semnătură (py-clob-client-v2): 0=EOA, 1=POLY_PROXY (Magic),
    # 2=POLY_GNOSIS_SAFE, 3=POLY_1271/EIP-1271 (conturi proxy — confirmat
    # funcțional pentru acest cont). Notă: cu chei API explicite, ordinele
    # folosesc întotdeauna 3 (vezi src/clob_client.py).
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
    # Take-profit / stop-loss ca fracție din avg_price (ex: 0.20 = ±20%; 0 = dezactivat).
    # Default: TP dezactivat, SL activ la -30%. Rațiune: edge-ul botului e ca YES
    # să rezolve la $1, deci câștigătorii se țin până la rezoluție; un TP procentual
    # unic nici nu poate funcționa pe toată fereastra de intrare (la intrare 0.85,
    # +25% ar cere preț 1.06 — imposibil). În schimb, -30% față de o intrare tipică
    # 0.65-0.90 duce prețul sub ~0.5, adică teza e invalidată — acolo tăiem pierderea.
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0"))
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.30"))
    # Toleranță față de best ask live la BUY: dacă ask-ul real depășește prețul
    # Gamma cu mai mult de această fracție, ordinul e blocat (0 = dezactivat).
    SLIPPAGE_TOLERANCE: float = float(os.getenv("SLIPPAGE_TOLERANCE", "0.05"))
    # Discord webhook pentru notificări buy/sell și raport orar P/L (gol = dezactivat).
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    # Interval raport portofoliu pe Discord, în ore (0 = dezactivat raportul orar).
    PNL_REPORT_INTERVAL_HOURS: float = float(os.getenv("PNL_REPORT_INTERVAL_HOURS", "1"))
    # Timeout (secunde) pentru apelurile LLM (Groq + fallback OpenRouter). SDK-ul OpenAI
    # are implicit 600s (10 min!) — inacceptabil pentru bucla single-threaded: un apel
    # blocat ar îngheța TP/SL-ul și Panic Sell-ul. 120s acoperă lejer și un batch mare pe
    # un model free OpenRouter, dar taie hang-urile de rețea.
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

settings = Settings()

def _resolve_log_level(name: str) -> int:
    """Nivelul de logging din LOG_LEVEL, tolerant la lowercase/valori invalide → INFO.

    `getattr(logging, "debug")` (lowercase) ar întoarce FUNCȚIA logging.debug, nu nivelul —
    și `basicConfig(level=<funcție>)` ar crăpa cu TypeError chiar la import (bot mort la
    pornire pentru un simplu LOG_LEVEL=debug în .env). Normalizăm la uppercase și acceptăm
    doar valori care sunt efectiv nivele numerice."""
    level = getattr(logging, str(name).strip().upper(), logging.INFO)
    return level if isinstance(level, int) else logging.INFO

logging.basicConfig(
    level=_resolve_log_level(settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
