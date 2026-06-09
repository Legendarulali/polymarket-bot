import logging
from py_clob_client_v2 import ClobClient
from config.settings import settings

logger = logging.getLogger(__name__)

_client: ClobClient | None = None


def get_clob_client() -> ClobClient:
    global _client
    if _client is not None:
        return _client

    logger.info(
        "Initializing Polymarket CLOB V2 client | host=%s chain_id=%d "
        "funder=%s signature_type=%d",
        settings.CLOB_HOST,
        settings.CHAIN_ID,
        settings.FUNDER_ADDRESS,
        settings.SIGNATURE_TYPE,
    )

    # signature_type=1 → POLY_PROXY: funder (proxy/safe) ≠ signer EOA.
    # Override via SIGNATURE_TYPE env var (0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE).
    client = ClobClient(
        host=settings.CLOB_HOST,
        chain_id=settings.CHAIN_ID,
        key=settings.PRIVATE_KEY,
        funder=settings.FUNDER_ADDRESS,
        signature_type=settings.SIGNATURE_TYPE,
    )

    # Încearcă înregistrarea activă pe server cu create_api_key().
    # Dacă cheile există deja serverul va returna eroare → fallback la derive_api_key().
    try:
        logger.info("[API] Încerc înregistrarea activă a API Key pe serverul Polymarket...")
        creds = client.create_api_key()
        logger.info("[API] Chei generate cu succes! API_KEY=%s", creds.api_key)
    except Exception as e:
        logger.warning("[API] Înregistrarea pe server a eșuat sau cheile există deja: %s", e)
        try:
            creds = client.derive_api_key()
            logger.info("API credentials derived locally from private key (no server round-trip).")
        except Exception as exc:
            logger.warning(
                "derive_api_key() failed (%s) — retrying with create_or_derive_api_key().", exc
            )
            creds = client.create_or_derive_api_key()
            logger.info("API credentials obtained via create_or_derive_api_key().")

    client.set_api_creds(creds)
    _client = client
    logger.info("CLOB client ready | funder=%s", settings.FUNDER_ADDRESS)
    return _client
