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

    # derive_api_key() derivă credențialele local din cheia privată (fără apel HTTP).
    # Evită erorile 400 de la /auth/api-key cauzate de creare duplicată sau nonce greșit.
    # Fallback la create_or_derive_api_key() doar dacă derivarea locală eșuează.
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
