import logging
from py_clob_client_v2 import ClobClient, ApiCreds
from config.settings import settings

logger = logging.getLogger(__name__)

_client: ClobClient | None = None

_HAS_EXPLICIT_CREDS = bool(
    settings.POLYMARKET_API_KEY
    and settings.POLYMARKET_API_SECRET
    and settings.POLYMARKET_API_PASSPHRASE
)


def get_clob_client() -> ClobClient:
    global _client
    if _client is not None:
        return _client

    if _HAS_EXPLICIT_CREDS:
        sig_type = 1
        logger.info(
            "Initializing Polymarket CLOB V2 client with explicit API credentials | "
            "host=%s chain_id=%d funder=%s signature_type=%d",
            settings.CLOB_HOST,
            settings.CHAIN_ID,
            settings.FUNDER_ADDRESS,
            sig_type,
        )
        client = ClobClient(
            host=settings.CLOB_HOST,
            chain_id=settings.CHAIN_ID,
            key=settings.PRIVATE_KEY,
            funder=settings.FUNDER_ADDRESS,
            signature_type=sig_type,
            api_key=settings.POLYMARKET_API_KEY,
            api_secret=settings.POLYMARKET_API_SECRET,
            api_passphrase=settings.POLYMARKET_API_PASSPHRASE,
        )
        logger.info("CLOB client ready (explicit creds) | API_KEY=%s", settings.POLYMARKET_API_KEY)
    else:
        logger.info(
            "Initializing Polymarket CLOB V2 client (derive mode) | "
            "host=%s chain_id=%d funder=%s signature_type=%d",
            settings.CLOB_HOST,
            settings.CHAIN_ID,
            settings.FUNDER_ADDRESS,
            settings.SIGNATURE_TYPE,
        )
        client = ClobClient(
            host=settings.CLOB_HOST,
            chain_id=settings.CHAIN_ID,
            key=settings.PRIVATE_KEY,
            funder=settings.FUNDER_ADDRESS,
            signature_type=settings.SIGNATURE_TYPE,
        )
        try:
            creds = client.derive_api_key()
            logger.info("API credentials derived locally from private key.")
        except Exception as exc:
            logger.warning("derive_api_key() failed (%s) — retrying with create_or_derive_api_key().", exc)
            creds = client.create_or_derive_api_key()
            logger.info("API credentials obtained via create_or_derive_api_key().")
        client.set_api_creds(creds)

    _client = client
    logger.info("CLOB client ready | funder=%s", settings.FUNDER_ADDRESS)
    return _client
