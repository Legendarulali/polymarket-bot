import logging
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
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

    # Tipuri de semnătură py-clob-client oficial: 0=EOA, 1=POLY_PROXY (Magic/
    # email wallet), 2=POLY_GNOSIS_SAFE (proxy Gnosis Safe creat de Polymarket).
    # Pentru conturi proxy: maker = FUNDER_ADDRESS, semnat de EOA.
    if _HAS_EXPLICIT_CREDS:
        logger.info(
            "Initializing Polymarket CLOB client with explicit API credentials | "
            "host=%s chain_id=%d funder=%s signature_type=%d",
            settings.CLOB_HOST,
            settings.CHAIN_ID,
            settings.FUNDER_ADDRESS,
            settings.SIGNATURE_TYPE,
        )
        api_creds = ApiCreds(
            api_key=settings.POLYMARKET_API_KEY,
            api_secret=settings.POLYMARKET_API_SECRET,
            api_passphrase=settings.POLYMARKET_API_PASSPHRASE,
        )
        client = ClobClient(
            host=settings.CLOB_HOST,
            chain_id=settings.CHAIN_ID,
            key=settings.PRIVATE_KEY,
            creds=api_creds,
            funder=settings.FUNDER_ADDRESS,
            signature_type=settings.SIGNATURE_TYPE,
        )
    else:
        logger.info(
            "Initializing Polymarket CLOB client (derive mode) | "
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
            logger.warning("derive_api_key() failed (%s) — retrying with create_or_derive_api_creds().", exc)
            creds = client.create_or_derive_api_creds()
            logger.info("API credentials obtained via create_or_derive_api_creds().")
        client.set_api_creds(creds)

    _client = client
    logger.info("CLOB client ready | funder=%s", settings.FUNDER_ADDRESS)
    return _client
