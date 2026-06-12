import logging
from py_clob_client_v2 import ClobClient
from py_clob_client_v2.clob_types import ApiCreds
from config.settings import settings

logger = logging.getLogger(__name__)

_client: ClobClient | None = None

_HAS_EXPLICIT_CREDS = bool(
    settings.POLYMARKET_API_KEY
    and settings.POLYMARKET_API_SECRET
    and settings.POLYMARKET_API_PASSPHRASE
)

# Conturile proxy Polymarket (Gnosis Safe / Magic Wallet) plasează ordine cu
# signature_type=3 (POLY_1271 / EIP-1271): maker = FUNDER_ADDRESS, semnat de
# EOA. Confirmat funcțional în producție; tipurile 1/2 produc "maker address
# not allowed" pentru acest cont. NU folosi settings.SIGNATURE_TYPE aici —
# acela poate fi setat diferit pentru endpoint-urile de balanță.
_ORDER_SIGNATURE_TYPE = 3


def get_signature_type() -> int:
    """Signature type efectiv folosit de client — pentru endpoint-urile de balanță."""
    if _HAS_EXPLICIT_CREDS:
        return _ORDER_SIGNATURE_TYPE
    return settings.SIGNATURE_TYPE


def get_clob_client() -> ClobClient:
    global _client
    if _client is not None:
        return _client

    if _HAS_EXPLICIT_CREDS:
        logger.info(
            "Initializing Polymarket CLOB V2 client with explicit API credentials (POLY_1271) | "
            "host=%s chain_id=%d funder=%s signature_type=%d",
            settings.CLOB_HOST,
            settings.CHAIN_ID,
            settings.FUNDER_ADDRESS,
            _ORDER_SIGNATURE_TYPE,
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
            signature_type=_ORDER_SIGNATURE_TYPE,
        )
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
