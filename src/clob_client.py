import logging
from py_clob_client_v2 import ClobClient
from config.settings import settings

logger = logging.getLogger(__name__)

_client: ClobClient | None = None


def get_clob_client() -> ClobClient:
    global _client
    if _client is not None:
        return _client

    logger.info("Initializing Polymarket CLOB V2 client for host=%s chain_id=%d",
                settings.CLOB_HOST, settings.CHAIN_ID)

    client = ClobClient(
        host=settings.CLOB_HOST,
        chain_id=settings.CHAIN_ID,
        key=settings.PRIVATE_KEY,
        funder=settings.FUNDER_ADDRESS,
    )

    creds = client.create_or_derive_api_key()
    client.set_api_creds(creds)

    _client = client
    logger.info("CLOB client authenticated successfully (funder=%s)", settings.FUNDER_ADDRESS)
    return _client
