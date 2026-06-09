import logging
from eth_account import Account
from web3 import Web3

from py_clob_client_v2 import BalanceAllowanceParams, AssetType
from src.clob_client import get_clob_client
from config.settings import settings

logger = logging.getLogger(__name__)

# Polymarket Polygon mainnet — collateral = USDC.e (6 decimals)
_USDC_ADDR     = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
_EXCHANGE_ADDR = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
_MAX_UINT256   = 2**256 - 1
_USDC_DECIMALS = 1_000_000  # 10^6

_ERC20_ABI = [
    {
        "name": "approve", "type": "function", "stateMutability": "nonpayable",
        "inputs":  [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "allowance", "type": "function", "stateMutability": "view",
        "inputs":  [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


def _sync_clob_server(client) -> None:
    """Cere serverului CLOB să re-citească starea on-chain a balanței/allowance."""
    try:
        client.update_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        logger.info("[Allowance] CLOB server: stare balanță/allowance sincronizată.")
    except Exception as exc:
        logger.warning("[Allowance] update_balance_allowance() sync eșuat: %s", exc)


def _read_on_chain_allowance(w3: Web3, owner: str) -> float:
    usdc = w3.eth.contract(address=_USDC_ADDR, abi=_ERC20_ABI)
    raw = usdc.functions.allowance(
        Web3.to_checksum_address(owner),
        _EXCHANGE_ADDR,
    ).call()
    return raw / _USDC_DECIMALS


def _send_approve_tx(w3: Web3, private_key: str, signer: str) -> str:
    usdc = w3.eth.contract(address=_USDC_ADDR, abi=_ERC20_ABI)
    tx = usdc.functions.approve(_EXCHANGE_ADDR, _MAX_UINT256).build_transaction({
        "from":     signer,
        "nonce":    w3.eth.get_transaction_count(signer),
        "gas":      80_000,
        "gasPrice": w3.eth.gas_price,
        "chainId":  137,
    })
    signed   = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash  = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt  = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt["status"] != 1:
        raise RuntimeError(f"Approve tx reverted: {tx_hash.hex()}")
    return tx_hash.hex()


def ensure_collateral_approved() -> None:
    """
    Verifică și setează allowance-ul ERC-20 al contractului CLOB Exchange pentru USDC.

    Logica în trepte:
      1. Dacă librăria expune approve_collateral() — o folosim direct (gestionează
         intern semnarea prin proxy).
      2. Conectare la Polygon RPC și citire allowance on-chain.
         - Dacă allowance > 0 → doar sincronizăm serverul CLOB și returnăm.
         - Dacă funder == EOA (SIGNATURE_TYPE=0) → trimitem tx approve direct.
         - Dacă funder ≠ EOA (proxy wallet) → nu putem semna din bot; afișăm
           instrucțiuni clare pentru aprobare manuală din web UI.
    """
    return
    client = get_clob_client()

    # Treaptă 1: metoda nativă din librărie (proxy-aware)
    if hasattr(client, "approve_collateral"):
        try:
            logger.info("[Allowance] Încerc client.approve_collateral()...")
            tx = client.approve_collateral()
            logger.info("[Allowance] approve_collateral() confirmat: tx=%s", tx)
            _sync_clob_server(client)
            return
        except Exception as exc:
            logger.warning("[Allowance] approve_collateral() eșuat: %s — trec la web3.", exc)

    # Treaptă 2: verificare on-chain via web3
    rpc = settings.POLYGON_RPC_URL
    logger.info("[Allowance] Conectare la Polygon RPC: %s", rpc)
    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        logger.error(
            "[Allowance] Imposibil de conectat la RPC %s. "
            "Setează POLYGON_RPC_URL cu un endpoint Polygon valabil (ex: Alchemy/Infura).",
            rpc,
        )
        return

    funder = settings.FUNDER_ADDRESS
    current_allowance = _read_on_chain_allowance(w3, funder)
    logger.info(
        "[Allowance] Allowance on-chain: funder=%s → %.6f USDC față de Exchange.",
        funder,
        current_allowance,
    )

    if current_allowance > 0:
        logger.info("[Allowance] Allowance deja setat. Sincronizez serverul CLOB...")
        _sync_clob_server(client)
        return

    # Treaptă 3: trimite approve direct dacă funder == EOA (signature_type=0)
    eoa_address = Account.from_key(settings.PRIVATE_KEY).address

    if eoa_address.lower() != funder.lower():
        logger.warning(
            "[Allowance] *** ACȚIUNE MANUALĂ NECESARĂ ***\n"
            "  Funder  : %s\n"
            "  EOA bot : %s\n"
            "  Cele două adrese diferă — funder-ul este un proxy wallet.\n"
            "  Botul nu poate semna approve() în numele contractului proxy.\n"
            "  Pași pentru rezolvare:\n"
            "    1. Deschide https://polymarket.com și conectează portofelul funder.\n"
            "    2. Mergi la Settings → Deposit sau aprobă CLOB-ul din interfață.\n"
            "    3. Alternativ: rulează scriptul one-shot de mai jos cu cheia proxy:\n"
            "       POLYGON_RPC_URL=... PRIVATE_KEY=<proxy_key> python -c \"\n"
            "       from src.allowance import ensure_collateral_approved; "
            "ensure_collateral_approved()\"",
            funder,
            eoa_address,
        )
        return

    # funder == EOA → putem trimite direct
    logger.info("[Allowance] Trimit approve(MAX_UINT256) on-chain de la %s...", eoa_address)
    try:
        tx_hash = _send_approve_tx(w3, settings.PRIVATE_KEY, eoa_address)
        logger.info("[Allowance] Approve confirmat pe Polygon! tx=%s", tx_hash)
        _sync_clob_server(client)
    except Exception as exc:
        logger.error("[Allowance] Aprobare on-chain eșuată: %s", exc, exc_info=True)
