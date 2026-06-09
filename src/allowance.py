import logging
from eth_account import Account
from web3 import Web3

from py_clob_client_v2 import BalanceAllowanceParams, AssetType
from src.clob_client import get_clob_client
from config.settings import settings

logger = logging.getLogger(__name__)

# Polymarket Polygon mainnet
_USDC_ADDR         = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
_CTF_ADDR          = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
_EXCHANGE_ADDR     = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
_NEG_RISK_EXCHANGE = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")
_NEG_RISK_ADAPTER  = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")
_ZERO_ADDR         = "0x0000000000000000000000000000000000000000"
_MAX_UINT256       = 2**256 - 1
_USDC_DECIMALS     = 1_000_000

# Toți operatorii care necesită aprobare CTF.setApprovalForAll
_CTF_OPERATORS = [_EXCHANGE_ADDR, _NEG_RISK_EXCHANGE, _NEG_RISK_ADAPTER]

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

_ERC1155_ABI = [
    {
        "name": "setApprovalForAll", "type": "function", "stateMutability": "nonpayable",
        "inputs":  [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}],
        "outputs": [],
    },
    {
        "name": "isApprovedForAll", "type": "function", "stateMutability": "view",
        "inputs":  [{"name": "account", "type": "address"}, {"name": "operator", "type": "address"}],
        "outputs": [{"name": "", "type": "bool"}],
    },
]

# Gnosis Safe minimal ABI
_SAFE_ABI = [
    {
        "name": "execTransaction", "type": "function", "stateMutability": "payable",
        "inputs": [
            {"name": "to",             "type": "address"},
            {"name": "value",          "type": "uint256"},
            {"name": "data",           "type": "bytes"},
            {"name": "operation",      "type": "uint8"},
            {"name": "safeTxGas",      "type": "uint256"},
            {"name": "baseGas",        "type": "uint256"},
            {"name": "gasPrice",       "type": "uint256"},
            {"name": "gasToken",       "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "signatures",     "type": "bytes"},
        ],
        "outputs": [{"name": "success", "type": "bool"}],
    },
]


def _sync_clob_server(client) -> None:
    sig_type = settings.SIGNATURE_TYPE
    try:
        client.update_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig_type)
        )
        logger.info("[Allowance] CLOB server: balanță/allowance sincronizate (sig_type=%d).", sig_type)
    except Exception as exc:
        logger.warning("[Allowance] update_balance_allowance() eșuat: %s", exc)


def _calldata(contract_fn) -> bytes:
    """Extrage calldata-ul ABI-encoded al unui ContractFunction fără a trimite tx."""
    dummy = "0x0000000000000000000000000000000000000001"
    hex_data = contract_fn.build_transaction(
        {"from": dummy, "gas": 200_000, "gasPrice": 0}
    )["data"]
    return bytes.fromhex(hex_data[2:] if hex_data.startswith("0x") else hex_data)


def _send_direct(w3: Web3, contract_fn, from_addr: str, private_key: str, gas: int = 100_000) -> str:
    """Trimite direct o tranzacție din contul EOA."""
    tx = contract_fn.build_transaction({
        "from":     from_addr,
        "nonce":    w3.eth.get_transaction_count(from_addr),
        "gas":      gas,
        "gasPrice": w3.eth.gas_price,
        "chainId":  137,
    })
    signed  = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt["status"] != 1:
        raise RuntimeError(f"Tx reverted: {tx_hash.hex()}")
    return tx_hash.hex()


def _execute_via_safe(
    w3: Web3,
    safe_address: str,
    to: str,
    data: bytes,
    eoa_address: str,
    private_key: str,
) -> str:
    """
    Execută un call arbitrar prin Gnosis Safe.
    Folosește semnătura pre-validată (tip 0x01): Safe verifică că msg.sender == owner,
    fără EIP-712 — funcționează când EOA este singurul owner cu threshold=1.
    """
    safe = w3.eth.contract(address=Web3.to_checksum_address(safe_address), abi=_SAFE_ABI)

    # Pre-validated signature: bytes32(owner_address) + bytes32(0) + bytes1(0x01)
    owner_b32 = bytes(12) + bytes.fromhex(eoa_address.lower().removeprefix("0x"))
    signature = owner_b32 + bytes(32) + bytes([0x01])  # 65 bytes total

    tx = safe.functions.execTransaction(
        Web3.to_checksum_address(to),
        0,          # value
        data,       # calldata
        0,          # operation: CALL
        0,          # safeTxGas
        0,          # baseGas
        0,          # gasPrice
        _ZERO_ADDR, # gasToken
        _ZERO_ADDR, # refundReceiver
        signature,
    ).build_transaction({
        "from":     eoa_address,
        "nonce":    w3.eth.get_transaction_count(eoa_address),
        "gas":      250_000,
        "gasPrice": w3.eth.gas_price,
        "chainId":  137,
    })
    signed  = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt["status"] != 1:
        raise RuntimeError(f"Safe.execTransaction reverted: {tx_hash.hex()}")
    return tx_hash.hex()


def ensure_collateral_approved() -> None:
    """
    Verifică și activează toate aprobările necesare pentru trading pe Polymarket CLOB:
      1. USDC.approve(CLOB_Exchange, MAX_UINT256)
      2. CTF.setApprovalForAll(CLOB_Exchange, true)
      3. CTF.setApprovalForAll(NegRisk_Exchange, true)
      4. CTF.setApprovalForAll(NegRisk_Adapter, true)

    Conturi EOA: trimite tranzacțiile direct.
    Conturi proxy (funder ≠ EOA): execută prin interfața Gnosis Safe.
    """
    client = get_clob_client()

    # Treaptă 1: metoda nativă SDK dacă există
    if hasattr(client, "approve_collateral"):
        try:
            logger.info("[Allowance] Încerc client.approve_collateral()...")
            tx = client.approve_collateral()
            logger.info("[Allowance] approve_collateral() confirmat: tx=%s", tx)
            _sync_clob_server(client)
            return
        except Exception as exc:
            logger.warning("[Allowance] approve_collateral() eșuat: %s — trec la web3.", exc)

    # Treaptă 2: web3 direct pe Polygon
    rpc = settings.POLYGON_RPC_URL
    logger.info("[Allowance] Conectare la Polygon RPC: %s", rpc)
    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        logger.error("[Allowance] Imposibil de conectat la RPC %s.", rpc)
        return

    eoa    = Account.from_key(settings.PRIVATE_KEY).address
    funder = Web3.to_checksum_address(settings.FUNDER_ADDRESS)
    is_proxy = eoa.lower() != funder.lower()
    owner    = funder if is_proxy else eoa

    logger.info(
        "[Allowance] Owner pentru aprobări: %s | mod=%s",
        owner, "proxy/Safe" if is_proxy else "EOA direct",
    )

    usdc = w3.eth.contract(address=_USDC_ADDR, abi=_ERC20_ABI)
    ctf  = w3.eth.contract(address=_CTF_ADDR,  abi=_ERC1155_ABI)

    # Citire stare on-chain
    usdc_allowance = usdc.functions.allowance(
        Web3.to_checksum_address(owner), _EXCHANGE_ADDR
    ).call() / _USDC_DECIMALS

    ctf_missing = [
        op for op in _CTF_OPERATORS
        if not ctf.functions.isApprovedForAll(
            Web3.to_checksum_address(owner),
            Web3.to_checksum_address(op),
        ).call()
    ]

    logger.info("[Allowance] USDC allowance: %.2f | CTF operatori lipsă: %d", usdc_allowance, len(ctf_missing))

    if usdc_allowance > 0 and not ctf_missing:
        logger.info("[Allowance] Toate aprobările active. Sincronizez CLOB server...")
        _sync_clob_server(client)
        return

    def _send(contract_fn):
        """Trimite via EOA sau prin Safe, în funcție de tipul contului."""
        if is_proxy:
            to   = contract_fn.address
            data = _calldata(contract_fn)
            return _execute_via_safe(w3, funder, to, data, eoa, settings.PRIVATE_KEY)
        return _send_direct(w3, contract_fn, eoa, settings.PRIVATE_KEY)

    if usdc_allowance == 0:
        logger.info("[Allowance] Trimit USDC.approve(EXCHANGE, MAX) de la %s...", owner)
        try:
            tx = _send(usdc.functions.approve(_EXCHANGE_ADDR, _MAX_UINT256))
            logger.info("[Allowance] USDC.approve confirmat: tx=%s", tx)
        except Exception as exc:
            logger.error("[Allowance] USDC.approve eșuat: %s", exc, exc_info=True)

    for op in ctf_missing:
        logger.info("[Allowance] Trimit CTF.setApprovalForAll(%s, true) de la %s...", op, owner)
        try:
            tx = _send(ctf.functions.setApprovalForAll(op, True))
            logger.info("[Allowance] CTF.setApprovalForAll(%s) confirmat: tx=%s", op, tx)
        except Exception as exc:
            logger.error("[Allowance] CTF.setApprovalForAll(%s) eșuat: %s", op, exc, exc_info=True)

    _sync_clob_server(client)
