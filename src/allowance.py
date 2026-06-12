import logging
from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from py_clob_client_v2 import BalanceAllowanceParams, AssetType
from src.clob_client import get_clob_client, get_signature_type
from config.settings import settings

logger = logging.getLogger(__name__)

# Polymarket Polygon mainnet — sistemul curent (CTF Exchange V2 + colateral pUSD).
# Adresele sunt cele din py-clob-client-v2 config (chain 137); vechiul exchange
# V1 (0x4bFb…) și USDC.e nu mai sunt folosite la decontarea ordinelor.
_COLLATERAL_ADDR      = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")  # pUSD
_CTF_ADDR             = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
_EXCHANGE_V2          = Web3.to_checksum_address("0xE111180000d2663C0091e4f400237545B87B996B")
_NEG_RISK_EXCHANGE_V2 = Web3.to_checksum_address("0xe2222d279d744050d28e00520010520000310F59")
_NEG_RISK_ADAPTER     = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")
_ZERO_ADDR            = "0x0000000000000000000000000000000000000000"
_MAX_UINT256          = 2**256 - 1
_USDC_DECIMALS        = 1_000_000
# Re-aprobă colateralul când allowance-ul scade sub valoarea unui ordin standard.
_REAPPROVE_THRESHOLD_USD = 5.0

# Cheltuitorii care necesită aprobare de colateral (pUSD.approve) și operatorii
# care necesită CTF.setApprovalForAll.
_COLLATERAL_SPENDERS = [_EXCHANGE_V2, _NEG_RISK_EXCHANGE_V2, _NEG_RISK_ADAPTER]
_CTF_OPERATORS       = [_EXCHANGE_V2, _NEG_RISK_EXCHANGE_V2, _NEG_RISK_ADAPTER]

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
    sig_type = get_signature_type()
    try:
        client.update_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig_type)
        )
        logger.info("[Allowance] CLOB server: balanță/allowance sincronizate (sig_type=%d).", sig_type)
    except Exception as exc:
        logger.warning("[Allowance] update_balance_allowance() eșuat: %s", exc)


def _checked_gas_price(w3: Web3) -> int:
    """Prețul curent de gas, cu refuz dacă depășește plafonul configurat."""
    gas_price = w3.eth.gas_price
    cap_gwei = settings.MAX_GAS_PRICE_GWEI
    if cap_gwei > 0 and gas_price > cap_gwei * 1e9:
        raise RuntimeError(
            f"Gas price {gas_price / 1e9:.1f} gwei peste plafonul de "
            f"{cap_gwei:.0f} gwei (MAX_GAS_PRICE_GWEI) — tranzacție amânată."
        )
    return gas_price


def _approval_amount() -> int:
    """Suma de aprobat pentru colateral (pUSD): plafonată via USDC_APPROVAL_USD sau nelimitată."""
    if settings.USDC_APPROVAL_USD > 0:
        return int(settings.USDC_APPROVAL_USD * _USDC_DECIMALS)
    logger.warning(
        "[Allowance] Aprobare colateral NELIMITATĂ (MAX_UINT256). Setează USDC_APPROVAL_USD "
        "pentru a limita expunerea în cazul compromiterii contractului de exchange."
    )
    return _MAX_UINT256


def _calldata(contract_fn) -> bytes:
    """Extrage calldata-ul ABI-encoded al unui ContractFunction fără a trimite tx."""
    # encode doar argumentele + selector, fără build_transaction: acela face
    # RPC calls (estimare gas / EIP-1559 din ultimul block) care pot eșua.
    return bytes.fromhex(contract_fn._encode_transaction_data()[2:])


def _send_direct(w3: Web3, contract_fn, from_addr: str, private_key: str, gas: int = 100_000) -> str:
    """Trimite direct o tranzacție din contul EOA."""
    tx = contract_fn.build_transaction({
        "from":     from_addr,
        "nonce":    w3.eth.get_transaction_count(from_addr),
        "gas":      gas,
        "gasPrice": _checked_gas_price(w3),
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
        "gasPrice": _checked_gas_price(w3),
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
    Verifică și activează toate aprobările necesare pentru trading pe Polymarket
    CLOB (sistemul V2 cu colateral pUSD):
      1. pUSD.approve(spender) pentru Exchange V2, NegRisk Exchange V2, NegRisk Adapter
      2. CTF.setApprovalForAll(operator, true) pentru aceiași trei operatori

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
    # Polygon e lanț POA: block-urile au extraData >32 bytes și web3 refuză să
    # le parseze fără acest middleware (eroarea "extraData is 324 bytes").
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
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

    pusd = w3.eth.contract(address=_COLLATERAL_ADDR, abi=_ERC20_ABI)
    ctf  = w3.eth.contract(address=_CTF_ADDR,        abi=_ERC1155_ABI)
    owner_cs = Web3.to_checksum_address(owner)

    def _read_missing() -> tuple[list, list]:
        """Citește on-chain ce aprobări lipsesc (colateral pUSD + CTF)."""
        # Cu aprobări mărginite, allowance-ul scade la fiecare fill — re-aprobă
        # înainte să coboare sub valoarea unui ordin, nu doar când ajunge la 0.
        collateral = [
            sp for sp in _COLLATERAL_SPENDERS
            if pusd.functions.allowance(owner_cs, sp).call() / _USDC_DECIMALS
            < _REAPPROVE_THRESHOLD_USD
        ]
        operators = [
            op for op in _CTF_OPERATORS
            if not ctf.functions.isApprovedForAll(owner_cs, op).call()
        ]
        return collateral, operators

    collateral_missing, ctf_missing = _read_missing()
    logger.info(
        "[Allowance] Aprobări pUSD lipsă: %d/%d | CTF operatori lipsă: %d/%d",
        len(collateral_missing), len(_COLLATERAL_SPENDERS),
        len(ctf_missing), len(_CTF_OPERATORS),
    )

    if not collateral_missing and not ctf_missing:
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

    amount = _approval_amount() if collateral_missing else 0
    for sp in collateral_missing:
        logger.info("[Allowance] Trimit pUSD.approve(%s, %s) de la %s...",
                    sp, "MAX" if amount == _MAX_UINT256 else f"{amount / _USDC_DECIMALS:.2f} USD", owner)
        try:
            tx = _send(pusd.functions.approve(sp, amount))
            logger.info("[Allowance] pUSD.approve(%s) confirmat: tx=%s", sp, tx)
        except Exception as exc:
            logger.warning(
                "[Allowance] On-chain approval failed, but skipping since manual "
                "setup might be active (pUSD.approve %s error: %s)", sp, exc,
            )

    for op in ctf_missing:
        logger.info("[Allowance] Trimit CTF.setApprovalForAll(%s, true) de la %s...", op, owner)
        try:
            tx = _send(ctf.functions.setApprovalForAll(op, True))
            logger.info("[Allowance] CTF.setApprovalForAll(%s) confirmat: tx=%s", op, tx)
        except Exception as exc:
            logger.warning(
                "[Allowance] On-chain approval failed, but skipping since manual "
                "setup might be active (CTF.setApprovalForAll %s error: %s)", op, exc,
            )

    # Verificare finală: dacă aprobările tot lipsesc după încercări, ordinele
    # vor fi respinse de CLOB în buclă — semnalează clar pentru operator.
    try:
        final_collateral_missing, final_ctf_missing = _read_missing()
        if final_collateral_missing or final_ctf_missing:
            logger.error(
                "[Allowance] APROBĂRI INCOMPLETE după toate încercările: "
                "pUSD spenders lipsă: %s | CTF operatori lipsă: %s. "
                "Ordinele pot fi respinse până la aprobare manuală din Polymarket UI.",
                [str(sp) for sp in final_collateral_missing] or "niciunul",
                [str(op) for op in final_ctf_missing] or "niciunul",
            )
        else:
            logger.info("[Allowance] Verificare finală OK: toate aprobările active.")
    except Exception as exc:
        logger.warning("[Allowance] Verificarea finală a aprobărilor a eșuat: %s", exc)

    _sync_clob_server(client)
