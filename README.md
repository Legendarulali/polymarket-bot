# Polymarket AI Trading Bot

Bot autonom de trading pentru [Polymarket](https://polymarket.com) care:

1. preia piețele active din Gamma API,
2. le filtrează după volum, fereastră de probabilitate și clustering pe eveniment,
3. culege știri recente (Google News RSS, opțional NewsAPI + X),
4. cere o decizie de trading unui LLM (Groq / Llama 3.1),
5. plasează ordine limită BUY prin Polymarket CLOB, cu verificări de risc,
6. opțional, închide pozițiile pe take-profit / stop-loss.

> ⚠️ **Disclaimer**: acest bot tranzacționează bani reali pe baza deciziilor unui
> LLM. Folosește-l doar cu fonduri pe care îți permiți să le pierzi și pornește
> cu limite stricte (`MAX_ACTIVE_POSITIONS`, `USDC_APPROVAL_USD`).

## Arhitectură

```
main.py                  bucla principală: fetch piețe → filtre → AI → ordin
├── config/settings.py   configurare din variabile de mediu
└── src/
    ├── clob_client.py        client Polymarket CLOB (EOA sau proxy Gnosis Safe)
    ├── execution.py          plasare/anulare ordine limită (tick size, sizing)
    ├── position_manager.py   exit pe take-profit / stop-loss (opt-in)
    ├── reconciliation.py     balanță + poziții deschise (Data API)
    ├── allowance.py          aprobări on-chain USDC/CTF (direct sau via Safe)
    ├── ai_analyser.py        decizie de trading via Groq LLM
    └── news_monitor.py       agregare știri (RSS / NewsAPI / X)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # completează valorile
python main.py
```

Toate variabilele de mediu sunt documentate în [.env.example](.env.example).
Minimul obligatoriu: `POLYMARKET_PRIVATE_KEY` și `FUNDER_ADDRESS`.

### Deploy (Railway / Docker)

Repo-ul include `Dockerfile` (multi-stage, non-root) și `railway.json`.
Secretele se setează ca variabile de mediu în platformă — fișierul `.env`
nu este inclus niciodată în imagine (vezi `.dockerignore`).

## Controale de risc

| Control | Variabilă | Default |
|---|---|---|
| Poziții simultane maxime | `MAX_ACTIVE_POSITIONS` | 5 |
| Plafon aprobare USDC on-chain | `USDC_APPROVAL_USD` | 0 (nelimitat — recomandat de setat) |
| Plafon gas Polygon | `MAX_GAS_PRICE_GWEI` | 500 |
| Toleranță slippage la BUY | `SLIPPAGE_TOLERANCE` | 0.05 |
| Take-profit | `TAKE_PROFIT_PCT` | 0 (dezactivat — câștigătorii se țin până la rezoluție) |
| Stop-loss | `STOP_LOSS_PCT` | 0.30 (−30% ≈ preț sub ~0.5 de la o intrare tipică → teză invalidată) |

În plus, hardcodate în `main.py`: volum minim $10k, fereastră de probabilitate
YES (0.65, 0.90), confidence minim 0.70, max 1–2 poziții per eveniment,
deduplicare pe token și verificare de balanță înainte de fiecare ordin.

## Note de securitate

- **Cheia privată** controlează fondurile. Folosește un portofel dedicat doar
  botului, cu sume limitate. Pe termen lung, mută cheia într-un secret manager.
- **Aprobarea USDC** este nelimitată by default (standardul Polymarket).
  Setează `USDC_APPROVAL_USD` ca să limitezi expunerea în cazul compromiterii
  contractului de exchange — cu costul unor re-aprobări periodice (gas).
- **RPC**: folosește un endpoint autentificat (Alchemy/Infura) în loc de
  `polygon-rpc.com` — un RPC public poate servi date false sau cenzura tranzacții.
- **Clientul CLOB** este `py-clob-client` — clientul **oficial** Polymarket
  (anterior se folosea fork-ul neoficial `py-clob-client-v2`).
- **Decizia LLM** se bazează pe titluri de piață și știri — conținut extern
  nesigur. Promptul are gardă anti-injection, output-ul e validat strict
  (doar YES/NO, confidence clamped), iar filtrele de risc rulează independent
  de decizia AI.

### Rollback migrare client CLOB

Clientul oficial suportă semnături `0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE`.
Fostul fork folosea `3` (POLY_1271/EIP-1271), normalizat acum automat la `2`.
Pentru majoritatea conturilor proxy Polymarket (Gnosis Safe) tipul `2`
funcționează identic. **La primul deploy după migrare, urmărește logurile**:
dacă ordinele sunt respinse repetat cu `maker address not allowed` sau erori
de semnătură, contul tău necesită fluxul EIP-1271, pe care clientul oficial
nu îl implementează — în acest caz fă rollback la commit-ul dinaintea
migrării (`git revert`) sau repune `py-clob-client-v2==1.0.1` în
`requirements.txt` împreună cu vechiul `src/clob_client.py`.

## Teste

```bash
pip install pytest
python -m pytest -q
```

Dependențele grele (`py_clob_client_v2`, `web3`, `openai`, …) sunt mock-uite în
`tests/conftest.py`, deci testele rulează fără mediu de producție. CI-ul
(GitHub Actions) rulează lint (`ruff`), testele și un audit de vulnerabilități
(`pip-audit`) la fiecare push/PR.
