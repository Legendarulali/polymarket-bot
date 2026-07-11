# Polymarket AI Trading Bot

Bot autonom de trading pentru [Polymarket](https://polymarket.com) care:

1. preia piețele active din Gamma API,
2. le filtrează după volum, fereastră de probabilitate și clustering pe eveniment,
3. culege știri recente (Google News RSS, opțional NewsAPI + X),
4. cere o decizie de trading unui LLM (Groq / gpt-oss-20b),
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
    ├── reconciliation.py     balanță + poziții + P/L (Data API), sumar portofoliu
    ├── allowance.py          aprobări on-chain USDC/CTF (direct sau via Safe)
    ├── ai_analyser.py        decizie de trading via Groq LLM
    ├── news_monitor.py       agregare știri (RSS / NewsAPI / X)
    └── notifications.py      alerte Discord (buy/sell) + raport orar P/L
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

Repo-ul include `Dockerfile` (multi-stage, non-root), `docker-compose.yml` și
`railway.json`. Secretele se setează ca variabile de mediu în platformă — fișierul
`.env` nu este inclus niciodată în imagine (vezi `.dockerignore`).

**Persistență (volum `/data`):** imaginea ține toată starea botului în `/data` —
`history.json` (istoricul analizelor + ultimele decizii AI + coada de oportunități
`next_in_line`) și logurile rotative (`/data/logs/bot.log`). Montează un volum acolo
și starea supraviețuiește distrugerii/reconstruirii/repornirii containerului:

```bash
docker compose up -d --build     # bind mount ./data:/data (vezi docker-compose.yml)
# sau, fără compose:
docker run -d --env-file .env -v "$(pwd)/data:/data" polymarket-bot
```

Pe Railway, atașează un volum montat pe `/data` — fără el starea rămâne efemeră
(comportamentul vechi), dar botul funcționează identic.

## Controale de risc

| Control | Variabilă | Default |
|---|---|---|
| Poziții simultane maxime | `MAX_ACTIVE_POSITIONS` | 5 |
| Plafon aprobare USDC on-chain | `USDC_APPROVAL_USD` | 0 (nelimitat — recomandat de setat) |
| Plafon gas Polygon | `MAX_GAS_PRICE_GWEI` | 500 |
| Toleranță slippage la BUY | `SLIPPAGE_TOLERANCE` | 0.05 |
| Take-profit | `TAKE_PROFIT_PCT` | 0 (dezactivat — câștigătorii se țin până la rezoluție) |
| Stop-loss | `STOP_LOSS_PCT` | 0.30 (−30% ≈ preț sub ~0.5 de la o intrare tipică → teză invalidată) |

În plus, configurabile din mediu (cu default-uri în `main.py`): volum minim $10k
(`MIN_VOLUME_USD`), o fereastră de probabilitate YES **unică** `[0.45, 0.95]`
(`PROBABILITY_MIN`/`PROBABILITY_MAX`, aceeași pentru meci/outright/general — îl ține
concentrat pe meciurile competitive și ignoră zgomotul de probabilitate mică; decizia
reală o ia LLM-ul + pragul de confidence 0.70), max 1–2 poziții per eveniment,
deduplicare pe token și verificare
de balanță înainte de fiecare ordin. Meciurile (head-to-head) sunt prioritizate la
începutul fiecărei scanări.

## Notificări Discord

Botul poate trimite pe un canal de Discord, prin webhook:

- **Alertă la fiecare BUY / SELL** — piață, preț, size, total, confidence (la BUY)
  sau motivul exit-ului (take-profit / stop-loss, la SELL).
- **Raport periodic** (implicit la fiecare oră), trimis ca **2 mesaje grupate**:
  1. **💵 Status Financiar** — valoare totală (cash + poziții), cash, P/L nerealizat
     și variația față de ora trecută / pornirea botului.
  2. **📦 Poziții Active** — inventarul pozițiilor deschise, numerotat (piață ·
     outcome · sumă investită · P/L), sau „Niciuna." dacă nu există poziții.

Lista piețelor scanate **nu** se trimite pe Discord — rămâne strict în consolă/loguri,
ca fluxul de Discord să conțină doar tranzacțiile și rapoartele periodice.

| Setare | Variabilă | Default |
|---|---|---|
| Webhook Discord | `DISCORD_WEBHOOK_URL` | gol (notificări dezactivate) |
| Interval raport P/L (ore) | `PNL_REPORT_INTERVAL_HOURS` | 1 |

Webhook-ul se creează din Discord: **Server Settings → Integrations → Webhooks →
New Webhook → Copy Webhook URL**. Notificările sunt complet opționale și izolate:
orice eroare de rețea către Discord e prinsă și logată, fără a afecta trading-ul.

## Note de securitate

- **Cheia privată** controlează fondurile. Folosește un portofel dedicat doar
  botului, cu sume limitate. Pe termen lung, mută cheia într-un secret manager.
- **Aprobarea USDC** este nelimitată by default (standardul Polymarket).
  Setează `USDC_APPROVAL_USD` ca să limitezi expunerea în cazul compromiterii
  contractului de exchange — cu costul unor re-aprobări periodice (gas).
- **RPC**: folosește un endpoint autentificat (Alchemy/Infura) în loc de
  `polygon-rpc.com` — un RPC public poate servi date false sau cenzura tranzacții.
- **Clientul CLOB** este `py-clob-client-v2` — un client comunitar, **necesar
  funcțional**: Polymarket a migrat la formatul de ordine V2 (CTF Exchange V2,
  colateral pUSD), iar clientul oficial `py-clob-client` (≤0.34.6) este
  respins de server cu `invalid order version` (verificat live 2026-06-12).
  Fiind neoficial, verifică proveniența înainte de orice bump de versiune.
- **Semnături**: ordinele folosesc `signature_type=3` (POLY_1271/EIP-1271),
  singurul confirmat funcțional pentru conturile proxy ale acestui bot;
  valoarea este fixată în `src/clob_client.py` când există chei API explicite.
- **Aprobările on-chain** vizează sistemul curent: pUSD
  (`0xC011a7E1…`) către Exchange V2 (`0xE1111800…`), NegRisk Exchange V2
  (`0xe2222d27…`) și NegRisk Adapter — nu vechiul exchange V1/USDC.e.
- **Decizia LLM** se bazează pe titluri de piață și știri — conținut extern
  nesigur. Promptul are gardă anti-injection, output-ul e validat strict
  (doar YES/NO, confidence clamped), iar filtrele de risc rulează independent
  de decizia AI.

## Teste

```bash
pip install pytest
python -m pytest -q
```

Dependențele grele (`py_clob_client_v2`, `web3`, `openai`, …) sunt mock-uite în
`tests/conftest.py`, deci testele rulează fără mediu de producție. CI-ul
(GitHub Actions) rulează lint (`ruff`), testele și un audit de vulnerabilități
(`pip-audit`) la fiecare push/PR.
