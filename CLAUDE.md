# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Autonomous trading bot for **Polymarket** (Polygon prediction markets). Continuous loop: pull active markets from the Gamma API → filter (volume, probability window, per-event clustering) → gather news → ask a Groq LLM whether to trade → risk-check → place limit BUY orders on the Polymarket CLOB. Optionally exits positions on take-profit/stop-loss and reports portfolio P/L to Discord. **No dry-run mode** — `python main.py` with a funded wallet trades real money.

`README.md` (Romanian) is the operator guide; this file is the engineering map. Source comments and most log messages are in **Romanian** — match that style when editing.

## Commands

```bash
pip install -r requirements.txt          # deps (Python 3.11+; 3.13 works locally)

python -m pytest -q                      # full suite
python -m pytest tests/test_bot.py::TestAnalyzedHistory          # one class
python -m pytest tests/test_bot.py::TestSettings::test_chain_id_is_polygon   # one test

python main.py                           # run the LIVE bot (needs a populated .env)
docker build -t polymarket-bot .         # container; deploys on Railway via railway.json
```

CI (GitHub Actions) gates each push/PR on `ruff check`, `pytest`, and `pip-audit` — run `ruff check .` before pushing if it's installed.

## Configuration

Env-driven via `config/settings.py` (loads `.env`). It **raises at import time** if a required var is missing, so importing any module needs the env populated (tests seed it in `conftest.py`). Full list in `.env.example` / README; the non-obvious ones:

- **Required:** `POLYMARKET_PRIVATE_KEY`, `FUNDER_ADDRESS`.
- `GROQ_API_KEY` — without it `analyse_market` returns `trade=False`, so the bot never trades.
- **Proxy accounts:** `POLYMARKET_API_KEY` / `_SECRET` / `_PASSPHRASE` switch to explicit-creds mode + `signature_type=3`. `/balance-allowance` returns 0 for proxies → set `OVERRIDE_BALANCE` to the real pUSD balance to unblock trading.
- **Risk knobs** (hardcoded thresholds live at the top of `main.py`): `MAX_ACTIVE_POSITIONS` (5), `SLIPPAGE_TOLERANCE` (0.05), `TAKE_PROFIT_PCT` (0 = off), `STOP_LOSS_PCT` (0.30), `USDC_APPROVAL_USD`, `MAX_GAS_PRICE_GWEI`.
- **Discord:** `DISCORD_WEBHOOK_URL` (empty = off), `PNL_REPORT_INTERVAL_HOURS` (1 — each interval sends the two-message report). The per-scan markets list is **never** sent to Discord; it stays in the console logs.
- **Scanning:** `FOOTBALL_ONLY` (true by default — process **only** individual football matches and drop outrights/general markets entirely; set false to restore the legacy YES/NO flow on all market types), `MARKET_SCAN_LIMIT` (200, markets paginated per loop), `MIN_VOLUME_USD` (10000 — lower it if today's low-volume matches get filtered out), `PROBABILITY_MIN`/`PROBABILITY_MAX` (0.45/0.95 — applied **per 1X2 outcome** (a Polymarket outcome's price *is* its YES probability); a match is analyzed only if at least one of its 1/X/2 legs lands in the window. Note: a draw's price rarely exceeds 0.45, so the floor favors 1/2 — lower `PROBABILITY_MIN` to fish for draws/underdogs), `GAMMA_ORDER`/`GAMMA_ASCENDING` (Gamma sort field, e.g. `volumeNum` or `startDate` ascending to surface imminent matches; empty by default, auto-falls-back to unordered if the API rejects it), `GAMMA_TAG_IDS` (optional CSV of numeric Gamma tag ids for a soccer/World-Cup deep-scan; off by default — needs the real tag id verified against the live API), `HISTORY_FILE` (override the `history.json` path, e.g. a Railway volume since `/app` is ephemeral + root-only).

## Architecture

`main.py` orchestrates; everything in `src/` is a focused helper. Startup: boot reconciliation → warm CLOB client → ensure on-chain approvals → loop. **Each loop iteration:** `get_account_state` → balance/exposure gates → maybe send the hourly P/L report → fetch markets (paginated up to `MARKET_SCAN_LIMIT`, deduped by id) → drop those already in the persistent history → shuffle, then **stable-sort match markets to the front** (`_is_match_market`) so head-to-head/imminent games are analyzed before per-loop limits/throttle hit → `process_market` each (throttled only after a real LLM call) → `check_exit_conditions`.

**`process_market` pipeline** (constants at the top of `main.py`): classify (logged as `[Tip: Meci/Outright/General]`, by `_classify_market` from title/slug; match takes precedence over outright) → global-exposure gate → **football-only gate** (`FOOTBALL_ONLY`: anything not `match` is dropped as `non-meci`) → volume ≥ `MIN_VOLUME_USD` → then the route branches:
  - **Match → 1X2 path** (`_process_match_1x2`): `parse_match_market` (`src/football.py`) splits the market's parallel `outcomes`/`outcomePrices`/`clobTokenIds` arrays into `1`=home / `X`=draw / `2`=away legs (draw detected by name, teams matched to the `A vs B` title), logged as `[Meci] A vs B | 1 (A)=.. | X (Draw)=.. | 2 (B)=..` → per-leg probability window (candidates = legs in `PROBABILITY_MIN`–`PROBABILITY_MAX`) → news → `analyse_match` returns a `1`/`X`/`2`/`NONE` pick (validated against the eligible legs) → the chosen leg's token/price flow into `_execute_trade`.
  - **Non-1X2 (binary Yes/No match, or any market when `FOOTBALL_ONLY` is off) → legacy path** (`_process_legacy_binary`): YES-price window → per-event cap via `_extract_event_id` (skipped for `match`) → news → `analyse_market` (YES/NO) → `_execute_trade`.

  `_execute_trade` is the shared tail: `confidence ≥ 0.70` is gated by the caller, then balance / duplicate-token / slippage checks → `place_limit_order` → `notify_trade`. Each filter bumps a `_scan_stats` counter; the loop logs a per-scan `[Scan] Rezultat filtrare ...` summary so you can see *why* markets are skipped → record to `history.json`.

**Module map:**
- `config/settings.py` — `Settings` singleton + root logging config.
- `src/clob_client.py` — singleton `ClobClient` via `get_clob_client()` (never construct directly). Explicit-creds mode (proxy, sig_type 3) vs derive mode. `get_signature_type()` centralizes the signature type used elsewhere.
- `src/execution.py` — `place_limit_order` / `cancel_order` / `compute_order_size`. **The `Side` enum is defined here** (string-based), not imported from the SDK. Size = `max($5/price, 2 shares)`; **a SELL must pass `size=` explicitly** or it sells a price-derived quantity instead of the held amount. Supports multiple tick sizes.
- `src/reconciliation.py` — `get_account_state()` → `{pusd_balance, allowance, active_positions}`. Balance/allowance are **micro-USD (÷1e6)**; positions come from the Data API enriched with P/L fields, `outcome`, and `event_id` (from `eventSlug`). `compute_portfolio_summary()` builds the Discord report.
- `src/position_manager.py` — `check_exit_conditions()`: opt-in take-profit/stop-loss → SELL at best bid; has its own 30-min per-token exit cooldown.
- `src/allowance.py` — one-time on-chain approvals (pUSD + CTF) targeting the **V2** exchanges; EOA sends tx directly, proxy routes through `Safe.execTransaction`.
- `src/football.py` — `parse_match_market(market)` maps a Gamma match into 1X2 legs (`1`/`X`/`2` with team names, prices, token ids); `extract_teams`, `format_legs`. Pure Python (no new deps → nothing to mock in `conftest`). Returns `None` for binary Yes/No or unparseable markets so the caller can fall back to the legacy path. **Assumes the 1X2 is a single market with parallel `outcomes`/`outcomePrices`/`clobTokenIds` arrays** — if Polymarket ever splits a game into separate per-outcome binary markets, those degrade to the legacy YES/NO path (still tradeable, just not grouped/logged as one 1X2).
- `src/ai_analyser.py` — Groq via the OpenAI-compatible SDK (`llama-3.1-8b-instant`) → strict JSON decision. `analyse_market` (binary YES/NO) and `analyse_match` (1X2 — picks `1`/`X`/`2`/`NONE`, validated against the eligible legs so it can't pick an out-of-window outcome). Prompt-injection hardened (untrusted title/news wrapped + sanitized); tenacity retries 4×.
- `src/news_monitor.py` — `fetch_free_news` (Google News RSS, no key, used by the loop) vs `fetch_latest_breaking_news` (NewsAPI + X).
- `src/notifications.py` — Discord webhooks, three message types: `notify_trade` (BUY/SELL, with outcome), `notify_portfolio` (financial status only — "💵 Status Financiar": totals + P/L), `notify_positions` (open-positions inventory — "📦 Poziții Active", one numbered line per position; empty → "Niciuna."). All **no-op without a webhook and never raise**, safe to call inline. `_maybe_send_pnl_report` (in `main.py`) sends **both** messages — portfolio **then** positions — once per `PNL_REPORT_INTERVAL_HOURS`, never per loop. There is **no** per-scan markets webhook (removed by design — the scanned-markets list lives only in the console logs).

## Cross-cutting concepts

- **Account-mode duality is the central complexity.** EOA (`key == funder`) vs proxy (key signs for a Safe/Magic wallet that holds the funds). It changes auth (`clob_client.py`), approval execution (`allowance.py` Safe path), and balance reporting (the `OVERRIDE_BALANCE` workaround). Touch any of those with both paths in mind.
- **`py-clob-client-v2` is required, not a typo.** Polymarket migrated to V2 orders / pUSD collateral; the official `py-clob-client` is rejected by the server with `invalid order version`. Don't "upgrade" to it.
- **Persistent analysis history (`history.json`):** loaded first in `main()`, it filters markets analyzed within `_CACHE_TTL_SECONDS` (4h cooldown) out of the fetched batch *before* processing, so Groq isn't re-called for them across restarts. Recorded via `_record_analyzed` after each LLM call; expired entries pruned on load; the file is gitignored.

## Testing

- **No `__init__.py` anywhere** — the project runs from the repo root so `config`, `src`, and `main` import as top-level modules. Keep new code root-relative.
- `tests/conftest.py` **mocks every heavy/native dependency** (`py_clob_client_v2`, `openai`, `web3`/`eth_account`, `tenacity`, `defusedxml`, `httpx`) and seeds required env vars before any project import — the suite needs only pytest installed. Adding a module that imports a new third-party package means mocking it there, or collection breaks.
- `test_bot.py` imports the **real** helpers from `main` / `src` (not local copies), so changing those functions is covered directly by the suite.

## Gotchas

- Units differ by layer: CLOB balance/allowance are micro-USD (÷1e6); market and order prices are plain 0–1 probabilities.
- Event clustering correlates Gamma markets to open positions via the **event slug** — `_extract_event_id` (markets) and `reconciliation` (positions) must both keep reading `eventSlug`, or the per-event cap silently stops limiting anything.
- The git tree contains a junk file whose *name* is a full Windows path (`C:...tests__init__.py`, with an escaped colon). It is not a real package file — ignore it.
