import logging
from py_clob_client_v2 import ClobClient, Side
from src.execution import place_limit_order
from src.news_monitor import fetch_latest_breaking_news, analyze_sentiment

logger = logging.getLogger(__name__)

# Piața este considerată supraevaluată când suma probabilităților depășește acest prag.
_OVERPRICED_THRESHOLD = 1.03
# Ordinul de test are size fixă până când strategia este validată în producție.
_TEST_ORDER_SIZE = 1.0
# Fonduri minime necesare înainte de a plasa orice ordin.
_MIN_BALANCE_PUSD = 5.0


class BaseStrategy:
    def __init__(self, client: ClobClient) -> None:
        self.client = client

    def tick(self, account_state: dict) -> None:
        pass


class MultiOutcomeArbStrategy(BaseStrategy):
    """
    Detectează ineficiențe de preț în piețe multi-opțiune Polymarket.

    Logică: suma prețurilor YES ale tuturor opțiunilor ar trebui să fie ~1.0.
    Când suma depășește _OVERPRICED_THRESHOLD, cel puțin un token este supraevaluat
    și poate fi short-at (SELL) la prețul Ask curent.
    """

    def __init__(self, client: ClobClient, token_ids: list[str]) -> None:
        super().__init__(client)
        if not token_ids:
            raise ValueError("MultiOutcomeArbStrategy necesită cel puțin un token_id.")
        self.token_ids = token_ids

    def _get_best_ask(self, token_id: str) -> float | None:
        """Returnează cel mai bun Ask pentru un token, sau None dacă orderbook-ul e gol."""
        try:
            ob = self.client.get_order_book(token_id=token_id)
            asks = ob.asks if hasattr(ob, "asks") else ob.get("asks", [])
            if not asks:
                return None
            return round(float(asks[0].price if hasattr(asks[0], "price") else asks[0]["price"]), 4)
        except Exception as exc:
            logger.warning("[ArbStrategy] Eroare la orderbook token=%s: %s", token_id, exc)
            return None

    def _has_open_position(self, token_id: str, active_positions: list[dict]) -> bool:
        return any(pos.get("token_id") == token_id for pos in active_positions)

    def tick(self, account_state: dict) -> None:
        try:
            pusd_balance: float = account_state.get("pusd_balance", 0.0)
            active_positions: list[dict] = account_state.get("active_positions", [])

            # --- Pas 1: Colectează cel mai bun Ask pentru fiecare token ---
            token_prices: dict[str, float] = {}
            for token_id in self.token_ids:
                ask = self._get_best_ask(token_id)
                if ask is not None:
                    token_prices[token_id] = ask
                else:
                    logger.debug("[ArbStrategy] token=%s omis (orderbook gol).", token_id)

            if len(token_prices) < 2:
                logger.info(
                    "[ArbStrategy] Date insuficiente (%d/%d tokeni cu preț). Tick omis.",
                    len(token_prices),
                    len(self.token_ids),
                )
                return

            # --- Pas 2: Calculează suma probabilităților implicite ---
            prob_sum = round(sum(token_prices.values()), 4)

            logger.info(
                "[ArbStrategy] Suma probabilităților pieței: %.4f | Tokeni activi: %d/%d",
                prob_sum,
                len(token_prices),
                len(self.token_ids),
            )
            for tid, price in token_prices.items():
                logger.debug("[ArbStrategy]   token=%-30s ask=$%.4f", tid, price)

            # --- Pas 3: Verifică dacă piața este supraevaluată ---
            if prob_sum <= _OVERPRICED_THRESHOLD:
                return

            logger.warning(
                "[ArbStrategy] SUMA PIETEI SUPRAEVALUATA: %.4f > %.2f",
                prob_sum,
                _OVERPRICED_THRESHOLD,
            )

            # --- Pas 4: Identifică tokenul cel mai speculat (preț maxim) ---
            target_token_id = max(token_prices, key=lambda tid: token_prices[tid])
            target_price = token_prices[target_token_id]

            logger.info(
                "[ArbStrategy] Token țintă: %s | Ask=$%.4f",
                target_token_id,
                target_price,
            )

            # --- Pas 5: Verificări de risc înainte de execuție ---
            if pusd_balance < _MIN_BALANCE_PUSD:
                logger.warning(
                    "[ArbStrategy] Fonduri insuficiente (%.4f pUSD < %.1f pUSD minim). Ordin blocat.",
                    pusd_balance,
                    _MIN_BALANCE_PUSD,
                )
                return

            if self._has_open_position(target_token_id, active_positions):
                logger.info(
                    "[ArbStrategy] Poziție deschisă deja pe token=%s. Ordin nou blocat.",
                    target_token_id,
                )
                return

            # --- Pas 6: Alertă de oportunitate și execuție ---
            logger.warning(
                "[OPORTUNITATE] Suma pieței este %.4f. Se plasează ordin limită defensiv SELL "
                "pe token=%s | price=$%.4f | size=%.1f",
                prob_sum,
                target_token_id,
                target_price,
                _TEST_ORDER_SIZE,
            )

            result = place_limit_order(
                token_id=target_token_id,
                price=target_price,
                size=_TEST_ORDER_SIZE,
                side=Side.SELL,
            )

            if result.get("success"):
                logger.info(
                    "[ArbStrategy] Ordin plasat cu succes | order_id=%s",
                    result.get("order_id"),
                )
            else:
                logger.error(
                    "[ArbStrategy] Ordin respins de CLOB: %s",
                    result.get("error"),
                )

        except Exception as exc:
            logger.error("[ArbStrategy] Eroare neașteptată în tick(): %s", exc, exc_info=True)


class NewsCatalystStrategy(BaseStrategy):
    """
    Monitorizează știri și tweet-uri în timp real și plasează ordine când
    detectează cuvinte cheie relevante pentru o piață specifică.
    """

    def __init__(
        self,
        client: ClobClient,
        token_id: str,
        keywords: list[str],
        side: Side = Side.BUY,
        news_query: str = "",
        order_size: float = 1.0,
        min_balance: float = 5.0,
    ) -> None:
        super().__init__(client)
        if not token_id:
            raise ValueError("NewsCatalystStrategy necesită un token_id.")
        if not keywords:
            raise ValueError("NewsCatalystStrategy necesită cel puțin un keyword.")
        self.token_id = token_id
        self.keywords = keywords
        self.side = side
        self.news_query = news_query or " ".join(keywords[:3])
        self.order_size = float(order_size)
        self.min_balance = float(min_balance)

    def _get_best_ask(self) -> float | None:
        try:
            ob = self.client.get_order_book(token_id=self.token_id)
            asks = ob.asks if hasattr(ob, "asks") else ob.get("asks", [])
            if not asks:
                return None
            return round(float(asks[0].price if hasattr(asks[0], "price") else asks[0]["price"]), 4)
        except Exception as exc:
            logger.warning("[NewsStrategy] Eroare la orderbook token=%s: %s", self.token_id, exc)
            return None

    def _has_open_position(self, active_positions: list[dict]) -> bool:
        return any(pos.get("token_id") == self.token_id for pos in active_positions)

    def tick(self, account_state: dict) -> None:
        try:
            pusd_balance: float = account_state.get("pusd_balance", 0.0)
            active_positions: list[dict] = account_state.get("active_positions", [])

            # --- Pas 1: Fetch știri și tweet-uri ---
            texts = fetch_latest_breaking_news(query=self.news_query)
            if not texts:
                logger.debug("[NewsStrategy] Niciun item de știri disponibil.")
                return

            # --- Pas 2: Analiză de sentiment prin cuvinte cheie ---
            triggered_text: str | None = None
            for text in texts:
                if analyze_sentiment(text, self.keywords):
                    triggered_text = text
                    break

            if triggered_text is None:
                logger.debug(
                    "[NewsStrategy] Nicio știre relevantă pentru keywords=%s.", self.keywords
                )
                return

            logger.critical(
                "[NEWS TRIGGER] Știre critică detectată: \"%s\"",
                triggered_text[:200],
            )

            # --- Pas 3: Verificări de risc ---
            if pusd_balance < self.min_balance:
                logger.warning(
                    "[NewsStrategy] Fonduri insuficiente (%.4f pUSD < %.1f minim). Ordin blocat.",
                    pusd_balance,
                    self.min_balance,
                )
                return

            if self._has_open_position(active_positions):
                logger.info(
                    "[NewsStrategy] Poziție deschisă deja pe token=%s. Ordin nou blocat.",
                    self.token_id,
                )
                return

            # --- Pas 4: Preia Ask-ul curent pentru execuție agresivă ---
            ask_price = self._get_best_ask()
            if ask_price is None:
                logger.warning("[NewsStrategy] Orderbook gol pentru token=%s. Ordin blocat.", self.token_id)
                return

            logger.warning(
                "[NewsStrategy] Plasare ordin %s | token=%s | price=$%.4f | size=%.1f",
                self.side.name,
                self.token_id,
                ask_price,
                self.order_size,
            )

            result = place_limit_order(
                token_id=self.token_id,
                price=ask_price,
                size=self.order_size,
                side=self.side,
            )

            if result.get("success"):
                logger.info(
                    "[NewsStrategy] Ordin plasat cu succes | order_id=%s",
                    result.get("order_id"),
                )
            else:
                logger.error(
                    "[NewsStrategy] Ordin respins de CLOB: %s",
                    result.get("error"),
                )

        except Exception as exc:
            logger.error("[NewsStrategy] Eroare neașteptată în tick(): %s", exc, exc_info=True)
