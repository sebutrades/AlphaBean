from ktg.interfaces import Strategy, Event


class AlphabeanOrderPlacer(Strategy):
    """Fire-and-forget order placer for AlphaBean live scanner.

    Started via: kti forward start <guid> --symbol AAPL --account 203979
                   --params '{"side":"buy","qty":10,"intent":"init"}'

    Places the order immediately in on_start, then stays alive to
    receive fill/reject events and log them.
    """

    __script_name__ = "alphabean_order_placer"

    # Default algo GUIDs (Market ARCA)
    DEFAULT_BUY_ALGO = "2b4fdc55-ff01-416e-a5ea-e1f1d4524c7d"
    DEFAULT_SELL_ALGO = "8fdee8fe-b772-46bd-b411-5544f7a0d917"

    def __init__(self, **kwargs):
        self.side = str(kwargs.get("side", "")).strip().lower()
        self.qty = int(kwargs.get("qty", 10))
        self.intent = str(kwargs.get("intent", "init")).strip().lower()
        self.algo = str(kwargs.get("algo", "")).strip()
        self.limit_price = float(kwargs.get("limit_price", 0.0))
        self.signal_id = str(kwargs.get("signal_id", "")).strip()

    @classmethod
    def is_symbol_qualified(cls, symbol, md, service, account):
        return False

    @classmethod
    def using_extra_symbols(cls, symbol, md, service, account):
        return False

    def on_start(self, md, order, service, account):
        service.clear_event_triggers()
        service.add_event_trigger(
            [md.symbol],
            [Event.ACK, Event.FILL, Event.REJECT, Event.CANCEL],
        )

        self.order_placed = False
        self.fill_count = 0

        # Resolve algo: use default based on side if not provided
        if not self.algo:
            if self.side == "buy":
                self.algo = self.DEFAULT_BUY_ALGO
            elif self.side == "sell":
                self.algo = self.DEFAULT_SELL_ALGO

        # Validate
        if self.side not in ("buy", "sell"):
            service.error(
                f"[AB] Invalid side='{self.side}' for {md.symbol} "
                f"signal_id={self.signal_id}"
            )
            return

        if self.intent not in ("init", "increase", "decrease", "exit", "none"):
            service.error(
                f"[AB] Invalid intent='{self.intent}' for {md.symbol}"
            )
            return

        if self.intent != "exit" and self.qty <= 0:
            service.error(
                f"[AB] Non-exit intent requires qty>0: got {self.qty}"
            )
            return

        # Place the order
        price = max(self.limit_price, 0.01) if self.limit_price > 0 else 0.0

        try:
            if self.intent == "exit":
                if self.side == "buy":
                    oid = order.algo_buy(
                        md.symbol, self.algo, self.intent,
                        price=price,
                    )
                else:
                    oid = order.algo_sell(
                        md.symbol, self.algo, self.intent,
                        price=price,
                    )
            else:
                if self.side == "buy":
                    oid = order.algo_buy(
                        md.symbol, self.algo, self.intent,
                        order_quantity=self.qty,
                        price=price,
                    )
                else:
                    oid = order.algo_sell(
                        md.symbol, self.algo, self.intent,
                        order_quantity=self.qty,
                        price=price,
                    )

            self.order_placed = True
            service.info(
                f"[AB] ORDER SENT: {self.side.upper()} {md.symbol} "
                f"qty={self.qty} intent={self.intent} "
                f"limit={price:.4f} signal={self.signal_id} oid={oid}"
            )
        except Exception as ex:
            service.error(
                f"[AB] ORDER FAILED: {self.side} {md.symbol} "
                f"qty={self.qty} err={ex}"
            )

    def on_ack(self, event, md, order, service, account):
        service.info(
            f"[AB] ACK: {event.symbol} oid={event.order_id} "
            f"signal={self.signal_id}"
        )

    def on_fill(self, event, md, order, service, account):
        self.fill_count += 1
        service.info(
            f"[AB] FILL: {event.symbol} {event.shares}@{event.price:.4f} "
            f"signal={self.signal_id} fill#{self.fill_count}"
        )

    def on_reject(self, event, md, order, service, account):
        service.error(
            f"[AB] REJECT: {event.symbol} reason={event.reason} "
            f"signal={self.signal_id}"
        )

    def on_cancel(self, event, md, order, service, account):
        service.info(
            f"[AB] CANCEL: {event.symbol} reason={event.reason} "
            f"signal={self.signal_id}"
        )

    def on_finish(self, md, order, service, account):
        if self.order_placed:
            service.info(
                f"[AB] Session end: {md.symbol} fills={self.fill_count} "
                f"signal={self.signal_id}"
            )
