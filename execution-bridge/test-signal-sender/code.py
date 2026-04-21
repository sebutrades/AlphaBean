from ktg.interfaces import Strategy, Event


class TestSignalSender(Strategy):
    __script_name__ = "test_signal_sender"

    # Event streams - must match execution_bridge streams
    SIGNAL_STREAM = "!order_signals"
    EVENTS_STREAM = "!order_events"

    # Symbols we will test with
    TEST_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "V", "MA",
    ]

    def __init__(self, **kwargs):
        self.test_qty = kwargs.get("test_qty", 10)
        self.stop_pct = kwargs.get("stop_pct", 0.5)
        self.buy_algo = str(kwargs.get("buy_algo", ""))
        self.sell_algo = str(kwargs.get("sell_algo", ""))

    @classmethod
    def on_strategy_start(cls, md, service, account):
        from ktg.interfaces import LastLoadedStrategy
        params = LastLoadedStrategy.parameters
        service.info("=" * 60)
        service.info("TEST SIGNAL SENDER - Starting")
        service.info(f"Parameters: {params}")
        service.info(f"Test symbols: {cls.TEST_SYMBOLS}")
        service.info("=" * 60)

    @classmethod
    def is_symbol_qualified(cls, symbol, md, service, account):
        # Only run on our test symbols
        return symbol in cls.TEST_SYMBOLS

    @classmethod
    def using_extra_symbols(cls, symbol, md, service, account):
        return False

    @classmethod
    def register_event_streams(cls, md, service, account):
        # Listen for order events (ack, fill, reject, cancel) from execution_bridge
        return {cls.EVENTS_STREAM: "on_order_event"}

    def on_start(self, md, order, service, account):
        # Per-symbol state
        self.buy_sent = False
        self.buy_filled = False
        self.sell_sent = False
        self.sell_filled = False
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.fill_count = 0
        self.ack_count = 0
        self.reject_count = 0
        self.order_tag = hash(md.symbol) & 0x7FFFFFFF  # unique tag per symbol

        # Subscribe to NBBO so we can monitor stop losses after fill
        service.clear_event_triggers()
        service.add_event_trigger(
            [md.symbol],
            [Event.NBBO_PRICE, Event.MINUTE_BAR],
        )

        # Set up a timer to send the buy signal 10 seconds after start
        # This gives the execution_bridge time to start too
        now = service.system_time
        delay = service.time_interval(0, 0, 10)  # 10 seconds
        service.add_time_trigger(now + delay, 0, timer_id="send_buy")

        # Safety timer: sell everything after 5 minutes
        safety_delay = service.time_interval(0, 5, 0)  # 5 minutes
        service.add_time_trigger(now + safety_delay, 0, timer_id="safety_exit")

        service.info(
            f"[{md.symbol}] on_start: "
            f"qty={self.test_qty} stop={self.stop_pct}% "
            f"tag={self.order_tag} "
            f"buy timer in 10s, safety exit in 5min"
        )

    # ==================== Timer handler ====================

    def on_timer(self, event, md, order, service, account):
        timer_id = getattr(event, "timer_id", None)

        if timer_id == "send_buy":
            self._send_buy_signal(md, service)

        elif timer_id == "safety_exit":
            if self.buy_filled and not self.sell_sent:
                service.info(
                    f"[{md.symbol}] SAFETY EXIT: 5min timeout reached, "
                    f"sending sell signal"
                )
                self._send_sell_signal(md, service, reason="safety_timeout")
            elif not self.buy_filled:
                service.info(
                    f"[{md.symbol}] SAFETY EXIT: buy never filled, nothing to sell"
                )

    # ==================== Signal publishing ====================

    def _send_buy_signal(self, md, service):
        if self.buy_sent:
            return

        bid = md.L1.bid
        ask = md.L1.ask
        if bid <= 0 or ask <= 0:
            service.warn(
                f"[{md.symbol}] Cannot send buy: invalid quotes "
                f"bid={bid} ask={ask}"
            )
            return

        # Send a market buy signal to execution_bridge via event stream
        payload = {
            "symbol": md.symbol,
            "side": "buy",
            "intent": "init",
            "order_quantity": self.test_qty,
            "limit_price": 0.0,  # market order
            "algo_override": self.buy_algo,
            "user_tag": self.order_tag,
        }

        service.info("=" * 50)
        service.info(f"[{md.symbol}] SENDING BUY SIGNAL")
        service.info(f"  Side:     BUY")
        service.info(f"  Qty:      {self.test_qty}")
        service.info(f"  Intent:   init")
        service.info(f"  Type:     MARKET (limit=0)")
        service.info(f"  Algo:     {self.buy_algo}")
        service.info(f"  Tag:      {self.order_tag}")
        service.info(f"  Bid/Ask:  ${bid:.2f} / ${ask:.2f}")
        service.info("=" * 50)

        published = service.publish_event(
            self.SIGNAL_STREAM, md.symbol, payload
        )
        if published:
            self.buy_sent = True
            service.info(f"[{md.symbol}] Buy signal PUBLISHED to {self.SIGNAL_STREAM}")
        else:
            service.error(
                f"[{md.symbol}] Buy signal FAILED to publish "
                f"(check ACL on {self.SIGNAL_STREAM})"
            )

    def _send_sell_signal(self, md, service, reason="manual"):
        if self.sell_sent:
            return
        if not self.buy_filled:
            service.warn(f"[{md.symbol}] Cannot sell: buy not filled yet")
            return

        bid = md.L1.bid
        ask = md.L1.ask

        payload = {
            "symbol": md.symbol,
            "side": "sell",
            "intent": "exit",
            "order_quantity": 0,  # exit intent handles qty automatically
            "limit_price": 0.0,  # market order
            "algo_override": self.sell_algo,
            "user_tag": self.order_tag,
        }

        service.info("=" * 50)
        service.info(f"[{md.symbol}] SENDING SELL SIGNAL ({reason})")
        service.info(f"  Side:     SELL")
        service.info(f"  Intent:   exit (full position)")
        service.info(f"  Type:     MARKET (limit=0)")
        service.info(f"  Algo:     {self.sell_algo}")
        service.info(f"  Entry:    ${self.entry_price:.2f}")
        if self.stop_price > 0:
            service.info(f"  Stop:     ${self.stop_price:.2f}")
        service.info(f"  Bid/Ask:  ${bid:.2f} / ${ask:.2f}")
        if self.entry_price > 0 and bid > 0:
            pnl = (bid - self.entry_price) * self.test_qty
            pnl_pct = (bid - self.entry_price) / self.entry_price * 100
            service.info(f"  Est P&L:  ${pnl:.2f} ({pnl_pct:+.2f}%)")
        service.info(f"  Reason:   {reason}")
        service.info("=" * 50)

        published = service.publish_event(
            self.SIGNAL_STREAM, md.symbol, payload
        )
        if published:
            self.sell_sent = True
            service.info(f"[{md.symbol}] Sell signal PUBLISHED to {self.SIGNAL_STREAM}")
        else:
            service.error(
                f"[{md.symbol}] Sell signal FAILED to publish "
                f"(check ACL on {self.SIGNAL_STREAM})"
            )

    # ==================== Order event handler ====================

    def on_order_event(self, event, md, order, service, account, userdata):
        """Receive ack/fill/reject/cancel events from execution_bridge."""
        data = event.field
        event_type = str(data.get("event_type", "unknown"))
        symbol = str(data.get("symbol", ""))
        order_id = str(data.get("order_id", ""))
        intent = str(data.get("intent", ""))
        state = str(data.get("state", ""))
        price = float(data.get("price", 0.0))
        shares = float(data.get("shares", 0.0))
        tag = int(data.get("user_tag", 0))

        service.info("-" * 50)
        service.info(f"[{symbol}] ORDER EVENT RECEIVED: {event_type.upper()}")
        service.info(f"  Order ID:  {order_id}")
        service.info(f"  Intent:    {intent}")
        service.info(f"  State:     {state}")
        service.info(f"  Price:     ${price:.4f}")
        service.info(f"  Shares:    {shares:.0f}")
        service.info(f"  Tag:       {tag}")

        if event_type == "ack":
            self.ack_count += 1
            service.info(f"  -> Order ACKNOWLEDGED (ack #{self.ack_count})")

        elif event_type == "fill":
            self.fill_count += 1
            service.info(f"  -> Order FILLED (fill #{self.fill_count})")

            if not self.buy_filled and intent == "init":
                # This is our buy fill
                self.buy_filled = True
                self.entry_price = price
                self.stop_price = price * (1.0 - self.stop_pct / 100.0)

                service.info(f"  -> BUY FILL CONFIRMED")
                service.info(f"     Entry price: ${self.entry_price:.4f}")
                service.info(f"     Stop price:  ${self.stop_price:.4f} ({self.stop_pct}% below)")
                service.info(f"     Position:    {self.test_qty} shares")
                service.info(f"     Monitoring NBBO for stop loss trigger...")

            elif intent == "exit":
                self.sell_filled = True
                exit_price = price
                if self.entry_price > 0:
                    pnl = (exit_price - self.entry_price) * self.test_qty
                    pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
                    service.info(f"  -> SELL FILL CONFIRMED")
                    service.info(f"     Entry:  ${self.entry_price:.4f}")
                    service.info(f"     Exit:   ${exit_price:.4f}")
                    service.info(f"     P&L:    ${pnl:.2f} ({pnl_pct:+.2f}%)")
                    if pnl >= 0:
                        service.info(f"     Result: WIN")
                    else:
                        service.info(f"     Result: LOSS (stop hit)")

        elif event_type == "reject":
            self.reject_count += 1
            reason = str(data.get("reason", "unknown"))
            service.error(f"  -> Order REJECTED: {reason}")

        elif event_type == "cancel":
            reason = str(data.get("reason", "unknown"))
            service.warn(f"  -> Order CANCELLED: {reason}")

        elif event_type == "cancel_reject":
            reason = str(data.get("reason", "unknown"))
            service.warn(f"  -> Cancel REJECTED: {reason}")

        service.info("-" * 50)

    # ==================== Price monitoring for stop loss ====================

    def on_nbbo_price(self, event, md, order, service, account):
        """Monitor price for stop loss trigger after buy fill."""
        if not self.buy_filled or self.sell_sent:
            return

        bid = event.bid
        if bid <= 0:
            return

        # Check if stop loss is hit
        if bid <= self.stop_price:
            service.info("!" * 50)
            service.info(
                f"[{md.symbol}] STOP LOSS TRIGGERED! "
                f"Bid=${bid:.4f} <= Stop=${self.stop_price:.4f}"
            )
            service.info("!" * 50)
            self._send_sell_signal(md, service, reason="stop_loss")

    def on_minute_bar(self, event, md, order, service, account, bar):
        """Log position status every minute bar."""
        if not self.buy_filled or self.sell_filled:
            return

        bid = md.L1.bid
        if bid <= 0 or self.entry_price <= 0:
            return

        pnl = (bid - self.entry_price) * self.test_qty
        pnl_pct = (bid - self.entry_price) / self.entry_price * 100
        dist_to_stop = (bid - self.stop_price) / bid * 100

        service.info(
            f"[{md.symbol}] POSITION UPDATE: "
            f"entry=${self.entry_price:.2f} "
            f"bid=${bid:.2f} "
            f"pnl=${pnl:.2f}({pnl_pct:+.2f}%) "
            f"stop=${self.stop_price:.2f}({dist_to_stop:.2f}% away)"
        )

    # ==================== Lifecycle ====================

    def on_finish(self, md, order, service, account):
        service.info("=" * 50)
        service.info(f"[{md.symbol}] TEST COMPLETE")
        service.info(f"  Buy sent:    {self.buy_sent}")
        service.info(f"  Buy filled:  {self.buy_filled}")
        service.info(f"  Sell sent:   {self.sell_sent}")
        service.info(f"  Sell filled: {self.sell_filled}")
        service.info(f"  Entry:       ${self.entry_price:.4f}")
        service.info(f"  Stop:        ${self.stop_price:.4f}")
        service.info(f"  Acks:        {self.ack_count}")
        service.info(f"  Fills:       {self.fill_count}")
        service.info(f"  Rejects:     {self.reject_count}")
        service.info("=" * 50)
