from ktg.interfaces import Strategy, Event


class ExecutionBridge(Strategy):
    __script_name__ = "execution_bridge"

    SIGNAL_STREAM = "!order_signals"
    EVENTS_STREAM = "!order_events"

    VALID_INTENTS = ("init", "increase", "decrease", "exit", "none")

    def __init__(self, **kwargs):
        self.default_buy_algo = str(kwargs.get("default_buy_algo", "") or "").strip()
        self.default_sell_algo = str(kwargs.get("default_sell_algo", "") or "").strip()

    @classmethod
    def on_strategy_start(cls, md, service, account):
        from ktg.interfaces import LastLoadedStrategy
        params = LastLoadedStrategy.parameters
        service.info(f"ExecutionBridge parameters: {params}")
        service.info(
            f"ExecutionBridge streams: in={cls.SIGNAL_STREAM} out={cls.EVENTS_STREAM}"
        )

    @classmethod
    def is_symbol_qualified(cls, symbol, md, service, account):
        return True

    @classmethod
    def using_extra_symbols(cls, symbol, md, service, account):
        return False

    @classmethod
    def register_event_streams(cls, md, service, account):
        return {cls.SIGNAL_STREAM: "on_signal_event"}

    def on_start(self, md, order, service, account):
        service.clear_event_triggers()
        service.add_event_trigger(
            [md.symbol],
            [Event.ACK, Event.FILL, Event.REJECT, Event.CANCEL, Event.CANCEL_REJECT],
        )

    # ==================== Inbound signal handler ====================

    def on_signal_event(self, event, md, order, service, account, userdata):
        """Place an order on behalf of an upstream signal producer."""
        data = event.field
        symbol = event.symbol or str(data.get("symbol", ""))
        if not symbol:
            service.warn("Signal dropped: missing symbol")
            return

        side = str(data.get("side", "")).lower()
        if side not in ("buy", "sell"):
            service.warn(f"Signal {symbol} dropped: invalid side '{side}'")
            return

        intent = str(data.get("intent", ""))
        if intent not in self.VALID_INTENTS:
            service.warn(f"Signal {symbol} dropped: invalid intent '{intent}'")
            return

        try:
            order_quantity = int(data.get("order_quantity", 0) or 0)
        except (TypeError, ValueError):
            service.warn(f"Signal {symbol} dropped: invalid order_quantity")
            return

        try:
            limit_price = float(data.get("limit_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            limit_price = 0.0
        if limit_price > 0.0:
            limit_price = max(limit_price, 0.01)

        algo_override = str(data.get("algo_override", "") or "").strip()

        try:
            user_tag = int(data.get("user_tag", 0) or 0)
        except (TypeError, ValueError):
            user_tag = 0

        if side == "buy":
            algo_uuid = algo_override or self.default_buy_algo
        else:
            algo_uuid = algo_override or self.default_sell_algo
        if not algo_uuid:
            service.error(
                f"Signal {symbol} dropped: no algo configured for side={side} "
                f"(set default_{side}_algo or pass algo_override)"
            )
            return

        order_id = None
        try:
            if intent == "exit":
                if side == "buy":
                    order_id = order.algo_buy(
                        symbol, algo_uuid, intent,
                        price=limit_price, user_tag=user_tag,
                    )
                else:
                    order_id = order.algo_sell(
                        symbol, algo_uuid, intent,
                        price=limit_price, user_tag=user_tag,
                    )
            else:
                if order_quantity <= 0:
                    service.warn(
                        f"Signal {symbol} dropped: intent={intent} requires "
                        f"positive order_quantity"
                    )
                    return
                if side == "buy":
                    order_id = order.algo_buy(
                        symbol, algo_uuid, intent,
                        order_quantity=order_quantity,
                        price=limit_price,
                        user_tag=user_tag,
                    )
                else:
                    order_id = order.algo_sell(
                        symbol, algo_uuid, intent,
                        order_quantity=order_quantity,
                        price=limit_price,
                        user_tag=user_tag,
                    )
        except Exception as ex:
            service.error(
                f"Order placement failed for {symbol} side={side} intent={intent}: {ex}"
            )
            return

        if order_id:
            service.info(
                f"Routed {side.upper()} {symbol} qty={order_quantity} "
                f"intent={intent} limit={limit_price:.4f} user_tag={user_tag} "
                f"order_id={order_id}"
            )

    # ==================== Outbound event republishing ====================

    def _publish_order_event(self, service, event_type, event, extra=None):
        payload = {
            "event_type": event_type,
            "symbol": str(event.symbol),
            "order_id": str(event.order_id),
            "clordid": str(event.clordid),
            "instruction_id": str(event.instruction_id),
            "pair_id": str(event.pair_id),
            "expected_direction": str(event.expected_direction),
            "intent": str(event.intent),
            "state": str(event.state),
            "order_algorithm": str(event.order_algorithm),
            "script_id": str(event.script_id),
            "script_class_name": str(event.script_class_name),
            "security_type": str(event.security_type),
            "price": float(event.price),
            "shares": float(event.shares),
            "contracts": float(event.contracts),
            "account_id": int(event.account_id),
            "user_tag": int(event.user_tag),
            "event_timestamp": int(event.timestamp),
        }
        if extra:
            payload.update(extra)
        published = service.publish_event(self.EVENTS_STREAM, event.symbol, payload)
        if not published:
            service.warn(
                f"publish_event failed: {event_type} {event.symbol} "
                f"order_id={event.order_id} (verify ACL on {self.EVENTS_STREAM})"
            )

    def on_ack(self, event, md, order, service, account):
        self._publish_order_event(service, "ack", event)

    def on_fill(self, event, md, order, service, account):
        extra = {
            "fill_id": int(event.fill_id),
            "fee": float(event.fee),
            "fee_currency": str(event.fee_currency),
            "commissions_per_share": float(event.commissions_per_share),
            "actual_direction": str(event.actual_direction),
        }
        self._publish_order_event(service, "fill", event, extra)

    def on_reject(self, event, md, order, service, account):
        self._publish_order_event(
            service, "reject", event, {"reason": str(event.reason)}
        )

    def on_cancel(self, event, md, order, service, account):
        self._publish_order_event(
            service, "cancel", event, {"reason": str(event.reason)}
        )

    def on_cancel_reject(self, event, md, order, service, account):
        self._publish_order_event(
            service, "cancel_reject", event, {"reason": str(event.reason)}
        )
