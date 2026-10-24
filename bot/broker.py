"""Alpaca paper account wrapper. paper=True is forced; there is no live path."""
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from .risk import AccountState


class PaperBroker:
    def __init__(self, cfg):
        assert cfg.paper is True, "live trading is not supported"
        self.client = TradingClient(cfg.alpaca_key, cfg.alpaca_secret, paper=True)

    def market_open(self) -> bool:
        return self.client.get_clock().is_open

    def state(self, symbol: str) -> AccountState:
        acct = self.client.get_account()
        try:
            pos_value = float(self.client.get_open_position(symbol).market_value)
        except Exception:
            pos_value = 0.0
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        orders = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=midnight))
        return AccountState(float(acct.equity), float(acct.last_equity), pos_value, len(orders))

    def buy(self, symbol: str, notional: float):
        return self.client.submit_order(MarketOrderRequest(
            symbol=symbol, notional=notional, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))

    def close(self, symbol: str):
        return self.client.close_position(symbol)
