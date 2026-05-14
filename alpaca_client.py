"""Thin wrapper around alpaca-py for trading + market data."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    GetAssetsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from alpaca.trading.enums import AssetClass, AssetStatus

log = logging.getLogger(__name__)


def _parse_timeframe(tf: str) -> TimeFrame:
    mapping = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[tf]


@dataclass
class AlpacaCredentials:
    api_key: str
    api_secret: str
    paper: bool = True


class AlpacaClient:
    def __init__(self, creds: AlpacaCredentials):
        self.creds = creds
        self.trading = TradingClient(creds.api_key, creds.api_secret, paper=creds.paper)
        self.data = StockHistoricalDataClient(creds.api_key, creds.api_secret)

    def account(self):
        return self.trading.get_account()

    def positions(self):
        return self.trading.get_all_positions()

    def get_position(self, symbol: str):
        try:
            return self.trading.get_open_position(symbol)
        except Exception:
            return None

    def is_market_open(self) -> bool:
        clock = self.trading.get_clock()
        return bool(clock.is_open)

    def tradable_symbols(self, symbols: list[str]) -> list[str]:
        req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        assets = self.trading.get_all_assets(req)
        wanted = set(symbols)
        return [a.symbol for a in assets if a.symbol in wanted and a.tradable]

    def get_bars(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        end = datetime.now(timezone.utc) - timedelta(minutes=20)  # avoid IEX 15min delay edge
        start = end - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_parse_timeframe(timeframe),
            start=start,
            end=end,
            adjustment="all",
        )
        resp = self.data.get_stock_bars(req)
        df = resp.df
        if df.empty:
            return df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df[["open", "high", "low", "close", "volume"]]

    def submit_bracket_buy(self, symbol: str, qty: int, take_profit: float, stop_loss: float):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            take_profit={"limit_price": round(take_profit, 2)},
            stop_loss={"stop_price": round(stop_loss, 2)},
        )
        return self.trading.submit_order(order)

    def submit_market_sell(self, symbol: str, qty: int):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.trading.submit_order(order)

    def close_all(self):
        return self.trading.close_all_positions(cancel_orders=True)
