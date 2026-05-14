"""Live trading loop. Pulls bars, scores them with the ML model, places trades."""
from __future__ import annotations

import logging
from datetime import datetime

from alpaca_client import AlpacaClient, AlpacaCredentials
from config import (
    DEFAULT_SYMBOLS,
    FEATURE_LOOKBACK,
    MIN_SIGNAL_CONFIDENCE,
    TRAIN_BAR_TIMEFRAME,
)
from ml_model import SignalModel
from risk import can_open_new_position, plan_trade, trading_halted

log = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, creds: AlpacaCredentials, symbols: list[str] | None = None):
        self.client = AlpacaClient(creds)
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.model: SignalModel | None = SignalModel.load()
        self.last_run: datetime | None = None
        self.last_status: str = "Idle"
        self.recent_trades: list[dict] = []

    def status(self) -> dict:
        acct = self.client.account()
        positions = self.client.positions()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "paper": self.client.creds.paper,
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "qty": int(p.qty),
                    "avg_entry": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            ],
            "model_loaded": self.model is not None,
            "model_auc": getattr(self.model, "auc", None),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_status": self.last_status,
            "recent_trades": self.recent_trades[-25:],
        }

    def run_once(self) -> None:
        """One evaluation cycle. Idempotent; safe to call on a schedule."""
        self.last_run = datetime.utcnow()
        if self.model is None:
            self.last_status = "No model trained yet. Run train.py first."
            log.warning(self.last_status)
            return

        if not self.client.is_market_open():
            self.last_status = "Market closed; skipping."
            return

        account = self.client.account()
        halted, reason = trading_halted(account)
        if halted:
            self.last_status = f"Halted: {reason}"
            log.warning(self.last_status)
            return

        positions = {p.symbol: p for p in self.client.positions()}
        equity = float(account.equity)
        opened = 0

        for symbol in self.symbols:
            try:
                if symbol in positions:
                    continue  # already in this name; bracket order handles exit
                if not can_open_new_position(len(positions) + opened):
                    break

                bars = self.client.get_bars(symbol, TRAIN_BAR_TIMEFRAME, days=30)
                if len(bars) < FEATURE_LOOKBACK:
                    continue

                prob_up = self.model.predict_proba(bars)
                if prob_up != prob_up:  # NaN
                    continue

                if prob_up < MIN_SIGNAL_CONFIDENCE:
                    continue

                price = float(bars["close"].iloc[-1])
                plan = plan_trade(symbol, price, equity, prob_up)
                if plan is None:
                    continue

                order = self.client.submit_bracket_buy(
                    symbol=plan.symbol,
                    qty=plan.qty,
                    take_profit=plan.target,
                    stop_loss=plan.stop,
                )
                opened += 1
                trade = {
                    "time": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "qty": plan.qty,
                    "entry": plan.entry,
                    "stop": plan.stop,
                    "target": plan.target,
                    "confidence": round(prob_up, 4),
                    "order_id": str(order.id),
                }
                self.recent_trades.append(trade)
                log.info("Opened %s", trade)
            except Exception as e:
                log.exception("Error evaluating %s: %s", symbol, e)

        self.last_status = f"Cycle complete. Opened {opened} new position(s)."
