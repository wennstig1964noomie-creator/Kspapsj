"""Live trading loop. Scores symbols with the rule-based strategy and trades."""
from __future__ import annotations

import logging
from datetime import datetime

from alpaca_client import AlpacaClient, AlpacaCredentials
from config import (
    BARS_LOOKBACK_DAYS,
    BARS_TIMEFRAME,
    DEFAULT_SYMBOLS,
    MIN_SIGNAL_SCORE,
    REGIME_SYMBOL,
)
from risk import can_open_new_position, plan_trade, trading_halted
from strategy import evaluate, market_regime_ok

log = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, creds: AlpacaCredentials, symbols: list[str] | None = None):
        self.client = AlpacaClient(creds)
        self.symbols = symbols or DEFAULT_SYMBOLS
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
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_status": self.last_status,
            "recent_trades": self.recent_trades[-25:],
        }

    def run_once(self) -> None:
        self.last_run = datetime.utcnow()

        if not self.client.is_market_open():
            self.last_status = "Market closed; skipping."
            return

        account = self.client.account()
        halted, reason = trading_halted(account)
        if halted:
            self.last_status = f"Halted: {reason}"
            log.warning(self.last_status)
            return

        # Market regime check — only trade when broad market is in an uptrend.
        try:
            spy_bars = self.client.get_bars(REGIME_SYMBOL, "1Day", days=400)
            if not market_regime_ok(spy_bars):
                self.last_status = f"Regime off: {REGIME_SYMBOL} below 200-day SMA."
                return
        except Exception as e:
            log.exception("regime check failed: %s", e)
            self.last_status = f"Regime check failed: {e}"
            return

        positions = {p.symbol: p for p in self.client.positions()}
        equity = float(account.equity)
        opened = 0
        evaluated = 0
        candidates: list[tuple[float, str, object]] = []

        # Score everything first, then enter the best candidates.
        for symbol in self.symbols:
            if symbol in positions:
                continue
            try:
                bars = self.client.get_bars(symbol, BARS_TIMEFRAME, days=BARS_LOOKBACK_DAYS)
                if len(bars) < 200:
                    continue
                sig = evaluate(bars)
                evaluated += 1
                if sig is None or sig.score < MIN_SIGNAL_SCORE:
                    continue
                candidates.append((sig.score, symbol, sig))
            except Exception as e:
                log.exception("scoring %s failed: %s", symbol, e)

        candidates.sort(key=lambda t: t[0], reverse=True)

        for score, symbol, sig in candidates:
            if not can_open_new_position(len(positions) + opened):
                break
            plan = plan_trade(symbol, sig.price, sig.atr, equity, score)
            if plan is None:
                continue
            try:
                order = self.client.submit_bracket_buy(
                    symbol=plan.symbol,
                    qty=plan.qty,
                    take_profit=plan.target,
                    stop_loss=plan.stop,
                )
                opened += 1
                self.recent_trades.append({
                    "time": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "qty": plan.qty,
                    "entry": plan.entry,
                    "stop": plan.stop,
                    "target": plan.target,
                    "confidence": round(score, 4),
                    "reasons": ", ".join(sig.reasons),
                    "order_id": str(order.id),
                })
                log.info("Opened %s qty=%d score=%.2f", symbol, plan.qty, score)
            except Exception as e:
                log.exception("order submit failed for %s: %s", symbol, e)

        self.last_status = (
            f"Evaluated {evaluated}, {len(candidates)} passed threshold, opened {opened}."
        )
