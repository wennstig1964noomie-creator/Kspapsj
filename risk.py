"""Risk management helpers."""
from __future__ import annotations

from dataclasses import dataclass

from config import (
    DAILY_MAX_LOSS_PCT,
    MAX_POSITION_PCT,
    MAX_POSITIONS,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)


@dataclass
class TradePlan:
    symbol: str
    qty: int
    entry: float
    stop: float
    target: float


def position_size(equity: float, price: float, confidence: float) -> int:
    """Kelly-lite sizing: scale position with model confidence, capped."""
    # confidence in [0.5, 1.0] → fraction in [0, MAX_POSITION_PCT]
    edge = max(0.0, min(1.0, (confidence - 0.5) * 2))
    fraction = MAX_POSITION_PCT * edge
    dollars = equity * fraction
    return max(0, int(dollars // price))


def plan_trade(symbol: str, price: float, equity: float, confidence: float) -> TradePlan | None:
    qty = position_size(equity, price, confidence)
    if qty <= 0:
        return None
    stop = price * (1 - STOP_LOSS_PCT)
    target = price * (1 + TAKE_PROFIT_PCT)
    return TradePlan(symbol=symbol, qty=qty, entry=price, stop=stop, target=target)


def trading_halted(account) -> tuple[bool, str]:
    """Halt trading if daily PnL breached or pattern-day-trader flagged."""
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    if last_equity > 0:
        daily_return = (equity - last_equity) / last_equity
        if daily_return <= -DAILY_MAX_LOSS_PCT:
            return True, f"Daily loss limit hit ({daily_return:.2%})"
    if getattr(account, "trading_blocked", False):
        return True, "Account trading_blocked=True"
    if getattr(account, "account_blocked", False):
        return True, "Account account_blocked=True"
    return False, ""


def can_open_new_position(open_count: int) -> bool:
    return open_count < MAX_POSITIONS
