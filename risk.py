"""Risk management helpers (ATR-based exits, vol-targeted sizing)."""
from __future__ import annotations

from dataclasses import dataclass

from config import (
    ATR_STOP_MULT,
    ATR_TARGET_MULT,
    DAILY_MAX_LOSS_PCT,
    MAX_POSITION_PCT,
    MAX_POSITIONS,
    RISK_PER_TRADE_PCT,
)


@dataclass
class TradePlan:
    symbol: str
    qty: int
    entry: float
    stop: float
    target: float


def plan_trade(symbol: str, price: float, atr: float, equity: float, score: float) -> TradePlan | None:
    """Volatility-targeted position sizing.

    Risk a fixed % of equity per trade. Position size = (equity * risk%) / stop_distance.
    Then cap at MAX_POSITION_PCT of equity. Confidence scales the risk fraction.
    """
    if atr <= 0 or price <= 0:
        return None
    stop_distance = ATR_STOP_MULT * atr
    target_distance = ATR_TARGET_MULT * atr
    stop = price - stop_distance
    target = price + target_distance
    if stop <= 0:
        return None

    # Score in [0,1] → risk fraction in [0, RISK_PER_TRADE_PCT]
    risk_fraction = RISK_PER_TRADE_PCT * max(0.0, min(1.0, score))
    dollar_risk = equity * risk_fraction
    qty_by_risk = int(dollar_risk // stop_distance)
    qty_by_cap = int((equity * MAX_POSITION_PCT) // price)
    qty = max(0, min(qty_by_risk, qty_by_cap))
    if qty <= 0:
        return None
    return TradePlan(symbol=symbol, qty=qty, entry=price, stop=stop, target=target)


def trading_halted(account) -> tuple[bool, str]:
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
