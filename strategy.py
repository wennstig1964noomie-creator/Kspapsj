"""Rule-based multi-factor trading strategy. No training required.

Combines several well-known quantitative signals:
  • Market regime filter (only buy when SPY > 200-day SMA)
  • Trend (price above rising 50 and 200 EMAs)
  • Momentum (60-bar return)
  • Mean reversion in an uptrend (pullback RSI 35-55)
  • Trend strength (ADX > 20)
  • Volume confirmation
  • MACD histogram positive
Position sizing is volatility-adjusted (ATR-based), exits are ATR-based
bracket orders so stop distance adapts to each symbol's noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange


@dataclass
class Signal:
    score: float          # 0..1, higher = stronger buy
    price: float          # latest close
    atr: float            # ATR(14) used for stop/target sizing
    reasons: list[str]    # human-readable reasons that fired


def market_regime_ok(spy_bars: pd.DataFrame) -> bool:
    """True when SPY is above its 200-bar SMA — broad-market uptrend filter."""
    if spy_bars is None or len(spy_bars) < 200:
        return False
    sma200 = spy_bars["close"].rolling(200).mean().iloc[-1]
    return bool(spy_bars["close"].iloc[-1] > sma200)


def evaluate(bars: pd.DataFrame) -> Signal | None:
    """Score a single symbol. Returns None if it doesn't pass hard filters."""
    if len(bars) < 200:
        return None

    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]

    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    rsi = RSIIndicator(close, window=14).rsi()
    adx = ADXIndicator(high, low, close, window=14).adx()
    macd_hist = MACD(close).macd_diff()
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()

    price = float(close.iloc[-1])
    e50 = float(ema50.iloc[-1])
    e200 = float(ema200.iloc[-1])
    e50_prev = float(ema50.iloc[-5])
    e200_prev = float(ema200.iloc[-5])
    rsi_v = float(rsi.iloc[-1])
    adx_v = float(adx.iloc[-1])
    macd_v = float(macd_hist.iloc[-1])
    atr_v = float(atr.iloc[-1])
    vol_ratio = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
    momentum = float(close.iloc[-1] / close.iloc[-60] - 1)  # ~60-bar return

    # Hard filters — must all pass.
    if not (price > e50 > e200):
        return None
    if not (e50 > e50_prev and e200 > e200_prev):  # both EMAs rising
        return None
    if rsi_v > 70:  # don't chase overbought
        return None
    if np.isnan(atr_v) or atr_v <= 0:
        return None

    reasons: list[str] = ["price > EMA50 > EMA200 (uptrend)"]
    score = 0.0

    # Momentum contribution (0..0.25)
    mom_score = float(np.clip(momentum / 0.10, 0, 1)) * 0.25
    score += mom_score
    if mom_score > 0.05:
        reasons.append(f"momentum +{momentum:.1%}")

    # Pullback in uptrend — RSI 35..55 is the sweet spot.
    if 35 <= rsi_v <= 55:
        score += 0.20
        reasons.append(f"healthy pullback (RSI {rsi_v:.0f})")
    elif 55 < rsi_v <= 65:
        score += 0.10
        reasons.append(f"momentum continuation (RSI {rsi_v:.0f})")

    # Trend strength
    if adx_v >= 25:
        score += 0.20
        reasons.append(f"strong trend (ADX {adx_v:.0f})")
    elif adx_v >= 20:
        score += 0.10
        reasons.append(f"trending (ADX {adx_v:.0f})")

    # MACD histogram positive and recently expanding
    macd_prev = float(macd_hist.iloc[-2])
    if macd_v > 0 and macd_v > macd_prev:
        score += 0.15
        reasons.append("MACD hist rising")
    elif macd_v > 0:
        score += 0.08

    # Volume confirmation
    if vol_ratio >= 1.2:
        score += 0.10
        reasons.append(f"volume {vol_ratio:.1f}× avg")
    elif vol_ratio >= 0.8:
        score += 0.05

    # Proximity to recent highs (breakout-ish)
    high_50 = float(close.rolling(50).max().iloc[-1])
    if price >= high_50 * 0.98:
        score += 0.10
        reasons.append("near 50-bar high")

    return Signal(score=min(score, 1.0), price=price, atr=atr_v, reasons=reasons)
