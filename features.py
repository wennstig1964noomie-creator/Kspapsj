"""Technical feature engineering for the ML signal model."""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi_14", "stoch_k", "stoch_d",
    "macd", "macd_signal", "macd_diff",
    "ema_fast_dist", "ema_slow_dist", "adx",
    "bb_pct", "bb_width", "atr_pct",
    "obv_chg", "mfi_14",
    "vol_ratio", "hl_range",
]


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features from OHLCV bars.

    `bars` must have columns: open, high, low, close, volume — indexed by time.
    """
    df = bars.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["ret_12"] = close.pct_change(12)

    df["rsi_14"] = RSIIndicator(close, window=14).rsi()
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    macd = MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    ema_fast = EMAIndicator(close, window=12).ema_indicator()
    ema_slow = EMAIndicator(close, window=50).ema_indicator()
    df["ema_fast_dist"] = (close - ema_fast) / ema_fast
    df["ema_slow_dist"] = (close - ema_slow) / ema_slow
    df["adx"] = ADXIndicator(high, low, close, window=14).adx()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_pct"] = atr / close

    df["obv_chg"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume().pct_change(5)
    df["mfi_14"] = MFIIndicator(high, low, close, volume, window=14).money_flow_index()

    df["vol_ratio"] = volume / volume.rolling(20).mean()
    df["hl_range"] = (high - low) / close

    return df


def make_training_frame(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add forward-return label for supervised training."""
    feats = build_features(bars)
    future_ret = feats["close"].shift(-horizon) / feats["close"] - 1
    feats["label"] = (future_ret > 0).astype(int)
    feats["future_ret"] = future_ret
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["label"])
    return feats
