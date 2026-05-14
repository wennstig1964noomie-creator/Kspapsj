import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "V", "MA", "JNJ", "UNH", "XOM", "CVX", "WMT", "HD",
    "COST", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "NFLX",
]
REGIME_SYMBOL = "SPY"   # used for the market-regime filter

TRADE_INTERVAL_MINUTES = 15
BARS_TIMEFRAME = "1Hour"
BARS_LOOKBACK_DAYS = 60   # enough for 200-bar EMA on hourly data

# Strategy entry threshold (signal score is 0..1)
MIN_SIGNAL_SCORE = 0.55

# Risk
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.15    # cap per-name exposure
RISK_PER_TRADE_PCT = 0.01  # risk 1% of equity per trade (at full confidence)
DAILY_MAX_LOSS_PCT = 0.03  # halt if equity drops >3% intraday
ATR_STOP_MULT = 2.0        # stop = price - 2*ATR
ATR_TARGET_MULT = 4.0      # target = price + 4*ATR (2:1 reward:risk)

# Flask
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production-please")
SESSION_TYPE = "filesystem"
