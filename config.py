import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Universe of liquid US equities the bot will consider.
DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "V", "MA", "JNJ", "UNH", "XOM", "CVX", "WMT", "HD",
]

# Trading parameters
TRADE_INTERVAL_MINUTES = 15           # how often the bot evaluates signals
LOOKBACK_DAYS = 365 * 3               # history pulled for training
FEATURE_LOOKBACK = 60                 # bars of context per prediction

# Risk parameters
MAX_POSITIONS = 5                     # cap concurrent open positions
MAX_POSITION_PCT = 0.15               # max % of equity per single position
DAILY_MAX_LOSS_PCT = 0.03             # halt trading if daily loss exceeds this
STOP_LOSS_PCT = 0.02                  # per-trade stop loss
TAKE_PROFIT_PCT = 0.04                # per-trade take profit
MIN_SIGNAL_CONFIDENCE = 0.58          # probability threshold to enter

# ML
MODEL_PATH = MODELS_DIR / "signal_model.joblib"
PREDICTION_HORIZON_BARS = 4           # predict direction this many bars ahead
TRAIN_BAR_TIMEFRAME = "1Hour"

# Flask
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production-please")
SESSION_TYPE = "filesystem"
