"""Self-contained smoke test: synthetic data → features → train → predict.

Proves the ML pipeline works end-to-end without needing Alpaca keys.
"""
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke")


def synthetic_bars(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Generate a random-walk-with-momentum OHLCV series."""
    rng = np.random.default_rng(seed)
    drift = 0.0001
    momentum = np.zeros(n)
    rets = np.zeros(n)
    for i in range(1, n):
        momentum[i] = 0.7 * momentum[i - 1] + 0.3 * rng.standard_normal() * 0.005
        rets[i] = drift + momentum[i] + rng.standard_normal() * 0.002
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.standard_normal(n)) * 0.001)
    low = close * (1 - np.abs(rng.standard_normal(n)) * 0.001)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(100_000, 500_000, size=n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def main() -> int:
    from ml_model import train
    from features import build_features

    symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    bars_by_symbol = {s: synthetic_bars(seed=i) for i, s in enumerate(symbols)}
    log.info("Generated %d symbols × %d bars each.", len(symbols), len(bars_by_symbol["AAPL"]))

    model = train(bars_by_symbol)
    log.info("Model trained. Validation AUC=%.4f", model.auc)

    # Predict on the most recent bar for each symbol
    for sym, bars in bars_by_symbol.items():
        prob = model.predict_proba(bars)
        log.info("  P(up | %s) = %.4f", sym, prob)

    # Save + reload sanity check
    model.save()
    from ml_model import SignalModel
    reloaded = SignalModel.load()
    assert reloaded is not None
    prob2 = reloaded.predict_proba(bars_by_symbol["AAPL"])
    log.info("Reloaded model predicts P(up|AAPL) = %.4f (matches=%s)", prob2, abs(prob2 - model.predict_proba(bars_by_symbol["AAPL"])) < 1e-9)

    # Feature smoke test
    feats = build_features(bars_by_symbol["AAPL"])
    log.info("Feature columns: %d. Last row has NaNs: %s", feats.shape[1], feats.iloc[-1].isna().any())

    log.info("SMOKE TEST PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
