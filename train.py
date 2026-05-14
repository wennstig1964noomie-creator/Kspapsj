"""Train the signal model from historical Alpaca bars.

Usage:
    APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... python train.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from alpaca_client import AlpacaClient, AlpacaCredentials
from config import DEFAULT_SYMBOLS, LOOKBACK_DAYS, TRAIN_BAR_TIMEFRAME
from ml_model import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--timeframe", default=TRAIN_BAR_TIMEFRAME)
    args = parser.parse_args()

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        log.error("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")
        return 1

    client = AlpacaClient(AlpacaCredentials(api_key=key, api_secret=secret, paper=True))

    bars_by_symbol = {}
    for sym in args.symbols:
        log.info("Fetching %s …", sym)
        try:
            df = client.get_bars(sym, args.timeframe, days=args.days)
            if df.empty:
                log.warning("  no bars for %s", sym)
                continue
            bars_by_symbol[sym] = df
            log.info("  %d bars", len(df))
        except Exception as e:
            log.exception("  failed: %s", e)

    if not bars_by_symbol:
        log.error("No data fetched. Aborting.")
        return 1

    model = train(bars_by_symbol)
    model.save()
    log.info("Saved model (val AUC=%.4f).", model.auc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
