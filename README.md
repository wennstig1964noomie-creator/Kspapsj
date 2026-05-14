# AI Alpaca Trading Bot

A research framework for an automated equities trader on [Alpaca](https://alpaca.markets).
A LightGBM classifier scores short-horizon direction from technical features; a Flask
dashboard lets users paste their API keys and start/stop the bot.

> **No bot is guaranteed to be profitable.** Markets are non-stationary; a model that
> backtests well can lose money live. Paper-trade first, monitor closely, and only
> commit capital you can afford to lose. This software is provided "as is" without warranty.

## What's in the box

| File | Purpose |
| --- | --- |
| `features.py` | Builds ~20 technical indicators (RSI, MACD, Bollinger, ADX, MFI, OBV, ATR, returns…) |
| `ml_model.py` | LightGBM binary classifier with time-ordered train/val split |
| `train.py` | CLI to pull history from Alpaca and train the model |
| `alpaca_client.py` | Wrapper around `alpaca-py` for bars + orders |
| `risk.py` | Position sizing (confidence-scaled), stop-loss, daily-loss kill switch |
| `trading_bot.py` | Evaluation loop: pull bars → score → place bracket order |
| `app.py` + `templates/` | Flask UI for key entry, start/stop, monitoring |
| `config.py` | Knobs: universe, risk caps, thresholds, intervals |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Train (uses your Alpaca paper keys to pull historical bars)
export APCA_API_KEY_ID=...
export APCA_API_SECRET_KEY=...
python train.py

# 2. Run the dashboard
python app.py
# → open http://localhost:5000, paste keys, start the bot
```

## How it works

1. **Universe:** ~20 large-cap US equities + major ETFs (editable in `config.py`).
2. **Features:** momentum, trend, volatility, and volume indicators on 1-hour bars.
3. **Label:** does close 4 bars from now exceed today's close? (binary)
4. **Model:** LightGBM with early-stopping on a time-ordered hold-out (no shuffling — that's the most common cause of fake backtest profits).
5. **Trade decision:** if `P(up) ≥ MIN_SIGNAL_CONFIDENCE` (default 0.58), open a bracket order — stop ~2% below entry, target ~4% above.
6. **Position sizing:** Kelly-lite — fraction of equity scales with how far the model's confidence is above 0.5, capped at 15% per name.
7. **Risk caps:** max 5 concurrent positions, halts trading for the day if equity drops >3%.

## Tuning honestly

- The defaults are *starting points*, not optimums. Re-train on fresh data, sweep `MIN_SIGNAL_CONFIDENCE`, `STOP_LOSS_PCT`, and `TAKE_PROFIT_PCT` against your own validation period.
- Watch for **distribution shift**: a model trained on a bull market can fail in a bear market. Re-train at least monthly.
- **Validation AUC** is shown in the dashboard. AUC near 0.5 means no edge — don't trade.
- Consider adding: per-symbol models, regime filters (e.g., trade only when SPY is above its 200-day SMA), or shorting logic.

## Security notes

- API keys live in server memory only for the duration of the session; nothing is written to disk.
- For multi-user deployment behind a public URL, put the app behind HTTPS and a real session backend, and rotate `FLASK_SECRET_KEY`.
- Paper-trading endpoint is the default and recommended.
