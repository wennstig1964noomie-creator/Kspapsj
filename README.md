# AI Alpaca Trading Bot

A 24/7 automated equities bot for [Alpaca](https://alpaca.markets). Paste your API keys
into the web dashboard and click **Start** — no training, no setup. The bot uses a
built-in rule-based multi-factor strategy.

> **No strategy is guaranteed to be profitable.** Markets change; what worked last year
> may not work this year. Paper-trade first. Only commit capital you can afford to lose.

## The built-in strategy

It's not a single signal but a weighted combination of well-established quant rules:

1. **Market regime filter** — only trades when SPY is above its 200-day SMA. Steps aside during confirmed downtrends.
2. **Trend filter** — symbol must trade above its 50-EMA, which must be above its 200-EMA, and both EMAs must be rising.
3. **Momentum** — recent 60-bar return contributes to the score.
4. **Pullback RSI** — gives extra weight to symbols with RSI 35–55 (healthy dip in an uptrend) and refuses to chase RSI > 70.
5. **Trend strength (ADX)** — higher ADX = stronger trend = bigger score.
6. **MACD histogram** — positive and rising adds to score.
7. **Volume confirmation** — recent volume above its 20-bar average adds to score.
8. **Proximity to 50-bar highs** — a breakout bonus.

A symbol must pass all hard filters AND score ≥ 0.55 to be eligible. The bot ranks
candidates by score and opens up to 5 positions, biggest-score-first.

### Position sizing & exits
- **Volatility-targeted sizing**: risk 1% of equity per trade, calculated as `equity * 1% / (2 × ATR)`. High-volatility names automatically get smaller positions.
- **Cap**: never more than 15% of equity in a single name.
- **Exits**: bracket orders with stop at `entry – 2×ATR` and target at `entry + 4×ATR` (2:1 reward:risk).
- **Kill switch**: bot halts for the day if equity drops >3%.

## Run it

```bash
pip install -r requirements.txt
python app.py
# → open http://localhost:5000, paste keys, click Start
```

## Deploy from your phone (Render.com)

1. Get Alpaca paper keys at app.alpaca.markets
2. render.com → New → Blueprint → connect this repo (auto-detects `render.yaml`)
3. Apply, wait ~3 min for the build
4. Open the Render URL in Safari, paste keys, click **Start**

The free Render tier sleeps after 15 min idle. For a 24/7 bot use the Starter plan
($7/mo) — already specified in `render.yaml`.

## Tuning

All knobs are in `config.py`:
- Universe (`DEFAULT_SYMBOLS`)
- Entry threshold (`MIN_SIGNAL_SCORE`)
- Per-trade risk (`RISK_PER_TRADE_PCT`)
- Position cap (`MAX_POSITION_PCT`), max concurrent (`MAX_POSITIONS`)
- Stop/target multiples (`ATR_STOP_MULT`, `ATR_TARGET_MULT`)
- Daily kill switch (`DAILY_MAX_LOSS_PCT`)
- Trade cadence (`TRADE_INTERVAL_MINUTES`)

All signal weights and filter thresholds live in `strategy.py` — edit them there.

## Honest caveats

- "Pre-equipped" doesn't mean "pre-profitable". This is a sensible default strategy, not magic.
- Backtest the strategy against your universe before live trading.
- Watch for regime shifts. The SPY 200-day filter helps but isn't a complete defense.
- Trading commissions on Alpaca are zero, but **slippage and the bid-ask spread are real costs** you'll see in live results that don't show in idealized backtests.
