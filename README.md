# Paper trading bot (Alpaca + AI via OpenRouter)

A small Python bot that looks at price charts for SPY and QQQ, asks an AI model through OpenRouter (or a simple
rule set) whether to buy, sell or hold, checks the answer against risk limits, and places
orders on your **Alpaca paper account only**. There is no live-trading code path:
`paper=True` is hard-coded and asserted.

## How it works (one pass)

1. **Fetch bars**: last ~120 fifteen-minute bars from Alpaca (free IEX feed). `bot/data.py`
2. **Indicators**: SMA 10, SMA 30, RSI 14. `bot/indicators.py`
3. **Chart**: draws a candlestick chart PNG. `bot/chart.py`
4. **Analyse**: sends the chart image to the model set in `AI_MODEL` (via OpenRouter) and gets back JSON
   `{action, confidence, reason}`. With no OpenRouter key it falls back to an SMA crossover rule. `bot/analyst.py`
5. **Risk check**: every order must pass `bot/risk.py`:
   - only symbols on the allow-list (SPY, QQQ)
   - at most 10% of equity in one symbol
   - at most 4 orders per day
   - stop for the day after a 2% drop in equity
   - ignore signals with confidence below 0.65
6. **Order**: buys a dollar amount (market order), or closes the position on "sell".
   With `DRY_RUN=true` (the default) it only logs what it *would* do.

Limits live in `bot/config.py` (`RiskLimits`).

## Setup on your computer

You need Python 3.11+.

```bash
cd trading-bot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` in a text editor and paste your keys there (never in chat, never in code):

- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`: from the Alpaca dashboard, make sure the
  toggle at the top says **Paper**, then "API Keys" → Generate/Regenerate.
- `OPENROUTER_API_KEY` (optional): from openrouter.ai → Keys. Each chart analysis is one
  API call, so a loop over 2 symbols every 15 minutes is roughly 50 calls per trading day.
- `AI_MODEL`: any OpenRouter model that accepts images. Default `anthropic/claude-sonnet-5`
  (about $0.20 per trading day); `google/gemini-3.6-flash` is about a third of that.

`.env` is listed in `.gitignore`, so it won't be uploaded if you put this on GitHub.

## Run

```bash
python -m bot.main          # one pass, then exit
python -m bot.main --loop   # repeat every 15 minutes (Ctrl+C to stop)
pytest -q                   # run the tests (no keys needed)
```

The US market is open 15:30–22:00 German time (Mon–Fri); outside those hours the bot
logs "market closed" and does nothing. Watch a few dry runs, then set `DRY_RUN=false` to
let it place paper orders, and check them in the Alpaca paper dashboard.

## Roadmap

1. Run in dry-run mode for a few days and read the reasons it gives.
2. Turn on paper orders; log every decision to a CSV so results can be reviewed.
3. Backtest the rules and the AI on past data to see whether either beats just holding SPY.
4. Add stop-loss / take-profit (bracket orders).
5. Only after weeks of paper results: think about real money (and re-check whether
   Alpaca accepts live accounts from Germany).

AI chart reading is an experiment, not a proven edge. Treat paper results with suspicion
until they hold up over many weeks.
