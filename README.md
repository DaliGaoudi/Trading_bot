# Paper trading bot (Alpaca + AI via OpenRouter)

A small Python bot that trades SPY and QQQ on your **Alpaca paper account only**. There is no
live-trading code path: `paper=True` is hard-coded and asserted.

It has two strategies, picked with `STRATEGY` in `.env`:

- **`trend` (default)**: once a day, hold a symbol while its last daily close is above its 200-day
  average, and move to cash when it closes below. This is the rule with the strongest long-run
  evidence. If `OPENROUTER_API_KEY` is set, the AI looks at a daily chart before each buy and can
  **veto** it (for example "wait, RSI is extreme"). The AI can never start a trade or block a sell.
  That is at most one call per symbol per day, usually a handful a month.
- **`intraday`**: the original 15-minute strategy below, where the AI reads the chart and decides.
  Its year-long backtest made less than simply holding, so it's kept for comparison.

## Trend strategy (one daily pass)

1. **Fetch daily bars**: about 400 days from Alpaca, completed days only. `bot/data.py`
2. **Rule**: close above the 200-day average and not holding → buy; close below and holding → sell;
   otherwise do nothing. `bot/strategy.py`
3. **AI veto** (buys only, optional): sends a daily chart (SMA 50/200, RSI) to `AI_MODEL`; anything
   but a confident "buy" (confidence ≥ 0.65) cancels the buy for today. Set `AI_VETO=false` to turn it off.
4. **Risk check and order**: the same `bot/risk.py` limits and dry-run switch as below.

With `--loop` the bot checks every 30 minutes and runs the daily pass once per trading day while the
market is open. Orders fill during the session after the close the decision was based on.

## Intraday strategy (one pass)

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
python -m bot.main --loop   # keep running (trend: once a day; intraday: every 15 min). Ctrl+C stops
pytest -q                   # run the tests (no keys needed)
```

## Backtest

```bash
python -m bot.backtest                          # trend rule, last 365 days of daily bars
python -m bot.backtest --days 1825              # trend rule over 5 years (more meaningful)
python -m bot.backtest --max-position-pct 0.5   # same, with 50% per symbol instead of 10%
python -m bot.backtest --analyst ai             # trend rule + AI veto: prints the max call count and cost first
python -m bot.backtest --strategy intraday      # the original 15-minute strategy
```

It replays the same decision code and `bot/risk.py` limits bar by bar, starting from $100,000,
filling each order at the next bar's open (plus 0.01% slippage), and prints profit/loss, max
drawdown and order count next to buy-and-hold SPY and a hold with the same maximum exposure. Results go to `backtest_results/`
(`summary.txt`, `trades.csv`, `equity.csv`, `equity.png`); downloaded bars are cached in `data/`.
Replaying the intraday AI over a year is about 12,500 chart calls (roughly $25-65 with Sonnet 5);
the trend veto needs at most a few hundred. Either way it asks before sending anything. A backtest of
any AI over dates it may have seen in training flatters it, so forward paper trading is the real test. Past results don't predict future returns.

The US market is open 15:30–22:00 German time (Mon–Fri); outside those hours the bot
logs "market closed" and does nothing. Watch a few dry runs, then set `DRY_RUN=false` to
let it place paper orders, and check them in the Alpaca paper dashboard.

## Roadmap

1. Run in dry-run mode for a few days and read the reasons it gives.
2. Turn on paper orders; log every decision to a CSV so results can be reviewed.
3. Backtest the rules and the AI on past data to see whether either beats just holding SPY
   (`python -m bot.backtest`).
4. Add stop-loss / take-profit (bracket orders).
5. Only after weeks of paper results: think about real money (and re-check whether
   Alpaca accepts live accounts from Germany).

AI chart reading is an experiment, not a proven edge. Treat paper results with suspicion
until they hold up over many weeks.
