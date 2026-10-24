"""Run one pass over the watchlist, or loop with --loop."""
import argparse
import logging
import time
from datetime import datetime

from .analyst import analyse
from .broker import PaperBroker
from .chart import render_chart
from .config import settings
from .data import fetch_bars, fetch_daily_bars
from .indicators import add_indicators
from .strategy import add_trend_indicators, decide
from . import risk

log = logging.getLogger("bot")


def place(cfg, broker, symbol, decision, acct):
    ok, notional, why = risk.check(cfg.risk, symbol, decision.action, decision.confidence, acct)
    if not ok:
        log.info("%s: no order, %s", symbol, why)
        return
    if cfg.dry_run:
        log.info("%s: DRY RUN would %s $%.2f", symbol, decision.action, notional)
        return
    if decision.action == "buy":
        broker.buy(symbol, notional)
    else:
        broker.close(symbol)
    log.info("%s: paper order sent: %s $%.2f", symbol, decision.action, notional)


def run_trend_once(cfg, broker):
    """Once a day while the market is open: compare yesterday's close with its 200-day average."""
    if not broker.market_open():
        log.info("market closed, skipping")
        return False
    for symbol in cfg.risk.symbols:
        df = fetch_daily_bars(cfg, symbol)
        if len(df) < cfg.trend_sma_days:
            log.info("%s: only %d daily bars, need %d", symbol, len(df), cfg.trend_sma_days)
            continue
        df = add_trend_indicators(df, cfg.trend_sma_days)
        acct = broker.state(symbol)
        decision = decide(cfg, df, symbol, holding=acct.position_value > 0)
        log.info("%s: %s %s", symbol, decision.action, decision.reason)
        place(cfg, broker, symbol, decision, acct)
    return True


def run_once(cfg, broker):
    if cfg.strategy == "trend":
        return run_trend_once(cfg, broker)
    if not broker.market_open():
        log.info("market closed, skipping")
        return
    for symbol in cfg.risk.symbols:
        df = fetch_bars(cfg, symbol)
        if len(df) < 35:
            log.info("%s: not enough bars", symbol)
            continue
        df = add_indicators(df)
        png = render_chart(df, symbol)
        decision = analyse(cfg, df, symbol, png)
        log.info("%s: %s (%.2f) %s", symbol, decision.action, decision.confidence, decision.reason)

        place(cfg, broker, symbol, decision, broker.state(symbol))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true",
                   help="keep running until stopped (trend: once per trading day; intraday: every bar)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not (settings.alpaca_key and settings.alpaca_secret):
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file first.")
    ai = settings.model if settings.openrouter_key else "off"
    if settings.strategy == "trend" and not settings.ai_veto:
        ai = "off"
    log.info("paper account, dry_run=%s, strategy=%s, AI=%s", settings.dry_run, settings.strategy, ai)
    broker = PaperBroker(settings)
    last_day = None
    while True:
        if settings.strategy == "trend":
            today = datetime.now().date()
            if today != last_day and run_trend_once(settings, broker):
                last_day = today
            wait = 30 * 60
        else:
            run_once(settings, broker)
            wait = settings.timeframe_minutes * 60
        if not args.loop:
            break
        time.sleep(wait)


if __name__ == "__main__":
    main()
