"""Run one pass over the watchlist, or loop with --loop."""
import argparse
import logging
import time

from .analyst import analyse
from .broker import PaperBroker
from .chart import render_chart
from .config import settings
from .data import fetch_bars
from .indicators import add_indicators
from . import risk

log = logging.getLogger("bot")


def run_once(cfg, broker):
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

        acct = broker.state(symbol)
        ok, notional, why = risk.check(cfg.risk, symbol, decision.action, decision.confidence, acct)
        if not ok:
            log.info("%s: no order, %s", symbol, why)
            continue
        if cfg.dry_run:
            log.info("%s: DRY RUN would %s $%.2f", symbol, decision.action, notional)
            continue
        if decision.action == "buy":
            broker.buy(symbol, notional)
        else:
            broker.close(symbol)
        log.info("%s: paper order sent: %s $%.2f", symbol, decision.action, notional)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true", help="repeat every bar until stopped")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not (settings.alpaca_key and settings.alpaca_secret):
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file first.")
    log.info("paper account, dry_run=%s, analyst=%s", settings.dry_run,
             settings.model if settings.openrouter_key else "rules")
    broker = PaperBroker(settings)
    while True:
        run_once(settings, broker)
        if not args.loop:
            break
        time.sleep(settings.timeframe_minutes * 60)


if __name__ == "__main__":
    main()
